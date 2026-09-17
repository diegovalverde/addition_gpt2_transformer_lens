"""Residual-space localization of the layer-1 MLP's raw-nine gate."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from addition_gpt.model import ModelConfig, build_model
from conditional_write_interaction import factorial
from mechanistic_dependency import paired_prompts
from residual_causal_trace import score, state
from train import resolve_device

MID = "blocks.1.hook_resid_mid"; MLP = "blocks.1.hook_mlp_out"

@dataclass(frozen=True)
class Run:
    name: str; checkpoint: Path; width: int

def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--train-pairs", type=int, default=400); p.add_argument("--examples", type=int, default=200); p.add_argument("--epsilon", type=float, default=.001); p.add_argument("--seed", type=int, default=3100)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"); p.add_argument("--output-dir", type=Path, default=Path("artifacts/layer1_mlp_residual_gate")); return p.parse_args()

def load(run: Run, device: str) -> torch.nn.Module:
    saved = torch.load(run.checkpoint, map_location=device, weights_only=False); model = build_model(ModelConfig(**saved["model_config"]), device); model.load_state_dict(saved["model_state_dict"]); model.eval(); return model

def direction(delta: torch.Tensor) -> torch.Tensor:
    d = torch.linalg.svd(delta, full_matrices=False).Vh[0]; return d if float((delta @ d).mean()) >= 0 else -d

def patch_logits(model: torch.nn.Module, prompts: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
    def edit(value: torch.Tensor, hook: object) -> torch.Tensor:
        changed = value.clone(); changed[:, -1] = replacement; return changed
    return model.run_with_hooks(prompts, fwd_hooks=[(MID, edit)])[:, -1]

def endpoint(model: torch.nn.Module, prompts: torch.Tensor, d: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    captured: list[torch.Tensor] = []
    def edit(value: torch.Tensor, hook: object) -> torch.Tensor:
        changed = value.clone(); changed[:, -1] += epsilon * d; return changed
    def capture(value: torch.Tensor, hook: object) -> torch.Tensor:
        captured.append(value[:, -1]); return value
    logits = model.run_with_hooks(prompts, fwd_hooks=[(MID, edit), (MLP, capture)])[:, -1]
    return captured[0].detach(), logits

def main() -> None:
    a = args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True); rows: list[dict[str, object]] = []
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device)
        tr_t, tr_s, _, _ = paired_prompts(run.width, a.train_pairs, a.seed + 100, device)
        d_mid = direction(state(model, tr_s, MID, -1) - state(model, tr_t, MID, -1)); d_mlp = direction(state(model, tr_s, MLP, -1) - state(model, tr_t, MLP, -1))
        data = factorial(run.width, a.examples, a.seed, device); summaries: dict[int, dict[str, float]] = {}
        for raw_sum in (8, 9):
            target_p, _, _, target_sum = data[f"c0r{raw_sum}"]; source_p, _, _, source_sum = data[f"c1r{raw_sum}"]
            base_s, _ = score(model(target_p)[:, -1], target_sum, source_sum)
            source_mid = state(model, source_p, MID, -1); out = patch_logits(model, target_p, source_mid); patched_s, sd = score(out, target_sum, source_sum)
            plus, logits_plus = endpoint(model, target_p, d_mid, a.epsilon); minus, logits_minus = endpoint(model, target_p, d_mid, -a.epsilon)
            mlp_jvp = (plus - minus) / (2 * a.epsilon); score_jvp = (score(logits_plus, target_sum, source_sum)[0] - score(logits_minus, target_sum, source_sum)[0]) / (2 * a.epsilon)
            summaries[raw_sum] = {"mlp_transport_projection": float((mlp_jvp @ d_mlp).mean()), "score_jvp": float(score_jvp.mean().detach()), "mlp_jvp_norm": float(mlp_jvp.norm(dim=-1).mean())}
            rows.append({"phase": "resid_mid_source_patch", "model": run.name, "raw_tens_sum": raw_sum, "mean_delta_s": float((patched_s - base_s).mean().detach()), "source_tens_argmax_fraction": float(out.argmax(-1).eq(sd).float().mean())})
            rows.append({"phase": "mlp_jvp", "model": run.name, "raw_tens_sum": raw_sum, **summaries[raw_sum]})
        rows.append({"phase": "raw9_minus_raw8_mlp_jvp", "model": run.name, "raw_tens_sum": 9, **{key: summaries[9][key] - summaries[8][key] for key in summaries[8]}})
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n"); fields = sorted({k for r in rows for k in r})
    with (a.output_dir / "results.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")

if __name__ == "__main__": main()
