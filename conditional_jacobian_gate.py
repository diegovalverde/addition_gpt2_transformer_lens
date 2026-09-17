"""Context-dependent JVP of the layer-1 carry-write transformation."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from addition_gpt.data import EQUALS_ID
from addition_gpt.model import ModelConfig, build_model
from conditional_write_interaction import factorial
from mechanistic_dependency import paired_prompts
from residual_causal_trace import score, state
from train import resolve_device

L0 = "blocks.0.hook_resid_post"; L1 = "blocks.1.hook_resid_post"

@dataclass(frozen=True)
class Run:
    name: str; checkpoint: Path; width: int

def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--train-pairs", type=int, default=400); p.add_argument("--examples", type=int, default=200); p.add_argument("--epsilon", type=float, default=.001); p.add_argument("--seed", type=int, default=1700)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"); p.add_argument("--output-dir", type=Path, default=Path("artifacts/conditional_jacobian_gate")); return p.parse_args()

def load(run: Run, device: str) -> torch.nn.Module:
    saved = torch.load(run.checkpoint, map_location=device, weights_only=False); model = build_model(ModelConfig(**saved["model_config"]), device); model.load_state_dict(saved["model_state_dict"]); model.eval(); return model

def leading(delta: torch.Tensor) -> torch.Tensor:
    d = torch.linalg.svd(delta, full_matrices=False).Vh[0]
    return d if float((delta @ d).mean()) >= 0 else -d

def endpoint(model: torch.nn.Module, prompts: torch.Tensor, direction: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    equals = int((prompts[0] == EQUALS_ID).nonzero()[0]); captured: list[torch.Tensor] = []
    def inject(value: torch.Tensor, hook: object) -> torch.Tensor:
        changed = value.clone(); changed[:, equals] += epsilon * direction; return changed
    def capture(value: torch.Tensor, hook: object) -> torch.Tensor:
        captured.append(value[:, -1]); return value
    logits = model.run_with_hooks(prompts, fwd_hooks=[(L0, inject), (L1, capture)])[:, -1]
    return captured[0].detach(), logits

def main() -> None:
    a = args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True); rows: list[dict[str, object]] = []
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device)
        tr_t, tr_s, _, _ = paired_prompts(run.width, a.train_pairs, a.seed + 100, device); equals = int((tr_t[0] == EQUALS_ID).nonzero()[0])
        d0 = leading(state(model, tr_s, L0, equals) - state(model, tr_t, L0, equals)); d1 = leading(state(model, tr_s, L1, -1) - state(model, tr_t, L1, -1))
        data = factorial(run.width, a.examples, a.seed, device); summaries: dict[int, dict[str, float]] = {}
        for raw_sum in (8, 9):
            prompts, _, _, target_sum = data[f"c0r{raw_sum}"]; _, _, _, source_sum = data[f"c1r{raw_sum}"]
            hp, lp = endpoint(model, prompts, d0, a.epsilon); hm, lm = endpoint(model, prompts, d0, -a.epsilon)
            jvp = (hp - hm) / (2 * a.epsilon); sp, _ = score(lp, target_sum, source_sum); sm, _ = score(lm, target_sum, source_sum); score_jvp = (sp - sm) / (2 * a.epsilon)
            projection = jvp @ d1
            summaries[raw_sum] = {"transported_projection": float(projection.mean()), "score_jvp": float(score_jvp.mean().detach()), "jvp_norm": float(jvp.norm(dim=-1).mean())}
            rows.append({"phase": "context_jvp", "model": run.name, "raw_tens_sum": raw_sum, **summaries[raw_sum]})
        rows.append({"phase": "raw9_minus_raw8", "model": run.name, "raw_tens_sum": 9, "transported_projection": summaries[9]["transported_projection"] - summaries[8]["transported_projection"], "score_jvp": summaries[9]["score_jvp"] - summaries[8]["score_jvp"], "jvp_norm": summaries[9]["jvp_norm"] - summaries[8]["jvp_norm"]})
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n"); fields = sorted({k for r in rows for k in r})
    with (a.output_dir / "results.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")

if __name__ == "__main__": main()
