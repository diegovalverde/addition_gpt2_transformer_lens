"""Test one-dimensional residual transport from layer 0 ``=`` to layer 1 units."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from addition_gpt.data import EQUALS_ID
from addition_gpt.model import ModelConfig, build_model
from mechanistic_dependency import paired_prompts
from residual_causal_trace import score, state
from train import resolve_device

L0 = "blocks.0.hook_resid_post"; L1 = "blocks.1.hook_resid_post"

@dataclass(frozen=True)
class Run:
    name: str; checkpoint: Path; width: int

def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--train-pairs", type=int, default=400); p.add_argument("--examples", type=int, default=200); p.add_argument("--epsilon", type=float, default=.001); p.add_argument("--seed", type=int, default=300)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"); p.add_argument("--output-dir", type=Path, default=Path("artifacts/residual_direction_transport")); return p.parse_args()

def load(run: Run, device: str) -> torch.nn.Module:
    saved = torch.load(run.checkpoint, map_location=device, weights_only=False); model = build_model(ModelConfig(**saved["model_config"]), device); model.load_state_dict(saved["model_state_dict"]); model.eval(); return model

def leading(delta: torch.Tensor) -> torch.Tensor: return torch.linalg.svd(delta, full_matrices=False).Vh[0]

def l0_patch_logits(model: torch.nn.Module, prompts: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
    equals = int((prompts[0] == EQUALS_ID).nonzero()[0])
    def edit(value: torch.Tensor, hook: object) -> torch.Tensor:
        changed = value.clone(); changed[:, equals] = replacement; return changed
    return model.run_with_hooks(prompts, fwd_hooks=[(L0, edit)])[:, -1]

def transport(model: torch.nn.Module, prompts: torch.Tensor, direction: torch.Tensor, epsilon: float) -> torch.Tensor:
    equals = int((prompts[0] == EQUALS_ID).nonzero()[0])
    def endpoint(sign: float) -> torch.Tensor:
        captured: list[torch.Tensor] = []
        def inject(value: torch.Tensor, hook: object) -> torch.Tensor:
            changed = value.clone(); changed[:, equals] += sign * epsilon * direction; return changed
        def capture(value: torch.Tensor, hook: object) -> torch.Tensor:
            captured.append(value[:, -1]); return value
        model.run_with_hooks(prompts, fwd_hooks=[(L0, inject), (L1, capture)])
        return captured[0].detach()
    return (endpoint(1) - endpoint(-1)) / (2 * epsilon)

def main() -> None:
    a = args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True); rows: list[dict[str, object]] = []
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device)
        tr_t, tr_s, _, _ = paired_prompts(run.width, a.train_pairs, a.seed + 100, device)
        equals_train = int((tr_t[0] == EQUALS_ID).nonzero()[0])
        d0 = leading(state(model, tr_s, L0, equals_train) - state(model, tr_t, L0, equals_train))
        d1 = leading(state(model, tr_s, L1, -1) - state(model, tr_t, L1, -1))
        target_p, source_p, target, source = paired_prompts(run.width, a.examples, a.seed, device)
        equals = int((target_p[0] == EQUALS_ID).nonzero()[0]); h0_t, h0_s = state(model, target_p, L0, equals), state(model, source_p, L0, equals)
        delta = h0_s - h0_t; projected = (delta @ d0[:, None]) * d0[None, :]
        baseline_s, _ = score(model(target_p)[:, -1], target, source)
        for kind, replacement in (("projected_l0_sufficiency", h0_t + projected), ("projected_l0_removal", h0_s - projected)):
            logits = l0_patch_logits(model, target_p, replacement); changed_s, sd = score(logits, target, source)
            rows.append({"phase": kind, "model": run.name, "mean_delta_s": float((changed_s - baseline_s).mean().detach()), "source_tens_argmax_fraction": float(logits.argmax(-1).eq(sd).float().mean())})
        jvp = transport(model, target_p, d0, a.epsilon)
        alignment = torch.nn.functional.cosine_similarity(jvp, d1, dim=-1)
        rows.append({"phase": "transport", "model": run.name, "l0_l1_direction_cosine": float(torch.dot(d0, d1).abs()), "mean_jvp_norm": float(jvp.norm(dim=-1).mean()), "mean_jvp_l1_alignment": float(alignment.mean()), "positive_jvp_l1_alignment_fraction": float((alignment > 0).float().mean())})
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n"); fields = sorted({k for r in rows for k in r})
    with (a.output_dir / "results.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")

if __name__ == "__main__": main()
