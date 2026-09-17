"""Compare and cross-intervene 1D residual directions across carry positions."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from addition_gpt.model import ModelConfig, build_model
from hundreds_dependency_subspace import pairs as hundreds_pairs, score as hundreds_score
from mechanistic_dependency import paired_prompts
from residual_causal_trace import score as tens_score, state
from train import resolve_device

SITE = "blocks.1.hook_resid_post"

@dataclass(frozen=True)
class Run:
    name: str; checkpoint: Path; width: int

def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--train-pairs", type=int, default=400); p.add_argument("--examples", type=int, default=200); p.add_argument("--seed", type=int, default=300)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"); p.add_argument("--output-dir", type=Path, default=Path("artifacts/compare_dependency_directions")); return p.parse_args()

def load(run: Run, device: str) -> torch.nn.Module:
    saved = torch.load(run.checkpoint, map_location=device, weights_only=False)
    model = build_model(ModelConfig(**saved["model_config"]), device); model.load_state_dict(saved["model_state_dict"]); model.eval(); return model

def direction(model: torch.nn.Module, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svd(state(model, source, SITE, -1) - state(model, target, SITE, -1), full_matrices=False).Vh[0]

def patch(model: torch.nn.Module, prompts: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
    def edit(value: torch.Tensor, hook: object) -> torch.Tensor:
        changed = value.clone(); changed[:, -1] = replacement; return changed
    return model.run_with_hooks(prompts, fwd_hooks=[(SITE, edit)])[:, -1]

def cross_row(model: torch.nn.Module, task: str, target_p: torch.Tensor, source_p: torch.Tensor, target: torch.Tensor, source: torch.Tensor, basis: torch.Tensor, score_fn: object) -> tuple[float, float]:
    h_target, h_source = state(model, target_p, SITE, -1), state(model, source_p, SITE, -1)
    delta = h_source - h_target; projected = (delta @ basis[:, None]) * basis[None, :]
    logits = patch(model, target_p, h_target + projected); baseline, _ = score_fn(model(target_p)[:, -1], target, source); changed, source_digit = score_fn(logits, target, source)
    return float((changed - baseline).mean().detach()), float(logits.argmax(-1).eq(source_digit).float().mean())

def main() -> None:
    a = args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True); rows: list[dict[str, object]] = []
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device)
        # Independent training templates for the two structural tasks.
        ut, us, _, _ = paired_prompts(run.width, a.train_pairs, a.seed + 100, device)
        ht, hs, _, _ = hundreds_pairs(run.width, a.train_pairs, a.seed + 500, device)
        unit_direction, hundred_direction = direction(model, ut, us), direction(model, ht, hs)
        cosine = float(torch.dot(unit_direction, hundred_direction).abs())
        rows.append({"phase": "direction_similarity", "model": run.name, "absolute_cosine": cosine})
        # Held-out cross-task projections.  "native" is retained as a positive control.
        eut, eus, eut_sum, eus_sum = paired_prompts(run.width, a.examples, a.seed, device)
        eht, ehs, eht_sum, ehs_sum = hundreds_pairs(run.width, a.examples, a.seed + 400, device)
        for task, target_p, source_p, target, source, scorer, own, other in (
            ("units_to_tens", eut, eus, eut_sum, eus_sum, tens_score, unit_direction, hundred_direction),
            ("tens_to_hundreds", eht, ehs, eht_sum, ehs_sum, hundreds_score, hundred_direction, unit_direction),
        ):
            for basis_name, basis in (("native", own), ("cross_position", other)):
                delta_s, source_rate = cross_row(model, task, target_p, source_p, target, source, basis, scorer)
                rows.append({"phase": "cross_task_projection", "model": run.name, "task": task, "basis": basis_name, "mean_delta_s": delta_s, "source_argmax_fraction": source_rate})
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = sorted({k for r in rows for k in r})
    with (a.output_dir / "results.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")

if __name__ == "__main__": main()
