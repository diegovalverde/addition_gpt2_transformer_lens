"""Test carry-direction plus interaction-subspace synergy at the raw-nine gate."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from addition_gpt.model import ModelConfig, build_model
from conditional_interaction_subspace import interaction, random_basis
from conditional_write_interaction import factorial
from mechanistic_dependency import paired_prompts
from raw_carry_branch_rescue import decode, generate
from residual_causal_trace import state
from train import resolve_device

SITE = "blocks.1.hook_resid_post"

@dataclass(frozen=True)
class Run:
    name: str; checkpoint: Path; width: int

def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--train-examples", type=int, default=400); p.add_argument("--examples", type=int, default=200); p.add_argument("--interaction-dimension", type=int, default=16); p.add_argument("--random-subspaces", type=int, default=10); p.add_argument("--seed", type=int, default=2500)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"); p.add_argument("--output-dir", type=Path, default=Path("artifacts/carry_interaction_synergy")); return p.parse_args()

def load(run: Run, device: str) -> torch.nn.Module:
    saved = torch.load(run.checkpoint, map_location=device, weights_only=False); model = build_model(ModelConfig(**saved["model_config"]), device); model.load_state_dict(saved["model_state_dict"]); model.eval(); return model

def main() -> None:
    a = args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True); rows: list[dict[str, object]] = []
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device)
        # Independent carry axis and interaction basis.
        ct, cs, _, _ = paired_prompts(run.width, a.train_examples, a.seed + 100, device); carry_delta_train = state(model, cs, SITE, -1) - state(model, ct, SITE, -1); carry_direction = torch.linalg.svd(carry_delta_train, full_matrices=False).Vh[0]
        if float((carry_delta_train @ carry_direction).mean()) < 0: carry_direction = -carry_direction
        train_basis = torch.linalg.svd(interaction(model, factorial(run.width, a.train_examples, a.seed + 200, device)), full_matrices=False).Vh[:a.interaction_dimension]
        data = factorial(run.width, a.examples, a.seed, device); target_p, _, _, target_sum = data["c0r9"]; source_p, _, _, source_sum = data["c1r9"]
        h_t, h_s = state(model, target_p, SITE, -1), state(model, source_p, SITE, -1); carry_component = (h_s - h_t) @ carry_direction[:, None] * carry_direction[None, :]
        i = interaction(model, data); interaction_component = (i @ train_basis.T) @ train_basis
        variants: list[tuple[str, torch.Tensor]] = [("carry_only", carry_component), ("interaction_only", interaction_component), ("carry_plus_interaction", carry_component + interaction_component)]
        for j in range(a.random_subspaces):
            basis = random_basis(h_t.shape[-1], a.interaction_dimension, a.seed + 10000 + j).to(device); random_component = (i @ basis.T) @ basis
            variants.append(("carry_plus_random", carry_component + random_component))
        for label, correction in variants:
            answers = decode(generate(model, target_p, run.width, correction), run.width)
            rows.append({"model": run.name, "intervention": label, "source_exact_fraction": float(answers.eq(source_sum).float().mean()), "target_exact_fraction": float(answers.eq(target_sum).float().mean()), "mean_answer_shift": float((answers - target_sum).float().mean())})
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n"); fields = sorted({key for row in rows for key in row})
    with (a.output_dir / "results.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")

if __name__ == "__main__": main()
