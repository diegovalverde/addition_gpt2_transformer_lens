"""Sweep raw tens sums to test whether the residual direction is a carry variable."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from addition_gpt.data import decode_answer, encode_addition
from addition_gpt.model import ModelConfig, build_model
from compositional_dependency import generate
from mechanistic_dependency import paired_prompts
from residual_causal_trace import state
from train import resolve_device

SITE = "blocks.1.hook_resid_post"

@dataclass(frozen=True)
class Run:
    name: str; checkpoint: Path; width: int

def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--train-pairs", type=int, default=400); p.add_argument("--examples-per-sum", type=int, default=200); p.add_argument("--raw-sums", default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated raw tens sums in [0,18]."); p.add_argument("--seed", type=int, default=1100)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"); p.add_argument("--output-dir", type=Path, default=Path("artifacts/carry_direction_generality")); return p.parse_args()

def load(run: Run, device: str) -> torch.nn.Module:
    saved = torch.load(run.checkpoint, map_location=device, weights_only=False); model = build_model(ModelConfig(**saved["model_config"]), device); model.load_state_dict(saved["model_state_dict"]); model.eval(); return model

def pairs(width: int, raw_tens: int, examples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Only units carry differs; all columns above tens have no raw carry."""
    if not 0 <= raw_tens <= 18: raise ValueError("raw_tens must be in [0, 18]")
    g = torch.Generator().manual_seed(seed)
    target_units_sum = torch.randint(9, (examples,), generator=g)
    target_left_units = torch.stack([torch.randint(int(x) + 1, (1,), generator=g)[0] for x in target_units_sum]); target_right_units = target_units_sum - target_left_units
    source_units_sum = target_units_sum + 10
    source_left_units = torch.stack([torch.randint(int(x) - 9, 10, (1,), generator=g)[0] for x in source_units_sum]); source_right_units = source_units_sum - source_left_units
    lower, upper = max(0, raw_tens - 9), min(9, raw_tens)
    tens_left = torch.randint(lower, upper + 1, (examples,), generator=g); tens_right = raw_tens - tens_left
    target_left = target_left_units + 10 * tens_left; target_right = target_right_units + 10 * tens_right
    source_left = source_left_units + 10 * tens_left; source_right = source_right_units + 10 * tens_right
    for col in range(2, width):
        raw = torch.randint(9, (examples,), generator=g); left = torch.stack([torch.randint(int(x) + 1, (1,), generator=g)[0] for x in raw]); right = raw - left
        target_left += 10**col * left; target_right += 10**col * right; source_left += 10**col * left; source_right += 10**col * right
    target = torch.tensor([encode_addition(int(l), int(r), width) for l, r in zip(target_left, target_right)])
    source = torch.tensor([encode_addition(int(l), int(r), width) for l, r in zip(source_left, source_right)])
    return target, source, target_left + target_right, source_left + source_right

def decode(tokens: torch.Tensor, width: int) -> torch.Tensor: return torch.tensor([decode_answer(row, width) for row in tokens])

def main() -> None:
    a = args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True); rows: list[dict[str, object]] = []
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device)
        tr_t, tr_s, _, _ = paired_prompts(run.width, a.train_pairs, a.seed + 100, device)
        train_delta = state(model, tr_s, SITE, -1) - state(model, tr_t, SITE, -1); direction = torch.linalg.svd(train_delta, full_matrices=False).Vh[0]
        if float((train_delta @ direction).mean()) < 0: direction = -direction
        scale = float((train_delta @ direction).abs().mean())
        for raw_tens in (int(value) for value in a.raw_sums.split(",")):
            target, source, target_sum, source_sum = pairs(run.width, raw_tens, a.examples_per_sum, a.seed + raw_tens)
            prompt_length = 2 * run.width + 3; prompts = target[:, :prompt_length].to(device)
            baseline = decode(generate(model, prompts, run.width, direction, scale, set()), run.width)
            edited = decode(generate(model, prompts, run.width, direction, scale, {0}), run.width)
            rows.append({"model": run.name, "raw_tens_sum": raw_tens, "baseline_target_exact_fraction": float(baseline.eq(target_sum).float().mean()), "edited_source_exact_fraction": float(edited.eq(source_sum).float().mean()), "edited_target_exact_fraction": float(edited.eq(target_sum).float().mean()), "mean_answer_shift": float((edited - target_sum).float().mean())})
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n"); fields = sorted({k for r in rows for k in r})
    with (a.output_dir / "results.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")

if __name__ == "__main__": main()
