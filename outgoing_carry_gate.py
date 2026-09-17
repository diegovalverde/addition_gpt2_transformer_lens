"""Directly test the raw-sum-nine outgoing-carry gate at the tens token."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from addition_gpt.model import ModelConfig, build_model
from carry_direction_generality import pairs
from mechanistic_dependency import paired_prompts
from residual_causal_trace import state
from train import resolve_device

SITE = "blocks.1.hook_resid_post"

@dataclass(frozen=True)
class Run:
    name: str; checkpoint: Path; width: int

def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--train-pairs", type=int, default=400); p.add_argument("--examples-per-sum", type=int, default=200); p.add_argument("--seed", type=int, default=1300)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"); p.add_argument("--output-dir", type=Path, default=Path("artifacts/outgoing_carry_gate")); return p.parse_args()

def load(run: Run, device: str) -> torch.nn.Module:
    saved = torch.load(run.checkpoint, map_location=device, weights_only=False); model = build_model(ModelConfig(**saved["model_config"]), device); model.load_state_dict(saved["model_state_dict"]); model.eval(); return model

def logits(model: torch.nn.Module, inputs: torch.Tensor, direction: torch.Tensor | None, scale: float) -> torch.Tensor:
    if direction is None: return model(inputs)[:, -1]
    def edit(value: torch.Tensor, hook: object) -> torch.Tensor:
        changed = value.clone(); changed[:, -1] += scale * direction; return changed
    return model.run_with_hooks(inputs, fwd_hooks=[(SITE, edit)])[:, -1]

def main() -> None:
    a = args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True); rows: list[dict[str, object]] = []
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device)
        tr_t, tr_s, _, _ = paired_prompts(run.width, a.train_pairs, a.seed + 100, device)
        train_delta = state(model, tr_s, SITE, -1) - state(model, tr_t, SITE, -1); direction = torch.linalg.svd(train_delta, full_matrices=False).Vh[0]
        if float((train_delta @ direction).mean()) < 0: direction = -direction
        scale = float((train_delta @ direction).abs().mean())
        for raw_tens in range(10):
            target, source, target_sum, source_sum = pairs(run.width, raw_tens, a.examples_per_sum, a.seed + raw_tens)
            prompt_length = 2 * run.width + 3
            # Teacher-force target units and tens. The next logit is hundreds.
            inputs = torch.cat((target[:, :prompt_length], target[:, prompt_length:prompt_length + 2]), 1).to(device)
            base, edited = logits(model, inputs, None, scale), logits(model, inputs, direction, scale)
            target_digit, source_digit = ((target_sum // 100) % 10).to(device), ((source_sum // 100) % 10).to(device)
            rows.append({"model": run.name, "raw_tens_sum": raw_tens,
                         "baseline_target_hundreds_fraction": float(base.argmax(-1).eq(target_digit).float().mean()),
                         "edited_target_hundreds_fraction": float(edited.argmax(-1).eq(target_digit).float().mean()),
                         "edited_source_hundreds_fraction": float(edited.argmax(-1).eq(source_digit).float().mean()),
                         "hundreds_changed_fraction": float(edited.argmax(-1).ne(base.argmax(-1)).float().mean())})
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n"); fields = sorted({k for r in rows for k in r})
    with (a.output_dir / "results.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")

if __name__ == "__main__": main()
