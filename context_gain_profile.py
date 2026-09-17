"""Locate the raw-nine signed-gain flip across later residual sites."""
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

SITES = [("blocks.1.hook_resid_post", "l1_post_units"), ("blocks.2.hook_resid_post", "l2_post_units"), ("blocks.3.hook_resid_post", "l3_post_units")]

@dataclass(frozen=True)
class Run:
    name: str; checkpoint: Path; width: int

def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--train-pairs", type=int, default=400); p.add_argument("--examples", type=int, default=200); p.add_argument("--seed", type=int, default=1900)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"); p.add_argument("--output-dir", type=Path, default=Path("artifacts/context_gain_profile")); return p.parse_args()

def load(run: Run, device: str) -> torch.nn.Module:
    saved = torch.load(run.checkpoint, map_location=device, weights_only=False); model = build_model(ModelConfig(**saved["model_config"]), device); model.load_state_dict(saved["model_state_dict"]); model.eval(); return model

def gain(model: torch.nn.Module, prompts: torch.Tensor, target: torch.Tensor, source: torch.Tensor, name: str, d: torch.Tensor) -> torch.Tensor:
    captured: list[torch.Tensor] = []
    def retain(value: torch.Tensor, hook: object) -> torch.Tensor:
        value.retain_grad(); captured.append(value); return value
    model.zero_grad(set_to_none=True); logits = model.run_with_hooks(prompts, fwd_hooks=[(name, retain)])[:, -1]; s, _ = score(logits, target, source); s.sum().backward()
    return (captured[0].grad[:, -1] * d).sum(-1).detach()

def main() -> None:
    a = args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True); rows: list[dict[str, object]] = []
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device)
        tr_t, tr_s, _, _ = paired_prompts(run.width, a.train_pairs, a.seed + 100, device)
        data = factorial(run.width, a.examples, a.seed, device)
        per_site: dict[str, dict[int, float]] = {}
        for name, label in SITES:
            delta = state(model, tr_s, name, -1) - state(model, tr_t, name, -1); d = torch.linalg.svd(delta, full_matrices=False).Vh[0]
            if float((delta @ d).mean()) < 0: d = -d
            per_site[label] = {}
            for raw_sum in (8, 9):
                prompts, _, _, target_sum = data[f"c0r{raw_sum}"]; _, _, _, source_sum = data[f"c1r{raw_sum}"]
                value = float(gain(model, prompts, target_sum, source_sum, name, d).mean()); per_site[label][raw_sum] = value
                rows.append({"phase": "context_gain", "model": run.name, "site": label, "raw_tens_sum": raw_sum, "mean_gain": value})
            rows.append({"phase": "raw9_minus_raw8", "model": run.name, "site": label, "raw_tens_sum": 9, "mean_gain": per_site[label][9] - per_site[label][8]})
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n"); fields = sorted({k for r in rows for k in r})
    with (a.output_dir / "results.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")

if __name__ == "__main__": main()
