"""Decompose the established layer-1 residual gate into attention/MLP writes."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from addition_gpt.model import ModelConfig, build_model
from conditional_write_interaction import factorial
from residual_causal_trace import score
from train import resolve_device

COMPONENTS = [("attn_out", "blocks.1.hook_attn_out"), ("mlp_out", "blocks.1.hook_mlp_out")]

@dataclass(frozen=True)
class Run:
    name: str; checkpoint: Path; width: int

def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--examples", type=int, default=200); p.add_argument("--seed", type=int, default=2700)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"); p.add_argument("--output-dir", type=Path, default=Path("artifacts/layer1_component_gate")); return p.parse_args()

def load(run: Run, device: str) -> torch.nn.Module:
    saved = torch.load(run.checkpoint, map_location=device, weights_only=False); model = build_model(ModelConfig(**saved["model_config"]), device); model.load_state_dict(saved["model_state_dict"]); model.eval(); return model

@torch.no_grad()
def component(model: torch.nn.Module, prompts: torch.Tensor, name: str) -> torch.Tensor:
    _, cache = model.run_with_cache(prompts, names_filter=lambda n: n == name); return cache[name][:, -1].clone()

def patched(model: torch.nn.Module, prompts: torch.Tensor, name: str, replacement: torch.Tensor) -> torch.Tensor:
    def edit(value: torch.Tensor, hook: object) -> torch.Tensor:
        changed = value.clone(); changed[:, -1] = replacement; return changed
    return model.run_with_hooks(prompts, fwd_hooks=[(name, edit)])[:, -1]

def measure(logits: torch.Tensor, baseline_s: torch.Tensor, target: torch.Tensor, source: torch.Tensor) -> tuple[float, float]:
    s, sd = score(logits, target, source); return float((s - baseline_s).mean().detach()), float(logits.argmax(-1).eq(sd).float().mean())

def main() -> None:
    a = args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True); rows: list[dict[str, object]] = []
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device); data = factorial(run.width, a.examples, a.seed, device)
        for raw_sum in (8, 9):
            target_p, _, _, target_sum = data[f"c0r{raw_sum}"]; source_p, _, _, source_sum = data[f"c1r{raw_sum}"]; baseline_s, _ = score(model(target_p)[:, -1], target_sum, source_sum)
            for label, name in COMPONENTS:
                target_value, source_value = component(model, target_p, name), component(model, source_p, name)
                for phase, replacement in (("source_patch", source_value), ("zero_ablate", torch.zeros_like(target_value)), ("clean_rescue", target_value)):
                    delta_s, source_rate = measure(patched(model, target_p, name, replacement), baseline_s, target_sum, source_sum)
                    rows.append({"model": run.name, "raw_tens_sum": raw_sum, "component": label, "phase": phase, "mean_delta_s": delta_s, "source_tens_argmax_fraction": source_rate})
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n"); fields = sorted({k for r in rows for k in r})
    with (a.output_dir / "results.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")

if __name__ == "__main__": main()
