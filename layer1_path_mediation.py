"""Causal path mediation between layer-1 attention and MLP writes at units."""
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

ATTN = "blocks.1.hook_attn_out"; MLP = "blocks.1.hook_mlp_out"

@dataclass(frozen=True)
class Run:
    name: str; checkpoint: Path; width: int

def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--examples", type=int, default=200); p.add_argument("--seed", type=int, default=2900)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"); p.add_argument("--output-dir", type=Path, default=Path("artifacts/layer1_path_mediation")); return p.parse_args()

def load(run: Run, device: str) -> torch.nn.Module:
    saved = torch.load(run.checkpoint, map_location=device, weights_only=False); model = build_model(ModelConfig(**saved["model_config"]), device); model.load_state_dict(saved["model_state_dict"]); model.eval(); return model

@torch.no_grad()
def values(model: torch.nn.Module, prompts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, cache = model.run_with_cache(prompts, names_filter=lambda n: n in {ATTN, MLP}); return cache[ATTN][:, -1].clone(), cache[MLP][:, -1].clone()

def patched(model: torch.nn.Module, prompts: torch.Tensor, attn: torch.Tensor | None, mlp: torch.Tensor | None) -> torch.Tensor:
    hooks = []
    if attn is not None:
        def edit_attn(value: torch.Tensor, hook: object) -> torch.Tensor:
            changed = value.clone(); changed[:, -1] = attn; return changed
        hooks.append((ATTN, edit_attn))
    if mlp is not None:
        def edit_mlp(value: torch.Tensor, hook: object) -> torch.Tensor:
            changed = value.clone(); changed[:, -1] = mlp; return changed
        hooks.append((MLP, edit_mlp))
    return model.run_with_hooks(prompts, fwd_hooks=hooks)[:, -1] if hooks else model(prompts)[:, -1]

def main() -> None:
    a = args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True); rows: list[dict[str, object]] = []
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device); data = factorial(run.width, a.examples, a.seed, device)
        for raw_sum in (8, 9):
            target_p, _, _, target_sum = data[f"c0r{raw_sum}"]; source_p, _, _, source_sum = data[f"c1r{raw_sum}"]
            target_attn, target_mlp = values(model, target_p); source_attn, source_mlp = values(model, source_p)
            baseline_s, _ = score(model(target_p)[:, -1], target_sum, source_sum)
            variants = [
                ("source_attn", source_attn, None),
                ("source_attn_clean_mlp_restore", source_attn, target_mlp),
                ("source_mlp", None, source_mlp),
                ("both_source", source_attn, source_mlp),
                ("zero_both", torch.zeros_like(target_attn), torch.zeros_like(target_mlp)),
                ("zero_attn_clean_mlp_restore", torch.zeros_like(target_attn), target_mlp),
            ]
            for label, attn, mlp in variants:
                out = patched(model, target_p, attn, mlp); s, sd = score(out, target_sum, source_sum)
                rows.append({"model": run.name, "raw_tens_sum": raw_sum, "intervention": label, "mean_delta_s": float((s - baseline_s).mean().detach()), "source_tens_argmax_fraction": float(out.argmax(-1).eq(sd).float().mean())})
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n"); fields = sorted({k for r in rows for k in r})
    with (a.output_dir / "results.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")

if __name__ == "__main__": main()
