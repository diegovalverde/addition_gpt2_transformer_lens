"""Held-out rank-k causal subspaces for carry x raw-nine residual interactions."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from addition_gpt.model import ModelConfig, build_model
from conditional_write_interaction import factorial
from residual_causal_trace import score, state
from train import resolve_device

SITE = "blocks.1.hook_resid_post"

@dataclass(frozen=True)
class Run:
    name: str; checkpoint: Path; width: int

def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--train-examples", type=int, default=400); p.add_argument("--examples", type=int, default=200); p.add_argument("--dimensions", default="1,2,4,8,16,32,64,128"); p.add_argument("--random-subspaces", type=int, default=10); p.add_argument("--seed", type=int, default=2300)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"); p.add_argument("--output-dir", type=Path, default=Path("artifacts/conditional_interaction_subspace")); return p.parse_args()

def load(run: Run, device: str) -> torch.nn.Module:
    saved = torch.load(run.checkpoint, map_location=device, weights_only=False); model = build_model(ModelConfig(**saved["model_config"]), device); model.load_state_dict(saved["model_state_dict"]); model.eval(); return model

def interaction(model: torch.nn.Module, data: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    h = {key: state(model, value[0], SITE, -1) for key, value in data.items()}
    return (h["c1r9"] - h["c0r9"]) - (h["c1r8"] - h["c0r8"])

def patch(model: torch.nn.Module, prompts: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
    def edit(value: torch.Tensor, hook: object) -> torch.Tensor:
        changed = value.clone(); changed[:, -1] = replacement; return changed
    return model.run_with_hooks(prompts, fwd_hooks=[(SITE, edit)])[:, -1]

def metrics(logits: torch.Tensor, reference: torch.Tensor, target: torch.Tensor, source: torch.Tensor) -> tuple[float, float]:
    s, sd = score(logits, target, source); return float((s - reference).mean().detach()), float(logits.argmax(-1).eq(sd).float().mean())

def random_basis(d: int, k: int, seed: int) -> torch.Tensor:
    return torch.linalg.qr(torch.randn((d, k), generator=torch.Generator().manual_seed(seed)), mode="reduced").Q.T

def main() -> None:
    a = args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True); dims = [int(x) for x in a.dimensions.split(",")]; rows: list[dict[str, object]] = []
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device)
        train_i = interaction(model, factorial(run.width, a.train_examples, a.seed + 100, device)); basis = torch.linalg.svd(train_i, full_matrices=False).Vh
        data = factorial(run.width, a.examples, a.seed, device); i = interaction(model, data)
        target_p, _, _, target_sum = data["c0r9"]; source_p, _, _, source_sum = data["c1r9"]
        h_target, h_source = state(model, target_p, SITE, -1), state(model, source_p, SITE, -1)
        target_s, _ = score(model(target_p)[:, -1], target_sum, source_sum); source_s, _ = score(model(source_p)[:, -1], target_sum, source_sum)
        for k in dims:
            learned = basis[:k].to(device)
            for kind, directions in [("learned", learned), *[("random", random_basis(h_target.shape[-1], k, a.seed + 10000 * j + k).to(device)) for j in range(a.random_subspaces)]]:
                projected = (i @ directions.T) @ directions
                for phase, prompts, replacement, reference in (("interaction_sufficiency", target_p, h_target + projected, target_s), ("interaction_removal", source_p, h_source - projected, source_s)):
                    delta_s, source_rate = metrics(patch(model, prompts, replacement), reference, target_sum, source_sum)
                    rows.append({"model": run.name, "phase": phase, "dimension": k, "subspace": kind, "mean_delta_s": delta_s, "source_tens_argmax_fraction": source_rate})
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n"); fields = sorted({key for row in rows for key in row})
    with (a.output_dir / "results.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")

if __name__ == "__main__": main()
