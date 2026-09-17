"""Held-out low-dimensional causal subspaces at layer-1 post, units token."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from addition_gpt.model import ModelConfig, build_model
from mechanistic_dependency import paired_prompts
from residual_causal_trace import score, state
from train import resolve_device


SITE = "blocks.1.hook_resid_post"


@dataclass(frozen=True)
class Run:
    name: str; checkpoint: Path; width: int


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--train-pairs", type=int, default=400)
    p.add_argument("--examples", type=int, default=200)
    p.add_argument("--dimensions", default="1,2,4,8,16,32,64,128")
    p.add_argument("--random-subspaces", type=int, default=10)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/residual_subspace")); return p.parse_args()


def load(run: Run, device: str) -> torch.nn.Module:
    saved = torch.load(run.checkpoint, map_location=device, weights_only=False)
    model = build_model(ModelConfig(**saved["model_config"]), device)
    model.load_state_dict(saved["model_state_dict"]); model.eval(); return model


def orthogonal_random(d_model: int, count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn((d_model, count), generator=generator)
    return torch.linalg.qr(matrix, mode="reduced").Q.T


def patch(model: torch.nn.Module, prompts: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
    def edit(value: torch.Tensor, hook: object) -> torch.Tensor:
        changed = value.clone(); changed[:, -1] = replacement; return changed
    return model.run_with_hooks(prompts, fwd_hooks=[(SITE, edit)])[:, -1]


def metrics(logits: torch.Tensor, baseline_s: torch.Tensor, target: torch.Tensor, source: torch.Tensor) -> tuple[float, float]:
    s, source_digit = score(logits, target, source)
    return float((s - baseline_s).mean().detach()), float(logits.argmax(-1).eq(source_digit).float().mean())


def main() -> None:
    a = args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True)
    dims = [int(x) for x in a.dimensions.split(",")]; rows: list[dict[str, object]] = []
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device)
        # Fit basis on an independent paired/template stream, never the fixed evaluation pairs.
        train_t, train_s, _, _ = paired_prompts(run.width, a.train_pairs, a.seed + 100, device)
        train_delta = state(model, train_s, SITE, -1) - state(model, train_t, SITE, -1)
        basis = torch.linalg.svd(train_delta, full_matrices=False).Vh  # uncentered: preserves the shared causal shift
        target_p, source_p, target, source = paired_prompts(run.width, a.examples, a.seed, device)
        h_target, h_source = state(model, target_p, SITE, -1), state(model, source_p, SITE, -1)
        delta = h_source - h_target
        baseline_s, _ = score(model(target_p)[:, -1], target, source)
        for dimension in dims:
            if not 1 <= dimension <= h_target.shape[-1]: raise ValueError("dimensions must be in [1, d_model]")
            directions = basis[:dimension].to(device)
            projected = (delta @ directions.T) @ directions
            # Sufficiency: add only the learned component to target. Necessity: remove it from full source state.
            for phase, replacement in (("projected_sufficiency", h_target + projected), ("projected_removal", h_source - projected)):
                delta_s, source_rate = metrics(patch(model, target_p, replacement), baseline_s, target, source)
                rows.append({"phase": phase, "model": run.name, "dimension": dimension, "subspace": "learned", "mean_delta_s": delta_s, "source_tens_argmax_fraction": source_rate})
            for control in range(a.random_subspaces):
                random_basis = orthogonal_random(h_target.shape[-1], dimension, a.seed + 10000 * control + dimension).to(device)
                random_projected = (delta @ random_basis.T) @ random_basis
                for phase, replacement in (("projected_sufficiency", h_target + random_projected), ("projected_removal", h_source - random_projected)):
                    delta_s, source_rate = metrics(patch(model, target_p, replacement), baseline_s, target, source)
                    rows.append({"phase": phase, "model": run.name, "dimension": dimension, "subspace": "random", "control": control, "mean_delta_s": delta_s, "source_tens_argmax_fraction": source_rate})
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = sorted({k for row in rows for k in row})
    with (a.output_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")


if __name__ == "__main__": main()
