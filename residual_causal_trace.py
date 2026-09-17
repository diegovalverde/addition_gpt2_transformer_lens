"""Residual causal tracing from layer-0 ``=`` to the units-token readout."""
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
from train import resolve_device


@dataclass(frozen=True)
class Run:
    name: str; checkpoint: Path; width: int


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--examples", type=int, default=200); p.add_argument("--seed", type=int, default=300)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/residual_causal_trace")); return p.parse_args()


def load(run: Run, device: str) -> torch.nn.Module:
    state = torch.load(run.checkpoint, map_location=device, weights_only=False)
    model = build_model(ModelConfig(**state["model_config"]), device)
    model.load_state_dict(state["model_state_dict"]); model.eval(); return model


def score(logits: torch.Tensor, target: torch.Tensor, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    source_digit, target_digit = ((source // 10) % 10).to(logits.device), ((target // 10) % 10).to(logits.device)
    s = logits.gather(1, source_digit[:, None]).squeeze(1) - logits.gather(1, target_digit[:, None]).squeeze(1)
    return s, source_digit


@torch.no_grad()
def state(model: torch.nn.Module, prompts: torch.Tensor, name: str, position: int) -> torch.Tensor:
    _, cache = model.run_with_cache(prompts, names_filter=lambda n: n == name)
    return cache[name][:, position].clone()


def units_sites() -> list[tuple[str, str]]:
    return [("blocks.0.hook_resid_pre", "l0_pre_units"), ("blocks.0.hook_resid_post", "l0_post_units"),
            ("blocks.1.hook_resid_pre", "l1_pre_units"), ("blocks.1.hook_resid_post", "l1_post_units"),
            ("blocks.2.hook_resid_pre", "l2_pre_units"), ("blocks.2.hook_resid_post", "l2_post_units"),
            ("blocks.3.hook_resid_pre", "l3_pre_units"), ("blocks.3.hook_resid_post", "l3_post_units")]


def logits_with_patches(model: torch.nn.Module, prompts: torch.Tensor, patches: list[tuple[str, int, torch.Tensor]]) -> torch.Tensor:
    """Apply any number of position-specific residual replacements safely."""
    grouped: dict[str, list[tuple[int, torch.Tensor]]] = {}
    for name, position, replacement in patches: grouped.setdefault(name, []).append((position, replacement))
    hooks = []
    for name, locations in grouped.items():
        def patch(value: torch.Tensor, hook: object, edits=locations) -> torch.Tensor:
            changed = value.clone()
            for position, replacement in edits: changed[:, position] = replacement
            return changed
        hooks.append((name, patch))
    return model.run_with_hooks(prompts, fwd_hooks=hooks)[:, -1]


def row(phase: str, model: str, site: str, logits: torch.Tensor, baseline_s: torch.Tensor, target: torch.Tensor, source: torch.Tensor) -> dict[str, object]:
    s, source_digit = score(logits, target, source)
    return {"phase": phase, "model": model, "site": site,
            "mean_delta_s": float((s - baseline_s).mean().detach()),
            "source_tens_argmax_fraction": float(logits.argmax(-1).eq(source_digit).float().mean())}


def main() -> None:
    a = args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True); rows: list[dict[str, object]] = []
    equals_name = "blocks.0.hook_resid_post"
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device)
        target_p, source_p, target, source = paired_prompts(run.width, a.examples, a.seed, device)
        equals = int((target_p[0] == EQUALS_ID).nonzero()[0]); units = -1
        baseline = model(target_p)[:, -1]; baseline_s, _ = score(baseline, target, source)
        source_equals = state(model, source_p, equals_name, equals)
        l0_patch_logits = logits_with_patches(model, target_p, [(equals_name, equals, source_equals)])
        l0_patch_s, _ = score(l0_patch_logits, target, source)
        rows.append(row("l0_equals_source_patch", run.name, "l0_post_equals", l0_patch_logits, baseline_s, target, source))
        for name, label in units_sites():
            source_units = state(model, source_p, name, units)
            target_units = state(model, target_p, name, units)
            # Is a source state at this units position sufficient on its own?
            suff = logits_with_patches(model, target_p, [(name, units, source_units)])
            rows.append(row("units_source_patch", run.name, label, suff, baseline_s, target, source))
            # Does restoring the clean units residual block the known l0 effect?
            mediated = logits_with_patches(model, target_p, [(equals_name, equals, source_equals), (name, units, target_units)])
            rows.append(row("l0_patch_clean_units_restore", run.name, label, mediated, l0_patch_s, target, source))
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = sorted({k for r in rows for k in r})
    with (a.output_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")


if __name__ == "__main__": main()
