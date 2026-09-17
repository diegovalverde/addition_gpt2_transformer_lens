"""Necessity, rescue, and finite-difference mediation tests for the layer-0 state.

This runner is intentionally downstream of ``mechanistic_dependency.py``: it
does not select heads by a single seed.  The candidate is the complete
distributed layer-0 attention write (all four heads), tested against randomly
chosen four-head writes elsewhere in the network.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from addition_gpt.data import EQUALS_ID
from addition_gpt.model import ModelConfig, build_model
from mechanistic_dependency import (
    edited_logits, gradient_jvp, hook_name, paired_prompts, score,
)
from train import resolve_device


@dataclass(frozen=True)
class Run:
    name: str
    checkpoint: Path
    width: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--directions", type=Path, default=Path("artifacts/mechanistic_dependency/dependency_directions.pt"))
    p.add_argument("--examples", type=int, default=200)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--random-controls", type=int, default=10)
    p.add_argument("--finite-difference", type=float, default=0.001)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/dependency_circuit"))
    return p.parse_args()


def load(run: Run, device: str) -> torch.nn.Module:
    saved = torch.load(run.checkpoint, map_location=device, weights_only=False)
    model = build_model(ModelConfig(**saved["model_config"]), device)
    model.load_state_dict(saved["model_state_dict"]); model.eval()
    return model


def digits(sums: torch.Tensor, device: str) -> torch.Tensor:
    return ((sums // 10) % 10).to(device)


def metrics(logits: torch.Tensor, target: torch.Tensor, source: torch.Tensor) -> dict[str, float]:
    sd, td = digits(source, str(logits.device)), digits(target, str(logits.device))
    s = logits.gather(1, sd[:, None]).squeeze(1) - logits.gather(1, td[:, None]).squeeze(1)
    return {"mean_s": float(s.mean().detach()), "source_tens_argmax_fraction": float(logits.argmax(-1).eq(sd).float().mean()), "target_tens_argmax_fraction": float(logits.argmax(-1).eq(td).float().mean())}


@torch.no_grad()
def cache_z(model: torch.nn.Module, prompts: torch.Tensor, layers: set[int]) -> dict[int, torch.Tensor]:
    names = {hook_name("z", layer) for layer in layers}
    _, cache = model.run_with_cache(prompts, names_filter=lambda n: n in names)
    return {layer: cache[hook_name("z", layer)].clone() for layer in layers}


@torch.no_grad()
def cache_attn_out(model: torch.nn.Module, prompts: torch.Tensor) -> torch.Tensor:
    name = hook_name("attn_out")
    _, cache = model.run_with_cache(prompts, names_filter=lambda n: n == name)
    return cache[name].clone()


def z_patch_logits(model: torch.nn.Module, prompts: torch.Tensor, replacements: dict[int, tuple[list[int], torch.Tensor]]) -> torch.Tensor:
    """Replace selected heads at equals; replacement tensors are [batch, head, d_head]."""
    equals = int((prompts[0] == EQUALS_ID).nonzero()[0])
    hooks = []
    for layer, (heads, values) in replacements.items():
        def patch(value: torch.Tensor, hook: object, selected=heads, replacement=values) -> torch.Tensor:
            changed = value.clone(); changed[:, equals, selected] = replacement; return changed
        hooks.append((hook_name("z", layer), patch))
    return model.run_with_hooks(prompts, fwd_hooks=hooks)[:, -1]


def z_ablate_then_restore_attn_out(
    model: torch.nn.Module, prompts: torch.Tensor, restored_attn_out: torch.Tensor
) -> torch.Tensor:
    """Zero all layer-0 head writes, then restore only their combined output."""
    equals = int((prompts[0] == EQUALS_ID).nonzero()[0])
    def zero_z(value: torch.Tensor, hook: object) -> torch.Tensor:
        changed = value.clone(); changed[:, equals] = 0; return changed
    def restore_out(value: torch.Tensor, hook: object) -> torch.Tensor:
        changed = value.clone(); changed[:, equals] = restored_attn_out[:, equals]; return changed
    return model.run_with_hooks(
        prompts,
        fwd_hooks=[(hook_name("z"), zero_z), (hook_name("attn_out"), restore_out)],
    )[:, -1]


def random_subsets(count: int, seed: int) -> list[list[tuple[int, int]]]:
    candidates = [(layer, head) for layer in range(1, 4) for head in range(4)]
    generator = torch.Generator().manual_seed(seed)
    return [[candidates[i] for i in torch.randperm(len(candidates), generator=generator)[:4].tolist()] for _ in range(count)]


def transport(model: torch.nn.Module, prompts: torch.Tensor, direction: torch.Tensor, epsilon: float) -> list[dict[str, float | str]]:
    """Per-context central differences from layer-0 attention output to later states."""
    source_name = hook_name("attn_out")
    equals = int((prompts[0] == EQUALS_ID).nonzero()[0])
    targets = [(hook_name("resid_post", 0), equals), (hook_name("resid_pre", 1), equals), (hook_name("resid_post", 1), equals), (hook_name("resid_post", 3), -1)]
    def captured(sign: float) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        def edit(value: torch.Tensor, hook: object) -> torch.Tensor:
            changed = value.clone(); changed[:, equals] += sign * epsilon * direction; return changed
        hooks: list[tuple[str, object]] = [(source_name, edit)]
        for name, pos in targets:
            def capture(value: torch.Tensor, hook: object, hook_name=name, position=pos) -> torch.Tensor:
                result[hook_name] = value[:, position].detach().clone(); return value
            hooks.append((name, capture))
        model.run_with_hooks(prompts, fwd_hooks=hooks)
        return result
    plus, minus = captured(1.0), captured(-1.0)
    rows = []
    for name, _ in targets:
        vector = (plus[name] - minus[name]) / (2 * epsilon)
        rows.append({"target_hook": name, "mean_transported_norm": float(vector.norm(dim=-1).mean())})
    return rows


def main() -> None:
    a = parse_args(); device = resolve_device(a.device)
    if device != "cpu": print("WARNING: CPU is the reference device.")
    directions = torch.load(a.directions, map_location="cpu", weights_only=False)
    output = a.output_dir; output.mkdir(parents=True, exist_ok=True); rows: list[dict[str, object]] = []
    for name, checkpoint, width_string in a.model:
        run = Run(name, Path(checkpoint), int(width_string)); model = load(run, device)
        target_p, source_p, target, source = paired_prompts(run.width, a.examples, a.seed, device)
        z_target = cache_z(model, target_p, {0, 1, 2, 3}); z_source = cache_z(model, source_p, {0})
        attn_target, attn_source = cache_attn_out(model, target_p), cache_attn_out(model, source_p)
        baseline = edited_logits(model, target_p)
        rows.append({"phase": "baseline", "model": run.name, **metrics(baseline, target, source)})
        equals = int((target_p[0] == EQUALS_ID).nonzero()[0])
        candidate_heads = [0, 1, 2, 3]
        # Mean/zero ablation and two complementary rescues.
        for intervention, replacement in (
            ("zero_ablate", torch.zeros_like(z_target[0][:, equals, candidate_heads])),
            ("mean_ablate", z_target[0][:, equals, candidate_heads].mean(0, keepdim=True).expand(a.examples, -1, -1)),
            ("clean_rescue", z_target[0][:, equals, candidate_heads]),
            ("matched_source_rescue", z_source[0][:, equals, candidate_heads]),
        ):
            logits = z_patch_logits(model, target_p, {0: (candidate_heads, replacement)})
            rows.append({"phase": "necessity_rescue", "model": run.name, "intervention": intervention, **metrics(logits, target, source)})
        # Proper mediation: the z write is absent, while only its downstream
        # combined attention output is restored at the same position.
        for intervention, restored in (
            ("zero_then_clean_attn_out_rescue", attn_target),
            ("zero_then_matched_source_attn_out_rescue", attn_source),
        ):
            logits = z_ablate_then_restore_attn_out(model, target_p, restored)
            rows.append({"phase": "mediation_rescue", "model": run.name, "intervention": intervention, **metrics(logits, target, source)})
        # Same-size controls: four unrelated heads, across layers 1--3.
        for number, subset in enumerate(random_subsets(a.random_controls, a.seed)):
            by_layer: dict[int, list[int]] = {}
            for layer, head in subset: by_layer.setdefault(layer, []).append(head)
            replacement = {layer: (heads, torch.zeros_like(z_target[layer][:, equals, heads])) for layer, heads in by_layer.items()}
            logits = z_patch_logits(model, target_p, replacement)
            rows.append({"phase": "random_zero_ablate", "model": run.name, "control": number, "heads": ";".join(f"{l}.{h}" for l, h in subset), **metrics(logits, target, source)})
        if run.name not in directions:
            raise KeyError(f"no dependency direction for {run.name}")
        direction = directions[run.name]["attn_out"].to(device)
        for row in transport(model, target_p, direction, a.finite_difference):
            rows.append({"phase": "transport", "model": run.name, "source_hook": hook_name("attn_out"), **row})
        # A scalar JVP confirms this exact source direction remains valid here.
        jvp = gradient_jvp(model, target_p, target, source, hook_name("attn_out"), direction)
        rows.append({"phase": "transport", "model": run.name, "source_hook": hook_name("attn_out"), "target_hook": "targeted_score", "mean_transported_norm": float(jvp.abs().mean())})
        print(f"completed {run.name}")
    (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = sorted({key for row in rows for key in row})
    with (output / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {output / 'results.json'}")


if __name__ == "__main__":
    main()
