"""Othello-style residual-stream analysis of the carry-dependency state."""
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--directions", type=Path, default=Path("artifacts/mechanistic_dependency/dependency_directions.pt"))
    p.add_argument("--examples", type=int, default=200)
    p.add_argument("--probe-examples", type=int, default=400)
    p.add_argument("--probe-epochs", type=int, default=100)
    p.add_argument("--epsilon", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/residual_dependency"))
    return p.parse_args()


def load(run: Run, device: str) -> torch.nn.Module:
    state = torch.load(run.checkpoint, map_location=device, weights_only=False)
    model = build_model(ModelConfig(**state["model_config"]), device)
    model.load_state_dict(state["model_state_dict"]); model.eval(); return model


def sites(width: int, prompts: torch.Tensor) -> list[tuple[str, int, str]]:
    equals = int((prompts[0] == EQUALS_ID).nonzero()[0])
    return [("blocks.0.hook_resid_post", equals, "l0_post_equals"),
            ("blocks.1.hook_resid_pre", equals, "l1_pre_equals"),
            ("blocks.1.hook_resid_post", equals, "l1_post_equals"),
            ("blocks.2.hook_resid_post", equals, "l2_post_equals"),
            ("blocks.3.hook_resid_post", -1, "l3_post_units")]


def score_from_logits(logits: torch.Tensor, target: torch.Tensor, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sd, td = ((source // 10) % 10).to(logits.device), ((target // 10) % 10).to(logits.device)
    return logits.gather(1, sd[:, None]).squeeze(1) - logits.gather(1, td[:, None]).squeeze(1), sd


@torch.no_grad()
def cached_site(model: torch.nn.Module, prompts: torch.Tensor, name: str, position: int) -> torch.Tensor:
    _, cache = model.run_with_cache(prompts, names_filter=lambda n: n == name)
    return cache[name][:, position].clone()


def fit_probe(x: torch.Tensor, y: torch.Tensor, train: torch.Tensor, test: torch.Tensor, epochs: int) -> float:
    probe = torch.nn.Linear(x.shape[1], 1); opt = torch.optim.AdamW(probe.parameters(), lr=.03, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(probe(x[train]).squeeze(-1), y[train]); loss.backward(); opt.step()
    with torch.no_grad(): return float(((probe(x[test]).squeeze(-1) >= 0) == y[test].bool()).float().mean())


def patch_logits(model: torch.nn.Module, prompts: torch.Tensor, name: str, position: int, replacement: torch.Tensor) -> torch.Tensor:
    def patch(value: torch.Tensor, hook: object) -> torch.Tensor:
        changed = value.clone(); changed[:, position] = replacement; return changed
    return model.run_with_hooks(prompts, fwd_hooks=[(name, patch)])[:, -1]


def transported_and_gradient(model: torch.nn.Module, prompts: torch.Tensor, target: torch.Tensor, source: torch.Tensor, direction: torch.Tensor, name: str, position: int, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Central-difference transport from l0 residual plus downstream score gradient."""
    source_name = "blocks.0.hook_resid_post"; equals = int((prompts[0] == EQUALS_ID).nonzero()[0])
    def endpoint(sign: float) -> torch.Tensor:
        captured: list[torch.Tensor] = []
        def inject(value: torch.Tensor, hook: object) -> torch.Tensor:
            changed = value.clone(); changed[:, equals] += sign * epsilon * direction; return changed
        def capture(value: torch.Tensor, hook: object) -> torch.Tensor:
            captured.append(value[:, position]); return value
        model.run_with_hooks(prompts, fwd_hooks=[(source_name, inject), (name, capture)])
        return captured[0].detach()
    transported = (endpoint(1) - endpoint(-1)) / (2 * epsilon)
    captured: list[torch.Tensor] = []
    def retain(value: torch.Tensor, hook: object) -> torch.Tensor:
        value.retain_grad(); captured.append(value); return value
    model.zero_grad(set_to_none=True)
    logits = model.run_with_hooks(prompts, fwd_hooks=[(name, retain)])[:, -1]
    s, _ = score_from_logits(logits, target, source); s.sum().backward()
    return transported, captured[0].grad[:, position].detach()


def main() -> None:
    a = parse_args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True)
    directions = torch.load(a.directions, map_location="cpu", weights_only=False); rows: list[dict[str, object]] = []
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device)
        target_p, source_p, target, source = paired_prompts(run.width, a.examples, a.seed, device)
        baseline_logits = model(target_p)[:, -1]; baseline_s, source_digit = score_from_logits(baseline_logits, target, source)
        direction = directions[run.name]["resid_post"].to(device)
        # Residual patches and local JVP/gradient alignment on the fixed pair set.
        for name, pos, label in sites(run.width, target_p):
            source_state = cached_site(model, source_p, name, pos)
            patched = patch_logits(model, target_p, name, pos, source_state)
            patched_s, _ = score_from_logits(patched, target, source)
            transported, gradient = transported_and_gradient(model, target_p, target, source, direction, name, pos, a.epsilon)
            cosine = torch.nn.functional.cosine_similarity(transported, gradient, dim=-1)
            rows.append({"phase": "residual_patch_transport", "model": run.name, "site": label,
                         "mean_delta_s": float((patched_s - baseline_s).mean().detach()),
                         "source_tens_argmax_fraction": float(patched.argmax(-1).eq(source_digit).float().mean()),
                         "mean_transported_norm": float(transported.norm(dim=-1).mean()),
                         "mean_gradient_alignment": float(cosine.mean())})
        # Fresh, pair/template-held-out probes at the same residual sites.
        pt, ps, _, _ = paired_prompts(run.width, a.probe_examples, a.seed + 100, device)
        both = torch.cat((pt, ps)); labels = torch.cat((torch.zeros(a.probe_examples), torch.ones(a.probe_examples)))
        pair_train = torch.arange(a.probe_examples) % 2 == 0; train = torch.cat((pair_train, pair_train)); test = ~train
        for name, pos, label in sites(run.width, pt):
            x = cached_site(model, both, name, pos).flatten(1).cpu()
            rows.append({"phase": "residual_probe", "model": run.name, "site": label,
                         "held_out_accuracy": fit_probe(x, labels, train, test, a.probe_epochs), "majority_baseline": .5})
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = sorted({k for r in rows for k in r})
    with (a.output_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")


if __name__ == "__main__": main()
