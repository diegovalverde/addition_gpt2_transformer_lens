"""Causal geometry and head-subset tests for carry-dependency addition.

This is deliberately a *paired* experiment.  Each pair has identical tens and
higher digits, while its source has a units carry that makes raw tens sum nine.
Appending the common units-answer token means the final logit predicts the
clean downstream (tens) counterfactual.

Example (CPU, the six IID-selected checkpoints)::

  uv run python mechanistic_dependency.py \
    --model excluded-1 checkpoints/carry-dependency-exposure-0-seed-1/widths-3-seed-1-step-1000.pt 3 \
    --model control-1 checkpoints/carry-dependency-random-control-seed-1/widths-3-seed-1-step-500.pt 3 \
    --model excluded-2 checkpoints/carry-dependency-exposure-0-seed-2/widths-3-seed-2-step-1000.pt 3 \
    --model control-2 checkpoints/carry-dependency-random-control-seed-2/widths-3-seed-2-step-1500.pt 3 \
    --model excluded-3 checkpoints/carry-dependency-exposure-0-seed-3/widths-3-seed-3-step-1000.pt 3 \
    --model control-3 checkpoints/carry-dependency-random-control-seed-3/widths-3-seed-3-step-1500.pt 3

The report calls probe accuracy "decodability" and reports intervention and
finite-difference agreement separately; neither is silently promoted to a
circuit claim.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from addition_gpt.data import EQUALS_ID
from addition_gpt.model import ModelConfig, build_model
from patch_carries import matched_carry_dependency_pairs, matched_units_carry_pairs
from train import resolve_device


@dataclass(frozen=True)
class Run:
    name: str
    checkpoint: Path
    width: int


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--examples", type=int, default=200, help="Fixed paired evaluation set (seed 300).")
    p.add_argument("--probe-examples", type=int, default=1000, help="Pairs used for probe train/test.")
    p.add_argument("--probe-epochs", type=int, default=300)
    p.add_argument("--probe-lr", type=float, default=0.03)
    p.add_argument("--amplitudes", default="0,0.001,0.003,0.01,0.03")
    p.add_argument("--finite-difference", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/mechanistic_dependency"))
    p.add_argument("--skip-subsets", action="store_true")
    return p.parse_args()


def parse_runs(raw: list[list[str]]) -> list[Run]:
    return [Run(n, Path(c), int(w)) for n, c, w in raw]


def load(run: Run, device: str) -> torch.nn.Module:
    state = torch.load(run.checkpoint, map_location=device, weights_only=False)
    model = build_model(ModelConfig(**state["model_config"]), device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def hook_name(kind: str, layer: int = 0) -> str:
    return {
        "resid_pre": f"blocks.{layer}.hook_resid_pre",
        "attn_out": f"blocks.{layer}.hook_attn_out",
        "resid_post": f"blocks.{layer}.hook_resid_post",
        "z": f"blocks.{layer}.attn.hook_z",
    }[kind]


@torch.no_grad()
def features(model: torch.nn.Module, prompts: torch.Tensor, kind: str) -> torch.Tensor:
    name = hook_name(kind)
    _, cache = model.run_with_cache(prompts, names_filter=lambda n: n == name)
    value = cache[name][:, -1]
    return value.flatten(1).cpu()


def paired_prompts(width: int, pairs: int, seed: int, device: str, ordinary: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    maker = matched_units_carry_pairs if ordinary else matched_carry_dependency_pairs
    target, source, target_sum, source_sum = maker(width, pairs, seed)
    prompt_length = 2 * width + 3
    # The units answer is shared by construction. Teacher-force it to ask for tens.
    shared_units = target[:, prompt_length : prompt_length + 1]
    return (torch.cat((target[:, :prompt_length], shared_units), 1).to(device),
            torch.cat((source[:, :prompt_length], shared_units), 1).to(device),
            target_sum, source_sum)


def fit_direction(x: torch.Tensor, y: torch.Tensor, train: torch.Tensor, test: torch.Tensor, epochs: int, lr: float) -> tuple[torch.Tensor, float]:
    probe = torch.nn.Linear(x.shape[1], 1)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(probe(x[train]).squeeze(-1), y[train])
        loss.backward(); opt.step()
    with torch.no_grad():
        accuracy = float(((probe(x[test]).squeeze(-1) >= 0) == y[test].bool()).float().mean())
        direction = probe.weight.detach().squeeze(0)
    return direction / direction.norm(), accuracy


def edited_logits(model: torch.nn.Module, prompts: torch.Tensor, name: str | None = None, direction: torch.Tensor | None = None, amplitude: float = 0.0) -> torch.Tensor:
    if name is None or direction is None or amplitude == 0:
        return model(prompts)[:, -1]
    else:
        equals = int((prompts[0] == EQUALS_ID).nonzero()[0])
        def edit(value: torch.Tensor, hook: object) -> torch.Tensor:
            changed = value.clone()
            changed[:, equals] += amplitude * direction.reshape(value.shape[2:])
            return changed
        return model.run_with_hooks(prompts, fwd_hooks=[(name, edit)])[:, -1]


def score(model: torch.nn.Module, prompts: torch.Tensor, target_sums: torch.Tensor, source_sums: torch.Tensor, name: str | None = None, direction: torch.Tensor | None = None, amplitude: float = 0.0) -> torch.Tensor:
    # Digits are token IDs 0..9. Tens is floor(sum / 10) % 10.
    source_digit = ((source_sums // 10) % 10).to(prompts.device)
    target_digit = ((target_sums // 10) % 10).to(prompts.device)
    logits = edited_logits(model, prompts, name, direction, amplitude)
    return logits.gather(1, source_digit[:, None]).squeeze(1) - logits.gather(1, target_digit[:, None]).squeeze(1)


def mean_ci(values: torch.Tensor) -> tuple[float, float]:
    values = values.detach().cpu().float()
    return float(values.mean()), float(1.96 * values.std(unbiased=True) / values.numel() ** 0.5)


def gradient_jvp(model: torch.nn.Module, prompts: torch.Tensor, target: torch.Tensor, source: torch.Tensor, name: str, direction: torch.Tensor) -> torch.Tensor:
    """Scalar-output JVP as grad(score) dot direction, without a full Jacobian."""
    equals = int((prompts[0] == EQUALS_ID).nonzero()[0])
    captured: list[torch.Tensor] = []
    def retain(value: torch.Tensor, hook: object) -> torch.Tensor:
        value.retain_grad(); captured.append(value); return value
    model.zero_grad(set_to_none=True)
    source_digit = ((source // 10) % 10).to(prompts.device)
    target_digit = ((target // 10) % 10).to(prompts.device)
    logits = model.run_with_hooks(prompts, fwd_hooks=[(name, retain)])[:, -1]
    s = logits.gather(1, source_digit[:, None]).sum() - logits.gather(1, target_digit[:, None]).sum()
    s.backward()
    return (captured[0].grad[:, equals].flatten(1) * direction.flatten()).sum(1).detach()


def subset_rows(model: torch.nn.Module, prompts_t: torch.Tensor, prompts_s: torch.Tensor, target: torch.Tensor, source: torch.Tensor, width: int) -> list[dict[str, object]]:
    """All nonempty layer-0 z-head subsets, with S and source-tens argmax."""
    name = hook_name("z")
    equals = int((prompts_t[0] == EQUALS_ID).nonzero()[0])
    with torch.no_grad():
        _, cache = model.run_with_cache(prompts_s, names_filter=lambda n: n == name)
        source_z = cache[name][:, equals].clone()
        base = score(model, prompts_t, target, source)
    rows = []
    for size in range(1, 5):
        for heads in itertools.combinations(range(4), size):
            def patch(value: torch.Tensor, hook: object, selected=heads) -> torch.Tensor:
                result = value.clone(); result[:, equals, list(selected)] = source_z[:, list(selected)]; return result
            with torch.no_grad():
                logits = model.run_with_hooks(prompts_t, fwd_hooks=[(name, patch)])[:, -1]
                sd, td = ((source // 10) % 10).to(logits.device), ((target // 10) % 10).to(logits.device)
                s = logits.gather(1, sd[:, None]).squeeze(1) - logits.gather(1, td[:, None]).squeeze(1)
                rows.append({"heads": "+".join(map(str, heads)), "head_count": size,
                             "mean_delta_s": float((s - base).mean()),
                             "source_tens_argmax_fraction": float(logits.argmax(-1).eq(sd).float().mean())})
    return rows


def main() -> None:
    a = args(); device = resolve_device(a.device)
    if device != "cpu": print("WARNING: CPU is the reference device; this run is exploratory.")
    runs = parse_runs(a.model); output = a.output_dir; output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []; directions: dict[str, dict[str, torch.Tensor]] = {}
    for i, run in enumerate(runs):
        model = load(run, device)
        # Splitting by pair index keeps a higher-digit template and its two labels together.
        pt, ps, target, source = paired_prompts(run.width, a.probe_examples, a.seed + i * 10, device)
        train_pairs = torch.arange(a.probe_examples) % 2 == 0
        # stack target then source; labels 0/1, and duplicate each pair split.
        inputs = torch.cat((pt, ps)); labels = torch.cat((torch.zeros(a.probe_examples), torch.ones(a.probe_examples)))
        split = torch.cat((train_pairs, train_pairs)); test = ~split
        directions[run.name] = {}
        for kind in ("resid_pre", "attn_out", "resid_post", "z"):
            x = features(model, inputs, kind)
            dep, acc = fit_direction(x, labels, split, test, a.probe_epochs, a.probe_lr)
            shuffled, _ = fit_direction(x, labels[torch.randperm(labels.numel(), generator=torch.Generator().manual_seed(a.seed))], split, test, a.probe_epochs, a.probe_lr)
            directions[run.name][kind] = dep
            all_rows.append({"phase": "probe", "model": run.name, "hook": kind, "held_out_accuracy": acc, "majority_baseline": 0.5})
            # Small signed intervention, JVP, and its central finite-difference check.
            et, es, eval_target, eval_source = paired_prompts(run.width, a.examples, a.seed, device)
            base = score(model, et, eval_target, eval_source)
            ordinary_t, ordinary_s, _, _ = paired_prompts(run.width, a.probe_examples, a.seed + 1000 + i, device, ordinary=True)
            ordinary_x = features(model, torch.cat((ordinary_t, ordinary_s)), kind)
            ordinary, _ = fit_direction(ordinary_x, labels, split, test, a.probe_epochs, a.probe_lr)
            for label, vector in (("dependency", dep), ("shuffled", shuffled), ("ordinary_units_carry", ordinary), ("random", torch.randn_like(dep) / dep.numel() ** 0.5)):
                vector = vector / vector.norm(); name = hook_name(kind); v = vector.to(device)
                for amp in map(float, a.amplitudes.split(",")):
                    changed = score(model, et, eval_target, eval_source, name, v, amp)
                    delta = changed - base; mean, ci = mean_ci(delta)
                    logits = edited_logits(model, et, name, v, amp)
                    source_digit = ((eval_source // 10) % 10).to(device)
                    all_rows.append({"phase": "intervention", "model": run.name, "hook": kind, "direction": label, "amplitude": amp, "mean_delta_s": mean, "ci95_halfwidth": ci, "source_tens_argmax_fraction": float(logits.argmax(-1).eq(source_digit).float().mean())})
                if label == "dependency":
                    eps = a.finite_difference
                    jvp = gradient_jvp(model, et, eval_target, eval_source, name, v)
                    fd = (score(model, et, eval_target, eval_source, name, v, eps) - score(model, et, eval_target, eval_source, name, v, -eps)) / (2 * eps)
                    corr = float(torch.corrcoef(torch.stack((jvp.cpu(), fd.detach().cpu())))[0, 1])
                    rel = float(((jvp - fd).abs().mean() / (fd.abs().mean() + 1e-8)).detach())
                    all_rows.append({"phase": "jvp_validation", "model": run.name, "hook": kind, "epsilon": eps, "correlation": corr, "sign_agreement": float((jvp.sign() == fd.sign()).float().mean()), "relative_error": rel})
        if not a.skip_subsets:
            et, es, eval_target, eval_source = paired_prompts(run.width, a.examples, a.seed, device)
            for row in subset_rows(model, et, es, eval_target, eval_source, run.width):
                row.update(phase="head_subset_patch", model=run.name, hook="z"); all_rows.append(row)
        print(f"completed {run.name}")
    torch.save(directions, output / "dependency_directions.pt")
    (output / "results.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    fields = sorted({key for row in all_rows for key in row})
    with (output / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
    print(f"saved {output / 'results.json'}")


if __name__ == "__main__":
    main()
