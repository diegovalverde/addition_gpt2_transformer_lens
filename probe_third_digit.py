"""Probe third-column arithmetic targets from residual-stream activations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from addition_gpt.data import AdditionBatchGenerator
from probe_first_digit import DEFAULT_SITES, ProbeRun, fit_probe, load_model, parse_runs
from train import resolve_device


TARGET_CHOICES = ("third-digit", "carry-in", "carry-out")
CONTEXT_CHOICES = ("equals", "after-second-digit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("NAME", "CHECKPOINT", "WIDTH"),
        required=True,
        help="Repeat for every model to compare.",
    )
    parser.add_argument("--target", choices=TARGET_CHOICES, required=True)
    parser.add_argument(
        "--context",
        choices=CONTEXT_CHOICES,
        default="after-second-digit",
        help="Probe at `=` or after teacher-forcing the first two answer digits.",
    )
    parser.add_argument("--sites", default=",".join(DEFAULT_SITES))
    parser.add_argument("--train-examples", type=int, default=10_000)
    parser.add_argument("--test-examples", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/third_digit_probes"))
    return parser.parse_args()


def third_column_labels(left: torch.Tensor, right: torch.Tensor, target: str) -> torch.Tensor:
    """Return the hundreds digit, its carry-in, or its carry-out bit."""
    units_carry = left % 10 + right % 10 >= 10
    raw_tens_sum = (left // 10) % 10 + (right // 10) % 10
    carry_in = (raw_tens_sum + units_carry >= 10).long()
    raw_hundreds_sum = (left // 100) % 10 + (right // 100) % 10
    if target == "third-digit":
        return ((raw_hundreds_sum + carry_in) % 10).long()
    if target == "carry-in":
        return carry_in
    if target == "carry-out":
        return (raw_hundreds_sum + carry_in >= 10).long()
    raise ValueError(f"unknown target {target!r}")


def target_classes(target: str) -> int:
    return 10 if target == "third-digit" else 2


@torch.no_grad()
def collect_activations(
    model: torch.nn.Module,
    width: int,
    sites: tuple[str, ...],
    target: str,
    context: str,
    examples: int,
    batch_size: int,
    seed: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, float | None]:
    """Cache residuals at the state used to predict the third answer digit."""
    if width < 3:
        raise ValueError("third-column probing requires width at least three")
    generator = AdditionBatchGenerator(width, seed)
    activations = {site: [] for site in sites}
    labels = []
    model_correct = 0
    remaining = examples
    equals_length = 2 * width + 3
    prompt_length = equals_length + 2 * (context == "after-second-digit")
    while remaining:
        count = min(batch_size, remaining)
        batch = generator.batch(count)
        prompts = batch.tokens[:, :prompt_length].to(device)
        logits, cache = model.run_with_cache(
            prompts,
            names_filter=lambda name: name in sites,
        )
        labels.append(third_column_labels(batch.left, batch.right, target))
        if context == "after-second-digit":
            third_digit = third_column_labels(batch.left, batch.right, "third-digit")
            model_correct += int((logits[:, -1].argmax(dim=-1).cpu() == third_digit).sum())
        for site in sites:
            if site not in cache:
                raise ValueError(f"model cache did not contain requested site {site!r}")
            activations[site].append(cache[site][:, -1].cpu())
        remaining -= count
    model_accuracy = model_correct / examples if context == "after-second-digit" else None
    return (
        {site: torch.cat(values, dim=0) for site, values in activations.items()},
        torch.cat(labels, dim=0),
        model_accuracy,
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    runs: list[ProbeRun] = parse_runs(args.model)
    sites = tuple(site.strip() for site in args.sites.split(",") if site.strip())
    if not sites:
        raise ValueError("at least one probe site is required")
    results: list[dict[str, object]] = []
    probe_weights: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    boundary_checks: list[dict[str, object]] = []
    for run_index, run in enumerate(runs):
        model = load_model(run, device)
        train_activations, train_labels, _ = collect_activations(
            model, run.width, sites, args.target, args.context, args.train_examples,
            args.batch_size, args.seed, device,
        )
        test_activations, test_labels, model_accuracy = collect_activations(
            model, run.width, sites, args.target, args.context, args.test_examples,
            args.batch_size, args.seed + 1, device,
        )
        majority_accuracy = float(
            torch.bincount(test_labels, minlength=target_classes(args.target)).max()
            / test_labels.shape[0]
        )
        for site_index, site in enumerate(sites):
            train_accuracy, test_accuracy, weights = fit_probe(
                train_activations[site], train_labels, test_activations[site], test_labels,
                args.epochs, args.learning_rate, args.seed + 100 * run_index + site_index,
                target_classes(args.target),
            )
            probe_weights.setdefault(run.name, {})[site] = weights
            results.append(
                {
                    "model": run.name,
                    "width": run.width,
                    "target": args.target,
                    "context": args.context,
                    "site": site,
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "majority_accuracy": majority_accuracy,
                    "model_generated_third_digit_accuracy": model_accuracy,
                }
            )
        first = "blocks.0.hook_resid_post"
        second = "blocks.1.hook_resid_pre"
        if first in test_activations and second in test_activations:
            boundary_checks.append(
                {
                    "model": run.name,
                    "first_site": first,
                    "second_site": second,
                    "max_abs_difference": float(
                        (test_activations[first] - test_activations[second]).abs().max()
                    ),
                }
            )
        print(f"completed {args.context} {args.target} probes for {run.name}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.context}_{args.target.replace('-', '_')}_probe"
    json_path = args.output_dir / f"{stem}_results.json"
    csv_path = args.output_dir / f"{stem}_results.csv"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    (args.output_dir / f"{stem}_boundary_checks.json").write_text(
        json.dumps(boundary_checks, indent=2) + "\n"
    )
    torch.save(probe_weights, args.output_dir / f"{stem}_weights.pt")
    print(json.dumps(results, indent=2))
    print(json.dumps(boundary_checks, indent=2))
    print(f"saved {json_path}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
