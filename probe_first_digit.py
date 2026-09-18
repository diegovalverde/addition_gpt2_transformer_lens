"""Probe first-column arithmetic targets from residual-stream activations."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from addition_gpt.data import AdditionBatchGenerator
from addition_gpt.model import ModelConfig, build_model
from train import resolve_device


DEFAULT_SITES = (
    "blocks.0.hook_resid_pre",
    "blocks.0.hook_resid_post",
    "blocks.1.hook_resid_pre",
    "blocks.1.hook_resid_post",
    "blocks.2.hook_resid_post",
    "blocks.3.hook_resid_post",
)
TARGET_CHOICES = ("first-digit", "units-carry-out")


@dataclass(frozen=True)
class ProbeRun:
    name: str
    checkpoint: Path
    width: int


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
    parser.add_argument("--sites", default=",".join(DEFAULT_SITES))
    parser.add_argument(
        "--target",
        choices=TARGET_CHOICES,
        default="first-digit",
        help="First-column arithmetic target to decode.",
    )
    parser.add_argument("--train-examples", type=int, default=10_000)
    parser.add_argument("--test-examples", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/first_digit_probes"))
    return parser.parse_args()


def parse_runs(raw_runs: list[list[str]]) -> list[ProbeRun]:
    runs = []
    for name, checkpoint, width in raw_runs:
        parsed_width = int(width)
        if parsed_width < 1:
            raise ValueError("probe widths must be positive")
        runs.append(ProbeRun(name, Path(checkpoint), parsed_width))
    return runs


def target_labels(left: torch.Tensor, right: torch.Tensor, target: str) -> torch.Tensor:
    """Return the requested first-column arithmetic target."""
    if target == "first-digit":
        return ((left + right) % 10).long()
    if target == "units-carry-out":
        return ((left % 10 + right % 10) >= 10).long()
    raise ValueError(f"unknown target {target!r}")


def target_classes(target: str) -> int:
    if target == "first-digit":
        return 10
    if target == "units-carry-out":
        return 2
    raise ValueError(f"unknown target {target!r}")


@torch.no_grad()
def collect_activations(
    model: torch.nn.Module,
    width: int,
    sites: tuple[str, ...],
    target: str,
    examples: int,
    batch_size: int,
    seed: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, float]:
    """Cache selected residuals at ``=`` and label one first-column target."""
    generator = AdditionBatchGenerator(width, seed)
    activations = {site: [] for site in sites}
    labels = []
    model_correct = 0
    remaining = examples
    prompt_length = 2 * width + 3
    while remaining:
        count = min(batch_size, remaining)
        batch = generator.batch(count)
        prompts = batch.tokens[:, :prompt_length].to(device)
        logits, cache = model.run_with_cache(
            prompts,
            names_filter=lambda name: name in sites,
        )
        label = target_labels(batch.left, batch.right, target)
        labels.append(label)
        first_digit = ((batch.left + batch.right) % 10).long()
        model_correct += int((logits[:, -1].argmax(dim=-1).cpu() == first_digit).sum())
        for site in sites:
            if site not in cache:
                raise ValueError(f"model cache did not contain requested site {site!r}")
            activations[site].append(cache[site][:, -1].cpu())
        remaining -= count
    return (
        {site: torch.cat(values, dim=0) for site, values in activations.items()},
        torch.cat(labels, dim=0),
        model_correct / examples,
    )


def fit_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    epochs: int,
    learning_rate: float,
    seed: int,
    classes: int,
) -> tuple[float, float, dict[str, torch.Tensor]]:
    """Fit a linear classifier and return train/test accuracy."""
    torch.manual_seed(seed)
    probe = torch.nn.Linear(train_features.shape[1], classes)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(probe(train_features), train_labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_accuracy = float((probe(train_features).argmax(dim=-1) == train_labels).float().mean())
        test_accuracy = float((probe(test_features).argmax(dim=-1) == test_labels).float().mean())
    weights = {key: value.detach().cpu() for key, value in probe.state_dict().items()}
    return train_accuracy, test_accuracy, weights


def load_model(run: ProbeRun, device: str) -> torch.nn.Module:
    loaded = torch.load(run.checkpoint, map_location=device, weights_only=False)
    model = build_model(ModelConfig(**loaded["model_config"]), device)
    model.load_state_dict(loaded["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    runs = parse_runs(args.model)
    sites = tuple(site.strip() for site in args.sites.split(",") if site.strip())
    if not sites:
        raise ValueError("at least one probe site is required")
    results: list[dict[str, object]] = []
    probe_weights: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    boundary_checks: list[dict[str, object]] = []
    for run_index, run in enumerate(runs):
        model = load_model(run, device)
        train_activations, train_labels, _ = collect_activations(
            model,
            run.width,
            sites,
            args.target,
            args.train_examples,
            args.batch_size,
            args.seed,
            device,
        )
        test_activations, test_labels, model_accuracy = collect_activations(
            model,
            run.width,
            sites,
            args.target,
            args.test_examples,
            args.batch_size,
            args.seed + 1,
            device,
        )
        majority_accuracy = float(
            torch.bincount(test_labels, minlength=target_classes(args.target)).max()
            / test_labels.shape[0]
        )
        for site_index, site in enumerate(sites):
            train_accuracy, test_accuracy, weights = fit_probe(
                train_activations[site],
                train_labels,
                test_activations[site],
                test_labels,
                args.epochs,
                args.learning_rate,
                args.seed + 100 * run_index + site_index,
                target_classes(args.target),
            )
            probe_weights.setdefault(run.name, {})[site] = weights
            results.append(
                {
                    "model": run.name,
                    "width": run.width,
                    "target": args.target,
                    "site": site,
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "majority_accuracy": majority_accuracy,
                    "model_generated_first_digit_accuracy": model_accuracy,
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
        print(f"completed {args.target} probes for {run.name}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "first_digit_probe_results.json"
    csv_path = args.output_dir / "first_digit_probe_results.csv"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    (args.output_dir / "boundary_checks.json").write_text(
        json.dumps(boundary_checks, indent=2) + "\n"
    )
    torch.save(probe_weights, args.output_dir / "first_digit_probe_weights.pt")
    print(json.dumps(results, indent=2))
    print(json.dumps(boundary_checks, indent=2))
    print(f"saved {json_path}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
