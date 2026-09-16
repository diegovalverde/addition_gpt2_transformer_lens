"""Probe carry bits from residual-stream activations before answer generation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from addition_gpt.data import AdditionBatchGenerator, carry_targets
from addition_gpt.model import ModelConfig, build_model
from train import resolve_device


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
    parser.add_argument("--train-examples", type=int, default=10_000)
    parser.add_argument("--test-examples", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/carry_probes"))
    return parser.parse_args()


def parse_runs(raw_runs: list[list[str]]) -> list[ProbeRun]:
    runs = []
    for name, checkpoint, width in raw_runs:
        parsed_width = int(width)
        if parsed_width < 1:
            raise ValueError("probe widths must be positive")
        runs.append(ProbeRun(name, Path(checkpoint), parsed_width))
    return runs


@torch.no_grad()
def collect_activations(
    model: torch.nn.Module,
    width: int,
    examples: int,
    batch_size: int,
    seed: int,
    device: str,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Cache residual activations at ``=`` from prompts that contain no answer tokens."""
    generator = AdditionBatchGenerator(width, seed)
    layers = model.cfg.n_layers
    activations = {layer: [] for layer in range(layers)}
    labels = []
    remaining = examples
    prompt_length = 2 * width + 3
    while remaining:
        count = min(batch_size, remaining)
        batch = generator.batch(count)
        prompts = batch.tokens[:, :prompt_length].to(device)
        _, cache = model.run_with_cache(
            prompts,
            names_filter=lambda name: name.endswith("hook_resid_post"),
        )
        for layer in range(layers):
            activations[layer].append(cache[f"blocks.{layer}.hook_resid_post"][:, -1].cpu())
        labels.append(carry_targets(batch.left, batch.right, width))
        remaining -= count
    return (
        {layer: torch.cat(values, dim=0) for layer, values in activations.items()},
        torch.cat(labels, dim=0),
    )


def fit_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    epochs: int,
    learning_rate: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fit independent linear classifiers for each carry bit and return accuracy."""
    probe = torch.nn.Linear(train_features.shape[1], train_labels.shape[1])
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(probe(train_features), train_labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predictions = probe(test_features).sigmoid() >= 0.5
        accuracy = (predictions == test_labels.bool()).float().mean(dim=0)
    return accuracy, {key: value.detach().cpu() for key, value in probe.state_dict().items()}


def load_model(run: ProbeRun, device: str) -> torch.nn.Module:
    loaded = torch.load(run.checkpoint, map_location=device, weights_only=False)
    model = build_model(ModelConfig(**loaded["model_config"]), device)
    model.load_state_dict(loaded["model_state_dict"])
    model.eval()
    return model


def write_results(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "carry_probe_results.json"
    csv_path = output_dir / "carry_probe_results.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("model", "width", "layer", "carry_column", "accuracy"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {json_path}")
    print(f"saved {csv_path}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    runs = parse_runs(args.model)
    results = []
    probe_weights: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for index, run in enumerate(runs):
        model = load_model(run, device)
        train_activations, train_labels = collect_activations(
            model, run.width, args.train_examples, args.batch_size, args.seed + 2 * index, device
        )
        test_activations, test_labels = collect_activations(
            model, run.width, args.test_examples, args.batch_size, args.seed + 2 * index + 1, device
        )
        for layer, train_features in train_activations.items():
            accuracy, weights = fit_probe(
                train_features,
                train_labels,
                test_activations[layer],
                test_labels,
                args.epochs,
                args.learning_rate,
            )
            probe_weights.setdefault(run.name, {})[layer] = weights
            for column, value in enumerate(accuracy.tolist(), start=1):
                results.append(
                    {
                        "model": run.name,
                        "width": run.width,
                        "layer": layer,
                        "carry_column": column,
                        "accuracy": value,
                    }
                )
        print(f"completed carry probes for {run.name}")
    write_results(results, args.output_dir)
    weights_path = args.output_dir / "carry_probe_weights.pt"
    torch.save(probe_weights, weights_path)
    print(f"saved {weights_path}")


if __name__ == "__main__":
    main()
