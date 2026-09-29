"""Test the minimum depth for reliable three-digit addition at a fixed width."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


METRIC = re.compile(r"greedy_exact_answer_accuracy=(\d+(?:\.\d+)?)%")


def run(command: list[str]) -> str:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print(result.stdout, end="")
    return result.stdout


def accuracy(output: str) -> float:
    match = METRIC.search(output)
    if match is None:
        raise ValueError("evaluation output did not contain greedy exact-answer accuracy")
    return float(match.group(1)) / 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="1,2,3,4")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--examples", type=int, default=10_000)
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/layer-sweep"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = tuple(int(value) for value in args.layers.split(","))
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if not layers or min(layers) < 1 or not seeds or not 0 < args.threshold <= 1:
        raise ValueError("layers and seeds must be non-empty; threshold must be in (0, 1]")
    if args.d_model < 4 or args.d_model % 4:
        raise ValueError("--d-model must be a multiple of four")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[int, int, float, float]] = []
    for layer_count in layers:
        for seed in seeds:
            checkpoint_dir = args.output_dir / f"l{layer_count}-seed{seed}"
            run([
                sys.executable, "train.py", "--width", "3", "--d-model", str(args.d_model),
                "--n-layers", str(layer_count), "--steps", str(args.steps),
                "--batch-size", str(args.batch_size), "--seed", str(seed),
                "--checkpoint-every", str(args.steps), "--device", args.device,
                "--checkpoint-dir", str(checkpoint_dir),
            ])
            checkpoint = checkpoint_dir / (
                f"widths-3-d{args.d_model}-h4-l{layer_count}-seed-{seed}.pt"
            )
            iid = accuracy(run([
                sys.executable, "evaluate.py", "--checkpoint", str(checkpoint), "--width", "3",
                "--split", "iid", "--examples", str(args.examples), "--device", args.device,
            ]))
            carry = accuracy(run([
                sys.executable, "evaluate.py", "--checkpoint", str(checkpoint), "--width", "3",
                "--split", "carry", "--examples", str(args.examples), "--device", args.device,
            ]))
            rows.append((layer_count, seed, iid, carry))

    report = args.output_dir / "summary.csv"
    report.write_text(
        "n_layers,seed,iid_greedy_exact,carry_greedy_exact,passes\n" + "".join(
            f"{layer_count},{seed},{iid:.6f},{carry:.6f},"
            f"{iid >= args.threshold and carry >= args.threshold}\n"
            for layer_count, seed, iid, carry in rows
        )
    )
    passing = [layer_count for layer_count in layers if all(
        iid >= args.threshold and carry >= args.threshold
        for candidate, _, iid, carry in rows if candidate == layer_count
    )]
    print(f"wrote {report}")
    print(f"smallest_passing_n_layers={min(passing) if passing else 'none'}")


if __name__ == "__main__":
    main()
