"""Train and evaluate a width sweep for three-digit reversed addition.

Each width is evaluated with several initialization/data seeds. A width passes
only when every seed reaches the requested greedy exact-answer accuracy on both
IID and carry-heavy held-out examples. This avoids calling a model an adder
because it gets teacher-forced digits right while failing to decode an answer.
"""

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
    parser.add_argument("--dimensions", default="4,8,16,32,64,128")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--examples", type=int, default=10_000)
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dimension-sweep"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dimensions = tuple(int(value) for value in args.dimensions.split(","))
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if not dimensions or min(dimensions) < 4 or any(width % 4 for width in dimensions):
        raise ValueError("dimensions must be non-empty multiples of four")
    if not seeds or not 0 < args.threshold <= 1:
        raise ValueError("provide seeds and a threshold in (0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[int, int, float, float]] = []
    for dimension in dimensions:
        for seed in seeds:
            checkpoint_dir = args.output_dir / f"d{dimension}-seed{seed}"
            run([
                sys.executable, "train.py", "--width", "3", "--d-model", str(dimension),
                "--steps", str(args.steps), "--batch-size", str(args.batch_size),
                "--seed", str(seed), "--checkpoint-every", str(args.steps),
                "--device", args.device, "--checkpoint-dir", str(checkpoint_dir),
            ])
            checkpoint = checkpoint_dir / f"widths-3-d{dimension}-h4-seed-{seed}.pt"
            iid = accuracy(run([
                sys.executable, "evaluate.py", "--checkpoint", str(checkpoint), "--width", "3",
                "--split", "iid", "--examples", str(args.examples), "--device", args.device,
            ]))
            carry = accuracy(run([
                sys.executable, "evaluate.py", "--checkpoint", str(checkpoint), "--width", "3",
                "--split", "carry", "--examples", str(args.examples), "--device", args.device,
            ]))
            rows.append((dimension, seed, iid, carry))

    report = args.output_dir / "summary.csv"
    report.write_text(
        "d_model,seed,iid_greedy_exact,carry_greedy_exact,passes\n" + "".join(
            f"{dimension},{seed},{iid:.6f},{carry:.6f},{iid >= args.threshold and carry >= args.threshold}\n"
            for dimension, seed, iid, carry in rows
        )
    )
    passing = [dimension for dimension in dimensions if all(
        iid >= args.threshold and carry >= args.threshold
        for candidate, _, iid, carry in rows if candidate == dimension
    )]
    print(f"wrote {report}")
    print(f"smallest_passing_d_model={min(passing) if passing else 'none'}")


if __name__ == "__main__":
    main()
