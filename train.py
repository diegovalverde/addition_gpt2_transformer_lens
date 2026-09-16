"""Train a four-layer transformer on reversed-digit addition."""

from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as functional

from addition_gpt.data import AdditionBatchGenerator, answer_target_mask
from addition_gpt.model import ModelConfig, build_model


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def answer_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Compute next-token cross entropy only on answer targets."""
    target = tokens[:, 1:]
    mask = answer_target_mask(tokens)
    token_loss = functional.cross_entropy(
        logits[:, :-1].transpose(1, 2), target, reduction="none"
    )
    return (token_loss * mask).sum() / mask.sum()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, help="Train on one fixed operand width.")
    parser.add_argument(
        "--train-widths",
        help="Comma-separated widths to cycle uniformly between, for example 1,2,3.",
    )
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-operand", type=int, default=0)
    parser.add_argument("--max-operand", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    if (args.width is None) == (args.train_widths is None):
        parser.error("provide exactly one of --width or --train-widths")
    return args


def resolve_train_widths(args: argparse.Namespace) -> tuple[int, ...]:
    if args.width is not None:
        return (args.width,)
    assert args.train_widths is not None
    widths = tuple(int(value) for value in args.train_widths.split(","))
    if not widths or min(widths) < 1:
        raise ValueError("train widths must be positive integers")
    return widths


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    train_widths = resolve_train_widths(args)
    generators = {
        width: AdditionBatchGenerator(width, args.seed + width) for width in train_widths
    }
    model_config = ModelConfig(seed=args.seed)
    maximum_sequence_length = max(generator.sequence_length for generator in generators.values())
    if maximum_sequence_length > model_config.n_ctx:
        raise ValueError(
            f"width {max(train_widths)} needs {maximum_sequence_length} tokens, "
            f"but the model context is {model_config.n_ctx}"
        )
    model = build_model(model_config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=min(500, args.steps)),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.steps - min(500, args.steps)), eta_min=1e-4
            ),
        ],
        milestones=[min(500, args.steps)],
    )

    model.train()
    for step in range(1, args.steps + 1):
        width = train_widths[(step - 1) % len(train_widths)]
        batch = generators[width].batch(
            args.batch_size,
            min_operand=args.min_operand,
            max_operand=args.max_operand,
        )
        tokens = batch.tokens.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = answer_loss(model(tokens), tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(
                f"step={step:>6} width={width} loss={loss.item():.5f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    width_label = "-".join(str(width) for width in train_widths)
    checkpoint = args.checkpoint_dir / f"widths-{width_label}-seed-{args.seed}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config.to_dict(),
            "training": vars(args) | {"device": device, "train_widths": train_widths},
            "metadata": {
                "torch": torch.__version__,
                "python": platform.python_version(),
                "git_revision": git_revision(),
            },
        },
        checkpoint,
    )
    print(f"saved checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
