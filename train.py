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


def save_checkpoint(
    checkpoint: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    model_config: ModelConfig,
    args: argparse.Namespace,
    device: str,
    train_widths: tuple[int, ...],
    step: int,
    generators: dict[int, AdditionBatchGenerator],
) -> None:
    """Save a resumable training state and the information needed to reproduce it."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "generator_states": {
                str(width): generator.generator.get_state() for width, generator in generators.items()
            },
            "model_config": model_config.to_dict(),
            "training": vars(args) | {"device": device, "train_widths": train_widths, "step": step},
            "metadata": {
                "torch": torch.__version__,
                "python": platform.python_version(),
                "git_revision": git_revision(),
            },
        },
        checkpoint,
    )


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
    parser.add_argument("--exclude-units-carry-chain", action="store_true")
    parser.add_argument(
        "--units-carry-chain-exposure",
        type=float,
        default=1.0,
        help=(
            "Probability of retaining a sampled units-to-tens carry-chain example. "
            "Non-chain examples are always retained."
        ),
    )
    parser.add_argument("--drop-probability", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument(
        "--schedule-steps",
        type=int,
        help="Total steps for the learning-rate schedule; defaults to --steps.",
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    if (args.width is None) == (args.train_widths is None):
        parser.error("provide exactly one of --width or --train-widths")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    if args.schedule_steps is not None and args.schedule_steps < 1:
        parser.error("--schedule-steps must be positive")
    if not 0 <= args.units_carry_chain_exposure <= 1:
        parser.error("--units-carry-chain-exposure must be in [0, 1]")
    if args.exclude_units_carry_chain and args.units_carry_chain_exposure != 1.0:
        parser.error(
            "use either --exclude-units-carry-chain or --units-carry-chain-exposure, not both"
        )
    return args


def resolve_train_widths(args: argparse.Namespace) -> tuple[int, ...]:
    if args.width is not None:
        return (args.width,)
    assert args.train_widths is not None
    widths = tuple(int(value) for value in args.train_widths.split(","))
    if not widths or min(widths) < 1:
        raise ValueError("train widths must be positive integers")
    return widths


def restore_generator_states(
    checkpoint: dict[str, object],
    generators: dict[int, AdditionBatchGenerator],
    train_widths: tuple[int, ...],
    batch_size: int,
    completed_steps: int,
) -> None:
    """Restore generator state, or deterministically replay old batches for legacy checkpoints."""
    saved_states = checkpoint.get("generator_states")
    if isinstance(saved_states, dict):
        for width, generator in generators.items():
            state = saved_states.get(str(width))
            if not isinstance(state, torch.Tensor):
                raise ValueError(f"checkpoint has no generator state for width {width}")
            generator.generator.set_state(state)
        return
    for step in range(1, completed_steps + 1):
        width = train_widths[(step - 1) % len(train_widths)]
        generators[width].batch(batch_size)


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
    schedule_steps = args.schedule_steps or args.steps
    warmup_steps = min(500, schedule_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, schedule_steps - warmup_steps), eta_min=1e-4
            ),
        ],
        milestones=[warmup_steps],
    )
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    width_label = "-".join(str(width) for width in train_widths)
    completed_steps = 0
    if args.resume_from is not None:
        loaded = torch.load(args.resume_from, map_location=device, weights_only=False)
        if not isinstance(loaded, dict):
            raise ValueError("resume checkpoint is invalid")
        saved_config = loaded.get("model_config")
        if not isinstance(saved_config, dict):
            raise ValueError("resume checkpoint has no model configuration")
        saved_model_config = ModelConfig(**saved_config)
        if saved_model_config != model_config:
            raise ValueError("resume checkpoint model configuration does not match this run")
        saved_training = loaded.get("training")
        if not isinstance(saved_training, dict):
            raise ValueError("resume checkpoint has no training metadata")
        saved_widths = tuple(int(width) for width in saved_training["train_widths"])
        if saved_widths != train_widths:
            raise ValueError("resume checkpoint train widths do not match this run")
        completed_steps = int(saved_training["step"])
        if completed_steps >= args.steps:
            raise ValueError("--steps must be greater than the resumed step")
        model.load_state_dict(loaded["model_state_dict"])
        optimizer.load_state_dict(loaded["optimizer_state_dict"])
        scheduler.load_state_dict(loaded["scheduler_state_dict"])
        restore_generator_states(loaded, generators, train_widths, args.batch_size, completed_steps)
        print(f"resumed from {args.resume_from} at step {completed_steps}")

    model.train()
    for step in range(completed_steps + 1, args.steps + 1):
        width = train_widths[(step - 1) % len(train_widths)]
        batch = generators[width].batch(
            args.batch_size,
            min_operand=args.min_operand,
            max_operand=args.max_operand,
            exclude_units_carry_chain=args.exclude_units_carry_chain,
            units_carry_chain_exposure=args.units_carry_chain_exposure,
            drop_probability=args.drop_probability,
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
        if step % args.checkpoint_every == 0 and step != args.steps:
            checkpoint = args.checkpoint_dir / f"widths-{width_label}-seed-{args.seed}-step-{step}.pt"
            save_checkpoint(
                checkpoint,
                model,
                optimizer,
                scheduler,
                model_config,
                args,
                device,
                train_widths,
                step,
                generators,
            )
            print(f"saved intermediate checkpoint to {checkpoint}")

    checkpoint = args.checkpoint_dir / f"widths-{width_label}-seed-{args.seed}.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        model_config,
        args,
        device,
        train_widths,
        args.steps,
        generators,
    )
    print(f"saved checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
