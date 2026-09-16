"""Evaluate a saved addition model with teacher-forced answer metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from addition_gpt.data import AdditionBatchGenerator, EQUALS_ID, answer_target_mask
from addition_gpt.model import ModelConfig, build_model
from train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--split", choices=("iid", "carry", "range"), default="iid")
    parser.add_argument("--min-operand", type=int, default=0)
    parser.add_argument("--max-operand", type=int)
    parser.add_argument("--examples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


@torch.no_grad()
def greedy_answers(model: torch.nn.Module, tokens: torch.Tensor, width: int) -> torch.Tensor:
    """Generate answer digits from the prompt through ``=`` without answer tokens."""
    equals_positions = (tokens == EQUALS_ID).nonzero(as_tuple=False)
    if equals_positions.shape[0] != tokens.shape[0]:
        raise ValueError("each sequence must contain exactly one equals token")
    prefix_length = int(equals_positions[0, 1]) + 1
    if not bool((equals_positions[:, 1] == prefix_length - 1).all()):
        raise ValueError("all sequences must place equals at the same position")
    generated = tokens[:, :prefix_length]
    for _ in range(width + 2):
        next_token = model(generated)[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
    return generated[:, prefix_length:]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    loaded = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ModelConfig(**loaded["model_config"]), device)
    model.load_state_dict(loaded["model_state_dict"])
    model.eval()
    generator = AdditionBatchGenerator(args.width, args.seed)
    if generator.sequence_length > model.cfg.n_ctx:
        raise ValueError(
            f"width {args.width} needs {generator.sequence_length} tokens, "
            f"but this checkpoint has context {model.cfg.n_ctx}"
        )
    exact_correct = 0
    greedy_exact_correct = 0
    digit_correct = 0
    total_digits = 0
    remaining = args.examples
    while remaining:
        count = min(args.batch_size, remaining)
        batch = generator.batch(
            count,
            require_carry=args.split == "carry",
            min_operand=args.min_operand,
            max_operand=args.max_operand,
        )
        tokens = batch.tokens.to(device)
        predictions = model(tokens).argmax(dim=-1)[:, :-1]
        targets = tokens[:, 1:]
        mask = answer_target_mask(tokens)
        correct = predictions == targets
        digit_correct += int((correct & mask).sum())
        total_digits += int(mask.sum())
        exact_correct += int(((correct | ~mask).all(dim=1)).sum())
        generated = greedy_answers(model, tokens, args.width)
        expected = tokens[:, -generated.shape[1] :]
        greedy_exact_correct += int((generated == expected).all(dim=1).sum())
        remaining -= count
    print(
        f"split={args.split} examples={args.examples} width={args.width} "
        f"range=[{args.min_operand}, {args.max_operand or 10**args.width})"
    )
    print(f"teacher_forced_digit_accuracy={digit_correct / total_digits:.4%}")
    print(f"teacher_forced_exact_answer_accuracy={exact_correct / args.examples:.4%}")
    print(f"greedy_exact_answer_accuracy={greedy_exact_correct / args.examples:.4%}")


if __name__ == "__main__":
    main()
