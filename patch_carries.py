"""Activation-patch matched carry pairs and measure generated counterfactual sums."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from addition_gpt.data import (
    EQUALS_ID,
    decode_answer,
    encode_addition,
    has_carry_dependency_chain,
)
from addition_gpt.model import ModelConfig, build_model
from train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--examples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument(
        "--pair-type",
        choices=("units-carry", "carry-dependency"),
        default="units-carry",
        help="Matched intervention pair construction.",
    )
    parser.add_argument("--blends", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/carry_patching"))
    return parser.parse_args()


def matched_units_carry_pairs(
    width: int, examples: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build pairs with identical higher digits and output units digit, but carry 0 vs 1."""
    if width < 1:
        raise ValueError("width must be positive")
    generator = torch.Generator().manual_seed(seed)
    high_limit = 10 ** (width - 1)
    left_high = torch.randint(high_limit, (examples,), generator=generator) * 10
    right_high = torch.randint(high_limit, (examples,), generator=generator) * 10
    target_sum_units = torch.randint(9, (examples,), generator=generator)
    target_left_units = torch.stack(
        [torch.randint(int(total) + 1, (1,), generator=generator)[0] for total in target_sum_units]
    )
    target_right_units = target_sum_units - target_left_units
    source_sum_units = target_sum_units + 10
    source_left_units = torch.stack(
        [
            torch.randint(int(total) - 9, 10, (1,), generator=generator)[0]
            for total in source_sum_units
        ]
    )
    source_right_units = source_sum_units - source_left_units
    target_left = left_high + target_left_units
    target_right = right_high + target_right_units
    source_left = left_high + source_left_units
    source_right = right_high + source_right_units
    target_tokens = torch.tensor(
        [encode_addition(int(left), int(right), width) for left, right in zip(target_left, target_right)]
    )
    source_tokens = torch.tensor(
        [encode_addition(int(left), int(right), width) for left, right in zip(source_left, source_right)]
    )
    return target_tokens, source_tokens, target_left + target_right, source_left + source_right


def matched_carry_dependency_pairs(
    width: int, examples: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Toggle only a units carry that triggers a tens carry in the source.

    Both prompts have identical tens and higher input digits. Their raw tens
    digits sum to nine, so the source's added units carry is necessary for a
    tens carry. Higher raw columns are sampled to sum to at most eight, avoiding
    a second dependency and isolating the units-to-tens transition.
    """
    if width < 2:
        raise ValueError("carry-dependency pairs require width at least two")
    generator = torch.Generator().manual_seed(seed)
    target_units_sum = torch.randint(9, (examples,), generator=generator)
    target_left_units = torch.stack(
        [torch.randint(int(total) + 1, (1,), generator=generator)[0] for total in target_units_sum]
    )
    target_right_units = target_units_sum - target_left_units
    source_units_sum = target_units_sum + 10
    source_left_units = torch.stack(
        [
            torch.randint(int(total) - 9, 10, (1,), generator=generator)[0]
            for total in source_units_sum
        ]
    )
    source_right_units = source_units_sum - source_left_units

    tens_left = torch.randint(10, (examples,), generator=generator)
    tens_right = 9 - tens_left
    target_left = target_left_units + 10 * tens_left
    target_right = target_right_units + 10 * tens_right
    source_left = source_left_units + 10 * tens_left
    source_right = source_right_units + 10 * tens_right
    for column in range(2, width):
        place = 10**column
        raw_sum = torch.randint(9, (examples,), generator=generator)
        left_digit = torch.stack(
            [torch.randint(int(total) + 1, (1,), generator=generator)[0] for total in raw_sum]
        )
        right_digit = raw_sum - left_digit
        target_left += place * left_digit
        target_right += place * right_digit
        source_left += place * left_digit
        source_right += place * right_digit

    target_tokens = torch.tensor(
        [encode_addition(int(left), int(right), width) for left, right in zip(target_left, target_right)]
    )
    source_tokens = torch.tensor(
        [encode_addition(int(left), int(right), width) for left, right in zip(source_left, source_right)]
    )
    if bool(has_carry_dependency_chain(target_left, target_right, width).any()):
        raise AssertionError("target pairs must not contain a carry dependency")
    if not bool(has_carry_dependency_chain(source_left, source_right, width).all()):
        raise AssertionError("source pairs must contain the toggled carry dependency")
    return target_tokens, source_tokens, target_left + target_right, source_left + source_right


@torch.no_grad()
def equals_residual(model: torch.nn.Module, prompts: torch.Tensor, layer: int) -> torch.Tensor:
    name = f"blocks.{layer}.hook_resid_post"
    _, cache = model.run_with_cache(prompts, names_filter=lambda hook_name: hook_name == name)
    return cache[name][:, -1].clone()


@torch.no_grad()
def generate_patched(
    model: torch.nn.Module,
    target_prompts: torch.Tensor,
    target_residual: torch.Tensor,
    source_residual: torch.Tensor,
    width: int,
    layer: int,
    blend: float,
) -> torch.Tensor:
    """Greedily decode target prompts while replacing ``=`` residuals by a source blend."""
    equals_position = int((target_prompts[0] == EQUALS_ID).nonzero()[0])
    replacement = (1 - blend) * target_residual + blend * source_residual
    generated = target_prompts
    for _ in range(width + 2):
        def patch(activation: torch.Tensor, hook: object) -> torch.Tensor:
            activation[:, equals_position] = replacement
            return activation

        logits = model.run_with_hooks(
            generated,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch)],
        )
        generated = torch.cat((generated, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)
    return generated[:, -width - 2 :]


def decode_answers(tokens: torch.Tensor, width: int) -> torch.Tensor:
    return torch.tensor([decode_answer(row, width) for row in tokens], dtype=torch.long)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    loaded = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ModelConfig(**loaded["model_config"]), device)
    model.load_state_dict(loaded["model_state_dict"])
    model.eval()
    pair_builder = (
        matched_units_carry_pairs
        if args.pair_type == "units-carry"
        else matched_carry_dependency_pairs
    )
    target_tokens, source_tokens, target_sums, source_sums = pair_builder(
        args.width, args.examples, args.seed
    )
    prompt_length = 2 * args.width + 3
    target_prompts = target_tokens[:, :prompt_length].to(device)
    source_prompts = source_tokens[:, :prompt_length].to(device)
    target_residual = equals_residual(model, target_prompts, args.layer)
    source_residual = equals_residual(model, source_prompts, args.layer)
    rows = []
    for blend in (float(value) for value in args.blends.split(",")):
        answers = decode_answers(
            generate_patched(
                model,
                target_prompts,
                target_residual,
                source_residual,
                args.width,
                args.layer,
                blend,
            ),
            args.width,
        )
        rows.append(
            {
                "pair_type": args.pair_type,
                "blend": blend,
                "target_exact_fraction": float((answers == target_sums).float().mean()),
                "source_counterfactual_fraction": float((answers == source_sums).float().mean()),
                "answer_changed_fraction": float((answers != target_sums).float().mean()),
                "mean_answer_shift": float((answers - target_sums).float().mean()),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "matched_carry_patching_results.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    with (args.output_dir / "matched_carry_patching_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(f"saved {json_path}")


if __name__ == "__main__":
    main()
