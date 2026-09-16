"""Test whether a linear carry-probe direction causally changes generated sums."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from addition_gpt.data import AdditionBatchGenerator, EQUALS_ID, carry_targets, decode_answer
from addition_gpt.model import ModelConfig, build_model
from train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe-weights", type=Path, required=True)
    parser.add_argument("--probe-model", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--carry-column", type=int, required=True)
    parser.add_argument("--examples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--source-carry", choices=("0", "1"), default="0")
    parser.add_argument("--amplitudes", default="0,0.05,0.1,0.2,0.4")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/carry_interventions"))
    return parser.parse_args()


def collect_examples(
    width: int, column: int, source_carry: int, examples: int, batch_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select examples where changing one carry produces an in-range counterfactual sum."""
    generator = AdditionBatchGenerator(width, seed)
    selected_tokens = []
    selected_left = []
    selected_right = []
    shift = 10**column
    while sum(item.shape[0] for item in selected_tokens) < examples:
        batch = generator.batch(batch_size)
        targets = carry_targets(batch.left, batch.right, width)
        sums = batch.left + batch.right
        mask = targets[:, column - 1].eq(source_carry)
        if source_carry == 0:
            mask &= sums + shift < 10 ** (width + 1)
        else:
            mask &= sums >= shift
        selected_tokens.append(batch.tokens[mask])
        selected_left.append(batch.left[mask])
        selected_right.append(batch.right[mask])
    return (
        torch.cat(selected_tokens)[:examples],
        torch.cat(selected_left)[:examples],
        torch.cat(selected_right)[:examples],
    )


@torch.no_grad()
def residual_norm(model: torch.nn.Module, prompts: torch.Tensor, layer: int) -> float:
    _, cache = model.run_with_cache(prompts, names_filter=lambda name: name == f"blocks.{layer}.hook_resid_post")
    return float(cache[f"blocks.{layer}.hook_resid_post"][:, -1].norm(dim=-1).mean())


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompts: torch.Tensor,
    width: int,
    layer: int | None = None,
    direction: torch.Tensor | None = None,
    amplitude: float = 0.0,
) -> torch.Tensor:
    """Greedily generate a full fixed-width answer, optionally patching at ``=``."""
    equals_position = int((prompts[0] == EQUALS_ID).nonzero()[0])
    generated = prompts
    for _ in range(width + 2):
        if layer is None or direction is None or amplitude == 0:
            logits = model(generated)
        else:
            def patch(activation: torch.Tensor, hook: object) -> torch.Tensor:
                activation[:, equals_position] += amplitude * direction
                return activation

            logits = model.run_with_hooks(
                generated,
                fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch)],
            )
        generated = torch.cat((generated, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)
    return generated[:, -width - 2 :]


@torch.no_grad()
def patched_logits(
    model: torch.nn.Module,
    prompts: torch.Tensor,
    layer: int,
    direction: torch.Tensor,
    amplitude: float,
) -> torch.Tensor:
    """Return first-answer logits with a patch at the final prompt position."""
    equals_position = int((prompts[0] == EQUALS_ID).nonzero()[0])

    def patch(activation: torch.Tensor, hook: object) -> torch.Tensor:
        activation[:, equals_position] += amplitude * direction
        return activation

    return model.run_with_hooks(
        prompts,
        fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch)],
    )


def decode_answers(tokens: torch.Tensor, width: int) -> torch.Tensor:
    return torch.tensor([decode_answer(row, width) for row in tokens], dtype=torch.long)


def main() -> None:
    args = parse_args()
    if not 1 <= args.carry_column <= args.width:
        raise ValueError("carry column must be within the operand width")
    device = resolve_device(args.device)
    loaded = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ModelConfig(**loaded["model_config"]), device)
    model.load_state_dict(loaded["model_state_dict"])
    model.eval()
    weights = torch.load(args.probe_weights, map_location="cpu", weights_only=False)
    direction = weights[args.probe_model][args.layer]["weight"][args.carry_column - 1].to(device)
    direction = direction / direction.norm()
    tokens, left, right = collect_examples(
        args.width,
        args.carry_column,
        int(args.source_carry),
        args.examples,
        args.batch_size,
        args.seed,
    )
    prompt_length = 2 * args.width + 3
    prompts = tokens[:, :prompt_length].to(device)
    scale = residual_norm(model, prompts, args.layer)
    baseline = decode_answers(generate(model, prompts, args.width), args.width)
    baseline_logits = model(prompts)[:, -1].detach()
    expected = left + right + (10**args.carry_column if args.source_carry == "0" else -(10**args.carry_column))
    rows = []
    for relative_amplitude in (float(value) for value in args.amplitudes.split(",")):
        scaled_amplitude = relative_amplitude * scale
        answers = decode_answers(
            generate(
                model,
                prompts,
                args.width,
                args.layer,
                direction,
                scaled_amplitude,
            ),
            args.width,
        )
        rows.append(
            {
                "relative_amplitude": relative_amplitude,
                "answer_changed_fraction": float((answers != baseline).float().mean()),
                "counterfactual_exact_fraction": float((answers == expected).float().mean()),
                "mean_answer_shift": float((answers - baseline).float().mean()),
                "baseline_exact_fraction": float((baseline == left + right).float().mean()),
                "first_answer_logit_mean_abs_change": float(
                    (patched_logits(model, prompts, args.layer, direction, scaled_amplitude)[:, -1]
                    - baseline_logits).abs().mean()
                ),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "carry_intervention_results.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    with (args.output_dir / "carry_intervention_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"residual_norm={scale:.4f}")
    print(json.dumps(rows, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
