"""Test whether the learned residual direction composes across two carries."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from addition_gpt.data import EQUALS_ID, decode_answer, encode_addition
from addition_gpt.model import ModelConfig, build_model
from mechanistic_dependency import paired_prompts
from residual_causal_trace import state
from train import resolve_device

SITE = "blocks.1.hook_resid_post"

@dataclass(frozen=True)
class Run:
    name: str; checkpoint: Path; width: int

def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--train-pairs", type=int, default=400); p.add_argument("--examples", type=int, default=200); p.add_argument("--seed", type=int, default=900)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"); p.add_argument("--output-dir", type=Path, default=Path("artifacts/compositional_dependency")); return p.parse_args()

def load(run: Run, device: str) -> torch.nn.Module:
    saved = torch.load(run.checkpoint, map_location=device, weights_only=False); model = build_model(ModelConfig(**saved["model_config"]), device); model.load_state_dict(saved["model_state_dict"]); model.eval(); return model

def pairs(width: int, examples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Source's units carry triggers carries at both tens and hundreds."""
    if width < 3: raise ValueError("requires width >= 3")
    g = torch.Generator().manual_seed(seed)
    target_units_sum = torch.randint(9, (examples,), generator=g)
    target_left_units = torch.stack([torch.randint(int(x) + 1, (1,), generator=g)[0] for x in target_units_sum]); target_right_units = target_units_sum - target_left_units
    source_units_sum = target_units_sum + 10
    source_left_units = torch.stack([torch.randint(int(x) - 9, 10, (1,), generator=g)[0] for x in source_units_sum]); source_right_units = source_units_sum - source_left_units
    tens_left = torch.randint(10, (examples,), generator=g); tens_right = 9 - tens_left
    hundreds_left = torch.randint(10, (examples,), generator=g); hundreds_right = 9 - hundreds_left
    target_left = target_left_units + 10 * tens_left + 100 * hundreds_left; target_right = target_right_units + 10 * tens_right + 100 * hundreds_right
    source_left = source_left_units + 10 * tens_left + 100 * hundreds_left; source_right = source_right_units + 10 * tens_right + 100 * hundreds_right
    target = torch.tensor([encode_addition(int(l), int(r), width) for l, r in zip(target_left, target_right)])
    source = torch.tensor([encode_addition(int(l), int(r), width) for l, r in zip(source_left, source_right)])
    return target, source, target_left + target_right, source_left + source_right

@torch.no_grad()
def generate(model: torch.nn.Module, prompts: torch.Tensor, width: int, direction: torch.Tensor, scale: float, intervene_after: set[int]) -> torch.Tensor:
    """``intervene_after`` uses answer-digit indices: 0=units, 1=tens."""
    prompt_length = 2 * width + 3; generated = prompts
    for answer_index in range(width + 2):
        # At this forward pass, last input is the prior answer token.
        prior_answer_index = answer_index - 1
        if prior_answer_index in intervene_after:
            def edit(value: torch.Tensor, hook: object) -> torch.Tensor:
                changed = value.clone(); changed[:, -1] += scale * direction; return changed
            out = model.run_with_hooks(generated, fwd_hooks=[(SITE, edit)])
        else: out = model(generated)
        generated = torch.cat((generated, out[:, -1].argmax(-1, keepdim=True)), 1)
    return generated[:, prompt_length:prompt_length + width + 2]

def decode(rows: torch.Tensor, width: int) -> torch.Tensor:
    return torch.tensor([decode_answer(row, width) for row in rows])

def main() -> None:
    a = args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True); rows: list[dict[str, object]] = []
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device)
        # Learn and orient the position-general direction on isolated units-to-tens pairs.
        tr_t, tr_s, _, _ = paired_prompts(run.width, a.train_pairs, a.seed + 100, device)
        train_delta = state(model, tr_s, SITE, -1) - state(model, tr_t, SITE, -1); direction = torch.linalg.svd(train_delta, full_matrices=False).Vh[0]
        if float((train_delta @ direction).mean()) < 0: direction = -direction
        # Scale is held out from the composition examples.
        scale = float((train_delta @ direction).abs().mean())
        target, source, target_sum, source_sum = pairs(run.width, a.examples, a.seed)
        prompt_length = 2 * run.width + 3; prompts = target[:, :prompt_length].to(device)
        for label, positions in (("baseline", set()), ("units_only", {0}), ("tens_only", {1}), ("both", {0, 1})):
            answers = decode(generate(model, prompts, run.width, direction, scale, positions), run.width)
            rows.append({"model": run.name, "intervention": label, "target_exact_fraction": float(answers.eq(target_sum).float().mean()), "source_exact_fraction": float(answers.eq(source_sum).float().mean()), "mean_answer_shift": float((answers - target_sum).float().mean())})
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n"); fields = sorted({k for r in rows for k in r})
    with (a.output_dir / "results.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")

if __name__ == "__main__": main()
