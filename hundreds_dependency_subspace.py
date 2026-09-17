"""Held-out 1D residual test for isolated tens-to-hundreds dependencies."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from addition_gpt.data import encode_addition
from addition_gpt.model import ModelConfig, build_model
from residual_causal_trace import state
from train import resolve_device

SITE = "blocks.1.hook_resid_post"

@dataclass(frozen=True)
class Run:
    name: str; checkpoint: Path; width: int

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--train-pairs", type=int, default=400); p.add_argument("--examples", type=int, default=200)
    p.add_argument("--random-subspaces", type=int, default=10); p.add_argument("--seed", type=int, default=700)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/hundreds_dependency_subspace")); return p.parse_args()

def load(run: Run, device: str) -> torch.nn.Module:
    saved = torch.load(run.checkpoint, map_location=device, weights_only=False)
    model = build_model(ModelConfig(**saved["model_config"]), device); model.load_state_dict(saved["model_state_dict"]); model.eval(); return model

def pairs(width: int, examples: int, seed: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Toggle only a tens carry that causes a hundreds carry (raw sum == 9)."""
    if width < 3: raise ValueError("tens-to-hundreds pairs require width >= 3")
    g = torch.Generator().manual_seed(seed)
    # Units are identical and cannot carry.
    units_sum = torch.randint(9, (examples,), generator=g)
    units_left = torch.stack([torch.randint(int(total) + 1, (1,), generator=g)[0] for total in units_sum])
    units_right = units_sum - units_left
    # Tens answer digit is shared; only the source carries into hundreds.
    target_tens_sum = torch.randint(9, (examples,), generator=g)
    target_tens_left = torch.stack([torch.randint(int(total) + 1, (1,), generator=g)[0] for total in target_tens_sum])
    target_tens_right = target_tens_sum - target_tens_left
    source_tens_sum = target_tens_sum + 10
    source_tens_left = torch.stack([torch.randint(int(total) - 9, 10, (1,), generator=g)[0] for total in source_tens_sum])
    source_tens_right = source_tens_sum - source_tens_left
    hundreds_left = torch.randint(10, (examples,), generator=g); hundreds_right = 9 - hundreds_left
    target_left = units_left + 10 * target_tens_left + 100 * hundreds_left
    target_right = units_right + 10 * target_tens_right + 100 * hundreds_right
    source_left = units_left + 10 * source_tens_left + 100 * hundreds_left
    source_right = units_right + 10 * source_tens_right + 100 * hundreds_right
    for column in range(3, width):
        raw = torch.randint(9, (examples,), generator=g); left = torch.stack([torch.randint(int(total) + 1, (1,), generator=g)[0] for total in raw])
        target_left += 10**column * left; target_right += 10**column * (raw - left)
        source_left += 10**column * left; source_right += 10**column * (raw - left)
    target = torch.tensor([encode_addition(int(l), int(r), width) for l, r in zip(target_left, target_right)])
    source = torch.tensor([encode_addition(int(l), int(r), width) for l, r in zip(source_left, source_right)])
    prompt_length = 2 * width + 3
    # Units and tens answers are identical; next position predicts hundreds.
    return (torch.cat((target[:, :prompt_length], target[:, prompt_length:prompt_length + 2]), 1).to(device),
            torch.cat((source[:, :prompt_length], source[:, prompt_length:prompt_length + 2]), 1).to(device),
            target_left + target_right, source_left + source_right)

def score(logits: torch.Tensor, target: torch.Tensor, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sd, td = ((source // 100) % 10).to(logits.device), ((target // 100) % 10).to(logits.device)
    return logits.gather(1, sd[:, None]).squeeze(1) - logits.gather(1, td[:, None]).squeeze(1), sd

def patch(model: torch.nn.Module, prompts: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
    def edit(value: torch.Tensor, hook: object) -> torch.Tensor:
        changed = value.clone(); changed[:, -1] = replacement; return changed
    return model.run_with_hooks(prompts, fwd_hooks=[(SITE, edit)])[:, -1]

def random_basis(d: int, seed: int) -> torch.Tensor:
    return torch.linalg.qr(torch.randn((d, 1), generator=torch.Generator().manual_seed(seed)), mode="reduced").Q.T

def main() -> None:
    a = parse_args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True); rows: list[dict[str, object]] = []
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device)
        tr_t, tr_s, _, _ = pairs(run.width, a.train_pairs, a.seed + 100, device)
        direction = torch.linalg.svd(state(model, tr_s, SITE, -1) - state(model, tr_t, SITE, -1), full_matrices=False).Vh[:1]
        te_t, te_s, target, source = pairs(run.width, a.examples, a.seed, device)
        h_t, h_s = state(model, te_t, SITE, -1), state(model, te_s, SITE, -1); delta = h_s - h_t
        baseline_s, _ = score(model(te_t)[:, -1], target, source)
        for kind, basis in [("learned", direction), *[("random", random_basis(h_t.shape[-1], a.seed + i)) for i in range(a.random_subspaces)]]:
            projected = (delta @ basis.T) @ basis
            for phase, replacement in (("projected_sufficiency", h_t + projected), ("projected_removal", h_s - projected)):
                logits = patch(model, te_t, replacement); s, sd = score(logits, target, source)
                rows.append({"phase": phase, "model": run.name, "subspace": kind, "mean_delta_s": float((s - baseline_s).mean().detach()), "source_hundreds_argmax_fraction": float(logits.argmax(-1).eq(sd).float().mean())})
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = sorted({k for r in rows for k in r})
    with (a.output_dir / "results.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")

if __name__ == "__main__": main()
