"""Causal 2x2 residual interaction test for the raw-sum-nine carry write."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from addition_gpt.data import encode_addition
from addition_gpt.model import ModelConfig, build_model
from residual_causal_trace import score, state
from train import resolve_device

SITE = "blocks.1.hook_resid_post"

@dataclass(frozen=True)
class Run:
    name: str; checkpoint: Path; width: int

def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--examples", type=int, default=200); p.add_argument("--seed", type=int, default=1500)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"); p.add_argument("--output-dir", type=Path, default=Path("artifacts/conditional_write_interaction")); return p.parse_args()

def load(run: Run, device: str) -> torch.nn.Module:
    saved = torch.load(run.checkpoint, map_location=device, weights_only=False); model = build_model(ModelConfig(**saved["model_config"]), device); model.load_state_dict(saved["model_state_dict"]); model.eval(); return model

def factorial(width: int, examples: int, seed: int, device: str) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Matched conditions: carry c in {0,1}, raw tens r in {8,9}."""
    g = torch.Generator().manual_seed(seed)
    target_units_sum = torch.randint(9, (examples,), generator=g)
    left0 = torch.stack([torch.randint(int(x) + 1, (1,), generator=g)[0] for x in target_units_sum]); right0 = target_units_sum - left0
    source_sum = target_units_sum + 10; left1 = torch.stack([torch.randint(int(x) - 9, 10, (1,), generator=g)[0] for x in source_sum]); right1 = source_sum - left1
    high_left = torch.zeros(examples, dtype=torch.long); high_right = torch.zeros(examples, dtype=torch.long)
    for col in range(2, width):
        total = torch.randint(9, (examples,), generator=g); left = torch.stack([torch.randint(int(x) + 1, (1,), generator=g)[0] for x in total])
        high_left += 10**col * left; high_right += 10**col * (total - left)
    tens_left = torch.randint(9, (examples,), generator=g)
    templates: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for raw in (8, 9):
        tens_right = raw - tens_left
        for carry, left_u, right_u in ((0, left0, right0), (1, left1, right1)):
            left = left_u + 10 * tens_left + high_left; right = right_u + 10 * tens_right + high_right; sums = left + right
            tokens = torch.tensor([encode_addition(int(a), int(b), width) for a, b in zip(left, right)])
            prompt_length = 2 * width + 3
            # The units answer is common within each raw-tens pair.
            prompts = torch.cat((tokens[:, :prompt_length], tokens[:, prompt_length:prompt_length + 1]), 1).to(device)
            templates[f"c{carry}r{raw}"] = (prompts, left, right, sums)
    return templates

def patch_logits(model: torch.nn.Module, prompts: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
    def edit(value: torch.Tensor, hook: object) -> torch.Tensor:
        changed = value.clone(); changed[:, -1] = replacement; return changed
    return model.run_with_hooks(prompts, fwd_hooks=[(SITE, edit)])[:, -1]

def main() -> None:
    a = args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True); rows: list[dict[str, object]] = []
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device); data = factorial(run.width, a.examples, a.seed, device)
        h = {key: state(model, value[0], SITE, -1) for key, value in data.items()}
        # Difference-in-differences isolates the non-additive carry x raw-nine state.
        interaction = (h["c1r9"] - h["c0r9"]) - (h["c1r8"] - h["c0r8"])
        source_p, _, _, source_sum = data["c1r9"]; target_p, _, _, target_sum = data["c0r9"]
        baseline_s, _ = score(model(target_p)[:, -1], target_sum, source_sum)
        # Sufficiency at c0,r9; necessity by removing the interaction from c1,r9.
        for phase, prompts, replacement, reference_s in (
            ("interaction_sufficiency", target_p, h["c0r9"] + interaction, baseline_s),
            ("interaction_removal", source_p, h["c1r9"] - interaction, None),
        ):
            out = patch_logits(model, prompts, replacement); s, sd = score(out, target_sum, source_sum)
            if reference_s is None:
                reference_s, _ = score(model(source_p)[:, -1], target_sum, source_sum)
            rows.append({"model": run.name, "phase": phase, "mean_delta_s": float((s - reference_s).mean().detach()), "source_tens_argmax_fraction": float(out.argmax(-1).eq(sd).float().mean()), "mean_interaction_norm": float(interaction.norm(dim=-1).mean())})
        # A norm-matched random per-example direction is a causal control.
        random = torch.randn_like(interaction); random = random / random.norm(dim=-1, keepdim=True) * interaction.norm(dim=-1, keepdim=True)
        out = patch_logits(model, target_p, h["c0r9"] + random); s, sd = score(out, target_sum, source_sum)
        rows.append({"model": run.name, "phase": "random_interaction_control", "mean_delta_s": float((s - baseline_s).mean().detach()), "source_tens_argmax_fraction": float(out.argmax(-1).eq(sd).float().mean()), "mean_interaction_norm": float(interaction.norm(dim=-1).mean())})
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n"); fields = sorted({k for r in rows for k in r})
    with (a.output_dir / "results.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")

if __name__ == "__main__": main()
