"""Try to rescue the raw-nine dependency using a model's ordinary raw-carry state."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from addition_gpt.data import decode_answer, encode_addition
from addition_gpt.model import ModelConfig, build_model
from residual_causal_trace import state
from train import resolve_device

SITE = "blocks.1.hook_resid_post"

@dataclass(frozen=True)
class Run:
    name: str; checkpoint: Path; width: int

def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--examples", type=int, default=200); p.add_argument("--seed", type=int, default=2100)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"); p.add_argument("--output-dir", type=Path, default=Path("artifacts/raw_carry_branch_rescue")); return p.parse_args()

def load(run: Run, device: str) -> torch.nn.Module:
    saved = torch.load(run.checkpoint, map_location=device, weights_only=False); model = build_model(ModelConfig(**saved["model_config"]), device); model.load_state_dict(saved["model_state_dict"]); model.eval(); return model

def contexts(width: int, examples: int, seed: int, device: str) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Match c1/raw9 to ordinary c0/raw10; same units and high templates."""
    g = torch.Generator().manual_seed(seed)
    unit_sum = torch.randint(9, (examples,), generator=g); left0 = torch.stack([torch.randint(int(x) + 1, (1,), generator=g)[0] for x in unit_sum]); right0 = unit_sum - left0
    source_unit_sum = unit_sum + 10; left1 = torch.stack([torch.randint(int(x) - 9, 10, (1,), generator=g)[0] for x in source_unit_sum]); right1 = source_unit_sum - left1
    # Left is at least one so raw 10 has valid right digit; r9 and r10 differ by only one right digit.
    tens_left = torch.randint(1, 10, (examples,), generator=g)
    high_left = torch.zeros(examples, dtype=torch.long); high_right = torch.zeros(examples, dtype=torch.long)
    for col in range(2, width):
        total = torch.randint(9, (examples,), generator=g); left = torch.stack([torch.randint(int(x) + 1, (1,), generator=g)[0] for x in total]); high_left += 10**col * left; high_right += 10**col * (total - left)
    result = {}
    for name, left_u, right_u, raw in (("c0r9", left0, right0, 9), ("c1r9", left1, right1, 9), ("c0r10", left0, right0, 10)):
        left = left_u + 10 * tens_left + high_left; right = right_u + 10 * (raw - tens_left) + high_right; sums = left + right
        tokens = torch.tensor([encode_addition(int(a), int(b), width) for a, b in zip(left, right)]); prompt_length = 2 * width + 3
        result[name] = (torch.cat((tokens[:, :prompt_length], tokens[:, prompt_length:prompt_length + 1]), 1).to(device), sums, tokens)
    return result

@torch.no_grad()
def generate(model: torch.nn.Module, prompts: torch.Tensor, width: int, correction: torch.Tensor | None) -> torch.Tensor:
    generated = prompts; prompt_length = 2 * width + 3
    for step in range(width + 1):  # units already teacher-forced; generate tens, hundreds, overflow, EOS
        if step == 0 and correction is not None:
            def edit(value: torch.Tensor, hook: object) -> torch.Tensor:
                changed = value.clone(); changed[:, -1] += correction; return changed
            logits = model.run_with_hooks(generated, fwd_hooks=[(SITE, edit)])
        else: logits = model(generated)
        generated = torch.cat((generated, logits[:, -1].argmax(-1, keepdim=True)), 1)
    return generated[:, prompt_length:prompt_length + width + 2]

def decode(rows: torch.Tensor, width: int) -> torch.Tensor: return torch.tensor([decode_answer(row, width) for row in rows])

def main() -> None:
    a = args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True); rows: list[dict[str, object]] = []
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device); data = contexts(run.width, a.examples, a.seed, device)
        h9, h10 = (state(model, data[key][0], SITE, -1) for key in ("c0r9", "c0r10"))
        correction = h10 - h9
        random = torch.randn_like(correction); random = random / random.norm(dim=-1, keepdim=True) * correction.norm(dim=-1, keepdim=True)
        prompts, source_sum, _ = data["c1r9"]
        for label, vector in (("baseline", None), ("raw_carry_branch", correction), ("random_norm_matched", random)):
            answers = decode(generate(model, prompts, run.width, vector), run.width)
            rows.append({"model": run.name, "intervention": label, "source_exact_fraction": float(answers.eq(source_sum).float().mean()), "mean_answer_error": float((answers - source_sum).float().mean()), "mean_correction_norm": float(correction.norm(dim=-1).mean())})
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n"); fields = sorted({k for r in rows for k in r})
    with (a.output_dir / "results.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")

if __name__ == "__main__": main()
