"""Signed residual-direction interventions and local readout gain at layer 1."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from addition_gpt.model import ModelConfig, build_model
from mechanistic_dependency import paired_prompts
from residual_causal_trace import score, state
from train import resolve_device

SITE = "blocks.1.hook_resid_post"

@dataclass(frozen=True)
class Run:
    name: str; checkpoint: Path; width: int

def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--train-pairs", type=int, default=400); p.add_argument("--examples", type=int, default=200); p.add_argument("--amplitudes", default="-1,-0.5,0,0.5,1"); p.add_argument("--seed", type=int, default=300)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"); p.add_argument("--output-dir", type=Path, default=Path("artifacts/residual_readout_gain")); return p.parse_args()

def load(run: Run, device: str) -> torch.nn.Module:
    saved = torch.load(run.checkpoint, map_location=device, weights_only=False); model = build_model(ModelConfig(**saved["model_config"]), device); model.load_state_dict(saved["model_state_dict"]); model.eval(); return model

def logits(model: torch.nn.Module, prompts: torch.Tensor, direction: torch.Tensor | None = None, amount: float = 0.) -> torch.Tensor:
    if direction is None or amount == 0: return model(prompts)[:, -1]
    def edit(value: torch.Tensor, hook: object) -> torch.Tensor:
        changed = value.clone(); changed[:, -1] += amount * direction; return changed
    return model.run_with_hooks(prompts, fwd_hooks=[(SITE, edit)])[:, -1]

def gradient_gain(model: torch.nn.Module, prompts: torch.Tensor, target: torch.Tensor, source: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    captured: list[torch.Tensor] = []
    def retain(value: torch.Tensor, hook: object) -> torch.Tensor:
        value.retain_grad(); captured.append(value); return value
    model.zero_grad(set_to_none=True); out = model.run_with_hooks(prompts, fwd_hooks=[(SITE, retain)])[:, -1]; s, _ = score(out, target, source); s.sum().backward()
    return (captured[0].grad[:, -1] * direction).sum(-1).detach()

def main() -> None:
    a = args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True); rows: list[dict[str, object]] = []
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device)
        tr_t, tr_s, _, _ = paired_prompts(run.width, a.train_pairs, a.seed + 100, device)
        direction = torch.linalg.svd(state(model, tr_s, SITE, -1) - state(model, tr_t, SITE, -1), full_matrices=False).Vh[0]
        # Orient source-minus-target as positive; SVD itself has an arbitrary sign.
        if float(((state(model, tr_s, SITE, -1) - state(model, tr_t, SITE, -1)) @ direction).mean()) < 0: direction = -direction
        target_p, source_p, target, source = paired_prompts(run.width, a.examples, a.seed, device)
        delta = state(model, source_p, SITE, -1) - state(model, target_p, SITE, -1)
        scale = float((delta @ direction).abs().mean())
        baseline_s, _ = score(model(target_p)[:, -1], target, source)
        for amp in (float(x) for x in a.amplitudes.split(",")):
            out = logits(model, target_p, direction, amp * scale); changed, sd = score(out, target, source)
            rows.append({"phase": "signed_intervention", "model": run.name, "amplitude_in_matched_scales": amp, "mean_delta_s": float((changed - baseline_s).mean().detach()), "source_tens_argmax_fraction": float(out.argmax(-1).eq(sd).float().mean()), "scale": scale})
        gain = gradient_gain(model, target_p, target, source, direction)
        eps = .001 * scale
        fd = (score(logits(model, target_p, direction, eps), target, source)[0] - score(logits(model, target_p, direction, -eps), target, source)[0]) / (2 * eps)
        rows.append({"phase": "readout_gain", "model": run.name, "mean_gradient_gain": float(gain.mean()), "mean_finite_difference_gain": float(fd.mean().detach()), "jvp_fd_correlation": float(torch.corrcoef(torch.stack((gain.cpu(), fd.detach().cpu())))[0, 1]), "sign_agreement": float((gain.sign() == fd.sign()).float().mean())})
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n"); fields = sorted({k for r in rows for k in r})
    with (a.output_dir / "results.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")

if __name__ == "__main__": main()
