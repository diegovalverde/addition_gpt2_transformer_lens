"""Causal signed-gain profile across units-token residual sites."""
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

SITES = [("blocks.1.hook_resid_post", "l1_post_units"), ("blocks.2.hook_resid_post", "l2_post_units"), ("blocks.3.hook_resid_post", "l3_post_units")]

@dataclass(frozen=True)
class Run:
    name: str; checkpoint: Path; width: int

def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--model", action="append", nargs=3, metavar=("NAME", "CHECKPOINT", "WIDTH"), required=True)
    p.add_argument("--train-pairs", type=int, default=400); p.add_argument("--examples", type=int, default=200); p.add_argument("--seed", type=int, default=300)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu"); p.add_argument("--output-dir", type=Path, default=Path("artifacts/residual_gain_profile")); return p.parse_args()

def load(run: Run, device: str) -> torch.nn.Module:
    saved = torch.load(run.checkpoint, map_location=device, weights_only=False); model = build_model(ModelConfig(**saved["model_config"]), device); model.load_state_dict(saved["model_state_dict"]); model.eval(); return model

def injected_logits(model: torch.nn.Module, prompts: torch.Tensor, name: str, d: torch.Tensor | None = None, amount: float = 0.) -> torch.Tensor:
    if d is None or amount == 0: return model(prompts)[:, -1]
    def edit(value: torch.Tensor, hook: object) -> torch.Tensor:
        changed = value.clone(); changed[:, -1] += amount * d; return changed
    return model.run_with_hooks(prompts, fwd_hooks=[(name, edit)])[:, -1]

def gradient(model: torch.nn.Module, prompts: torch.Tensor, target: torch.Tensor, source: torch.Tensor, name: str, d: torch.Tensor) -> torch.Tensor:
    got: list[torch.Tensor] = []
    def retain(value: torch.Tensor, hook: object) -> torch.Tensor:
        value.retain_grad(); got.append(value); return value
    model.zero_grad(set_to_none=True); out = model.run_with_hooks(prompts, fwd_hooks=[(name, retain)])[:, -1]; s, _ = score(out, target, source); s.sum().backward()
    return (got[0].grad[:, -1] * d).sum(-1).detach()

def main() -> None:
    a = args(); device = resolve_device(a.device); a.output_dir.mkdir(parents=True, exist_ok=True); rows: list[dict[str, object]] = []
    for raw in a.model:
        run = Run(raw[0], Path(raw[1]), int(raw[2])); model = load(run, device)
        tr_t, tr_s, _, _ = paired_prompts(run.width, a.train_pairs, a.seed + 100, device)
        te_t, te_s, target, source = paired_prompts(run.width, a.examples, a.seed, device); baseline_s, _ = score(model(te_t)[:, -1], target, source)
        for name, label in SITES:
            train_delta = state(model, tr_s, name, -1) - state(model, tr_t, name, -1); d = torch.linalg.svd(train_delta, full_matrices=False).Vh[0]
            if float((train_delta @ d).mean()) < 0: d = -d
            eval_delta = state(model, te_s, name, -1) - state(model, te_t, name, -1); scale = float((eval_delta @ d).abs().mean())
            out = injected_logits(model, te_t, name, d, scale); changed, sd = score(out, target, source)
            gain = gradient(model, te_t, target, source, name, d)
            eps = .001 * scale; fd = (score(injected_logits(model, te_t, name, d, eps), target, source)[0] - score(injected_logits(model, te_t, name, d, -eps), target, source)[0]) / (2 * eps)
            rows.append({"model": run.name, "site": label, "mean_gradient_gain": float(gain.mean()), "mean_finite_difference_gain": float(fd.mean().detach()), "jvp_fd_correlation": float(torch.corrcoef(torch.stack((gain.cpu(), fd.detach().cpu())))[0, 1]), "mean_delta_s_at_positive_scale": float((changed - baseline_s).mean().detach()), "source_tens_argmax_fraction": float(out.argmax(-1).eq(sd).float().mean()), "scale": scale})
        print(f"completed {run.name}")
    (a.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n"); fields = sorted({k for r in rows for k in r})
    with (a.output_dir / "results.csv").open("w", newline="") as f: writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"saved {a.output_dir / 'results.json'}")

if __name__ == "__main__": main()
