# Addition GPT-2 with TransformerLens

A small, from-scratch GPT-2-style transformer trained to add integers represented
as reversed digit sequences. See [DESIGN.md](DESIGN.md) for the experiment design.

## Setup

```bash
uv sync --group dev
uv run pytest
```

## First training run

```bash
uv run python train.py --width 2 --steps 1_000 --batch-size 256 --device mps
uv run python evaluate.py --checkpoint checkpoints/widths-2-seed-1.pt --width 2 --split iid --device mps
```

Use `--device cpu` if MPS is unavailable. The two-digit run is a smoke test; use
the three-digit configuration in `DESIGN.md` for the baseline experiment.

## MPS caveat

The current TransformerLens release warns that PyTorch MPS may silently produce
incorrect results. MPS runs are useful for exploratory work, but validate any
reported metric on CPU using the same seed and checkpoint. Do not suppress that
warning merely to obtain cleaner logs.

## Mixed-width and out-of-distribution runs

Train widths 1--3 in an even cycle, then evaluate an unseen length or range:

```bash
uv run python train.py --train-widths 1,2,3 --steps 20_000 --batch-size 256 --device mps
uv run python evaluate.py --checkpoint checkpoints/widths-1-2-3-seed-1.pt --width 4 --split iid --device cpu
uv run python train.py --width 3 --max-operand 100 --steps 20_000 --batch-size 256 --device mps
uv run python evaluate.py --checkpoint checkpoints/widths-3-seed-1.pt --width 3 --split range --min-operand 900 --max-operand 1000 --device cpu
```

## Carry probes

Probe pre-answer residual activations: every prompt ends at `=`, so the probe
cannot read teacher-forced answer digits. The command below writes held-out
layer-by-carry accuracies to ignored files under `artifacts/`.

```bash
uv run python probe_carries.py \
  --model fixed3 checkpoints/widths-3-seed-1-step-1000.pt 3 \
  --model mixed3 checkpoints/widths-1-2-3-seed-1.pt 3 \
  --model direct4 checkpoints/widths-4-seed-1-step-1000.pt 4 \
  --train-examples 10000 --test-examples 5000 --device cpu
```

## Carry-direction interventions

Use a saved pre-answer probe direction to patch the residual stream while
greedily decoding. For a positive carry intervention, examples begin with that
carry bit equal to zero and the counterfactual target is `sum + 10**column`.

```bash
uv run python intervene_carries.py \
  --checkpoint checkpoints/widths-3-seed-1-step-1000.pt \
  --probe-weights artifacts/carry_probes/pre_answer/carry_probe_weights.pt \
  --probe-model fixed3 --width 3 --layer 2 --carry-column 1 \
  --examples 200 --amplitudes 0,0.1,0.2,0.4,0.8,1.6 --device cpu
```
