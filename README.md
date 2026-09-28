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

## Minimum residual-stream width

The baseline residual stream has `d_model=128`. To identify the smallest width
that still performs three-digit addition reliably, sweep widths with the MLP
kept at `4*d_model` and four heads. A candidate must reach 99% greedy exact
accuracy on both IID and carry-required examples for every seed.

```bash
uv run python sweep_dimensions.py --device mps
```

The machine-readable results are written to `artifacts/dimension-sweep/summary.csv`.

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

## Matched activation patching

This stronger causal test replaces the layer-0 `=` residual of a target prompt
with the matching residual from a source prompt. The source and target have the
same higher digits and output units digit, but differ in whether units addition
creates a carry; their correct answers differ by exactly 10.

```bash
uv run python patch_carries.py \
  --checkpoint checkpoints/widths-3-seed-1-step-1000.pt \
  --width 3 --layer 0 --examples 1000 --blends 0,0.5,0.75,1 --device cpu
```

## Structured carry-chain holdout

Hold out the composition in which a units carry triggers a tens carry because
the raw tens digits sum to nine. Compare it against a 4.5% random-drop control;
the structured pattern occurs with probability 4.5% under uniform operands.

```bash
uv run python train.py --width 3 --steps 1000 --batch-size 256 \
  --exclude-units-carry-chain --device cpu \
  --checkpoint-dir checkpoints/carry-chain-holdout
uv run python evaluate.py --checkpoint checkpoints/carry-chain-holdout/widths-3-seed-1.pt \
  --width 3 --split carry-chain --examples 10000 --device cpu

uv run python train.py --width 3 --steps 1000 --batch-size 256 \
  --drop-probability 0.045 --device cpu \
  --checkpoint-dir checkpoints/carry-chain-random-control

# Use a coherent longer learning-rate schedule for a duration curve.
uv run python train.py --width 3 --steps 3000 --schedule-steps 3000 \
  --batch-size 256 --exclude-units-carry-chain --checkpoint-every 1000 \
  --device cpu --checkpoint-dir checkpoints/carry-chain-holdout-3k
```

## Fractional carry-chain exposure curve

`--units-carry-chain-exposure p` retains each naturally sampled units-to-tens
carry-chain example independently with probability `p`; all non-chain examples
are retained. Chains occur about 4.5% of uniform three-digit operand pairs, so
`p=0.05` produces roughly 0.24% chains in the final training stream (not 5%).

Run exposures `0, 0.001, 0.01, 0.05, 1` across seeds `1, 2, 3` with CPU, a
lower learning rate, and a coherent schedule. Record checkpoints every 500
steps, select a checkpoint on IID accuracy only, then measure carry-chain
accuracy at that step.

```bash
for exposure in 0 0.001 0.01 0.05 1; do
  for seed in 1 2 3; do
    uv run python -u train.py --width 3 --steps 3000 --schedule-steps 3000 \
      --batch-size 256 --learning-rate 0.0003 --seed "$seed" \
      --units-carry-chain-exposure "$exposure" --checkpoint-every 500 --device cpu \
      --checkpoint-dir "checkpoints/carry-chain-exposure-$exposure-seed-$seed"
  done
done
```

A sharp recovery between `0.001` and `0.05` would mean the model needs examples
of the composition but learns it sample-efficiently. Weak accuracy through
`0.05` would instead point to a stronger compositional limitation of this
architecture and objective.
