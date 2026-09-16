# Experiment results

## Two-digit smoke baseline

**Status:** exploratory. The model was trained on MPS, for which TransformerLens
emits a warning about possible silent numerical errors. Metrics below were
computed on CPU, but that validates checkpoint inference rather than MPS training
numerics. Reproduce a final baseline on CPU before making interpretability claims.

| Property | Value |
| --- | --- |
| Model | 4 layers, 4 heads, `d_model=128`, `d_mlp=512` |
| Training representation | fixed-width, reversed decimal digits |
| Training operand width | 2 |
| Training steps | 1,000 |
| Batch size | 256 |
| Seed | 1 |
| Training device | MPS |
| Checkpoint | `checkpoints/widths-2-seed-1.pt` (gitignored) |

The answer-only loss fell from 3.30666 at step 1 to 0.00973 at step 200.

### CPU evaluation

| Split | Examples | Teacher-forced digit accuracy | Teacher-forced exact answer | Greedy exact answer |
| --- | ---: | ---: | ---: | ---: |
| IID two-digit | 10,000 | 100.0000% | 100.0000% | 100.0000% |
| At least one carry | 10,000 | 100.0000% | 100.0000% | 100.0000% |

This demonstrates fitting and short-range generalization inside the two-digit
training distribution. It does **not** demonstrate length or numerical-range
generalization; those are the next experiments.

## Three-digit fixed-width CPU baseline

This is the first numerically conservative baseline: both training and evaluation
ran on CPU. Training was configured for 20,000 steps with checkpoints every
1,000 steps, then intentionally stopped after evaluating the first checkpoint.
The step-1,000 model had already fit the in-distribution task, and additional
steps could not answer the fixed-width generalization question.

| Property | Value |
| --- | --- |
| Model | 4 layers, 4 heads, `d_model=128`, `d_mlp=512` |
| Training representation | fixed-width, reversed decimal digits |
| Training operand width | 3 |
| Evaluated checkpoint | step 1,000 |
| Batch size | 256 |
| Seed | 1 |
| Training and evaluation device | CPU |
| Checkpoint | `checkpoints/widths-3-seed-1-step-1000.pt` (gitignored) |

| Split | Examples | Teacher-forced digit accuracy | Teacher-forced exact answer | Greedy exact answer |
| --- | ---: | ---: | ---: | ---: |
| IID three-digit | 10,000 | 99.9400% | 99.7000% | 99.7000% |
| Three-digit, at least one carry | 10,000 | 99.9540% | 99.7700% | 99.7700% |
| Unseen four-digit | 10,000 | 9.7817% | 0.0000% | 0.0000% |

The result cleanly establishes the fixed-width limitation: the model learns
three-digit addition, including carries, but does not extrapolate to a fourth
digit. The next experiment is mixed-width training on widths 1--3 and the same
four-digit evaluation.

## Commands

```bash
uv run python -u train.py --width 2 --steps 1000 --batch-size 256 --seed 1 --device mps
uv run python evaluate.py --checkpoint checkpoints/widths-2-seed-1.pt --width 2 --split iid --examples 10000 --batch-size 256 --seed 2 --device cpu
uv run python evaluate.py --checkpoint checkpoints/widths-2-seed-1.pt --width 2 --split carry --examples 10000 --batch-size 256 --seed 3 --device cpu

uv run python -u train.py --width 3 --steps 20000 --batch-size 256 --seed 1 --checkpoint-every 1000 --device cpu
uv run python evaluate.py --checkpoint checkpoints/widths-3-seed-1-step-1000.pt --width 3 --split iid --examples 10000 --batch-size 256 --seed 2 --device cpu
uv run python evaluate.py --checkpoint checkpoints/widths-3-seed-1-step-1000.pt --width 3 --split carry --examples 10000 --batch-size 256 --seed 3 --device cpu
uv run python evaluate.py --checkpoint checkpoints/widths-3-seed-1-step-1000.pt --width 4 --split iid --examples 10000 --batch-size 256 --seed 4 --device cpu
```
