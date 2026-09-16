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

## Mixed-width CPU baseline

This matched the prior baseline's model, batch size, seed, CPU device, and
step-1,000 checkpoint. Instead of seeing only three-digit operands, training
cycled evenly between widths 1, 2, and 3.

| Property | Value |
| --- | --- |
| Training operand widths | 1, 2, and 3, evenly cycled |
| Evaluated checkpoint | step 1,000 |
| Batch size | 256 |
| Seed | 1 |
| Training and evaluation device | CPU |
| Checkpoint | `checkpoints/widths-1-2-3-seed-1-step-1000.pt` (gitignored) |

| Split | Examples | Teacher-forced digit accuracy | Teacher-forced exact answer | Greedy exact answer |
| --- | ---: | ---: | ---: | ---: |
| IID width 1 | 10,000 | 100.0000% | 100.0000% | 100.0000% |
| IID width 2 | 10,000 | 100.0000% | 100.0000% | 100.0000% |
| IID width 3 | 10,000 | 99.9460% | 99.7300% | 99.7300% |
| Width 3, at least one carry | 10,000 | 99.9400% | 99.7000% | 99.7000% |
| Unseen width 4 | 10,000 | 18.6567% | 0.0000% | 0.0000% |

Mixed-width training improves width-4 digit accuracy compared with the
fixed-width model (18.6567% versus 9.7817%), but it still never generates an
entire correct four-digit answer. At this scale and training duration, seeing
multiple lengths is not sufficient for exact length extrapolation.

### Mixed-width learning curve

The mixed-width step-1,000 checkpoint was resumed with its optimizer, scheduler,
and data-generator state intact. Increasing training duration did not yield
exact width-4 answers.

| Checkpoint step | Width-4 digit accuracy | Width-4 greedy exact answer |
| ---: | ---: | ---: |
| 1,000 | 18.6567% | 0.0000% |
| 2,000 | 18.8667% | 0.0000% |
| 3,000 | 18.9867% | 0.0000% |

## Four-digit capacity control

To distinguish an extrapolation failure from insufficient model capacity, the
same architecture was trained directly on four-digit addition. Both training and
evaluation used CPU; the checkpoint was saved at step 1,000.

| Split | Examples | Teacher-forced digit accuracy | Teacher-forced exact answer | Greedy exact answer |
| --- | ---: | ---: | ---: | ---: |
| IID width 4 | 10,000 | 99.9867% | 99.9200% | 99.9200% |
| Width 4, at least one carry | 10,000 | 99.9917% | 99.9500% | 99.9500% |

The architecture therefore has ample capacity for four-digit addition. The
mixed-width model's zero exact accuracy on width 4 is a genuine
length-generalization failure, not underfitting of the task.

## Commands

```bash
uv run python -u train.py --width 2 --steps 1000 --batch-size 256 --seed 1 --device mps
uv run python evaluate.py --checkpoint checkpoints/widths-2-seed-1.pt --width 2 --split iid --examples 10000 --batch-size 256 --seed 2 --device cpu
uv run python evaluate.py --checkpoint checkpoints/widths-2-seed-1.pt --width 2 --split carry --examples 10000 --batch-size 256 --seed 3 --device cpu

uv run python -u train.py --width 3 --steps 20000 --batch-size 256 --seed 1 --checkpoint-every 1000 --device cpu
uv run python evaluate.py --checkpoint checkpoints/widths-3-seed-1-step-1000.pt --width 3 --split iid --examples 10000 --batch-size 256 --seed 2 --device cpu
uv run python evaluate.py --checkpoint checkpoints/widths-3-seed-1-step-1000.pt --width 3 --split carry --examples 10000 --batch-size 256 --seed 3 --device cpu
uv run python evaluate.py --checkpoint checkpoints/widths-3-seed-1-step-1000.pt --width 4 --split iid --examples 10000 --batch-size 256 --seed 4 --device cpu

uv run python -u train.py --train-widths 1,2,3 --steps 20000 --batch-size 256 --seed 1 --checkpoint-every 1000 --device cpu
uv run python evaluate.py --checkpoint checkpoints/widths-1-2-3-seed-1-step-1000.pt --width 4 --split iid --examples 10000 --batch-size 256 --seed 14 --device cpu
uv run python train.py --train-widths 1,2,3 --steps 3000 --batch-size 256 --seed 1 --checkpoint-every 1000 --device cpu --resume-from checkpoints/widths-1-2-3-seed-1-step-1000.pt
uv run python evaluate.py --checkpoint checkpoints/widths-1-2-3-seed-1-step-2000.pt --width 4 --split iid --examples 10000 --batch-size 256 --seed 14 --device cpu
uv run python evaluate.py --checkpoint checkpoints/widths-1-2-3-seed-1.pt --width 4 --split iid --examples 10000 --batch-size 256 --seed 14 --device cpu

uv run python -u train.py --width 4 --steps 20000 --batch-size 256 --seed 1 --checkpoint-every 1000 --device cpu
uv run python evaluate.py --checkpoint checkpoints/widths-4-seed-1-step-1000.pt --width 4 --split iid --examples 10000 --batch-size 256 --seed 21 --device cpu
uv run python evaluate.py --checkpoint checkpoints/widths-4-seed-1-step-1000.pt --width 4 --split carry --examples 10000 --batch-size 256 --seed 22 --device cpu
```
