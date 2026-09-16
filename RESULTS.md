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

## Commands

```bash
uv run python -u train.py --width 2 --steps 1000 --batch-size 256 --seed 1 --device mps
uv run python evaluate.py --checkpoint checkpoints/widths-2-seed-1.pt --width 2 --split iid --examples 10000 --batch-size 256 --seed 2 --device cpu
uv run python evaluate.py --checkpoint checkpoints/widths-2-seed-1.pt --width 2 --split carry --examples 10000 --batch-size 256 --seed 3 --device cpu
```
