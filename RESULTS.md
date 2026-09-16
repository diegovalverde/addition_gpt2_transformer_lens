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

Although this model was trained for 1,000 steps of 256 independently sampled
problems (256,000 draws), the three-digit input space contains 1,000,000 ordered
operand pairs. Its 99.70% exact accuracy on a separately seeded stream of 10,000
evaluation examples therefore rules out a literal table containing every input
pair. Some evaluation pairs can still collide with training draws, so this is
evidence against exhaustive pair memorization—not proof that the model has
learned a fully general algorithm. The structured carry-chain holdout below is
the stronger compositional test.

## Structured carry-chain holdout

The training set excluded every problem where the units column produces a carry
and the raw tens digits sum to nine. In those cases the incoming units carry is
necessary to trigger a second, tens carry. The pattern occurs in 4.5% of uniform
three-digit problems. A matched control rejected 4.5% of examples uniformly at
random instead, so it has the same expected amount of training data.

| Model, 1,000 CPU steps | IID exact accuracy | Held-out carry-chain exact accuracy |
| --- | ---: | ---: |
| Unrestricted baseline | 99.70% | 99.99% |
| Carry-chain-excluded | 97.22% | 36.86% |
| 4.5% random-drop control | 100.00% | 100.00% |

All metrics are greedy exact-answer accuracy on independently seeded 10,000
example streams. This is a strong single-seed indication of compositional
generalization failure: removing a small, structured combination of familiar
local rules causes a 63-point accuracy collapse, while discarding an equal
amount of random data does not. Repeat the three conditions with additional
seeds before treating the effect size as final.

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

## Linear carry probes before answer generation

For each model, fresh prompts were truncated immediately after `=`. A separate
linear classifier was fitted for every layer to predict the carry-out bit of
each decimal column. The probe training set has 10,000 examples and the held-out
test set has 5,000; all computation used CPU. Since the majority-class baseline
for these carry labels is 50.1--56.0%, values substantially above that range
show decodable carry information.

| Model / carry-out column | Layer 0 | Layer 1 | Layer 2 | Layer 3 |
| --- | ---: | ---: | ---: | ---: |
| Fixed width 3: carry 1 | 100.00% | 100.00% | 100.00% | 100.00% |
| Fixed width 3: carry 2 | 54.94% | 72.14% | 73.06% | 75.64% |
| Fixed width 3: carry 3 | 51.16% | 62.44% | 68.18% | 70.50% |
| Mixed widths 1--3: carry 1 | 100.00% | 100.00% | 100.00% | 100.00% |
| Mixed widths 1--3: carry 2 | 54.80% | 68.02% | 67.48% | 64.60% |
| Mixed widths 1--3: carry 3 | 50.92% | 62.52% | 66.02% | 50.90% |
| Direct width 4: carry 1 | 100.00% | 100.00% | 100.00% | 100.00% |
| Direct width 4: carry 2 | 54.52% | 66.60% | 71.54% | 67.84% |
| Direct width 4: carry 3 | 53.78% | 64.00% | 65.68% | 64.44% |
| Direct width 4: carry 4 | 64.44% | 67.82% | 67.92% | 65.54% |

The models expose the first carry linearly from the first layer, while later
carries become more decodable through middle and late layers. The fixed-width
model has the strongest late-layer carry-2 and carry-3 decodability; the
mixed-width model's final-layer carry-3 accuracy falls back to baseline. These
are **decodability** results, not evidence that a probe direction is causally
used. The next experiment should intervene on a held-out carry direction and
measure the change in generated answer digits.

## Carry-direction intervention

The fixed-width three-digit model was patched at layer 2, at the `=` position,
using its normalized carry-1 probe direction. The 200 held-out source examples
all had carry 1 equal to zero; a successful positive intervention would generate
`sum + 10`. Amplitudes are fractions of the mean residual-stream norm (43.45).

| Relative amplitude | Mean absolute first-logit change | Greedy answers changed | Exact `sum + 10` counterfactual |
| ---: | ---: | ---: | ---: |
| 0.0 | 0.000 | 0.0% | 0.0% |
| 0.1 | 0.074 | 0.0% | 0.0% |
| 0.2 | 0.146 | 0.0% | 0.0% |
| 0.4 | 0.307 | 0.0% | 0.0% |
| 0.8 | 0.728 | 0.0% | 0.0% |
| 1.6 | 1.463 | 3.0% | 0.0% |

The hook demonstrably changes logits, but this linear probe direction is not a
clean causal carry-control direction: even the largest tested perturbation does
not produce the intended counterfactual answer. The appropriate follow-up is
activation patching between matched source and target prompts, rather than
assuming a discriminative probe direction is generative.

## Matched carry activation patching

To test causal use without assuming a linear probe is a control direction, we
constructed 1,000 source/target pairs. Each pair has identical higher digits and
the same output units digit. Target units sum to 0--8 (no carry); source units
sum to 10--18 (carry), so the source answer is exactly `target + 10`.

We replaced the target model's layer-0 residual at `=` with a blend of its own
residual and the matched source residual, then greedily decoded the target.

| Source-residual blend | Target answer retained | Exact source (`target + 10`) answer | Answers changed | Mean answer shift |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 100.0% | 0.0% | 0.0% | 0.00 |
| 0.50 | 59.9% | 40.0% | 40.1% | 4.10 |
| 0.75 | 1.6% | 98.4% | 98.4% | 9.84 |
| 1.00 | 0.5% | 99.2% | 99.5% | 10.04 |

This is strong causal evidence that the early residual state at `=` contains
information the model uses to propagate the units carry. It also explains the
earlier probe-direction null result: the relevant computation is causally
distributed in the matched activation state, while a discriminative linear
direction alone is not a sufficient control vector.

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
