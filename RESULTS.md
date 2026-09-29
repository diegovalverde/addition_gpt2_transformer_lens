# Experiment results

## Minimum residual-stream width

We trained fixed-width three-digit addition models for 1,000 CPU steps with a
batch size of 256. Each candidate used four attention heads, four transformer
blocks, and an MLP width of `4*d_model`. The held-out evaluation contains 10,000
examples per split and per seed. A width passes only if **every** seed reaches at
least 99% greedy exact-answer accuracy on both IID and carry-required examples.

| `d_model` | IID greedy exact (seeds 1, 2, 3) | Carry greedy exact (seeds 1, 2, 3) | Passes all seeds? |
| ---: | --- | --- | :---: |
| 4 | 0.06%, 0.13%, 0.17% | 0.13%, 0.08%, 0.18% | No |
| 8 | 0.15%, 0.17%, 0.19% | 0.12%, 0.13%, 0.17% | No |
| 16 | 6.05%, 7.41%, 1.12% | 6.04%, 7.93%, 1.16% | No |
| 32 | 11.71%, 100.00%, 99.99% | 10.89%, 100.00%, 99.98% | No |
| 64 | 100.00%, 100.00%, 100.00% | 100.00%, 100.00%, 100.00% | **Yes** |

The smallest robust width in this experiment is therefore **64**, half the
128-dimensional baseline. At 32 dimensions the model is near the transition but
not reliable: two seeds solve addition almost perfectly, while one remains near
chance-level exact-answer accuracy. These results were run on an Intel macOS
host with PyTorch 2.2.2 and TransformerLens 2.15.4 because current PyTorch
releases do not ship Intel macOS wheels. This runtime difference should be kept
in mind when comparing against the project’s current Torch >=2.6 setup.

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

### Replication and additional-training check

Two additional carry-chain-excluded seeds reproduce the qualitative gap, though
its size varies. The two random-drop controls tested (seeds 1 and 2) remain at
100% exact accuracy on both IID and carry-chain streams.

| Carry-chain-excluded seed, 1,000 steps | IID exact accuracy | Carry-chain exact accuracy | Gap |
| --- | ---: | ---: | ---: |
| 1 | 97.22% | 36.86% | 60.36 points |
| 2 | 99.40% | 87.41% | 11.99 points |
| 3 | 99.08% | 79.64% | 19.44 points |
| Mean | 98.57% | 67.97% | 30.60 points |

More training did not reliably repair the omission in a fresh seed-1,
three-thousand-step run with its learning-rate schedule set for 3,000 steps:
carry-chain exact accuracy was 52.32% at step 1,000, then 0.02% at steps 2,000
and 3,000. IID exact accuracy at step 3,000 remained 95.46%. This non-monotonic
result means extra examples under the current optimization setup are not a
reliable fix; it warrants a more stable multi-seed learning-curve study before
spending compute on 5,000- or 10,000-step runs.

### Next: stabilized fractional-exposure curve

The next study will retain only a controlled fraction of the carry-chain cases:
`0`, `0.001`, `0.01`, `0.05`, or `1.0`. Here an exposure is the probability of
keeping a naturally sampled chain example, while all non-chain examples remain.
Because chains arise in roughly 4.5% of uniform operand pairs, these settings
produce approximately 0%, 0.0045%, 0.045%, 0.24%, and 4.5% chain examples in
the training stream.

Each condition will use seeds 1--3, CPU, a 3,000-step schedule, learning rate
`3e-4`, and checkpoints every 500 steps. We will select a checkpoint by IID
accuracy before reading its carry-chain score. This separates a genuine
sample-complexity threshold from the unstable training dynamics seen in the
earlier 3,000-step run.

#### Stabilized zero-exposure replications

All three replications used exposure 0, CPU, batch size 256, a 3,000-step
cosine schedule, learning rate `3e-4`, and checkpoints every 500 steps. IID
evaluation used a fresh 10,000-example stream with seed 41; checkpoint
selection used only greedy IID exact accuracy. Carry-chain evaluation used a
separate fresh 10,000-example stream with seed 42.

| Checkpoint | IID greedy exact accuracy | Carry-chain greedy exact accuracy |
| ---: | ---: | ---: |
| 500 | 98.74% | not evaluated |
| 1,000 | 99.74% | not evaluated |
| 1,500 | 99.86% | 97.14% |
| 2,000 | 99.97% | 99.41% |
| 3,000 | 99.99% | 99.91% |

For seed 1, the final step-3,000 checkpoint was IID-selected and achieved
99.91% greedy exact on the carry-chain stream. Seeds 2 and 3 completed the same
protocol. When multiple checkpoints tied on IID score, the earliest tied
checkpoint was selected before reading the carry-chain score.

| Seed | IID-selected checkpoint | IID greedy exact accuracy | Carry-chain greedy exact accuracy |
| ---: | ---: | ---: | ---: |
| 1 | 3,000 | 99.99% | 99.91% |
| 2 | 1,500 | 100.00% | 100.00% |
| 3 | 2,500 | 99.99% | 99.85% |

Despite seeing no training examples of this carry-chain pattern, all three
IID-selected models score at least 99.85% greedy exact on it. The earlier
structured-holdout failure was therefore strongly sensitive to optimization;
this narrow holdout is not sufficient to distinguish literal lookup from a
broader compositional carry computation. Per the preregistered decision rule,
the next experiment should be a broader compositional holdout that removes all
examples where an incoming carry is necessary for the following column to carry,
while retaining each local operation separately. Do not run the fractional
exposure curve or compare mechanisms until that broader success/failure contrast
exists.

### Broader carry-dependency holdout

The units-to-tens chain is too narrow: all zero-exposure replications solved
it. The broader split instead marks an addition whenever an incoming carry is
necessary for a later column to carry. At a given column this means the incoming
carry is one and the raw digits sum to nine. For three-digit additions the split
therefore includes both units-to-tens and tens-to-hundreds dependencies, while
still allowing each raw local digit operation in other contexts. It occurs in
exactly 9.0% of uniform operand pairs: 4.5% are units-to-tens dependencies,
4.95% are tens-to-hundreds dependencies, and 0.45% satisfy both.

All three seeds used the same CPU, batch size 256, 3,000-step cosine schedule,
learning rate `3e-4`, and checkpoint cadence as the stabilized study. A matched
random-drop control conditionally resampled 9% of candidates uniformly. Each
condition selected its earliest IID-tied-best checkpoint using the independent
10,000-example IID stream (seed 41), then received one 10,000-example
carry-dependency evaluation (seed 42).

| Condition | Seed | IID-selected checkpoint | IID greedy exact accuracy | Carry-dependency greedy exact accuracy |
| --- | ---: | ---: | ---: | ---: |
| Dependency excluded | 1 | 1,000 | 90.55% | 0.00% |
| Dependency excluded | 2 | 1,000 | 90.55% | 0.00% |
| Dependency excluded | 3 | 1,000 | 90.55% | 0.00% |
| 9% random-drop control | 1 | 500 | 100.00% | 100.00% |
| 9% random-drop control | 2 | 1,500 | 100.00% | 100.00% |
| 9% random-drop control | 3 | 1,500 | 100.00% | 100.00% |

This is a reproducible success/failure contrast: the broad structural omission,
rather than uniform resampling at the same rate, prevents both strong IID
performance and zero-shot success on the omitted dependencies. Mechanistic
comparisons between matched excluded and control models are now justified, but
their goal is to explain this behavioral contrast—not to treat probe
decodability as causal evidence.

#### Initial mechanistic comparison

Linear probes at the `=` residual position were fit on 10,000 IID additions and
tested on 5,000 fresh IID additions for every IID-selected model. Units-carry
decoding was 100% at every layer in every model. At layer 2, mean decoding
accuracy across seeds was 81.45% (excluded) versus 81.95% (control) for the
tens carry, and 68.61% versus 70.39% for the hundreds carry. Thus this probe
does not provide a clean explanation of the behavioral contrast.

We also patched the entire `=` residual from matched no-units-carry prompts to
matched units-carry prompts, at every layer, on 200 fixed pairs per model. Full
layer-0 replacement switched to the source sum in 86.5%, 86.5%, and 42.5% of
excluded-model pairs (seeds 1--3) and 100.0%, 100.0%, and 13.5% of control
pairs. This is strongly seed-dependent rather than a consistent condition
effect. Moreover, these legacy pairs are matched only on the units carry, not
on the broader carry-dependency predicate, so they cannot identify the cause of
the broad-holdout failure. The next causal test should construct pairs that
toggle a specific carry dependency while holding the remaining columns fixed.

#### Predicate-conditioned activation patching

We then constructed pairs that hold all tens and higher input digits fixed,
make the raw tens sum nine, and toggle only the units carry. Consequently, the
source—but not the target—has exactly one units-to-tens carry dependency; higher
raw digit sums are at most eight to prevent a second dependency. On the same
200 fixed pairs, replacing the full layer-0 `=` residual with the source
residual produced the source counterfactual in 98.5%, 100.0%, and 22.0% of
control-model pairs for seeds 1--3, respectively, but 0.0% for every excluded
model. No later-layer full-residual replacement produced source counterfactuals
in either condition.

This is causal localization evidence: in the controls, the layer-0 `=` residual
can carry the specific dependency information needed to change the answer,
whereas excluded models do not use that state to produce the omitted transition.
The third control's weaker 22.0% effect means its magnitude is not yet robust;
the next step is targeted component or direction patching within layer 0, with
the same fixed pairs and all three seeds.

Replacing only the layer-0 attention output reproduced the full-residual rates
(98.5%, 100.0%, and 22.0% in controls; 0.0% in excluded models). Replacing only
the MLP output was weaker (85.0%, 80.5%, and 3.5% in controls), and replacing
the layer-0 pre-residual was null. A complete four-head grid found no individual
attention head sufficient: its maximum source-counterfactual rate was 3.5%.
Thus the attention effect is distributed across heads or depends on a head
combination; the next targeted test should patch every pre-specified nonempty
head subset rather than nominate a single head from this screen.

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

## First generated digit probe

`probe_first_digit.py` fits a ten-way linear classifier to residual activations
at the `=` position, using prompts that contain no answer tokens.  The label is
the first reversed answer digit, `(left + right) % 10`.  Probes used 10,000
training examples and 5,000 disjoint test examples (seed 500), with a 10.52%
test-set majority baseline.

For the standard fixed-width checkpoint, layer-0 residual pre is at chance
(10.52%), while layer-0 residual post and layer-1 residual pre both reach
100.00% held-out accuracy.  Those latter activations are exactly identical
(maximum absolute difference 0), because they name the same residual boundary.
Layer-1, layer-2, and layer-3 residual post probes also reach 100.00%, and the
model itself predicts the first digit with 100.00% accuracy on this stream.

The same result replicates across all three dependency-excluded models and all
three random-drop controls: chance before layer 0 and 100.00% from layer-0 post
onward.  Thus the correct first digit becomes linearly available during the
first transformer block, before autoregressive answer generation begins.  This
is decodability evidence; a direction or component intervention is still needed
to establish which layer-0 write causally supplies the generated digit.

## Units carry-out probe

The same `probe_first_digit.py` runner also supports the binary
`--target units-carry-out`: whether the units digits sum to at least ten.  It
uses the same prompt-only 10,000/5,000 train/test split as the first-digit
probe.  The test-set carry rate is 44.94%, so the majority-class baseline is
55.06%.

The baseline model and all six dependency-excluded/control models are at that
majority baseline on layer-0 residual pre, then reach 100.00% held-out accuracy
at layer-0 residual post, layer-1 residual pre, and every later tested residual
site.  Thus layer 0 makes both the units output digit and its carry-out linearly
available before the first answer token is generated.  The two labels are not
the same—for example, raw units totals 0 and 10 share output digit 0 but have
different carry bits—so the carry probe captures information beyond merely
recovering the output digit.  It nevertheless remains a decoding result, not
proof that the linear probe direction is the model's causal carry representation.

## Second-column probes

`probe_second_digit.py` independently probes three quantities for the tens
column: its output digit `(raw_tens_sum + units_carry_in) % 10`, the incoming
units carry bit, and the tens carry-out bit.  Its natural `after-first-digit`
context teacher-forces the correct units answer digit and reads the residual at
that new final position: exactly the state used to predict the tens digit.  An
optional `equals` context tests precomputation before any answer token is
provided.

On the standard width-three checkpoint (10,000 training examples, 5,000
disjoint test examples, seed 600), layer-0 residual pre is effectively chance
for the completed tens digit and its carry-out, whereas the layer-0 post write
already exposes partial information.  Layer-1 residual post makes all three
quantities essentially linearly exact:

| Target | Majority baseline | Layer-0 pre | Layer-0 post | Layer-1 post | Layer-3 post |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tens output digit | 10.42% | 10.84% | 66.14% | 99.96% | 99.96% |
| Units carry-in | 53.88% | 75.50% | 75.18% | 99.98% | 99.98% |
| Tens carry-out | 50.00% | 52.26% | 97.42% | 99.88% | 99.98% |

Layer-0 post and layer-1 pre remain exactly the same residual boundary.  The
model predicts the teacher-forced tens digit with 99.72% accuracy on this
stream.  Thus the second step is not represented merely as a copied carry bit:
the layer-1 computation produces a residual from which the incoming carry, the
completed digit, and the next carry are separately recoverable.  These are
independent linear decoding tests, not causal interventions.

Run, for example:

```bash
uv run python probe_second_digit.py \
  --target carry-out --context after-first-digit \
  --model baseline checkpoints/widths-3-seed-1-step-1000.pt 3 \
  --device cpu
```

## Third-column probes

`probe_third_digit.py` repeats the independent-probe design for the hundreds
column: the hundreds output digit, the carry into hundreds from the tens
column, and the hundreds carry-out (the final overflow bit for width-three
addition).  In its natural `after-second-digit` context it teacher-forces the
correct units and tens answer digits, then reads the state that predicts the
hundreds digit.  As with the second-column tool, `equals` can instead test
precomputation before answer generation.

On the same baseline checkpoint (10,000 training examples, 5,000 disjoint test
examples, seed 700), the third-step pattern is closely analogous to the tens
step:

| Target | Majority baseline | Layer-0 pre | Layer-0 post | Layer-1 post | Layer-3 post |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hundreds output digit | 10.82% | 10.56% | 67.96% | 99.58% | 99.98% |
| Tens carry-in | 51.22% | 75.68% | 75.40% | 100.00% | 99.94% |
| Hundreds carry-out | 50.52% | 53.90% | 97.46% | 99.98% | 100.00% |

The model predicts the teacher-forced hundreds digit with 99.92% accuracy on
this stream.  Thus, after the first two answer tokens supply the generated
history, layer 1 again forms a state that linearly exposes both adjacent carry
bits and the completed column digit.  This is a separate decoding measurement
for each label, not proof that the probes identify causal directions.

```bash
uv run python probe_third_digit.py \
  --target carry-in --context after-second-digit \
  --model baseline checkpoints/widths-3-seed-1-step-1000.pt 3 \
  --device cpu
```

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

## Carry-dependency causal geometry and layer-0 head subsets

`mechanistic_dependency.py` implements the dependency-conditioned experiment
from `MECHANISTIC_PLAN.md`.  It uses the fixed 200 paired examples (seed 300),
teacher-forces their shared units answer digit, and scores the source tens-digit
logit minus the target tens-digit logit.  Checkpoints are the IID-selected ones
documented in `AGENT_HANDOFF.md`.  Probe training/testing used 400 matched
pairs (held-out by pair/template), 100 optimization epochs, and CPU.

The exact dependency is decodable at layer 0's attention output in both
families, so this observation does **not** explain the behavioral difference:

| Model | Excluded/control | Held-out dependency-probe accuracy |
| --- | --- | ---: |
| seed 1 | excluded | 68.5% |
| seed 1 | control | 79.5% |
| seed 2 | excluded | 95.0% |
| seed 2 | control | 84.8% |
| seed 3 | excluded | 99.8% |
| seed 3 | control | 100.0% |

Small normalized dependency-direction edits at the attention output were
locally differentiable: JVP versus central finite differences had correlation
at least 0.9997, 98.0--100.0% sign agreement, and 0.2--2.6% relative error
across the six models.  However, through amplitude 0.03 no signed probe edit
made a target's tens-digit argmax become the source digit.  Thus the result is
local causal geometry, not a usable semantic control direction.

The direct causal account instead remains a distributed layer-0 attention
write.  Replacing all four layer-0 `hook_z` writes from the matched source gave
the following source-tens argmax rates (and mean `Delta S`):

| Seed | Excluded | Control |
| --- | ---: | ---: |
| 1 | 0.0% (0.47) | 99.0% (15.01) |
| 2 | 0.0% (2.12) | 100.0% (36.27) |
| 3 | 0.0% (-1.10) | 22.0% (11.46) |

All 15 nonempty subsets were retained in
`artifacts/mechanistic_dependency/results.{json,csv}`.  No individual head was
sufficient; the strongest small subset differs by seed (for example 0+2+3 is
98.0% in control seed 1, whereas 0+1+2 is 98.5% in control seed 2 and 19.0% in
control seed 3).  There is therefore no seed-robust small-head circuit to call
necessary yet.  The supported conclusion is narrower: controls use a
multi-head layer-0 attention state to transmit the units-to-tens dependency,
whereas excluded models do not make that matched source state causally drive
the tens decision.  Necessity, rescue, and later-state mediation remain needed
before claiming a complete circuit.

## Distributed-state necessity, rescue, and transport

We next tested the complete four-head layer-0 attention write as the candidate
state, rather than selecting a different small subset per seed.  On the same
200 pairs, zero-ablation selectively altered the control tens decision, while
ten same-size random four-head ablations drawn from layers 1--3 retained the
target tens digit on 100% of pairs for every model.  The effect is variable,
and should be reported that way:

| Model | Clean mean S | Layer-0 all-head zero-ablation mean S | Source tens argmax after ablation | Matched-source write source-tens argmax |
| --- | ---: | ---: | ---: | ---: |
| excluded 1 | -12.87 | -13.09 | 0.0% | 0.0% |
| control 1 | -9.30 | 0.63 | 61.5% | 99.0% |
| excluded 2 | -15.22 | -15.62 | 0.0% | 0.0% |
| control 2 | -19.32 | -12.30 | 5.5% | 100.0% |
| excluded 3 | -16.65 | -17.24 | 0.0% | 0.0% |
| control 3 | -18.22 | -14.37 | 0.0% | 22.0% |

Mean ablation was near-null, so the evidence is specifically about removing
the structured write, not merely replacing it with a typical activation.  The
matched-source condition is a state restoration/counterfactual patch (not an
independent post-ablation recovery sequence); it replicates the prior direct
sufficiency result.  A literal ablate-then-rescue experiment is the next
stronger test.

That stricter test is now complete.  We zeroed all four layer-0 `hook_z`
head writes, then restored **only** `blocks.0.hook_attn_out` at `=`.  Restoring
the clean combined output exactly restored each model's baseline score and
target tens digit.  Restoring its matched-source output instead yielded source
tens argmax rates of 99.0%, 100.0%, and 22.0% in control seeds 1--3, versus
0.0% in all excluded seeds.  This identifies the combined layer-0 attention
output as the mediating state for the observed patch effect.  It still does not
separate the individual head contributions, which remain distributed and
seed-variable.

Central finite differences also show that the normalized dependency direction
at layer-0 attention output propagates to layer-1 residual states and the
final residual (mean transported norms were nonzero in all seeds).  These
transport values establish that a local perturbation reaches downstream state,
but do not yet distinguish the model families; they are not circuit evidence
on their own.

The script and complete, per-seed random-control results are in
`test_dependency_circuit.py` and `artifacts/dependency_circuit/`.

## Residual-stream transport and readout

We then followed the Othello-style residual-stream analysis directly, using
only residual hooks for the representation and causal patches.  At each site,
a logistic dependency probe was trained/tested on disjoint paired templates
(400 pairs, 100 epochs).  Every model, excluded and control, reached
99.5--100.0% held-out accuracy at all tested sites.  This is strong
decodability but, again, does not explain behavior.

The causal patches do.  Source residuals were copied to the target at each
site; `Delta S` is the matched source-versus-target tens-logit contrast.

| Site | Controls: mean Delta S / source-tens rate | Excluded: mean Delta S / source-tens rate |
| --- | --- | --- |
| layer-0 residual post at `=` | 15.01, 36.27, 11.46 / 99%, 100%, 22% | 0.47, 2.12, -1.10 / 0%, 0%, 0% |
| layer-1 residual pre at `=` | identical to layer 0 post | identical to layer 0 post |
| layer-1 residual post at `=` | 0.13, 0.01, 0.13 / 0%, 0%, 0% | -0.37, -0.27, -0.37 / 0%, 0%, 0% |
| layer-2 residual post at `=` | -0.05, 0.02, 0.18 / 0%, 0%, 0% | -0.15, -0.02, 0.02 / 0%, 0%, 0% |
| final (layer-3) residual post at units token | 15.84, 37.79, 36.31 / 100%, 100%, 100% | 0.84, 3.02, 0.12 / 0%, 0%, 0% |

Layer-0 residual post and layer-1 residual pre are the same architectural
boundary, explaining their identical result.  The decisive source state is
not directly usable at `=` after the layer-1 update, but a causally effective
counterpart appears at the final residual position that predicts the tens
digit.  This gives a residual-path account of the contrast:

`layer-0 residual at =` → `later residual processing` → `layer-3 residual at
the units token` → `tens-digit logits`.

Per-context central finite differences confirm nonzero transport of the
layer-0 residual probe direction to every tested residual site.  Gradient
alignment at the final units position is positive in controls (0.19, 0.17,
0.21), supporting local readout alignment, though it is not by itself enough
to identify the intervening computation.  Full machine-readable results are
in `artifacts/residual_dependency/`; the runner is `residual_dependency.py`.

## Residual causal trace at the units token

To identify the first causal readout state, we swept source-to-target residual
patches across every units-token residual boundary, and then performed the
complementary mediation test: apply the causal source patch at layer-0 `=`,
but restore one units-token residual to its clean target value.

The first sufficient site is **layer-1 residual post at the units token**.
Patching that residual produced source tens-digit argmax in 100% of all three
control seeds, and 0% of all excluded seeds.  The preceding layer-0 residual
post / layer-1 residual pre units state was insufficient (0%, 0%, and 24.5%
in the controls; 0% in excluded models).  Later units-token residual sites
remain sufficient, as expected once the state has been written.

The reverse intervention establishes mediation.  With the source layer-0 `=`
residual patch active, restoring the clean layer-1 residual post at units
returns every control to 0% source-tens argmax and removes its score effect
(`Delta S`: -14.90, -36.26, -11.40).  Restoring the earlier layer-0-post /
layer-1-pre units residual leaves the source effect intact (99%, 100%, 22%).
The same operations are null in excluded models.

This refines the residual pathway without making an attention-head claim:

`layer-0 residual post at =` → `layer-1 transformation` →
`layer-1 residual post at units` → `later residual stream` → `tens logits`.

The complete trace is in `artifacts/residual_causal_trace/`, generated by
`residual_causal_trace.py`.

## Held-out low-dimensional residual subspace

At layer-1 residual post at the units token, the matched source-minus-target
state is effectively one-dimensional for this task.  We learned an uncentered
SVD basis from 400 disjoint paired templates and evaluated its projections on
the fixed 200 pairs.  The leading direction alone is sufficient for source
tens-digit argmax on 100% of every control seed and 0% of every excluded seed.
Control mean `Delta S` is 14.69, 37.02, and 34.00; excluded values are 0.60,
2.50, and -0.45.

Removing that same one-dimensional component from the full source residual
eliminates the source-tens effect in every control.  Ten independently sampled
random subspaces provide the matched control: at dimension 8, their mean
`Delta S` is only 0.66, 1.21, and 0.83 in controls and they produce 0% source
tens argmax.  This is held-out causal evidence for a one-dimensional residual
state at the first causal units-token readout site, not merely a high-accuracy
probe.  See `residual_subspace.py` and `artifacts/residual_subspace/`.

## Structural replication: tens-to-hundreds dependency

We repeated the one-dimensional test on isolated tens-to-hundreds dependency
pairs: units never carry, source and target differ only in the tens carry,
raw hundreds digits sum to nine, and the common units and tens answer digits
are teacher-forced before scoring the hundreds digit.  A separately trained
one-dimensional layer-1-post residual subspace is sufficient for source
hundreds-digit argmax in 99.5%, 100%, and 100% of the control seeds, versus
0% for every excluded seed.  Its mean `Delta S` is 13.89, 36.21, and 33.49 in
controls; randomly sampled 1D subspaces are null (mean `Delta S` 0.03, 0.18,
0.06; 0% source argmax).  Removing the learned component from the full source
residual returns all controls to 0% source-hundreds argmax.

This confirms the low-dimensional residual result on a structurally different
dependency transition.  It does not yet show that the two learned directions
are the same representation; `hundreds_dependency_subspace.py` and
`artifacts/hundreds_dependency_subspace/` contain the full results.

## Cross-position direction comparison

The independently fitted one-dimensional residual directions are nearly
identical within each model.  Absolute cosines for units-to-tens versus
tens-to-hundreds are 0.978, 0.985, and 0.978 in controls (0.985, 0.986, and
0.966 in excluded models).  Cross-task interventions confirm that this is a
shared causal direction in controls: a hundreds-trained direction gives 100%,
100%, and 100% source-tens argmax on units pairs, while a units-trained
direction gives 98.0%, 100%, and 100% source-hundreds argmax on hundreds
pairs.  Excluded models remain 0% in all cross-task tests.

The supported interpretation is a position-general carry-dependency residual
direction at layer-1 post, with a family difference in whether that direction
is causally routed into the next-digit decision.  See
`compare_dependency_directions.py` and
`artifacts/compare_dependency_directions/`.

## Layer-0 residual-direction transport

An independently learned leading source-minus-target direction at layer-0
residual post at `=` is itself causally useful: its held-out one-dimensional
projection yields source-tens argmax of 100%, 96.5%, and 16.5% in controls and
0% in excluded models.  Removing that component from the full source layer-0
state eliminates the source effect in all controls.  The layer-0 and layer-1
directions occupy different coordinates (absolute cosine 0.01--0.09 in five
models), but central finite-difference transport from the former to the latter
has mean alignment 0.86 and 0.81 in control seeds 1 and 2.

The same local alignment is also present in excluded seeds 1 and 2, so it
cannot be the family-level explanation.  This is a useful negative result:
local residual transport is not equivalent to causally using the state for the
next-digit decision.  See `residual_direction_transport.py` and
`artifacts/residual_direction_transport/`.

## Signed layer-1 residual readout gain

We injected the independently learned layer-1-post units direction alone,
oriented positive by held-out source-minus-target differences and scaled by
the mean matched projection.  A positive one-scale intervention produces
source-tens argmax at 99%, 100%, and 100% in controls and 0% in excluded
models.  More directly, the local derivative of the tens score along this
direction is +0.22, +0.50, and +0.42 in controls, versus -0.59, -0.24, and
-0.74 in excluded models.  Autodiff JVP and central finite differences have
correlations above 0.999999 in every model and complete sign agreement.

On the raw-sum-nine dependency contexts, this separates representation/transport
from readout: excluded models encode the direction but give it the wrong local
signed effect on the dependent tens decision.  Complete curves and validation are in `residual_readout_gain.py`
and `artifacts/residual_readout_gain/`.

## Residual gain profile

The signed causal distinction persists at each later units-token residual site.
For layer 1, layer 2, and layer 3 respectively, control derivative gains are
positive in every seed: (0.22, 0.33, 0.43), (0.50, 0.64, 0.43), and
(0.42, 0.58, 0.41).  Excluded gains are negative throughout: (-0.59, -0.42,
-0.17), (-0.24, -0.16, -0.02), and (-0.74, -0.41, -0.18).  Positive
matched-scale injections at layers 2 and 3 yield 100% source-tens argmax in
all controls and 0% in all excluded models.  Every local gradient was
validated against a central finite difference (correlation above 0.99999).

On the dependency contexts, the family difference is therefore established at
the first causal layer-1 units residual write and retained by the later residual
computation, rather than being introduced by a later sign flip.  See `residual_gain_profile.py`
and `artifacts/residual_gain_profile/`.

## Compositional two-dependency carry chain

We generated answers for prompts where the units carry triggers a tens carry
and the tens carry triggers a hundreds carry (both downstream raw sums are
nine).  The same learned layer-1 residual direction was injected after the
units answer, the tens answer, or both.  In controls, a tens-only injection
causes the independent hundreds-carry counterfactual (mean answer shift +95,
+100, +100); units-only produces the full source answer in 94.5%, 100%, and
100%; and both positions yield 99.5%, 100%, and 100% full-source exactness.
All excluded models retain the target answer exactly under every intervention.

The direction therefore composes along two successive dependent carries during
greedy decoding, rather than only controlling an isolated local contrast.  See
`compositional_dependency.py` and `artifacts/compositional_dependency/`.

## Raw-tens-sum generality sweep

The learned direction is not a generic malformed digit-shift perturbation.
We swept raw tens sums 0--9 while holding the units-carry source/target
contrast fixed, injecting the direction after the units answer and greedily
decoding the full result.  For raw sums 0--8, excluded seeds 1 and 2 produce
the exact source (`+10`) answer at 99--100%; seed 3 is also mostly successful
(88--100%).  At the dependency boundary, raw sum 9, all excluded models remain
at the target answer (0% source exact, 100% target exact).  Controls produce
the source answer at the boundary in 95%, 100%, and 100% of cases.

The mechanistic failure is therefore conditional: excluded models can use the
direction as an incoming carry to increment ordinary digits, but do not invoke
the modulo-10 wrap and outgoing-carry branch when the receiving raw digit sum
is nine.  This refines—not overturns—the negative local-gain result, which is
specific to the dependency boundary.  See `carry_direction_generality.py` and
`artifacts/carry_direction_generality/`.

## Direct downstream-carry injection

To distinguish a missing carry readout from a missing carry-state write, we
teacher-forced target units and tens digits and injected the direction directly
at the tens-token residual while predicting hundreds.  At raw tens sum nine,
this produces the source hundreds digit in 98%, 100%, and 88% of excluded
models (85%, 100%, and 73% of controls).  This bypasses the earlier failure:
excluded models can use an explicitly written carry state at the next token.

The same edit changes hundreds predictions for raw sums below nine too, as it
directly asserts an incoming carry irrespective of the forced tens context;
that is expected and makes this a bypass experiment rather than a selectivity
test.  The result localizes the missing operation to the conditional write of
the next carry state from incoming carry plus raw sum nine, not to the later
readout of a supplied carry state.  See `outgoing_carry_gate.py` and
`artifacts/outgoing_carry_gate/`.

## Factorial conditional-write interaction

We formed matched layer-1-post residual states for the 2×2 factorial of
incoming carry (0/1) and raw tens sum (8/9), then used the
difference-in-differences as a candidate conditional-write interaction.  It is
not a complete minimal mechanism.  Removing it from source raw-nine control
states reduces `S` by 3.99, 6.91, and 9.30, but source tens argmax remains
84.5%, 100%, and 99.5%; adding it to target raw-nine states is not sufficient
(at most 4.5% source argmax).  Excluded models remain null.

Thus the raw-nine conditional state is distributed within the layer-1 residual
transformation or depends on its surrounding state, rather than residing in a
single factorial residual vector.  See `conditional_write_interaction.py` and
`artifacts/conditional_write_interaction/`.

## Low-rank conditional-interaction subspace

SVD on interaction vectors from 400 disjoint templates yields a selectively
necessary but insufficient subspace.  Removing the learned 16D subspace from
source raw-nine states reduces control `S` by 3.88, 6.68, and 9.12, compared
with 0.55, 0.61, and 1.02 for matched random 16D subspaces.  Yet source tens
argmax remains 89.5%, 100%, and 98.5%, and adding the learned subspace to raw-
nine targets produces at most 3.5% source argmax.  Excluded models remain
null.  The conditional state is low-rank enough to be selectively necessary,
but requires its residual context to be causally expressed.  See
`conditional_interaction_subspace.py` and
`artifacts/conditional_interaction_subspace/`.

## Carry-plus-interaction synergy

We tested whether the gate could be reconstructed additively by combining the
one-dimensional incoming-carry component with the learned 16D interaction
subspace.  It cannot: carry-only, interaction-only, carry-plus-interaction,
and carry-plus-random-subspace edits all give 0% source exact in every excluded
model.  Controls are already driven to the source answer by carry-only edits
(97%, 100%, and 100%), and the interaction term does not add a rescue effect.
The conditional rule therefore requires the native context-dependent residual
update rather than additive assembly of its observed residual components.  See
`carry_interaction_synergy.py` and
`artifacts/carry_interaction_synergy/`.

## Context-dependent Jacobian gate

We injected the same small layer-0 carry direction into matched raw-tens-8
and raw-tens-9 contexts, measured its finite-difference transport to layer-1
post at units, and measured the tens-score JVP.  In excluded models, transport
along the layer-1 carry direction is still positive and nearly unchanged at
raw 9 (0.90 → 0.78, 0.45 → 0.45, 0.21 → 0.21).  The score JVP, however, flips
from positive at raw 8 (+0.25, +0.24, +0.20) to negative at raw 9 (-0.54,
-0.21, -0.18).  Control score JVPs remain positive in both contexts.

This localizes the missing behavior to a context-dependent nonlinear gate in
the layer-1 residual transformation: raw sum nine does not prevent carry-state
transport, but changes how that state affects the next-digit decision.  See
`conditional_jacobian_gate.py` and `artifacts/conditional_jacobian_gate/`.

## Direct context-gain profile

Directly differentiating the score with respect to each units-token residual
site confirms the ordering of the raw-nine error.  At layer-1 post, excluded
models have positive raw-eight gain (+0.16, +0.12, +0.42) and negative raw-nine
gain (-0.59, -0.23, -0.72); controls remain positive in both contexts.  Later
residual sites retain the raw-nine negative effect rather than repairing it.
The first causal sign error is therefore at the layer-1-post units boundary.
See `context_gain_profile.py` and `artifacts/context_gain_profile/`.

## Layer-1 component write localization

As an implementation-level follow-up to the residual localization, we copied
source layer-1 attention-output or MLP-output writes at the units token into
raw-nine targets.  In controls, attention-write patches yield source tens
argmax of 100%, 100%, and 65%; MLP-write patches yield 98%, 96%, and 100%.
Both are null in excluded models.  At raw eight, the same source component
patches work in both families.  Thus the failed boundary computation is
reflected in both layer-1 component writes, but these sufficiency patches do
not identify a unique minimal component circuit; zero ablations are broad.
See `layer1_component_gate.py` and `artifacts/layer1_component_gate/`.

## Layer-1 attention-to-MLP path mediation

The source attention-output patch at raw nine is mediated by the MLP write.
It produces source tens argmax in controls at 100%, 100%, and 70.5%, but adding
a clean target MLP-output restore cancels the effect (0% source argmax in every
control).  Conversely, source MLP-output patches alone yield 97%, 98.5%, and
100% source argmax.  Both source component writes recover the full source
effect.  This establishes the causal path
`layer-1 attention output → layer-1 MLP output → downstream residual readout`
for the boundary computation, while remaining agnostic about which internal
attention features drive the MLP.  See `layer1_path_mediation.py` and
`artifacts/layer1_path_mediation/`.

## Pre-MLP residual boundary test

Copying the full layer-1 units residual after attention and before the MLP
(`blocks.1.hook_resid_mid`) makes the boundary localization sharper.  At raw
tens sum eight, the matched source-to-target copy yields the source tens answer
in every model.  At raw sum nine it yields the source answer in 100% of cases
for each control, but 0% for each excluded model; mean score changes are
15.4--37.4 logits in controls and only 0.09--2.89 in excluded models.

The family difference is thus already causally expressed in the state the MLP
receives, not wholly created by the MLP from a common pre-MLP residual.  This
remains compatible with the path mediation result: attention supplies causal
context, and the native MLP update remains required to express it downstream.

For a small source-oriented pre-MLP residual perturbation, the local MLP map
transports into its learned output direction comparably in excluded raw-eight
and raw-nine contexts (positive projection 1.08--1.19).  But the downstream
score JVP flips from positive at raw eight (+0.53, +0.77, +1.09) to negative
at raw nine (-1.13, -0.54, -1.14).  Controls stay positive in both contexts.
The result supports a context-sensitive residual computation/readout at the
raw-nine branch rather than loss of carry-direction transport.  See
`layer1_mlp_residual_gate.py` and `artifacts/layer1_mlp_residual_gate/`.

## Threshold neighborhood sweep

To test whether the defect is truly specific to the dependency boundary, we
repeated the full-answer intervention at raw tens sums 7, 8, 9, 10, and 11.
Excluded models produce the source answer at 91--100% for 7 and 8 and 100% for
10 and 11, but all three give 0% source exact and 100% target exact at exactly
9.  Controls are highly successful throughout the neighborhood.  Thus excluded
models possess both neighboring behaviors—ordinary increment below the
threshold and raw carry above it—but miss the singleton transition
`9 + incoming carry`.  The threshold runner reuses
`carry_direction_generality.py`; results are in
`artifacts/carry_direction_threshold/`.

## Ordinary raw-carry branch rescue

We attempted a within-model residual rescue using the layer-1-post offset from
matched `raw 10, no incoming carry` minus `raw 9, no incoming carry` contexts,
added to failing `raw 9, incoming carry` states.  It restores 0% source exact
in every excluded model and disrupts correct controls; norm-matched random
offsets are also null in excluded models.  The ordinary raw-carry branch is
therefore not a portable additive correction for the dependency hole.  The
missing behavior requires the raw-nine context-sensitive transformation.
See `raw_carry_branch_rescue.py` and
`artifacts/raw_carry_branch_rescue/`.

Run the complete replication with:

```bash
uv run python mechanistic_dependency.py \
  --model excluded-1 checkpoints/carry-dependency-exposure-0-seed-1/widths-3-seed-1-step-1000.pt 3 \
  --model control-1 checkpoints/carry-dependency-random-control-seed-1/widths-3-seed-1-step-500.pt 3 \
  --model excluded-2 checkpoints/carry-dependency-exposure-0-seed-2/widths-3-seed-2-step-1000.pt 3 \
  --model control-2 checkpoints/carry-dependency-random-control-seed-2/widths-3-seed-2-step-1500.pt 3 \
  --model excluded-3 checkpoints/carry-dependency-exposure-0-seed-3/widths-3-seed-3-step-1000.pt 3 \
  --model control-3 checkpoints/carry-dependency-random-control-seed-3/widths-3-seed-3-step-1500.pt 3 \
  --probe-examples 400 --probe-epochs 100
```

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
