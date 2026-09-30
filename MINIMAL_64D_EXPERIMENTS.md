# What Does the 1-Layer, 64D Adder Compute?

## Question and model

The smallest robust model found so far has one transformer block, four attention
heads, a 64-dimensional residual stream, 16-dimensional heads, and a
256-dimensional MLP. It reaches 100% greedy exact accuracy on 10,000 IID and
10,000 carry-required three-digit examples for each of three seeds after 1,000
training steps.

That is evidence that the model predicts addition well in distribution. It is
not by itself evidence that the 64 dimensions contain a digit-by-digit addition
algorithm: training samples are drawn from a finite one-million-pair input
space, so some evaluation prompts can also have appeared during training.

This plan separates three claims:

1. **Behavioral generalization:** it answers combinations absent from training.
2. **Representational structure:** before answer tokens are supplied, its `=`
   activation contains arithmetic variables such as carry bits.
3. **Causal use:** changing a variable's activation changes exactly the answer
   digits downstream of that variable.

Only the combination supports an arithmetic-computation interpretation.

## Common protocol

Use the three selected one-layer checkpoints, not the best seed only. Keep the
training budget fixed at 1,000 steps and batch size 256. Use CPU for all
reported results. Select any checkpoint using IID validation only; never select
on a held-out composition or a mechanistic score.

For each experiment report every seed, a mean, and the minimum seed result. The
minimum matters because the width sweep showed that a configuration can look
perfect in two seeds while failing in a third.

Use the following controlled baseline conditions whenever possible:

| Condition | What it controls |
| --- | --- |
| 1-layer, 64D adder | The model of interest |
| 4-layer, 64D adder | Whether depth changes the mechanism |
| 1-layer, 32D successes and failures | The capacity/optimization boundary |
| Label-shuffled or random-direction control | Generic fitting or generic disruption |

## Experiment 1: exact unseen-pair evaluation

### Goal

Measure performance on operand pairs that were *provably absent* from training.
This is the cleanest first test against literal pair lookup.

### Design

Generate a deterministic train/test partition of all `1000 × 1000` ordered
three-digit operand pairs, for example by hashing `(left, right, seed)` into an
80/20 split. During training draw only pairs in the train partition and write a
compact manifest containing the split seed and rule. Evaluate only test-partition
pairs, stratified by no-carry, one-carry, two-carry, and carry-chain examples.

Run three partition seeds and three model seeds. The existing on-the-fly random
generator must be extended or wrapped for this experiment; do not infer absence
from a random evaluation seed.

### Predictions and decision rule

| Result | Interpretation |
| --- | --- |
| Near-100% exact on all strata | Rules out exact-pair lookup; continue to compositional tests. |
| High no-carry but low carry-chain accuracy | The model has local facts but not carry composition. |
| Accuracy near the train-pair collision rate | Consistent with lookup-style behavior. |

As an auxiliary sanity check, evaluate all one million pairs in batches. This
measures total task coverage, but only the split-restricted test identifies
unseen training pairs.

## Experiment 2: structured compositional holdouts

### Goal

Test whether the model composes familiar rules rather than interpolating a dense
table of examples.

### Designs

Run each of these as a fresh training condition, with a matched random-drop
control that removes the same expected number of examples.

1. **Carry dependency:** exclude cases in which an incoming carry is necessary
   for the next column to carry (`incoming_carry=1`, raw digit sum `=9`). The
   repository already supports this with `--exclude-carry-dependency-chain`.
2. **Units-pair composition:** withhold a balanced set of units digit pairs
   whose sums cover every answer digit and both carry values; test those pairs
   combined with unseen tens/hundreds templates. This prevents success by a
   single digit-pair lookup while retaining the primitive rules elsewhere.
3. **Position transfer:** train with the dependency predicate only in the
   units-to-tens transition and test it at tens-to-hundreds. This asks whether
   the one block shares a position-general carry rule or stores position-specific
   associations.

### Decision rule

Passing random-drop controls but failing the structured condition is evidence
for a missing composition, not simple data scarcity. Passing all conditions is
strong behavioral evidence for a compact addition rule, although it does not
uniquely identify the circuit.

## Experiment 3: pre-answer arithmetic probes

### Goal

Ask what information is present at the `=` position before the model receives
any answer digit. This avoids a probe merely reading teacher-forced output.

### Measurements

Cache these one-block activations at `=`:

```text
blocks.0.hook_resid_pre
blocks.0.attn.hook_z             # separately for each of four heads
blocks.0.attn.hook_result
blocks.0.hook_resid_post
```

Fit separate linear probes for:

- units carry, tens carry, and hundreds carry;
- each raw column sum (`a_i + b_i`) and its threshold-at-10 bit;
- each output digit and final overflow;
- the carry-dependency predicate (`incoming_carry AND raw_sum == 9`).

Split probe train/test sets by high-digit template, so near-identical prompts
cannot occur in both sets. Report probe accuracy alongside a majority-class and
a shuffled-label baseline. Compare the 1-layer 64D model with the 4-layer 64D
baseline and both successful and failed 32D seeds.

The existing starting point is:

```bash
uv run python probe_carries.py \
  --model one64 CHECKPOINT.pt 3 \
  --train-examples 10000 --test-examples 5000 --device cpu
```

### Interpretation

A high probe score means that a variable is decodable, not that the model uses
it. The interesting pattern is *where* each variable becomes decodable: e.g.,
units carry in a head write and the tens carry only after the MLP would support
a staged carry computation even in one block.

## Experiment 4: head roles and information flow

### Goal

Determine whether individual heads read digit columns, write carry evidence, or
whether the computation is distributed across heads and the MLP.

### Design

On matched prompt pairs that differ only in a units carry:

1. Record every head's attention pattern from `=` to the six operand-digit
   positions. Quantify attention mass by source column; do not rely on a single
   visualization.
2. Patch one `hook_z` head at a time from a carry-1 source into a carry-0 target
   at `=`. Then test every non-empty subset of the four heads.
3. Patch `attn` output, MLP output, and `resid_post` separately. Measure the
   logit contrast for the correct downstream tens digit and greedy answer, not
   only total loss.
4. Mean-ablate candidate heads or the MLP write; compare against random heads
   and same-norm random directions. Attempt a matched-source rescue after a
   successful ablation.

`patch_carries.py` supplies the matched-pair construction and component patching
interface. For the one-layer model, use `--layer 0`.

### Evidence standard

Call a head or component a carry mechanism only if it has selective
sufficiency, necessity relative to matched random controls, and rescue. A
plausible attention map or a useful attribution score is hypothesis generation,
not a circuit result.

## Experiment 5: causal carry directions

### Goal

Test whether the probe-defined carry code is read by the output computation.

### Design

At `blocks.0.hook_resid_post` and `=`, take the normalized positive-minus-
negative direction from a carry probe. For carry-0 targets, add small positive
and negative multiples of that direction. Measure the logit difference between
the counterfactual and original tens digit, while teacher-forcing the unchanged
units answer digit. Validate the smallest edits with a central finite difference
or a Jacobian-vector product.

Required controls are the opposite sign, a shuffled-label probe direction, a
same-norm random direction, and an ordinary units-carry direction when testing
the more specific carry-dependency direction.

Selective signed effects support causal use of that feature. Broad degradation
of many output digits means the edit is merely disruptive.

## Experiment 6: compactness and memorization stress tests

### Goal

Check whether success tracks arithmetic structure rather than the ability to
store arbitrary input-output associations.

### Design

Train the same 1-layer, 64D architecture on three altered targets:

1. A fixed random permutation of correct answer strings across operand pairs.
2. Correct answers with one output digit independently permuted.
3. The true task but with the exact unseen-pair split from Experiment 1.

Compare training loss, train exact accuracy, and disjoint-test exact accuracy.
If the model fits arbitrary labels as easily as addition, arithmetic success is
weak evidence of a special computation. If arbitrary labels do not generalize
or fit poorly while addition passes the structured splits, a compact-rule
account becomes more credible.

## Recommended order and stopping points

1. Run Experiment 1 and the carry-dependency holdout first. Stop any claim of
   algorithmic generalization if either fails.
2. Run Experiment 3 on the ordinary 1-layer 64D model and on the failed 32D
   seed. This gives a useful contrast even before interventions.
3. Run Experiments 4 and 5 only for variables with stable, cross-seed probe
   geometry.
4. Run Experiment 6 if behavioral and causal evidence agree but a lookup-like
   explanation remains plausible.

The desired end state is deliberately narrower than “we understand all 64
coordinates”: a reproducible account of which subspaces encode carry and digit
facts, which components write them, and which downstream reads are necessary
to produce the sum.
