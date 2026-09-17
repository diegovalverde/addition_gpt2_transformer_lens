# Mechanistic Plan: Carry-Dependency Computation

## Purpose

Explain the reproducible behavioral difference between two three-digit addition
models:

- **Dependency-excluded models:** trained without every case where an incoming
  carry is necessary for the following column to carry; all three seeds score
  0% exact on that omitted family.
- **Random-drop controls:** trained with an equal 9% uniform resampling rate;
  all three seeds score 100% exact on the same family.

The goal is a causal account of the difference, not another demonstration that
carry labels are linearly decodable. This plan adapts the Othello methodology:

```text
probe -> semantic direction -> small intervention -> JVP validation
-> component localization -> ablation / rescue -> generalization
```

## Exact latent predicate

For a decimal column, define:

```text
incoming_carry = carry into the column
raw_sum        = left_digit + right_digit, excluding incoming carry
dependency     = incoming_carry AND (raw_sum == 9)
```

The broader holdout removes every three-digit addition containing this predicate
at either units-to-tens or tens-to-hundreds. It occurs in exactly 9% of uniform
three-digit operand pairs.

For the first mechanistic experiments, isolate a units-to-tens dependency:

- hold all tens and higher input digits fixed;
- require raw tens sum to equal 9;
- toggle only the units carry;
- ensure higher raw sums are at most 8, preventing a second dependency.

The source has `dependency = 1`; the target has `dependency = 0`. The two
prompts share their units answer digit, so their tens answer digit is the clean
downstream counterfactual.

## Established evidence

- Linear carry probes distinguish ordinary units carry perfectly in both model
  families. They do **not** explain the behavioral contrast.
- Full `=` residual replacement at layer 0 changes target to source answers in
  controls but not excluded models when using dependency-conditioned pairs.
- Layer-0 attention-output replacement reproduces the full-residual effect.
  MLP-output replacement is weaker; pre-residual replacement is null.
- No individual layer-0 head is sufficient. The largest single-head source
  counterfactual rate was 3.5%, so the relevant attention computation is
  distributed or interaction-dependent.

Do not say that a model “represents” the dependency merely because a probe can
decode it. Do not say that a head implements it merely because it has a useful
attention pattern or attribution score.

## Models and evaluation discipline

Use the IID-selected checkpoints already documented in `RESULTS.md` and
`AGENT_HANDOFF.md`. The main comparison is each excluded seed against its
matched random-drop-control seed.

- Keep the 200 fixed predicate-conditioned source/target pairs, seed 300.
- Report every seed, including weak or null effects.
- Use CPU as the reference device; do not use MPS as final evidence.
- Select model checkpoints using IID accuracy only. Never select on the holdout
  score or on a mechanistic metric.

## Phase 1: dependency-conditioned probe geometry

### Question

Is the exact dependency predicate linearly readable at the causal site:
layer 0, `=` position, especially its attention output?

### Design

Fit logistic probes separately at these hooks:

```text
blocks.0.hook_resid_pre
blocks.0.attn.hook_z or attention output
blocks.0.hook_resid_post
```

Train and test on fresh matched predicate-conditioned examples. Balance source
and target labels. Split by underlying higher-digit template so no template
appears in both train and test.

Let `v_dependency` be the normalized positive-minus-negative probe direction.
Also retain directions for:

- units carry alone, with raw tens sum not fixed to 9;
- a shuffled dependency label;
- a random normalized residual direction.

### Report

- held-out probe accuracy and majority baseline;
- probe accuracy by model family and seed;
- cosine similarities between dependency, ordinary-units-carry, and control
  directions;
- norms and pairwise alignment of directions across seeds.

High accuracy establishes **decodability only**. It supplies an operational
direction for the next phase.

## Phase 2: small signed semantic interventions

### Targeted score

Whole greedy answers are too coarse for a local test. Define a tens-digit logit
contrast after conditioning on the shared correct units answer token:

\[
S = z(\text{source-correct tens digit}) - z(\text{target-correct tens digit}).
\]

The input may include the common units answer digit teacher-forced after `=`;
the activation edit remains at layer 0, `=`. This makes `S` the immediate
decision affected by the dependency.

### Intervention

For each target prompt, edit the chosen activation:

\[
h' = h + \epsilon v_{dependency}.
\]

Evaluate both signs and small amplitudes, for example residual-norm-scaled
`epsilon = 0, +/-0.001, +/-0.003, +/-0.01, +/-0.03` before testing larger
values. Report mean `Delta S`, its confidence interval over the 200 pairs, and
the fraction whose tens-digit argmax flips to the source value.

### Required controls

- opposite-sign intervention;
- ordinary-units-carry direction;
- shuffled-label direction;
- random direction matched for norm;
- source prompts as a sign-specificity check;
- multiple amplitudes, restricted to the approximately linear regime.

The expected signature is selective positive `Delta S` in control models and a
weaker or absent effect in excluded models. A generic disruption that moves many
unrelated logits is not evidence for dependency use.

## Phase 3: Jacobian/JVP validation

### Local quantity

For each prompt `x`, source hook, `=` position, and direction `v`, compute the
Jacobian-vector product for the targeted score:

\[
J_xv = \frac{\partial S}{\partial h_{0,=}} v.
\]

Use autodiff without materializing a full Jacobian. Always record the source
hook, source position, output score, direction, and model checkpoint.

### Validation gate

For a small `epsilon`, compare the JVP with a central finite difference:

\[
\frac{S(h + \epsilon v) - S(h - \epsilon v)}{2\epsilon}.
\]

Then compare both with the actual hook intervention. Report correlation, sign
agreement, and relative error across the fixed pair set. Do not use JVP results
whose finite-difference validation fails.

This establishes **local causal geometry**: whether the model’s downstream
computation locally treats the probe-defined dependency direction as evidence
for the dependent tens digit. It does not yet identify a circuit.

## Phase 4: J-space transport

Transport `v_dependency` from layer-0 `=` to later hidden states using local
JVPs, rather than assuming a direction retains the same meaning across layers.

Candidate targets:

```text
blocks.0.hook_resid_post at =
blocks.1.hook_resid_pre at =
blocks.1.hook_resid_post at =
final residual at the position predicting the tens digit
```

For each context `x`, compute the local transported vector `J_x v_dependency`.
Compare its norm and cosine with:

- the transported vector in the paired source context;
- the average transported vector across the pair dataset;
- the downstream gradient of `S`;
- the corresponding vectors from excluded models.

Validate at least one source-to-target JVP with central finite differences.
Average transformed vectors only after computing per-context JVPs; do not claim
that an average Jacobian is a universal representation.

## Phase 5: layer-0 attention localization

### Existing next task: complete head-subset patching

Run all 15 nonempty subsets of the four layer-0 heads on the fixed
predicate-conditioned pairs. Patch only the selected `hook_z` heads from source
to target. Report source-counterfactual rate and `Delta S` for every subset and
every seed.

The purpose is to test whether a small head combination is sufficient. Do not
choose a subset based only on one seed, and do not hide null subsets.

### Attribution as a hypothesis generator

For score `S`, calculate each head write’s local alignment:

\[
A_h = \nabla_{c_h}S^\top c_h.
\]

Compare real heads with same-size random head subsets. Attribution ranks
candidates; it is not an intervention and cannot establish causal necessity.

Inspect candidate attention patterns only after a patching signal exists. The
specific hypothesis is that the candidate head set reads units digits and tens
digits and writes dependency evidence at `=`.

## Phase 6: necessity, mediation, and rescue

For any head subset that is sufficient in Phase 5:

1. **Necessity:** mean-ablate or zero-ablate it in control models. Measure `S`,
   tens-digit accuracy, and carry-dependency exact accuracy. Compare with random
   subsets of the same size.
2. **Rescue:** after ablating the subset, restore its clean or matched-source
   write. A selective recovery of `S` and the source tens digit is stronger
   evidence than ablation alone.
3. **Path mediation:** patch the candidate layer-0 head writes while keeping
   later components clean, then test whether the effect appears at the predicted
   downstream residual target from Phase 4 and at `S`.

The strongest defensible circuit claim needs all three: selective sufficiency,
necessity relative to matched controls, and selective rescue/mediation.

## Decision rules

| Result | Interpretation and next action |
| --- | --- |
| Dependency probe fails | The predicate is not linearly accessible at the tested hook; sweep later hooks before using a direction intervention. |
| Probe succeeds but intervention/JVP is null | Decodability without demonstrated local causal use; try a different hook or representation, not stronger arbitrary edits. |
| JVP predicts selective intervention | Strong local causal-geometry evidence; proceed to transport and component work. |
| No head subset is sufficient | Treat the computation as distributed or interaction-dependent; test attention-output subspaces and head ablations rather than forcing a single-head story. |
| Subset is sufficient and necessary, with rescue | Candidate causal pathway; test it on a second broad holdout before calling it general. |

## What remains after this plan

Replicate any final circuit result on a second structurally different holdout,
such as isolated tens-to-hundreds dependencies. The present broad-holdout result
shows an important compositional failure; it does not by itself prove a general
theory of how the model implements all carries.

## References in this workspace

- `AGENT_HANDOFF.md` — project status and checkpoint mapping.
- `RESULTS.md` — all reported behavioral and patching results.
- `patch_carries.py` — matched and predicate-conditioned activation patching.
- `probe_carries.py` — existing residual carry-probe infrastructure.
- `../Mechanistic-Journey-OthelloGPT/docs/part1/chapter03_probing_isnt_enough.md`
- `../Mechanistic-Journey-OthelloGPT/docs/part2/chapter04_jacobians.md`
- `../Mechanistic-Journey-OthelloGPT/docs/part2/chapter05_jspace.md`
- `../Mechanistic-Journey-OthelloGPT/docs/part2/chapter06_information_flow.md`
