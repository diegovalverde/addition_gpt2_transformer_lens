# Current mechanistic interpretation

## Claim in one sentence

For the units-to-tens carry-dependency task, successful control models route a
dependency-sensitive state from the layer-0 residual at `=` into the layer-1
residual at the units-answer token, where it becomes causally sufficient for
the next (tens) digit; dependency-excluded models can linearly decode the same
predicate but do not route a matched source state into that readout.

This is a current, deliberately narrow belief about the three-digit models and
the fixed predicate-conditioned pairs.  It is not yet a claim about all decimal
carries or a fully identified head-level circuit.

## The task variable

The relevant latent predicate is:

```text
incoming units carry AND raw tens-digit sum == 9
```

It determines whether the tens column carries only because of the incoming
units carry.  Source and target prompts hold the tens and higher input digits
fixed, share the correct units answer digit, and differ only in whether the
units column carries.  After teacher-forcing that shared units digit, the next
token is the clean downstream tens-digit decision.

## Residual-stream account

```text
input at =
    │
    ▼
layer-0 residual post at =
    │  dependency-sensitive state; source-to-target patch is causal in controls
    ▼
layer-1 transformation
    │
    ▼
layer-1 residual post at units token
    │  first units-token residual state sufficient for the source tens digit
    │  restoring its clean target value blocks the layer-0 source-patch effect
    ▼
later residual stream
    ▼
tens-digit logits
```

`blocks.0.hook_resid_post` at `=` and `blocks.1.hook_resid_pre` at `=` are the
same architectural boundary, so their identical patch effects are a sanity
check rather than two independent discoveries.

## Causal evidence

On the fixed 200 source/target pairs:

- Copying the source layer-0 residual at `=` into targets produces source tens
  digits in control seeds 1--3 at 99%, 100%, and 22%; excluded seeds are 0%,
  0%, and 0%.
- Copying the source final layer-3 residual at the units token produces source
  tens digits in controls at 100%, 100%, and 100%; excluded models are 0%.
- The first units-token residual that is itself sufficient is layer-1 residual
  post: source patches produce source tens digits in all three controls and in
  none of the excluded models.
- With a layer-0 source patch active, restoring the clean layer-1-post units
  residual removes the source effect in every control.  Restoring the earlier
  layer-0-post/layer-1-pre units residual does not.  This identifies the
  layer-1 transition as a causal mediator of the residual pathway.

Earlier component-level experiments localize the layer-0 state to its combined
attention output: zeroing the four layer-0 head writes is selectively
disruptive in controls, and restoring just the clean combined attention output
restores baseline behavior.  This is useful localization, but the primary
mechanistic account above is intentionally stated in residual-stream terms.

## What probes do and do not say

Dependency probes reach roughly perfect held-out accuracy at tested residual
sites in **both** families.  This establishes that the predicate is linearly
decodable; it does not distinguish successful computation from failure.

Small edits along a normalized layer-0 dependency-probe direction have
finite-difference-validated local effects, but do not flip tens-digit argmax
in the tested small-amplitude regime.  The useful causal object is therefore a
matched residual state (and its downstream routing), not yet a single global
semantic direction that controls every context.

## What is still unknown

- The exact computation inside the layer-1 transformation that moves the state
  from `=` to the units token.  We have localized its residual input and output,
  not identified a minimal component circuit.
- Why control seed 3 has a weaker layer-0 direct patch effect despite a strong
  final-residual effect.
- The semantic interpretation of the one-dimensional layer-1 residual
  direction: it is a robust causal control direction for the paired task, but
  we have not yet compared its forms across dependency positions or tested
  operand width.

## Low-dimensional causal state

The full matched residual state is not required at the layer-1-post units
site.  On a disjoint 400-pair training stream, we took the uncentered leading
singular direction of source-minus-target residual differences.  On the fixed
200 held-out pairs, its one-dimensional projection is sufficient for the
source tens digit in 100% of all three controls and 0% of all excluded models.
It also accounts for nearly the full logit effect (control `Delta S`: 14.69,
37.02, 34.00).

Conversely, removing this one-dimensional projection from the full source
residual returns all controls to 0% source-tens argmax.  Same-dimensional
random orthogonal subspaces have no source-tens flips (the 8D random-control
mean `Delta S` is only 0.66, 1.21, and 0.83 for control seeds 1--3).  Thus the
current best candidate for the carried computation is a **one-dimensional
causal residual direction at layer-1-post units**, learned independently per
seed from training templates and validated on held-out templates.

## Structural replication: tens-to-hundreds

The same experiment was repeated on an isolated tens-to-hundreds dependency.
Units cannot carry; source and target differ only in whether the tens column
carries; raw hundreds digits sum to nine; the shared units and tens answers are
teacher-forced before predicting the hundreds digit.  A separately learned,
one-dimensional layer-1-post residual projection is sufficient for the source
hundreds digit in 99.5%, 100%, and 100% of the three controls, and 0% of every
excluded model.  Removing it from the full source residual returns all controls
to 0% source-hundreds argmax.  One-dimensional random-subspace controls are
null.

Thus the causal residual-state phenomenon is not confined to the original
units-to-tens holdout.

## Position-general direction

The two independently learned directions are nearly the same vector within
each model: their absolute cosine is 0.978, 0.985, and 0.978 in controls
(excluded models are similarly aligned at 0.985, 0.986, and 0.966).  This is
not merely geometric alignment.  Cross-position intervention transfers the
causal effect: applying a control's tens-to-hundreds direction to units-to-tens
pairs gives 100%, 100%, and 100% source-tens argmax; applying its
units-to-tens direction to tens-to-hundreds pairs gives 98.0%, 100%, and 100%
source-hundreds argmax.  Excluded models remain at 0% in every cross-task test.

Our current best interpretation is therefore a single, position-general
**carry-dependency residual direction** in the layer-1-post residual stream.
Successful models route it into the relevant next-digit readout; excluded
models encode an aligned direction but fail to use it at the carry-propagation
boundary described below.

## Layer-0 formation and transport

A separate leading direction at layer-0 residual post at `=` is also a useful
causal source state.  Its one-dimensional source-minus-target projection is
sufficient for source tens argmax in control seeds 1--3 at 100%, 96.5%, and
16.5%, and is null in excluded models.  Removing that projection from the full
source layer-0 residual removes the source effect in every control.

This layer-0 direction is not the same coordinate vector as the layer-1 units
direction (absolute cosine 0.01--0.09 in five of six models).  Per-context
central-difference transport nevertheless maps it substantially into the
layer-1 direction in control seeds 1 and 2 (mean cosine 0.86 and 0.81).
Comparable local alignment appears in excluded seeds 1 and 2 as well, so this
transport measurement does **not** explain the family difference on its own.
It reinforces the causal story's boundary: the failure is not simply that the
predicate cannot move locally out of layer 0; it is that excluded models do not
turn the resulting residual state into the next-digit counterfactual.

## Residual readout gain separates the families

We isolated the layer-1-post direction itself, oriented so that positive means
"more source-like" on independent source-minus-target examples, and injected
it into target units-token residuals.  A +1 matched-state-scale edit produces
source tens argmax in 99%, 100%, and 100% of controls, and 0% of all excluded
models.  The local tens-score derivative along the direction is positive in
every control (+0.22, +0.50, +0.42) and negative in every excluded model
(-0.59, -0.24, -0.74).  Central finite differences agree with the
autodiff JVP essentially perfectly.

On the raw-sum-nine dependency contexts, this is the sharpest current account
of the failure: excluded models contain and locally transport the direction,
but their downstream readout gives its source-oriented sign a negative local
effect on the dependent next digit.  Controls assign it positive readout gain.

## Gain profile through the residual stream

The sign distinction is already present at layer-1-post units and persists
through layers 2 and 3.  At every tested units-token residual site, all control
models have positive local derivative along that site's independently learned
source-oriented direction; all excluded models have negative derivative.
For example, layer-3 gains are +0.43, +0.43, and +0.41 in controls, versus
-0.17, -0.02, and -0.18 in excluded models.  Positive matched-scale injections
at layer 2 or 3 still yield 100% source-tens argmax in all controls and 0% in
all excluded models.

Thus, on raw-sum-nine dependency contexts, there is no evidence that a later
layer reverses an otherwise correct signal.  The control/excluded sign
difference is established by the first causal units-token residual write and
carried through the final residual readout.

## Boundary-specific failure, not ordinary carry failure

Sweeping the raw tens sum while injecting the same direction separates the
ordinary carry operation from carry propagation.  For raw tens sums 0--8, the
positive intervention produces the correct full `+10` source counterfactual
in excluded seeds 1 and 2 at 99--100% (and mostly succeeds in seed 3).  The
same models fail completely at raw tens sum 9: 0% source exact and 100% target
exact.  Controls retain high source exactness at the boundary (95%, 100%, and
100%).

So excluded models are not missing an incoming-carry variable, nor do they
globally read it with the wrong sign.  They use it to increment an ordinary
next digit, but fail at the **mod-10 wrap plus outgoing-carry** branch required
when that digit's raw sum is nine.  The negative gain results above should be
read as a local signature of this boundary-specific failure, not a claim about
all carry contexts.

## The missing operation is a conditional write, not final carry readout

We also injected the direction directly at the tens-token residual while
predicting the hundreds digit.  At raw tens sum nine, this makes excluded
models emit the source hundreds digit in 88--100% of cases (controls: 73--100%).
Thus excluded models can still **read** an explicitly supplied incoming-carry
state to increment the following digit.  The intervention is intentionally
not selective for raw sums 0--8, because it directly asserts a carry into the
hundreds column even when the teacher-forced tens digit does not warrant one.

Together with the raw-sum sweep, this identifies the missing computation more
precisely: controls conditionally *write a new carry state* after combining an
incoming carry with `raw sum = 9`; excluded models fail to create that state
endogenously.  Directly writing it at the next token bypasses their failure.

## Conditional-write interaction is distributed

A matched 2×2 residual factorial contrast (`incoming carry` × `raw tens is
9 rather than 8`) at layer-1-post units detects a non-additive interaction
state.  In controls, subtracting that interaction from the source raw-nine
state weakens its tens-score by 3.99, 6.91, and 9.30 logit units.  However, it still leaves the
source tens argmax on 84.5%, 100%, and 99.5% of examples; adding the
interaction to the raw-nine target is not sufficient either.

The conditional carry write is therefore not captured by one simple
difference-in-differences residual vector.  It is distributed within the
layer-1 residual transformation (or requires its interaction with the
surrounding residual state).  This is a useful constraint on the next circuit
search, not a completed minimal decomposition.

Learning a held-out low-rank subspace of these interaction vectors makes the
necessity result sharper.  A 16-dimensional learned interaction subspace
reduces control source-state score by 3.88, 6.68, and 9.12 logit units when
removed, compared with only 0.55, 0.61, and 1.02 for same-dimensional random
subspaces.  It still does not make source tens argmax fall below 89.5%, 100%,
and 98.5%, and adding it to targets is insufficient.  The gate is therefore a
compact but context-dependent distributed residual interaction, not an
independent additive feature.

Combining the two observed components does not rescue the excluded models:
carry-only, interaction-only, carry-plus-interaction, and carry-plus-random
subspace edits all yield 0% source exact in excluded seeds.  Controls are
already driven to the source answer by the carry component alone.  Thus the
raw-nine gate is not an additive conjunction of a carry vector and an
interaction subspace; it requires the native context-dependent residual update
itself.

## Context-dependent Jacobian gate

The layer-1 JVP distinguishes state transport from the failing conditional
operation.  We injected the same small layer-0 carry direction into otherwise
matched raw-tens-8 and raw-tens-9 contexts, then measured both its transport
into layer-1-post units and its tens-score derivative.

In excluded models, the transported projection remains positive and nearly
unchanged from raw 8 to raw 9 (for example 0.90 → 0.78, 0.45 → 0.45, and
0.21 → 0.21).  But the score JVP flips from positive at raw 8 (+0.25, +0.24,
+0.20) to negative at raw 9 (-0.54, -0.21, -0.18).  Controls retain positive
score JVPs in both contexts.  Thus raw sum nine does not block the residual
carry state from arriving; it changes how the surrounding layer-1 computation
maps that arriving state into the tens decision.

This is the most specific current localization: the absent behavior is a
**context-dependent nonlinear Jacobian gate** in the layer-1 residual
transformation, not missing state representation, state transport, or a later
carry-state readout.

Direct gain measurements at each residual site confirm the ordering.  At
layer-1-post units, excluded models have positive gain at raw 8 (+0.16, +0.12,
+0.42) and negative gain at raw 9 (-0.59, -0.23, -0.72); controls are positive
in both contexts.  The raw-nine negative effect is retained at later residual
sites, rather than corrected downstream.  This places the first causal sign
error at the layer-1 units residual boundary.

At the implementation level, copying either the layer-1 attention output or
the layer-1 MLP output from source to target at the units token is sufficient
to drive the raw-nine source tens digit in controls (attention: 100%, 100%,
65%; MLP: 98%, 96%, 100%) and null in excluded models.  At raw eight, the same
component patches work in both families.  This is compatible with a
context-specific failure in the layer-1 writes, but does not identify a unique
attention or MLP circuit: source-component patches can be sufficient through
redundancy, and zero ablations are broadly disruptive.

Path mediation resolves one important relationship.  At raw nine in controls,
a source attention-output patch flips the tens decision, but restoring only the
clean target MLP output at the same units token eliminates that flip (0% source
tens argmax in all three controls).  A source MLP-output patch alone remains
sufficient (97%, 98.5%, and 100%).  Therefore the attention-path contribution
to this boundary computation is mediated through the layer-1 MLP update; the
MLP output is the immediate write that carries the causal effect onward.

Copying the *complete* layer-1 units-token residual immediately after attention
and immediately before that MLP (`resid_mid`) sharpens the residual
localization.  At raw sum eight, this matched source-to-target patch produces
the source tens answer in every model.  At raw sum nine, it produces the source
answer in 100% of cases in every control but 0% in every excluded model (mean
score change 15.4--37.4 logits versus 0.09--2.89).  The family difference is
therefore already causally expressed in the residual state supplied to the
MLP; it is not wholly synthesized by the MLP from an interchangeable pre-MLP
state.  This is compatible with the mediation result: attention supplies
causal context, while the MLP's native update is still needed to express it.

A small source-oriented `resid_mid` perturbation is locally transported by the
MLP into its independently learned source-oriented output direction by a
similar positive amount in excluded raw-eight and raw-nine contexts
(1.17→1.19, 1.08→1.13, and 1.13→1.19 across seeds).  Yet its downstream
tens-score derivative flips from positive at raw eight (+0.53, +0.77, +1.09)
to negative at raw nine (-1.13, -0.54, -1.14); controls stay positive in both
contexts.  The defect is consequently not loss of local carry-direction
transport, but a context-dependent residual computation/readout at raw nine,
beginning no later than this pre-MLP state.  See `layer1_mlp_residual_gate.py`
and `artifacts/layer1_mlp_residual_gate/`.

## A singleton threshold hole

The threshold sweep around the boundary rules out a broader inability to
process raw carries.  With the same injected incoming-carry direction, excluded
models produce the correct source answer at raw tens sums 7, 8, 10, and 11
(91--100%, aside from ordinary seed-level noise), but fail at **exactly 9**:
0% source exact and 100% target exact in all three seeds.  Controls remain
highly successful across the same range.

The mechanistic defect is therefore exceptionally specific: the models have
both neighboring behaviors—ordinary increment below the threshold and ordinary
raw carry above it—but did not learn the interpolation point
`raw sum 9 + incoming carry → wrap and emit carry`.

## Raw-carry branch is not an additive rescue

We tested whether excluded models merely fail to invoke an otherwise available
ordinary raw-carry state.  At layer-1-post units, we formed the within-model
offset from matched `raw 10, no incoming carry` minus `raw 9, no incoming
carry` contexts and added it to the failing `raw 9, incoming carry` state.
It rescues 0% of excluded examples and disrupts controls as well.  Norm-matched
random offsets also do not rescue excluded models.

Thus the missing raw-nine behavior is not restored by adding a portable
"ordinary raw carry" residual vector.  The required computation is genuinely
context-sensitive and distributed, consistent with the Jacobian-gate result.

## Compositional carry chain

We tested prompts with two adjacent dependencies simultaneously: a units carry
causes a tens carry (`raw tens = 9`), which in turn causes a hundreds carry
(`raw hundreds = 9`).  During greedy decoding we injected the same learned
direction after the units answer, after the tens answer, or after both.

Controls compose the intervention.  A tens-only injection creates the
independent hundreds-carry counterfactual (mean answer shift +95 or +100);
the units-only intervention produces the correct source answer in 94.5%, 100%,
and 100% of controls; applying it at both positions gives the full source
answer in 99.5%, 100%, and 100%.  Excluded models remain exactly at the target
answer under every condition.

This is behavioral causal evidence that the position-general residual
direction can be applied repeatedly along a carry chain, rather than being a
single isolated-predicate control.

## Next discriminating experiment

Test whether the same direction generalizes across operand width or multiple
simultaneous dependencies.  Separately, identify the minimal layer-1
computation that converts the locally transported state into a usable
next-digit residual readout.

For complete protocols, per-seed values, and artifacts, see `RESULTS.md`.
