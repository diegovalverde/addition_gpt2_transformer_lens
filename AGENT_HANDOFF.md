# Addition Transformer: Agent Handoff

## Goal

Determine whether a small, four-layer GPT-2-style transformer trained from
scratch performs digit addition through compositional carry computation rather
than literal lookup. This project uses TransformerLens and reversed decimal
digits: `<bos> a_ones ... + b_ones ... = sum_ones ... <eos>`.

## Verified current state

- The model has 4 layers, `d_model=128`, 4 heads, `d_head=32`, and
  `d_mlp=512`; training uses answer-only next-token cross-entropy.
- CPU is the reference device. TransformerLens warns that MPS may be silently
  incorrect, so do not use MPS results as final evidence.
- Fixed-width, three-digit addition reaches about 99.7% exact after 1,000 CPU
  steps; it generalizes poorly to unseen four-digit width. Directly trained
  four-digit models reach about 99.9%, so this is a length-generalization
  failure, not a capacity failure.
- Pre-answer linear probes decode the units carry perfectly and later carries
  increasingly well. Matched activation patching at layer 0, `=` position can
  switch a target answer to its matched `sum + 10` source answer in 99.2% of
  examples. Probe directions themselves were not clean causal controls.

See `RESULTS.md` for all established numbers and methods.

## Carry-chain holdout: resolved conclusion

The held-out pattern is: units digits create a carry and raw tens digits sum to
9, so the incoming units carry causes the tens column to carry. It occurs in
about 4.5% of uniform three-digit operand pairs.

Earlier runs that excluded this pattern and used the old optimization setup had
large, variable accuracy drops. A stabilized run instead excluded it completely
(`--units-carry-chain-exposure 0`) while using CPU, batch size 256, 3,000 steps,
cosine schedule length 3,000, and learning rate `3e-4`.

| Seed 1 checkpoint | IID exact, 10k | Held-out chain exact, 10k |
| ---: | ---: | ---: |
| 500 | 98.74% | not evaluated |
| 1,000 | 99.74% | not evaluated |
| 1,500 | 99.86% | 97.14% |
| 2,000 | 99.97% | 99.41% |
| 3,000 | 99.99% | 99.91% |

IID uses evaluation seed 41 and the chain stream uses seed 42. Seeds 2 and 3
completed the same protocol: their IID-selected checkpoints scored 100.00% and
99.85%, respectively, on the chain stream. The narrow holdout is therefore
solved robustly and is not diagnostic of a literal lookup mechanism.

## Recent commits

- `5006066` — records the seed-1 trajectory through step 2,000.
- `59fb93c` — adds fractional `--units-carry-chain-exposure` and its tests.
- `08b4ca2` — prior holdout replications and their documentation.
- `4bc1ddb` — adds the broader carry-dependency holdout, control, and tests.

## Broader carry-dependency holdout: replicated result

The broader holdout excludes every example where an incoming carry is necessary
for a following column to carry: the incoming carry is one and the following raw
digit pair sums to nine. It contains both units-to-tens and tens-to-hundreds
dependencies and occurs in exactly 9% of uniform three-digit pairs.

All excluded models (seeds 1--3) selected their step-1,000 checkpoint by IID
greedy exact accuracy: 90.55% IID and 0.00% carry-dependency exact. Matched 9%
random-resampling controls selected IID-optimal checkpoints of 500, 1,500, and
1,500 for seeds 1--3, respectively; all three score 100.00% IID and 100.00%
carry-dependency exact. Evaluation streams contain 10,000 examples with seeds
41 (IID) and 42 (holdout).

## Immediate next experiment: predicate-conditioned causal test

IID carry probes were nearly indistinguishable between excluded and control
models, and legacy units-carry activation patches were seed-variable. They do
not isolate the broader holdout because their source/target pairs do not control
the later carry-dependency predicate.

Build matched source/target pairs that differ only in whether one selected
incoming carry causes the following column to carry, while fixing all other
columns and answer digits where possible. Pre-specify the layer/position grid,
run the same grid over all seeds, and compare excluded models with their
same-seed controls. Treat whole-residual replacement as a localization screen;
only then test more targeted components or directions.

## Implementation notes

- `addition_gpt/data.py`: data, carry targets, and chain predicate. The
  exposure control performs conditional resampling: non-chain examples are
  always kept; sampled chain examples are retained with probability `p`.
  Thus `p=0.05` means roughly 0.24% chain examples in the final stream, not 5%.
- `train.py`: stores model, optimizer, scheduler, and generator state in every
  checkpoint. Resume exactly with `--resume-from`; preserve the other training
  arguments.
- `evaluate.py`: teacher-forced and greedy exact metrics. Greedy exact is the
  primary metric.
- `probe_carries.py`, `intervene_carries.py`, and `patch_carries.py`: existing
  mechanistic tools. Do not treat probe decodability as causality.
- Tests: `uv run pytest` should pass (13 tests after `59fb93c`).

## Guardrails

- Work in this repository and use `uv`, never `pip`.
- Keep checkpoints and artifacts untracked; commit code and markdown results.
- Do not compare checkpoint scores selected on the carry-chain split; selection
  must use IID only.
- Treat the existing one-seed success as evidence that optimization matters, not
  proof of a fully general addition algorithm.
