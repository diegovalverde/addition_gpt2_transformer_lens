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

## Carry-chain holdout: revised conclusion

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

IID uses evaluation seed 41 and the chain stream uses seed 42. The final result
is promising zero-shot composition evidence, but it is **one seed**, so do not
claim robustness yet.

## Recent commits

- `5006066` — records the seed-1 trajectory through step 2,000.
- `59fb93c` — adds fractional `--units-carry-chain-exposure` and its tests.
- `08b4ca2` — prior holdout replications and their documentation.

## Immediate next experiment: replication, not probing

Run the exact zero-exposure protocol for seeds 2 and 3. Do not tune based on
carry-chain accuracy. Evaluate every saved checkpoint on IID first, choose the
best IID checkpoint per seed, then evaluate that one checkpoint on carry-chain.

```bash
for seed in 2 3; do
  uv run python -u train.py --width 3 --steps 3000 --schedule-steps 3000 \
    --batch-size 256 --learning-rate 0.0003 --seed "$seed" \
    --units-carry-chain-exposure 0 --checkpoint-every 500 --device cpu \
    --checkpoint-dir "checkpoints/carry-chain-exposure-0-seed-$seed"
done
```

Use `evaluate.py` with `--examples 10000 --batch-size 256 --seed 41` for IID,
and the same arguments with `--split carry-chain --seed 42` for holdout. Record
greedy exact-answer accuracy in `RESULTS.md`.

## Decision rule after replication

- If seeds 2 and 3 also score high on the unseen chain: do **not** spend compute
  on the fractional exposure curve. The narrow holdout is not hard enough.
  Build a broader compositional holdout, such as excluding all examples where
  an incoming carry is necessary for the following column to carry, while
  retaining each local operation separately.
- If one or more seeds fail substantially: run exposures `0.001`, `0.01`, and
  `0.05` with the same protocol, then estimate the threshold with three seeds.
- Only after a reproducible success/failure contrast exists should agents run
  probes or activation patching to compare mechanisms between conditions.

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
