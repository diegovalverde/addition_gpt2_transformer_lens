# Addition GPT-2 / TransformerLens experiment

## Goal

Train a small decoder-only transformer from scratch to add non-negative integers,
then test whether it has learned an addition procedure rather than memorizing
training pairs. The project is deliberately separate from the Othello experiments.

The first milestone is reliable fixed-width addition with carries. The second is
out-of-distribution evaluation on operands, carry patterns, and lengths withheld
from training.

## Why this representation

A causal transformer predicts output left-to-right. Ordinary decimal addition
propagates carries right-to-left, which makes it a needlessly difficult first
experiment. We therefore reverse each number's digits before tokenizing.

For example, the human-readable problem `12 + 34 = 46` becomes:

```text
2 1 + 4 3 = 6 4 <eos>
```

The first answer digit is now determined by the first operand digits and a carry
from the immediately previous position. This makes a learned algorithm plausible
with a small causal model, while retaining non-trivial carry handling.

All operands initially use a fixed width. For width three, `7 + 35 = 42` is:

```text
7 0 0 + 5 3 0 = 2 4 0 0 <eos>
```

The answer must have `width + 1` digits so that overflow is represented. The
extra leading zero in the reversed representation is included consistently.

## Token vocabulary

Use individual digit tokens, never an atomic token per integer:

```text
0 1 2 3 4 5 6 7 8 9 + = <bos> <eos> <pad>
```

Atomic-number tokens would turn the experiment into a bounded lookup-table task
and do not support meaningful tests at unseen lengths.

The training target is next-token cross-entropy, but loss is masked before the
`=` token. This focuses the model on computing the answer rather than spending
most of its capacity predicting independently sampled operands.

## Model: AdditionGPT

This is GPT-2-style in architecture, not a fine-tuned OpenAI GPT-2 checkpoint.
Use `transformer_lens.HookedTransformer` so hooks and activations are available
for later mechanistic-interpretability work.

| Setting | Initial value |
| --- | ---: |
| Layers | 4 |
| Attention heads | 4 |
| Model width (`d_model`) | 128 |
| Head width (`d_head`) | 32 |
| MLP width (`d_mlp`) | 512 |
| Context length | 30 (enough for width 8 plus markers and overflow) |
| Positional embeddings | learned absolute |
| Activation | GELU |
| Dropout | 0.0 |
| Parameters | roughly 0.8M |

No dropout is intentional: the data generator supplies effectively unlimited,
independent examples, and deterministic activations simplify later analysis.

## Training data

Generate examples on demand from a seeded `torch.Generator`; do not create a
large static dataset. Each batch samples operands uniformly from the configured
range and uses exact Python integer arithmetic for the target.

Initial curriculum:

1. Train 2-digit, zero-padded operands for a quick end-to-end smoke test.
2. Train 3-digit operands, including all carry cases.
3. Train widths 1--5 uniformly mixed, retaining positional room for width 8.

For the fixed-width runs, permit all values `0..10**width - 1`. A validation
stream must use a distinct seed. Do not split a finite saved dataset: the
on-demand generator gives a cleaner independence guarantee.

## Optimization

| Setting | Initial value |
| --- | ---: |
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Betas | `(0.9, 0.95)` |
| Weight decay | `0.01` |
| Batch size | 256 (reduce if memory requires it) |
| Training steps | 20,000 for 3-digit baseline |
| LR schedule | 500-step linear warmup, cosine decay to `1e-4` |
| Precision | float32 initially |
| Gradient clipping | global norm `1.0` |

Run on Apple Silicon through PyTorch MPS when available. Keep the first version
in float32; MPS mixed precision can be tried only after a correct baseline.

TransformerLens currently warns that PyTorch MPS can silently produce incorrect
results. Use MPS for exploratory training only, and reproduce any metric used in
analysis on CPU from the same checkpoint and fixed evaluation seed. Do not set
`TRANSFORMERLENS_ALLOW_MPS=1` merely to hide this warning.

## Success metrics

Report more than token loss:

- **Exact-answer accuracy:** every answer digit, including overflow, is correct.
- **Digit accuracy:** accuracy only after `=`.
- **Carry accuracy:** exact-answer accuracy on examples containing at least one
  carry.
- **Maximum-length accuracy:** evaluate each operand width separately.

Greedy decoding should be evaluated autoregressively after the `=` marker. Also
evaluate teacher-forced answer logits, but never present that as generation
accuracy.

## Required test splits

The experiment is only persuasive if it distinguishes different generalization
claims.

| Split | Train | Evaluate | Interpretation |
| --- | --- | --- | --- |
| IID | 3-digit operands | new seeded 3-digit operands | baseline fitting |
| Held-out pairs | fixed finite sampled pairs | unseen pairs in same range | pair memorization check |
| Carry | all pairs | subset with one or more carries | local algorithm test |
| Long length | widths 1--3 | width 4 or more | length extrapolation |
| Range | operands below 100 | 100--999 | numerical-range extrapolation |

Do not expect vanilla learned absolute positional embeddings to generalize far
beyond their trained width. Failure on the long-length split is a result, not a
reason to change the test.

## Reproducible workflow

Once the project files are implemented, reproduce the 3-digit baseline as
follows:

```bash
cd ~/workspace/projects/addition_gpt2_transformer_lens
uv sync
uv run python train.py --width 3 --steps 20000 --batch-size 256 --seed 1 --device mps
uv run python evaluate.py --checkpoint checkpoints/widths-3-seed-1.pt --width 3 --split iid --seed 2 --device mps
uv run python evaluate.py --checkpoint checkpoints/widths-3-seed-1.pt --width 3 --split carry --seed 3 --device mps
```

For a CPU-only machine, replace `--device mps` with `--device cpu`. The training
program must save the full model configuration, vocabulary mapping, optimizer
state, random seed, package versions, and git revision with every checkpoint.

## Implementation order

1. Add `pyproject.toml` with PyTorch and TransformerLens dependencies.
2. Add a deterministic tokenizer/data generator with unit tests for formatting,
   padding, overflow, and carries.
3. Add `train.py`, including answer-only masked loss and checkpoint metadata.
4. Add `evaluate.py` with teacher-forced and greedy metrics for every split.
5. Train the 2-digit smoke test, then the 3-digit baseline.
6. Add a notebook or script for activation caching and carry-direction probes.

## Non-goals for version one

- Natural-language math prompting.
- Pretrained language-model fine-tuning.
- Claims of arbitrary-length addition.
- Changing the token representation mid-comparison.

These constraints keep the first results interpretable and inexpensive enough to
run repeatedly on a MacBook.
