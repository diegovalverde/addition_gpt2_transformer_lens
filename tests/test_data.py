import torch

from addition_gpt.data import (
    AdditionBatchGenerator,
    EQUALS_ID,
    answer_target_mask,
    carry_targets,
    contains_carry,
    decode_answer,
    encode_addition,
    reverse_digits,
    has_carry_dependency_chain,
    has_units_carry_chain,
)
from evaluate import greedy_answers
from patch_carries import matched_carry_dependency_pairs, matched_units_carry_pairs


class PlannedAnswerModel(torch.nn.Module):
    """Minimal model that emits a known answer sequence for greedy-decoding tests."""

    def __init__(self, prompt_length: int, planned_tokens: torch.Tensor) -> None:
        super().__init__()
        self.prompt_length = prompt_length
        self.planned_tokens = planned_tokens

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        step = tokens.shape[1] - self.prompt_length
        logits = torch.full((*tokens.shape, 15), -float("inf"))
        logits[:, -1].scatter_(1, self.planned_tokens[:, step].unsqueeze(1), 0.0)
        return logits


def test_reverse_digits_preserves_fixed_width() -> None:
    assert reverse_digits(7, 3) == [7, 0, 0]
    assert reverse_digits(420, 3) == [0, 2, 4]


def test_encoding_handles_overflow() -> None:
    tokens = torch.tensor(encode_addition(99, 99, 2))
    equals = int((tokens == EQUALS_ID).nonzero()[0])
    assert decode_answer(tokens[equals + 1 :], 2) == 198


def test_answer_mask_starts_at_equals_logit() -> None:
    tokens = torch.tensor([encode_addition(7, 35, 3)])
    mask = answer_target_mask(tokens)
    equals = int((tokens[0] == EQUALS_ID).nonzero()[0])
    assert not bool(mask[0, :equals].any())
    assert bool(mask[0, equals:].all())


def test_carry_detection() -> None:
    assert contains_carry(7, 5, 2)
    assert not contains_carry(12, 34, 2)


def test_carry_targets_follow_each_decimal_column() -> None:
    targets = carry_targets(torch.tensor([7, 99]), torch.tensor([5, 1]), width=2)
    assert torch.equal(targets, torch.tensor([[1.0, 0.0], [1.0, 1.0]]))


def test_units_carry_chain_requires_incoming_units_carry() -> None:
    left = torch.tensor([19, 15, 18])
    right = torch.tensor([81, 94, 81])
    assert torch.equal(has_units_carry_chain(left, right), torch.tensor([True, False, False]))


def test_carry_dependency_chain_covers_each_adjacent_column() -> None:
    # 19 + 81 depends on the units carry for the tens carry; 190 + 810
    # depends on the tens carry for the hundreds carry.
    left = torch.tensor([19, 190, 18, 100])
    right = torch.tensor([81, 810, 81, 809])
    assert torch.equal(
        has_carry_dependency_chain(left, right, width=3),
        torch.tensor([True, True, False, False]),
    )


def test_generator_can_exclude_or_require_units_carry_chains() -> None:
    excluded = AdditionBatchGenerator(width=3, seed=1).batch(128, exclude_units_carry_chain=True)
    required = AdditionBatchGenerator(width=3, seed=2).batch(128, require_units_carry_chain=True)
    assert not bool(has_units_carry_chain(excluded.left, excluded.right).any())
    assert bool(has_units_carry_chain(required.left, required.right).all())


def test_generator_can_exclude_or_require_carry_dependency_chains() -> None:
    excluded = AdditionBatchGenerator(width=3, seed=1).batch(
        128, exclude_carry_dependency_chain=True
    )
    required = AdditionBatchGenerator(width=3, seed=2).batch(
        128, require_carry_dependency_chain=True
    )
    assert not bool(has_carry_dependency_chain(excluded.left, excluded.right, width=3).any())
    assert bool(has_carry_dependency_chain(required.left, required.right, width=3).all())


def test_zero_carry_chain_exposure_excludes_carry_chains() -> None:
    batch = AdditionBatchGenerator(width=3, seed=1).batch(
        128, units_carry_chain_exposure=0.0
    )
    assert not bool(has_units_carry_chain(batch.left, batch.right).any())


def test_partial_carry_chain_exposure_retains_some_but_not_all_chains() -> None:
    batch = AdditionBatchGenerator(width=3, seed=9).batch(
        4_000, units_carry_chain_exposure=0.05
    )
    chain_rate = has_units_carry_chain(batch.left, batch.right).float().mean().item()
    assert 0.001 < chain_rate < 0.005


def test_seeded_generators_match() -> None:
    first = AdditionBatchGenerator(width=3, seed=42).batch(8)
    second = AdditionBatchGenerator(width=3, seed=42).batch(8)
    assert torch.equal(first.tokens, second.tokens)


def test_generator_respects_operand_range() -> None:
    batch = AdditionBatchGenerator(width=3, seed=7).batch(64, min_operand=100, max_operand=200)
    assert bool((batch.left >= 100).all() and (batch.left < 200).all())
    assert bool((batch.right >= 100).all() and (batch.right < 200).all())


def test_greedy_answers_does_not_read_teacher_forced_answer_tokens() -> None:
    tokens = torch.tensor([encode_addition(7, 35, 3), encode_addition(99, 9, 3)])
    equals = int((tokens[0] == EQUALS_ID).nonzero()[0])
    expected = tokens[:, equals + 1 :]
    model = PlannedAnswerModel(equals + 1, expected)
    assert torch.equal(greedy_answers(model, tokens, width=3), expected)


def test_matched_pairs_toggle_only_units_carry_and_shift_answer_by_ten() -> None:
    _, _, target_sums, source_sums = matched_units_carry_pairs(width=3, examples=32, seed=9)
    assert torch.equal(source_sums, target_sums + 10)


def test_matched_dependency_pairs_toggle_only_the_units_digits() -> None:
    target, source, target_sums, source_sums = matched_carry_dependency_pairs(
        width=3, examples=32, seed=9
    )
    # Operand digits above units are held fixed, and only the source has a
    # carry dependency; matching units output keeps the answer change at ten.
    assert torch.equal(target[:, 2:4], source[:, 2:4])
    assert torch.equal(target[:, 6:8], source[:, 6:8])
    assert torch.equal(source_sums, target_sums + 10)
