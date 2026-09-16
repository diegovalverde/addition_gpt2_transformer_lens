"""Tokenization and deterministic data generation for reversed-digit addition."""

from __future__ import annotations

from dataclasses import dataclass

import torch


VOCAB = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "=", "<bos>", "<eos>", "<pad>")
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}
ID_TO_TOKEN = {index: token for token, index in TOKEN_TO_ID.items()}
DIGIT_IDS = tuple(TOKEN_TO_ID[str(digit)] for digit in range(10))
BOS_ID = TOKEN_TO_ID["<bos>"]
EOS_ID = TOKEN_TO_ID["<eos>"]
PLUS_ID = TOKEN_TO_ID["+"]
EQUALS_ID = TOKEN_TO_ID["="]
PAD_ID = TOKEN_TO_ID["<pad>"]


def reverse_digits(value: int, width: int) -> list[int]:
    """Return exactly ``width`` least-significant-first decimal digits."""
    if value < 0 or value >= 10**width:
        raise ValueError(f"{value} does not fit in width {width}")
    return [int(character) for character in f"{value:0{width}d}"[::-1]]


def encode_addition(left: int, right: int, width: int) -> list[int]:
    """Encode a fixed-width addition problem with a width-plus-one answer."""
    result = left + right
    tokens = [BOS_ID]
    tokens.extend(reverse_digits(left, width))
    tokens.append(PLUS_ID)
    tokens.extend(reverse_digits(right, width))
    tokens.append(EQUALS_ID)
    tokens.extend(reverse_digits(result, width + 1))
    tokens.append(EOS_ID)
    return tokens


def decode_answer(tokens: torch.Tensor, width: int) -> int:
    """Decode reversed answer-digit tokens, excluding the trailing EOS token."""
    digit_tokens = tokens[: width + 1].tolist()
    digits = "".join(ID_TO_TOKEN[int(token)] for token in reversed(digit_tokens))
    if not digits.isdecimal():
        raise ValueError("answer contains a non-digit token")
    return int(digits)


def contains_carry(left: int, right: int, width: int) -> bool:
    """Return whether ordinary column addition creates any carry."""
    carry = 0
    for left_digit, right_digit in zip(reverse_digits(left, width), reverse_digits(right, width)):
        carry, _ = divmod(left_digit + right_digit + carry, 10)
        if carry:
            return True
    return False


def carry_targets(left: torch.Tensor, right: torch.Tensor, width: int) -> torch.Tensor:
    """Return the carry-out bit for every least-significant-first column."""
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("left and right must be matching one-dimensional tensors")
    targets = torch.zeros((left.shape[0], width), dtype=torch.float32)
    carry = torch.zeros_like(left)
    for column in range(width):
        place = 10**column
        left_digit = (left // place) % 10
        right_digit = (right // place) % 10
        carry = (left_digit + right_digit + carry >= 10).long()
        targets[:, column] = carry
    return targets


def has_units_carry_chain(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Identify cases where a units carry causes a tens carry because raw tens sum to nine."""
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("left and right must be matching one-dimensional tensors")
    units_carry = (left % 10) + (right % 10) >= 10
    raw_tens_sum = ((left // 10) % 10) + ((right // 10) % 10)
    return units_carry & raw_tens_sum.eq(9)


def answer_target_mask(tokens: torch.Tensor) -> torch.Tensor:
    """Mask shifted next-token labels to answer digits plus EOS only."""
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [batch, sequence]")
    equals_positions = (tokens == EQUALS_ID).nonzero(as_tuple=False)
    if equals_positions.shape[0] != tokens.shape[0]:
        raise ValueError("each sequence must contain exactly one equals token")
    positions = torch.arange(tokens.shape[1] - 1, device=tokens.device).unsqueeze(0)
    equals = equals_positions[:, 1].to(tokens.device).unsqueeze(1)
    return positions >= equals


@dataclass
class AdditionBatch:
    """One generated batch and metadata needed for split-level metrics."""

    tokens: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    carries: torch.Tensor


class AdditionBatchGenerator:
    """Generate independent, seedable fixed-width addition batches on CPU."""

    def __init__(self, width: int, seed: int) -> None:
        if width < 1:
            raise ValueError("width must be positive")
        self.width = width
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    @property
    def sequence_length(self) -> int:
        return 3 * self.width + 5

    def batch(
        self,
        batch_size: int,
        require_carry: bool = False,
        exclude_units_carry_chain: bool = False,
        require_units_carry_chain: bool = False,
        drop_probability: float = 0.0,
        min_operand: int = 0,
        max_operand: int | None = None,
    ) -> AdditionBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        limit = 10**self.width
        upper = limit if max_operand is None else max_operand
        if min_operand < 0 or upper > limit or min_operand >= upper:
            raise ValueError(f"operand range must be within [0, {limit})")
        if exclude_units_carry_chain and require_units_carry_chain:
            raise ValueError("a carry chain cannot be both excluded and required")
        if not 0 <= drop_probability < 1:
            raise ValueError("drop probability must be in [0, 1)")
        left = torch.randint(min_operand, upper, (batch_size,), generator=self.generator)
        right = torch.randint(min_operand, upper, (batch_size,), generator=self.generator)
        carries = torch.tensor(
            [contains_carry(int(a), int(b), self.width) for a, b in zip(left, right)],
            dtype=torch.bool,
        )
        chains = has_units_carry_chain(left, right)
        random_drop = (
            torch.rand(batch_size, generator=self.generator) < drop_probability
            if drop_probability
            else torch.zeros(batch_size, dtype=torch.bool)
        )
        invalid = (
            (require_carry & ~carries)
            | (exclude_units_carry_chain & chains)
            | (require_units_carry_chain & ~chains)
            | random_drop
        )
        while bool(invalid.any()):
            count = int(invalid.sum())
            left[invalid] = torch.randint(min_operand, upper, (count,), generator=self.generator)
            right[invalid] = torch.randint(min_operand, upper, (count,), generator=self.generator)
            carries = torch.tensor(
                [contains_carry(int(a), int(b), self.width) for a, b in zip(left, right)],
                dtype=torch.bool,
            )
            chains = has_units_carry_chain(left, right)
            random_drop = (
                torch.rand(batch_size, generator=self.generator) < drop_probability
                if drop_probability
                else torch.zeros(batch_size, dtype=torch.bool)
            )
            invalid = (
                (require_carry & ~carries)
                | (exclude_units_carry_chain & chains)
                | (require_units_carry_chain & ~chains)
                | random_drop
            )
        rows = [encode_addition(int(a), int(b), self.width) for a, b in zip(left, right)]
        return AdditionBatch(torch.tensor(rows, dtype=torch.long), left, right, carries)
