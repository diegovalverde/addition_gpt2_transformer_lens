"""Model construction for the addition experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from transformer_lens import HookedTransformer, HookedTransformerConfig

from addition_gpt.data import VOCAB


@dataclass(frozen=True)
class ModelConfig:
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_head: int = 32
    d_mlp: int = 512
    n_ctx: int = 30
    seed: int = 1

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def build_model(config: ModelConfig, device: str) -> HookedTransformer:
    """Build a randomly initialized GPT-2-style HookedTransformer."""
    hooked_config = HookedTransformerConfig(
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_model=config.d_model,
        d_head=config.d_head,
        d_mlp=config.d_mlp,
        n_ctx=config.n_ctx,
        d_vocab=len(VOCAB),
        d_vocab_out=len(VOCAB),
        act_fn="gelu",
        normalization_type="LN",
        positional_embedding_type="standard",
        seed=config.seed,
        device=device,
    )
    return HookedTransformer(hooked_config)
