import pytest
import torch
from jaxtyping import TypeCheckError

from cs336_basics.nn import functional as F
from cs336_basics.nn.attention import RotaryPositionalEmbedding
from cs336_basics.nn.modules import Embedding, Linear


pytestmark = pytest.mark.skipif(
    not hasattr(F.gelu, "__wrapped__"),
    reason="runtime annotation checks require the jaxtyping import hook",
)


def test_runtime_checker_rejects_invalid_literal() -> None:
    with pytest.raises(TypeCheckError):
        F.gelu(torch.randn(2, 3), "invalid")  # ty: ignore[invalid-argument-type]


def test_runtime_checker_rejects_wrong_linear_width() -> None:
    with pytest.raises(TypeCheckError):
        Linear(3, 4)(torch.randn(2, 5))


def test_runtime_checker_rejects_float_embedding_indices() -> None:
    with pytest.raises(TypeCheckError):
        Embedding(5, 3)(torch.randn(2, 4))


def test_runtime_checker_rejects_wrong_rope_width() -> None:
    rope = RotaryPositionalEmbedding(10_000.0, 8, 16)

    with pytest.raises(TypeCheckError):
        rope(torch.randn(2, 4, 6), torch.arange(4))


def test_runtime_hook_wraps_rope_once() -> None:
    depth = 0
    forward = RotaryPositionalEmbedding.forward
    while hasattr(forward, "__wrapped__"):
        depth += 1
        forward = forward.__wrapped__

    assert depth == 1
