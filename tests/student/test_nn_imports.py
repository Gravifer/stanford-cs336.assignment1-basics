"""Tests for the top-level neural-network façade."""

import cs336_basics.nn as course_nn
from cs336_basics.nn.attention import MultiheadAttention, MultiheadSelfAttention, RotaryPositionalEmbedding
from cs336_basics.nn.feed_forward import SwiGLU, SwiGLU_delegate, SwiGLU_own_weights, SwiGLU_packed_input
from cs336_basics.nn.model import GPTDecoderLayer, TransformerBlock, TransformerLM
from cs336_basics.nn.modules import (
    DeltaLayer,
    Embedding,
    Linear,
    Module,
    RMSNorm,
    SiLU,  # ty: ignore[deprecated]
    SoftMax,
)


def test_nn_facade_reexports_all_supported_module_classes() -> None:
    expected = {
        "DeltaLayer": DeltaLayer,
        "Embedding": Embedding,
        "GPTDecoderLayer": GPTDecoderLayer,
        "Linear": Linear,
        "Module": Module,
        "MultiheadAttention": MultiheadAttention,
        "MultiheadSelfAttention": MultiheadSelfAttention,
        "RMSNorm": RMSNorm,
        "RotaryPositionalEmbedding": RotaryPositionalEmbedding,
        "SiLU": SiLU,  # ty: ignore[deprecated]
        "SoftMax": SoftMax,
        "SwiGLU": SwiGLU,
        "SwiGLU_delegate": SwiGLU_delegate,
        "SwiGLU_own_weights": SwiGLU_own_weights,
        "SwiGLU_packed_input": SwiGLU_packed_input,
        "TransformerBlock": TransformerBlock,
        "TransformerLM": TransformerLM,
    }

    assert course_nn.__all__ == list(expected)
    for name, implementation in expected.items():
        assert getattr(course_nn, name) is implementation


def test_nn_facade_exposes_implementation_namespaces() -> None:
    assert course_nn.analytics.__name__ == "cs336_basics.nn.analytics"
    assert course_nn.attention.__name__ == "cs336_basics.nn.attention"
    assert course_nn.feed_forward.__name__ == "cs336_basics.nn.feed_forward"
    assert course_nn.functional.__name__ == "cs336_basics.nn.functional"
    assert course_nn.initializer.__name__ == "cs336_basics.nn.initializer"
    assert course_nn.model.__name__ == "cs336_basics.nn.model"
    assert course_nn.modules.__name__ == "cs336_basics.nn.modules"
