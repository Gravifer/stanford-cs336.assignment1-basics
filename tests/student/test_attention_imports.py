"""Import-contract tests for the attention package façade."""

from cs336_basics.nn.attention import (
    MultiheadAttention,
    MultiheadSelfAttention,
    RotaryPositionalEmbedding,
)
from cs336_basics.nn.attention.modules import (
    MultiheadAttention as DirectMultiheadAttention,
)
from cs336_basics.nn.attention.modules import (
    MultiheadSelfAttention as DirectMultiheadSelfAttention,
)
from cs336_basics.nn.attention.rope import (
    RotaryPositionalEmbedding as DirectRotaryPositionalEmbedding,
)


def test_attention_package_exports_direct_module_classes() -> None:
    """Package and direct imports resolve to the same class objects."""
    assert MultiheadAttention is DirectMultiheadAttention
    assert MultiheadSelfAttention is DirectMultiheadSelfAttention
    assert RotaryPositionalEmbedding is DirectRotaryPositionalEmbedding
