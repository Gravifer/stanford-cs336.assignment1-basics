"""Attention layers and rotary positional embeddings.

The package façade exposes the public attention API while implementations live
in focused modules that can also be imported directly.
"""

from .modules import MultiheadAttention, MultiheadSelfAttention
from .rope import RotaryPositionalEmbedding

__all__ = [
    "MultiheadAttention",
    "MultiheadSelfAttention",
    "RotaryPositionalEmbedding",
]
