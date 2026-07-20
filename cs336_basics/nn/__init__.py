"""Public neural-network API for the course implementations."""

from . import analytics as analytics
from . import attention as attention
from . import feed_forward as feed_forward
from . import functional as functional
from . import initializer as initializer
from . import model as model
from . import modules as modules
from .attention import MultiheadAttention, MultiheadSelfAttention, RotaryPositionalEmbedding
from .feed_forward import SwiGLU, SwiGLU_delegate, SwiGLU_own_weights, SwiGLU_packed_input
from .model import GPTDecoderLayer, TransformerBlock, TransformerLM
from .modules import DeltaLayer, Embedding, Linear, Module, RMSNorm, SiLU, SoftMax  # ty: ignore[deprecated]


__all__ = [
    "DeltaLayer",
    "Embedding",
    "GPTDecoderLayer",
    "Linear",
    "Module",
    "MultiheadAttention",
    "MultiheadSelfAttention",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SiLU",
    "SoftMax",
    "SwiGLU",
    "SwiGLU_delegate",
    "SwiGLU_own_weights",
    "SwiGLU_packed_input",
    "TransformerBlock",
    "TransformerLM",
]
