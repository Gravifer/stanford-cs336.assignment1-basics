from typing import Never, assert_type

import torch

from cs336_basics.nn.attention import MultiheadSelfAttention
from cs336_basics.nn.feed_forward import SwiGLU, SwiGLU_packed_input
from cs336_basics.nn.modules import Embedding, Linear, RMSNorm


def check_embedding_return() -> None:
    assert_type(Embedding.from_pretrained(torch.randn(5, 3)), Embedding)


def check_linear_bias() -> None:
    assert_type(Linear(3, 4).bias, Never)


def check_rmsnorm_weight() -> None:
    assert_type(RMSNorm(4).weight, torch.Tensor | None)


def check_swiglu_export() -> None:
    assert_type(SwiGLU(4, 8), SwiGLU_packed_input)


def check_self_attention_constructor_aliases() -> None:
    assert_type(MultiheadSelfAttention(8, 2, 4, 4), MultiheadSelfAttention)
    assert_type(
        MultiheadSelfAttention(
            embed_dim=8,
            num_heads=2,
            qk_head_dim=4,
            value_head_dim=4,
        ),
        MultiheadSelfAttention,
    )
