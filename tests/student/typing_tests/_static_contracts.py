from typing import Never, assert_type

import torch

from cs336_basics.nn.attention import MultiheadAttention, MultiheadSelfAttention, RotaryPositionalEmbedding
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
    assert_type(
        MultiheadSelfAttention(
            8,
            num_q_heads=2,
            num_kv_heads=1,
            _layout_strategy="head_after_sequence",
            _qk_execution_strategy="stacked",
        ),
        MultiheadSelfAttention,
    )
    assert_type(MultiheadAttention(8, 2, num_kv_heads=1), MultiheadAttention)
    assert_type(MultiheadAttention(8, num_q_heads=2, num_kv_heads=1), MultiheadAttention)


def check_rope_execution_contracts() -> None:
    rope = RotaryPositionalEmbedding(10_000.0, 8, 16, use_matrix_form=False)
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn(2, 3, 4, 8)
    assert_type(rope.forward(query, torch.arange(4)), torch.Tensor)
    assert_type(rope.apply_qk(query, key), tuple[torch.Tensor, torch.Tensor])


def check_attention_position_layout_aliases() -> None:
    attention = MultiheadAttention(8, 2)
    self_attention = MultiheadSelfAttention(8, 2)
    x = torch.randn(2, 4, 8)
    positions = torch.arange(4).expand(2, -1)
    batch_mask = torch.ones(2, 4, 4, dtype=torch.bool)
    head_mask = torch.ones(2, 4, 4, dtype=torch.bool)
    assert_type(
        attention.forward(x, x, x, query_positions=positions, key_positions=positions, position_layout="batch"),
        torch.Tensor,
    )
    assert_type(
        attention.forward(
            x,
            x,
            x,
            query_positions=positions,
            key_positions=positions,
            query_position_layout="batch",
            key_position_layout="head",
        ),
        torch.Tensor,
    )
    assert_type(attention.forward(x, x, x, batch_mask, mask_layout="batch"), torch.Tensor)
    assert_type(attention.forward(x, x, x, head_mask, mask_layout="head"), torch.Tensor)
    assert_type(self_attention.forward(x, token_positions=positions, position_layout="batch"), torch.Tensor)
    assert_type(self_attention.forward(x, batch_mask, mask_layout="batch"), torch.Tensor)
