"""Private attention contractions for explicit head-axis layouts."""

from typing import Literal

import einx
import torch

from cs336_basics.nn import functional as F


type _LayoutStrategy = Literal["head_before_sequence", "head_after_sequence"]
type _CausalStrategy = bool | Literal["torch", "compose", "noclone"]


def _grouped_attention_dimensions(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layout_strategy: _LayoutStrategy,
) -> tuple[int, int, int, int]:
    """Validate headed GQA inputs and return (group, seq_q, seq_k, d_k)."""
    if query.ndim < 3 or key.ndim < 3 or value.ndim < 3:
        raise ValueError("grouped query attention inputs must include sequence, head, and feature dimensions")
    if query.shape[:-3] != key.shape[:-3] or key.shape[:-3] != value.shape[:-3]:
        raise ValueError("grouped query attention inputs must have identical leading batch shapes")

    if layout_strategy == "head_before_sequence":
        query_heads, seq_q = query.shape[-3:-1]
        key_heads, seq_k = key.shape[-3:-1]
        value_heads, value_seq = value.shape[-3:-1]
    else:
        seq_q, query_heads = query.shape[-3:-1]
        seq_k, key_heads = key.shape[-3:-1]
        value_seq, value_heads = value.shape[-3:-1]

    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            f"query and key head widths must match, got {query.shape[-1]} and {key.shape[-1]}"
        )
    if key_heads != value_heads:
        raise ValueError(f"key and value head counts must match, got {key_heads} and {value_heads}")
    if seq_k != value_seq:
        raise ValueError(f"key and value sequence lengths must match, got {seq_k} and {value_seq}")
    if key_heads <= 0 or query_heads % key_heads != 0:
        raise ValueError(
            f"query head count {query_heads} must be divisible by key/value head count {key_heads}"
        )
    return query_heads // key_heads, seq_q, seq_k, query.shape[-1]


def _grouped_scaled_dot_product_attention_head_before_sequence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: _CausalStrategy = False,
    scale: float | None = None,
    *,
    attn_mask: torch.Tensor | None = None,
    attn_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run production GQA/MQA on (*batch, head, sequence, feature) tensors."""
    group, seq_q, seq_k, d_k = _grouped_attention_dimensions(
        query,
        key,
        value,
        layout_strategy="head_before_sequence",
    )
    grouped_query = einx.id(
        "batch... (kv_head group) query d_k -> batch... kv_head group query d_k",
        query,
        group=group,
    )
    scores = einx.dot(
        "batch... kv_head group query [d_k], batch... kv_head key [d_k] "
        "-> batch... (kv_head group) query key",
        grouped_query,
        key,
    ) * (1 / d_k**0.5 if scale is None else scale)
    attn_weights = F._attention_weights(
        scores,
        seq_q,
        seq_k,
        mask,
        dropout_p,
        is_causal,
        attn_mask=attn_mask,
        attn_bias=attn_bias,
    )
    grouped_weights = einx.id(
        "batch... (kv_head group) query key -> batch... kv_head group query key",
        attn_weights,
        group=group,
    )
    return einx.dot(
        "batch... kv_head group query key, batch... kv_head key d_v "
        "-> batch... (kv_head group) query d_v",
        grouped_weights,
        value,
    )


def _grouped_scaled_dot_product_attention_head_after_sequence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: _CausalStrategy = False,
    scale: float | None = None,
    *,
    attn_mask: torch.Tensor | None = None,
    attn_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run experimental GQA/MQA on (*batch, sequence, head, feature) tensors."""
    group, seq_q, seq_k, d_k = _grouped_attention_dimensions(
        query,
        key,
        value,
        layout_strategy="head_after_sequence",
    )
    grouped_query = einx.id(
        "batch... query (kv_head group) d_k -> batch... kv_head group query d_k",
        query,
        group=group,
    )
    scores = einx.dot(
        "batch... kv_head group query [d_k], batch... key kv_head [d_k] "
        "-> batch... (kv_head group) query key",
        grouped_query,
        key,
    ) * (1 / d_k**0.5 if scale is None else scale)
    attn_weights = F._attention_weights(
        scores,
        seq_q,
        seq_k,
        mask,
        dropout_p,
        is_causal,
        attn_mask=attn_mask,
        attn_bias=attn_bias,
    )
    grouped_weights = einx.id(
        "batch... (kv_head group) query key -> batch... kv_head group query key",
        attn_weights,
        group=group,
    )
    return einx.dot(
        "batch... kv_head group query key, batch... key kv_head d_v "
        "-> batch... query (kv_head group) d_v",
        grouped_weights,
        value,
    )


def _scaled_dot_product_attention_head_after_sequence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: _CausalStrategy = False,
    scale: float | None = None,
    *,
    attn_mask: torch.Tensor | None = None,
    attn_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run experimental ordinary attention with the head axis after sequence."""
    assert query.shape[-1] == key.shape[-1], (
        f"Cannot determine d_k; queries are {query.shape[-1]}-D while keys are {key.shape[-1]}-D."
    )
    seq_q, seq_k, d_k = query.shape[-3], key.shape[-3], key.shape[-1]
    scores = einx.dot("... q head [d_k], ... k head [d_k] -> ... head q k", query, key) * (
        1 / d_k**0.5 if scale is None else scale
    )
    attn_weights = F._attention_weights(
        scores,
        seq_q,
        seq_k,
        mask,
        dropout_p,
        is_causal,
        attn_mask=attn_mask,
        attn_bias=attn_bias,
    )
    return einx.dot("... head q k, ... k head d_v -> ... q head d_v", attn_weights, value)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: _CausalStrategy = False,
    scale: float | None = None,
    *,
    layout_strategy: _LayoutStrategy,
) -> torch.Tensor:
    """Dispatch headed attention by activation layout and Q/KV head counts.

    The head-before-sequence GQA/MQA contraction is the production grouped
    path. Both head-after-sequence contractions are private experimental
    benchmark variants.
    """
    head_axis = -3 if layout_strategy == "head_before_sequence" else -2
    if query.shape[head_axis] == key.shape[head_axis]:
        attention = (
            F.scaled_dot_product_attention
            if layout_strategy == "head_before_sequence"
            else _scaled_dot_product_attention_head_after_sequence
        )
    else:
        attention = (
            _grouped_scaled_dot_product_attention_head_before_sequence
            if layout_strategy == "head_before_sequence"
            else _grouped_scaled_dot_product_attention_head_after_sequence
        )
    return attention(
        query,
        key,
        value,
        mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
