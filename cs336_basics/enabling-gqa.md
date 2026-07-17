# Enabling Grouped-Query Attention

## Overview

Multi-head attention ordinarily gives every query head its own key and value heads. Grouped-query attention reduces the number of key/value heads while retaining the full query-head count. Several consecutive query heads then share one key head and one value head. Multi-query attention is the endpoint where every query head shares the same single key/value head.

Let

$$
H_q = \text{number of query heads},
\qquad
H_{kv} = \text{number of key/value heads}.
$$

The supported relation is

$$
H_q \bmod H_{kv} = 0,
$$

and the number of query heads sharing each key/value head is

$$
G = \frac{H_q}{H_{kv}}.
$$

This yields three configurations:

- ordinary MHA when $H_{kv}=H_q$;
- proper GQA when $1<H_{kv}<H_q$;
- MQA when $H_{kv}=1<H_q$.

The one-head case $H_q=H_{kv}=1$ follows the ordinary MHA path because no sharing distinction remains.

## Module interface

[`MultiheadAttention`](nn/attention/__init__.py#sym:MultiheadAttention) keeps `num_heads` as the canonical query-head count. The keyword-only `num_q_heads` is a mutually exclusive, read-only alias that makes the distinction explicit at call sites. `num_kv_heads=None` resolves to the query-head count.

```python
# Eight query heads and eight key/value heads: ordinary MHA
MultiheadAttention(512, 8)

# Eight query heads sharing two key/value heads: GQA
MultiheadAttention(512, 8, num_kv_heads=2)

# Eight query heads sharing one key/value head: MQA
MultiheadAttention(512, num_q_heads=8, num_kv_heads=1)
```

[`MultiheadSelfAttention`](nn/attention/__init__.py#sym:MultiheadSelfAttention) exposes the same head-count controls. Its course-native positional arguments remain `d_model`, `num_heads`, `d_k`, `d_v`, and `dropout`. Calls using the `num_q_heads` alias keep the remaining dimensions keyword-only, avoiding an ambiguous positional gap where `num_heads` would otherwise appear.

The head counts are constructor state because they determine parameter shapes. A forward-time Boolean switch would not say how many key/value heads the module owns.

## Projection dimensions and packing

For per-head query/key width $d_k$ and value width $d_v$, the logical projection matrices are

$$
W_Q \in \mathbb{R}^{H_qd_k\times d_{\mathrm{model}}},
$$

$$
W_K \in \mathbb{R}^{H_{kv}d_k\times d_{\mathrm{key\ input}}},
\qquad
W_V \in \mathbb{R}^{H_{kv}d_v\times d_{\mathrm{value\ input}}}.
$$

Every query head still produces an attended value vector, so the output projection consumes

$$
H_qd_v
$$

features. The reduced stored width $H_{kv}d_v$ applies to the V projection before attention, not to the attended result after sharing.

When query, key, and value have the same raw input width, their unequal output segments remain packable into one rectangular parameter:

$$
W_{QKV}
\in
\mathbb{R}^{(H_qd_k+H_{kv}d_k+H_{kv}d_v)\times d_{\mathrm{model}}}.
$$

Consequently, packed self-attention still computes Q, K, and V in one projection matrix multiplication. Conditional packing continues to depend on raw input widths rather than projected widths. Separate raw key or value widths retain the existing three-parameter representation.

Initialization follows the same logical segments in either representation. Q uses fan-out $H_qd_k$, K uses $H_{kv}d_k$, V uses $H_{kv}d_v$, and the output projection uses input width $H_qd_v$. Loading delegated or separate weights into packed storage uses those same boundaries. Loading an ordinary MHA checkpoint into a GQA module is intentionally left to an explicit model-conversion policy: selecting or averaging key/value heads changes parameter meaning rather than storage syntax.

## Grouped attention as a view and contraction rule

After projection and head decomposition, the head-before-sequence layout is

$$
Q\in\mathbb{R}^{*B\times H_q\times L\times d_k},
$$

$$
K\in\mathbb{R}^{*B\times H_{kv}\times S\times d_k},
\qquad
V\in\mathbb{R}^{*B\times H_{kv}\times S\times d_v}.
$$

The query-head axis is reinterpreted as two adjacent logical axes:

$$
H_q \longleftrightarrow (H_{kv},G).
$$

This is a view of the existing query activation. The score contraction is then

$$
(*B,H_{kv},G,L,d_k)
\mathbin{@}
(*B,H_{kv},S,d_k)^\mathsf{T}
\longrightarrow
(*B,H_{kv},G,L,S).
$$

Before masking and softmax, $(H_{kv},G)$ is joined back into the ordinary query-head axis:

$$
(*B,H_q,L,S).
$$

The existing mask, causal composition, softmax, and dropout core therefore receives exactly the score layout it already understands. In particular, a head-dependent mask is indexed by query head, because each score row belongs to a query head even when its K/V source is shared.

After softmax, the query-head axis is viewed as $(H_{kv},G)$ once more for the value contraction:

$$
(*B,H_{kv},G,L,S)
\mathbin{@}
(*B,H_{kv},S,d_v)
\longrightarrow
(*B,H_q,L,d_v).
$$

The head-after-sequence path performs the same logical operations while returning a contiguous `(*batch, query, query_head, d_v)` result. Grouping `(query_head, d_v)` for the output projection remains a view in that pipeline.

The implementation expresses these transformations through grouped `einx` axes. It does not materialize repeated K or V tensors. An explicit `repeat_interleave` implementation remains valuable as a transparent reference because it states the sharing rule directly:

```python
queries_per_kv_head = num_q_heads // num_kv_heads
expanded_k = k.repeat_interleave(queries_per_kv_head, dim=head_axis)
expanded_v = v.repeat_interleave(queries_per_kv_head, dim=head_axis)
```

The test suite compares forward values and gradients against this reference.

## Generic SDPA and head-aware dispatch

The course-facing [`scaled_dot_product_attention`](nn/functional.py#sym:scaled_dot_product_attention) accepts

$$
(*\mathrm{batch},\mathrm{query},d_k),
\quad
(*\mathrm{batch},\mathrm{key},d_k),
\quad
(*\mathrm{batch},\mathrm{key},d_v).
$$

Its leading axes are deliberately generic. A three-dimensional input may use its leading axis as a batch axis and need not contain heads at all. Inferring GQA there would require silently assigning a head interpretation to one member of `*batch`.

The module has already created explicit query-head and KV-head axes, so it dispatches unequal head counts to private, layout-specific grouped contractions. Equal counts continue through the ordinary SDPA paths. This keeps the generic mathematical operation free of a mode flag while letting the module provide the structural information required by GQA.

PyTorch takes a different compatibility-oriented approach: its SDPA contract names axis `-3` as the head axis and uses `enable_gqa=True` to opt into unequal head counts. Its documented mathematical reference expands K and V. ([PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)) FlashAttention accepts explicitly headed Q, K, and V tensors and derives MQA/GQA directly when K/V have fewer heads. ([FlashAttention interface](https://github.com/Dao-AILab/flash-attention#how-to-use-flashattention)) NVIDIA Transformer Engine configures attention modules with a separate number of GQA groups. ([Transformer Engine attention API](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html))

These APIs serve different abstraction boundaries. The presence or absence of a Boolean switch does not change the sharing rule; it determines where the head-axis interpretation is declared.

## RoPE behavior

RoPE still applies to Q and K and never to V. Q and K have the same per-head feature width but different head counts under GQA, so they cannot use the equal-shaped packed-QK rotation view.

Packed self-projection remains active, but it exposes combined QK rotation metadata only for ordinary MHA. GQA and MQA use the existing separate Q/K RoPE path. Attention positions are batch-aligned by default, while an explicit head layout preserves head-dependent selection. Generic cross-attention may choose the query and key interpretations independently; self-attention uses one shared interpretation. The private forced-stacking benchmark strategy rejects unequal Q/K shapes.

Separate application may select equivalent cache data twice when Q and K share positions. Removing that duplication would require generalizing shared positional selection across unequal target head shapes, especially for head-dependent position tensors. It is an independent optimization rather than part of the logical GQA contract.

## Performance boundary

GQA and MQA reduce K/V parameter size, activation width, and KV-cache size. Realizing their largest runtime benefits typically depends on fused attention kernels and cache-aware lower-level layouts. This implementation instead emphasizes a direct correspondence among parameter dimensions, head grouping, masks, RoPE, and output reconstruction.

The grouped contractions avoid the most obvious Python-level penalty—materializing repeated K/V tensors—but they are not claimed to match a fused GQA kernel. Backend lowering, compilation, device architecture, dtype, sequence length, and memory layout remain decisive. The implementation is consequently useful as a logical dispatch reference and a correctness baseline for later backend integration.
