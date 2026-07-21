# KV-Cache Readiness

## Scope

The current attention implementation is stateless. This note records the boundaries that should remain available for a later autoregressive KV cache without choosing a cache API or implementing generation ahead of course guidance.

The existing representation and grouped-attention decisions are documented separately in [Weight Packing and Activation Locality in Multi-Head Attention](mha-weights.md) and [Enabling Grouped-Query Attention](enabling-gqa.md). This note concerns only their implications for future cached execution.

## Ownership and stored tensors

An autoregressive KV cache belongs to a model's externally managed inference state, not permanently to `MultiheadAttention`. It represents one generation session across layers, batch or beam slots, and independently advancing sequence lengths. Ordinary training and full-sequence attention should remain stateless.

The cache should hold projected activations rather than raw hidden states. Its logical shapes are

$$
K_{cache}
\in
\mathbb{R}^{*\mathrm{cache\_batch}\times H_{kv}\times S\times d_k},
$$

$$
V_{cache}
\in
\mathbb{R}^{*\mathrm{cache\_batch}\times H_{kv}\times S\times d_v}.
$$

Equivalently, with named axes:

- K: `(*cache_batch, kv_head, sequence, d_k)`;
- V: `(*cache_batch, kv_head, sequence, d_v)`.

Cached K is stored after RoPE has been applied at its absolute positions. Existing K must not be rotated again, and V is never rotated. Under GQA or MQA, both tensors retain the KV-head count $H_{kv}$. Expanding them to the query-head count would forfeit the memory reduction and is not part of the grouped-attention contract.

## Absolute positions and causality

For each batch entry with previous cache length $L$ and a new chunk of length $S_{new}$, the new positions are

$$
L + [0, 1, \ldots, S_{new}-1].
$$

Those absolute positions apply to both new Q and new K in self-attention. Cache-aware causality must likewise compare absolute positions:

$$
\mathrm{key\_position} \le \mathrm{query\_position}.
$$

A one-token query at absolute position $t$ must see cached keys $0$ through $t$. This is equivalent to lower-right rectangular alignment for a contiguous prefix. It must not silently replace the generic course SDPA's existing upper-left rectangular causal convention.

## Future projected-attention seam

The useful conceptual decomposition is:

1. project only new input tokens;
2. split query and KV heads;
3. rotate new Q and K at absolute positions;
4. update the external cache or select complete K/V;
5. run projected grouped attention;
6. merge query heads and apply the output projection.

This boundary preserves packed projection opportunities for prefill while allowing a short-query decode path to consume projected cached K/V. It also keeps the existing grouped contraction available so cached GQA and MQA do not need materialized repeated K/V heads.

## Deliberately deferred choices

The public cache API, allocation and update policy, capacity and per-entry length representation, batch reordering, physical backend layout, and fused-kernel integration remain unspecified. They should be chosen when generation requirements or course guidance make the necessary caller-facing contract concrete.

In particular, this note does not propose paged attention, quantized caches, offloading, speculative decoding, or a serving engine. It records readiness constraints rather than an implementation plan.
