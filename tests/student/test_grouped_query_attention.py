from collections.abc import Callable
from typing import Any, Literal, cast

import pytest
import torch

from cs336_basics.nn import functional as F
from cs336_basics.nn import initializer as init
from cs336_basics.nn.attention import MultiheadAttention, MultiheadSelfAttention, RotaryPositionalEmbedding


def _expanded_kv_reference(
    module: MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    is_causal: bool = False,
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
    query_position_layout: Literal["batch", "head"] = "batch",
    key_position_layout: Literal["batch", "head"] = "batch",
) -> torch.Tensor:
    projected_q, projected_k, projected_v, projected_qk = module._in_projection(query, key, value)
    q, k, v, packed_qk = module._split_heads(projected_q, projected_k, projected_v, projected_qk)
    if module.rope is not None:
        q, k = module._apply_rope(
            q,
            k,
            packed_qk,
            query_positions,
            key_positions,
            query_position_layout,
            key_position_layout,
        )

    queries_per_kv_head = module.num_heads // module.num_kv_heads
    if module._layout_strategy == "head_after_sequence":
        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)
    k = k.repeat_interleave(queries_per_kv_head, dim=-3)
    v = v.repeat_interleave(queries_per_kv_head, dim=-3)
    attended = F.scaled_dot_product_attention(
        q,
        k,
        v,
        mask,
        dropout_p=module.dropout if module.training else 0.0,
        is_causal=is_causal,
        scale=module.qk_head_dim**-0.5,
    )
    joined = attended.transpose(-3, -2).reshape(*attended.shape[:-3], attended.shape[-2], -1)
    return module.output_proj(joined)


@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize("layout", ["head_before_sequence", "head_after_sequence"])
@pytest.mark.parametrize("mask_kind", ["boolean", "additive"])
def test_grouped_attention_matches_expanded_kv_with_gradients(
    num_kv_heads: int,
    layout: Literal["head_before_sequence", "head_after_sequence"],
    mask_kind: Literal["boolean", "additive"],
) -> None:
    torch.manual_seed(18)
    actual_module = MultiheadAttention(
        embed_dim=10,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        kdim=7,
        vdim=9,
        qk_head_dim=4,
        value_head_dim=3,
        _layout_strategy=layout,
    )
    reference_module = MultiheadAttention(
        embed_dim=10,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        kdim=7,
        vdim=9,
        qk_head_dim=4,
        value_head_dim=3,
        _layout_strategy=layout,
    )
    reference_module.load_state_dict(actual_module.state_dict())

    query = torch.randn(2, 3, 4, 10, requires_grad=True)
    key = torch.randn(2, 3, 6, 7, requires_grad=True)
    value = torch.randn(2, 3, 6, 9, requires_grad=True)
    reference_query = query.detach().clone().requires_grad_()
    reference_key = key.detach().clone().requires_grad_()
    reference_value = value.detach().clone().requires_grad_()
    allowed = torch.ones(4, 4, 6, dtype=torch.bool)
    allowed[:, :, -1] = False
    allowed[1, :, 1] = False
    mask = allowed if mask_kind == "boolean" else torch.zeros_like(allowed, dtype=query.dtype).masked_fill(~allowed, -7.0)

    actual = actual_module(query, key, value, mask, is_causal=True)
    expected = _expanded_kv_reference(
        reference_module,
        reference_query,
        reference_key,
        reference_value,
        mask,
        is_causal=True,
    )
    torch.testing.assert_close(actual, expected)

    actual.square().sum().backward()
    expected.square().sum().backward()
    for actual_grad, expected_grad in (
        (query.grad, reference_query.grad),
        (key.grad, reference_key.grad),
        (value.grad, reference_value.grad),
    ):
        torch.testing.assert_close(actual_grad, expected_grad)
    for (_, actual_parameter), (_, reference_parameter) in zip(
        actual_module.named_parameters(),
        reference_module.named_parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad)


def test_packed_gqa_projection_keeps_one_matmul_without_qk_rotation_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = MultiheadSelfAttention(12, 6, 2, 3, num_kv_heads=2)
    assert module.in_proj_weight is not None
    assert module.in_proj_weight.shape == (6 * 2 + 2 * 2 + 2 * 3, 12)
    assert module.output_proj.weight.shape == (12, 6 * 3)

    calls: list[tuple[int, ...]] = []
    original: Callable[..., torch.Tensor] = F.linear

    def traced(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(weight.shape))
        return original(input, weight)

    monkeypatch.setattr(F, "linear", traced)
    x = torch.randn(2, 5, 12)
    projected_q, projected_k, projected_v, projected_qk = module._in_projection(x, x, x)
    _, _, _, packed_qk = module._split_heads(projected_q, projected_k, projected_v, projected_qk)
    assert projected_qk is None
    assert packed_qk is None

    calls.clear()
    module(x)
    assert calls == [(22, 12), (12, 18)]


def test_num_q_heads_alias_and_mha_default() -> None:
    aliased = MultiheadAttention(8, num_q_heads=4, num_kv_heads=2)
    ordinary_default = MultiheadAttention(8, 4)
    ordinary_explicit = MultiheadAttention(8, num_q_heads=4, num_kv_heads=4)

    assert aliased.num_heads == aliased.num_q_heads == 4
    assert aliased.num_kv_heads == 2
    assert MultiheadAttention.num_q_heads.fset is None
    assert ordinary_default.num_kv_heads == ordinary_default.num_heads
    assert ordinary_explicit.num_kv_heads == ordinary_explicit.num_heads
    assert "num_kv_heads=2" in repr(aliased)

    ordinary_explicit.load_state_dict(ordinary_default.state_dict())
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 6, 8)
    value = torch.randn(2, 6, 8)
    torch.testing.assert_close(
        ordinary_default(query, key, value),
        ordinary_explicit(query, key, value),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        cast(Any, MultiheadAttention)(8, 4, num_q_heads=4)
    with pytest.raises(ValueError, match="mutually exclusive"):
        cast(Any, MultiheadSelfAttention)(8, 4, num_q_heads=4)


def test_self_attention_accepts_keyword_only_query_head_alias() -> None:
    module = MultiheadSelfAttention(
        12,
        num_q_heads=6,
        num_kv_heads=2,
        d_k=2,
        d_v=3,
    )

    assert (module.num_q_heads, module.num_kv_heads, module.d_k, module.d_v) == (6, 2, 2, 3)
    assert module(torch.randn(2, 4, 12)).shape == (2, 4, 12)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_q_heads": 4, "num_kv_heads": 0}, "num_kv_heads"),
        ({"num_q_heads": 6, "num_kv_heads": 4}, "divisible"),
    ],
)
def test_grouped_head_count_validation(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MultiheadAttention(8, **kwargs)


def test_gqa_rope_matches_expanded_reference_and_stacked_rejects() -> None:
    actual_rope = RotaryPositionalEmbedding(10_000.0, 4, 16)
    reference_rope = RotaryPositionalEmbedding(10_000.0, 4, 16)
    actual = MultiheadSelfAttention(12, 6, 4, 2, num_kv_heads=2, rope=actual_rope)
    reference = MultiheadSelfAttention(12, 6, 4, 2, num_kv_heads=2, rope=reference_rope)
    reference.load_state_dict(actual.state_dict())
    x = torch.randn(2, 5, 12)
    positions = torch.arange(5).unsqueeze(0)

    for selected_positions in (None, positions):
        observed = actual(x, token_positions=selected_positions)
        expected = _expanded_kv_reference(
            reference,
            x,
            x,
            x,
            is_causal=True,
            query_positions=selected_positions,
            key_positions=selected_positions,
        )
        torch.testing.assert_close(observed, expected)

    stacked = MultiheadSelfAttention(
        12,
        6,
        4,
        2,
        num_kv_heads=2,
        rope=RotaryPositionalEmbedding(10_000.0, 4, 16),
        _qk_execution_strategy="stacked",
    )
    with pytest.raises(ValueError, match="equal shapes"):
        stacked(x)


def test_gqa_self_attention_rejects_nonbroadcastable_shared_head_positions() -> None:
    module = MultiheadSelfAttention(
        8,
        4,
        num_kv_heads=2,
        rope=RotaryPositionalEmbedding(10_000.0, 2, 16),
    )

    with pytest.raises(ValueError, match="not compatible"):
        module(
            torch.randn(2, 5, 8),
            token_positions=torch.arange(5).expand(4, -1),
            position_layout="head",
        )


def test_gqa_cross_attention_rope_supports_distinct_positions() -> None:
    actual = MultiheadAttention(
        10,
        4,
        num_kv_heads=2,
        kdim=7,
        vdim=9,
        qk_head_dim=4,
        value_head_dim=2,
        rope=RotaryPositionalEmbedding(10_000.0, 4, 16),
    )
    reference = MultiheadAttention(
        10,
        4,
        num_kv_heads=2,
        kdim=7,
        vdim=9,
        qk_head_dim=4,
        value_head_dim=2,
        rope=RotaryPositionalEmbedding(10_000.0, 4, 16),
    )
    reference.load_state_dict(actual.state_dict())
    query = torch.randn(2, 4, 10)
    key = torch.randn(2, 6, 7)
    value = torch.randn(2, 6, 9)
    query_positions = torch.arange(4)
    key_positions = torch.arange(6).flip(0)

    observed = actual(
        query,
        key,
        value,
        query_positions=query_positions,
        key_positions=key_positions,
    )
    expected = _expanded_kv_reference(
        reference,
        query,
        key,
        value,
        query_positions=query_positions,
        key_positions=key_positions,
    )
    torch.testing.assert_close(observed, expected)


def test_gqa_delegated_state_translation_uses_unequal_segments() -> None:
    module = MultiheadAttention(8, 4, num_kv_heads=2, qk_head_dim=3, value_head_dim=2)
    q_weight = torch.randn(12, 8)
    k_weight = torch.randn(6, 8)
    v_weight = torch.randn(4, 8)
    output_weight = torch.randn(8, 8)

    result = module.load_state_dict(
        {
            "q_proj.weight": q_weight,
            "k_proj.weight": k_weight,
            "v_proj.weight": v_weight,
            "output_proj.weight": output_weight,
        }
    )

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert module.in_proj_weight is not None
    torch.testing.assert_close(module.in_proj_weight, torch.cat((q_weight, k_weight, v_weight)))


def test_gqa_initializes_logical_projection_widths(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[int, ...], int, int]] = []

    def traced(weight: torch.Tensor, d_in: int, d_out: int) -> None:
        calls.append((tuple(weight.shape), d_in, d_out))
        torch.nn.init.zeros_(weight)

    monkeypatch.setattr(init, "starter_trunc_normal_for_linear_", traced)
    MultiheadAttention(
        10,
        4,
        num_kv_heads=2,
        qk_head_dim=3,
        value_head_dim=2,
    )

    assert calls == [
        ((10, 8), 8, 10),
        ((12, 10), 10, 12),
        ((6, 10), 10, 6),
        ((4, 10), 10, 4),
        ((10, 8), 8, 10),
    ]


def test_head_after_sequence_gqa_joins_heads_as_a_contiguous_view(monkeypatch: pytest.MonkeyPatch) -> None:
    module = MultiheadSelfAttention(
        8,
        4,
        num_kv_heads=2,
        _layout_strategy="head_after_sequence",
    )
    observed: list[tuple[tuple[int, ...], bool]] = []
    original: Callable[..., torch.Tensor] = module.output_proj.forward

    def traced(input: torch.Tensor) -> torch.Tensor:
        observed.append((input.stride(), input.is_contiguous()))
        return original(input)

    monkeypatch.setattr(module.output_proj, "forward", traced)
    module(torch.randn(2, 5, 8))

    assert observed == [((40, 8, 1), True)]
