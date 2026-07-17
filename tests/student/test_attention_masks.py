from typing import Literal

import pytest
import torch

from cs336_basics.nn.functional import scaled_dot_product_attention


def make_tensors():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    return q, k, v


def make_uniform_score_tensors():
    q = torch.zeros(1, 4, 1)
    k = torch.zeros(1, 4, 1)
    v = torch.tensor([[[1.0], [2.0], [4.0], [8.0]]])
    return q, k, v


def test_boolean_mask():
    q, k, v = make_uniform_score_tensors()
    bool_mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    out = scaled_dot_product_attention(q, k, v, mask=bool_mask, dropout_p=0.0, is_causal=False)
    torch.testing.assert_close(out.squeeze(), torch.tensor([1.0, 1.5, 7 / 3, 15 / 4]))


def test_numeric_mask():
    q, k, v = make_uniform_score_tensors()
    bool_mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    num_mask = torch.zeros(4, 4)
    num_mask[~bool_mask] = float("-inf")
    out = scaled_dot_product_attention(q, k, v, mask=num_mask, dropout_p=0.0, is_causal=False)
    torch.testing.assert_close(out.squeeze(), torch.tensor([1.0, 1.5, 7 / 3, 15 / 4]))


def test_compose_with_boolean_mask():
    q, k, v = make_uniform_score_tensors()
    mask2 = torch.ones(4, 4, dtype=torch.bool)
    mask2[3, 0] = False
    out = scaled_dot_product_attention(q, k, v, mask=mask2, dropout_p=0.0, is_causal="compose")
    torch.testing.assert_close(out.squeeze(), torch.tensor([1.0, 1.5, 7 / 3, 14 / 3]))


def test_compose_with_numeric_mask():
    q, k, v = make_uniform_score_tensors()
    num_mask2 = torch.zeros(4, 4)
    num_mask2[3, 0] = float("-inf")
    out = scaled_dot_product_attention(q, k, v, mask=num_mask2, dropout_p=0.0, is_causal="compose")
    torch.testing.assert_close(out.squeeze(), torch.tensor([1.0, 1.5, 7 / 3, 14 / 3]))


def test_broadcast_mask_suffix():
    # mask has a batch-like leading dim that should broadcast to query batch
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 8)
    mask = torch.tril(torch.ones(1, 4, 4, dtype=torch.bool))
    out = scaled_dot_product_attention(q, k, v, mask=mask, dropout_p=0.0, is_causal=False)
    for batch_index in range(2):
        expected = scaled_dot_product_attention(
            q[batch_index], k[batch_index], v[batch_index], mask=mask[0], dropout_p=0.0
        )
        torch.testing.assert_close(out[batch_index], expected)


def test_mask_matches_batch():
    # mask matches the leading batch dims exactly
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 8)
    mask = torch.stack(
        [
            torch.tril(torch.ones(4, 4, dtype=torch.bool)),
            torch.triu(torch.ones(4, 4, dtype=torch.bool)),
        ]
    )
    out = scaled_dot_product_attention(q, k, v, mask=mask, dropout_p=0.0, is_causal=False)
    for batch_index in range(2):
        expected = scaled_dot_product_attention(
            q[batch_index], k[batch_index], v[batch_index], mask=mask[batch_index], dropout_p=0.0
        )
        torch.testing.assert_close(out[batch_index], expected)


def test_attn_bias_alias_and_dtype_promotion():
    # attn_bias (additive) and mask aliasing; check dtype promotions don't crash
    q, k, v = make_tensors()
    bias = torch.zeros(4, 4, dtype=torch.float64)
    bias[0, 3] = float("-inf")
    out1 = scaled_dot_product_attention(q, k, v, attn_bias=bias, dropout_p=0.0, is_causal=False)
    out2 = scaled_dot_product_attention(q, k, v, mask=bias, dropout_p=0.0, is_causal=False)
    torch.testing.assert_close(out1, out2)
    assert out1.dtype == q.dtype


@pytest.mark.parametrize("query_len,key_len", [(4, 4), (3, 5), (5, 3)])
def test_causal_mask_matches_broadcasted_index_reference(query_len: int, key_len: int):
    q = torch.randn(2, query_len, 4)
    k = torch.randn(2, key_len, 4)
    v = torch.randn(2, key_len, 3)
    causal = torch.arange(key_len).unsqueeze(0) <= torch.arange(query_len).unsqueeze(1)

    actual = scaled_dot_product_attention(q, k, v, is_causal=True)
    expected = scaled_dot_product_attention(q, k, v, mask=causal)

    torch.testing.assert_close(actual, expected)


def test_rectangular_causal_mask_composes_with_boolean_and_additive_masks():
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 5, 4)
    v = torch.randn(2, 5, 3)
    allowed = torch.ones(3, 5, dtype=torch.bool)
    allowed[:, 1] = False
    causal = torch.arange(5).unsqueeze(0) <= torch.arange(3).unsqueeze(1)

    bool_actual = scaled_dot_product_attention(q, k, v, mask=allowed, is_causal="compose")
    bool_expected = scaled_dot_product_attention(q, k, v, mask=allowed & causal)
    torch.testing.assert_close(bool_actual, bool_expected)

    bias = torch.zeros(3, 5).masked_fill(~allowed, -10.0)
    float_actual = scaled_dot_product_attention(q, k, v, attn_bias=bias, is_causal="compose")
    float_expected = scaled_dot_product_attention(q, k, v, attn_bias=bias.masked_fill(~causal, float("-inf")))
    torch.testing.assert_close(float_actual, float_expected)


@pytest.mark.parametrize("mask_kind", ["boolean", "additive"])
def test_fully_masked_examples_match_torch_with_finite_zero_gradients(mask_kind: str) -> None:
    torch.manual_seed(12)
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 5, 4)
    v = torch.randn(2, 5, 6)
    allowed = torch.ones(2, 3, 5, dtype=torch.bool)
    allowed[0] = False
    mask = allowed if mask_kind == "boolean" else torch.zeros_like(allowed, dtype=q.dtype).masked_fill(~allowed, -torch.inf)

    actual_inputs = [tensor.detach().clone().requires_grad_() for tensor in (q, k, v)]
    expected_inputs = [tensor.detach().clone().requires_grad_() for tensor in (q, k, v)]
    actual = scaled_dot_product_attention(*actual_inputs, mask=mask)
    expected = torch.nn.functional.scaled_dot_product_attention(*expected_inputs, attn_mask=mask)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[0], torch.zeros_like(actual[0]))
    actual.sum().backward()
    expected.sum().backward()
    for actual_input, expected_input in zip(actual_inputs, expected_inputs, strict=True):
        assert actual_input.grad is not None and expected_input.grad is not None
        assert torch.isfinite(actual_input.grad).all()
        torch.testing.assert_close(actual_input.grad, expected_input.grad)
        torch.testing.assert_close(actual_input.grad[0], torch.zeros_like(actual_input.grad[0]))


def test_causal_composition_zeroes_rows_with_no_permitted_keys() -> None:
    q, k, v = make_tensors()
    allowed = torch.ones(4, 4, dtype=torch.bool)
    allowed[2] = False

    output = scaled_dot_product_attention(q, k, v, mask=allowed, is_causal="compose")

    torch.testing.assert_close(output[:, 2], torch.zeros_like(output[:, 2]))
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("num_kv_heads", [2, 4])
@pytest.mark.parametrize("layout", ["head_before_sequence", "head_after_sequence"])
def test_attention_modules_zero_fully_masked_rows_for_all_head_paths(
    num_kv_heads: int,
    layout: Literal["head_before_sequence", "head_after_sequence"],
) -> None:
    from cs336_basics.nn.attention import MultiheadAttention

    module = MultiheadAttention(
        8,
        4,
        num_kv_heads=num_kv_heads,
        _layout_strategy=layout,
    )
    x = torch.randn(2, 4, 8, requires_grad=True)
    allowed = torch.ones(4, 4, dtype=torch.bool)
    allowed[1] = False

    output = module(x, x, x, allowed)

    torch.testing.assert_close(output[:, 1], torch.zeros_like(output[:, 1]))
    output.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
