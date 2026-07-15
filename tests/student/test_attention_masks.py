import math

import torch

from cs336_basics.nn.functional import scaled_dot_product_attention


def make_tensors():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    return q, k, v


def test_boolean_mask():
    q, k, v = make_tensors()
    bool_mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    out = scaled_dot_product_attention(q, k, v, mask=bool_mask, dropout_p=0.0, is_causal=False)
    assert out.shape == (1, 4, 8)


def test_numeric_mask():
    q, k, v = make_tensors()
    bool_mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    num_mask = torch.zeros(4, 4)
    num_mask[~bool_mask] = float("-inf")
    out = scaled_dot_product_attention(q, k, v, mask=num_mask, dropout_p=0.0, is_causal=False)
    assert out.shape == (1, 4, 8)


def test_compose_with_boolean_mask():
    q, k, v = make_tensors()
    mask2 = torch.ones(4, 4, dtype=torch.bool)
    mask2[0, 3] = False
    out = scaled_dot_product_attention(q, k, v, mask=mask2, dropout_p=0.0, is_causal="compose")
    assert out.shape == (1, 4, 8)


def test_compose_with_numeric_mask():
    q, k, v = make_tensors()
    num_mask2 = torch.zeros(4, 4)
    num_mask2[0, 3] = float("-inf")
    out = scaled_dot_product_attention(q, k, v, mask=num_mask2, dropout_p=0.0, is_causal="compose")
    assert out.shape == (1, 4, 8)


def test_broadcast_mask_suffix():
    # mask has a batch-like leading dim that should broadcast to query batch
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 8)
    mask = torch.tril(torch.ones(1, 4, 4, dtype=torch.bool))
    out = scaled_dot_product_attention(q, k, v, mask=mask, dropout_p=0.0, is_causal=False)
    assert out.shape == (2, 4, 8)


def test_mask_matches_batch():
    # mask matches the leading batch dims exactly
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 8)
    mask = torch.stack([torch.tril(torch.ones(4, 4, dtype=torch.bool)) for _ in range(2)], dim=0)
    out = scaled_dot_product_attention(q, k, v, mask=mask, dropout_p=0.0, is_causal=False)
    assert out.shape == (2, 4, 8)


def test_attn_bias_alias_and_dtype_promotion():
    # attn_bias (additive) and mask aliasing; check dtype promotions don't crash
    q, k, v = make_tensors()
    bias = torch.zeros(4, 4, dtype=torch.float64)
    bias[0, 3] = float("-inf")
    out1 = scaled_dot_product_attention(q, k, v, attn_bias=bias, dropout_p=0.0, is_causal=False)
    out2 = scaled_dot_product_attention(q, k, v, mask=bias, dropout_p=0.0, is_causal=False)
    assert out1.shape == (1, 4, 8)
    assert out2.shape == (1, 4, 8)
