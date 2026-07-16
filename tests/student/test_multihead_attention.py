from collections.abc import Callable
from typing import Any

import pytest
import torch

from cs336_basics.nn import functional as F
from cs336_basics.nn.attention import MultiheadAttention, MultiheadSelfAttention, RotaryPositionalEmbedding


def test_mha_matches_torch_with_distinct_input_widths() -> None:
    torch.manual_seed(0)
    actual = MultiheadAttention(embed_dim=12, num_heads=3, kdim=5, vdim=7)
    expected = torch.nn.MultiheadAttention(
        embed_dim=12,
        num_heads=3,
        kdim=5,
        vdim=7,
        bias=False,
        batch_first=True,
    )
    with torch.no_grad():
        expected.q_proj_weight.copy_(actual.q_proj.weight)
        expected.k_proj_weight.copy_(actual.k_proj.weight)
        expected.v_proj_weight.copy_(actual.v_proj.weight)
        expected.out_proj.weight.copy_(actual.output_proj.weight)

    query = torch.randn(2, 4, 12)
    key = torch.randn(2, 6, 5)
    value = torch.randn(2, 6, 7)

    expected_output, _ = expected(query, key, value, need_weights=False)
    torch.testing.assert_close(actual(query, key, value), expected_output)


def test_generalized_head_widths_and_leading_batch_axes() -> None:
    torch.manual_seed(1)
    module = MultiheadAttention(
        embed_dim=10,
        num_heads=3,
        qk_head_dim=4,
        value_head_dim=2,
    )
    query = torch.randn(2, 3, 4, 10)
    key = torch.randn(2, 3, 6, 10)
    value = torch.randn(2, 3, 6, 10)

    output = module(query, key, value)

    assert output.shape == (2, 3, 4, 10)
    assert module.q_proj.weight.shape == (12, 10)
    assert module.k_proj.weight.shape == (12, 10)
    assert module.v_proj.weight.shape == (6, 10)
    assert module.output_proj.weight.shape == (10, 6)


def test_rope_infers_positions_and_broadcasts_over_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(2)
    rope = RotaryPositionalEmbedding(theta=10_000.0, d_k=4, max_seq_len=8)
    module = MultiheadAttention(embed_dim=8, num_heads=2, rope=rope)
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn(2, 3, 6, 8)
    value = torch.randn(2, 3, 6, 8)
    query_positions = torch.arange(4).unsqueeze(0)
    key_positions = torch.arange(6).unsqueeze(0)
    observed_shapes: list[tuple[torch.Size, torch.Size, torch.Size]] = []
    original: Callable[..., torch.Tensor] = F.scaled_dot_product_attention

    def traced(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        observed_shapes.append((query.shape, key.shape, value.shape))
        return original(query, key, value, *args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", traced)

    inferred = module(query, key, value)
    explicit = module(
        query,
        key,
        value,
        query_positions=query_positions,
        key_positions=key_positions,
    )

    torch.testing.assert_close(inferred, explicit)
    expected_shapes = (
        torch.Size((2, 3, 2, 4, 4)),
        torch.Size((2, 3, 2, 6, 4)),
        torch.Size((2, 3, 2, 6, 4)),
    )
    assert observed_shapes == [expected_shapes, expected_shapes]


def test_self_attention_is_causal_by_default() -> None:
    torch.manual_seed(3)
    module = MultiheadSelfAttention(d_model=8, num_heads=2)
    x = torch.randn(2, 5, 8)

    default = module(x)
    explicit = module(x, is_causal=True)
    noncausal = module(x, is_causal=False)

    torch.testing.assert_close(default, explicit)
    assert not torch.allclose(default, noncausal)


def test_module_disables_attention_dropout_during_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[float] = []
    original: Callable[..., torch.Tensor] = F.scaled_dot_product_attention

    def traced(*args: Any, dropout_p: float = 0.0, **kwargs: Any) -> torch.Tensor:
        observed.append(dropout_p)
        return original(*args, dropout_p=dropout_p, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", traced)
    module = MultiheadSelfAttention(d_model=8, num_heads=2, dropout=0.25)
    x = torch.randn(2, 4, 8)

    module.eval()
    module(x)
    module.train()
    module(x)

    assert observed == [0.0, 0.25]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"embed_dim": 7, "num_heads": 2}, "divisible"),
        ({"embed_dim": 8, "num_heads": 2, "dropout": 1.1}, "dropout"),
        ({"embed_dim": 8, "num_heads": 2, "kdim": 0}, "kdim"),
        ({"embed_dim": 8, "num_heads": 2, "qk_head_dim": 0}, "qk_head_dim"),
    ],
)
def test_constructor_validation(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MultiheadAttention(**kwargs)


def test_rejects_rope_width_mismatch() -> None:
    rope = RotaryPositionalEmbedding(theta=10_000.0, d_k=6, max_seq_len=8)

    with pytest.raises(ValueError, match="RoPE width"):
        MultiheadAttention(embed_dim=8, num_heads=2, rope=rope)


def test_rejects_incompatible_input_shapes() -> None:
    module = MultiheadAttention(embed_dim=8, num_heads=2, kdim=5, vdim=7)

    with pytest.raises(ValueError, match="sequence lengths"):
        module(torch.randn(2, 4, 8), torch.randn(2, 6, 5), torch.randn(2, 5, 7))

    with pytest.raises(ValueError, match="widths"):
        module(torch.randn(2, 4, 8), torch.randn(2, 6, 6), torch.randn(2, 6, 7))
