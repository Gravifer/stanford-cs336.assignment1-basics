from collections.abc import Callable
from typing import Any

import einx
import pytest
import torch
from jaxtyping import TypeCheckError
from torch.nn.modules.module import _IncompatibleKeys

from cs336_basics.nn import functional as F
from cs336_basics.nn.attention import RotaryPositionalEmbedding
from cs336_basics.nn.feed_forward import SwiGLU_delegate, SwiGLU_own_weights, SwiGLU_packed_input
from cs336_basics.nn.modules import Embedding, Linear, RMSNorm


def test_embedding_from_pretrained_contract() -> None:
    weight = torch.randn(5, 3)
    embedding = Embedding.from_pretrained(weight)

    assert type(embedding) is Embedding
    assert embedding.weight.data_ptr() == weight.data_ptr()
    assert embedding.weight.requires_grad is False
    assert not hasattr(embedding, "freeze")


def test_linear_registers_no_bias() -> None:
    linear = Linear(3, 4)

    assert linear.bias is None
    assert linear.state_dict().keys() == {"weight"}


def test_non_affine_rmsnorm_has_optional_weight() -> None:
    norm = RMSNorm(4, elementwise_affine=False)

    assert norm.weight is None
    assert norm(torch.randn(2, 3, 4)).shape == (2, 3, 4)


def test_gelu_supports_both_declared_modes() -> None:
    x = torch.randn(2, 3)

    torch.testing.assert_close(F.gelu(x, "none"), torch.nn.functional.gelu(x, approximate="none"))
    torch.testing.assert_close(F.gelu(x, "tanh"), torch.nn.functional.gelu(x, approximate="tanh"))


def test_gelu_rejects_unknown_mode() -> None:
    with pytest.raises((NotImplementedError, TypeCheckError)):
        F.gelu(torch.randn(2, 3), "invalid")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("module_type", [SwiGLU_delegate, SwiGLU_own_weights, SwiGLU_packed_input])
def test_swiglu_load_state_dict_returns_incompatible_keys(module_type: Callable[..., torch.nn.Module]) -> None:
    module = module_type(4, 8)
    state_dict = {
        "value_weight": torch.randn(8, 4),
        "gate_weight": torch.randn(8, 4),
        "out_weight": torch.randn(4, 8),
    }

    result = module.load_state_dict(state_dict)

    assert isinstance(result, _IncompatibleKeys)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


@pytest.mark.parametrize("module_type", [SwiGLU_delegate, SwiGLU_own_weights, SwiGLU_packed_input])
def test_swiglu_load_state_dict_accepts_packed_input(module_type: Callable[..., torch.nn.Module]) -> None:
    module = module_type(4, 8)
    state_dict = {
        "in_weight": torch.randn(16, 4),
        "out_weight": torch.randn(4, 8),
    }

    result = module.load_state_dict(state_dict)

    assert isinstance(result, _IncompatibleKeys)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


@pytest.mark.parametrize("module_type", [SwiGLU_delegate, SwiGLU_own_weights, SwiGLU_packed_input])
def test_swiglu_load_state_dict_reports_incomplete_layout(module_type: Callable[..., torch.nn.Module]) -> None:
    module = module_type(4, 8)

    result = module.load_state_dict({"out_weight": torch.randn(4, 8), "surprise": torch.tensor(1)}, strict=False)

    assert isinstance(result, _IncompatibleKeys)
    assert result.missing_keys
    assert result.unexpected_keys == ["surprise"]


@pytest.mark.parametrize("module_type", [SwiGLU_delegate, SwiGLU_own_weights, SwiGLU_packed_input])
def test_swiglu_rejects_conflicting_weight_layouts(module_type: Callable[..., torch.nn.Module]) -> None:
    module = module_type(4, 8)
    state_dict = {
        "in_weight": torch.randn(16, 4),
        "value_weight": torch.randn(8, 4),
        "out_weight": torch.randn(4, 8),
    }

    with pytest.raises(ValueError, match="contains both"):
        module.load_state_dict(state_dict)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_rope_promotes_computation_and_preserves_input_dtype(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    rope = RotaryPositionalEmbedding(10_000.0, 8, 16).to(dtype=dtype)
    x = torch.randn(2, 4, 8, dtype=dtype)
    positions = torch.arange(4)
    observed_dtypes: list[tuple[torch.dtype, torch.dtype]] = []
    original_dot: Callable[..., Any] = einx.dot

    def traced_dot(description: str, left: torch.Tensor, right: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        observed_dtypes.append((left.dtype, right.dtype))
        return original_dot(description, left, right, **kwargs)

    monkeypatch.setattr(einx, "dot", traced_dot)
    output = rope(x, positions)

    expected_op_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert observed_dtypes == [(expected_op_dtype, expected_op_dtype)]
    assert output.dtype == dtype
    assert output.shape == x.shape


def test_rope_broadcasts_position_batch_suffix() -> None:
    rope = RotaryPositionalEmbedding(10_000.0, 8, 16)
    x = torch.randn(2, 3, 4, 8)
    positions = torch.arange(4).expand(3, -1)

    output = rope(x, positions)

    assert output.shape == x.shape


def test_rope_rejects_non_suffix_position_batch() -> None:
    rope = RotaryPositionalEmbedding(10_000.0, 8, 16)
    x = torch.randn(3, 2, 4, 8)
    positions = torch.arange(4).expand(3, -1)

    with pytest.raises(ValueError, match="not compatible"):
        rope(x, positions)


def test_rope_rejects_scalar_positions() -> None:
    rope = RotaryPositionalEmbedding(10_000.0, 8, 16)

    with pytest.raises((ValueError, TypeCheckError)):
        rope(torch.randn(2, 4, 8), torch.tensor(0))


def test_rope_warns_and_returns_empty_input() -> None:
    rope = RotaryPositionalEmbedding(10_000.0, 8, 16)
    x = torch.empty(2, 0, 8)

    with pytest.warns(UserWarning, match="empty tensors is a no-op"):
        output = rope(x, torch.empty(0, dtype=torch.int64))

    assert output is x


def test_rope_validates_shapes_before_empty_noop() -> None:
    rope = RotaryPositionalEmbedding(10_000.0, 8, 16)
    x = torch.empty(0, 2, 8)

    with pytest.raises((ValueError, TypeCheckError)):
        rope(x, torch.empty(0, dtype=torch.int64))
