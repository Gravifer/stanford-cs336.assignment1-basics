from collections.abc import Callable
from typing import Any

import pytest
import torch

from cs336_basics.nn.attention import RotaryPositionalEmbedding


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("position_shape", ["sequence", "singleton_head", "per_head"])
def test_matrix_and_elementwise_rope_match_with_gradients(dtype: torch.dtype, position_shape: str) -> None:
    matrix = RotaryPositionalEmbedding(10_000.0, 8, 16, use_matrix_form=True).to(dtype=dtype)
    elementwise = RotaryPositionalEmbedding(10_000.0, 8, 16, use_matrix_form=False).to(dtype=dtype)
    x_matrix = torch.randn(2, 3, 5, 8, dtype=dtype, requires_grad=True)
    x_elementwise = x_matrix.detach().clone().requires_grad_()
    positions = torch.arange(5)
    if position_shape == "singleton_head":
        positions = positions.unsqueeze(0)
    elif position_shape == "per_head":
        positions = torch.stack((positions, positions.flip(0), torch.zeros_like(positions)))

    matrix_output = matrix(x_matrix, positions)
    elementwise_output = elementwise(x_elementwise, positions)
    torch.testing.assert_close(matrix_output, elementwise_output)

    matrix_output.square().sum().backward()
    elementwise_output.square().sum().backward()
    torch.testing.assert_close(x_matrix.grad, x_elementwise.grad)


def test_rope_registers_only_selected_cache_representation() -> None:
    matrix = RotaryPositionalEmbedding(10_000.0, 8, 16, use_matrix_form=True)
    elementwise = RotaryPositionalEmbedding(10_000.0, 8, 16, use_matrix_form=False)

    assert set(dict(matrix.named_buffers())) == {"rot"}
    assert set(dict(elementwise.named_buffers())) == {"cos", "sin"}
    assert matrix.rot is not None and elementwise.cos is not None and elementwise.sin is not None
    assert matrix.rot.numel() == 2 * (elementwise.cos.numel() + elementwise.sin.numel())


def test_singleton_position_selection_expands_after_gather() -> None:
    rope = RotaryPositionalEmbedding(10_000.0, 8, 16)
    x = torch.randn(2, 3, 5, 8)

    selection = rope._select_rotations(
        x,
        torch.arange(5).unsqueeze(0),
        broadcast_positions=True,
        layout_strategy="head_before_sequence",
    )

    assert isinstance(selection, torch.Tensor)
    assert selection.shape == (2, 3, 5, 4, 2, 2)
    assert selection.stride(0) == 0
    assert selection.stride(1) == 0


@pytest.mark.parametrize("use_matrix_form", [True, False])
def test_apply_qk_matches_two_forward_calls_and_selects_once(
    use_matrix_form: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    rope = RotaryPositionalEmbedding(10_000.0, 8, 16, use_matrix_form=use_matrix_form)
    query = torch.randn(2, 3, 5, 8)
    key = torch.randn(2, 3, 5, 8)
    positions = torch.arange(5).unsqueeze(0)
    expected = rope(query, positions), rope(key, positions)
    selection_count = 0
    original: Callable[..., Any] = rope._select_rotations

    def traced(*args: Any, **kwargs: Any) -> Any:
        nonlocal selection_count
        selection_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(rope, "_select_rotations", traced)
    actual = rope.apply_qk(query, key, positions)

    assert selection_count == 1
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.parametrize("stacked", [False, True])
def test_apply_qk_consecutive_positions_and_autograd(stacked: bool) -> None:
    rope = RotaryPositionalEmbedding(10_000.0, 8, 16, use_matrix_form=False)
    query = torch.randn(2, 3, 5, 8, requires_grad=True)
    key = torch.randn(2, 3, 5, 8, requires_grad=True)

    actual = rope.apply_qk(query, key, _stacked=stacked)
    expected = rope.apply_qk(query, key, torch.arange(5), _stacked=False)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])

    (actual[0].square().sum() + actual[1].square().sum()).backward()
    assert query.grad is not None
    assert key.grad is not None


def test_apply_qk_supports_head_after_sequence_layout() -> None:
    rope = RotaryPositionalEmbedding(10_000.0, 8, 16, use_matrix_form=False)
    query = torch.randn(2, 3, 5, 8)
    key = torch.randn(2, 3, 5, 8)
    positions = torch.arange(5).expand(3, -1)
    expected = rope.apply_qk(query, key, positions)

    actual = rope.apply_qk(
        query.transpose(-3, -2),
        key.transpose(-3, -2),
        positions,
        _layout_strategy="head_after_sequence",
    )

    torch.testing.assert_close(actual[0].transpose(-3, -2), expected[0])
    torch.testing.assert_close(actual[1].transpose(-3, -2), expected[1])


def test_elementwise_rope_empty_and_invalid_shapes() -> None:
    rope = RotaryPositionalEmbedding(10_000.0, 8, 16, use_matrix_form=False)
    x = torch.empty(2, 0, 8)
    with pytest.warns(UserWarning, match="empty tensors is a no-op"):
        assert rope(x, torch.empty(0, dtype=torch.int64)) is x
    with pytest.raises(ValueError, match="not compatible"):
        rope(torch.randn(2, 3, 5, 8), torch.arange(4))
