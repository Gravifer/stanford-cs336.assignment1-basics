import warnings
from collections.abc import Callable
from typing import Final

import pytest
import torch

from cs336_basics.nn.feed_forward import SwiGLU_delegate, SwiGLU_own_weights, SwiGLU_packed_input


SWIGLU_TYPES: Final[tuple[Callable[..., torch.nn.Module], ...]] = (
    SwiGLU_delegate,
    SwiGLU_own_weights,
    SwiGLU_packed_input,
)


class _FeedForwardLayer(torch.nn.Module):
    def __init__(self, ffn: torch.nn.Module) -> None:
        super().__init__()
        self.ffn = ffn


class _FeedForwardStack(torch.nn.Module):
    def __init__(self, ffn: torch.nn.Module) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_FeedForwardLayer(ffn)])


@pytest.mark.parametrize("module_type", SWIGLU_TYPES)
@pytest.mark.parametrize(
    ("d_model", "expected_d_ff"),
    [(4, 64), (256, 704), (512, 1344), (1600, 4288)],
)
def test_swiglu_default_width_is_nearest_positive_multiple_of_64(
    module_type: Callable[..., torch.nn.Module],
    d_model: int,
    expected_d_ff: int,
) -> None:
    module = module_type(d_model, device=torch.device("meta"))

    assert module.d_ff == expected_d_ff


@pytest.mark.parametrize("module_type", SWIGLU_TYPES)
def test_swiglu_retains_explicit_width(module_type: Callable[..., torch.nn.Module]) -> None:
    module = module_type(16, 37, device=torch.device("meta"))

    assert module.d_ff == 37


def test_owning_swiglu_initialization_matches_delegated_linear_modules() -> None:
    torch.manual_seed(7)
    delegated = SwiGLU_delegate(16, 64)

    torch.manual_seed(7)
    owned = SwiGLU_own_weights(16, 64)
    torch.testing.assert_close(owned.value_weight, delegated.value_linear.weight)
    torch.testing.assert_close(owned.gate_weight, delegated.gate_linear.weight)
    torch.testing.assert_close(owned.out_weight, delegated.out_linear.weight)

    torch.manual_seed(7)
    packed = SwiGLU_packed_input(16, 64)
    torch.testing.assert_close(
        packed.in_weight,
        torch.cat((delegated.value_linear.weight, delegated.gate_linear.weight)),
    )
    torch.testing.assert_close(packed.out_weight, delegated.out_linear.weight)


@pytest.mark.parametrize("module_type", SWIGLU_TYPES)
def test_course_swiglu_keys_compose_through_module_list_and_preserve_mapping(
    module_type: Callable[..., torch.nn.Module],
) -> None:
    module = module_type(4, 6)
    stack = _FeedForwardStack(module)
    w1 = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
    w2 = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 20
    w3 = -torch.arange(24, dtype=torch.float32).reshape(6, 4) / 30
    state_dict = {
        "layers.0.ffn.w1.weight": w1,
        "layers.0.ffn.w2.weight": w2,
        "layers.0.ffn.w3.weight": w3,
    }
    original_items = tuple(state_dict.items())

    with pytest.warns(UserWarning, match=r"w1\.weight maps to gate.*w2\.weight maps to output.*w3\.weight maps to value") as caught:
        result = stack.load_state_dict(state_dict)

    assert len(caught) == 1
    assert caught[0].filename == __file__
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert tuple(state_dict.items()) == original_items

    x = torch.randn(2, 3, 4)
    gate = torch.nn.functional.linear(x, w1)
    value = torch.nn.functional.linear(x, w3)
    expected = torch.nn.functional.linear(value * torch.nn.functional.silu(gate), w2)
    torch.testing.assert_close(module(x), expected)


@pytest.mark.parametrize("module_type", SWIGLU_TYPES)
@pytest.mark.parametrize("layout", ["semantic", "packed"])
def test_compatibility_layouts_compose_through_module_list_without_course_warning(
    module_type: Callable[..., torch.nn.Module],
    layout: str,
) -> None:
    module = module_type(4, 6)
    stack = _FeedForwardStack(module)
    value = torch.full((6, 4), 1.0)
    gate = torch.full((6, 4), 2.0)
    output = torch.full((4, 6), 3.0)
    if layout == "semantic":
        state_dict = {
            "layers.0.ffn.value_weight": value,
            "layers.0.ffn.gate_weight": gate,
            "layers.0.ffn.out_weight": output,
        }
    else:
        state_dict = {
            "layers.0.ffn.in_weight": torch.cat((value, gate)),
            "layers.0.ffn.out_weight": output,
        }

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = stack.load_state_dict(state_dict)

    assert result.missing_keys == []
    assert result.unexpected_keys == []


@pytest.mark.parametrize(
    ("module_type", "expected_keys"),
    [
        (SwiGLU_delegate, {"value_linear.weight", "gate_linear.weight", "out_linear.weight"}),
        (SwiGLU_own_weights, {"value_weight", "gate_weight", "out_weight"}),
        (SwiGLU_packed_input, {"in_weight", "out_weight"}),
    ],
)
def test_native_swiglu_round_trip_preserves_saved_layout(
    module_type: Callable[..., torch.nn.Module],
    expected_keys: set[str],
) -> None:
    source = module_type(4, 6)
    target = module_type(4, 6)
    state_dict = source.state_dict()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = target.load_state_dict(state_dict)

    assert set(state_dict) == expected_keys
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_nested_native_swiglu_load_preserves_assign_behavior() -> None:
    source = _FeedForwardStack(SwiGLU_packed_input(4, 6))
    target_module = SwiGLU_packed_input(4, 6)
    target = _FeedForwardStack(target_module)
    state_dict = source.state_dict()
    old_parameter = target_module.in_weight

    result = target.load_state_dict(state_dict, assign=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert target_module.in_weight is not old_parameter
    assert target_module.in_weight.data_ptr() == state_dict["layers.0.ffn.in_weight"].data_ptr()


def test_packed_swiglu_rejects_incomplete_unpacked_input_group() -> None:
    module = SwiGLU_packed_input(4, 6)

    with pytest.raises(ValueError, match="incomplete"):
        module.load_state_dict({"gate_weight": torch.randn(6, 4)}, strict=False)


@pytest.mark.parametrize("module_type", SWIGLU_TYPES)
def test_swiglu_rejects_mixed_course_and_semantic_layouts(
    module_type: Callable[..., torch.nn.Module],
) -> None:
    module = module_type(4, 6)

    with pytest.raises(ValueError, match="contains both"):
        module.load_state_dict(
            {
                "w1.weight": torch.randn(6, 4),
                "out_weight": torch.randn(4, 6),
            },
            strict=False,
        )
