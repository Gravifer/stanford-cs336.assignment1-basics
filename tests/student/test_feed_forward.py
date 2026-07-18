from collections.abc import Callable

import pytest
import torch

from cs336_basics.nn.feed_forward import SwiGLU_delegate, SwiGLU_own_weights, SwiGLU_packed_input


SWIGLU_TYPES: tuple[Callable[..., torch.nn.Module], ...] = (
    SwiGLU_delegate,
    SwiGLU_own_weights,
    SwiGLU_packed_input,
)


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
