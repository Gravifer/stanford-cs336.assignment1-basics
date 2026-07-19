"""Tests for static symbolic model analytics."""

import pytest
import sympy
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from cs336_basics.nn.analytics import CostRepr, Module, TensorRepr, cost_repr, matmul_flops
from cs336_basics.nn.modules import Linear


def _tensor(value: object) -> TensorRepr:
    assert isinstance(value, TensorRepr)
    return value


def test_repository_module_base_preserves_torch_identity() -> None:
    linear = Linear(3, 5)

    assert isinstance(linear, Module)
    assert isinstance(linear, nn.Module)
    assert linear.state_dict().keys() == {"weight"}


def test_symbolic_tensor_and_arguments_are_immutable_copies() -> None:
    arguments: dict[str, object] = {
        "self": TensorRepr((1, 2, 3), torch.float32),
        "mat2": TensorRepr((1, 3, 4), torch.float32),
    }
    cost = CostRepr("test bmm", torch.ops.aten.bmm.default, arguments)
    arguments["self"] = TensorRepr((1, 9, 3), torch.float32)

    assert cost.arguments["self"] == TensorRepr((1, 2, 3), torch.float32)
    with pytest.raises(TypeError):
        cost.arguments["self"] = TensorRepr((1, 9, 3), torch.float32)  # ty: ignore[invalid-assignment]


def test_cost_repr_requires_exact_overload_and_schema_arguments() -> None:
    operands = {
        "self": TensorRepr((1, 2, 3)),
        "mat2": TensorRepr((1, 3, 4)),
    }

    with pytest.raises(TypeError, match="exact torch.ops overload"):
        CostRepr("packet", torch.ops.aten.bmm, operands)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="requires schema arguments: mat2"):
        CostRepr("incomplete", torch.ops.aten.bmm.default, {"self": operands["self"]})
    with pytest.raises(ValueError, match="no schema arguments named: right"):
        CostRepr("conflict", torch.ops.aten.bmm.default, {**operands, "right": operands["mat2"]})


def test_linear_cost_uses_symbolic_aten_operands() -> None:
    tree = Linear(3, 5).cost_repr()
    cost = tree.costs[0]

    assert cost.operation is torch.ops.aten.bmm.default
    assert cost.name == "linear projection"
    assert tuple(tree.symbols.values()) == ("tokens", "d_in", "d_out")
    assert _tensor(cost.arguments["self"]).shape[1:] == (
        tree.find_symbols("tokens")[0],
        tree.find_symbols("d_in")[0],
    )


def test_scoped_symbols_do_not_collide_between_modules() -> None:
    first = Linear(3, 5).cost_repr()
    second = Linear(3, 5).cost_repr()

    assert first.find_symbols("d_in")[0] != second.find_symbols("d_in")[0]
    assert str(first.find_symbols("d_in")[0]) == str(second.find_symbols("d_in")[0])


def test_course_policy_keeps_symbolic_and_bound_views() -> None:
    tree = cost_repr(Linear(3, 5))
    tokens = tree.find_symbols("tokens")[0]
    d_in = tree.find_symbols("d_in")[0]
    d_out = tree.find_symbols("d_out")[0]
    report = matmul_flops(tree)

    assert sympy.simplify(report.symbolic_total - 2 * tokens * d_in * d_out) == 0
    assert report.bound_total == 30 * tokens
    assert report.substitute({tokens: 7}).bound_total == 210
    assert report.unsupported == ()


def test_linear_symbolic_cost_matches_meta_flop_counter() -> None:
    linear = Linear(3, 5, device=torch.device("meta"))
    tree = linear.cost_repr()
    tokens = tree.find_symbols("tokens")[0]
    counter = FlopCounterMode(display=False)

    with counter:
        linear(torch.empty((7, 3), device="meta"))

    assert set(counter.get_flop_counts()["Global"]) == {torch.ops.aten.bmm}
    assert matmul_flops(tree, substitutions={tokens: 7}).bound_total == counter.get_total_flops()


def test_unsupported_operations_and_external_modules_remain_visible() -> None:
    unsupported = CostRepr(
        "pointwise sine",
        torch.ops.aten.sin.default,
        {"self": TensorRepr((3, 4))},
    )

    class Unsupported(Module):
        def _cost_repr(self, scope):
            del scope
            return (unsupported,)

    report = matmul_flops(Unsupported().cost_repr())
    unclassified_report = matmul_flops(Module().cost_repr())
    external_report = matmul_flops(cost_repr(nn.Linear(3, 4)))

    assert report.symbolic_total == 0
    assert "no matmul policy for aten.sin.default" in report.unsupported[0]
    assert any("has not classified its static local matmul work" in message for message in unclassified_report.unsupported)
    assert any("has no static local-cost provider" in message for message in external_report.unsupported)
    with pytest.raises(NotImplementedError, match="unsupported symbolic costs"):
        matmul_flops(Unsupported().cost_repr(), strict=True)
