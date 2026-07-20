"""Tests for static symbolic model analytics."""

from collections import OrderedDict
from typing import Literal

import pytest
import sympy
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

import cs336_basics.nn.analytics as analytics
from cs336_basics.nn import DeltaLayer, Module
from cs336_basics.nn.analytics import (
    CostObserver,
    CostRepr,
    CostTerm,
    ModuleStateFootprint,
    SymbolRepr,
    TensorRepr,
    cost_repr,
    matmul_flops,
    module_state_footprint,
    observe_costs,
)
from cs336_basics.nn.attention import MultiheadAttention, MultiheadSelfAttention, RotaryPositionalEmbedding
from cs336_basics.nn.feed_forward import SwiGLU_delegate, SwiGLU_own_weights, SwiGLU_packed_input
from cs336_basics.nn.model import GPTDecoderLayer, TransformerLM
from cs336_basics.nn.modules import DeltaLayer as ModulesDeltaLayer
from cs336_basics.nn.modules import Linear, Module as ModulesModule


def _tensor(value: object) -> TensorRepr:
    assert isinstance(value, TensorRepr)
    return value


def test_repository_module_base_preserves_torch_identity() -> None:
    linear = Linear(3, 5)

    assert isinstance(linear, Module)
    assert isinstance(linear, nn.Module)
    assert linear.state_dict().keys() == {"weight"}
    assert Module is ModulesModule
    assert DeltaLayer is ModulesDeltaLayer
    assert not hasattr(analytics, "Module")


def test_external_torch_module_can_implement_cost_provider_structurally() -> None:
    class External(nn.Module):
        def _cost_repr(self, scope):
            s = scope.symbols
            s.unbound("tokens")
            return (
                CostRepr(
                    "external projection",
                    torch.ops.aten.bmm.default,
                    {
                        "self": TensorRepr((1, s.tokens, 3)),
                        "mat2": TensorRepr((1, 3, 4)),
                    },
                ),
            )

        def _cost_children(self, scope):
            return ()

    tree = cost_repr(External())

    assert tree.costs[0].name == "external projection"
    assert matmul_flops(tree, substitutions={tree.find_symbols("tokens")[0]: 5}, strict=True).bound_total == 120


class _ObservableProjection(Module):
    def __init__(self, *, invalid_name: bool = False) -> None:
        super().__init__()
        self.invalid_name = invalid_name

    def forward(self, x):
        return x + 1

    def _cost_repr(self, scope):
        s = scope.symbols
        s.unbound("tokens")
        return (
            CostRepr(
                "observed projection",
                torch.ops.aten.mm.default,
                {"self": TensorRepr((s.tokens, 3)), "mat2": TensorRepr((3, 4))},
            ),
        )

    def _cost_call_bindings(self, args, kwargs, output):
        del output
        x = args[0] if args else kwargs["x"]
        return {"undeclared" if self.invalid_name else "tokens": x.shape[0]}


def test_cost_observer_records_repeated_root_calls_without_retaining_tensors() -> None:
    module = _ObservableProjection()
    first = torch.empty((2, 3))
    second = torch.empty((5, 3))
    observer = observe_costs(module)

    with observer:
        assert module(first).shape == first.shape
        assert module(x=second).shape == second.shape

    reports = observer.matmul_flops(strict=True)
    assert observer.call_count == 2
    assert tuple(report.bound_total for report in reports) == (48, 120)
    assert reports[0].symbolic_total == reports[1].symbolic_total
    assert all(not isinstance(value, torch.Tensor) for record in observer._substitutions for value in record.values())


def test_cost_observer_lifecycle_and_hook_isolation() -> None:
    module = _ObservableProjection()
    observer = module.observe_costs()
    assert isinstance(observer, CostObserver)
    existing_hook_outputs: list[torch.Tensor] = []
    existing_handle = module.register_forward_hook(lambda _module, _args, output: existing_hook_outputs.append(output))

    with pytest.raises(RuntimeError, match="has not been entered"):
        observer.matmul_flops()
    with observer:
        with pytest.raises(RuntimeError, match="cannot be entered more than once"):
            observer.__enter__()
        output = module(torch.empty((2, 3)))
    module(torch.empty((7, 3)))
    existing_handle.remove()

    assert output.shape == (2, 3)
    assert observer.call_count == 1
    assert len(existing_hook_outputs) == 2
    assert observer.matmul_flops(strict=True)[0].bound_total == 48
    with pytest.raises(RuntimeError, match="cannot be entered more than once"):
        observer.__enter__()


def test_cost_observer_does_not_record_a_forward_without_output() -> None:
    class Fails(_ObservableProjection):
        def forward(self, x):
            raise LookupError("no output")

    observer = observe_costs(Fails())
    with observer, pytest.raises(LookupError, match="no output"):
        observer._module(torch.empty((2, 3)))

    assert observer.call_count == 0
    assert observer.matmul_flops() == ()


def test_cost_observer_defers_binding_failures_until_reporting() -> None:
    observer = observe_costs(_ObservableProjection(invalid_name=True))

    with observer:
        output = observer._module(torch.empty((2, 3)))

    assert output.shape == (2, 3)
    assert observer.call_count == 0
    with pytest.raises(RuntimeError, match="undeclared root symbols"):
        observer.matmul_flops()

    rank_error = observe_costs(_ObservableProjection())
    with rank_error:
        scalar_output = rank_error._module(torch.tensor(1.0))
    assert scalar_output.shape == ()
    with pytest.raises(RuntimeError, match="IndexError"):
        rank_error.matmul_flops()


def test_cost_observer_nested_sessions_record_their_own_completion_windows() -> None:
    module = _ObservableProjection()
    outer = observe_costs(module)
    inner = observe_costs(module)

    with outer:
        module(torch.empty((2, 3)))
        with inner:
            module(torch.empty((3, 3)))
        module(torch.empty((5, 3)))

    assert tuple(report.bound_total for report in outer.matmul_flops(strict=True)) == (48, 72, 120)
    assert tuple(report.bound_total for report in inner.matmul_flops(strict=True)) == (72,)


def test_cost_observer_requires_a_structural_call_provider() -> None:
    class External(nn.Module):
        def _cost_repr(self, scope):
            return ()

        def _cost_children(self, scope):
            return ()

    with pytest.raises(TypeError, match="does not provide root-invocation"):
        observe_costs(External())


@pytest.mark.parametrize("provider_method", ["_cost_repr", "_cost_children"])
def test_structural_cost_providers_reject_malformed_results(provider_method: str) -> None:
    class Malformed(nn.Module):
        def _cost_repr(self, scope):
            return (object(),) if provider_method == "_cost_repr" else ()

        def _cost_children(self, scope):
            return (object(),) if provider_method == "_cost_children" else ()

    with pytest.raises(TypeError, match=provider_method):
        cost_repr(Malformed())


def test_structural_containers_preserve_repeated_shared_module_invocations() -> None:
    shared = Linear(3, 3, device=torch.device("meta"))
    module = nn.Sequential(shared, shared)
    tree = cost_repr(module)
    counter = FlopCounterMode(display=False)

    with counter:
        module(torch.empty((5, 3), device="meta"))

    assert tuple(child.name for child in tree.children) == ("0", "1")
    assert tree.children[0].module_type == tree.children[1].module_type == "Linear"
    assert all(child.edge_role == "call" for child in tree.children)
    substitutions = {symbol: 5 for symbol in tree.find_symbols("tokens")}
    assert matmul_flops(tree, substitutions=substitutions, strict=True).bound_total == counter.get_total_flops()


def test_authored_module_recursion_preserves_repeated_registered_child_calls() -> None:
    class SharedParent(Module):
        def __init__(self) -> None:
            super().__init__()
            shared = Linear(3, 3, device=torch.device("meta"))
            self.first = shared
            self.second = shared

        def forward(self, x):
            return self.second(self.first(x))

        def _cost_repr(self, scope):
            return ()

        def _cost_children(self, scope):
            return (scope.child("first", self.first), scope.child("second", self.second))

    module = SharedParent()
    tree = module.cost_repr()
    counter = FlopCounterMode(display=False)

    with counter:
        module(torch.empty((5, 3), device="meta"))

    assert tuple(child.name for child in tree.children) == ("first", "second")
    substitutions = {symbol: 5 for symbol in tree.find_symbols("tokens")}
    assert matmul_flops(tree, substitutions=substitutions, strict=True).bound_total == counter.get_total_flops()


def test_default_module_children_are_inventory_until_calls_are_authored() -> None:
    class RegistrationOnly(Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = Linear(3, 4)

        def _cost_repr(self, scope):
            return ()

    tree = RegistrationOnly().cost_repr()
    report = matmul_flops(tree, strict=True)

    assert tree.children[0].edge_role == "inventory"
    assert report.terms == ()
    assert report.bound_total == 0
    assert matmul_flops(tree.children[0], strict=True).bound_total == 0


def test_unresolved_local_work_keeps_explicitly_authored_child_calls() -> None:
    class Partial(Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = Linear(3, 4)

        def _cost_children(self, scope):
            return (scope.child("projection", self.projection),)

    tree = Partial().cost_repr()
    report = matmul_flops(tree)

    assert tree.children[0].edge_role == "call"
    assert len(report.terms) == 1
    assert len(report.unsupported) == 1
    with pytest.raises(NotImplementedError, match="unsupported symbolic costs"):
        matmul_flops(tree, strict=True)


def test_structural_containers_preserve_authored_slot_names() -> None:
    module = nn.Sequential(OrderedDict((name, Linear(3, 3)) for name in ("attention", "feed_forward")))

    assert tuple(child.name for child in cost_repr(module).children) == ("attention", "feed_forward")


def test_structural_container_cycles_reach_the_collector_guard() -> None:
    module = nn.Sequential()
    module.add_module("loop", module)

    with pytest.raises(ValueError, match="child graph contains a module cycle"):
        cost_repr(module)


def test_structural_container_subclasses_must_classify_custom_local_work() -> None:
    class SpecializedSequential(nn.Sequential):
        pass

    tree = cost_repr(SpecializedSequential(Linear(3, 3)))

    assert tuple(child.name for child in tree.children) == ("0",)
    assert tree.children[0].edge_role == "inventory"
    assert len(tree.unresolved) == 1
    assert "SpecializedSequential has no static local-cost provider" in tree.unresolved[0]
    assert matmul_flops(tree).terms == ()
    with pytest.raises(NotImplementedError, match="unsupported symbolic costs"):
        matmul_flops(tree, strict=True)


@pytest.mark.parametrize(
    "container",
    [
        nn.ModuleList([Linear(3, 4), Linear(4, 5)]),
        nn.ModuleDict({"input": Linear(3, 4), "output": Linear(4, 5)}),
    ],
)
def test_registration_only_torch_containers_do_not_imply_invocation(container: nn.Module) -> None:
    tree = cost_repr(container)
    report = matmul_flops(tree)

    assert len(tree.children) == 2
    assert all(child.edge_role == "inventory" for child in tree.children)
    assert len(tree.unresolved) == 1
    assert "has no static local-cost provider" in tree.unresolved[0]
    assert report.terms == ()
    assert report.bound_total == 0
    with pytest.raises(NotImplementedError, match="unsupported symbolic costs"):
        matmul_flops(tree, strict=True)


def test_directed_cost_child_graph_rejects_active_ancestor_cycles() -> None:
    class Cyclic(Module):
        def _cost_repr(self, scope):
            return ()

        def _cost_children(self, scope):
            return (scope.child("self", self),)

    with pytest.raises(ValueError, match="child graph contains a module cycle"):
        Cyclic().cost_repr()


def test_directed_cost_children_require_unambiguous_paths() -> None:
    child = Linear(3, 4)

    class InvalidChildren(Module):
        def __init__(self, names: tuple[str, ...]) -> None:
            super().__init__()
            self.names = names

        def _cost_repr(self, scope):
            return ()

        def _cost_children(self, scope):
            return tuple(scope.child(name, child) for name in self.names)

    with pytest.raises(ValueError, match="without dots"):
        InvalidChildren(("nested.child",)).cost_repr()
    with pytest.raises(ValueError, match="duplicate directed cost child names"):
        InvalidChildren(("child", "child")).cost_repr()


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

    tensors = [TensorRepr((2, 3)), TensorRepr((2, 4))]
    concatenation = CostRepr("concatenation", torch.ops.aten.cat.default, {"tensors": tensors})
    tensors.append(TensorRepr((2, 5)))

    assert concatenation.arguments["tensors"] == (TensorRepr((2, 3)), TensorRepr((2, 4)))


def test_symbolic_tensor_requires_a_torch_dtype() -> None:
    with pytest.raises(TypeError, match="torch.dtype or None"):
        TensorRepr((2, 3), "float32")  # ty: ignore[invalid-argument-type]


def test_symbolic_tensor_reports_only_its_intrinsic_logical_footprint() -> None:
    tokens = sympy.Dummy("tokens", integer=True, nonnegative=True)
    symbolic = TensorRepr((tokens, 3), torch.float16)
    symbolic_nbytes = symbolic.logical_nbytes

    assert TensorRepr((), torch.float32).numel == 1
    assert TensorRepr((), torch.float32).logical_nbytes == 4
    assert TensorRepr((2, 0, 7), torch.float64).numel == 0
    assert sympy.simplify(symbolic.numel - 3 * tokens) == 0
    assert symbolic_nbytes is not None
    assert sympy.simplify(symbolic_nbytes - 6 * tokens) == 0
    assert TensorRepr((tokens, 3)).logical_nbytes is None


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


def test_public_cost_records_reject_malformed_structure_at_the_boundary() -> None:
    cost = CostRepr(
        "valid",
        torch.ops.aten.mm.default,
        {"self": TensorRepr((2, 3)), "mat2": TensorRepr((3, 4))},
    )

    with pytest.raises(ValueError, match="non-empty semantic name"):
        CostRepr(1, torch.ops.aten.mm.default, cost.arguments)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="costs must be CostRepr"):
        analytics.CostTree("tree", "Module", costs=(object(),))  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="SymPy Symbol"):
        SymbolRepr("width", "width", object())
    with pytest.raises(TypeError, match="source must be CostRepr"):
        CostTerm("tree", object(), sympy.Integer(1))  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="scalar symbolic expressions"):
        CostTerm("tree", cost, [])
    with pytest.raises(TypeError, match="conditions must be SymPy equalities"):
        analytics.CostReport((), 0, 0, {}, conditions=("invalid",))
    with pytest.raises(TypeError, match="expects a CostTree"):
        matmul_flops(object())  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="edge role"):
        analytics.CostTree("tree", "Module", edge_role="unknown")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="cannot carry call arguments or repetitions"):
        analytics.CostTree("tree", "Module", repetitions=2, edge_role="inventory")
    with pytest.raises(ValueError, match="cannot carry call arguments or repetitions"):
        analytics._CostChild("child", nn.Identity(), arguments={"tokens": 1}, edge_role="inventory")


def test_public_cost_trees_preserve_unambiguous_child_paths_and_symbol_scopes() -> None:
    identity = sympy.Dummy("width", integer=True, nonnegative=True)
    symbol = SymbolRepr("width", "width", identity)
    child = analytics.CostTree("child", "Module")

    with pytest.raises(ValueError, match="must not contain dots"):
        analytics.CostTree("parent", "Module", children=(analytics.CostTree("nested.child", "Module"),))
    with pytest.raises(ValueError, match="child names must be unique"):
        analytics.CostTree("parent", "Module", children=(child, child))
    with pytest.raises(ValueError, match="locally declared"):
        analytics.CostTree("tree", "Module", symbols=(symbol,), arguments={sympy.Dummy("other"): 3})

    shared_identity = sympy.Symbol("shared", integer=True, nonnegative=True)
    first = analytics.CostTree(
        "first",
        "Module",
        symbols=(SymbolRepr("width", "width", shared_identity, 2),),
    )
    second = analytics.CostTree(
        "second",
        "Module",
        symbols=(SymbolRepr("width", "width", shared_identity, 3),),
    )
    with pytest.raises(ValueError, match="unique across the subtree"):
        analytics.CostTree("parent", "Module", children=(first, second))


def test_public_cost_reports_validate_closed_consistent_symbolic_results() -> None:
    identity = sympy.Dummy("tokens", integer=True, nonnegative=True)
    foreign = sympy.Dummy("foreign", integer=True, nonnegative=True)
    cost = CostRepr(
        "projection",
        torch.ops.aten.mm.default,
        {"self": TensorRepr((identity, 3)), "mat2": TensorRepr((3, 4))},
    )
    term = CostTerm("model", cost, 24 * identity)

    with pytest.raises(ValueError, match="symbolic total"):
        analytics.CostReport((term,), 25 * identity, 50, {identity: 2}, known_symbols=frozenset({identity}))
    with pytest.raises(ValueError, match="bound total"):
        analytics.CostReport((term,), 24 * identity, 999, {identity: 2}, known_symbols=frozenset({identity}))
    with pytest.raises(ValueError, match="unknown symbolic identities"):
        analytics.CostReport((term,), 24 * identity, 24 * foreign, {}, known_symbols=frozenset({identity}))
    with pytest.raises(ValueError, match="definitions contain a cycle"):
        analytics.CostReport(
            (term,),
            24 * identity,
            24 * identity,
            {identity: foreign, foreign: identity},
            known_symbols=frozenset({identity, foreign}),
        )
    with pytest.raises(ValueError, match="symbolic facts are inconsistent"):
        analytics.CostReport(
            (term,),
            24 * identity,
            48,
            {identity: 2},
            conditions=(sympy.Eq(identity, 3, evaluate=False),),
            known_symbols=frozenset({identity}),
        )

    dependent = sympy.Dummy("dependent", integer=True, nonnegative=True)
    deferred_domain = analytics.CostReport(
        (),
        0,
        0,
        {identity: dependent - 1},
        known_symbols=frozenset({identity, dependent}),
    )
    with pytest.raises(ValueError, match="must be nonnegative"):
        deferred_domain.substitute({dependent: 0})


def test_cost_repr_requires_symbolic_metadata_for_tensor_schema_arguments() -> None:
    with pytest.raises(TypeError, match="must be represented by TensorRepr"):
        CostRepr("live tensor", torch.ops.aten.sin.default, {"self": torch.empty(2, 3)})
    with pytest.raises(TypeError, match="sequence of TensorRepr"):
        CostRepr("not a sequence", torch.ops.aten.cat.default, {"tensors": TensorRepr((2, 3))})
    with pytest.raises(TypeError, match="must be represented by TensorRepr"):
        CostRepr(
            "live tensor list",
            torch.ops.aten.cat.default,
            {"tensors": [TensorRepr((2, 3)), torch.empty(2, 4)]},
        )
    with pytest.raises(TypeError, match="rather than live torch.Tensor"):
        CostRepr(
            "tensor in scalar slot",
            torch.ops.aten.cat.default,
            {"tensors": [TensorRepr((2, 3))], "dim": torch.tensor(0)},
        )
    with pytest.raises(TypeError, match="non-tensor argument 'dim'"):
        CostRepr(
            "tensor metadata in scalar slot",
            torch.ops.aten.cat.default,
            {"tensors": [TensorRepr((2, 3))], "dim": TensorRepr(())},
        )


def test_cost_repr_recurses_through_optional_tensor_lists() -> None:
    valid = CostRepr(
        "advanced index",
        torch.ops.aten.index.Tensor,
        {"self": TensorRepr((3, 4)), "indices": [TensorRepr((2,)), None]},
    )

    assert valid.arguments["indices"] == (TensorRepr((2,)), None)
    with pytest.raises(TypeError, match="must be represented by TensorRepr"):
        CostRepr(
            "invalid advanced index",
            torch.ops.aten.index.Tensor,
            {"self": TensorRepr((3, 4)), "indices": [1, None]},
        )


def test_cost_repr_rejects_tensor_metadata_nested_in_nontensor_lists() -> None:
    with pytest.raises(TypeError, match="non-tensor argument 'shape'"):
        CostRepr(
            "invalid reshape",
            torch.ops.aten.reshape.default,
            {"self": TensorRepr((2, 3)), "shape": [TensorRepr(())]},
        )


def test_module_state_footprint_uses_torch_registered_state_traversal() -> None:
    class MixedState(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trainable = nn.Parameter(torch.empty((2, 3), dtype=torch.float32))
            self.frozen = nn.Parameter(torch.empty(4, dtype=torch.float16), requires_grad=False)
            self.register_buffer("persistent", torch.empty(5, dtype=torch.uint8))
            self.register_buffer("temporary", torch.empty(2, dtype=torch.float64), persistent=False)

    module = MixedState()
    report = module_state_footprint(nn.ModuleList([module, module]))

    assert report == ModuleStateFootprint(
        parameter_numel=10,
        parameter_bytes=32,
        trainable_parameter_numel=6,
        trainable_parameter_bytes=24,
        buffer_numel=7,
        buffer_bytes=21,
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("parameter_numel", True, TypeError),
        ("parameter_bytes", 1.5, TypeError),
        ("buffer_numel", -1, ValueError),
    ],
)
def test_module_state_footprint_rejects_invalid_concrete_counts(field, value, error) -> None:
    values = {
        "parameter_numel": 4,
        "parameter_bytes": 16,
        "trainable_parameter_numel": 3,
        "trainable_parameter_bytes": 12,
        "buffer_numel": 2,
        "buffer_bytes": 8,
    }
    values[field] = value

    with pytest.raises(error):
        ModuleStateFootprint(**values)


@pytest.mark.parametrize(
    "changes",
    [
        {"parameter_numel": 17},
        {"buffer_numel": 9},
        {"trainable_parameter_numel": 5},
        {"trainable_parameter_bytes": 17},
        {"trainable_parameter_bytes": 2},
        {"trainable_parameter_numel": 0},
    ],
)
def test_module_state_footprint_rejects_inconsistent_subtotals(changes: dict[str, int]) -> None:
    values = {
        "parameter_numel": 4,
        "parameter_bytes": 16,
        "trainable_parameter_numel": 3,
        "trainable_parameter_bytes": 12,
        "buffer_numel": 2,
        "buffer_bytes": 8,
    }
    values.update(changes)

    with pytest.raises(ValueError):
        ModuleStateFootprint(**values)


def test_module_state_footprint_works_for_meta_models_and_rejects_nonmodules() -> None:
    module = Linear(3, 5, device=torch.device("meta"), dtype=torch.float16)
    report = module_state_footprint(module)

    assert report.parameter_numel == 15
    assert report.parameter_bytes == 30
    assert report.trainable_parameter_numel == 15
    assert report.buffer_numel == 0
    assert module.state_footprint() == report
    with pytest.raises(TypeError, match="torch.nn.Module"):
        module_state_footprint(object())  # ty: ignore[invalid-argument-type]


def test_module_state_footprint_counts_logical_views_and_deduplicates_tied_parameters() -> None:
    class TiedAndViewed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            shared = nn.Parameter(torch.empty((1, 6)))
            self.first = nn.Linear(6, 1, bias=False)
            self.second = nn.Linear(6, 1, bias=False)
            self.first.weight = shared
            self.second.weight = shared

            storage = torch.empty(8)
            self.register_buffer("left", storage[:6])
            self.register_buffer("right", storage[2:])

    report = module_state_footprint(TiedAndViewed())

    assert report.parameter_numel == 6
    assert report.parameter_bytes == 24
    assert report.buffer_numel == 12
    assert report.buffer_bytes == 48


def test_module_state_footprint_counts_cross_category_registration_per_category() -> None:
    module = nn.Module()
    tensor = nn.Parameter(torch.empty(3, dtype=torch.float16))
    module.register_parameter("weight", tensor)
    module.register_buffer("mirror", tensor)

    report = module_state_footprint(module)

    assert report.parameter_numel == report.buffer_numel == 3
    assert report.parameter_bytes == report.buffer_bytes == 6


def test_module_state_footprint_rejects_uninitialized_lazy_state() -> None:
    with pytest.raises(ValueError, match="requires initialized"):
        module_state_footprint(nn.LazyLinear(4))


def test_linear_cost_uses_symbolic_aten_operands() -> None:
    tree = Linear(3, 5).cost_repr()
    cost = tree.costs[0]

    assert cost.operation is torch.ops.aten.bmm.default
    assert cost.name == "linear projection"
    assert tuple(record.local_name for record in tree.symbols) == ("tokens", "d_in", "d_out")
    assert all(isinstance(record, SymbolRepr) for record in tree.symbols)
    assert _tensor(cost.arguments["self"]).shape[1:] == (
        tree.find_symbols("tokens")[0],
        tree.find_symbols("d_in")[0],
    )


def test_scoped_symbols_do_not_collide_between_modules() -> None:
    first = Linear(3, 5).cost_repr()
    second = Linear(3, 5).cost_repr()

    assert first.find_symbols("d_in")[0] != second.find_symbols("d_in")[0]
    assert str(first.find_symbols("d_in")[0]) == str(second.find_symbols("d_in")[0])


def test_symbol_environment_is_a_focused_builder_for_immutable_records() -> None:
    captured: dict[str, object] = {}

    class Described(Module):
        def _cost_repr(self, scope):
            s = scope.symbols
            captured["symbols"] = s

            assert s.unbound("batch", "sequence") is s
            assert s.bind(width=4) is s
            assert s.bind(twice_width=2 * s.width) is s
            assert s.batch is s["batch"]
            assert s["batch", "sequence"] == (s.batch, s.sequence)
            assert tuple(s) == (s.batch, s.sequence, s.width, s.twice_width)
            return ()

    tree = Described().cost_repr()
    records = tree.symbols

    assert tuple(record.local_name for record in records) == ("batch", "sequence", "width", "twice_width")
    assert tuple(record.display_name for record in records) == ("batch", "sequence", "width", "twice_width")
    assert all(isinstance(record.symbol, sympy.Dummy) for record in records)
    assert records[0].binding is None
    assert records[2].binding == 4
    assert records[3].binding == 2 * records[2].symbol
    with pytest.raises(RuntimeError, match="environment is frozen"):
        captured["symbols"].unbound("late")  # ty: ignore[unresolved-attribute]


def test_symbol_environment_keeps_display_names_separate_from_local_identity() -> None:
    class Named(Module):
        def _cost_repr(self, scope):
            s = scope.symbols
            s.unbound("query_heads", "kv_heads")
            assert s.display(query_heads="H_q", kv_heads="H_kv") is s
            return ()

    tree = Named().cost_repr()

    assert tuple(record.local_name for record in tree.symbols) == ("query_heads", "kv_heads")
    assert tuple(record.display_name for record in tree.symbols) == ("H_q", "H_kv")
    assert tree.find_symbols("H_q") == (tree.symbols[0].symbol,)
    assert tree.find_symbols("query_heads") == ()


def test_symbol_environment_rejects_invalid_display_names() -> None:
    class InvalidDisplay(Module):
        def __init__(self, local_name: str, display_name: str) -> None:
            super().__init__()
            self.local_name = local_name
            self.display_name = display_name

        def _cost_repr(self, scope):
            scope.symbols.unbound("width")
            scope.symbols.display(**{self.local_name: self.display_name})
            return ()

    with pytest.raises(ValueError, match="undeclared"):
        InvalidDisplay("depth", "D").cost_repr()
    with pytest.raises(ValueError, match="non-empty"):
        InvalidDisplay("width", "").cost_repr()
    with pytest.raises(ValueError, match="non-empty"):
        InvalidDisplay("width", "   ").cost_repr()

    class CollidingDisplay(Module):
        def _cost_repr(self, scope):
            scope.symbols.unbound("width", "depth")
            scope.symbols.display(width="depth")
            return ()

    with pytest.raises(ValueError, match="unique"):
        CollidingDisplay().cost_repr()


@pytest.mark.parametrize("name", ["bind", "_private", "not-valid", "class"])
def test_symbol_environment_reserves_api_and_invalid_names(name: str) -> None:
    class InvalidSymbol(Module):
        def _cost_repr(self, scope):
            scope.symbols.unbound(name)
            return ()

    with pytest.raises(ValueError, match="symbol name"):
        InvalidSymbol().cost_repr()


def test_symbol_environment_rejects_incompatible_definitions_and_cycles() -> None:
    class Incompatible(Module):
        def _cost_repr(self, scope):
            scope.symbols.bind(width=3)
            scope.symbols.bind(width=4)
            return ()

    class Cyclic(Module):
        def _cost_repr(self, scope):
            s = scope.symbols
            s.unbound("left", "right")
            s.bind(left=s.right, right=s.left)
            return ()

    with pytest.raises(ValueError, match="incompatible bindings"):
        Incompatible().cost_repr()
    with pytest.raises(ValueError, match="contain a cycle"):
        Cyclic().cost_repr()


@pytest.mark.parametrize("location", ["binding", "cost", "child_argument", "child_repetition"])
def test_cost_providers_cannot_introduce_undeclared_symbol_identities(location: str) -> None:
    foreign = sympy.Symbol("foreign", integer=True, nonnegative=True)

    class ForeignSymbol(Module):
        def __init__(self) -> None:
            super().__init__()
            self.child = Linear(3, 4)

        def _cost_repr(self, scope):
            s = scope.symbols
            s.unbound("tokens")
            if location == "binding":
                s.bind(width=foreign)
            if location == "cost":
                return (
                    CostRepr(
                        "foreign projection",
                        torch.ops.aten.bmm.default,
                        {
                            "self": TensorRepr((1, s.tokens, foreign)),
                            "mat2": TensorRepr((1, foreign, 4)),
                        },
                    ),
                )
            return ()

        def _cost_children(self, scope):
            if location == "child_argument":
                return (scope.child("child", self.child, arguments={"d_in": foreign}),)
            if location == "child_repetition":
                return (scope.child("child", self.child, repetitions=foreign),)
            return ()

    with pytest.raises(ValueError, match="references undeclared scoped symbols: foreign"):
        ForeignSymbol().cost_repr()


def test_symbolic_dimensions_and_repetitions_reject_definite_negative_values() -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        TensorRepr((-1, 3))
    with pytest.raises(ValueError, match="must be nonnegative"):
        CostRepr(
            "negative repetition",
            torch.ops.aten.bmm.default,
            {"self": TensorRepr((1, 2, 3)), "mat2": TensorRepr((1, 3, 4))},
            repetitions=-1,
        )


@pytest.mark.parametrize("value", [True, False, 1.0, 1.5, float("nan"), float("inf"), sympy.true])
def test_symbolic_dimensions_reject_concrete_noninteger_values(value) -> None:
    with pytest.raises(TypeError):
        TensorRepr((value,))


def test_symbolic_dimensions_retain_unresolved_symbolic_expressions() -> None:
    dimension = sympy.Symbol("dimension")

    assert TensorRepr((dimension,)).shape == (dimension,)


@pytest.mark.parametrize("location", ["axis", "cost_repetition", "child_repetition"])
def test_domain_checks_reject_values_that_become_invalid_after_binding(location: str) -> None:
    class DeferredDomain(Module):
        def __init__(self) -> None:
            super().__init__()
            self.child = Linear(2, 2)

        def _cost_repr(self, scope):
            s = scope.symbols
            s.unbound("tokens")
            axis = s.tokens / 2 if location == "axis" else s.tokens
            repetitions = s.tokens / 2 if location == "cost_repetition" else 1
            return (
                CostRepr(
                    "deferred domain",
                    torch.ops.aten.bmm.default,
                    {
                        "self": TensorRepr((1, axis, 2)),
                        "mat2": TensorRepr((1, 2, 2)),
                    },
                    repetitions=repetitions,
                ),
            )

        def _cost_children(self, scope):
            if location != "child_repetition":
                return ()
            return (scope.child("child", self.child, repetitions=scope.symbols.tokens / 2),)

    tree = DeferredDomain().cost_repr()
    tokens = tree.find_symbols("tokens")[0]
    report = matmul_flops(tree, strict=True)

    assert tokens in report.bound_total.free_symbols
    with pytest.raises(ValueError, match="must be integers"):
        report.substitute({tokens: 3})


def test_domain_checks_reject_values_that_become_negative_after_binding() -> None:
    class DeferredDomain(Module):
        def _cost_repr(self, scope):
            s = scope.symbols
            s.unbound("tokens")
            return (
                CostRepr(
                    "deferred domain",
                    torch.ops.aten.bmm.default,
                    {
                        "self": TensorRepr((1, s.tokens - 10, 2)),
                        "mat2": TensorRepr((1, 2, 2)),
                    },
                ),
            )

    tree = DeferredDomain().cost_repr()
    tokens = tree.find_symbols("tokens")[0]

    with pytest.raises(ValueError, match="must be nonnegative"):
        matmul_flops(tree, substitutions={tokens: 3}, strict=True)


def test_domain_checks_retain_child_bindings_shadowed_by_parent_arguments() -> None:
    class Child(Module):
        def _cost_repr(self, scope):
            s = scope.symbols
            s.unbound("width", "offset")
            s.bind(width=s.offset - 10)
            return ()

    class Parent(Module):
        def __init__(self) -> None:
            super().__init__()
            self.child = Child()

        def _cost_repr(self, scope):
            scope.symbols.unbound("width")
            return ()

        def _cost_children(self, scope):
            return (scope.child("child", self.child, arguments={"width": scope.symbols.width}),)

    tree = Parent().cost_repr()
    offset = tree.find_symbols("offset")[0]

    with pytest.raises(ValueError, match="must be nonnegative"):
        matmul_flops(tree, substitutions={offset: 3}, strict=True)


def test_domain_expressions_survive_incremental_report_substitution() -> None:
    class DeferredDomains(Module):
        def _cost_repr(self, scope):
            s = scope.symbols
            s.unbound("tokens", "width")
            return (
                CostRepr(
                    "deferred domains",
                    torch.ops.aten.bmm.default,
                    {
                        "self": TensorRepr((1, s.tokens / 2, s.width - 10)),
                        "mat2": TensorRepr((1, s.width - 10, 2)),
                    },
                ),
            )

    tree = DeferredDomains().cost_repr()
    tokens = tree.find_symbols("tokens")[0]
    width = tree.find_symbols("width")[0]
    partly_bound = matmul_flops(tree, strict=True).substitute({tokens: 4})

    with pytest.raises(ValueError, match="must be nonnegative"):
        partly_bound.substitute({width: 3})


class _DirectedLinearParent(Module):
    def __init__(self, width: int | None = None, *, argument_name: str = "d_in") -> None:
        super().__init__()
        self.width = width
        self.argument_name = argument_name
        self.projection = Linear(3, 4)

    def _cost_repr(self, scope):
        return ()

    def _cost_children(self, scope):
        s = scope.symbols
        s.unbound("tokens", "width")
        if self.width is not None:
            s.bind(width=self.width)
        return (
            scope.child(
                "projection",
                self.projection,
                arguments={"tokens": s.tokens, self.argument_name: s.width, "d_out": 4},
            ),
        )


def test_child_arguments_create_checked_parent_child_equalities() -> None:
    tree = _DirectedLinearParent().cost_repr()
    tokens = tree.find_symbols("tokens")[0]
    width = tree.find_symbols("width")[0]
    report = matmul_flops(tree, strict=True)

    assert len(report.conditions) == 1
    assert report.conditions[0].subs({width: 3}) == sympy.true
    assert sympy.simplify(report.bound_total - 8 * tokens * width) == 0

    resolved = report.substitute({tokens: 5, width: 3})
    assert resolved.bound_total == 120
    assert resolved.conditions == ()
    with pytest.raises(ValueError, match="symbolic facts are inconsistent"):
        report.substitute({width: 5})


def test_child_arguments_preserve_and_validate_instance_bindings() -> None:
    matching = _DirectedLinearParent(3).cost_repr()
    assert matmul_flops(matching, strict=True).conditions == ()

    with pytest.raises(ValueError, match="symbolic facts are inconsistent"):
        _DirectedLinearParent(5).cost_repr()
    with pytest.raises(ValueError, match="does not declare child argument symbols: input_width"):
        _DirectedLinearParent(argument_name="input_width").cost_repr()


def test_report_substitutions_are_additive_facts_not_overrides() -> None:
    tree = Linear(3, 5).cost_repr()
    tokens = tree.find_symbols("tokens")[0]
    d_in = tree.find_symbols("d_in")[0]

    assert matmul_flops(tree, substitutions={d_in: 3}).conditions == ()
    with pytest.raises(ValueError, match="symbolic facts are inconsistent"):
        matmul_flops(tree, substitutions={d_in: 4})
    with pytest.raises(ValueError, match="unknown symbolic identities"):
        matmul_flops(tree, substitutions={sympy.Dummy("external", integer=True): 1})
    with pytest.raises(ValueError, match="substitution values contain unknown symbolic identities"):
        matmul_flops(tree, substitutions={tokens: sympy.Dummy("external", integer=True)})


def test_report_substitutions_may_relate_known_scoped_symbols() -> None:
    tree = Linear(3, 5).cost_repr()
    tokens = tree.find_symbols("tokens")[0]
    d_in = tree.find_symbols("d_in")[0]

    assert matmul_flops(tree, substitutions={tokens: d_in}, strict=True).bound_total == 90


def test_report_substitutions_reject_symbolic_definition_cycles() -> None:
    class Pair(Module):
        def _cost_repr(self, scope):
            scope.symbols.unbound("left", "right")
            return ()

    tree = Pair().cost_repr()
    left = tree.find_symbols("left")[0]
    right = tree.find_symbols("right")[0]

    with pytest.raises(ValueError, match="definitions contain a cycle"):
        matmul_flops(tree, substitutions={left: right, right: left})


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
    assert isinstance(report.terms[0], CostTerm)

    resolved = report.substitute({tokens: 7})
    assert sympy.simplify(resolved.terms[0].expression - 2 * tokens * d_in * d_out) == 0
    assert resolved.bound_terms[0].expression == 210
    assert sum(term.expression for term in resolved.bound_terms) == resolved.bound_total


@pytest.mark.parametrize(
    ("cost", "invoke", "packet"),
    [
        (
            CostRepr(
                "matrix product",
                torch.ops.aten.mm.default,
                {"self": TensorRepr((2, 3)), "mat2": TensorRepr((3, 4))},
            ),
            lambda: torch.mm(torch.empty((2, 3), device="meta"), torch.empty((3, 4), device="meta")),
            torch.ops.aten.mm,
        ),
        (
            CostRepr(
                "added matrix product",
                torch.ops.aten.addmm.default,
                {
                    "self": TensorRepr(()),
                    "mat1": TensorRepr((2, 3)),
                    "mat2": TensorRepr((3, 4)),
                },
            ),
            lambda: torch.addmm(
                torch.empty((), device="meta"),
                torch.empty((2, 3), device="meta"),
                torch.empty((3, 4), device="meta"),
                beta=0,
                alpha=0,
            ),
            torch.ops.aten.addmm,
        ),
        (
            CostRepr(
                "added batched matrix product",
                torch.ops.aten.baddbmm.default,
                {
                    "self": TensorRepr((1, 2, 1)),
                    "batch1": TensorRepr((5, 2, 3)),
                    "batch2": TensorRepr((5, 3, 4)),
                },
            ),
            lambda: torch.baddbmm(
                torch.empty((1, 2, 1), device="meta"),
                torch.empty((5, 2, 3), device="meta"),
                torch.empty((5, 3, 4), device="meta"),
                beta=0,
                alpha=0,
            ),
            torch.ops.aten.baddbmm,
        ),
    ],
)
def test_torch_registered_dense_product_policies_match_flop_counter(cost, invoke, packet) -> None:
    class Described(Module):
        def _cost_repr(self, scope):
            return (cost,)

    counter = FlopCounterMode(display=False)
    with counter:
        invoke()

    report = matmul_flops(Described().cost_repr(), strict=True)
    assert report.bound_total == counter.get_total_flops()
    assert set(counter.get_flop_counts()["Global"]) == {packet}


def test_dense_product_policy_preserves_symbolic_dimensions_and_operation_repetition() -> None:
    class SymbolicProduct(Module):
        def _cost_repr(self, scope):
            s = scope.symbols
            s.unbound("rows", "inner", "columns", "calls")
            return (
                CostRepr(
                    "repeated matrix product",
                    torch.ops.aten.mm.default,
                    {
                        "self": TensorRepr((s.rows, s.inner)),
                        "mat2": TensorRepr((s.inner, s.columns)),
                    },
                    repetitions=s.calls,
                ),
            )

    tree = SymbolicProduct().cost_repr()
    rows, inner, columns, calls = (tree.find_symbols(name)[0] for name in ("rows", "inner", "columns", "calls"))
    report = matmul_flops(tree, strict=True)

    assert sympy.simplify(report.symbolic_total - 2 * calls * rows * inner * columns) == 0
    assert report.substitute({rows: 3, inner: 0, columns: 5, calls: 7}).bound_total == 0


@pytest.mark.parametrize(
    "cost",
    [
        CostRepr(
            "wrong-rank matrix product",
            torch.ops.aten.mm.default,
            {"self": TensorRepr((1, 2, 3)), "mat2": TensorRepr((3, 4))},
        ),
        CostRepr(
            "mismatched added product",
            torch.ops.aten.addmm.default,
            {
                "self": TensorRepr((2, 4)),
                "mat1": TensorRepr((2, 3)),
                "mat2": TensorRepr((5, 4)),
            },
        ),
        CostRepr(
            "unbroadcastable added product",
            torch.ops.aten.addmm.default,
            {
                "self": TensorRepr((3, 5)),
                "mat1": TensorRepr((2, 3)),
                "mat2": TensorRepr((3, 4)),
            },
        ),
        CostRepr(
            "mismatched batched product",
            torch.ops.aten.baddbmm.default,
            {
                "self": TensorRepr((5, 2, 4)),
                "batch1": TensorRepr((5, 2, 3)),
                "batch2": TensorRepr((7, 3, 4)),
            },
        ),
        CostRepr(
            "over-ranked batched addend",
            torch.ops.aten.baddbmm.default,
            {
                "self": TensorRepr((1, 5, 2, 4)),
                "batch1": TensorRepr((5, 2, 3)),
                "batch2": TensorRepr((5, 3, 4)),
            },
        ),
    ],
)
def test_dense_product_policies_reject_structurally_invalid_operands(cost) -> None:
    class Described(Module):
        def _cost_repr(self, scope):
            return (cost,)

    with pytest.raises(ValueError):
        matmul_flops(Described().cost_repr(), strict=True)


@pytest.mark.parametrize(
    ("cost", "invoke"),
    [
        (
            CostRepr(
                "directionally invalid addmm addend",
                torch.ops.aten.addmm.default,
                {
                    "self": TensorRepr((2, 4)),
                    "mat1": TensorRepr((1, 3)),
                    "mat2": TensorRepr((3, 4)),
                },
            ),
            lambda: torch.addmm(torch.empty((2, 4)), torch.empty((1, 3)), torch.empty((3, 4))),
        ),
        (
            CostRepr(
                "directionally invalid baddbmm addend",
                torch.ops.aten.baddbmm.default,
                {
                    "self": TensorRepr((2, 1, 4)),
                    "batch1": TensorRepr((1, 1, 3)),
                    "batch2": TensorRepr((1, 3, 4)),
                },
            ),
            lambda: torch.baddbmm(
                torch.empty((2, 1, 4)),
                torch.empty((1, 1, 3)),
                torch.empty((1, 3, 4)),
            ),
        ),
        (
            CostRepr(
                "over-ranked addmm addend",
                torch.ops.aten.addmm.default,
                {
                    "self": TensorRepr((1, 2, 4)),
                    "mat1": TensorRepr((2, 3)),
                    "mat2": TensorRepr((3, 4)),
                },
            ),
            lambda: torch.addmm(torch.empty((1, 2, 4)), torch.empty((2, 3)), torch.empty((3, 4))),
        ),
    ],
)
def test_added_product_policy_matches_torch_directional_expansion(cost, invoke) -> None:
    class Described(Module):
        def _cost_repr(self, scope):
            return (cost,)

    with pytest.raises(RuntimeError):
        invoke()
    with pytest.raises(ValueError, match="addend"):
        matmul_flops(Described().cost_repr(), strict=True)


def test_linear_symbolic_cost_matches_meta_flop_counter() -> None:
    linear = Linear(3, 5, device=torch.device("meta"))
    tree = linear.cost_repr()
    tokens = tree.find_symbols("tokens")[0]
    counter = FlopCounterMode(display=False)

    with counter:
        linear(torch.empty((7, 3), device="meta"))

    assert set(counter.get_flop_counts()["Global"]) == {torch.ops.aten.bmm}
    assert matmul_flops(tree, substitutions={tokens: 7}).bound_total == counter.get_total_flops()


def test_logical_matmul_cost_survives_einx_unit_axis_strength_reduction() -> None:
    linear = Linear(1, 5, device=torch.device("meta"))
    linear_tree = linear.cost_repr()
    linear_tokens = linear_tree.find_symbols("tokens")[0]
    linear_counter = FlopCounterMode(display=False)

    with linear_counter:
        linear(torch.empty((8, 1), device="meta"))

    linear_report = matmul_flops(linear_tree, substitutions={linear_tokens: 8}, strict=True)
    assert linear_tree.costs[0].operation is torch.ops.aten.bmm.default
    assert linear_report.bound_total == 80
    assert linear_counter.get_total_flops() == 0

    swiglu = SwiGLU_packed_input(1, 5, device=torch.device("meta"))
    swiglu_tree = swiglu.cost_repr()
    swiglu_tokens = swiglu_tree.find_symbols("tokens")[0]
    swiglu_counter = FlopCounterMode(display=False)

    with swiglu_counter:
        swiglu(torch.empty((8, 1), device="meta"))

    swiglu_report = matmul_flops(swiglu_tree, substitutions={swiglu_tokens: 8}, strict=True)
    assert swiglu_report.bound_total == 240
    assert swiglu_counter.get_total_flops() == 80


def test_unsupported_operations_and_external_modules_remain_visible() -> None:
    unsupported = CostRepr(
        "pointwise sine",
        torch.ops.aten.sin.default,
        {"self": TensorRepr((3, 4))},
    )

    class Unsupported(Module):
        def _cost_repr(self, scope):
            return (unsupported,)

    report = matmul_flops(Unsupported().cost_repr())
    unclassified_report = matmul_flops(Module().cost_repr())
    external_report = matmul_flops(cost_repr(nn.Linear(3, 4)))

    assert report.symbolic_total == 0
    assert "no matmul policy for aten.sin.default" in report.unsupported[0]
    assert any(
        "has not classified its static local matmul work" in message for message in unclassified_report.unsupported
    )
    assert any("has no static local-cost provider" in message for message in external_report.unsupported)
    with pytest.raises(NotImplementedError, match="unsupported symbolic costs"):
        matmul_flops(Unsupported().cost_repr(), strict=True)


@pytest.mark.parametrize(
    ("module_type", "term_count"),
    [
        (SwiGLU_delegate, 3),
        (SwiGLU_own_weights, 3),
        (SwiGLU_packed_input, 2),
    ],
)
def test_swiglu_variants_preserve_distinct_operations_and_equal_totals(module_type, term_count) -> None:
    module = module_type(4, 7, device=torch.device("meta"))
    tree = module.cost_repr()
    tokens = tree.find_symbols("tokens")[0]
    report = matmul_flops(tree, substitutions={tokens: 6}, strict=True)
    counter = FlopCounterMode(display=False)

    with counter:
        module(torch.empty((2, 3, 4), device="meta"))

    assert len(report.terms) == term_count
    assert report.bound_total == counter.get_total_flops()
    assert set(counter.get_flop_counts()["Global"]) == {torch.ops.aten.bmm}


@pytest.mark.parametrize("num_kv_heads", [4, 2, 1])
def test_self_attention_symbolic_cost_matches_meta_flop_counter(num_kv_heads: int) -> None:
    module = MultiheadSelfAttention(
        d_model=8,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        d_k=2,
        d_v=2,
        rope=RotaryPositionalEmbedding(10_000.0, 2, 8, device=torch.device("meta")),
        device=torch.device("meta"),
    )
    tree = module.cost_repr()
    substitutions = {
        tree.find_symbols("batch")[0]: 2,
        tree.find_symbols("sequence")[0]: 3,
    }
    report = matmul_flops(tree, substitutions=substitutions, strict=True)
    counter = FlopCounterMode(display=False)

    with counter:
        module(torch.empty((2, 3, 8), device="meta"))

    assert len(report.terms) == 4
    assert report.bound_total == counter.get_total_flops()
    assert set(counter.get_flop_counts()["Global"]) == {torch.ops.aten.bmm}


@pytest.mark.parametrize("activation_layout", ["head_before_sequence", "head_after_sequence"])
@pytest.mark.parametrize("num_kv_heads", [4, 2, 1])
def test_self_attention_cost_matches_generalized_batch_and_head_layouts(
    activation_layout: Literal["head_before_sequence", "head_after_sequence"],
    num_kv_heads: int,
) -> None:
    module = MultiheadSelfAttention(
        d_model=12,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        d_k=4,
        d_v=2,
        rope=RotaryPositionalEmbedding(10_000.0, 4, 8, device=torch.device("meta")),
        _layout_strategy=activation_layout,
        device=torch.device("meta"),
    )
    tree = module.cost_repr()
    report = matmul_flops(
        tree,
        substitutions={
            tree.find_symbols("batch")[0]: 6,
            tree.find_symbols("sequence")[0]: 5,
        },
        strict=True,
    )
    counter = FlopCounterMode(display=False)

    with counter:
        module(torch.empty((2, 3, 5, 12), device="meta"))

    assert report.bound_total == counter.get_total_flops()
    assert set(counter.get_flop_counts()["Global"]) == {torch.ops.aten.bmm}


def test_grouped_attention_encodes_batching_and_heads_in_operands() -> None:
    tree = MultiheadSelfAttention(8, 4, num_kv_heads=2, d_k=2, d_v=2).cost_repr()
    scores = tree.costs[1]
    batch = tree.find_symbols("batch")[0]
    sequence = tree.find_symbols("sequence")[0]

    bound_shape = tuple(axis.subs(tree.bindings) for axis in _tensor(scores.arguments["self"]).shape)
    assert bound_shape == (2 * batch, 2 * sequence, 2)
    assert scores.repetitions == 1


def test_generic_attention_remains_visible_as_invocation_dependent() -> None:
    tree = MultiheadAttention(8, 4).cost_repr()
    report = matmul_flops(tree)

    assert any("has not classified its static local matmul work" in message for message in report.unsupported)
    with pytest.raises(NotImplementedError, match="unsupported symbolic costs"):
        matmul_flops(tree, strict=True)


def test_transformer_lm_summarizes_authored_layers_and_matches_meta_flop_counter() -> None:
    model = TransformerLM(
        vocab_size=17,
        context_length=8,
        d_model=8,
        num_layers=3,
        num_heads=4,
        d_ff=12,
        rope_theta=10_000.0,
        device=torch.device("meta"),
    )
    tree = model.cost_repr()
    batch = tree.find_symbols("batch")[0]
    sequence = tree.find_symbols("sequence")[0]
    num_layers = tree.find_symbols("num_layers")[0]
    report = matmul_flops(tree, substitutions={batch: 2, sequence: 5}, strict=True)
    counter = FlopCounterMode(display=False)

    with counter:
        model(torch.empty((2, 5), device="meta", dtype=torch.long))

    assert tuple(child.name for child in tree.children) == ("token_embeddings", "layers", "ln_final", "lm_head")
    assert tree.children[1].repetitions == num_layers
    assert tree.bindings[num_layers] == 3
    assert len(report.terms) == 7
    assert report.bound_total == counter.get_total_flops()
    assert report.unsupported == ()


def test_transformer_lm_keeps_invocation_shape_symbolic_until_bound() -> None:
    tree = TransformerLM(17, 8, 8, 2, 4, 12, 10_000.0, device=torch.device("meta")).cost_repr()
    report = matmul_flops(tree, strict=True)
    batch = tree.find_symbols("batch")[0]
    sequence = tree.find_symbols("sequence")[0]

    assert batch in report.bound_total.free_symbols
    assert sequence in report.bound_total.free_symbols
    assert report.substitute({batch: 1, sequence: 8}).bound_total.is_Integer


def test_transformer_lm_observer_binds_positional_and_keyword_call_shapes() -> None:
    model = TransformerLM(
        vocab_size=17,
        context_length=8,
        d_model=8,
        num_layers=2,
        num_heads=4,
        d_ff=12,
        rope_theta=10_000.0,
        device=torch.device("meta"),
    )
    observer = model.observe_costs()
    counter = FlopCounterMode(display=False)

    with observer, counter:
        positional = model(torch.empty((2, 3, 5), device="meta", dtype=torch.long))
        keyword = model(token_ids=torch.empty((3, 4), device="meta", dtype=torch.long))

    reports = observer.matmul_flops(strict=True)
    assert positional.shape == (2, 3, 5, 17)
    assert keyword.shape == (3, 4, 17)
    assert len(reports) == 2
    assert reports[0].symbolic_total == reports[1].symbolic_total
    assert reports[0].bound_total != reports[1].bound_total
    assert all(report.bound_total.is_Integer for report in reports)
    assert sum(report.bound_total for report in reports) == counter.get_total_flops()


@pytest.mark.parametrize(
    ("factory", "input_shape"),
    [
        (lambda: Linear(4, 5, device=torch.device("meta")), (2, 3, 4)),
        (lambda: SwiGLU_delegate(4, 7, device=torch.device("meta")), (2, 3, 4)),
        (lambda: SwiGLU_own_weights(4, 7, device=torch.device("meta")), (2, 3, 4)),
        (lambda: SwiGLU_packed_input(4, 7, device=torch.device("meta")), (2, 3, 4)),
        (
            lambda: MultiheadSelfAttention(
                d_model=4,
                num_heads=2,
                d_k=2,
                d_v=3,
                device=torch.device("meta"),
            ),
            (2, 3, 5, 4),
        ),
        (
            lambda: GPTDecoderLayer(
                d_model=4,
                num_heads=2,
                d_ff=7,
                max_seq_len=8,
                theta=10_000.0,
                device=torch.device("meta"),
            ),
            (2, 3, 5, 4),
        ),
        (
            lambda: GPTDecoderLayer(
                d_model=4,
                num_heads=2,
                d_ff=7,
                max_seq_len=8,
                theta=10_000.0,
                device=torch.device("meta"),
            ).attn,
            (2, 3, 5, 4),
        ),
        (
            lambda: GPTDecoderLayer(
                d_model=4,
                num_heads=2,
                d_ff=7,
                max_seq_len=8,
                theta=10_000.0,
                device=torch.device("meta"),
            ).ffn,
            (2, 3, 5, 4),
        ),
    ],
)
def test_core_module_observers_bind_interface_axes_and_match_flop_counter(factory, input_shape) -> None:
    module = factory()
    observer = module.observe_costs()
    counter = FlopCounterMode(display=False)

    with observer, counter:
        module(x=torch.empty(input_shape, device="meta"))

    report, = observer.matmul_flops(strict=True)
    assert report.bound_total.is_Integer
    assert report.bound_total == counter.get_total_flops()


def test_transformer_lm_folding_rejects_layer_count_drift() -> None:
    model = TransformerLM(17, 8, 8, 2, 4, 12, 10_000.0, device=torch.device("meta"))
    model.num_layers = 3

    with pytest.raises(ValueError, match="num_layers to match the registered layer count"):
        model.cost_repr()


def test_transformer_lm_folding_requires_structure_not_weight_identity() -> None:
    model = TransformerLM(17, 8, 8, 2, 4, 12, 10_000.0, device=torch.device("meta"))
    model.layers[1] = GPTDecoderLayer(8, 4, 12, 8, 10_000.0, device=torch.device("meta"))

    model.cost_repr()

    model.layers[1] = GPTDecoderLayer(8, 4, 16, 8, 10_000.0, device=torch.device("meta"))
    with pytest.raises(ValueError, match="requires homogeneous cost-driving configuration"):
        model.cost_repr()


def test_transformer_lm_folding_checks_same_shape_attention_configurations() -> None:
    model = TransformerLM(17, 8, 4, 2, 2, 12, 10_000.0, device=torch.device("meta"))
    configurations = ((1, 1, 2, 4), (2, 1, 2, 2))
    for layer, (query_heads, kv_heads, d_k, d_v) in zip(model.layers, configurations, strict=True):
        assert isinstance(layer, GPTDecoderLayer)
        layer.attn.update = MultiheadSelfAttention(
            d_model=4,
            num_heads=query_heads,
            num_kv_heads=kv_heads,
            d_k=d_k,
            d_v=d_v,
            rope=RotaryPositionalEmbedding(10_000.0, d_k, 8, device=torch.device("meta")),
            device=torch.device("meta"),
        )

    first_layer, second_layer = model.layers
    assert isinstance(first_layer, GPTDecoderLayer)
    assert isinstance(second_layer, GPTDecoderLayer)
    first_state = tuple(parameter.shape for parameter in first_layer.attn.update.parameters())
    second_state = tuple(parameter.shape for parameter in second_layer.attn.update.parameters())
    assert first_state == second_state
    with pytest.raises(ValueError, match="requires homogeneous cost-driving configuration"):
        model.cost_repr()


def test_transformer_lm_folding_ignores_rope_capacity_when_cost_is_unchanged() -> None:
    model = TransformerLM(17, 8, 8, 2, 4, 12, 10_000.0, device=torch.device("meta"))
    model.layers[1] = GPTDecoderLayer(8, 4, 12, 16, 10_000.0, device=torch.device("meta"))

    model.cost_repr()


def test_transformer_lm_folding_checks_delegated_module_types() -> None:
    model = TransformerLM(17, 8, 8, 2, 4, 12, 10_000.0, device=torch.device("meta"))
    second_layer = model.layers[1]
    assert isinstance(second_layer, GPTDecoderLayer)
    second_layer.attn.norm = Linear(8, 8, device=torch.device("meta"))

    with pytest.raises(ValueError, match="requires homogeneous cost-driving configuration"):
        model.cost_repr()


def test_transformer_lm_state_footprint_matches_architectural_formula() -> None:
    vocab_size = 17
    context_length = 8
    d_model = 8
    num_layers = 3
    num_heads = 4
    d_ff = 12
    model = TransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        10_000.0,
        device=torch.device("meta"),
        dtype=torch.float16,
    )

    expected_parameters = (
        2 * vocab_size * d_model
        + num_layers * (4 * d_model**2 + 3 * d_model * d_ff + 2 * d_model)
        + d_model
    )
    expected_rope_buffers = num_layers * context_length * (d_model // num_heads)
    footprint = model.state_footprint()

    assert footprint.parameter_numel == footprint.trainable_parameter_numel == expected_parameters
    assert footprint.parameter_bytes == footprint.trainable_parameter_bytes == 2 * expected_parameters
    assert footprint.buffer_numel == expected_rope_buffers
    assert footprint.buffer_bytes == 4 * expected_rope_buffers
