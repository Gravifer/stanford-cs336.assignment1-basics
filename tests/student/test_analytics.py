"""Tests for static symbolic model analytics."""

import pytest
import sympy
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

import cs336_basics.nn.analytics as analytics
from cs336_basics.nn import DeltaLayer, Module
from cs336_basics.nn.analytics import CostRepr, CostTerm, SymbolRepr, TensorRepr, cost_repr, matmul_flops
from cs336_basics.nn.attention import MultiheadAttention, MultiheadSelfAttention, RotaryPositionalEmbedding
from cs336_basics.nn.feed_forward import SwiGLU_delegate, SwiGLU_own_weights, SwiGLU_packed_input
from cs336_basics.nn.model import TransformerLM
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
    d_in = tree.find_symbols("d_in")[0]

    assert matmul_flops(tree, substitutions={d_in: 3}).conditions == ()
    with pytest.raises(ValueError, match="symbolic facts are inconsistent"):
        matmul_flops(tree, substitutions={d_in: 4})
    with pytest.raises(ValueError, match="unknown symbolic identities"):
        matmul_flops(tree, substitutions={sympy.Dummy("external", integer=True): 1})


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
