"""Symbolic descriptions and observations of neural-network resource use."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch
from torch import nn


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


__all__ = [
    "CostReport",
    "CostRepr",
    "CostTree",
    "Module",
    "TensorRepr",
    "cost_repr",
    "matmul_flops",
]


_UNBOUND = object()


def _sympy() -> Any:
    """Import the development-only symbolic dependency when analytics is used."""
    try:
        import sympy
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "symbolic model analytics requires the development dependency; "
            "install the project's dev dependency group"
        ) from error
    return sympy


def _immutable_mapping(mapping: Mapping[Any, Any]) -> Mapping[Any, Any]:
    """Copy a mapping before exposing an immutable view of it."""
    return MappingProxyType(dict(mapping))


def _expression(value: object) -> Any:
    """Normalize an integer or symbolic value into a SymPy expression."""
    sympy = _sympy()
    expression = sympy.sympify(value)
    if expression.is_integer is False:
        raise TypeError(f"cost dimensions and repetitions must be integer expressions, got {value!r}")
    return expression


@dataclass(frozen=True)
class TensorRepr:
    """Symbolic metadata for one tensor operand without allocating tensor data."""

    shape: tuple[Any, ...]
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(_expression(axis) for axis in self.shape))


@dataclass(frozen=True)
class CostRepr:
    """One semantically named invocation of an exact ATen overload."""

    name: str
    operation: torch._ops.OpOverload
    arguments: Mapping[str, Any]
    repetitions: Any = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a cost representation requires a non-empty semantic name")
        if not isinstance(self.operation, torch._ops.OpOverload):
            raise TypeError("operation must be an exact torch.ops overload, such as torch.ops.aten.bmm.default")

        arguments = dict(self.arguments)
        schema_arguments = {argument.name: argument for argument in self.operation._schema.arguments}
        unexpected = arguments.keys() - schema_arguments.keys()
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ValueError(f"{self.operation} has no schema arguments named: {names}")
        missing = {
            name
            for name, argument in schema_arguments.items()
            if name not in arguments and not argument.has_default_value()
        }
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{self.operation} requires schema arguments: {names}")

        object.__setattr__(self, "arguments", _immutable_mapping(arguments))
        object.__setattr__(self, "repetitions", _expression(self.repetitions))


@dataclass(frozen=True)
class _CostChild:
    """Directed traversal from one module description to an immediate child."""

    name: str
    module: nn.Module
    repetitions: Any = 1
    bindings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a directed cost child requires a non-empty name")
        object.__setattr__(self, "repetitions", _expression(self.repetitions))
        object.__setattr__(self, "bindings", _immutable_mapping(self.bindings))


class _CostScope:
    """Allocate identity-distinct symbols and bindings local to one module."""

    def __init__(self) -> None:
        self._symbols: dict[Any, str] = {}
        self._by_name: dict[str, Any] = {}
        self._bindings: dict[Any, Any] = {}

    @property
    def symbols(self) -> Mapping[Any, str]:
        """Return local symbol identities and their human-facing names."""
        return _immutable_mapping(self._symbols)

    @property
    def bindings(self) -> Mapping[Any, Any]:
        """Return optional instance bindings for local symbols."""
        return _immutable_mapping(self._bindings)

    def symbol(self, name: str, value: object = _UNBOUND, *, positive: bool = True) -> Any:
        """Return one locally scoped integer symbol, optionally with a binding."""
        if not name:
            raise ValueError("a symbolic dimension requires a non-empty display name")
        if name in self._by_name:
            symbol = self._by_name[name]
            if value is not _UNBOUND:
                binding = _expression(value)
                previous = self._bindings.get(symbol, binding)
                if previous != binding:
                    raise ValueError(f"symbol {name!r} received incompatible bindings {previous} and {binding}")
                self._bindings[symbol] = binding
            return symbol

        sympy = _sympy()
        symbol = sympy.Dummy(name, integer=True, positive=positive)
        self._symbols[symbol] = name
        self._by_name[name] = symbol
        if value is not _UNBOUND:
            self._bindings[symbol] = _expression(value)
        return symbol

    def child(
        self,
        name: str,
        module: nn.Module,
        *,
        repetitions: Any = 1,
        bindings: Mapping[str, Any] | None = None,
    ) -> _CostChild:
        """Describe how collection should enter one immediate child module."""
        return _CostChild(name, module, repetitions, {} if bindings is None else bindings)


@dataclass(frozen=True)
class CostTree:
    """Immutable symbolic costs arranged according to the authored module tree."""

    name: str
    module_type: str
    costs: tuple[CostRepr, ...] = ()
    children: tuple[CostTree, ...] = ()
    repetitions: Any = 1
    symbols: Mapping[Any, str] = field(default_factory=dict)
    bindings: Mapping[Any, Any] = field(default_factory=dict)
    unresolved: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "costs", tuple(self.costs))
        object.__setattr__(self, "children", tuple(self.children))
        object.__setattr__(self, "repetitions", _expression(self.repetitions))
        object.__setattr__(self, "symbols", _immutable_mapping(self.symbols))
        object.__setattr__(self, "bindings", _immutable_mapping(self.bindings))
        object.__setattr__(self, "unresolved", tuple(self.unresolved))

    def find_symbols(self, name: str) -> tuple[Any, ...]:
        """Return every scoped symbol displayed with ``name`` in this subtree."""
        found = tuple(symbol for symbol, display_name in self.symbols.items() if display_name == name)
        return found + tuple(symbol for child in self.children for symbol in child.find_symbols(name))


@dataclass(frozen=True)
class _CostTerm:
    """One policy result attributed to a module path and source representation."""

    path: str
    source: CostRepr
    expression: Any


@dataclass(frozen=True)
class CostReport:
    """Structured symbolic result of applying one policy to a cost tree."""

    terms: tuple[_CostTerm, ...]
    symbolic_total: Any
    bound_total: Any
    bindings: Mapping[Any, Any]
    unsupported: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        object.__setattr__(self, "bindings", _immutable_mapping(self.bindings))
        object.__setattr__(self, "unsupported", tuple(self.unsupported))

    def substitute(self, bindings: Mapping[Any, Any]) -> CostReport:
        """Return a report with additional symbolic bindings applied."""
        combined = dict(self.bindings)
        combined.update({symbol: _expression(value) for symbol, value in bindings.items()})
        return replace(self, bound_total=_substitute(self.symbolic_total, combined), bindings=combined)


class Module(nn.Module):
    """Torch module with an optional, composable symbolic-cost description."""

    def cost_repr(self) -> CostTree:
        """Collect this module's static symbolic cost representation."""
        return cost_repr(self)

    def _cost_repr(self, scope: _CostScope) -> Iterable[CostRepr] | None:
        """Return local operations, or ``None`` until local work is classified."""
        del scope
        return None

    def _cost_children(self, scope: _CostScope) -> Iterable[_CostChild]:
        """Direct symbolic collection into immediate children."""
        return tuple(scope.child(name, child) for name, child in self.named_children())


_STRUCTURAL_TORCH_MODULES = (nn.ModuleDict, nn.ModuleList, nn.Sequential)


def _collect_cost_tree(
    module: nn.Module,
    name: str,
    *,
    repetitions: Any = 1,
    parent_bindings: Mapping[str, Any] | None = None,
) -> CostTree:
    scope = _CostScope()
    unresolved: tuple[str, ...] = ()

    if isinstance(module, Module):
        local_costs = module._cost_repr(scope)
        costs = () if local_costs is None else tuple(local_costs)
        child_specs = tuple(module._cost_children(scope))
        if local_costs is None:
            unresolved = (f"{type(module).__qualname__} has not classified its static local matmul work",)
    else:
        costs = ()
        child_specs = tuple(scope.child(child_name, child) for child_name, child in module.named_children())
        if not isinstance(module, _STRUCTURAL_TORCH_MODULES):
            unresolved = (f"{type(module).__qualname__} has no static local-cost provider",)

    children = [
        _collect_cost_tree(
            child_spec.module,
            child_spec.name,
            repetitions=child_spec.repetitions,
            parent_bindings=child_spec.bindings,
        )
        for child_spec in child_specs
    ]

    bindings = dict(scope.bindings)
    if parent_bindings:
        by_name = {display_name: symbol for symbol, display_name in scope.symbols.items()}
        unknown = parent_bindings.keys() - by_name.keys()
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"{type(module).__qualname__} does not declare child-bound symbols: {names}")
        for display_name, expression in parent_bindings.items():
            bindings[by_name[display_name]] = _expression(expression)

    return CostTree(
        name=name,
        module_type=type(module).__qualname__,
        costs=costs,
        children=tuple(children),
        repetitions=repetitions,
        symbols=scope.symbols,
        bindings=bindings,
        unresolved=unresolved,
    )


def cost_repr(module: nn.Module) -> CostTree:
    """Collect a static symbolic cost tree from any Torch module."""
    if not isinstance(module, nn.Module):
        raise TypeError(f"cost_repr expects a torch.nn.Module, got {type(module).__qualname__}")
    return _collect_cost_tree(module, type(module).__qualname__)


def _substitute(expression: Any, bindings: Mapping[Any, Any]) -> Any:
    """Apply transitive bindings until substitution reaches a fixed point."""
    sympy = _sympy()
    result = sympy.sympify(expression)
    for _ in range(len(bindings) + 1):
        substituted = result.subs(bindings, simultaneous=False)
        if substituted == result:
            return sympy.simplify(result)
        result = substituted
    return sympy.simplify(result)


def _require_tensor(cost: CostRepr, name: str) -> TensorRepr:
    operand = cost.arguments[name]
    if not isinstance(operand, TensorRepr):
        raise TypeError(f"{cost.operation} argument {name!r} must be represented by TensorRepr")
    return operand


def _same_dimension(left: Any, right: Any, description: str) -> None:
    sympy = _sympy()
    if sympy.simplify(left - right) != 0:
        raise ValueError(f"{description} must match, got {left} and {right}")


def _bmm_flops(cost: CostRepr) -> Any:
    left = _require_tensor(cost, "self")
    right = _require_tensor(cost, "mat2")
    if len(left.shape) != 3 or len(right.shape) != 3:
        raise ValueError("aten.bmm operands must both be rank three")
    batch, rows, inner = left.shape
    other_batch, other_inner, columns = right.shape
    _same_dimension(batch, other_batch, "aten.bmm batch dimensions")
    _same_dimension(inner, other_inner, "aten.bmm inner dimensions")
    return cost.repetitions * batch * rows * columns * 2 * inner


_MATMUL_POLICIES = {torch.ops.aten.bmm.default: _bmm_flops}


def _tree_bindings(tree: CostTree) -> dict[Any, Any]:
    bindings = dict(tree.bindings)
    for child in tree.children:
        bindings.update(_tree_bindings(child))
    return bindings


def matmul_flops(
    tree: CostTree,
    *,
    substitutions: Mapping[Any, Any] | None = None,
    strict: bool = False,
) -> CostReport:
    """Apply the course's conventional matrix-operation FLOP policy."""
    sympy = _sympy()
    terms: list[_CostTerm] = []
    unsupported: list[str] = []

    def visit(node: CostTree, path: str, repetition: Any) -> None:
        effective_repetition = repetition * node.repetitions
        unsupported.extend(f"{path}: {message}" for message in node.unresolved)
        for cost in node.costs:
            policy = _MATMUL_POLICIES.get(cost.operation)
            if policy is None:
                unsupported.append(f"{path}: no matmul policy for {cost.operation} ({cost.name})")
                continue
            expression = sympy.expand(effective_repetition * policy(cost))
            terms.append(_CostTerm(path, cost, expression))
        for child in node.children:
            visit(child, f"{path}.{child.name}", effective_repetition)

    visit(tree, tree.name, sympy.Integer(1))
    if strict and unsupported:
        raise NotImplementedError("unsupported symbolic costs:\n" + "\n".join(unsupported))

    symbolic_total = sympy.expand(sum((term.expression for term in terms), sympy.Integer(0)))
    bindings = _tree_bindings(tree)
    if substitutions:
        bindings.update({symbol: _expression(value) for symbol, value in substitutions.items()})
    return CostReport(
        terms=tuple(terms),
        symbolic_total=symbolic_total,
        bound_total=_substitute(symbolic_total, bindings),
        bindings=bindings,
        unsupported=tuple(unsupported),
    )
