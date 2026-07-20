"""Symbolic descriptions and observations of neural-network resource use."""

from __future__ import annotations

import keyword
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, overload, runtime_checkable

import torch
from torch import nn


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


__all__ = [
    "CostReport",
    "CostRepr",
    "CostTerm",
    "CostTree",
    "ModuleStateFootprint",
    "SymbolRepr",
    "TensorRepr",
    "cost_repr",
    "matmul_flops",
    "module_state_footprint",
]


def _sympy() -> Any:
    """Import the development-only symbolic dependency when analytics is used."""
    try:
        import sympy
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "symbolic model analytics requires the development dependency; install the project's dev dependency group"
        ) from error
    return sympy


def _immutable_value(value: Any) -> Any:
    """Copy and freeze standard containers used in symbolic metadata."""
    if isinstance(value, torch.Tensor):
        raise TypeError("symbolic cost metadata must use TensorRepr rather than live torch.Tensor values")
    if isinstance(value, Mapping):
        return MappingProxyType({key: _immutable_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_immutable_value(item) for item in value)
    return value


def _immutable_mapping(mapping: Mapping[Any, Any]) -> Mapping[Any, Any]:
    """Recursively copy a mapping before exposing an immutable view of it."""
    return MappingProxyType({key: _immutable_value(value) for key, value in mapping.items()})


def _expression(value: object) -> Any:
    """Normalize a nonnegative integer or symbolic value into a SymPy expression."""
    sympy = _sympy()
    expression = sympy.sympify(value)
    if expression.is_Boolean:
        raise TypeError(f"cost dimensions and repetitions cannot be booleans, got {value!r}")
    if expression.is_number and expression.is_integer is not True:
        raise TypeError(f"concrete cost dimensions and repetitions must be integers, got {value!r}")
    if expression.is_integer is False:
        raise TypeError(f"cost dimensions and repetitions must be integer expressions, got {value!r}")
    if expression.is_nonnegative is False:
        raise ValueError(f"cost dimensions and repetitions must be nonnegative, got {value!r}")
    return expression


def _validate_tensor_argument(schema_type: Any, value: Any, operation: Any, name: str) -> None:
    """Require symbolic metadata wherever an ATen schema expects tensor data."""
    if isinstance(schema_type, torch.TensorType):
        if not isinstance(value, TensorRepr):
            raise TypeError(f"{operation} tensor argument {name!r} must be represented by TensorRepr")
        return
    if isinstance(schema_type, torch.OptionalType):
        if value is not None:
            _validate_tensor_argument(schema_type.getElementType(), value, operation, name)
        return
    if isinstance(schema_type, torch.ListType) and isinstance(schema_type.getElementType(), torch.TensorType):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{operation} tensor-list argument {name!r} must be a sequence of TensorRepr values")
        for item in value:
            _validate_tensor_argument(schema_type.getElementType(), item, operation, name)


@dataclass(frozen=True)
class TensorRepr:
    """Symbolic metadata for one tensor operand without allocating tensor data."""

    shape: tuple[Any, ...]
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if self.dtype is not None and not isinstance(self.dtype, torch.dtype):
            raise TypeError(f"tensor metadata dtype must be a torch.dtype or None, got {self.dtype!r}")
        object.__setattr__(self, "shape", tuple(_expression(axis) for axis in self.shape))


@dataclass(frozen=True)
class SymbolRepr:
    """One immutable locally named symbolic dimension and optional definition."""

    local_name: str
    display_name: str
    symbol: Any
    binding: Any | None = None

    def __post_init__(self) -> None:
        if not self.local_name:
            raise ValueError("a symbolic dimension requires a non-empty local name")
        if not self.display_name:
            raise ValueError("a symbolic dimension requires a non-empty display name")
        if self.binding is not None:
            object.__setattr__(self, "binding", _expression(self.binding))


@dataclass(frozen=True)
class ModuleStateFootprint:
    """Logical element and byte counts for a module's registered tensor state."""

    parameter_numel: int
    parameter_bytes: int
    trainable_parameter_numel: int
    trainable_parameter_bytes: int
    buffer_numel: int
    buffer_bytes: int


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
        for name, value in arguments.items():
            _validate_tensor_argument(schema_arguments[name].type, value, self.operation, name)

        object.__setattr__(self, "arguments", _immutable_mapping(arguments))
        object.__setattr__(self, "repetitions", _expression(self.repetitions))


@dataclass(frozen=True)
class _CostChild:
    """One directed symbolic invocation of an immediate child module."""

    name: str
    module: nn.Module
    repetitions: Any = 1
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or "." in self.name:
            raise ValueError("a directed cost child name must be a non-empty string without dots")
        if not isinstance(self.module, nn.Module):
            raise TypeError(f"a directed cost child must be a torch.nn.Module, got {type(self.module).__qualname__}")
        if any(not isinstance(name, str) for name in self.arguments):
            raise TypeError("directed cost child argument names must be strings")
        object.__setattr__(self, "repetitions", _expression(self.repetitions))
        object.__setattr__(
            self,
            "arguments",
            _immutable_mapping({name: _expression(value) for name, value in self.arguments.items()}),
        )


def _metadata_free_symbols(value: Any) -> frozenset[Any]:
    """Find SymPy identities used by supported symbolic metadata containers."""
    if isinstance(value, TensorRepr):
        return frozenset().union(*(axis.free_symbols for axis in value.shape))
    if isinstance(value, Mapping):
        return frozenset().union(*(_metadata_free_symbols(item) for item in value.values()))
    if isinstance(value, (tuple, list, set, frozenset)):
        return frozenset().union(*(_metadata_free_symbols(item) for item in value))
    return frozenset(getattr(value, "free_symbols", ()))


def _require_local_symbols(
    module: nn.Module,
    local_symbols: frozenset[Any],
    value: Any,
    description: str,
) -> None:
    """Reject symbolic identities that were not declared by this module scope."""
    foreign = _metadata_free_symbols(value) - local_symbols
    if foreign:
        names = ", ".join(sorted(map(str, foreign)))
        raise ValueError(
            f"{type(module).__qualname__} {description} references undeclared scoped symbols: {names}"
        )


_RESERVED_SYMBOL_NAMES = frozenset(
    {
        "bind",
        "declare",
        "define",
        "denote",
        "display",
        "freeze",
        "get",
        "introduce",
        "items",
        "keys",
        "records",
        "render",
        "resolve",
        "substitute",
        "unbound",
        "update",
        "values",
    }
)


def _check_definition_cycles(definitions: Mapping[Any, Any]) -> None:
    """Reject cycles among symbolic definitions before fixed-point substitution."""
    identities = set(definitions)
    dependencies = {
        symbol: set(expression.free_symbols) & identities for symbol, expression in definitions.items()
    }
    visiting: set[Any] = set()
    visited: set[Any] = set()

    def visit(symbol: Any) -> None:
        if symbol in visiting:
            raise ValueError("symbolic definitions contain a cycle")
        if symbol in visited:
            return
        visiting.add(symbol)
        for dependency in dependencies.get(symbol, ()):
            visit(dependency)
        visiting.remove(symbol)
        visited.add(symbol)

    for symbol in dependencies:
        visit(symbol)


class _SymbolEnvironment:
    """Mutable symbol-table view used while one module authors its costs."""

    def __init__(self) -> None:
        self._symbols: dict[str, Any] = {}
        self._bindings: dict[str, Any] = {}
        self._display_names: dict[str, str] = {}
        self._frozen = False

    def _validate_name(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError(f"symbol names must be strings, got {type(name).__qualname__}")
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"symbol name must be a valid non-keyword identifier, got {name!r}")
        if name.startswith("_") or name in _RESERVED_SYMBOL_NAMES or hasattr(type(self), name):
            raise ValueError(f"symbol name {name!r} is reserved by the symbolic environment")

    def _ensure_mutable(self) -> None:
        if self._frozen:
            raise RuntimeError("the symbolic environment is frozen")

    def _symbol(self, name: str) -> Any:
        self._validate_name(name)
        symbol = self._symbols.get(name)
        if symbol is None:
            symbol = _sympy().Dummy(name, integer=True, nonnegative=True)
            self._symbols[name] = symbol
        return symbol

    def unbound(self, *names: str) -> _SymbolEnvironment:
        """Declare locally unbound symbols and return this focused environment."""
        self._ensure_mutable()
        if not names:
            raise ValueError("unbound() requires at least one symbol name")
        for name in names:
            self._symbol(name)
        return self

    def bind(self, **bindings: object) -> _SymbolEnvironment:
        """Declare locally defined symbols and return this focused environment."""
        self._ensure_mutable()
        if not bindings:
            raise ValueError("bind() requires at least one symbolic binding")
        for name, value in bindings.items():
            self._symbol(name)
            binding = _expression(value)
            previous = self._bindings.get(name, binding)
            if previous != binding:
                raise ValueError(f"symbol {name!r} received incompatible bindings {previous} and {binding}")
            self._bindings[name] = binding
        return self

    def display(self, **names: str) -> _SymbolEnvironment:
        """Assign human-facing names to symbols already declared in this scope."""
        self._ensure_mutable()
        if not names:
            raise ValueError("display() requires at least one symbol name")
        proposed = dict(self._display_names)
        for local_name, display_name in names.items():
            if local_name not in self._symbols:
                raise ValueError(f"cannot name undeclared symbol {local_name!r}")
            if not isinstance(display_name, str) or not display_name.strip():
                raise ValueError(f"display name for {local_name!r} must be a non-empty string")
            existing = self._display_names.get(local_name)
            if existing is not None and existing != display_name:
                raise ValueError(
                    f"symbol {local_name!r} already has display name {existing!r}, got {display_name!r}"
                )
            proposed[local_name] = display_name
        effective_names = [proposed.get(local_name, local_name) for local_name in self._symbols]
        if len(set(effective_names)) != len(effective_names):
            raise ValueError("display names must be unique within one symbolic scope")
        self._display_names = proposed
        return self

    def __getattr__(self, name: str) -> Any:
        try:
            return self._symbols[name]
        except KeyError as error:
            raise AttributeError(f"no local symbol named {name!r}") from error

    @overload
    def __getitem__(self, name: str) -> Any: ...

    @overload
    def __getitem__(self, name: tuple[str, ...]) -> tuple[Any, ...]: ...

    def __getitem__(self, name: str | tuple[str, ...]) -> Any | tuple[Any, ...]:
        """Look up one or several identities by local name."""
        if isinstance(name, tuple):
            return tuple(self[item] for item in name)
        try:
            return self._symbols[name]
        except KeyError as error:
            raise KeyError(f"no local symbol named {name!r}") from error

    def __iter__(self) -> Iterator[Any]:
        """Iterate over identities in first-declaration order."""
        return iter(self._symbols.values())

    def __len__(self) -> int:
        """Return the number of declared local symbols."""
        return len(self._symbols)

    def _check_cycles(self) -> None:
        _check_definition_cycles(
            {self._symbols[name]: binding for name, binding in self._bindings.items()}
        )

    def _freeze(self) -> tuple[SymbolRepr, ...]:
        """Freeze this builder and return immutable records in declaration order."""
        self._check_cycles()
        self._frozen = True
        return tuple(
            SymbolRepr(name, self._display_names.get(name, name), symbol, self._bindings.get(name))
            for name, symbol in self._symbols.items()
        )


class _CostScope:
    """Author one module's local symbols and directed child invocations."""

    def __init__(self) -> None:
        self.symbols = _SymbolEnvironment()

    def child(
        self,
        name: str,
        module: nn.Module,
        *,
        repetitions: Any = 1,
        arguments: Mapping[str, Any] | None = None,
    ) -> _CostChild:
        """Pass symbolic arguments into one immediate child module."""
        return _CostChild(name, module, repetitions, {} if arguments is None else arguments)


@dataclass(frozen=True)
class CostTree:
    """Immutable symbolic costs arranged according to the authored module tree."""

    name: str
    module_type: str
    costs: tuple[CostRepr, ...] = ()
    children: tuple[CostTree, ...] = ()
    repetitions: Any = 1
    symbols: tuple[SymbolRepr, ...] = ()
    arguments: Mapping[Any, Any] = field(default_factory=dict)
    unresolved: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "costs", tuple(self.costs))
        object.__setattr__(self, "children", tuple(self.children))
        object.__setattr__(self, "repetitions", _expression(self.repetitions))
        object.__setattr__(self, "symbols", tuple(self.symbols))
        object.__setattr__(self, "arguments", _immutable_mapping(self.arguments))
        object.__setattr__(self, "unresolved", tuple(self.unresolved))

    @property
    def bindings(self) -> Mapping[Any, Any]:
        """Return immutable local symbol definitions for inspection."""
        return _immutable_mapping(
            {record.symbol: record.binding for record in self.symbols if record.binding is not None}
        )

    def find_symbols(self, name: str) -> tuple[Any, ...]:
        """Return every scoped symbol displayed with ``name`` in this subtree."""
        found = tuple(record.symbol for record in self.symbols if record.display_name == name)
        return found + tuple(symbol for child in self.children for symbol in child.find_symbols(name))


@dataclass(frozen=True)
class CostTerm:
    """One policy result attributed to a module path and source representation."""

    path: str
    source: CostRepr
    expression: Any


@dataclass(frozen=True)
class CostReport:
    """Structured symbolic result of applying one policy to a cost tree."""

    terms: tuple[CostTerm, ...]
    symbolic_total: Any
    bound_total: Any
    bindings: Mapping[Any, Any]
    unsupported: tuple[str, ...] = ()
    conditions: tuple[Any, ...] = ()
    known_symbols: frozenset[Any] = field(default_factory=frozenset, repr=False)
    _domain_expressions: tuple[Any, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        object.__setattr__(self, "bindings", _immutable_mapping(self.bindings))
        object.__setattr__(self, "unsupported", tuple(self.unsupported))
        object.__setattr__(self, "conditions", tuple(self.conditions))
        object.__setattr__(self, "known_symbols", frozenset(self.known_symbols))
        object.__setattr__(self, "_domain_expressions", tuple(self._domain_expressions))

    def substitute(self, bindings: Mapping[Any, Any]) -> CostReport:
        """Return a report with additional symbolic bindings applied."""
        combined = dict(self.bindings)
        relations = [(condition.lhs, condition.rhs) for condition in self.conditions]
        _add_substitutions(combined, relations, bindings, self.known_symbols)
        _validate_domains(self._domain_expressions, combined)
        conditions = _validate_relations(relations, combined)
        return replace(
            self,
            bound_total=_substitute(self.symbolic_total, combined),
            bindings=combined,
            conditions=conditions,
        )

    @property
    def bound_terms(self) -> tuple[CostTerm, ...]:
        """Return component terms after applying this report's known bindings."""
        return tuple(
            replace(term, expression=_substitute(term.expression, self.bindings)) for term in self.terms
        )


@runtime_checkable
class _CostProvider(Protocol):
    """Protected structural contract for modules that author symbolic costs."""

    def _cost_repr(self, scope: _CostScope) -> Iterable[CostRepr] | None: ...

    def _cost_children(self, scope: _CostScope) -> Iterable[_CostChild]: ...


_STRUCTURAL_TORCH_MODULES = (nn.ModuleDict, nn.ModuleList, nn.Sequential)


def _structural_children(module: nn.Module) -> tuple[tuple[str, nn.Module], ...]:
    """Preserve every registered slot in Torch's public structural containers."""
    if isinstance(module, _STRUCTURAL_TORCH_MODULES):
        return tuple((name, child) for name, child in module._modules.items() if child is not None)
    return tuple(module.named_children())


def _collect_cost_tree(
    module: nn.Module,
    name: str,
    *,
    repetitions: Any = 1,
    parent_arguments: Mapping[str, Any] | None = None,
    ancestors: frozenset[int] = frozenset(),
) -> CostTree:
    module_identity = id(module)
    if module_identity in ancestors:
        raise ValueError("the directed symbolic cost child graph contains a module cycle")
    child_ancestors = ancestors | {module_identity}
    scope = _CostScope()
    unresolved: tuple[str, ...] = ()

    if isinstance(module, _CostProvider):
        local_costs = module._cost_repr(scope)
        costs = () if local_costs is None else tuple(local_costs)
        child_specs = tuple(module._cost_children(scope))
        if local_costs is None:
            unresolved = (f"{type(module).__qualname__} has not classified its static local matmul work",)
    else:
        costs = ()
        child_specs = tuple(scope.child(child_name, child) for child_name, child in _structural_children(module))
        if type(module) not in _STRUCTURAL_TORCH_MODULES:
            unresolved = (f"{type(module).__qualname__} has no static local-cost provider",)

    child_names = tuple(child_spec.name for child_spec in child_specs)
    if len(set(child_names)) != len(child_names):
        raise ValueError(f"{type(module).__qualname__} describes duplicate directed cost child names")

    symbol_records = scope.symbols._freeze()
    local_symbols = frozenset(record.symbol for record in symbol_records)
    for record in symbol_records:
        if record.binding is not None:
            _require_local_symbols(module, local_symbols, record.binding, f"binding for {record.local_name!r}")
    for cost in costs:
        _require_local_symbols(module, local_symbols, cost.arguments, f"cost {cost.name!r}")
        _require_local_symbols(module, local_symbols, cost.repetitions, f"cost {cost.name!r} repetitions")
    for child_spec in child_specs:
        _require_local_symbols(module, local_symbols, child_spec.arguments, f"child {child_spec.name!r} arguments")
        _require_local_symbols(module, local_symbols, child_spec.repetitions, f"child {child_spec.name!r} repetitions")

    by_name = {record.local_name: record.symbol for record in symbol_records}
    arguments: dict[Any, Any] = {}
    if parent_arguments:
        unknown = parent_arguments.keys() - by_name.keys()
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"{type(module).__qualname__} does not declare child argument symbols: {names}")
        arguments = {by_name[name]: expression for name, expression in parent_arguments.items()}

    children = [
        _collect_cost_tree(
            child_spec.module,
            child_spec.name,
            repetitions=child_spec.repetitions,
            parent_arguments=child_spec.arguments,
            ancestors=child_ancestors,
        )
        for child_spec in child_specs
    ]

    return CostTree(
        name=name,
        module_type=type(module).__qualname__,
        costs=costs,
        children=tuple(children),
        repetitions=repetitions,
        symbols=symbol_records,
        arguments=arguments,
        unresolved=unresolved,
    )


def cost_repr(module: nn.Module) -> CostTree:
    """Collect a static symbolic cost tree from any Torch module."""
    if not isinstance(module, nn.Module):
        raise TypeError(f"cost_repr expects a torch.nn.Module, got {type(module).__qualname__}")
    tree = _collect_cost_tree(module, type(module).__qualname__)
    definitions, relations, _ = _tree_facts(tree)
    _validate_domains(_tree_domains(tree), definitions)
    _validate_relations(relations, definitions)
    return tree


def _tensor_footprint_totals(tensors: Iterable[torch.Tensor]) -> tuple[int, int]:
    """Count logical tensor elements and their dtype-sized bytes."""
    count = 0
    byte_count = 0
    for tensor in tensors:
        if torch.nn.parameter.is_lazy(tensor):
            raise ValueError("module state footprint requires initialized parameters and buffers")
        count += tensor.numel()
        byte_count += tensor.numel() * tensor.element_size()
    return count, byte_count


def module_state_footprint(module: nn.Module) -> ModuleStateFootprint:
    """Summarize registered parameters and buffers without allocating tensor data.

    Byte counts are logical ``numel * element_size`` values aggregated across
    devices and dtypes, not allocator peaks or sparse/compressed physical-storage
    measurements. Torch suppresses duplicate identities within each parameter or
    buffer traversal; distinct views sharing storage are counted separately, and
    an object registered in both categories contributes to both.
    """
    if not isinstance(module, nn.Module):
        raise TypeError(f"module_state_footprint expects a torch.nn.Module, got {type(module).__qualname__}")

    parameters = tuple(module.parameters())
    parameter_numel, parameter_bytes = _tensor_footprint_totals(parameters)
    trainable_parameter_numel, trainable_parameter_bytes = _tensor_footprint_totals(
        parameter for parameter in parameters if parameter.requires_grad
    )
    buffer_numel, buffer_bytes = _tensor_footprint_totals(module.buffers())
    return ModuleStateFootprint(
        parameter_numel=parameter_numel,
        parameter_bytes=parameter_bytes,
        trainable_parameter_numel=trainable_parameter_numel,
        trainable_parameter_bytes=trainable_parameter_bytes,
        buffer_numel=buffer_numel,
        buffer_bytes=buffer_bytes,
    )


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


def _require_broadcastable(left: tuple[Any, ...], right: tuple[Any, ...], description: str) -> None:
    """Require every aligned dimension to be provably equal or singleton."""
    sympy = _sympy()
    rank = max(len(left), len(right))
    padded_left = (sympy.Integer(1),) * (rank - len(left)) + left
    padded_right = (sympy.Integer(1),) * (rank - len(right)) + right
    for left_axis, right_axis in zip(padded_left, padded_right, strict=True):
        if sympy.simplify(left_axis - right_axis) == 0:
            continue
        if sympy.simplify(left_axis - 1) == 0 or sympy.simplify(right_axis - 1) == 0:
            continue
        raise ValueError(f"{description} are not broadcastable: {left} and {right}")


def _mm_flops(cost: CostRepr, left_name: str = "self", right_name: str = "mat2") -> Any:
    """Apply Torch's conventional dense rank-two matrix-product formula."""
    left = _require_tensor(cost, left_name)
    right = _require_tensor(cost, right_name)
    if len(left.shape) != 2 or len(right.shape) != 2:
        raise ValueError(f"{cost.operation} matrix operands must both be rank two")
    rows, inner = left.shape
    other_inner, columns = right.shape
    _same_dimension(inner, other_inner, f"{cost.operation} inner dimensions")
    return cost.repetitions * rows * columns * 2 * inner


def _bmm_flops(cost: CostRepr, left_name: str = "self", right_name: str = "mat2") -> Any:
    """Apply Torch's conventional dense rank-three batched-product formula."""
    left = _require_tensor(cost, left_name)
    right = _require_tensor(cost, right_name)
    if len(left.shape) != 3 or len(right.shape) != 3:
        raise ValueError(f"{cost.operation} batch-matrix operands must both be rank three")
    batch, rows, inner = left.shape
    other_batch, other_inner, columns = right.shape
    _same_dimension(batch, other_batch, f"{cost.operation} batch dimensions")
    _same_dimension(inner, other_inner, f"{cost.operation} inner dimensions")
    return cost.repetitions * batch * rows * columns * 2 * inner


def _addmm_flops(cost: CostRepr) -> Any:
    """Count only the matrix product in an ``addmm`` invocation."""
    addend = _require_tensor(cost, "self")
    left = _require_tensor(cost, "mat1")
    right = _require_tensor(cost, "mat2")
    flops = _mm_flops(cost, "mat1", "mat2")
    _require_broadcastable(addend.shape, (left.shape[0], right.shape[1]), "aten.addmm addend and product")
    return flops


def _baddbmm_flops(cost: CostRepr) -> Any:
    """Count only the batched matrix products in a ``baddbmm`` invocation."""
    addend = _require_tensor(cost, "self")
    left = _require_tensor(cost, "batch1")
    right = _require_tensor(cost, "batch2")
    flops = _bmm_flops(cost, "batch1", "batch2")
    if len(addend.shape) > 3:
        raise ValueError("aten.baddbmm addend must have rank at most three")
    _require_broadcastable(
        addend.shape,
        (left.shape[0], left.shape[1], right.shape[2]),
        "aten.baddbmm addend and product",
    )
    return flops


_MATMUL_POLICIES = {
    torch.ops.aten.mm.default: _mm_flops,
    torch.ops.aten.addmm.default: _addmm_flops,
    torch.ops.aten.bmm.default: _bmm_flops,
    torch.ops.aten.baddbmm.default: _baddbmm_flops,
}


def _tree_facts(tree: CostTree) -> tuple[dict[Any, Any], list[tuple[Any, Any]], frozenset[Any]]:
    """Collect canonical definitions, consistency relations, and known identities."""
    definitions: dict[Any, Any] = {}
    relations: list[tuple[Any, Any]] = []
    known_symbols: set[Any] = set()

    def visit(node: CostTree) -> None:
        local_bindings = dict(node.bindings)
        for record in node.symbols:
            symbol = record.symbol
            known_symbols.add(symbol)
            if symbol in node.arguments:
                definitions[symbol] = node.arguments[symbol]
                if symbol in local_bindings:
                    relations.append((node.arguments[symbol], local_bindings[symbol]))
            elif symbol in local_bindings:
                definitions[symbol] = local_bindings[symbol]
        for child in node.children:
            visit(child)

    visit(tree)
    _check_definition_cycles(definitions)
    return definitions, relations, frozenset(known_symbols)


def _metadata_dimensions(value: Any) -> tuple[Any, ...]:
    """Collect tensor-axis expressions without treating arbitrary integer arguments as dimensions."""
    if isinstance(value, TensorRepr):
        return value.shape
    if isinstance(value, Mapping):
        return tuple(axis for item in value.values() for axis in _metadata_dimensions(item))
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(axis for item in value for axis in _metadata_dimensions(item))
    return ()


def _tree_domains(tree: CostTree) -> tuple[Any, ...]:
    """Collect every expression whose value is a tensor dimension or repetition."""
    local = [record.symbol for record in tree.symbols]
    local.extend(record.binding for record in tree.symbols if record.binding is not None)
    local.extend(tree.arguments.values())
    local.append(tree.repetitions)
    for cost in tree.costs:
        local.extend(_metadata_dimensions(cost.arguments))
        local.append(cost.repetitions)
    return tuple(local) + tuple(expression for child in tree.children for expression in _tree_domains(child))


def _validate_domains(expressions: Iterable[Any], bindings: Mapping[Any, Any]) -> None:
    """Reject dimensions or repetitions that become definitely non-integral or negative."""
    for expression in expressions:
        resolved = _substitute(expression, bindings)
        if resolved.is_Boolean or resolved.is_integer is False:
            raise ValueError(f"resolved cost dimensions and repetitions must be integers, got {resolved}")
        if resolved.is_number and resolved.is_integer is not True:
            raise ValueError(f"resolved cost dimensions and repetitions must be integers, got {resolved}")
        if resolved.is_nonnegative is False:
            raise ValueError(f"resolved cost dimensions and repetitions must be nonnegative, got {resolved}")


def _add_substitutions(
    definitions: dict[Any, Any],
    relations: list[tuple[Any, Any]],
    substitutions: Mapping[Any, Any],
    known_symbols: frozenset[Any],
) -> None:
    """Add caller facts without replacing architectural definitions."""
    unknown = substitutions.keys() - known_symbols
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise ValueError(f"substitutions contain unknown symbolic identities: {names}")
    for symbol, value in substitutions.items():
        expression = _expression(value)
        foreign = expression.free_symbols - known_symbols
        if foreign:
            names = ", ".join(sorted(map(str, foreign)))
            raise ValueError(f"substitution values contain unknown symbolic identities: {names}")
        if symbol in definitions:
            relations.append((expression, definitions[symbol]))
        else:
            definitions[symbol] = expression
    _check_definition_cycles(definitions)


def _validate_relations(relations: Iterable[tuple[Any, Any]], definitions: Mapping[Any, Any]) -> tuple[Any, ...]:
    """Reject definite contradictions and retain genuinely symbolic equalities."""
    sympy = _sympy()
    unresolved: list[Any] = []
    for left, right in relations:
        resolved_left = _substitute(left, definitions)
        resolved_right = _substitute(right, definitions)
        difference = sympy.simplify(resolved_left - resolved_right)
        if difference == 0 or difference.is_zero is True:
            continue
        if difference.is_zero is False:
            raise ValueError(f"symbolic facts are inconsistent: {resolved_left} != {resolved_right}")
        unresolved.append(sympy.Eq(resolved_left, resolved_right, evaluate=False))
    return tuple(unresolved)


def matmul_flops(
    tree: CostTree,
    *,
    substitutions: Mapping[Any, Any] | None = None,
    strict: bool = False,
) -> CostReport:
    """Apply the course's conventional matrix-operation FLOP policy."""
    sympy = _sympy()
    terms: list[CostTerm] = []
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
            terms.append(CostTerm(path, cost, expression))
        for child in node.children:
            visit(child, f"{path}.{child.name}", effective_repetition)

    visit(tree, tree.name, sympy.Integer(1))
    if strict and unsupported:
        raise NotImplementedError("unsupported symbolic costs:\n" + "\n".join(unsupported))

    symbolic_total = sympy.expand(sum((term.expression for term in terms), sympy.Integer(0)))
    bindings, relations, known_symbols = _tree_facts(tree)
    if substitutions:
        _add_substitutions(bindings, relations, substitutions, known_symbols)
    domains = _tree_domains(tree)
    _validate_domains(domains, bindings)
    conditions = _validate_relations(relations, bindings)
    return CostReport(
        terms=tuple(terms),
        symbolic_total=symbolic_total,
        bound_total=_substitute(symbolic_total, bindings),
        bindings=bindings,
        unsupported=tuple(unsupported),
        conditions=conditions,
        known_symbols=known_symbols,
        _domain_expressions=domains,
    )
