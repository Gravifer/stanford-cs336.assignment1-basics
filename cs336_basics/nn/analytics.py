"""Symbolic descriptions and observations of neural-network resource use."""

from __future__ import annotations

import keyword
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, overload, runtime_checkable

import torch
from torch import nn


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


__all__ = [
    "CostObserver",
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
    "observe_costs",
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
    try:
        expression = sympy.sympify(value)
    except (TypeError, sympy.SympifyError) as error:
        raise TypeError(f"cost dimensions and repetitions must be scalar symbolic expressions, got {value!r}") from error
    if getattr(expression, "is_Boolean", False):
        raise TypeError(f"cost dimensions and repetitions cannot be booleans, got {value!r}")
    if not isinstance(expression, sympy.Expr):
        raise TypeError(f"cost dimensions and repetitions must be scalar symbolic expressions, got {value!r}")
    if expression.is_number and expression.is_integer is not True:
        raise TypeError(f"concrete cost dimensions and repetitions must be integers, got {value!r}")
    if expression.is_integer is False:
        raise TypeError(f"cost dimensions and repetitions must be integer expressions, got {value!r}")
    if expression.is_nonnegative is False:
        raise ValueError(f"cost dimensions and repetitions must be nonnegative, got {value!r}")
    return expression


def _validate_tensor_argument(schema_type: Any, value: Any, operation: Any, name: str) -> None:
    """Validate symbolic tensor compatibility with one ATen schema argument."""
    if isinstance(schema_type, torch.TensorType):
        if not isinstance(value, TensorRepr):
            raise TypeError(f"{operation} tensor argument {name!r} must be represented by TensorRepr")
        return
    if isinstance(schema_type, torch.OptionalType):
        if value is not None:
            _validate_tensor_argument(schema_type.getElementType(), value, operation, name)
        return
    if isinstance(schema_type, torch.ListType):
        element_type = schema_type.getElementType()
        if _schema_contains_tensor(element_type):
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"{operation} tensor-list argument {name!r} must be a sequence of TensorRepr values")
            for item in value:
                _validate_tensor_argument(element_type, item, operation, name)
            return
    if _contains_tensor_repr(value):
        raise TypeError(f"{operation} non-tensor argument {name!r} cannot contain TensorRepr")


def _schema_contains_tensor(schema_type: Any) -> bool:
    """Return whether optional/list composition ultimately contains Tensor."""
    if isinstance(schema_type, torch.TensorType):
        return True
    if isinstance(schema_type, (torch.OptionalType, torch.ListType)):
        return _schema_contains_tensor(schema_type.getElementType())
    return False


def _contains_tensor_repr(value: Any) -> bool:
    """Find tensor metadata nested in standard symbolic containers."""
    if isinstance(value, TensorRepr):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor_repr(item) for item in value.values())
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_tensor_repr(item) for item in value)
    return False


@dataclass(frozen=True)
class TensorRepr:
    """Symbolic metadata for one tensor operand without allocating tensor data."""

    shape: tuple[Any, ...]
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if self.dtype is not None and not isinstance(self.dtype, torch.dtype):
            raise TypeError(f"tensor metadata dtype must be a torch.dtype or None, got {self.dtype!r}")
        object.__setattr__(self, "shape", tuple(_expression(axis) for axis in self.shape))

    @property
    def numel(self) -> Any:
        """Return the symbolic number of logical elements in this tensor."""
        return _sympy().prod(self.shape)

    @property
    def logical_nbytes(self) -> Any | None:
        """Return logical dtype-sized bytes, or ``None`` when dtype is unknown.

        This intrinsic extent does not say whether storage is materialized,
        aliased, retained, or simultaneously live with another tensor.
        """
        if self.dtype is None:
            return None
        return self.numel * self.dtype.itemsize


@dataclass(frozen=True)
class SymbolRepr:
    """One immutable locally named symbolic dimension and optional definition."""

    local_name: str
    display_name: str
    symbol: Any
    binding: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.local_name, str) or not self.local_name:
            raise ValueError("a symbolic dimension requires a non-empty local name")
        if not isinstance(self.display_name, str) or not self.display_name:
            raise ValueError("a symbolic dimension requires a non-empty display name")
        if not isinstance(self.symbol, _sympy().Symbol):
            raise TypeError("a symbolic dimension identity must be a SymPy Symbol")
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

    def __post_init__(self) -> None:
        values = {
            "parameter_numel": self.parameter_numel,
            "parameter_bytes": self.parameter_bytes,
            "trainable_parameter_numel": self.trainable_parameter_numel,
            "trainable_parameter_bytes": self.trainable_parameter_bytes,
            "buffer_numel": self.buffer_numel,
            "buffer_bytes": self.buffer_bytes,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"module state footprint {name} must be an integer")
            if value < 0:
                raise ValueError(f"module state footprint {name} must be nonnegative")
        if self.parameter_bytes < self.parameter_numel or self.buffer_bytes < self.buffer_numel:
            raise ValueError("module state footprint bytes cannot be smaller than logical element counts")
        if self.trainable_parameter_numel > self.parameter_numel:
            raise ValueError("trainable parameter elements cannot exceed all parameter elements")
        if self.trainable_parameter_bytes > self.parameter_bytes:
            raise ValueError("trainable parameter bytes cannot exceed all parameter bytes")
        if self.trainable_parameter_bytes < self.trainable_parameter_numel:
            raise ValueError("trainable parameter bytes cannot be smaller than trainable element counts")
        for category, numel, byte_count in (
            ("parameter", self.parameter_numel, self.parameter_bytes),
            ("trainable parameter", self.trainable_parameter_numel, self.trainable_parameter_bytes),
            ("buffer", self.buffer_numel, self.buffer_bytes),
        ):
            if numel == 0 and byte_count != 0:
                raise ValueError(f"zero {category} elements must occupy zero logical bytes")


@dataclass(frozen=True)
class CostRepr:
    """One semantically named invocation of an exact Torch operator overload."""

    name: str
    operation: torch._ops.OpOverload
    arguments: Mapping[str, Any]
    repetitions: Any = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
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
    """One directed symbolic relationship to an immediate child module."""

    name: str
    module: nn.Module
    repetitions: Any = 1
    arguments: Mapping[str, Any] = field(default_factory=dict)
    edge_role: Literal["call", "inventory"] = "call"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or "." in self.name:
            raise ValueError("a directed cost child name must be a non-empty string without dots")
        if not isinstance(self.module, nn.Module):
            raise TypeError(f"a directed cost child must be a torch.nn.Module, got {type(self.module).__qualname__}")
        if any(not isinstance(name, str) for name in self.arguments):
            raise TypeError("directed cost child argument names must be strings")
        if self.edge_role not in ("call", "inventory"):
            raise ValueError(f"unknown directed cost child edge role: {self.edge_role!r}")
        repetitions = _expression(self.repetitions)
        if self.edge_role == "inventory" and (self.arguments or repetitions != 1):
            raise ValueError("inventory edges cannot carry call arguments or repetitions")
        object.__setattr__(self, "repetitions", repetitions)
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
        """Describe one immediate child invocation and pass its symbolic arguments."""
        return _CostChild(
            name,
            module,
            repetitions,
            {} if arguments is None else arguments,
            edge_role="call",
        )

    def inventory(self, name: str, module: nn.Module) -> _CostChild:
        """Retain a registered child without claiming that the parent invokes it."""
        return _CostChild(name, module, edge_role="inventory")


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
    edge_role: Literal["call", "inventory"] = "call"

    def __post_init__(self) -> None:
        costs = tuple(self.costs)
        children = tuple(self.children)
        symbols = tuple(self.symbols)
        unresolved = tuple(self.unresolved)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a cost tree requires a non-empty name")
        if not isinstance(self.module_type, str) or not self.module_type:
            raise ValueError("a cost tree requires a non-empty module type")
        if any(not isinstance(cost, CostRepr) for cost in costs):
            raise TypeError("cost tree costs must be CostRepr values")
        if any(not isinstance(child, CostTree) for child in children):
            raise TypeError("cost tree children must be CostTree values")
        if any(not isinstance(symbol, SymbolRepr) for symbol in symbols):
            raise TypeError("cost tree symbols must be SymbolRepr values")
        if any(not isinstance(message, str) for message in unresolved):
            raise TypeError("cost tree unresolved messages must be strings")
        if self.edge_role not in ("call", "inventory"):
            raise ValueError(f"unknown cost tree edge role: {self.edge_role!r}")
        repetitions = _expression(self.repetitions)
        arguments = dict(self.arguments)
        if self.edge_role == "inventory" and (arguments or repetitions != 1):
            raise ValueError("inventory cost trees cannot carry call arguments or repetitions")
        child_names = tuple(child.name for child in children)
        if any("." in name for name in child_names):
            raise ValueError("cost tree child names must not contain dots")
        if len(set(child_names)) != len(child_names):
            raise ValueError("cost tree child names must be unique")
        local_names = tuple(symbol.local_name for symbol in symbols)
        display_names = tuple(symbol.display_name for symbol in symbols)
        identities = tuple(symbol.symbol for symbol in symbols)
        if len(set(local_names)) != len(local_names):
            raise ValueError("cost tree local symbol names must be unique")
        if len(set(display_names)) != len(display_names):
            raise ValueError("cost tree display symbol names must be unique")
        if len(set(identities)) != len(identities):
            raise ValueError("cost tree symbol identities must be unique")
        subtree_identities = list(identities)

        def append_child_identities(child: CostTree) -> None:
            subtree_identities.extend(symbol.symbol for symbol in child.symbols)
            for descendant in child.children:
                append_child_identities(descendant)

        for child in children:
            append_child_identities(child)
        if len(set(subtree_identities)) != len(subtree_identities):
            raise ValueError("cost tree symbol identities must be unique across the subtree")
        local_identity_set = frozenset(identities)
        for symbol in symbols:
            if symbol.binding is not None and _metadata_free_symbols(symbol.binding) - local_identity_set:
                raise ValueError("cost tree symbol bindings must use locally declared identities")
        for cost in costs:
            if _metadata_free_symbols(cost.arguments) - local_identity_set:
                raise ValueError("cost tree local costs must use locally declared identities")
            if _metadata_free_symbols(cost.repetitions) - local_identity_set:
                raise ValueError("cost tree local cost repetitions must use locally declared identities")
        for child in children:
            if _metadata_free_symbols(child.arguments) - local_identity_set:
                raise ValueError("cost tree child arguments must use parent-local identities")
            if _metadata_free_symbols(child.repetitions) - local_identity_set:
                raise ValueError("cost tree child repetitions must use parent-local identities")
        unknown_arguments = arguments.keys() - set(identities)
        if unknown_arguments:
            raise ValueError("cost tree arguments must target locally declared symbol identities")
        object.__setattr__(self, "costs", costs)
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "repetitions", repetitions)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(
            self,
            "arguments",
            _immutable_mapping({symbol: _expression(value) for symbol, value in arguments.items()}),
        )
        object.__setattr__(self, "unresolved", unresolved)

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

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("a cost term requires a non-empty module path")
        if not isinstance(self.source, CostRepr):
            raise TypeError("a cost term source must be CostRepr")
        object.__setattr__(self, "expression", _expression(self.expression))


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
        terms = tuple(self.terms)
        unsupported = tuple(self.unsupported)
        conditions = tuple(self.conditions)
        if any(not isinstance(term, CostTerm) for term in terms):
            raise TypeError("cost report terms must be CostTerm values")
        if any(not isinstance(message, str) for message in unsupported):
            raise TypeError("cost report unsupported messages must be strings")
        if conditions:
            equality_type = _sympy().Equality
            if any(not isinstance(condition, equality_type) for condition in conditions):
                raise TypeError("cost report conditions must be SymPy equalities")
        sympy = _sympy()
        known_symbols = frozenset(self.known_symbols)
        if any(not isinstance(symbol, _sympy().Symbol) for symbol in known_symbols):
            raise TypeError("cost report known symbols must be SymPy Symbol identities")
        bindings = dict(self.bindings)
        if any(not isinstance(symbol, _sympy().Symbol) for symbol in bindings):
            raise TypeError("cost report binding keys must be SymPy Symbol identities")
        if bindings.keys() - known_symbols:
            raise ValueError("cost report bindings must target known symbolic identities")
        normalized_bindings = {symbol: _expression(value) for symbol, value in bindings.items()}
        _check_definition_cycles(normalized_bindings)
        symbolic_total = _expression(self.symbolic_total)
        bound_total = _expression(self.bound_total)
        source_domains = tuple(
            expression
            for term in terms
            for expression in (
                *_metadata_dimensions(term.source.arguments),
                term.source.repetitions,
            )
        )
        domain_expressions = tuple(
            dict.fromkeys(
                _expression(expression)
                for expression in (
                    *self._domain_expressions,
                    *known_symbols,
                    *normalized_bindings.values(),
                    *source_domains,
                )
            )
        )
        report_metadata: tuple[Any, ...] = (
            symbolic_total,
            bound_total,
            tuple(term.expression for term in terms),
            tuple(term.source.arguments for term in terms),
            tuple(term.source.repetitions for term in terms),
            tuple(normalized_bindings.values()),
            conditions,
            domain_expressions,
        )
        foreign_symbols = _metadata_free_symbols(report_metadata) - known_symbols
        if foreign_symbols:
            names = ", ".join(sorted(map(str, foreign_symbols)))
            raise ValueError(f"cost report expressions reference unknown symbolic identities: {names}")
        expected_symbolic_total = sympy.expand(sum((term.expression for term in terms), sympy.Integer(0)))
        if sympy.simplify(symbolic_total - expected_symbolic_total) != 0:
            raise ValueError("cost report symbolic total must equal the sum of its terms")
        expected_bound_total = _substitute(symbolic_total, normalized_bindings)
        if sympy.simplify(bound_total - expected_bound_total) != 0:
            raise ValueError("cost report bound total must equal its symbolic total under its bindings")
        conditions = _validate_relations(
            ((condition.lhs, condition.rhs) for condition in conditions),
            normalized_bindings,
        )
        _validate_domains(domain_expressions, normalized_bindings)
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "symbolic_total", symbolic_total)
        object.__setattr__(self, "bound_total", bound_total)
        object.__setattr__(
            self,
            "bindings",
            _immutable_mapping(normalized_bindings),
        )
        object.__setattr__(self, "unsupported", unsupported)
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "known_symbols", known_symbols)
        object.__setattr__(self, "_domain_expressions", domain_expressions)

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


@runtime_checkable
class _CostCallProvider(Protocol):
    """Protected contract for binding one completed root-module forward."""

    def _cost_call_bindings(
        self,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> Mapping[str, Any]: ...


class CostObserver:
    """Observe call-specific root bindings without retaining call tensors.

    The session installs one ordinary per-module forward hook. It observes every
    invocation that reaches that hook while the context is active, in forward-hook
    completion order. It retains only normalized scalar facts and never the call
    tensors themselves. A session is not thread-safe: forwards must not overlap
    context exit or report generation.
    """

    def __init__(self, module: nn.Module) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError(f"cost observation expects a torch.nn.Module, got {type(module).__qualname__}")
        if not isinstance(module, _CostCallProvider):
            raise TypeError(
                f"{type(module).__qualname__} does not provide root-invocation cost bindings"
            )
        self._module = module
        self._tree: CostTree | None = None
        self._root_symbols: Mapping[str, Any] = MappingProxyType({})
        self._substitutions: list[Mapping[Any, Any]] = []
        self._failures: list[str] = []
        self._handle: Any | None = None
        self._entered = False
        self._closed = False

    def __enter__(self) -> CostObserver:
        if self._entered or self._closed:
            raise RuntimeError("a cost observation session cannot be entered more than once")
        tree = cost_repr(self._module)
        self._tree = tree
        self._root_symbols = MappingProxyType(
            {record.local_name: record.symbol for record in tree.symbols}
        )
        self._handle = self._module.register_forward_hook(
            self._observe,
            with_kwargs=True,
            always_call=False,
        )
        self._entered = True
        return self

    def __exit__(self, *exc_info: object) -> None:
        del exc_info
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._entered = False
        self._closed = True

    def _observe(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> None:
        """Copy call-specific facts without modifying or retaining the output."""
        try:
            provider = cast(_CostCallProvider, module)
            bindings = provider._cost_call_bindings(args, kwargs, output)
            if not isinstance(bindings, Mapping):
                raise TypeError("_cost_call_bindings() must return a mapping")
            if any(not isinstance(name, str) for name in bindings):
                raise TypeError("_cost_call_bindings() keys must be local symbol names")
            unknown = bindings.keys() - self._root_symbols.keys()
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"_cost_call_bindings() returned undeclared root symbols: {names}")
            substitutions = {
                self._root_symbols[name]: _expression(value) for name, value in bindings.items()
            }
        except Exception as error:  # observation must not turn a valid forward into a failure
            self._failures.append(f"{type(error).__qualname__}: {error}")
            return None
        self._substitutions.append(MappingProxyType(substitutions))
        return None

    @property
    def tree(self) -> CostTree:
        """Return the static tree snapshot after the session has been entered."""
        if self._tree is None:
            raise RuntimeError("cost observation has not been entered")
        return self._tree

    @property
    def call_count(self) -> int:
        """Return the number of root forwards whose facts were bound successfully."""
        return len(self._substitutions)

    def matmul_flops(self, *, strict: bool = False) -> tuple[CostReport, ...]:
        """Apply the matrix-product policy separately to every observed call."""
        tree = self.tree
        if self._failures:
            raise RuntimeError("cost observation failed:\n" + "\n".join(self._failures))
        return tuple(
            matmul_flops(tree, substitutions=substitutions, strict=strict)
            for substitutions in self._substitutions
        )


def observe_costs(module: nn.Module) -> CostObserver:
    """Create a single-use observation session for one root module."""
    return CostObserver(module)


_REGISTERED_SLOT_CONTAINERS = (nn.ModuleDict, nn.ModuleList, nn.Sequential)
_EXECUTING_TORCH_CONTAINERS = (nn.Sequential,)


def _structural_children(module: nn.Module) -> tuple[tuple[str, nn.Module], ...]:
    """Preserve every registered slot in Torch's public structural containers."""
    if isinstance(module, _REGISTERED_SLOT_CONTAINERS):
        return tuple((name, child) for name, child in module._modules.items() if child is not None)
    return tuple(module.named_children())


def _collect_cost_tree(
    module: nn.Module,
    name: str,
    *,
    repetitions: Any = 1,
    parent_arguments: Mapping[str, Any] | None = None,
    edge_role: Literal["call", "inventory"] = "call",
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
        if any(not isinstance(cost, CostRepr) for cost in costs):
            raise TypeError(
                f"{type(module).__qualname__}._cost_repr() must return CostRepr values or None"
            )
        if any(not isinstance(child, _CostChild) for child in child_specs):
            raise TypeError(
                f"{type(module).__qualname__}._cost_children() must return directed cost children"
            )
        if local_costs is None:
            unresolved = (f"{type(module).__qualname__} has not classified its static local matmul work",)
    else:
        costs = ()
        executes_registered_slots = type(module) in _EXECUTING_TORCH_CONTAINERS
        describe_child = scope.child if executes_registered_slots else scope.inventory
        child_specs = tuple(
            describe_child(child_name, child) for child_name, child in _structural_children(module)
        )
        if not executes_registered_slots:
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
            edge_role=child_spec.edge_role,
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
        edge_role=edge_role,
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


def _require_expandable_to(source: tuple[Any, ...], target: tuple[Any, ...], description: str) -> None:
    """Require ``source`` to be provably expandable to the fixed ``target`` shape."""
    sympy = _sympy()
    if len(source) > len(target):
        raise ValueError(f"{description}: source rank exceeds target rank ({source} to {target})")
    padded_source = (sympy.Integer(1),) * (len(target) - len(source)) + source
    for source_axis, target_axis in zip(padded_source, target, strict=True):
        if sympy.simplify(source_axis - target_axis) == 0:
            continue
        if sympy.simplify(source_axis - 1) == 0:
            continue
        raise ValueError(f"{description}: {source} cannot expand to {target}")


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
    _require_expandable_to(addend.shape, (left.shape[0], right.shape[1]), "aten.addmm addend")
    return flops


def _baddbmm_flops(cost: CostRepr) -> Any:
    """Count only the batched matrix products in a ``baddbmm`` invocation."""
    addend = _require_tensor(cost, "self")
    left = _require_tensor(cost, "batch1")
    right = _require_tensor(cost, "batch2")
    flops = _bmm_flops(cost, "batch1", "batch2")
    _require_expandable_to(
        addend.shape,
        (left.shape[0], left.shape[1], right.shape[2]),
        "aten.baddbmm addend",
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
    """Apply the course's matrix-operation policy to call edges in a cost tree.

    Inventory edges remain available for structural inspection but contribute
    no terms. With ``strict=False``, unsupported executed work is reported next
    to the sum of supported terms; that sum is not a complete execution total.
    """
    if not isinstance(tree, CostTree):
        raise TypeError(f"matmul_flops expects a CostTree, got {type(tree).__qualname__}")
    sympy = _sympy()
    terms: list[CostTerm] = []
    unsupported: list[str] = []

    def visit(node: CostTree, path: str, repetition: Any) -> None:
        if node.edge_role == "inventory":
            return
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
