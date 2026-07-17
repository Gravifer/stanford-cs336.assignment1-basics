"""Focused runtime probes for the typing conventions used in ``cs336_basics.nn``.

This is deliberately separate from the assignment test suite. It reports observed
behavior rather than asserting a policy that the project has not chosen yet.
"""

from collections.abc import Callable
from typing import Any

import einx
import torch
from beartype import beartype
from jaxtyping import Float, Shaped, jaxtyped

from cs336_basics.nn.modules import Linear


type ModelVec = Float[torch.Tensor, "{self.width}"]  # noqa: F821 - evaluated against the bound method


def report(label: str, probe: Callable[[], Any]) -> None:
    """Run one probe and print either its value or the exception it raises."""
    try:
        value = probe()
    except Exception as exc:  # probes intentionally exercise invalid calls
        print(f"{label}: {type(exc).__name__}: {exc}")
    else:
        if isinstance(value, torch.Tensor):
            value = f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
        print(f"{label}: OK: {value}")


class GlobalAliasChecked:
    """Mimic the module-level ``ModelVec`` alias used by the NN modules."""

    def __init__(self, width: int) -> None:
        self.width = width

    @jaxtyped(typechecker=beartype)
    def identity(self, x: Shaped[ModelVec, "*batch"]) -> Shaped[ModelVec, "*batch"]:
        return x


class GlobalAliasUnchecked:
    def __init__(self, width: int) -> None:
        self.width = width

    @jaxtyped(typechecker=None)
    def identity(self, x: Shaped[ModelVec, "*batch"]) -> Shaped[ModelVec, "*batch"]:
        return x


class ClassAliasChecked:
    """Mimic the class-local aliases used by RotaryPositionalEmbedding."""

    type Vec = Float[torch.Tensor, "{self.width}"]  # noqa: F821 - evaluated against the bound method

    def __init__(self, width: int) -> None:
        self.width = width

    @jaxtyped(typechecker=beartype)
    def identity(self, x: Shaped[Vec, "*batch"]) -> Shaped[Vec, "*batch"]:
        return x


class AnnotatedAttribute:
    """Check what an annotation does (and does not do) for an attribute."""

    type Vec = Float[torch.Tensor, "{self.width}"]  # noqa: F821 - evaluated in manually_matches

    width: int
    value: Vec

    def __init__(self, width: int, value: torch.Tensor) -> None:
        self.width = width
        self.value = value

    @jaxtyped(typechecker=None)
    def manually_matches_alias_directly(self) -> bool:
        return isinstance(self.value, self.Vec)  # ty: ignore[invalid-argument-type]

    @jaxtyped(typechecker=None)
    def manually_matches_nested_alias(self) -> bool:
        return isinstance(self.value, Shaped[self.Vec, ""])  # ty: ignore[invalid-argument-type]

    @jaxtyped(typechecker=None)
    def manually_matches_alias_value(self) -> bool:
        return isinstance(self.value, self.Vec.__value__)


class ConstructorArgumentsChecked:
    """Reference constructor arguments rather than not-yet-assigned attributes."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        rows: int,
        columns: int,
        value: Float[torch.Tensor, "{rows} {columns}"] | None = None,
    ) -> None:
        self.value = torch.empty(rows, columns) if value is None else value


@jaxtyped(typechecker=beartype)
def named_variadic(
    x: Float[torch.Tensor, "*mapped channels"],
    y: Float[torch.Tensor, "*mapped channels"],
) -> Float[torch.Tensor, "*mapped channels"]:
    return y


@jaxtyped(typechecker=beartype)
def anonymous_variadic(
    x: Float[torch.Tensor, "... channels"],
    y: Float[torch.Tensor, "... channels"],
) -> Float[torch.Tensor, "... channels"]:
    return y


@jaxtyped(typechecker=None)
def named_variadic_unchecked(
    x: Float[torch.Tensor, "*mapped channels"],
    y: Float[torch.Tensor, "*mapped channels"],
) -> Float[torch.Tensor, "*mapped channels"]:
    return y


@jaxtyped(typechecker=beartype)
def named_return(x: Float[torch.Tensor, "*mapped channels"]) -> Float[torch.Tensor, "*mapped channels"]:
    return torch.zeros(6, x.shape[-1])


@jaxtyped(typechecker=beartype)
def anonymous_return(x: Float[torch.Tensor, "... channels"]) -> Float[torch.Tensor, "... channels"]:
    return torch.zeros(6, x.shape[-1])


@jaxtyped(typechecker=beartype)
def accept_linear(module: Float[Linear, "out_features in_features"]) -> tuple[int, ...]:
    return module.shape


@jaxtyped(typechecker=beartype)
def locals_are_not_checked(x: Float[torch.Tensor, "*shape"]) -> Float[torch.Tensor, "*shape"]:
    deliberately_wrong: Float[torch.Tensor, "4"] = torch.zeros(5)  # noqa: UP037 - jaxtyping shape
    del deliberately_wrong
    return x


def probe_dependent_aliases() -> None:
    print("\n[dependent aliases]")
    global_checked = GlobalAliasChecked(width=4)
    global_unchecked = GlobalAliasUnchecked(width=4)
    class_checked = ClassAliasChecked(width=4)
    good = torch.zeros(2, 3, 4)
    bad = torch.zeros(2, 3, 5)

    report("global alias / checked / good", lambda: global_checked.identity(good))
    report("global alias / checked / bad", lambda: global_checked.identity(bad))
    report("global alias / unchecked / bad", lambda: global_unchecked.identity(bad))
    report("class alias / checked / good", lambda: class_checked.identity(good))
    report("class alias / checked / bad", lambda: class_checked.identity(bad))

    good_attribute = AnnotatedAttribute(width=4, value=torch.zeros(4))
    bad_attribute = AnnotatedAttribute(width=4, value=torch.zeros(5))
    report("attribute assignment / good", lambda: good_attribute.value)
    report("attribute assignment / bad (no automatic check)", lambda: bad_attribute.value)
    report("attribute / direct TypeAliasType isinstance / good", good_attribute.manually_matches_alias_directly)
    report("attribute / nested alias isinstance / good", good_attribute.manually_matches_nested_alias)
    report("attribute / nested alias isinstance / bad", bad_attribute.manually_matches_nested_alias)
    report("attribute / alias __value__ isinstance / good", good_attribute.manually_matches_alias_value)
    report("attribute / alias __value__ isinstance / bad", bad_attribute.manually_matches_alias_value)
    report(
        "constructor argument dimensions / good",
        lambda: ConstructorArgumentsChecked(4, 3, torch.zeros(4, 3)).value,
    )
    report(
        "constructor argument dimensions / bad",
        lambda: ConstructorArgumentsChecked(4, 3, torch.zeros(5, 3)).value,
    )


def probe_duck_linear() -> None:
    print("\n[Linear as a duck array]")
    linear = Linear(3, 4, dtype=torch.float32)
    float_linear = Float[Linear, "4 3"]
    shaped_linear = Shaped[Linear, "4 3"]

    report("Linear.shape", lambda: linear.shape)
    report("Linear.dtype", lambda: linear.dtype)
    report(
        "isinstance / Shaped / correct shape",
        lambda: isinstance(linear, shaped_linear),  # ty: ignore[invalid-argument-type]
    )
    report(
        "isinstance / Float / correct shape",
        lambda: isinstance(linear, float_linear),  # ty: ignore[invalid-argument-type]
    )
    report(
        "isinstance / Float / wrong shape",
        lambda: isinstance(linear, Float[Linear, "3 4"]),  # ty: ignore[invalid-argument-type]
    )
    report("beartype argument / Float[Linear]", lambda: accept_linear(linear))


def probe_jaxtyping_variadics() -> None:
    print("\n[jaxtyping named versus anonymous variadics]")
    x = torch.zeros(2, 3, 4)
    same = torch.zeros(2, 3, 4)
    different_prefix = torch.zeros(6, 4)

    report("named / same prefix", lambda: named_variadic(x, same))
    report("named / different prefix", lambda: named_variadic(x, different_prefix))
    report("anonymous / different prefix", lambda: anonymous_variadic(x, different_prefix))
    report("named / unchecked / different prefix", lambda: named_variadic_unchecked(x, different_prefix))
    report("named / mismatched return prefix", lambda: named_return(x))
    report("anonymous / mismatched return prefix", lambda: anonymous_return(x))
    report("local annotation mismatch is not checked", lambda: locals_are_not_checked(torch.zeros(2, 4)))


def probe_einx_ellipses() -> None:
    print("\n[einx named versus anonymous ellipses]")
    x = torch.zeros(2, 3, 4)
    same = torch.ones(2, 3, 4)
    different_prefix = torch.ones(6, 4)

    report(
        "named / same prefix",
        lambda: einx.add("mapped... d, mapped... d -> mapped... d", x, same),
    )
    report(
        "anonymous / same prefix",
        lambda: einx.add("... d, ... d -> ... d", x, same),
    )
    report(
        "named / different prefix",
        lambda: einx.add("mapped... d, mapped... d -> mapped... d", x, different_prefix),
    )
    report(
        "anonymous / different prefix",
        lambda: einx.add("... d, ... d -> ... d", x, different_prefix),
    )

    mapped_batch = torch.zeros(2, 3, 5, 4)
    batch = torch.zeros(5, 4)
    report(
        "two distinct named groups",
        lambda: einx.add(
            "mapped... batch... d, batch... d -> mapped... batch... d",
            mapped_batch,
            batch,
            mapped=(2, 3),
            batch=(5,),
        ),
    )
    report(
        "two anonymous groups in one expression",
        lambda: einx.add("... ... d, ... d -> ... ... d", mapped_batch, batch),
    )


def main() -> None:
    print(f"torch={torch.__version__}")
    probe_dependent_aliases()
    probe_duck_linear()
    probe_jaxtyping_variadics()
    probe_einx_ellipses()


if __name__ == "__main__":
    main()
