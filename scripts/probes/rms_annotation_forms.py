"""Probe ways to express a dynamic normalized suffix with jaxtyping."""

from collections.abc import Callable
from typing import Any

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped


def report(label: str, probe: Callable[[], Any]) -> None:
    try:
        value = probe()
    except Exception as exc:  # probes intentionally exercise invalid calls
        message = str(exc) if label.endswith("/ valid") else str(exc).splitlines()[0]
        print(f"{label}: {type(exc).__name__}: {message}")
    else:
        print(f"{label}: OK: shape={tuple(value.shape)}")


@jaxtyped(typechecker=beartype)
def current_form(
    input: Float[torch.Tensor, "*mapped {*dims}"],
    dims: tuple[int, ...],
    weight: Float[torch.Tensor, "{*dims}"],  # noqa: F821 - deliberately probing this shape string
) -> Float[torch.Tensor, "*mapped {*dims}"]:
    del dims, weight
    return input


@jaxtyped(typechecker=beartype)
def tuple_substitution(
    input: Float[torch.Tensor, "*mapped {dims}"],
    dims: tuple[int, ...],
    weight: Float[torch.Tensor, "{dims}"],  # noqa: F821 - deliberately probing this shape string
) -> Float[torch.Tensor, "*mapped {dims}"]:
    del dims, weight
    return input


@jaxtyped(typechecker=beartype)
def joined_substitution(
    input: Float[torch.Tensor, "*mapped {' '.join(map(str, dims))}"],
    dims: tuple[int, ...],
    weight: Float[torch.Tensor, "{' '.join(map(str, dims))}"],  # noqa: F821 - deliberate probe
) -> Float[torch.Tensor, "*mapped {' '.join(map(str, dims))}"]:
    del dims, weight
    return input


@jaxtyped(typechecker=beartype)
def names_only(
    input: Float[torch.Tensor, "*input_shape"],
    dims: tuple[int, ...],
    weight: Float[torch.Tensor, "*weight_shape"],
) -> Float[torch.Tensor, "*input_shape"]:
    del dims, weight
    return input


def exercise(name: str, function: Callable[..., torch.Tensor]) -> None:
    good_input = torch.zeros(2, 3, 4)
    bad_input = torch.zeros(2, 3, 5)
    good_weight = torch.ones(3, 4)
    bad_weight = torch.ones(3, 5)
    dims = (3, 4)

    report(f"{name} / valid", lambda: function(good_input, dims, good_weight))
    report(f"{name} / wrong input suffix", lambda: function(bad_input, dims, good_weight))
    report(f"{name} / wrong weight", lambda: function(good_input, dims, bad_weight))


def main() -> None:
    exercise("current", current_form)
    exercise("tuple substitution", tuple_substitution)
    exercise("joined substitution", joined_substitution)
    exercise("names only", names_only)


if __name__ == "__main__":
    main()
