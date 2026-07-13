"""Probe beartype support for terminal return annotations."""

from collections.abc import Callable
from typing import Any, Never, NoReturn

from beartype import beartype
from jaxtyping import jaxtyped


def report(label: str, probe: Callable[[], Any]) -> None:
    try:
        value = probe()
    except Exception as exc:  # decoration failures are the expected observation
        print(f"{label}: {type(exc).__name__}: {str(exc).splitlines()[0]}")
    else:
        print(f"{label}: OK: {value}")


def decorate_never_with_beartype() -> Callable[[], Never]:
    def terminal() -> Never:
        raise RuntimeError

    return beartype(terminal)


def decorate_no_return_with_beartype() -> Callable[[], NoReturn]:
    def terminal() -> NoReturn:
        raise RuntimeError

    return beartype(terminal)


def decorate_none_with_beartype() -> Callable[[], None]:
    def terminal() -> None:
        raise RuntimeError

    return beartype(terminal)


def decorate_never_with_jaxtyped() -> Callable[[], Never]:
    def terminal() -> Never:
        raise RuntimeError

    return jaxtyped(typechecker=beartype)(terminal)


def decorate_no_return_with_jaxtyped() -> Callable[[], NoReturn]:
    def terminal() -> NoReturn:
        raise RuntimeError

    return jaxtyped(typechecker=beartype)(terminal)


def main() -> None:
    report("beartype / Never", decorate_never_with_beartype)
    report("beartype / NoReturn", decorate_no_return_with_beartype)
    report("beartype / None", decorate_none_with_beartype)
    report("jaxtyped+beartype / Never", decorate_never_with_jaxtyped)
    report("jaxtyped+beartype / NoReturn", decorate_no_return_with_jaxtyped)


if __name__ == "__main__":
    main()
