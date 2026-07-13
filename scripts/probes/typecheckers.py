"""Probe-only runtime typechecker adapters."""

from collections.abc import Callable
from typing import Any

from beartype import beartype
from beartype.roar import BeartypeDecorHintPepUnsupportedException


def beartype_skipping_unsupported(function: Callable[..., Any]) -> Callable[..., Any]:
    """Apply beartype, leaving only callables with unsupported hints unchecked."""
    try:
        return beartype(function)
    except BeartypeDecorHintPepUnsupportedException:
        return function
