"""Probe the dtypes and shapes accepted by the NN initializer helpers."""

from collections.abc import Callable
from typing import Any

import torch

from cs336_basics.nn.initializer import starter_trunc_normal_for_embedding_, starter_trunc_normal_for_linear_


def report(label: str, probe: Callable[[], Any]) -> None:
    try:
        value = probe()
    except Exception as exc:  # probes intentionally exercise unsupported dtypes
        print(f"{label}: {type(exc).__name__}: {str(exc).splitlines()[0]}")
    else:
        print(f"{label}: OK: shape={tuple(value.shape)}, dtype={value.dtype}")


def initialize_linear(dtype: torch.dtype, shape: tuple[int, ...] = (4, 3)) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=dtype)
    return starter_trunc_normal_for_linear_(tensor, d_in=3, d_out=4)


def initialize_embedding(dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.empty((7, 3), dtype=dtype)
    return starter_trunc_normal_for_embedding_(tensor)


def main() -> None:
    for dtype in (torch.float16, torch.float32, torch.float64, torch.int64, torch.complex64):
        report(f"linear / {dtype}", lambda dtype=dtype: initialize_linear(dtype))
        report(f"embedding / {dtype}", lambda dtype=dtype: initialize_embedding(dtype))

    report("linear / packed first dimension", lambda: initialize_linear(torch.float32, (8, 3)))
    report("linear / unrelated shape", lambda: initialize_linear(torch.float32, (2, 5, 7)))
    report("linear / zero element", lambda: initialize_linear(torch.float32, (0, 3)))


if __name__ == "__main__":
    main()
