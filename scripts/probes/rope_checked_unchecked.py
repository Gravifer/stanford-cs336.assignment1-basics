"""Compare the current checked RoPE forward with its unwrapped implementation."""

import inspect
from collections.abc import Callable
from typing import Any

import torch

from cs336_basics.nn.attention import RotaryPositionalEmbedding


def outcome(probe: Callable[[], Any]) -> str:
    try:
        value = probe()
    except Exception as exc:  # probes intentionally exercise invalid calls
        return f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    else:
        return f"OK: shape={tuple(value.shape)}, dtype={value.dtype}"


def compare(
    label: str,
    rope: RotaryPositionalEmbedding,
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    broadcast_positions: bool = True,
) -> None:
    original_forward = inspect.unwrap(type(rope).forward)
    checked = outcome(lambda: rope(x, positions, broadcast_positions=broadcast_positions))
    unchecked = outcome(lambda: original_forward(rope, x, positions, broadcast_positions=broadcast_positions))
    print(f"{label}\n  checked:   {checked}\n  unchecked: {unchecked}")


def main() -> None:
    rope = RotaryPositionalEmbedding(theta=10_000.0, d_k=4, max_seq_len=8)
    exact_x = torch.zeros(2, 3, 4)
    exact_positions = torch.arange(3).expand(2, 3)

    compare("exact batch", rope, exact_x, exact_positions)
    compare("unbatched positions broadcast", rope, exact_x, torch.arange(3))
    compare(
        "nonempty batch suffix broadcast",
        rope,
        torch.zeros(2, 3, 5, 4),
        torch.arange(5).expand(3, 5),
    )
    compare("wrong key width", rope, torch.zeros(2, 3, 6), exact_positions)
    compare("wrong sequence length", rope, exact_x, torch.arange(2).expand(2, 2))
    compare("floating positions", rope, exact_x, exact_positions.float())
    compare("int32 positions", rope, exact_x, exact_positions.to(torch.int32))
    compare("negative positions", rope, exact_x, exact_positions - 1)
    compare("out-of-range positions", rope, exact_x, exact_positions + 8)
    compare("float16 input", rope, exact_x.to(torch.float16), exact_positions)
    compare("float64 input", rope, exact_x.to(torch.float64), exact_positions)
    compare("integer input", rope, exact_x.to(torch.int64), exact_positions)
    compare(
        "broadcast disabled",
        rope,
        exact_x,
        torch.arange(3),
        broadcast_positions=False,
    )


if __name__ == "__main__":
    main()
