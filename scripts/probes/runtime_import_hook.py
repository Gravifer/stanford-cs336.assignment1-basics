"""Compare imports of an NN-like module with jaxtyping checking on or off.

Run as either::

    python scripts/probes/runtime_import_hook.py on
    python scripts/probes/runtime_import_hook.py off
"""

import argparse
import importlib
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Final

import torch
from jaxtyping import install_import_hook


TARGET_MODULE: Final = "scripts.probes.import_hook_target"


def report(label: str, probe: Callable[[], Any]) -> None:
    try:
        value = probe()
    except Exception as exc:  # probes intentionally exercise invalid calls
        print(f"{label}: {type(exc).__name__}: {exc}")
    else:
        if isinstance(value, torch.Tensor):
            value = f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
        print(f"{label}: OK: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checking", choices=("on", "off"))
    args = parser.parse_args()

    hook = install_import_hook(TARGET_MODULE, "beartype.beartype") if args.checking == "on" else nullcontext()
    with hook:
        target = importlib.import_module(TARGET_MODULE)

    print(f"runtime checking: {args.checking}")
    report(
        "named variadic mismatch",
        lambda: target.same_prefix(torch.zeros(2, 3, 4), torch.zeros(6, 4)),
    )
    report("bad width with explicit guard", lambda: target.explicit_width_guard(torch.zeros(2, 5)))
    report(
        "integer input with annotation only", lambda: target.annotation_only_dtype(torch.zeros(2, dtype=torch.int64))
    )


if __name__ == "__main__":
    main()
