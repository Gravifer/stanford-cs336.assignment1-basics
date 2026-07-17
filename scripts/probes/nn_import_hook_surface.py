"""Exercise the implemented NN surface with the jaxtyping import hook on/off.

This intentionally imports ``cs336_basics.nn`` only after configuring the hook.
Run this script in a fresh process for each mode.
"""

import argparse
import inspect
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, cast

import torch
import beartype as beartype_module
from jaxtyping import install_import_hook

from scripts.probes.typecheckers import beartype_skipping_unsupported


VERBOSE = False


def report(label: str, probe: Callable[[], Any]) -> None:
    try:
        value = probe()
    except Exception as exc:  # probes intentionally exercise invalid calls
        message = str(exc) if VERBOSE else str(exc).splitlines()[0]
        print(f"{label}: {type(exc).__name__}: {message}")
    else:
        if isinstance(value, torch.Tensor):
            value = f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
        print(f"{label}: OK: {value}")


def wrapper_depth(function: Callable[..., Any]) -> int:
    depth = 0
    while hasattr(function, "__wrapped__"):
        depth += 1
        function = cast("Callable[..., Any]", function.__wrapped__)
    return depth


def main() -> None:
    global VERBOSE
    parser = argparse.ArgumentParser()
    parser.add_argument("checking", choices=("on", "off"))
    parser.add_argument(
        "--scope",
        choices=("package", "functional", "feed_forward", "attention", "modules"),
        default="package",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--typechecker",
        choices=("strict", "skip-unsupported"),
        default="strict",
    )
    args = parser.parse_args()
    VERBOSE = args.verbose

    hook_target = {
        "package": "cs336_basics.nn",
        "functional": "cs336_basics.nn.functional",
        "feed_forward": "cs336_basics.nn.feed_forward",
        "attention": "cs336_basics.nn.attention",
        "modules": "cs336_basics.nn.modules",
    }[args.scope]
    if args.typechecker == "skip-unsupported":
        beartype_module._cs336_probe = beartype_skipping_unsupported
        typechecker = "beartype._cs336_probe"
    else:
        typechecker = "beartype.beartype"
    hook_manager = install_import_hook(hook_target, typechecker) if args.checking == "on" else None
    if hook_manager is not None:
        print(f"hook target: {hook_target}")
        for candidate in (
            "cs336_basics.nn.functional",
            "cs336_basics.nn.feed_forward",
            "cs336_basics.nn.attention",
            "cs336_basics.nn.modules",
        ):
            print(f"instrument {candidate}: {hook_manager.hook.should_instrument(candidate)}")
    hook = hook_manager if hook_manager is not None else nullcontext()
    with hook:
        from cs336_basics.nn import functional as functional
        from cs336_basics.nn.attention import RotaryPositionalEmbedding
        from cs336_basics.nn.feed_forward import SwiGLU_delegate
        from cs336_basics.nn.modules import Embedding, Linear, RMSNorm

    print(f"runtime checking: {args.checking}; scope: {args.scope}; typechecker: {args.typechecker}")

    print("\n[functional]")
    report("gelu / valid", lambda: functional.gelu(torch.zeros(2, 3)))
    report("gelu / documented exact mode", lambda: functional.gelu(torch.zeros(2, 3), approximate="none"))
    report(
        "gelu / invalid mode",
        lambda: functional.gelu(
            torch.zeros(2, 3),
            approximate="invalid",  # ty: ignore[invalid-argument-type]
        ),
    )
    report(
        "linear / valid",
        lambda: functional.linear(torch.zeros(2, 3), torch.zeros(4, 3)),
    )
    report(
        "rms_norm / valid",
        lambda: functional.rms_norm(torch.zeros(2, 3, 4), (3, 4), torch.ones(3, 4)),
    )

    print("\n[module construction and attributes]")
    pretrained_weight = torch.zeros(7, 3)
    report("Embedding / default weight", lambda: Embedding(7, 3))
    report("Embedding / supplied weight", lambda: Embedding(7, 3, _weight=pretrained_weight))
    report("Embedding.freeze exists", lambda: hasattr(Embedding(7, 3), "freeze"))
    report("RMSNorm affine weight", lambda: RMSNorm(4).weight)
    report("RMSNorm non-affine weight", lambda: RMSNorm(4, elementwise_affine=False).weight)
    report("Linear.bias", lambda: Linear(3, 4).bias)

    print("\n[module forwards]")
    embedding = Embedding(7, 3)
    linear = Linear(3, 4)
    rms_norm = RMSNorm(4)
    swiglu = SwiGLU_delegate(4, 8)
    report("Embedding.forward / valid", lambda: embedding(torch.tensor([[0, 1]])))
    report("Embedding.forward / float indices", lambda: embedding(torch.zeros(1, 2)))
    report("Linear.forward / valid", lambda: linear(torch.zeros(2, 3)))
    report("Linear.forward / wrong final width", lambda: linear(torch.zeros(2, 5)))
    report("RMSNorm.forward / valid", lambda: rms_norm(torch.zeros(2, 3, 4)))
    report("RMSNorm.forward / wrong final width", lambda: rms_norm(torch.zeros(2, 3, 5)))
    report("SwiGLU.forward / valid", lambda: swiglu(torch.zeros(2, 3, 4)))
    report("SwiGLU.forward / wrong final width", lambda: swiglu(torch.zeros(2, 3, 5)))

    print("\n[RoPE wrapper]")
    rope = RotaryPositionalEmbedding(theta=10_000.0, d_k=4, max_seq_len=8)
    report("forward wrapper depth", lambda: wrapper_depth(type(rope).forward))
    report(
        "forward / valid",
        lambda: rope(torch.zeros(2, 3, 4), torch.arange(3).expand(2, 3)),
    )
    original_forward = inspect.unwrap(type(rope).forward)
    report(
        "unwrapped forward / valid",
        lambda: original_forward(rope, torch.zeros(2, 3, 4), torch.arange(3).expand(2, 3)),
    )


if __name__ == "__main__":
    main()
