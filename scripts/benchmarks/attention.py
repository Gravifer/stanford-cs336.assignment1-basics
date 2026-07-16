"""Benchmark RoPE representations and experimental MHA activation paths.

Examples:
    uv run python scripts/benchmarks/attention.py rope --device auto
    uv run python scripts/benchmarks/attention.py mha --backward --compile-mode default
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import torch

from cs336_basics.nn.attention import MultiheadSelfAttention, RotaryPositionalEmbedding

type Layout = Literal["head_before_sequence", "head_after_sequence"]
type QKStrategy = Literal["auto", "separate", "stacked"]
type PositionKind = Literal["sequence", "singleton_head", "per_head"]


@dataclass(frozen=True)
class Result:
    case: str
    milliseconds: float
    peak_megabytes: float | None
    strides: str
    contiguous: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("rope", "mha"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        subparser.add_argument(
            "--dtype",
            choices=("float16", "bfloat16", "float32", "float64"),
            default="float32",
        )
        subparser.add_argument("--batch-size", type=int, default=4)
        subparser.add_argument("--sequence-length", type=int, default=128)
        subparser.add_argument("--num-heads", type=int, default=8)
        subparser.add_argument("--head-dim", type=int, default=64)
        subparser.add_argument("--warmup", type=int, default=10)
        subparser.add_argument("--repeats", type=int, default=30)
        subparser.add_argument("--backward", action="store_true")
        subparser.add_argument(
            "--compile-mode",
            choices=("none", "default", "reduce-overhead", "max-autotune"),
            default="none",
        )
        subparser.add_argument(
            "--positions",
            choices=("all", "sequence", "singleton_head", "per_head"),
            default="all",
        )
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    return torch.device(name)


def _dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _positions(
    kind: PositionKind,
    sequence_length: int,
    num_heads: int,
    device: torch.device,
) -> torch.Tensor:
    base = torch.arange(sequence_length, device=device)
    if kind == "sequence":
        return base
    if kind == "singleton_head":
        return base.unsqueeze(0)
    offsets = torch.arange(num_heads, device=device).unsqueeze(-1)
    return (base + offsets) % sequence_length


def _position_kinds(name: Literal["all", "sequence", "singleton_head", "per_head"]) -> Iterable[PositionKind]:
    if name == "all":
        return ("sequence", "singleton_head", "per_head")
    return (name,)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _compiled(function: Callable[..., torch.Tensor | tuple[torch.Tensor, torch.Tensor]], mode: str) -> Callable:
    if mode == "none":
        return function
    return torch.compile(function, mode=mode)


def _time(
    function: Callable[[], torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    leaves: tuple[torch.Tensor, ...],
    module: torch.nn.Module | None,
    *,
    backward: bool,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> tuple[float, float | None]:
    def iteration() -> None:
        if module is not None:
            module.zero_grad(set_to_none=True)
        for leaf in leaves:
            leaf.grad = None
        output = function()
        if backward:
            tensors = output if isinstance(output, tuple) else (output,)
            loss = tensors[0].square().mean()
            for tensor in tensors[1:]:
                loss = loss + tensor.square().mean()
            loss.backward()

    for _ in range(warmup):
        iteration()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(repeats):
        iteration()
    _sync(device)
    elapsed = (time.perf_counter() - start) * 1_000 / repeats
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    return elapsed, peak


def _assert_parity(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    dtype: torch.dtype,
) -> None:
    tolerance = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-5
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=tolerance, atol=tolerance)


def _rope_evaluation(
    rope: RotaryPositionalEmbedding,
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    *,
    stacked: bool,
    layout: Layout,
) -> tuple[torch.Tensor, ...]:
    q = query.detach().clone().requires_grad_()
    k = key.detach().clone().requires_grad_()
    output = rope.apply_qk(q, k, positions, _stacked=stacked, _layout_strategy=layout)
    loss = output[0].square().mean() + output[1].square().mean()
    gradients = torch.autograd.grad(loss, (q, k))
    return output[0].detach(), output[1].detach(), gradients[0], gradients[1]


def _run_rope(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> list[Result]:
    if args.head_dim % 2:
        raise SystemExit("--head-dim must be even for RoPE")
    canonical_q = torch.randn(
        args.batch_size,
        args.num_heads,
        args.sequence_length,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    canonical_k = torch.randn_like(canonical_q)
    results: list[Result] = []
    baseline: dict[tuple[Layout, PositionKind], tuple[torch.Tensor, ...]] = {}
    for layout in ("head_before_sequence", "head_after_sequence"):
        query = canonical_q if layout == "head_before_sequence" else canonical_q.transpose(-3, -2)
        key = canonical_k if layout == "head_before_sequence" else canonical_k.transpose(-3, -2)
        for position_kind in _position_kinds(args.positions):
            positions = _positions(position_kind, args.sequence_length, args.num_heads, device)
            for matrix_form in (True, False):
                rope = RotaryPositionalEmbedding(
                    10_000.0,
                    args.head_dim,
                    args.sequence_length,
                    device=device,
                    use_matrix_form=matrix_form,
                ).to(dtype=dtype)
                for strategy in ("separate", "stacked"):
                    stacked = strategy == "stacked"
                    evaluation = _rope_evaluation(
                        rope,
                        query,
                        key,
                        positions,
                        stacked=stacked,
                        layout=layout,
                    )
                    key_name = (layout, position_kind)
                    if key_name not in baseline:
                        baseline[key_name] = evaluation
                    else:
                        _assert_parity(evaluation, baseline[key_name], dtype)

                    q = query.detach().clone().requires_grad_(args.backward)
                    k = key.detach().clone().requires_grad_(args.backward)
                    run = _compiled(
                        lambda: rope.apply_qk(
                            q,
                            k,
                            positions,
                            _stacked=stacked,
                            _layout_strategy=layout,
                        ),
                        args.compile_mode,
                    )
                    milliseconds, peak = _time(
                        run,
                        (q, k),
                        rope,
                        backward=args.backward,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        device=device,
                    )
                    results.append(
                        Result(
                            f"{'matrix' if matrix_form else 'elementwise'}/"
                            f"{'zero-copy-packed' if strategy == 'auto' else strategy}/{position_kind}/{layout}",
                            milliseconds,
                            peak,
                            str(query.stride()),
                            str(query.is_contiguous()),
                        )
                    )
    return results


def _mha_evaluation(
    module: MultiheadSelfAttention,
    x: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    input_ = x.detach().clone().requires_grad_()
    output = module(input_, token_positions=positions)
    gradients = torch.autograd.grad(output.square().mean(), (input_, *module.parameters()))
    return output.detach(), *(gradient.detach() for gradient in gradients)


def _run_mha(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> list[Result]:
    if args.head_dim % 2:
        raise SystemExit("--head-dim must be even for RoPE")
    d_model = args.num_heads * args.head_dim
    x_template = torch.randn(
        args.batch_size,
        args.sequence_length,
        d_model,
        device=device,
        dtype=dtype,
    )
    reference = MultiheadSelfAttention(d_model, args.num_heads, device=device, dtype=dtype)
    reference_state = reference.state_dict()
    results: list[Result] = []
    baseline: dict[PositionKind, tuple[torch.Tensor, ...]] = {}
    for position_kind in _position_kinds(args.positions):
        positions = _positions(position_kind, args.sequence_length, args.num_heads, device)
        for matrix_form in (True, False):
            for layout in ("head_before_sequence", "head_after_sequence"):
                for strategy in ("auto", "separate", "stacked"):
                    rope = RotaryPositionalEmbedding(
                        10_000.0,
                        args.head_dim,
                        args.sequence_length,
                        device=device,
                        use_matrix_form=matrix_form,
                    ).to(dtype=dtype)
                    module = MultiheadSelfAttention(
                        d_model,
                        args.num_heads,
                        rope=rope,
                        device=device,
                        dtype=dtype,
                        _layout_strategy=layout,
                        _qk_execution_strategy=strategy,
                    )
                    module.load_state_dict(reference_state)
                    evaluation = _mha_evaluation(module, x_template, positions)
                    if position_kind not in baseline:
                        baseline[position_kind] = evaluation
                    else:
                        _assert_parity(evaluation, baseline[position_kind], dtype)

                    x = x_template.detach().clone().requires_grad_(args.backward)
                    run = _compiled(lambda: module(x, token_positions=positions), args.compile_mode)
                    projected = module._in_projection(x, x, x)
                    q, _, _, _ = module._split_heads(*projected)
                    milliseconds, peak = _time(
                        run,
                        (x,),
                        module,
                        backward=args.backward,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        device=device,
                    )
                    results.append(
                        Result(
                            f"{'matrix' if matrix_form else 'elementwise'}/"
                            f"{'zero-copy-packed' if strategy == 'auto' else strategy}/{position_kind}/{layout}",
                            milliseconds,
                            peak,
                            str(q.stride()),
                            str(q.is_contiguous()),
                        )
                    )
    return results


def _print_metadata(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"CUDA: {torch.version.cuda}; cuDNN: {torch.backends.cudnn.version()}")
    else:
        print("CUDA: untested (unavailable or not selected)")
    print(f"dtype: {dtype}; compile: {args.compile_mode}; backward: {args.backward}")


def _print_results(results: list[Result]) -> None:
    print()
    print("| case | ms | peak MiB | strides | contiguous |")
    print("|---|---:|---:|---|---|")
    for result in results:
        peak = "n/a" if result.peak_megabytes is None else f"{result.peak_megabytes:.1f}"
        print(f"| {result.case} | {result.milliseconds:.3f} | {peak} | `{result.strides}` | {result.contiguous} |")


def main() -> None:
    args = _parser().parse_args()
    device = _device(args.device)
    dtype = _dtype(args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        print("Warning: float16 CPU kernels may be unsupported or unrepresentative", file=sys.stderr)
    torch.manual_seed(0)
    _print_metadata(args, device, dtype)
    results = _run_rope(args, device, dtype) if args.command == "rope" else _run_mha(args, device, dtype)
    _print_results(results)


if __name__ == "__main__":
    main()
