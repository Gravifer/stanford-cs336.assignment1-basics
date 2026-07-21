"""Produce supporting statistics for the Transformer resource-accounting problem.

The analytical tables use the course convention that an ``(m, n) @ (n, p)``
matrix multiplication costs ``2 * m * n * p`` FLOPs. Torch observations are
reported separately: :class:`torch.utils.flop_counter.FlopCounterMode` sees a
forward pass with concrete shapes, even when the tensors live on ``meta``.

Examples:
    uv run --group cuda --no-group cpu python scripts/transformer_accounting.py
    uv run --group cuda --no-group cpu python scripts/transformer_accounting.py --torch-observation xl
    uv run --group cuda --no-group cpu python scripts/transformer_accounting.py --torch-observation none
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch.utils.flop_counter import FlopCounterMode

from cs336_basics.nn.model import TransformerLM


@dataclass(frozen=True)
class ModelConfig:
    """Assignment-facing Transformer configuration."""

    key: str
    name: str
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int

    @property
    def head_dim(self) -> int:
        """Return the per-head query, key, and value width."""
        if self.d_model % self.num_heads:
            raise ValueError(f"{self.name}: d_model must be divisible by num_heads")
        return self.d_model // self.num_heads


@dataclass(frozen=True)
class Matmul:
    """One semantically named family of equally shaped matrix multiplications."""

    name: str
    repetitions: int
    m: int
    n: int
    p: int

    @property
    def flops(self) -> int:
        """Evaluate this family under the course's ``2mnp`` convention."""
        return self.repetitions * 2 * self.m * self.n * self.p


@dataclass(frozen=True)
class ModelObservation:
    """Numeric resource figures from one concrete-shape meta model and forward."""

    total: int
    first_layer: int
    lm_head: int
    operators: tuple[tuple[str, int], ...]
    parameter_count: int
    parameter_bytes: int
    buffer_count: int
    buffer_bytes: int


def _default_d_ff(d_model: int) -> int:
    """Match the assignment's nearest-multiple-of-64 SwiGLU width."""
    return 64 * max(1, (d_model + 12) // 24)


def _configs() -> tuple[ModelConfig, ...]:
    vocab_size = 50_257
    context_length = 1_024
    return (
        ModelConfig("small", "GPT-2 small", vocab_size, context_length, 12, 768, 12, _default_d_ff(768)),
        ModelConfig("medium", "GPT-2 medium", vocab_size, context_length, 24, 1_024, 16, _default_d_ff(1_024)),
        ModelConfig("large", "GPT-2 large", vocab_size, context_length, 36, 1_280, 20, _default_d_ff(1_280)),
        ModelConfig("xl", "GPT-2 XL", vocab_size, context_length, 48, 1_600, 25, _default_d_ff(1_600)),
        ModelConfig("xl-long", "GPT-2 XL, 16,384 context", vocab_size, 16_384, 48, 1_600, 25, 4_288),
    )


def _matmuls(config: ModelConfig, batch_size: int) -> tuple[Matmul, ...]:
    """Describe the concrete packed-projection implementation used by the model."""
    tokens = batch_size * config.context_length
    layers = config.num_layers
    heads = config.num_heads
    sequence = config.context_length
    d_model = config.d_model
    d_ff = config.d_ff
    d_head = config.head_dim
    return (
        Matmul("attention packed QKV projection", layers, tokens, d_model, 3 * d_model),
        Matmul("attention output projection", layers, tokens, d_model, d_model),
        Matmul("attention scores (QK^T)", layers * batch_size * heads, sequence, d_head, sequence),
        Matmul("attention value aggregation (AV)", layers * batch_size * heads, sequence, sequence, d_head),
        Matmul("SwiGLU packed gate/value projection", layers, tokens, d_model, 2 * d_ff),
        Matmul("SwiGLU output projection", layers, tokens, d_ff, d_model),
        Matmul("language-model output head", 1, tokens, d_model, config.vocab_size),
    )


def _parameter_breakdown(config: ModelConfig) -> tuple[tuple[str, int], ...]:
    """Return parameter counts for the assignment architecture without weight tying."""
    d_model = config.d_model
    return (
        ("token embeddings", config.vocab_size * d_model),
        ("attention projections", config.num_layers * 4 * d_model**2),
        ("SwiGLU projections", config.num_layers * 3 * d_model * config.d_ff),
        ("block and final RMSNorm gains", (2 * config.num_layers + 1) * d_model),
        ("language-model output head", config.vocab_size * d_model),
    )


def _make_model(config: ModelConfig, dtype: torch.dtype) -> TransformerLM:
    return TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=10_000.0,
        dropout=0.0,
        device=torch.device("meta"),
        dtype=dtype,
    )


def _module_flops(counts: dict[str, dict[object, int]], path: str) -> int:
    return sum(counts.get(path, {}).values())


def _observe_torch(config: ModelConfig, batch_size: int, dtype: torch.dtype) -> ModelObservation:
    """Run one meta forward and return Torch counts plus concrete model storage statistics."""
    model = _make_model(config, dtype)
    model.eval()
    tokens = torch.empty((batch_size, config.context_length), device="meta", dtype=torch.long)
    counter = FlopCounterMode(display=False, depth=4)
    with torch.inference_mode(), counter:
        model(tokens)

    counts = counter.get_flop_counts()
    global_counts = counts.get("Global", {})
    operators = tuple(sorted((str(operation), flops) for operation, flops in global_counts.items()))
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    buffers = tuple(model.buffers())
    return ModelObservation(
        total=counter.get_total_flops(),
        first_layer=_module_flops(counts, "TransformerLM.layers.0"),
        lm_head=_module_flops(counts, "TransformerLM.lm_head"),
        operators=operators,
        parameter_count=sum(parameter.numel() for parameter in parameters),
        parameter_bytes=sum(parameter.numel() * parameter.element_size() for parameter in parameters),
        buffer_count=sum(buffer.numel() for buffer in buffers),
        buffer_bytes=sum(buffer.numel() * buffer.element_size() for buffer in buffers),
    )


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _fmt_percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.2f}%"


def _print_parameter_report(
    config: ModelConfig,
    dtype: torch.dtype,
    observed: ModelObservation | None,
) -> None:
    breakdown = _parameter_breakdown(config)
    total = sum(count for _, count in breakdown)
    element_size = torch.empty((), dtype=dtype).element_size()
    parameter_bytes = total * element_size

    print("## GPT-2 XL parameters and storage")
    print()
    print("| Component | Parameters | Share |")
    print("|---|---:|---:|")
    for name, count in breakdown:
        print(f"| {name} | {_fmt_int(count)} | {_fmt_percent(count, total)} |")
    print(f"| **Total** | **{_fmt_int(total)}** | **100.00%** |")
    print()
    print(f"- Parameter dtype: `{dtype}` ({element_size} bytes per parameter)")
    print(f"- Parameter storage: {_fmt_int(parameter_bytes)} bytes = {parameter_bytes / 10**9:.3f} GB = {parameter_bytes / 2**30:.3f} GiB")
    if observed is not None:
        print(
            f"- Instantiated meta model: {_fmt_int(observed.parameter_count)} trainable parameters, "
            f"{_fmt_int(observed.parameter_bytes)} parameter bytes"
        )
        print(
            f"- Formula-minus-model deltas: {_fmt_int(total - observed.parameter_count)} parameters, "
            f"{_fmt_int(parameter_bytes - observed.parameter_bytes)} bytes"
        )
        print(
            f"- Separately allocated non-parameter buffers: {_fmt_int(observed.buffer_count)} elements, "
            f"{_fmt_int(observed.buffer_bytes)} bytes"
        )
    print()


def _print_xl_matmuls(config: ModelConfig, batch_size: int) -> None:
    matmuls = _matmuls(config, batch_size)
    total = sum(matmul.flops for matmul in matmuls)
    print("## GPT-2 XL forward matrix multiplications")
    print()
    print("| Matrix multiplication | Repetitions | m | n | p | FLOPs | Share |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for matmul in matmuls:
        print(
            f"| {matmul.name} | {_fmt_int(matmul.repetitions)} | {_fmt_int(matmul.m)} | "
            f"{_fmt_int(matmul.n)} | {_fmt_int(matmul.p)} | {_fmt_int(matmul.flops)} | "
            f"{_fmt_percent(matmul.flops, total)} |"
        )
    print(f"| **Total** |  |  |  |  | **{_fmt_int(total)}** | **100.00%** |")
    print()


def _print_model_comparison(configs: tuple[ModelConfig, ...], batch_size: int) -> None:
    print("## Component FLOP proportions at context length 1,024")
    print()
    print("| Model | Component | FLOPs | Share |")
    print("|---|---|---:|---:|")
    for config in configs:
        matmuls = _matmuls(config, batch_size)
        total = sum(matmul.flops for matmul in matmuls)
        for matmul in matmuls:
            print(f"| {config.name} | {matmul.name} | {_fmt_int(matmul.flops)} | {_fmt_percent(matmul.flops, total)} |")
        print(f"| **{config.name}** | **Total** | **{_fmt_int(total)}** | **100.00%** |")
    print()


def _print_context_comparison(short: ModelConfig, long: ModelConfig, batch_size: int) -> None:
    short_matmuls = _matmuls(short, batch_size)
    long_matmuls = _matmuls(long, batch_size)
    short_total = sum(matmul.flops for matmul in short_matmuls)
    long_total = sum(matmul.flops for matmul in long_matmuls)
    print("## GPT-2 XL context-length comparison")
    print()
    print("| Context length | Total FLOPs | Relative to 1,024 |")
    print("|---:|---:|---:|")
    print(f"| {_fmt_int(short.context_length)} | {_fmt_int(short_total)} | 1.000x |")
    print(f"| {_fmt_int(long.context_length)} | {_fmt_int(long_total)} | {long_total / short_total:.3f}x |")
    print()
    print("| Component | Share at 1,024 | Share at 16,384 |")
    print("|---|---:|---:|")
    for short_matmul, long_matmul in zip(short_matmuls, long_matmuls, strict=True):
        print(
            f"| {short_matmul.name} | {_fmt_percent(short_matmul.flops, short_total)} | "
            f"{_fmt_percent(long_matmul.flops, long_total)} |"
        )
    print()


def _print_torch_observations(
    configs: tuple[ModelConfig, ...],
    observations: dict[str, ModelObservation],
    batch_size: int,
) -> None:
    if not observations:
        return
    print("## Torch `FlopCounterMode` observations")
    print()
    print(
        "These are numeric observations from concrete-shape meta forwards. They are not symbolic formulas, "
        "and Torch's operator registry currently attributes all counted contractions here to `aten.bmm`."
    )
    print()
    print("| Model | Analytical 2mnp | Torch observed | Delta | First layer | LM head | Global operators |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for config in configs:
        result = observations.get(config.key)
        if result is None:
            continue
        observation = result
        analytical = sum(matmul.flops for matmul in _matmuls(config, batch_size))
        operators = ", ".join(f"{name}: {_fmt_int(flops)}" for name, flops in observation.operators)
        print(
            f"| {config.name} | {_fmt_int(analytical)} | {_fmt_int(observation.total)} | "
            f"{_fmt_int(observation.total - analytical)} | {_fmt_int(observation.first_layer)} | "
            f"{_fmt_int(observation.lm_head)} | {operators} |"
        )
    print()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1, help="number of independent input sequences")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32", "float64"),
        default="float32",
        help="parameter dtype used for storage figures and meta models",
    )
    parser.add_argument(
        "--torch-observation",
        choices=("all", "xl", "none"),
        default="all",
        help="which assignment configurations to run through FlopCounterMode",
    )
    return parser


def main() -> None:
    """Print a Markdown report for the assignment configurations."""
    args = _parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")
    dtype = getattr(torch, args.dtype)
    configs = _configs()
    short_configs = configs[:4]
    xl = configs[3]
    xl_long = configs[4]

    selected = ()
    if args.torch_observation == "all":
        selected = configs
    elif args.torch_observation == "xl":
        selected = (xl,)

    observations: dict[str, ModelObservation] = {}
    for config in selected:
        observations[config.key] = _observe_torch(config, args.batch_size, dtype)

    print("# Transformer accounting supporting statistics")
    print()
    print("## Assumptions")
    print()
    print(f"- Batch size: {args.batch_size}")
    print("- Forward pass only; dropout disabled and gradients disabled")
    print("- Untied token-embedding and language-model-head weights, matching this repository")
    print("- Packed QKV and packed SwiGLU gate/value projections, matching the selected implementations")
    print("- Matrix multiplication cost: `2 * m * n * p` FLOPs")
    print()

    _print_parameter_report(xl, dtype, observations.get("xl"))
    _print_xl_matmuls(xl, args.batch_size)
    _print_model_comparison(short_configs, args.batch_size)
    _print_context_comparison(xl, xl_long, args.batch_size)
    _print_torch_observations(configs, observations, args.batch_size)


if __name__ == "__main__":
    main()
