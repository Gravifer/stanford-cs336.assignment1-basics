"""Generate self-contained RoPE and attention parity bundles."""

from __future__ import annotations

from importlib.metadata import version

import numpy as np
import torch

from cs336_basics.nn.attention import MultiheadSelfAttention, RotaryPositionalEmbedding
from tests.adapters import (
    run_multihead_self_attention,
    run_multihead_self_attention_with_rope,
    run_rope,
    run_scaled_dot_product_attention,
)

from fixture_tools import descriptor, source_metadata, write_bundle


def _producer() -> dict[str, object]:
    return {
        "language": "Python",
        "runtime_version": version("cs336_basics"),
        "packages": {"numpy": np.__version__, "torch": torch.__version__},
    }


def _gradient_metadata() -> dict[str, object]:
    return {
        "present": True,
        "objective": "sum(expected.output ** 2)",
        "physical_representation": "dense",
    }


def _write_rope(source: dict[str, object]) -> None:
    input_tensor = (torch.arange(48, dtype=torch.float32).reshape(2, 3, 8) / 13 - 1).requires_grad_()
    positions = torch.tensor([[0, 2, 5], [1, 3, 4]], dtype=torch.int64)
    output = run_rope(
        d_k=8,
        theta=10_000.0,
        max_seq_len=6,
        in_query_or_key=input_tensor,
        token_positions=positions,
    )
    (input_gradient,) = torch.autograd.grad(output.square().sum(), (input_tensor,))
    arrays = {
        "input.x": input_tensor.detach().numpy(),
        "input.positions": positions.numpy(),
        "expected.output": output.detach().numpy(),
        "expected.gradient.input.x": input_gradient.detach().numpy(),
    }
    axes = ["batch", "sequence", "feature"]
    metadata = {
        "contract_version": 1,
        "source": source,
        "operation": "run_rope",
        "producer": _producer(),
        "array_file": "rope.npz",
        "arrays": {
            "input.x": descriptor("input", arrays["input.x"], axes),
            "input.positions": descriptor(
                "input",
                arrays["input.positions"],
                ["batch", "sequence"],
                zero_based_values=True,
            ),
            "expected.output": descriptor("expected_output", arrays["expected.output"], axes),
            "expected.gradient.input.x": descriptor(
                "expected_input_gradient", arrays["expected.gradient.input.x"], axes
            ),
        },
        "scalars": {"d_k": 8, "theta": 10_000.0, "max_seq_len": 6},
        "tolerances": {"rtol": 2e-6, "atol": 2e-6, "equal_nan": False},
        "gradients": _gradient_metadata(),
        "notes": [
            "Output and input gradient are produced through tests.adapters.run_rope.",
            "Positions are deliberately non-consecutive and differ by batch.",
        ],
    }
    write_bundle("rope", arrays, metadata)


def _write_sdpa(source: dict[str, object]) -> None:
    generator = torch.Generator().manual_seed(336)
    query = torch.randn(2, 2, 3, 4, generator=generator, dtype=torch.float32, requires_grad=True)
    key = torch.randn(2, 2, 4, 4, generator=generator, dtype=torch.float32, requires_grad=True)
    value = torch.randn(2, 2, 4, 5, generator=generator, dtype=torch.float32, requires_grad=True)
    mask = torch.ones(2, 2, 3, 4, dtype=torch.bool).tril()
    mask[0, 0, 1, :] = False
    output = run_scaled_dot_product_attention(Q=query, K=key, V=value, mask=mask)
    gradients = torch.autograd.grad(output.square().sum(), (query, key, value))
    arrays = {
        "input.query": query.detach().numpy(),
        "input.key": key.detach().numpy(),
        "input.value": value.detach().numpy(),
        "input.mask": mask.numpy(),
        "expected.output": output.detach().numpy(),
        "expected.gradient.input.query": gradients[0].detach().numpy(),
        "expected.gradient.input.key": gradients[1].detach().numpy(),
        "expected.gradient.input.value": gradients[2].detach().numpy(),
    }
    metadata = {
        "contract_version": 1,
        "source": source,
        "operation": "run_scaled_dot_product_attention",
        "producer": _producer(),
        "array_file": "scaled_dot_product_attention.npz",
        "arrays": {
            "input.query": descriptor("input", arrays["input.query"], ["batch", "head", "query", "qk_feature"]),
            "input.key": descriptor("input", arrays["input.key"], ["batch", "head", "key", "qk_feature"]),
            "input.value": descriptor("input", arrays["input.value"], ["batch", "head", "key", "value_feature"]),
            "input.mask": descriptor("input", arrays["input.mask"], ["batch", "head", "query", "key"]),
            "expected.output": descriptor(
                "expected_output",
                arrays["expected.output"],
                ["batch", "head", "query", "value_feature"],
            ),
            "expected.gradient.input.query": descriptor(
                "expected_input_gradient",
                arrays["expected.gradient.input.query"],
                ["batch", "head", "query", "qk_feature"],
            ),
            "expected.gradient.input.key": descriptor(
                "expected_input_gradient",
                arrays["expected.gradient.input.key"],
                ["batch", "head", "key", "qk_feature"],
            ),
            "expected.gradient.input.value": descriptor(
                "expected_input_gradient",
                arrays["expected.gradient.input.value"],
                ["batch", "head", "key", "value_feature"],
            ),
        },
        "scalars": {"scale": "1/sqrt(qk_feature)", "is_causal": False},
        "tolerances": {"rtol": 2e-5, "atol": 2e-5, "equal_nan": False},
        "gradients": _gradient_metadata(),
        "notes": [
            "The boolean mask is adapter input; true permits attention.",
            "One query row is fully masked and must produce finite zero output and gradients.",
        ],
    }
    write_bundle("scaled_dot_product_attention", arrays, metadata)


def _attention_bundle(source: dict[str, object], *, with_rope: bool) -> None:
    generator = torch.Generator().manual_seed(337 if with_rope else 338)
    d_model = 8
    num_heads = 2
    sequence_length = 4
    batch_size = 2
    input_tensor = torch.randn(
        batch_size,
        sequence_length,
        d_model,
        generator=generator,
        dtype=torch.float32,
        requires_grad=True,
    )
    weights = {
        name: torch.randn(d_model, d_model, generator=generator, dtype=torch.float32) / 4
        for name in ("query", "key", "value", "output")
    }
    positions = torch.tensor([[0, 1, 3, 5], [1, 2, 4, 5]], dtype=torch.int64)
    rope = RotaryPositionalEmbedding(theta=10_000.0, d_k=d_model // num_heads, max_seq_len=6) if with_rope else None
    attention = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, rope=rope)
    attention.load_state_dict(
        {
            "q_proj.weight": weights["query"],
            "k_proj.weight": weights["key"],
            "v_proj.weight": weights["value"],
            "output_proj.weight": weights["output"],
        }
    )
    output = attention(input_tensor, token_positions=positions if with_rope else None)
    if with_rope:
        adapter_output = run_multihead_self_attention_with_rope(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=6,
            theta=10_000.0,
            q_proj_weight=weights["query"],
            k_proj_weight=weights["key"],
            v_proj_weight=weights["value"],
            o_proj_weight=weights["output"],
            in_features=input_tensor,
            token_positions=positions,
        )
    else:
        adapter_output = run_multihead_self_attention(
            d_model=d_model,
            num_heads=num_heads,
            q_proj_weight=weights["query"],
            k_proj_weight=weights["key"],
            v_proj_weight=weights["value"],
            o_proj_weight=weights["output"],
            in_features=input_tensor,
        )
    torch.testing.assert_close(output, adapter_output, rtol=0, atol=0)

    assert attention.in_proj_weight is not None
    input_gradient, packed_gradient, output_gradient = torch.autograd.grad(
        output.square().sum(),
        (input_tensor, attention.in_proj_weight, attention.output_proj.weight),
    )
    query_gradient, key_gradient, value_gradient = packed_gradient.split(d_model, dim=0)
    stem = "multihead_self_attention_with_rope" if with_rope else "multihead_self_attention"
    operation = "run_multihead_self_attention_with_rope" if with_rope else "run_multihead_self_attention"
    arrays = {
        "input.x": input_tensor.detach().numpy(),
        "parameter.query_weight": weights["query"].numpy(),
        "parameter.key_weight": weights["key"].numpy(),
        "parameter.value_weight": weights["value"].numpy(),
        "parameter.output_weight": weights["output"].numpy(),
        "expected.output": output.detach().numpy(),
        "expected.gradient.input.x": input_gradient.detach().numpy(),
        "expected.gradient.parameter.query_weight": query_gradient.detach().numpy(),
        "expected.gradient.parameter.key_weight": key_gradient.detach().numpy(),
        "expected.gradient.parameter.value_weight": value_gradient.detach().numpy(),
        "expected.gradient.parameter.output_weight": output_gradient.detach().numpy(),
    }
    if with_rope:
        arrays["input.positions"] = positions.numpy()

    activation_axes = ["batch", "sequence", "model_feature"]
    projection_axes = ["projected_feature", "input_feature"]
    descriptors = {
        "input.x": descriptor("input", arrays["input.x"], activation_axes),
        "parameter.query_weight": descriptor("parameter", arrays["parameter.query_weight"], projection_axes),
        "parameter.key_weight": descriptor("parameter", arrays["parameter.key_weight"], projection_axes),
        "parameter.value_weight": descriptor("parameter", arrays["parameter.value_weight"], projection_axes),
        "parameter.output_weight": descriptor(
            "parameter",
            arrays["parameter.output_weight"],
            ["output_feature", "joined_head_feature"],
        ),
        "expected.output": descriptor("expected_output", arrays["expected.output"], activation_axes),
        "expected.gradient.input.x": descriptor(
            "expected_input_gradient", arrays["expected.gradient.input.x"], activation_axes
        ),
        "expected.gradient.parameter.query_weight": descriptor(
            "expected_parameter_gradient",
            arrays["expected.gradient.parameter.query_weight"],
            projection_axes,
        ),
        "expected.gradient.parameter.key_weight": descriptor(
            "expected_parameter_gradient",
            arrays["expected.gradient.parameter.key_weight"],
            projection_axes,
        ),
        "expected.gradient.parameter.value_weight": descriptor(
            "expected_parameter_gradient",
            arrays["expected.gradient.parameter.value_weight"],
            projection_axes,
        ),
        "expected.gradient.parameter.output_weight": descriptor(
            "expected_parameter_gradient",
            arrays["expected.gradient.parameter.output_weight"],
            ["output_feature", "joined_head_feature"],
        ),
    }
    if with_rope:
        descriptors["input.positions"] = descriptor(
            "input",
            arrays["input.positions"],
            ["batch", "sequence"],
            zero_based_values=True,
        )
    metadata = {
        "contract_version": 1,
        "source": source,
        "operation": operation,
        "producer": _producer(),
        "array_file": f"{stem}.npz",
        "arrays": descriptors,
        "scalars": {
            "d_model": d_model,
            "num_heads": num_heads,
            "is_causal": True,
            "theta": 10_000.0 if with_rope else None,
            "max_seq_len": 6 if with_rope else None,
        },
        "tolerances": {"rtol": 3e-5, "atol": 3e-5, "equal_nan": False},
        "gradients": _gradient_metadata(),
        "notes": [
            "Forward values are checked exactly against the working tests adapter.",
            "Python stores Q/K/V in one packed parameter; its gradient is unpacked into semantic arrays.",
            "The module is causal by default.",
        ],
    }
    write_bundle(stem, arrays, metadata)


def main() -> None:
    source = source_metadata()
    _write_rope(source)
    _write_sdpa(source)
    _attention_bundle(source, with_rope=False)
    _attention_bundle(source, with_rope=True)


if __name__ == "__main__":
    main()
