"""Generate self-contained transformer block and language-model parity bundles."""

from __future__ import annotations

from importlib.metadata import version

import numpy as np
import torch

from cs336_basics.nn.model import TransformerBlock, TransformerLM
from tests.adapters import run_transformer_block, run_transformer_lm

from fixture_tools import descriptor, source_metadata, write_bundle


def _producer() -> dict[str, object]:
    return {
        "language": "Python",
        "runtime_version": version("cs336_basics"),
        "packages": {"numpy": np.__version__, "torch": torch.__version__},
    }


def _random_weight(generator: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=generator, dtype=torch.float32) / 5


def _course_block_weights(generator: torch.Generator, d_model: int, d_ff: int) -> dict[str, torch.Tensor]:
    return {
        "ln1.weight": 0.5 + torch.rand(d_model, generator=generator),
        "attn.q_proj.weight": _random_weight(generator, d_model, d_model),
        "attn.k_proj.weight": _random_weight(generator, d_model, d_model),
        "attn.v_proj.weight": _random_weight(generator, d_model, d_model),
        "attn.output_proj.weight": _random_weight(generator, d_model, d_model),
        "ln2.weight": 0.5 + torch.rand(d_model, generator=generator),
        "ffn.w1.weight": _random_weight(generator, d_ff, d_model),
        "ffn.w2.weight": _random_weight(generator, d_model, d_ff),
        "ffn.w3.weight": _random_weight(generator, d_ff, d_model),
    }


def _semantic_block_gradients(
    native_gradients: dict[str, torch.Tensor],
    *,
    prefix: str,
    d_model: int,
    d_ff: int,
) -> dict[str, torch.Tensor]:
    attention_input = native_gradients[f"{prefix}attn.update.in_proj_weight"]
    query, key, value = attention_input.split(d_model, dim=0)
    feed_forward_input = native_gradients[f"{prefix}ffn.update.in_weight"]
    value_ffn, gate_ffn = feed_forward_input.split(d_ff, dim=0)
    return {
        "attention_norm": native_gradients[f"{prefix}attn.norm.weight"],
        "query_weight": query,
        "key_weight": key,
        "value_weight": value,
        "attention_output_weight": native_gradients[f"{prefix}attn.update.output_proj.weight"],
        "feed_forward_norm": native_gradients[f"{prefix}ffn.norm.weight"],
        "w1": gate_ffn,
        "w2": native_gradients[f"{prefix}ffn.update.out_weight"],
        "w3": value_ffn,
    }


def _semantic_block_parameters(weights: dict[str, torch.Tensor], *, prefix: str = "") -> dict[str, torch.Tensor]:
    return {
        "attention_norm": weights[f"{prefix}ln1.weight"],
        "query_weight": weights[f"{prefix}attn.q_proj.weight"],
        "key_weight": weights[f"{prefix}attn.k_proj.weight"],
        "value_weight": weights[f"{prefix}attn.v_proj.weight"],
        "attention_output_weight": weights[f"{prefix}attn.output_proj.weight"],
        "feed_forward_norm": weights[f"{prefix}ln2.weight"],
        "w1": weights[f"{prefix}ffn.w1.weight"],
        "w2": weights[f"{prefix}ffn.w2.weight"],
        "w3": weights[f"{prefix}ffn.w3.weight"],
    }


def _parameter_axes(name: str) -> list[str]:
    if name.endswith("norm"):
        return ["model_feature"]
    if name.endswith("token_embedding"):
        return ["vocabulary", "model_feature"]
    if name.endswith("lm_head"):
        return ["vocabulary", "model_feature"]
    if name.endswith("w1") or name.endswith("w3"):
        return ["feed_forward_feature", "model_feature"]
    if name.endswith("w2"):
        return ["model_feature", "feed_forward_feature"]
    return ["projected_feature", "input_feature"]


def _add_parameters(
    arrays: dict[str, np.ndarray],
    descriptors: dict[str, dict[str, object]],
    parameters: dict[str, torch.Tensor],
    gradients: dict[str, torch.Tensor],
    *,
    namespace: str = "",
) -> None:
    for name, tensor in parameters.items():
        qualified = f"{namespace}.{name}" if namespace else name
        parameter_key = f"parameter.{qualified}"
        gradient_key = f"expected.gradient.parameter.{qualified}"
        arrays[parameter_key] = tensor.detach().numpy()
        arrays[gradient_key] = gradients[name].detach().numpy()
        axes = _parameter_axes(name)
        descriptors[parameter_key] = descriptor("parameter", arrays[parameter_key], axes)
        descriptors[gradient_key] = descriptor("expected_parameter_gradient", arrays[gradient_key], axes)


def _write_block(source: dict[str, object]) -> None:
    generator = torch.Generator().manual_seed(340)
    d_model, num_heads, d_ff, context_length = 8, 2, 16, 6
    theta = 10_000.0
    weights = _course_block_weights(generator, d_model, d_ff)
    input_tensor = torch.randn(2, 4, d_model, generator=generator, requires_grad=True)
    block = TransformerBlock(d_model, num_heads, d_ff, context_length, theta)
    block.load_state_dict(weights)
    output = block(input_tensor)
    adapter_output = run_transformer_block(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=context_length,
        theta=theta,
        weights=weights,
        in_features=input_tensor,
    )
    torch.testing.assert_close(output, adapter_output, rtol=0, atol=0)

    named_parameters = dict(block.named_parameters())
    gradient_values = torch.autograd.grad(
        output.square().sum(),
        (input_tensor, *named_parameters.values()),
    )
    native_gradients = dict(zip(named_parameters, gradient_values[1:], strict=True))
    semantic_parameters = _semantic_block_parameters(weights)
    semantic_gradients = _semantic_block_gradients(
        native_gradients,
        prefix="",
        d_model=d_model,
        d_ff=d_ff,
    )
    arrays = {
        "input.x": input_tensor.detach().numpy(),
        "expected.output": output.detach().numpy(),
        "expected.gradient.input.x": gradient_values[0].detach().numpy(),
    }
    activation_axes = ["batch", "sequence", "model_feature"]
    descriptors = {
        "input.x": descriptor("input", arrays["input.x"], activation_axes),
        "expected.output": descriptor("expected_output", arrays["expected.output"], activation_axes),
        "expected.gradient.input.x": descriptor(
            "expected_input_gradient", arrays["expected.gradient.input.x"], activation_axes
        ),
    }
    _add_parameters(arrays, descriptors, semantic_parameters, semantic_gradients)
    metadata = {
        "contract_version": 1,
        "source": source,
        "operation": "run_transformer_block",
        "producer": _producer(),
        "array_file": "transformer_block.npz",
        "arrays": descriptors,
        "scalars": {
            "d_model": d_model,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "context_length": context_length,
            "theta": theta,
        },
        "tolerances": {"rtol": 1e-4, "atol": 1e-4, "equal_nan": False},
        "gradients": {
            "present": True,
            "objective": "sum(expected.output ** 2)",
            "physical_representation": "dense",
        },
        "notes": [
            "Forward values are checked exactly against tests.adapters.run_transformer_block.",
            "Packed Python attention and SwiGLU gradients are unpacked into semantic arrays.",
        ],
    }
    write_bundle("transformer_block", arrays, metadata)


def _write_model(source: dict[str, object]) -> None:
    generator = torch.Generator().manual_seed(341)
    vocab_size, context_length, d_model = 19, 6, 8
    num_layers, num_heads, d_ff, theta = 2, 2, 16, 10_000.0
    weights: dict[str, torch.Tensor] = {
        "token_embeddings.weight": _random_weight(generator, vocab_size, d_model),
        "ln_final.weight": 0.5 + torch.rand(d_model, generator=generator),
        "lm_head.weight": _random_weight(generator, vocab_size, d_model),
    }
    for index in range(num_layers):
        weights.update(
            {
                f"layers.{index}.{name}": tensor
                for name, tensor in _course_block_weights(generator, d_model, d_ff).items()
            }
        )
    token_ids = torch.tensor([[0, 3, 5, 7], [2, 4, 8, 11]], dtype=torch.int64)
    model = TransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        theta,
    )
    model.load_state_dict(weights)
    output = model(token_ids)
    adapter_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=weights,
        in_indices=token_ids,
    )
    torch.testing.assert_close(output, adapter_output, rtol=0, atol=0)

    named_parameters = dict(model.named_parameters())
    gradient_values = torch.autograd.grad(output.square().sum(), tuple(named_parameters.values()))
    native_gradients = dict(zip(named_parameters, gradient_values, strict=True))
    arrays = {
        "input.token_ids": token_ids.numpy(),
        "expected.output": output.detach().numpy(),
    }
    descriptors = {
        "input.token_ids": descriptor(
            "input",
            arrays["input.token_ids"],
            ["batch", "sequence"],
            zero_based_values=True,
        ),
        "expected.output": descriptor(
            "expected_output",
            arrays["expected.output"],
            ["batch", "sequence", "vocabulary"],
        ),
    }
    top_parameters = {
        "token_embedding": weights["token_embeddings.weight"],
        "final_norm": weights["ln_final.weight"],
        "lm_head": weights["lm_head.weight"],
    }
    top_gradients = {
        "token_embedding": native_gradients["token_embeddings.weight"],
        "final_norm": native_gradients["ln_final.weight"],
        "lm_head": native_gradients["lm_head.weight"],
    }
    _add_parameters(arrays, descriptors, top_parameters, top_gradients)
    for index in range(num_layers):
        prefix = f"layers.{index}."
        semantic_parameters = _semantic_block_parameters(weights, prefix=prefix)
        semantic_gradients = _semantic_block_gradients(
            native_gradients,
            prefix=prefix,
            d_model=d_model,
            d_ff=d_ff,
        )
        _add_parameters(
            arrays,
            descriptors,
            semantic_parameters,
            semantic_gradients,
            namespace=f"block_{index}",
        )
    metadata = {
        "contract_version": 1,
        "source": source,
        "operation": "run_transformer_lm",
        "producer": _producer(),
        "array_file": "transformer_lm.npz",
        "arrays": descriptors,
        "scalars": {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "theta": theta,
        },
        "tolerances": {"rtol": 2e-4, "atol": 2e-4, "equal_nan": False},
        "gradients": {
            "present": True,
            "objective": "sum(expected.output ** 2)",
            "physical_representation": "dense",
        },
        "notes": [
            "Forward values are checked exactly against tests.adapters.run_transformer_lm.",
            "The tiny fixture covers two independent blocks and a truncated-length input.",
            "Packed Python attention/SwiGLU gradients are stored as semantic arrays.",
        ],
    }
    write_bundle("transformer_lm", arrays, metadata)


def main() -> None:
    source = source_metadata()
    _write_block(source)
    _write_model(source)


if __name__ == "__main__":
    main()
