"""Generate self-contained RMSNorm and SwiGLU parity bundles."""

from __future__ import annotations

from importlib.metadata import version

import numpy as np
import torch

from cs336_basics.nn import functional as functional
from cs336_basics.nn.feed_forward import SwiGLU
from tests.adapters import run_rmsnorm, run_swiglu

from fixture_tools import descriptor, source_metadata, write_bundle


def _producer() -> dict[str, object]:
    return {
        "language": "Python",
        "runtime_version": version("cs336_basics"),
        "packages": {"numpy": np.__version__, "torch": torch.__version__},
    }


def _rmsnorm(source: dict[str, object]) -> None:
    eps = 1e-5
    input_tensor = (torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 7 - 1).requires_grad_()
    weight = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32, requires_grad=True)
    output = functional.rms_norm(input_tensor, (4,), weight, eps)
    torch.testing.assert_close(output, run_rmsnorm(4, eps, weight.detach(), input_tensor.detach()), rtol=0, atol=0)
    input_gradient, weight_gradient = torch.autograd.grad(output.square().sum(), (input_tensor, weight))

    arrays = {
        "input.x": input_tensor.detach().numpy(),
        "parameter.weight": weight.detach().numpy(),
        "expected.output": output.detach().numpy(),
        "expected.gradient.input.x": input_gradient.detach().numpy(),
        "expected.gradient.parameter.weight": weight_gradient.detach().numpy(),
    }
    metadata = {
        "contract_version": 1,
        "source": source,
        "operation": "run_rmsnorm",
        "producer": _producer(),
        "array_file": "rmsnorm.npz",
        "arrays": {
            "input.x": descriptor("input", arrays["input.x"], ["batch", "sequence", "feature"]),
            "parameter.weight": descriptor("parameter", arrays["parameter.weight"], ["feature"]),
            "expected.output": descriptor(
                "expected_output", arrays["expected.output"], ["batch", "sequence", "feature"]
            ),
            "expected.gradient.input.x": descriptor(
                "expected_input_gradient",
                arrays["expected.gradient.input.x"],
                ["batch", "sequence", "feature"],
            ),
            "expected.gradient.parameter.weight": descriptor(
                "expected_parameter_gradient",
                arrays["expected.gradient.parameter.weight"],
                ["feature"],
            ),
        },
        "scalars": {"d_model": 4, "eps": eps},
        "tolerances": {"rtol": 1e-5, "atol": 1e-6, "equal_nan": False},
        "gradients": {
            "present": True,
            "objective": "sum(expected.output ** 2)",
            "physical_representation": "dense",
        },
        "notes": ["Output is verified through tests.adapters.run_rmsnorm; gradients use explicit CS336 arguments."],
    }
    write_bundle("rmsnorm", arrays, metadata)


def _swiglu(source: dict[str, object]) -> None:
    d_model = 4
    d_ff = 6
    input_tensor = (torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 9 - 1).requires_grad_()
    w1 = (torch.arange(24, dtype=torch.float32).reshape(6, 4) / 20 - 0.5).requires_grad_()
    w2 = (torch.arange(24, dtype=torch.float32).reshape(4, 6) / 18 - 0.4).requires_grad_()
    w3 = (torch.arange(24, dtype=torch.float32).flip(0).reshape(6, 4) / 21 - 0.6).requires_grad_()

    adapter_output = run_swiglu(
        d_model,
        d_ff,
        w1.detach(),
        w2.detach(),
        w3.detach(),
        input_tensor.detach(),
    )
    module = SwiGLU(d_model, d_ff, dtype=torch.float32)
    module.load_state_dict(
        {"w1.weight": w1.detach(), "w2.weight": w2.detach(), "w3.weight": w3.detach()}
    )
    output = module(input_tensor)
    torch.testing.assert_close(output, adapter_output, rtol=0, atol=0)
    input_gradient, packed_input_gradient, output_weight_gradient = torch.autograd.grad(
        output.square().sum(),
        (input_tensor, module.in_weight, module.out_weight),
    )
    value_weight_gradient, gate_weight_gradient = packed_input_gradient.split(d_ff)

    arrays = {
        "input.x": input_tensor.detach().numpy(),
        "parameter.w1": w1.detach().numpy(),
        "parameter.w2": w2.detach().numpy(),
        "parameter.w3": w3.detach().numpy(),
        "expected.output": output.detach().numpy(),
        "expected.gradient.input.x": input_gradient.detach().numpy(),
        "expected.gradient.parameter.w1": gate_weight_gradient.detach().numpy(),
        "expected.gradient.parameter.w2": output_weight_gradient.detach().numpy(),
        "expected.gradient.parameter.w3": value_weight_gradient.detach().numpy(),
    }
    arrays_metadata: dict[str, dict[str, object]] = {
        "input.x": descriptor("input", arrays["input.x"], ["batch", "sequence", "model_feature"]),
        "parameter.w1": descriptor(
            "parameter", arrays["parameter.w1"], ["hidden_feature", "model_feature"]
        ),
        "parameter.w2": descriptor(
            "parameter", arrays["parameter.w2"], ["model_feature", "hidden_feature"]
        ),
        "parameter.w3": descriptor(
            "parameter", arrays["parameter.w3"], ["hidden_feature", "model_feature"]
        ),
        "expected.output": descriptor(
            "expected_output", arrays["expected.output"], ["batch", "sequence", "model_feature"]
        ),
        "expected.gradient.input.x": descriptor(
            "expected_input_gradient",
            arrays["expected.gradient.input.x"],
            ["batch", "sequence", "model_feature"],
        ),
    }
    for name, axes in (
        ("w1", ["hidden_feature", "model_feature"]),
        ("w2", ["model_feature", "hidden_feature"]),
        ("w3", ["hidden_feature", "model_feature"]),
    ):
        key = f"expected.gradient.parameter.{name}"
        arrays_metadata[key] = descriptor("expected_parameter_gradient", arrays[key], axes)

    metadata = {
        "contract_version": 1,
        "source": source,
        "operation": "run_swiglu",
        "producer": _producer(),
        "array_file": "swiglu.npz",
        "arrays": arrays_metadata,
        "scalars": {"d_model": d_model, "d_ff": d_ff},
        "tolerances": {"rtol": 1e-5, "atol": 1e-6, "equal_nan": False},
        "gradients": {
            "present": True,
            "objective": "sum(expected.output ** 2)",
            "physical_representation": "dense",
        },
        "notes": [
            "Output is verified through tests.adapters.run_swiglu.",
            "The selected Python module uses one packed value/gate GEMM; its gradient is unpacked into semantic w3/w1 arrays.",
            "Julia keeps semantic weights explicit, so contraction-order roundoff is covered by the declared tolerance.",
        ],
    }
    write_bundle("swiglu", arrays, metadata)


def main() -> None:
    source = source_metadata()
    _rmsnorm(source)
    _swiglu(source)


if __name__ == "__main__":
    main()
