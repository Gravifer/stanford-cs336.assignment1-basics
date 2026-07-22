"""Generate self-contained SiLU and softmax parity bundles."""

from __future__ import annotations

from importlib.metadata import version

import numpy as np
import torch

from tests.adapters import run_silu, run_softmax

from fixture_tools import descriptor, source_metadata, write_bundle


def _producer() -> dict[str, object]:
    return {
        "language": "Python",
        "runtime_version": version("cs336_basics"),
        "packages": {"numpy": np.__version__, "torch": torch.__version__},
    }


def _gradient_bundle(
    *,
    stem: str,
    operation: str,
    source: dict[str, object],
    input_tensor: torch.Tensor,
    output: torch.Tensor,
    axes: list[str],
    scalars: dict[str, object],
    notes: list[str],
) -> None:
    objective = output.square().sum()
    (input_gradient,) = torch.autograd.grad(objective, (input_tensor,))
    arrays = {
        "input.x": input_tensor.detach().numpy(),
        "expected.output": output.detach().numpy(),
        "expected.gradient.input.x": input_gradient.detach().numpy(),
    }
    metadata = {
        "contract_version": 1,
        "source": source,
        "operation": operation,
        "producer": _producer(),
        "array_file": f"{stem}.npz",
        "arrays": {
            "input.x": descriptor("input", arrays["input.x"], axes),
            "expected.output": descriptor("expected_output", arrays["expected.output"], axes),
            "expected.gradient.input.x": descriptor(
                "expected_input_gradient", arrays["expected.gradient.input.x"], axes
            ),
        },
        "scalars": scalars,
        "tolerances": {"rtol": 1e-6, "atol": 1e-6, "equal_nan": False},
        "gradients": {
            "present": True,
            "objective": "sum(expected.output ** 2)",
            "physical_representation": "dense",
        },
        "notes": notes,
    }
    write_bundle(stem, arrays, metadata)


def main() -> None:
    source = source_metadata()

    silu_input = torch.tensor(
        [[-20.0, -2.0, -0.5, 0.0], [0.5, 2.0, 8.0, 20.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    _gradient_bundle(
        stem="silu",
        operation="run_silu",
        source=source,
        input_tensor=silu_input,
        output=run_silu(silu_input),
        axes=["row", "feature"],
        scalars={},
        notes=["Output and input gradient are produced through tests.adapters.run_silu."],
    )

    softmax_input = (
        torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 3
        + torch.tensor([[[100.0], [-50.0], [0.0]]], dtype=torch.float32)
    ).requires_grad_()
    _gradient_bundle(
        stem="softmax",
        operation="run_softmax",
        source=source,
        input_tensor=softmax_input,
        output=run_softmax(softmax_input, dim=1),
        axes=["batch", "choice", "feature"],
        scalars={"python_dim": 1, "axis": "choice"},
        notes=[
            "Output and input gradient are produced through tests.adapters.run_softmax.",
            "The large offsets exercise maximum-subtraction stability along the choice axis.",
        ],
    )


if __name__ == "__main__":
    main()
