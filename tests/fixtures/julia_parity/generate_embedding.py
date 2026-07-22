"""Generate the self-contained embedding parity bundle from the Python adapter surface."""

from __future__ import annotations

from importlib.metadata import version

import numpy as np
import torch

from cs336_basics.nn.modules import Embedding
from tests.adapters import run_embedding

from fixture_tools import descriptor, source_metadata, write_bundle


def main() -> None:
    source = source_metadata()
    weights = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10 - 1
    token_ids = torch.tensor([[0, 2, 2], [5, 1, 2]], dtype=torch.int64)

    adapter_output = run_embedding(6, 4, weights, token_ids)
    module = Embedding(6, 4, dtype=torch.float32)
    with torch.no_grad():
        module.weight.copy_(weights)
    output = module(token_ids)
    torch.testing.assert_close(output, adapter_output, rtol=0, atol=0)

    objective = output.square().sum()
    objective.backward()
    assert module.weight.grad is not None

    arrays = {
        "input.token_ids": token_ids.numpy(),
        "parameter.weight": weights.numpy(),
        "expected.output": output.detach().numpy(),
        "expected.gradient.parameter.weight": module.weight.grad.detach().numpy(),
    }
    descriptors = {
        "input.token_ids": descriptor(
            "input",
            arrays["input.token_ids"],
            ["batch", "sequence"],
            zero_based_values=True,
        ),
        "parameter.weight": descriptor(
            "parameter", arrays["parameter.weight"], ["vocabulary", "feature"]
        ),
        "expected.output": descriptor(
            "expected_output", arrays["expected.output"], ["batch", "sequence", "feature"]
        ),
        "expected.gradient.parameter.weight": descriptor(
            "expected_parameter_gradient",
            arrays["expected.gradient.parameter.weight"],
            ["vocabulary", "feature"],
        ),
    }
    metadata = {
        "contract_version": 1,
        "source": source,
        "operation": "run_embedding",
        "producer": {
            "language": "Python",
            "runtime_version": version("cs336_basics"),
            "packages": {"numpy": np.__version__, "torch": torch.__version__},
        },
        "array_file": "embedding.npz",
        "arrays": descriptors,
        "scalars": {"vocab_size": 6, "d_model": 4},
        "tolerances": {"rtol": 1e-6, "atol": 1e-6, "equal_nan": False},
        "gradients": {
            "present": True,
            "objective": "sum(expected.output ** 2)",
            "physical_representation": "dense",
        },
        "notes": [
            "Output is produced through tests.adapters.run_embedding.",
            "Repeated token ID 2 verifies gradient accumulation; the stored weight gradient is dense.",
        ],
    }
    write_bundle("embedding", arrays, metadata)


if __name__ == "__main__":
    main()
