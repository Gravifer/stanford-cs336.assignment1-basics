"""Generate the self-contained linear parity bundle from the Python adapter surface."""

from __future__ import annotations

import io
import json
import subprocess
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from cs336_basics.nn.modules import Linear
from tests.adapters import run_linear


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "v1"
STEM = "linear"


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _descriptor(role: str, array: np.ndarray, axes: list[str]) -> dict[str, object]:
    return {
        "role": role,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "axes": axes,
        "physical_representation": "dense",
        "finiteness": "required",
    }


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an uncompressed NPZ with stable member order and timestamps."""
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_STORED
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue())


def main() -> None:
    source_commit = _git("rev-parse", "HEAD")
    source_timestamp = _git("show", "-s", "--format=%cI", "HEAD")
    working_tree_clean = not _git("status", "--porcelain")

    weights = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 8 - 0.5
    input_features = (torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10 - 1).requires_grad_()

    adapter_output = run_linear(4, 3, weights, input_features.detach())
    module = Linear(4, 3, dtype=torch.float32)
    with torch.no_grad():
        module.weight.copy_(weights)
    output = module(input_features)
    torch.testing.assert_close(output, adapter_output, rtol=0, atol=0)

    objective = output.square().sum()
    objective.backward()
    assert input_features.grad is not None
    assert module.weight.grad is not None

    arrays = {
        "input.x": input_features.detach().numpy(),
        "parameter.weight": weights.numpy(),
        "expected.output": output.detach().numpy(),
        "expected.gradient.input.x": input_features.grad.detach().numpy(),
        "expected.gradient.parameter.weight": module.weight.grad.detach().numpy(),
    }
    descriptors = {
        "input.x": _descriptor("input", arrays["input.x"], ["batch", "sequence", "input_feature"]),
        "parameter.weight": _descriptor(
            "parameter", arrays["parameter.weight"], ["output_feature", "input_feature"]
        ),
        "expected.output": _descriptor(
            "expected_output", arrays["expected.output"], ["batch", "sequence", "output_feature"]
        ),
        "expected.gradient.input.x": _descriptor(
            "expected_input_gradient",
            arrays["expected.gradient.input.x"],
            ["batch", "sequence", "input_feature"],
        ),
        "expected.gradient.parameter.weight": _descriptor(
            "expected_parameter_gradient",
            arrays["expected.gradient.parameter.weight"],
            ["output_feature", "input_feature"],
        ),
    }
    metadata = {
        "contract_version": 1,
        "source": {
            "git_commit": source_commit,
            "generated_at": source_timestamp,
            "working_tree_clean": working_tree_clean,
        },
        "operation": "run_linear",
        "producer": {
            "language": "Python",
            "runtime_version": version("cs336_basics"),
            "packages": {"numpy": np.__version__, "torch": torch.__version__},
        },
        "array_file": f"{STEM}.npz",
        "arrays": descriptors,
        "scalars": {"d_in": 4, "d_out": 3},
        "tolerances": {"rtol": 1e-6, "atol": 1e-6, "equal_nan": False},
        "gradients": {
            "present": True,
            "objective": "sum(expected.output ** 2)",
            "physical_representation": "dense",
        },
        "notes": [
            "Output is produced through tests.adapters.run_linear.",
            "Gradients use the same CS336 Linear module because load_state_dict intentionally severs the source tensor graph.",
        ],
    }

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_deterministic_npz(OUTPUT_DIRECTORY / f"{STEM}.npz", arrays)
    (OUTPUT_DIRECTORY / f"{STEM}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
