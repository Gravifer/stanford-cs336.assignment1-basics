"""Small deterministic writer shared by maintainer-only parity producers."""

from __future__ import annotations

import io
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "v1"


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def source_metadata() -> dict[str, object]:
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "generated_at": _git("show", "-s", "--format=%cI", "HEAD"),
        "working_tree_clean": not _git("status", "--porcelain"),
    }


def descriptor(
    role: str,
    array: np.ndarray,
    axes: list[str],
    *,
    zero_based_values: bool | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "role": role,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "axes": axes,
        "physical_representation": "dense",
        "finiteness": "required" if np.issubdtype(array.dtype, np.floating) else "not_applicable",
    }
    if zero_based_values is not None:
        result["zero_based_values"] = zero_based_values
    return result


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


def write_bundle(stem: str, arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_deterministic_npz(OUTPUT_DIRECTORY / f"{stem}.npz", arrays)
    (OUTPUT_DIRECTORY / f"{stem}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
