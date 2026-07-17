import argparse
from collections import Counter
from collections.abc import Callable
from typing import Any

import pytest
import torch

from scripts.benchmarks import attention as benchmark


def make_args(command: str) -> argparse.Namespace:
    return argparse.Namespace(
        command=command,
        batch_size=1,
        sequence_length=3,
        num_heads=2,
        head_dim=4,
        warmup=0,
        repeats=1,
        backward=False,
        compile_mode="default",
        positions="sequence",
    )


@pytest.mark.parametrize(
    ("command", "expected_per_execution"),
    [("rope", 8), ("mha", 12)],
)
def test_benchmark_reports_eager_and_compiled_rows(
    command: str,
    expected_per_execution: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark, "_compiled", lambda function, _mode: function)
    args = make_args(command)

    if command == "rope":
        results = benchmark._run_rope(args, torch.device("cpu"), torch.float32)
    else:
        results = benchmark._run_mha(args, torch.device("cpu"), torch.float32)

    assert Counter(result.execution for result in results) == {
        "eager": expected_per_execution,
        "compiled (default)": expected_per_execution,
    }


@pytest.mark.parametrize("command", ["rope", "mha"])
def test_benchmark_checks_compiled_callable_before_timing(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def corrupt_compiled(function: Callable[..., Any], mode: str) -> Callable[..., Any]:
        if mode == "none":
            return function

        def corrupted(*args: Any, **kwargs: Any) -> Any:
            output = function(*args, **kwargs)
            if isinstance(output, tuple):
                return tuple(tensor + 1 for tensor in output)
            return output + 1

        return corrupted

    monkeypatch.setattr(benchmark, "_compiled", corrupt_compiled)
    args = make_args(command)

    with pytest.raises(AssertionError):
        if command == "rope":
            benchmark._run_rope(args, torch.device("cpu"), torch.float32)
        else:
            benchmark._run_mha(args, torch.device("cpu"), torch.float32)
