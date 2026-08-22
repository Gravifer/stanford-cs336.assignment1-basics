"""Tests for the class-last negative log-likelihood loss."""

from typing import Literal

import pytest
import torch
import torch.nn.functional as torch_f

from cs336_basics.nn.functional import nll_loss


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_nll_loss_matches_torch_values_and_gradients(
    weighted: bool,
    reduction: Literal["none", "sum", "mean"],
) -> None:
    input = torch.randn(2, 3, 5, dtype=torch.float64)
    target = torch.tensor([[2, -100, 0], [4, 1, 3]])
    weight = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5], dtype=torch.float64) if weighted else None

    actual_input = input.clone().requires_grad_()
    expected_input = input.clone().requires_grad_()
    actual = nll_loss(
        torch_f.log_softmax(actual_input, dim=-1),
        target,
        weight=weight,
        ignore_index=-100,
        reduction=reduction,
    )
    expected = torch_f.nll_loss(
        torch_f.log_softmax(expected_input, dim=-1).movedim(-1, 1),
        target,
        weight=weight,
        ignore_index=-100,
        reduction=reduction,
    )

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual_input.grad, expected_input.grad)


@pytest.mark.parametrize("invalid_target", [-1, 5])
def test_nll_loss_rejects_nonignored_out_of_range_targets(invalid_target: int) -> None:
    logprobs = torch_f.log_softmax(torch.randn(2, 3, 5), dim=-1)
    target = torch.tensor([[2, invalid_target, 0], [4, 1, 3]])

    with pytest.raises(RuntimeError, match="out of bounds"):
        nll_loss(logprobs, target)


def test_nll_loss_ignored_dummy_gather_does_not_leak_nonfinite_values() -> None:
    logprobs = torch.tensor([[float("nan"), -1.0], [-2.0, -3.0]])
    target = torch.tensor([-100, 1])

    actual = nll_loss(logprobs, target, ignore_index=-100, reduction="none")

    torch.testing.assert_close(actual, torch.tensor([0.0, 3.0]))
