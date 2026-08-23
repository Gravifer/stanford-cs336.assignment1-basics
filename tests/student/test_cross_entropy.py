"""Tests for class-last cross entropy with hard and soft targets."""

from typing import Literal

import pytest
import torch
import torch.nn.functional as torch_f

from cs336_basics.nn.functional import cross_entropy


@pytest.mark.parametrize("target_kind", ["indices", "probabilities"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.2])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_cross_entropy_matches_torch_values_and_gradients(
    target_kind: Literal["indices", "probabilities"],
    weighted: bool,
    label_smoothing: float,
    reduction: Literal["none", "sum", "mean"],
) -> None:
    logits = torch.randn(2, 3, 5, dtype=torch.float64)
    weight = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5], dtype=torch.float64) if weighted else None
    if target_kind == "indices":
        target = torch.tensor([[2, -100, 0], [4, 1, 3]])
        torch_target = target
        ignore_index = -100
    else:
        target = torch.softmax(torch.randn(2, 3, 5, dtype=torch.float64), dim=-1)
        torch_target = target.movedim(-1, 1)
        ignore_index = None

    actual_logits = logits.clone().requires_grad_()
    expected_logits = logits.clone().requires_grad_()
    actual = cross_entropy(
        actual_logits,
        target,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    expected = torch_f.cross_entropy(
        expected_logits.movedim(-1, 1),
        torch_target,
        weight=weight,
        ignore_index=-100,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual_logits.grad, expected_logits.grad)


def test_probability_target_mean_uses_prediction_count() -> None:
    logits = torch.randn(2, 3, 5, dtype=torch.float64)
    target = 2 * torch.softmax(torch.randn(2, 3, 5, dtype=torch.float64), dim=-1)

    actual = cross_entropy(logits, target, reduction="mean")
    expected = torch_f.cross_entropy(logits.movedim(-1, 1), target.movedim(-1, 1), reduction="mean")

    torch.testing.assert_close(actual, expected)


def test_probability_targets_reject_ignore_index() -> None:
    logits = torch.randn(2, 3, 5)
    target = torch.softmax(torch.randn_like(logits), dim=-1)

    with pytest.raises(ValueError, match="ignore_index is not applicable"):
        cross_entropy(logits, target, ignore_index=-100)


def test_cross_entropy_rejects_target_dtype_incompatible_with_shape() -> None:
    logits = torch.randn(2, 3, 5)

    with pytest.raises(TypeError, match="class-probability targets must have a floating-point dtype"):
        cross_entropy(logits, torch.zeros_like(logits, dtype=torch.long))
    with pytest.raises(TypeError, match="class-index targets must have an integer dtype"):
        cross_entropy(logits, torch.zeros(2, 3))
