"""Tests for distribution-aware categorical KL divergence."""

from typing import Literal

import pytest
import torch
import torch.nn.functional as torch_f

from cs336_basics.nn.functional import kl_div


@pytest.mark.parametrize("log_target", [False, True])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean", "batchmean"])
def test_kl_div_matches_reduced_torch_values_and_gradients(
    log_target: bool,
    reduction: Literal["none", "sum", "mean", "batchmean"],
) -> None:
    logprobs = torch.log_softmax(torch.randn(2, 3, 5, dtype=torch.float64), dim=-1)
    probabilities = torch.softmax(torch.randn(2, 3, 5, dtype=torch.float64), dim=-1)
    target = probabilities.log() if log_target else probabilities

    actual_logprobs = logprobs.detach().clone().requires_grad_()
    expected_logprobs = logprobs.detach().clone().requires_grad_()
    actual_target = target.detach().clone().requires_grad_()
    expected_target = target.detach().clone().requires_grad_()

    actual = kl_div(
        actual_logprobs,
        actual_target,
        reduction=reduction,
        log_target=log_target,
    )
    expected_sites = torch_f.kl_div(
        expected_logprobs,
        expected_target,
        reduction="none",
        log_target=log_target,
    ).sum(dim=-1)
    if reduction == "none":
        expected = expected_sites
    elif reduction == "sum":
        expected = expected_sites.sum()
    elif reduction == "mean":
        expected = expected_sites.mean()
    else:
        expected = expected_sites.sum() / logprobs.shape[0]

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual_logprobs.grad, expected_logprobs.grad)
    torch.testing.assert_close(actual_target.grad, expected_target.grad)


def test_kl_div_probability_target_handles_zero_mass() -> None:
    logprobs = torch.log_softmax(torch.randn(2, 3, dtype=torch.float64), dim=-1)
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.25, 0.75]], dtype=torch.float64)

    actual = kl_div(logprobs, target, reduction="none")
    expected = torch_f.kl_div(logprobs, target, reduction="none").sum(dim=-1)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


def test_kl_div_mean_defaults_to_prediction_site_mean() -> None:
    logprobs = torch.log_softmax(torch.randn(2, 3, 5), dim=-1)
    target = torch.softmax(torch.randn_like(logprobs), dim=-1)

    torch.testing.assert_close(kl_div(logprobs, target), kl_div(logprobs, target, reduction="mean"))


def test_kl_div_single_distribution_has_no_batchmean() -> None:
    logprobs = torch.log_softmax(torch.randn(5), dim=-1)
    target = torch.softmax(torch.randn_like(logprobs), dim=-1)

    assert kl_div(logprobs, target, reduction="none").shape == ()
    with pytest.raises(ValueError, match="requires a leading batch axis"):
        kl_div(logprobs, target, reduction="batchmean")


def test_kl_div_rejects_incompatible_shapes_and_reductions() -> None:
    logprobs = torch.log_softmax(torch.randn(2, 3, 5), dim=-1)

    with pytest.raises(ValueError, match="target shape must equal logprobs shape"):
        kl_div(logprobs, torch.rand(2, 5))
    with pytest.raises(ValueError, match="unsupported reduction"):
        kl_div(logprobs, torch.rand_like(logprobs), reduction="median")  # ty: ignore[no-matching-overload]
