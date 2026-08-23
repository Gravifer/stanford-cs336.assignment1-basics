"""Tests for optimizer state and mutation behavior."""

from collections.abc import Callable

import pytest
import torch

from cs336_basics.optim import AdamW, SGD


@pytest.mark.parametrize(
    "make_optimizer",
    [
        lambda parameter: SGD([parameter], lr=0.1),
        lambda parameter: AdamW([parameter], lr=0.1),
    ],
)
def test_optimizer_updates_participate_in_version_tracking(
    make_optimizer: Callable[[torch.nn.Parameter], torch.optim.Optimizer],
) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = make_optimizer(parameter)
    loss = parameter.square().sum()
    parameter.grad = torch.ones_like(parameter)

    optimizer.step()

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@pytest.mark.parametrize(
    "make_optimizer",
    [
        lambda parameter: SGD([parameter], lr=0.1),
        lambda parameter: AdamW([parameter], lr=0.1),
    ],
)
def test_optimizer_closure_runs_with_gradients_enabled(
    make_optimizer: Callable[[torch.nn.Parameter], torch.optim.Optimizer],
) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = make_optimizer(parameter)
    parameter.grad = torch.ones_like(parameter)

    def closure() -> float:
        assert torch.is_grad_enabled()
        return parameter.square().sum().item()

    loss = optimizer.step(closure)

    assert loss is not None
    assert loss == pytest.approx(1.0)


def test_adamw_initializes_and_reuses_moment_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = AdamW([parameter])
    zeros_like = torch.zeros_like
    allocations = 0

    def counted_zeros_like(*args, **kwargs):
        nonlocal allocations
        allocations += 1
        return zeros_like(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros_like", counted_zeros_like)

    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    first_moment = optimizer.state[parameter]["m1"]
    second_moment = optimizer.state[parameter]["m2"]

    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert allocations == 2
    assert optimizer.state[parameter]["m1"] is first_moment
    assert optimizer.state[parameter]["m2"] is second_moment
