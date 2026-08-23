"""Tests for optimizer state and mutation behavior."""

from collections.abc import Callable

import pytest
import torch

from cs336_basics.optim import AdamW, SGD, clip_grad_norm_, cosine_annealing_learning_rate


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


def test_clip_grad_norm_accepts_one_shot_iterables_and_returns_original_norm() -> None:
    parameters = (
        torch.nn.Parameter(torch.zeros(2)),
        torch.nn.Parameter(torch.zeros(2)),
    )
    parameters[0].grad = torch.tensor([3.0, 4.0])
    parameters[1].grad = torch.tensor([0.0, 12.0])

    total_norm = clip_grad_norm_((parameter for parameter in parameters), 6.5)

    assert isinstance(total_norm, torch.Tensor)
    assert total_norm == pytest.approx(13.0)
    first_gradient = parameters[0].grad
    second_gradient = parameters[1].grad
    assert first_gradient is not None
    assert second_gradient is not None
    clipped_norm = torch.linalg.vector_norm(torch.stack((first_gradient.norm(), second_gradient.norm())))
    assert clipped_norm == pytest.approx(6.5, rel=1e-5)


def test_cosine_annealing_learning_rate_is_public_and_covers_each_phase() -> None:
    schedule = cosine_annealing_learning_rate

    assert schedule(0, 1.0, 0.1, 2, 6) == pytest.approx(0.0)
    assert schedule(2, 1.0, 0.1, 2, 6) == pytest.approx(1.0)
    assert schedule(t=4, alpha_max=1.0, alpha_min=0.1, T_warmup=2, T_annealing=6) == pytest.approx(0.55)
    assert schedule(6, 1.0, 0.1, 2, 6) == pytest.approx(0.1)
