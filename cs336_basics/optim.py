import math
from collections.abc import Callable

import torch
from torch.optim.optimizer import ParamsT


class SGD(torch.optim.Optimizer):
    def __init__(self, params: ParamsT, lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr: float = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t: int = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad  # Get the gradient of loss with respect to p.
                p.add_(grad, alpha=-lr / math.sqrt(t + 1))  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),  # Typical applications set to this,
        # but LLMs like LLaMA and GPT-3 are often trained with (0.9, 0.95)
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            α: float = group["lr"]  # Get the learning rate
            β1, β2 = group["betas"]
            λ: float = group["weight_decay"]
            ε: float = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                # State load/init
                if not state:
                    state["t"] = 1
                    state["m1"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["m2"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                t: int = state["t"]
                m: torch.Tensor = state["m1"]  # First moment m
                v: torch.Tensor = state["m2"]  # Second moment v

                g = p.grad
                step_size: float = α * math.sqrt(1 - β2**t) / (1 - β1**t)  # t starts from 1 so that this works

                p.mul_(1 - λ * α)  # Apply weight decay

                # Update moment estimates
                m.mul_(β1).add_(g, alpha=1 - β1)
                v.mul_(β2).addcmul_(g, g, value=1 - β2)

                denom = v.sqrt().add_(ε)

                # Update parameters with decoupled weight decay
                p.addcdiv_(m, denom, value=-step_size)

                state["m1"] = m
                state["m2"] = v
                state["t"] = t + 1

        return loss
