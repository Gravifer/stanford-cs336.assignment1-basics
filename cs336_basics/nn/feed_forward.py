from collections.abc import Mapping
from typing import Any

import einx
import torch
from jaxtyping import Float, Shaped
from torch import dtype, nn

from cs336_basics.nn import functional as F
from cs336_basics.nn import initializer as init

from .modules import Linear, ModelVec


class SwiGLU_delegate(nn.Module):
    """SwiGLU module made of :cls:`Linear`s and functional SiLU.

    Given a value tensor :math:`𝑥`,
    compute the output as follows:

    .. math::
        FFN(𝑥) = SwiGLU(𝑥,𝑊_1,𝑊_2,𝑊_3) = 𝑊_2(SiLU(𝑊_1 𝑥)⊙𝑊_3 𝑥)

    where :math:`SiLU` is the Sigmoid Linear Unit activation function.
    """

    __constants__ = ["d_ff", "d_model"]
    d_ff: int
    d_model: int
    value_linear: Float[Linear, "{self.d_ff} {self.d_model}"]
    gate_linear: Float[Linear, "{self.d_ff} {self.d_model}"]
    out_linear: Float[Linear, "{self.d_model} {self.d_ff}"]

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: dtype | None = None,
    ):
        super().__init__()
        # if no d_ff supplied use 8/3 * d_model rounded to the nearest multiple of 64
        d_ff = d_ff or (int(8 * d_model / 3.0) >> 6) << 6
        self.d_ff = d_ff
        self.d_model = d_model
        self.value_linear = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.gate_linear = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.out_linear = Linear(d_ff, d_model, device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        """Reset the parameters of the module."""
        self.value_linear.reset_parameters()
        self.gate_linear.reset_parameters()
        self.out_linear.reset_parameters()

    def forward(self, x: Shaped[ModelVec, "*mapped"]) -> Shaped[ModelVec, "*mapped"]:
        """Apply the SwiGLU transformation to the input."""
        type HiddenVec = Float[torch.Tensor, "{self.d_ff}"]  # noqa: F821
        value: Shaped[HiddenVec, "*mapped"] = self.value_linear(x)
        gate: Shaped[HiddenVec, "*mapped"] = self.gate_linear(x)
        gated: Shaped[HiddenVec, "*mapped"] = F.swiglu(value, gate)
        return self.out_linear(gated)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}"

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        """Load the state dict into the module.

        We need to load into the Linear members
        """
        out_weight = state_dict.get("out_weight")
        if "in_weight" in state_dict:
            if "value_weight" in state_dict or "gate_weight" in state_dict:
                raise ValueError("state_dict contains both 'in_weight' and 'value_weight'/'gate_weight'; cannot load")
            in_weight = state_dict["in_weight"]
            # unpack in_weight into value_weight and gate_weight; we'll let errors propagate, if any
            value_weight, gate_weight = einx.id(
                "(v + g) d_model -> v d_model, g d_model", in_weight, v=self.d_ff, g=self.d_ff
            )
        else:
            value_weight = state_dict.get("value_weight")
            gate_weight = state_dict.get("gate_weight")
        self.value_linear.load_state_dict({"weight": value_weight}, strict, assign)
        self.gate_linear.load_state_dict({"weight": gate_weight}, strict, assign)
        self.out_linear.load_state_dict({"weight": out_weight}, strict, assign)


class SwiGLU_own_weights(nn.Module):
    """SwiGLU feed-forward layer.

    Given a value tensor :math:`𝑥`,
    compute the output as follows:

    .. math::
        FFN(𝑥) = SwiGLU(𝑥,𝑊_1,𝑊_2,𝑊_3) = 𝑊_2(SiLU(𝑊_1 𝑥)⊙𝑊_3 𝑥)

    where :math:`SiLU` is the Sigmoid Linear Unit activation function.
    """

    __constants__ = ["d_ff", "d_model"]
    d_ff: int
    d_model: int
    value_weight: Float[torch.Tensor, "{self.d_ff} {self.d_model}"]
    gate_weight: Float[torch.Tensor, "{self.d_ff} {self.d_model}"]
    out_weight: Float[torch.Tensor, "{self.d_model} {self.d_ff}"]

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: dtype | None = None,
    ):
        super().__init__()
        # if no d_ff supplied use 8/3 * d_model rounded to the nearest multiple of 64
        d_ff = d_ff or (int(8 * d_model / 3.0) >> 6) << 6
        self.d_ff = d_ff
        self.d_model = d_model
        self.value_weight = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.gate_weight = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.out_weight = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))

    def reset_parameters(self) -> None:
        """Reset the parameters of the module."""
        init.starter_trunc_normal_for_linear_(self.value_weight, self.d_model, self.d_ff)
        init.starter_trunc_normal_for_linear_(self.gate_weight, self.d_model, self.d_ff)
        init.starter_trunc_normal_for_linear_(self.out_weight, self.d_ff, self.d_model)

    def forward(
        self, x: Float[torch.Tensor, "*mapped {self.d_model}"]
    ) -> Float[torch.Tensor, "*mapped {self.d_model}"]:
        """Apply the SwiGLU transformation to the input."""
        value = F.linear(x, self.value_weight)
        gate = F.linear(x, self.gate_weight)
        return F.linear(F.swiglu(value, gate), self.out_weight)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}; weights owned"

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        """Load the state dict into the module.

        We need to load into the Linear members
        """
        out_weight = state_dict.get("out_weight")
        if "in_weight" in state_dict:
            if "value_weight" in state_dict or "gate_weight" in state_dict:
                raise ValueError("state_dict contains both 'in_weight' and 'value_weight'/'gate_weight'; cannot load")
            in_weight = state_dict["in_weight"]
            # unpack in_weight into value_weight and gate_weight; we'll let errors propagate, if any
            value_weight, gate_weight = einx.id(
                "(v + g) d_model -> v d_model, g d_model", in_weight, v=self.d_ff, g=self.d_ff
            )
        else:
            value_weight = state_dict.get("value_weight")
            gate_weight = state_dict.get("gate_weight")
        super().load_state_dict(
            {"value_weight": value_weight, "gate_weight": gate_weight, "out_weight": out_weight}, strict, assign
        )


class SwiGLU_packed_input(nn.Module):
    """SwiGLU feed-forward layer.

    Given a value tensor :math:`𝑥`,
    compute the output as follows:

    .. math::
        FFN(𝑥) = SwiGLU(𝑥,𝑊_1,𝑊_2,𝑊_3) = 𝑊_2(SiLU(𝑊_1 𝑥)⊙𝑊_3 𝑥)

    where :math:`SiLU` is the Sigmoid Linear Unit activation function.
    """

    __constants__ = ["d_ff", "d_model"]
    d_ff: int
    d_model: int
    in_weight: Float[torch.Tensor, "2*{self.d_ff} {self.d_model}"]
    out_weight: Float[torch.Tensor, "{self.d_model} {self.d_ff}"]

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: dtype | None = None,
    ):
        super().__init__()
        # if no d_ff supplied use 8/3 * d_model rounded to the nearest multiple of 64
        d_ff = d_ff or (int(8 * d_model / 3.0) >> 6) << 6
        self.d_ff = d_ff
        self.d_model = d_model
        self.in_weight = nn.Parameter(torch.empty((2 * d_ff, d_model), device=device, dtype=dtype))
        self.out_weight = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))

    def reset_parameters(self) -> None:
        """Reset the parameters of the module."""
        init.starter_trunc_normal_for_linear_(self.in_weight, self.d_model, self.d_ff)
        init.starter_trunc_normal_for_linear_(self.out_weight, self.d_ff, self.d_model)

    def forward(
        self, x: Float[torch.Tensor, "*mapped {self.d_model}"]
    ) -> Float[torch.Tensor, "*mapped {self.d_model}"]:
        """Apply the SwiGLU transformation to the input."""
        vg = F.linear(x, self.in_weight)
        value, gate = einx.id("mapped... (v + g) -> mapped... v, mapped... g", vg, v=self.d_ff, g=self.d_ff)
        return F.linear(F.swiglu(value, gate), self.out_weight)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}; value and gate weights packed"

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        """Load the state dict into the module.

        We need to pack the value and gate weights
        """
        out_weight = state_dict.get("out_weight")
        if "in_weight" not in state_dict:
            value_weight = state_dict.get("value_weight")
            gate_weight = state_dict.get("gate_weight")
            # pack value_weight and gate_weight into in_weight; we'll let errors propagate, if any
            in_weight = einx.id(
                "v d_model, g d_model -> (v + g) d_model", value_weight, gate_weight, v=self.d_ff, g=self.d_ff
            )
        else:
            if "value_weight" in state_dict or "gate_weight" in state_dict:
                raise ValueError("state_dict contains both 'in_weight' and 'value_weight'/'gate_weight'; cannot load")
            in_weight = state_dict["in_weight"]
        super().load_state_dict({"in_weight": in_weight, "out_weight": out_weight}, strict, assign)


SwiGLU: type = SwiGLU_packed_input
