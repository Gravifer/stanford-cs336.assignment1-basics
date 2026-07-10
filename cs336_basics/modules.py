import einx
import torch
from torch import dtype, nn

from .initializer import starter_trunc_normal_for_embedding_, starter_trunc_normal_for_linear_  # noqa: F401


class Linear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    # @property
    # def bias(self) -> None:
    #     return None
    #     # raise AttributeError("This module has no bias.")

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: dtype | None = None
    ):
        # factory_kwargs = {"device": device, "dtype": dtype} # ! in modern days a typing pain
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.register_parameter("bias", None)
        self.reset_parameters()  # Object of type `Tensor` is not callable  ty:(call-non-callable)

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        starter_trunc_normal_for_linear_(self.weight, self.in_features, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input."""
        # return torch.nn.functional.linear(x, self.weight, self.bias)
        return einx.dot("... d_in, d_out d_in->... d_out", x, self.weight)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"in_features={self.in_features}, out_features={self.out_features}, NO bias"
