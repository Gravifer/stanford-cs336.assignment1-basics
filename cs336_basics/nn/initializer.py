"""Course initialization recipes for linear and embedding weights."""

import warnings

import torch
from jaxtyping import Float
from torch.nn.init import trunc_normal_


def starter_trunc_normal_for_linear_(
    tensor: Float[torch.Tensor, "*shape"],
    d_in: int,
    d_out: int,
    *,
    generator: torch.Generator | None = None,
) -> Float[torch.Tensor, "*shape"]:
    """
    Fills the input `tensor` with values drawn from a truncated normal distribution:

    .. math::
        𝒩︀(𝜇 = 0, 𝜎² = 2 / (𝑑_{in}+𝑑_{out}) ) truncated at [-3𝜎, 3𝜎]

    Intended for linear weights.

    Args:
        tensor (torch.Tensor): The tensor to be filled.
        d_in (int): The number of input features.
        d_out (int): The number of output features.
        generator (torch.Generator): an RNG to sample from (default: None)

    Returns:
        torch.Tensor: The filled tensor.
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op", stacklevel=2)
        return tensor
    std: float = (2.0 / (d_in + d_out)) ** 0.5
    trunc_normal_(tensor, 0.0, std, -3 * std, 3 * std, generator)
    return tensor


def starter_trunc_normal_for_embedding_(
    tensor: Float[torch.Tensor, "*shape"],
    *,
    generator: torch.Generator | None = None,
) -> Float[torch.Tensor, "*shape"]:
    """
    Fills the input `tensor` with values drawn from a truncated normal distribution:

    .. math::
        𝒩︀(𝜇 = 0, 𝜎² = 1 ) truncated at [-3, 3]

    Intended for embedding.

    Args:
        tensor (torch.Tensor): The tensor to be filled.
        generator (torch.Generator): an RNG to sample from (default: None)

    Returns:
        torch.Tensor: The filled tensor.
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op", stacklevel=2)
        return tensor
    trunc_normal_(tensor, 0.0, 1.0, -3.0, 3.0, generator)
    return tensor
