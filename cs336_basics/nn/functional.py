import warnings
from typing import Literal

import einx
import torch
from jaxtyping import Float, Int
from torch import sigmoid  # we're re-exporting it because the assignment explicitly allows it


def glu(
    value: Float[torch.Tensor, "*shape"],
    gate: Float[torch.Tensor, "*shape"],
) -> Float[torch.Tensor, "*shape"]:
    """Gated linear unit.

    Note that we expect unpacked input; with packed input, pass views.
    """
    if value.shape != gate.shape:
        raise ValueError("Value and gate tensors must have the same shape for GLU.")
    return value * sigmoid(gate)


def gelu(
    input: Float[torch.Tensor, "*shape"],
    approximate: Literal["none", "tanh"] = "tanh",
) -> Float[torch.Tensor, "*shape"]:
    """GELU activation function.

    Args:
        input: Input tensor of shape (*mapped, *dims).
        approximate: Whether to use the approximate GELU implementation.

    Returns:
        Output tensor of shape (*mapped, *dims).
    """
    if approximate == "none":
        return 0.5 * input * (1 + torch.erf(input / (2**0.5)))
    elif approximate == "tanh":
        return 0.5 * input * (1 + torch.tanh(input * 0.7978845608 * (1 + 0.044715 * input * input)))
    else:
        raise NotImplementedError(f"Unknown approximate mode: {approximate}")


def softmax(
    input: Float[torch.Tensor, "*shape"],
    dim: int | None = None,
    _stacklevel: int = 3,
    dtype: torch.dtype | None = None,
) -> Float[torch.Tensor, "*shape"]:
    """Softmax activation function.

    Takes two parameters (so that the signature matches torch.nn.functional.softmax):
        a tensor and a dimension 𝑖, and apply softmax to the 𝑖-th dimension of the input tensor.
    The output tensor should have the same shape as the input tensor,
        but its 𝑖-th dimension will now have a normalized probability distribution.
    Use the trick of subtracting the maximum value in the 𝑖-th dimension
        from all elements of the 𝑖-th dimension to avoid numerical stability issues.
    """
    type InputShaped   = Float[torch.Tensor, "*shape"]
    type ReducedShaped = Float[torch.Tensor, "*reduced"]
    # type KeptShaped  = Float[torch.Tensor, "*kept"]
    # derive einx strings
    dim: int = (dim if dim is not None else -1) % input.ndim
    axes_map: dict[str, int] = {(f"n{i}"): d                        for i, d in enumerate(input.shape)}
    _shape:    str = " ".join(   f"n{i}"                            for i    in range(input.ndim))
    _targeted: str = " ".join(   f"n{i}" if i != dim else f"[n{i}]" for i    in range(input.ndim))
    _kept:     str = " ".join(   f"n{i}" if i != dim else "()"      for i    in range(input.ndim))
    _reduced:  str = " ".join(   f"n{i}"                            for i    in range(input.ndim) if i != dim)
    # return einx.softmax(_shape, input, **axes_map) # * arguably cheating
    # max_kept: KeptShaped = torch.max(input, dim=dim, keepdim=True).values
    # max_kept: KeptShaped = einx.max(f"{_targeted} -> {_kept}", input, **axes_map)
    max_reduced: ReducedShaped = einx.max(f"{_targeted} -> {_reduced}", input, **axes_map)
    # exp_input: InputShaped = torch.exp(input - max_kept)
    # exp_input: InputShaped = einx.subtract(f"{_shape}, {_kept} -> {_shape}", input, max_kept, **axes_map).exp()
    exp_input: InputShaped = einx.subtract(f"{_shape}, {_reduced} -> {_shape}", input, max_reduced, **axes_map).exp()
    # sum_exp: KeptShaped = torch.sum(exp_input, dim=dim, keepdim=True)
    # sum_exp: KeptShaped = einx.sum(f"{_targeted} -> {_kept}", exp_input, **axes_map)
    sum_exp: ReducedShaped = einx.sum(f"{_targeted} -> {_reduced}", exp_input, **axes_map)
    # return exp_input / sum_exp
    # return einx.divide(f"{_shape}, {_kept} -> {_shape}", exp_input, sum_exp, **axes_map)
    return einx.divide(f"{_shape}, {_reduced} -> {_shape}", exp_input, sum_exp, **axes_map)


def linear(
    input: Float[torch.Tensor, "*mapped in_features"],
    weight: Float[torch.Tensor, "out_features in_features"],
) -> Float[torch.Tensor, "*mapped out_features"]:
    """Linear transformation without bias.

    Args:
        input: Input tensor of shape (..., in_features).
        weight: Weight tensor of shape (out_features, in_features).

    Returns:
        Output tensor of shape (..., out_features).
    """
    return einx.dot("mapped... d_in, d_out d_in -> mapped... d_out", input, weight)


def silu(
    input: Float[torch.Tensor, "*shape"],
) -> Float[torch.Tensor, "*shape"]:
    """SiLU activation function."""
    return input * sigmoid(input)


def swiglu(
    value: Float[torch.Tensor, "*shape"],
    gate: Float[torch.Tensor, "*shape"],
) -> Float[torch.Tensor, "*shape"]:
    """Swish GLU.

    Note that we expect unpacked input; with packed input, pass views.
    """
    if value.shape != gate.shape:
        raise ValueError("Value and gate tensors must have the same shape for SwiGLU.")
    return value * silu(gate)


def embedding(
    input: Int[torch.Tensor, "*batch sequence_length"],
    weight: Float[torch.Tensor, "num_embeddings embedding_dim"],
) -> Float[torch.Tensor, "*batch sequence_length embedding_dim"]:
    """Embedding lookup.

    Args:
        input: LongTensor of shape (*batch, sequence_length) containing indices.
        weight: Weight tensor of shape (num_embeddings, embedding_dim).

    Returns:
        Output tensor of shape (*batch, sequence_length, embedding_dim).
    """
    return einx.get_at(
        "[vocab_size] d_model, batch... sequence_length -> batch... sequence_length d_model",
        weight,
        input,
    )


def rms_norm(  # torch flavored
    input: Float[torch.Tensor, "*batch_shape"],
    dims: tuple[int, ...],
    weight: Float[torch.Tensor, "*shape"] | None = None,
    eps: float = 1e-5,
) -> Float[torch.Tensor, "*batch_shape"]:
    r"""Apply Root Mean Square Layer Normalization.

    See :class:`~modules.RMSNorm` for details.
    """
    if dims == ():
        warnings.warn("RMSNorm with empty normalized shape is a no-op", stacklevel=2)
        return input
    in_dtype = input.dtype
    weight_dtype = weight.dtype if weight is not None else in_dtype
    op_dtype = torch.promote_types(in_dtype, weight_dtype)
    if op_dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        op_dtype = torch.float32  # up-cast to float32 for numerical stability
    input = input.to(op_dtype)
    axes_map: dict[str, int] = {f"n{i}": d for i, d in enumerate(dims)}
    _normed: str = " ".join(axes_map)
    # # Compute the root mean square
    # mean_square = torch.mean(input**2, dim=dims, keepdim=True)
    # rms = torch.sqrt(mean_square + eps)
    rms: Float[torch.Tensor, "*mapped"] = (
        einx.mean(f"mapped... [{_normed}]", input.abs() ** 2, **axes_map) + eps
    ) ** 0.5
    # # Normalize the input
    # normalized_input = input / rms
    normed: Float[torch.Tensor, "*batch_shape"] = einx.divide(
        f"mapped... {_normed}, mapped... -> mapped... {_normed}", input, rms, **axes_map
    )
    # # Apply weights if provided
    if weight is not None:
        # normalized_input = normalized_input * weights
        normed: Float[torch.Tensor, "*batch_shape"] = einx.multiply(
            f"mapped... {_normed}, {_normed} -> mapped... {_normed}", normed, weight, **axes_map
        )
    return normed
