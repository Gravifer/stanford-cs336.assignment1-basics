import warnings
from typing import Literal

import einx
import torch
from jaxtyping import Float, Int
from torch.nn.functional import sigmoid


def glu(
    input: Float[torch.Tensor, "*mapped {*dims}"],
) -> Float[torch.Tensor, "*mapped {*dims}"]:
    """Gated linear unit.

    Args:
        input: Input tensor of shape (*mapped, *dims).

    Returns:
        Output tensor of shape (*mapped, *dims).
    """
    if input.shape[-1] % 2 != 0:
        raise ValueError("The last dimension of the input must be even for GLU.")
    split_size = input.shape[-1] // 2
    a, b = torch.split(input, split_size, dim=-1)
    return a * sigmoid(b)


def gelu(
    input: Float[torch.Tensor, "*mapped {*dims}"],
    approximate: Literal["none", "tanh"] = "tanh",
) -> Float[torch.Tensor, "*mapped {*dims}"]:
    """GELU activation function.

    Args:
        input: Input tensor of shape (*mapped, *dims).
        approximate: Whether to use the approximate GELU implementation.

    Returns:
        Output tensor of shape (*mapped, *dims).
    """
    if not approximate:
        return 0.5 * input * (1 + torch.erf(input / (2**0.5)))
    elif approximate == "tanh":
        return 0.5 * input * (1 + torch.tanh(input * 0.7978845608 * (1 + 0.044715 * input * input)))
    else:
        raise NotImplementedError(f"Unknown approximate mode: {approximate}")


def linear(
    input: Float[torch.Tensor, "... in_features"],
    weight: Float[torch.Tensor, "out_features in_features"],
) -> Float[torch.Tensor, "... out_features"]:
    """Linear transformation without bias.

    Args:
        input: Input tensor of shape (..., in_features).
        weight: Weight tensor of shape (out_features, in_features).

    Returns:
        Output tensor of shape (..., out_features).
    """
    return einx.dot("... d_in, d_out d_in->... d_out", input, weight)


def silu(
    input: Float[torch.Tensor, "*mapped {*dims}"],
) -> Float[torch.Tensor, "*mapped {*dims}"]:
    """SiLU activation function.

    Args:
        input: Input tensor of shape (*mapped, *dims).

    Returns:
        Output tensor of shape (*mapped, *dims).
    """
    return einx.multiply(
        "*mapped {*dims}, *mapped {*dims} -> *mapped {*dims}",
        input,
        sigmoid(input),
    )


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
    input: Float[torch.Tensor, "*mapped {*dims}"],
    dims: tuple[int, ...],
    weight: Float[torch.Tensor, "{*dims}"] | None = None,  # noqa: F821
    eps: float = 1e-5,
) -> Float[torch.Tensor, "*mapped {*dims}"]:
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
    dim_map: dict[str, int] = {f"n{i}": d for i, d in enumerate(dims)}
    _normed: str = " ".join(d for d in dim_map.keys())
    # # Compute the root mean square
    # mean_square = torch.mean(input**2, dim=dims, keepdim=True)
    # rms = torch.sqrt(mean_square + eps)
    rms: Float[torch.Tensor, "*mapped"] = (
        einx.mean(f"mapped... [{_normed}]", input.abs() ** 2, **dim_map) + eps
    ) ** 0.5
    # # Normalize the input
    # normalized_input = input / rms
    normed: Float[torch.Tensor, "*mapped {*dims}"] = einx.divide(
        f"mapped... {_normed}, mapped... -> mapped... {_normed}", input, rms, **dim_map
    )
    # # Apply weights if provided
    if weight is not None:
        # normalized_input = normalized_input * weights
        normed: Float[torch.Tensor, "*mapped {*dims}"] = einx.multiply(
            f"mapped... {_normed}, {_normed} -> mapped... {_normed}", normed, weight, **dim_map
        )
    return normed
