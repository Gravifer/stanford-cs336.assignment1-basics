import warnings
from typing import Never

import einx
import regex
import torch
from jaxtyping import Float, Int, Shaped
from torch import dtype, nn

from cs336_basics import initializer as init

type ModelVec = Float[torch.Tensor, "{self.d_model}"]  # ruff takes issue with this # noqa: F821

BRACKETED: regex.Pattern[str] = regex.compile(r"(?P<bracket>\[(?:[^\[\]]++|(?&bracket))*\])")


class Linear(nn.Module):  # mimicking :cls:`torch.nn.Linear`, but NO bias
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Float[torch.Tensor, "{self.out_features} {self.in_features}"]

    @property
    def bias(self) -> Never:
        raise AttributeError("This module has no bias.")

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: dtype | None = None
    ):
        # factory_kwargs = {"device": device, "dtype": dtype} # ! in modern days a typing pain
        super().__init__()
        self.in_features = in_features  # d_in
        self.out_features = out_features  # d_out
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.register_parameter("bias", None)
        self.reset_parameters()  # Object of type `Tensor` is not callable  ty:(call-non-callable)

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        init.starter_trunc_normal_for_linear_(self.weight, self.in_features, self.out_features)

    def forward(
        self, x: Float[torch.Tensor, "... {self.in_features}"]
    ) -> Float[torch.Tensor, "... {self.out_features}"]:
        """Apply the linear transformation to the input."""
        # return torch.nn.functional.linear(x, self.weight, self.bias)
        return einx.dot("... d_in, d_out d_in->... d_out", x, self.weight)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, NO bias"


class RMSNorm(nn.Module):  # mimicking :cls:`torch.nn.RMSNorm`; used for layer normalization
    """RMSNorm (B. Zhang et al. NeurIPS 2019 <https://arxiv.org/abs/1910.07467>, eq 4) for layer normalization.

    Given a vector :math:`𝑎 ∈ ℝ^{𝑑_model}` of activations,
    RMSNorm will rescale each activation :math:`𝑎ᵢ` as follows:

    .. math::
        RMSNorm(𝑎ᵢ) = 𝑎ᵢ / RMS(𝑎) * 𝑔ᵢ

    where :math:`RMS(𝑎) = √(1/𝑑_{model} * Σᵢ 𝑎ᵢ² + eps)`,
    :math:`𝑔 ∈ ℝ^{𝑑_model}` is a learnable *gain* vector,
    and :math:`eps` is a hyperparameter that is often fixed at `1e-5`.
    """

    __constants__ = ["d_model", "eps"]
    d_model: int
    eps: float
    weight: ModelVec  # the `gᵢ`s # noqa: F821
    elementwise_affine: bool

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: torch.device | None = None,
        dtype: dtype | None = None,
    ):
        super().__init__()
        self.d_model: int = d_model
        self.eps: float = eps
        self.elementwise_affine: bool = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty((d_model,), device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def forward(self, x: Shaped[ModelVec, "*batch sequence_length"]) -> Shaped[ModelVec, "*batch sequence_length"]:
        in_dtype = x.dtype
        return self.rms_norm(x, (self.d_model,), self.weight, self.eps).to(in_dtype)

    @classmethod
    def rms_norm(  # torch flavored
        cls,
        input: Float[torch.Tensor, "*mapped {*dims}"],
        dims: tuple[int, ...],
        weights: Float[torch.Tensor, "{*dims}"] | None = None,  # noqa: F821
        eps: float = 1e-5,
    ) -> Float[torch.Tensor, "*mapped {*dims}"]:
        """Functional interface to RMSNorm

        dims is a tuple representing the normalized shape (last ones of the dimensions).
        """
        if dims == ():
            warnings.warn("RMSNorm with empty normalized shape is a no-op", stacklevel=2)
            return input
        in_dtype = input.dtype
        weight_dtype = weights.dtype if weights is not None else in_dtype
        op_dtype = torch.promote_types(in_dtype, weight_dtype)
        if op_dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
            op_dtype = torch.float32  # up-cast to float32 for numerical stability
        input = input.to(op_dtype)
        dim_map: dict[str, int] = {f"n{i}": d for i, d in enumerate(dims)}
        _normed: str = " ".join(d for d in dim_map.keys())
        rms: Float[torch.Tensor, "*mapped"] = (
            einx.mean(f"mapped... [{_normed}]", input.abs() ** 2, **dim_map) + eps
        ) ** 0.5
        normed: Float[torch.Tensor, "*mapped {*dims}"] = einx.divide(
            f"mapped... {_normed}, mapped... -> mapped... {_normed}", input, rms, **dim_map
        )
        if weights is not None:
            normed: Float[torch.Tensor, "*mapped {*dims}"] = einx.multiply(
                f"mapped... {_normed}, {_normed} -> mapped... {_normed}", normed, weights, **dim_map
            )
        return normed

    @classmethod
    def rms_norm_einx(  # einx flavored
        cls,
        desc: str,
        input: Float[torch.Tensor, "*mapped {*dims}"],
        weights: Float[torch.Tensor, "{*dims}"] | None = None,  # noqa: F821
        eps: float = 1e-5,
        **kwargs,
    ) -> Float[torch.Tensor, "*mapped {*dims}"]:
        """Functional interface to RMSNorm

        desc is a einx description string of the operation.
        """
        # TODO: redo this completely
        pass


class Embedding(nn.Module):  # mimicking :cls:`torch.nn.Embedding` in :module:`torch.nn.sparse`
    __constants__ = ["num_embeddings", "embedding_dim"]
    num_embeddings: int  # vocab_size
    embedding_dim: int  # d_model
    weight: Float[torch.Tensor, "{self.num_embeddings} {self.embedding_dim}"]  # matrix of embeddings
    freeze: bool

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        _weight: Float[torch.Tensor, "{self.num_embeddings} {self.embedding_dim}"] | None = None,
        _freeze: bool = False,
        device: torch.device | None = None,
        dtype: dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if _weight is None:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype),
                requires_grad=not _freeze,
            )
            self.reset_parameters()
        else:
            if list(_weight.shape) != [num_embeddings, embedding_dim]:
                raise AssertionError("Shape of weight does not match num_embeddings and embedding_dim")
            self.weight = nn.Parameter(_weight, requires_grad=not _freeze)

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        init.starter_trunc_normal_for_embedding_(self.weight)

    def forward(
        self, token_ids: Int[torch.Tensor, "*batch sequence_length"]
    ) -> Float[torch.Tensor, "*batch sequence_length {self.embedding_dim}"]:
        """Apply the embedding transformation to the input."""
        return einx.get_at(
            "[vocab_size] d_model, batch... sequence_length -> batch... sequence_length d_model",
            self.weight,
            token_ids,
        )

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"

    @classmethod
    def from_pretrained(
        cls,
        embeddings,
        freeze=True,
    ):
        r"""Create Embedding instance from given 2-dimensional FloatTensor.

        Args:
            embeddings (Tensor): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as ``num_embeddings``, second as ``embedding_dim``.
            freeze (bool, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``.
        """
        if embeddings.dim() != 2:
            raise AssertionError("Embeddings parameter is expected to be 2-dimensional")
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            _freeze=freeze,
        )
        return embedding
