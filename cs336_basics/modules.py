import warnings
from typing import Never

import einx
import einx._src.namedtensor.stage1 as einx_stage1
import torch
from einx._src.adapter.einx_from_namedtensor import Invocation as EinxInvocation
from einx._src.adapter.einx_from_namedtensor import _parse_op as parse_einx_op
from jaxtyping import Float, Int, Shaped
from torch import dtype, nn

from cs336_basics import initializer as init

type ModelVec = Float[torch.Tensor, "{self.d_model}"]  # ruff takes issue with this # noqa: F821


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
        input: Float[torch.Tensor, "*in_dims"],
        weights: Float[torch.Tensor, "*shape"] | None = None,
        eps: float = 1e-5,
        **kwargs,
    ) -> Float[torch.Tensor, "*in_dims"]:
        """Functional interface to RMSNorm using an einx description.

        With one input expression, bracketed axes are normalized and ``weights`` uses
        those axes in their order of appearance, e.g. ``"batch... [d]"``. With two
        input expressions, the second expression describes ``weights`` and its axes
        are normalized, e.g. ``"a b c d e, b d -> a b c d e"``.
        """

        parsed_desc = einx_stage1.parse_op(desc)
        input_count = len(parsed_desc.children[0].children)
        if input_count == 2 and weights is None:
            raise ValueError("An einx description with two input expressions requires weights.")
        if input_count not in (1, 2):
            raise ValueError(f"RMSNorm expects one or two input expressions, but found {input_count}.")

        if input_count == 1:
            # Unary reduction syntax is convenient sugar: brackets (or an explicit
            # reduction output) identify the normalized axes, and the weight layout
            # is inferred from those axes.
            invocation = EinxInvocation(desc, name="rms_norm_einx", tensors=(input,), kwargs=kwargs)
            exprs_in, exprs_out = parse_einx_op(
                desc,
                el_op=lambda op: f"{op.children[0].children[0]} ->",
                invocation=invocation,
                implicit_output="bijective",
                mark_reduced_axes=True,
            )
            (expr_in,) = exprs_in
            (expr_reduced,) = exprs_out

            input_desc = str(einx_stage1.remove(expr_in, einx_stage1.Brackets, keep_children=True))
            weight_desc = " ".join(
                str(expr.inner) for expr in expr_in.nodes() if isinstance(expr, einx_stage1.Brackets)
            )
            mean_desc = desc
            affine_desc = f"{input_desc}, {weight_desc} -> {input_desc}"
            parameters = kwargs
        else:
            # Binary syntax follows einx's elementwise API. The second expression
            # both describes the weight tensor and selects the normalized axes.
            assert weights is not None
            invocation = EinxInvocation(desc, name="rms_norm_einx", tensors=(input, weights), kwargs=kwargs)

            def elementwise_signature(op):
                inputs = ", ".join("" for _ in op.children[0].children)
                return f"{inputs} ->"

            exprs_in, exprs_out = parse_einx_op(
                desc,
                el_op=elementwise_signature,
                invocation=invocation,
                implicit_output="bijective",
            )
            expr_in, expr_weights = exprs_in
            (_expr_out,) = exprs_out

            normalized_axis_names = {expr.name for expr in expr_weights.nodes() if isinstance(expr, einx_stage1.Axis)}
            expr_with_reduction = einx_stage1.map(
                expr_in,
                lambda expr: (
                    einx_stage1.Brackets.create(expr.__deepcopy__())
                    if isinstance(expr, einx_stage1.Axis) and expr.name in normalized_axis_names
                    else None
                ),
                include_children=False,
            )

            input_desc = str(expr_in)
            expr_reduced = einx_stage1.remove(expr_with_reduction, einx_stage1.Brackets, keep_children=False)
            mean_desc = str(expr_with_reduction)
            affine_desc = desc

            # Tensor shapes disambiguate named ellipses in the binary form. Preserve
            # that information when the derived unary reduction is evaluated.
            parameters = dict(kwargs)
            parameters.update(einx.solve_axes(f"{expr_in}, {expr_weights}", input, weights, **kwargs))

        in_dtype = input.dtype
        weight_dtype = weights.dtype if weights is not None else in_dtype
        op_dtype = torch.promote_types(in_dtype, weight_dtype)
        if op_dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
            op_dtype = torch.float32
        input = input.to(op_dtype)

        rms: Float[torch.Tensor, "*mapped"] = (einx.mean(mean_desc, input.abs() ** 2, **parameters) + eps) ** 0.5
        normed: Float[torch.Tensor, "*in_dims"] = einx.divide(
            f"{input_desc}, {expr_reduced} -> {input_desc}", input, rms, **parameters
        )
        if weights is not None:
            normed = einx.multiply(affine_desc, normed, weights.to(op_dtype), **parameters)
        return normed


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
