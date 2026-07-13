from typing import Never, NoReturn

import einx
import einx._src.namedtensor.stage1 as einx_stage1
import torch
from einx._src.adapter.einx_from_namedtensor import Invocation as EinxInvocation
from einx._src.adapter.einx_from_namedtensor import _parse_op as parse_einx_op
from einx._src.frontend.errors import SemanticError as EinxSemanticError
from jaxtyping import Float, Int, Shaped
from torch import dtype, nn
from typing_extensions import deprecated  # ? ruff doesn't see that python 3.13 have deprecated in warnings already

from cs336_basics.nn import functional as F
from cs336_basics.nn import initializer as init

# from .feed_forward import SwiGLU  # ! would be circular

type ModelVec = Float[torch.Tensor, "{self.d_model}"]  # ruff takes issue with this # noqa: F821


class Embedding(nn.Module):  # mimicking :cls:`torch.nn.Embedding` in :module:`torch.nn.sparse`
    __constants__ = ["num_embeddings", "embedding_dim"]
    num_embeddings: int  # vocab_size
    embedding_dim: int  # d_model
    weight: Float[torch.Tensor, "{self.num_embeddings} {self.embedding_dim}"]  # matrix of embeddings

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        _weight: Float[torch.Tensor, "{num_embeddings} {embedding_dim}"] | None = None,
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
        return F.embedding(token_ids, self.weight)

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"

    @classmethod
    def from_pretrained(
        cls,
        embeddings: Float[torch.Tensor, "num_embeddings embedding_dim"],
        freeze: bool = True,
    ) -> "Embedding":  # noqa: UP037
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


class RMSNorm(nn.Module):  # mimicking :cls:`torch.nn.RMSNorm`; used for layer normalization
    """RMSNorm (B. Zhang et al. NeurIPS 2019 <https://arxiv.org/abs/1910.07467>, eq 4) for layer normalization.

    Given a vector :math:`𝑎 ∈ ℝ^{𝑑_model}` of activations,
    RMSNorm will rescale each activation :math:`𝑎ᵢ` as follows:

    .. math::
        RMSNorm(𝑎ᵢ) = 𝑎ᵢ / RMS(𝑎) * 𝑔ᵢ

    where :math:`RMS(𝑎) = √(1/𝑑_model * Σᵢ 𝑎ᵢ² + eps)`,
    :math:`𝑔 ∈ ℝ^{𝑑_model}` is a learnable *gain* vector,
    and :math:`eps` is a hyperparameter that is often fixed at `1e-5`.
    """

    __constants__ = ["d_model", "eps"]
    d_model: int
    eps: float
    weight: ModelVec | None  # the `gᵢ`s # noqa: F821
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
            assert self.weight is not None, "RMSNorm.weight is required when elementwise_affine=True"
            nn.init.ones_(self.weight)

    def forward(self, x: Shaped[ModelVec, "*batch sequence_length"]) -> Shaped[ModelVec, "*batch sequence_length"]:
        in_dtype = x.dtype
        return self.rms_norm(x, (self.d_model,), self.weight, self.eps).to(in_dtype)

    @staticmethod
    def rms_norm(  # torch flavored
        input: Float[torch.Tensor, "*batch_shape"],
        dims: tuple[int, ...],
        weight: Float[torch.Tensor, "*shape"] | None = None,
        eps: float = 1e-5,
    ) -> Float[torch.Tensor, "*batch_shape"]:
        """Functional interface to RMSNorm
        Root Mean Square Layer Normalization.

        Args:
            input: Input tensor of shape (*mapped, *dims).
            dims: Dimensions to normalize over.
            weight: Optional weight tensor of shape (*dims).
            eps: Small value to avoid division by zero.

        Returns:
            Normalized tensor of shape (*mapped, *dims).
        """
        return F.rms_norm(input, dims, weight, eps)

    @staticmethod
    def rms_norm_einx(  # einx flavored
        desc: str,
        input: Float[torch.Tensor, "*in_dims"],
        weight: Float[torch.Tensor, "*shape"] | None = None,
        eps: float = 1e-5,
        **kwargs,
    ) -> Float[torch.Tensor, "*in_dims"]:
        """Functional interface to RMSNorm using an einx description.

        With one input expression, bracketed axes are normalized and ``weight`` uses
        those axes in their order of appearance, e.g. ``"batch... [d]"``. With two
        unbracketed inputs, the second expression describes ``weight`` and selects
        the normalized axes, e.g. ``"a b c d e, b d -> a b c d e"``. A bracketed,
        targeted-preserving description such as
        ``"a [b] c [d] e, [d b] -> e [b d] c a"`` may rearrange the mapped axes.
        """

        parsed_desc = einx_stage1.parse_op(desc)
        input_count = len(parsed_desc.children[0].children)
        if input_count == 2 and weight is None:
            raise EinxSemanticError("An einx description with two input expressions requires weight.")
        if input_count not in (1, 2):
            raise EinxSemanticError(f"RMSNorm expects one or two input expressions, but found {input_count}.")

        has_brackets = any(isinstance(expr, einx_stage1.Brackets) for expr in parsed_desc.nodes())
        has_targeted_output = len(parsed_desc.children) == 2 and any(
            isinstance(expr, einx_stage1.Brackets) for expr in parsed_desc.children[1].nodes()
        )
        targeted_preserving = (input_count == 2 and has_brackets) or (input_count == 1 and has_targeted_output)
        rearrange_desc = None

        if input_count == 1 and not targeted_preserving:
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
            axes_map = kwargs
        elif targeted_preserving:
            # In this form brackets describe RMSNorm's true elementary signature:
            # targeted axes (and optionally their weights) map to the same axes.
            tensors = (input,) if input_count == 1 else (input, weight)
            invocation = EinxInvocation(desc, name="rms_norm_einx", tensors=tensors, kwargs=kwargs)

            def targeted_signature(op):
                inputs = ", ".join(str(expr) for expr in op.children[0].children)
                return f"{inputs} -> {op.children[0].children[0]}"

            exprs_in, exprs_out = parse_einx_op(
                desc,
                el_op=targeted_signature,
                invocation=invocation,
                implicit_output=0,
            )
            expr_in = exprs_in[0]
            (expr_out,) = exprs_out

            targeted_axes_in = [
                expr.name
                for expr in expr_in.nodes()
                if isinstance(expr, einx_stage1.Axis) and einx_stage1.is_in_brackets(expr)
            ]
            targeted_axes_out = [
                expr.name
                for expr in expr_out.nodes()
                if isinstance(expr, einx_stage1.Axis) and einx_stage1.is_in_brackets(expr)
            ]
            if targeted_axes_in != targeted_axes_out:
                targeted_axes = set(targeted_axes_in).union(targeted_axes_out)
                raise EinxSemanticError(
                    invocation=invocation,
                    pos=invocation.indicator.get_pos_for_axisnames(exprs_in + exprs_out, targeted_axes),
                    message=(
                        "All axes marked with brackets must appear in the same order in the primary input and output "
                        "expressions.\n%EXPR%"
                    ),
                )

            input_desc = str(einx_stage1.remove(expr_in, einx_stage1.Brackets, keep_children=True))
            output_desc = str(einx_stage1.remove(expr_out, einx_stage1.Brackets, keep_children=True))
            expr_reduced = einx_stage1.remove(expr_in, einx_stage1.Brackets, keep_children=False)
            mean_desc = str(expr_in)

            if input_count == 2:
                assert weight is not None
                expr_weights = exprs_in[1]
                weight_desc = str(einx_stage1.remove(expr_weights, einx_stage1.Brackets, keep_children=True))
                affine_desc = f"{input_desc}, {weight_desc} -> {output_desc}"
                axes_map = dict(kwargs)
                axes_map.update(einx.solve_axes(f"{input_desc}, {weight_desc}", input, weight, **kwargs))
            else:
                weight_desc = " ".join(
                    str(expr.inner) for expr in expr_in.nodes() if isinstance(expr, einx_stage1.Brackets)
                )
                affine_desc = f"{input_desc}, {weight_desc} -> {output_desc}"
                rearrange_desc = f"{input_desc} -> {output_desc}"
                axes_map = kwargs
        else:
            # Binary syntax follows einx's elementwise API. The second expression
            # both describes the weight tensor and selects the normalized axes.
            assert weight is not None
            invocation = EinxInvocation(desc, name="rms_norm_einx", tensors=(input, weight), kwargs=kwargs)

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
            axes_map = dict(kwargs)
            axes_map.update(einx.solve_axes(f"{expr_in}, {expr_weights}", input, weight, **kwargs))

        in_dtype = input.dtype
        weight_dtype = weight.dtype if weight is not None else in_dtype
        op_dtype = torch.promote_types(in_dtype, weight_dtype)
        if op_dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
            op_dtype = torch.float32
        input = input.to(op_dtype)

        rms: Float[torch.Tensor, "*mapped"] = (einx.mean(mean_desc, input.abs() ** 2, **axes_map) + eps) ** 0.5
        normed: Float[torch.Tensor, "*in_dims"] = einx.divide(
            f"{input_desc}, {expr_reduced} -> {input_desc}", input, rms, **axes_map
        )
        if weight is not None:
            normed = einx.multiply(affine_desc, normed, weight.to(op_dtype), **axes_map)
        elif rearrange_desc is not None:
            normed = einx.id(rearrange_desc, normed, **axes_map)
        return normed


class Linear(nn.Module):  # mimicking :cls:`torch.nn.Linear`, but NO bias
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Float[torch.Tensor, "{self.out_features} {self.in_features}"]
    bias: Never

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
        self, x: Float[torch.Tensor, "*mapped {self.in_features}"]
    ) -> Float[torch.Tensor, "*mapped {self.out_features}"]:
        """Apply the linear transformation to the input."""
        # return torch.nn.functional.linear(x, self.weight)
        return F.linear(x, self.weight)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, NO bias"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.weight.shape

    @property
    def dtype(self) -> dtype:
        return self.weight.dtype


@deprecated("we won't have a SiLU module; without the in-place option, not using the functional version is moot.")
class SiLU(nn.Module):
    def __init__(self, inplace: bool = False) -> NoReturn:
        raise NotImplementedError(
            "we won't have a SiLU module; without the in-place option, not using the functional version is moot."
        )
