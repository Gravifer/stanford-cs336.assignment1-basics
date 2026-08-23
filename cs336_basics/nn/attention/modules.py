"""Multi-head, grouped-query, and multi-query attention modules."""

from collections.abc import Mapping
from math import prod
from typing import Any, Final, Literal, overload

import einx
import torch
from jaxtyping import Float, Int, Shaped
from torch import nn

from cs336_basics.nn import functional as F
from cs336_basics.nn.analytics import CostRepr, TensorRepr, _CostChild, _CostScope

from .. import initializer as init
from .._typing import MaskBias
from ..modules import Linear, Module
from . import _head_layout
from .rope import RotaryPositionalEmbedding


type _PositionLayout = Literal["batch", "head"]
type _MaskLayout = Literal["batch", "head"]


class MultiheadAttention(Module):
    """Multi-head attention over tensors shaped ``(*batch, sequence, features)``.

    ``kdim`` and ``vdim`` describe the raw key and value input widths. The
    projected per-head widths are ``qk_head_dim`` and ``value_head_dim``.
    These pairs are independent. ``num_heads`` is the query-head count and
    ``num_kv_heads`` is the shared key/value-head count. The K projection has
    shape ``(num_kv_heads * qk_head_dim, kdim)``, while the V projection has
    shape ``(num_kv_heads * value_head_dim, vdim)``.

    Omitting ``num_kv_heads`` gives ordinary MHA. For eight query heads,
    ``num_kv_heads=2`` gives GQA and ``num_kv_heads=1`` gives MQA::

        MultiheadAttention(512, 8)
        MultiheadAttention(512, 8, num_kv_heads=2)
        MultiheadAttention(512, num_q_heads=8, num_kv_heads=1)

    Boolean masks follow :func:`cs336_basics.nn.functional.scaled_dot_product_attention`:
    ``True`` permits attention and ``False`` masks it.
    Masks are batch-aligned by default; pass ``mask_layout="head"`` to interpret
    their leading suffix as an explicit query-head axis instead.
    """

    embed_dim: Final[int]
    num_heads: Final[int]
    num_kv_heads: Final[int]
    kdim: Final[int]
    vdim: Final[int]
    qk_head_dim: Final[int]
    value_head_dim: Final[int]
    dropout: float
    q_proj_dim: Final[int]
    k_proj_dim: Final[int]
    v_proj_dim: Final[int]
    _qkv_same_input_dim: Final[bool]
    _layout_strategy: Final[Literal["head_before_sequence", "head_after_sequence"]]
    _qk_execution_strategy: Final[Literal["auto", "separate", "stacked"]]

    type QueryVec = Float[torch.Tensor, "{self.embed_dim}"]  # noqa: F821
    type KeyVec = Float[torch.Tensor, "{self.kdim}"]  # noqa: F821
    type ValueVec = Float[torch.Tensor, "{self.vdim}"]  # noqa: F821
    type OutputVec = Float[torch.Tensor, "{self.embed_dim}"]  # noqa: F821

    @property
    def num_q_heads(self) -> int:
        """Query-head count, exposed as an explicit alias for ``num_heads``."""
        return self.num_heads

    @staticmethod
    def _coalesce_alias(
        canonical_name: str,
        canonical_value: int | None,
        alias_name: str,
        alias_value: int | None,
    ) -> int | None:
        """Resolve mutually exclusive course and Torch-style argument names."""
        if canonical_value is not None and alias_value is not None:
            raise ValueError(f"{canonical_name} and {alias_name} are mutually exclusive")
        return canonical_value if canonical_value is not None else alias_value

    @overload
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        *,
        num_q_heads: None = None,
        num_kv_heads: int | None = None,
        kdim: int | None = None,
        vdim: int | None = None,
        qk_head_dim: int | None = None,
        value_head_dim: int | None = None,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        _layout_strategy: Literal["head_before_sequence", "head_after_sequence"] = "head_before_sequence",
        _qk_execution_strategy: Literal["auto", "separate", "stacked"] = "auto",
    ) -> None: ...

    @overload
    def __init__(
        self,
        embed_dim: int,
        num_heads: None = None,
        dropout: float = 0.0,
        *,
        num_q_heads: int,
        num_kv_heads: int | None = None,
        kdim: int | None = None,
        vdim: int | None = None,
        qk_head_dim: int | None = None,
        value_head_dim: int | None = None,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        _layout_strategy: Literal["head_before_sequence", "head_after_sequence"] = "head_before_sequence",
        _qk_execution_strategy: Literal["auto", "separate", "stacked"] = "auto",
    ) -> None: ...

    def __init__(
        self,
        embed_dim: int,
        num_heads: int | None = None,
        dropout: float = 0.0,
        *,
        num_q_heads: int | None = None,
        num_kv_heads: int | None = None,
        kdim: int | None = None,
        vdim: int | None = None,
        qk_head_dim: int | None = None,
        value_head_dim: int | None = None,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        _layout_strategy: Literal["head_before_sequence", "head_after_sequence"] = "head_before_sequence",
        _qk_execution_strategy: Literal["auto", "separate", "stacked"] = "auto",
    ) -> None:
        resolved_num_heads = self._coalesce_alias("num_heads", num_heads, "num_q_heads", num_q_heads)
        if resolved_num_heads is None:
            raise TypeError("missing query-head count: supply num_heads or num_q_heads")
        num_heads = resolved_num_heads
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                "embed_dim and num_heads must be greater than 0, "
                f"got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be greater than 0, got {num_kv_heads}")
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads must be divisible by num_kv_heads, got {num_heads} and {num_kv_heads}")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")

        kdim = embed_dim if kdim is None else kdim
        vdim = embed_dim if vdim is None else vdim
        if kdim <= 0 or vdim <= 0:
            raise ValueError(f"kdim and vdim must be greater than 0, got kdim={kdim} and vdim={vdim}")

        if qk_head_dim is None:
            if embed_dim % num_heads != 0:
                raise ValueError("embed_dim must be divisible by num_heads when qk_head_dim is omitted")
            qk_head_dim = embed_dim // num_heads
        if value_head_dim is None:
            value_head_dim = qk_head_dim
        if qk_head_dim <= 0 or value_head_dim <= 0:
            raise ValueError(
                "qk_head_dim and value_head_dim must be greater than 0, "
                f"got qk_head_dim={qk_head_dim} and value_head_dim={value_head_dim}"
            )
        if rope is not None and rope.d_k != qk_head_dim:
            raise ValueError(f"RoPE width {rope.d_k} does not match qk_head_dim {qk_head_dim}")
        if _layout_strategy not in ("head_before_sequence", "head_after_sequence"):
            raise ValueError(f"unknown layout strategy: {_layout_strategy!r}")
        if _qk_execution_strategy not in ("auto", "separate", "stacked"):
            raise ValueError(f"unknown Q/K execution strategy: {_qk_execution_strategy!r}")

        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.kdim = kdim
        self.vdim = vdim
        self.qk_head_dim = qk_head_dim
        self.value_head_dim = value_head_dim
        self.dropout = dropout
        self.rope = rope
        self.q_proj_dim = num_heads * qk_head_dim
        self.k_proj_dim = num_kv_heads * qk_head_dim
        self.v_proj_dim = num_kv_heads * value_head_dim
        self._qkv_same_input_dim = kdim == embed_dim and vdim == embed_dim
        self._layout_strategy = _layout_strategy
        self._qk_execution_strategy = _qk_execution_strategy

        if self._qkv_same_input_dim:
            self.in_proj_weight = nn.Parameter(
                torch.empty(
                    (self.q_proj_dim + self.k_proj_dim + self.v_proj_dim, embed_dim),
                    device=device,
                    dtype=dtype,
                )
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)
        else:
            self.register_parameter("in_proj_weight", None)
            self.q_proj_weight = nn.Parameter(torch.empty((self.q_proj_dim, embed_dim), device=device, dtype=dtype))
            self.k_proj_weight = nn.Parameter(torch.empty((self.k_proj_dim, kdim), device=device, dtype=dtype))
            self.v_proj_weight = nn.Parameter(torch.empty((self.v_proj_dim, vdim), device=device, dtype=dtype))
        self.output_proj = Linear(num_heads * value_head_dim, embed_dim, device=device, dtype=dtype)
        self.register_load_state_dict_pre_hook(self._translate_projection_state_dict)
        self.reset_parameters()

    def _separate_projection_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logical Q/K/V weights from separate physical storage."""
        assert self.q_proj_weight is not None, (
            "projection storage invariant broken: in_proj_weight is None but q_proj_weight is also None; "
            f"the separate Q weight should have shape ({self.q_proj_dim}, {self.embed_dim})"
        )
        assert self.k_proj_weight is not None, (
            "projection storage invariant broken: in_proj_weight is None but k_proj_weight is also None; "
            f"the separate K weight should have shape ({self.k_proj_dim}, {self.kdim})"
        )
        assert self.v_proj_weight is not None, (
            "projection storage invariant broken: in_proj_weight is None but v_proj_weight is also None; "
            f"the separate V weight should have shape ({self.v_proj_dim}, {self.vdim})"
        )
        return self.q_proj_weight, self.k_proj_weight, self.v_proj_weight

    # Public and complete by design: unlike torch.nn.MultiheadAttention's private
    # helper, this resets both the input projections and the delegated output projection.
    def reset_parameters(self) -> None:
        """Reset all projection parameters."""
        if self.in_proj_weight is not None:
            q_weight, k_weight, v_weight = einx.id(
                "(q + k + v) input -> q input, k input, v input",
                self.in_proj_weight,
                q=self.q_proj_dim,
                k=self.k_proj_dim,
                v=self.v_proj_dim,
            )
            init.starter_trunc_normal_for_linear_(q_weight, self.embed_dim, self.q_proj_dim)
            init.starter_trunc_normal_for_linear_(k_weight, self.kdim, self.k_proj_dim)
            init.starter_trunc_normal_for_linear_(v_weight, self.vdim, self.v_proj_dim)
        else:
            q_weight, k_weight, v_weight = self._separate_projection_weights()
            init.starter_trunc_normal_for_linear_(q_weight, self.embed_dim, self.q_proj_dim)
            init.starter_trunc_normal_for_linear_(k_weight, self.kdim, self.k_proj_dim)
            init.starter_trunc_normal_for_linear_(v_weight, self.vdim, self.v_proj_dim)
        self.output_proj.reset_parameters()

    def _in_projection(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Project Q/K/V using the cheapest path allowed by storage and input identity."""
        if self.in_proj_weight is None:
            q_weight, k_weight, v_weight = self._separate_projection_weights()
            return (
                F.linear(query, q_weight),
                F.linear(key, k_weight),
                F.linear(value, v_weight),
                None,
            )

        if query is key and key is value:
            projected = F.linear(query, self.in_proj_weight)
            projected_qk, projected_v = einx.id(
                "batch... (qk + v) -> batch... qk, batch... v",
                projected,
                qk=self.q_proj_dim + self.k_proj_dim,
                v=self.v_proj_dim,
            )
            projected_q, projected_k = einx.id(
                "batch... (q + k) -> batch... q, batch... k",
                projected_qk,
                q=self.q_proj_dim,
                k=self.k_proj_dim,
            )
            return (
                projected_q,
                projected_k,
                projected_v,
                projected_qk if self.num_heads == self.num_kv_heads else None,
            )

        if key is value:
            q_weight, kv_weight = einx.id(
                "(q + kv) input -> q input, kv input",
                self.in_proj_weight,
                q=self.q_proj_dim,
                kv=self.k_proj_dim + self.v_proj_dim,
            )
            projected_q = F.linear(query, q_weight)
            projected_kv = F.linear(key, kv_weight)
            projected_k, projected_v = einx.id(
                "batch... (k + v) -> batch... k, batch... v",
                projected_kv,
                k=self.k_proj_dim,
                v=self.v_proj_dim,
            )
            return projected_q, projected_k, projected_v, None

        q_weight, k_weight, v_weight = einx.id(
            "(q + k + v) input -> q input, k input, v input",
            self.in_proj_weight,
            q=self.q_proj_dim,
            k=self.k_proj_dim,
            v=self.v_proj_dim,
        )
        return F.linear(query, q_weight), F.linear(key, k_weight), F.linear(value, v_weight), None

    def _split_heads(
        self,
        projected_q: torch.Tensor,
        projected_k: torch.Tensor,
        projected_v: torch.Tensor,
        projected_qk: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Expose projected Q/K/V head axes in the configured activation layout."""
        if self._layout_strategy == "head_before_sequence":
            q_description = "batch... query (query_head d_k) -> batch... query_head query d_k"
            k_description = "batch... key (kv_head d_k) -> batch... kv_head key d_k"
            v_description = "batch... key (kv_head d_v) -> batch... kv_head key d_v"
            qk_description = "batch... query (qk query_head d_k) -> qk batch... query_head query d_k"
        else:
            q_description = "batch... query (query_head d_k) -> batch... query query_head d_k"
            k_description = "batch... key (kv_head d_k) -> batch... key kv_head d_k"
            v_description = "batch... key (kv_head d_v) -> batch... key kv_head d_v"
            qk_description = "batch... query (qk query_head d_k) -> qk batch... query query_head d_k"

        qk = None
        if projected_qk is not None:
            qk = einx.id(
                qk_description,
                projected_qk,
                qk=2,
                query_head=self.num_heads,
                d_k=self.qk_head_dim,
            )
            q, k = qk[0], qk[1]
        else:
            q = einx.id(
                q_description,
                projected_q,
                query_head=self.num_heads,
                d_k=self.qk_head_dim,
            )
            k = einx.id(
                k_description,
                projected_k,
                kv_head=self.num_kv_heads,
                d_k=self.qk_head_dim,
            )
        v = einx.id(
            v_description,
            projected_v,
            kv_head=self.num_kv_heads,
            d_v=self.value_head_dim,
        )
        return q, k, v, qk

    def _prepare_rope_positions(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None,
        *,
        position_layout: _PositionLayout,
    ) -> torch.Tensor | None:
        """Align batch-oriented positions with one projected head tensor."""
        if token_positions is None or position_layout == "head":
            return token_positions

        if token_positions.ndim == 0:
            raise ValueError("token_positions must include a sequence dimension")
        if self._layout_strategy == "head_before_sequence":
            batch_sequence_shape = (*x.shape[:-3], x.shape[-2])
        else:
            batch_sequence_shape = (*x.shape[:-3], x.shape[-3])
        if token_positions.ndim > len(batch_sequence_shape):
            raise ValueError(
                f"batch-aligned token_positions shape {token_positions.shape} "
                f"is not compatible with batch/sequence shape {batch_sequence_shape}"
            )
        suffix_shape = batch_sequence_shape[-token_positions.ndim :]
        if token_positions.shape[-1] != suffix_shape[-1]:
            raise ValueError(
                f"batch-aligned token_positions shape {token_positions.shape} "
                f"is not compatible with batch/sequence shape {batch_sequence_shape}"
            )
        try:
            token_positions.expand(suffix_shape)
        except RuntimeError as error:
            raise ValueError(
                f"batch-aligned token_positions shape {token_positions.shape} "
                f"is not compatible with batch/sequence shape {batch_sequence_shape}"
            ) from error
        return einx.id("batch... sequence -> batch... 1 sequence", token_positions)

    def _apply_single_rope(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None,
        *,
        position_layout: _PositionLayout,
    ) -> torch.Tensor:
        """Apply RoPE to one projected tensor under its position-layout contract."""
        assert self.rope is not None, "RoPE application requires a registered RotaryPositionalEmbedding"
        token_positions = self._prepare_rope_positions(x, token_positions, position_layout=position_layout)
        selection = self.rope._select_rotations(
            x,
            token_positions,
            broadcast_positions=True,
            layout_strategy=self._layout_strategy,
        )
        return self.rope._apply_rotations(x, selection, layout_strategy=self._layout_strategy)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        packed_qk: torch.Tensor | None,
        query_positions: torch.Tensor | None,
        key_positions: torch.Tensor | None,
        *,
        query_position_layout: _PositionLayout,
        key_position_layout: _PositionLayout,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate Q/K together when their storage and position contracts permit."""
        assert self.rope is not None, "Q/K RoPE application requires a registered RotaryPositionalEmbedding"
        shared_positions = q.shape == k.shape and (
            (query_positions is None and key_positions is None)
            or (query_positions is key_positions and query_position_layout == key_position_layout)
        )
        if not shared_positions:
            if self._qk_execution_strategy == "stacked":
                raise ValueError("stacked Q/K RoPE requires equal shapes and shared token positions")
            return (
                self._apply_single_rope(q, query_positions, position_layout=query_position_layout),
                self._apply_single_rope(k, key_positions, position_layout=key_position_layout),
            )

        positions = self._prepare_rope_positions(q, query_positions, position_layout=query_position_layout)
        if self._qk_execution_strategy == "auto" and packed_qk is not None:
            selection = self.rope._select_rotations(
                q,
                positions,
                broadcast_positions=True,
                layout_strategy=self._layout_strategy,
            )
            selection = self.rope._prepend_selection_axis(selection, 2)
            rotated_qk = self.rope._apply_rotations(
                packed_qk,
                selection,
                layout_strategy=self._layout_strategy,
            )
            return rotated_qk[0], rotated_qk[1]
        return self.rope.apply_qk(
            q,
            k,
            positions,
            _stacked=self._qk_execution_strategy == "stacked",
            _layout_strategy=self._layout_strategy,
        )

    def _validate_inputs(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
        """Validate raw Q/K/V batch, sequence, and feature dimensions."""
        if query.shape[:-2] != key.shape[:-2] or key.shape[:-2] != value.shape[:-2]:
            raise ValueError(
                "query, key, and value must have identical batch shapes, "
                f"got {query.shape[:-2]}, {key.shape[:-2]}, and {value.shape[:-2]}"
            )
        if key.shape[-2] != value.shape[-2]:
            raise ValueError(f"key and value sequence lengths must match, got {key.shape[-2]} and {value.shape[-2]}")
        expected_widths = (self.embed_dim, self.kdim, self.vdim)
        actual_widths = (query.shape[-1], key.shape[-1], value.shape[-1])
        if actual_widths != expected_widths:
            raise ValueError(f"expected query/key/value widths {expected_widths}, got {actual_widths}")

    @staticmethod
    def _prepare_attention_mask(
        mask: torch.Tensor | None,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        mask_layout: _MaskLayout,
    ) -> torch.Tensor | None:
        """Insert a singleton head axis for batch-aligned masks."""
        if mask_layout not in ("batch", "head"):
            raise ValueError(f"mask_layout must be 'batch' or 'head', got {mask_layout!r}")
        if mask is None or mask_layout == "head":
            return mask

        batch_query_key_shape = (*query.shape[:-2], query.shape[-2], key.shape[-2])
        if mask.ndim < 2 or mask.ndim > len(batch_query_key_shape):
            raise ValueError(
                f"batch-aligned mask shape {mask.shape} is not compatible with "
                f"batch/query/key shape {batch_query_key_shape}"
            )
        suffix_shape = batch_query_key_shape[-mask.ndim :]
        if tuple(mask.shape[-2:]) != tuple(suffix_shape[-2:]):
            raise ValueError(
                f"batch-aligned mask shape {mask.shape} is not compatible with "
                f"batch/query/key shape {batch_query_key_shape}"
            )
        try:
            mask.expand(suffix_shape)
        except RuntimeError as error:
            raise ValueError(
                f"batch-aligned mask shape {mask.shape} is not compatible with "
                f"batch/query/key shape {batch_query_key_shape}"
            ) from error
        return einx.id("batch... query key -> batch... 1 query key", mask)

    @staticmethod
    def _resolve_position_layouts(
        *,
        position_layout: _PositionLayout | None,
        query_position_layout: _PositionLayout | None,
        key_position_layout: _PositionLayout | None,
    ) -> tuple[_PositionLayout, _PositionLayout]:
        """Resolve the shared position-layout alias or independent Q/K layouts."""
        if position_layout is not None and (query_position_layout is not None or key_position_layout is not None):
            raise ValueError("position_layout is mutually exclusive with query_position_layout and key_position_layout")
        valid_layouts = ("batch", "head")
        for name, layout in (
            ("position_layout", position_layout),
            ("query_position_layout", query_position_layout),
            ("key_position_layout", key_position_layout),
        ):
            if layout is not None and layout not in valid_layouts:
                raise ValueError(f"{name} must be 'batch' or 'head', got {layout!r}")
        if position_layout is not None:
            return position_layout, position_layout
        return query_position_layout or "batch", key_position_layout or "batch"

    @overload
    def forward(
        self,
        query: Shaped[QueryVec, "*batch query_len"],
        key: Shaped[KeyVec, "*batch key_len"],
        value: Shaped[ValueVec, "*batch key_len"],
        mask: MaskBias[torch.Tensor, "*slew query_len key_len"] | None = None,  # ty: ignore[invalid-syntax-in-forward-annotation]
        *,
        is_causal: bool = False,
        mask_layout: _MaskLayout = "batch",
        query_positions: Int[torch.Tensor, "*query_position_batch query_len"] | None = None,
        key_positions: Int[torch.Tensor, "*key_position_batch key_len"] | None = None,
        position_layout: _PositionLayout,
        query_position_layout: None = None,
        key_position_layout: None = None,
    ) -> Shaped[OutputVec, "*batch query_len"]: ...

    @overload
    def forward(
        self,
        query: Shaped[QueryVec, "*batch query_len"],
        key: Shaped[KeyVec, "*batch key_len"],
        value: Shaped[ValueVec, "*batch key_len"],
        mask: MaskBias[torch.Tensor, "*slew query_len key_len"] | None = None,  # ty: ignore[invalid-syntax-in-forward-annotation]
        *,
        is_causal: bool = False,
        mask_layout: _MaskLayout = "batch",
        query_positions: Int[torch.Tensor, "*query_position_batch query_len"] | None = None,
        key_positions: Int[torch.Tensor, "*key_position_batch key_len"] | None = None,
        position_layout: None = None,
        query_position_layout: _PositionLayout | None = None,
        key_position_layout: _PositionLayout | None = None,
    ) -> Shaped[OutputVec, "*batch query_len"]: ...

    def forward(
        self,
        query: Shaped[QueryVec, "*batch query_len"],
        key: Shaped[KeyVec, "*batch key_len"],
        value: Shaped[ValueVec, "*batch key_len"],
        mask: MaskBias[torch.Tensor, "*slew query_len key_len"] | None = None,  # ty: ignore[invalid-syntax-in-forward-annotation]
        *,
        is_causal: bool = False,
        mask_layout: _MaskLayout = "batch",
        query_positions: Int[torch.Tensor, "*query_position_batch query_len"] | None = None,
        key_positions: Int[torch.Tensor, "*key_position_batch key_len"] | None = None,
        position_layout: _PositionLayout | None = None,
        query_position_layout: _PositionLayout | None = None,
        key_position_layout: _PositionLayout | None = None,
    ) -> Shaped[OutputVec, "*batch query_len"]:
        """Project Q/K/V, apply optional RoPE, attend, and project the result.

        Masks and position tensors are batch-aligned by default. Use
        ``mask_layout="head"`` for a mask with an explicit query-head axis.
        Use ``position_layout``
        to set one interpretation for Q and K, or the mutually exclusive
        ``query_position_layout`` and ``key_position_layout`` controls to set
        them independently. ``"head"`` preserves explicit head-dependent
        position axes.
        """
        self._validate_inputs(query, key, value)
        mask = self._prepare_attention_mask(mask, query, key, mask_layout=mask_layout)
        resolved_query_layout, resolved_key_layout = self._resolve_position_layouts(
            position_layout=position_layout,
            query_position_layout=query_position_layout,
            key_position_layout=key_position_layout,
        )

        projected_q, projected_k, projected_v, projected_qk = self._in_projection(query, key, value)
        q, k, v, packed_qk = self._split_heads(projected_q, projected_k, projected_v, projected_qk)

        if self.rope is not None:
            q, k = self._apply_rope(
                q,
                k,
                packed_qk,
                query_positions,
                key_positions,
                query_position_layout=resolved_query_layout,
                key_position_layout=resolved_key_layout,
            )

        attended = _head_layout.scaled_dot_product_attention(
            q,
            k,
            v,
            mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=self.qk_head_dim**-0.5,
            layout_strategy=self._layout_strategy,
        )
        if self._layout_strategy == "head_before_sequence":
            joined = einx.id(
                "batch... query_head query d_v -> batch... query (query_head d_v)",
                attended,
                query_head=self.num_heads,
                d_v=self.value_head_dim,
            )
        else:
            joined = einx.id(
                "batch... query query_head d_v -> batch... query (query_head d_v)",
                attended,
                query_head=self.num_heads,
                d_v=self.value_head_dim,
            )
        return self.output_proj(joined)

    def extra_repr(self) -> str:
        """Return constructor-relevant attention settings for module repr."""
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, "
            f"kdim={self.kdim}, vdim={self.vdim}, qk_head_dim={self.qk_head_dim}, "
            f"value_head_dim={self.value_head_dim}, dropout={self.dropout}, rope={self.rope is not None}, "
            f"packed={self.in_proj_weight is not None}, layout={self._layout_strategy}, "
            f"qk_execution={self._qk_execution_strategy}"
        )

    def _translate_projection_state_dict(
        self,
        module: nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Translate compatible projection layouts before PyTorch loads this module."""
        del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        delegated_keys = tuple(prefix + key for key in ("q_proj.weight", "k_proj.weight", "v_proj.weight"))
        separate_keys = tuple(prefix + key for key in ("q_proj_weight", "k_proj_weight", "v_proj_weight"))
        packed_key = prefix + "in_proj_weight"
        has_delegated = any(key in state_dict for key in delegated_keys)
        has_separate = any(key in state_dict for key in separate_keys)
        has_packed = packed_key in state_dict
        if sum((has_delegated, has_separate, has_packed)) > 1:
            raise ValueError("state_dict contains conflicting packed and unpacked Q/K/V weight layouts")

        if has_delegated:
            present = [key in state_dict for key in delegated_keys]
            if self.in_proj_weight is not None:
                if not all(present):
                    raise ValueError("delegated Q/K/V weight layout is incomplete")
                q_weight, k_weight, v_weight = (state_dict[key] for key in delegated_keys)
                state_dict[packed_key] = einx.id(
                    "q input, k input, v input -> (q + k + v) input",
                    q_weight,
                    k_weight,
                    v_weight,
                    q=self.q_proj_dim,
                    k=self.k_proj_dim,
                    v=self.v_proj_dim,
                )
                for key in delegated_keys:
                    del state_dict[key]
            else:
                for source, target in zip(delegated_keys, separate_keys, strict=True):
                    if source in state_dict:
                        state_dict[target] = state_dict.pop(source)
        elif has_separate:
            present = [key in state_dict for key in separate_keys]
            if self.in_proj_weight is not None:
                if not all(present):
                    raise ValueError("separate Q/K/V weight layout is incomplete for packed storage")
                q_weight, k_weight, v_weight = (state_dict[key] for key in separate_keys)
                state_dict[packed_key] = einx.id(
                    "q input, k input, v input -> (q + k + v) input",
                    q_weight,
                    k_weight,
                    v_weight,
                    q=self.q_proj_dim,
                    k=self.k_proj_dim,
                    v=self.v_proj_dim,
                )
                for key in separate_keys:
                    del state_dict[key]
        elif has_packed:
            if self.in_proj_weight is None:
                raise ValueError("cannot load packed Q/K/V weights when key or value input widths differ")


class MultiheadSelfAttention(MultiheadAttention):
    """Causal-by-default self-attention with shared query, key, and value inputs.

    ``d_model`` is the input and output width. ``d_k`` and ``d_v`` are the
    projected per-head query/key and value widths. The corresponding
    :class:`MultiheadAttention` names are available as keyword-only aliases.
    ``num_kv_heads`` selects ordinary MHA, GQA, or MQA in the same way as on
    :class:`MultiheadAttention`.

    Note:
        This class inherits from :class:`MultiheadAttention` to reuse its
        implementation and parameter representation, but it is intentionally
        not a substitutable subtype: its constructor uses course notation, its
        forward method accepts one shared input, and causal attention is the
        default.
    """

    type ModelVec = Float[torch.Tensor, "{self.d_model}"]  # noqa: F821

    @property
    def d_model(self) -> int:
        """Model input and output width."""
        return self.embed_dim

    @property
    def d_k(self) -> int:
        """Projected query/key width per head."""
        return self.qk_head_dim

    @property
    def d_v(self) -> int:
        """Projected value width per head."""
        return self.value_head_dim

    @overload
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int | None = None,
        d_v: int | None = None,
        dropout: float = 0.0,
        *,
        num_q_heads: None = None,
        num_kv_heads: int | None = None,
        embed_dim: None = None,
        qk_head_dim: int | None = None,
        value_head_dim: int | None = None,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        _layout_strategy: Literal["head_before_sequence", "head_after_sequence"] = "head_before_sequence",
        _qk_execution_strategy: Literal["auto", "separate", "stacked"] = "auto",
    ) -> None: ...

    @overload
    def __init__(
        self,
        d_model: int,
        *,
        num_q_heads: int,
        num_kv_heads: int | None = None,
        d_k: int | None = None,
        d_v: int | None = None,
        dropout: float = 0.0,
        embed_dim: None = None,
        qk_head_dim: int | None = None,
        value_head_dim: int | None = None,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        _layout_strategy: Literal["head_before_sequence", "head_after_sequence"] = "head_before_sequence",
        _qk_execution_strategy: Literal["auto", "separate", "stacked"] = "auto",
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_q_heads: None = None,
        num_kv_heads: int | None = None,
        d_k: int | None = None,
        d_v: int | None = None,
        qk_head_dim: int | None = None,
        value_head_dim: int | None = None,
        dropout: float = 0.0,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        _layout_strategy: Literal["head_before_sequence", "head_after_sequence"] = "head_before_sequence",
        _qk_execution_strategy: Literal["auto", "separate", "stacked"] = "auto",
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int | None = None,
        d_k: int | None = None,
        d_v: int | None = None,
        qk_head_dim: int | None = None,
        value_head_dim: int | None = None,
        dropout: float = 0.0,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        _layout_strategy: Literal["head_before_sequence", "head_after_sequence"] = "head_before_sequence",
        _qk_execution_strategy: Literal["auto", "separate", "stacked"] = "auto",
    ) -> None: ...

    def __init__(
        self,
        d_model: int | None = None,
        num_heads: int | None = None,
        d_k: int | None = None,
        d_v: int | None = None,
        dropout: float = 0.0,
        *,
        num_q_heads: int | None = None,
        num_kv_heads: int | None = None,
        embed_dim: int | None = None,
        qk_head_dim: int | None = None,
        value_head_dim: int | None = None,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        _layout_strategy: Literal["head_before_sequence", "head_after_sequence"] = "head_before_sequence",
        _qk_execution_strategy: Literal["auto", "separate", "stacked"] = "auto",
    ) -> None:
        model_dim = self._coalesce_alias("d_model", d_model, "embed_dim", embed_dim)
        query_heads = self._coalesce_alias("num_heads", num_heads, "num_q_heads", num_q_heads)
        projected_qk_dim = self._coalesce_alias("d_k", d_k, "qk_head_dim", qk_head_dim)
        projected_value_dim = self._coalesce_alias("d_v", d_v, "value_head_dim", value_head_dim)
        if model_dim is None:
            raise TypeError("missing model width: supply d_model or embed_dim")
        if query_heads is None:
            raise TypeError("missing query-head count: supply num_heads or num_q_heads")

        super().__init__(
            embed_dim=model_dim,
            num_heads=query_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            kdim=model_dim,
            vdim=model_dim,
            qk_head_dim=projected_qk_dim,
            value_head_dim=projected_value_dim,
            rope=rope,
            device=device,
            dtype=dtype,
            _layout_strategy=_layout_strategy,
            _qk_execution_strategy=_qk_execution_strategy,
        )

    def forward(  # ty: ignore[invalid-method-override]
        self,
        x: Shaped[ModelVec, "*batch sequence"],
        mask: MaskBias[torch.Tensor, "*slew sequence sequence"] | None = None,  # ty: ignore[invalid-syntax-in-forward-annotation]
        *,
        token_positions: Int[torch.Tensor, "*position_batch sequence"] | None = None,
        is_causal: bool = True,
        mask_layout: _MaskLayout = "batch",
        position_layout: _PositionLayout = "batch",
    ) -> Shaped[ModelVec, "*batch sequence"]:
        """Apply self-attention with shared Q/K/V inputs, masks, and positions."""
        return super().forward(
            x,
            x,
            x,
            mask,
            is_causal=is_causal,
            mask_layout=mask_layout,
            query_positions=token_positions,
            key_positions=token_positions,
            position_layout=position_layout,
        )

    def _cost_repr(self, scope: _CostScope) -> tuple[CostRepr, ...]:
        """Describe packed projection and grouped attention at symbolic shapes."""
        s = scope.symbols
        s.unbound("batch", "sequence")
        s.bind(
            d_model=self.d_model,
            query_heads=self.num_heads,
            kv_heads=self.num_kv_heads,
            d_k=self.d_k,
            d_v=self.d_v,
        )

        tokens = s.batch * s.sequence
        group = s.query_heads / s.kv_heads
        projected_width = s.query_heads * s.d_k + s.kv_heads * s.d_k + s.kv_heads * s.d_v
        dtype = self.output_proj.weight.dtype
        return (
            CostRepr(
                "packed query, key, and value projection",
                torch.ops.aten.bmm.default,
                {
                    "self": TensorRepr((1, tokens, s.d_model), dtype),
                    "mat2": TensorRepr((1, s.d_model, projected_width), dtype),
                },
            ),
            CostRepr(
                "grouped query-key scores",
                torch.ops.aten.bmm.default,
                {
                    "self": TensorRepr((s.batch * s.kv_heads, group * s.sequence, s.d_k), dtype),
                    "mat2": TensorRepr((s.batch * s.kv_heads, s.d_k, s.sequence), dtype),
                },
            ),
            CostRepr(
                "grouped attention-value product",
                torch.ops.aten.bmm.default,
                {
                    "self": TensorRepr((s.batch * s.kv_heads, group * s.sequence, s.sequence), dtype),
                    "mat2": TensorRepr((s.batch * s.kv_heads, s.sequence, s.d_v), dtype),
                },
            ),
        )

    def _cost_call_bindings(
        self,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> Mapping[str, Any]:
        """Bind flattened batch and sequence axes from one self-attention call."""
        del output
        x = args[0] if args else kwargs["x"]
        if not isinstance(x, torch.Tensor):
            raise TypeError("self-attention cost observation requires sequence-feature tensor input")
        if x.ndim < 2:
            raise ValueError("self-attention cost observation requires sequence and feature axes")
        return {
            "batch": prod(x.shape[:-2], start=1),
            "sequence": x.shape[-2],
        }

    def _cost_children(self, scope: _CostScope) -> tuple[_CostChild, ...]:
        """Pass dimensions to the output projection and retain optional RoPE."""
        s = scope.symbols
        s.unbound("batch", "sequence")
        s.bind(query_heads=self.num_heads, d_v=self.d_v, d_model=self.d_model)
        children = [
            scope.child(
                "output_proj",
                self.output_proj,
                arguments={
                    "tokens": s.batch * s.sequence,
                    "d_in": s.query_heads * s.d_v,
                    "d_out": s.d_model,
                },
            )
        ]
        if self.rope is not None:
            children.append(scope.child("rope", self.rope))
        return tuple(children)
