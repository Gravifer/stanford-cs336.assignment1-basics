import warnings
from collections.abc import Mapping
from typing import Any, Literal, overload

import einops
import einx
import torch
from jaxtyping import Float, Int, Shaped
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys

from cs336_basics.nn import functional as F

from .._typing import MaskBias
from .. import initializer as init
from ..modules import Linear


class RotaryPositionalEmbedding(nn.Module):
    """RoPE for attention, with cached rotation data.

    Matrix-form caches store a 2-by-2 rotation for every frequency and are
    therefore twice the size of the equivalent cosine/sine caches.
    """

    __constants__ = ["theta", "d_pair", "max_seq_len", "use_matrix_form"]
    theta: float
    d_pair: int
    max_seq_len: int
    # sin_angles: Float[torch.Tensor, "{self.max_seq_len} {self.d_pair}"]
    # cos_angles: Float[torch.Tensor, "{self.max_seq_len} {self.d_pair}"]
    # trigs: Float[torch.Tensor, "2 {self.max_seq_len} {self.d_pair}"]
    use_matrix_form: bool
    rot: Float[torch.Tensor, "{self.max_seq_len} {self.d_pair} 2 2"] | None
    cos: Float[torch.Tensor, "{self.max_seq_len} {self.d_pair}"] | None
    sin: Float[torch.Tensor, "{self.max_seq_len} {self.d_pair}"] | None

    type KeyVec = Float[torch.Tensor, "{self.d_k}"]  # noqa: F821
    type HalfKey = Float[torch.Tensor, "{self.d_pair}"]  # noqa: F821

    @property
    def d_k(self) -> int:
        return 2 * self.d_pair

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device
        | None = None,  # usual torch convention is to keep construction of this device-agnostic; we are following the handout-signature
        *,
        use_matrix_form: bool = True,
    ):
        """
        theta        Θ value for the RoPE
        d_k          dimension of query and key vectors
        max_seq_len  Maximum sequence length that will be input
        device       Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        if d_k % 2:
            raise ValueError("RoPE expects even dimensional queries and keys")
        self.d_pair = d_k // 2
        self.max_seq_len = max_seq_len
        self.use_matrix_form = use_matrix_form

        # Precompute the rotation angles for efficiency
        inv_freq: self.HalfKey = 1.0 / (self.theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        freqs: Shaped[self.HalfKey, "{self.max_seq_len}"] = einx.multiply(  # noqa: UP037
            "max_seq_len, d_pair -> max_seq_len d_pair",
            torch.arange(max_seq_len, device=device),
            inv_freq,
            d_pair=self.d_pair,
        )
        # self.register_buffer("sin_angles", freqs.sin(), persistent=False)
        # self.register_buffer("cos_angles", freqs.cos(), persistent=False)
        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        if use_matrix_form:
            rot = einops.rearrange(
                [cos, -sin, sin, cos],
                "(col row) max_seq_len d_pair -> max_seq_len d_pair col row",
                d_pair=self.d_pair,
                col=2,
                row=2,
            )
            self.register_buffer("rot", rot, persistent=False)
            self.cos = None
            self.sin = None
        else:
            self.rot = None
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _position_target_shape(
        x: torch.Tensor,
        layout_strategy: Literal["head_before_sequence", "head_after_sequence"],
    ) -> tuple[int, ...]:
        if layout_strategy == "head_before_sequence":
            return tuple(x.shape[:-1])
        if layout_strategy == "head_after_sequence":
            if x.ndim < 3:
                raise ValueError("head-after-sequence RoPE inputs must include sequence and head dimensions")
            return (*x.shape[:-3], x.shape[-2], x.shape[-3])
        raise ValueError(f"unknown RoPE layout strategy: {layout_strategy!r}")

    def _validate_token_positions(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
        *,
        broadcast_positions: bool,
        layout_strategy: Literal["head_before_sequence", "head_after_sequence"],
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        target_shape = self._position_target_shape(x, layout_strategy)
        token_positions = token_positions.to(x.device)
        if token_positions.ndim == 0:
            raise ValueError("token_positions must include a sequence dimension")

        if broadcast_positions:
            if token_positions.ndim > len(target_shape):
                raise ValueError(
                    f"token_positions shape {token_positions.shape} is not compatible with x shape {x.shape}"
                )
            suffix_shape = target_shape[-token_positions.ndim :]
            if token_positions.shape[-1] != suffix_shape[-1]:
                raise ValueError(
                    f"token_positions shape {token_positions.shape} is not compatible with x shape {x.shape}"
                )
            try:
                # Validation only: retaining the unexpanded index avoids gathering
                # duplicate cache blocks for singleton head or batch dimensions.
                token_positions.expand(suffix_shape)
            except RuntimeError as error:
                raise ValueError(
                    f"token_positions shape {token_positions.shape} is not compatible with x shape {x.shape}"
                ) from error
        elif tuple(token_positions.shape) != target_shape:
            raise ValueError(
                f"token_positions shape {token_positions.shape} does not match with x shape {x.shape}; is broadcasting needed?"
            )
        return token_positions, target_shape

    def _select_rotations(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None,
        *,
        broadcast_positions: bool,
        layout_strategy: Literal["head_before_sequence", "head_after_sequence"],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        target_shape = self._position_target_shape(x, layout_strategy)
        if token_positions is None:
            seq_len = target_shape[-1]
            if seq_len > self.max_seq_len:
                raise ValueError(f"sequence length {seq_len} exceeds the RoPE cache length {self.max_seq_len}")
            index_shape = (seq_len,)
            prefix_ndim = len(target_shape) - 1
        else:
            token_positions, target_shape = self._validate_token_positions(
                x,
                token_positions,
                broadcast_positions=broadcast_positions,
                layout_strategy=layout_strategy,
            )
            index_shape = tuple(token_positions.shape)
            prefix_ndim = len(target_shape) - token_positions.ndim

        def select(cache: torch.Tensor) -> torch.Tensor:
            selected = cache[: target_shape[-1]] if token_positions is None else cache[token_positions]
            cache_shape = tuple(cache.shape[1:])
            selected = selected.reshape((1,) * prefix_ndim + index_shape + cache_shape)
            return selected.expand(target_shape + cache_shape)

        if self.use_matrix_form:
            assert self.rot is not None, "matrix-form RoPE must own a rotation-matrix cache"
            return select(self.rot)
        assert self.cos is not None, "elementwise RoPE must own a cosine cache"
        assert self.sin is not None, "elementwise RoPE must own a sine cache"
        return select(self.cos), select(self.sin)

    @staticmethod
    def _selection_to_physical_layout(
        selection: torch.Tensor,
        *,
        cache_ndim: int,
        layout_strategy: Literal["head_before_sequence", "head_after_sequence"],
    ) -> torch.Tensor:
        if layout_strategy == "head_before_sequence":
            return selection
        return selection.transpose(-cache_ndim - 2, -cache_ndim - 1)

    def _apply_rotations(
        self,
        x: torch.Tensor,
        selection: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        *,
        layout_strategy: Literal["head_before_sequence", "head_after_sequence"],
    ) -> torch.Tensor:
        in_dtype = x.dtype
        cache = selection if isinstance(selection, torch.Tensor) else selection[0]
        op_dtype = torch.promote_types(in_dtype, cache.dtype)
        if op_dtype not in (torch.float32, torch.float64):
            op_dtype = torch.float32
        x_split = einx.id("... (d_pair p) -> ... d_pair p", x.to(op_dtype), d_pair=self.d_pair, p=2)

        if isinstance(selection, torch.Tensor):
            rot = self._selection_to_physical_layout(
                selection,
                cache_ndim=3,
                layout_strategy=layout_strategy,
            ).to(op_dtype)
            x_split_rotated = einx.dot(
                "... d_pair [row], ... d_pair col [row] -> ... d_pair col",
                x_split,
                rot,
                d_pair=self.d_pair,
                col=2,
                row=2,
            )
        else:
            cos, sin = (
                self._selection_to_physical_layout(part, cache_ndim=1, layout_strategy=layout_strategy).to(op_dtype)
                for part in selection
            )
            x_even, x_odd = x_split[..., 0], x_split[..., 1]
            x_split_rotated = torch.stack(
                (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
                dim=-1,
            )
        return einx.id(
            "... d_pair p -> ... (d_pair p)",
            x_split_rotated,
            d_pair=self.d_pair,
            p=2,
        ).to(in_dtype)

    @staticmethod
    def _prepend_selection_axis(
        selection: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        size: int,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        def prepend(x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(0).expand((size, *x.shape))

        return (
            prepend(selection)
            if isinstance(selection, torch.Tensor)
            else (prepend(selection[0]), prepend(selection[1]))
        )

    def forward(
        self,
        x: Shaped[KeyVec, "*leading seq_len"],
        token_positions: Int[torch.Tensor, "*slew seq_len"],
        *,
        broadcast_positions: bool = True,  # if False, second case of token_positions is not allowed
    ) -> Shaped[KeyVec, "*leading seq_len"]:
        """
        Apply RoPE to the input tensor `𝑥` based on token positions.

        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        With position broadcasting, the leading axes are interpreted as
        ``(*mapped, *slew)``: token positions cover the ``*slew`` suffix, while
        the selected rotations are reused over the ``*mapped`` prefix.

        Args:
            x: Input tensor of shape (*mapped, *slew, seq_len, d_k).
            token_positions: Token positions of shape (*slew, seq_len).

        Returns:
            Tensor with the same shape as ``x`` and RoPE applied.
        """
        selection = self._select_rotations(
            x,
            token_positions,
            broadcast_positions=broadcast_positions,
            layout_strategy="head_before_sequence",
        )
        if x.numel() == 0 and token_positions.numel() == 0:
            warnings.warn("Applying RoPE to empty tensors is a no-op", stacklevel=2)
            return x
        return self._apply_rotations(x, selection, layout_strategy="head_before_sequence")

    def apply_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        *,
        broadcast_positions: bool = True,
        _stacked: bool = False,
        _layout_strategy: Literal["head_before_sequence", "head_after_sequence"] = "head_before_sequence",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one shared positional selection to query and key tensors.

        ``None`` selects consecutive zero-based positions by slicing the cache.
        ``_stacked`` is an allocation-forcing benchmark control, not an
        automatically selected execution policy.
        """
        if query.shape != key.shape:
            raise ValueError(f"query and key must have equal shapes for shared RoPE, got {query.shape} and {key.shape}")
        selection = self._select_rotations(
            query,
            token_positions,
            broadcast_positions=broadcast_positions,
            layout_strategy=_layout_strategy,
        )
        if query.numel() == 0 and (token_positions is None or token_positions.numel() == 0):
            warnings.warn("Applying RoPE to empty tensors is a no-op", stacklevel=2)
            return query, key
        if not _stacked:
            return (
                self._apply_rotations(query, selection, layout_strategy=_layout_strategy),
                self._apply_rotations(key, selection, layout_strategy=_layout_strategy),
            )
        qk = torch.stack((query, key), dim=0)
        stacked_selection = self._prepend_selection_axis(selection, 2)
        qk = self._apply_rotations(qk, stacked_selection, layout_strategy=_layout_strategy)
        return qk[0], qk[1]


class MultiheadAttention(nn.Module):
    """Multi-head attention over tensors shaped ``(*batch, sequence, features)``.

    ``kdim`` and ``vdim`` describe the raw key and value input widths. The
    projected per-head widths are ``qk_head_dim`` and ``value_head_dim``.
    These pairs are independent: the K projection has shape
    ``(num_heads * qk_head_dim, kdim)``, while the V projection has shape
    ``(num_heads * value_head_dim, vdim)``.
    Boolean masks follow :func:`cs336_basics.nn.functional.scaled_dot_product_attention`:
    ``True`` permits attention and ``False`` masks it.
    """

    __constants__ = [
        "embed_dim",
        "num_heads",
        "kdim",
        "vdim",
        "qk_head_dim",
        "value_head_dim",
        "dropout",
        "q_proj_dim",
        "k_proj_dim",
        "v_proj_dim",
        "_qkv_same_input_dim",
        "_layout_strategy",
        "_qk_execution_strategy",
    ]

    type QueryVec = Float[torch.Tensor, "{self.embed_dim}"]  # noqa: F821
    type KeyVec = Float[torch.Tensor, "{self.kdim}"]  # noqa: F821
    type ValueVec = Float[torch.Tensor, "{self.vdim}"]  # noqa: F821
    type OutputVec = Float[torch.Tensor, "{self.embed_dim}"]  # noqa: F821

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        *,
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
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                "embed_dim and num_heads must be greater than 0, "
                f"got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
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
        self.kdim = kdim
        self.vdim = vdim
        self.qk_head_dim = qk_head_dim
        self.value_head_dim = value_head_dim
        self.dropout = dropout
        self.rope = rope
        self.q_proj_dim = num_heads * qk_head_dim
        self.k_proj_dim = num_heads * qk_head_dim
        self.v_proj_dim = num_heads * value_head_dim
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
        self.reset_parameters()

    def _separate_projection_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            return projected_q, projected_k, projected_v, projected_qk

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
        if self._layout_strategy == "head_before_sequence":
            q_description = "batch... query (head d_k) -> batch... head query d_k"
            k_description = "batch... key (head d_k) -> batch... head key d_k"
            v_description = "batch... key (head d_v) -> batch... head key d_v"
            qk_description = "batch... query (qk head d_k) -> qk batch... head query d_k"
        else:
            q_description = "batch... query (head d_k) -> batch... query head d_k"
            k_description = "batch... key (head d_k) -> batch... key head d_k"
            v_description = "batch... key (head d_v) -> batch... key head d_v"
            qk_description = "batch... query (qk head d_k) -> qk batch... query head d_k"

        qk = None
        if projected_qk is not None:
            qk = einx.id(
                qk_description,
                projected_qk,
                qk=2,
                head=self.num_heads,
                d_k=self.qk_head_dim,
            )
            q, k = qk[0], qk[1]
        else:
            q = einx.id(
                q_description,
                projected_q,
                head=self.num_heads,
                d_k=self.qk_head_dim,
            )
            k = einx.id(
                k_description,
                projected_k,
                head=self.num_heads,
                d_k=self.qk_head_dim,
            )
        v = einx.id(
            v_description,
            projected_v,
            head=self.num_heads,
            d_v=self.value_head_dim,
        )
        return q, k, v, qk

    def _apply_single_rope(self, x: torch.Tensor, token_positions: torch.Tensor | None) -> torch.Tensor:
        assert self.rope is not None, "RoPE application requires a registered RotaryPositionalEmbedding"
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.rope is not None, "Q/K RoPE application requires a registered RotaryPositionalEmbedding"
        shared_positions = q.shape == k.shape and (
            (query_positions is None and key_positions is None) or query_positions is key_positions
        )
        if not shared_positions:
            if self._qk_execution_strategy == "stacked":
                raise ValueError("stacked Q/K RoPE requires equal shapes and shared token positions")
            return self._apply_single_rope(q, query_positions), self._apply_single_rope(k, key_positions)

        positions = query_positions
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

    def forward(
        self,
        query: Shaped[QueryVec, "*batch query_len"],
        key: Shaped[KeyVec, "*batch key_len"],
        value: Shaped[ValueVec, "*batch key_len"],
        mask: MaskBias[torch.Tensor, "*slew query_len key_len"] | None = None,  # ty: ignore[invalid-syntax-in-forward-annotation]
        *,
        is_causal: bool = False,
        query_positions: Int[torch.Tensor, "*query_position_batch query_len"] | None = None,
        key_positions: Int[torch.Tensor, "*key_position_batch key_len"] | None = None,
    ) -> Shaped[OutputVec, "*batch query_len"]:
        """Project Q/K/V, apply optional RoPE, attend, and project the result."""
        self._validate_inputs(query, key, value)

        projected_q, projected_k, projected_v, projected_qk = self._in_projection(query, key, value)
        q, k, v, packed_qk = self._split_heads(projected_q, projected_k, projected_v, projected_qk)

        if self.rope is not None:
            q, k = self._apply_rope(q, k, packed_qk, query_positions, key_positions)

        attention = (
            F.scaled_dot_product_attention
            if self._layout_strategy == "head_before_sequence"
            else F._scaled_dot_product_attention_head_after_sequence
        )
        attended = attention(
            q,
            k,
            v,
            mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=self.qk_head_dim**-0.5,
        )
        if self._layout_strategy == "head_before_sequence":
            joined = einx.id(
                "batch... head query d_v -> batch... query (head d_v)",
                attended,
                head=self.num_heads,
                d_v=self.value_head_dim,
            )
        else:
            joined = einx.id(
                "batch... query head d_v -> batch... query (head d_v)",
                attended,
                head=self.num_heads,
                d_v=self.value_head_dim,
            )
        return self.output_proj(joined)

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"kdim={self.kdim}, vdim={self.vdim}, qk_head_dim={self.qk_head_dim}, "
            f"value_head_dim={self.value_head_dim}, dropout={self.dropout}, rope={self.rope is not None}, "
            f"packed={self.in_proj_weight is not None}, layout={self._layout_strategy}, "
            f"qk_execution={self._qk_execution_strategy}"
        )

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> _IncompatibleKeys:
        """Load native weights or translate the earlier delegated projection layout."""
        delegated_keys = ("q_proj.weight", "k_proj.weight", "v_proj.weight")
        separate_keys = ("q_proj_weight", "k_proj_weight", "v_proj_weight")
        has_delegated = any(key in state_dict for key in delegated_keys)
        has_separate = any(key in state_dict for key in separate_keys)
        has_packed = "in_proj_weight" in state_dict
        if sum((has_delegated, has_separate, has_packed)) > 1:
            raise ValueError("state_dict contains conflicting packed and unpacked Q/K/V weight layouts")

        translated: dict[str, Any] = {}
        source_keys: tuple[str, str, str] | None = None
        if has_delegated:
            source_keys = delegated_keys
        elif has_separate:
            source_keys = separate_keys

        if source_keys is not None:
            present = [key in state_dict for key in source_keys]
            if all(present):
                q_weight, k_weight, v_weight = (state_dict[key] for key in source_keys)
                if self.in_proj_weight is not None:
                    translated["in_proj_weight"] = einx.id(
                        "q input, k input, v input -> (q + k + v) input",
                        q_weight,
                        k_weight,
                        v_weight,
                        q=self.q_proj_dim,
                        k=self.k_proj_dim,
                        v=self.v_proj_dim,
                    )
                else:
                    translated.update(
                        q_proj_weight=q_weight,
                        k_proj_weight=k_weight,
                        v_proj_weight=v_weight,
                    )
        elif has_packed:
            if self.in_proj_weight is None:
                raise ValueError("cannot load packed Q/K/V weights when key or value input widths differ")
            translated["in_proj_weight"] = state_dict["in_proj_weight"]

        known_projection_keys = {"in_proj_weight", *delegated_keys, *separate_keys}
        translated.update((key, value) for key, value in state_dict.items() if key not in known_projection_keys)
        return super().load_state_dict(translated, strict, assign)


class MultiheadSelfAttention(MultiheadAttention):
    """Causal-by-default self-attention with shared query, key, and value inputs.

    ``d_model`` is the input and output width. ``d_k`` and ``d_v`` are the
    projected per-head query/key and value widths. The corresponding
    :class:`MultiheadAttention` names are available as keyword-only aliases.

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
        return self.embed_dim

    @property
    def d_k(self) -> int:
        return self.qk_head_dim

    @property
    def d_v(self) -> int:
        return self.value_head_dim

    @staticmethod
    def _coalesce_dimension_alias(
        course_name: str,
        course_value: int | None,
        mha_name: str,
        mha_value: int | None,
    ) -> int | None:
        if course_value is not None and mha_value is not None:
            raise ValueError(f"{course_name} and {mha_name} are mutually exclusive")
        return course_value if course_value is not None else mha_value

    @overload
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int | None = None,
        d_v: int | None = None,
        dropout: float = 0.0,
        *,
        embed_dim: int | None = None,
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
        embed_dim: int | None = None,
        qk_head_dim: int | None = None,
        value_head_dim: int | None = None,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        _layout_strategy: Literal["head_before_sequence", "head_after_sequence"] = "head_before_sequence",
        _qk_execution_strategy: Literal["auto", "separate", "stacked"] = "auto",
    ) -> None:
        model_dim = self._coalesce_dimension_alias("d_model", d_model, "embed_dim", embed_dim)
        projected_qk_dim = self._coalesce_dimension_alias("d_k", d_k, "qk_head_dim", qk_head_dim)
        projected_value_dim = self._coalesce_dimension_alias("d_v", d_v, "value_head_dim", value_head_dim)
        if model_dim is None:
            raise TypeError("missing model width: supply d_model or embed_dim")
        if num_heads is None:
            raise TypeError("missing required argument: num_heads")

        super().__init__(
            embed_dim=model_dim,
            num_heads=num_heads,
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
    ) -> Shaped[ModelVec, "*batch sequence"]:
        """Apply self-attention with shared Q/K/V inputs and positions."""
        return super().forward(
            x,
            x,
            x,
            mask,
            is_causal=is_causal,
            query_positions=token_positions,
            key_positions=token_positions,
        )
