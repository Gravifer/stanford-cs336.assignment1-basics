"""Rotary positional embeddings for headed attention activations."""

import warnings
from typing import Final, Literal

import einops
import einx
import torch
from jaxtyping import Float, Int, Shaped

from cs336_basics.nn.analytics import CostRepr, _CostScope
from cs336_basics.nn.modules import Module


class RotaryPositionalEmbedding(Module):
    """RoPE for attention, with cached rotation data.

    Elementwise cosine/sine rotation is the default. Optional matrix-form
    caches store a 2-by-2 rotation for every frequency and are therefore twice
    the size of the equivalent cosine/sine caches.
    """

    theta: Final[float]
    d_pair: Final[int]
    max_seq_len: Final[int]
    # sin_angles: Float[torch.Tensor, "{self.max_seq_len} {self.d_pair}"]
    # cos_angles: Float[torch.Tensor, "{self.max_seq_len} {self.d_pair}"]
    # trigs: Float[torch.Tensor, "2 {self.max_seq_len} {self.d_pair}"]
    use_matrix_form: Final[bool]
    rot: Float[torch.Tensor, "{self.max_seq_len} {self.d_pair} 2 2"] | None
    cos: Float[torch.Tensor, "{self.max_seq_len} {self.d_pair}"] | None
    sin: Float[torch.Tensor, "{self.max_seq_len} {self.d_pair}"] | None

    type KeyVec = Float[torch.Tensor, "{self.d_k}"]  # noqa: F821
    type HalfKey = Float[torch.Tensor, "{self.d_pair}"]  # noqa: F821

    @property
    def d_k(self) -> int:
        """Query/key feature width rotated by this module."""
        return 2 * self.d_pair

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device
        | None = None,  # usual torch convention is to keep construction of this device-agnostic; we are following the handout-signature
        *,
        use_matrix_form: bool = False,
    ):
        """
        theta        Θ value for the RoPE
        d_k          dimension of query and key vectors
        max_seq_len  Maximum sequence length that will be input
        device       Device to store the buffer on
        use_matrix_form  Use cached 2-by-2 matrices instead of the default cosine/sine form
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
        *,
        layout_strategy: Literal["head_before_sequence", "head_after_sequence"],
    ) -> tuple[int, ...]:
        """Return logical position axes for the selected head layout."""
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
        """Validate positions while retaining their smallest broadcastable view."""
        target_shape = self._position_target_shape(x, layout_strategy=layout_strategy)
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
        """Select cached rotations without duplicating broadcastable indices."""
        target_shape = self._position_target_shape(x, layout_strategy=layout_strategy)
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

    def _apply_rotations(
        self,
        x: torch.Tensor,
        selection: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        *,
        layout_strategy: Literal["head_before_sequence", "head_after_sequence"],
    ) -> torch.Tensor:
        """Apply selected matrix or cosine/sine rotations in a stable dtype."""
        in_dtype = x.dtype
        cache = selection if isinstance(selection, torch.Tensor) else selection[0]
        op_dtype = torch.promote_types(in_dtype, cache.dtype)
        if op_dtype not in (torch.float32, torch.float64):
            op_dtype = torch.float32
        x_split = einx.id("... (d_pair p) -> ... d_pair p", x.to(op_dtype), d_pair=self.d_pair, p=2)

        if isinstance(selection, torch.Tensor):
            rot = (
                selection
                if layout_strategy == "head_before_sequence"
                else einx.id(
                    "mapped... head sequence trailing... -> mapped... sequence head trailing...",
                    selection,
                    mapped=tuple(x.shape[:-3]),
                    trailing=(self.d_pair, 2, 2),
                )
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
                (
                    part
                    if layout_strategy == "head_before_sequence"
                    else einx.id(
                        "mapped... head sequence trailing... -> mapped... sequence head trailing...",
                        part,
                        mapped=tuple(x.shape[:-3]),
                        trailing=(self.d_pair,),
                    )
                ).to(op_dtype)
                for part in selection
            )
            x_even, x_odd = x_split[..., 0], x_split[..., 1]
            x_split_rotated = einops.rearrange(
                [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos],
                "pair ... -> ... pair",
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
        """Share selected rotation data across a newly stacked Q/K axis."""

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

    def _cost_repr(self, scope: _CostScope) -> tuple[CostRepr, ...] | None:
        """Classify default elementwise RoPE as containing no matrix products."""
        return None if self.use_matrix_form else ()

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
        qk = einops.rearrange([query, key], "qk ... -> qk ...")
        stacked_selection = self._prepend_selection_axis(selection, 2)
        qk = self._apply_rotations(qk, stacked_selection, layout_strategy=_layout_strategy)
        return qk[0], qk[1]
