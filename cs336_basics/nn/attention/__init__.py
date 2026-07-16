import warnings

import einops
import einx
import torch
from jaxtyping import Float, Int, Shaped
from torch import nn

from cs336_basics.nn import functional as F

from .._typing import MaskBias
from ..modules import Linear


class RotaryPositionalEmbedding(nn.Module):
    """RoPE for attention, with trigs cached"""

    __constants__ = ["theta", "d_pair", "max_seq_len"]
    theta: float
    d_pair: int
    max_seq_len: int
    # sin_angles: Float[torch.Tensor, "{self.max_seq_len} {self.d_pair}"]
    # cos_angles: Float[torch.Tensor, "{self.max_seq_len} {self.d_pair}"]
    # trigs: Float[torch.Tensor, "2 {self.max_seq_len} {self.d_pair}"]
    rot: Float[torch.Tensor, "{self.max_seq_len} {self.d_pair} 2 2"]

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
        s = torch.sin(freqs)
        c = torch.cos(freqs)
        # self.register_buffer(
        #     "trigs",
        #     einx.id("max_seq_len d_pair, max_seq_len d_pair -> 2 max_seq_len d_pair", s, c),
        #     persistent=False,
        # )
        rot = einops.rearrange(
            [c, -s, s, c],
            "(col row) max_seq_len d_pair -> max_seq_len d_pair col row",
            d_pair=self.d_pair,
            col=2,
            row=2,
        )
        self.register_buffer("rot", rot, persistent=False)

    def forward(
        self,
        x: Shaped[KeyVec, "*map_batch seq_len"],
        token_positions: Int[torch.Tensor, "*batch seq_len"],
        *,
        broadcast_positions: bool = True,  # if False, second case of token_positions is not allowed
    ) -> Shaped[KeyVec, "*map_batch seq_len"]:
        """
        Apply RoPE to the input tensor `𝑥` based on token positions.

        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        The token positions are a tensor of shape (..., seq_len)
        specifying the token positions of `𝑥` along the sequence dimension.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_k)
            token_positions: Tensor of shape (batch_size, seq_len) containing the positions of tokens

        Returns:
            Tensor of shape (batch_size, seq_len, d_k) with RoPE applied
        """
        map_batch: tuple[int, ...] = tuple(x.shape[:-2])

        # Ensure token_positions is of the same device as x
        token_positions = token_positions.to(x.device)

        if token_positions.ndim == 0:
            raise ValueError("token_positions must include a sequence dimension")

        if broadcast_positions:
            suffix_start = -token_positions.ndim - 1
            target_shape = x.shape[suffix_start:-1]
            if token_positions.ndim >= x.ndim or len(token_positions.shape) != len(target_shape):
                raise ValueError(
                    f"token_positions shape {token_positions.shape} is not compatible with x shape {x.shape}"
                )
            if any(
                position_dim not in (1, target_dim)
                for position_dim, target_dim in zip(token_positions.shape, target_shape)
            ):
                raise ValueError(
                    f"token_positions shape {token_positions.shape} is not compatible with x shape {x.shape}"
                )
            token_positions = token_positions.expand(target_shape)
            mapped: tuple[int, ...] = tuple(x.shape[:suffix_start])
            batch: tuple[int, ...] = tuple(target_shape[:-1])
        elif token_positions.ndim != x.ndim - 1 or token_positions.shape != x.shape[:-1]:
            raise ValueError(
                f"token_positions shape {token_positions.shape} does not match with x shape {x.shape}; is broadcasting needed?"
            )
        else:  # *batch === *map_batch
            mapped: tuple[int, ...] = tuple()
            batch: tuple[int, ...] = map_batch

        if x.numel() == 0 and token_positions.numel() == 0:
            warnings.warn("Applying RoPE to empty tensors is a no-op", stacklevel=2)
            return x

        axes_map: dict[str, tuple[int, ...] | int] = {
            "map_batch": map_batch,
            "mapped": mapped,
            "batch": batch,
            "d_pair": self.d_pair,
            "col": 2,
            "row": 2,
        }

        # # Compute the rotation angles for the given token positions
        # angles = self.freqs[token_positions]
        #
        # # Compute the sine and cosine components (cached)
        # sin_angles = torch.sin(angles)
        # cos_angles = torch.cos(angles)
        rot = einx.get_at(
            "[max_seq_len] d_pair col row, batch... seq_len -> batch... seq_len d_pair col row",
            self.rot,
            token_positions,
            **axes_map,
        )

        in_dtype = x.dtype
        op_dtype = torch.promote_types(in_dtype, rot.dtype)
        if op_dtype not in (torch.float32, torch.float64):
            op_dtype = torch.float32
        rot = rot.to(op_dtype)

        # # Split x into even and odd parts
        # x_even = x[..., 0::2]
        # x_odd = x[..., 1::2]
        x_split = einx.id(
            "mapped... batch... seq_len (d_pair p) -> mapped... batch... seq_len d_pair p",
            x.to(op_dtype),
            **axes_map,
        )

        # # Apply the rotation
        # x_rotated_even = x_even * cos_angles - x_odd * sin_angles
        # x_rotated_odd = x_even * sin_angles + x_odd * cos_angles
        x_split_rotated = einx.dot(
            "mapped... batch... seq_len d_pair [row], batch... seq_len d_pair col [row] -> mapped... batch... seq_len d_pair col",
            x_split,
            rot,
            **axes_map,
        )

        # # Interleave the rotated even and odd parts back together
        # x_rotated = torch.stack((x_rotated_even, x_rotated_odd), dim=-1).reshape_as(x)
        x_rotated = einx.id(
            "mapped... batch... seq_len d_pair p -> mapped... batch... seq_len (d_pair p)",
            x_split_rotated,
            **axes_map,
        )

        return x_rotated.to(in_dtype)


class MultiheadAttention(nn.Module):
    """Multi-head attention over tensors shaped ``(*batch, sequence, features)``.

    ``kdim`` and ``vdim`` describe the raw key and value input widths. The
    projected per-head widths are ``qk_head_dim`` and ``value_head_dim``.
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

        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim
        self.vdim = vdim
        self.qk_head_dim = qk_head_dim
        self.value_head_dim = value_head_dim
        self.dropout = dropout
        self.rope = rope

        self.q_proj = Linear(embed_dim, num_heads * qk_head_dim, device=device, dtype=dtype)
        self.k_proj = Linear(kdim, num_heads * qk_head_dim, device=device, dtype=dtype)
        self.v_proj = Linear(vdim, num_heads * value_head_dim, device=device, dtype=dtype)
        self.output_proj = Linear(num_heads * value_head_dim, embed_dim, device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        """Reset all projection parameters."""
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.output_proj.reset_parameters()

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

        projected_q = self.q_proj(query)
        projected_k = self.k_proj(key)
        projected_v = self.v_proj(value)

        # Heads follow the ordinary batch axes so RoPE can treat them as its position-broadcast suffix.
        q = einx.id(
            "batch... query (head d_k) -> batch... head query d_k",
            projected_q,
            head=self.num_heads,
            d_k=self.qk_head_dim,
        )
        k = einx.id(
            "batch... key (head d_k) -> batch... head key d_k",
            projected_k,
            head=self.num_heads,
            d_k=self.qk_head_dim,
        )
        v = einx.id(
            "batch... key (head d_v) -> batch... head key d_v",
            projected_v,
            head=self.num_heads,
            d_v=self.value_head_dim,
        )

        if self.rope is not None:
            if query_positions is None:
                query_positions = torch.arange(query.shape[-2], device=query.device)
            if key_positions is None:
                key_positions = torch.arange(key.shape[-2], device=key.device)
            q = self.rope(q, query_positions)
            k = self.rope(k, key_positions)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=self.qk_head_dim**-0.5,
        )
        joined = einx.id(
            "batch... head query d_v -> batch... query (head d_v)",
            attended,
            head=self.num_heads,
            d_v=self.value_head_dim,
        )
        return self.output_proj(joined)

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"kdim={self.kdim}, vdim={self.vdim}, qk_head_dim={self.qk_head_dim}, "
            f"value_head_dim={self.value_head_dim}, dropout={self.dropout}, rope={self.rope is not None}"
        )


class MultiheadSelfAttention(MultiheadAttention):
    """Causal-by-default self-attention specialization of :class:`MultiheadAttention`."""

    type ModelVec = Float[torch.Tensor, "{self.d_model}"]  # noqa: F821

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        *,
        qk_head_dim: int | None = None,
        value_head_dim: int | None = None,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            kdim=d_model,
            vdim=d_model,
            qk_head_dim=qk_head_dim,
            value_head_dim=value_head_dim,
            rope=rope,
            device=device,
            dtype=dtype,
        )
        self.d_model = d_model

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
