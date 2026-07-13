import einops
import einx
import torch
from beartype import beartype
from jaxtyping import Float, Int, Shaped, jaxtyped
from torch import nn


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
        inv_freq: self.HalfKey = 1.0 / (  # noqa: UP037
            self.theta ** (torch.arange(0, d_k, 2, device=device) / d_k)
        )
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

    @jaxtyped(typechecker=beartype)
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

        if broadcast_positions:
            if token_positions.shape == x.shape[:-1]:
                broadcast_positions = False
                mapped: tuple[int, ...] = tuple()
                batch: tuple[int, ...] = map_batch
            elif token_positions.ndim >= x.ndim or token_positions.shape != x.shape[token_positions.ndim - x.ndim : -1]:
                raise ValueError(
                    f"token_positions shape {token_positions.shape} is not compatible with x shape {x.shape}"
                )
            else:  # *batch is a valid suffix of *map_batch, so we can broadcast
                mapped: tuple[int, ...] = tuple(x.shape[: token_positions.ndim - x.ndim])
                batch: tuple[int, ...] = tuple(token_positions.shape[:-1])
        elif token_positions.ndim != x.ndim - 1 or token_positions.shape != x.shape[:-1]:
            raise ValueError(
                f"token_positions shape {token_positions.shape} does not match with x shape {x.shape}; is broadcasting needed?"
            )
        else:  # *batch === *map_batch
            mapped: tuple[int, ...] = tuple()
            batch: tuple[int, ...] = map_batch
        shape_dict: dict[str, tuple[int, ...] | int] = {
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
            **shape_dict,
        )

        # # Split x into even and odd parts
        # x_even = x[..., 0::2]
        # x_odd = x[..., 1::2]
        x_split = einx.id(
            "mapped... batch... seq_len (d_pair p) -> mapped... batch... seq_len d_pair p", x, **shape_dict
        )

        # # Apply the rotation
        # x_rotated_even = x_even * cos_angles - x_odd * sin_angles
        # x_rotated_odd = x_even * sin_angles + x_odd * cos_angles
        x_split_rotated = einx.dot(
            "mapped... batch... seq_len d_pair [row], batch... seq_len d_pair col [row] -> mapped... batch... seq_len d_pair col",
            x_split,
            rot,
            **shape_dict,
        )

        # # Interleave the rotated even and odd parts back together
        # x_rotated = torch.stack((x_rotated_even, x_rotated_odd), dim=-1).reshape_as(x)
        x_rotated = einx.id(
            "mapped... batch... seq_len d_pair p -> mapped... batch... seq_len (d_pair p)",
            x_split_rotated,
            **shape_dict,
        )

        return x_rotated
