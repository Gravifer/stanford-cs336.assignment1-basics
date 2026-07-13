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
        x: Shaped[KeyVec, "*mapped seq_len"],
        token_positions: Int[
            torch.Tensor, "*mapped seq_len"
        ]  # otherwise, the same positions are broadcast for all x[*mapped]
        | Int[torch.Tensor, "seq_len"],  # noqa: F821
        *,
        broadcast_positions: bool = True,  # if False, second case of token_positions is not allowed
    ) -> Shaped[KeyVec, "*mapped seq_len"]:
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
        # Ensure token_positions is of the same device as x
        token_positions = token_positions.to(x.device)

        # print the shapes
        print(f"x.shape = {x.shape}")
        print(f"token_positions.shape = {token_positions.shape} ({token_positions!r})")
        if broadcast_positions:
            # | "*mapped seq_len" -> check shapes are consistent; toggle broadcast off
            # | "seq_len" -> check shapes are consistent
            if token_positions.shape == x.shape[:-1]:
                broadcast_positions = False
            elif token_positions.shape != x.shape[-2:-1]:
                raise ValueError(
                    f"token_positions shape {token_positions.shape} is not compatible with x shape {x.shape}"
                )
        print(f"broadcast_positions = {broadcast_positions}")

        # # Compute the rotation angles for the given token positions
        # angles = self.freqs[token_positions]
        #
        # # Compute the sine and cosine components (cached)
        # sin_angles = torch.sin(angles)
        # cos_angles = torch.cos(angles)
        if broadcast_positions:
            rot = einx.get_at(
                "[max_seq_len] d_pair col row, seq_len -> seq_len d_pair col row",
                self.rot,
                token_positions,
                d_pair=self.d_pair,
                col=2,
                row=2,
            )
        else:
            rot = einx.get_at(
                "[max_seq_len] d_pair col row, mapped... seq_len -> mapped... seq_len d_pair col row",
                self.rot,
                token_positions,
                d_pair=self.d_pair,
                col=2,
                row=2,
            )
        print(f"rot.shape = {rot.shape}")

        # # Split x into even and odd parts
        # x_even = x[..., 0::2]
        # x_odd = x[..., 1::2]
        x_split = einx.id("mapped... seq_len (d_pair p) -> mapped... seq_len d_pair p", x, d_pair=self.d_pair, p=2)

        # # Apply the rotation
        # x_rotated_even = x_even * cos_angles - x_odd * sin_angles
        # x_rotated_odd = x_even * sin_angles + x_odd * cos_angles
        if broadcast_positions:
            x_split_rotated = einx.dot(
                "mapped... seq_len d_pair [row], seq_len d_pair col [row] -> mapped... seq_len d_pair col",
                x_split,
                rot,
                d_pair=self.d_pair,
                col=2,
                row=2,
            )
        else:
            x_split_rotated = einx.dot(
                "mapped... seq_len d_pair [row], mapped... seq_len d_pair col [row] -> mapped... seq_len d_pair col",
                x_split,
                rot,
                d_pair=self.d_pair,
                col=2,
                row=2,
            )

        # # Interleave the rotated even and odd parts back together
        # x_rotated = torch.stack((x_rotated_even, x_rotated_odd), dim=-1).reshape_as(x)
        x_rotated = einx.id(
            "mapped... seq_len d_pair p -> mapped... seq_len (d_pair p)", x_split_rotated, d_pair=self.d_pair, p=2
        )

        return x_rotated
