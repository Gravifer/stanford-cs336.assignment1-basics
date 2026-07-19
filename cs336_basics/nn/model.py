"""Decoder-only Transformer language-model modules."""

from collections.abc import Callable
from typing import Any, cast

import einx
import torch
from jaxtyping import Float, Int
from torch import nn

from cs336_basics.nn.attention import MultiheadSelfAttention, RotaryPositionalEmbedding
from cs336_basics.nn.analytics import Module
from cs336_basics.nn.feed_forward import SwiGLU
from cs336_basics.nn.modules import Embedding, Linear, RMSNorm


__all__ = [
    "DeltaLayer",
    "GPTDecoderLayer",
    "TransformerBlock",
    "TransformerLM",
]


type ModelActivations = Float[torch.Tensor, "*batch sequence_length d_model"]
type TokenIndices = Int[torch.Tensor, "*batch sequence_length"]
type Logits = Float[torch.Tensor, "*batch sequence_length vocab_size"]


def DeltaLayer[ModuleT: nn.Module](module_type: type[ModuleT]) -> type[ModuleT]:  # noqa: N802
    """Turn an ordinary module forward pass into an additive layer.

    The module's authored ``forward`` is exposed as ``delta``. The decorator
    replaces the public forward pass with ``forward(x) = x + delta(x)``. It
    introduces no wrapper module, parameters, child prefixes, or state-loading
    behavior.
    """
    delta = cast(Callable[..., torch.Tensor] | None, module_type.__dict__.get("forward"))
    if delta is None:
        raise TypeError("a DeltaLayer must define its own forward method")
    if "delta" in module_type.__dict__:
        raise TypeError("a DeltaLayer cannot define delta separately from forward")

    def forward(self: ModuleT, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        update = delta(self, x, *args, **kwargs)
        return einx.add("... d_model, ... d_model -> ... d_model", x, update)

    setattr(module_type, "delta", delta)
    setattr(module_type, "forward", forward)
    return module_type


_COURSE_BLOCK_KEY_TRANSLATION = {
    "ln1.weight": "attn.norm.weight",
    "attn.q_proj.weight": "attn.update.q_proj.weight",
    "attn.k_proj.weight": "attn.update.k_proj.weight",
    "attn.v_proj.weight": "attn.update.v_proj.weight",
    "attn.output_proj.weight": "attn.update.output_proj.weight",
    "ln2.weight": "ffn.norm.weight",
    "ffn.w1.weight": "ffn.update.w1.weight",
    "ffn.w2.weight": "ffn.update.w2.weight",
    "ffn.w3.weight": "ffn.update.w3.weight",
}


class GPTDecoderLayer(Module):
    """Pre-norm GPT decoder layer composed of two additive updates."""

    @DeltaLayer
    class Attention(Module):
        """Normalized causal self-attention update."""

        def __init__(self, norm: RMSNorm, update: MultiheadSelfAttention) -> None:
            super().__init__()
            self.norm = norm
            self.update = update

        def forward(self, x: ModelActivations) -> ModelActivations:
            """Compute the normalized attention update."""
            return self.update(self.norm(x))

    @DeltaLayer
    class FeedForward(Module):
        """Normalized position-wise SwiGLU update."""

        def __init__(self, norm: RMSNorm, update: SwiGLU) -> None:
            super().__init__()
            self.norm = norm
            self.update = update

        def forward(self, x: ModelActivations) -> ModelActivations:
            """Compute the normalized feed-forward update."""
            return self.update(self.norm(x))

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        *,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if d_model % num_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        d_k = d_model // num_heads

        self.attn = self.Attention(
            norm=RMSNorm(d_model, eps=norm_eps, device=device, dtype=dtype),
            update=MultiheadSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                d_k=d_k,
                d_v=d_k,
                dropout=dropout,
                rope=RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=device),
                device=device,
                dtype=dtype,
            ),
        )
        self.ffn = self.FeedForward(
            norm=RMSNorm(d_model, eps=norm_eps, device=device, dtype=dtype),
            update=SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype),
        )
        self.register_load_state_dict_pre_hook(self._translate_course_state_dict)

    def forward(self, x: ModelActivations) -> ModelActivations:
        """Pass the attention result directly into the feed-forward layer."""
        return self.ffn(self.attn(x))

    def _translate_course_state_dict(
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
        """Translate the flat course block layout into nested sublayers."""
        del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        for course_suffix, native_suffix in _COURSE_BLOCK_KEY_TRANSLATION.items():
            source = prefix + course_suffix
            destination = prefix + native_suffix
            if source not in state_dict:
                continue
            if destination in state_dict:
                raise ValueError(
                    "state_dict contains both course and native keys for the same parameter: "
                    f"{source!r} and {destination!r}"
                )
            state_dict[destination] = state_dict.pop(source)


TransformerBlock = GPTDecoderLayer


class TransformerLM(Module):
    """Decoder-only Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        *,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                GPTDecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    dropout=dropout,
                    norm_eps=norm_eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, eps=norm_eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: TokenIndices) -> Logits:
        """Compute unnormalized next-token logits."""
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
