"""Decoder-only Transformer language-model modules."""

from collections.abc import Mapping
from math import prod
from typing import Any

import torch
from jaxtyping import Float, Int
from torch import nn

from cs336_basics.nn.attention import MultiheadSelfAttention, RotaryPositionalEmbedding
from cs336_basics.nn.analytics import CostRepr, _CostChild, _CostScope
from cs336_basics.nn.feed_forward import SwiGLU
from cs336_basics.nn.modules import DeltaLayer, Embedding, Linear, Module, RMSNorm


__all__ = [
    "GPTDecoderLayer",
    "TransformerBlock",
    "TransformerLM",
]


type ModelActivations = Float[torch.Tensor, "*batch sequence_length d_model"]
type TokenIndices = Int[torch.Tensor, "*batch sequence_length"]
type Logits = Float[torch.Tensor, "*batch sequence_length vocab_size"]


def _activation_call_bindings(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    output: Any,
) -> Mapping[str, Any]:
    """Bind flattened batch and sequence axes from model activations."""
    del output
    x = args[0] if args else kwargs["x"]
    if not isinstance(x, torch.Tensor):
        raise TypeError("decoder cost observation requires sequence-feature tensor input")
    if x.ndim < 2:
        raise ValueError("decoder cost observation requires sequence and feature axes")
    return {
        "batch": prod(x.shape[:-2], start=1),
        "sequence": x.shape[-2],
    }


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
    class PrenormRoPEAttention(Module):
        """Normalized causal self-attention update."""

        def __init__(self, norm: RMSNorm, update: MultiheadSelfAttention) -> None:
            super().__init__()
            self.norm = norm
            self.update = update

        def forward(self, x: ModelActivations) -> ModelActivations:
            """Compute the normalized attention update."""
            return self.update(self.norm(x))

        def _cost_repr(self, scope: _CostScope) -> tuple[CostRepr, ...]:
            """Classify residual addition and delegate normalized attention."""
            return ()

        def _cost_call_bindings(
            self,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
            output: Any,
        ) -> Mapping[str, Any]:
            """Bind this sublayer's activation batch and sequence axes."""
            return _activation_call_bindings(args, kwargs, output)

        def _cost_children(self, scope: _CostScope) -> tuple[_CostChild, ...]:
            """Pass this sublayer's invocation shape to self-attention."""
            s = scope.symbols
            s.unbound("batch", "sequence")
            s.bind(d_model=self.update.d_model)
            return (
                scope.child("norm", self.norm),
                scope.child(
                    "update",
                    self.update,
                    arguments={"batch": s.batch, "sequence": s.sequence, "d_model": s.d_model},
                ),
            )

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

        def _cost_repr(self, scope: _CostScope) -> tuple[CostRepr, ...]:
            """Classify residual addition and delegate normalized SwiGLU."""
            return ()

        def _cost_call_bindings(
            self,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
            output: Any,
        ) -> Mapping[str, Any]:
            """Bind this sublayer's activation batch and sequence axes."""
            return _activation_call_bindings(args, kwargs, output)

        def _cost_children(self, scope: _CostScope) -> tuple[_CostChild, ...]:
            """Pass this sublayer's invocation shape to SwiGLU."""
            s = scope.symbols
            s.unbound("batch", "sequence")
            s.bind(d_model=self.update.d_model)
            return (
                scope.child("norm", self.norm),
                scope.child(
                    "update",
                    self.update,
                    arguments={
                        "tokens": s.batch * s.sequence,
                        "d_model": s.d_model,
                        "d_ff": self.update.d_ff,
                    },
                ),
            )

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

        self.attn = self.PrenormRoPEAttention(
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

    def _cost_fold_key(self) -> tuple[object, ...]:
        """Describe the configuration read by this layer's symbolic cost hooks."""
        attention = self.attn.update
        feed_forward = self.ffn.update
        rope = attention.rope
        delegated_types = tuple(
            (name, type(module)) for name, module in self.named_modules(remove_duplicate=False)
        )
        attention_state = tuple(
            (name, tuple(parameter.shape), parameter.dtype)
            for name, parameter in attention.named_parameters(remove_duplicate=False)
        )
        feed_forward_state = tuple(
            (name, tuple(parameter.shape), parameter.dtype)
            for name, parameter in feed_forward.named_parameters(remove_duplicate=False)
        )
        rope_form = None if rope is None else (type(rope), rope.use_matrix_form)
        return (
            delegated_types,
            type(self),
            type(self.attn),
            type(attention),
            attention.d_model,
            attention.num_heads,
            attention.num_kv_heads,
            attention.d_k,
            attention.d_v,
            attention_state,
            rope_form,
            type(self.ffn),
            type(feed_forward),
            feed_forward.d_model,
            feed_forward.d_ff,
            feed_forward_state,
        )

    def _cost_repr(self, scope: _CostScope) -> tuple[CostRepr, ...]:
        """Classify block orchestration and delegate its two sublayers."""
        return ()

    def _cost_call_bindings(
        self,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> Mapping[str, Any]:
        """Bind this layer's activation batch and sequence axes."""
        return _activation_call_bindings(args, kwargs, output)

    def _cost_children(self, scope: _CostScope) -> tuple[_CostChild, ...]:
        """Pass one shared block shape to attention and feed-forward."""
        s = scope.symbols
        s.unbound("batch", "sequence")
        s.bind(d_model=self.attn.update.d_model)
        arguments = {"batch": s.batch, "sequence": s.sequence, "d_model": s.d_model}
        return (
            scope.child("attn", self.attn, arguments=arguments),
            scope.child("ffn", self.ffn, arguments=arguments),
        )

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

    def _cost_repr(self, scope: _CostScope) -> tuple[CostRepr, ...]:
        """Classify model orchestration and delegate its numerical work."""
        return ()

    def _cost_call_bindings(
        self,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> Mapping[str, Any]:
        """Bind flattened batch and sequence dimensions from one token-ID call."""
        del output
        token_ids = args[0] if args else kwargs["token_ids"]
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("TransformerLM cost observation requires tensor token_ids")
        if token_ids.ndim < 1:
            raise ValueError("TransformerLM cost observation requires token_ids with a sequence axis")
        return {
            "batch": prod(token_ids.shape[:-1], start=1),
            "sequence": token_ids.shape[-1],
        }

    def _cost_children(self, scope: _CostScope) -> tuple[_CostChild, ...]:
        """Summarize identical blocks while traversing other children normally."""
        if len(self.layers) != self.num_layers:
            raise ValueError(
                "TransformerLM symbolic layer folding requires num_layers to match the registered layer count, "
                f"got num_layers={self.num_layers} and {len(self.layers)} registered layers"
            )
        if self.layers:
            representative = self.layers[0]
            if not isinstance(representative, GPTDecoderLayer):
                raise ValueError("TransformerLM symbolic layer folding requires GPTDecoderLayer instances")
            representative_signature = representative._cost_fold_key()
            for index, layer in enumerate(list(self.layers)[1:], start=1):
                if not isinstance(layer, GPTDecoderLayer) or layer._cost_fold_key() != representative_signature:
                    raise ValueError(
                        "TransformerLM symbolic layer folding requires homogeneous cost-driving configuration; "
                        f"layer {index} differs from layer 0"
                    )

        s = scope.symbols
        s.unbound("batch", "sequence")
        s.bind(d_model=self.d_model, num_layers=self.num_layers, vocab_size=self.vocab_size)
        children = [scope.child("token_embeddings", self.token_embeddings)]
        if self.layers:
            children.append(
                scope.child(
                    "layers",
                    self.layers[0],
                    repetitions=s.num_layers,
                    arguments={"batch": s.batch, "sequence": s.sequence, "d_model": s.d_model},
                )
            )
        children.extend(
            (
                scope.child("ln_final", self.ln_final),
                scope.child(
                    "lm_head",
                    self.lm_head,
                    arguments={"tokens": s.batch * s.sequence, "d_in": s.d_model, "d_out": s.vocab_size},
                ),
            )
        )
        return tuple(children)
