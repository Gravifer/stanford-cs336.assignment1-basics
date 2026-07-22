"""
The Julia package for the CS336 parallel implementation.

Adapter-exposed Python behavior is introduced incrementally as idiomatic Julia
functions and explicit model components after its parity evidence is established.
"""
module CS336

import LinearAlgebra
import LuxCore
import NNlib
import Random
import WeightInitializers

export BPETokenizer,
    ElementwiseRotaryEmbedding,
    MatrixRotaryEmbedding,
    RotaryEmbedding,
    TransformerBlock,
    TransformerLM,
    apply_rope,
    apply_rope_qk,
    decode,
    embedding,
    encode,
    encode_iterable,
    linear,
    headed_scaled_dot_product_attention,
    multihead_attention,
    multihead_self_attention,
    rmsnorm,
    silu,
    scaled_dot_product_attention,
    softmax,
    swiglu,
    swiglu_packed,
    train_bpe

include("functional.jl")
include("attention.jl")
include("model.jl")
include("tokenizer.jl")

end
