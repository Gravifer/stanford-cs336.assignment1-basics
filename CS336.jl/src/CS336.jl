"""
The Julia package for the CS336 parallel implementation.

Adapter-exposed Python behavior is introduced incrementally as idiomatic Julia
functions and explicit model components after its parity evidence is established.
"""
module CS336

export BPETokenizer,
    decode,
    embedding,
    encode,
    encode_iterable,
    linear,
    rmsnorm,
    silu,
    softmax,
    swiglu,
    swiglu_packed,
    train_bpe

include("functional.jl")
include("tokenizer.jl")

end
