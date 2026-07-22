function _linear_parameters(rng, ::Type{T}, output_features, input_features) where {T}
    standard_deviation = sqrt(T(2) / T(input_features + output_features))
    return WeightInitializers.truncated_normal(
        rng,
        T,
        output_features,
        input_features;
        mean=zero(T),
        std=standard_deviation,
        lo=-T(3) * standard_deviation,
        hi=T(3) * standard_deviation,
    )
end

function _embedding_parameters(rng, ::Type{T}, feature_count, vocabulary_size) where {T}
    return WeightInitializers.truncated_normal(
        rng,
        T,
        feature_count,
        vocabulary_size;
        mean=zero(T),
        std=one(T),
        lo=-T(3),
        hi=T(3),
    )
end

function _rope_state(layer)
    rope = RotaryEmbedding(
        layer.theta,
        layer.d_model ÷ layer.num_heads,
        layer.context_length;
        variant=layer.rope_variant,
        T=typeof(layer.theta),
    )
    if rope isa ElementwiseRotaryEmbedding
        return (
            variant=:elementwise,
            theta=rope.theta,
            cosine=rope.cosine,
            sine=rope.sine,
        )
    end
    return (variant=:matrix, theta=rope.theta, rotation=rope.rotation)
end

function _rope_from_state(state)
    if state.variant === :elementwise
        return ElementwiseRotaryEmbedding(state.theta, state.cosine, state.sine)
    elseif state.variant === :matrix
        return MatrixRotaryEmbedding(state.theta, state.rotation)
    end
    throw(ArgumentError("unknown RoPE state variant: $(state.variant)"))
end

function _layer_input(input)
    if input isa Tuple
        length(input) == 2 ||
            throw(ArgumentError("Lux transformer tuple input must be (activations, positions)"))
        return input
    end
    return input, nothing
end

"""
    TransformerBlock(d_model, num_heads, d_ff, context_length, theta; ...)

An explicit-parameter LuxCore decoder block. The architecture object stores
only configuration; trainable arrays live in the `NamedTuple` returned by
`LuxCore.setup`, while the RoPE cache and default zero-based positions live in
state. Attention and SwiGLU use packed projections by default.
"""
struct TransformerBlock{T<:AbstractFloat} <: LuxCore.AbstractLuxLayer
    d_model::Int
    num_heads::Int
    d_ff::Int
    context_length::Int
    theta::T
    norm_eps::T
    rope_variant::Symbol
end

function TransformerBlock(
    d_model::Integer,
    num_heads::Integer,
    d_ff::Integer,
    context_length::Integer,
    theta::Real;
    norm_eps::Real=1e-5,
    rope_variant::Symbol=:elementwise,
    T::Type{<:AbstractFloat}=Float32,
)
    d_model > 0 || throw(ArgumentError("d_model must be positive"))
    num_heads > 0 && d_model % num_heads == 0 ||
        throw(ArgumentError("d_model must be divisible by a positive num_heads"))
    d_ff > 0 || throw(ArgumentError("d_ff must be positive"))
    context_length > 0 || throw(ArgumentError("context_length must be positive"))
    rope_variant in (:elementwise, :matrix) ||
        throw(ArgumentError("unknown RoPE variant: $rope_variant"))
    return TransformerBlock{T}(
        d_model,
        num_heads,
        d_ff,
        context_length,
        T(theta),
        T(norm_eps),
        rope_variant,
    )
end

function LuxCore.initialparameters(
    rng::Random.AbstractRNG,
    layer::TransformerBlock{T},
) where {T}
    query_weight = _linear_parameters(rng, T, layer.d_model, layer.d_model)
    key_weight = _linear_parameters(rng, T, layer.d_model, layer.d_model)
    value_weight = _linear_parameters(rng, T, layer.d_model, layer.d_model)
    output_weight = _linear_parameters(rng, T, layer.d_model, layer.d_model)
    gate_weight = _linear_parameters(rng, T, layer.d_ff, layer.d_model)
    feed_forward_output = _linear_parameters(rng, T, layer.d_model, layer.d_ff)
    value_feed_forward_weight = _linear_parameters(rng, T, layer.d_ff, layer.d_model)
    return (
        attention_norm=ones(T, layer.d_model),
        attention=(
            input_weight=vcat(query_weight, key_weight, value_weight),
            output_weight=output_weight,
        ),
        feed_forward_norm=ones(T, layer.d_model),
        feed_forward=(
            input_weight=vcat(value_feed_forward_weight, gate_weight),
            output_weight=feed_forward_output,
        ),
    )
end

function LuxCore.initialstates(::Random.AbstractRNG, layer::TransformerBlock)
    return (
        rope=_rope_state(layer),
        positions=collect(Int32, 0:(layer.context_length - 1)),
    )
end

LuxCore.parameterlength(layer::TransformerBlock) =
    3 * layer.d_model^2 + layer.d_model^2 + 2 * layer.d_ff * layer.d_model +
    layer.d_model * layer.d_ff + 2 * layer.d_model

function LuxCore.statelength(layer::TransformerBlock)
    pair_count = layer.d_model ÷ layer.num_heads ÷ 2
    rope_values =
        layer.rope_variant === :elementwise ?
        2 * pair_count * layer.context_length :
        4 * pair_count * layer.context_length
    return rope_values + layer.context_length
end

function _apply_transformer_block(layer, x, parameters, state, positions)
    size(x, 1) == layer.d_model || throw(
        DimensionMismatch("transformer block input width does not match d_model"),
    )
    sequence_length = size(x, 2)
    sequence_length <= layer.context_length || throw(
        DimensionMismatch("input sequence exceeds the configured context length"),
    )
    selected_positions =
        positions === nothing ? state.positions[1:sequence_length] : positions
    rope = _rope_from_state(state.rope)
    normalized_attention = rmsnorm(
        x,
        parameters.attention_norm;
        eps=layer.norm_eps,
    )
    attention_update = multihead_self_attention(
        normalized_attention,
        parameters.attention.input_weight,
        parameters.attention.output_weight;
        num_heads=layer.num_heads,
        rope,
        positions=selected_positions,
    )
    after_attention = x .+ attention_update
    normalized_feed_forward = rmsnorm(
        after_attention,
        parameters.feed_forward_norm;
        eps=layer.norm_eps,
    )
    feed_forward_update = swiglu(
        normalized_feed_forward,
        parameters.feed_forward.input_weight,
        parameters.feed_forward.output_weight,
    )
    return after_attention .+ feed_forward_update
end

function (layer::TransformerBlock)(input, parameters, state)
    x, positions = _layer_input(input)
    return _apply_transformer_block(layer, x, parameters, state, positions), state
end

"""
    TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads,
                  d_ff, rope_theta; ...)

Decoder-only language model implementing the LuxCore explicit
parameter/state interface. One RoPE cache and default-position state is shared
across every block; each block retains independent parameters.
"""
struct TransformerLM{T<:AbstractFloat,B<:TransformerBlock{T}} <: LuxCore.AbstractLuxLayer
    vocab_size::Int
    context_length::Int
    d_model::Int
    num_layers::Int
    block::B
end

function TransformerLM(
    vocab_size::Integer,
    context_length::Integer,
    d_model::Integer,
    num_layers::Integer,
    num_heads::Integer,
    d_ff::Integer,
    rope_theta::Real;
    norm_eps::Real=1e-5,
    rope_variant::Symbol=:elementwise,
    T::Type{<:AbstractFloat}=Float32,
)
    vocab_size > 0 || throw(ArgumentError("vocab_size must be positive"))
    num_layers >= 0 || throw(ArgumentError("num_layers must be non-negative"))
    block = TransformerBlock(
        d_model,
        num_heads,
        d_ff,
        context_length,
        rope_theta;
        norm_eps,
        rope_variant,
        T,
    )
    return TransformerLM{T,typeof(block)}(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        block,
    )
end

function LuxCore.initialparameters(
    rng::Random.AbstractRNG,
    model::TransformerLM{T},
) where {T}
    return (
        token_embedding=_embedding_parameters(rng, T, model.d_model, model.vocab_size),
        blocks=ntuple(_ -> LuxCore.initialparameters(rng, model.block), model.num_layers),
        final_norm=ones(T, model.d_model),
        lm_head=_linear_parameters(rng, T, model.vocab_size, model.d_model),
    )
end

function LuxCore.initialstates(::Random.AbstractRNG, model::TransformerLM)
    return (
        rope=_rope_state(model.block),
        positions=collect(Int32, 0:(model.context_length - 1)),
    )
end

function LuxCore.parameterlength(model::TransformerLM)
    return model.d_model * model.vocab_size +
           model.num_layers * LuxCore.parameterlength(model.block) + model.d_model +
           model.vocab_size * model.d_model
end

LuxCore.statelength(model::TransformerLM) = LuxCore.statelength(model.block)

function (model::TransformerLM)(input, parameters, state)
    token_ids, positions = _layer_input(input)
    ndims(token_ids) >= 1 ||
        throw(DimensionMismatch("TransformerLM token IDs must include a sequence axis"))
    sequence_length = size(token_ids, 1)
    sequence_length <= model.context_length || throw(
        DimensionMismatch("input sequence exceeds the configured context length"),
    )
    selected_positions =
        positions === nothing ? state.positions[1:sequence_length] : positions
    x = embedding(token_ids, parameters.token_embedding)
    block_state = (rope=state.rope, positions=state.positions)
    for index in 1:model.num_layers
        x = _apply_transformer_block(
            model.block,
            x,
            parameters.blocks[index],
            block_state,
            selected_positions,
        )
    end
    normalized = rmsnorm(x, parameters.final_norm; eps=model.block.norm_eps)
    return linear(normalized, parameters.lm_head), state
end
