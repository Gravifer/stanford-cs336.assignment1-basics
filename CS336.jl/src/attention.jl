abstract type RotaryEmbedding end

"""Cosine/sine RoPE cache used by the allocation-light elementwise lane."""
struct ElementwiseRotaryEmbedding{T<:Real,C<:AbstractMatrix{T}} <: RotaryEmbedding
    theta::T
    cosine::C
    sine::C
end

"""Explicit 2×2 RoPE matrices used by the comparison/benchmark lane."""
struct MatrixRotaryEmbedding{T<:Real,R<:AbstractArray{T,4}} <: RotaryEmbedding
    theta::T
    rotation::R
end

function ElementwiseRotaryEmbedding(theta::Real, cosine::AbstractMatrix, sine::AbstractMatrix)
    size(cosine) == size(sine) ||
        throw(DimensionMismatch("RoPE cosine and sine caches must have equal shapes"))
    T = promote_type(typeof(theta), eltype(cosine), eltype(sine))
    promoted_cosine = eltype(cosine) === T ? cosine : T.(cosine)
    promoted_sine = eltype(sine) === T ? sine : T.(sine)
    C = typeof(promoted_cosine)
    typeof(promoted_sine) === C ||
        throw(ArgumentError("RoPE cosine and sine caches must use the same array type"))
    return ElementwiseRotaryEmbedding{T,C}(T(theta), promoted_cosine, promoted_sine)
end

function MatrixRotaryEmbedding(theta::Real, rotation::AbstractArray{<:Real,4})
    size(rotation, 1) == 2 && size(rotation, 2) == 2 || throw(
        DimensionMismatch("matrix-form RoPE cache must have shape (2, 2, pair, position)"),
    )
    T = promote_type(typeof(theta), eltype(rotation))
    promoted_rotation = eltype(rotation) === T ? rotation : T.(rotation)
    return MatrixRotaryEmbedding{T,typeof(promoted_rotation)}(T(theta), promoted_rotation)
end

"""
    RotaryEmbedding(theta, feature_count, max_sequence_length;
                    variant=:elementwise, T=Float32)

Construct a zero-based rotary-position cache. `variant=:elementwise` stores one
cosine and sine value per feature pair and position. `variant=:matrix` stores
an explicit 2×2 matrix and is retained as a benchmark comparison rather than
the default representation.
"""
function RotaryEmbedding(
    theta::Real,
    feature_count::Integer,
    max_sequence_length::Integer;
    variant::Symbol=:elementwise,
    T::Type{<:AbstractFloat}=Float32,
)
    feature_count > 0 && iseven(feature_count) ||
        throw(ArgumentError("RoPE feature count must be a positive even integer"))
    max_sequence_length >= 0 ||
        throw(ArgumentError("RoPE maximum sequence length must be non-negative"))
    theta > 0 || throw(ArgumentError("RoPE theta must be positive"))

    pair_count = feature_count ÷ 2
    pair_indices = T.(0:(pair_count - 1))
    inverse_frequencies = inv.(T(theta) .^ (T(2) .* pair_indices ./ T(feature_count)))
    positions = T.(0:(max_sequence_length - 1))
    angles = reshape(inverse_frequencies, pair_count, 1) .* reshape(positions, 1, :)
    cosine = cos.(angles)
    sine = sin.(angles)

    if variant === :elementwise
        return ElementwiseRotaryEmbedding(T(theta), cosine, sine)
    elseif variant === :matrix
        top = cat(
            reshape(cosine, 1, 1, pair_count, max_sequence_length),
            reshape(-sine, 1, 1, pair_count, max_sequence_length);
            dims=2,
        )
        bottom = cat(
            reshape(sine, 1, 1, pair_count, max_sequence_length),
            reshape(cosine, 1, 1, pair_count, max_sequence_length);
            dims=2,
        )
        return MatrixRotaryEmbedding(T(theta), cat(top, bottom; dims=1))
    end
    throw(ArgumentError("unknown RoPE variant: $variant"))
end

rope_feature_count(rope::ElementwiseRotaryEmbedding) = 2 * size(rope.cosine, 1)
rope_feature_count(rope::MatrixRotaryEmbedding) = 2 * size(rope.rotation, 3)
rope_max_sequence_length(rope::ElementwiseRotaryEmbedding) = size(rope.cosine, 2)
rope_max_sequence_length(rope::MatrixRotaryEmbedding) = size(rope.rotation, 4)

function _validate_rope_input(rope::RotaryEmbedding, x, positions, sequence_dim, position_layout)
    ndims(x) >= 2 || throw(DimensionMismatch("RoPE input must include feature and sequence axes"))
    2 <= sequence_dim <= ndims(x) || throw(
        ArgumentError("RoPE sequence_dim must be in 2:$(ndims(x)), received $sequence_dim"),
    )
    size(x, 1) == rope_feature_count(rope) || throw(
        DimensionMismatch(
            "RoPE cache width $(rope_feature_count(rope)) does not match input width $(size(x, 1))",
        ),
    )
    position_layout in (:batch, :head) ||
        throw(ArgumentError("unknown RoPE position layout: $position_layout"))
    position_sequence_axis = position_layout === :batch ? 1 : 2
    minimum_rank = position_layout === :batch ? 1 : 2
    ndims(positions) >= minimum_rank ||
        throw(ArgumentError("RoPE positions do not contain the required layout axes"))
    size(positions, position_sequence_axis) == size(x, sequence_dim) || throw(
        DimensionMismatch("RoPE position sequence length does not match input length"),
    )
    if position_layout === :head
        sequence_dim >= 3 || throw(
            DimensionMismatch("head-aligned positions require an explicit head axis"),
        )
        size(positions, 1) == size(x, sequence_dim - 1) || throw(
            DimensionMismatch("RoPE position head count does not match input heads"),
        )
    end

    batch_shape = size(x)[(sequence_dim + 1):end]
    position_batch_shape = size(positions)[(position_sequence_axis + 1):end]
    length(position_batch_shape) <= length(batch_shape) || throw(
        DimensionMismatch("RoPE positions have more batch axes than the input"),
    )
    aligned_batch_shape = batch_shape[(end - length(position_batch_shape) + 1):end]
    all(
        position_size == 1 || position_size == input_size for
        (position_size, input_size) in zip(position_batch_shape, aligned_batch_shape)
    ) || throw(DimensionMismatch("RoPE position batch axes are not broadcast-compatible"))

    maximum_position = rope_max_sequence_length(rope) - 1
    all(position -> 0 <= position <= maximum_position, positions) || throw(
        ArgumentError("RoPE positions must be in 0:$maximum_position"),
    )
    return nothing
end

function _rope_selection_shape(
    x,
    positions,
    sequence_dim,
    cache_prefix,
    position_layout,
)
    mapped_axes = sequence_dim - 2
    batch_axes = ndims(x) - sequence_dim
    position_prefix_axes = position_layout === :batch ? 1 : 2
    position_batch_axes = ndims(positions) - position_prefix_axes
    skipped_batch_axes = batch_axes - position_batch_axes
    if position_layout === :batch
        return (
            cache_prefix...,
            ntuple(_ -> 1, mapped_axes)...,
            size(positions, 1),
            ntuple(_ -> 1, skipped_batch_axes)...,
            size(positions)[2:end]...,
        )
    end
    return (
        cache_prefix...,
        ntuple(_ -> 1, mapped_axes - 1)...,
        size(positions, 1),
        size(positions, 2),
        ntuple(_ -> 1, skipped_batch_axes)...,
        size(positions)[3:end]...,
    )
end

function _select_rope_cache(cache::AbstractMatrix, x, positions, sequence_dim, position_layout)
    selected = cache[:, vec(positions .+ one(eltype(positions)))]
    return reshape(
        selected,
        _rope_selection_shape(
            x,
            positions,
            sequence_dim,
            (size(cache, 1),),
            position_layout,
        ),
    )
end

function _select_rope_cache(
    cache::AbstractArray{<:Real,4},
    x,
    positions,
    sequence_dim,
    position_layout,
)
    selected = cache[:, :, :, vec(positions .+ one(eltype(positions)))]
    return reshape(
        selected,
        _rope_selection_shape(
            x,
            positions,
            sequence_dim,
            (2, 2, size(cache, 3)),
            position_layout,
        ),
    )
end

function _paired_features(x, operation_type)
    operation_input = eltype(x) === operation_type ? x : operation_type.(x)
    even = operation_input[1:2:end, ntuple(_ -> Colon(), ndims(x) - 1)...]
    odd = operation_input[2:2:end, ntuple(_ -> Colon(), ndims(x) - 1)...]
    paired = cat(
        reshape(even, 1, size(even)...),
        reshape(odd, 1, size(odd)...);
        dims=1,
    )
    return paired, even, odd
end

function _apply_rope_selection(
    rope::ElementwiseRotaryEmbedding,
    x,
    selection::Tuple,
)
    cosine, sine = selection
    operation_type = promote_type(eltype(x), eltype(rope.cosine))
    operation_type = operation_type in (Float32, Float64) ? operation_type : Float32
    _, even, odd = _paired_features(x, operation_type)
    rotated_even = even .* operation_type.(cosine) .- odd .* operation_type.(sine)
    rotated_odd = even .* operation_type.(sine) .+ odd .* operation_type.(cosine)
    paired = cat(
        reshape(rotated_even, 1, size(rotated_even)...),
        reshape(rotated_odd, 1, size(rotated_odd)...);
        dims=1,
    )
    output = reshape(paired, size(x))
    return eltype(output) === eltype(x) ? output : eltype(x).(output)
end

function _apply_rope_selection(rope::MatrixRotaryEmbedding, x, rotation)
    operation_type = promote_type(eltype(x), eltype(rope.rotation))
    operation_type = operation_type in (Float32, Float64) ? operation_type : Float32
    paired, even, _ = _paired_features(x, operation_type)
    expanded_rotation = operation_type.(rotation) .+ reshape(
        zero.(even),
        1,
        1,
        size(even)...,
    )
    rotated = NNlib.batched_vec(
        reshape(expanded_rotation, 2, 2, :),
        reshape(paired, 2, :),
    )
    output = reshape(rotated, size(x))
    return eltype(output) === eltype(x) ? output : eltype(x).(output)
end

function _rope_selection(
    rope::ElementwiseRotaryEmbedding,
    x,
    positions,
    sequence_dim,
    position_layout,
)
    return (
        _select_rope_cache(rope.cosine, x, positions, sequence_dim, position_layout),
        _select_rope_cache(rope.sine, x, positions, sequence_dim, position_layout),
    )
end

function _rope_selection(
    rope::MatrixRotaryEmbedding,
    x,
    positions,
    sequence_dim,
    position_layout,
)
    return _select_rope_cache(rope.rotation, x, positions, sequence_dim, position_layout)
end

"""
    apply_rope(rope, x, positions; sequence_dim=2)

Apply cached RoPE to a feature-first tensor. Positions are zero-based and use
`(sequence, batch...)`; their batch axes align to the suffix of the input axes
after `sequence_dim`. Axes between feature and sequence (for example, heads)
share the selected positions.
"""
function apply_rope(
    rope::RotaryEmbedding,
    x::AbstractArray,
    positions::AbstractArray{<:Integer};
    sequence_dim::Integer=2,
    position_layout::Symbol=:batch,
)
    _validate_rope_input(rope, x, positions, sequence_dim, position_layout)
    isempty(x) && return x
    selection = _rope_selection(rope, x, positions, sequence_dim, position_layout)
    return _apply_rope_selection(rope, x, selection)
end

function apply_rope_qk(
    rope::RotaryEmbedding,
    query::AbstractArray,
    key::AbstractArray,
    positions::AbstractArray{<:Integer};
    sequence_dim::Integer=2,
    strategy::Symbol=:auto,
    position_layout::Symbol=:batch,
)
    strategy in (:auto, :separate, :stacked) ||
        throw(ArgumentError("unknown Q/K RoPE strategy: $strategy"))
    if strategy === :separate || size(query) != size(key)
        strategy === :stacked && throw(
            DimensionMismatch("stacked Q/K RoPE requires equal query and key shapes"),
        )
        return (
            apply_rope(rope, query, positions; sequence_dim, position_layout),
            apply_rope(rope, key, positions; sequence_dim, position_layout),
        )
    end

    _validate_rope_input(rope, query, positions, sequence_dim, position_layout)
    if strategy === :auto
        selection = _rope_selection(rope, query, positions, sequence_dim, position_layout)
        return (
            _apply_rope_selection(rope, query, selection),
            _apply_rope_selection(rope, key, selection),
        )
    end

    stacked = cat(
        reshape(query, size(query, 1), 1, size(query)[2:end]...),
        reshape(key, size(key, 1), 1, size(key)[2:end]...);
        dims=2,
    )
    rotated = apply_rope(
        rope,
        stacked,
        positions;
        sequence_dim=sequence_dim + 1,
        position_layout,
    )
    return (
        selectdim(rotated, 2, 1),
        selectdim(rotated, 2, 2),
    )
end

function _aligned_attention_mask(mask, score_shape)
    ndims(mask) >= 2 || throw(DimensionMismatch("attention mask must include key and query axes"))
    ndims(mask) <= length(score_shape) ||
        throw(DimensionMismatch("attention mask has too many batch axes"))
    for axis in 1:2
        size(mask, axis) in (1, score_shape[axis]) || throw(
            DimensionMismatch("attention mask key/query axes do not match attention scores"),
        )
    end
    mask_batch = size(mask)[3:end]
    score_batch = score_shape[3:end]
    aligned_score_batch = score_batch[(end - length(mask_batch) + 1):end]
    all(
        mask_size == 1 || mask_size == score_size for
        (mask_size, score_size) in zip(mask_batch, aligned_score_batch)
    ) || throw(DimensionMismatch("attention mask batch axes are not broadcast-compatible"))
    return reshape(
        mask,
        size(mask, 1),
        size(mask, 2),
        ntuple(_ -> 1, length(score_batch) - length(mask_batch))...,
        mask_batch...,
    )
end

function _causal_mask(scores)
    key_count, query_count = size(scores, 1), size(scores, 2)
    allowed = fill!(similar(scores, Bool, key_count, query_count), true)
    return LinearAlgebra.triu(allowed)
end

function _attention_weights(scores; mask=nothing, is_causal::Bool=false)
    key_count, query_count = size(scores, 1), size(scores, 2)
    aligned_mask = mask === nothing ? nothing : _aligned_attention_mask(mask, size(scores))
    causal = is_causal ? _causal_mask(scores) : nothing
    negative_infinity = convert(eltype(scores), -Inf)

    fully_masked = nothing
    if aligned_mask === nothing && causal === nothing
        masked_scores = scores
    elseif aligned_mask === nothing || eltype(aligned_mask) <: Bool
        allowed = aligned_mask === nothing ? causal : aligned_mask
        causal === nothing || aligned_mask === nothing || (allowed = allowed .& causal)
        fully_masked = .!any(allowed; dims=1)
        masked_scores = ifelse.(allowed, scores, negative_infinity)
    else
        bias = eltype(aligned_mask) === eltype(scores) ? aligned_mask : eltype(scores).(aligned_mask)
        causal === nothing || (bias = ifelse.(causal, bias, negative_infinity))
        maximum_bias = maximum(bias; dims=1)
        fully_masked = isinf.(maximum_bias) .& (maximum_bias .< zero(eltype(scores)))
        masked_scores = scores .+ bias
    end

    fully_masked === nothing ||
        (masked_scores = ifelse.(fully_masked, zero(eltype(scores)), masked_scores))
    weights = softmax(masked_scores; dims=1)
    fully_masked === nothing ||
        (weights = ifelse.(fully_masked, zero(eltype(weights)), weights))
    return weights
end

"""
    scaled_dot_product_attention(query, key, value; mask=nothing,
                                 is_causal=false, scale=nothing)

Compute attention on feature-first tensors shaped `(feature, sequence,
batch...)`. Masks use `(key, query, batch...)`, with shorter batch suffixes
broadcast over leading batch axes. Boolean `true` permits attention; numeric
masks are additive. Fully masked query columns produce finite zero outputs.
"""
function scaled_dot_product_attention(
    query::AbstractArray,
    key::AbstractArray,
    value::AbstractArray;
    mask=nothing,
    is_causal::Bool=false,
    scale=nothing,
    return_weights::Bool=false,
)
    ndims(query) >= 2 && ndims(key) >= 2 && ndims(value) >= 2 || throw(
        DimensionMismatch("attention inputs must include feature and sequence axes"),
    )
    size(query, 1) == size(key, 1) || throw(
        DimensionMismatch("query and key feature counts must match"),
    )
    size(key, 2) == size(value, 2) || throw(
        DimensionMismatch("key and value sequence lengths must match"),
    )
    size(query)[3:end] == size(key)[3:end] == size(value)[3:end] || throw(
        DimensionMismatch("query, key, and value batch axes must match"),
    )
    key_feature_count = size(key, 1)
    key_feature_count > 0 || throw(DimensionMismatch("attention key width must be positive"))

    query_matrix = reshape(query, size(query, 1), size(query, 2), :)
    key_matrix = reshape(key, size(key, 1), size(key, 2), :)
    value_matrix = reshape(value, size(value, 1), size(value, 2), :)
    score_scale =
        scale === nothing ? inv(sqrt(eltype(query)(key_feature_count))) : eltype(query)(scale)
    score_matrix = NNlib.batched_mul(NNlib.batched_transpose(key_matrix), query_matrix)
    scores = reshape(
        score_matrix .* score_scale,
        size(key, 2),
        size(query, 2),
        size(query)[3:end]...,
    )
    weights = _attention_weights(scores; mask, is_causal)
    output_matrix = NNlib.batched_mul(value_matrix, reshape(weights, size(weights, 1), size(weights, 2), :))
    output = reshape(output_matrix, size(value, 1), size(query, 2), size(query)[3:end]...)
    return return_weights ? (output, weights) : output
end

scaled_dot_product_attention(query, key, value, mask; kwargs...) =
    scaled_dot_product_attention(query, key, value; mask, kwargs...)

function _head_sequence_permutation(rank, layout::Symbol)
    if layout === :head_before_sequence
        return (1, 3, 2, ntuple(axis -> axis, rank - 3) .+ 3...)
    elseif layout === :head_after_sequence
        return ntuple(identity, rank)
    end
    throw(ArgumentError("unknown attention layout: $layout"))
end

function _grouped_headed_attention(
    query,
    key,
    value;
    mask,
    is_causal,
    scale,
    layout,
)
    if layout === :head_before_sequence
        query = permutedims(query, _head_sequence_permutation(ndims(query), layout))
        key = permutedims(key, _head_sequence_permutation(ndims(key), layout))
        value = permutedims(value, _head_sequence_permutation(ndims(value), layout))
    elseif layout !== :head_after_sequence
        throw(ArgumentError("unknown attention layout: $layout"))
    end

    query_heads = size(query, 3)
    key_heads = size(key, 3)
    group = query_heads ÷ key_heads
    query_count = size(query, 2)
    key_count = size(key, 2)
    batch_shape = size(query)[4:end]
    batch_count = prod(batch_shape; init=1)

    grouped_query = reshape(query, size(query, 1), query_count, group, key_heads, batch_shape...)
    grouped_query = reshape(grouped_query, size(query, 1), query_count * group, key_heads * batch_count)
    key_matrix = reshape(key, size(key, 1), key_count, key_heads * batch_count)
    score_scale = scale === nothing ? inv(sqrt(eltype(query)(size(key, 1)))) : eltype(query)(scale)
    score_matrix = NNlib.batched_mul(NNlib.batched_transpose(key_matrix), grouped_query)
    scores = reshape(score_matrix .* score_scale, key_count, query_count, group, key_heads, batch_shape...)
    weights = _attention_weights(scores; mask, is_causal)
    value_matrix = reshape(value, size(value, 1), key_count, key_heads * batch_count)
    attended = NNlib.batched_mul(
        value_matrix,
        reshape(weights, key_count, query_count * group, key_heads * batch_count),
    )
    attended = reshape(attended, size(value, 1), query_count, group, key_heads, batch_shape...)
    attended = reshape(attended, size(value, 1), query_count, query_heads, batch_shape...)
    if layout === :head_before_sequence
        attended = permutedims(attended, (1, 3, 2, ntuple(axis -> axis, ndims(attended) - 3) .+ 3...))
    end
    return attended
end

"""
    headed_scaled_dot_product_attention(query, key, value; layout=:head_before_sequence, ...)

Attend headed feature-first activations. The canonical layout is
`(head_feature, head, sequence, batch...)`; `:head_after_sequence` selects the
experimental `(head_feature, sequence, head, batch...)` lane. Unequal query and
key/value head counts use grouped contractions without materializing repeated
key/value heads.
"""
function headed_scaled_dot_product_attention(
    query::AbstractArray,
    key::AbstractArray,
    value::AbstractArray;
    mask=nothing,
    is_causal::Bool=false,
    scale=nothing,
    layout::Symbol=:head_before_sequence,
)
    all(ndims(array) >= 3 for array in (query, key, value)) || throw(
        DimensionMismatch("headed attention inputs need feature, head, and sequence axes"),
    )
    head_axis = layout === :head_before_sequence ? 2 : layout === :head_after_sequence ? 3 : 0
    head_axis == 0 && throw(ArgumentError("unknown attention layout: $layout"))
    query_heads = size(query, head_axis)
    key_heads = size(key, head_axis)
    value_heads = size(value, head_axis)
    key_heads == value_heads || throw(DimensionMismatch("key and value head counts must match"))
    key_heads > 0 && query_heads % key_heads == 0 || throw(
        DimensionMismatch("query head count must be divisible by key/value head count"),
    )

    if query_heads != key_heads
        return _grouped_headed_attention(
            query,
            key,
            value;
            mask,
            is_causal,
            scale,
            layout,
        )
    end

    permutation = _head_sequence_permutation(ndims(query), layout)
    query_sequence_first = layout === :head_before_sequence ? permutedims(query, permutation) : query
    key_sequence_first = layout === :head_before_sequence ? permutedims(key, permutation) : key
    value_sequence_first = layout === :head_before_sequence ? permutedims(value, permutation) : value
    attended = scaled_dot_product_attention(
        query_sequence_first,
        key_sequence_first,
        value_sequence_first;
        mask,
        is_causal,
        scale,
    )
    return layout === :head_before_sequence ? permutedims(attended, permutation) : attended
end

function _split_projection(projected, head_feature_count, head_count, layout)
    sequence_and_batch = size(projected)[2:end]
    headed = reshape(projected, head_feature_count, head_count, sequence_and_batch...)
    if layout === :head_before_sequence
        return headed
    elseif layout === :head_after_sequence
        return permutedims(
            headed,
            (1, 3, 2, ntuple(axis -> axis, ndims(headed) - 3) .+ 3...),
        )
    end
    throw(ArgumentError("unknown attention layout: $layout"))
end

function _apply_attention_rope(
    rope,
    query,
    key,
    positions,
    layout,
    strategy,
    position_layout,
)
    if layout === :head_before_sequence
        return apply_rope_qk(
            rope,
            query,
            key,
            positions;
            sequence_dim=3,
            strategy,
            position_layout,
        )
    end
    permutation = (1, 3, 2, ntuple(axis -> axis, ndims(query) - 3) .+ 3...)
    query_head_first = permutedims(query, permutation)
    key_head_first = permutedims(key, permutation)
    rotated_query, rotated_key = apply_rope_qk(
        rope,
        query_head_first,
        key_head_first,
        positions;
        sequence_dim=3,
        strategy,
        position_layout,
    )
    return permutedims(rotated_query, permutation), permutedims(rotated_key, permutation)
end

function _apply_single_attention_rope(rope, x, positions, layout, position_layout)
    if layout === :head_before_sequence
        return apply_rope(rope, x, positions; sequence_dim=3, position_layout)
    elseif layout === :head_after_sequence
        permutation = (1, 3, 2, ntuple(axis -> axis, ndims(x) - 3) .+ 3...)
        head_first = permutedims(x, permutation)
        rotated = apply_rope(
            rope,
            head_first,
            positions;
            sequence_dim=3,
            position_layout,
        )
        return permutedims(rotated, permutation)
    end
    throw(ArgumentError("unknown attention layout: $layout"))
end

function _attention_from_projections(
    projected_query,
    projected_key,
    projected_value,
    output_weight;
    num_heads,
    num_kv_heads,
    rope,
    positions,
    key_positions=positions,
    query_position_layout=:batch,
    key_position_layout=query_position_layout,
    mask,
    is_causal,
    layout,
    qk_strategy,
)
    size(projected_query, 1) % num_heads == 0 ||
        throw(DimensionMismatch("query projection width must be divisible by num_heads"))
    size(projected_key, 1) % num_kv_heads == 0 ||
        throw(DimensionMismatch("key projection width must be divisible by num_kv_heads"))
    size(projected_value, 1) % num_kv_heads == 0 ||
        throw(DimensionMismatch("value projection width must be divisible by num_kv_heads"))
    query_feature_count = size(projected_query, 1) ÷ num_heads
    key_feature_count = size(projected_key, 1) ÷ num_kv_heads
    value_feature_count = size(projected_value, 1) ÷ num_kv_heads
    query_feature_count == key_feature_count ||
        throw(DimensionMismatch("query and key per-head widths must match"))

    query = _split_projection(projected_query, query_feature_count, num_heads, layout)
    key = _split_projection(projected_key, key_feature_count, num_kv_heads, layout)
    value = _split_projection(projected_value, value_feature_count, num_kv_heads, layout)
    rope === nothing || begin
        positions === nothing && (positions = collect(0:(size(projected_query, 2) - 1)))
        key_positions === nothing &&
            (key_positions = collect(0:(size(projected_key, 2) - 1)))
        if size(query) == size(key) &&
           positions === key_positions &&
           query_position_layout === key_position_layout
            query, key = _apply_attention_rope(
                rope,
                query,
                key,
                positions,
                layout,
                qk_strategy,
                query_position_layout,
            )
        else
            qk_strategy === :stacked && throw(
                DimensionMismatch(
                    "stacked Q/K RoPE requires equal shapes and shared positions",
                ),
            )
            query = _apply_single_attention_rope(
                rope,
                query,
                positions,
                layout,
                query_position_layout,
            )
            key = _apply_single_attention_rope(
                rope,
                key,
                key_positions,
                layout,
                key_position_layout,
            )
        end
    end
    attended = headed_scaled_dot_product_attention(
        query,
        key,
        value;
        mask,
        is_causal,
        scale=inv(sqrt(eltype(query)(query_feature_count))),
        layout,
    )
    if layout === :head_before_sequence
        joined = reshape(attended, num_heads * value_feature_count, size(attended)[3:end]...)
    else
        head_first = permutedims(
            attended,
            (1, 3, 2, ntuple(axis -> axis, ndims(attended) - 3) .+ 3...),
        )
        joined = reshape(head_first, num_heads * value_feature_count, size(head_first)[3:end]...)
    end
    return linear(joined, output_weight)
end

"""
    multihead_attention(query, key, value,
                        query_weight, key_weight, value_weight, output_weight; ...)

General feature-first attention with separate projection storage. Inputs use
`(input_feature, sequence, batch...)`; query and key sequence lengths may
differ, while key and value lengths must agree.
"""
function multihead_attention(
    query_input::AbstractArray,
    key_input::AbstractArray,
    value_input::AbstractArray,
    query_weight::AbstractMatrix,
    key_weight::AbstractMatrix,
    value_weight::AbstractMatrix,
    output_weight::AbstractMatrix;
    num_heads::Integer,
    num_kv_heads::Integer=num_heads,
    rope::Union{Nothing,RotaryEmbedding}=nothing,
    query_positions=nothing,
    key_positions=nothing,
    query_position_layout::Symbol=:batch,
    key_position_layout::Symbol=:batch,
    mask=nothing,
    is_causal::Bool=false,
    layout::Symbol=:head_before_sequence,
    qk_strategy::Symbol=:auto,
)
    num_heads > 0 && num_kv_heads > 0 && num_heads % num_kv_heads == 0 ||
        throw(ArgumentError("num_heads must be positive and divisible by num_kv_heads"))
    return _attention_from_projections(
        linear(query_input, query_weight),
        linear(key_input, key_weight),
        linear(value_input, value_weight),
        output_weight;
        num_heads,
        num_kv_heads,
        rope,
        positions=query_positions,
        key_positions,
        query_position_layout,
        key_position_layout,
        mask,
        is_causal,
        layout,
        qk_strategy,
    )
end

"""
    multihead_attention(query, key, value, packed_input_weight, output_weight; ...)

General attention with Q/K/V rows packed into one input weight. Projection
execution follows input identity: shared Q/K/V uses one GEMM, shared K/V uses
two, and three distinct inputs use three. This preserves the authored Python
execution choices without storing mutable module aliases.
"""
function multihead_attention(
    query_input::AbstractArray,
    key_input::AbstractArray,
    value_input::AbstractArray,
    packed_input_weight::AbstractMatrix,
    output_weight::AbstractMatrix;
    num_heads::Integer,
    num_kv_heads::Integer=num_heads,
    qk_head_feature_count::Integer=size(output_weight, 2) ÷ num_heads,
    value_head_feature_count::Integer=size(output_weight, 2) ÷ num_heads,
    rope::Union{Nothing,RotaryEmbedding}=nothing,
    query_positions=nothing,
    key_positions=nothing,
    query_position_layout::Symbol=:batch,
    key_position_layout::Symbol=:batch,
    mask=nothing,
    is_causal::Bool=false,
    layout::Symbol=:head_before_sequence,
    qk_strategy::Symbol=:auto,
)
    num_heads > 0 && num_kv_heads > 0 && num_heads % num_kv_heads == 0 ||
        throw(ArgumentError("num_heads must be positive and divisible by num_kv_heads"))
    query_width = num_heads * qk_head_feature_count
    key_width = num_kv_heads * qk_head_feature_count
    value_width = num_kv_heads * value_head_feature_count
    size(packed_input_weight, 1) == query_width + key_width + value_width || throw(
        DimensionMismatch("packed attention projection rows do not match Q/K/V widths"),
    )
    trailing = ntuple(_ -> Colon(), ndims(query_input) - 1)

    if query_input === key_input && key_input === value_input
        projected = linear(query_input, packed_input_weight)
        projected_query = projected[1:query_width, trailing...]
        projected_key = projected[(query_width + 1):(query_width + key_width), trailing...]
        projected_value = projected[
            (query_width + key_width + 1):(query_width + key_width + value_width),
            trailing...,
        ]
    elseif key_input === value_input
        query_weight = packed_input_weight[1:query_width, :]
        key_value_weight = packed_input_weight[(query_width + 1):end, :]
        projected_query = linear(query_input, query_weight)
        projected_key_value = linear(key_input, key_value_weight)
        key_trailing = ntuple(_ -> Colon(), ndims(projected_key_value) - 1)
        projected_key = projected_key_value[1:key_width, key_trailing...]
        projected_value = projected_key_value[(key_width + 1):end, key_trailing...]
    else
        query_weight = packed_input_weight[1:query_width, :]
        key_weight = packed_input_weight[(query_width + 1):(query_width + key_width), :]
        value_weight = packed_input_weight[(query_width + key_width + 1):end, :]
        projected_query = linear(query_input, query_weight)
        projected_key = linear(key_input, key_weight)
        projected_value = linear(value_input, value_weight)
    end

    return _attention_from_projections(
        projected_query,
        projected_key,
        projected_value,
        output_weight;
        num_heads,
        num_kv_heads,
        rope,
        positions=query_positions,
        key_positions,
        query_position_layout,
        key_position_layout,
        mask,
        is_causal,
        layout,
        qk_strategy,
    )
end

"""
    multihead_self_attention(x, q_weight, k_weight, v_weight, output_weight; ...)

Self-attention with explicit projection weights. `x` uses `(model_feature,
sequence, batch...)`; weights use `(output_feature, input_feature)`. This is
the separate-projection storage lane.
"""
function multihead_self_attention(
    x::AbstractArray,
    query_weight::AbstractMatrix,
    key_weight::AbstractMatrix,
    value_weight::AbstractMatrix,
    output_weight::AbstractMatrix;
    num_heads::Integer,
    num_kv_heads::Integer=num_heads,
    rope::Union{Nothing,RotaryEmbedding}=nothing,
    positions=nothing,
    position_layout::Symbol=:batch,
    mask=nothing,
    is_causal::Bool=true,
    layout::Symbol=:head_before_sequence,
    qk_strategy::Symbol=:auto,
)
    num_heads > 0 && num_kv_heads > 0 && num_heads % num_kv_heads == 0 ||
        throw(ArgumentError("num_heads must be positive and divisible by num_kv_heads"))
    return _attention_from_projections(
        linear(x, query_weight),
        linear(x, key_weight),
        linear(x, value_weight),
        output_weight;
        num_heads,
        num_kv_heads,
        rope,
        positions,
        query_position_layout=position_layout,
        key_position_layout=position_layout,
        mask,
        is_causal,
        layout,
        qk_strategy,
    )
end

"""
    multihead_self_attention(x, packed_input_weight, output_weight; ...)

Packed-projection self-attention. Rows of `packed_input_weight` contain query,
key, then value projections, allowing one input GEMM. `qk_head_feature_count`
and `value_head_feature_count` make GQA/MQA splits unambiguous.
"""
function multihead_self_attention(
    x::AbstractArray,
    packed_input_weight::AbstractMatrix,
    output_weight::AbstractMatrix;
    num_heads::Integer,
    num_kv_heads::Integer=num_heads,
    qk_head_feature_count::Integer=size(output_weight, 2) ÷ num_heads,
    value_head_feature_count::Integer=size(output_weight, 2) ÷ num_heads,
    rope::Union{Nothing,RotaryEmbedding}=nothing,
    positions=nothing,
    position_layout::Symbol=:batch,
    mask=nothing,
    is_causal::Bool=true,
    layout::Symbol=:head_before_sequence,
    qk_strategy::Symbol=:auto,
)
    query_width = num_heads * qk_head_feature_count
    key_width = num_kv_heads * qk_head_feature_count
    value_width = num_kv_heads * value_head_feature_count
    size(packed_input_weight, 1) == query_width + key_width + value_width || throw(
        DimensionMismatch("packed attention projection rows do not match Q/K/V widths"),
    )
    projected = linear(x, packed_input_weight)
    projected_query = projected[1:query_width, ntuple(_ -> Colon(), ndims(projected) - 1)...]
    projected_key = projected[
        (query_width + 1):(query_width + key_width),
        ntuple(_ -> Colon(), ndims(projected) - 1)...,
    ]
    projected_value = projected[
        (query_width + key_width + 1):(query_width + key_width + value_width),
        ntuple(_ -> Colon(), ndims(projected) - 1)...,
    ]
    return _attention_from_projections(
        projected_query,
        projected_key,
        projected_value,
        output_weight;
        num_heads,
        num_kv_heads,
        rope,
        positions,
        query_position_layout=position_layout,
        key_position_layout=position_layout,
        mask,
        is_causal,
        layout,
        qk_strategy,
    )
end
