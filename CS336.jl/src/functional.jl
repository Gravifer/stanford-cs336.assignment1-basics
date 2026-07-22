"""
    linear(x, weight)

Apply a bias-free linear transformation to the leading feature axis of `x`.

`x` uses the package's canonical `(input_feature, mapped...)` layout and
`weight` uses `(output_feature, input_feature)`. The result has shape
`(output_feature, mapped...)`. With Julia's column-major arrays, flattening the
mapped axes makes every feature vector a contiguous matrix column, so the
kernel lowers to `weight * x₂d` without an internal layout conversion.
"""
function linear(x::AbstractArray, weight::AbstractMatrix)
    ndims(x) > 0 || throw(DimensionMismatch("linear input must have a feature axis"))

    input_features = size(x, 1)
    size(weight, 2) == input_features || throw(
        DimensionMismatch(
            "linear weight has $(size(weight, 2)) input features, but input has $input_features",
        ),
    )

    mapped_shape = ntuple(axis -> size(x, axis + 1), ndims(x) - 1)
    output = weight * reshape(x, input_features, :)
    return reshape(output, size(weight, 1), mapped_shape...)
end

"""
    embedding(token_ids, weight)

Look up zero-based external token IDs in a `(feature, vocabulary)` weight
matrix and return `(feature, size(token_ids)...)`.

Vocabulary entries are columns so each embedding vector is contiguous in
Julia's column-major storage. The conversion to one-based indices happens only
at this lookup boundary; `token_ids` itself is never rewritten.
"""
function embedding(token_ids::AbstractArray{<:Integer}, weight::AbstractMatrix)
    vocabulary_size = size(weight, 2)
    all(token_id -> 0 <= token_id < vocabulary_size, token_ids) || throw(
        ArgumentError(
            "embedding token IDs must be in 0:$(vocabulary_size - 1)",
        ),
    )

    indices = vec(token_ids .+ one(eltype(token_ids)))
    selected = weight[:, indices]
    return reshape(selected, size(weight, 1), size(token_ids)...)
end

"""
    silu(x)

Apply the sigmoid linear unit elementwise. Arrays retain their shape and use a
fully broadcasted implementation so the same function specializes for CPU and
accelerator array types.
"""
silu(x::Number) = x / (one(x) + exp(-x))
silu(x::AbstractArray) = silu.(x)

"""
    softmax(x; dims)

Compute a numerically stable softmax over the one-based Julia dimension
`dims`. The maximum is retained along the reduced dimension, making the
broadcasting contract explicit for arrays of any rank.
"""
function softmax(x::AbstractArray; dims::Integer)
    1 <= dims <= ndims(x) || throw(
        ArgumentError("softmax dims must be in 1:$(ndims(x)), received $dims"),
    )

    shifted = x .- maximum(x; dims)
    exponentials = exp.(shifted)
    return exponentials ./ sum(exponentials; dims)
end

"""
    rmsnorm(x, weight; eps=1e-5)

Apply RMS normalization over the leading feature axis of `x`, followed by the
explicit affine `weight`. Float16 inputs are reduced in Float32 and the result
is converted back to the input element type, matching the adapter contract.
"""
function rmsnorm(x::AbstractArray, weight::AbstractVector; eps::Real=1e-5)
    ndims(x) > 0 || throw(DimensionMismatch("RMSNorm input must have a feature axis"))
    feature_count = size(x, 1)
    feature_count > 0 || throw(DimensionMismatch("RMSNorm feature axis must not be empty"))
    length(weight) == feature_count || throw(
        DimensionMismatch(
            "RMSNorm weight has $(length(weight)) features, but input has $feature_count",
        ),
    )

    promoted_type = promote_type(eltype(x), eltype(weight))
    operation_type =
        promoted_type in (Float32, Float64, ComplexF32, ComplexF64) ?
        promoted_type : Float32
    operation_input = eltype(x) === operation_type ? x : operation_type.(x)
    operation_weight =
        eltype(weight) === operation_type ? weight : operation_type.(weight)

    mean_square = sum(abs2, operation_input; dims=1) ./ feature_count
    scale = inv.(sqrt.(mean_square .+ operation_type(eps)))
    affine_shape = (feature_count, ntuple(_ -> 1, ndims(x) - 1)...)
    output = operation_input .* scale .* reshape(operation_weight, affine_shape)
    return eltype(output) === eltype(x) ? output : eltype(x).(output)
end

"""
    default_feed_forward_width(d_model)

Return the authored SwiGLU default width: approximately `8d_model/3`, rounded
to the course implementation's nearest positive multiple-of-64 convention.
Explicit widths remain valid and bypass this sizing policy.
"""
function default_feed_forward_width(d_model::Integer)
    d_model > 0 || throw(ArgumentError("d_model must be positive"))
    return 64 * max(1, (d_model + 12) ÷ 24)
end

"""
    swiglu(x, w1, w2, w3)

Apply the adapter-semantic SwiGLU transformation with explicit weights:
`w2 * (silu(w1 * x) .* (w3 * x))`. All activations use the canonical leading
feature axis. The three-argument method selects packed value/gate storage.
"""
function swiglu(
    x::AbstractArray,
    w1::AbstractMatrix,
    w2::AbstractMatrix,
    w3::AbstractMatrix,
)
    gate = linear(x, w1)
    value = linear(x, w3)
    size(gate) == size(value) || throw(
        DimensionMismatch("SwiGLU w1 and w3 projections must have equal shapes"),
    )
    return linear(silu(gate) .* value, w2)
end

"""
    swiglu_packed(x, input_weight, output_weight)

Apply SwiGLU with a packed input projection. `input_weight` stores value rows
followed by gate rows, matching the adapter-selected Python variant, and must
therefore have an even number of rows. This uses one input GEMM while retaining
an explicit output weight.
"""
function swiglu_packed(
    x::AbstractArray,
    input_weight::AbstractMatrix,
    output_weight::AbstractMatrix,
)
    packed_features = size(input_weight, 1)
    iseven(packed_features) || throw(
        DimensionMismatch("packed SwiGLU input weight must have an even row count"),
    )
    hidden_features = packed_features ÷ 2
    hidden_features > 0 || throw(
        DimensionMismatch("packed SwiGLU hidden feature count must be positive"),
    )

    projected = linear(x, input_weight)
    value = selectdim(projected, 1, 1:hidden_features)
    gate = selectdim(projected, 1, (hidden_features + 1):packed_features)
    return linear(silu(gate) .* value, output_weight)
end

"""
    swiglu(x, input_weight, output_weight)

Select the packed-input SwiGLU implementation through the common `swiglu`
name. This arity stores value rows before gate rows and performs one input
projection; `swiglu_packed` remains available when a descriptive name is
clearer at a call site.
"""
swiglu(
    x::AbstractArray,
    input_weight::AbstractMatrix,
    output_weight::AbstractMatrix,
) = swiglu_packed(x, input_weight, output_weight)
