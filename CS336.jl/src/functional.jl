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
