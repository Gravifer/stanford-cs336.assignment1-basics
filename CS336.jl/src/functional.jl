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
