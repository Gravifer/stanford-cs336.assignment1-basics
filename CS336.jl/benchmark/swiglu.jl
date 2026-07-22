using BenchmarkTools
using CS336
using Random
using Statistics
using Zygote

function positive_argument(
    arguments::Vector{String},
    index::Integer,
    default::T,
    name::AbstractString,
) where {T<:Real}
    index > length(arguments) && return default
    value = parse(T, arguments[index])
    value > zero(T) || throw(ArgumentError("$name must be positive, got $value"))
    return value
end

default_hidden_width(model_width::Integer) =
    64 * max(1, (model_width + 12) ÷ 24)

explicit_objective(x, w1, w2, w3) = sum(abs2, swiglu(x, w1, w2, w3))
packed_objective(x, input_weight, output_weight) =
    sum(abs2, swiglu(x, input_weight, output_weight))

function assert_parity(x, w1, w2, w3, packed; rtol=1.0f-4, atol=1.0f-5)
    explicit_output = swiglu(x, w1, w2, w3)
    packed_output = swiglu(x, packed, w2)
    isapprox(explicit_output, packed_output; rtol, atol) ||
        error("explicit and packed SwiGLU forward paths disagree")

    explicit_gradients = Zygote.gradient(explicit_objective, x, w1, w2, w3)
    packed_gradients = Zygote.gradient(packed_objective, x, packed, w2)
    expected_packed_gradient = vcat(explicit_gradients[4], explicit_gradients[2])
    comparisons = (
        (explicit_gradients[1], packed_gradients[1], "input"),
        (explicit_gradients[3], packed_gradients[3], "output weight"),
        (expected_packed_gradient, packed_gradients[2], "packed input weight"),
    )
    for (expected, actual, name) in comparisons
        size(actual) == size(expected) ||
            error("unexpected $name gradient shape $(size(actual)); expected $(size(expected))")
        isapprox(actual, expected; rtol, atol) ||
            error("explicit and packed SwiGLU $name gradients disagree")
    end
    return nothing
end

function print_trial(name::AbstractString, trial::BenchmarkTools.Trial)
    estimate = median(trial)
    println(
        join(
            (
                name,
                string(estimate.time),
                string(estimate.memory),
                string(estimate.allocs),
                string(length(trial)),
            ),
            '\t',
        ),
    )
end

function main(arguments::Vector{String}=ARGS)
    model_width = positive_argument(arguments, 1, 512, "d_model")
    hidden_width = positive_argument(
        arguments,
        2,
        default_hidden_width(model_width),
        "d_ff",
    )
    tokens = positive_argument(arguments, 3, 512, "tokens")
    samples = positive_argument(arguments, 4, 20, "samples")
    seconds = positive_argument(arguments, 5, 5.0, "seconds")

    rng = Xoshiro(0)
    x = randn(rng, Float32, model_width, tokens)
    w1 = randn(rng, Float32, hidden_width, model_width)
    w2 = randn(rng, Float32, model_width, hidden_width)
    w3 = randn(rng, Float32, hidden_width, model_width)
    packed = vcat(w3, w1)
    assert_parity(x, w1, w2, w3, packed)

    println("d_model=$model_width d_ff=$hidden_width tokens=$tokens dtype=Float32")
    println("variant\tmedian_ns\tmemory_bytes\tallocations\tsamples")
    explicit_loss = explicit_objective
    packed_loss = packed_objective
    explicit_forward_benchmark = @benchmarkable swiglu($x, $w1, $w2, $w3)
    packed_forward_benchmark = @benchmarkable swiglu($x, $packed, $w2)
    explicit_backward_benchmark = @benchmarkable Zygote.gradient(
        $explicit_loss,
        $x,
        $w1,
        $w2,
        $w3,
    )
    packed_backward_benchmark = @benchmarkable Zygote.gradient(
        $packed_loss,
        $x,
        $packed,
        $w2,
    )
    benchmark_options = (; samples, seconds, evals=1)
    explicit_forward = BenchmarkTools.run(explicit_forward_benchmark; benchmark_options...)
    packed_forward = BenchmarkTools.run(packed_forward_benchmark; benchmark_options...)
    explicit_backward =
        BenchmarkTools.run(explicit_backward_benchmark; benchmark_options...)
    packed_backward =
        BenchmarkTools.run(packed_backward_benchmark; benchmark_options...)

    print_trial("explicit_forward", explicit_forward)
    print_trial("packed_forward", packed_forward)
    print_trial("explicit_backward", explicit_backward)
    print_trial("packed_backward", packed_backward)
    return nothing
end

main()
