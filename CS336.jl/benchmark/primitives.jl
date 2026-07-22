using BenchmarkTools
using CS336
using Random
using Statistics
using Zygote

function positive_argument(arguments, index, default::T, name) where {T<:Real}
    index > length(arguments) && return default
    value = parse(T, arguments[index])
    value > zero(T) || throw(ArgumentError("$name must be positive, got $value"))
    return value
end

function print_trial(name, trial)
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

squared_sum(array) = sum(array .* array)

function main(arguments=ARGS)
    feature_count = positive_argument(arguments, 1, 512, "feature_count")
    token_count = positive_argument(arguments, 2, 512, "token_count")
    vocabulary_size = positive_argument(arguments, 3, 8_192, "vocabulary_size")
    samples = positive_argument(arguments, 4, 20, "samples")
    seconds = positive_argument(arguments, 5, 5.0, "seconds")

    rng = Xoshiro(344)
    input = randn(rng, Float32, feature_count, token_count)
    linear_weight = randn(rng, Float32, feature_count, feature_count)
    norm_weight = randn(rng, Float32, feature_count)
    embedding_weight = randn(rng, Float32, feature_count, vocabulary_size)
    token_ids = Int32.(mod.(0:(token_count - 1), vocabulary_size))

    @assert size(linear(input, linear_weight)) == size(input)
    @assert size(embedding(token_ids, embedding_weight)) == size(input)
    @assert all(isfinite, silu(input))
    @assert all(isapprox.(vec(sum(softmax(input; dims=1); dims=1)), 1.0f0))
    @assert size(rmsnorm(input, norm_weight)) == size(input)

    linear_objective = (x, weight) -> squared_sum(linear(x, weight))
    embedding_objective = weight -> squared_sum(embedding(token_ids, weight))
    activation_objective = x -> squared_sum(silu(x))
    softmax_objective = x -> squared_sum(softmax(x; dims=1))
    norm_objective = (x, weight) -> squared_sum(rmsnorm(x, weight))

    linear_forward = @benchmarkable linear($input, $linear_weight)
    embedding_forward = @benchmarkable embedding($token_ids, $embedding_weight)
    silu_forward = @benchmarkable silu($input)
    softmax_forward = @benchmarkable softmax($input; dims=1)
    rmsnorm_forward = @benchmarkable rmsnorm($input, $norm_weight)
    linear_backward = @benchmarkable Zygote.gradient(
        $linear_objective,
        $input,
        $linear_weight,
    )
    embedding_backward = @benchmarkable Zygote.gradient(
        $embedding_objective,
        $embedding_weight,
    )
    silu_backward = @benchmarkable Zygote.gradient($activation_objective, $input)
    softmax_backward = @benchmarkable Zygote.gradient($softmax_objective, $input)
    rmsnorm_backward = @benchmarkable Zygote.gradient(
        $norm_objective,
        $input,
        $norm_weight,
    )
    benchmarks = [
        "linear_forward" => linear_forward,
        "embedding_forward" => embedding_forward,
        "silu_forward" => silu_forward,
        "softmax_forward" => softmax_forward,
        "rmsnorm_forward" => rmsnorm_forward,
        "linear_backward" => linear_backward,
        "embedding_backward" => embedding_backward,
        "silu_backward" => silu_backward,
        "softmax_backward" => softmax_backward,
        "rmsnorm_backward" => rmsnorm_backward,
    ]
    println(
        "features=$feature_count tokens=$token_count vocabulary=$vocabulary_size dtype=Float32",
    )
    println("variant\tmedian_ns\tmemory_bytes\tallocations\tsamples")
    options = (; samples, seconds, evals=1)
    for (name, benchmark) in benchmarks
        print_trial(name, BenchmarkTools.run(benchmark; options...))
    end
    return nothing
end

main()
