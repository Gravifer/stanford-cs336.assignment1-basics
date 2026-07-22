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
    d_model = positive_argument(arguments, 1, 512, "d_model")
    num_heads = positive_argument(arguments, 2, 8, "num_heads")
    num_kv_heads = positive_argument(arguments, 3, 2, "num_kv_heads")
    sequence_length = positive_argument(arguments, 4, 128, "sequence_length")
    batch_size = positive_argument(arguments, 5, 4, "batch_size")
    samples = positive_argument(arguments, 6, 20, "samples")
    seconds = positive_argument(arguments, 7, 5.0, "seconds")
    d_model % num_heads == 0 || error("d_model must be divisible by num_heads")
    num_heads % num_kv_heads == 0 || error("num_heads must be divisible by num_kv_heads")
    head_width = d_model ÷ num_heads

    rng = Xoshiro(343)
    input = randn(rng, Float32, d_model, sequence_length, batch_size)
    query_weight = randn(rng, Float32, d_model, d_model)
    key_weight = randn(rng, Float32, d_model, d_model)
    value_weight = randn(rng, Float32, d_model, d_model)
    output_weight = randn(rng, Float32, d_model, d_model)
    packed_weight = vcat(query_weight, key_weight, value_weight)
    cross_query = randn(rng, Float32, d_model, sequence_length ÷ 2 + 1, batch_size)
    cross_key = randn(rng, Float32, d_model, sequence_length, batch_size)
    cross_value = randn(rng, Float32, d_model, sequence_length, batch_size)
    rope_elementwise = RotaryEmbedding(10_000.0f0, head_width, sequence_length)
    rope_matrix = RotaryEmbedding(
        10_000.0f0,
        head_width,
        sequence_length;
        variant=:matrix,
    )
    positions = Int32.(0:(sequence_length - 1))
    headed_input = reshape(input, head_width, num_heads, sequence_length, batch_size)

    kv_width = num_kv_heads * head_width
    grouped_key_weight = randn(rng, Float32, kv_width, d_model)
    grouped_value_weight = randn(rng, Float32, kv_width, d_model)
    grouped_weight = vcat(query_weight, grouped_key_weight, grouped_value_weight)
    multiquery_key_weight = randn(rng, Float32, head_width, d_model)
    multiquery_value_weight = randn(rng, Float32, head_width, d_model)
    multiquery_weight = vcat(query_weight, multiquery_key_weight, multiquery_value_weight)

    separate_output = multihead_self_attention(
        input,
        query_weight,
        key_weight,
        value_weight,
        output_weight;
        num_heads,
    )
    packed_output = multihead_self_attention(input, packed_weight, output_weight; num_heads)
    isapprox(separate_output, packed_output; rtol=1.0f-4, atol=1.0f-4) ||
        error("packed and separate self-attention disagree")
    head_after_output = multihead_self_attention(
        input,
        packed_weight,
        output_weight;
        num_heads,
        layout=:head_after_sequence,
    )
    isapprox(packed_output, head_after_output; rtol=1.0f-4, atol=1.0f-4) ||
        error("head layouts disagree")
    cross_separate = multihead_attention(
        cross_query,
        cross_key,
        cross_value,
        query_weight,
        key_weight,
        value_weight,
        output_weight;
        num_heads,
    )
    cross_packed = multihead_attention(
        cross_query,
        cross_key,
        cross_value,
        packed_weight,
        output_weight;
        num_heads,
    )
    isapprox(cross_separate, cross_packed; rtol=1.0f-4, atol=1.0f-4) ||
        error("packed and separate cross-attention disagree")
    rope_elementwise_output = apply_rope(
        rope_elementwise,
        headed_input,
        positions;
        sequence_dim=3,
    )
    rope_matrix_output = apply_rope(
        rope_matrix,
        headed_input,
        positions;
        sequence_dim=3,
    )
    isapprox(rope_elementwise_output, rope_matrix_output; rtol=1.0f-4, atol=1.0f-4) ||
        error("RoPE representations disagree")

    packed_objective = (x, input_weight, output) -> squared_sum(
        multihead_self_attention(x, input_weight, output; num_heads),
    )
    rope_elementwise_benchmark = @benchmarkable apply_rope(
            $rope_elementwise,
            $headed_input,
            $positions;
            sequence_dim=3,
        )
    rope_matrix_benchmark = @benchmarkable apply_rope(
            $rope_matrix,
            $headed_input,
            $positions;
            sequence_dim=3,
        )
    self_separate_benchmark = @benchmarkable multihead_self_attention(
            $input,
            $query_weight,
            $key_weight,
            $value_weight,
            $output_weight;
            num_heads=$num_heads,
        )
    self_packed_benchmark = @benchmarkable multihead_self_attention(
            $input,
            $packed_weight,
            $output_weight;
            num_heads=$num_heads,
        )
    self_head_after_benchmark = @benchmarkable multihead_self_attention(
            $input,
            $packed_weight,
            $output_weight;
            num_heads=$num_heads,
            layout=:head_after_sequence,
        )
    gqa_packed_benchmark = @benchmarkable multihead_self_attention(
            $input,
            $grouped_weight,
            $output_weight;
            num_heads=$num_heads,
            num_kv_heads=$num_kv_heads,
        )
    mqa_packed_benchmark = @benchmarkable multihead_self_attention(
            $input,
            $multiquery_weight,
            $output_weight;
            num_heads=$num_heads,
            num_kv_heads=1,
        )
    cross_separate_benchmark = @benchmarkable multihead_attention(
            $cross_query,
            $cross_key,
            $cross_value,
            $query_weight,
            $key_weight,
            $value_weight,
            $output_weight;
            num_heads=$num_heads,
        )
    cross_packed_distinct_benchmark = @benchmarkable multihead_attention(
            $cross_query,
            $cross_key,
            $cross_value,
            $packed_weight,
            $output_weight;
            num_heads=$num_heads,
        )
    cross_packed_shared_kv_benchmark = @benchmarkable multihead_attention(
            $cross_query,
            $cross_key,
            $cross_key,
            $packed_weight,
            $output_weight;
            num_heads=$num_heads,
        )
    self_packed_backward_benchmark = @benchmarkable Zygote.gradient(
            $packed_objective,
            $input,
            $packed_weight,
            $output_weight,
        )
    benchmarks = [
        "rope_elementwise_forward" => rope_elementwise_benchmark,
        "rope_matrix_forward" => rope_matrix_benchmark,
        "self_separate_forward" => self_separate_benchmark,
        "self_packed_forward" => self_packed_benchmark,
        "self_head_after_forward" => self_head_after_benchmark,
        "gqa_packed_forward" => gqa_packed_benchmark,
        "mqa_packed_forward" => mqa_packed_benchmark,
        "cross_separate_forward" => cross_separate_benchmark,
        "cross_packed_distinct_forward" => cross_packed_distinct_benchmark,
        "cross_packed_shared_kv_forward" => cross_packed_shared_kv_benchmark,
        "self_packed_backward" => self_packed_backward_benchmark,
    ]

    println(
        "d_model=$d_model heads=$num_heads kv_heads=$num_kv_heads " *
        "sequence=$sequence_length batch=$batch_size dtype=Float32",
    )
    println("variant\tmedian_ns\tmemory_bytes\tallocations\tsamples")
    options = (; samples, seconds, evals=1)
    for (name, benchmark) in benchmarks
        print_trial(name, BenchmarkTools.run(benchmark; options...))
    end
    return nothing
end

main()
