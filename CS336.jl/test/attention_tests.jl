@testset "RoPE variants and execution strategies" begin
    theta = 10_000.0f0
    elementwise = RotaryEmbedding(theta, 4, 8)
    matrix = RotaryEmbedding(theta, 4, 8; variant=:matrix)
    input = reshape(Float32.(1:24), 4, 3, 2) ./ 10
    positions = repeat(reshape(Int32[0, 1, 2], 3, 1), 1, 2)

    elementwise_output = apply_rope(elementwise, input, positions)
    matrix_output = apply_rope(matrix, input, positions)
    @test size(elementwise_output) == size(input)
    @test eltype(elementwise_output) === Float32
    @test elementwise_output[:, 1, :] == input[:, 1, :]
    @test matrix_output ≈ elementwise_output rtol = 2.0f-6 atol = 2.0f-6

    position_one_angle = Float32[1, 1 / sqrt(theta)]
    expected_pair_one = Float32[
        input[1, 2, 1] * cos(position_one_angle[1]) - input[2, 2, 1] * sin(position_one_angle[1]),
        input[1, 2, 1] * sin(position_one_angle[1]) + input[2, 2, 1] * cos(position_one_angle[1]),
    ]
    @test elementwise_output[1:2, 2, 1] ≈ expected_pair_one

    vector_positions = Int32[0, 1, 2]
    @test apply_rope(elementwise, input, vector_positions)[:, :, 1] ≈
          apply_rope(elementwise, input[:, :, 1], vector_positions)

    query = reshape(Float32.(1:48), 4, 2, 3, 2) ./ 20
    key = reverse(query; dims=1)
    auto_query, auto_key = apply_rope_qk(
        elementwise,
        query,
        key,
        positions;
        sequence_dim=3,
        strategy=:auto,
    )
    separate_query, separate_key = apply_rope_qk(
        elementwise,
        query,
        key,
        positions;
        sequence_dim=3,
        strategy=:separate,
    )
    stacked_query, stacked_key = apply_rope_qk(
        elementwise,
        query,
        key,
        positions;
        sequence_dim=3,
        strategy=:stacked,
    )
    @test auto_query == separate_query == stacked_query
    @test auto_key == separate_key == stacked_key

    head_positions = Array{Int32}(undef, 2, 3, 2)
    head_positions[1, :, :] .= reshape(Int32[0, 1, 2], 3, 1)
    head_positions[2, :, :] .= reshape(Int32[2, 3, 4], 3, 1)
    head_output = apply_rope(
        elementwise,
        query,
        head_positions;
        sequence_dim=3,
        position_layout=:head,
    )
    for batch in axes(query, 4), head in axes(query, 2)
        expected = apply_rope(
            elementwise,
            query[:, head, :, batch],
            head_positions[head, :, batch],
        )
        @test head_output[:, head, :, batch] ≈ expected
    end
    head_stacked_query, head_stacked_key = apply_rope_qk(
        elementwise,
        query,
        key,
        head_positions;
        sequence_dim=3,
        strategy=:stacked,
        position_layout=:head,
    )
    @test head_stacked_query ≈ head_output
    @test size(head_stacked_key) == size(key)

    half_output = apply_rope(
        RotaryEmbedding(theta, 4, 3),
        Float16.(input[:, :, 1]),
        vector_positions,
    )
    @test eltype(half_output) === Float16
    @test_throws ArgumentError RotaryEmbedding(theta, 3, 8)
    @test_throws ArgumentError RotaryEmbedding(theta, 4, 8; variant=:unknown)
    @test_throws ArgumentError apply_rope(elementwise, input, Int32[0, 1, 8])
    @test_throws DimensionMismatch apply_rope(elementwise, input, Int32[0, 1])
    @test_throws DimensionMismatch apply_rope(
        elementwise,
        query,
        zeros(Int32, 3, 3, 2);
        sequence_dim=3,
        position_layout=:head,
    )
end

@testset "scaled dot-product attention semantics" begin
    query = zeros(Float32, 2, 3, 2)
    key = zeros(Float32, 2, 4, 2)
    value = reshape(Float32.(1:24), 3, 4, 2)
    output, weights = scaled_dot_product_attention(query, key, value; return_weights=true)
    expected_means = sum(value; dims=2) ./ 4
    @test size(output) == (3, 3, 2)
    @test weights == fill(0.25f0, 4, 3, 2)
    @test output ≈ repeat(expected_means, 1, 3, 1)

    boolean_mask = Bool[
        1 0 1
        0 0 1
        0 0 0
        0 0 0
    ]
    masked_output, masked_weights = scaled_dot_product_attention(
        query,
        key,
        value;
        mask=boolean_mask,
        return_weights=true,
    )
    @test masked_weights[:, 1, :] == repeat(Float32[1, 0, 0, 0], 1, 2)
    @test all(iszero, masked_weights[:, 2, :])
    @test all(iszero, masked_output[:, 2, :])
    @test all(isfinite, masked_output)

    additive_mask = ifelse.(boolean_mask, 0.0f0, -Inf32)
    @test scaled_dot_product_attention(query, key, value; mask=additive_mask) == masked_output

    causal_output, causal_weights = scaled_dot_product_attention(
        query,
        key,
        value;
        is_causal=true,
        return_weights=true,
    )
    @test causal_weights[:, 1, 1] == Float32[1, 0, 0, 0]
    @test causal_weights[:, 2, 1] == Float32[0.5, 0.5, 0, 0]
    @test causal_weights[:, 3, 1] ≈ Float32[1 / 3, 1 / 3, 1 / 3, 0]
    @test all(isfinite, causal_output)

    gradient_query = randn(Float32, 2, 3, 2)
    gradient_key = randn(Float32, 2, 4, 2)
    gradient_value = randn(Float32, 3, 4, 2)
    gradients = Zygote.gradient(
        (q, k, v) -> sum(abs2, scaled_dot_product_attention(q, k, v)),
        gradient_query,
        gradient_key,
        gradient_value,
    )
    @test size.(gradients) == size.((gradient_query, gradient_key, gradient_value))
    @test all(gradient -> all(isfinite, gradient), gradients)
end

@testset "headed MHA, GQA, and MQA layouts" begin
    query = randn(Float32, 3, 4, 5, 2)
    key = randn(Float32, 3, 2, 5, 2)
    value = randn(Float32, 2, 2, 5, 2)
    grouped = headed_scaled_dot_product_attention(
        query,
        key,
        value;
        is_causal=true,
    )
    expanded_key = repeat(key; inner=(1, 2, 1, 1))
    expanded_value = repeat(value; inner=(1, 2, 1, 1))
    expanded = headed_scaled_dot_product_attention(
        query,
        expanded_key,
        expanded_value;
        is_causal=true,
    )
    @test grouped ≈ expanded rtol = 2.0f-5 atol = 2.0f-5

    head_after_query = permutedims(query, (1, 3, 2, 4))
    head_after_key = permutedims(key, (1, 3, 2, 4))
    head_after_value = permutedims(value, (1, 3, 2, 4))
    head_after = headed_scaled_dot_product_attention(
        head_after_query,
        head_after_key,
        head_after_value;
        is_causal=true,
        layout=:head_after_sequence,
    )
    @test permutedims(head_after, (1, 3, 2, 4)) ≈ grouped rtol = 2.0f-5 atol = 2.0f-5

    multiquery_key = key[:, 1:1, :, :]
    multiquery_value = value[:, 1:1, :, :]
    multiquery = headed_scaled_dot_product_attention(
        query,
        multiquery_key,
        multiquery_value;
        is_causal=true,
    )
    @test size(multiquery) == (2, 4, 5, 2)
end

@testset "self-attention storage and execution variants" begin
    d_model = 8
    num_heads = 4
    input = randn(Float32, d_model, 5, 2)
    query_weight = randn(Float32, d_model, d_model)
    key_weight = randn(Float32, d_model, d_model)
    value_weight = randn(Float32, d_model, d_model)
    output_weight = randn(Float32, d_model, d_model)
    packed_weight = vcat(query_weight, key_weight, value_weight)

    separate = multihead_self_attention(
        input,
        query_weight,
        key_weight,
        value_weight,
        output_weight;
        num_heads,
    )
    packed = multihead_self_attention(input, packed_weight, output_weight; num_heads)
    head_after = multihead_self_attention(
        input,
        packed_weight,
        output_weight;
        num_heads,
        layout=:head_after_sequence,
    )
    @test separate == packed == head_after

    rope = RotaryEmbedding(10_000.0f0, d_model ÷ num_heads, 5)
    positions = repeat(reshape(Int32.(0:4), 5, 1), 1, 2)
    rope_auto = multihead_self_attention(
        input,
        packed_weight,
        output_weight;
        num_heads,
        rope,
        positions,
        qk_strategy=:auto,
    )
    rope_separate = multihead_self_attention(
        input,
        packed_weight,
        output_weight;
        num_heads,
        rope,
        positions,
        qk_strategy=:separate,
    )
    rope_stacked = multihead_self_attention(
        input,
        packed_weight,
        output_weight;
        num_heads,
        rope,
        positions,
        qk_strategy=:stacked,
    )
    @test rope_auto == rope_separate == rope_stacked

    input_gradient = only(
        Zygote.gradient(
            x -> sum(abs2, multihead_self_attention(x, packed_weight, output_weight; num_heads)),
            input,
        ),
    )
    @test size(input_gradient) == size(input)
    @test all(isfinite, input_gradient)
end

@testset "general attention input-sharing paths" begin
    d_model = 8
    num_heads = 2
    query_input = randn(Float32, d_model, 3, 2)
    key_input = randn(Float32, d_model, 4, 2)
    value_input = randn(Float32, d_model, 4, 2)
    query_weight = randn(Float32, d_model, d_model)
    key_weight = randn(Float32, d_model, d_model)
    value_weight = randn(Float32, d_model, d_model)
    output_weight = randn(Float32, d_model, d_model)
    packed_weight = vcat(query_weight, key_weight, value_weight)

    separate = multihead_attention(
        query_input,
        key_input,
        value_input,
        query_weight,
        key_weight,
        value_weight,
        output_weight;
        num_heads,
    )
    packed_distinct = multihead_attention(
        query_input,
        key_input,
        value_input,
        packed_weight,
        output_weight;
        num_heads,
    )
    @test packed_distinct ≈ separate rtol = 3.0f-5 atol = 3.0f-5

    separate_shared_kv = multihead_attention(
        query_input,
        key_input,
        key_input,
        query_weight,
        key_weight,
        value_weight,
        output_weight;
        num_heads,
    )
    packed_shared_kv = multihead_attention(
        query_input,
        key_input,
        key_input,
        packed_weight,
        output_weight;
        num_heads,
    )
    @test packed_shared_kv ≈ separate_shared_kv rtol = 3.0f-5 atol = 3.0f-5

    self_input = randn(Float32, d_model, 3, 2)
    packed_shared_qkv = multihead_attention(
        self_input,
        self_input,
        self_input,
        packed_weight,
        output_weight;
        num_heads,
        is_causal=true,
    )
    self_attention = multihead_self_attention(
        self_input,
        packed_weight,
        output_weight;
        num_heads,
    )
    @test packed_shared_qkv == self_attention

    rope = RotaryEmbedding(10_000.0f0, d_model ÷ num_heads, 5)
    query_positions = Int32[0, 2, 4]
    key_positions = Int32[0, 1, 3, 4]
    rope_separate = multihead_attention(
        query_input,
        key_input,
        value_input,
        query_weight,
        key_weight,
        value_weight,
        output_weight;
        num_heads,
        rope,
        query_positions,
        key_positions,
    )
    rope_packed = multihead_attention(
        query_input,
        key_input,
        value_input,
        packed_weight,
        output_weight;
        num_heads,
        rope,
        query_positions,
        key_positions,
    )
    @test rope_packed ≈ rope_separate rtol = 3.0f-5 atol = 3.0f-5
    @test_throws DimensionMismatch multihead_attention(
        query_input,
        key_input,
        value_input,
        packed_weight,
        output_weight;
        num_heads,
        rope,
        query_positions,
        key_positions,
        qk_strategy=:stacked,
    )

    gradients = Zygote.gradient(
        (q, k, v, packed, output) -> sum(
            abs2,
            multihead_attention(q, k, v, packed, output; num_heads),
        ),
        query_input,
        key_input,
        value_input,
        packed_weight,
        output_weight,
    )
    @test size.(gradients) == size.(
        (query_input, key_input, value_input, packed_weight, output_weight),
    )
    @test all(gradient -> all(isfinite, gradient), gradients)

    head_query_positions = repeat(
        reshape(Int32[0, 2, 4, 1, 3, 4], num_heads, 3, 1),
        1,
        1,
        2,
    )
    head_key_positions = repeat(
        reshape(Int32[0, 1, 3, 4, 1, 2, 3, 4], num_heads, 4, 1),
        1,
        1,
        2,
    )
    head_position_output = multihead_attention(
        query_input,
        key_input,
        value_input,
        packed_weight,
        output_weight;
        num_heads,
        rope,
        query_positions=head_query_positions,
        key_positions=head_key_positions,
        query_position_layout=:head,
        key_position_layout=:head,
    )
    @test size(head_position_output) == size(query_input)
    @test all(isfinite, head_position_output)
end

attention_fixture_path(stem) = normpath(
    joinpath(
        @__DIR__,
        "..",
        "..",
        "tests",
        "fixtures",
        "julia_parity",
        "v1",
        "$stem.json",
    ),
)

@testset "RoPE Python parity" begin
    bundle = load_bundle(attention_fixture_path("rope"))
    arrays = bundle.arrays
    tolerances = bundle.metadata["tolerances"]
    input = permutedims(arrays["input.x"], (3, 2, 1))
    positions = permutedims(arrays["input.positions"], (2, 1))
    expected_output = permutedims(arrays["expected.output"], (3, 2, 1))
    expected_gradient =
        permutedims(arrays["expected.gradient.input.x"], (3, 2, 1))
    scalars = bundle.metadata["scalars"]
    elementwise = RotaryEmbedding(
        scalars["theta"],
        scalars["d_k"],
        scalars["max_seq_len"],
    )
    matrix = RotaryEmbedding(
        scalars["theta"],
        scalars["d_k"],
        scalars["max_seq_len"];
        variant=:matrix,
    )

    output = apply_rope(elementwise, input, positions)
    matrix_output = apply_rope(matrix, input, positions)
    gradient = only(
        Zygote.gradient(x -> sum(abs2, apply_rope(elementwise, x, positions)), input),
    )
    rtol = tolerances["rtol"]
    atol = tolerances["atol"]
    nans = tolerances["equal_nan"]
    @test bundle.metadata["operation"] == "run_rope"
    @test isapprox(output, expected_output; rtol, atol, nans)
    @test isapprox(matrix_output, expected_output; rtol, atol, nans)
    @test isapprox(gradient, expected_gradient; rtol, atol, nans)
end

@testset "scaled attention Python parity" begin
    bundle = load_bundle(attention_fixture_path("scaled_dot_product_attention"))
    arrays = bundle.arrays
    tolerances = bundle.metadata["tolerances"]
    to_feature_first(array) = permutedims(array, (4, 3, 2, 1))
    query = to_feature_first(arrays["input.query"])
    key = to_feature_first(arrays["input.key"])
    value = to_feature_first(arrays["input.value"])
    mask = to_feature_first(arrays["input.mask"])
    expected_output = to_feature_first(arrays["expected.output"])
    expected_gradients = (
        to_feature_first(arrays["expected.gradient.input.query"]),
        to_feature_first(arrays["expected.gradient.input.key"]),
        to_feature_first(arrays["expected.gradient.input.value"]),
    )

    output = scaled_dot_product_attention(query, key, value; mask)
    gradients = Zygote.gradient(
        (q, k, v) -> sum(abs2, scaled_dot_product_attention(q, k, v; mask)),
        query,
        key,
        value,
    )
    rtol = tolerances["rtol"]
    atol = tolerances["atol"]
    nans = tolerances["equal_nan"]
    @test bundle.metadata["operation"] == "run_scaled_dot_product_attention"
    @test isapprox(output, expected_output; rtol, atol, nans)
    for (gradient, expected) in zip(gradients, expected_gradients)
        @test isapprox(gradient, expected; rtol, atol, nans)
    end
    @test all(iszero, output[:, 2, 1, 1])
end

@testset "self-attention Python parity" begin
    for (stem, operation, with_rope) in (
        ("multihead_self_attention", "run_multihead_self_attention", false),
        (
            "multihead_self_attention_with_rope",
            "run_multihead_self_attention_with_rope",
            true,
        ),
    )
        bundle = load_bundle(attention_fixture_path(stem))
        arrays = bundle.arrays
        tolerances = bundle.metadata["tolerances"]
        scalars = bundle.metadata["scalars"]
        input = permutedims(arrays["input.x"], (3, 2, 1))
        query_weight = arrays["parameter.query_weight"]
        key_weight = arrays["parameter.key_weight"]
        value_weight = arrays["parameter.value_weight"]
        output_weight = arrays["parameter.output_weight"]
        packed_weight = vcat(query_weight, key_weight, value_weight)
        expected_output = permutedims(arrays["expected.output"], (3, 2, 1))
        expected_gradients = (
            permutedims(arrays["expected.gradient.input.x"], (3, 2, 1)),
            arrays["expected.gradient.parameter.query_weight"],
            arrays["expected.gradient.parameter.key_weight"],
            arrays["expected.gradient.parameter.value_weight"],
            arrays["expected.gradient.parameter.output_weight"],
        )
        rope =
            with_rope ?
            RotaryEmbedding(
                scalars["theta"],
                scalars["d_model"] ÷ scalars["num_heads"],
                scalars["max_seq_len"],
            ) : nothing
        positions = with_rope ? permutedims(arrays["input.positions"], (2, 1)) : nothing
        keywords = (;
            num_heads=scalars["num_heads"],
            rope,
            positions,
        )

        separate_output = multihead_self_attention(
            input,
            query_weight,
            key_weight,
            value_weight,
            output_weight;
            keywords...,
        )
        packed_output = multihead_self_attention(
            input,
            packed_weight,
            output_weight;
            keywords...,
        )
        separate_gradients = Zygote.gradient(
            (x, q, k, v, o) -> sum(
                abs2,
                multihead_self_attention(x, q, k, v, o; keywords...),
            ),
            input,
            query_weight,
            key_weight,
            value_weight,
            output_weight,
        )
        packed_gradients = Zygote.gradient(
            (x, packed, o) -> sum(
                abs2,
                multihead_self_attention(x, packed, o; keywords...),
            ),
            input,
            packed_weight,
            output_weight,
        )
        expected_packed_gradients = (
            expected_gradients[1],
            vcat(expected_gradients[2], expected_gradients[3], expected_gradients[4]),
            expected_gradients[5],
        )
        rtol = tolerances["rtol"]
        atol = tolerances["atol"]
        nans = tolerances["equal_nan"]
        @test bundle.metadata["operation"] == operation
        @test isapprox(separate_output, expected_output; rtol, atol, nans)
        @test isapprox(packed_output, expected_output; rtol, atol, nans)
        for (gradient, expected) in zip(separate_gradients, expected_gradients)
            @test isapprox(gradient, expected; rtol, atol, nans)
        end
        for (gradient, expected) in zip(packed_gradients, expected_packed_gradients)
            @test isapprox(gradient, expected; rtol, atol, nans)
        end
    end
end
