@testset "LuxCore TransformerBlock interface" begin
    rng = Xoshiro(336)
    block = TransformerBlock(8, 2, 16, 6, 10_000.0f0)
    parameters, state = LuxCore.setup(rng, block)
    input = randn(rng, Float32, 8, 4, 2)
    output, returned_state = block(input, parameters, state)

    @test block isa LuxCore.AbstractLuxLayer
    @test size(output) == size(input)
    @test eltype(output) === Float32
    @test returned_state === state
    @test keys(parameters) ==
          (:attention_norm, :attention, :feed_forward_norm, :feed_forward)
    @test size(parameters.attention.input_weight) == (24, 8)
    @test size(parameters.attention.output_weight) == (8, 8)
    @test size(parameters.feed_forward.input_weight) == (32, 8)
    @test size(parameters.feed_forward.output_weight) == (8, 16)
    @test keys(state) == (:rope, :positions)
    @test state.positions == Int32.(0:5)
    @test LuxCore.parameterlength(block) == 656
    @test LuxCore.statelength(block) == 30

    explicit_positions = repeat(reshape(Int32.(0:3), 4, 1), 1, 2)
    explicit_output, _ = block((input, explicit_positions), parameters, state)
    @test explicit_output ≈ output

    matrix_block = TransformerBlock(8, 2, 16, 6, 10_000.0f0; rope_variant=:matrix)
    _, matrix_state = LuxCore.setup(Xoshiro(337), matrix_block)
    matrix_output, _ = matrix_block(input, parameters, matrix_state)
    @test matrix_output ≈ output rtol = 3.0f-5 atol = 3.0f-5

    input_gradient = only(
        Zygote.gradient(
            x -> sum(abs2, first(block(x, parameters, state))),
            input,
        ),
    )
    parameter_gradient = only(
        Zygote.gradient(
            ps -> sum(abs2, first(block(input, ps, state))),
            parameters,
        ),
    )
    @test size(input_gradient) == size(input)
    @test all(isfinite, input_gradient)
    @test size(parameter_gradient.attention.input_weight) == (24, 8)
    @test size(parameter_gradient.feed_forward.input_weight) == (32, 8)

    @test_throws ArgumentError TransformerBlock(8, 3, 16, 6, 10_000.0)
    @test_throws ArgumentError TransformerBlock(8, 2, 16, 0, 10_000.0)
    @test_throws DimensionMismatch block(randn(Float32, 7, 4), parameters, state)
    @test_throws DimensionMismatch block(randn(Float32, 8, 7), parameters, state)
end

@testset "LuxCore TransformerLM interface" begin
    rng = Xoshiro(338)
    model = TransformerLM(32, 6, 8, 2, 2, 16, 10_000.0f0)
    parameters, state = LuxCore.setup(rng, model)
    token_ids = reshape(Int32.(0:7), 4, 2)
    logits, returned_state = model(token_ids, parameters, state)

    @test model isa LuxCore.AbstractLuxLayer
    @test size(logits) == (32, 4, 2)
    @test eltype(logits) === Float32
    @test returned_state === state
    @test keys(parameters) == (:token_embedding, :blocks, :final_norm, :lm_head)
    @test length(parameters.blocks) == 2
    @test size(parameters.token_embedding) == (8, 32)
    @test size(parameters.lm_head) == (32, 8)
    @test keys(state) == (:rope, :positions)
    @test !haskey(state, :blocks)
    @test LuxCore.parameterlength(model) == 1_832
    @test LuxCore.statelength(model) == 30

    truncated_logits, _ = model(token_ids[1:2, :], parameters, state)
    @test size(truncated_logits) == (32, 2, 2)
    explicit_positions = repeat(reshape(Int32.(0:3), 4, 1), 1, 2)
    explicit_logits, _ = model((token_ids, explicit_positions), parameters, state)
    @test explicit_logits ≈ logits

    parameter_gradient = only(
        Zygote.gradient(
            ps -> sum(abs2, first(model(token_ids, ps, state))),
            parameters,
        ),
    )
    @test size(parameter_gradient.token_embedding) == (8, 32)
    @test length(parameter_gradient.blocks) == 2
    @test size(parameter_gradient.blocks[1].attention.input_weight) == (24, 8)
    @test size(parameter_gradient.lm_head) == (32, 8)

    empty_model = TransformerLM(32, 6, 8, 0, 2, 16, 10_000.0f0)
    empty_parameters, empty_state = LuxCore.setup(Xoshiro(339), empty_model)
    empty_logits, _ = empty_model(token_ids, empty_parameters, empty_state)
    @test size(empty_logits) == (32, 4, 2)
    @test isempty(empty_parameters.blocks)

    @test_throws ArgumentError TransformerLM(0, 6, 8, 2, 2, 16, 10_000.0)
    @test_throws ArgumentError TransformerLM(32, 6, 8, -1, 2, 16, 10_000.0)
    @test_throws ArgumentError model(Int32[-1], parameters, state)
    @test_throws DimensionMismatch model(reshape(Int32.(0:6), 7, 1), parameters, state)
end

model_fixture_path(stem) = normpath(
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

function fixture_block_tree(arrays; namespace="", gradient=false)
    role_prefix = gradient ? "expected.gradient.parameter." : "parameter."
    prefix = isempty(namespace) ? role_prefix : "$role_prefix$namespace."
    getarray(name) = arrays["$prefix$name"]
    return (
        attention_norm=getarray("attention_norm"),
        attention=(
            input_weight=vcat(
                getarray("query_weight"),
                getarray("key_weight"),
                getarray("value_weight"),
            ),
            output_weight=getarray("attention_output_weight"),
        ),
        feed_forward_norm=getarray("feed_forward_norm"),
        feed_forward=(
            input_weight=vcat(getarray("w3"), getarray("w1")),
            output_weight=getarray("w2"),
        ),
    )
end

function test_block_tree_approx(actual, expected; rtol, atol, nans)
    @test isapprox(actual.attention_norm, expected.attention_norm; rtol, atol, nans)
    @test isapprox(
        actual.attention.input_weight,
        expected.attention.input_weight;
        rtol,
        atol,
        nans,
    )
    @test isapprox(
        actual.attention.output_weight,
        expected.attention.output_weight;
        rtol,
        atol,
        nans,
    )
    @test isapprox(
        actual.feed_forward_norm,
        expected.feed_forward_norm;
        rtol,
        atol,
        nans,
    )
    @test isapprox(
        actual.feed_forward.input_weight,
        expected.feed_forward.input_weight;
        rtol,
        atol,
        nans,
    )
    @test isapprox(
        actual.feed_forward.output_weight,
        expected.feed_forward.output_weight;
        rtol,
        atol,
        nans,
    )
end

@testset "TransformerBlock Python parity" begin
    bundle = load_bundle(model_fixture_path("transformer_block"))
    arrays = bundle.arrays
    scalars = bundle.metadata["scalars"]
    tolerances = bundle.metadata["tolerances"]
    block = TransformerBlock(
        scalars["d_model"],
        scalars["num_heads"],
        scalars["d_ff"],
        scalars["context_length"],
        scalars["theta"],
    )
    _, state = LuxCore.setup(Xoshiro(340), block)
    parameters = fixture_block_tree(arrays)
    expected_parameter_gradient = fixture_block_tree(arrays; gradient=true)
    input = permutedims(arrays["input.x"], (3, 2, 1))
    expected_output = permutedims(arrays["expected.output"], (3, 2, 1))
    expected_input_gradient =
        permutedims(arrays["expected.gradient.input.x"], (3, 2, 1))

    output = first(block(input, parameters, state))
    input_gradient, parameter_gradient = Zygote.gradient(
        (x, ps) -> sum(abs2, first(block(x, ps, state))),
        input,
        parameters,
    )
    rtol = tolerances["rtol"]
    atol = tolerances["atol"]
    nans = tolerances["equal_nan"]
    @test bundle.metadata["operation"] == "run_transformer_block"
    @test isapprox(output, expected_output; rtol, atol, nans)
    @test isapprox(input_gradient, expected_input_gradient; rtol, atol, nans)
    test_block_tree_approx(parameter_gradient, expected_parameter_gradient; rtol, atol, nans)
end

@testset "TransformerLM Python parity" begin
    bundle = load_bundle(model_fixture_path("transformer_lm"))
    arrays = bundle.arrays
    scalars = bundle.metadata["scalars"]
    tolerances = bundle.metadata["tolerances"]
    model = TransformerLM(
        scalars["vocab_size"],
        scalars["context_length"],
        scalars["d_model"],
        scalars["num_layers"],
        scalars["num_heads"],
        scalars["d_ff"],
        scalars["theta"],
    )
    _, state = LuxCore.setup(Xoshiro(341), model)
    parameters = (
        token_embedding=permutedims(arrays["parameter.token_embedding"], (2, 1)),
        blocks=ntuple(
            index -> fixture_block_tree(arrays; namespace="block_$(index - 1)"),
            scalars["num_layers"],
        ),
        final_norm=arrays["parameter.final_norm"],
        lm_head=arrays["parameter.lm_head"],
    )
    expected_parameter_gradient = (
        token_embedding=permutedims(
            arrays["expected.gradient.parameter.token_embedding"],
            (2, 1),
        ),
        blocks=ntuple(
            index -> fixture_block_tree(
                arrays;
                namespace="block_$(index - 1)",
                gradient=true,
            ),
            scalars["num_layers"],
        ),
        final_norm=arrays["expected.gradient.parameter.final_norm"],
        lm_head=arrays["expected.gradient.parameter.lm_head"],
    )
    token_ids = permutedims(arrays["input.token_ids"], (2, 1))
    expected_output = permutedims(arrays["expected.output"], (3, 2, 1))

    output = first(model(token_ids, parameters, state))
    parameter_gradient = only(
        Zygote.gradient(
            ps -> sum(abs2, first(model(token_ids, ps, state))),
            parameters,
        ),
    )
    rtol = tolerances["rtol"]
    atol = tolerances["atol"]
    nans = tolerances["equal_nan"]
    @test bundle.metadata["operation"] == "run_transformer_lm"
    @test isapprox(output, expected_output; rtol, atol, nans)
    @test isapprox(
        parameter_gradient.token_embedding,
        expected_parameter_gradient.token_embedding;
        rtol,
        atol,
        nans,
    )
    @test isapprox(
        parameter_gradient.final_norm,
        expected_parameter_gradient.final_norm;
        rtol,
        atol,
        nans,
    )
    @test isapprox(
        parameter_gradient.lm_head,
        expected_parameter_gradient.lm_head;
        rtol,
        atol,
        nans,
    )
    for index in eachindex(parameter_gradient.blocks)
        test_block_tree_approx(
            parameter_gradient.blocks[index],
            expected_parameter_gradient.blocks[index];
            rtol,
            atol,
            nans,
        )
    end
end
