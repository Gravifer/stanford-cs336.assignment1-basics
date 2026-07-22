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
