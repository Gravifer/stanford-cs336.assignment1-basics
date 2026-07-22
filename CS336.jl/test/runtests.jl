using CS336
using JSON
using NPZ
using Test

@testset "CS336 package smoke test" begin
    @test nameof(CS336) === :CS336
    @test Base.pkgversion(CS336) == v"0.1.0"
    @test samefile(pathof(CS336), joinpath(@__DIR__, "..", "src", "CS336.jl"))
end

@testset "model metadata compatibility" begin
    config_path = normpath(joinpath(@__DIR__, "..", "..", "tests", "fixtures", "ts_tests", "model_config.json"))
    config = JSON.parsefile(config_path)
    expected = Dict{String, Any}(
        "vocab_size" => 10_000,
        "context_length" => 16,
        "d_model" => 64,
        "num_layers" => 3,
        "num_heads" => 4,
        "d_ff" => 128,
        "remove_rmsnorm" => false,
        "use_post_norm" => false,
        "remove_rope" => false,
        "rope_theta" => 10_000.0,
        "ffn_type" => nothing,
    )

    @test config isa AbstractDict{String, Any}
    @test Set(keys(config)) == Set(keys(expected))
    @test config == expected
end

@testset "repository fixture smoke test" begin
    fixture = normpath(joinpath(@__DIR__, "..", "..", "tests", "fixtures", "address.txt"))
    @test isfile(fixture)
    @test !isempty(read(fixture, String))
end

@testset "tokenizer fixture compatibility" begin
    fixture_root = normpath(joinpath(@__DIR__, "..", "..", "tests", "fixtures"))
    vocab_cases = (
        ("gpt2_vocab.json", 50_257, 50_256),
        ("train-bpe-reference-vocab.json", 500, 0),
    )

    for (filename, count, endoftext_id) in vocab_cases
        vocab = JSON.parsefile(joinpath(fixture_root, filename))
        @test vocab isa AbstractDict{String, Any}
        @test length(vocab) == count
        @test Set(values(vocab)) == Set(0:(count - 1))
        @test vocab["<|endoftext|>"] == endoftext_id
    end

    merge_cases = (
        ("gpt2_merges.txt", 50_000, "Ġ t", "Ġg azed"),
        ("train-bpe-reference-merges.txt", 243, "Ġ t", "Ġ ver"),
    )

    for (filename, count, first_merge, last_merge) in merge_cases
        merges = readlines(joinpath(fixture_root, filename))
        @test length(merges) == count
        @test first(merges) == first_merge
        @test last(merges) == last_merge
    end
end

@testset "NumPy snapshot compatibility" begin
    snapshot_root = normpath(joinpath(@__DIR__, "..", "..", "tests", "_snapshots"))
    expected_shapes = Dict(
        "test_4d_scaled_dot_product_attention.npz" => (2, 2, 12, 64),
        "test_adamw.npz" => (2, 3),
        "test_embedding.npz" => (4, 12, 64),
        "test_linear.npz" => (4, 12, 128),
        "test_multihead_self_attention.npz" => (4, 12, 64),
        "test_multihead_self_attention_with_rope.npz" => (4, 12, 64),
        "test_positionwise_feedforward.npz" => (4, 12, 64),
        "test_rmsnorm.npz" => (4, 12, 64),
        "test_rope.npz" => (4, 12, 64),
        "test_scaled_dot_product_attention.npz" => (4, 12, 64),
        "test_swiglu.npz" => (4, 12, 64),
        "test_transformer_block.npz" => (4, 12, 64),
        "test_transformer_lm.npz" => (4, 12, 10_000),
        "test_transformer_lm_truncated_input.npz" => (4, 6, 10_000),
    )

    files = sort(filter(path -> endswith(path, ".npz"), readdir(snapshot_root; join=true)))
    @test basename.(files) == sort!(collect(keys(expected_shapes)))

    for file in files
        snapshot = npzread(file)
        @test Set(keys(snapshot)) == Set(["array"])
        array = only(values(snapshot))
        @test array isa Array{Float32}
        @test size(array) == expected_shapes[basename(file)]
    end
end
