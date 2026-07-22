using CS336
using NPZ
using Test

@testset "CS336 package smoke test" begin
    @test nameof(CS336) === :CS336
    @test Base.pkgversion(CS336) == v"0.1.0"
    @test samefile(pathof(CS336), joinpath(@__DIR__, "..", "src", "CS336.jl"))
end

@testset "repository fixture smoke test" begin
    fixture = normpath(joinpath(@__DIR__, "..", "..", "tests", "fixtures", "address.txt"))
    @test isfile(fixture)
    @test !isempty(read(fixture, String))
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
