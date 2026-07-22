using CS336
using JSON
using NPZ
using Test
using TOML
using Zygote

include("fixture_contract.jl")
using .FixtureContract

@testset "CS336 package smoke test" begin
    @test nameof(CS336) === :CS336
    @test Base.pkgversion(CS336) == v"0.1.0"
    @test samefile(pathof(CS336), joinpath(@__DIR__, "..", "src", "CS336.jl"))
end

@testset "workspace topology" begin
    repository_root = normpath(joinpath(@__DIR__, "..", ".."))
    root_project = TOML.parsefile(joinpath(repository_root, "Project.toml"))
    expected_projects = [
        "CS336.jl",
        "CS336.jl/test",
        "CS336.jl/benchmark",
        "CS336.jl/environments/cuda",
        "CS336.jl/environments/lux",
        "CS336.jl/environments/lux_cuda",
    ]

    @test root_project["workspace"]["projects"] == expected_projects

    manifest_paths = String[]
    excluded_directories = Set([".git", ".venv"])
    for (directory, subdirectories, files) in walkdir(repository_root)
        filter!(name -> name ∉ excluded_directories, subdirectories)
        "Manifest.toml" in files && push!(manifest_paths, joinpath(directory, "Manifest.toml"))
    end

    @test manifest_paths == [joinpath(repository_root, "Manifest.toml")]

    root_manifest = TOML.parsefile(only(manifest_paths))
    @test root_manifest["julia_version"] == "1.12.6"
    @test root_manifest["manifest_format"] == "2.0"
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

@testset "neutral fixture schema compatibility" begin
    schema_path = normpath(
        joinpath(@__DIR__, "..", "..", "tests", "fixtures", "julia_parity", "schema-v1.json"),
    )
    schema = JSON.parsefile(schema_path)
    required_properties = Set([
        "contract_version",
        "source",
        "operation",
        "producer",
        "array_file",
        "arrays",
        "scalars",
        "tolerances",
        "gradients",
        "notes",
    ])

    @test schema isa AbstractDict{String, Any}
    @test schema["\$schema"] == "https://json-schema.org/draft/2020-12/schema"
    @test schema["additionalProperties"] === false
    @test Set(schema["required"]) == required_properties
    @test Set(keys(schema["properties"])) == required_properties
    @test schema["properties"]["contract_version"]["const"] == 1

    array_descriptor = schema["\$defs"]["array_descriptor"]
    @test array_descriptor["additionalProperties"] === false
    @test Set(array_descriptor["required"]) ==
          Set(["role", "dtype", "shape", "axes", "physical_representation", "finiteness"])
end

@testset "neutral fixture bundle validation" begin
    function descriptor(role, dtype, shape, axes; finiteness="required", zero_based=nothing)
        result = Dict{String, Any}(
            "role" => role,
            "dtype" => dtype,
            "shape" => shape,
            "axes" => axes,
            "physical_representation" => "dense",
            "finiteness" => finiteness,
        )
        zero_based === nothing || (result["zero_based_values"] = zero_based)
        return result
    end

    function metadata(name, descriptors; gradients_present=false)
        return Dict{String, Any}(
            "contract_version" => 1,
            "source" => Dict{String, Any}(
                "git_commit" => repeat("0", 40),
                "generated_at" => "2026-07-22T15:00:00Z",
                "working_tree_clean" => true,
            ),
            "operation" => "synthetic.transport",
            "producer" => Dict{String, Any}(
                "language" => "Julia test",
                "runtime_version" => string(VERSION),
                "packages" => Dict{String, Any}(
                    "JSON" => string(pkgversion(JSON)),
                    "NPZ" => string(pkgversion(NPZ)),
                ),
            ),
            "array_file" => "$name.npz",
            "arrays" => descriptors,
            "scalars" => Dict{String, Any}("purpose" => "transport validation"),
            "tolerances" =>
                Dict{String, Any}("rtol" => 1.0e-5, "atol" => 1.0e-6, "equal_nan" => false),
            "gradients" => Dict{String, Any}(
                "present" => gradients_present,
                "objective" => gradients_present ? "sum(expected.output)" : nothing,
                "physical_representation" => gradients_present ? "dense" : "none",
            ),
            "notes" => Any["Synthetic temporary bundle; not parity evidence."],
        )
    end

    function write_bundle(directory, name, bundle_metadata, arrays)
        metadata_path = joinpath(directory, "$name.json")
        npzwrite(joinpath(directory, "$name.npz"), arrays)
        JSON.json(metadata_path, bundle_metadata)
        return metadata_path
    end

    mktempdir() do directory
        input = reshape(Float32.(1:4), 2, 2)
        output = input .+ 1
        arrays = Dict("input.x" => input, "expected.output" => output)
        descriptors = Dict{String, Any}(
            "input.x" => descriptor("input", "float32", Any[2, 2], Any["feature", "batch"]),
            "expected.output" =>
                descriptor("expected_output", "float32", Any[2, 2], Any["feature", "batch"]),
        )

        valid_path = write_bundle(directory, "valid", metadata("valid", descriptors), arrays)
        bundle = load_bundle(valid_path)
        @test bundle.arrays["input.x"] == input
        @test bundle.arrays["expected.output"] == output
        @test bundle.metadata["operation"] == "synthetic.transport"
        @test basename(bundle.array_path) == "valid.npz"

        shape_metadata = metadata("shape_mismatch", deepcopy(descriptors))
        shape_metadata["arrays"]["input.x"]["shape"] = Any[4]
        shape_path = write_bundle(directory, "shape_mismatch", shape_metadata, arrays)
        @test_throws ArgumentError load_bundle(shape_path)

        axis_metadata = metadata("axis_mismatch", deepcopy(descriptors))
        axis_metadata["arrays"]["input.x"]["axes"] = Any["flat"]
        axis_path = write_bundle(directory, "axis_mismatch", axis_metadata, arrays)
        @test_throws ArgumentError load_bundle(axis_path)

        dtype_metadata = metadata("dtype_mismatch", deepcopy(descriptors))
        dtype_metadata["arrays"]["input.x"]["dtype"] = "float64"
        dtype_path = write_bundle(directory, "dtype_mismatch", dtype_metadata, arrays)
        @test_throws ArgumentError load_bundle(dtype_path)

        key_metadata = metadata("key_mismatch", deepcopy(descriptors))
        delete!(key_metadata["arrays"], "input.x")
        key_path = write_bundle(directory, "key_mismatch", key_metadata, arrays)
        @test_throws ArgumentError load_bundle(key_path)

        nonfinite_arrays = deepcopy(arrays)
        nonfinite_arrays["expected.output"][1] = NaN32
        nonfinite_path = write_bundle(
            directory, "nonfinite", metadata("nonfinite", deepcopy(descriptors)), nonfinite_arrays
        )
        @test_throws ArgumentError load_bundle(nonfinite_path)

        indexed_arrays = Dict(
            "input.ids" => Int64[0, -1],
            "expected.output" => reshape(Float32.(1:4), 2, 2),
        )
        indexed_descriptors = Dict{String, Any}(
            "input.ids" => descriptor(
                "input", "int64", Any[2], Any["token"]; finiteness="not_applicable", zero_based=true
            ),
            "expected.output" =>
                descriptor("expected_output", "float32", Any[2, 2], Any["feature", "batch"]),
        )
        indexed_path = write_bundle(
            directory,
            "negative_index",
            metadata("negative_index", indexed_descriptors),
            indexed_arrays,
        )
        @test_throws ArgumentError load_bundle(indexed_path)

        gradient_arrays = merge(
            arrays,
            Dict("expected.gradient.parameter.weight" => fill(1.0f0, 2, 2)),
        )
        gradient_descriptors = merge(
            descriptors,
            Dict{String, Any}(
                "expected.gradient.parameter.weight" => descriptor(
                    "expected_parameter_gradient",
                    "float32",
                    Any[2, 2],
                    Any["output_feature", "input_feature"],
                ),
            ),
        )
        inconsistent_gradient_path = write_bundle(
            directory,
            "gradient_inconsistent",
            metadata("gradient_inconsistent", gradient_descriptors),
            gradient_arrays,
        )
        @test_throws ArgumentError load_bundle(inconsistent_gradient_path)

        valid_gradient_path = write_bundle(
            directory,
            "gradient_valid",
            metadata("gradient_valid", gradient_descriptors; gradients_present=true),
            gradient_arrays,
        )
        gradient_bundle = load_bundle(valid_gradient_path)
        @test gradient_bundle.metadata["gradients"]["present"] === true

        extra_metadata = metadata("extra_top_level", deepcopy(descriptors))
        extra_metadata["unexpected"] = true
        extra_path = write_bundle(directory, "extra_top_level", extra_metadata, arrays)
        @test_throws ArgumentError load_bundle(extra_path)

        boolean_version_metadata = metadata("boolean_version", deepcopy(descriptors))
        boolean_version_metadata["contract_version"] = true
        boolean_version_path =
            write_bundle(directory, "boolean_version", boolean_version_metadata, arrays)
        @test_throws ArgumentError load_bundle(boolean_version_path)

        boolean_tolerance_metadata = metadata("boolean_tolerance", deepcopy(descriptors))
        boolean_tolerance_metadata["tolerances"]["rtol"] = true
        boolean_tolerance_path =
            write_bundle(directory, "boolean_tolerance", boolean_tolerance_metadata, arrays)
        @test_throws ArgumentError load_bundle(boolean_tolerance_path)

        boolean_shape_metadata = metadata("boolean_shape", deepcopy(descriptors))
        boolean_shape_metadata["arrays"]["input.x"]["shape"] = Any[true, 2]
        boolean_shape_path = write_bundle(directory, "boolean_shape", boolean_shape_metadata, arrays)
        @test_throws ArgumentError load_bundle(boolean_shape_path)

        representation_metadata = metadata("representation", deepcopy(descriptors))
        representation_metadata["arrays"]["input.x"]["physical_representation"] = "opaque"
        representation_path =
            write_bundle(directory, "representation", representation_metadata, arrays)
        @test_throws ArgumentError load_bundle(representation_path)

        source_metadata = metadata("source", deepcopy(descriptors))
        source_metadata["source"]["git_commit"] = "not-a-commit"
        source_path = write_bundle(directory, "source", source_metadata, arrays)
        @test_throws ArgumentError load_bundle(source_path)
    end
end

@testset "linear parity bundle transport" begin
    metadata_path = normpath(
        joinpath(
            @__DIR__,
            "..",
            "..",
            "tests",
            "fixtures",
            "julia_parity",
            "v1",
            "linear.json",
        ),
    )
    bundle = load_bundle(metadata_path)

    @test bundle.metadata["operation"] == "run_linear"
    @test bundle.metadata["source"]["working_tree_clean"] === true
    @test bundle.metadata["gradients"]["present"] === true
    @test bundle.metadata["scalars"] == Dict{String, Any}("d_in" => 4, "d_out" => 3)
    @test Set(keys(bundle.arrays)) == Set([
        "input.x",
        "parameter.weight",
        "expected.output",
        "expected.gradient.input.x",
        "expected.gradient.parameter.weight",
    ])
end

@testset "linear primitive" begin
    weight = reshape(Float32.(1:12), 3, 4) ./ 8
    input = reshape(Float32.(1:24), 4, 3, 2) ./ 10
    output = linear(input, weight)

    @test size(output) == (3, 3, 2)
    @test output[:, 1, 1] ≈ weight * input[:, 1, 1]
    @test linear(input[:, 1, 1], weight) ≈ weight * input[:, 1, 1]
    @test eltype(output) === Float32
    @test_throws DimensionMismatch linear(reshape(input, 6, 4), weight)
    @test_throws DimensionMismatch linear(fill(1.0f0), weight)
end

@testset "linear Python parity" begin
    metadata_path = normpath(
        joinpath(
            @__DIR__,
            "..",
            "..",
            "tests",
            "fixtures",
            "julia_parity",
            "v1",
            "linear.json",
        ),
    )
    bundle = load_bundle(metadata_path)
    arrays = bundle.arrays
    tolerances = bundle.metadata["tolerances"]

    input = permutedims(arrays["input.x"], (3, 2, 1))
    weight = arrays["parameter.weight"]
    expected_output = permutedims(arrays["expected.output"], (3, 2, 1))
    expected_input_gradient =
        permutedims(arrays["expected.gradient.input.x"], (3, 2, 1))
    expected_weight_gradient = arrays["expected.gradient.parameter.weight"]

    output = linear(input, weight)
    weight_gradient, input_gradient =
        Zygote.gradient((w, x) -> sum(abs2, linear(x, w)), weight, input)

    rtol = tolerances["rtol"]
    atol = tolerances["atol"]
    nans = tolerances["equal_nan"]
    @test isapprox(output, expected_output; rtol, atol, nans)
    @test isapprox(input_gradient, expected_input_gradient; rtol, atol, nans)
    @test isapprox(weight_gradient, expected_weight_gradient; rtol, atol, nans)
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
