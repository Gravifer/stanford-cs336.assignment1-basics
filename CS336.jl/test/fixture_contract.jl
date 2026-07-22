module FixtureContract

using JSON
using NPZ

export load_bundle

const TOP_LEVEL_KEYS = Set([
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

const ARRAY_REQUIRED_KEYS =
    Set(["role", "dtype", "shape", "axes", "physical_representation", "finiteness"])
const ARRAY_OPTIONAL_KEYS = Set(["zero_based_values", "description"])
const ARRAY_ROLES = Set([
    "input",
    "parameter",
    "state",
    "expected_output",
    "expected_input_gradient",
    "expected_parameter_gradient",
    "expected_state",
    "expected_updated_parameter",
    "expected_optimizer_state",
])
const GRADIENT_ROLES = Set([
    "expected_input_gradient",
    "expected_parameter_gradient",
])
const ARRAY_REPRESENTATIONS =
    Set(["dense", "coo", "csr", "csc", "indexed_rows", "indexed_columns"])
const GRADIENT_REPRESENTATIONS = Set(["none", "dense", "indexed", "sparse_matrix", "mixed"])
const OPERATION_PATTERN = r"^[A-Za-z0-9_.-]+$"
const ARRAY_NAME_PATTERN = r"^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$"
const COMMIT_PATTERN = r"^[0-9a-f]{40}$"
const TIMESTAMP_PATTERN =
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
const DTYPE_TYPES = Dict(
    "bool" => Bool,
    "uint8" => UInt8,
    "int32" => Int32,
    "int64" => Int64,
    "float16" => Float16,
    "float32" => Float32,
    "float64" => Float64,
)

fail(message) = throw(ArgumentError(message))
is_json_integer(value) = value isa Integer && !(value isa Bool)
is_json_number(value) = value isa Real && !(value isa Bool)
is_nonempty_string(value) = value isa String && !isempty(value)

function require_object(value, location)
    value isa AbstractDict{String, Any} || fail("$location must be a JSON object")
    return value
end

function require_exact_keys(value, expected, location)
    object = require_object(value, location)
    actual = Set(keys(object))
    actual == expected || fail(
        "$location keys differ: missing=$(sort!(collect(setdiff(expected, actual)))) " *
        "extra=$(sort!(collect(setdiff(actual, expected))))",
    )
    return object
end

function require_array_descriptor(descriptor, name, array)
    location = "arrays.$name"
    object = require_object(descriptor, location)
    actual_keys = Set(keys(object))
    missing = setdiff(ARRAY_REQUIRED_KEYS, actual_keys)
    extra = setdiff(actual_keys, union(ARRAY_REQUIRED_KEYS, ARRAY_OPTIONAL_KEYS))
    isempty(missing) || fail("$location is missing keys $(sort!(collect(missing)))")
    isempty(extra) || fail("$location has unknown keys $(sort!(collect(extra)))")

    role = object["role"]
    role isa String && role in ARRAY_ROLES || fail("$location.role is unsupported")

    dtype_name = object["dtype"]
    dtype_name isa String && haskey(DTYPE_TYPES, dtype_name) || fail("$location.dtype is unsupported")
    eltype(array) === DTYPE_TYPES[dtype_name] || fail(
        "$location dtype mismatch: declared=$dtype_name actual=$(eltype(array))",
    )

    declared_shape = object["shape"]
    declared_shape isa AbstractVector || fail("$location.shape must be an array")
    all(dimension -> is_json_integer(dimension) && dimension >= 0, declared_shape) ||
        fail("$location.shape must contain non-negative integers")
    Tuple(declared_shape) == size(array) || fail(
        "$location shape mismatch: declared=$(Tuple(declared_shape)) actual=$(size(array))",
    )

    axes = object["axes"]
    axes isa AbstractVector || fail("$location.axes must be an array")
    all(axis -> axis isa String && !isempty(axis), axes) ||
        fail("$location.axes must contain non-empty strings")
    length(axes) == length(declared_shape) || fail(
        "$location axis count $(length(axes)) does not match rank $(length(declared_shape))",
    )
    allunique(axes) || fail("$location.axes must be unique")

    representation = object["physical_representation"]
    representation isa String && representation in ARRAY_REPRESENTATIONS ||
        fail("$location.physical_representation is unsupported")

    finiteness = object["finiteness"]
    finiteness in ("required", "allow_nonfinite", "not_applicable") ||
        fail("$location.finiteness is unsupported")
    if finiteness == "required" && eltype(array) <: AbstractFloat
        all(isfinite, array) || fail("$location requires finite values")
    end

    if haskey(object, "zero_based_values")
        object["zero_based_values"] isa Bool ||
            fail("$location.zero_based_values must be boolean")
    end
    if get(object, "zero_based_values", false)
        eltype(array) <: Integer || fail("$location.zero_based_values requires an integer dtype")
        all(value -> value >= 0, array) || fail("$location contains a negative external index")
    end
    if haskey(object, "description")
        object["description"] isa String || fail("$location.description must be a string")
    end

    return role
end

function load_bundle(metadata_path::AbstractString)
    isfile(metadata_path) || fail("metadata file does not exist: $metadata_path")
    splitext(metadata_path)[2] == ".json" || fail("metadata path must end in .json")

    metadata = require_exact_keys(JSON.parsefile(metadata_path), TOP_LEVEL_KEYS, "metadata")
    is_json_integer(metadata["contract_version"]) && metadata["contract_version"] == 1 ||
        fail("unsupported contract version")
    operation = metadata["operation"]
    is_nonempty_string(operation) && occursin(OPERATION_PATTERN, operation) ||
        fail("operation must match $OPERATION_PATTERN")
    source = require_exact_keys(
        metadata["source"],
        Set(["git_commit", "generated_at", "working_tree_clean"]),
        "source",
    )
    source["git_commit"] isa String && occursin(COMMIT_PATTERN, source["git_commit"]) ||
        fail("source.git_commit must be 40 lowercase hexadecimal characters")
    source["generated_at"] isa String &&
        occursin(TIMESTAMP_PATTERN, source["generated_at"]) ||
        fail("source.generated_at must be an ISO 8601 date-time")
    source["working_tree_clean"] isa Bool ||
        fail("source.working_tree_clean must be boolean")

    producer = require_exact_keys(
        metadata["producer"],
        Set(["language", "runtime_version", "packages"]),
        "producer",
    )
    is_nonempty_string(producer["language"]) || fail("producer.language must be non-empty")
    is_nonempty_string(producer["runtime_version"]) ||
        fail("producer.runtime_version must be non-empty")
    packages = require_object(producer["packages"], "producer.packages")
    all(is_nonempty_string, values(packages)) ||
        fail("producer.packages values must be non-empty strings")

    require_object(metadata["scalars"], "scalars")
    notes = metadata["notes"]
    notes isa AbstractVector && all(note -> note isa String, notes) ||
        fail("notes must be an array of strings")

    tolerances = require_exact_keys(
        metadata["tolerances"], Set(["rtol", "atol", "equal_nan"]), "tolerances"
    )
    is_json_number(tolerances["rtol"]) && tolerances["rtol"] >= 0 ||
        fail("tolerances.rtol must be non-negative")
    is_json_number(tolerances["atol"]) && tolerances["atol"] >= 0 ||
        fail("tolerances.atol must be non-negative")
    tolerances["equal_nan"] isa Bool || fail("tolerances.equal_nan must be boolean")

    array_file = metadata["array_file"]
    array_file isa String || fail("array_file must be a string")
    basename(array_file) == array_file || fail("array_file must be a sibling filename")
    expected_array_file = splitext(basename(metadata_path))[1] * ".npz"
    array_file == expected_array_file || fail(
        "array_file must share the metadata stem: expected=$expected_array_file actual=$array_file",
    )
    array_path = joinpath(dirname(metadata_path), array_file)
    isfile(array_path) || fail("array file does not exist: $array_path")

    descriptors = require_object(metadata["arrays"], "arrays")
    isempty(descriptors) && fail("arrays must not be empty")
    arrays = npzread(array_path)
    Set(keys(descriptors)) == Set(keys(arrays)) || fail(
        "NPZ keys differ from metadata descriptors",
    )

    roles = Dict{String, String}()
    for (name, descriptor) in descriptors
        occursin(ARRAY_NAME_PATTERN, name) || fail("array name is unsupported: $name")
        roles[name] = require_array_descriptor(descriptor, name, arrays[name])
    end
    any(==("expected_output"), values(roles)) || fail("bundle has no expected output")

    gradients = require_exact_keys(
        metadata["gradients"],
        Set(["present", "objective", "physical_representation"]),
        "gradients",
    )
    gradients["present"] isa Bool || fail("gradients.present must be boolean")
    representation = gradients["physical_representation"]
    representation isa String && representation in GRADIENT_REPRESENTATIONS ||
        fail("gradients.physical_representation is unsupported")
    objective = gradients["objective"]
    objective === nothing || objective isa String ||
        fail("gradients.objective must be a string or null")
    gradient_arrays_present = any(role -> role in GRADIENT_ROLES, values(roles))
    if gradients["present"]
        is_nonempty_string(objective) ||
            fail("gradient bundles require a non-empty objective")
        representation != "none" ||
            fail("gradient bundles require a physical representation")
        gradient_arrays_present || fail("gradient metadata is present but gradient arrays are absent")
    else
        objective === nothing || fail("forward-only bundles require a null objective")
        representation == "none" ||
            fail("forward-only bundles require physical_representation=none")
        gradient_arrays_present && fail("gradient arrays exist while gradients.present is false")
    end

    return (; metadata, arrays, metadata_path=abspath(metadata_path), array_path=abspath(array_path))
end

end
