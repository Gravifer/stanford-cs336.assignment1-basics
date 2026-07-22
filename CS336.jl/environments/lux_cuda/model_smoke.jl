using CS336
using CUDA
using Lux
using LuxCUDA
using Random
using Zygote

CUDA.functional(true)

const RTOL = 5.0f-4
const ATOL = 5.0f-5

root_storage(array) = parent(array) === array ? array : root_storage(parent(array))

function assert_gpu_close(actual, expected; rtol=RTOL, atol=ATOL)
    @assert root_storage(actual) isa CuArray
    @assert isapprox(Array(actual), expected; rtol, atol)
end

function assert_block_gradient(gpu_gradient, cpu_gradient)
    assert_gpu_close(gpu_gradient.attention_norm, cpu_gradient.attention_norm)
    assert_gpu_close(
        gpu_gradient.attention.input_weight,
        cpu_gradient.attention.input_weight,
    )
    assert_gpu_close(
        gpu_gradient.attention.output_weight,
        cpu_gradient.attention.output_weight,
    )
    assert_gpu_close(gpu_gradient.feed_forward_norm, cpu_gradient.feed_forward_norm)
    assert_gpu_close(
        gpu_gradient.feed_forward.input_weight,
        cpu_gradient.feed_forward.input_weight,
    )
    assert_gpu_close(
        gpu_gradient.feed_forward.output_weight,
        cpu_gradient.feed_forward.output_weight,
    )
end

rng = Xoshiro(342)
model = TransformerLM(19, 6, 8, 2, 2, 16, 10_000.0f0)
parameters, state = Lux.setup(rng, model)
token_ids = reshape(Int32[0, 3, 5, 7, 2, 4, 8, 11], 4, 2)
cpu_output, next_cpu_state = Lux.apply(model, token_ids, parameters, state)

gpu = gpu_device()
gpu_parameters = parameters |> gpu
gpu_state = state |> gpu
gpu_token_ids = token_ids |> gpu
gpu_output, next_gpu_state = Lux.apply(model, gpu_token_ids, gpu_parameters, gpu_state)

cpu_gradient = only(
    Zygote.gradient(
        candidate -> begin
            output, _ = Lux.apply(model, token_ids, candidate, state)
            sum(output .* output)
        end,
        parameters,
    ),
)
gpu_gradient = only(
    Zygote.gradient(
        candidate -> begin
            output, _ = Lux.apply(model, gpu_token_ids, candidate, gpu_state)
            sum(output .* output)
        end,
        gpu_parameters,
    ),
)

CUDA.synchronize()
assert_gpu_close(gpu_output, cpu_output)
assert_gpu_close(gpu_gradient.token_embedding, cpu_gradient.token_embedding)
for index in eachindex(gpu_gradient.blocks)
    assert_block_gradient(gpu_gradient.blocks[index], cpu_gradient.blocks[index])
end
assert_gpu_close(gpu_gradient.final_norm, cpu_gradient.final_norm)
assert_gpu_close(gpu_gradient.lm_head, cpu_gradient.lm_head)

@assert next_cpu_state === state
@assert next_gpu_state === gpu_state
@assert keys(gpu_state) == (:rope, :positions)
@assert root_storage(gpu_state.rope.cosine) isa CuArray
@assert root_storage(gpu_state.rope.sine) isa CuArray
@assert root_storage(gpu_state.positions) isa CuArray
@assert !haskey(gpu_state, :blocks)

println(
    "CS336 LuxCore model CUDA smoke passed: Lux=",
    pkgversion(Lux),
    " LuxCUDA=",
    pkgversion(LuxCUDA),
    " CUDA=",
    pkgversion(CUDA),
    " device=",
    CUDA.name(CUDA.device()),
)
