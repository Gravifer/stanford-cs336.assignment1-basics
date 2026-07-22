using CS336
using CUDA
using Random
using Zygote

CUDA.functional(true)

const RTOL = 1.0f-4
const ATOL = 1.0f-5

cpu(array) = Array(array)
squared_sum(array) = sum(array .* array)

function assert_gpu_close(actual, expected; rtol=RTOL, atol=ATOL)
    @assert actual isa CuArray
    @assert isapprox(cpu(actual), expected; rtol, atol)
end

rng = Xoshiro(0)

linear_input = randn(rng, Float32, 4, 3, 2)
linear_weight = randn(rng, Float32, 5, 4)
gpu_linear_input = CuArray(linear_input)
gpu_linear_weight = CuArray(linear_weight)
assert_gpu_close(
    linear(gpu_linear_input, gpu_linear_weight),
    linear(linear_input, linear_weight),
)
cpu_linear_gradients = Zygote.gradient(
    (x, w) -> squared_sum(linear(x, w)),
    linear_input,
    linear_weight,
)
gpu_linear_gradients = Zygote.gradient(
    (x, w) -> squared_sum(linear(x, w)),
    gpu_linear_input,
    gpu_linear_weight,
)
foreach(assert_gpu_close, gpu_linear_gradients, cpu_linear_gradients)

embedding_weight = randn(rng, Float32, 3, 8)
token_ids = Int32[2, 2, 5, 0]
gpu_embedding_weight = CuArray(embedding_weight)
gpu_token_ids = CuArray(token_ids)
assert_gpu_close(
    embedding(gpu_token_ids, gpu_embedding_weight),
    embedding(token_ids, embedding_weight),
)
cpu_embedding_gradient = only(
    Zygote.gradient(
        weight -> squared_sum(embedding(token_ids, weight)),
        embedding_weight,
    ),
)
gpu_embedding_gradient = only(
    Zygote.gradient(
        weight -> squared_sum(embedding(gpu_token_ids, weight)),
        gpu_embedding_weight,
    ),
)
assert_gpu_close(gpu_embedding_gradient, cpu_embedding_gradient)

activation_input = randn(rng, Float32, 6, 5)
gpu_activation_input = CuArray(activation_input)
for operation in (silu, x -> softmax(x; dims=1))
    assert_gpu_close(operation(gpu_activation_input), operation(activation_input))
    cpu_gradient = only(Zygote.gradient(x -> squared_sum(operation(x)), activation_input))
    gpu_gradient = only(
        Zygote.gradient(x -> squared_sum(operation(x)), gpu_activation_input),
    )
    assert_gpu_close(gpu_gradient, cpu_gradient)
end

rms_weight = randn(rng, Float32, 6)
gpu_rms_weight = CuArray(rms_weight)
assert_gpu_close(
    rmsnorm(gpu_activation_input, gpu_rms_weight),
    rmsnorm(activation_input, rms_weight),
)
cpu_rms_gradients = Zygote.gradient(
    (x, w) -> squared_sum(rmsnorm(x, w)),
    activation_input,
    rms_weight,
)
gpu_rms_gradients = Zygote.gradient(
    (x, w) -> squared_sum(rmsnorm(x, w)),
    gpu_activation_input,
    gpu_rms_weight,
)
foreach(assert_gpu_close, gpu_rms_gradients, cpu_rms_gradients)

half_input = Float16.(activation_input)
half_weight = ones(Float16, size(half_input, 1))
gpu_half_output = rmsnorm(CuArray(half_input), CuArray(half_weight))
@assert eltype(gpu_half_output) === Float16
assert_gpu_close(gpu_half_output, rmsnorm(half_input, half_weight); rtol=2.0f-3, atol=2.0f-3)

swiglu_input = randn(rng, Float32, 4, 5)
w1 = randn(rng, Float32, 7, 4)
w2 = randn(rng, Float32, 4, 7)
w3 = randn(rng, Float32, 7, 4)
packed = vcat(w3, w1)
gpu_swiglu_input = CuArray(swiglu_input)
gpu_w1, gpu_w2, gpu_w3, gpu_packed = CuArray.((w1, w2, w3, packed))
assert_gpu_close(
    swiglu(gpu_swiglu_input, gpu_w1, gpu_w2, gpu_w3),
    swiglu(swiglu_input, w1, w2, w3),
)
assert_gpu_close(
    swiglu(gpu_swiglu_input, gpu_packed, gpu_w2),
    swiglu(swiglu_input, packed, w2),
)

cpu_explicit_gradients = Zygote.gradient(
    (x, gate, output, value) -> squared_sum(swiglu(x, gate, output, value)),
    swiglu_input,
    w1,
    w2,
    w3,
)
gpu_explicit_gradients = Zygote.gradient(
    (x, gate, output, value) -> squared_sum(swiglu(x, gate, output, value)),
    gpu_swiglu_input,
    gpu_w1,
    gpu_w2,
    gpu_w3,
)
foreach(assert_gpu_close, gpu_explicit_gradients, cpu_explicit_gradients)

cpu_packed_gradients = Zygote.gradient(
    (x, input, output) -> squared_sum(swiglu(x, input, output)),
    swiglu_input,
    packed,
    w2,
)
gpu_packed_gradients = Zygote.gradient(
    (x, input, output) -> squared_sum(swiglu(x, input, output)),
    gpu_swiglu_input,
    gpu_packed,
    gpu_w2,
)
foreach(assert_gpu_close, gpu_packed_gradients, cpu_packed_gradients)

CUDA.synchronize()
println(
    "CS336 primitive CUDA smoke passed: CUDA=",
    pkgversion(CUDA),
    " device=",
    CUDA.name(CUDA.device()),
)
