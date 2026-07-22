using CS336
using CUDA
using Random
using Zygote

CUDA.functional(true)

const RTOL = 3.0f-4
const ATOL = 3.0f-5

cpu(array) = Array(array)
squared_sum(array) = sum(array .* array)
root_storage(array) = parent(array) === array ? array : root_storage(parent(array))

function assert_gpu_close(actual, expected; rtol=RTOL, atol=ATOL)
    @assert root_storage(actual) isa CuArray
    @assert isapprox(cpu(actual), expected; rtol, atol)
end

function gpu_rope(rope::ElementwiseRotaryEmbedding)
    return ElementwiseRotaryEmbedding(
        rope.theta,
        CuArray(rope.cosine),
        CuArray(rope.sine),
    )
end

function gpu_rope(rope::MatrixRotaryEmbedding)
    return MatrixRotaryEmbedding(rope.theta, CuArray(rope.rotation))
end

rng = Xoshiro(336)

rope_input = randn(rng, Float32, 4, 2, 3, 2)
positions = repeat(reshape(Int32[0, 2, 4], 3, 1), 1, 2)
gpu_rope_input = CuArray(rope_input)
gpu_positions = CuArray(positions)
for variant in (:elementwise, :matrix)
    cpu_cache = RotaryEmbedding(10_000.0f0, 4, 5; variant)
    gpu_cache = gpu_rope(cpu_cache)
    assert_gpu_close(
        apply_rope(gpu_cache, gpu_rope_input, gpu_positions; sequence_dim=3),
        apply_rope(cpu_cache, rope_input, positions; sequence_dim=3),
    )
end

elementwise_cpu = RotaryEmbedding(10_000.0f0, 4, 5)
elementwise_gpu = gpu_rope(elementwise_cpu)
cpu_rope_gradient = only(
    Zygote.gradient(
        x -> squared_sum(apply_rope(elementwise_cpu, x, positions; sequence_dim=3)),
        rope_input,
    ),
)
gpu_rope_gradient = only(
    Zygote.gradient(
        x -> squared_sum(
            apply_rope(elementwise_gpu, x, gpu_positions; sequence_dim=3),
        ),
        gpu_rope_input,
    ),
)
assert_gpu_close(gpu_rope_gradient, cpu_rope_gradient)

query = randn(rng, Float32, 4, 3, 2)
key = randn(rng, Float32, 4, 4, 2)
value = randn(rng, Float32, 5, 4, 2)
mask = trues(4, 3, 2)
mask[:, 2, 1] .= false
gpu_query, gpu_key, gpu_value, gpu_mask = CuArray.((query, key, value, mask))
cpu_attention = scaled_dot_product_attention(query, key, value; mask)
gpu_attention = scaled_dot_product_attention(gpu_query, gpu_key, gpu_value; mask=gpu_mask)
assert_gpu_close(gpu_attention, cpu_attention)
cpu_attention_gradients = Zygote.gradient(
    (q, k, v) -> squared_sum(scaled_dot_product_attention(q, k, v; mask)),
    query,
    key,
    value,
)
gpu_attention_gradients = Zygote.gradient(
    (q, k, v) -> squared_sum(
        scaled_dot_product_attention(q, k, v; mask=gpu_mask),
    ),
    gpu_query,
    gpu_key,
    gpu_value,
)
foreach(assert_gpu_close, gpu_attention_gradients, cpu_attention_gradients)
@assert all(iszero, cpu(gpu_attention)[:, 2, 1])

d_model = 8
num_heads = 2
model_input = randn(rng, Float32, d_model, 4, 2)
packed_weight = randn(rng, Float32, 3 * d_model, d_model)
output_weight = randn(rng, Float32, d_model, d_model)
model_positions = repeat(reshape(Int32.(0:3), 4, 1), 1, 2)
model_rope = RotaryEmbedding(10_000.0f0, d_model ÷ num_heads, 4)
gpu_model_input, gpu_packed_weight, gpu_output_weight, gpu_model_positions =
    CuArray.((model_input, packed_weight, output_weight, model_positions))
gpu_model_rope = gpu_rope(model_rope)
cpu_model_output = multihead_self_attention(
    model_input,
    packed_weight,
    output_weight;
    num_heads,
    rope=model_rope,
    positions=model_positions,
)
gpu_model_output = multihead_self_attention(
    gpu_model_input,
    gpu_packed_weight,
    gpu_output_weight;
    num_heads,
    rope=gpu_model_rope,
    positions=gpu_model_positions,
)
assert_gpu_close(gpu_model_output, cpu_model_output)
cpu_model_gradients = Zygote.gradient(
    (x, input_weight, out_weight) -> squared_sum(
        multihead_self_attention(
            x,
            input_weight,
            out_weight;
            num_heads,
            rope=model_rope,
            positions=model_positions,
        ),
    ),
    model_input,
    packed_weight,
    output_weight,
)
gpu_model_gradients = Zygote.gradient(
    (x, input_weight, out_weight) -> squared_sum(
        multihead_self_attention(
            x,
            input_weight,
            out_weight;
            num_heads,
            rope=gpu_model_rope,
            positions=gpu_model_positions,
        ),
    ),
    gpu_model_input,
    gpu_packed_weight,
    gpu_output_weight,
)
foreach(assert_gpu_close, gpu_model_gradients, cpu_model_gradients)

CUDA.synchronize()
println(
    "CS336 attention CUDA smoke passed: CUDA=",
    pkgversion(CUDA),
    " device=",
    CUDA.name(CUDA.device()),
)
