using CS336
using CUDA
using Lux
using LuxCUDA
using NNlib
using Optimisers
using Random
using Zygote

CUDA.functional(true)

rng = Xoshiro(0)
model = Lux.Dense(3 => 2)
parameters, state = Lux.setup(rng, model)
input = rand(rng, Float32, 3, 4)
cpu_output, _ = Lux.apply(model, input, parameters, state)

gpu = gpu_device()
cpu = cpu_device()
gpu_parameters = parameters |> gpu
gpu_state = state |> gpu
gpu_input = input |> gpu

gpu_output, next_gpu_state = Lux.apply(model, gpu_input, gpu_parameters, gpu_state)
loss(candidate) = sum(abs2, first(Lux.apply(model, gpu_input, candidate, gpu_state)))
gradient = only(Zygote.gradient(loss, gpu_parameters))

optimizer_state = Optimisers.setup(Optimisers.Adam(1.0f-3), gpu_parameters)
next_optimizer_state, next_gpu_parameters =
    Optimisers.update(optimizer_state, gpu_parameters, gradient)
gpu_probabilities = NNlib.softmax(gpu_output; dims=1)

CUDA.synchronize()
output = gpu_output |> cpu
probabilities = gpu_probabilities |> cpu
parameter_delta = sum(abs, (next_gpu_parameters.weight .- gpu_parameters.weight) |> cpu)

@assert nameof(CS336) === :CS336
@assert occursin("CUDA", string(typeof(gpu)))
@assert gpu_output isa CuArray{Float32, 2}
@assert output isa Matrix{Float32}
@assert size(output) == (2, 4)
@assert next_gpu_state == gpu_state
@assert isapprox(output, cpu_output; rtol=8f-5, atol=8f-6)
@assert parameter_delta > 0
@assert all(isapprox.(vec(sum(probabilities; dims=1)), 1.0f0; atol=8eps(Float32)))
@assert typeof(next_optimizer_state) === typeof(optimizer_state)

println(
    "Lux CUDA smoke passed: ",
    "LuxCUDA=", pkgversion(LuxCUDA),
    " CUDA=", pkgversion(CUDA),
    " device=", CUDA.name(CUDA.device()),
)
