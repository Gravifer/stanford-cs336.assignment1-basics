using CS336
using Lux
using NNlib
using Optimisers
using Random
using Zygote

rng = Xoshiro(0)
model = Lux.Dense(3 => 2)
parameters, state = Lux.setup(rng, model)
input = rand(rng, Float32, 3, 4)

output, next_state = Lux.apply(model, input, parameters, state)
loss(candidate) = sum(abs2, first(Lux.apply(model, input, candidate, state)))
gradient = only(Zygote.gradient(loss, parameters))

optimizer_state = Optimisers.setup(Optimisers.Adam(1.0f-3), parameters)
next_optimizer_state, next_parameters = Optimisers.update(optimizer_state, parameters, gradient)
probabilities = NNlib.softmax(output; dims=1)

@assert nameof(CS336) === :CS336
@assert size(output) == (2, 4)
@assert next_state == state
@assert isfinite(loss(parameters))
@assert sum(abs, next_parameters.weight .- parameters.weight) > 0
@assert all(isapprox.(vec(sum(probabilities; dims=1)), 1.0f0; atol=8eps(Float32)))
@assert typeof(next_optimizer_state) === typeof(optimizer_state)

println(
    "Lux CPU smoke passed: ",
    "Lux=", pkgversion(Lux),
    " NNlib=", pkgversion(NNlib),
    " Optimisers=", pkgversion(Optimisers),
    " Zygote=", pkgversion(Zygote),
)
