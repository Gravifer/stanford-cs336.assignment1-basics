"""
The Julia package for the CS336 parallel implementation.

Adapter-exposed Python behavior is introduced incrementally as idiomatic Julia
functions and explicit model components after its parity evidence is established.
"""
module CS336

export embedding, linear

include("functional.jl")

end
