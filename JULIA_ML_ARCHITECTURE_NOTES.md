# Julia ML architecture notes

These notes preserve the architectural hypotheses that the Julia parallel will test. They distinguish what Julia can structurally simplify from what only sustained ecosystem investment can provide.

## Short answer

Julia can streamline meaningful parts of PyTorch's framework machinery, but it cannot make the underlying capability matrix disappear.

Generic functions, multiple dispatch, compiler specialization, the `AbstractArray` ecosystem, package extensions, and independent AD rule packages let ordinary Julia types and functions participate in CPU/GPU execution and differentiation without every combination being enumerated inside one central tensor class and dispatcher. Lux can keep architecture, parameters, and state explicit instead of relying on a mutable module-registration protocol. These are real reductions in framework-specific indirection.

The remaining hard work includes correct kernels, layout selection, fusion, mixed precision, vendor libraries, graph/whole-step compilation, collectives, distributed failure handling, serialization, profiler integration, and compatibility across releases and accelerators. Julia distributes much of that work across packages; PyTorch centralizes more of it. Distribution can make the architecture smaller and more composable, but it also exposes integration boundaries and the smaller funding/maintainer base.

The practical claim for this repository is therefore conditional: comparable steady-state performance is plausible for paths that reach the same vendor libraries or competitive generated kernels. It is not guaranteed for every operation or composition. The benchmark protocol treats a greater-than-10× matched steady-state gap as a failed comparability result until profiling demonstrates a genuinely missing ecosystem capability.

## What can be streamlined

### Operation dispatch and device-generic code

Julia's official methods manual describes dispatch over the types of all arguments and compiler specialization for concrete argument tuples. An educational operation can be an ordinary generic function over arrays; CPU arrays, CUDA arrays, other GPU arrays, element types, and specialized layouts can supply more specific methods without requiring a single framework-owned tensor object to encode every key.

GPUArrays explicitly presents itself as an `AbstractArray`-like interface shared by GPU backends. This provides a route for generic array code and reusable algorithms across CUDA, AMDGPU, oneAPI, and Metal. It does not imply that every generic expression is efficient or supported on every device.

### Model structure

Lux models do not own parameters and state. Setup returns them explicitly, and application accepts the model, input, parameters, and state. For this project that reduces hidden parameter discovery, mutation-based registration, state-dict traversal conventions, and device-transfer magic at the model boundary. It also makes cross-language weight fixtures and whole-step compilation easier to reason about.

This does not mean “no framework.” Lux, Optimisers, device adaptation, serialization, and training helpers still define contracts, and their compatibility must be tested and pinned.

### Automatic differentiation

ChainRules specifies forward and reverse differentiation rules independently of one particular array or model framework. An operation may acquire a custom pullback without being built into a central autograd operator registry. Zygote consumes source-level Julia and explicit arguments; Enzyme differentiates compiled LLVM and uses explicit activity/shadow information. The existence of multiple AD backends is an architectural strength, not automatic interchangeability.

Zygote's documented mutation limitations, Enzyme activity constraints, and backend-specific custom-rule behavior remain real. A function is only “AD-generic” after tests demonstrate the required outputs, gradients, mutation behavior, devices, and compiler path.

### Optional integrations

Julia Pkg extensions allow integration code to load when optional dependencies are present, while weak dependencies avoid making every backend an unconditional install/load-time cost. This can reduce the monolithic feature burden of a baseline package.

The cost is a compatibility graph spread across independently released packages. A package extension removes unconditional coupling; it does not fund or test all combinations.

## What is not streamlined away

### Vendor and hardware behavior

Dense matrix multiplication, convolutions, communication collectives, and many fused kernels ultimately depend on vendor libraries, compiler backends, or carefully tuned hardware-specific code. Multiple dispatch can select and compose those implementations cleanly. It cannot substitute for their existence, coverage, or tuning.

### Sparse embedding gradients

The PyTorch-internals embedding example concerns a dense embedding table whose backward result is row-sparse. Neither Julia's generic AD nor Lux automatically chooses a sparse physical tangent for indexed lookup. The baseline must manually decide its gradient representation contract.

The course's current PyTorch embedding path produces a dense gradient, so dense is the parity behavior. A Julia row-sparse experiment is meaningful only if repeated token IDs coalesce correctly and the sparse representation survives the parameter tree, optimizer state/update, weight decay decision, device movement, checkpointing, and any distributed reduction. Supporting the full path requires deliberate work even if Julia lets it live outside a central tensor implementation.

### Ecosystem completeness

PyTorch's internal complexity partly records years of funded backward compatibility and broad product requirements. Julia's modularity may avoid putting all of that code in one repository, but missing integrations remain missing capabilities. For this reason the project records both architectural cleanliness and operational coverage; one must not be used as a proxy for the other.

This is principally a resourcing question when the technical design already permits an integration but no maintainers have implemented, tested, documented, or sustained it. The benchmark report should name that evidence specifically rather than saying only “Julia is fragmented.”

## Why Lux is the baseline

Lux is selected because its current documented interface fits the experiment:

- architecture is separate from explicit parameter and state trees;
- it supports ordinary Julia functions/layers and explicit differentiation;
- its current manual documents multiple AD choices;
- its current manual documents Reactant+Enzyme compilation rather than leaving the compiler path as a speculative integration;
- its device management is built around the Julia accelerator ecosystem.

Lux is not selected because it is assumed faster than Flux or PyTorch, nor because every course primitive should be replaced by a built-in Lux layer. The reference math remains visible; a practical/library path is added separately when needed for a fair performance comparison.

## Side note: “tabular ML”

Tabular ML means prediction from row-and-column datasets such as database tables or spreadsheets—classification/regression with methods such as linear models, trees, random forests, and boosted trees. It was relevant to a broad Julia ecosystem survey (for example, MLJ), but it is not central to this transformer-language-model package.

## Documentation checked 2026-07-22

- [Julia methods and multiple dispatch](https://docs.julialang.org/en/v1/manual/methods/)
- [Julia performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/)
- [Julia Pkg weak dependencies and extensions](https://pkgdocs.julialang.org/v1/creating-packages/#Weak-dependencies)
- [GPUArrays interface](https://juliagpu.github.io/GPUArrays.jl/dev/interface/)
- [ChainRules introduction](https://juliadiff.org/ChainRulesCore.jl/stable/)
- [Lux introduction](https://lux.csail.mit.edu/stable/introduction/)
- [Lux automatic differentiation](https://lux.csail.mit.edu/stable/manual/autodiff)
- [Lux Reactant compilation](https://lux.csail.mit.edu/stable/manual/compiling_lux_models)
- [Zygote limitations](https://fluxml.ai/Zygote.jl/stable/limitations/)
- [Enzyme documentation](https://enzymead.github.io/Enzyme.jl/stable/)
- [Reactant automatic differentiation](https://enzymead.github.io/Reactant.jl/stable/tutorials/automatic-differentiation)
