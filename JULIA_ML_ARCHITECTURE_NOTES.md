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

The isolation is observable in this repository. With compilation caches already populated, three fresh processes per case produced these full process-start-plus-load medians on the provisional host:

| Scope | Median elapsed | Fresh-process peak RSS |
| --- | ---: | ---: |
| bare Julia process | 0.389 s | 199,352,320 bytes |
| dependency-free `CS336` | 0.438 s | 201,306,112 bytes |
| `CS336` + Lux/NNlib/Optimisers/Zygote | 2.862 s | 381,984,768 bytes |
| preceding stack + LuxCUDA/CUDA initialization | 7.400 s | 1,057,271,808 bytes |

These are three-sample boundary diagnostics, not cross-language benchmarks. They demonstrate both sides of the design: Julia does not make accelerator/framework loading free, but the baseline package can avoid paying it until the relevant optional environment is selected. PyTorch comparison requires the final cold-start protocol on matched processes.

## What is not streamlined away

### Vendor and hardware behavior

Dense matrix multiplication, convolutions, communication collectives, and many fused kernels ultimately depend on vendor libraries, compiler backends, or carefully tuned hardware-specific code. Multiple dispatch can select and compose those implementations cleanly. It cannot substitute for their existence, coverage, or tuning.

### Sparse embedding gradients

The PyTorch-internals embedding example concerns a dense embedding table whose backward result is row-sparse. Neither Julia's generic AD nor Lux automatically chooses a sparse physical tangent for indexed lookup. The baseline must manually decide its gradient representation contract.

The course's current PyTorch embedding path produces a dense gradient, so dense is the parity behavior. A Julia row-sparse experiment is meaningful only if repeated token IDs coalesce correctly and the sparse representation survives the parameter tree, optimizer state/update, weight decay decision, device movement, checkpointing, and any distributed reduction. Supporting the full path requires deliberate work even if Julia lets it live outside a central tensor implementation.

Consequently, the Julia port does **not** need to manually implement sparse storage merely to match this repository. It can use a dense embedding parameter and dense tangent, just as the current PyTorch course path does. Manual sparse-gradient work begins only if we deliberately add the optional row-sparse experiment; at that point, specializing lookup backward alone is insufficient because the optimizer and state semantics are part of the feature.

This was verified empirically with the locked stack on 2026-07-22. For `Lux.Embedding(8 => 3)` and indices `[2, 2, 5]`, Zygote returned a full `Matrix{Float32}` of size `(3, 8)`: all 24 elements were stored, only six were nonzero, and the repeated index accumulated correctly to value 2 in the selected embedding column. Optimisers Adam allocated two dense `(3, 8)` moment matrices. The native CUDA path produced the same representation as a full `CuArray{Float32,2}` with dense GPU moment arrays.

Lux stores embedding weights as `(embedding_feature, vocabulary)`, so the selected vocabulary items are physical columns. PyTorch conventionally stores `(vocabulary, embedding_feature)`, where the same mathematical support is described as row-sparse. The axis orientation changes the name, not the sparsity issue: neither tested Lux path preserved an indexed sparse tangent automatically.

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

### Compatibility result on this repository

On 2026-07-22, the isolated `CS336.jl/environments/lux` project resolved Lux 1.31.4, NNlib 0.9.38, Optimisers 0.4.7, and Zygote 0.7.11 on Julia 1.12.6. A fresh process successfully exercised the documented explicit path: `Lux.setup`, `Lux.apply`, a Zygote gradient with respect to the parameter tree, an Optimisers Adam update, and NNlib softmax. The package itself remains dependency-free.

This establishes that the chosen interfaces compose today; it is not a speed result. The separate `environments/lux_cuda` project subsequently validated LuxCUDA 0.3.6 with CUDA.jl 6.2.1: documented device selection, Dense forward, Zygote backward, Optimisers Adam update, NNlib softmax, synchronization, and host round-trip all succeeded on the RTX 3070 Ti. The tiny CPU/GPU forward comparison differed by at most one `Float32` ulp at the observed scale. Operation/model performance and the “not more than 10×” comparability question remain gated on matched implementations and the benchmark protocol.

A transformer-relevant integration probe also passed. Lux `MultiHeadAttention(64; nheads=4)` accepted `(feature=64, sequence=12, batch=4)`, returned output `(64, 12, 4)` and scores `(key=12, query=12, head=4, batch=4)`, differentiated all projection parameter subtrees through Zygote, and completed an Optimisers Adam update on CPU and CUDA. With identical initialized parameters and inputs, maximum CPU/GPU differences were `3.5762787e-7` for output and `2.9802322e-8` for attention scores. This demonstrates current eager integration coverage, not course-implementation parity, fused-kernel coverage, or throughput.

A Pkg outdated audit found no held-back direct dependency in either the Lux CPU or LuxCUDA project. In the shared manifest, GPUCompiler 1.23.0 is constrained below the available 2.1.1 by CUDACore/CUDATools from the current CUDA.jl 6.2.1 stack. This is a concrete compatibility edge worth rechecking at phase boundaries, but not evidence that the selected public stack is stale: the direct CUDA/Lux packages themselves were current and their integration tests passed.

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
