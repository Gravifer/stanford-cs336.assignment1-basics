# Julia development guide

The Julia package lives in `CS336.jl/`, while the reproducible Julia workspace lives at repository root beside `pyproject.toml`.

## Toolchain and environment

Julia is managed through Juliaup. Check the selected stable channel with:

```powershell
juliaup status
juliaup update release
```

All package work uses the repository project. Do not add course packages to Julia's user-global environment.

```powershell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.status(; workspace=true)'
julia --project=. -e 'using CS336; println(pathof(CS336))'
```

`Project.toml` declares the workspace and directly depends on the local `CS336.jl` package through `[sources]`. Consequently conventional tooling activated at repository root can load `CS336`, while the package's own `CS336.jl/Project.toml` remains authoritative for runtime dependencies. `Manifest.toml` is the generated shared lockfile; child manifests are not expected.

## Tests

Run the package suite through Pkg from repository root:

```powershell
julia --project=. -e 'using Pkg; Pkg.test("CS336")'
```

For a direct fresh-process smoke run of the test project:

```powershell
julia --startup-file=no --history-file=no --project=CS336.jl/test CS336.jl/test/runtests.jl
```

CPU smoke tests must remain runnable without installing or loading an accelerator package. Optional compiler and accelerator test environments will be added only when their corresponding phases begin.

The optional CUDA readiness environment is activated separately:

```powershell
julia --project=CS336.jl/environments/cuda -e 'using CUDA; CUDA.versioninfo(); @assert CUDA.functional(true)'
```

CUDA v6.2 is constrained there and v6.2.1 is locked. Importing it may download substantial toolkit artifacts into Julia's shared package/artifact cache, but it does not add CUDA to the runtime package or CPU tests.

The baseline Lux integration remains isolated from the runtime package:

```powershell
julia --project=CS336.jl/environments/lux -e 'using Lux, NNlib, Optimisers, Zygote, CS336; println((Lux=pkgversion(Lux), NNlib=pkgversion(NNlib), Optimisers=pkgversion(Optimisers), Zygote=pkgversion(Zygote)))'
```

The lock currently resolves Lux 1.31.4, NNlib 0.9.38, Optimisers 0.4.7, and
Zygote 0.7.11. Phase 3 promotes NNlib 0.9.38 alone to a direct runtime
dependency for its ChainRules-aware, CPU/CUDA-extensible batched matrix
operations. Lux, Optimisers, and Zygote remain integration/test concerns.

Run its committed integration smoke with:

```powershell
julia --startup-file=no --history-file=no --project=CS336.jl/environments/lux CS336.jl/environments/lux/smoke.jl
```

The combined native NVIDIA integration is kept separate so it cannot silently become a CPU requirement:

```powershell
julia --project=CS336.jl/environments/lux_cuda -e 'using CUDA, Lux, LuxCUDA; CUDA.functional(true); println((device=typeof(gpu_device()), LuxCUDA=pkgversion(LuxCUDA), CUDA=pkgversion(CUDA)))'
```

It locks LuxCUDA 0.3.6 and CUDA.jl 6.2.1, including cuDNN.jl 6.2.1. A forward/backward/update smoke has passed on the recorded RTX 3070 Ti host. This is readiness evidence, not a benchmark.

Rerun that committed integration check with:

```powershell
julia --startup-file=no --history-file=no --project=CS336.jl/environments/lux_cuda CS336.jl/environments/lux_cuda/smoke.jl
```

The same optional environment contains a direct-primitives smoke:

```powershell
julia --project=CS336.jl/environments/lux_cuda CS336.jl/environments/lux_cuda/primitives_smoke.jl
```

It compares CPU and CUDA forward results plus Zygote gradients for linear,
repeated-ID embedding, SiLU, softmax, RMSNorm, and both SwiGLU storage forms;
it also checks a Float16 RMSNorm forward pass. This passed with CUDA.jl 6.2.1
on the recorded RTX 3070 Ti. It is correctness evidence, not a performance
benchmark, and does not make CUDA a runtime or CPU-test dependency.

Phase 3 adds a separate attention smoke in the same optional environment:

```powershell
julia --project=CS336.jl/environments/lux_cuda CS336.jl/environments/lux_cuda/attention_smoke.jl
```

It compares CPU/CUDA forward values and Zygote gradients for both RoPE cache
representations, masked SDPA (including a fully masked row), and causal packed
self-attention with RoPE. Lazy GPU-backed gradient wrappers are preserved and
checked by their root storage rather than being materialized solely for a type
assertion.

## Benchmarks

Benchmark dependencies live in their own workspace member and do not belong in the runtime or test projects:

```powershell
julia --project=CS336.jl/benchmark -e 'using BenchmarkTools, CS336; println(Base.pkgversion(BenchmarkTools))'
julia --project=CS336.jl/benchmark CS336.jl/benchmark/swiglu.jl 512 1344 512 20 5.0
```

BenchmarkTools v1.8 and benchmark-only Zygote v0.7 are constrained in
`CS336.jl/benchmark/Project.toml`. `swiglu.jl` compares exactly two Julia
representations—separate semantic weights and packed input weights—after
forward and explicit-gradient parity checks. Its five positional arguments are
`d_model`, `d_ff`, flattened feature-first token count, samples, and seconds.
The command above is a development workload, not a frozen comparative result.
Follow `JULIA_BENCHMARK_PROTOCOL.md` for setup exclusion, interpolation,
warm-up, synchronization, raw samples, and cold-versus-warm reporting.

## Tokenizer training boundary

The Julia `BPETokenizer`, `encode`, `decode`, `encode_iterable`, and
`train_bpe` APIs are runtime exports. Automated trainer verification uses only
four tiny contrived corpora and compares their exact byte merges with the
Python implementation. Do not run `corpus.en`, TinyStories, or another real
training corpus as part of routine agent verification. Full-corpus training
and throughput measurement are explicitly deferred to the user.

## Cross-language conventions

These conventions apply before numerical parity fixtures are introduced:

- External token IDs retain the Python/course convention: zero-based integer values. Convert a token ID to Julia's one-based array index only at the actual lookup boundary; never mutate stored token IDs merely to suit indexing.
- Use `Int64` for token-ID fixture interchange with PyTorch `torch.long`. Julia-internal sizes and axes may use `Int` when they are not serialized or compared across languages.
- Use `Float32` as the default neural-network parity dtype. Tests for other dtypes must name the dtype explicitly and use dtype-appropriate tolerances.
- The canonical Julia hidden-state layout is `(feature, sequence, batch)`. Token-ID batches are `(sequence, batch)`. This matches the documented Lux attention layout and keeps feature vectors in the leading, contiguous dimension.
- The Python boundary layout remains `(batch, sequence, feature)`. Convert once in fixture/import adapters and keep layout conversion out of mathematical kernels.
- Attention scores use `(query_sequence, key_sequence, head, batch)` when exposed. Mask adapters must document their logical axes and verify broadcasting rather than relying on positional coincidence.
- Linear weights retain the logical `(output_feature, input_feature)` matrix shape used by both Julia matrix multiplication and PyTorch storage. Every other imported parameter receives an explicit mapping test before use.

Text and JSON files under `tests/fixtures` are declared LF in `.gitattributes` because tokenizer inputs are byte-sensitive. A worktree populated before that rule was added may retain CRLF files until a normal clean refresh; check with `git ls-files --eol tests/fixtures`. Do not silently normalize bytes inside only one language's tokenizer path.

These are logical contracts, not permission to assume that a particular physical layout is fast. The benchmark protocol requires profiling copies, strides, and vendor-library selection.

## Randomness and reproducibility

- Pass an explicit RNG into initializers and randomized helpers; do not hide reliance on the task-global RNG.
- Use an explicitly seeded `Random.Xoshiro` for repeatable Julia-local tests where exact values are not the contract.
- Do not expect Julia and PyTorch RNG algorithms to produce matching tensors from the same seed.
- Use committed fixtures for exact cross-language parity. Julia's own documentation warns that seeded streams can change across minor releases; the root manifest and recorded Julia version are necessary but saved data is the strongest exact-reproduction evidence.

## Documentation-first dependency rule

Before adding a package or relying on an API:

1. read its current official, version-matched documentation;
2. confirm Julia compatibility and maintained release status;
3. decide whether it is a runtime, test, benchmark, weak, or optional dependency;
4. add a compatibility bound to the owning project;
5. resolve from repository root and inspect the root-manifest change;
6. add a minimal integration test and record the decision in `work-log.md`.

Lux, Optimisers, and Zygote remain in isolated integration/test environments
and have passed a documented interface smoke test. NNlib is the one promoted
runtime dependency: Phase 3 uses its documented batched GEMM and ChainRules
support rather than maintaining private batch loops. Reactant and Enzyme
remain uninstalled until the compiled-path phase exercises them. CUDA remains
isolated and is activated only by optional accelerator environments.

## Planning records

- `JULIA_PORT_PLAN.md` — syllabus phases and design decisions.
- `JULIA_PHASE0_CHECKLIST.md` — scaffold acceptance gates.
- `JULIA_PARITY_MATRIX.md` — Python/Julia parity surface.
- `JULIA_ML_ARCHITECTURE_NOTES.md` — framework architecture findings.
- `JULIA_BENCHMARK_PROTOCOL.md` — eventual comparison methodology.
- `work-log.md` — timestamped operational record.

Documentation consulted for these conventions includes the official [Lux layer documentation](https://lux.csail.mit.edu/stable/api/Lux/layers), [NNlib batched-operation documentation](https://fluxml.ai/Flux.jl/stable/reference/models/nnlib/), [Julia random-number documentation](https://docs.julialang.org/en/v1/stdlib/Random/), and [Pkg environment documentation](https://pkgdocs.julialang.org/v1/environments/).
