# Julia parallel implementation plan

## Resume state

- Branch: `codex/julia-basics` (local only; never push unless the user changes that instruction).
- Base at creation: local `dev` at `8c032c9` (`chore: prepare for training`), matching `origin/dev` on 2026-07-22.
- Branch policy: grow only from `dev`. Do not merge or rebase Python feature branches into this branch.
- Verified toolchain: Juliaup 1.20.8 with the stable `release` channel at Julia 1.12.6. `juliaup update release` reported no update on 2026-07-22.
- Current stage: Phases 0–4 have reached their present adapter-defined gate.
  Phases 0–3 are implemented and verified under the user's tiny-corpus
  training constraint. Feature-first linear, embedding, SiLU, softmax,
  RMSNorm, and explicit/packed SwiGLU pass Python forward/gradient parity.
  Tokenizer encoding matches GPT-2 fixture IDs and BPE training matches four
  tiny Python probes in serial and threaded modes. Full-corpus BPE training is
  explicitly deferred to the user. Phase 3 includes Python output/gradient
  parity for RoPE, attention, a transformer block, and a two-layer language
  model, plus CPU and optional CUDA execution. Phase 4 was re-audited on
  2026-07-23; all seven training/optimization adapters still raise
  `NotImplementedError`, so the phase is complete as a scope audit and remains
  gated rather than implemented. Phases 5–6 remain future compiled-path and
  final comparative-benchmark work.

When resuming, first run:

```powershell
git status --short --branch
git switch codex/julia-basics
git fetch --all --prune  # only if the user wants a network refresh
git rebase dev
```

Before rebasing, require a clean worktree or make a normal checkpoint commit. Do not use destructive reset/checkout commands to clear user work.

## Objective

Build a Julia package containing an idiomatic semantic port of the student implementation already reachable through the working functions in `tests/adapters.py`. Preserve mathematical behavior without mechanically cloning PyTorch's object model. Benchmark correctness, eager performance, compiled performance, memory, and startup/compilation costs against the Python/PyTorch implementation.

This project is about language/framework architecture and an implementation comparison. Keep the port's explicit mathematical implementation visible even when Julia or a vendor library has a faster built-in implementation.

Adapters define semantics and progress only. They do not prescribe Julia structure. Runtime kernels should be pure generic array functions where practical; composed models should use explicit Lux parameter/state trees; AD should differentiate explicit arguments through Zygote first and Enzyme/Reactant in the compiled track. Do not reproduce `nn.Module`, parameter registration, `state_dict`, or PyTorch's dispatcher.

## Scope and non-goals

`tests/adapters.py` is the authoritative progress boundary. Port a surface only when its adapter has a working Python body. An adapter that still raises `NotImplementedError` remains out of scope even if an old plan, snapshot filename, Python test, or implementation file suggests future work.

The Julia package is not a second CS336 course distribution. Do not recreate assignment prompts, TODOs, grading adapters, submission hooks, or a student-facing starter harness. Julia tests are maintainer regression and cross-language parity tests. Shared fixture producers exist only to make comparisons reproducible.

## Companion records

- `JULIA_ML_ARCHITECTURE_NOTES.md` preserves the ecosystem conclusions and the boundary between structural streamlining and funded capability.
- `JULIA_PARITY_MATRIX.md` tracks the evolving Python adapter/test surface and assigns it to baseline or experimental Julia tracks.
- `JULIA_VARIANT_AUDIT.md` prevents adapter parity from erasing meaningful Python storage, contraction, layout, and compilation experiments.
- `JULIA_FIXTURE_CONTRACT.md` defines self-contained neutral inputs, weights, outputs, gradients, and metadata for cross-language parity.
- `JULIA_SNAPSHOT_PROVENANCE.md` maps every legacy NPZ output to its live producer, semantic axes, dependencies, tolerances, and migration gap.
- `JULIA_PHASE0_CHECKLIST.md` is the acceptance checklist for the root workspace and minimal package scaffold.
- `JULIA_DEVELOPMENT.md` contains the executable project/test commands and cross-language boundary conventions.
- `JULIA_BENCHMARK_PROTOCOL.md` defines correctness gates, timing controls, reproducibility metadata, and the greater-than-10× investigation threshold.
- `JULIA_BENCHMARK_HOST.md` records provisional hardware/runtime readiness and the fields that still must be frozen for final trials.
- `work-log.md` is the timestamped operational record and commit ledger.

## Repository and environment layout

```text
pyproject.toml
Project.toml                 # root Julia workspace/environment
Manifest.toml                # the one Julia lockfile, committed
work-log.md
CS336.jl/                    # actual Julia package root
  Project.toml               # package identity and direct dependencies
  src/
    CS336.jl
    functional.jl
    layers.jl
    attention.jl
    transformer.jl
    tokenizer.jl
    data.jl
    optimizers.jl
    training.jl
  test/
    Project.toml             # workspace member, if extra test deps are needed
    runtests.jl
    ...
  benchmark/
    Project.toml             # workspace member with benchmark-only deps
    correctness.jl
    kernels.jl
    model.jl
    training.jl
  environments/
    cuda/
      Project.toml           # optional workspace member, isolated CUDA stack
    lux/
      Project.toml           # optional workspace member, baseline ML integration
    lux_cuda/
      Project.toml           # optional workspace member, combined native NVIDIA path
    reactant/
      Project.toml           # optional workspace member, isolated compiler stack
```

The directory is exactly `CS336.jl/`, and the Julia module/package name is `CS336`. The `.jl` suffix is conventional for a Julia package repository or package directory, while Julia identifiers themselves cannot contain the dot.

The root `Project.toml` is a Julia 1.12 workspace project whose `[workspace]` members include `CS336.jl` and the test/benchmark environments as they are introduced. The package's own `CS336.jl/Project.toml` declares its direct dependencies. The root `Manifest.toml` is the single resolved dependency graph shared by the workspace and must be committed beside `pyproject.toml`. In Julia terminology, `Manifest.toml` **is** the lockfile; do not invent or maintain a second lock format.

Activate and instantiate from the repository root with `julia --project=.`. Do not add course dependencies to Julia's user-global environment. Keep the Python package untouched except for deliberately shared fixtures or benchmark launchers.

This layout follows the official Julia 1.12 [workspace](https://pkgdocs.julialang.org/dev/toml-files/#The-%5Bworkspace%5D-section), [environment](https://pkgdocs.julialang.org/v1/environments/), and [package-creation](https://pkgdocs.julialang.org/v1/creating-packages/) documentation.

## Working protocol

- Work only on `codex/julia-basics`, which grows from `dev`. Rebase onto the local `dev` tip when useful and the worktree is clean. Never merge Python feature branches into it.
- Keep all work local. Do not push.
- Make fine-grained commits whose subjects describe one coherent change. Do not bundle planning, scaffolding, dependency resolution, and implementation into one commit.
- Maintain `work-log.md` as the chronological operational record. Add ISO 8601 timestamps with the `+08:00` offset while working, and record material commands, documentation consulted, decisions, verification results, dependency/toolchain changes, commits, and blockers. Do not record secrets or dump noisy command output.
- Treat official Julia, Pkg, Juliaup, and package documentation as the default authority. Before adopting a Julia API, AD behavior, accelerator path, or dependency, verify it against current upstream documentation or source and record the reference when it affects design.
- Manage the user-global Julia runtime only with Juliaup. At the start of a work period, check `juliaup status`, then run `juliaup update release` if the stable channel is behind. Updating the runtime does not authorize installing course packages globally.
- For the 2026-07-22 work period, continue useful documentation/planning work until 23:30 Asia/Shanghai. An earlier stop is allowed only after all useful in-scope work is exhausted and ten consecutive minutes have been spent idle; record the stop time and reason in `work-log.md`.

## Blessed stack

Avoid an open-ended matrix of interchangeable frameworks. Start with one path:

- Julia's stable `release` channel managed by Juliaup (currently Julia 1.12.6).
- `Lux.jl` for model/layer structure and explicit parameters/state.
- `NNlib.jl` for established neural-network primitives where a library-backed comparison is useful.
- `Zygote.jl` for the initial reverse-mode AD baseline.
- Julia's `Test` plus `BenchmarkTools.jl` for tests and measurements.
- A small fixture reader such as `NPZ.jl` and `JSON3.jl` only if needed to consume the existing Python snapshots.
- `Enzyme.jl` plus `Reactant.jl` as an optional second-stage compiled path, in an isolated environment after the baseline is correct.

Do not adopt Flux and Lux simultaneously. Do not switch AD systems opportunistically inside the baseline. Pin the manifest and add a smoke test for the complete dependency combination.

## Why Lux

Lux treats a model primarily as an architecture description. Initialization produces a parameter tree and a state tree, and application is conceptually:

```julia
y, new_state = Lux.apply(model, x, parameters, state)
```

This is useful here because:

1. Parameters, non-trainable state, and architecture are not hidden behind mutable `nn.Module` registration rules.
2. The same parameter tree can be passed explicitly to differentiation, optimizers, serialization, CPU/GPU transfer, and benchmarks.
3. A directly ported primitive can remain a plain Julia function; a small Lux layer can wrap it without making the framework own the math.
4. The explicit interface is suitable for Reactant/Enzyme whole-step compilation later.
5. It becomes easier to compare identical weights and state across the Python and Julia implementations.

Lux is not intended to replace the direct port. Implement the currently exposed softmax, RMSNorm, RoPE, attention, and SwiGLU behavior explicitly first. Use Lux to compose and initialize the resulting layers. Keep an optimized/library-backed variant separate when it helps answer the performance question. AdamW and other adapters that still raise `NotImplementedError` are not current targets.

## Adapter-aligned phases

### Phase 0: scaffold and cross-language fixtures — complete

- Create the root workspace `Project.toml`, `CS336.jl/Project.toml`, `CS336.jl/src/CS336.jl`, tests, and documented commands. Resolve and commit the one root `Manifest.toml`.
- Use the verified Julia 1.12.6 runtime and record any later runtime upgrade in this plan and the benchmark metadata.
- Read existing `.npz`, JSON, text, vocabulary, and merge fixtures without rewriting them.
- Establish deterministic seeds, dtype conventions, tensor-axis conventions, and tolerances.
- Add one self-contained Python-to-Julia numerical parity test before implementing the model stack. The fixture producer is maintainer comparison infrastructure, not starter code.

### Phase 1: tokenizer — complete under tiny-corpus constraint

- BPE training, encoding, replacement decoding, special-token behavior, and
  lazy chunk encoding are implemented without runtime dependencies.
- Exact GPT-2 encoding and four tiny Python trainer cases establish semantic
  parity, including bytewise tie-breaking and special-token boundaries.
- Per the user's instruction, do not run the repository's real/full BPE
  training workloads. Full-corpus correctness and throughput benchmarking are
  a documented user-deferred action; tiny contrived probes are the acceptance
  evidence for this phase.

### Phase 2: numerical primitives

- Complete as of 2026-07-23: all listed primitives pass CPU fixture/gradient
  parity, and the optional CUDA smoke passes forward/gradient comparison against
  CPU on the recorded NVIDIA host.
- Stable softmax, linear, embedding, SiLU/SwiGLU, and RMSNorm.
- Preserve exactly two SwiGLU execution lanes: explicit separate weights and
  packed input weights. The Python delegated/owned class distinction collapses
  under explicit Julia parameters. The common `swiglu` function selects the
  two storage forms by positional arity; the model default uses packed input
  projection.
- Test output values, edge cases, gradients, dtype promotion, allocations, and CPU/GPU behavior.
- For embedding, test both gradient values and the physical gradient representation; a mathematically row-sparse result is not necessarily stored sparsely by either framework.
- Keep array shapes explicit in docstrings and tests; Julia does not need a direct imitation of `jaxtyping`.

### Phase 3: attention and transformer layers

- Complete as of 2026-07-23 through Python fixture parity, LuxCore structural
  tests, and optional CPU/CUDA forward/gradient comparisons.
- RoPE, scaled dot-product attention, causal and additive/boolean masks, fully masked rows, MHA/GQA/MQA, transformer block, and language model.
- RoPE, SDPA, and self-attention are complete through Python forward/gradient
  parity and an optional CUDA forward/gradient smoke. Transformer block and
  language-model composition now implement the LuxCore parameter/state
  interface with one RoPE cache shared across layers. Both match small
  self-contained Python forward/gradient bundles; a full two-layer LuxCUDA
  forward and parameter-tree gradient comparison also passes.
- Use canonical feature-first `(head_feature, head, sequence, batch...)` attention activations and preserve `(head_feature, sequence, head, batch...)` as the authored experimental layout. Boundary adapters may transpose imported PyTorch weights or fixtures; do not scatter layout conversions through kernels.
- Preserve packed/separate projection storage, input-sharing projection paths,
  elementwise/matrix RoPE caches, and automatic/separate/stacked Q/K rotation
  paths as named benchmark lanes.
- Match existing Python snapshots and gradient checks before optimizing.

### Phase 4: optimization and training — audited and gated by Python adapters

- `run_get_batch`, `run_cross_entropy`, `run_gradient_clipping`, `get_adamw_cls`, `run_get_lr_cosine_schedule`, `run_save_checkpoint`, and `run_load_checkpoint` currently raise `NotImplementedError`.
- Do not implement these surfaces in Julia until their Python adapters become working reference boundaries on `dev`.
- When `dev` advances, rebase this branch and update `JULIA_PARITY_MATRIX.md` before expanding scope; do not import intermediate Python feature branches.
- Re-audited on 2026-07-23 after completing Phases 1–3: all seven adapters
  remain gated, `dev` remains an ancestor of the Julia branch, and no Phase 4
  runtime implementation has been fabricated.

### Phase 5: compiled and accelerator paths

- Establish native Julia CPU and CUDA-array baselines first.
- CUDA toolchain readiness is verified in `CS336.jl/environments/cuda`, Lux
  forward/backward/update composition is verified in
  `CS336.jl/environments/lux_cuda`, and the Phase 2 direct primitives pass a
  CPU/CUDA forward-and-gradient smoke there. The directly ported attention
  and two-layer LuxCore model also pass optional CUDA forward/gradient smokes;
  compiled Reactant/Enzyme measurement remains pending.
- Lux's built-in multi-head attention has passed a separate CPU/CUDA forward, Zygote backward, Adam update, and identical-input numerical comparison; this is ecosystem readiness only and does not substitute for the directly ported attention path.
- Evaluate Enzyme for mutation-friendly differentiation.
- Evaluate Reactant+Enzyme for compiled inference and a compiled training step.
- Treat unsupported tracing/control-flow cases as measured limitations, not reasons to distort the baseline implementation prematurely.

### Phase 6: final comparative benchmark

- Freeze manifests, hardware settings, dataset slices, seeds, shapes, dtypes, and synchronization rules.
- Produce machine-readable raw results plus a human-readable report.
- Profile any result that crosses the investigation thresholds below.

## Performance question and acceptance policy

The central question is not whether Julia wins every benchmark. It is whether the Julia ecosystem can reach the same order of magnitude as PyTorch without heroic private infrastructure.

Benchmark four execution modes when they exist:

1. PyTorch eager.
2. `torch.compile` after successful compilation.
3. Julia/Lux eager or native-array execution.
4. Julia Reactant+Enzyme compiled execution.

Measure cold and warm behavior separately:

- startup/package load;
- first invocation and compilation;
- steady-state latency and throughput;
- peak and steady memory;
- forward only;
- forward plus backward;
- complete optimizer/training step.

For GPU timing, synchronize before and after measured regions. Use enough work to amortize launch overhead. Verify outputs and gradients before timing. Record device, driver, library versions, BLAS backend, thread counts, dtype, shapes, batch size, compiler flags, and framework versions.

Investigation thresholds:

- A vendor-backed dense primitive such as GEMM or convolution more than 2x behind PyTorch is suspicious and must be profiled for layout, conversion, synchronization, or dispatch mistakes.
- An important steady-state kernel or model path 3x to 10x behind requires a written explanation and an optimized alternative if feasible.
- More than 10x steady-state lag is a failure for the comparability goal unless the report demonstrates a missing ecosystem feature rather than an implementation or benchmark error.
- Cold compilation may exceed 10x, but it must be reported independently and amortized over realistic run lengths rather than hidden.

Do not compare a handwritten Julia reference kernel only against PyTorch's fused production kernel and call that a language result. Report both direct-port/reference and best practical paths.

## Design record: embedding-gradient sparsity

### Corrected reading of the PyTorch-internals example

The sparsity example in Edward Yang's [PyTorch internals](https://blog.ezyang.com/2019/05/pytorch-internals/) is specifically about embedding backward, not sparse forward weights. The embedding table can remain dense while the gradient with respect to it is row-sparse: only vocabulary rows selected by the batch receive lookup-gradient contributions. PyTorch uses this as a capability test for making sparse tensors participate directly in autograd rather than representing them as an opaque wrapper.

PyTorch does not enable that representation automatically. [`torch.nn.Embedding`](https://docs.pytorch.org/docs/2.13/generated/torch.nn.modules.sparse.Embedding.html) defaults to `sparse=False`; opting into `sparse=True` produces a sparse weight gradient, but the documented compatible optimizers are limited to SGD, SparseAdam, and CPU Adagrad.

The current Python course implementation calls `torch.nn.functional.embedding` without requesting a sparse gradient. Its embedding gradient is therefore dense. Unless `dev` changes this behavior, dense gradients are the required cross-language parity target.

### Current Julia finding

The investigated Julia stack also produces a dense embedding-weight gradient by default:

- Current [`Lux.Embedding`](https://github.com/LuxDL/Lux.jl/blob/82b8efede2a7f8523fd43da7c492e44d6ee7cd1a/src/layers/embedding.jl) performs ordinary indexed selection from a dense parameter array.
- The general [`ChainRules` indexing pullback](https://github.com/JuliaDiff/ChainRules.jl/blob/main/src/rulesets/Base/indexing.jl) allocates a gradient shaped like the source and scatter-adds into it. Its lightweight `OneElement` representation covers scalar indexing, not a batch of embedding rows.
- A locked-stack experiment on 2026-07-22 confirmed the behavior on both CPU and CUDA: `Lux.Embedding(8 => 3)` with repeated indices `[2, 2, 5]` produced a full dense `(3, 8)` Zygote tangent, correctly accumulated repeats, and led Optimisers Adam to allocate dense moment arrays. Lux's `(feature, vocabulary)` storage makes this physically column-sparse rather than PyTorch-style row-sparse, but it is still stored densely.
- Passing equivalent values as `SparseMatrixCSC` to Optimisers is not a shortcut to sparse-optimizer semantics. Generic Descent accepts it, but Adam retains dense moments and treats absent stored entries as numerical zeros: momentum from a previously touched embedding column continues updating that column on a later step where it is absent. A sparse experiment must specify whether absent entries mean zero-gradient evolution or a skipped update.
- [`NNlib.gather`](https://github.com/FluxML/NNlib.jl/blob/master/src/gather.jl) likewise creates a full-size zero gradient and scatters contributions into it.
- Enzyme shadows follow the primal memory layout. A dense parameter array therefore naturally has a dense derivative shadow unless a more specialized rule or representation is introduced.

Julia's multiple dispatch makes it possible to introduce a specialized tangent and optimizer path without placing every layout inside a PyTorch-style central `TensorImpl`. It does not cause sparse embedding gradients to emerge automatically.

### Project decision

1. Implement and validate the dense-gradient embedding path first because it matches the Python course implementation and dense AdamW training semantics.
2. Do not label the baseline as supporting sparse embedding gradients merely because the mathematical gradient has many zero rows.
3. Preserve sparse embedding gradients as a separate framework-architecture experiment after baseline training works.
4. Revisit this decision if the Python implementation enables sparse gradients, ties the token embedding to another parameter, or changes optimizer semantics.

A genuine Julia sparse-gradient experiment must cover the complete backward-to-update contract:

- represent selected vocabulary rows and their accumulated row gradients;
- coalesce repeated token IDs correctly;
- pass that tangent through the Lux parameter tree and selected AD backend without densification;
- update compatible optimizer state without scanning or allocating the entire table;
- avoid treating generic `SparseMatrixCSC` acceptance as proof of skipped-entry or sparse-state behavior;
- define behavior for momentum, checkpointing, device transfer, and distributed reduction;
- define whether untouched rows receive decoupled weight decay.

The last item prevents silently treating a row-sparse update as equivalent to dense AdamW. PyTorch's own sparse-gradient embedding path does not claim AdamW compatibility.

### Benchmark addition

Add an embedding-specific benchmark whose vocabulary size grows while the number of token occurrences stays fixed. Report separately:

1. Python/PyTorch dense backward using the course implementation.
2. Julia/Lux dense backward using the parity implementation.
3. PyTorch sparse backward with a documented compatible optimizer.
4. Julia row-sparse backward plus update, only if the full tangent/optimizer path is implemented correctly.

Measure forward time, backward time, update time, gradient storage, optimizer-state storage, repeated-ID behavior, and CPU/GPU support. Do not compare sparse backward alone while excluding a densifying or semantically different optimizer step.

## Ecosystem/resourcing risk

The fragmentation issue is substantially a money and maintainer-bandwidth issue, not merely a technical-design flaw. PyTorch can fund centralized compatibility across accelerators, compilation, distributed training, mixed precision, model formats, and releases. Julia projects distribute that burden among smaller teams.

Mitigations for this repository:

- Keep the baseline dependency graph small and pinned.
- Choose one supported integration path instead of testing every Julia AD/backend combination.
- Isolate Reactant in its own environment until it is needed.
- Add integration smoke tests and record known-good versions.
- Prefer stable public interfaces and ordinary arrays/functions at package boundaries.
- Preserve simple reference implementations so a framework integration failure does not block the course.
- Re-evaluate dependency health at phase boundaries, not continuously during implementation.

The documentation-first rule is also a mitigation: when ecosystem familiarity is weak, do not fill gaps from Python analogies. Check the relevant Julia package manual and, for behavior not promised by the manual, the tagged source and a minimal experiment. Log the evidence used for choices that could affect correctness, compatibility, or benchmarks.

## Terminology note: tabular ML

"Tabular ML" means learning from rows and named columns, such as a spreadsheet or SQL table: credit-risk prediction, customer churn, house-price regression, and similar classification/regression problems. Typical methods include linear/logistic regression, decision trees, random forests, and gradient-boosted trees. It is not central to this transformer-language-model assignment; MLJ was relevant only to the earlier broad survey of Julia's ML ecosystem.

## Next-session checklist

1. Read this file and confirm `git status --short --branch` shows `codex/julia-basics`.
2. Inspect whether `dev` moved; rebase this branch onto local `dev` if appropriate.
3. Reconfirm `julia --version`; Julia 1.12.6 was verified when this plan was created.
4. Run `juliaup status` and update the stable release channel through Juliaup only if it is behind.
5. Run `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test("CS336")'` and confirm the root manifest remains unchanged.
6. Do not install packages globally; invoke Julia with `--project=.` from the repository root.
7. Before Phase 1, consult and log official documentation for any proposed tokenizer/data dependency. Keep the already validated ML stack isolated; do not add it to the runtime package preemptively.
8. Update `JULIA_PARITY_MATRIX.md` after rebasing if the Python adapter surface changed.
9. Update `work-log.md` during the session and commit small coherent checkpoints.
