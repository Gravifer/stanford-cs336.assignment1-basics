# Julia parallel implementation plan

## Resume state

- Branch: `codex/julia-basics` (local only; never push unless the user changes that instruction).
- Base at creation: local `dev` at `8c032c9` (`chore: prepare for training`), matching `origin/dev` on 2026-07-22.
- Branch policy: grow only from `dev`. Do not merge or rebase Python feature branches into this branch.
- Verified toolchain: Juliaup 1.20.8 with the stable `release` channel at Julia 1.12.6. `juliaup update release` reported no update on 2026-07-22.
- Current stage: Phase 0 scaffold complete, with optional readiness environments in progress. The root workspace, `CS336.jl` package boundary, test project, fixture readers, benchmark project, sole root manifest, CUDA readiness environment, and Lux compatibility environment are committed. The runtime package remains dependency-free and no assignment operation has been added.

When resuming, first run:

```powershell
git status --short --branch
git switch codex/julia-basics
git fetch --all --prune  # only if the user wants a network refresh
git rebase dev
```

Before rebasing, require a clean worktree or make a normal checkpoint commit. Do not use destructive reset/checkout commands to clear user work.

## Objective

Build a Julia package parallel to the repository's `cs336_basics` Python package, following the CS336 assignment sequence. Preserve mathematical behavior and course intent, but use idiomatic Julia rather than mechanically cloning PyTorch's object model. At the end of the syllabus, benchmark correctness, eager performance, compiled performance, memory, and startup/compilation costs against the Python/PyTorch implementation.

This project is about language/framework architecture as well as completing the course. Keep low-level educational implementations visible even when Julia or a vendor library has a faster built-in implementation.

## Companion records

- `JULIA_ML_ARCHITECTURE_NOTES.md` preserves the ecosystem conclusions and the boundary between structural streamlining and funded capability.
- `JULIA_PARITY_MATRIX.md` tracks the evolving Python adapter/test surface and assigns it to baseline or experimental Julia tracks.
- `JULIA_FIXTURE_CONTRACT.md` defines self-contained neutral inputs, weights, outputs, gradients, and metadata for cross-language parity.
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
- `NNlib.jl` for established neural-network primitives where using a library is consistent with the assignment.
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
3. A course-authored primitive can remain a plain Julia function; a small Lux layer can wrap it without making the framework own the math.
4. The explicit interface is suitable for Reactant/Enzyme whole-step compilation later.
5. It becomes easier to compare identical weights and state across the Python and Julia implementations.

Lux is not intended to replace the educational code. Implement softmax, RMSNorm, RoPE, attention, SwiGLU, AdamW, and related assignment targets explicitly first. Use Lux to compose and initialize the resulting layers. Keep an optimized/library-backed variant separate when it helps answer the performance question.

## Syllabus-aligned phases

### Phase 0: scaffold and cross-language fixtures — complete 2026-07-22

- Create the root workspace `Project.toml`, `CS336.jl/Project.toml`, `CS336.jl/src/CS336.jl`, tests, and documented commands. Resolve and commit the one root `Manifest.toml`.
- Use the verified Julia 1.12.6 runtime and record any later runtime upgrade in this plan and the benchmark metadata.
- Read existing `.npz`, JSON, text, vocabulary, and merge fixtures without rewriting them.
- Establish deterministic seeds, dtype conventions, tensor-axis conventions, and tolerances.
- Add one Python-to-Julia parity test before implementing the model stack.

### Phase 1: tokenizer and data

- Port BPE training, encoding, decoding, special-token behavior, and document chunking.
- Match existing fixture semantics byte-for-byte where practical.
- Benchmark BPE training throughput and tokenizer throughput separately from neural-network work.

### Phase 2: numerical primitives

- Stable softmax, linear, embedding, SiLU/SwiGLU, RMSNorm, cross-entropy, gradient clipping, and initialization.
- Test output values, edge cases, gradients, dtype promotion, allocations, and CPU/GPU behavior.
- For embedding, test both gradient values and the physical gradient representation; a mathematically row-sparse result is not necessarily stored sparsely by either framework.
- Keep array shapes explicit in docstrings and tests; Julia does not need a direct imitation of `jaxtyping`.

### Phase 3: attention and transformer layers

- RoPE, scaled dot-product attention, causal and additive/boolean masks, fully masked rows, MHA/GQA, transformer block, and language model.
- Decide and document one canonical Julia axis layout. Boundary adapters may transpose imported PyTorch weights or fixtures; do not scatter layout conversions through kernels.
- Match existing Python snapshots and gradient checks before optimizing.

### Phase 4: optimization and training

- Port AdamW, learning-rate scheduling, batching, checkpoint state, training loop, and sampling as they land on `dev`.
- Because this branch follows `dev`, rebase when the syllabus implementation advances there; do not import the intermediate Python feature branch.
- Add a one-step parity test, then a short loss-curve parity test.

### Phase 5: compiled and accelerator paths

- Establish native Julia CPU and CUDA-array baselines first.
- CUDA toolchain readiness is verified in `CS336.jl/environments/cuda`, and Lux forward/backward/update composition is verified in `CS336.jl/environments/lux_cuda`; operation/model baselines remain pending correctness implementations.
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

Do not compare a handwritten Julia reference kernel only against PyTorch's fused production kernel and call that a language result. Report both educational/reference and best practical paths.

## Design record: embedding-gradient sparsity

### Corrected reading of the PyTorch-internals example

The sparsity example in Edward Yang's [PyTorch internals](https://blog.ezyang.com/2019/05/pytorch-internals/) is specifically about embedding backward, not sparse forward weights. The embedding table can remain dense while the gradient with respect to it is row-sparse: only vocabulary rows selected by the batch receive lookup-gradient contributions. PyTorch uses this as a capability test for making sparse tensors participate directly in autograd rather than representing them as an opaque wrapper.

PyTorch does not enable that representation automatically. [`torch.nn.Embedding`](https://docs.pytorch.org/docs/2.13/generated/torch.nn.modules.sparse.Embedding.html) defaults to `sparse=False`; opting into `sparse=True` produces a sparse weight gradient, but the documented compatible optimizers are limited to SGD, SparseAdam, and CPU Adagrad.

The current Python course implementation calls `torch.nn.functional.embedding` without requesting a sparse gradient. Its embedding gradient is therefore dense. Unless `dev` changes this behavior, dense gradients are the required cross-language parity target.

### Current Julia finding

The investigated Julia stack also produces a dense embedding-weight gradient by default:

- Current [`Lux.Embedding`](https://github.com/LuxDL/Lux.jl/blob/82b8efede2a7f8523fd43da7c492e44d6ee7cd1a/src/layers/embedding.jl) performs ordinary indexed selection from a dense parameter array.
- The general [`ChainRules` indexing pullback](https://github.com/JuliaDiff/ChainRules.jl/blob/main/src/rulesets/Base/indexing.jl) allocates a gradient shaped like the source and scatter-adds into it. Its lightweight `OneElement` representation covers scalar indexing, not a batch of embedding rows.
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
