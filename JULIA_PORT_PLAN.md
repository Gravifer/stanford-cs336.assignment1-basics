# Julia parallel implementation plan

## Resume state

- Branch: `codex/julia-basics` (local only; never push unless the user changes that instruction).
- Base at creation: local `dev` at `8c032c9` (`chore: prepare for training`), matching `origin/dev` on 2026-07-22.
- Branch policy: grow only from `dev`. Do not merge or rebase Python feature branches into this branch.
- Verified runtime: Julia 1.12.6.
- Current stage: planning only. No Julia package or dependencies have been added yet.

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

## Proposed repository layout

```text
julia/
  Project.toml
  Manifest.toml
  src/
    CS336Basics.jl
    functional.jl
    layers.jl
    attention.jl
    transformer.jl
    tokenizer.jl
    data.jl
    optimizers.jl
    training.jl
  test/
    runtests.jl
    ...
  benchmark/
    correctness.jl
    kernels.jl
    model.jl
    training.jl
  environments/
    reactant/       # optional compiler stack, isolated from the baseline environment
```

Use the conventional Julia package name `CS336Basics`. Keep the Python package untouched except for deliberately shared fixtures or benchmark launchers.

## Blessed stack

Avoid an open-ended matrix of interchangeable frameworks. Start with one path:

- Julia 1.12 or newer stable release available on the machine.
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
y, new_state = model(x, parameters, state)
```

This is useful here because:

1. Parameters, non-trainable state, and architecture are not hidden behind mutable `nn.Module` registration rules.
2. The same parameter tree can be passed explicitly to differentiation, optimizers, serialization, CPU/GPU transfer, and benchmarks.
3. A course-authored primitive can remain a plain Julia function; a small Lux layer can wrap it without making the framework own the math.
4. The explicit interface is suitable for Reactant/Enzyme whole-step compilation later.
5. It becomes easier to compare identical weights and state across the Python and Julia implementations.

Lux is not intended to replace the educational code. Implement softmax, RMSNorm, RoPE, attention, SwiGLU, AdamW, and related assignment targets explicitly first. Use Lux to compose and initialize the resulting layers. Keep an optimized/library-backed variant separate when it helps answer the performance question.

## Syllabus-aligned phases

### Phase 0: scaffold and cross-language fixtures

- Create `julia/Project.toml`, module skeleton, tests, and documented commands.
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

## Terminology note: tabular ML

"Tabular ML" means learning from rows and named columns, such as a spreadsheet or SQL table: credit-risk prediction, customer churn, house-price regression, and similar classification/regression problems. Typical methods include linear/logistic regression, decision trees, random forests, and gradient-boosted trees. It is not central to this transformer-language-model assignment; MLJ was relevant only to the earlier broad survey of Julia's ML ecosystem.

## Next-session checklist

1. Read this file and confirm `git status --short --branch` shows `codex/julia-basics`.
2. Inspect whether `dev` moved; rebase this branch onto local `dev` if appropriate.
3. Reconfirm `julia --version`; Julia 1.12.6 was verified when this plan was created.
4. Create Phase 0 only: package scaffold, manifest, smoke test, fixture-loading test, and commands in the root README.
5. Do not install packages globally; use the `julia/` project environment.
6. Commit a small checkpoint before beginning tokenizer or neural-network implementation.
