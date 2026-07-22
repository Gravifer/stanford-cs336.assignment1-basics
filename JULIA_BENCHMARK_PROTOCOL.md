# Julia–PyTorch benchmark protocol

This protocol exists to answer a narrow question credibly: can the Julia implementation reach the same performance order of magnitude as the Python/PyTorch implementation, or is there a material ecosystem gap? A result is not evidence about either language until correctness, workload, timing, and backend equivalence have been checked.

## Comparison lanes

Report each available lane separately:

| Lane | Purpose | Included costs |
| --- | --- | --- |
| PyTorch eager | course baseline | steady execution; separate process startup |
| PyTorch compiled | mature compiled reference | graph capture/compile reported separately from warm execution |
| Julia reference | educational implementation | native Julia/Lux execution without claiming production fusion |
| Julia practical eager | best supported eager/library path | vendor-backed NNlib/accelerator operations where appropriate |
| Julia compiled | Reactant+Enzyme experiment | compilation reported separately from warm execution |

Do not collapse unavailable or failed lanes into zero, and do not substitute a different model or mathematical operation merely to make a lane compile.

## Required gates before timing

1. Match inputs, weights, dtypes, shapes, masks, layouts, seeds, and training/evaluation mode.
2. Check forward outputs against agreed tolerances.
3. Check parameter and input gradients where applicable.
4. For optimizer benchmarks, check updated parameters and optimizer state after more than one step.
5. Confirm both sides execute on the intended device and backend.
6. Confirm the measured region includes the same work and excludes equivalent setup.

A benchmark result that fails a gate is diagnostic data, not a comparative result.

## Cold and warm accounting

Record these independently:

- process startup and package import/load;
- accelerator initialization;
- first ordinary invocation;
- graph or ahead-of-time compilation;
- first execution of compiled code;
- steady-state distribution after warm-up.

Report the break-even iteration count when a compiled lane has a material up-front cost. Never amortize compilation into an unspecified number of steps or omit it from the report.

## CPU timing controls

- Use fresh processes for cold measurements and a stable long-lived process for warm trials.
- Fix Julia threads, PyTorch intra/inter-op threads, and BLAS threads. Record the BLAS implementation.
- Prepare inputs outside the timed expression unless data creation is the workload.
- Use the official BenchmarkTools interpolation/setup conventions so global lookup or setup allocation does not contaminate Julia timings.
- Use `evals=1` for mutating training steps unless a reset-per-evaluation setup is proven equivalent.
- Preserve raw samples and report at least median, a tail percentile, sample count, and dispersion rather than only the minimum.
- Run lanes in an order that avoids systematic thermal or background-load bias, and record material machine-load anomalies.

## Accelerator timing controls

GPU launches are asynchronous. Synchronize the relevant accelerator before and after each measured sample on both sides, using the backend's documented mechanism. The official CUDA.jl documentation warns that unsynchronized timing measures launch latency rather than completion; PyTorch's benchmark timer likewise synchronizes accelerators.

- Warm allocator pools and kernels before steady-state trials.
- Keep host-to-device transfer outside kernel/model timing unless transfer is explicitly the benchmark.
- Record device model, memory, driver, runtime, vendor libraries, framework/backend versions, power mode, and dtype features such as TF32 or BF16.
- Capture both host wall time and device/kernel profiles when investigating a gap.
- Report peak device memory with a consistent reset/measurement procedure and report host allocation separately.

## Workload levels

Benchmark at four levels so a high-level gap can be localized:

1. primitives: GEMM-facing linear, embedding, normalization, activation, softmax;
2. composed kernels: RoPE, attention, SwiGLU;
3. model: transformer block and full LM forward/backward;
4. training: loss, backward, clipping, optimizer update, and realistic data batch.

Use at least one tiny correctness shape and multiple representative performance shapes. Tiny shapes diagnose overhead but do not establish accelerator throughput.

The embedding study additionally holds token occurrences fixed while vocabulary size grows. Compare dense course paths first. Sparse variants form a separate lane and must include optimizer update and state, not backward alone.

## Lag triage and the 10× criterion

Calculate the Julia/PyTorch steady-state ratio for matched lanes. Interpret it as follows:

- up to 2× for vendor-backed dense primitives: ordinarily comparable, while still checking obvious layout/backend differences;
- 2×–3×: profile and explain;
- 3×–10×: material gap requiring localization and, when feasible, a supported optimized Julia path;
- over 10×: fails the comparability objective unless evidence isolates a missing Julia ecosystem capability rather than a benchmark or implementation defect.

For every result above 3×, inspect in this order:

1. correctness or workload mismatch;
2. missing synchronization or setup leakage;
3. dtype, device, tensor layout, or hidden copy/conversion;
4. unintended allocations, scalar indexing, dynamic dispatch, or type instability;
5. vendor-library selection and algorithm choice;
6. missing fusion or compilation;
7. genuinely unavailable backend feature.

The final category is where maintainer/industry investment becomes a defensible explanation. Framework fragmentation or package count alone is not performance evidence.

## Reproducibility record

Each result bundle must include:

- Git commit and dirty-worktree state;
- root `Manifest.toml` and Python lock state;
- Julia, Juliaup channel, Python, PyTorch, Lux, AD, compiler, and accelerator versions;
- OS, CPU, GPU, RAM, drivers, thread counts, and relevant environment settings;
- benchmark configuration, warm-up count, samples, seed, dtype, shapes, and synchronization method;
- machine-readable raw samples and a human-readable analysis;
- correctness tolerances and gate results.

Starting documentation, checked 2026-07-22:

- [BenchmarkTools manual](https://juliaci.github.io/BenchmarkTools.jl/stable/manual/)
- [CUDA.jl benchmarking and profiling](https://cuda.juliagpu.org/stable/development/profiling/)
- [CUDA.jl introductory synchronization example](https://cuda.juliagpu.org/stable/tutorials/introduction/)
- [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
- [Lux GPU management](https://lux.csail.mit.edu/stable/manual/gpu_management)
