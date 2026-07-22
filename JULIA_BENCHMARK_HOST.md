# Provisional benchmark host record

Captured 2026-07-22T20:35:52+08:00 on commit `a57c0f08081f50383e39f8d89a223d1ec1b54735` (`codex/julia-basics`). This is an environment-readiness snapshot, not a benchmark result or the final frozen configuration.

## Host

| Item | Observed value |
| --- | --- |
| OS | Microsoft Windows 11 Pro for Workstations, version 10.0.26200, 64-bit |
| CPU | 12th Gen Intel Core i7-12800H |
| CPU topology | 14 physical cores, 20 logical processors |
| Reported nominal/max clock field | 2400 MHz |
| Physical memory | 34,027,638,784 bytes (about 31.7 GiB) |
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| GPU memory | 8192 MiB |
| Compute capability | 8.6 |
| NVIDIA driver | 591.74 |
| GPU state during query | P5 |

The P5 state is evidence that the GPU was not in a steady benchmark load/power state. Final GPU trials must control warm-up, power/thermal conditions, synchronization, and background load rather than reusing this state as a benchmark setting.

## Julia baseline

| Item | Observed value |
| --- | --- |
| Julia | 1.12.6 through Juliaup `release` |
| Machine | `x86_64-w64-mingw32`, 64-bit |
| `Sys.CPU_THREADS` | 20 |
| Julia threads in default process | 1 |
| BLAS dispatcher/vendor | libblastrampoline (`lbt`) |
| BLAS backend | ILP64 OpenBLAS (`libopenblas64_.dll`) |
| BLAS threads in default process | 10 |

One Julia thread plus ten BLAS threads is merely the default observed configuration. CPU comparisons must explicitly choose and record Julia, BLAS, and PyTorch thread counts; otherwise nested or unequal parallelism can dominate the result.

## Python lock baseline

`uv tree --depth 1` resolved the current project to:

| Item | Locked value |
| --- | --- |
| Project | `cs336-basics` 26.0.0 |
| uv executable | 0.11.26; project dependency lock includes uv 0.11.29 |
| PyTorch | 2.11.0+cu130 |
| NumPy | 2.4.4 |
| einops | 0.8.2 |
| einx | 0.4.2 |

These are lock-resolution observations, not proof that a particular wheel/device backend successfully executed. The final benchmark record must capture runtime-reported PyTorch, CUDA runtime, cuDNN, device, and compiler information from the benchmark process.

## Freeze requirements before results

- Record the exact clean Git commit after all benchmark code and fixtures are present.
- Run correctness/gradient gates on that commit.
- Fix explicit CPU/BLAS/framework thread counts and affinity policy.
- Record laptop power source/profile, temperatures, GPU clocks/power mode, and background workload policy.
- Record Julia and Python environment/manifest hashes.
- Record runtime accelerator and vendor-library versions from inside each framework.
- Separate CPU and GPU trials and separate cold process/compilation costs from warm steady state.
- Preserve raw samples and synchronization metadata as required by `JULIA_BENCHMARK_PROTOCOL.md`.

## CUDA readiness check

Verified 2026-07-22T20:45:36+08:00 through the isolated `CS336.jl/environments/cuda` workspace member:

| Item | Runtime-reported value |
| --- | --- |
| CUDA.jl | 6.2.1 |
| CUDA runtime | 13.3.0, artifact installation |
| CUDA compiler | 13.3.33, artifact installation |
| NVIDIA driver | 591.74.0, reported for CUDA 13.1 |
| Device target | sm_86; PTX 9.3 (LLVM target PTX 9.0) |
| cuBLAS | 13.6.0 |
| cuSPARSE | 12.8.2 |
| cuSOLVER | 12.2.6 |
| cuFFT | 12.3.0 |
| cuRAND | 10.4.3 |
| GPU memory at query | 6.217 GiB available of 8.000 GiB |

`CUDA.functional(true)` succeeded. A 256-element `Float32` CuArray broadcast was executed, explicitly synchronized, copied to host, and verified exactly. The artifact runtime is newer than the native driver's reported CUDA level, so final benchmarks must record whether CUDA's compatibility mechanism is active and must not assume this combination has identical performance characteristics to a natively matching driver/runtime.

## LuxCUDA composition check

Verified 2026-07-22T20:58:55+08:00 through the isolated `CS336.jl/environments/lux_cuda` member. LuxCUDA 0.3.6 loaded with Lux 1.31.4, CUDA.jl 6.2.1, and cuDNN.jl 6.2.1. Lux's documented `gpu_device()` selected a CUDA device; Dense forward, explicit Zygote parameter backward, Optimisers Adam update, NNlib softmax, explicit synchronization, and CPU round-trip passed. For the tiny deterministic smoke input, the maximum absolute CPU/GPU forward difference was 1.1920929e-7.

This removes a basic integration unknown but does not establish throughput, kernel coverage, memory efficiency, or model-scale stability. Those remain subject to the correctness gates and controlled benchmark protocol.

## Provisional vendor-GEMM spot check

At 2026-07-22T21:14:45+08:00, a readiness-only Julia process allocated three 4096×4096 `Float32` CuArrays, warmed five allocation-free `mul!` calls, synchronized, then collected 20 `CUDA.@elapsed` samples. CUDA.jl documents CuArray matrix multiplication as a high-level route to cuBLAS.

Observed device times were 20.812 ms minimum, 23.499 ms median, and 37.931 ms maximum, corresponding to 5.85 decimal TFLOP/s at the median using `2n³/time`. Immediately after the process, the laptop GPU reported 87 °C and had returned to P8; power limit was unavailable through the query. Raw per-sample data was not preserved.

This demonstrates an operational vendor-library path, not Julia/PyTorch comparability. The thermal state, absent PyTorch lane, unverified TF32/math-mode equivalence, non-randomized lane order, and lack of raw samples disqualify it from final results. It is retained so later work does not accidentally promote this number into the benchmark report.
