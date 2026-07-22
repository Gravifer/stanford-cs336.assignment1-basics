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
