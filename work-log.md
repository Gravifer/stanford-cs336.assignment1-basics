# Julia parallel work log

Timezone: Asia/Shanghai (UTC+08:00). Timestamps use ISO 8601.

## 2026-07-22

- 2026-07-22T19:52:31+08:00 — Resumed on the local-only `codex/julia-basics` branch. Confirmed the existing work is documentation-only and the worktree is clean.
- 2026-07-22T19:54:00+08:00 — Checked the user-global toolchain through Juliaup. Juliaup is version 1.20.8; the `release` channel is Julia 1.12.6. Ran `juliaup update release`; no newer stable Julia was available.
- 2026-07-22T19:54:00+08:00 — Consulted the official [Juliaup documentation](https://github.com/JuliaLang/juliaup), [Pkg environment documentation](https://pkgdocs.julialang.org/v1/environments/), [Pkg package documentation](https://pkgdocs.julialang.org/v1/creating-packages/), and [Project/Manifest documentation](https://pkgdocs.julialang.org/dev/toml-files/) before choosing the repository layout. Julia 1.12 workspaces support a root project and a single root manifest shared by child packages.
- 2026-07-22T19:54:00+08:00 — Began revising `JULIA_PORT_PLAN.md` to make `CS336.jl/` the package root, keep the Julia project environment and lockfile at repository root beside `pyproject.toml`, require documentation-first dependency decisions, and formalize branch, commit, work-log, and session stopping policies.
- 2026-07-22T19:55:14+08:00 — Committed the initial log as `1e51054` (`docs: start Julia work log`) and the revised workspace plan as `f98702e` (`docs: align Julia plan with root workspace`).
- 2026-07-22T19:56:26+08:00 — Inventoried the current `cs336_basics` package, course adapters, student tests, and fixtures on `dev`. The current surface includes GQA/layout variants, checkpointing, tokenizer CLI behavior, and symbolic cost analytics in addition to the core snapshot-tested transformer operations.
- 2026-07-22T19:56:26+08:00 — Consulted the current official Lux, Zygote, Enzyme, and Reactant manuals. Added `JULIA_PARITY_MATRIX.md` to separate baseline parity from compiled, sparse-embedding, and later analytics tracks and to retain the source register.
- 2026-07-22T19:57:38+08:00 — Committed the parity inventory as `a1c6849` (`docs: inventory Julia parity surface`).
- 2026-07-22T19:57:38+08:00 — Consulted the official BenchmarkTools, CUDA.jl, PyTorch benchmark utility, and Lux device-management documentation. Added `JULIA_BENCHMARK_PROTOCOL.md` to define correctness gates, cold/warm accounting, CPU/GPU timing controls, raw-result metadata, gap triage, and the greater-than-10× failure criterion.
- 2026-07-22T19:58:56+08:00 — Committed the benchmark protocol as `274b125` (`docs: define Julia benchmark protocol`).
- 2026-07-22T19:58:56+08:00 — Consulted the official Julia multiple-dispatch/performance manuals, Pkg extension documentation, GPUArrays interface, and ChainRules documentation. Added `JULIA_ML_ARCHITECTURE_NOTES.md` to preserve what Julia can streamline, what remains vendor/ecosystem work, why Lux is the baseline, the embedding-gradient conclusion, and the role of maintainer funding.
- 2026-07-22T19:59:57+08:00 — Committed the architecture findings as `b900099` (`docs: record Julia ML architecture findings`).
- 2026-07-22T19:59:57+08:00 — Consulted the official Pkg environment/package/testing/compatibility manuals and Julia `Test` documentation. Added `JULIA_PHASE0_CHECKLIST.md` with acceptance gates for the root workspace, exact `CS336.jl/` package skeleton, non-global dependency use, one root manifest, minimal fixture reader, optional compiler environments, and fine-grained commits.
- 2026-07-22T20:00:43+08:00 — Committed the Phase 0 checklist as `ab1bf05` (`docs: define Julia phase zero gates`). Verified the branch remains `codex/julia-basics`, local `dev` and the branch merge-base remain `8c032c9`, and no stale `julia/` or `CS336Basics` package naming remains in the Julia documents.
- 2026-07-22T20:00:43+08:00 — Added a companion-record index to `JULIA_PORT_PLAN.md` so the architecture notes, parity matrix, Phase 0 checklist, benchmark protocol, and work log are discoverable from the main plan.
- 2026-07-22T20:01:00+08:00 — Committed the companion-record index as `0d9df4f` (`docs: link Julia planning records`). Began the final documentation, history, and clean-worktree audit.
