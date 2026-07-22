# Julia parallel work log

Timezone: Asia/Shanghai (UTC+08:00). Timestamps use ISO 8601.

## 2026-07-22

- 2026-07-22T19:52:31+08:00 — Resumed on the local-only `codex/julia-basics` branch. Confirmed the existing work is documentation-only and the worktree is clean.
- 2026-07-22T19:54:00+08:00 — Checked the user-global toolchain through Juliaup. Juliaup is version 1.20.8; the `release` channel is Julia 1.12.6. Ran `juliaup update release`; no newer stable Julia was available.
- 2026-07-22T19:54:00+08:00 — Consulted the official [Juliaup documentation](https://github.com/JuliaLang/juliaup), [Pkg environment documentation](https://pkgdocs.julialang.org/v1/environments/), [Pkg package documentation](https://pkgdocs.julialang.org/v1/creating-packages/), and [Project/Manifest documentation](https://pkgdocs.julialang.org/dev/toml-files/) before choosing the repository layout. Julia 1.12 workspaces support a root project and a single root manifest shared by child packages.
- 2026-07-22T19:54:00+08:00 — Began revising `JULIA_PORT_PLAN.md` to make `CS336.jl/` the package root, keep the Julia project environment and lockfile at repository root beside `pyproject.toml`, require documentation-first dependency decisions, and formalize branch, commit, work-log, and session stopping policies.
