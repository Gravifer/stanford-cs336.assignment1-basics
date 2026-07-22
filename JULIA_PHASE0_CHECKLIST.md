# Julia Phase 0 checklist

Phase 0 establishes a reproducible, minimal package and parity harness. It must not implement assignment components. Complete and commit each checkpoint independently.

## 0. Preflight

- Confirm the current branch is `codex/julia-basics`, the worktree is understood, and the branch is based on the intended local `dev` tip.
- Confirm the active Juliaup default channel is `release`; run its update check and record the Julia/Juliaup versions in `work-log.md`.
- Read the version-matched Pkg package, workspace, environment, test, and compatibility documentation before writing project metadata.
- Confirm no root `Project.toml` or `Manifest.toml` has appeared on `dev` since this checklist was written.

Acceptance: no branch switch, rebase, toolchain change, or generated file is left unrecorded.

## 1. Root workspace checkpoint

- Add a repository-root `Project.toml` whose purpose is the Julia workspace/environment.
- Register `CS336.jl` as a workspace member. Add test, benchmark, or compiler experiment members only when their projects actually exist.
- Do not create a root Julia source module; the package lives below `CS336.jl/`.
- Declare a Julia compatibility floor consistent with the workspace feature and verified runtime.
- Do not hand-author `Manifest.toml`; let Pkg resolve/generate it after the package metadata exists.

Acceptance: launching Julia with `--project=.` reports the repository-root project as active, and Pkg's workspace status sees the intended member without touching the user-global project.

## 2. Package skeleton checkpoint

- Create the exact directory `CS336.jl/`.
- Give the child package the Julia name `CS336`, a stable generated UUID, an initial development version, and explicit compatibility entries.
- Create the conventional entry point `CS336.jl/src/CS336.jl` containing only the module boundary and enough harmless package metadata or smoke behavior to prove loading works.
- Keep runtime dependencies empty until a documented Phase 0 need justifies each one. Do not add the entire proposed ML stack preemptively.

Acceptance: the package loads from the root workspace in a fresh Julia process with no use of the global environment and without implementing any CS336 assignment operation.

## 3. Test environment checkpoint

- Create `CS336.jl/test/runtests.jl` and, because this repository uses a Julia 1.12 workspace, a test project only when extra test dependencies require it.
- Start with Julia's `Test` standard library and a package-load/identity smoke test.
- Keep accelerator tests conditional and separately selectable; a CPU-only machine must be able to run the baseline suite.
- Record deterministic RNG policy, dtype conventions, Julia/Python indexing boundary, and the canonical tensor-axis convention before numerical fixtures are consumed.
- Do not mark unimplemented behavior as a passing or broken test merely to populate a test tree.

Acceptance: the documented test command succeeds in a fresh process, discovers only the root manifest, and proves the package under test is the local workspace member.

## 4. Fixture-reader checkpoint

- Inventory which existing fixtures are neutral formats: NPZ, JSON, UTF-8 text, and raw data.
- Select the minimum reader dependency needed for the first parity fixture only after checking its current official docs, release health, Julia compatibility, and license.
- Treat pickle and PyTorch checkpoint fixtures as Python-owned source artifacts. If Julia cannot safely read them without a Python bridge, plan a committed, provenance-recorded neutral fixture rather than making Python a hidden Julia test dependency.
- Verify one non-assignment-specific fixture-loading smoke case before any model operation is ported.

Acceptance: fixture bytes/shapes/dtypes are inspected deterministically and the new dependency is constrained in the child/test project as appropriate.

## 5. Manifest and reproducibility checkpoint

- Resolve and instantiate from the repository root with the root project active.
- Commit the generated root `Manifest.toml`; it is the only Julia lockfile.
- Confirm Pkg did not create `CS336.jl/Manifest.toml`, a test manifest, or files in the user-global environment.
- Run instantiate and package-load/test checks in a fresh process.
- Record exact resolved direct-package versions in `work-log.md`; the manifest remains the authority for all transitives.

Acceptance: cloning the repository and instantiating the root environment is sufficient to reproduce the Phase 0 package/test environment.

## 6. Optional environments—not yet baseline

Do not add Reactant, Enzyme, CUDA, AMDGPU, or other accelerator/compiler dependencies to the baseline merely because they appear in the eventual plan. Introduce `CS336.jl/environments/reactant/Project.toml` or accelerator-specific dependencies only at the phase that exercises them, as workspace members sharing the root manifest where Pkg supports the intended isolation.

Before doing so, verify from current documentation whether a weak dependency/package extension, a workspace subproject, or a separate benchmark environment best matches the loading and compatibility requirements. “Optional” must mean the CPU baseline can instantiate, load, and test without activating that feature.

## 7. Documentation and commit checkpoint

- Document the root activation, instantiate, test, and status commands in the repository README or a Julia-specific contributor note.
- Link `JULIA_PORT_PLAN.md`, `JULIA_PARITY_MATRIX.md`, `JULIA_BENCHMARK_PROTOCOL.md`, and `JULIA_ML_ARCHITECTURE_NOTES.md` from one discoverable place.
- Update `work-log.md` as each material action occurs.
- Keep commits fine-grained: workspace metadata, package skeleton, smoke test, fixture dependency/reader, generated manifest, and documentation should be independently reviewable when practical.
- Do not push.

## Official references checked 2026-07-22

- [Pkg environments and `--project`](https://pkgdocs.julialang.org/v1/environments/)
- [Pkg Project/Manifest and workspace files](https://pkgdocs.julialang.org/v1/toml-files/)
- [Creating packages and test-specific workspace dependencies](https://pkgdocs.julialang.org/v1/creating-packages/)
- [Pkg compatibility constraints](https://pkgdocs.julialang.org/v1/compatibility/)
- [Julia `Test` standard library](https://docs.julialang.org/en/v1/stdlib/Test/)
