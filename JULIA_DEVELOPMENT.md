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
```

`Project.toml` declares the workspace and `Manifest.toml` is its generated lockfile. The package and test subprojects share that one root manifest; child manifests are not expected.

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

## Cross-language conventions

These conventions apply before numerical parity fixtures are introduced:

- External token IDs retain the Python/course convention: zero-based integer values. Convert a token ID to Julia's one-based array index only at the actual lookup boundary; never mutate stored token IDs merely to suit indexing.
- Use `Int64` for token-ID fixture interchange with PyTorch `torch.long`. Julia-internal sizes and axes may use `Int` when they are not serialized or compared across languages.
- Use `Float32` as the default neural-network parity dtype. Tests for other dtypes must name the dtype explicitly and use dtype-appropriate tolerances.
- The canonical Julia hidden-state layout is `(feature, sequence, batch)`. Token-ID batches are `(sequence, batch)`. This matches the documented Lux attention layout and keeps feature vectors in the leading, contiguous dimension.
- The Python boundary layout remains `(batch, sequence, feature)`. Convert once in fixture/import adapters and keep layout conversion out of mathematical kernels.
- Attention scores use `(query_sequence, key_sequence, head, batch)` when exposed. Mask adapters must document their logical axes and verify broadcasting rather than relying on positional coincidence.
- Linear weights retain the logical `(output_feature, input_feature)` matrix shape used by both Julia matrix multiplication and PyTorch storage. Every other imported parameter receives an explicit mapping test before use.

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

Do not add Lux, an AD backend, accelerator libraries, or Reactant until the phase that exercises and tests that dependency.

## Planning records

- `JULIA_PORT_PLAN.md` — syllabus phases and design decisions.
- `JULIA_PHASE0_CHECKLIST.md` — scaffold acceptance gates.
- `JULIA_PARITY_MATRIX.md` — Python/Julia parity surface.
- `JULIA_ML_ARCHITECTURE_NOTES.md` — framework architecture findings.
- `JULIA_BENCHMARK_PROTOCOL.md` — eventual comparison methodology.
- `work-log.md` — timestamped operational record.

Documentation consulted for these conventions includes the official [Lux layer documentation](https://lux.csail.mit.edu/stable/api/Lux/layers), [NNlib batched-operation documentation](https://fluxml.ai/Flux.jl/stable/reference/models/nnlib/), [Julia random-number documentation](https://docs.julialang.org/en/v1/stdlib/Random/), and [Pkg environment documentation](https://pkgdocs.julialang.org/v1/environments/).
