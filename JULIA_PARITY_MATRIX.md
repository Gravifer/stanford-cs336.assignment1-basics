# Julia parity matrix

This is the evolving scope ledger for the Julia package under `CS336.jl/`. It records what is reachable through working functions in `tests/adapters.py`, what constitutes parity, and which work belongs to the direct port versus an optional ecosystem experiment. It is not a student-facing assignment specification.

Status vocabulary: **planned** means exposed by a working adapter and assigned to a phase, **gated** means the named adapter still raises `NotImplementedError`, and **excluded** means Python functionality is not part of the adapter progress boundary.

Harness status as of 2026-07-23: the root Julia workspace reads every legacy
NPZ snapshot plus versioned self-contained numerical bundles through test-only
NPZ and JSON dependencies. Linear, embedding, SiLU, softmax, RMSNorm, and both
SwiGLU representations now have independent forward/gradient evidence. The
tokenizer additionally matches exact GPT-2 IDs on ASCII, Unicode, and special
token cases, while the trainer matches Python on four deliberately tiny
corpora. The Phase 2 primitives also pass an optional CPU/CUDA forward and
Zygote-gradient comparison on CUDA.jl 6.2.1. RoPE, four-dimensional SDPA,
causal MHA, and causal MHA with RoPE now have self-contained Python
forward/gradient bundles as well. The LuxCore transformer block and tiny
two-layer LM have semantic parameter-tree gradient bundles.
`JULIA_SNAPSHOT_PROVENANCE.md` records the remaining limitations of the legacy
output-only snapshots.

## Public parity surface on `dev`

| Area | Current Python adapter or evidence | Julia phase | Initial status | Parity evidence |
| --- | --- | ---: | --- | --- |
| Linear | `run_linear`; self-contained v1 bundle | 2 | parity-validated | feature-first output plus dense input/weight gradients through explicit Zygote arguments |
| Embedding | `run_embedding`; self-contained v1 bundle | 2 | parity-validated | zero-based lookup, repeated-ID accumulation, output, and full dense weight gradient through explicit Zygote arguments |
| SiLU / SwiGLU | `run_silu`, `run_swiglu`; self-contained v1 bundles | 2 | SiLU parity-validated; SwiGLU parity-validated | two Julia lanes only: explicit separate and packed input projection; output/input/all-parameter gradients |
| Stable softmax | `run_softmax`; self-contained v1 bundle | 2 | parity-validated | explicit Julia dimension mapping, large-offset stability, dtype, output, and input gradient |
| RMSNorm | `run_rmsnorm`; self-contained v1 bundle | 2 | parity-validated | feature-axis output, stable reduction dtype, input and affine-weight gradients |
| Cross-entropy | `run_cross_entropy` raises `NotImplementedError` | gated | gated | expand scope only after Python adapter implementation |
| Gradient clipping | `run_gradient_clipping` raises `NotImplementedError` | gated | gated | expand scope only after Python adapter implementation |
| RoPE | `run_rope`; self-contained v1 bundle | 3 | parity-validated | elementwise/matrix rotation values, irregular batched positions, Float16 dtype, input gradient, automatic/separate/stacked Q/K execution, and optional CUDA forward/gradient smoke |
| Scaled dot-product attention | `run_scaled_dot_product_attention`; self-contained 4-D v1 bundle | 3 | parity-validated | output and Q/K/V gradients, boolean/additive/causal mask semantics, and finite zero fully masked rows |
| MHA / self-attention | attention adapters; self-contained plain/RoPE v1 bundles | 3 | parity-validated | packed/separate projections, causal outputs and all parameter/input gradients, two feature-first head layouts, and three Q/K RoPE policies |
| Grouped/multi-query attention | no direct adapter; implemented inside adapter-reachable attention family | 3 experiment | structurally validated execution lane | grouped contraction matches expanded-KV MHA without materializing repeated K/V; MHA/GQA/MQA timing and grouped CUDA remain pending |
| Transformer block | `run_transformer_block`; self-contained v1 bundle | 3 | parity-validated | LuxCore explicit parameter/state interface; packed attention/SwiGLU; output, input gradient, and every semantic parameter gradient |
| Transformer LM | `run_transformer_lm`; tiny two-layer self-contained v1 bundle | 3 | parity-validated | feature-first logits, truncated/context behavior, one shared device-movable RoPE state, every semantic parameter-tree gradient, and full LuxCUDA forward/gradient-tree agreement |
| Batch sampling | `run_get_batch` raises `NotImplementedError` | gated | gated | expand scope only after Python adapter implementation |
| AdamW | `get_adamw_cls` raises `NotImplementedError` | gated | gated | legacy NPZ output alone is not an adapter boundary |
| Cosine schedule | `run_get_lr_cosine_schedule` raises `NotImplementedError` | gated | gated | expand scope only after Python adapter implementation |
| Checkpoints | save/load adapters raise `NotImplementedError` | gated | gated | expand scope only after Python adapter implementation |
| BPE training | `run_train_bpe`; tiny Python probes plus repository fixtures | 1 | tiny-corpus parity-validated | vocabulary bytes, merge order/tie-break, special boundaries, merge exhaustion, serial/threaded determinism; full-corpus run user-deferred |
| BPE tokenizer | `get_tokenizer`; GPT-2 vocabulary/merges and tokenizer tests | 1 | parity-validated | exact encode IDs, replacement decode, overlapping/disallowed special tokens, streaming input, and GPT-2 fixture round trips |
| Symbolic cost analytics | no adapter function | excluded | excluded | outside current port boundary |

The matrix follows the adapter boundary for parity status, not as permission to
erase meaningful implementation choices inside an adapter-reachable component.
`JULIA_VARIANT_AUDIT.md` records which Python distinctions become Julia
execution lanes, which collapse naturally under explicit parameters, and which
remain outside scope. When `dev` advances, update both records after rebasing
and before expanding implementation scope. Do not turn either ledger into a
Julia assignment harness.

## Cross-language fixture policy

- Prefer language-neutral NPZ, JSON, UTF-8 text, and raw binary fixtures.
- Existing pickle and PyTorch checkpoint files may be treated as source evidence, but do not make the Julia test suite depend on a Python interpreter merely to decode every test case. Produce any new neutral fixture deliberately and record its provenance.
- Verify mathematical results before performance. Matching an output snapshot is insufficient when the backward representation or optimizer semantics are part of the question.
- Establish explicit dtype, indexing, seed, and tensor-axis conventions at the boundary. Julia's one-based indexing and column-major storage must not leak into externally visible token IDs or silently change the benchmark workload.
- Preserve a distinction between course-authored reference paths and best practical library/compiler paths.

Current numeric snapshot inventory:

| Snapshot | Stored logical shape |
| --- | --- |
| `test_4d_scaled_dot_product_attention.npz` | `(2, 2, 12, 64)` |
| `test_adamw.npz` | `(2, 3)` |
| `test_embedding.npz` | `(4, 12, 64)` |
| `test_linear.npz` | `(4, 12, 128)` |
| `test_multihead_self_attention.npz` | `(4, 12, 64)` |
| `test_multihead_self_attention_with_rope.npz` | `(4, 12, 64)` |
| `test_positionwise_feedforward.npz` | `(4, 12, 64)` |
| `test_rmsnorm.npz` | `(4, 12, 64)` |
| `test_rope.npz` | `(4, 12, 64)` |
| `test_scaled_dot_product_attention.npz` | `(4, 12, 64)` |
| `test_swiglu.npz` | `(4, 12, 64)` |
| `test_transformer_block.npz` | `(4, 12, 64)` |
| `test_transformer_lm.npz` | `(4, 12, 10000)` |
| `test_transformer_lm_truncated_input.npz` | `(4, 6, 10000)` |

NPZ.jl returns these arrays with the same logical dimension order stored by NumPy. Import adapters must convert Python `(batch, sequence, feature)` tensors to the canonical Julia `(feature, sequence, batch)` layout exactly once. The fixture tests intentionally do not perform that conversion.

These output-only snapshots are not self-contained parity fixtures. Their inputs come from seeded PyTorch RNG fixtures and their weights come from a PyTorch checkpoint, neither of which Julia should reproduce implicitly. `JULIA_FIXTURE_CONTRACT.md` defines the versioned neutral bundle required before an operation can move from planned to parity-validated.

## Ecosystem tracks

The baseline track uses ordinary Julia functions, Lux's explicit parameter/state interface, Zygote for the first reverse-mode path, and Optimisers where a library optimizer is being used for comparison. The official Lux documentation states that models do not own parameters/state and documents `Lux.setup` plus `Lux.apply(model, x, ps, st)`; this is the architectural reason for selecting Lux, not an assumption copied from Python.

The compiled track is separate. Current official Lux documentation directly documents Reactant compilation and `AutoEnzyme`, while Reactant documents Enzyme-based differentiation. That makes Reactant+Enzyme a credible benchmark target, but not a prerequisite for correctness. Zygote's official limitations still identify array mutation as a major constraint, so mutation-heavy educational code must not be advertised as AD-portable until tested.

The sparse-embedding track is also separate. The baseline matches the course's dense embedding gradient. A row-sparse Julia gradient counts as implemented only when the tangent, repeated-index coalescing, parameter-tree traversal, optimizer update/state, weight decay, serialization, device transfer, and benchmark all avoid accidental densification or semantic substitution.

## Documentation register

Use current upstream manuals before implementation. Starting references, verified 2026-07-22:

- [Julia Pkg environments](https://pkgdocs.julialang.org/v1/environments/)
- [Julia Pkg Project and Manifest files, including workspaces](https://pkgdocs.julialang.org/dev/toml-files/)
- [Creating Julia packages](https://pkgdocs.julialang.org/v1/creating-packages/)
- [Lux introduction and explicit parameter/state quickstart](https://lux.csail.mit.edu/stable/introduction/)
- [Lux layer interface](https://lux.csail.mit.edu/stable/manual/interface)
- [Lux automatic differentiation](https://lux.csail.mit.edu/stable/manual/autodiff)
- [Compiling Lux models with Reactant](https://lux.csail.mit.edu/stable/manual/compiling_lux_models)
- [Zygote documentation](https://fluxml.ai/Zygote.jl/stable/)
- [Zygote limitations](https://fluxml.ai/Zygote.jl/stable/limitations/)
- [Enzyme.jl documentation](https://enzymead.github.io/Enzyme.jl/stable/)
- [Reactant automatic differentiation](https://enzymead.github.io/Reactant.jl/stable/tutorials/automatic-differentiation)

The isolated compatibility project locks Lux 1.31.4, NNlib 0.9.38,
Optimisers 0.4.7, and Zygote 0.7.11 on Julia 1.12.6. Its explicit
setup/apply/gradient/update/softmax smoke test passes. Phase 3 now uses NNlib
0.9.38 for batched contractions and LuxCore 1.5.3 for explicit model-layer
interfaces as direct runtime dependencies; full Lux, Optimisers, and Zygote
have not been promoted merely because the integration stack loads.
