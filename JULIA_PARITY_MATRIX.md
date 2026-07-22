# Julia parity matrix

This is the evolving scope ledger for the Julia package under `CS336.jl/`. It records what is reachable through working functions in `tests/adapters.py`, what constitutes parity, and which work belongs to the direct port versus an optional ecosystem experiment. It is not a student-facing assignment specification.

Status vocabulary: **planned** means exposed by a working adapter and assigned to a phase, **gated** means the named adapter still raises `NotImplementedError`, and **excluded** means Python functionality is not part of the adapter progress boundary.

Harness status as of 2026-07-22: the root Julia workspace can read every existing `.npz` snapshot through test-only NPZ v0.4.3 and structured fixture metadata through test-only JSON.jl v1.6.1. The 65-assertion suite locks all 14 NPZ filenames/key/dtype/shapes, the complete model configuration, and tokenizer vocabulary/merge transport. This proves fixture interoperability only; none of the operations below gains parity status until its Julia outputs and gradients are compared independently. `JULIA_SNAPSHOT_PROVENANCE.md` records that one legacy NPZ is orphaned and that existing output snapshots omit neutral inputs, parameters, gradients, and tolerance metadata.

## Public parity surface on `dev`

| Area | Current Python adapter or evidence | Julia phase | Initial status | Parity evidence |
| --- | --- | ---: | --- | --- |
| Linear | `run_linear`; self-contained v1 bundle | 2 | parity-validated | feature-first output plus dense input/weight gradients through explicit Zygote arguments |
| Embedding | `run_embedding`; self-contained v1 bundle | 2 | parity-validated | zero-based lookup, repeated-ID accumulation, output, and full dense weight gradient through explicit Zygote arguments |
| SiLU / SwiGLU | `run_silu`, `run_swiglu`; SiLU v1 bundle | 2 | SiLU parity-validated; SwiGLU planned | SiLU output/input gradient and extreme-value stability; SwiGLU remains |
| Stable softmax | `run_softmax`; self-contained v1 bundle | 2 | parity-validated | explicit Julia dimension mapping, large-offset stability, dtype, output, and input gradient |
| RMSNorm | `run_rmsnorm`; NPZ snapshot | 2 | planned | output and gradient fixtures |
| Cross-entropy | `run_cross_entropy` raises `NotImplementedError` | gated | gated | expand scope only after Python adapter implementation |
| Gradient clipping | `run_gradient_clipping` raises `NotImplementedError` | gated | gated | expand scope only after Python adapter implementation |
| RoPE | `run_rope`; NPZ snapshot | 3 | planned | rotation values, positions, dtype, and device behavior |
| Scaled dot-product attention | `run_scaled_dot_product_attention`; 3-D and 4-D snapshots | 3 | planned | outputs, gradients, mask semantics, and fully masked rows |
| MHA / self-attention | attention adapters and snapshots | 3 | planned | parameter mapping, outputs, gradients, and shape contracts |
| Grouped-query attention | no adapter function | excluded | excluded | outside current port boundary |
| Transformer block | `run_transformer_block`; NPZ snapshot | 3 | planned | parameter import, output, and gradients |
| Transformer LM | `run_transformer_lm`; full and truncated snapshots | 3 | planned | logits, truncation behavior, state mapping, and gradients |
| Batch sampling | `run_get_batch` raises `NotImplementedError` | gated | gated | expand scope only after Python adapter implementation |
| AdamW | `get_adamw_cls` raises `NotImplementedError` | gated | gated | legacy NPZ output alone is not an adapter boundary |
| Cosine schedule | `run_get_lr_cosine_schedule` raises `NotImplementedError` | gated | gated | expand scope only after Python adapter implementation |
| Checkpoints | save/load adapters raise `NotImplementedError` | gated | gated | expand scope only after Python adapter implementation |
| BPE training | `run_train_bpe`; text/JSON/pickle fixtures | 1 | planned | vocabulary bytes, merge order, special tokens, and determinism |
| BPE tokenizer | `get_tokenizer`; tokenizer and CLI tests | 1 | planned | encode/decode, special tokens, streaming input, and GPT-2 fixtures |
| Symbolic cost analytics | no adapter function | excluded | excluded | outside current port boundary |

The matrix follows the adapter boundary because that is the explicit progress contract selected for this port. Internal Python class structure and Python-only tests are not automatically Julia requirements. When `dev` advances, update this matrix after rebasing and before expanding implementation scope. Do not turn this ledger into a Julia assignment harness.

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

The isolated compatibility project now locks Lux 1.31.4, NNlib 0.9.38, Optimisers 0.4.7, and Zygote 0.7.11 on Julia 1.12.6. Its explicit setup/apply/gradient/update/softmax smoke test passes. These packages have not been promoted to runtime dependencies, and no operation in this matrix becomes “implemented” merely because the integration stack loads.
