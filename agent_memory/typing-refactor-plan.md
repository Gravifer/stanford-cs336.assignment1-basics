# NN typing refactor plan

## Scope and principles

- Work only in the already implemented `cs336_basics.nn` surface. Do not touch
  tokenizer code or implement missing assignment components.
- Keep assignment checkpoint tests unchanged. Put any new runtime-typing tests
  in a clearly separate top-level `typing_tests/` directory; retain exploratory
  scripts in `scripts/probes/`.
- Treat runtime checking as an optional diagnostic mode. Valid calls must work
  with checking both on and off. Essential project policy must still be
  enforced by implementation logic/einx when checking is off, but a beartype
  exception is sufficient when checking is on.
- Preserve named jaxtyping/einx variadics whenever they express a relationship.
  Use anonymous `...` only when neither system loses a constraint and the
  visual correspondence is not useful.
- Do not try to encode relationships jaxtyping cannot represent. Prefer a
  truthful weaker boundary plus existing einx/logic validation over an
  impressive-looking annotation that rejects valid calls.

## Phase 1: make the object model truthful and unblock strict runtime import

### `cs336_basics/nn/modules.py`

1. Replace `Linear.bias -> Never` with the truthful no-bias state (`bias: None`)
   while retaining `register_parameter("bias", None)`. Remove the property whose
   `AttributeError` currently falls through to `nn.Module` and returns `None`.
   This fixes both the beartype import failure and ty's false unreachable-code
   inference.
2. Change the deliberately unusable `SiLU.__init__` terminal annotation from
   `Never` to `NoReturn`. The installed beartype supports `NoReturn` in return
   position. Keep the existing raised exception and deprecation behavior.
3. Remove the stale `Embedding.freeze` annotation, since the runtime API stores
   freeze state in `weight.requires_grad` and exposes no `freeze` attribute.
4. Change `Embedding.__init__._weight` dimensions to refer to constructor
   arguments (`{num_embeddings} {embedding_dim}`), not attributes assigned only
   inside the body.
5. Fully annotate `Embedding.from_pretrained`: floating 2-D tensor input,
   `freeze: bool = True`, and `Self` return. This removes its current `Unknown`
   result in ty.
6. Make `RMSNorm.weight` optional to match the non-affine runtime state. Add an
   explicit internal narrowing assertion where `elementwise_affine` guarantees
   a parameter before initialization.

### Gate

- A strict import hook over `cs336_basics.nn` must import successfully without
  the probe-only skip-unsupported typechecker.
- Ruff and ty must remain clean.
- Static reveal spot checks should show `Linear.bias` as `None`,
  `Embedding.from_pretrained` as `Embedding`/`Self`, and non-affine
  `RMSNorm.weight` as optional.

## Phase 2: repair and strengthen tensor contracts

### `cs336_basics/nn/functional.py`

1. Replace GELU's invalid `*mapped {*dims}` input/return with one repeated named
   full shape such as `*shape`.
2. Make the implementation honor the documented/type-admitted
   `approximate="none"` mode. Keep `Literal["none", "tanh"]`; invalid strings
   should be rejected by beartype in checked mode and by the implementation in
   unchecked mode.
3. Change linear's anonymous mapped prefix to a named one in both systems:
   `*mapped` in jaxtyping and `mapped...` in einx. This verifies that the output
   preserves the complete input prefix.
4. Keep `glu`, `silu`, and `swiglu` named full-shape contracts; they already
   express useful input/return and cross-input relationships.
5. Keep embedding's named batch relationship; it already checks integer index
   dtype, sequence shape, and output prefix usefully.
6. Replace generic RMSNorm's invalid dynamic suffix annotations with truthful
   named full shapes: one repeated input/return shape and an independent weight
   shape. Let `dims`, einx constraints, and existing logic validate the
   normalized suffix. Update intermediate annotations for readable alignment,
   without treating local annotations as runtime checks.

### `cs336_basics/nn/modules.py`

7. Apply the same truthful generic RMSNorm signature to the static wrapper.
   Preserve the stronger `RMSNorm.forward` contract tied to `self.d_model`.
8. Leave `rms_norm_einx` out of this refactor as agreed.

### `cs336_basics/nn/initializer.py`

9. Annotate both in-place helpers as floating tensors with a repeated named
   `*shape` across input and return. Do not tie physical tensor dimensions to
   `d_in`/`d_out`: packed SwiGLU weights intentionally violate that matrix-shape
   assumption while using those values for initialization statistics.

### Gate

- All valid functional/module calls in `nn_import_hook_surface.py` must succeed
  under the strict package hook.
- Wrong related shapes must fail in checked mode; named output relationships
  must also be covered.
- Existing implemented checkpoint tests (`linear`, `embedding`, `swiglu`,
  `rmsnorm`, `rope`) must remain green without the hook.

## Phase 3: preserve concrete exported and override types

### `cs336_basics/nn/feed_forward.py`

1. Replace `SwiGLU: type = SwiGLU_packed_input` with either an inferred direct
   assignment or `type[SwiGLU_packed_input]`. Prefer the inferred assignment
   unless an explicit public annotation materially improves readability. Both
   preserve the concrete constructor and instance type across modules; broad
   `type` currently turns consumers into `Any`.
2. Review the three `load_state_dict` overrides as a separate, bounded change:
   - first decide whether their intentional public contract is current `None`
     or PyTorch-compatible `_IncompatibleKeys`;
   - then annotate mapping values and the return truthfully;
   - add direct tests for packed and unpacked input state dicts before changing
     behavior.
   Do not silently claim the PyTorch return contract while still discarding it.

### Gate

- A cross-module ty reveal should infer `SwiGLU_packed_input` for `SwiGLU(...)`,
  not `Any`.
- State-dict tests must cover all three accepted layouts and strict-mode errors.

## Phase 4: centralize optional runtime checking

1. Remove the explicit `@jaxtyped(typechecker=beartype)` from
   `RotaryPositionalEmbedding.forward` once strict package import succeeds.
   Otherwise the package hook double-wraps it and creates redundant dynamic
   contexts.
2. Use jaxtyping's installed pytest hook as the project-wide switch rather than
   enabling it in default assignment configuration:

   ```powershell
   # unchecked
   .\.venv\Scripts\python.exe -m pytest -q typing_tests

   # checked
   .\.venv\Scripts\python.exe -m pytest -q typing_tests `
     --jaxtyping-packages=cs336_basics.nn,beartype.beartype
   ```

3. Design `typing_tests/` around behavior categories:
   - valid calls must pass and produce equivalent outputs in both modes;
   - project-policy violations must fail in both modes, without requiring the
     same exception class;
   - annotation-only violations are expected to be rejected only in checked
     mode;
   - package import itself is a checked-mode smoke test.
4. Keep the default `pyproject.toml` pytest options unchanged so Stanford's
   checkpoint suite is not silently converted into a runtime-typing suite.

### Gate

- Run the separate typing suite once without the hook and once with the strict
  package hook.
- Run the implemented assignment-test subset in both modes. Checked mode must
  no longer fail during import.
- Confirm RoPE wrapper depth is one in checked mode and zero in normal mode.

## Phase 5: RoPE annotation cleanup and focused regression matrix

### `cs336_basics/nn/attention/__init__.py`

1. Retain dependent class-local `KeyVec`/`HalfKey` aliases and the distinct
   `*map_batch`/`*batch` names. Probes confirm these work and that the two
   variadics intentionally cannot express the suffix relationship.
2. Remove only the stale `UP037` suppression from `inv_freq`; retain the
   necessary suppression on the nested `Shaped[self.HalfKey, ...]` expression.
3. Add checked/unchecked cases for exact batches, unbatched position broadcast,
   wrong key width, wrong sequence length, non-integer positions, and
   `broadcast_positions=False`.

## Explicit parking lot: do not hide these in typing edits

1. **RoPE nonempty batch-suffix broadcasting:** comments say it is valid, but
   both checked and unchecked implementations reject it. Decide separately
   whether to repair the shape guard and add a behavioral regression test.
2. **RoPE floating precision:** `Float` reasonably accepts float16/32/64, but
   the float32 cached rotation currently makes float16/64 calls fail. This is a
   buffer-dtype/operation policy decision, not expressible as the current shape
   annotation.
3. **Negative/out-of-range positions:** jaxtyping checks dtype/shape, not value
   ranges. Decide whether project logic should reject them; do not add guards
   merely to duplicate type checking.
4. **State-dict return compatibility:** resolve in Phase 3 with tests rather
   than opportunistically during annotation cleanup.

## Final verification

```powershell
.\.venv\Scripts\ruff.exe format --check cs336_basics\nn scripts\probes typing_tests
.\.venv\Scripts\ruff.exe check cs336_basics\nn scripts\probes typing_tests
.\.venv\Scripts\ty.exe check cs336_basics\nn scripts\probes typing_tests
.\.venv\Scripts\python.exe -m pytest -q `
  tests\test_model.py::test_linear `
  tests\test_model.py::test_embedding `
  tests\test_model.py::test_swiglu `
  tests\test_model.py::test_rmsnorm `
  tests\test_model.py::test_rope
.\.venv\Scripts\python.exe -m pytest -q typing_tests
.\.venv\Scripts\python.exe -m pytest -q typing_tests `
  --jaxtyping-packages=cs336_basics.nn,beartype.beartype
```

Completion requires all gates above, a clean strict package import, no new
`Any`/`Unknown` at the public surfaces identified by the probes, and no changes
to tokenizer or missing assignment implementations.
