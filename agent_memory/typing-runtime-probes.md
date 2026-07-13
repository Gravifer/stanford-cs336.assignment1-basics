# Typing runtime probes

Observed on 2026-07-13 with Python 3.12.2, PyTorch 2.11.0+cpu,
jaxtyping 0.3.9, beartype 0.22.9, einx 0.4.2, and einops 0.8.2.

The probes live in `scripts/probes/` and are intentionally separate from the
assignment checkpoint tests.

## Stable observations

- In a `@jaxtyped(typechecker=beartype)` method signature, `{self.width}` is
  resolved correctly through both a module-level PEP 695 alias and a class-local
  PEP 695 alias nested in `Shaped[...]`.
- An ordinary annotated instance attribute is not checked when assigned.
- A class-local PEP 695 alias cannot be passed directly as the second argument
  to `isinstance`. Nesting it as `Shaped[Alias, ""]` works inside a
  `jaxtyped` context, as does checking against `Alias.__value__`.
- The project's `Linear` works as a jaxtyping duck array. Its `torch.dtype`
  property is accepted by `Shaped[Linear, ...]`, `Float[Linear, ...]`, and a
  beartype-checked function parameter. A string-valued dtype is not required in
  these installed versions.
- A repeated named jaxtyping variadic such as `*mapped` relates the full prefix
  shape across parameters and return values. Anonymous `...` does not establish
  that relationship.
- For one shared vectorized group in an einx expression, named `mapped...` and
  anonymous `...` enforce the same shape relationship. Two independent groups,
  such as `mapped...` and `batch...`, require distinct names; replacing both
  with anonymous ellipses is invalid.
- `install_import_hook` provides a working global on/off comparison. With the
  beartype hook enabled, annotation violations fail before the function body.
  Without the hook, explicit project guards still run, while annotation-only
  constraints are not enforced.
- Local variable annotations are not runtime-checked, even inside a
  `@jaxtyped(typechecker=beartype)` function. They remain documentation/static
  metadata only.
- Constructor annotations can safely use earlier constructor arguments, e.g.
  `{rows} {columns}`. This accepts a matching supplied tensor and rejects a
  mismatch before the constructor body.

## Findings on the current NN surface

- A strict package-wide import hook currently fails during import because
  beartype 0.22.9 cannot decorate `typing.Never`. The blockers are
  `Linear.bias` and the deliberately unusable `SiLU.__init__`.
- `typing.NoReturn` is supported by both beartype and
  `jaxtyped(typechecker=beartype)`. It is a viable terminal-return annotation;
  `Never` is not in this runtime configuration.
- `Linear.bias` remains deliberately terminal to ty through a class-only
  `Never` annotation, while runtime registered-parameter lookup returns `None`.
  This preserves the static warning without exposing `Never` to beartype.
- `RotaryPositionalEmbedding.forward` is no longer explicitly decorated. The
  package hook supplies exactly one wrapper, while normal imports supply none.
- `Embedding.__init__` works under the hook when `_weight` is `None`, but a
  supplied tensor fails type checking because `{self.num_embeddings}` and
  `{self.embedding_dim}` are evaluated before those attributes are assigned.
  Constructor-argument expressions are the tested replacement.
- `*mapped {*dims}`, `{*dims}`, tuple substitution, and attempted joined-string
  substitution all reject valid generic RMSNorm inputs. Jaxtyping splits the
  shape expression into axes before f-string evaluation and permits only one
  variadic group, so it cannot express an arbitrary mapped prefix plus an
  arbitrary runtime-sized normalized suffix this way.
- A conservative generic RMSNorm annotation using named full shapes
  (`*input_shape`, `*weight_shape`) works and preserves the input/return shape
  relationship, but intentionally cannot relate the normalized suffix to
  `dims`; einx/project logic must enforce that relationship.
- `gelu` has the same invalid `*mapped {*dims}` annotation even though it needs
  only one arbitrary shape. A single named `*shape` is sufficient and stronger
  across its input and return.
- The documented GELU exact mode (`approximate="none"`) currently reaches the
  implementation's unknown-mode branch. This is a behavior/annotation
  inconsistency, not a runtime-checker issue.
- Initializer helpers accept float16, float32, and float64 tensors and reject
  integer and complex tensors in PyTorch internals. They also intentionally
  accept packed or otherwise arbitrary tensor shapes, so a named `*shape`
  floating-tensor contract is appropriate, while tying the physical shape to
  `d_in`/`d_out` is not.

## Static reveal findings

- `Embedding.freeze` is inferred as `bool` but the attribute is absent at
  runtime.
- `RMSNorm.weight` is inferred as `Tensor` in both affine modes, but is `None`
  when `elementwise_affine=False`.
- `Embedding.from_pretrained` returns `Unknown` because its signature is
  unannotated.
- Exporting `SwiGLU` as `SwiGLU: type = SwiGLU_packed_input` makes construction
  and calls `Any` in consumers. Either an inferred assignment or
  `type[SwiGLU_packed_input]` preserves the concrete type across modules.
- The custom `load_state_dict` overrides are inferred as returning `None`,
  matching their current behavior but differing from the usual PyTorch return
  contract.

## RoPE checked/unchecked boundary

- Both modes accept exact batches, unbatched positions broadcast over mapped
  axes, int32 positions, and negative positions.
- Checked mode improves failures for wrong key width, wrong sequence length,
  and floating positions; unchecked mode fails later through manual guards or
  einx/PyTorch. Integer activations are an annotation-only violation: checked
  mode rejects them, while unchecked promoted computation can execute them.
- Float16, bfloat16, float32, and float64 activations now use a promoted
  operation dtype (at least float32) and return in the activation dtype.
- Out-of-range position values reach einx/PyTorch in both modes, as expected for
  a shape/dtype checker.
- Nonempty batch-suffix broadcasting now uses the corrected negative slice
  `-token_positions.ndim - 1` and matches the documented einx layout.

## Project policy captured from discussion

- Prefer `torch.dtype` directly for tensor-like objects and verify behavior with
  probes rather than converting it to a string for jaxtyping.
- Replace named variadics with `...` only when doing so preserves the intended
  constraints in both jaxtyping and the corresponding einx context. Preserve
  the visual correspondence between `*mapped` and `mapped...` when it conveys a
  real relationship. Intermediate-only annotations may be treated more loosely.
- Runtime checking is diagnostic and should be flexibly comparable with
  beartype on and off. If beartype rejects an invalid input that project logic
  would otherwise reject, its exception is sufficient; do not catch it merely
  to translate it.
- When behavior differs between checked and unchecked modes, reconsider whether
  the annotation expresses reasonable policy before adding or changing logic
  guards.
- Additional typing tests/probes must remain clearly separated from the existing
  assignment checkpoint suite.
- `RMSNorm.rms_norm_einx` is an experiment and is outside the current typing
  refactor scope.
- Later in this branch, remove the stale `UP037` suppression on `inv_freq` in
  `RotaryPositionalEmbedding.__init__`.

## Commands

```powershell
.\.venv\Scripts\python.exe scripts\probes\jaxtyping_runtime.py
.\.venv\Scripts\python.exe scripts\probes\runtime_import_hook.py on
.\.venv\Scripts\python.exe scripts\probes\runtime_import_hook.py off
.\.venv\Scripts\python.exe scripts\probes\beartype_terminal_hints.py
.\.venv\Scripts\python.exe scripts\probes\initializer_surface.py
.\.venv\Scripts\python.exe scripts\probes\rms_annotation_forms.py
.\.venv\Scripts\python.exe scripts\probes\rope_checked_unchecked.py
.\.venv\Scripts\python.exe scripts\probes\nn_import_hook_surface.py off
.\.venv\Scripts\python.exe scripts\probes\nn_import_hook_surface.py on --scope functional
# Probe-only shim to see past unsupported Never annotations:
.\.venv\Scripts\python.exe scripts\probes\nn_import_hook_surface.py on --scope package --typechecker skip-unsupported
.\.venv\Scripts\ruff.exe check scripts\probes
.\.venv\Scripts\ty.exe check scripts\probes
```
