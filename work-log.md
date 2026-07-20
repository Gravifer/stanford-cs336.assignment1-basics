# Work Log

This chronological log records implementation activity for later audit. It is not a project design document, user guide,
or source of repository policy.

## 2026-07-20

- 18:26 UTC+8 — Began the unattended continuation on a clean `feat/transformer-lm` worktree at `3147f32`.
- Review triage: treat the package façade, defensive `DeltaLayer` alias, and unused hook-parameter style as actionable.
  Keep the nested feed-forward name and dependency-injection constructors. Do not rename the attention wrapper to a name
  that falsely guarantees RoPE.
- Deferred-work priority: first investigate a homogeneous, independently constructed layer stack against PyTorch's
  `TransformerEncoder`/`ModuleList` precedent. Preserve `layers.N.*` checkpoint keys and distinct parameter objects.
- 18:31–18:40 UTC+8 — Added a coherent `cs336_basics.nn` façade in `5118044`, then kept deprecated `SiLU` off that
  façade after review in `7e118ad`. Removed defensive `DeltaLayer` import aliases and retained implementation-module
  access for every exported class.
- 18:42 UTC+8 — Replaced explicit unused-scope deletion with ordinary unused parameters in `75c8ba4`. Renaming those
  parameters was rejected because it breaks the keyword-compatible protected hook signature under `ty`.
- 18:43–18:50 UTC+8 — Prototyped and independently reviewed a `GPTDecoderStack`, then discarded it before commit.
  An indexed factory and inherited `ModuleList` mutation make a homogeneous/folded-cost promise unsound; an immutable
  special container would add more machinery than the model needs. Retained explicit `ModuleList` construction and
  model-authored representative-layer folding.
- 18:51 UTC+8 — Renamed the nested attention update to `PrenormRoPEAttention` in `580e7c3`. Unlike a general reusable
  container, this course decoder constructor does always assemble pre-norm causal self-attention with RoPE.
- 18:53–19:03 UTC+8 — Extended the exact-overload cost policy in `a592fd3` to Torch's registered dense-product family:
  `mm.default`, `addmm.default`, and `baddbmm.default` alongside the existing `bmm.default`. Meta execution and
  `FlopCounterMode` are the oracle, including zero `alpha`/`beta`; review added conservative addend-broadcast checks.
  Rank-sensitive `matmul`, `mv`, `addbmm`, scaled, and output overloads remain visibly unsupported.
- 19:05–19:15 UTC+8 — Added `module_state_footprint()` and `ModuleStateFootprint` in `7deec80`. The report uses Torch's
  registered-state traversal and works on meta tensors, but deliberately reports logical per-category `numel` and
  dtype-sized bytes rather than physical storage. Tests cover mixed dtypes, tied parameters, shared-storage views,
  cross-category registration, nonpersistent buffers, and uninitialized lazy state.
- Assignment-boundary check: forward Transformer accounting requires parameter load memory and matmul FLOPs. The later
  AdamW problem asks for retained training activations and peak memory. Do not mislabel operator-output volume as
  activation peak; training/backward lifetime analysis remains deferred.
- 19:17 UTC+8 — Added the thin `Module.state_footprint()` façade in `afc8bc7`; arbitrary Torch modules continue using
  `analytics.module_state_footprint()`, and the traversal remains implemented only once.
- 19:19–19:25 UTC+8 — Completed the existing local/display/identity/binding symbol record in `84829e1` with explicit
  `scope.symbols.display(...)`. Display names are unique within one scope, may repeat in distant scopes, and do not
  change parent-child argument matching or internal SymPy identity.
- 19:28 UTC+8 — Added `CostReport.bound_terms` in `709eb52`, preserving authored symbolic component terms while making
  the same report's architectural and invocation bindings available term by term.
- 19:31–19:34 UTC+8 — Broad regression: 278 passed and 6 skipped. Three failures are the deliberately disabled course
  SwiGLU warning from `cb9dc06`; two are untouched, unimplemented later-assignment adapters. Added the missing symbolic
  operation-repetition and zero-inner-dimension policy regression in `f32bf38`.
- 19:35–19:36 UTC+8 — Verified that Torch's `flop_registry` and shape-wrapped formulas are private runtime machinery,
  not a stable symbolic API. Recorded the reuse boundary in `4785b13`: exact ATen identities and meta/FlopCounterMode
  remain shared, while the small symbolic formula policy stays local.
- 19:38 UTC+8 — Added `b18f59e`, an end-to-end meta-`TransformerLM` check against an independent parameter formula and
  per-layer RoPE buffer formula. This guards untied embedding/output weights, block multiplicity, dtype-sized parameter
  bytes, and float32 nonpersistent rotation buffers.
- 19:46–19:48 UTC+8 — Tightened symbolic dimension normalization to reject Python/SymPy booleans, finite non-integral
  numbers, NaN, and infinity while retaining unresolved symbolic expressions. The focused analytics suite has 55 passing
  tests; Ruff and `ty` are clean for the changed files.
- 19:49–19:54 UTC+8 — Audited directed cost-tree traversal. Structural Torch containers now retain every registered
  invocation even when slots share a module identity; active-ancestor cycles are rejected without rejecting sibling DAG
  reuse; child paths are unique and unambiguous. Review caught and corrected loss of authored `Sequential` slot names by
  using public `named_modules(remove_duplicate=False)`. Focused analytics: 59 passed; Ruff and `ty` clean.
- 19:55–19:57 UTC+8 — Restricted caller substitutions to the report's scoped symbol identities. Expressions may relate
  any symbols already declared in the tree, but cannot introduce a foreign identity that later report calls cannot name
  or bind. Focused analytics: 60 passed; Ruff and `ty` clean.
- 19:58–20:02 UTC+8 — Recursively copied and froze standard list/tuple, set, and mapping containers in symbolic metadata;
  validated `TensorRepr.dtype`. Review confirmed tuple-normalizing ATen list arguments is appropriate because the IR is
  descriptive and never dispatched directly. Real tensor leaves remain a separate schema-validation concern. Focused
  analytics: 61 passed; Ruff and `ty` clean.
- 20:03–20:07 UTC+8 — Closed the live-tensor gap using the exact ATen schema already carried by `CostRepr`: `Tensor`,
  optional Tensor, and Tensor-list operands require `TensorRepr`, and live tensors are rejected anywhere in nested
  symbolic metadata rather than cloned. Focused analytics: 62 passed; Ruff and `ty` clean.
- Wider student regression after the robustness commits: 281 passed and 6 skipped. The only three failures remain the
  course-key SwiGLU warning assertions; translation works, while `cb9dc06` deliberately disabled the warning to avoid a
  submission flag. No analytics, attention, adapter-device, benchmark, tokenizer, or typing regression appeared.
- API audit found that recursive `named_modules()` could itself recurse through a cyclic structural container before the
  collector's guard ran. Direct registered-slot traversal now puts every edge through that guard. Only exact official
  `ModuleList`, `ModuleDict`, and `Sequential` types receive known-zero-local-work classification; subclasses without a
  provider remain visibly unsupported because they may override `forward`. Focused analytics: 64 passed; Ruff/`ty` clean.
- Enforced lexical symbol scope for authored analytics. Local bindings and costs may use only identities declared by that
  module; child arguments and repetitions may use only identities declared by their parent scope. Parent-to-child
  identity transfer remains explicit through named arguments. Focused analytics: 68 passed; Ruff/`ty` clean.
- Added post-binding domain validation for every declared dimension, tensor axis, local/parent dimension expression, and
  local/tree repetition. Expressions may remain unresolved, but a later substitution that makes one definitely negative
  or non-integral is rejected, including across incremental `CostReport.substitute()` calls. Independent review added
  child-local bindings shadowed by parent arguments. Focused analytics: 74 passed; Ruff/`ty` clean.
- Completed symbolic tensor compatibility checks for composed ATen schema types. Optional Tensor lists recurse correctly,
  and `TensorRepr` cannot be placed inside scalar or non-tensor-list slots. This remains deliberately narrower than full
  dispatcher argument validation. Focused analytics: 76 passed; Ruff/`ty` clean.
- Hardened exported record boundaries: names and member kinds are validated at construction, report conditions must be
  SymPy equalities, iterable fields are normalized without consuming generators twice, and `matmul_flops()` rejects a
  non-`CostTree` argument directly. Focused analytics: 77 passed; Ruff/`ty` clean.
- Enforced `TransformerLM`'s authored layer-folding preconditions. Count drift is rejected, and an explicit decoder-layer
  fold key covers delegated module types, query/KV cost configuration, RoPE execution form, feed-forward configuration,
  and cost-owner parameter shape/dtype without comparing values or cost-irrelevant RoPE capacity. Two review rounds
  supplied same-state-shape grouped-attention and delegated-norm counterexamples before approval. Model-focused suite:
  102 passed; Ruff/`ty` clean.
- Formula audit identified einx strength reduction for unit contraction axes. Preserved the architectural/course-level
  `bmm` record and documented why concrete meta `FlopCounterMode` totals diverge for those degenerate eager programs;
  added pinned Linear and packed-SwiGLU examples. Focused analytics: 83 passed; Ruff/`ty` clean.
- Qualified the memo's `SymInt` boundary: concrete/backed values can normalize now, while genuine `ShapeEnv` identities
  require explicit scope remapping and remain deferred with dynamic invocation observation.
- Aligned the reusable `Module` default recursion with its documented one-invocation-per-registered-slot policy. Direct
  child slots preserve repeated aliases; custom call semantics still override `_cost_children()`. Focused analytics and
  model-architecture suites: 91 passed; Ruff/`ty` clean.
- Broadened the nondegenerate attention oracle across MHA/GQA/MQA, unequal QK/value widths, both retained private layout
  strategies, and two leading batch axes represented by their product. Focused analytics: 90 passed; Ruff/`ty` clean.
