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
- Distinguished execution containers from registration-only containers. Exact `Sequential` supplies a known chained
  execution; `ModuleList` and `ModuleDict` expose inventory slots but remain unresolved under strict cost reporting until
  an authored parent directs invocation. Focused analytics: 92 passed; Ruff/`ty` clean.
- Added caller-directed validation at the protected structural-provider boundary. Malformed `_cost_repr()` or
  `_cost_children()` results now fail before the collector dereferences implementation details.
- Corrected `addmm`/`baddbmm` addend validation to follow Torch's directional expansion contract. A source addend may
  expand to the fixed product shape, but the product is not symmetrically broadcast with it; CPU Torch oracles cover the
  cases where meta dispatch is overly permissive.
- Hardened the exported analytics records as construction boundaries. Symbol identities and expressions are normalized
  eagerly, reports validate known binding identities, and directly assembled trees preserve local symbol and child-path
  uniqueness rather than failing later inside aggregation.
- Began invocation observation as a private lifecycle experiment. One official per-module forward hook snapshots and
  binds only root-local scalar facts for each completed call; reports remain readable after exit, repeated calls stay
  separate, hook failures do not alter a successful forward, and tensors are not retained. Public naming/export remains
  deferred until the lifecycle contract survives review.
- Exposed the reviewed root-session surface as `CostObserver`, `observe_costs(module)`, and the repository `Module`
  convenience method. Concrete classes still opt into call bindings through the protected local-name mapping hook.
- Added the first concrete invocation binding at the model interface. `TransformerLM` maps all leading token-ID axes to
  flattened `batch` and the final axis to `sequence`; positional and keyword meta forwards produce separate fully bound
  reports whose combined total matches Torch's observed FLOP counter.
- Extended the same interface-local binding to direct `Linear`, all three SwiGLU representations, self-attention, and
  decoder-layer roots. Multi-axis meta calls for each now resolve their authored symbols and agree with Torch's FLOP
  counter without recursive runtime tracing.
- Reassessed activation-memory terminology against Torch's profiler, CUDA allocator peaks, meta/fake tensors, and
  autograd saved-tensor hooks. The memo now keeps registered state, logical tensor footprints, operator allocation
  traffic, training retention, and concrete allocator peaks distinct; no output-volume proxy is labeled as peak memory.
- Broad student regression after observation work: 333 passed and 6 skipped. The only three failures remain the
  deliberately disarmed SwiGLU course-key warnings from `cb9dc06`; translation, loading, analytics, model, attention,
  adapter, and typing coverage otherwise passed.
- Independent final audit found no concrete analytics/observation gaps. Additional probes confirmed forward and
  gradient transparency, complete hook removal, unchanged native state-dict keys, and `load_state_dict(assign=True)`
  behavior during an active session. Documented non-thread-safety and deferred recursive/SymInt work remain explicit.
- A follow-up public-API and memo audit found no code defect, but corrected five documentation boundaries: symbolic
  `SymInt` remapping, root binding versus operator observation, exact Torch overload terminology, deferred observation
  failures, and registered buffers versus serialized checkpoint state.
- A registration-only container audit found that non-strict reports still summed every registered slot once. Directed
  cost edges now distinguish actual calls from structural inventory; inventory remains inspectable but contributes no
  execution terms until a model author declares the calls.
- Tightened the new edge-role boundary: inventory cannot carry symbolic call arguments or repetitions, preventing an
  extension provider or manually constructed public tree from smuggling invocation semantics into structural slots.
- Added one non-aggregating tensor-footprint primitive: symbolic logical element count and optional dtype-sized bytes.
  Its API and memo explicitly avoid activation-lifetime or allocator-peak claims.
- Hardened the public concrete state-footprint record against negative, non-integral, and impossible byte/subset totals;
  normal module traversal continues to produce the same values.
- A Torch export probe confirmed that backed dynamic `SymInt` values still sympify to foreign `ShapeEnv` symbols while
  concrete hints and range constraints remain separate. The memo records the evidence; remapping remains deferred.
- Aligned public analytics record boundaries so whitespace-only names, paths, module types, and diagnostic messages no
  longer satisfy their documented non-empty contracts.
- Refined inventory semantics so ignored structural subtrees no longer leak symbols, bindings, or domain constraints
  into execution reports; full-tree validation still checks them when the static representation is collected.
- A concrete CUDA probe separated logical model/output bytes from allocator observations. Repeated forwards retained a
  stable 8,519,680-byte allocation; Torch's private diagnostic clear for cuBLAS workspaces returned allocated memory
  exactly to the pre-forward level. This confirms why allocator peaks and persistent library workspaces must not be
  reported as activation footprints; no private CUDA API enters the package implementation.
- Final broad student regression: 346 passed and 6 skipped. The only three failures remain the deliberately disarmed
  SwiGLU course-key warning assertions; all analytics, model, attention, loading, adapter, and typing coverage passed.
- Checked the homogeneous-layer container idea against Torch. Python's `TransformerEncoder` and `TransformerDecoder`
  deep-copy one prototype through a private `_get_clones()` helper into `ModuleList`, while the public generic choices
  remain `Sequential` and `ModuleList`. That same-initial-state policy differs from this model's independently
  constructed layers, so no generic cloned-stack abstraction was added during the audit.
- Independent review found that dense-product policy checks compared scoped symbols before applying their definitions.
  Matrix policies now consume tree and caller facts first, retain unresolved batch/contraction equalities as report
  conditions, and compose through parent-shadowed child bindings. Direct aliases, caller substitutions, contradictory
  shapes, and condition deduplication are covered; unresolved broadcast disjunction remains explicitly deferred.
- The same final review found that `DeltaLayer` preserved Torch state but erased Python runtime signature metadata.
  Selective wrapper metadata now restores the authored signature, annotations, and qualified name while an explicit
  additive-forward docstring avoids presenting the update-only documentation as the exposed residual behavior.
- Post-fix broad student regression: 350 passed and 6 skipped, with only the same three intentionally disarmed SwiGLU
  warning assertions failing. Independent review reported no remaining concrete defect in analytics composition,
  state loading, model folding, tensor footprints, or DeltaLayer behavior.
- A deterministic randomized meta probe varied Transformer layer count (including zero), vocabulary, width, head count,
  feed-forward width, batch, and sequence across twelve cases. Every authored matrix total exactly matched Torch's
  `FlopCounterMode`; the temporary probe was removed rather than retained as a randomized student test.
- A second deterministic randomized meta probe covered sixteen MHA/GQA/MQA cases across unequal QK/value widths, both
  retained activation layouts, optional RoPE, and one or two leading batch axes. Every authored matrix total again
  matched `FlopCounterMode`; the temporary probe was removed.
- Reopened the memo's PyTorch, Torch source, einx, jaxtyping, and Netlib references. The source paths and documentation
  targets remain live; version-selector redirects are expected. No task framing or implementation log was added to the
  memo during the link audit.
- An independent final API audit found no defect in the `Module`/`DeltaLayer` ownership, root façade, decorator
  transparency, nested decoder naming, or constructor boundaries. It also confirmed that Torch's nearest homogeneous
  Transformer container deep-copies one prototype with identical initial values, unlike this model's independently
  initialized layers, so no speculative generic stack container was introduced.
- Re-ran the veteran `scripts/transformer_accounting.py` unchanged, then independently queried the new symbolic API on
  its GPT-2 XL meta model. Both paths report 3,516,769,894,400 forward matrix FLOPs; the new state-footprint result also
  matches the script's 1,640,452,800 parameters, 6,561,811,200 parameter bytes, and 12,582,912 buffer bytes exactly.
