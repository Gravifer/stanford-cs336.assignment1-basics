# Symbolic Model Analytics in PyTorch

## Scope

This note records how PyTorch represents and observes computation, which parts can support a reusable symbolic
description, and which semantic information still has to come from the model author. It is a research memo and a set of
refined minutes. Names and interfaces mentioned here are not stability promises.

## Reference map

The most useful introductions to the machinery discussed below are:

- [How Does the Dispatcher Work?](https://docs.pytorch.org/devlogs/dispatcher/2026-04-16-how-does-the-dispatcher-work/)
  develops the current dispatcher model from first principles.
- [PyTorch internals](https://blog.ezyang.com/2019/05/pytorch-internals/) gives the broader conceptual map of
  tensors, storage, autograd, ATen, and generated bindings. Some implementation details are historical.
- [Let's talk about the PyTorch dispatcher](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)
  explains dispatch keys, registration, boxing, and redispatch in greater depth.
- [Extending PyTorch](https://docs.pytorch.org/docs/stable/notes/extending.html) introduces `__torch_function__`,
  `__torch_dispatch__`, modes, and the boundaries among the extension mechanisms.
- [`torch.library`](https://docs.pytorch.org/docs/stable/library.html) and the
  [custom-operator guide](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html) cover the public
  operator-registration system, including schemas, fake implementations, autograd, transforms, and `opcheck`.
- The [meta-device guide](https://docs.pytorch.org/docs/stable/meta.html),
  [`ShapeEnv` reference](https://docs.pytorch.org/docs/main/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html),
  and [`torch.export` programming model](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export/programming_model.html)
  cover metadata-only execution and Torch's symbolic-shape machinery.
- The [`torch.export` overview](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export.html),
  [API reference](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export/api_reference.html),
  [Export IR specification](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/ir_spec.html), and
  [Core ATen and Prims IR catalog](https://docs.pytorch.org/docs/stable/torch.compiler_ir.html) describe successively
  more explicit captured representations and decomposition levels.
- The [`FlopCounterMode` source](https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py) is the closest
  existing analogue to the observer and formula-policy split used here.

These sources do not all describe the same PyTorch release. The current official documentation should govern public API
usage; the older internals essays remain valuable for their conceptual explanations.

## ATen operators and their schemas

ATen is the operator layer reached after Python conveniences such as tensor methods have entered the PyTorch dispatcher.
An object such as `torch.ops.aten.bmm.default` identifies one overload; its schema describes formal arguments, return
values, defaults, aliasing, and mutation. The schema language is data interpreted by PyTorch, not C++ syntax.

For example, an annotation such as `Tensor(a!)` says that the tensor belongs to alias set `a` and is written. The runtime
path is implemented by the C++ [`SchemaParser`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/frontend/function_schema_parser.cpp),
which delegates type and alias annotations to `SchemaTypeParser`. `torch.library.Library.define()` passes operator-schema
strings into this machinery; its Python entry point lives in
[`torch/library.py`](https://github.com/pytorch/pytorch/blob/main/torch/library.py), while the supported interface is
documented under [`torch.library`](https://docs.pytorch.org/docs/stable/library.html). A private Python binding is useful
for inspection:

```python
schema = torch._C.parse_schema(
    "demo::add_(Tensor(a!) self, Tensor other) -> Tensor(a!)"
)
assert schema.arguments[0].alias_info.is_write
```

PyTorch also has a separate build-time parser in
[`torchgen/model.py`](https://github.com/pytorch/pytorch/blob/main/torchgen/model.py). It turns `native_functions.yaml`
entries into immutable, semantic dataclasses used by code generation. Torchgen's preference for immutable records and
lossless round trips is instructive, but Torchgen is not a runtime API on which this package should depend.

Python's exact overload objects do not currently provide a comparably public schema accessor. The initial record
validation reads `OpOverload._schema`, a narrow version-sensitive dependency kept behind the repository's pinned Torch
version and direct schema-validation tests; it should not be mistaken for a settled third-party extension contract.

An operator schema does not contain an analytical cost formula. Nor does it preserve a human reason such as "attention
packed QKV projection." The useful division is consequently:

```text
ATen schema            operator identity and argument structure
symbolic operands      shapes, dtype, and architectural expressions
cost policy            one interpretation of those operands
semantic name          why this operation exists in the model
```

Using exact Torch operator identities--normally ATen overloads for built-in tensor work--avoids creating a second,
weaker operation namespace. A semantic name remains separate because one logical operation may lower differently across
implementations, and several semantic operations may share the same operator overload. Exact custom Torch overloads can
remain visible too, even when the initial policies do not know how to interpret them.

## Dispatch and Torch's FLOP counter

`TorchDispatchMode` observes calls after Python tensor syntax has been normalized into dispatcher operations. PyTorch's
[extension notes](https://docs.pytorch.org/docs/stable/notes/extending.html#extending-torch-native-api) explain that, at
this level, `torch.add(a, 2)` and `a + 2` arrive as the same ATen call. The current
[dispatcher walkthrough](https://docs.pytorch.org/devlogs/dispatcher/2026-04-16-how-does-the-dispatcher-work/) explains
how per-tensor keys, thread-local included and excluded sets, and operator dispatch tables select an implementation.

[`FlopCounterMode`](https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py) is a particularly close
precedent. It is a context manager backed by `TorchDispatchMode`. Its registry is keyed by ATen operator packets, while
formula wrappers replace tensor arguments with their shapes. The formulas for `mm` and `bmm` therefore receive operand
shapes rather than model-specific `m`, `n`, and `p` supplied by each caller.

The public surface of that module exports `FlopCounterMode` and `register_flop_formula`, not the module-level
`flop_registry`. A registered formula is also wrapped to extract shapes from actual tensor arguments; it does not accept
`TensorRepr` records carrying arbitrary SymPy dimensions. Reaching through the wrapper's `__wrapped__` attribute would
depend on private implementation structure. Static symbolic policies consequently mirror the small supported formula
set under exact ATen overload identities, while meta execution under `FlopCounterMode` remains the executable oracle.

The counter uses [`ModuleTracker`](https://github.com/pytorch/pytorch/blob/main/torch/utils/module_tracker.py) to attribute
observed operations to fully qualified module names. The
[`ModuleTracker` documentation](https://docs.pytorch.org/docs/stable/module_tracker.html) explicitly describes this use.
Root invocation binding already uses an ordinary official forward hook; any later recursive observation should likewise
prefer Torch's tracking machinery. This is not a reason to install permanent analytics hooks in every module
constructor.

Torch's registry is intentionally incomplete. When an unregistered operation has a decomposition, the counter may enter
that decomposition; an operation without a registered formula or usable decomposition contributes no recorded FLOPs.
That behavior is suitable for a counter but insufficient for a report that claims analytical coverage. Our terminology
must distinguish:

- **supported:** a selected policy supplies a formula;
- **known zero-cost:** the policy deliberately assigns no arithmetic work, as it might for a view;
- **unsupported:** the operation or module-local computation is visible, but the policy cannot estimate it.

Unsupported work must remain visible. A partial total is not a complete total merely because missing terms were assigned
an implicit zero.

## `mm`, `bmm`, and related vocabularies

PyTorch operator names and traditional BLAS routine names overlap conceptually but are not one namespace.
PyTorch groups the relevant public functions under its
[BLAS and LAPACK operations](https://docs.pytorch.org/docs/stable/torch.html#blas-and-lapack-operations) index, but their
Python and ATen names need not be classic BLAS routine names.

| PyTorch/ATen name | Meaning |
|---|---|
| `mm` | two-dimensional matrix-matrix multiplication |
| `mv` | matrix-vector multiplication |
| `bmm` | three-dimensional batched matrix-matrix multiplication |
| `matmul` | rank-sensitive matrix product with broadcasting over leading axes |
| `addmm` | a matrix product plus a scaled addend |
| `addmv` | a matrix-vector product plus a scaled addend |
| `baddbmm` | independent batched matrix products plus batched addends |
| `addbmm` | batched matrix products reduced into one added matrix |
| `addr` | a matrix plus a rank-one outer-product update |
| `linear` | an affine projection, conventionally `x @ weight.T + bias` |
| `einsum` | a general Einstein-notation tensor contraction |

`bmm` has exactly one leading batch dimension:

```text
(batch, m, n) @ (batch, n, p) -> (batch, m, p)
```

It does not know that a flattened dimension means batch times head, beam, or group. If an implementation lowers attention
to operands shaped `(B * H, S_q, D_k)` and `(B * H, D_k, S_k)`, then `B * H` is already part of that operation and must not
also be placed in an external repetition count. A different lowering may preserve, flatten, or repeatedly invoke those
logical axes differently. The general rule is that repetition must not duplicate multiplicity already represented in
the recorded operands.

Traditional BLAS divides routines into vector-vector, matrix-vector, and matrix-matrix levels. Common names include
`AXPY` for a scaled vector addition, `GEMV` for a general matrix-vector product, `GER` for a rank-one update, and `GEMM`
for a general matrix-matrix product. The authoritative
[Netlib BLAS quick reference](https://www.netlib.org/lapack/lug/node145.html) defines `GEMM` as
$C \leftarrow \alpha\,op(A)op(B)+\beta C$.

LAPACK builds higher-level factorizations and solvers over BLAS. Representative names include `GETRF` for LU,
`POTRF` for Cholesky, `GEQRF` for QR, `GESV` for a linear solve through LU, and `GESVD` for singular-value decomposition.
An ATen `bmm` may eventually use a vendor batched-GEMM kernel, but it remains a backend-neutral PyTorch operator rather
than a classic BLAS routine name.

## Symbolic shapes, meta tensors, and fake tensors

A [`SymInt`](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/ir_spec.html#symint) behaves like an
integer whose value may instead be represented symbolically. Internally it participates in Torch's `SymNode` and
[`ShapeEnv`](https://docs.pytorch.org/docs/main/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html) machinery,
which tracks hints, constraints, guards, and symbol provenance. This is substantially more than a Python alias for
`int | sympy.Expr`.

That machinery is appropriate for dimensions discovered while tracing tensor programs. It is less natural as the
canonical algebra for architectural quantities such as an unbound layer count. The authored representation therefore
uses SymPy expressions directly. Ordinary integers and SymPy-compatible constant expressions can be normalized at its
boundary. A symbolic `ShapeEnv` identity, whether its `SymInt` is backed or unbacked, must instead be remapped into the
receiving analytics scope or deliberately specialized to its hint. The current boundary rejects such foreign symbolic
identities. Concrete classes may import SymPy lazily inside their own analytics hooks when their symbolic description
needs it.

An exported dynamic dimension makes the distinction observable. Its `SymInt` may have a concrete node hint from the
example input while its node expression is a separate SymPy symbol; `sympy.sympify()` returns that symbol, not the hint.
Range constraints likewise remain separately attached to the exported program. Importing the expression without its
scope and constraints would therefore lose provenance even for a backed value. The
[`torch.export` symbolic-shape model](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export/programming_model.html#basics-of-symbolic-shapes)
describes how fake implementations propagate expressions such as sums of input dimensions and how `ShapeEnv` records
guards over them.

The [`meta` device](https://docs.pytorch.org/docs/stable/meta.html) stores tensor metadata without allocating tensor data.
Most operations can produce meta outputs with the shapes, strides, and dtypes that real execution would have produced,
but no numerical result exists and data-dependent operations such as `item()` cannot succeed. This makes a GPT-scale
forward inexpensive enough to serve as an executable structural oracle.

[`FakeTensor`](https://docs.pytorch.org/docs/2.9/torch.compiler_fake_tensor.html) also carries no tensor data, but it models
the device and aliasing behavior of a real tensor and is coupled to a `ShapeEnv`. Fake tensors are therefore more capable
for symbolic tracing, though not automatically cheaper than meta tensors at small sizes because their Python machinery
and decompositions have overhead.

Neither representation supplies the human semantics of an architectural formula. A meta forward answers what one
configured model executed. It does not by itself reconstruct the family expression containing symbols such as
`num_layers`.

## Memory quantities are not interchangeable

"Activation memory" is ambiguous unless the report names both the lifetime model and the execution mode. At least five
useful quantities differ:

| Quantity | What it measures | Suitable Torch evidence |
|---|---|---|
| registered state | logical parameter and buffer elements or dtype-sized bytes | module parameter/buffer traversal |
| one tensor footprint | `numel * element_size` for a known logical tensor | real, fake, or meta tensor metadata |
| operator allocation traffic | bytes allocated or released by individual executed operators | [`torch.profiler`](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) with `profile_memory=True` |
| saved-for-backward tensors | tensors autograd explicitly packs for a later backward | [`saved_tensors_hooks`](https://docs.pytorch.org/docs/stable/notes/autograd.html#hooks-for-saved-tensors) |
| allocator peak | maximum bytes reported as occupied by PyTorch's allocator on one concrete device run | [`max_memory_allocated`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html) after resetting peak statistics |

Here registered state means all tensors registered as parameters or buffers. It includes nonpersistent buffers that
`state_dict()` deliberately omits, so it is not synonymous with serialized checkpoint state.

For one symbolic operand, `TensorRepr.numel` is the product of its logical shape and
`TensorRepr.logical_nbytes` additionally multiplies by the known dtype size. The latter is `None` when dtype is unknown.
These intrinsic values deliberately do not aggregate operands or claim that a tensor is materialized, retained,
nonaliasing, or simultaneously live with any other tensor.

The sum of operator output sizes is not generally any of these. An output may alias an input, be a view, be released
before a later output exists, remain live because the caller retains it, or be saved specifically for backward. The CUDA
allocator may also round requests, cache freed blocks, and report allocated versus reserved bytes separately. The
profiler records allocation and release events, then its tables attribute or aggregate their memory effects across
executed operators; those tables are not by themselves an architectural peak formula.

Allocator observations can also include library workspaces rather than model tensors. In particular, Torch's
[CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#cublas-workspaces) explain that cuBLAS workspaces
are allocated per handle-and-stream combination and retained for reuse. A stable allocation left after an output is
released is therefore not sufficient evidence of a retained activation or a leak.

Meta and fake tensors can supply logical shapes and dtypes, so their tensor footprints remain meaningful. They allocate
no ordinary tensor storage and therefore cannot validate an allocator peak. Conversely, a concrete CUDA peak is useful
systems evidence for one implementation, dtype, device, allocator configuration, and call shape, but it does not recover
a symbolic family expression.

Training retention is a separate plane again. Autograd saved-tensor hooks can inventory what a concrete forward saves
for backward, but installing them can change Tensor-object packing behavior, and saved objects may share storage. They
are an executable oracle for a later training analysis rather than a substitute for a module-authored retention formula.

A symbolic peak requires an execution schedule, alias information, and liveness intervals. FX/export metadata may
eventually provide an observed graph on which to perform that analysis. Until then, reports should keep registered state,
logical tensor footprints, allocation traffic, saved-for-backward bytes, and allocator peaks as separately named results.

## Module structure, FX, and export

Torch's module tree carries parameters, buffers, state-dict prefixes, and a useful semantic hierarchy. Official forward
hooks and `ModuleTracker` can associate a successful call with that hierarchy. Torch has no standard hook asking a module
for an authored symbolic cost description, so a small protected provider remains necessary. Official hooks should still
transport future invocation information; the provider need not reinvent execution interception.

The module-specific [`register_forward_pre_hook()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook)
and [`register_forward_hook()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)
methods return removable handles, making a temporary context-managed observer possible without permanent instrumentation.
These hooks participate in `module(...)`; callers that invoke `module.forward(...)` directly bypass `Module.__call__`
and therefore the hook machinery. [`ModuleTracker`](https://docs.pytorch.org/docs/stable/module_tracker.html) is the
official context manager for associating execution with the active module hierarchy and is already used by
`FlopCounterMode`.

Container registration is not always execution. `Sequential` defines a chained forward and can be interpreted directly;
`ModuleList` and `ModuleDict` only register slots. Their children may be displayed as an inventory, but a strict static
cost report remains unresolved until an authored parent states which slots are invoked and with what repetition.
Inventory edges remain inspectable in the cost tree but do not contribute terms to an execution report. The same
conservative default applies to registered children of the repository's base `Module`: a concrete delegating module must
use `scope.child(...)` to identify actual calls.

[`torch.fx`](https://docs.pytorch.org/docs/stable/fx.html) represents executable dataflow as a graph of calls and values.
[`torch.export`](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export.html) captures an ahead-of-time
`ExportedProgram` backed by FX. It lifts parameters and buffers into a graph signature, records shape constraints, and
normalizes tensor computation into ATen and custom operators. The
[`export()` API](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export/api_reference.html#torch.export.export)
accepts example inputs and a `dynamic_shapes` specification and returns the program together with its captured state and
constraints. Export can retain the default training-oriented ATen IR, functionalize it, or decompose it into the smaller
Core ATen operator set; the
[Export IR specification](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/ir_spec.html) describes the
resulting graph and state. The separate
[Core ATen and Prims IR catalog](https://docs.pytorch.org/docs/stable/torch.compiler_ir.html) lists their schemas and
explains that Prims makes type promotion and broadcasting more explicit than Core ATen.

Export is a valuable observed frontend and lowering inspector. It also specializes parameter shapes and concrete Python
structure: a `ModuleList` length is normally unrolled, and semantic labels such as "packed QKV projection" are not the
canonical graph identity. FX is consequently not the canonical container for authored hierarchy, scoped bindings, and
symbolic repetition. A lightweight cost tree may generate or annotate FX later if a concrete transformation needs it.

## Relationship to jaxtyping and einx

The representation deliberately reuses the repository's axis vocabulary without parsing the surface language of either
library.

[`jaxtyping`](https://docs.kidger.site/jaxtyping/) supplies dtype and shape contracts. Its
[runtime contexts](https://docs.kidger.site/jaxtyping/api/runtime-type-checking/) bind axis names locally to a checked
function call, which is useful inspiration for scoped analytics symbols. Its annotations are contracts, however, not
architectural expressions with repetition, substitution, and alternative cost policies.

`einx` describes tensor transformations and solves relationships needed to issue backend operations. Its
[JIT documentation](https://einx.readthedocs.io/en/latest/more/jit.html) notes that generated code can be inspected to
verify which backend calls were made. That is useful evidence for a particular lowering. The analytics plane instead
records symbolic operands and why a contraction exists; it should not make an einx expression string into a second API.

The three layers can share names such as `batch`, `head`, `sequence`, and `d_model` while remaining separately useful:

```text
jaxtyping       interface contract
einx            tensor transformation and backend lowering
analytics       architectural algebra, repetition, and interpretation
```

## Refined working model

The current working separation is:

```text
module declaration      symbolic operation templates
module instance         architectural bindings such as d_model = 1600
forward observation     invocation bindings such as batch = 1 and sequence = 1024
report policy           selected substitutions and cost interpretation
```

One expression may consequently be shown as

$$
2 L B H S^2 d_h,
$$

partially bound as

$$
2 \cdot 48 \cdot B \cdot 25 \cdot S^2 \cdot 64,
$$

and finally evaluated for a particular invocation. The static declaration remains useful without running the model.
Root observation can bind invocation dimensions; meta execution under `FlopCounterMode`, or a later recursive operator
observer, separately checks the selected ATen lowering.

Symbols are local to a module description. Two distant modules may both display `d_model` without their variables being
identical. Internally, identity-distinct SymPy symbols prevent accidental capture, while immediate parent-child
arguments establish the identities that the model author intends.

Matrix-product policies use that same constraint plane. Known architectural facts and caller substitutions are applied
before checking batch and contraction dimensions; an equality that remains symbolic is retained as a report condition
rather than rejected. Directional broadcasting is different: an addend axis is valid when it equals either one or the
product axis. Until report conditions represent that disjunction, `addmm` and `baddbmm` addend expansion is accepted
only when the existing facts prove it directly.

During a module's protected cost hook, `scope.symbols` is a focused mutable view of that module's symbol table.
`unbound()` introduces invocation dimensions, `bind()` adds instance or architectural definitions, `display()` may assign
a distinct human-facing label, and attribute or bracket access retrieves the corresponding identity-distinct SymPy
symbols. Collection then freezes that builder into immutable records carrying a local name, a display name, the unique
symbolic identity, and an optional local binding; the mutable view does not escape into the resulting cost tree.

A parent's `scope.child(..., arguments=...)` mapping declares a call edge and supplies expressions for the child's formal
local symbols. These arguments do not overwrite definitions contributed by the child instance. Instead, both facts are
retained as an equality: a definitely false equality is rejected, while one that still contains unbound symbols appears
in the cost report as a condition. Caller substitutions are additional facts under the same rule rather than mutable
overrides. Registration-only inventory uses a distinct edge role, preserving structure without assigning one fictional
invocation to every slot.

Call-specific facts can be supplied by a root observation session without changing the authored tree:

```python
with model.observe_costs() as observation:
    logits = model(token_ids)

report, = observation.matmul_flops(strict=True)
```

The session snapshots the static tree on entry and uses an ordinary per-module Torch forward hook with keyword support.
After each root forward reaches the hook, the concrete class maps its interface arguments to its own local symbolic
names. `TransformerLM`, for example, binds `sequence` from the final token-ID axis and binds `batch` to the product of
all preceding axes. The generic observer then converts those names to the root's identity-distinct symbols and applies
the same report policy used for explicit substitutions. Each report consequently retains its symbolic total alongside
the total bound for that call.

Reports correspond only to calls whose bindings were collected successfully. A binding error never turns an otherwise
valid forward into a failure: the observer records the error, excludes that call from `call_count`, and raises the
deferred failure when report generation is requested.

Only normalized scalar facts are retained; argument and output tensors are not. Multiple root calls remain separate in
forward-hook completion order. The session is single-use and not thread-safe, so forwards must not overlap context exit
or report generation. It is deliberately a root-binding facility, not yet a recursive runtime trace: static authored
parent-child arguments continue to propagate the root facts through a Transformer model.

The model author also owns recursion policy. Ordinary modules contribute only their local work; modules that actually
delegate computation author call edges to the relevant children. A repeated Transformer stack may deliberately describe
one representative block with symbolic `num_layers` repetition and avoid traversing the remaining concrete blocks;
embeddings, final normalization, and logit emission are still traversed normally. Such authored folding must validate
that the concrete layer count and cost-driving configuration still match the representative; independently initialized
parameter values and cost-irrelevant buffer capacity need not match. Automatically discovering and folding repeated
siblings is a separate presentation problem.

The first representation is matmul-focused. Concrete parameter and buffer counts already come from Torch's module
introspection. A unified resource record should not be introduced until at least an ordinary parameter, the RoPE
trigonometric buffer, a temporary activation, and a retained activation have made the differences concrete.

## Lowering levels must not be added together

Several useful descriptions coexist:

1. semantic model operations, such as a packed QKV projection;
2. eager ATen calls emitted by the current implementation;
3. functional ATen after mutation and alias removal;
4. Core ATen after broader decomposition;
5. fused or backend-specific kernels.

A report must name the level it interprets. Counting one semantic operation and then also counting all operations into
which it decomposes is double counting. The authored plane may keep a human name and an expected eager-ATen anchor;
`FlopCounterMode` or a future operator-level observer determines whether that expectation still matches the
implementation. Root `CostObserver` sessions bind call dimensions but do not inspect operator lowering.

An expected eager anchor is not a promise that every concrete shape reaches that operator. In particular, einx may
strength-reduce a contraction whose reduced axis has size one into elementwise multiplication. The authored record still
describes the logical matrix product and the course formula still assigns it $2mnp$ work; a concrete
`FlopCounterMode` run then reports the strength-reduced eager program. Exact oracle-equality tests therefore use
nondegenerate contraction axes, while degenerate tests preserve and explain the distinction rather than conditionally
changing the architectural formula.

## Deliberately deferred questions

The following are not prerequisites for the current authored and root-observed work:

- recursively observing and reconciling submodule invocations;
- importing and remapping symbolic `ShapeEnv`/`SymInt` identities during observation;
- training and backward-pass costs;
- exact activation lifetime and peak-memory analysis;
- public third-party extension contracts;
- automatic repeated-block folding;
- FX generation from the authored representation;
- backend- and fusion-specific estimates;
- numerical Strassen-like estimates for rectangular products;
- disjunctive symbolic shape conditions such as unresolved broadcasting alternatives.

The asymptotic fast-multiplication comparison remains an interesting later policy. It should not be chased before the
ordinary symbolic representation, course convention, and Torch oracle agree.
