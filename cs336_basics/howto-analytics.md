# Symbolic Model Analytics in PyTorch

## Scope

This note records how PyTorch represents and observes computation, which parts can support a reusable symbolic
description, and which semantic information still has to come from the model author. It is a research memo and a set of
refined minutes. Names and interfaces mentioned here are not stability promises.

## ATen operators and their schemas

ATen is the operator layer reached after Python conveniences such as tensor methods have entered the PyTorch dispatcher.
An object such as `torch.ops.aten.bmm.default` identifies one overload; its schema describes formal arguments, return
values, defaults, aliasing, and mutation. The schema language is data interpreted by PyTorch, not C++ syntax.

For example, an annotation such as `Tensor(a!)` says that the tensor belongs to alias set `a` and is written. The runtime
path is implemented by the C++ [`SchemaParser`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/frontend/function_schema_parser.cpp),
which delegates type and alias annotations to `SchemaTypeParser`. `torch.library.Library.define()` passes operator-schema
strings into this machinery; its Python entry point lives in
[`torch/library.py`](https://github.com/pytorch/pytorch/blob/main/torch/library.py). A private Python binding is useful for
inspection:

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

An operator schema does not contain an analytical cost formula. Nor does it preserve a human reason such as "attention
packed QKV projection." The useful division is consequently:

```text
ATen schema            operator identity and argument structure
symbolic operands      shapes, dtype, and architectural expressions
cost policy            one interpretation of those operands
semantic name          why this operation exists in the model
```

Using actual ATen identities avoids creating a second, weaker operation namespace. A semantic name remains separate
because one logical operation may lower differently across implementations, and several semantic operations may share
the same ATen overload.

## Dispatch and Torch's FLOP counter

`TorchDispatchMode` observes calls after Python tensor syntax has been normalized into dispatcher operations. PyTorch's
[extension notes](https://docs.pytorch.org/docs/stable/notes/extending.html#extending-torch-native-api) explain that, at
this level, `torch.add(a, 2)` and `a + 2` arrive as the same ATen call.

[`FlopCounterMode`](https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py) is a particularly close
precedent. It is a context manager backed by `TorchDispatchMode`. Its registry is keyed by ATen operator packets, while
formula wrappers replace tensor arguments with their shapes. The formulas for `mm` and `bmm` therefore receive operand
shapes rather than model-specific `m`, `n`, and `p` supplied by each caller.

The counter uses [`ModuleTracker`](https://github.com/pytorch/pytorch/blob/main/torch/utils/module_tracker.py) to attribute
observed operations to fully qualified module names. The
[`ModuleTracker` documentation](https://docs.pytorch.org/docs/stable/module_tracker.html) explicitly describes this use.
This is strong evidence for using Torch's official hooks and tracking machinery when invocation observation is added.
It is not a reason to install permanent analytics hooks in every module constructor.

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
uses SymPy expressions directly and accepts observed `SymInt` values at its boundary. Concrete classes may import SymPy
lazily inside their own analytics hooks when their symbolic description needs it.

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

## Module structure, FX, and export

Torch's module tree carries parameters, buffers, state-dict prefixes, and a useful semantic hierarchy. Official forward
hooks and `ModuleTracker` can associate a successful call with that hierarchy. Torch has no standard hook asking a module
for an authored symbolic cost description, so a small protected provider remains necessary. Official hooks should still
transport future invocation information; the provider need not reinvent execution interception.

[`torch.fx`](https://docs.pytorch.org/docs/stable/fx.html) represents executable dataflow as a graph of calls and values.
[`torch.export`](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export.html) captures an ahead-of-time
`ExportedProgram` backed by FX. It lifts parameters and buffers into a graph signature, records shape constraints, and
normalizes tensor computation into ATen and custom operators. Export can retain the default training-oriented ATen IR,
functionalize it, or decompose it into the smaller Core ATen operator set; the
[Export IR specification](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/ir_spec.html) describes the
resulting graph and state.

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

and finally evaluated for a particular invocation. The static declaration remains useful without running the model;
future observation can bind it and check its chosen ATen lowering.

Symbols are local to a module description. Two distant modules may both display `d_model` without their variables being
identical. Internally, identity-distinct SymPy symbols prevent accidental capture, while immediate parent-child
substitutions establish the identities that the model author intends.

The model author also owns recursion policy. Ordinary modules contribute only their local work and delegate to their
children. A repeated Transformer stack may deliberately describe one representative block with symbolic `num_layers`
repetition and avoid traversing the remaining concrete blocks; embeddings, final normalization, and logit emission are
still traversed normally. Automatically discovering and folding repeated siblings is a separate presentation problem.

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
which it decomposes is double counting. The authored plane may keep a human name and an expected eager-ATen anchor; the
observed plane determines whether that expectation still matches the implementation.

## Deliberately deferred questions

The following are not prerequisites for the current static work:

- binding call-specific variables with hooks or a context manager;
- training and backward-pass costs;
- exact activation lifetime and peak-memory analysis;
- public third-party extension contracts;
- automatic repeated-block folding;
- FX generation from the authored representation;
- backend- and fusion-specific estimates;
- numerical Strassen-like estimates for rectangular products.

The asymptotic fast-multiplication comparison remains an interesting later policy. It should not be chased before the
ordinary symbolic representation, course convention, and Torch oracle agree.
