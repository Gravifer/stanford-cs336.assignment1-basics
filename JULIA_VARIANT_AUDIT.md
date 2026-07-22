# Julia semantic and execution-variant audit

This record prevents the Julia port from quietly reducing the authored Python
implementation to the narrowest code path exercised by `tests/adapters.py`.
Adapters remain the progress and parity boundary, but meaningful storage,
contraction, layout, and compilation choices inside an adapter-reachable
component remain first-class implementation or benchmark lanes.

The audit uses four dispositions:

- **parity path** — must match the selected Python adapter in outputs and
  gradients;
- **execution lane** — a mathematically equivalent implementation that must be
  preserved and benchmarked separately;
- **Julia collapse** — a Python distinction that disappears under explicit
  functions or Lux parameter/state trees, with the behavior still represented;
- **out of scope** — authored Python functionality outside every working
  adapter, recorded rather than silently forgotten.

Julia implementations use feature-first arrays. For attention, the production
candidate is `(head_feature, head, sequence, batch...)`; the experimental
alternative is `(head_feature, sequence, head, batch...)`. Both keep the
fastest-moving feature coordinate first for Julia's column-major storage.

## Feed-forward and initialization

| Python work | Julia disposition | Required evidence |
| --- | --- | --- |
| `SwiGLU_delegate`: three delegated `Linear` modules | Julia collapse into the explicit lane | same semantic parameters and arithmetic as owned separate weights; no third kernel |
| `SwiGLU_own_weights`: separate value, gate, and output arrays | explicit benchmark lane: `swiglu(x, w1, w2, w3)` | output and all explicit-argument gradients; two input GEMMs plus output GEMM |
| `SwiGLU_packed_input`: packed `[value; gate]` rows and one input GEMM | adapter-selected parity path: `swiglu(x, input_weight, output_weight)`; descriptive alias `swiglu_packed` | exact packing order, one input contraction, output and packed gradients |
| `SwiGLU = SwiGLU_packed_input` | preserve the packed lane as the model-composition default, without a mutable global alias | Lux architecture selects a concrete parameter layout at construction |
| default `d_ff = 64 * max(1, (d_model + 12) // 24)` | preserve as a shared Julia sizing function for every composition lane | exact width cases from the Python tests |
| delegated/owned/packed checkpoint translation | Julia collapse plus import boundary | one semantic mapper between course names, explicit Lux trees, and packed arrays; no `state_dict` clone |
| linear truncated normal with `σ² = 2/(d_in+d_out)`, bounded at `±3σ`; embedding standard normal bounded at `±3` | port into Lux initialization | distribution bounds and deterministic Julia RNG tests; cross-language RNG streams need not match |

The delegated and owned Python classes do identical arithmetic. Their
distinction is parameter ownership and delegation inside PyTorch, so it
collapses under Julia's explicit function and Lux parameter-tree model. The
Julia benchmark therefore has two SwiGLU lanes, not three: separate semantic
weights and packed input weights. Lux may wrap either kernel, but wrapper
organization does not create another numerical benchmark variant.

## Embedding and normalization

| Python work | Julia disposition | Required evidence |
| --- | --- | --- |
| dense embedding lookup with repeated IDs | parity path | zero-based external IDs, feature-first output, repeated-ID accumulation, dense full weight tangent |
| sparse embedding-gradient question | optional execution lane, not baseline parity | tangent representation, coalescing, optimizer state/update, weight decay, device transfer, serialization, and no accidental densification |
| feature-axis affine RMSNorm with stable reduction dtype | parity path | Float16/BFloat16-style promotion policy, output dtype, input and weight gradients |
| generalized `rms_norm` axes and optional weight | Julia collapse | ordinary `dims` plus optional explicit weight; no Python-shaped axis DSL required |
| `RMSNorm.rms_norm_einx` parser/invocation surface | out of scope as an API, because no working adapter selects it | record only; do not port einx internals into Julia |
| `SoftMax` module, implicit-dimension heuristic, and einx wrapper | Julia collapse | require an explicit Julia `dims`; dtype conversion remains an ordinary composable array operation |
| Python-only `GLU` and `GELU` helpers | out of scope | revisit only if a working adapter or model path begins using them |

## Attention and RoPE

The Python attention implementation is a family of kernels, not one MHA call.
The Julia port must retain the following axes of variation.

| Axis | Authored Python paths | Julia requirement |
| --- | --- | --- |
| query/KV sharing | MHA (`q_heads == kv_heads`), GQA, and MQA (`kv_heads == 1`) | one grouped contraction parameterized by query and KV head counts; ordinary MHA is the parity baseline, GQA/MQA are execution lanes |
| raw input widths | common Q/K/V input width or independent `kdim`/`vdim` | explicit projection shapes; cross-attention must not assume one model width |
| projected widths | independent `qk_head_dim` and `value_head_dim` | preserve distinct head dimensions throughout split, contraction, join, and output projection |
| projection storage | packed QKV for common widths; separate Q/K/V arrays for distinct widths | both physical parameter layouts |
| projection execution | one QKV GEMM for identical self-attention input; Q plus packed KV for shared K/V; three GEMMs otherwise | dispatch on an explicit architecture/layout value and input relation, not Python object identity hidden inside a module |
| activation layout | head before sequence (production) and head after sequence (experimental) | feature-first `(d, h, s, batch...)` and `(d, s, h, batch...)` lanes, with view/contiguity and allocation checks |
| Q/K RoPE execution | automatic zero-copy packed view, two separate applications, forced stacked application | preserve all three benchmark lanes and their fallbacks/rejections |
| RoPE representation | elementwise cosine/sine cache (default) and cached 2×2 matrices | both cache representations, forward and input-gradient parity, cache footprint and timing |
| positions | inferred consecutive, batch-aligned, head-aligned, independent query/key, and broadcast suffixes | preserve selection/broadcast semantics without gathering duplicate cache blocks |
| masks | boolean permit mask, additive bias, causal composition, batch/head alignment | preserve broadcasting and ensure fully masked rows return finite zeros with finite zero gradients |
| eager/compiled benchmark | eager and `torch.compile` rows, correctness checked before timing | native Julia and Reactant/Enzyme rows, with the same variant grid and correctness gate |

`tests/adapters.py` directly selects ordinary causal self-attention, with and
without default elementwise RoPE. That is the cross-language parity baseline.
GQA/MQA and the experimental layouts are nevertheless part of the user's
authored attention work and therefore move from “excluded” to Julia execution
lanes. They do not need a fake course adapter to justify their existence.

## Transformer composition

The authored model has one semantic architecture: two pre-norm additive
updates, causal RoPE self-attention followed by packed-input SwiGLU, then final
RMSNorm and an output projection. `DeltaLayer` and the nested PyTorch module
classes are composition machinery, not additional mathematical variants.

Julia will express the residual update as a pure connection or Lux composition
and keep parameters/state explicit. Course checkpoint names still require a
tested importer into the Lux parameter tree. Native PyTorch key preservation,
`assign=True`, registration hooks, and `ModuleList` bookkeeping are not runtime
requirements for Julia.

## Tokenizer

The working tokenizer adapters cover more than a naive merge loop. The Julia
port must preserve:

- byte-level initial vocabulary with special tokens first;
- special tokens as training boundaries excluded from merge statistics;
- deterministic highest-count selection with lexicographically greatest byte
  pair as the tie-break;
- the lazy heap plus authoritative-count invalidation algorithm;
- serial and parallel pretokenization producing identical counts and merges;
- longest allowed special-token precedence and explicit disallowed-token
  errors;
- merge-rank encoding, UTF-8 replacement behavior on decode, invalid-ID
  errors, and lazy iterable encoding;
- neutral vocabulary/merge file import. Python pickle remains fixture-source
  evidence, not a Julia runtime format.

Julia may use tasks, threads, or processes instead of Python
`multiprocessing`, but parallelism must not change merge order or vocabulary.

## Recorded exclusions

The Python repository also contains a substantial symbolic cost-analysis
system, `GLU`/`GELU` helpers, runtime/static typing experiments, and module
registration/checkpoint compatibility tests. They are real authored work, but
no working adapter makes them part of the requested student-part port. They
remain explicitly excluded unless the user broadens scope. The final benchmark
may use Julia-native profiling and cost accounting; it should not clone the
PyTorch/ATen symbolic observer merely to claim structural parity.

## Documentation basis

The layout decision follows Julia's official performance guidance that arrays
are column-major and the first index varies fastest. The common `swiglu` name
uses ordinary positional-arity dispatch, consistent with Julia's documented
method system; keyword arguments do not participate in dispatch. Lux model
composition will follow the current explicit parameter/state layer interface
rather than PyTorch ownership conventions.
