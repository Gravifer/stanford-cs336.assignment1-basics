# Weight Packing and Activation Locality in Multi-Head Attention

## Abstract

Multi-head attention has several related but independent representations: the mathematical operators, the registered parameter tensors, the output ordering of the projection, and the activation layout consumed by RoPE and attention. A statement such as “head-major” is therefore incomplete unless it identifies both the tensor being discussed and the lifetime over which the ordering is preserved.

This note develops a more precise account. The weight tensors may be described with explicit head axes while remaining registered as conventional two-dimensional linear weights. Packed storage may contain unequal query/key and value widths. Its segmentation determines which views are cheap, while the activation strides produced by the projection determine locality downstream. RoPE adds another choice between a matrix cache and cosine/sine caches, and between separate, already-packed, and newly stacked Q/K execution. Initialization is a further independent policy: logical projection boundaries need not be inferred from a packed container's outer shape.

## Logical projection operators

For $h$ attention heads, the query, key, and value operators of head $i$ are

$$
W_Q^{(i)},W_K^{(i)}
\in
\mathbb{R}^{d_k \times d_{\mathrm{model}}},
\qquad
W_V^{(i)}
\in
\mathbb{R}^{d_v \times d_{\mathrm{model}}}.
$$

Collecting heads makes their structure explicit:

$$
W_Q,W_K
\in
\mathbb{R}^{h \times d_k \times d_{\mathrm{model}}},
\qquad
W_V
\in
\mathbb{R}^{h \times d_v \times d_{\mathrm{model}}}.
$$

The course notation groups the first two axes:

$$
W_Q,W_K
\in
\mathbb{R}^{h d_k \times d_{\mathrm{model}}},
\qquad
W_V
\in
\mathbb{R}^{h d_v \times d_{\mathrm{model}}}.
$$

These are equivalent views when their scalar ordering agrees:

$$
(h,d_k,d_{\mathrm{model}})
\longleftrightarrow
(h d_k,d_{\mathrm{model}}).
$$

An explicit head axis permits a direct contraction:

```python
einx.dot(
    "... [d_model], head d_k [d_model] -> ... head d_k",
    x,
    W_K,
)
```

No Python-level loop or materialized flattening is inherent in this notation. `einx` lowers the description through its tensor backend; compiled graphs and benchmarks remain the authority for a particular expression and backend. See the [`einx` backend documentation](https://einx.readthedocs.io/en/latest/gettingstarted/backends.html), [compiled-code examples](https://einx.readthedocs.io/en/stable/more/compiledcode.html), and [FAQ](https://einx.readthedocs.io/en/latest/faq/universal.html).

## Packing is a segmented ordering

When Q, K, and V have the same raw input width, their weights can share one registered tensor even when $d_k \ne d_v$. A useful logical description is

$$
W_{QKV}
\in
\mathbb{R}^{h \times (d_k+d_k+d_v) \times d_{\mathrm{model}}}.
$$

The middle axis is segmented rather than homogeneous: its Q and K sections have width $d_k$, and its V section has width $d_v$. The head axis can appear before or after this segmented axis without changing the mathematical operator. What changes is scalar order and therefore which partitions are views.

For example, projection-segmented ordering is

$$
[Q_1,\ldots,Q_h\ ;\ K_1,\ldots,K_h\ ;\ V_1,\ldots,V_h],
$$

whereas head-segmented ordering is

$$
[Q_1,K_1,V_1\ ;\ Q_2,K_2,V_2\ ;\ \ldots\ ;\ Q_h,K_h,V_h].
$$

These are two members of a larger design space, not an exhaustive binary. A representation must also specify whether Q/K/V is a true tensor axis, a heterogeneous grouped axis, three separately registered tensors, or merely three views into a two-dimensional parameter. Further subdivisions arise for packed QK with separate V, packed KV with separate Q, grouped-query attention, quantized tiles, and backend-specific blocked formats.

Equal projected widths permit the segmented dimension to be factored more explicitly. If $d_k=d_v=d$, the head-segmented representation becomes

$$
W_{QKV}^{\mathrm{head}}
\in
\mathbb{R}^{h\times 3d\times d_{\mathrm{model}}}
\cong
\mathbb{R}^{h\times 3\times d\times d_{\mathrm{model}}}.
$$

The axis of size three selects query, key, or value; it is part of the weight representation and introduces neither batch nor sequence dimensions. A projection-segmented representation instead exposes

$$
W_{QKV}^{\mathrm{projection}}
\in
\mathbb{R}^{3\times h\times d\times d_{\mathrm{model}}}.
$$

Both orderings flatten to $(3hd,d_{\mathrm{model}})$, but their scalar order differs. Converting between them requires a permutation rather than a reshape.

The implementation registers a conventional two-dimensional parameter with projection-segmented rows:

$$
W_{QKV}^{\mathrm{registered}}
\in
\mathbb{R}^{(2h d_k+h d_v)\times d_{\mathrm{model}}}.
$$

This gives stable linear-weight and checkpoint conventions, makes each complete Q, K, and V projection a view, and still permits grouped `einx` descriptions. When raw input widths differ, the implementation owns three separate two-dimensional parameters because no single rectangular linear operator can consume all three inputs. This storage decision is implemented by [`MultiheadAttention.__init__`](nn/attention/__init__.py#sym:MultiheadAttention.__init__) and translated at state-loading boundaries by [`MultiheadAttention.load_state_dict`](nn/attention/__init__.py#sym:MultiheadAttention.load_state_dict).

## Activation lifetimes determine locality

Weight ordering alone does not predict the locality of the complete attention pipeline. For a public input shaped $(B,S,d_{\mathrm{model}})$, a packed projection produces a contiguous output whose last dimension contains segmented Q, K, and V blocks:

$$
(B,S,\underbrace{h d_k}_{Q}+\underbrace{h d_k}_{K}+\underbrace{h d_v}_{V}).
$$

Splitting this tensor is free, but an individual Q view is not generally contiguous across sequence elements. Its next token begins after the complete packed QKV width, not after $h d_k$. Giving Q the logical shape $(B,S,H,d_k)$ does not alter those strides. This is why a shape label such as BSHD is insufficient: shape order and physical contiguity are different facts.

Two activation pipelines are useful to compare:

- **Head before sequence:** $(B,H,S,D)$. Heads act as a batch-like axis for the local attention implementation. Reaching this order from projection output is a view, but joining attended heads for the output projection ordinarily requires the significant BHSD-to-BSHD rearrangement.
- **Head after sequence:** $(B,S,H,D)$. If the attention contraction writes a contiguous tensor in this order, grouping $(H,D)$ into the model feature is a reshape view. Whether the dimensions are written as separate `H D` symbols or as a grouped `(H D)` symbol does not change memory layout.

Producer-consumer agreement on strides determines the material layout cost, most visibly when a BHSD activation must become BSHD. The head-after-sequence path in this implementation preserves BSHD through RoPE and a layout-specific local attention contraction, so the attended result can enter the output projection through a view. Projection-segmented Q/K/V views retain their original strides along this path; eliminating those strides would require a different projection output ordering or a fused producer.

Modern fused attention interfaces confirm that the consumer matters. The official FlashAttention packed interface accepts QKV as $(B,S,3,H,D)$ and returns $(B,S,H,D)$; its separate interface accepts Q, K, and V as $(B,S,H,D)$. It also states that an already stacked QKV tensor is preferable to concatenating one for the call, especially in backward. ([FlashAttention interface](https://github.com/Dao-AILab/flash-attention#how-to-use-flashattention)) NVIDIA Transformer Engine exposes several packed and separate layout groups, including BS3HD, BSH3D, BSHD/BS2HD, and three separate BSHD tensors. ([Transformer Engine fused-attention layouts](https://nvidia.github.io/TransformerEngine/api/c/fused_attn.html))

These interfaces are evidence for preserving compatible producer layouts, not for one universal order. Kernel, precision, architecture, cross- versus self-attention, and the placement of RoPE can change the preferable path. The experimental layout controls therefore remain private rather than becoming a public `head_first` or `head_last` API.

## Q/K reuse and RoPE

RoPE applies to Q and K, never V. The same frequency basis is used for every head by default, while the position tensor may still contain a non-singleton head axis and thereby select different positions for different heads. A singleton head axis is also accepted because it appears in the course tests.

Two mathematically equivalent cache representations are implemented:

$$
R_{p,j}
=
\begin{bmatrix}
\cos \theta_{p,j} & -\sin \theta_{p,j}\\
\sin \theta_{p,j} & \cos \theta_{p,j}
\end{bmatrix},
$$

or separate cached values $(\cos\theta_{p,j},\sin\theta_{p,j})$. The matrix cache stores four scalars per position-frequency pair; the cosine/sine cache stores two. The matrix cache is therefore exactly twice the size of the paired elementwise cache.

Explicit token positions are validated by attempting suffix expansion. Cache gathering nevertheless uses the original index shape. A position tensor shaped $(1,S)$ therefore gathers one head-independent selection and expands it afterward as a zero-stride view, rather than gathering $H$ duplicate cache blocks. Non-singleton batch or head indices remain distinct. When positions are omitted for self-attention, consecutive zero-based positions use a cache slice and avoid index gathering altogether.

Q and K share positional selection whenever their shapes and positions permit it. Three execution modes expose the remaining distinction:

- **Already-packed QK:** a packed self-projection supplies an existing zero-copy QK view. Automatic execution rotates that view once and then exposes Q and K.
- **Separate:** the rotation cache is selected once, then applied independently to Q and K. This is the automatic fallback for cross-attention or incompatible positions.
- **Allocated stack:** Q and K are stacked into a new tensor and rotated once. This path exists only as an explicit benchmark control; local timing does not promote it into automatic production behavior.

The paired behavior is exposed by [`RotaryPositionalEmbedding.apply_qk`](nn/attention/__init__.py#sym:RotaryPositionalEmbedding.apply_qk), while cache selection and application remain separate protected operations. Matrix and elementwise implementations have the same position contract, output dtype, shape, and gradients.

## Projection count under course assumptions

The course notes that Q, K, and V can be computed in three matrix multiplications. The generalized implementation retains that upper bound and opportunistically does less work when identity relationships are known:

- shared Q/K/V self-attention with packed weights uses one projection matrix multiplication;
- distinct Q with shared K/V uses one Q projection and one packed KV projection;
- general cross-attention uses three projections;
- the output projection remains a separate linear operation in every case.

The course case therefore does not pay for general cross-attention dispatch in its dominant tensor operations. It receives the packed one-projection path, shared position selection, optional zero-copy combined QK rotation, explicit $d_k^{-1/2}$ scaling, and dropout disabled during evaluation.

The current educational local SDPA still materializes attention scores and masks rather than using FlashAttention. Its causal mask is formed by a broadcasted query/key index comparison rather than allocating an all-ones square followed by `tril`. Square and rectangular masks, boolean True-means-allowed masks, additive masks, and composition modes share the same masking and softmax core.

## Benchmark evidence

### Methodology

[`scripts/benchmarks/attention.py`](../scripts/benchmarks/attention.py) treats layout and execution claims as empirical questions. Its `rope` and `mha` subcommands compare:

- matrix and elementwise RoPE;
- separate, allocated-stacked, and already-packed Q/K execution;
- sequence-only, singleton-head, and head-dependent position tensors;
- head-before-sequence and head-after-sequence activations;
- eager and `torch.compile` execution;
- forward and forward-plus-backward timing.

Every case checks forward and gradient parity before timing. Reports include Python, PyTorch, device and CUDA metadata, dtype, compilation mode, strides, contiguity, elapsed time, and peak CUDA allocation when available. Warmup and repeats are explicit command-line controls, and the script writes no artifacts by default.

### Local measurements

The expanded measurements used an NVIDIA GeForce RTX 3070 Ti Laptop GPU with Windows 11, Python 3.12.2, PyTorch 2.11.0+cu130, CUDA 13.0, cuDNN 9.19, and Triton 3.6. They covered FP32, FP16, and BF16; eager and compiled execution; forward and forward-plus-backward timing; sequence lengths from 128 through 1024; and every position, layout, and Q/K strategy exposed by the harness. The FP32 measurements used PyTorch's `highest` matrix-multiplication precision with TF32 disabled.

The first comprehensive sweep retained the harness's fixed case order. Sustained runs drove the laptop GPU from 78 °C to 89 °C. Its configured target temperature was 87 °C, and `nvidia-smi` reported accumulated software-thermal-slowdown time. Background GPU utilization also remained near 26--29 percent after the benchmark processes exited. Some late results consequently became non-monotonic. Post-run low clock readings were accompanied by NVIDIA's `Idle` reason and do not by themselves prove that those clocks applied inside timed kernels. Fixed-order measurements are therefore useful for parity, allocation, and large directional differences, but not for ranking close alternatives.

Randomized controls reduced this ordering bias. Cases were shuffled for four rounds, with three warmup iterations and either ten MHA or twenty RoPE repetitions per round. The tables report medians of the four per-round means. For inferred positions and head-before-sequence RoPE, the isolated CUDA results were:

| Shape | Matrix separate (ms) | Elementwise separate (ms) | Matrix stacked (ms) | Elementwise stacked (ms) |
|---|---:|---:|---:|---:|
| $B=2,S=256$ | 0.929 | 0.648 | 0.953 | 0.447 |
| $B=2,S=512$ | 1.869 | 0.699 | 1.900 | 0.477 |
| $B=1,S=1024$ | 1.875 | 0.697 | 1.886 | 0.444 |
| $B=4,S=512$ | 3.722 | 0.705 | 3.755 | 0.405 |

Matrix-form latency grew approximately with the number of rotated values. Elementwise execution remained much cheaper over this range, with allocated stacking helping the CUDA elementwise kernel further. The result held for inferred, explicit one-dimensional, singleton-head, and per-head positions and for both activation layouts. On CPU at $B=4,H=8,S=512,D=64$, matrix RoPE took approximately 11.5--15.0 ms while elementwise RoPE took approximately 2.1--3.7 ms. Unlike CUDA, CPU generally favored separate elementwise execution over an allocated stack.

The RoPE difference is partially hidden by projections and quadratic attention. Randomized FP32 MHA medians at $B=2,H=8,S=256,D=64$ for the zero-copy automatic path were:

| Positions | Matrix, head before (ms) | Elementwise, head before (ms) | Matrix, head after (ms) | Elementwise, head after (ms) |
|---|---:|---:|---:|---:|
| inferred | 1.905 | 2.051 | 2.044 | 2.034 |
| explicit one-dimensional | 1.999 | 2.072 | 1.841 | 2.073 |
| singleton head | 1.964 | 2.127 | 1.873 | 2.238 |

At this moderate shape, complete-MHA differences were small enough that the projection and attention work obscured the isolated RoPE ranking. Neither activation layout won consistently. Allocated stacking sometimes improved forward latency by a small amount, but it materialized Q/K storage and was less consistent in backward and on CPU. Separate execution was usually slower than the automatic zero-copy packed path. These observations support retaining automatic zero-copy Q/K execution and keeping activation layout as a private experimental control.

Elementwise caches also reduced memory. Representative eager FP32 MHA peaks at $B=2,S=512$ were approximately 89--94 MiB for elementwise RoPE and 97--102 MiB for matrix RoPE. FP16 peaks were approximately 57--59 MiB and 61--63 MiB respectively. The harness stores parity snapshots on CPU and discards the duplicate GPU reference module, so these figures no longer include retained validation outputs and parameter gradients.

Real `torch.compile` runs completed for RoPE and MHA, and every compiled forward and gradient parity check passed. Compiler state was reset between cases. At $B=2,H=8,S=256,D=64$, compiled FP16 elementwise MHA cases took approximately 1.18--1.61 ms, while compiled matrix cases ranged from approximately 2.76--5.22 ms. FP32 compilation results were more mixed and remained sensitive to thermal and ordering effects. Compilation time itself was excluded from the timed region.

Taken together, the smaller cache, lower isolated latency, better scaling, lower observed memory, and strong CPU behavior make elementwise cosine/sine rotation the implementation default. Matrix form remains available for comparison and for studying the direct $2\times2$ formulation. The benchmark does not justify automatically allocating a Q/K stack or exposing activation layout as a stable public option.

For more reproducible future measurements, the harness should randomize case order, report distributions over multiple rounds, expose inferred positions directly, and sample temperature, clock-event reasons, P-state, clocks, and background utilization during rather than after timed execution.

## Initialization follows logical boundaries

The starter initializer samples each logical linear projection from a truncated normal distribution with nominal standard deviation

$$
\sigma
=
\sqrt{\frac{2}{d_{\mathrm{in}}+d_{\mathrm{out}}}},
$$

truncated to

$$
[-3\sigma,3\sigma].
$$

Because of truncation, the realized variance is slightly below the nominal variance of the corresponding untruncated normal distribution. The implementation is [`starter_trunc_normal_for_linear_`](nn/initializer.py#sym:starter_trunc_normal_for_linear_).

Packed storage does not cause the complete width $2h d_k+h d_v$ to be treated as one undifferentiated fan-out. [`MultiheadAttention.reset_parameters`](nn/attention/__init__.py#sym:MultiheadAttention.reset_parameters) obtains Q, K, and V views and applies the starter recipe using

$$
d_{\mathrm{out},Q}=h d_k,
\qquad
d_{\mathrm{out},K}=h d_k,
\qquad
d_{\mathrm{out},V}=h d_v.
$$

Packed and unpacked storage therefore receive the same logical baseline. This is an implementation choice, not a claim that the starter recipe is optimal. Later initialization-superparameter experiments can vary the scale without confounding that experiment with the storage branch.

## Whole-projection and per-head initialization

Even after Q, K, and V are distinguished, two plausible logical units remain.

Treating the complete key projection as one linear map gives

$$
d_{\mathrm{out}}=h d_k
$$

and hence

$$
\sigma_{\mathrm{whole}}
=
\sqrt{
\frac{2}
{d_{\mathrm{model}}+h d_k}
}.
$$

Treating each head projection as its own linear map gives

$$
d_{\mathrm{out}}=d_k
$$

and hence

$$
\sigma_{\mathrm{head}}
=
\sqrt{
\frac{2}
{d_{\mathrm{model}}+d_k}
}.
$$

The two choices express different hypotheses about the relevant signal propagation. The complete projection produces $h d_k$ coordinates from a shared input, while attention subsequently processes heads independently before concatenating them and applying the output projection.

The correct scaling cannot be established from tensor rank alone. Attention includes the bilinear interaction

$$
\frac{QK^\top}{\sqrt{d_k}},
$$

a softmax, value aggregation, head concatenation, and an output projection. An initialization analysis may consequently consider:

- variances of individual query and key coordinates;
- variance and concentration of attention logits;
- the effect of the $d_k^{-1/2}$ scale;
- softmax entropy and saturation;
- variance of aggregated values;
- mixing through $W_O$;
- forward and backward signal propagation across residual blocks.

The whole-projection starter recipe is one explicit baseline. Per-head scaling and attention-specific alternatives remain meaningful subjects for empirical optimization.

## Evidence from initialization research

The original Glorot analysis motivates initialization through forward and backward signal propagation and layer Jacobians. The relevant object is the modeled transformation, not an arbitrary container into which several transformations happen to be packed. ([Glorot and Bengio, 2010](https://proceedings.mlr.press/v9/glorot10a))

PyTorch provides a useful historical counterexample: packed and unpacked `MultiheadAttention` branches have applied Xavier initialization to differently shaped containers, yielding different variances for corresponding logical projections. The issue demonstrates how storage can accidentally influence initialization when fan values are inferred from the whole packed tensor. ([PyTorch issue #166378](https://github.com/pytorch/pytorch/issues/166378))

Transformer-specific work increasingly reasons about parameters through their computational roles:

- T-Fixup scales weights according to their positions in Transformer and residual computations. ([Huang et al., 2020](https://proceedings.mlr.press/v119/huang20f.html))
- Mimetic initialization analyzes the coupled products $W_QW_K^\top$ and $W_VW_O$. ([Trockman and Kolter, 2023](https://proceedings.mlr.press/v202/trockman23a.html))
- DeepScaleLM derives moment propagation through complete Transformer blocks. ([Kedia et al., 2024](https://proceedings.mlr.press/v235/kedia24a.html))
- Conditioned initialization treats individual-head Q and K maps as meaningful operators and uses independent semi-orthogonal projections. ([Saratchandran and Lucey, 2026](https://openreview.net/forum?id=cKNOCYPo2W))

These approaches differ in purpose and assumptions, but all go beyond treating packed tensor shape as a complete account of parameter semantics.

## Conclusions

The logical attention weights admit explicit head structure, unequal Q/K and V widths, and several valid segmented orderings. The registered two-dimensional parameter is a practical canonical representation, but its row order must be stated and should not silently determine initialization.

Locality is an end-to-end property. Projection segmentation determines the strides of Q/K/V views; RoPE determines whether selected cache data or Q/K activations are materialized; attention determines its preferred layout; and the output projection benefits when $(H,D)$ can be joined as a view. A contiguous BSHD tensor already has the desired head-feature locality whether those axes are written separately or grouped.

The implementation keeps these choices observable: packed self-projection is retained, Q/K selection is shared, automatic combination requires an existing zero-copy view, both activation orders are benchmarkable, and matrix versus elementwise RoPE is selectable. Elementwise RoPE is the default on the accumulated benchmark evidence, while matrix form remains an explicit comparison path. None of those representation choices settles the later initialization question. Whole-projection, per-head, and attention-specific scaling remain separate hypotheses to test.
