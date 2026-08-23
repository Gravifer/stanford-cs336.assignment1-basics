"""Functional neural-network primitives with named tensor transformations."""

import warnings
from typing import Literal, overload

import einx
import torch
from jaxtyping import Bool, Float, Int
from torch import sigmoid  # we're re-exporting it because the assignment explicitly allows it

from ._typing import MaskBias


def glu(
    value: Float[torch.Tensor, "*shape"],
    gate: Float[torch.Tensor, "*shape"],
) -> Float[torch.Tensor, "*shape"]:
    """Gated linear unit.

    Note that we expect unpacked input; with packed input, pass views.
    """
    if value.shape != gate.shape:
        raise ValueError("Value and gate tensors must have the same shape for GLU.")
    return value * sigmoid(gate)


def gelu(
    input: Float[torch.Tensor, "*shape"],
    approximate: Literal["none", "tanh"] = "tanh",
) -> Float[torch.Tensor, "*shape"]:
    """GELU activation function.

    Args:
        input: Input tensor of shape (*mapped, *dims).
        approximate: Whether to use the approximate GELU implementation.

    Returns:
        Output tensor of shape (*mapped, *dims).
    """
    if approximate == "none":
        return 0.5 * input * (1 + torch.erf(input / (2**0.5)))
    elif approximate == "tanh":
        return 0.5 * input * (1 + torch.tanh(input * 0.7978845608 * (1 + 0.044715 * input * input)))
    else:
        raise NotImplementedError(f"Unknown approximate mode: {approximate}")


def softmax(
    input: Float[torch.Tensor, "*shape"],
    dim: int,
    _stacklevel: int = 3,
    dtype: torch.dtype | None = None,
) -> Float[torch.Tensor, "*shape"]:
    """Softmax activation function.

    Takes two parameters (so that the signature matches torch.nn.functional.softmax):
        a tensor and a dimension 𝑖, and apply softmax to the 𝑖-th dimension of the input tensor.
    The output tensor should have the same shape as the input tensor,
        but its 𝑖-th dimension will now have a normalized probability distribution.
    Use the trick of subtracting the maximum value in the 𝑖-th dimension
        from all elements of the 𝑖-th dimension to avoid numerical stability issues.
    """
    type InputShaped = Float[torch.Tensor, "*shape"]
    # type ReducedShaped = Float[torch.Tensor, "*reduced"]
    type KeptShaped = Float[torch.Tensor, "*kept"]
    if dim is None:
        # Preserve PyTorch's deprecated implicit-axis compatibility; see
        # torch.nn.functional._get_softmax_dim.
        warnings.warn(
            "Implicit dimension choice for softmax has been deprecated. "
            "Change the call to include dim=X as an argument.",
            stacklevel=_stacklevel,
        )
        dim = 0 if input.ndim in (0, 1, 3) else 1
    if dtype is not None:
        input = input.to(dtype)
    # # derive einx strings
    # dim: int = (dim if dim is not None else -1) % input.ndim
    # axes_map: dict[str, int] = {(f"n{i}"): d                        for i, d in enumerate(input.shape)}
    # _shape:    str = " ".join(   f"n{i}"                            for i    in range(input.ndim))
    # _targeted: str = " ".join(   f"n{i}" if i != dim else f"[n{i}]" for i    in range(input.ndim))
    # _kept:     str = " ".join(   f"n{i}" if i != dim else "()"      for i    in range(input.ndim))
    # _reduced:  str = " ".join(   f"n{i}"                            for i    in range(input.ndim) if i != dim)
    # return einx.softmax(_shape, input, **axes_map) # * arguably cheating
    max_kept: KeptShaped = torch.max(input, dim=dim, keepdim=True).values
    # max_kept: KeptShaped = einx.max(f"{_targeted} -> {_kept}", input, **axes_map)
    # max_reduced: ReducedShaped = einx.max(f"{_targeted} -> {_reduced}", input, **axes_map)
    exp_input: InputShaped = torch.exp(input - max_kept)
    # exp_input: InputShaped = einx.subtract(f"{_shape}, {_kept} -> {_shape}", input, max_kept, **axes_map).exp()
    # exp_input: InputShaped = einx.subtract(f"{_shape}, {_reduced} -> {_shape}", input, max_reduced, **axes_map).exp()
    sum_exp: KeptShaped = torch.sum(exp_input, dim=dim, keepdim=True)
    # sum_exp: KeptShaped = einx.sum(f"{_targeted} -> {_kept}", exp_input, **axes_map)
    # sum_exp: ReducedShaped = einx.sum(f"{_targeted} -> {_reduced}", exp_input, **axes_map)
    return exp_input / sum_exp
    # return einx.divide(f"{_shape}, {_kept} -> {_shape}", exp_input, sum_exp, **axes_map)
    # return einx.divide(f"{_shape}, {_reduced} -> {_shape}", exp_input, sum_exp, **axes_map)


def log_softmax(
    input: Float[torch.Tensor, "*shape"],
    dim: int,
    _stacklevel: int = 3,
    dtype: torch.dtype | None = None,
) -> Float[torch.Tensor, "*shape"]:
    """Apply a softmax followed by a logarithm.

    While mathematically equivalent to log(softmax(x)), doing these two
    operations separately is slower and numerically unstable.
    """  # so we copy our softmax but replace multiplicative stuff with additive stuff
    type InputShaped = Float[torch.Tensor, "*shape"]
    type KeptShaped = Float[torch.Tensor, "*kept"]
    if dim is None:
        warnings.warn(
            "Implicit dimension choice for softmax has been deprecated. "
            "Change the call to include dim=X as an argument.",
            stacklevel=_stacklevel,
        )
        dim = 0 if input.ndim in (0, 1, 3) else 1
    if dtype is not None:
        input = input.to(dtype)
    # return einx.log_softmax(_shape, input, **axes_map) # * arguably cheating
    max_kept: KeptShaped = torch.max(input, dim=dim, keepdim=True).values
    sub_input: InputShaped = input - max_kept
    sum_exp: KeptShaped = torch.sum(sub_input.exp(), dim=dim, keepdim=True)
    return sub_input - torch.log(sum_exp)


def linear(
    input: Float[torch.Tensor, "*mapped in_features"],
    weight: Float[torch.Tensor, "out_features in_features"],
) -> Float[torch.Tensor, "*mapped out_features"]:
    """Linear transformation without bias.

    Args:
        input: Input tensor of shape (..., in_features).
        weight: Weight tensor of shape (out_features, in_features).

    Returns:
        Output tensor of shape (..., out_features).
    """
    return einx.dot("mapped... d_in, d_out d_in -> mapped... d_out", input, weight)


def silu(
    input: Float[torch.Tensor, "*shape"],
) -> Float[torch.Tensor, "*shape"]:
    """SiLU activation function."""
    return input * sigmoid(input)


def swiglu(
    value: Float[torch.Tensor, "*shape"],
    gate: Float[torch.Tensor, "*shape"],
) -> Float[torch.Tensor, "*shape"]:
    """Swish GLU.

    Note that we expect unpacked input; with packed input, pass views.
    """
    if value.shape != gate.shape:
        raise ValueError("Value and gate tensors must have the same shape for SwiGLU.")
    return value * silu(gate)


def embedding(
    input: Int[torch.Tensor, "*batch sequence_length"],
    weight: Float[torch.Tensor, "num_embeddings embedding_dim"],
) -> Float[torch.Tensor, "*batch sequence_length embedding_dim"]:
    """Embedding lookup.

    Args:
        input: LongTensor of shape (*batch, sequence_length) containing indices.
        weight: Weight tensor of shape (num_embeddings, embedding_dim).

    Returns:
        Output tensor of shape (*batch, sequence_length, embedding_dim).
    """
    return einx.get_at(
        "[vocab_size] d_model, batch... sequence_length -> batch... sequence_length d_model",
        weight,
        input,
    )


def rms_norm(  # torch flavored
    input: Float[torch.Tensor, "*batch_shape"],
    dims: tuple[int, ...],
    weight: Float[torch.Tensor, "*shape"] | None = None,
    eps: float = 1e-5,
) -> Float[torch.Tensor, "*batch_shape"]:
    r"""Apply Root Mean Square Layer Normalization.

    See :class:`~modules.RMSNorm` for details.
    """
    if dims == ():
        warnings.warn("RMSNorm with empty normalized shape is a no-op", stacklevel=2)
        return input
    in_dtype = input.dtype
    weight_dtype = weight.dtype if weight is not None else in_dtype
    op_dtype = torch.promote_types(in_dtype, weight_dtype)
    if op_dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        op_dtype = torch.float32  # up-cast to float32 for numerical stability
    input = input.to(op_dtype)
    axes_map: dict[str, int] = {f"n{i}": d for i, d in enumerate(dims)}
    _normed: str = " ".join(axes_map)
    # # Compute the root mean square
    # mean_square = torch.mean(input**2, dim=dims, keepdim=True)
    # rms = torch.sqrt(mean_square + eps)
    rms: Float[torch.Tensor, "*mapped"] = (
        einx.mean(f"mapped... [{_normed}]", input.abs() ** 2, **axes_map) + eps
    ) ** 0.5
    # # Normalize the input
    # normalized_input = input / rms
    normed: Float[torch.Tensor, "*batch_shape"] = einx.divide(
        f"mapped... {_normed}, mapped... -> mapped... {_normed}", input, rms, **axes_map
    )
    # # Apply weights if provided
    if weight is not None:
        # normalized_input = normalized_input * weights
        normed: Float[torch.Tensor, "*batch_shape"] = einx.multiply(
            f"mapped... {_normed}, {_normed} -> mapped... {_normed}", normed, weight, **axes_map
        )
    return normed


@overload
def nll_loss(
    logprobs: Float[torch.Tensor, "*shape classes"],
    target: Int[torch.Tensor, "*shape"],
    *,
    weight: Float[torch.Tensor, "classes"] | None = None,  # noqa: F821
    ignore_index: int | None = None,
    reduction: Literal["none"],
) -> Float[torch.Tensor, "*shape"]: ...


@overload
def nll_loss(
    logprobs: Float[torch.Tensor, "*shape classes"],
    target: Int[torch.Tensor, "*shape"],
    *,
    weight: Float[torch.Tensor, "classes"] | None = None,  # noqa: F821
    ignore_index: int | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> Float[torch.Tensor, ""]: ...


def nll_loss(
    logprobs: Float[torch.Tensor, "*shape classes"],
    target: Int[torch.Tensor, "*shape"],
    *,
    weight: Float[torch.Tensor, "classes"] | None = None,  # noqa: F821
    ignore_index: int | None = None,
    # ^ torch 7 had the default sentinel -100 for whatever reasons; we mostly won't use it, so we will use None here
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Float[torch.Tensor, "*shape"] | Float[torch.Tensor, ""]:
    r"""Negative log likelihood loss.

    Args:
        logprobs: Predicted log probabilities of shape (*shape, classes).
        target: Class indices of shape (*shape).
        weight: Optional per-class weights of shape (classes,).
        ignore_index: Exact target value whose prediction sites are omitted.
            The value may lie outside the valid class range. (optional)
        reduction: ``"none"`` preserves *shape; ``"sum"`` and ``"mean"``
            return a scalar.

    Returns:
        The negative log likelihood loss of shape (*shape) when unreduced,
        or a scalar when reduced by summation or averaging.
    """
    type LossShaped = Float[torch.Tensor, "*shape"]
    type TargetShaped = Int[torch.Tensor, "*shape"]
    type ScalarShaped = Float[torch.Tensor, ""]
    if logprobs.shape[:-1] != target.shape:
        raise ValueError(
            "target shape must equal logprobs shape without its classes axis; "
            f"got logprobs of shape {logprobs.shape} and target of {target.shape}"
        )
    classes: int = logprobs.shape[-1]

    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"unsupported reduction: {reduction!r}")

    if weight is not None and weight.shape != (classes,):
        raise ValueError(f"weight must have shape ({classes},), got {weight.shape}")

    is_ignored: Bool[torch.Tensor, "*shape"] = (
        torch.zeros_like(target, dtype=torch.bool)
        if ignore_index is None
        else einx.equal("shape..., -> shape...", target, ignore_index)
    )

    # Apply ignoring without changing the prediction shape. Ignored sites carry
    # class 0 only as a valid dummy gather index; is_ignored retains their actual
    # meaning, so a genuine class-0 target remains distinguishable.
    masked_target: TargetShaped = torch.where(is_ignored, 0, target)

    # Turn every remaining negative target into a positive out-of-range index
    # so that gathering rejects invalid input instead of wrapping around to a
    # class at the end. Ignored negative sentinels have already become the dummy
    # index 0. Unlike einx.get_at's ravelled indexing, torch.gather preserves
    # the class-axis bound for every prediction site.
    gather_target: TargetShaped = torch.where(masked_target < 0, classes, masked_target)

    # ! as of writing (einx 0.4.2), einx.get_at allowing out-of-range class indices to spill into adjacent prediction sites <https://github.com/fferflo/einx/issues/37>
    selected_logprobs: LossShaped = torch.gather(logprobs, dim=-1, index=gather_target.unsqueeze(-1)).squeeze(-1)

    loss: LossShaped = -selected_logprobs
    if weight is None:
        reduction_weight: LossShaped = ~is_ignored  # like we got all 1 weights
    else:
        selected_weight: LossShaped = einx.get_at(
            "[classes], shape... -> shape...",
            weight,
            gather_target,
        )
        reduction_weight: LossShaped = selected_weight.masked_fill(is_ignored, 0)
        loss: LossShaped = loss * reduction_weight
    loss: LossShaped = loss.masked_fill(is_ignored, 0)

    if reduction == "none":
        return loss

    total: ScalarShaped = einx.sum("[shape...]", loss)

    if reduction == "sum":
        return total

    denominator: ScalarShaped = einx.sum("[shape...]", reduction_weight)
    return total / denominator


@overload
def kl_div(
    logprobs: Float[torch.Tensor, "*shape classes"],
    target: Float[torch.Tensor, "*shape classes"],
    *,
    reduction: Literal["none"],
    log_target: bool = False,
) -> Float[torch.Tensor, "*shape"]: ...


@overload
def kl_div(
    logprobs: Float[torch.Tensor, "*shape classes"],
    target: Float[torch.Tensor, "*shape classes"],
    *,
    reduction: Literal["batchmean", "mean", "sum"] = "mean",
    log_target: bool = False,
) -> Float[torch.Tensor, ""]: ...


def kl_div(
    logprobs: Float[torch.Tensor, "*shape classes"],
    target: Float[torch.Tensor, "*shape classes"],
    *,
    reduction: Literal["none", "batchmean", "mean", "sum"] = "mean",
    log_target: bool = False,
) -> Float[torch.Tensor, "*shape"] | Float[torch.Tensor, ""]:
    r"""Compute categorical KL divergence from target to prediction.

    For every prediction site, this computes
    :math:`D_{KL}(P\Vert Q)=\sum_c P_c(\log P_c-\log Q_c)`, where
    ``logprobs`` contains :math:`\log Q` and ``target`` describes :math:`P`.
    Inputs are assumed to describe distributions but are not normalized or
    otherwise validated as such.

    Unlike :func:`torch.nn.functional.kl_div`, the class reduction is
    intrinsic: an unreduced result has shape ``(*shape)`` rather than
    ``(*shape, classes)``.

    Args:
        logprobs: Predicted log probabilities of shape (*shape, classes).
        target: Target probabilities, or target log probabilities when
            ``log_target`` is true, with the same shape as ``logprobs``.
        reduction: ``"none"`` returns one KL value per prediction site;
            ``"sum"`` sums all sites; ``"mean"`` averages all sites; and
            ``"batchmean"`` sums all sites and divides by the first leading
            dimension, which is interpreted as batch.
        log_target: Whether ``target`` contains log probabilities. Passing log
            probabilities can avoid numerical issues from taking their logarithm.

    Returns:
        KL divergence of shape (*shape) when unreduced, or a scalar when reduced.
    """
    type ProbsShaped = Float[torch.Tensor, "*shape classes"]
    type LossShaped = Float[torch.Tensor, "*shape"]
    type ScalarShaped = Float[torch.Tensor, ""]

    if logprobs.shape != target.shape:
        raise ValueError(f"target shape must equal logprobs shape; got {target.shape} and {logprobs.shape}")
    if reduction not in ("none", "batchmean", "mean", "sum"):
        raise ValueError(f"unsupported reduction: {reduction!r}")
    if reduction == "batchmean" and logprobs.ndim < 2:
        raise ValueError("batchmean requires a leading batch axis before the classes axis")

    if log_target:
        classwise_loss: ProbsShaped = target.exp() * (target - logprobs)
    else:
        classwise_loss = torch.xlogy(target, target) - target * logprobs
    loss: LossShaped = einx.sum("shape... [classes]", classwise_loss)

    if reduction == "none":
        return loss
    if reduction == "mean":
        return einx.mean("[shape...]", loss)

    total: ScalarShaped = einx.sum("[shape...]", loss)
    if reduction == "sum":
        return total
    return total / logprobs.shape[0]


@overload
def cross_entropy(
    logits: Float[torch.Tensor, "*shape classes"],
    target: Int[torch.Tensor, "*shape"] | Float[torch.Tensor, "*shape classes"],
    *,
    weight: Float[torch.Tensor, "classes"] | None = None,  # noqa: F821
    ignore_index: int | None = None,
    reduction: Literal["none"],
    label_smoothing: float = 0.0,
) -> Float[torch.Tensor, "*shape"]: ...


@overload
def cross_entropy(
    logits: Float[torch.Tensor, "*shape classes"],
    target: Int[torch.Tensor, "*shape"] | Float[torch.Tensor, "*shape classes"],
    *,
    weight: Float[torch.Tensor, "classes"] | None = None,  # noqa: F821
    ignore_index: int | None = None,
    reduction: Literal["mean", "sum"] = "mean",
    label_smoothing: float = 0.0,
) -> Float[torch.Tensor, ""]: ...


def cross_entropy(
    logits: Float[torch.Tensor, "*shape classes"],
    target: Int[torch.Tensor, "*shape"] | Float[torch.Tensor, "*shape classes"],
    *,
    weight: Float[torch.Tensor, "classes"] | None = None,  # noqa: F821
    ignore_index: int | None = None,
    # ^ torch 7 had the default sentinel -100 for whatever reasons; we mostly won't use it, so we will use None here
    reduction: Literal["none", "mean", "sum"] = "mean",
    label_smoothing: float = 0.0,
) -> Float[torch.Tensor, "*shape"] | Float[torch.Tensor, ""]:
    r"""Cross entropy loss.

    Args:
        logits: Predicted logits of shape (*shape, classes).
        target: Ground truth class indices of shape (*shape) or class probabilities of shape (*shape, classes).
        weight: Optional per-class weights of shape (classes,).
        ignore_index: Exact target value whose prediction sites are omitted.
            The value may lie outside the valid class range. Only
            applicable when the target contains class indices. (optional)
        reduction: ``"none"`` preserves *shape; ``"sum"`` and ``"mean"``
            return a scalar.
        label_smoothing:  A float in [0.0, 1.0]. Specifies the amount
            of smoothing when computing the loss, where 0.0 means no smoothing. The targets
            become a mixture of the original ground truth and a uniform distribution as described in
            `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.

    Returns:
        The cross entropy loss of shape (*shape) when unreduced,
        or a scalar when reduced by summation or averaging.
    """
    type LogitsShaped = Float[torch.Tensor, "*shape classes"]
    type LossShaped = Float[torch.Tensor, "*shape"]
    type ScalarShaped = Float[torch.Tensor, ""]

    target_probs = False
    if logits.shape == target.shape:
        if not target.is_floating_point():
            raise TypeError("class-probability targets must have a floating-point dtype")
        if ignore_index is not None:
            raise ValueError("ignore_index is not applicable when the target contains class probabilities.")
        target_probs = True
    elif logits.shape[:-1] != target.shape:
        raise ValueError(
            "target shape must equal logprobs shape without its classes axis; "
            f"got logprobs of shape {logits.shape} and target of {target.shape}"
        )
    elif target.is_floating_point():
        raise TypeError("class-index targets must have an integer dtype")
    classes: int = logits.shape[-1]

    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"unsupported reduction: {reduction!r}")

    if weight is not None and weight.shape != (classes,):
        raise ValueError(f"weight must have shape ({classes},), got {weight.shape}")

    if label_smoothing < 0.0 or label_smoothing > 1.0:
        raise ValueError("label_smoothing must be in [0, 1]")

    logprobs: LogitsShaped = log_softmax(logits, dim=-1)

    if target_probs:  # target is class probabilities
        if label_smoothing > 0.0:
            target = target * (1.0 - label_smoothing) + label_smoothing / classes
        if weight is not None:
            target = einx.multiply("shape... classes, classes", target, weight)
        loss: LossShaped = -einx.sum(
            "shape... [classes]",
            einx.multiply("shape... classes, shape... classes", target, logprobs),
        )
        if reduction == "none":
            return loss
        total: ScalarShaped = einx.sum("[shape...]", loss)
        if reduction == "sum":
            return total
        return einx.mean("[shape...]", loss)

    nll = nll_loss(
        logprobs,
        target,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction if label_smoothing == 0.0 else "none",
    )
    if label_smoothing == 0.0:
        return nll

    uniform_terms: LogitsShaped = logprobs
    if weight is not None:
        uniform_terms = einx.multiply("shape... classes, classes", uniform_terms, weight)
    uniform_loss: LossShaped = -einx.mean("shape... [classes]", uniform_terms)

    is_ignored: Bool[torch.Tensor, "*shape"] = (
        torch.zeros_like(target, dtype=torch.bool)
        if ignore_index is None
        else einx.equal("shape..., -> shape...", target, ignore_index)
    )
    uniform_loss = uniform_loss.masked_fill(is_ignored, 0)
    loss = nll * (1.0 - label_smoothing) + uniform_loss * label_smoothing
    if reduction == "none":
        return loss

    total = einx.sum("[shape...]", loss)
    if reduction == "sum":
        return total

    if weight is None:
        reduction_weight: LossShaped = ~is_ignored
    else:
        masked_target = torch.where(is_ignored, 0, target)
        selected_weight: LossShaped = einx.get_at("[classes], shape... -> shape...", weight, masked_target)
        reduction_weight = selected_weight.masked_fill(is_ignored, 0)
    denominator: ScalarShaped = einx.sum("[shape...]", reduction_weight)
    return total / denominator


def _attention_weights(
    scores: torch.Tensor,
    seq_q: int,
    seq_k: int,
    mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool | Literal["torch", "compose", "noclone"],
    *,
    attn_mask: torch.Tensor | None,
    attn_bias: torch.Tensor | None,
) -> torch.Tensor:
    """Apply mask composition, softmax, and dropout to precomputed scores."""
    type _Mask = Bool[torch.Tensor, "*slew query_len key_len"]
    type _Bias = Float[torch.Tensor, "*slew query_len key_len"]
    type _MaskBias = MaskBias[torch.Tensor, "*slew query_len key_len"]  # ty:ignore[invalid-syntax-in-forward-annotation]
    if sum(x is None for x in (mask, attn_mask, attn_bias)) < 2:
        raise ValueError("Only one of `mask` `attn_mask` `attn_bias` should be provided.")
    attn_bias: _MaskBias | None = attn_bias if attn_bias is not None else mask if mask is not None else attn_mask
    if is_causal == "torch":
        assert attn_bias is None, "torch's sdpa does not accept a mask with is_causal on; compose the mask yourself"
    elif is_causal == "noclone":
        assert attn_bias is None or attn_bias.dtype == torch.bool, (
            f"{attn_bias.dtype} mask will need to be cloned to apply causal masking"
        )
    if attn_bias is not None:
        assert attn_bias.shape[-2:] == (seq_q, seq_k), (
            f"Mask shape must be compatible with query and key; expected ({seq_q}, {seq_k}), got {attn_bias.shape[-2:]}."
        )

    if attn_bias is not None:
        if attn_bias.device != scores.device:
            warnings.warn(
                f"The mask is on device {attn_bias.device}, "
                f"but attention scores are on device {scores.device}. "
                "Moving the mask to the scores' device.",
                stacklevel=2,
            )
            attn_bias: _MaskBias = attn_bias.to(device=scores.device)
        if attn_bias.dtype != torch.bool:
            attn_bias: _Bias = attn_bias.to(dtype=scores.dtype)

    if is_causal:
        query_positions = einx.id("query -> query 1", torch.arange(seq_q, device=scores.device))
        key_positions = einx.id("key -> 1 key", torch.arange(seq_k, device=scores.device))
        causal_mask: _Mask = key_positions <= query_positions
        if attn_bias is None:
            attn_bias: _Mask = causal_mask
        elif attn_bias.dtype == torch.bool:
            attn_bias: _Mask = attn_bias & causal_mask
        else:
            # Clone attn_bias to avoid mutating the caller's tensor
            attn_bias: _Bias = attn_bias.clone().masked_fill_(~causal_mask, float("-inf"))

    fully_masked: torch.Tensor | None = None
    if attn_bias is not None:
        # Support both boolean masks and additive numeric biases.
        if attn_bias.dtype == torch.bool:
            fully_masked = ~einx.any("... query [key] -> ... query 1", attn_bias)
            scores = scores.masked_fill(~attn_bias, float("-inf"))
        else:
            fully_masked = torch.isneginf(einx.max("... query [key] -> ... query 1", attn_bias))
            scores += attn_bias
        # Softmax is undefined for an all-negative-infinity row. Give it a
        # finite input, then restore the SDPA convention of zero attention.
        scores = scores.masked_fill(fully_masked, 0.0)
    attn_weights = torch.softmax(scores, dim=-1)
    if fully_masked is not None:
        attn_weights = attn_weights.masked_fill(fully_masked, 0.0)
    if dropout_p > 0.0:
        torch.dropout_(attn_weights, p=dropout_p, train=True)
    return attn_weights


def scaled_dot_product_attention(  # mimicking :func:`torch.nn.functional.scaled_dot_product_attention`
    query: Float[torch.Tensor, "*batch query_len d_k"],
    key: Float[torch.Tensor, "*batch key_len d_k"],
    value: Float[torch.Tensor, "*batch key_len d_v"],
    mask: MaskBias[torch.Tensor, "*slew query_len key_len"] | None = None,  # ty:ignore[invalid-syntax-in-forward-annotation]
    dropout_p: float = 0.0,
    is_causal: bool | Literal["torch"] | Literal["compose"] | Literal["noclone"] = False,
    scale: float | None = None,
    *,
    attn_mask: MaskBias[torch.Tensor, "*slew query_len key_len"] | None = None,  # ty:ignore[invalid-syntax-in-forward-annotation]
    attn_bias: Float[torch.Tensor, "*slew query_len key_len"] | None = None,
) -> Float[torch.Tensor, "*batch query_len d_v"]:
    """Compute scaled dot-product attention.

    Handles keys and queries of shape (batch_size, ..., seq_len, d_k)
    and values of shape (batch_size, ..., seq_len, d_v),
    where ... represents any number of other batch-like dimensions (if provided).
    Also support an optional user-provided boolean mask of shape (seq_len, seq_len);
    can have additional dimensions, but have to be a suffix of the input batch shapes.

    Args:
        query: Query tensor of shape (*batch, query_len, d_k).
        key: Key tensor of shape (*batch, key_len, d_k).
        value: Value tensor of shape (*batch, key_len, d_v).
        mask: Optional mask tensor of shape (*slew, query_len, key_len). (the alias `attn_mask` is allowed)
            The alternatives `attn_mask` and `attn_bias` (this one can only be float, not boolean) are also accepted
            to mimic the torch counterpart, but only one of the three should be provided.
            if numeric rather than boolean, it will be treated as additive mask.
            Passing a boolean mask is equivalent to 0 for True and -inf for False.
            A query row with no permitted keys produces a zero output row.
        dropout_p: Dropout probability.
        is_causal: Whether to apply causal masking.
            when set to "torch", the passed mask will be rejected.
        scale: Scaling factor for the attention scores; default to 1 / √d_k.
    Returns:
        Output tensor of shape (*batch, query_len, d_v).
    """
    assert query.shape[-1] == key.shape[-1], (
        f"Cannot determine d_k; queries are {query.shape[-1]}-D while keys are {key.shape[-1]}-D."
    )
    seq_q, seq_k, d_k = query.shape[-2], key.shape[-2], key.shape[-1]
    scores = einx.dot("... q [d_k], ... k [d_k] -> ... q k", query, key) * (1 / d_k**0.5 if scale is None else scale)
    attn_weights = _attention_weights(
        scores,
        seq_q,
        seq_k,
        mask,
        dropout_p,
        is_causal,
        attn_mask=attn_mask,
        attn_bias=attn_bias,
    )
    return einx.dot("... q k, ... k d_v -> ... q d_v", attn_weights, value)
