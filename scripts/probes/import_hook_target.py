"""Undecorated functions imported by ``runtime_import_hook.py``."""

import torch
from jaxtyping import Float


def same_prefix(
    x: Float[torch.Tensor, "*mapped channels"],
    y: Float[torch.Tensor, "*mapped channels"],
) -> Float[torch.Tensor, "*mapped channels"]:
    return y


def explicit_width_guard(x: Float[torch.Tensor, "... 4"]) -> Float[torch.Tensor, "... 4"]:
    if x.shape[-1] != 4:
        raise ValueError(f"the final axis must have width 4, got shape {tuple(x.shape)}")
    return x


def annotation_only_dtype(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    return x
