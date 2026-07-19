"""Symbolic descriptions and observations of neural-network resource use."""

from torch import nn


__all__ = ["Module"]


class Module(nn.Module):
    """Base class for repository modules that can participate in analytics.

    The initial base deliberately adds no runtime behavior. Symbolic analytics
    is layered onto it without changing Torch's module registration, state
    loading, hooks, device movement, or call semantics.
    """

