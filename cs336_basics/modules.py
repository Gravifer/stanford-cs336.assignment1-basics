from typing import Never

import einx
import torch
from torch import dtype, nn

from cs336_basics import initializer as init


class Linear(nn.Module):  # mimicking :cls:`torch.nn.Linear`, but NO bias
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    @property
    def bias(self) -> Never:
        raise AttributeError("This module has no bias.")

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: dtype | None = None
    ):
        # factory_kwargs = {"device": device, "dtype": dtype} # ! in modern days a typing pain
        super().__init__()
        self.in_features = in_features  # d_in
        self.out_features = out_features  # d_out
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.register_parameter("bias", None)
        self.reset_parameters()  # Object of type `Tensor` is not callable  ty:(call-non-callable)

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        init.starter_trunc_normal_for_linear_(self.weight, self.in_features, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input."""
        # return torch.nn.functional.linear(x, self.weight, self.bias)
        return einx.dot("... d_in, d_out d_in->... d_out", x, self.weight)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"in_features={self.in_features}, out_features={self.out_features}, NO bias"


class Embedding(nn.Module):  # mimicking :cls:`torch.nn.Embedding` in :module:`torch.nn.sparse`
    __constants__ = ["num_embeddings", "embedding_dim"]
    num_embeddings: int
    embedding_dim: int
    embeddings: torch.Tensor

    def __init__(
        self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: dtype | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings  # vocab_size
        self.embedding_dim = embedding_dim  # d_model
        self.embeddings = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        init.starter_trunc_normal_for_embedding_(self.embeddings)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Apply the embedding transformation to the input."""
        return einx.get_at(
            "[vocab_size] d_model, batches... sequence_length -> batches... sequence_length d_model",
            self.embeddings,
            token_ids,
        )
