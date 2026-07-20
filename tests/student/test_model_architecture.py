"""Tests for decoder composition and course checkpoint compatibility."""

import warnings
from collections.abc import Callable
from typing import cast

import torch
from torch import nn

import cs336_basics.nn.model as model_module
from cs336_basics.nn import DeltaLayer
from cs336_basics.nn.model import GPTDecoderLayer, TransformerBlock, TransformerLM


def test_delta_layer_is_owned_by_reusable_modules_surface() -> None:
    assert "DeltaLayer" not in model_module.__all__


@DeltaLayer
class _ScaleDelta(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def _course_block_state(seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "ln1.weight": torch.randn(8, generator=generator),
        "attn.q_proj.weight": torch.randn(8, 8, generator=generator),
        "attn.k_proj.weight": torch.randn(8, 8, generator=generator),
        "attn.v_proj.weight": torch.randn(8, 8, generator=generator),
        "attn.output_proj.weight": torch.randn(8, 8, generator=generator),
        "ln2.weight": torch.randn(8, generator=generator),
        "ffn.w1.weight": torch.randn(16, 8, generator=generator),
        "ffn.w2.weight": torch.randn(8, 16, generator=generator),
        "ffn.w3.weight": torch.randn(16, 8, generator=generator),
    }


def test_delta_layer_adds_delta_without_changing_state_layout() -> None:
    layer = _ScaleDelta(2.0)
    x = torch.tensor([1.0, 3.0])
    delta = cast(Callable[[torch.Tensor], torch.Tensor], getattr(layer, "delta"))

    torch.testing.assert_close(layer(x), 3.0 * x)
    torch.testing.assert_close(delta(x), 2.0 * x)
    assert layer.state_dict().keys() == {"scale"}


def test_block_uses_nested_delta_sublayers() -> None:
    block = GPTDecoderLayer(8, 2, 16, 12, 10_000.0)
    keys = block.state_dict().keys()

    assert isinstance(block.attn, GPTDecoderLayer.Attention)
    assert isinstance(block.ffn, GPTDecoderLayer.FeedForward)
    assert "attn.norm.weight" in keys
    assert "attn.update.in_proj_weight" in keys
    assert "attn.update.output_proj.weight" in keys
    assert "ffn.norm.weight" in keys
    assert "ffn.update.in_weight" in keys
    assert "ffn.update.out_weight" in keys
    assert TransformerBlock is GPTDecoderLayer


def test_block_passes_completed_attention_output_to_feed_forward() -> None:
    block = GPTDecoderLayer(8, 2, 16, 12, 10_000.0)
    x = torch.randn(2, 5, 8)
    attention_delta = cast(Callable[[torch.Tensor], torch.Tensor], getattr(block.attn, "delta"))
    feed_forward_delta = cast(Callable[[torch.Tensor], torch.Tensor], getattr(block.ffn, "delta"))

    after_attention = x + attention_delta(x)
    expected = after_attention + feed_forward_delta(after_attention)

    torch.testing.assert_close(block(x), expected)


def test_block_loads_course_checkpoint_through_nested_hooks() -> None:
    block = GPTDecoderLayer(8, 2, 16, 12, 10_000.0)
    course_state = _course_block_state()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = block.load_state_dict(course_state)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert course_state.keys() == {
        "ln1.weight",
        "attn.q_proj.weight",
        "attn.k_proj.weight",
        "attn.v_proj.weight",
        "attn.output_proj.weight",
        "ln2.weight",
        "ffn.w1.weight",
        "ffn.w2.weight",
        "ffn.w3.weight",
    }
    torch.testing.assert_close(block.attn.norm.weight, course_state["ln1.weight"])
    torch.testing.assert_close(block.ffn.norm.weight, course_state["ln2.weight"])


def test_lm_native_state_loading_remains_composable() -> None:
    source = TransformerLM(19, 12, 8, 2, 2, 16, 10_000.0)
    target = TransformerLM(19, 12, 8, 2, 2, 16, 10_000.0)
    state = source.state_dict()

    result = target.load_state_dict(state, assign=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    first_layer = target.layers[0]
    assert isinstance(first_layer, GPTDecoderLayer)
    assert first_layer.attn.update.in_proj_weight is not None
    assert first_layer.attn.update.in_proj_weight.data_ptr() == state["layers.0.attn.update.in_proj_weight"].data_ptr()


def test_lm_loads_course_block_keys_under_module_list_prefix() -> None:
    model = TransformerLM(19, 12, 8, 1, 2, 16, 10_000.0)
    course_state = {
        "token_embeddings.weight": torch.randn(19, 8),
        **{f"layers.0.{key}": value for key, value in _course_block_state().items()},
        "ln_final.weight": torch.randn(8),
        "lm_head.weight": torch.randn(19, 8),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = model.load_state_dict(course_state)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
