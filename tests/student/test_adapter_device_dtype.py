import pytest
import torch
import torch.nn.functional as torch_f

from tests.adapters import (
    run_embedding,
    run_linear,
    run_multihead_self_attention,
    run_multihead_self_attention_with_rope,
    run_rmsnorm,
    run_rope,
    run_swiglu,
)


def test_weight_adapters_preserve_float64() -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 8, dtype=torch.float64, generator=generator)

    linear_weight = torch.randn(5, 8, dtype=torch.float64, generator=generator)
    linear = run_linear(8, 5, linear_weight, x)
    torch.testing.assert_close(linear, torch_f.linear(x, linear_weight))

    embedding_weight = torch.randn(11, 8, dtype=torch.float64, generator=generator)
    token_ids = torch.tensor([[0, 3, 7], [2, 5, 10]])
    embedding = run_embedding(11, 8, embedding_weight, token_ids)
    torch.testing.assert_close(embedding, torch_f.embedding(token_ids, embedding_weight))

    w1 = torch.randn(13, 8, dtype=torch.float64, generator=generator)
    w2 = torch.randn(8, 13, dtype=torch.float64, generator=generator)
    w3 = torch.randn(13, 8, dtype=torch.float64, generator=generator)
    swiglu = run_swiglu(8, 13, w1, w2, w3, x)
    expected_swiglu = torch_f.linear(torch_f.silu(torch_f.linear(x, w1)) * torch_f.linear(x, w3), w2)
    torch.testing.assert_close(swiglu, expected_swiglu)

    norm_weight = torch.randn(8, dtype=torch.float64, generator=generator)
    rmsnorm = run_rmsnorm(8, 1e-5, norm_weight, x)
    torch.testing.assert_close(rmsnorm, torch_f.rms_norm(x, (8,), norm_weight, 1e-5))

    for output in (linear, embedding, swiglu, rmsnorm):
        assert output.dtype == torch.float64
        assert output.device == x.device


def test_attention_and_rope_adapters_preserve_float64() -> None:
    generator = torch.Generator().manual_seed(1)
    q_weight, k_weight, v_weight, output_weight = [
        torch.randn(8, 8, dtype=torch.float64, generator=generator) for _ in range(4)
    ]
    x = torch.randn(2, 4, 8, dtype=torch.float64, generator=generator)
    positions = torch.arange(4).expand(2, 4)

    attention = run_multihead_self_attention(8, 2, q_weight, k_weight, v_weight, output_weight, x)
    attention_with_rope = run_multihead_self_attention_with_rope(
        8,
        2,
        8,
        10_000.0,
        q_weight,
        k_weight,
        v_weight,
        output_weight,
        x,
        positions,
    )
    rope = run_rope(8, 10_000.0, 8, x, positions)

    for output in (attention, attention_with_rope, rope):
        assert output.dtype == torch.float64
        assert output.device == x.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rope_attention_adapter_runs_on_accelerator() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(2)
    q_weight, k_weight, v_weight, output_weight = [
        torch.randn(8, 8, device=device, generator=generator) for _ in range(4)
    ]
    x = torch.randn(2, 4, 8, device=device, generator=generator)
    positions = torch.arange(4, device=device).expand(2, 4)

    output = run_multihead_self_attention_with_rope(
        8,
        2,
        8,
        10_000.0,
        q_weight,
        k_weight,
        v_weight,
        output_weight,
        x,
        positions,
    )

    assert output.device.type == "cuda"
    assert output.dtype == x.dtype
