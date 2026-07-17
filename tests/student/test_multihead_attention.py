from collections.abc import Callable
from typing import Any, Literal, cast

import pytest
import torch

from cs336_basics.nn import functional as F
from cs336_basics.nn.attention import MultiheadAttention, MultiheadSelfAttention, RotaryPositionalEmbedding


def test_mha_matches_torch_with_distinct_input_widths() -> None:
    torch.manual_seed(0)
    actual = MultiheadAttention(embed_dim=12, num_heads=3, kdim=5, vdim=7)
    expected = torch.nn.MultiheadAttention(
        embed_dim=12,
        num_heads=3,
        kdim=5,
        vdim=7,
        bias=False,
        batch_first=True,
    )
    q_weight = torch.randn(12, 12)
    k_weight = torch.randn(12, 5)
    v_weight = torch.randn(12, 7)
    output_weight = torch.randn(12, 12)
    actual.load_state_dict(
        {
            "q_proj.weight": q_weight,
            "k_proj.weight": k_weight,
            "v_proj.weight": v_weight,
            "output_proj.weight": output_weight,
        }
    )
    with torch.no_grad():
        expected.q_proj_weight.copy_(q_weight)
        expected.k_proj_weight.copy_(k_weight)
        expected.v_proj_weight.copy_(v_weight)
        expected.out_proj.weight.copy_(output_weight)

    query = torch.randn(2, 4, 12)
    key = torch.randn(2, 6, 5)
    value = torch.randn(2, 6, 7)

    expected_output, _ = expected(query, key, value, need_weights=False)
    torch.testing.assert_close(actual(query, key, value), expected_output)


def test_generalized_head_widths_and_leading_batch_axes() -> None:
    torch.manual_seed(1)
    module = MultiheadAttention(
        embed_dim=10,
        num_heads=3,
        qk_head_dim=4,
        value_head_dim=2,
    )
    query = torch.randn(2, 3, 4, 10)
    key = torch.randn(2, 3, 6, 10)
    value = torch.randn(2, 3, 6, 10)

    output = module(query, key, value)

    assert output.shape == (2, 3, 4, 10)
    assert module.in_proj_weight is not None
    assert module.in_proj_weight.shape == (30, 10)
    assert module.q_proj_weight is None
    assert module.k_proj_weight is None
    assert module.v_proj_weight is None
    assert module.output_proj.weight.shape == (10, 6)


def test_rope_infers_positions_and_broadcasts_over_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(2)
    rope = RotaryPositionalEmbedding(theta=10_000.0, d_k=4, max_seq_len=8)
    module = MultiheadAttention(embed_dim=8, num_heads=2, rope=rope)
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn(2, 3, 6, 8)
    value = torch.randn(2, 3, 6, 8)
    query_positions = torch.arange(4).unsqueeze(0)
    key_positions = torch.arange(6).unsqueeze(0)
    observed_shapes: list[tuple[torch.Size, torch.Size, torch.Size]] = []
    original: Callable[..., torch.Tensor] = F.scaled_dot_product_attention

    def traced(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        observed_shapes.append((query.shape, key.shape, value.shape))
        return original(query, key, value, *args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", traced)

    inferred = module(query, key, value)
    explicit = module(
        query,
        key,
        value,
        query_positions=query_positions,
        key_positions=key_positions,
    )

    torch.testing.assert_close(inferred, explicit)
    expected_shapes = (
        torch.Size((2, 3, 2, 4, 4)),
        torch.Size((2, 3, 2, 6, 4)),
        torch.Size((2, 3, 2, 6, 4)),
    )
    assert observed_shapes == [expected_shapes, expected_shapes]


def test_self_attention_is_causal_by_default() -> None:
    torch.manual_seed(3)
    module = MultiheadSelfAttention(d_model=8, num_heads=2)
    x = torch.randn(2, 5, 8)

    default = module(x)
    explicit = module(x, is_causal=True)
    noncausal = module(x, is_causal=False)

    torch.testing.assert_close(default, explicit)
    assert not torch.allclose(default, noncausal)


def test_self_attention_course_dimensions_are_read_only_mha_aliases() -> None:
    course_names = MultiheadSelfAttention(12, 3, 2, 5)
    mha_names = MultiheadSelfAttention(
        embed_dim=12,
        num_heads=3,
        qk_head_dim=2,
        value_head_dim=5,
    )

    for module in (course_names, mha_names):
        assert (module.d_model, module.d_k, module.d_v) == (12, 2, 5)
        assert (module.embed_dim, module.qk_head_dim, module.value_head_dim) == (12, 2, 5)
        assert (module.kdim, module.vdim) == (12, 12)

    assert MultiheadSelfAttention.d_model.fset is None
    assert MultiheadSelfAttention.d_k.fset is None
    assert MultiheadSelfAttention.d_v.fset is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 8, "embed_dim": 8, "num_heads": 2},
        {"d_model": 8, "num_heads": 2, "d_k": 4, "qk_head_dim": 4},
        {"d_model": 8, "num_heads": 2, "d_v": 4, "value_head_dim": 4},
    ],
)
def test_self_attention_rejects_conflicting_dimension_aliases(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        MultiheadSelfAttention(**kwargs)


def test_self_attention_requires_one_model_width_name() -> None:
    kwargs: dict[str, Any] = {"num_heads": 2}

    with pytest.raises(TypeError, match="d_model or embed_dim"):
        MultiheadSelfAttention(**kwargs)


def test_module_disables_attention_dropout_during_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[float] = []
    original: Callable[..., torch.Tensor] = F.scaled_dot_product_attention

    def traced(*args: Any, dropout_p: float = 0.0, **kwargs: Any) -> torch.Tensor:
        observed.append(dropout_p)
        return original(*args, dropout_p=dropout_p, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", traced)
    module = MultiheadSelfAttention(d_model=8, num_heads=2, dropout=0.25)
    x = torch.randn(2, 4, 8)

    module.eval()
    module(x)
    module.train()
    module(x)

    assert observed == [0.0, 0.25]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"embed_dim": 7, "num_heads": 2}, "divisible"),
        ({"embed_dim": 8, "num_heads": 2, "dropout": 1.1}, "dropout"),
        ({"embed_dim": 8, "num_heads": 2, "kdim": 0}, "kdim"),
        ({"embed_dim": 8, "num_heads": 2, "qk_head_dim": 0}, "qk_head_dim"),
    ],
)
def test_constructor_validation(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MultiheadAttention(**kwargs)


def test_rejects_rope_width_mismatch() -> None:
    rope = RotaryPositionalEmbedding(theta=10_000.0, d_k=6, max_seq_len=8)

    with pytest.raises(ValueError, match="RoPE width"):
        MultiheadAttention(embed_dim=8, num_heads=2, rope=rope)


def test_rejects_incompatible_input_shapes() -> None:
    module = MultiheadAttention(embed_dim=8, num_heads=2, kdim=5, vdim=7)

    with pytest.raises(ValueError, match="sequence lengths"):
        module(torch.randn(2, 4, 8), torch.randn(2, 6, 5), torch.randn(2, 5, 7))

    with pytest.raises(ValueError, match="widths"):
        module(torch.randn(2, 4, 8), torch.randn(2, 6, 6), torch.randn(2, 6, 7))


def test_uses_separate_owned_weights_for_distinct_input_widths() -> None:
    module = MultiheadAttention(
        embed_dim=10,
        num_heads=3,
        kdim=5,
        vdim=7,
        qk_head_dim=4,
        value_head_dim=2,
    )

    assert module.in_proj_weight is None
    assert module.q_proj_weight is not None and module.q_proj_weight.shape == (12, 10)
    assert module.k_proj_weight is not None and module.k_proj_weight.shape == (12, 5)
    assert module.v_proj_weight is not None and module.v_proj_weight.shape == (6, 7)


def test_packed_projection_uses_identity_sensitive_matmul_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    module = MultiheadAttention(embed_dim=8, num_heads=2)
    original: Callable[..., torch.Tensor] = F.linear
    calls: list[tuple[int, ...]] = []

    def traced(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(weight.shape))
        return original(input, weight)

    monkeypatch.setattr(F, "linear", traced)
    query = torch.randn(2, 4, 8)
    key_value = torch.randn(2, 6, 8)

    module(query, query, query)
    assert len(calls) == 2  # one packed QKV projection and one output projection

    calls.clear()
    module(query, key_value, key_value)
    assert len(calls) == 3  # Q, packed KV, and output

    calls.clear()
    module(query, key_value, key_value.clone())
    assert len(calls) == 4  # Q, K, V, and output


def test_load_state_dict_translates_delegated_weights_into_packed_storage() -> None:
    module = MultiheadAttention(embed_dim=8, num_heads=2)
    q_weight = torch.randn(8, 8)
    k_weight = torch.randn(8, 8)
    v_weight = torch.randn(8, 8)
    output_weight = torch.randn(8, 8)

    result = module.load_state_dict(
        {
            "q_proj.weight": q_weight,
            "k_proj.weight": k_weight,
            "v_proj.weight": v_weight,
            "output_proj.weight": output_weight,
        }
    )

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert module.in_proj_weight is not None
    torch.testing.assert_close(module.in_proj_weight, torch.cat((q_weight, k_weight, v_weight)))
    torch.testing.assert_close(module.output_proj.weight, output_weight)


def test_load_state_dict_translates_delegated_weights_into_separate_storage() -> None:
    module = MultiheadAttention(embed_dim=8, num_heads=2, kdim=5, vdim=7)
    q_weight = torch.randn(8, 8)
    k_weight = torch.randn(8, 5)
    v_weight = torch.randn(8, 7)

    result = module.load_state_dict(
        {
            "q_proj.weight": q_weight,
            "k_proj.weight": k_weight,
            "v_proj.weight": v_weight,
            "output_proj.weight": torch.randn(8, 8),
        }
    )

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert module.q_proj_weight is not None
    assert module.k_proj_weight is not None
    assert module.v_proj_weight is not None
    torch.testing.assert_close(module.q_proj_weight, q_weight)
    torch.testing.assert_close(module.k_proj_weight, k_weight)
    torch.testing.assert_close(module.v_proj_weight, v_weight)


@pytest.mark.parametrize(
    "module",
    [
        MultiheadAttention(embed_dim=8, num_heads=2),
        MultiheadAttention(embed_dim=8, num_heads=2, kdim=5, vdim=7),
    ],
)
def test_load_state_dict_accepts_native_layout(module: MultiheadAttention) -> None:
    clone = MultiheadAttention(
        embed_dim=module.embed_dim,
        num_heads=module.num_heads,
        kdim=module.kdim,
        vdim=module.vdim,
        qk_head_dim=module.qk_head_dim,
        value_head_dim=module.value_head_dim,
    )

    result = clone.load_state_dict(module.state_dict())

    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_load_state_dict_rejects_conflicting_projection_layouts() -> None:
    module = MultiheadAttention(embed_dim=8, num_heads=2)

    with pytest.raises(ValueError, match="conflicting"):
        module.load_state_dict(
            {
                "in_proj_weight": torch.randn(24, 8),
                "q_proj.weight": torch.randn(8, 8),
            },
            strict=False,
        )


def test_load_state_dict_preserves_missing_and_unexpected_key_reporting() -> None:
    module = MultiheadAttention(embed_dim=8, num_heads=2)

    result = module.load_state_dict(
        {
            "output_proj.weight": torch.randn(8, 8),
            "surprise": torch.tensor(1),
        },
        strict=False,
    )

    assert result.missing_keys == ["in_proj_weight"]
    assert result.unexpected_keys == ["surprise"]


@pytest.mark.parametrize("delegated", [False, True])
def test_load_state_dict_loads_partial_weights_into_separate_storage(delegated: bool) -> None:
    module = MultiheadAttention(embed_dim=8, num_heads=2, kdim=5, vdim=7)
    q_weight = torch.randn(8, 8)
    key = "q_proj.weight" if delegated else "q_proj_weight"

    result = module.load_state_dict({key: q_weight}, strict=False)

    assert result.missing_keys == ["k_proj_weight", "v_proj_weight", "output_proj.weight"]
    assert result.unexpected_keys == []
    assert module.q_proj_weight is not None
    torch.testing.assert_close(module.q_proj_weight, q_weight)


@pytest.mark.parametrize("delegated", [False, True])
def test_load_state_dict_rejects_incomplete_compatibility_layout_for_packed_storage(delegated: bool) -> None:
    module = MultiheadAttention(embed_dim=8, num_heads=2)
    key = "q_proj.weight" if delegated else "q_proj_weight"

    with pytest.raises(ValueError, match="incomplete"):
        module.load_state_dict({key: torch.randn(8, 8)}, strict=False)


@pytest.mark.parametrize("use_rope", [False, True])
@pytest.mark.parametrize("use_matrix_form", [False, True])
def test_head_sequence_layouts_match_with_masks_and_gradients(
    use_rope: bool,
    use_matrix_form: bool,
) -> None:
    torch.manual_seed(4)
    rope_before = RotaryPositionalEmbedding(10_000.0, 4, 16, use_matrix_form=use_matrix_form) if use_rope else None
    rope_after = RotaryPositionalEmbedding(10_000.0, 4, 16, use_matrix_form=use_matrix_form) if use_rope else None
    before = MultiheadAttention(
        embed_dim=10,
        num_heads=3,
        kdim=7,
        vdim=9,
        qk_head_dim=4,
        value_head_dim=2,
        rope=rope_before,
        _layout_strategy="head_before_sequence",
    )
    after = MultiheadAttention(
        embed_dim=10,
        num_heads=3,
        kdim=7,
        vdim=9,
        qk_head_dim=4,
        value_head_dim=2,
        rope=rope_after,
        _layout_strategy="head_after_sequence",
    )
    after.load_state_dict(before.state_dict())
    query_before = torch.randn(2, 3, 4, 10, requires_grad=True)
    key_before = torch.randn(2, 3, 6, 7, requires_grad=True)
    value_before = torch.randn(2, 3, 6, 9, requires_grad=True)
    query_after = query_before.detach().clone().requires_grad_()
    key_after = key_before.detach().clone().requires_grad_()
    value_after = value_before.detach().clone().requires_grad_()
    mask = torch.ones(1, 4, 6, dtype=torch.bool)
    mask[..., -1] = False

    output_before = before(query_before, key_before, value_before, mask)
    output_after = after(query_after, key_after, value_after, mask)
    torch.testing.assert_close(output_before, output_after)

    output_before.square().sum().backward()
    output_after.square().sum().backward()
    for before_grad, after_grad in (
        (query_before.grad, query_after.grad),
        (key_before.grad, key_after.grad),
        (value_before.grad, value_after.grad),
    ):
        torch.testing.assert_close(before_grad, after_grad)


@pytest.mark.parametrize(
    ("strategy", "expected_rotation_calls", "expected_stack_calls"),
    [("auto", 1, 0), ("separate", 2, 0), ("stacked", 1, 1)],
)
def test_self_attention_qk_execution_strategies(
    strategy: Literal["auto", "separate", "stacked"],
    expected_rotation_calls: int,
    expected_stack_calls: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Matrix form avoids counting the elementwise implementation's internal pair stack.
    rope = RotaryPositionalEmbedding(10_000.0, 4, 16, use_matrix_form=True)
    module = MultiheadSelfAttention(
        8,
        2,
        rope=rope,
        _qk_execution_strategy=strategy,
    )
    rotation_calls = 0
    stack_calls = 0
    original_apply: Callable[..., torch.Tensor] = rope._apply_rotations
    original_stack: Callable[..., torch.Tensor] = torch.stack

    def traced_apply(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal rotation_calls
        rotation_calls += 1
        return original_apply(*args, **kwargs)

    def traced_stack(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal stack_calls
        stack_calls += 1
        return original_stack(*args, **kwargs)

    monkeypatch.setattr(rope, "_apply_rotations", traced_apply)
    monkeypatch.setattr(torch, "stack", traced_stack)
    module(torch.randn(2, 5, 8))

    assert rotation_calls == expected_rotation_calls
    assert stack_calls == expected_stack_calls


def test_auto_qk_execution_falls_back_for_cross_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    rope = RotaryPositionalEmbedding(10_000.0, 4, 16)
    module = MultiheadAttention(8, 2, kdim=5, vdim=7, rope=rope, _qk_execution_strategy="auto")
    calls = 0
    original: Callable[..., torch.Tensor] = rope._apply_rotations

    def traced(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(rope, "_apply_rotations", traced)
    module(torch.randn(2, 4, 8), torch.randn(2, 6, 5), torch.randn(2, 6, 7))

    assert calls == 2


def test_packed_self_projection_exposes_zero_copy_qk_head_view() -> None:
    module = MultiheadAttention(8, 2)
    x = torch.randn(2, 5, 8)

    projected_q, projected_k, projected_v, projected_qk = module._in_projection(x, x, x)
    _, _, _, packed_qk = module._split_heads(projected_q, projected_k, projected_v, projected_qk)

    assert projected_qk is not None and packed_qk is not None
    assert projected_qk.untyped_storage().data_ptr() == packed_qk.untyped_storage().data_ptr()
    assert packed_qk.shape == (2, 2, 2, 5, 4)


@pytest.mark.parametrize("strategy", ["auto", "separate", "stacked"])
def test_packed_self_attention_layouts_match_with_head_dependent_positions_and_gradients(
    strategy: Literal["auto", "separate", "stacked"],
) -> None:
    rope_before = RotaryPositionalEmbedding(10_000.0, 4, 16, use_matrix_form=False)
    rope_after = RotaryPositionalEmbedding(10_000.0, 4, 16, use_matrix_form=False)
    before = MultiheadSelfAttention(
        10,
        3,
        4,
        2,
        rope=rope_before,
        _layout_strategy="head_before_sequence",
        _qk_execution_strategy=strategy,
    )
    after = MultiheadSelfAttention(
        10,
        3,
        4,
        2,
        rope=rope_after,
        _layout_strategy="head_after_sequence",
        _qk_execution_strategy=strategy,
    )
    after.load_state_dict(before.state_dict())
    x_before = torch.randn(2, 3, 5, 10, requires_grad=True)
    x_after = x_before.detach().clone().requires_grad_()
    positions = torch.stack((torch.arange(5), torch.arange(5).flip(0), torch.zeros(5, dtype=torch.int64)))
    mask = torch.ones(3, 5, 5, dtype=torch.bool)
    mask[:, :, -1] = False

    output_before = before(x_before, mask, token_positions=positions, is_causal=False)
    output_after = after(x_after, mask, token_positions=positions, is_causal=False)
    torch.testing.assert_close(output_before, output_after)

    output_before.square().sum().backward()
    output_after.square().sum().backward()
    torch.testing.assert_close(x_before.grad, x_after.grad)


def test_head_after_sequence_joins_attended_heads_as_a_view(monkeypatch: pytest.MonkeyPatch) -> None:
    module = MultiheadSelfAttention(8, 2, _layout_strategy="head_after_sequence")
    observed: list[tuple[tuple[int, ...], bool]] = []
    original: Callable[..., torch.Tensor] = module.output_proj.forward

    def traced(input: torch.Tensor) -> torch.Tensor:
        observed.append((input.stride(), input.is_contiguous()))
        return original(input)

    monkeypatch.setattr(module.output_proj, "forward", traced)
    module(torch.randn(2, 5, 8))

    assert observed == [((40, 8, 1), True)]


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("_layout_strategy", "sequence_first", "layout strategy"),
        ("_qk_execution_strategy", "combined", "execution strategy"),
    ],
)
def test_rejects_unknown_experimental_strategies(keyword: str, value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cast(Any, MultiheadAttention)(8, 2, **{keyword: value})
