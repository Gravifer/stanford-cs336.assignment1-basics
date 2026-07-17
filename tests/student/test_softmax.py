import pytest
import torch
import torch.nn.functional as torch_f

from cs336_basics.nn import functional as F
from cs336_basics.nn.modules import SoftMax


@pytest.mark.parametrize("shape", [(), (5,), (2, 3), (2, 3, 4), (2, 3, 4, 5)])
def test_softmax_none_matches_torch_implicit_dimension(shape: tuple[int, ...]) -> None:
    input = torch.randn(shape)

    with pytest.warns(UserWarning, match="Implicit dimension choice"):
        expected = torch_f.softmax(input, dim=None)
    with pytest.warns(UserWarning, match="Implicit dimension choice"):
        actual = F.softmax(input, dim=None)

    torch.testing.assert_close(actual, expected)


def test_softmax_dtype_controls_computation_and_output_dtype() -> None:
    input = torch.tensor([[0.1234, 1.234, -2.345], [7.89, -6.78, 5.67]], dtype=torch.float16)

    actual = F.softmax(input, dim=1, dtype=torch.float32)
    expected = torch_f.softmax(input, dim=1, dtype=torch.float32)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_softmax_module_preserves_none_dimension_semantics() -> None:
    input = torch.randn(2, 3, 4)
    module = SoftMax()

    assert module.dim is None
    with pytest.warns(UserWarning, match="Implicit dimension choice"):
        actual = module(input)
    with pytest.warns(UserWarning, match="Implicit dimension choice"):
        expected = torch.nn.Softmax(dim=None)(input)

    torch.testing.assert_close(actual, expected)
