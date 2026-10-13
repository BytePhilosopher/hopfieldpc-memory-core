"""Tests for HopfieldMemory memory mode options.

Validates three memory modes:
- External memory passed during forward
- Internal fixed memory (registered buffer, no gradient)
- Internal learnable memory (nn.Parameter, requires_grad=True)
"""

import pytest
import torch
from torch import nn

from hopfieldpc import HopfieldMemory


def test_external_memory_works() -> None:
    module = HopfieldMemory(dim=4, beta=1.0)
    query = torch.randn(8, 4)
    memory = torch.randn(10, 4)

    output = module(query, memory)

    assert output.retrieved.shape == (8, 4)
    assert output.weights.shape == (8, 10)
    assert module.memory is None


def test_external_memory_takes_precedence_over_internal() -> None:
    module = HopfieldMemory(dim=4, memory_size=3, beta=1.0)
    query = torch.randn(2, 4)
    external = torch.randn(7, 4)

    _, weights, diag = module(query, external)

    assert weights.shape == (2, 7)
    assert diag["memory_shape"] == (7, 4)


def test_internal_fixed_memory_works() -> None:
    module = HopfieldMemory(dim=4, memory_size=5, learnable_memory=False)
    query = torch.randn(8, 4)

    output = module(query)

    assert output.retrieved.shape == (8, 4)
    assert output.weights.shape == (8, 5)


def test_internal_fixed_memory_is_buffer() -> None:
    module = HopfieldMemory(dim=4, memory_size=5, learnable_memory=False)

    assert isinstance(module.memory, torch.Tensor)
    assert not isinstance(module.memory, nn.Parameter)


def test_fixed_memory_does_not_require_gradient() -> None:
    module = HopfieldMemory(dim=4, memory_size=5, learnable_memory=False)

    assert module.memory.requires_grad is False


def test_fixed_memory_shape() -> None:
    module = HopfieldMemory(dim=8, memory_size=12, learnable_memory=False)

    assert module.memory.shape == (12, 8)


def test_internal_learnable_memory_works() -> None:
    module = HopfieldMemory(dim=4, memory_size=5, learnable_memory=True)
    query = torch.randn(8, 4)

    output = module(query)

    assert output.retrieved.shape == (8, 4)
    assert output.weights.shape == (8, 5)


def test_learnable_memory_is_parameter() -> None:
    module = HopfieldMemory(dim=4, memory_size=5, learnable_memory=True)

    assert isinstance(module.memory, nn.Parameter)


def test_learnable_memory_has_requires_grad() -> None:
    module = HopfieldMemory(dim=4, memory_size=5, learnable_memory=True)

    assert module.memory.requires_grad is True


def test_learnable_memory_shape() -> None:
    module = HopfieldMemory(dim=8, memory_size=12, learnable_memory=True)

    assert module.memory.shape == (12, 8)


def test_learnable_memory_appears_in_parameters() -> None:
    module = HopfieldMemory(dim=4, memory_size=5, learnable_memory=True)
    param_list = list(module.parameters())

    assert len(param_list) == 1
    assert param_list[0] is module.memory


def test_fixed_memory_not_in_parameters() -> None:
    module = HopfieldMemory(dim=4, memory_size=5, learnable_memory=False)
    param_list = list(module.parameters())

    assert len(param_list) == 0


def test_no_memory_raises_on_forward() -> None:
    module = HopfieldMemory(dim=4)
    query = torch.randn(2, 4)

    with pytest.raises(ValueError, match="No memory was provided"):
        module(query)


def test_learnable_memory_requires_memory_size() -> None:
    with pytest.raises(ValueError, match="learnable_memory=True requires"):
        HopfieldMemory(dim=4, learnable_memory=True)


def test_learnable_memory_gradient_flows() -> None:
    module = HopfieldMemory(dim=3, memory_size=4, learnable_memory=True, beta=1.0)
    query = torch.randn(2, 3)

    output = module(query)
    loss = output.retrieved.sum()
    loss.backward()

    assert module.memory.grad is not None
    assert torch.isfinite(module.memory.grad).all()
