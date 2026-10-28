"""Numerical-stability tests for Hopfield retrieval."""

import math
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hopfieldpc import (
    HopfieldMemory,
    attention_entropy,
    hopfield_energy,
    hopfield_retrieve,
    multi_step_retrieve,
)

DTYPE_TOLERANCES = {
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
    torch.float32: (1e-6, 1e-5),
    torch.float64: (1e-12, 1e-10),
}


def _assert_probability_distribution(weights: torch.Tensor) -> None:
    atol, rtol = DTYPE_TOLERANCES[weights.dtype]
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    assert (weights <= 1).all()
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones(weights.shape[0], device=weights.device, dtype=weights.dtype),
        atol=atol,
        rtol=rtol,
    )


def _assert_diagnostics_finite(diagnostics: dict[str, Any]) -> None:
    for value in diagnostics.values():
        tensors = value if isinstance(value, list) else [value]
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                assert torch.isfinite(tensor).all()


@pytest.mark.parametrize("beta", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
@pytest.mark.parametrize("normalize", [False, True])
def test_retrieval_is_finite_across_beta_sweep(beta: float, normalize: bool) -> None:
    generator = torch.Generator().manual_seed(42)
    query = torch.randn(8, 64, generator=generator)
    memory = torch.randn(20, 64, generator=generator)

    retrieved, weights = hopfield_retrieve(
        query,
        memory,
        beta=beta,
        normalize=normalize,
    )

    assert torch.isfinite(retrieved).all()
    _assert_probability_distribution(weights)


@pytest.mark.parametrize("beta", [20.0, 50.0, 100.0])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("num_updates", [1, 3])
def test_module_diagnostics_are_finite(
    beta: float,
    normalize: bool,
    num_updates: int,
) -> None:
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(4, 16, generator=generator)
    memory = torch.randn(10, 16, generator=generator)
    module = HopfieldMemory(
        dim=16,
        beta=beta,
        normalize=normalize,
        num_updates=num_updates,
        return_states=True,
    )

    output = module(query, memory)

    assert torch.isfinite(output.retrieved).all()
    _assert_probability_distribution(output.weights)
    _assert_diagnostics_finite(output.diagnostics)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_large_low_precision_scores_use_safe_working_precision(
    dtype: torch.dtype,
) -> None:
    query = torch.full((1, 64), 5.0, dtype=dtype)
    memory = torch.stack(
        [
            torch.full((64,), 5.0, dtype=dtype),
            torch.zeros(64, dtype=dtype),
        ]
    )

    retrieved, weights = hopfield_retrieve(query, memory, beta=100.0)

    assert retrieved.dtype == dtype
    assert weights.dtype == dtype
    assert torch.isfinite(retrieved).all()
    _assert_probability_distribution(weights)
    assert weights.argmax(dim=-1).item() == 0


def test_centering_prevents_beta_scaling_overflow() -> None:
    query = torch.tensor([[1e19, 0.0]])
    memory = torch.tensor([[1e18, 0.0], [0.0, 1.0]])

    retrieved, weights = hopfield_retrieve(query, memory, beta=100.0)

    assert torch.isfinite(retrieved).all()
    assert torch.equal(weights, torch.tensor([[1.0, 0.0]]))


def test_energy_is_finite_when_beta_scaling_would_overflow() -> None:
    query = torch.tensor([[1e19, 0.0]])
    memory = torch.tensor([[1e18, 0.0], [0.0, 1.0]])

    energy = hopfield_energy(query, memory, beta=100.0)

    assert torch.isfinite(energy).all()
    expected = -(query @ memory.T).amax(dim=-1) + 0.5 * (query ** 2).sum(dim=-1)
    torch.testing.assert_close(energy, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_energy_is_finite_for_large_scores(dtype: torch.dtype) -> None:
    query = torch.full((1, 64), 5.0, dtype=dtype)
    memory = torch.stack(
        [
            torch.full((64,), 5.0, dtype=dtype),
            torch.zeros(64, dtype=dtype),
        ]
    )

    energy = hopfield_energy(query, memory, beta=100.0)

    assert energy.dtype == dtype
    # -1600 from the best match, +800 from the query term.
    torch.testing.assert_close(energy, torch.tensor([-800.0], dtype=dtype))


@pytest.mark.parametrize("beta", [0.1, 1.0, 10.0])
def test_energy_matches_float64_reference(beta: float) -> None:
    generator = torch.Generator().manual_seed(5)
    query = torch.randn(
        8, 16, generator=generator, dtype=torch.float64, requires_grad=True
    )
    memory = torch.randn(20, 16, generator=generator, dtype=torch.float64)

    energy = hopfield_energy(query, memory, beta=beta)
    reference = (
        -torch.logsumexp(beta * query @ memory.T, dim=-1) / beta
        + 0.5 * (query ** 2).sum(dim=-1)
    )
    (grad,) = torch.autograd.grad(energy.sum(), query)
    (reference_grad,) = torch.autograd.grad(reference.sum(), query)

    torch.testing.assert_close(energy, reference, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(grad, reference_grad, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("beta", [0.5, 2.0, 10.0])
def test_stabilized_retrieval_matches_float64_reference(
    beta: float,
    normalize: bool,
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator().manual_seed(11)
    query = (torch.randn(4, 8, generator=generator) * 0.1).to(dtype)
    memory = (torch.randn(6, 8, generator=generator) * 0.1).to(dtype)

    retrieved, weights = hopfield_retrieve(
        query,
        memory,
        beta=beta,
        normalize=normalize,
    )

    query64 = query.double()
    memory64 = memory.double()
    if normalize:
        query_scores = F.normalize(query64, dim=-1)
        memory_scores = F.normalize(memory64, dim=-1)
    else:
        query_scores, memory_scores = query64, memory64
    reference_weights = torch.softmax(beta * query_scores @ memory_scores.T, dim=-1)
    reference_retrieved = reference_weights @ memory64

    atol, rtol = {
        torch.float32: (1e-6, 1e-5),
        torch.float64: (1e-12, 1e-10),
    }[dtype]
    assert weights.dtype == dtype
    assert retrieved.dtype == dtype
    torch.testing.assert_close(
        weights.double(), reference_weights, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        retrieved.double(), reference_retrieved, atol=atol, rtol=rtol
    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_zero_query_has_uniform_normalized_weights(dtype: torch.dtype) -> None:
    query = torch.zeros((2, 4), dtype=dtype)
    memory = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=dtype,
    )

    retrieved, weights = hopfield_retrieve(
        query,
        memory,
        beta=100.0,
        normalize=True,
    )

    expected_weights = torch.full((2, 2), 0.5, dtype=dtype)
    atol, rtol = DTYPE_TOLERANCES[dtype]
    torch.testing.assert_close(weights, expected_weights, atol=atol, rtol=rtol)
    assert torch.isfinite(retrieved).all()


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("query_fill", [0.0, 1.0])
def test_all_zero_memory_is_finite_with_normalization(
    query_fill: float,
    dtype: torch.dtype,
) -> None:
    query = torch.full((2, 4), query_fill, dtype=dtype)
    memory = torch.zeros((3, 4), dtype=dtype)

    retrieved, weights = hopfield_retrieve(
        query,
        memory,
        beta=100.0,
        normalize=True,
    )

    assert torch.equal(retrieved, torch.zeros_like(retrieved))
    _assert_probability_distribution(weights)
    atol, rtol = DTYPE_TOLERANCES[dtype]
    torch.testing.assert_close(
        weights,
        torch.full((2, 3), 1.0 / 3.0, dtype=dtype),
        atol=atol,
        rtol=rtol,
    )


def test_normalized_weights_are_scale_invariant() -> None:
    generator = torch.Generator().manual_seed(23)
    query = torch.randn(4, 8, generator=generator)
    memory = torch.randn(6, 8, generator=generator)

    _, weights = hopfield_retrieve(query, memory, beta=10.0, normalize=True)
    _, scaled_weights = hopfield_retrieve(
        query * 10_000.0,
        memory * 10_000.0,
        beta=10.0,
        normalize=True,
    )

    torch.testing.assert_close(weights, scaled_weights, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("scale", [1.0, 100.0, 10_000.0])
@pytest.mark.parametrize("normalize", [False, True])
def test_large_float32_norms_remain_finite(scale: float, normalize: bool) -> None:
    generator = torch.Generator().manual_seed(31)
    query = torch.randn(4, 16, generator=generator) * scale
    memory = torch.randn(10, 16, generator=generator) * scale

    retrieved, weights = hopfield_retrieve(
        query,
        memory,
        beta=100.0,
        normalize=normalize,
    )

    assert torch.isfinite(retrieved).all()
    _assert_probability_distribution(weights)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_internal_low_precision_memory_path_is_finite(dtype: torch.dtype) -> None:
    module = HopfieldMemory(dim=64, memory_size=2, beta=100.0).to(dtype=dtype)
    assert module.memory is not None
    with torch.no_grad():
        module.memory[0].fill_(5.0)
        module.memory[1].zero_()
    query = torch.full((1, 64), 5.0, dtype=dtype)

    output = module(query)

    assert output.retrieved.dtype == dtype
    assert output.weights.dtype == dtype
    assert torch.isfinite(output.retrieved).all()
    _assert_probability_distribution(output.weights)
    _assert_diagnostics_finite(output.diagnostics)


def test_internal_memory_dtype_mismatch_raises() -> None:
    module = HopfieldMemory(dim=4, memory_size=3)
    query = torch.randn(2, 4, dtype=torch.float16)

    with pytest.raises(TypeError, match="must share a dtype"):
        module(query)


def test_external_memory_dtype_mismatch_raises() -> None:
    query = torch.randn(2, 4, dtype=torch.float32)
    memory = torch.randn(3, 4, dtype=torch.float64)

    with pytest.raises(TypeError, match="must share a dtype"):
        hopfield_retrieve(query, memory)


@pytest.mark.parametrize(
    "dtype", [torch.int64, torch.int32, torch.bool, torch.complex64]
)
def test_non_floating_tensors_are_rejected(dtype: torch.dtype) -> None:
    query = torch.ones((2, 4), dtype=dtype)
    memory = torch.ones((3, 4), dtype=dtype)

    with pytest.raises(TypeError, match="must be floating-point tensors"):
        hopfield_retrieve(query, memory)

    with pytest.raises(TypeError, match="must be floating-point tensors"):
        HopfieldMemory(dim=4)(query, memory)

    with pytest.raises(TypeError, match="must be floating-point tensors"):
        hopfield_energy(query, memory)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_multi_step_states_are_finite(dtype: torch.dtype) -> None:
    query = torch.full((1, 64), 5.0, dtype=dtype)
    memory = torch.stack(
        [
            torch.full((64,), 5.0, dtype=dtype),
            torch.zeros(64, dtype=dtype),
        ]
    )

    retrieved, weights, states = multi_step_retrieve(
        query,
        memory,
        beta=100.0,
        num_updates=10,
        return_states=True,
    )

    assert len(states) == 11
    for state in states:
        assert state.dtype == dtype
        assert state.shape == query.shape
        assert torch.isfinite(state).all()
    assert retrieved.dtype == dtype
    assert torch.isfinite(retrieved).all()
    _assert_probability_distribution(weights)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("normalize", [False, True])
def test_large_beta_gradients_are_finite(
    normalize: bool,
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator().manual_seed(47)
    # Unscaled vectors saturate the softmax at beta=100.
    query = torch.randn(4, 64, generator=generator).to(dtype).requires_grad_()
    memory = torch.randn(10, 64, generator=generator).to(dtype).requires_grad_()

    retrieved, weights = hopfield_retrieve(
        query,
        memory,
        beta=100.0,
        normalize=normalize,
    )
    (retrieved.float().square().mean() + weights.float().square().mean()).backward()

    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert memory.grad is not None and torch.isfinite(memory.grad).all()


def test_large_beta_learnable_memory_gradient_is_finite() -> None:
    module = HopfieldMemory(
        dim=4,
        memory_size=3,
        beta=100.0,
        learnable_memory=True,
    )
    query = torch.randn(2, 4) * 0.1

    module(query).retrieved.square().mean().backward()

    assert module.memory is not None
    assert module.memory.grad is not None
    assert torch.isfinite(module.memory.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_entropy_and_gradients_are_finite(dtype: torch.dtype) -> None:
    weights = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype, requires_grad=True)

    entropy = attention_entropy(weights)
    normalized_entropy = attention_entropy(weights, normalize=True)
    (entropy + normalized_entropy).sum().backward()

    assert entropy.dtype == torch.float32
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(normalized_entropy).all()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_uniform_entropy_is_correct(dtype: torch.dtype) -> None:
    weights = torch.full((1, 4), 0.25, dtype=dtype)

    entropy = attention_entropy(weights)
    normalized_entropy = attention_entropy(weights, normalize=True)

    torch.testing.assert_close(
        entropy,
        torch.full_like(entropy, math.log(4.0)),
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        normalized_entropy,
        torch.ones_like(normalized_entropy),
        atol=1e-6,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "beta",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, "1.0"],
)
def test_invalid_beta_is_rejected_consistently(beta: Any) -> None:
    query = torch.randn(2, 4)
    memory = torch.randn(3, 4)

    with pytest.raises(ValueError, match="beta must be positive and finite"):
        hopfield_retrieve(query, memory, beta=beta)

    with pytest.raises(ValueError, match="beta must be positive and finite"):
        HopfieldMemory(dim=4, beta=beta)

    with pytest.raises(ValueError, match="beta must be positive and finite"):
        hopfield_energy(query, memory, beta=beta)


@pytest.mark.parametrize(
    "beta",
    [1, 1.0, np.float32(1.0), np.float64(1.0)],
)
def test_real_numeric_beta_types_are_accepted(beta: Any) -> None:
    query = torch.randn(2, 4)
    memory = torch.randn(3, 4)

    retrieved, weights = hopfield_retrieve(query, memory, beta=beta)
    module = HopfieldMemory(dim=4, beta=beta)
    energy = hopfield_energy(query, memory, beta=beta)

    assert module.beta == 1.0
    assert torch.isfinite(retrieved).all()
    _assert_probability_distribution(weights)
    assert torch.isfinite(energy).all()
