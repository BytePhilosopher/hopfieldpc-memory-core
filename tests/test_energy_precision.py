"""Working-precision regression tests for Hopfield energy."""

import math

import pytest
import torch

from hopfieldpc import hopfield_energy

LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
ALL_DTYPES = (*LOW_PRECISION_DTYPES, torch.float32, torch.float64)


def _reference_energy(
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    beta: float,
    include_constants: bool,
    reduce: str,
) -> torch.Tensor:
    """Independent float64 formula for the controlled inputs in these tests."""
    q = query.to(dtype=torch.float64)
    m = memory.to(dtype=torch.float64)
    energy = (
        0.5 * q.square().sum(dim=-1)
        - torch.logsumexp(beta * (q @ m.T), dim=-1) / beta
    )
    if include_constants:
        energy = energy + math.log(m.shape[0]) / beta
        energy = energy + 0.5 * m.square().sum(dim=-1).max()
    return energy.mean() if reduce == "mean" else energy


@pytest.mark.parametrize("dtype", LOW_PRECISION_DTYPES)
@pytest.mark.parametrize("include_constants", [False, True])
@pytest.mark.parametrize("reduce", ["none", "mean"])
@pytest.mark.parametrize(
    "query_values,memory_values,expected_without,expected_with",
    [
        ([[200.0, 200.0]], [[0.0, 0.0]], 40000.0, 40000.0),
        ([[0.0, 0.0]], [[200.0, 200.0]], 0.0, 40000.0),
        ([[200.0, 200.0]], [[200.0, 200.0]], -40000.0, 0.0),
    ],
    ids=["query_norm", "memory_norm", "similarity_and_norms"],
)
def test_low_precision_energy_handles_large_intermediates(
    dtype: torch.dtype,
    include_constants: bool,
    reduce: str,
    query_values: list[list[float]],
    memory_values: list[list[float]],
    expected_without: float,
    expected_with: float,
) -> None:
    query = torch.tensor(query_values, dtype=dtype)
    memory = torch.tensor(memory_values, dtype=dtype)
    result = hopfield_energy(
        query, memory, include_constants=include_constants, reduce=reduce
    )
    value = expected_with if include_constants else expected_without
    expected = torch.tensor([value] if reduce == "none" else value, dtype=dtype)

    assert result.dtype == dtype
    assert result.device == query.device
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", LOW_PRECISION_DTYPES)
@pytest.mark.parametrize("include_constants", [False, True])
def test_energy_mean_is_computed_before_output_cast(
    dtype: torch.dtype, include_constants: bool
) -> None:
    # Per-query values are 80000 and 0; only their mean fits in float16.
    query = torch.tensor([[400.0, 0.0], [0.0, 0.0]], dtype=dtype)
    memory = torch.zeros((1, 2), dtype=dtype)
    result = hopfield_energy(
        query, memory, include_constants=include_constants, reduce="mean"
    )

    assert result.ndim == 0
    assert result.dtype == dtype
    assert torch.isfinite(result)
    torch.testing.assert_close(
        result, torch.tensor(40000.0, dtype=dtype), atol=0, rtol=0
    )


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("include_constants", [False, True])
@pytest.mark.parametrize("reduce", ["none", "mean"])
def test_energy_precision_matches_float64_reference(
    dtype: torch.dtype, include_constants: bool, reduce: str
) -> None:
    query = torch.tensor([[0.1, -0.3], [0.9, 0.2]], dtype=dtype)
    memory = torch.tensor([[-0.2, 0.7], [0.8, -0.4], [1.3, 1.1]], dtype=dtype)
    result = hopfield_energy(
        query, memory, beta=2.0, include_constants=include_constants, reduce=reduce
    )
    reference = _reference_energy(
        query, memory, beta=2.0, include_constants=include_constants, reduce=reduce
    ).to(dtype=dtype)
    tolerance = {
        torch.float16: 1e-3,
        torch.bfloat16: 1e-2,
        torch.float32: 1e-6,
        torch.float64: 1e-12,
    }[dtype]

    assert result.dtype == dtype
    assert result.device == query.device
    torch.testing.assert_close(result, reference, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", LOW_PRECISION_DTYPES)
@pytest.mark.parametrize("include_constants", [False, True])
@pytest.mark.parametrize("reduce", ["none", "mean"])
def test_energy_precision_preserves_autograd(
    dtype: torch.dtype, include_constants: bool, reduce: str
) -> None:
    query = torch.tensor(
        [[200.0, 200.0], [150.0, -100.0]], dtype=dtype, requires_grad=True
    )
    memory = torch.tensor(
        [[0.5, -0.25], [-0.25, 0.75]], dtype=dtype, requires_grad=True
    )
    result = hopfield_energy(
        query, memory, beta=0.5, include_constants=include_constants, reduce=reduce
    )
    assert torch.isfinite(result).all()
    actual_grads = torch.autograd.grad(result.float().sum(), (query, memory))

    query64 = query.detach().double().requires_grad_(True)
    memory64 = memory.detach().double().requires_grad_(True)
    reference = _reference_energy(
        query64, memory64, beta=0.5,
        include_constants=include_constants, reduce=reduce,
    )
    reference_grads = torch.autograd.grad(reference.sum(), (query64, memory64))
    tolerance = 1e-3 if dtype == torch.float16 else 1e-2

    for actual, expected in zip(actual_grads, reference_grads):
        assert actual.dtype == dtype
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual, expected.to(dtype=dtype), atol=tolerance, rtol=tolerance
        )


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("include_constants", [False, True])
def test_energy_precision_is_preserved_inside_cpu_autocast(
    dtype: torch.dtype, include_constants: bool
) -> None:
    query = torch.tensor([[200.0, 200.0]], dtype=dtype)
    memory = query.clone()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = hopfield_energy(query, memory, include_constants=include_constants)

    expected = torch.tensor([0.0 if include_constants else -40000.0], dtype=dtype)
    assert result.dtype == dtype
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("include_constants", [False, True])
def test_energy_precision_is_preserved_inside_cuda_autocast(
    include_constants: bool,
) -> None:
    query = torch.tensor([[200.0, 200.0]], device="cuda", dtype=torch.float16)
    memory = query.clone()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        result = hopfield_energy(query, memory, include_constants=include_constants)

    expected = query.new_tensor([0.0 if include_constants else -40000.0])
    assert result.dtype == query.dtype
    assert result.device == query.device
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected, atol=0, rtol=0)