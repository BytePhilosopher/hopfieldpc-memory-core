"""The default hopfield_energy omits a q-dependent term; the full energy is what retrieval descends."""

import pytest
import torch

from hopfieldpc import (
    hopfield_energy,
    hopfield_retrieve,
    multi_step_retrieve,
)

BETAS = [0.5, 2.0, 8.0]


def full_energy(q, m, beta, **kw):
    """The full energy: hopfield_energy with every term included."""
    return hopfield_energy(q, m, beta, include_constants=True, **kw)


def _setup(seed=0):
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(10, 16, generator=g, dtype=torch.float64)
    q = torch.randn(4, 16, generator=g, dtype=torch.float64)
    return q, m / m.norm(dim=-1, keepdim=True)


@pytest.mark.parametrize("beta", BETAS)
def test_full_energy_never_increases_under_retrieval(beta):
    q, m = _setup()
    previous = full_energy(q, m, beta)
    for _ in range(6):
        q, _ = multi_step_retrieve(q, m, beta=beta, num_updates=1, normalize=False)
        current = full_energy(q, m, beta)
        assert torch.all(current <= previous + 1e-9)
        previous = current


def test_default_energy_is_not_a_descent_function():
    memory = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    q0 = torch.tensor([[0.8, 0.5]])
    q1, _ = hopfield_retrieve(q0, memory, beta=5.0, normalize=False)

    assert hopfield_energy(q1, memory, beta=5.0).item() > hopfield_energy(q0, memory, beta=5.0).item()
    assert full_energy(q1, memory, beta=5.0).item() < full_energy(q0, memory, beta=5.0).item()


@pytest.mark.parametrize("beta", BETAS)
def test_full_energy_gradient_is_query_minus_retrieved(beta):
    q, m = _setup()
    x = q.clone().requires_grad_(True)
    full_energy(x, m, beta).sum().backward()
    retrieved, _ = hopfield_retrieve(q, m, beta=beta, normalize=False)
    torch.testing.assert_close(x.grad, q - retrieved)


@pytest.mark.parametrize("beta", BETAS)
def test_retrieval_decreases_energy_by_at_least_half_the_squared_step(beta):
    # Hessian of the full energy is I - beta * Cov_p(m), which is <= I.
    # Retrieval is a unit gradient step, so E(q) - E(q_new) >= 0.5 * |q - q_new|^2.
    q0, m = _setup()
    q1, _ = hopfield_retrieve(q0, m, beta=beta, normalize=False)
    drop = full_energy(q0, m, beta) - full_energy(q1, m, beta)
    step_sq = ((q0 - q1) ** 2).sum(-1)
    assert torch.all(drop >= 0.5 * step_sq - 1e-9)
