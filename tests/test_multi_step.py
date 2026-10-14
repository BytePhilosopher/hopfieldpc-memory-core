"""Tests for optional multi-step Hopfield retrieval."""

import pytest
import torch

from hopfieldpc import HopfieldMemory, hopfield_retrieve
from hopfieldpc.functional import multi_step_retrieve


def test_default_num_updates_is_one() -> None:
    assert HopfieldMemory(dim=4).num_updates == 1


def test_one_step_matches_hopfield_retrieve() -> None:
    q, m = torch.randn(3, 4), torch.randn(6, 4)
    r1, w1 = hopfield_retrieve(q, m, beta=2.0)
    r2, w2 = multi_step_retrieve(q, m, beta=2.0, num_updates=1)
    assert torch.allclose(r1, r2) and torch.allclose(w1, w2)


@pytest.mark.parametrize("steps", [1, 2, 3, 10])
def test_shapes_valid_after_multiple_updates(steps: int) -> None:
    module = HopfieldMemory(dim=8, beta=2.0, num_updates=steps)
    out = module(torch.randn(5, 8), torch.randn(12, 8))
    assert out.retrieved.shape == (5, 8)
    assert out.weights.shape == (5, 12)


def test_no_nan_or_inf() -> None:
    module = HopfieldMemory(dim=16, beta=8.0, num_updates=20)
    out = module(torch.randn(4, 16) * 5, torch.randn(10, 16))
    assert torch.isfinite(out.retrieved).all()
    assert torch.isfinite(out.weights).all()


def test_weights_sum_to_one_after_multi_step() -> None:
    module = HopfieldMemory(dim=8, beta=2.0, num_updates=3)
    out = module(torch.randn(4, 8), torch.randn(9, 8))
    assert torch.allclose(out.weights.sum(-1), torch.ones(4), atol=1e-5)


def test_distance_to_nearest_memory_does_not_explode() -> None:
    torch.manual_seed(0)
    memory = torch.randn(10, 16)
    query = memory[:4] + 0.1 * torch.randn(4, 16)
    out = HopfieldMemory(dim=16, beta=4.0, num_updates=5)(query, memory)
    q_end = torch.cdist(out.retrieved, memory).min(dim=-1).values
    assert torch.isfinite(q_end).all()
    # retrieved state is a convex combination of memories, so its norm is bounded
    assert out.retrieved.norm(dim=-1).max() <= memory.norm(dim=-1).max() + 1e-4


def test_multi_step_moves_toward_clear_memory_in_controlled_case() -> None:
    memory = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    query = torch.tensor([[0.8, 0.1]])
    start_dist = torch.cdist(query, memory[:1]).item()
    out = HopfieldMemory(dim=2, beta=8.0, num_updates=3)(query, memory)
    end_dist = torch.cdist(out.retrieved, memory[:1]).item()
    assert end_dist < start_dist


def test_return_states_gives_q0_to_qT() -> None:
    q, m = torch.randn(2, 4), torch.randn(5, 4)
    r, _, states = multi_step_retrieve(q, m, beta=2.0, num_updates=3, return_states=True)
    assert len(states) == 4
    assert torch.equal(states[0], q)
    assert torch.equal(states[-1], r)


def test_module_exposes_states_in_diagnostics() -> None:
    module = HopfieldMemory(dim=4, beta=2.0, num_updates=3, return_states=True)
    out = module(torch.randn(2, 4), torch.randn(5, 4))
    assert len(out.diagnostics["states"]) == 4


def test_states_not_returned_by_default() -> None:
    out = HopfieldMemory(dim=4, num_updates=3)(torch.randn(2, 4), torch.randn(5, 4))
    assert "states" not in out.diagnostics


def test_return_states_respects_disabled_diagnostics() -> None:
    module = HopfieldMemory(
        dim=4,
        num_updates=3,
        return_states=True,
        return_diagnostics=False,
    )
    out = module(torch.randn(2, 4), torch.randn(5, 4))
    assert out.diagnostics == {}


def test_invalid_num_updates_raises() -> None:
    with pytest.raises(ValueError, match="num_updates must be"):
        multi_step_retrieve(torch.randn(2, 4), torch.randn(5, 4), num_updates=0)


def test_module_rejects_bool_num_updates() -> None:
    with pytest.raises(ValueError, match="num_updates must be"):
        HopfieldMemory(dim=4, num_updates=True)


def test_function_rejects_bool_num_updates() -> None:
    with pytest.raises(ValueError, match="num_updates must be"):
        multi_step_retrieve(torch.randn(2, 4), torch.randn(5, 4), num_updates=True)


def test_gradient_flows_through_multi_step() -> None:
    module = HopfieldMemory(dim=3, memory_size=4, learnable_memory=True, num_updates=3)
    module(torch.randn(2, 3)).retrieved.sum().backward()
    assert module.memory.grad is not None
    assert torch.isfinite(module.memory.grad).all()
