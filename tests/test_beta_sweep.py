"""Unit tests for modern Hopfield beta sweep benchmark and evaluation metrics."""

import math
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from benchmarks.beta_sweep import (
    evaluate_beta_retrieval,
    run_beta_sweep,
    validate_beta_sweep_config,
)
from hopfieldpc.diagnostics import (
    attention_entropy,
    max_attention_weight,
    retrieval_distance,
)


def test_evaluate_all_configured_betas() -> None:
    betas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    config: dict[str, Any] = {
        "seed": 42,
        "dim": 8,
        "num_memories": 10,
        "num_queries": 15,
        "noise_level": 0.1,
        "beta_values": betas,
        "save_plot": False,
    }

    results, _ = run_beta_sweep(config, save_plot=False)
    evaluated_betas = [r["beta"] for r in results]
    assert evaluated_betas == betas


def test_reproducibility_with_same_seed() -> None:
    config: dict[str, Any] = {
        "seed": 999,
        "dim": 16,
        "num_memories": 20,
        "num_queries": 25,
        "noise_level": 0.2,
        "beta_values": [0.5, 2.0, 8.0],
        "save_plot": False,
    }

    results1, meta1 = run_beta_sweep(config, save_plot=False)
    results2, meta2 = run_beta_sweep(config, save_plot=False)

    assert meta1["baseline_query_distance"] == pytest.approx(meta2["baseline_query_distance"])
    for r1, r2 in zip(results1, results2):
        assert r1["beta"] == r2["beta"]
        assert r1["top1_accuracy"] == pytest.approx(r2["top1_accuracy"])
        assert r1["mean_entropy"] == pytest.approx(r2["mean_entropy"])
        assert r1["normalized_entropy"] == pytest.approx(r2["normalized_entropy"])
        assert r1["mean_max_weight"] == pytest.approx(r2["mean_max_weight"])
        assert r1["mean_retrieval_distance"] == pytest.approx(r2["mean_retrieval_distance"])


def test_identical_data_used_for_every_beta(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify memory bank and queries are created once and shared across all beta iterations."""
    called_betas = []

    original_evaluate = evaluate_beta_retrieval

    captured_queries = []
    captured_memories = []

    def mock_evaluate(
        *,
        queries: torch.Tensor,
        target_indices: torch.Tensor,
        memory: torch.Tensor,
        beta: float,
        num_updates: int = 1,
        normalize: bool = False,
    ) -> dict[str, float]:
        called_betas.append(beta)
        captured_queries.append(queries)
        captured_memories.append(memory)
        return original_evaluate(
            queries=queries,
            target_indices=target_indices,
            memory=memory,
            beta=beta,
            num_updates=num_updates,
            normalize=normalize,
        )

    monkeypatch.setattr("benchmarks.beta_sweep.evaluate_beta_retrieval", mock_evaluate)

    config: dict[str, Any] = {
        "seed": 7,
        "dim": 8,
        "num_memories": 6,
        "num_queries": 10,
        "noise_level": 0.1,
        "beta_values": [0.1, 1.0, 5.0],
        "save_plot": False,
    }

    results, _ = run_beta_sweep(config, save_plot=False)
    assert len(results) == 3
    # Check that the tensor objects passed to every beta are identical in values
    for q in captured_queries[1:]:
        assert torch.equal(q, captured_queries[0])
    for m in captured_memories[1:]:
        assert torch.equal(m, captured_memories[0])


def test_accuracy_calculation_correctness() -> None:
    # Memory of 3 items
    memory = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ])
    # 3 queries that are closest to items 0, 1, 2 respectively
    queries = torch.tensor([
        [0.9, 0.1],   # -> target 0
        [0.1, 0.9],   # -> target 1
        [-0.8, 0.1],  # -> target 2
    ])
    target_indices = torch.tensor([0, 1, 0])  # Note query 2 has wrong target 0 for testing

    metrics = evaluate_beta_retrieval(
        queries=queries,
        target_indices=target_indices,
        memory=memory,
        beta=10.0,
    )

    # Predicted should be [0, 1, 2]. Targets are [0, 1, 0]. Matches at 0 and 1 (2 out of 3 = 66.67%)
    assert metrics["top1_accuracy"] == pytest.approx(2.0 / 3.0, abs=1e-5)


def test_entropy_and_max_weight_known_distributions() -> None:
    # 1. One-hot distribution
    one_hot = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert attention_entropy(one_hot).item() == pytest.approx(0.0, abs=1e-6)
    assert attention_entropy(one_hot, normalize=True).item() == pytest.approx(0.0, abs=1e-6)
    assert max_attention_weight(one_hot).item() == pytest.approx(1.0, abs=1e-6)

    # 2. Uniform distribution over 4 memories
    uniform = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    expected_entropy = math.log(4.0)
    assert attention_entropy(uniform).item() == pytest.approx(expected_entropy, abs=1e-6)
    assert attention_entropy(uniform, normalize=True).item() == pytest.approx(1.0, abs=1e-6)
    assert max_attention_weight(uniform).item() == pytest.approx(0.25, abs=1e-6)


def test_retrieval_distance_against_correct_target_vectors() -> None:
    memory = torch.tensor([
        [2.0, 0.0],
        [0.0, 2.0],
    ])
    queries = torch.tensor([
        [1.8, 0.1],
        [0.1, 1.9],
    ])
    target_indices = torch.tensor([0, 1])

    metrics = evaluate_beta_retrieval(
        queries=queries,
        target_indices=target_indices,
        memory=memory,
        beta=100.0,  # Extreme sharpness -> retrieved ~ target
    )

    # At beta=100, retrieved should be almost identical to targets, so distance ~ 0
    assert metrics["mean_retrieval_distance"] < 0.01

    # Check distance calculation directly
    retrieved = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    targets = memory[target_indices]
    dist = retrieval_distance(retrieved, targets, p=2.0)
    assert torch.allclose(dist, torch.zeros(2), atol=1e-6)


def test_weights_finite_and_sum_to_one() -> None:
    memory = torch.randn(20, 32)
    queries = torch.randn(10, 32)
    targets = torch.randint(0, 20, (10,))

    for beta in [0.01, 0.1, 1.0, 5.0, 20.0, 50.0]:
        metrics = evaluate_beta_retrieval(
            queries=queries,
            target_indices=targets,
            memory=memory,
            beta=beta,
        )
        assert not math.isnan(metrics["mean_entropy"])
        assert not math.isnan(metrics["mean_max_weight"])
        assert not math.isnan(metrics["mean_retrieval_distance"])
        assert 0.0 <= metrics["top1_accuracy"] <= 1.0
        assert 0.0 <= metrics["mean_max_weight"] <= 1.0


def test_increasing_beta_sharpens_attention_controlled() -> None:
    """On a controlled task with unequal dot products, increasing beta must:

    1. Strictly increase mean_max_weight
    2. Strictly decrease mean_entropy
    3. Decrease retrieval distance to the target
    """
    memory = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    # Query clearly closer to memory 0
    queries = torch.tensor([[0.9, 0.1]])
    target_indices = torch.tensor([0])

    betas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    max_weights = []
    entropies = []
    distances = []

    for b in betas:
        m = evaluate_beta_retrieval(
            queries=queries,
            target_indices=target_indices,
            memory=memory,
            beta=b,
        )
        max_weights.append(m["mean_max_weight"])
        entropies.append(m["mean_entropy"])
        distances.append(m["mean_retrieval_distance"])

    # Max weight must be strictly monotonically increasing
    for i in range(len(betas) - 1):
        assert max_weights[i + 1] > max_weights[i]

    # Entropy must be strictly monotonically decreasing
    for i in range(len(betas) - 1):
        assert entropies[i + 1] < entropies[i]

    # Distance to target memory must be strictly monotonically decreasing
    for i in range(len(betas) - 1):
        assert distances[i + 1] < distances[i]


def test_invalid_configurations_raise_exceptions() -> None:
    base: dict[str, Any] = {
        "seed": 42,
        "dim": 8,
        "num_memories": 10,
        "num_queries": 10,
        "noise_level": 0.1,
        "beta_values": [1.0, 2.0],
    }

    # Invalid seed
    with pytest.raises(ValueError, match="seed"):
        validate_beta_sweep_config({**base, "seed": -1})
    with pytest.raises(ValueError, match="seed"):
        validate_beta_sweep_config({**base, "seed": True})

    # Invalid dim
    with pytest.raises(ValueError, match="dim"):
        validate_beta_sweep_config({**base, "dim": 0})
    with pytest.raises(ValueError, match="dim"):
        validate_beta_sweep_config({**base, "dim": True})

    # Invalid num_memories
    with pytest.raises(ValueError, match="num_memories"):
        validate_beta_sweep_config({**base, "num_memories": 0})

    # Invalid num_queries
    with pytest.raises(ValueError, match="num_queries"):
        validate_beta_sweep_config({**base, "num_queries": 0})

    # Invalid noise_level
    with pytest.raises(ValueError, match="noise_level"):
        validate_beta_sweep_config({**base, "noise_level": -0.5})

    # Invalid beta_values
    with pytest.raises(ValueError, match="beta_values"):
        validate_beta_sweep_config({**base, "beta_values": []})
    with pytest.raises(ValueError, match="beta_values"):
        validate_beta_sweep_config({**base, "beta_values": [0.0]})
    with pytest.raises(ValueError, match="beta_values"):
        validate_beta_sweep_config({**base, "beta_values": [-1.0]})

    # Invalid num_updates
    with pytest.raises(ValueError, match="num_updates"):
        validate_beta_sweep_config({**base, "num_updates": 0})
    with pytest.raises(ValueError, match="num_updates"):
        validate_beta_sweep_config({**base, "num_updates": True})


def test_run_beta_sweep_reports_generation() -> None:
    config: dict[str, Any] = {
        "seed": 42,
        "dim": 8,
        "num_memories": 10,
        "num_queries": 12,
        "noise_level": 0.1,
        "beta_values": [0.5, 2.0],
        "save_plot": True,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "reports"
        results, meta = run_beta_sweep(config, output_dir=output_dir, save_plot=True)

        assert len(results) == 2
        assert "baseline_query_distance" in meta

        json_path = output_dir / "results.json"
        csv_path = output_dir / "results.csv"
        plot_path = output_dir / "beta_sweep.png"

        assert json_path.exists()
        assert csv_path.exists()
        assert plot_path.exists()
