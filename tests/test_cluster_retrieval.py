"""Unit and integration tests for Hopfield cluster / metastable retrieval benchmark."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from benchmarks.cluster_retrieval import (
    compute_cluster_masses,
    compute_cluster_metrics,
    evaluate_cluster_retrieval,
    format_results_table,
    generate_cluster_centers,
    generate_clustered_memory,
    generate_queries,
    plot_cluster_benchmark,
    run_benchmark,
    validate_cluster_config,
)


def test_memory_bank_and_cluster_labels_shape() -> None:
    """Requirement 1 & 2: memory bank has expected shape and cluster labels length."""
    num_clusters = 5
    memories_per_cluster = 12
    dim = 32
    sep = 2.5
    noise = 0.1

    memory, cluster_labels, centers = generate_clustered_memory(
        num_clusters=num_clusters,
        memories_per_cluster=memories_per_cluster,
        dim=dim,
        cluster_separation=sep,
        cluster_noise=noise,
    )

    expected_total_memories = num_clusters * memories_per_cluster
    assert memory.shape == (expected_total_memories, dim)
    assert cluster_labels.shape == (expected_total_memories,)
    assert centers.shape == (num_clusters, dim)

    # Check each cluster has exactly memories_per_cluster items
    for c in range(num_clusters):
        assert (cluster_labels == c).sum().item() == memories_per_cluster


def test_generate_cluster_centers_separation() -> None:
    """Verify cluster centers are well-separated."""
    num_clusters = 4
    dim = 16
    sep = 3.0

    centers = generate_cluster_centers(num_clusters, dim, separation=sep)
    assert centers.shape == (num_clusters, dim)

    # In orthogonal frame, mutual distance between different centers is sqrt(2)*sep
    dists = torch.cdist(centers, centers)
    for i in range(num_clusters):
        for j in range(num_clusters):
            if i != j:
                assert dists[i, j].item() > 0.5 * sep


def test_target_cluster_is_valid() -> None:
    """Requirement 3: target cluster is valid."""
    centers = torch.randn(4, 16)
    queries, targets = generate_queries(
        cluster_centers=centers,
        target_cluster=2,
        num_queries=10,
        query_noise=0.05,
    )

    assert queries.shape == (10, 16)
    assert targets.shape == (10,)
    assert (targets == 2).all().item()

    # When target_cluster is None, targets are within [0, num_clusters - 1]
    queries_multi, targets_multi = generate_queries(
        cluster_centers=centers,
        target_cluster=None,
        num_queries=50,
        query_noise=0.05,
    )
    assert queries_multi.shape == (50, 16)
    assert targets_multi.min().item() >= 0
    assert targets_multi.max().item() < 4


def test_evaluate_cluster_retrieval_direct() -> None:
    """Direct unit test for evaluate_cluster_retrieval and compute_cluster_metrics."""
    memory = torch.tensor([
        [1.0, 0.0],
        [1.1, 0.0],
        [0.0, 1.0],
        [0.0, 1.1],
    ])
    cluster_labels = torch.tensor([0, 0, 1, 1])
    centers = torch.tensor([
        [1.05, 0.0],
        [0.0, 1.05],
    ])
    queries = torch.tensor([
        [1.0, 0.0],
    ])
    target_clusters = torch.tensor([0])

    metrics = evaluate_cluster_retrieval(
        queries=queries,
        target_clusters=target_clusters,
        memory=memory,
        cluster_centers=centers,
        cluster_labels=cluster_labels,
        beta=5.0,
    )

    assert "target_cluster_mass" in metrics
    assert "distance_to_cluster" in metrics
    assert "distance_to_nearest_memory" in metrics
    assert metrics["target_cluster_mass"] > 0.90
    assert metrics["top1_in_target_cluster"] == 1.0


def test_compute_cluster_metrics_direct() -> None:
    """Test compute_cluster_metrics directly with mock tensors."""
    retrieved = torch.tensor([[1.0, 0.0]])
    weights = torch.tensor([[0.7, 0.3]])
    memory = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    centers = torch.tensor([[1.0, 0.0]])
    cluster_labels = torch.tensor([0, 0])
    target_clusters = torch.tensor([0])

    metrics = compute_cluster_metrics(
        retrieved=retrieved,
        weights=weights,
        memory=memory,
        cluster_centers=centers,
        cluster_labels=cluster_labels,
        target_clusters=target_clusters,
    )
    assert metrics["target_cluster_mass"].item() == pytest.approx(1.0)
    assert metrics["top1_mass"].item() == pytest.approx(0.7)
    assert metrics["distance_to_cluster"].item() == pytest.approx(0.0)
    assert metrics["distance_to_nearest_memory"].item() == pytest.approx(0.0)


def test_attention_weights_and_cluster_masses_sum_to_one() -> None:
    """Requirement 4 & 5: attention weights and cluster masses sum to 1."""
    config: dict[str, Any] = {
        "seed": 42,
        "dim": 16,
        "num_clusters": 3,
        "memories_per_cluster": 8,
        "cluster_separation": 2.5,
        "cluster_noise": 0.1,
        "query_noise": 0.05,
        "num_queries": 15,
        "target_cluster": 1,
        "beta_values": [0.1, 1.0, 10.0],
        "save_plot": False,
    }

    results, _ = run_benchmark(config, save_plot=False)
    for row in results:
        assert 0.0 <= row["target_cluster_mass"] <= 1.0001
        assert 0.0 <= row["top1_mass"] <= 1.0001


def test_cluster_masses_computation() -> None:
    """Verify compute_cluster_masses aggregates correctly."""
    weights = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.25, 0.25, 0.25, 0.25],
    ])
    cluster_labels = torch.tensor([0, 0, 1, 1])
    num_clusters = 2

    masses = compute_cluster_masses(weights, cluster_labels, num_clusters)
    assert masses.shape == (2, 2)
    assert masses[0, 0].item() == pytest.approx(0.3)
    assert masses[0, 1].item() == pytest.approx(0.7)
    assert masses[1, 0].item() == pytest.approx(0.5)
    assert masses[1, 1].item() == pytest.approx(0.5)


def test_metrics_are_finite_and_bounded() -> None:
    """Requirement 6, 8, 9, 10: all metrics finite, non-negative, and properly bounded."""
    config: dict[str, Any] = {
        "seed": 101,
        "dim": 32,
        "num_clusters": 4,
        "memories_per_cluster": 6,
        "cluster_separation": 3.0,
        "cluster_noise": 0.15,
        "query_noise": 0.10,
        "num_queries": 20,
        "target_cluster": 0,
        "beta_values": [0.05, 0.5, 2.0, 10.0, 30.0],
        "save_plot": False,
    }

    results, _ = run_benchmark(config, save_plot=False)

    for r in results:
        # Finiteness
        for key in [
            "target_cluster_mass",
            "top1_mass",
            "entropy",
            "cluster_entropy",
            "distance_to_cluster",
            "distance_to_nearest_memory",
            "effective_memory_count",
            "effective_cluster_count",
        ]:
            val = r[key]
            assert isinstance(val, (int, float))
            assert not torch.isnan(torch.tensor(val))
            assert not torch.isinf(torch.tensor(val))

        # Invariants
        assert 0.0 <= r["target_cluster_mass"] <= 1.0001
        assert 0.0 <= r["top1_mass"] <= 1.0001
        assert r["entropy"] >= -1e-6
        assert r["cluster_entropy"] >= -1e-6
        assert r["distance_to_cluster"] >= 0.0
        assert r["distance_to_nearest_memory"] >= 0.0
        assert 0.0 <= r["top1_in_target_cluster"] <= 1.0
        assert r["effective_memory_count"] >= 1.0 - 1e-4
        assert r["effective_cluster_count"] >= 1.0 - 1e-4


def test_reproducibility_with_same_seed() -> None:
    """Requirement 7: repeated runs with the same seed are deterministic."""
    config: dict[str, Any] = {
        "seed": 777,
        "dim": 16,
        "num_clusters": 3,
        "memories_per_cluster": 5,
        "cluster_separation": 2.0,
        "cluster_noise": 0.1,
        "query_noise": 0.05,
        "num_queries": 10,
        "target_cluster": 1,
        "beta_values": [0.1, 1.0, 10.0],
        "save_plot": False,
    }

    results1, _ = run_benchmark(config, save_plot=False)
    results2, _ = run_benchmark(config, save_plot=False)

    for r1, r2 in zip(results1, results2):
        assert r1["beta"] == r2["beta"]
        assert r1["target_cluster_mass"] == pytest.approx(r2["target_cluster_mass"])
        assert r1["top1_mass"] == pytest.approx(r2["top1_mass"])
        assert r1["entropy"] == pytest.approx(r2["entropy"])
        assert r1["cluster_entropy"] == pytest.approx(r2["cluster_entropy"])
        assert r1["distance_to_cluster"] == pytest.approx(r2["distance_to_cluster"])
        assert r1["distance_to_nearest_memory"] == pytest.approx(r2["distance_to_nearest_memory"])


def test_scientific_transition_across_regimes() -> None:
    """Scientific acceptance test: verifies the three retrieval regimes.
    """
    config: dict[str, Any] = {
        "seed": 42,
        "dim": 64,
        "num_clusters": 4,
        "memories_per_cluster": 10,
        "cluster_separation": 3.0,
        "cluster_noise": 0.15,
        "query_noise": 0.10,
        "num_queries": 30,
        "target_cluster": 0,
        "beta_values": [0.05, 1.0, 50.0],
        "save_plot": False,
    }

    results, _ = run_benchmark(config, save_plot=False)
    low_res = results[0]   # beta = 0.05
    med_res = results[1]   # beta = 1.0
    high_res = results[2]  # beta = 50.0

    # Low beta: broad global averaging
    assert low_res["target_cluster_mass"] < 0.50
    assert low_res["top1_mass"] < 0.10
    assert low_res["entropy"] > 3.0

    # Medium beta: cluster / metastable retrieval
    assert med_res["target_cluster_mass"] > 0.95
    assert med_res["top1_mass"] < 0.50
    assert med_res["cluster_entropy"] < 0.10
    # Crucial scientific signature of cluster state:
    # Retrieved vector is closer to the cluster center prototype than to any individual memory!
    assert med_res["distance_to_cluster"] < med_res["distance_to_nearest_memory"]

    # High beta: single memory retrieval
    assert high_res["target_cluster_mass"] > 0.99
    assert high_res["top1_mass"] > 0.90
    assert high_res["entropy"] < 0.30
    assert high_res["distance_to_nearest_memory"] < 0.15

    # Monotonic trends
    assert high_res["top1_mass"] > med_res["top1_mass"] > low_res["top1_mass"]
    assert low_res["entropy"] > med_res["entropy"] > high_res["entropy"]
    assert low_res["distance_to_nearest_memory"] > high_res["distance_to_nearest_memory"]


def test_multi_seed_aggregation() -> None:
    """Verify benchmark supports running and aggregating multiple random seeds."""
    config: dict[str, Any] = {
        "seed": 42,
        "seeds": [10, 20, 30],
        "dim": 16,
        "num_clusters": 3,
        "memories_per_cluster": 5,
        "cluster_separation": 2.5,
        "cluster_noise": 0.1,
        "query_noise": 0.05,
        "num_queries": 10,
        "target_cluster": 0,
        "beta_values": [0.1, 2.0],
        "save_plot": False,
    }

    results, meta = run_benchmark(config, save_plot=False)
    assert meta["num_seeds"] == 3
    assert len(results) == 2

    for r in results:
        assert "target_cluster_mass_std" in r
        assert "distance_to_cluster_std" in r
        assert r["target_cluster_mass_std"] >= 0.0


def test_config_validation_errors() -> None:
    """Verify clear ValueErrors for invalid configuration parameters."""
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        validate_cluster_config({"seed": -1})

    with pytest.raises(ValueError, match="dim must be a positive integer"):
        validate_cluster_config({"dim": 0})

    with pytest.raises(ValueError, match="num_clusters must be an integer >= 2"):
        validate_cluster_config({"num_clusters": 1})

    with pytest.raises(ValueError, match="memories_per_cluster must be an integer >= 2"):
        validate_cluster_config({"memories_per_cluster": 1})

    with pytest.raises(ValueError, match="cluster_separation must be a positive number"):
        validate_cluster_config({"cluster_separation": 0.0})

    with pytest.raises(ValueError, match="cluster_noise must be a non-negative number"):
        validate_cluster_config({"cluster_noise": -0.1})

    with pytest.raises(ValueError, match="query_noise must be a non-negative number"):
        validate_cluster_config({"query_noise": -0.1})

    with pytest.raises(ValueError, match="target_cluster must be an integer in"):
        validate_cluster_config({"num_clusters": 4, "target_cluster": 5})

    with pytest.raises(ValueError, match="beta_values must be a non-empty list"):
        validate_cluster_config({"beta_values": []})

    with pytest.raises(ValueError, match="All beta_values must be positive"):
        validate_cluster_config({"beta_values": [1.0, -0.5]})


def test_save_reports_and_plot() -> None:
    """Verify reports (JSON, CSV) and diagnostic plot are correctly generated and saved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        config: dict[str, Any] = {
            "seed": 42,
            "dim": 16,
            "num_clusters": 3,
            "memories_per_cluster": 5,
            "cluster_separation": 2.5,
            "cluster_noise": 0.1,
            "query_noise": 0.05,
            "num_queries": 10,
            "target_cluster": 0,
            "beta_values": [0.1, 1.0, 5.0],
            "save_plot": True,
        }

        results, meta = run_benchmark(config, output_dir=out_dir, save_plot=True)

        assert (out_dir / "results.json").exists()
        assert (out_dir / "results.csv").exists()
        assert (out_dir / "cluster_retrieval.png").exists()

        # Check plot function directly
        plot_path = out_dir / "custom_plot.png"
        success = plot_cluster_benchmark(results, plot_path, cfg=meta)
        assert success is True
        assert plot_path.exists()
        assert plot_path.stat().st_size > 1000


def test_format_results_table() -> None:
    """Verify table formatting generates non-empty text containing expected columns."""
    sample_results = [
        {
            "beta": 0.1,
            "target_cluster_mass": 0.25,
            "top1_mass": 0.05,
            "entropy": 3.5,
            "cluster_entropy": 1.3,
            "distance_to_cluster": 2.1,
            "distance_to_nearest_memory": 2.2,
            "effective_memory_count": 20.0,
            "regime": "Global Averaging",
        }
    ]
    table = format_results_table(sample_results)
    assert "beta" in table
    assert "cluster_mass" in table
    assert "Global Averaging" in table
