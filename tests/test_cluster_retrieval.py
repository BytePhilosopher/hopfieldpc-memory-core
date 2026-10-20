"""Unit and integration tests for Hopfield cluster retrieval benchmark."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from benchmarks.cluster_retrieval import (
    classify_retrieval_regime,
    compute_cluster_masses,
    compute_cluster_metrics,
    compute_dynamics_diagnostics,
    evaluate_cluster_retrieval,
    format_results_table,
    generate_cluster_centers,
    generate_clustered_memory,
    generate_queries,
    main,
    metric_uncertainty,
    plot_cluster_benchmark,
    run_benchmark,
    run_single_seed_experiment,
    sample_perturbations,
    validate_cluster_config,
)
from hopfieldpc.functional import multi_step_retrieve


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
    """Verify cluster centers have norm=separation and orthogonal distance=sqrt(2)*separation."""
    num_clusters = 4
    dim = 16
    sep = 3.0

    centers = generate_cluster_centers(num_clusters, dim, separation=sep)
    assert centers.shape == (num_clusters, dim)

    # Each center vector has Euclidean norm equal to sep
    for i in range(num_clusters):
        assert torch.linalg.vector_norm(centers[i]).item() == pytest.approx(sep, rel=1e-4)

    # In orthogonal frame, mutual distance between different centers is sqrt(2)*sep
    dists = torch.cdist(centers, centers)
    for i in range(num_clusters):
        for j in range(num_clusters):
            if i != j:
                assert dists[i, j].item() == pytest.approx(math.sqrt(2) * sep, rel=1e-4)


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
    assert "dynamics_residual" in metrics
    assert "trajectory_separation_ratio" in metrics
    assert metrics["dynamics_evaluated"] is True
    assert "regime" in metrics
    assert "query_count" in metrics
    assert "seed_count" in metrics
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
    assert metrics["correct_cluster"].item() == 1.0
    assert metrics["wrong_cluster"].item() == 0.0


def test_classify_retrieval_regime_distribution_based() -> None:
    """Verify classify_retrieval_regime classifies by distribution, independently of target."""
    # 1. Single Memory concentration (top-1 >= 0.70)
    assert classify_retrieval_regime(1.0, 1.0, 0.0) == "Single Memory"
    assert classify_retrieval_regime(0.90, 0.75, 0.2) == "Single Memory"

    # 2. Cluster Averaging concentration (max cluster mass >= 0.85, top-1 < 0.70)
    assert classify_retrieval_regime(0.95, 0.20, 0.5) == "Cluster Averaging"
    assert classify_retrieval_regime(0.85, 0.50, 0.4) == "Cluster Averaging"

    # 3. Global Averaging requires an explicit near-uniformity criterion
    assert classify_retrieval_regime(0.30, 0.03, 0.99) == "Global Averaging"
    assert classify_retrieval_regime(0.40, 0.10, 0.90) == "Global Averaging"


def test_classify_retrieval_regime_broad_mixed_fallback() -> None:
    """Strongly concentrated distributions below both thresholds are not Global Averaging."""
    # top1=0.69, max cluster mass=0.84: concentrated, far from uniform
    assert classify_retrieval_regime(0.84, 0.69, 0.25) == "Broad/Mixed"
    # Mass split evenly across two of four clusters
    assert classify_retrieval_regime(0.50, 0.05, 0.80) == "Broad/Mixed"


def test_classify_retrieval_regime_rejects_target_cluster_mass() -> None:
    """The helper has no target-dependent input to fall back on."""
    with pytest.raises(TypeError):
        classify_retrieval_regime(  # type: ignore[call-arg]
            target_cluster_mass=0.9, top1_mass=0.2, normalized_entropy=0.5
        )


def test_classify_retrieval_regime_wrong_cluster_concentration() -> None:
    """Verify that concentration in a wrong cluster is NOT mislabeled as Global Averaging.

    Even if target_cluster_mass is 0.0, if retrieval is concentrated on a single memory
    in the wrong cluster, the regime is 'Single Memory', with wrong_cluster flagged.
    """
    memory = torch.tensor([
        [10.0, 0.0],
        [10.0, 0.1],
        [0.0, 10.0],
        [0.0, 10.1],
    ])
    cluster_labels = torch.tensor([0, 0, 1, 1])
    centers = torch.tensor([
        [10.0, 0.0],
        [0.0, 10.0],
    ])
    # Query is close to cluster 1 (so retrieval concentrates on cluster 1)
    queries = torch.tensor([
        [0.0, 10.0],
    ])
    # But target cluster is erroneously specified as 0!
    target_clusters = torch.tensor([0])

    row = evaluate_cluster_retrieval(
        queries=queries,
        target_clusters=target_clusters,
        memory=memory,
        cluster_centers=centers,
        cluster_labels=cluster_labels,
        beta=10.0,
    )

    # Attention is concentrated on cluster 1 / memory in cluster 1
    # Target cluster 0 receives near zero mass
    assert row["target_cluster_mass"] < 0.01
    assert row["wrong_cluster_mass"] > 0.99
    # The concentration regime is Single Memory (NOT Global Averaging!)
    assert row["regime"] == "Single Memory"
    assert row["fraction_single_memory"] == 1.0
    assert row["fraction_global_averaging"] == 0.0
    # Correct vs wrong cluster metrics clearly separate correctness
    assert row["correct_cluster_fraction"] == 0.0
    assert row["wrong_cluster_fraction"] == 1.0
    assert row["wrong_single_memory_fraction"] == 1.0


def test_classify_retrieval_regime_mixed_outcomes() -> None:
    """Verify per-query classification aggregates mixed regime fractions properly."""
    memory = torch.tensor([
        [6.0, 0.0],
        [4.0, 0.0],
        [0.0, 6.0],
        [0.0, 4.0],
    ])
    cluster_labels = torch.tensor([0, 0, 1, 1])
    centers = torch.tensor([
        [5.0, 0.0],
        [0.0, 5.0],
    ])
    # Query 1: sharp single memory concentration (closest to memory 0 [6.0, 0.0])
    # Query 2: equidistant between clusters (diffuse global averaging)
    queries = torch.tensor([
        [5.0, 0.0],
        [0.0, 0.0],
    ])
    target_clusters = torch.tensor([0, 0])

    row = evaluate_cluster_retrieval(
        queries=queries,
        target_clusters=target_clusters,
        memory=memory,
        cluster_centers=centers,
        cluster_labels=cluster_labels,
        beta=5.0,
    )

    fractions = row["regime_fractions"]
    assert fractions["Single Memory"] == 0.5
    assert fractions["Global Averaging"] == 0.5
    assert row["fraction_single_memory"] == 0.5
    assert row["fraction_global_averaging"] == 0.5


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
            "dynamics_residual",
            "trajectory_separation_final",
            "trajectory_separation_ratio",
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
        assert r["dynamics_residual"] >= 0.0


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
        assert r1["dynamics_residual"] == pytest.approx(r2["dynamics_residual"])


def test_scientific_transition_across_regimes_controlled_example() -> None:
    """Controlled example verifying the three qualitative retrieval regimes under standard geometry."""
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
    assert low_res["regime"] == "Global Averaging"

    # Medium beta: cluster averaging
    assert med_res["target_cluster_mass"] > 0.95
    assert med_res["top1_mass"] < 0.50
    assert med_res["cluster_entropy"] < 0.10
    # Closer to cluster generating prototype than to any individual stored memory:
    assert med_res["distance_to_cluster"] < med_res["distance_to_nearest_memory"]
    assert med_res["regime"] == "Cluster Averaging"

    # High beta: single memory concentration
    assert high_res["target_cluster_mass"] > 0.99
    assert high_res["top1_mass"] > 0.90
    assert high_res["entropy"] < 0.30
    assert high_res["distance_to_nearest_memory"] < 0.15
    assert high_res["regime"] == "Single Memory"

    # Monotonic trends
    assert high_res["top1_mass"] > med_res["top1_mass"] > low_res["top1_mass"]
    assert low_res["entropy"] > med_res["entropy"] > high_res["entropy"]
    assert low_res["distance_to_nearest_memory"] > high_res["distance_to_nearest_memory"]


def test_negative_control_zero_within_cluster_noise() -> None:
    """Negative control: zero intra-cluster noise creates duplicate memories that beta cannot distinguish.

    When cluster_noise == 0.0, all memories in a cluster are identical duplicates.
    Attention within the cluster is uniformly divided across all memories (1 / memories_per_cluster),
    meaning top-1 mass can never exceed 1 / memories_per_cluster regardless of how high beta is!
    """
    mem_per_cluster = 10
    config: dict[str, Any] = {
        "seed": 42,
        "dim": 32,
        "num_clusters": 3,
        "memories_per_cluster": mem_per_cluster,
        "cluster_separation": 3.0,
        "cluster_noise": 0.0,  # Duplicate memories within cluster
        "query_noise": 0.05,
        "num_queries": 20,
        "target_cluster": 0,
        "beta_values": [0.1, 1.0, 10.0, 100.0],
        "save_plot": False,
    }

    results, _ = run_benchmark(config, save_plot=False)

    expected_max_top1 = 1.0 / mem_per_cluster  # 0.10

    for r in results:
        # top1_mass is strictly bounded by 1 / memories_per_cluster
        assert r["top1_mass"] <= expected_max_top1 + 1e-4
        # Single Memory regime is NEVER reached because duplicate memories cannot be distinguished
        assert r["regime"] != "Single Memory"
        assert r["fraction_single_memory"] == 0.0


def test_one_step_vs_repeated_dynamics() -> None:
    """Test iterative Hopfield dynamics: fixed-point residual and perturbation stability."""
    memory, cluster_labels, centers = generate_clustered_memory(
        num_clusters=4,
        memories_per_cluster=8,
        dim=32,
        cluster_separation=3.0,
        cluster_noise=0.15,
        generator=torch.Generator().manual_seed(42),
    )
    queries, targets = generate_queries(
        cluster_centers=centers,
        target_cluster=0,
        num_queries=15,
        query_noise=0.10,
        generator=torch.Generator().manual_seed(42),
    )

    # 1-step retrieval vs 5-step repeated retrieval at high beta
    res_step1 = evaluate_cluster_retrieval(
        queries=queries,
        target_clusters=targets,
        memory=memory,
        cluster_centers=centers,
        cluster_labels=cluster_labels,
        beta=20.0,
        num_updates=1,
    )

    res_step5 = evaluate_cluster_retrieval(
        queries=queries,
        target_clusters=targets,
        memory=memory,
        cluster_centers=centers,
        cluster_labels=cluster_labels,
        beta=20.0,
        num_updates=5,
    )

    # Both report dynamics diagnostics
    assert res_step1["dynamics_evaluated"] and res_step5["dynamics_evaluated"]
    assert "trajectory_separation_ratio" in res_step5

    # Multi-step relaxation at high beta drives the residual ||R(q_T) - q_T|| toward 0
    assert res_step5["dynamics_residual"] <= res_step1["dynamics_residual"] + 1e-4
    assert res_step5["dynamics_residual"] < 0.05

    # Direct test of compute_dynamics_diagnostics on relaxed states:
    relaxed_states, _ = multi_step_retrieve(
        queries[0:4],
        memory,
        beta=20.0,
        num_updates=10,
    )
    perturbation = sample_perturbations(
        relaxed_states.shape, 0.05, generator=torch.Generator().manual_seed(0)
    )
    diag = compute_dynamics_diagnostics(
        retrieved=relaxed_states,
        memory=memory,
        beta=20.0,
        perturbation=perturbation,
    )
    # The relaxed state is (numerically) stationary: residual ||R(q*) - q*|| ~ 0
    assert diag["dynamics_residual"].mean().item() == pytest.approx(0.0, abs=1e-3)
    # Tested perturbations shrink relative to the unperturbed trajectory
    assert diag["trajectory_separation_ratio"].max().item() < 1.0


def _symmetric_two_cluster_memory() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two clusters, each with four memories placed symmetrically around its prototype."""
    e = torch.eye(6)
    c0, c1 = 3.0 * e[0], 3.0 * e[3]
    a = 0.3
    memory = torch.stack([
        c0 + a * e[1], c0 - a * e[1], c0 + a * e[2], c0 - a * e[2],
        c1 + a * e[4], c1 - a * e[4], c1 + a * e[5], c1 - a * e[5],
    ])
    cluster_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    centers = torch.stack([c0, c1])
    return memory, cluster_labels, centers


def test_persistent_cluster_state_controlled_case() -> None:
    """Controlled case where the cluster-averaged state persists under repeated updates.

    With memories symmetric around the prototype, the within-cluster average is a
    fixed point. Its local Jacobian is beta * Cov_p(memory) (eigenvalue 0.045 * beta
    here), so at moderate beta tested perturbations shrink. This is empirical evidence
    under the sampled perturbations, not a proof of basin stability.
    """
    memory, cluster_labels, centers = _symmetric_two_cluster_memory()
    queries = centers[0].repeat(8, 1) + 0.05 * torch.randn(
        8, 6, generator=torch.Generator().manual_seed(0)
    )
    targets = torch.zeros(8, dtype=torch.long)
    perturbation = sample_perturbations(
        queries.shape, 0.02, generator=torch.Generator().manual_seed(1)
    )

    for beta in (1.0, 5.0):
        row = evaluate_cluster_retrieval(
            queries=queries,
            target_clusters=targets,
            memory=memory,
            cluster_centers=centers,
            cluster_labels=cluster_labels,
            beta=beta,
            num_updates=20,
            perturbation=perturbation,
            perturbation_steps=3,
        )
        assert row["regime"] == "Cluster Averaging"
        assert row["fraction_cluster_averaging"] == 1.0
        assert row["top1_mass"] == pytest.approx(0.25, abs=1e-3)
        # Persistent: the relaxed state is stationary and sits at the prototype,
        # not at any individual stored memory.
        assert row["dynamics_residual"] < 1e-3
        assert row["distance_to_cluster"] < 1e-2
        assert row["distance_to_nearest_memory"] == pytest.approx(0.3, abs=1e-2)
        assert row["trajectory_separation_ratio"] < 0.1


def test_stationary_but_unstable_state_separates() -> None:
    """A stationary symmetric state at high beta separates under perturbation.

    The residual alone cannot distinguish this from a persistent state; the
    trajectory separation ratio can.
    """
    memory, _, centers = _symmetric_two_cluster_memory()
    states = centers[0].repeat(8, 1)
    perturbation = sample_perturbations(
        states.shape, 0.02, generator=torch.Generator().manual_seed(1)
    )

    diag = compute_dynamics_diagnostics(
        retrieved=states, memory=memory, beta=50.0, perturbation=perturbation
    )
    assert diag["dynamics_residual"].max().item() < 1e-4
    assert diag["trajectory_separation_ratio"].min().item() > 1.0


def test_trajectory_separation_excludes_drift_of_nonstationary_state() -> None:
    """Separation compares against the unperturbed trajectory, not the starting state."""
    memory, _, centers = _symmetric_two_cluster_memory()
    # Midway between prototypes: far from stationary at this beta.
    states = (0.5 * (centers[0] + centers[1]) + 0.3 * centers[0] / 3.0).unsqueeze(0)
    perturbation = 1e-4 * torch.ones_like(states) / math.sqrt(states.shape[-1])

    diag = compute_dynamics_diagnostics(
        retrieved=states,
        memory=memory,
        beta=1.0,
        perturbation=perturbation,
        perturbation_steps=3,
    )
    relaxed, _ = multi_step_retrieve(states, memory, beta=1.0, num_updates=3)
    drift = torch.linalg.vector_norm(relaxed - states).item()

    assert diag["dynamics_residual"].item() > 0.1
    # Measuring ||R^k(q_T + delta) - q_T|| / ||delta|| would be dominated by drift.
    assert drift / perturbation.norm().item() > 1e3
    assert diag["trajectory_separation_ratio"].item() < 10.0


def test_dynamics_disabled_reports_null_not_zero() -> None:
    """Skipped dynamics are reported as None with dynamics_evaluated=False."""
    memory, cluster_labels, centers = _symmetric_two_cluster_memory()
    row = evaluate_cluster_retrieval(
        queries=centers[0:1].repeat(3, 1),
        target_clusters=torch.zeros(3, dtype=torch.long),
        memory=memory,
        cluster_centers=centers,
        cluster_labels=cluster_labels,
        beta=1.0,
        compute_dynamics=False,
    )
    assert row["dynamics_evaluated"] is False
    for key in (
        "dynamics_residual",
        "trajectory_separation_initial",
        "trajectory_separation_final",
        "trajectory_separation_ratio",
    ):
        assert row[key] is None
        assert row[f"{key}_query_std"] is None


def test_perturbations_are_paired_across_betas() -> None:
    """The same perturbation sample is used at every beta within a seed."""
    cfg = validate_cluster_config({
        "dim": 16,
        "num_clusters": 3,
        "memories_per_cluster": 5,
        "num_queries": 12,
        "beta_values": [0.1, 1.0, 10.0],
        "perturbation_noise": 0.03,
        "perturbation_steps": 2,
    })
    results, metadata = run_single_seed_experiment(cfg, seed=5)

    initial = [r["trajectory_separation_initial"] for r in results]
    assert initial[0] == pytest.approx(initial[1], rel=1e-6)
    assert initial[0] == pytest.approx(initial[2], rel=1e-6)
    assert metadata["perturbation_noise"] == 0.03
    assert metadata["perturbation_steps"] == 2
    assert metadata["compute_dynamics"] is True


def test_multi_seed_uncertainty_consistency() -> None:
    """Uncertainty fields have one meaning per mode and are never silently zero-filled."""
    single_cfg: dict[str, Any] = {
        "seed": 42,
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
    single_res, single_meta = run_benchmark(single_cfg, save_plot=False)
    assert single_meta["num_queries"] == 10
    for r in single_res:
        assert r["seed_count"] == 1
        assert r["query_count"] == 10
        assert r["target_cluster_mass_query_std"] >= 0.0
        # A seed std cannot be measured from one seed, and the ambiguous alias is gone.
        assert "target_cluster_mass_seed_std" not in r
        assert "target_cluster_mass_std" not in r
        # Fraction metrics carry a query std computed from per-query indicators.
        for key in ("correct_cluster_fraction", "fraction_cluster_averaging"):
            p = r[key]
            n = r["query_count"]
            expected = math.sqrt(p * (1 - p) * n / (n - 1))
            assert r[f"{key}_query_std"] == pytest.approx(expected, abs=1e-6)

    multi_cfg: dict[str, Any] = {**single_cfg, "seeds": [10, 20, 30]}
    multi_res, multi_meta = run_benchmark(multi_cfg, save_plot=False)
    assert multi_meta["num_seeds"] == 3
    assert "seed" not in multi_meta
    for r in multi_res:
        assert r["seed_count"] == 3
        assert r["query_count"] == 10
        assert r["target_cluster_mass_seed_std"] >= 0.0
        assert r["target_cluster_mass_mean_within_seed_query_std"] >= 0.0
        assert "target_cluster_mass_query_std" not in r
        assert "target_cluster_mass_std" not in r
        assert r["correct_cluster_fraction_mean_within_seed_query_std"] is not None


def test_config_validation_errors() -> None:
    """Verify clear ValueErrors for invalid configuration parameters including NaN, Inf, and non-booleans."""
    # Non-mapping
    with pytest.raises(TypeError, match="Configuration must be a dictionary/mapping"):
        validate_cluster_config("invalid")  # type: ignore

    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        validate_cluster_config({"seed": -1})
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        validate_cluster_config({"seed": True})

    with pytest.raises(ValueError, match="dim must be a positive integer"):
        validate_cluster_config({"dim": 0})
    with pytest.raises(ValueError, match="dim must be a positive integer"):
        validate_cluster_config({"dim": True})

    with pytest.raises(ValueError, match="num_clusters must be an integer >= 2"):
        validate_cluster_config({"num_clusters": 1})

    with pytest.raises(ValueError, match="memories_per_cluster must be an integer >= 2"):
        validate_cluster_config({"memories_per_cluster": 1})

    # cluster_separation finite checks
    with pytest.raises(ValueError, match="cluster_separation must be a positive finite number"):
        validate_cluster_config({"cluster_separation": 0.0})
    with pytest.raises(ValueError, match="cluster_separation must be a positive finite number"):
        validate_cluster_config({"cluster_separation": float("nan")})
    with pytest.raises(ValueError, match="cluster_separation must be a positive finite number"):
        validate_cluster_config({"cluster_separation": float("inf")})

    # cluster_noise finite checks
    with pytest.raises(ValueError, match="cluster_noise must be a non-negative finite number"):
        validate_cluster_config({"cluster_noise": -0.1})
    with pytest.raises(ValueError, match="cluster_noise must be a non-negative finite number"):
        validate_cluster_config({"cluster_noise": float("nan")})
    with pytest.raises(ValueError, match="cluster_noise must be a non-negative finite number"):
        validate_cluster_config({"cluster_noise": float("inf")})

    # query_noise finite checks
    with pytest.raises(ValueError, match="query_noise must be a non-negative finite number"):
        validate_cluster_config({"query_noise": -0.1})
    with pytest.raises(ValueError, match="query_noise must be a non-negative finite number"):
        validate_cluster_config({"query_noise": float("nan")})

    with pytest.raises(ValueError, match="target_cluster must be an integer in"):
        validate_cluster_config({"num_clusters": 4, "target_cluster": 5})

    # beta_values checks (reused Task 12 validator)
    with pytest.raises(ValueError, match="beta_values must be a non-empty list"):
        validate_cluster_config({"beta_values": []})
    with pytest.raises(ValueError, match="All beta_values must be positive finite numbers"):
        validate_cluster_config({"beta_values": [1.0, -0.5]})
    with pytest.raises(ValueError, match="All beta_values must be positive finite numbers"):
        validate_cluster_config({"beta_values": [float("nan")]})
    with pytest.raises(ValueError, match="All beta_values must be positive finite numbers"):
        validate_cluster_config({"beta_values": [float("inf")]})

    # Genuine boolean checks (prevent bool("false") -> True trap)
    with pytest.raises(ValueError, match="normalize must be a genuine boolean"):
        validate_cluster_config({"normalize": "false"})
    with pytest.raises(ValueError, match="save_plot must be a genuine boolean"):
        validate_cluster_config({"save_plot": "false"})

    with pytest.raises(ValueError, match="num_updates must be a positive integer"):
        validate_cluster_config({"num_updates": 0})
    with pytest.raises(ValueError, match="num_updates must be a positive integer"):
        validate_cluster_config({"num_updates": True})

    # Dynamics settings
    with pytest.raises(ValueError, match="compute_dynamics must be a genuine boolean"):
        validate_cluster_config({"compute_dynamics": "yes"})
    with pytest.raises(ValueError, match="perturbation_noise must be a positive finite number"):
        validate_cluster_config({"perturbation_noise": 0.0})
    with pytest.raises(ValueError, match="perturbation_noise must be a positive finite number"):
        validate_cluster_config({"perturbation_noise": float("nan")})
    with pytest.raises(ValueError, match="perturbation_steps must be a positive integer"):
        validate_cluster_config({"perturbation_steps": 0})


def test_missing_config_file_raises_filenotfound(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that an explicitly missing config file fails clearly with FileNotFoundError."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["cluster_retrieval.py", "--config", "non_existent_config_file.yaml"],
    )
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        main()


def _expected_target_cluster_mass(cfg: dict[str, Any], seed: int, beta: float) -> float:
    """Recompute mean target-cluster mass for one seed directly from the generators."""
    generator = torch.Generator().manual_seed(seed)
    memory, labels, centers = generate_clustered_memory(
        num_clusters=cfg["num_clusters"],
        memories_per_cluster=cfg["memories_per_cluster"],
        dim=cfg["dim"],
        cluster_separation=cfg["cluster_separation"],
        cluster_noise=cfg["cluster_noise"],
        generator=generator,
    )
    queries, _ = generate_queries(
        cluster_centers=centers,
        target_cluster=cfg["target_cluster"],
        num_queries=cfg["num_queries"],
        query_noise=cfg["query_noise"],
        generator=generator,
    )
    _, weights = multi_step_retrieve(queries, memory, beta=beta, num_updates=1)
    target_mass = weights[:, labels == cfg["target_cluster"]].sum(dim=-1)
    return target_mass.mean().item()


def test_run_benchmark_saves_seed_specific_reports_end_to_end() -> None:
    """run_benchmark(output_dir=...) writes per-seed and aggregate values matching recomputation."""
    seeds = [11, 22]
    config: dict[str, Any] = {
        "seeds": seeds,
        "dim": 16,
        "num_clusters": 3,
        "memories_per_cluster": 5,
        "cluster_separation": 2.5,
        "cluster_noise": 0.1,
        "query_noise": 0.05,
        "num_queries": 10,
        "target_cluster": 0,
        "beta_values": [0.1, 1.0],
        "save_plot": False,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        run_benchmark(config, output_dir=out_dir, save_plot=False)

        with open(out_dir / "results.json", "r", encoding="utf-8") as f:
            payload = json.load(f)
        with open(out_dir / "results.csv", "r", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        with open(out_dir / "per_seed_results.csv", "r", encoding="utf-8") as f:
            per_seed_rows = list(csv.DictReader(f))

    assert payload["metadata"]["seeds"] == seeds
    assert payload["metadata"]["perturbation_noise"] == 0.05
    assert len(payload["per_seed_results"]) == len(seeds)
    assert len(per_seed_rows) == len(seeds) * len(config["beta_values"])

    for b_idx, beta in enumerate(config["beta_values"]):
        expected = [_expected_target_cluster_mass(config, s, beta) for s in seeds]
        # Seeds produce genuinely different values
        assert expected[0] != pytest.approx(expected[1], abs=1e-6)

        # Saved per-seed values (JSON and CSV) match independent recomputation
        for s_idx, s in enumerate(seeds):
            json_row = payload["per_seed_results"][s_idx][b_idx]
            assert json_row["target_cluster_mass"] == pytest.approx(expected[s_idx], rel=1e-5)
            csv_row = next(
                r for r in per_seed_rows if int(r["seed"]) == s and float(r["beta"]) == beta
            )
            assert float(csv_row["target_cluster_mass"]) == pytest.approx(expected[s_idx], rel=1e-5)

        # Aggregate statistics match independently computed mean and sample std
        agg = payload["results"][b_idx]
        assert agg["target_cluster_mass"] == pytest.approx(statistics.fmean(expected), rel=1e-5)
        assert agg["target_cluster_mass_seed_std"] == pytest.approx(
            statistics.stdev(expected), rel=1e-4, abs=1e-9
        )
        assert float(csv_rows[b_idx]["target_cluster_mass_seed_std"]) == pytest.approx(
            agg["target_cluster_mass_seed_std"]
        )
        within = [
            payload["per_seed_results"][i][b_idx]["target_cluster_mass_query_std"]
            for i in range(len(seeds))
        ]
        assert agg["target_cluster_mass_mean_within_seed_query_std"] == pytest.approx(
            statistics.fmean(within)
        )


def test_metric_uncertainty_is_decided_per_metric() -> None:
    """A metric with zero spread gets no band without suppressing other metrics' bands."""
    results = [
        {"beta": 0.1, "seed_count": 1, "target_cluster_mass_query_std": 0.0, "top1_mass_query_std": 0.02},
        {"beta": 1.0, "seed_count": 1, "target_cluster_mass_query_std": 0.0, "top1_mass_query_std": 0.05},
    ]
    assert metric_uncertainty(results, "target_cluster_mass") is None
    assert metric_uncertainty(results, "top1_mass") == [0.02, 0.05]
    # Missing or undefined std means no band for that metric
    assert metric_uncertainty(results, "entropy") is None
    results[1]["top1_mass_query_std"] = None
    assert metric_uncertainty(results, "top1_mass") is None
    # Multi-seed rows use seed std
    seeded = [{"beta": 1.0, "seed_count": 3, "top1_mass_seed_std": 0.1, "top1_mass_query_std": 0.4}]
    assert metric_uncertainty(seeded, "top1_mass") == [0.1]


def test_plot_cluster_benchmark_generation() -> None:
    """Verify plot generation produces a valid non-empty PNG when Matplotlib is present."""
    pytest.importorskip("matplotlib")
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
        plot_path = out_dir / "cluster_retrieval.png"
        assert plot_path.exists()
        assert plot_path.stat().st_size > 1000

        # Direct call test
        custom_plot = out_dir / "direct_plot.png"
        success = plot_cluster_benchmark(results, custom_plot, cfg=meta)
        assert success is True
        assert custom_plot.exists()
        assert custom_plot.stat().st_size > 1000


def test_plot_fallback_when_matplotlib_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify plot_cluster_benchmark gracefully warns and returns False when Matplotlib is absent."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if "matplotlib" in name:
            raise ImportError("Simulated missing matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_path = Path(tmpdir) / "test.png"
        sample_results = [
            {
                "beta": 1.0,
                "target_cluster_mass": 0.9,
                "top1_mass": 0.2,
                "entropy": 2.0,
                "cluster_entropy": 0.1,
                "distance_to_cluster": 0.5,
                "distance_to_nearest_memory": 1.5,
            }
        ]

        with pytest.warns(UserWarning, match="matplotlib is not installed"):
            result = plot_cluster_benchmark(sample_results, dummy_path)

        assert result is False
        assert not dummy_path.exists()


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
