"""Unit tests for synthetic nearest-memory retrieval benchmark and diagnostics."""

import tempfile
from pathlib import Path

import torch

from benchmarks.synthetic_retrieval import run_benchmark
from hopfieldpc.diagnostics import (
    attention_entropy,
    retrieval_distance,
    topk_mass,
    topk_recall,
)


def test_diagnostics_functions() -> None:
    weights = torch.tensor([
        [0.7, 0.2, 0.1, 0.0],
        [0.4, 0.3, 0.2, 0.1],
    ])

    entropy = attention_entropy(weights)
    assert entropy.shape == (2,)
    assert entropy[0].item() < entropy[1].item()

    top2_mass = topk_mass(weights, k=2)
    assert top2_mass.shape == (2,)
    assert torch.isclose(top2_mass[0], torch.tensor(0.9))
    assert torch.isclose(top2_mass[1], torch.tensor(0.7))

    targets = torch.tensor([0, 3])
    recall_top1 = topk_recall(weights, targets, k=1)
    assert recall_top1[0].item() == 1.0

    recall_top2 = topk_recall(weights, targets, k=2)
    assert recall_top2[0].item() == 1.0
    assert recall_top2[1].item() == 0.0

    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    tgt = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    dist = retrieval_distance(q, tgt)
    assert torch.isclose(dist[0], torch.tensor(0.0))
    assert torch.isclose(dist[1], torch.tensor(1.0))


def test_run_benchmark_tiny_config_and_output_files() -> None:
    tiny_config = {
        "seed": 123,
        "dim": 8,
        "num_trials": 4,
        "beta_values": [1.0, 5.0],
        "noise_levels": [0.01],
        "memory_sizes": [5],
        "num_updates": [1, 2],
        "top_k": 2,
        "normalize": False,
        "capacity_threshold": 0.95,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "test_reports"
        results, capacity_summary = run_benchmark(tiny_config, output_dir=output_dir)

        assert len(results) == 4
        expected_metrics = [
            "beta",
            "noise",
            "memory_size",
            "num_updates",
            "top1_acc",
            "topk_recall",
            "query_distance",
            "retrieved_distance",
            "retrieval_improvement",
            "entropy",
            "topk_mass",
            "energy_initial",
            "energy_final",
            "energy_delta",
        ]

        for row in results:
            for metric in expected_metrics:
                assert metric in row
                assert not torch.isnan(torch.tensor(row[metric]))

        assert len(capacity_summary) == 4
        assert "capacity_at_95" in capacity_summary[0]

        assert (output_dir / "results.json").exists()
        assert (output_dir / "results.csv").exists()
        assert (output_dir / "capacity_summary.csv").exists()
