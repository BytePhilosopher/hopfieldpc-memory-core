"""Synthetic nearest-memory retrieval benchmark.

Evaluates Hopfield memory retrieval performance across noise levels,
beta values, memory sizes, and update steps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from hopfieldpc.diagnostics import (
    attention_entropy,
    retrieval_distance,
    topk_mass,
    topk_recall,
)
from hopfieldpc.energy import hopfield_energy
from hopfieldpc.functional import multi_step_retrieve


def generate_memory(num_memories: int, dim: int) -> torch.Tensor:
    """Generate a random memory bank of shape [num_memories, dim]."""
    std = 1.0 / math.sqrt(dim)
    memory = torch.empty(num_memories, dim)
    torch.nn.init.normal_(memory, mean=0.0, std=std)
    return memory


def generate_noisy_queries(
    memory: torch.Tensor, noise_level: float, num_trials: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate queries by adding Gaussian noise to randomly sampled target memories."""
    num_memories, _ = memory.shape
    target_indices = torch.randint(0, num_memories, (num_trials,))
    targets = memory[target_indices]
    noise = torch.randn_like(targets) * noise_level
    queries = targets + noise
    return queries, target_indices


def evaluate_retrieval(
    queries: torch.Tensor,
    target_indices: torch.Tensor,
    memory: torch.Tensor,
    beta: float,
    num_updates: int,
    top_k: int,
    normalize: bool,
) -> dict[str, float]:
    """Run retrieval on an existing query-memory set and compute diagnostics."""
    retrieved, weights = multi_step_retrieve(
        query=queries,
        memory=memory,
        beta=beta,
        num_updates=num_updates,
        normalize=normalize,
    )

    targets = memory[target_indices]
    predicted_indices = weights.argmax(dim=-1)

    top1_acc = (predicted_indices == target_indices).float().mean().item()
    topk_rec = topk_recall(weights, target_indices, k=top_k).mean().item()

    query_dist = retrieval_distance(queries, targets).mean().item()
    retrieved_dist = retrieval_distance(retrieved, targets).mean().item()
    retrieval_improv = query_dist - retrieved_dist

    entropy = attention_entropy(weights).mean().item()
    mass = topk_mass(weights, k=top_k).mean().item()

    energy_initial = hopfield_energy(
        queries, memory, beta=beta, include_constants=True, reduce="mean"
    ).item()
    energy_final = hopfield_energy(
        retrieved, memory, beta=beta, include_constants=True, reduce="mean"
    ).item()
    energy_delta = energy_initial - energy_final

    return {
        "top1_acc": top1_acc,
        "topk_recall": topk_rec,
        "query_distance": query_dist,
        "retrieved_distance": retrieved_dist,
        "retrieval_improvement": retrieval_improv,
        "entropy": entropy,
        "topk_mass": mass,
        "energy_initial": energy_initial,
        "energy_final": energy_final,
        "energy_delta": energy_delta,
    }


def compute_capacity_summary(
    results: list[dict[str, Any]], threshold: float = 0.95
) -> list[dict[str, Any]]:
    """Compute largest memory_size where top1_acc >= threshold for each (beta, noise, num_updates)."""
    groups: dict[tuple[float, float, int], list[dict[str, Any]]] = {}
    for r in results:
        key = (r["beta"], r["noise"], r["num_updates"])
        groups.setdefault(key, []).append(r)

    summary = []
    for (beta, noise, num_updates), group in groups.items():
        qualifying = [r["memory_size"] for r in group if r["top1_acc"] >= threshold]
        capacity = max(qualifying) if qualifying else 0
        summary.append({
            "beta": beta,
            "noise": noise,
            "num_updates": num_updates,
            f"capacity_at_{int(threshold*100)}": capacity,
        })
    return summary


def print_results_table(results: list[dict[str, Any]]) -> None:
    """Print results in a clean formatted table."""
    header = (
        f"{'beta':>7} | {'noise':>7} | {'mem_sz':>6} | {'upd':>3} | "
        f"{'top1_acc':>8} | {'topk_rec':>8} | {'query_dst':>9} | "
        f"{'ret_dst':>8} | {'improv':>7} | {'entropy':>7} | {'topk_mass':>9} | "
        f"{'e_init':>8} | {'e_final':>8} | {'e_delta':>8}"
    )
    separator = "-" * len(header)

    print(header)
    print(separator)
    for row in results:
        print(
            f"{row['beta']:>7.2f} | {row['noise']:>7.2f} | {row['memory_size']:>6d} | {row['num_updates']:>3d} | "
            f"{row['top1_acc']:>8.4f} | {row['topk_recall']:>8.4f} | {row['query_distance']:>9.4f} | "
            f"{row['retrieved_distance']:>8.4f} | {row['retrieval_improvement']:>7.4f} | {row['entropy']:>7.4f} | "
            f"{row['topk_mass']:>9.4f} | {row['energy_initial']:>8.2f} | {row['energy_final']:>8.2f} | "
            f"{row['energy_delta']:>8.2f}"
        )


def print_capacity_table(summary: list[dict[str, Any]], threshold: float = 0.95) -> None:
    """Print capacity summary table."""
    cap_key = f"capacity_at_{int(threshold*100)}"
    header = f"{'beta':>7} | {'noise':>7} | {'updates':>7} | {cap_key:>15}"
    separator = "-" * len(header)

    print(header)
    print(separator)
    for row in summary:
        print(
            f"{row['beta']:>7.2f} | {row['noise']:>7.2f} | {row['num_updates']:>7d} | {row[cap_key]:>15d}"
        )


def save_reports(
    results: list[dict[str, Any]],
    capacity_summary: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save benchmark results and summary to CSV and JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_path = output_dir / "results.csv"
    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    cap_csv_path = output_dir / "capacity_summary.csv"
    if capacity_summary:
        with open(cap_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(capacity_summary[0].keys()))
            writer.writeheader()
            writer.writerows(capacity_summary)


def run_benchmark(
    config: dict[str, Any], output_dir: Path | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run full synthetic retrieval sweep according to configuration."""
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    dim = config["dim"]
    num_trials = config["num_trials"]
    top_k = config.get("top_k", 3)
    normalize = config.get("normalize", False)
    threshold = config.get("capacity_threshold", 0.95)

    beta_values = config["beta_values"]
    noise_levels = config["noise_levels"]
    memory_sizes = config["memory_sizes"]
    num_updates_list = config.get("num_updates", [1])

    results = []

    for memory_size in memory_sizes:
        for noise in noise_levels:
            memory = generate_memory(memory_size, dim)
            queries, target_indices = generate_noisy_queries(memory, noise, num_trials)

            for beta in beta_values:
                for updates in num_updates_list:
                    metrics = evaluate_retrieval(
                        queries=queries,
                        target_indices=target_indices,
                        memory=memory,
                        beta=beta,
                        num_updates=updates,
                        top_k=top_k,
                        normalize=normalize,
                    )
                    results.append({
                        "beta": beta,
                        "noise": noise,
                        "memory_size": memory_size,
                        "num_updates": updates,
                        **metrics,
                    })

    capacity_summary = compute_capacity_summary(results, threshold=threshold)

    if output_dir is not None:
        save_reports(results, capacity_summary, output_dir)

    return results, capacity_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic nearest-memory retrieval benchmark.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthetic_retrieval.yaml"),
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save persistent results."
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found at {args.config}")
        raise SystemExit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = args.output_dir
    if output_dir is None:
        configured_dir = config.get("reports_dir", "reports/synthetic_retrieval")
        output_dir = Path(configured_dir)

    print(f"Running benchmark with config from {args.config}...")
    print(f"Parameters: dim={config['dim']}, num_trials={config['num_trials']}, top_k={config.get('top_k', 3)}")
    print(f"Output reports directory: {output_dir}\n")

    results, capacity_summary = run_benchmark(config, output_dir=output_dir)

    print("\n--- Detailed Retrieval Results ---")
    print_results_table(results)

    threshold = config.get("capacity_threshold", 0.95)
    print(f"\n--- Capacity @ {int(threshold * 100)}% Accuracy Summary ---")
    print_capacity_table(capacity_summary, threshold=threshold)
    print(f"\nReports saved successfully to {output_dir}")


if __name__ == "__main__":
    main()
