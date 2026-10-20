"""Beta sweep experiment for modern Hopfield memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Any

import torch
import yaml

from benchmarks.synthetic_retrieval import generate_memory, generate_noisy_queries
from benchmarks.validation import validate_beta_values
from hopfieldpc.diagnostics import (
    attention_entropy,
    max_attention_weight,
    retrieval_distance,
)
from hopfieldpc.functional import multi_step_retrieve

DEFAULT_BETAS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def validate_beta_sweep_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize configuration parameters for the beta sweep.

    Note:
        beta_sweep.py evaluates retrieval metrics across a range of beta values while
        keeping other experimental parameters (memory size, noise level, query count,
        and update steps) fixed. Multi-value lists for these parameters are rejected
        with an explicit ValueError to prevent unintended silent truncation.
    """
    seed = config.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}.")

    dim = config.get("dim")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"dim must be a positive integer, got {dim!r}.")

    # Resolve num_memories / memory_size / memory_sizes
    # Check memory_sizes first so that multi-value lists are detected even if num_memories is also present
    if "memory_sizes" in config:
        raw_memories = config["memory_sizes"]
    elif "memory_size" in config:
        raw_memories = config["memory_size"]
    elif "num_memories" in config:
        raw_memories = config["num_memories"]
    else:
        raise ValueError("Configuration must specify num_memories or memory_size.")

    if isinstance(raw_memories, list):
        if len(raw_memories) > 1:
            raise ValueError(
                f"beta_sweep.py sweeps only beta and fixes memory size to a single value. "
                f"Got multi-value list for memory_sizes: {raw_memories}. "
                f"Please specify a single integer via 'num_memories' or 'memory_size' (or a single-element list)."
            )
        if len(raw_memories) == 0:
            raise ValueError("memory_sizes list cannot be empty.")
        num_memories = raw_memories[0]
    else:
        num_memories = raw_memories

    if (
        isinstance(num_memories, bool)
        or not isinstance(num_memories, int)
        or num_memories <= 0
    ):
        raise ValueError(f"num_memories must be a positive integer, got {num_memories!r}.")

    # Resolve num_queries / num_trials
    if "num_queries" in config:
        raw_queries = config["num_queries"]
    elif "num_trials" in config:
        raw_queries = config["num_trials"]
    else:
        raise ValueError("Configuration must specify num_queries or num_trials.")

    if isinstance(raw_queries, list):
        if len(raw_queries) > 1:
            raise ValueError(
                f"beta_sweep.py sweeps only beta and fixes query count to a single value. "
                f"Got multi-value list for queries/trials: {raw_queries}. "
                f"Please specify a single integer via 'num_queries' or 'num_trials' (or a single-element list)."
            )
        if len(raw_queries) == 0:
            raise ValueError("num_queries list cannot be empty.")
        num_queries = raw_queries[0]
    else:
        num_queries = raw_queries

    if (
        isinstance(num_queries, bool)
        or not isinstance(num_queries, int)
        or num_queries <= 0
    ):
        raise ValueError(f"num_queries must be a positive integer, got {num_queries!r}.")

    # Resolve noise_level / noise_levels
    # Check noise_levels first so that multi-value lists are detected even if noise_level is also present
    if "noise_levels" in config:
        raw_noise = config["noise_levels"]
    elif "noise_level" in config:
        raw_noise = config["noise_level"]
    else:
        raise ValueError("Configuration must specify noise_level or noise_levels.")

    if isinstance(raw_noise, list):
        if len(raw_noise) > 1:
            raise ValueError(
                f"beta_sweep.py sweeps only beta and fixes noise level to a single value. "
                f"Got multi-value list for noise_levels: {raw_noise}. "
                f"Please specify a single numeric value via 'noise_level' (or a single-element list)."
            )
        if len(raw_noise) == 0:
            raise ValueError("noise_levels list cannot be empty.")
        raw_noise = raw_noise[0]

    if (
        isinstance(raw_noise, bool)
        or not isinstance(raw_noise, (int, float))
        or raw_noise < 0
    ):
        raise ValueError(f"noise_level must be a non-negative number, got {raw_noise!r}.")
    noise_level = float(raw_noise)

    beta_values = config.get("beta_values", DEFAULT_BETAS)
    resolved_betas = validate_beta_values(beta_values)

    normalize = bool(config.get("normalize", False))

    raw_updates = config.get("num_updates", 1)
    if isinstance(raw_updates, list):
        if len(raw_updates) > 1:
            raise ValueError(
                f"beta_sweep.py sweeps only beta and fixes num_updates to a single value. "
                f"Got multi-value list for num_updates: {raw_updates}. "
                f"Please specify a single positive integer for 'num_updates' (or a single-element list)."
            )
        if len(raw_updates) == 0:
            raise ValueError("num_updates list cannot be empty.")
        num_updates = raw_updates[0]
    else:
        num_updates = raw_updates

    if (
        isinstance(num_updates, bool)
        or not isinstance(num_updates, int)
        or num_updates <= 0
    ):
        raise ValueError(f"num_updates must be a positive integer, got {num_updates!r}.")

    reports_dir = config.get("reports_dir", "reports/beta_sweep")
    save_plot = bool(config.get("save_plot", True))

    return {
        "seed": seed,
        "dim": dim,
        "num_memories": num_memories,
        "num_queries": num_queries,
        "noise_level": noise_level,
        "beta_values": resolved_betas,
        "normalize": normalize,
        "num_updates": num_updates,
        "reports_dir": reports_dir,
        "save_plot": save_plot,
    }


def evaluate_beta_retrieval(
    *,
    queries: torch.Tensor,
    target_indices: torch.Tensor,
    memory: torch.Tensor,
    beta: float,
    num_updates: int = 1,
    normalize: bool = False,
) -> dict[str, float]:
    """Run retrieval with a specific beta and calculate evaluation metrics.

    Metric Behavior across Beta:
    - top1_accuracy: Largely beta-invariant for single-step retrieval because any
      positive beta preserves the argmax ordering of logits (dot products). It serves
      as a recall sanity check rather than a sensitivity indicator.
    - mean_entropy & normalized_entropy: Main beta-sensitive metrics measuring attention
      dispersion across memories (decreases monotonically as beta increases).
    - mean_max_weight: Main beta-sensitive metric measuring attention sharpness and
      attractor concentration (increases monotonically towards 1.0 as beta increases).
    - mean_retrieval_distance: Main beta-sensitive metric measuring Euclidean distance
      to the ground-truth target memory (decreases as beta increases and retrieval
      transitions from a diffuse blend to the exact target attractor).
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta!r}.")

    retrieved, weights = multi_step_retrieve(
        query=queries,
        memory=memory,
        beta=beta,
        num_updates=num_updates,
        normalize=normalize,
    )

    # Numerical validity checks
    if not torch.isfinite(weights).all():
        raise FloatingPointError(f"Encountered non-finite weights for beta={beta}.")
    if not torch.isfinite(retrieved).all():
        raise FloatingPointError(f"Encountered non-finite retrieved state for beta={beta}.")

    weights_sum = weights.sum(dim=-1)
    expected_sum = torch.ones_like(weights_sum)
    if not torch.allclose(weights_sum, expected_sum, atol=1e-4):
        raise ValueError(f"Attention weights do not sum to 1 for beta={beta}.")

    # 1. Top-1 nearest-memory retrieval accuracy
    # (Argmax ordering is preserved under positive beta scaling, making this metric
    # mostly beta-invariant; useful as a recall sanity check).
    predicted_indices = weights.argmax(dim=-1)
    top1_accuracy = (predicted_indices == target_indices).float().mean().item()

    # 2. Attention entropy (primary beta-sensitive metric for dispersion)
    entropy = attention_entropy(weights)
    mean_entropy = entropy.mean().item()
    norm_entropy = attention_entropy(weights, normalize=True).mean().item()

    # 3. Maximum attention weight (primary beta-sensitive metric for sharpness)
    max_weights = max_attention_weight(weights)
    mean_max_weight = max_weights.mean().item()

    # 4. Mean Euclidean distance to target memory (primary beta-sensitive metric for attractor fidelity)
    targets = memory[target_indices]
    distances = retrieval_distance(retrieved, targets, p=2.0)
    mean_retrieval_distance = distances.mean().item()

    return {
        "beta": beta,
        "top1_accuracy": top1_accuracy,
        "mean_entropy": mean_entropy,
        "normalized_entropy": norm_entropy,
        "mean_max_weight": mean_max_weight,
        "mean_retrieval_distance": mean_retrieval_distance,
    }


def run_beta_sweep(
    config: dict[str, Any],
    output_dir: Path | None = None,
    save_plot: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run beta sweep under strictly controlled conditions.
    """
    cfg = validate_beta_sweep_config(config)

    torch.manual_seed(cfg["seed"])

    dim = cfg["dim"]
    num_memories = cfg["num_memories"]
    num_queries = cfg["num_queries"]
    noise_level = cfg["noise_level"]
    beta_values = cfg["beta_values"]
    normalize = cfg["normalize"]
    num_updates = cfg["num_updates"]

    # Generate single shared memory bank and query set
    memory = generate_memory(num_memories, dim)
    queries, target_indices = generate_noisy_queries(memory, noise_level, num_queries)

    # Compute baseline noisy query distance to target memories
    targets = memory[target_indices]
    baseline_query_distance = retrieval_distance(queries, targets, p=2.0).mean().item()

    results: list[dict[str, Any]] = []

    for beta in beta_values:
        metrics = evaluate_beta_retrieval(
            queries=queries,
            target_indices=target_indices,
            memory=memory,
            beta=beta,
            num_updates=num_updates,
            normalize=normalize,
        )
        results.append(metrics)

    metadata: dict[str, Any] = {
        "seed": cfg["seed"],
        "dim": dim,
        "num_memories": num_memories,
        "num_queries": num_queries,
        "noise_level": noise_level,
        "num_updates": num_updates,
        "normalize": normalize,
        "baseline_query_distance": baseline_query_distance,
    }

    if output_dir is not None:
        save_reports(results, output_dir, metadata=metadata)
        if save_plot and cfg.get("save_plot", True):
            plot_path = output_dir / "beta_sweep.png"
            plot_beta_sweep(results, plot_path, baseline_query_distance=baseline_query_distance)

    return results, metadata


def print_beta_sweep_table(
    results: list[dict[str, Any]],
    baseline_query_distance: float | None = None,
) -> None:
    header = (
        f"{'beta':>7} | {'top1_accuracy':>13} | {'mean_entropy':>12} | "
        f"{'normalized_entropy':>18} | {'mean_max_weight':>15} | {'mean_retrieval_distance':>23}"
    )
    separator = "-" * len(header)

    print(header)
    print(separator)
    for row in results:
        print(
            f"{row['beta']:>7.2f} | "
            f"{row['top1_accuracy']:>13.4f} | "
            f"{row['mean_entropy']:>12.4f} | "
            f"{row['normalized_entropy']:>18.4f} | "
            f"{row['mean_max_weight']:>15.4f} | "
            f"{row['mean_retrieval_distance']:>23.4f}"
        )
    print(separator)
    if baseline_query_distance is not None:
        print(f"Baseline mean query distance to target memory (L2): {baseline_query_distance:.4f}")
    print("Distance measure: Euclidean (L2) distance in feature space.")
    print("Note: Top-1 accuracy is largely beta-invariant (positive beta preserves argmax);")
    print("      entropy, max attention weight, and retrieval distance are the primary beta-sensitive metrics.\n")


def save_reports(
    results: list[dict[str, Any]],
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save beta sweep results and metadata to JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "metadata": metadata or {},
        "results": results,
    }
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    csv_path = output_dir / "results.csv"
    if results:
        fieldnames = [
            "beta",
            "top1_accuracy",
            "mean_entropy",
            "normalized_entropy",
            "mean_max_weight",
            "mean_retrieval_distance",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in fieldnames})


def plot_beta_sweep(
    results: list[dict[str, Any]],
    output_path: Path,
    baseline_query_distance: float | None = None,
) -> bool:
    """Generate a 4-panel diagnostic plot of beta sweep results.

    Returns:
        bool: True if the plot was generated and saved successfully, False if
        matplotlib is unavailable or plot generation was skipped.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn(
            "matplotlib is not installed; skipping plot generation. "
            "To enable diagnostic plots, install matplotlib (e.g., pip install matplotlib).",
            UserWarning,
            stacklevel=2,
        )
        return False

    betas = [r["beta"] for r in results]
    accs = [r["top1_accuracy"] for r in results]
    entropies = [r["mean_entropy"] for r in results]
    norm_entropies = [r["normalized_entropy"] for r in results]
    max_weights = [r["mean_max_weight"] for r in results]
    distances = [r["mean_retrieval_distance"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(r"Hopfield Memory Retrieval: Inverse Temperature ($\beta$) Sweep", fontsize=14)

    # Panel 1: Top-1 Accuracy
    ax = axes[0, 0]
    ax.plot(betas, accs, marker="o", color="#1f77b4", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"Inverse Temperature $\beta$ (log scale)")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Nearest-Memory Retrieval Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Entropy
    ax = axes[0, 1]
    ax.plot(betas, entropies, marker="s", color="#ff7f0e", linewidth=2, label="Mean Shannon Entropy")
    ax.plot(betas, norm_entropies, marker="^", color="#2ca02c", linestyle="--", label="Normalized Entropy")
    ax.set_xscale("log")
    ax.set_xlabel(r"Inverse Temperature $\beta$ (log scale)")
    ax.set_ylabel("Entropy")
    ax.set_title("Attention Weight Entropy (Dispersion)")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Maximum Attention Weight
    ax = axes[1, 0]
    ax.plot(betas, max_weights, marker="d", color="#d62728", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"Inverse Temperature $\beta$ (log scale)")
    ax.set_ylabel("Mean Max Attention Weight")
    ax.set_title(r"Retrieval Sharpness ($\max_i p_i$)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Panel 4: Retrieval Distance
    ax = axes[1, 1]
    ax.plot(betas, distances, marker="o", color="#9467bd", linewidth=2, label="Retrieved-to-Target Distance")
    if baseline_query_distance is not None:
        ax.axhline(
            baseline_query_distance,
            color="#7f7f7f",
            linestyle=":",
            linewidth=1.8,
            label=f"Query-to-Target Baseline ({baseline_query_distance:.3f})",
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"Inverse Temperature $\beta$ (log scale)")
    ax.set_ylabel("Mean Euclidean Distance")
    ax.set_title("Euclidean Distance to Target Memory")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run modern Hopfield beta sweep benchmark.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/beta_sweep.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save persistent results.",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=None,
        help="Override Gaussian query noise level.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed.",
    )
    parser.add_argument(
        "--num-memories",
        type=int,
        default=None,
        help="Override number of memories in the bank.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Override number of evaluation queries.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation.",
    )
    args = parser.parse_args()

    # Load configuration
    config: dict[str, Any]
    if args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        # Fallback to defaults if custom config not present
        config = {
            "seed": 42,
            "dim": 64,
            "num_memories": 50,
            "num_queries": 100,
            "noise_level": 0.2,
            "beta_values": DEFAULT_BETAS,
        }

    # Apply CLI overrides
    if args.noise_level is not None:
        config["noise_level"] = args.noise_level
    if args.seed is not None:
        config["seed"] = args.seed
    if args.num_memories is not None:
        config["num_memories"] = args.num_memories
    if args.num_queries is not None:
        config["num_queries"] = args.num_queries

    output_dir = args.output_dir
    if output_dir is None:
        configured_dir = str(config.get("reports_dir", "reports/beta_sweep"))
        output_dir = Path(configured_dir)

    print(f"Running Hopfield Beta Sweep with configuration from: {args.config}")
    print(
        f"Parameters: dim={config.get('dim', 64)}, "
        f"num_memories={config.get('num_memories', 50)}, "
        f"num_queries={config.get('num_queries', 100)}, "
        f"noise_level={config.get('noise_level', 0.2)}, "
        f"seed={config.get('seed', 42)}"
    )
    print(f"Beta values: {config.get('beta_values', DEFAULT_BETAS)}")
    print(f"Reports output directory: {output_dir}\n")

    results, metadata = run_beta_sweep(
        config=config,
        output_dir=output_dir,
        save_plot=not args.no_plot,
    )

    print("\n--- Beta Sweep Results Table ---")
    print_beta_sweep_table(
        results,
        baseline_query_distance=metadata.get("baseline_query_distance"),
    )
    print(f"Reports saved successfully to: {output_dir}")


if __name__ == "__main__":
    main()
