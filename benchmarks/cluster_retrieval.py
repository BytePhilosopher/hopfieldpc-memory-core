"""Cluster and metastable retrieval benchmark for modern Hopfield memory.
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

from hopfieldpc.diagnostics import (
    argmax_memory_index,
    attention_entropy,
    max_attention_weight,
    retrieval_distance,
)
from hopfieldpc.functional import multi_step_retrieve

DEFAULT_BETAS = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


def validate_cluster_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize configuration parameters for cluster retrieval."""
    seed = config.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}.")

    seeds_raw = config.get("seeds", None)
    resolved_seeds: list[int] | None = None
    if seeds_raw is not None:
        if not isinstance(seeds_raw, list) or len(seeds_raw) == 0:
            raise ValueError("seeds must be a non-empty list of non-negative integers.")
        resolved_seeds = []
        for s in seeds_raw:
            if isinstance(s, bool) or not isinstance(s, int) or s < 0:
                raise ValueError(f"All seeds must be non-negative integers, got {s!r}.")
            resolved_seeds.append(s)

    dim = config.get("dim", 64)
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"dim must be a positive integer, got {dim!r}.")

    num_clusters = config.get("num_clusters", 4)
    if isinstance(num_clusters, bool) or not isinstance(num_clusters, int) or num_clusters < 2:
        raise ValueError(f"num_clusters must be an integer >= 2, got {num_clusters!r}.")

    memories_per_cluster = config.get("memories_per_cluster", 10)
    if (
        isinstance(memories_per_cluster, bool)
        or not isinstance(memories_per_cluster, int)
        or memories_per_cluster < 2
    ):
        raise ValueError(
            f"memories_per_cluster must be an integer >= 2, got {memories_per_cluster!r}."
        )

    cluster_separation = config.get("cluster_separation", 3.0)
    if (
        isinstance(cluster_separation, bool)
        or not isinstance(cluster_separation, (int, float))
        or cluster_separation <= 0
    ):
        raise ValueError(
            f"cluster_separation must be a positive number, got {cluster_separation!r}."
        )
    cluster_separation = float(cluster_separation)

    cluster_noise = config.get("cluster_noise", 0.15)
    if (
        isinstance(cluster_noise, bool)
        or not isinstance(cluster_noise, (int, float))
        or cluster_noise < 0
    ):
        raise ValueError(f"cluster_noise must be a non-negative number, got {cluster_noise!r}.")
    cluster_noise = float(cluster_noise)

    query_noise = config.get("query_noise", 0.10)
    if (
        isinstance(query_noise, bool)
        or not isinstance(query_noise, (int, float))
        or query_noise < 0
    ):
        raise ValueError(f"query_noise must be a non-negative number, got {query_noise!r}.")
    query_noise = float(query_noise)

    num_queries = config.get("num_queries", 50)
    if isinstance(num_queries, bool) or not isinstance(num_queries, int) or num_queries <= 0:
        raise ValueError(f"num_queries must be a positive integer, got {num_queries!r}.")

    target_cluster = config.get("target_cluster", 0)
    if target_cluster is not None and (
        isinstance(target_cluster, bool)
        or not isinstance(target_cluster, int)
        or target_cluster < 0
        or target_cluster >= num_clusters
    ):
        raise ValueError(
            f"target_cluster must be an integer in [0, {num_clusters - 1}], got {target_cluster!r}."
        )

    beta_values = config.get("beta_values", DEFAULT_BETAS)
    if not isinstance(beta_values, list) or len(beta_values) == 0:
        raise ValueError("beta_values must be a non-empty list of positive numbers.")

    resolved_betas: list[float] = []
    for b in beta_values:
        if isinstance(b, bool) or not isinstance(b, (int, float)) or b <= 0:
            raise ValueError(f"All beta_values must be positive numbers, got {b!r}.")
        resolved_betas.append(float(b))

    normalize = bool(config.get("normalize", False))

    num_updates = config.get("num_updates", 1)
    if isinstance(num_updates, bool) or not isinstance(num_updates, int) or num_updates <= 0:
        raise ValueError(f"num_updates must be a positive integer, got {num_updates!r}.")

    reports_dir = config.get("reports_dir", "reports/cluster_retrieval")
    save_plot = bool(config.get("save_plot", True))

    return {
        "seed": seed,
        "seeds": resolved_seeds,
        "dim": dim,
        "num_clusters": num_clusters,
        "memories_per_cluster": memories_per_cluster,
        "cluster_separation": cluster_separation,
        "cluster_noise": cluster_noise,
        "query_noise": query_noise,
        "num_queries": num_queries,
        "target_cluster": target_cluster,
        "beta_values": resolved_betas,
        "normalize": normalize,
        "num_updates": num_updates,
        "reports_dir": reports_dir,
        "save_plot": save_plot,
    }


def generate_cluster_centers(
    num_clusters: int,
    dim: int,
    separation: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate clearly separated cluster centers in feature space.

    When num_clusters <= dim, cluster centers are constructed from an orthogonal
    frame (via QR decomposition) scaled by separation. This mathematically guarantees
    equidistant, non-overlapping cluster centers with zero mutual dot product.
    When num_clusters > dim, centers are normalized random Gaussian vectors
    scaled by separation.
    """
    if num_clusters <= dim:
        random_matrix = torch.randn(dim, num_clusters, generator=generator)
        q, _ = torch.linalg.qr(random_matrix)
        centers = q.T * separation  # [num_clusters, dim]
    else:
        raw_centers = torch.randn(num_clusters, dim, generator=generator)
        norm = torch.linalg.vector_norm(raw_centers, dim=-1, keepdim=True).clamp_min(1e-8)
        centers = (raw_centers / norm) * separation
    return centers


def generate_clustered_memory(
    num_clusters: int,
    memories_per_cluster: int,
    dim: int,
    cluster_separation: float,
    cluster_noise: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct a synthetic memory bank containing multiple separated clusters.

    Formula:
        M_{c, j} = center_c + noise_{c, j}
        where noise ~ N(0, cluster_noise^2 * I)

    Returns:
        memory: Tensor with shape [num_memories, dim].
        cluster_labels: LongTensor with shape [num_memories] containing cluster IDs.
        cluster_centers: Tensor with shape [num_clusters, dim].
    """
    centers = generate_cluster_centers(
        num_clusters=num_clusters,
        dim=dim,
        separation=cluster_separation,
        generator=generator,
    )

    memories_list: list[torch.Tensor] = []
    labels_list: list[int] = []

    for c in range(num_clusters):
        center = centers[c : c + 1]
        noise = torch.randn(
            memories_per_cluster,
            dim,
            generator=generator,
            dtype=center.dtype,
            device=center.device,
        ) * cluster_noise
        cluster_memories = center + noise
        memories_list.append(cluster_memories)
        labels_list.extend([c] * memories_per_cluster)

    memory = torch.cat(memories_list, dim=0)
    cluster_labels = torch.tensor(labels_list, dtype=torch.long, device=memory.device)

    return memory, cluster_labels, centers


def generate_queries(
    cluster_centers: torch.Tensor,
    target_cluster: int | None,
    num_queries: int,
    query_noise: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate evaluation queries near designated target clusters.

    Args:
        cluster_centers: Tensor of cluster centers [num_clusters, dim].
        target_cluster: Specific target cluster index, or None for uniform sampling.
        num_queries: Number of query vectors to produce.
        query_noise: Standard deviation of Gaussian perturbation around cluster center.
        generator: Optional PyTorch random number generator.

    Returns:
        queries: Tensor of shape [num_queries, dim].
        query_targets: LongTensor of shape [num_queries] with ground-truth cluster IDs.
    """
    num_clusters, dim = cluster_centers.shape

    if target_cluster is not None:
        target_indices = torch.full(
            (num_queries,),
            target_cluster,
            dtype=torch.long,
            device=cluster_centers.device,
        )
    else:
        target_indices = torch.randint(
            0,
            num_clusters,
            (num_queries,),
            generator=generator,
            device=cluster_centers.device,
        )

    targets = cluster_centers[target_indices]
    noise = torch.randn(
        num_queries,
        dim,
        generator=generator,
        dtype=cluster_centers.dtype,
        device=cluster_centers.device,
    ) * query_noise
    queries = targets + noise

    return queries, target_indices


def compute_cluster_masses(
    weights: torch.Tensor,
    cluster_labels: torch.Tensor,
    num_clusters: int,
) -> torch.Tensor:
    """Compute attention mass assigned to each cluster.

    Args:
        weights: Softmax retrieval weights [batch_size, num_memories].
        cluster_labels: Cluster membership per memory [num_memories].
        num_clusters: Total number of clusters.

    Returns:
        Tensor of shape [batch_size, num_clusters] where entry [i, c] is the sum
        of attention weights assigned to memories belonging to cluster c.
    """
    batch_size, num_memories = weights.shape
    if cluster_labels.shape[0] != num_memories:
        raise ValueError(
            f"cluster_labels length ({cluster_labels.shape[0]}) must match "
            f"number of memories in weights ({num_memories})."
        )

    cluster_masses = torch.zeros(
        batch_size,
        num_clusters,
        dtype=weights.dtype,
        device=weights.device,
    )

    for c in range(num_clusters):
        mask = cluster_labels == c
        cluster_masses[:, c] = weights[:, mask].sum(dim=-1)

    return cluster_masses


def compute_cluster_metrics(
    retrieved: torch.Tensor,
    weights: torch.Tensor,
    memory: torch.Tensor,
    cluster_centers: torch.Tensor,
    cluster_labels: torch.Tensor,
    target_clusters: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute per-query cluster retrieval diagnostics across all metric dimensions.

    Args:
        retrieved: Retrieved memory vectors [batch_size, dim].
        weights: Softmax attention weights [batch_size, num_memories].
        memory: Stored memory bank [num_memories, dim].
        cluster_centers: Cluster prototype vectors [num_clusters, dim].
        cluster_labels: Cluster assignment per memory [num_memories].
        target_clusters: Ground truth target cluster per query [batch_size].

    Returns:
        Dictionary of per-query metric tensors:
            - target_cluster_mass: Probability mass allocated to the target cluster [batch_size]
            - all_cluster_masses: Probability mass per cluster [batch_size, num_clusters]
            - top1_mass: Highest single memory attention weight [batch_size]
            - entropy: Shannon entropy of memory-level attention distribution [batch_size]
            - cluster_entropy: Shannon entropy of cluster-level probability distribution [batch_size]
            - distance_to_cluster: Euclidean distance to target cluster center [batch_size]
            - distance_to_nearest_memory: Minimum Euclidean distance to any stored memory [batch_size]
            - top1_in_target_cluster: Binary indicator whether top-1 memory is in target cluster [batch_size]
            - effective_memory_count: Inverse participation ratio 1 / sum(w_i^2) [batch_size]
            - effective_cluster_count: Inverse participation ratio 1 / sum(c_k^2) [batch_size]
    """
    num_clusters = cluster_centers.shape[0]
    cluster_masses = compute_cluster_masses(weights, cluster_labels, num_clusters)

    batch_indices = torch.arange(retrieved.shape[0], device=retrieved.device)
    target_cluster_mass = cluster_masses[batch_indices, target_clusters]

    top1 = max_attention_weight(weights)
    mem_entropy = attention_entropy(weights)
    clus_entropy = attention_entropy(cluster_masses)

    target_centers = cluster_centers[target_clusters]
    dist_cluster = retrieval_distance(retrieved, target_centers, p=2.0)

    # Minimum distance to any stored memory vector
    pairwise_distances = torch.cdist(retrieved, memory, p=2.0)
    dist_nearest_memory = pairwise_distances.min(dim=-1).values

    top1_indices = argmax_memory_index(weights)
    top1_clusters = cluster_labels[top1_indices]
    top1_in_target = (top1_clusters == target_clusters).float()

    weights_sq_sum = (weights ** 2).sum(dim=-1).clamp_min(1e-12)
    effective_memories = 1.0 / weights_sq_sum

    cluster_masses_sq_sum = (cluster_masses ** 2).sum(dim=-1).clamp_min(1e-12)
    effective_clusters = 1.0 / cluster_masses_sq_sum

    return {
        "target_cluster_mass": target_cluster_mass,
        "all_cluster_masses": cluster_masses,
        "top1_mass": top1,
        "entropy": mem_entropy,
        "cluster_entropy": clus_entropy,
        "distance_to_cluster": dist_cluster,
        "distance_to_nearest_memory": dist_nearest_memory,
        "top1_in_target_cluster": top1_in_target,
        "effective_memory_count": effective_memories,
        "effective_cluster_count": effective_clusters,
    }


def evaluate_cluster_retrieval(
    *,
    queries: torch.Tensor,
    target_clusters: torch.Tensor,
    memory: torch.Tensor,
    cluster_centers: torch.Tensor,
    cluster_labels: torch.Tensor,
    beta: float,
    num_updates: int = 1,
    normalize: bool = False,
) -> dict[str, Any]:
    """Execute Hopfield retrieval with specified beta and calculate aggregated cluster metrics."""
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta!r}.")

    retrieved, weights = multi_step_retrieve(
        query=queries,
        memory=memory,
        beta=beta,
        num_updates=num_updates,
        normalize=normalize,
    )

    if not torch.isfinite(weights).all():
        raise FloatingPointError(f"Encountered non-finite weights for beta={beta}.")
    if not torch.isfinite(retrieved).all():
        raise FloatingPointError(f"Encountered non-finite retrieved state for beta={beta}.")

    weights_sum = weights.sum(dim=-1)
    if not torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-4):
        raise ValueError(f"Attention weights do not sum to 1 for beta={beta}.")

    metrics = compute_cluster_metrics(
        retrieved=retrieved,
        weights=weights,
        memory=memory,
        cluster_centers=cluster_centers,
        cluster_labels=cluster_labels,
        target_clusters=target_clusters,
    )

    cluster_masses_sum = metrics["all_cluster_masses"].sum(dim=-1)
    if not torch.allclose(cluster_masses_sum, torch.ones_like(cluster_masses_sum), atol=1e-4):
        raise ValueError(f"Cluster masses do not sum to 1 for beta={beta}.")

    return {
        "beta": beta,
        "target_cluster_mass": metrics["target_cluster_mass"].mean().item(),
        "target_cluster_mass_std": metrics["target_cluster_mass"].std().item() if queries.shape[0] > 1 else 0.0,
        "top1_mass": metrics["top1_mass"].mean().item(),
        "top1_mass_std": metrics["top1_mass"].std().item() if queries.shape[0] > 1 else 0.0,
        "entropy": metrics["entropy"].mean().item(),
        "entropy_std": metrics["entropy"].std().item() if queries.shape[0] > 1 else 0.0,
        "cluster_entropy": metrics["cluster_entropy"].mean().item(),
        "distance_to_cluster": metrics["distance_to_cluster"].mean().item(),
        "distance_to_cluster_std": metrics["distance_to_cluster"].std().item() if queries.shape[0] > 1 else 0.0,
        "distance_to_nearest_memory": metrics["distance_to_nearest_memory"].mean().item(),
        "distance_to_nearest_memory_std": metrics["distance_to_nearest_memory"].std().item() if queries.shape[0] > 1 else 0.0,
        "top1_in_target_cluster": metrics["top1_in_target_cluster"].mean().item(),
        "effective_memory_count": metrics["effective_memory_count"].mean().item(),
        "effective_cluster_count": metrics["effective_cluster_count"].mean().item(),
    }


def classify_retrieval_regime(
    target_cluster_mass: float,
    top1_mass: float,
    num_clusters: int,
    memories_per_cluster: int,
) -> str:
    """Classify the qualitative retrieval regime based on mass distribution."""
    cluster_metastable_threshold = 0.85
    single_memory_threshold = 0.70

    if target_cluster_mass < cluster_metastable_threshold:
        return "Global Averaging"
    elif top1_mass >= single_memory_threshold:
        return "Single Memory"
    else:
        return "Cluster / Metastable"


def run_single_seed_experiment(
    cfg: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run full beta sweep on a single reproducible random seed."""
    generator = torch.Generator().manual_seed(seed)

    memory, cluster_labels, centers = generate_clustered_memory(
        num_clusters=cfg["num_clusters"],
        memories_per_cluster=cfg["memories_per_cluster"],
        dim=cfg["dim"],
        cluster_separation=cfg["cluster_separation"],
        cluster_noise=cfg["cluster_noise"],
        generator=generator,
    )

    queries, target_clusters = generate_queries(
        cluster_centers=centers,
        target_cluster=cfg["target_cluster"],
        num_queries=cfg["num_queries"],
        query_noise=cfg["query_noise"],
        generator=generator,
    )

    results: list[dict[str, Any]] = []
    for beta in cfg["beta_values"]:
        row = evaluate_cluster_retrieval(
            queries=queries,
            target_clusters=target_clusters,
            memory=memory,
            cluster_centers=centers,
            cluster_labels=cluster_labels,
            beta=beta,
            num_updates=cfg["num_updates"],
            normalize=cfg["normalize"],
        )
        row["regime"] = classify_retrieval_regime(
            target_cluster_mass=row["target_cluster_mass"],
            top1_mass=row["top1_mass"],
            num_clusters=cfg["num_clusters"],
            memories_per_cluster=cfg["memories_per_cluster"],
        )
        results.append(row)

    metadata: dict[str, Any] = {
        "seed": seed,
        "dim": cfg["dim"],
        "num_clusters": cfg["num_clusters"],
        "memories_per_cluster": cfg["memories_per_cluster"],
        "num_memories": cfg["num_clusters"] * cfg["memories_per_cluster"],
        "cluster_separation": cfg["cluster_separation"],
        "cluster_noise": cfg["cluster_noise"],
        "query_noise": cfg["query_noise"],
        "num_queries": cfg["num_queries"],
        "target_cluster": cfg["target_cluster"],
        "beta_values": cfg["beta_values"],
        "normalize": cfg["normalize"],
        "num_updates": cfg["num_updates"],
    }

    return results, metadata


def run_benchmark(
    config: dict[str, Any],
    output_dir: Path | None = None,
    save_plot: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run cluster retrieval benchmark across single or multiple random seeds."""
    cfg = validate_cluster_config(config)

    seeds = cfg.get("seeds")
    if seeds is None:
        seeds = [cfg["seed"]]

    all_seed_results: list[list[dict[str, Any]]] = []
    first_metadata: dict[str, Any] | None = None

    for s in seeds:
        seed_results, metadata = run_single_seed_experiment(cfg, seed=s)
        all_seed_results.append(seed_results)
        if first_metadata is None:
            first_metadata = metadata

    assert first_metadata is not None

    if len(seeds) == 1:
        final_results = all_seed_results[0]
        final_metadata = first_metadata
    else:
        # Aggregate across seeds
        final_results = []
        num_betas = len(cfg["beta_values"])
        metric_keys = [
            "target_cluster_mass",
            "top1_mass",
            "entropy",
            "cluster_entropy",
            "distance_to_cluster",
            "distance_to_nearest_memory",
            "top1_in_target_cluster",
            "effective_memory_count",
            "effective_cluster_count",
        ]

        for b_idx in range(num_betas):
            beta = cfg["beta_values"][b_idx]
            agg_row: dict[str, Any] = {"beta": beta}
            for k in metric_keys:
                vals = [seed_res[b_idx][k] for seed_res in all_seed_results]
                mean_val = float(sum(vals) / len(vals))
                variance = float(sum((v - mean_val) ** 2 for v in vals) / len(vals))
                std_val = float(variance ** 0.5)
                agg_row[k] = mean_val
                agg_row[f"{k}_std"] = std_val

            agg_row["regime"] = classify_retrieval_regime(
                target_cluster_mass=agg_row["target_cluster_mass"],
                top1_mass=agg_row["top1_mass"],
                num_clusters=cfg["num_clusters"],
                memories_per_cluster=cfg["memories_per_cluster"],
            )
            final_results.append(agg_row)

        final_metadata = {
            **first_metadata,
            "seeds": seeds,
            "num_seeds": len(seeds),
        }

    if output_dir is not None:
        save_reports(final_results, output_dir, metadata=final_metadata)
        if save_plot and cfg.get("save_plot", True):
            plot_path = output_dir / "cluster_retrieval.png"
            plot_cluster_benchmark(final_results, plot_path, cfg=cfg)

    return final_results, final_metadata


def format_results_table(
    results: list[dict[str, Any]],
    show_extended: bool = True,
) -> str:
    """Format benchmark results into a clean, professional ASCII table."""
    lines: list[str] = []

    if show_extended:
        header = (
            f"{'beta':>7} | {'cluster_mass':>12} | {'top1_mass':>9} | {'entropy':>7} | "
            f"{'cluster_dist':>12} | {'nearest_dist':>12} | {'eff_mem':>7} | {'clus_ent':>8} | {'regime':<20}"
        )
    else:
        header = (
            f"{'beta':>7} | {'cluster_mass':>12} | {'top1_mass':>9} | {'entropy':>7} | "
            f"{'cluster_dist':>12} | {'nearest_dist':>12}"
        )
    separator = "-" * len(header)

    lines.append(header)
    lines.append(separator)

    for row in results:
        b_str = f"{row['beta']:>7.2f}"
        cm_str = f"{row['target_cluster_mass']:>12.4f}"
        top1_str = f"{row['top1_mass']:>9.4f}"
        ent_str = f"{row['entropy']:>7.4f}"
        cdist_str = f"{row['distance_to_cluster']:>12.4f}"
        ndist_str = f"{row['distance_to_nearest_memory']:>12.4f}"

        if show_extended:
            eff_mem_str = f"{row.get('effective_memory_count', 0.0):>7.2f}"
            clus_ent_str = f"{row.get('cluster_entropy', 0.0):>8.4f}"
            regime_str = f"{row.get('regime', 'Unknown'):<20}"
            line = (
                f"{b_str} | {cm_str} | {top1_str} | {ent_str} | "
                f"{cdist_str} | {ndist_str} | {eff_mem_str} | {clus_ent_str} | {regime_str}"
            )
        else:
            line = f"{b_str} | {cm_str} | {top1_str} | {ent_str} | {cdist_str} | {ndist_str}"

        lines.append(line)

    lines.append(separator)
    return "\n".join(lines)


def print_cluster_results_table(
    results: list[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
) -> None:
    """Print the formatted cluster retrieval table and qualitative interpretation."""
    print(format_results_table(results, show_extended=True))

    if cfg is not None:
        num_clusters = cfg.get("num_clusters", 4)
        mem_per_cluster = cfg.get("memories_per_cluster", 10)
        tot_mem = num_clusters * mem_per_cluster
        print(f"\nExperimental Setup: {num_clusters} clusters, {mem_per_cluster} memories/cluster (Total: {tot_mem} memories).")
        print(f"Theoretical limits: Uniform mass across clusters = {1.0 / num_clusters:.4f}, Uniform mass across all memories = {1.0 / tot_mem:.4f}.")
    print("\nQualitative Transition Regimes:")
    print("  1. LOW beta:      Global Averaging (attention spread broadly across clusters, high entropy).")
    print("  2. MEDIUM beta:   Cluster / Metastable Retrieval (target cluster mass ~1.0, low cluster dist, distributed within cluster).")
    print("  3. HIGH beta:     Single-Memory Retrieval (top-1 mass dominates, distance to nearest memory -> 0).\n")


def save_reports(
    results: list[dict[str, Any]],
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save cluster benchmark results and metadata to JSON and CSV files."""
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
            "target_cluster_mass",
            "top1_mass",
            "entropy",
            "cluster_entropy",
            "distance_to_cluster",
            "distance_to_nearest_memory",
            "top1_in_target_cluster",
            "effective_memory_count",
            "effective_cluster_count",
            "regime",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow(r)


def plot_cluster_benchmark(
    results: list[dict[str, Any]],
    output_path: Path,
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Generate a 4-panel scientific diagnostic plot of the cluster retrieval transition.

    Panel 1: Target Cluster Mass vs beta (Regimes: Global average -> Cluster concentration)
    Panel 2: Top-1 Memory Mass vs beta (Regimes: Distributed -> Single-memory collapse)
    Panel 3: Entropy vs beta (Memory-level entropy vs Cluster-level entropy)
    Panel 4: Retrieval Distances vs beta (Cluster center distance vs Nearest memory distance)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn(
            "matplotlib is not installed; skipping plot generation.",
            UserWarning,
            stacklevel=2,
        )
        return False

    betas = [r["beta"] for r in results]
    cluster_masses = [r["target_cluster_mass"] for r in results]
    top1_masses = [r["top1_mass"] for r in results]
    mem_entropies = [r["entropy"] for r in results]
    clus_entropies = [r["cluster_entropy"] for r in results]
    dist_clusters = [r["distance_to_cluster"] for r in results]
    dist_memories = [r["distance_to_nearest_memory"] for r in results]

    num_clusters = cfg.get("num_clusters", 4) if cfg else 4
    mem_per_cluster = cfg.get("memories_per_cluster", 10) if cfg else 10
    total_memories = num_clusters * mem_per_cluster

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        r"Hopfield Memory: Cluster / Metastable Retrieval Benchmark ($\beta$ Transition)",
        fontsize=14,
        fontweight="bold",
    )

    # Panel 1: Target Cluster Mass
    ax = axes[0, 0]
    ax.plot(betas, cluster_masses, marker="o", color="#1f77b4", linewidth=2.2, label="Target Cluster Mass")
    ax.axhline(
        1.0 / num_clusters,
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.8,
        label=f"Global Uniform Baseline ({1.0/num_clusters:.2f})",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"Inverse Temperature $\beta$ (log scale)")
    ax.set_ylabel("Target Cluster Probability Mass")
    ax.set_title(r"Cluster Concentration vs $\beta$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Top-1 Memory Mass
    ax = axes[0, 1]
    ax.plot(betas, top1_masses, marker="s", color="#d62728", linewidth=2.2, label=r"Top-1 Mass ($\max_i p_i$)")
    ax.axhline(
        1.0 / total_memories,
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.5,
        label=f"All-Memory Uniform ({1.0/total_memories:.3f})",
    )
    ax.axhline(
        1.0 / mem_per_cluster,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.5,
        label=f"Intra-Cluster Uniform ({1.0/mem_per_cluster:.2f})",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"Inverse Temperature $\beta$ (log scale)")
    ax.set_ylabel("Top-1 Memory Mass")
    ax.set_title(r"Single-Memory Sharpness vs $\beta$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Entropy (Memory-Level vs Cluster-Level)
    ax = axes[1, 0]
    ax.plot(betas, mem_entropies, marker="^", color="#ff7f0e", linewidth=2.2, label="Memory Entropy (all memories)")
    ax.plot(betas, clus_entropies, marker="v", color="#9467bd", linewidth=2.2, linestyle="--", label="Cluster Entropy (inter-cluster)")
    ax.set_xscale("log")
    ax.set_xlabel(r"Inverse Temperature $\beta$ (log scale)")
    ax.set_ylabel("Shannon Entropy")
    ax.set_title("Attention Entropy Decomposition")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Panel 4: Distances (Target Cluster Center vs Nearest Stored Memory)
    ax = axes[1, 1]
    ax.plot(
        betas,
        dist_clusters,
        marker="o",
        color="#17becf",
        linewidth=2.2,
        label="Distance to Cluster Mean (Prototype)",
    )
    ax.plot(
        betas,
        dist_memories,
        marker="D",
        color="#8c564b",
        linewidth=2.2,
        label="Distance to Nearest Memory",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"Inverse Temperature $\beta$ (log scale)")
    ax.set_ylabel("Euclidean Distance (L2)")
    ax.set_title(r"State Convergence: Cluster Prototype $\to$ Individual Attractor")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Hopfield cluster / metastable retrieval benchmark."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cluster_retrieval.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save persistent results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override multiple random seeds for aggregation.",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=None,
        help="Override list of beta values to evaluate.",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=None,
        help="Override number of memory clusters.",
    )
    parser.add_argument(
        "--memories-per-cluster",
        type=int,
        default=None,
        help="Override number of memories per cluster.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Override feature dimension.",
    )
    parser.add_argument(
        "--cluster-separation",
        type=float,
        default=None,
        help="Override separation distance between cluster centers.",
    )
    parser.add_argument(
        "--cluster-noise",
        type=float,
        default=None,
        help="Override intra-cluster Gaussian noise standard deviation.",
    )
    parser.add_argument(
        "--query-noise",
        type=float,
        default=None,
        help="Override query Gaussian noise standard deviation.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Override number of evaluation queries.",
    )
    parser.add_argument(
        "--target-cluster",
        type=int,
        default=None,
        help="Override target cluster index (default: 0).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation.",
    )
    args = parser.parse_args()

    # Load base configuration
    config: dict[str, Any]
    if args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {
            "seed": 42,
            "dim": 64,
            "num_clusters": 4,
            "memories_per_cluster": 10,
            "cluster_separation": 3.0,
            "cluster_noise": 0.15,
            "query_noise": 0.10,
            "num_queries": 50,
            "target_cluster": 0,
            "beta_values": DEFAULT_BETAS,
        }

    # Apply CLI overrides
    if args.seed is not None:
        config["seed"] = args.seed
    if args.seeds is not None:
        config["seeds"] = args.seeds
    if args.betas is not None:
        config["beta_values"] = args.betas
    if args.num_clusters is not None:
        config["num_clusters"] = args.num_clusters
    if args.memories_per_cluster is not None:
        config["memories_per_cluster"] = args.memories_per_cluster
    if args.dim is not None:
        config["dim"] = args.dim
    if args.cluster_separation is not None:
        config["cluster_separation"] = args.cluster_separation
    if args.cluster_noise is not None:
        config["cluster_noise"] = args.cluster_noise
    if args.query_noise is not None:
        config["query_noise"] = args.query_noise
    if args.num_queries is not None:
        config["num_queries"] = args.num_queries
    if args.target_cluster is not None:
        config["target_cluster"] = args.target_cluster

    output_dir = args.output_dir
    if output_dir is None:
        configured_dir = str(config.get("reports_dir", "reports/cluster_retrieval"))
        output_dir = Path(configured_dir)

    print(f"Running Hopfield Cluster Retrieval Benchmark from: {args.config}")
    print(
        f"Parameters: dim={config.get('dim', 64)}, "
        f"num_clusters={config.get('num_clusters', 4)}, "
        f"memories_per_cluster={config.get('memories_per_cluster', 10)}, "
        f"cluster_separation={config.get('cluster_separation', 3.0)}, "
        f"cluster_noise={config.get('cluster_noise', 0.15)}, "
        f"query_noise={config.get('query_noise', 0.10)}, "
        f"num_queries={config.get('num_queries', 50)}"
    )
    print(f"Beta values: {config.get('beta_values', DEFAULT_BETAS)}")
    print(f"Reports output directory: {output_dir}\n")

    results, metadata = run_benchmark(
        config=config,
        output_dir=output_dir,
        save_plot=not args.no_plot,
    )

    print("\n--- Cluster Retrieval Benchmark Results Table ---")
    print_cluster_results_table(results, cfg=metadata)
    print(f"Reports saved successfully to: {output_dir}")


if __name__ == "__main__":
    main()
