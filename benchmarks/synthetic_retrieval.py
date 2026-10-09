"""Synthetic nearest-memory retrieval benchmark.

Checks whether Hopfield memory retrieves the correct stored vector from a noisy query.
"""

import argparse
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional as F

from hopfieldpc.functional import hopfield_retrieve


def generate_memory(num_memories: int, dim: int) -> torch.Tensor:
    """Generate a random memory bank of shape [num_memories, dim]."""
    std = 1.0 / math.sqrt(dim)
    memory = torch.empty(num_memories, dim)
    torch.nn.init.normal_(memory, mean=0.0, std=std)
    return memory


def generate_noisy_queries(
    memory: torch.Tensor, noise_level: float, num_trials: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate queries by adding noise to randomly selected target memories.
    
    Returns:
        queries: [num_trials, dim]
        target_indices: [num_trials]
    """
    num_memories, dim = memory.shape
    target_indices = torch.randint(0, num_memories, (num_trials,))
    targets = memory[target_indices]
    
    noise = torch.randn_like(targets) * noise_level
    queries = targets + noise
    
    return queries, target_indices


def compute_entropy(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute softmax entropy per query: -sum(p * log(p))."""
    return -(weights * torch.log(weights + eps)).sum(dim=-1)


def compute_topk_mass(weights: torch.Tensor, k: int) -> torch.Tensor:
    """Compute sum of top-k weights per query."""
    # Ensure we don't ask for more elements than exist
    k = min(k, weights.shape[-1])
    topk_weights, _ = torch.topk(weights, k, dim=-1)
    return topk_weights.sum(dim=-1)


def compute_distance(retrieved: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute L2 distance between retrieved vectors and target memory vectors."""
    return torch.norm(retrieved - targets, p=2, dim=-1)


def evaluate_condition(
    beta: float,
    noise_level: float,
    memory_size: int,
    dim: int,
    num_trials: int,
    top_k: int,
    normalize: bool,
) -> dict[str, float]:
    """Run one evaluation condition and return average metrics."""
    
    memory = generate_memory(memory_size, dim)
    queries, target_indices = generate_noisy_queries(memory, noise_level, num_trials)
    
    retrieved, weights = hopfield_retrieve(queries, memory, beta=beta, normalize=normalize)
    
    targets = memory[target_indices]
    
    # 1. Top-1 Accuracy
    predicted_indices = weights.argmax(dim=-1)
    top1_acc = (predicted_indices == target_indices).float().mean().item()
    
    # 2. Distance to Target
    distance = compute_distance(retrieved, targets).mean().item()
    
    # 3. Entropy
    entropy = compute_entropy(weights).mean().item()
    
    # 4. Top-k Mass
    topk_mass = compute_topk_mass(weights, top_k).mean().item()
    
    return {
        "beta": beta,
        "noise": noise_level,
        "memory_size": memory_size,
        "top1_acc": top1_acc,
        "distance": distance,
        "entropy": entropy,
        "topk_mass": topk_mass,
    }


def print_results_table(results: list[dict[str, float]]) -> None:
    """Print results in a formatted table."""
    header = (
        f"{'beta':>8} | {'noise':>8} | {'mem_size':>8} | "
        f"{'top1_acc':>8} | {'distance':>8} | {'entropy':>8} | {'topk_mass':>8}"
    )
    separator = "-" * len(header)
    
    print(header)
    print(separator)
    
    for row in results:
        print(
            f"{row['beta']:>8.2f} | {row['noise']:>8.2f} | {row['memory_size']:>8d} | "
            f"{row['top1_acc']:>8.4f} | {row['distance']:>8.4f} | "
            f"{row['entropy']:>8.4f} | {row['topk_mass']:>8.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic nearest-memory retrieval benchmark.")
    parser.add_argument(
        "--config", 
        type=Path, 
        default=Path("configs/synthetic_retrieval.yaml"),
        help="Path to YAML config file."
    )
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"Error: Config file not found at {args.config}")
        return
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Set random seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    
    dim = config["dim"]
    num_trials = config["num_trials"]
    top_k = config.get("top_k", 3)
    normalize = config.get("normalize", False)
    
    beta_values = config["beta_values"]
    noise_levels = config["noise_levels"]
    memory_sizes = config["memory_sizes"]
    
    results = []
    
    print(f"Running benchmark with {num_trials} trials per condition...")
    print(f"Parameters: dim={dim}, top_k={top_k}, normalize={normalize}")
    print()
    
    for beta in beta_values:
        for noise in noise_levels:
            for memory_size in memory_sizes:
                metrics = evaluate_condition(
                    beta=beta,
                    noise_level=noise,
                    memory_size=memory_size,
                    dim=dim,
                    num_trials=num_trials,
                    top_k=top_k,
                    normalize=normalize,
                )
                results.append(metrics)
                
    print_results_table(results)


if __name__ == "__main__":
    main()
