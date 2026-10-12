"""Diagnostics for Hopfield memory retrieval.

These utilities are for analysis/logging. They help answer whether retrieval is
focused on one memory, spread across many memories, or concentrated in a small
top-k set.
"""

from __future__ import annotations

from typing import Any

import torch


def _validate_weights(weights: torch.Tensor) -> None:
    if not isinstance(weights, torch.Tensor):
        raise TypeError(f"weights must be a torch.Tensor, got {type(weights).__name__}.")
    if weights.ndim != 2:
        raise ValueError(
            f"weights must have shape [batch_size, num_memories], got {tuple(weights.shape)}."
        )
    if weights.shape[1] <= 0:
        raise ValueError("weights must contain at least one memory column.")


def attention_entropy(
    weights: torch.Tensor,
    *,
    eps: float = 1e-12,
    normalize: bool = False,
) -> torch.Tensor:
    """Compute entropy of Hopfield memory weights.

    Args:
        weights: Softmax weights with shape [batch_size, num_memories].
        eps: Numerical stability value for log.
        normalize: If True, divide by log(num_memories), so uniform entropy is 1.

    Returns:
        Entropy tensor with shape [batch_size].
    """

    _validate_weights(weights)

    safe_weights = weights.clamp_min(eps)
    entropy = -(weights * safe_weights.log()).sum(dim=-1)

    if normalize:
        num_memories = weights.shape[-1]
        if num_memories == 1:
            return torch.zeros_like(entropy)
        entropy = entropy / torch.log(
            torch.tensor(float(num_memories), device=weights.device, dtype=weights.dtype)
        )

    return entropy


def topk_mass(weights: torch.Tensor, *, k: int = 1) -> torch.Tensor:
    """Compute total probability mass in the top-k memories.

    Args:
        weights: Softmax weights with shape [batch_size, num_memories].
        k: Number of largest weights to sum.

    Returns:
        Tensor with shape [batch_size].
    """

    _validate_weights(weights)

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}.")

    k = min(k, weights.shape[-1])
    return torch.topk(weights, k=k, dim=-1).values.sum(dim=-1)


def max_attention_weight(weights: torch.Tensor) -> torch.Tensor:
    """Return the maximum memory weight per query."""

    _validate_weights(weights)
    return weights.max(dim=-1).values


def argmax_memory_index(weights: torch.Tensor) -> torch.Tensor:
    """Return the index of the strongest memory per query."""

    _validate_weights(weights)
    return weights.argmax(dim=-1)


def retrieval_distance(
    retrieved: torch.Tensor,
    target: torch.Tensor,
    *,
    p: float = 2.0,
) -> torch.Tensor:
    """Compute vector distance between retrieved vectors and target vectors.

    Args:
        retrieved: Tensor with shape [batch_size, dim].
        target: Tensor with shape [batch_size, dim].
        p: Norm order.

    Returns:
        Distance tensor with shape [batch_size].
    """

    if retrieved.shape != target.shape:
        raise ValueError(
            "retrieved and target must have the same shape, got "
            f"{tuple(retrieved.shape)} and {tuple(target.shape)}."
        )

    return torch.linalg.vector_norm(retrieved - target, ord=p, dim=-1)


def selected_memory_distance(
    retrieved: torch.Tensor,
    memory: torch.Tensor,
    weights: torch.Tensor,
    *,
    p: float = 2.0,
) -> torch.Tensor:
    """Distance from retrieved vector to the highest-weight memory vector."""

    _validate_weights(weights)

    if retrieved.ndim != 2:
        raise ValueError(
            f"retrieved must have shape [batch_size, dim], got {tuple(retrieved.shape)}."
        )

    if memory.ndim != 2:
        raise ValueError(
            f"memory must have shape [num_memories, dim], got {tuple(memory.shape)}."
        )

    if retrieved.shape[-1] != memory.shape[-1]:
        raise ValueError(
            "retrieved and memory must share the same feature dimension, got "
            f"{retrieved.shape[-1]} and {memory.shape[-1]}."
        )

    top_indices = argmax_memory_index(weights)
    selected = memory[top_indices]
    return retrieval_distance(retrieved, selected, p=p)


def topk_recall(
    weights: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    k: int,
) -> torch.Tensor:
    """Check whether the target memory index is inside the top-k retrieved memories.

    Args:
        weights: Softmax memory weights with shape [batch_size, num_memories].
        target_indices: Correct memory indices with shape [batch_size].
        k: Number of top memories to check.

    Returns:
        Float tensor with shape [batch_size], where 1.0 means the target
        memory is in the top-k and 0.0 means it is not.
    """

    _validate_weights(weights)

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}.")

    if target_indices.ndim != 1:
        raise ValueError(
            "target_indices must have shape [batch_size], "
            f"got {tuple(target_indices.shape)}."
        )

    if target_indices.shape[0] != weights.shape[0]:
        raise ValueError(
            "target_indices batch size must match weights batch size: "
            f"{target_indices.shape[0]} != {weights.shape[0]}."
        )

    k = min(k, weights.shape[-1])
    _, topk_indices = torch.topk(weights, k, dim=-1)
    expanded_targets = target_indices.unsqueeze(-1).expand_as(topk_indices)
    return (topk_indices == expanded_targets).any(dim=-1).float()


def build_retrieval_diagnostics(
    *,
    query: torch.Tensor,
    memory: torch.Tensor,
    retrieved: torch.Tensor,
    weights: torch.Tensor,
    beta: float,
    num_updates: int,
    normalize: bool,
    top_k: int = 3,
    detach: bool = True,
) -> dict[str, Any]:
    """Build standard Hopfield retrieval diagnostics."""

    entropy = attention_entropy(weights)
    norm_entropy = attention_entropy(weights, normalize=True)
    top1 = topk_mass(weights, k=1)
    topk = topk_mass(weights, k=top_k)
    max_weight = max_attention_weight(weights)
    argmax_index = argmax_memory_index(weights)
    query_distance = retrieval_distance(retrieved, query)
    selected_distance = selected_memory_distance(retrieved, memory, weights)

    diagnostics: dict[str, Any] = {
        "beta": beta,
        "num_updates": num_updates,
        "normalize": normalize,
        "query_shape": tuple(query.shape),
        "memory_shape": tuple(memory.shape),
        "retrieved_shape": tuple(retrieved.shape),
        "weights_shape": tuple(weights.shape),
        "top_k": min(top_k, weights.shape[-1]),
        "entropy": entropy,
        "normalized_entropy": norm_entropy,
        "top1_mass": top1,
        "topk_mass": topk,
        "max_attention_weight": max_weight,
        "argmax_memory_index": argmax_index,
        "query_retrieved_distance": query_distance,
        "selected_memory_distance": selected_distance,
        "entropy_mean": entropy.mean(),
        "top1_mass_mean": top1.mean(),
        "topk_mass_mean": topk.mean(),
        "selected_memory_distance_mean": selected_distance.mean(),
    }

    if detach:
        diagnostics = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in diagnostics.items()
        }

    return diagnostics
