"""Hopfield energy computation for logging and analysis.

Implements the modern continuous Hopfield energy (Ramsauer et al., 2020):

    E(q, M, β) = -(1/β) · logsumexp(β · q Mᵀ) + ½ ‖q‖² + ½ M² + (1/β) · log(N)

where M = max_i ‖mᵢ‖ is the largest memory norm.
"""

from __future__ import annotations

import math

import torch

from .api import validate_beta, validate_query_memory


def hopfield_energy(
    query: torch.Tensor,
    memory: torch.Tensor,
    beta: float = 1.0,
    *,
    include_constants: bool = False,
    reduce: str = "none",
) -> torch.Tensor:
    """Compute modern Hopfield energy for each query.

    Args:
        query: Query tensor with shape [batch_size, dim].
        memory: Memory bank tensor with shape [num_memories, dim].
        beta: Retrieval sharpness (inverse temperature). Must be positive
            and finite.
        include_constants: If True, add the terms that complete the full energy of
            Ramsauer et al. (2020): ½‖q‖², ½M² and (1/β)·log(N). Only the last two
            are constants. ½‖q‖² depends on q, so with the default False the value
            is NOT the energy that retrieval descends and it can rise during
            retrieval. Use include_constants=True whenever energy descent or
            gradients matter. For unnormalized retrieval (normalize=False) with a
            fixed memory and finite positive beta, dE/dq = q - retrieved(q), so one
            retrieval update is a unit gradient step on this energy and cannot
            increase it. With normalize=True this guarantee does not apply.
        reduce: ``"none"`` returns per-query energy [batch_size],
                ``"mean"`` returns the scalar batch mean.

    Returns:
        Hopfield energy tensor.
    """

    beta = validate_beta(beta)

    if reduce not in ("none", "mean"):
        raise ValueError(f"reduce must be 'none' or 'mean', got {reduce!r}.")

    validate_query_memory(query=query, memory=memory)

    similarities = torch.matmul(query, memory.transpose(0, 1))
    # Same as -logsumexp(beta * similarities) / beta, but beta never scales the
    # uncentered similarities, so large scores cannot overflow.
    max_similarity = similarities.amax(dim=-1, keepdim=True)
    log_sum = torch.logsumexp(beta * (similarities - max_similarity), dim=-1)
    energy = -max_similarity.squeeze(-1) - log_sum / beta

    if include_constants:
        num_memories = memory.shape[0]
        query_norm_sq = 0.5 * (query ** 2).sum(dim=-1)
        log_n_term = (1.0 / beta) * math.log(num_memories)
        max_memory_norm_sq = (memory ** 2).sum(dim=-1).max()
        memory_bound_term = 0.5 * max_memory_norm_sq
        energy = energy + query_norm_sq + log_n_term + memory_bound_term

    if reduce == "mean":
        return energy.mean()

    return energy
