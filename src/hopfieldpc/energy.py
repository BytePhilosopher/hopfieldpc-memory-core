"""Hopfield energy computation for logging and analysis.

Implements the modern continuous Hopfield energy (Ramsauer et al., 2020):

    E(q, M, β) = -(1/β) · logsumexp(β · q Mᵀ) + ½ ‖q‖² + ½ M² + (1/β) · log(N)

where M = max_i ‖mᵢ‖ is the largest memory norm.

hopfield_energy always includes -(1/β)·logsumexp(β q Mᵀ) and ½‖q‖², the two
terms that depend on q. Its gradient with respect to q is q - retrieved(q).
include_constants=True adds only the two q-independent terms, ½ M² and
(1/β)·log(N), which shift the value but not the gradient.
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

    The energy is -(1/β)·logsumexp(β q Mᵀ) + ½‖q‖². Its gradient with respect
    to q is q - retrieved(q), so one retrieval step is a unit gradient step and
    cannot raise it. This holds for a fixed memory, a fixed beta and
    unnormalized retrieval (normalize=False), with or without the optional
    constants.

    Changed: earlier versions left ½‖q‖² out unless include_constants=True.
    That term depends on q, so the old default could rise during correct
    retrieval. The default value is now ½‖q‖² higher than before.

    Args:
        query: Query tensor with shape [batch_size, dim].
        memory: Memory bank tensor with shape [num_memories, dim].
        beta: Retrieval sharpness (inverse temperature). Must be positive
            and finite.
        include_constants: If True, also add the two terms that do not depend on
            q: (1/β)·log(N) and ½·max_i‖mᵢ‖². They shift the value but not its
            gradient, so they never affect convergence checks.
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
    energy = (
        -max_similarity.squeeze(-1)
        - log_sum / beta
        + 0.5 * (query ** 2).sum(dim=-1)
    )

    if include_constants:
        log_n_term = (1.0 / beta) * math.log(memory.shape[0])
        memory_bound_term = 0.5 * (memory ** 2).sum(dim=-1).max()
        energy = energy + log_n_term + memory_bound_term

    if reduce == "mean":
        return energy.mean()

    return energy
