"""Modern Hopfield energy with explicit working-precision handling.

E(q) = 0.5 * ||q||^2 - logsumexp(beta * q @ memory.T) / beta + C

C = log(N) / beta + 0.5 * max_i ||memory_i||^2 is optional and is
independent of the query when memory and beta are fixed.
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
    """Compute the state-dependent Hopfield energy and optional offsets.

    Both modes include the quadratic query term. For fixed memory, fixed
    positive beta, and unnormalized auto-associative retrieval, each query's
    energy gradient is q - retrieved(q). The retrieval update does not increase
    this energy in exact arithmetic; numerical comparisons need tolerances.

    Args:
        query: Floating-point tensor with shape [batch_size, dim].
        memory: Tensor with shape [num_memories, dim], matching query dtype
            and device.
        beta: Finite positive inverse temperature.
        include_constants: Add log(N) / beta and half the largest squared
            memory norm. These offsets do not change query gradients or
            energy differences between states for fixed memory and beta.
        reduce: "none" returns one value per query; "mean" returns the
            batch mean. The gradient of the mean has a 1 / batch_size factor.

    Returns:
        Energy with the query's dtype and device. Its shape is [batch_size]
        for "none" and scalar for "mean".

    Numerical behavior:
        Float16 and bfloat16 inputs use float32 working precision for all
        calculations, including squared norms and batch reduction. Float32
        and float64 inputs retain their precision. Autocast is disabled
        locally so it cannot lower the selected working precision.

        Conversion back to the input dtype happens only after reduction.
        This avoids unnecessary intermediate overflow, but does not guarantee
        finiteness for arbitrary input magnitudes. Results and gradients still
        need to fit their destination dtypes; no values are clipped.
    """

    beta = validate_beta(beta)

    if reduce not in ("none", "mean"):
        raise ValueError(f"reduce must be 'none' or 'mean', got {reduce!r}.")

    validate_query_memory(query=query, memory=memory)

    input_dtype = query.dtype
    working_dtype = (
        torch.float32
        if input_dtype in (torch.float16, torch.bfloat16)
        else input_dtype
    )

    # Keep ambient mixed-precision contexts from lowering matmul precision.
    with torch.autocast(device_type=query.device.type, enabled=False):
        query_work = query.to(dtype=working_dtype)
        memory_work = memory.to(dtype=working_dtype)

        similarities = torch.matmul(query_work, memory_work.transpose(0, 1))
        max_similarity = similarities.amax(dim=-1, keepdim=True)

        # For finite similarities, centering avoids scaling positive scores
        # by beta. The matmul and squared norms still have finite dtype limits.
        log_sum = torch.logsumexp(
            beta * (similarities - max_similarity), dim=-1
        )
        energy = (
            -max_similarity.squeeze(-1)
            - log_sum / beta
            + 0.5 * query_work.square().sum(dim=-1)
        )

        if include_constants:
            log_n_term = math.log(memory_work.shape[0]) / beta
            memory_bound_term = 0.5 * memory_work.square().sum(dim=-1).max()
            energy = energy + log_n_term + memory_bound_term

        # Reduce before casting: individual energies can exceed the output
        # dtype's range even when their mean is representable.
        if reduce == "mean":
            energy = energy.mean()

    return energy.to(dtype=input_dtype)