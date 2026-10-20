"""Functional Hopfield retrieval operations.

Implements the pure one-step modern Hopfield retrieval equation:

    weights = softmax(beta * query @ memory.T)
    retrieved = weights @ memory

and optional multi-step retrieval, which repeats the one-step update:

    q_{t+1} = HopfieldRetrieval(q_t, M)

The function in this file is intentionally independent from nn.Module so it can
be tested directly and reused by HopfieldMemory.
"""

from __future__ import annotations

from typing import Literal, overload

import torch
import torch.nn.functional as F

from .api import validate_beta, validate_query_memory


def _normalize_for_scores(
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize query and memory only for similarity scoring.

    The retrieved vector is still computed from the original memory vectors.
    This keeps normalization from changing the memory values themselves.
    """

    query_norm = F.normalize(query, p=2, dim=-1, eps=eps)
    memory_norm = F.normalize(memory, p=2, dim=-1, eps=eps)
    return query_norm, memory_norm


def hopfield_retrieve(
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    beta: float = 1.0,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one-step modern Hopfield retrieval.

    Args:
        query:
            Query tensor with shape [batch_size, dim].
        memory:
            Memory bank tensor with shape [num_memories, dim].
            Each row is one stored pattern.
        beta:
            Finite positive retrieval sharpness. Larger beta produces sharper
            memory selection.
        normalize:
            If True, query and memory are L2-normalized for similarity scoring.
            The retrieved vector is still computed from the original memory.

    Returns:
        retrieved:
            Retrieved memory vectors with shape [batch_size, dim].
        weights:
            Softmax memory weights with shape [batch_size, num_memories].

    Formula:
        scores = beta * query @ memory.T
        weights = softmax(scores)
        retrieved = weights @ memory
    """

    beta = validate_beta(beta)

    validate_query_memory(query=query, memory=memory)

    input_dtype = query.dtype
    working_dtype = (
        torch.float32
        if input_dtype in (torch.float16, torch.bfloat16)
        else input_dtype
    )
    query_work = query.to(dtype=working_dtype)
    memory_work = memory.to(dtype=working_dtype)

    if normalize:
        query_for_scores, memory_for_scores = _normalize_for_scores(
            query_work, memory_work
        )
    else:
        query_for_scores, memory_for_scores = query_work, memory_work

    similarities = torch.matmul(query_for_scores, memory_for_scores.transpose(0, 1))
    # Center before beta scaling to prevent positive overflow without changing softmax.
    similarities = similarities - similarities.amax(dim=-1, keepdim=True)
    scores = beta * similarities
    weights_work = torch.softmax(scores, dim=-1)
    retrieved_work = torch.matmul(weights_work, memory_work)

    return (
        retrieved_work.to(dtype=input_dtype),
        weights_work.to(dtype=input_dtype),
    )


@overload
def multi_step_retrieve(
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    beta: float = ...,
    num_updates: int = ...,
    normalize: bool = ...,
    return_states: Literal[False] = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def multi_step_retrieve(
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    beta: float = ...,
    num_updates: int = ...,
    normalize: bool = ...,
    return_states: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]: ...


def multi_step_retrieve(
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    beta: float = 1.0,
    num_updates: int = 1,
    normalize: bool = False,
    return_states: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
):
    """Run repeated Hopfield retrieval: q0 -> q1 -> ... -> qT.

    Args:
        query:
            Initial query q0 with shape [batch_size, dim].
        memory:
            Memory bank with shape [num_memories, dim].
        beta:
            Retrieval sharpness (positive).
        num_updates:
            Number of retrieval steps T (positive integer). Default 1
            reproduces plain one-step retrieval.
        normalize:
            Passed to every one-step retrieval (scoring-only normalization).
        return_states:
            If True, also return the list [q0, q1, ..., qT] for debugging.

    Returns:
        retrieved:
            Final state qT with shape [batch_size, dim].
        weights:
            Softmax weights from the last step, shape [batch_size, num_memories].
        states:
            Only if return_states=True: list of T + 1 tensors, q0 first.
    """

    if isinstance(num_updates, bool) or not isinstance(num_updates, int) or num_updates <= 0:
        raise ValueError(f"num_updates must be a positive integer, got {num_updates!r}.")

    state = query
    states: list[torch.Tensor] | None = [query] if return_states else None
    weights: torch.Tensor | None = None

    for _ in range(num_updates):
        state, weights = hopfield_retrieve(
            query=state,
            memory=memory,
            beta=beta,
            normalize=normalize,
        )
        if states is not None:
            states.append(state)

    # num_updates > 0 is validated above, so the loop ran at least once.
    assert weights is not None

    if states is not None:
        return state, weights, states
    return state, weights
