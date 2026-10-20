"""Public API types and shape validation for HopfieldPC memory core.

This file defines the tensor contract used by the standalone HopfieldMemory
module. Retrieval operations are implemented separately in functional.py.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch


class HopfieldShapeError(ValueError):
    """Raised when query or memory tensors violate the Hopfield shape contract."""


@dataclass(frozen=True)
class HopfieldShapeContract:
    """Resolved tensor shape information for one Hopfield retrieval call."""

    batch_size: int
    dim: int
    memory_size: int


class HopfieldMemoryOutput(NamedTuple):
    """Standard output container for HopfieldMemory.

    This tuple-compatible container supports both unpacking and named access::

        retrieved, weights, diagnostics = module(query, memory)
        output = module(query, memory)
        retrieved = output.retrieved

    Attributes:
        retrieved:
            Retrieved memory vector with shape [batch_size, dim].
        weights:
            Softmax memory weights with shape [batch_size, num_memories].
        diagnostics:
            Dictionary for optional analysis values such as entropy,
            top-k mass, retrieved distance, or Hopfield energy.
    """

    retrieved: torch.Tensor
    weights: torch.Tensor
    diagnostics: dict[str, Any]


def _require_tensor(name: str, value: object) -> torch.Tensor:
    """Validate that a value is a torch.Tensor and return it."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    return value


def validate_beta(beta: object) -> float:
    """Return beta as a finite positive float."""

    if (
        isinstance(beta, bool)
        or not isinstance(beta, numbers.Real)
        or not math.isfinite(beta)
        or beta <= 0
    ):
        raise ValueError(f"beta must be positive and finite, got {beta!r}.")

    return float(beta)


def validate_query_memory(
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    expected_dim: int | None = None,
) -> HopfieldShapeContract:
    """Validate query and memory tensors for Hopfield retrieval.

    Expected shape contract:
        query:  [batch_size, dim]
        memory: [num_memories, dim]

    Args:
        query:
            Query tensor. Usually a hidden state or final representation.
        memory:
            Memory bank tensor. Each row is one stored pattern.
        expected_dim:
            Optional expected embedding/hidden dimension.

    Returns:
        HopfieldShapeContract with resolved batch size, dimension, and memory size.

    Raises:
        TypeError:
            If query or memory is not a floating-point torch.Tensor, or their
            dtypes differ.
        HopfieldShapeError:
            If tensor ranks or dimensions are invalid.
    """

    query = _require_tensor("query", query)
    memory = _require_tensor("memory", memory)

    if not query.is_floating_point() or not memory.is_floating_point():
        raise TypeError(
            "query and memory must be floating-point tensors, "
            f"got {query.dtype} and {memory.dtype}."
        )

    if query.dtype != memory.dtype:
        raise TypeError(
            "query and memory must share a dtype, "
            f"got {query.dtype} and {memory.dtype}."
        )

    if query.ndim != 2:
        raise HopfieldShapeError(
            f"query must have shape [batch_size, dim], got shape {tuple(query.shape)}."
        )

    if memory.ndim != 2:
        raise HopfieldShapeError(
            f"memory must have shape [num_memories, dim], got shape {tuple(memory.shape)}."
        )

    batch_size, query_dim = query.shape
    memory_size, memory_dim = memory.shape

    if batch_size <= 0:
        raise HopfieldShapeError("query batch_size must be greater than 0.")

    if memory_size <= 0:
        raise HopfieldShapeError("memory_size must be greater than 0.")

    if query_dim <= 0:
        raise HopfieldShapeError("query dim must be greater than 0.")

    if memory_dim <= 0:
        raise HopfieldShapeError("memory dim must be greater than 0.")

    if query_dim != memory_dim:
        raise HopfieldShapeError(
            "query and memory must share the same final dimension: "
            f"query dim={query_dim}, memory dim={memory_dim}."
        )

    if expected_dim is not None and query_dim != expected_dim:
        raise HopfieldShapeError(
            f"expected dim={expected_dim}, but query/memory dim={query_dim}."
        )

    return HopfieldShapeContract(
        batch_size=int(batch_size),
        dim=int(query_dim),
        memory_size=int(memory_size),
    )