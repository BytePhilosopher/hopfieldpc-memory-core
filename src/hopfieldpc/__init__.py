"""HopfieldPC Memory Core.

Standalone modern Hopfield memory module for validating associative retrieval
before FabricPC integration.
"""

from .api import HopfieldMemoryOutput, HopfieldShapeContract, HopfieldShapeError
from .energy import hopfield_energy
from .functional import hopfield_retrieve, multi_step_retrieve
from .memory import HopfieldMemory
from .diagnostics import (
    argmax_memory_index,
    attention_entropy,
    build_retrieval_diagnostics,
    max_attention_weight,
    retrieval_distance,
    selected_memory_distance,
    topk_mass,
    topk_recall,
)

__all__ = [
    "HopfieldMemory",
    "HopfieldMemoryOutput",
    "HopfieldShapeContract",
    "HopfieldShapeError",
    "hopfield_energy",
    "hopfield_retrieve",
    "multi_step_retrieve",
    "argmax_memory_index",
    "attention_entropy",
    "build_retrieval_diagnostics",
    "max_attention_weight",
    "retrieval_distance",
    "selected_memory_distance",
    "topk_mass",
    "topk_recall",
]