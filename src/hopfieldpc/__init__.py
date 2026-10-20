"""HopfieldPC Memory Core.

Standalone modern Hopfield memory module for validating associative retrieval
before FabricPC integration.
"""

from .api import HopfieldMemoryOutput, HopfieldShapeContract, HopfieldShapeError
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
from .energy import hopfield_energy, hopfield_full_energy
from .functional import hopfield_retrieve, multi_step_retrieve
from .memory import HopfieldMemory

__all__ = [
    "HopfieldMemory",
    "HopfieldMemoryOutput",
    "HopfieldShapeContract",
    "HopfieldShapeError",
    "argmax_memory_index",
    "attention_entropy",
    "build_retrieval_diagnostics",
    "hopfield_energy",
    "hopfield_full_energy",
    "hopfield_retrieve",
    "max_attention_weight",
    "multi_step_retrieve",
    "retrieval_distance",
    "selected_memory_distance",
    "topk_mass",
    "topk_recall",
]