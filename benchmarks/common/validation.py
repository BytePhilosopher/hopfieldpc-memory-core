"""Shared configuration validators for benchmark runners.

Kept separate from the individual experiment runners so that benchmarks can
share validation without importing one another.
"""

from __future__ import annotations

from typing import Any

from hopfieldpc.api import validate_beta


def validate_beta_values(beta_values: Any) -> list[float]:
    """Validate a non-empty list of beta values.

    Each element is checked with the shared scalar validator
    ``hopfieldpc.api.validate_beta`` (finite, positive, non-boolean real).
    """
    if not isinstance(beta_values, list) or len(beta_values) == 0:
        raise ValueError("beta_values must be a non-empty list of positive floats.")

    resolved_betas: list[float] = []
    for b in beta_values:
        try:
            resolved_betas.append(validate_beta(b))
        except ValueError as exc:
            raise ValueError(
                f"All beta_values must be positive finite numbers, got {b!r}."
            ) from exc
    return resolved_betas
