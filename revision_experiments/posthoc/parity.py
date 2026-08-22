from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SCORE_RTOL = 1e-6


@dataclass(frozen=True)
class ParityResult:
    passed: bool
    max_abs_error: float
    max_rel_error: float


def compare_scores(
    source: np.ndarray,
    rebuilt: np.ndarray,
    *,
    atol: float,
    rtol: float = SCORE_RTOL,
) -> ParityResult:
    """Compare serialized scores with a scale-aware absolute/relative tolerance."""
    lhs = np.asarray(source, dtype=np.float64)
    rhs = np.asarray(rebuilt, dtype=np.float64)
    if lhs.shape != rhs.shape:
        raise ValueError(f"Score shapes differ: {lhs.shape} != {rhs.shape}")
    if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        raise ValueError("Scores contain NaN or Inf")
    absolute = np.abs(lhs - rhs)
    scale = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), np.finfo(np.float64).tiny)
    relative = absolute / scale
    tolerance = float(atol) + float(rtol) * scale
    return ParityResult(
        passed=bool(np.all(absolute <= tolerance)),
        max_abs_error=float(np.max(absolute, initial=0.0)),
        max_rel_error=float(np.max(relative, initial=0.0)),
    )
