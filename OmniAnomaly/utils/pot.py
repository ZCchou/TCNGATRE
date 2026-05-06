from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import genpareto


def fit_pot_threshold(
    scores: np.ndarray,
    init_level: float = 0.98,
    risk: float = 1e-3,
    fallback_quantile: float = 0.995,
    min_excess_count: int = 5,
) -> dict[str, float | int | str]:
    """
    Fit a simple POT threshold on normal validation scores.
    Falls back to a high quantile when exceedances are too few or GPD fitting fails.
    """
    x = np.asarray(scores, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size <= 0:
        raise ValueError("Cannot fit POT threshold on empty score array")

    init_level = float(np.clip(init_level, 0.50, 0.999999))
    fallback_quantile = float(np.clip(fallback_quantile, 0.50, 0.999999))
    base_threshold = float(np.quantile(x, init_level))
    excess = x[x > base_threshold] - base_threshold

    if excess.size < int(min_excess_count):
        threshold = float(np.quantile(x, fallback_quantile))
        return {
            "fit_mode": "pot_fallback_quantile",
            "threshold": threshold,
            "base_threshold": base_threshold,
            "fallback_quantile": fallback_quantile,
            "num_samples": int(x.size),
            "num_excess": int(excess.size),
        }

    try:
        shape, _, scale = genpareto.fit(excess, floc=0.0)
        tail_prob = float(max(risk, 1e-9)) / max(float(excess.size) / float(x.size), 1e-9)
        tail_prob = float(min(max(tail_prob, 1e-9), 0.999999))
        q_excess = float(genpareto.ppf(1.0 - tail_prob, c=shape, loc=0.0, scale=scale))
        threshold = float(base_threshold + max(q_excess, 0.0))
        if not np.isfinite(threshold):
            raise FloatingPointError("non-finite POT threshold")
        return {
            "fit_mode": "pot",
            "threshold": threshold,
            "base_threshold": base_threshold,
            "gpd_shape": float(shape),
            "gpd_scale": float(scale),
            "risk": float(risk),
            "init_level": init_level,
            "num_samples": int(x.size),
            "num_excess": int(excess.size),
        }
    except Exception:
        threshold = float(np.quantile(x, fallback_quantile))
        return {
            "fit_mode": "pot_fallback_quantile",
            "threshold": threshold,
            "base_threshold": base_threshold,
            "fallback_quantile": fallback_quantile,
            "num_samples": int(x.size),
            "num_excess": int(excess.size),
        }


def fit_threshold(
    scores: np.ndarray,
    mode: str = "pot",
    manual_threshold: float = 0.0,
    pot_init_level: float = 0.98,
    pot_risk: float = 1e-3,
    pot_fallback_quantile: float = 0.995,
) -> dict[str, float | int | str]:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "manual":
        return {
            "fit_mode": "manual",
            "threshold": float(manual_threshold),
        }
    if mode_norm == "pot":
        return fit_pot_threshold(
            scores=scores,
            init_level=pot_init_level,
            risk=pot_risk,
            fallback_quantile=pot_fallback_quantile,
        )
    raise ValueError(f"Unsupported threshold mode: {mode}")


def save_threshold(payload: dict, path: Path):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_threshold(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
