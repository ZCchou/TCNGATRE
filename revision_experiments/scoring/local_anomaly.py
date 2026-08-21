from __future__ import annotations

import numpy as np


KINDS = ("bias", "drift", "freeze", "noise")


def inject_local_anomaly(
    values: np.ndarray,
    channels: list[int],
    start: int,
    duration: int,
    kind: str,
    scale: np.ndarray,
    severity: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject an auditable local anomaly into a copy of [time, channel] data."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("values must be [time, channel]")
    if kind not in KINDS:
        raise ValueError(f"Unsupported injection kind: {kind}")
    if not channels:
        raise ValueError("channels cannot be empty")
    end = min(int(start) + max(int(duration), 1), x.shape[0])
    if start < 1 or start >= end:
        raise ValueError("injection start must leave a causal reference sample")
    selected = np.asarray(channels, dtype=np.int64)
    result = x.copy()
    robust_scale = np.asarray(scale, dtype=np.float64)[selected]
    robust_scale = np.maximum(robust_scale, 1e-8)
    rng = np.random.default_rng(int(seed))

    if kind == "bias":
        result[start:end, selected] += float(severity) * robust_scale
    elif kind == "drift":
        ramp = np.linspace(0.0, float(severity), end - start, dtype=np.float64)[:, None]
        result[start:end, selected] += ramp * robust_scale[None, :]
    elif kind == "freeze":
        result[start:end, selected] = result[start - 1, selected]
    elif kind == "noise":
        result[start:end, selected] += rng.normal(
            0.0, float(severity), size=(end - start, len(selected))
        ) * robust_scale[None, :]

    labels = np.zeros_like(result, dtype=np.int8)
    labels[start:end, selected] = 1
    return result, labels
