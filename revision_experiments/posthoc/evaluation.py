from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from revision_experiments.core.evaluation import evaluate_scores
from revision_experiments.scoring.aggregators import aggregate_dataframe

from .constants import EMA_ALPHA
from .inference import LoadedSourceModel, primary_proxy


def _parse_vector(value: Any) -> list[float]:
    if not isinstance(value, str):
        return np.asarray(value, dtype=np.float64).reshape(-1).astype(float).tolist()
    try:
        return np.asarray(json.loads(value), dtype=np.float64).reshape(-1).astype(float).tolist()
    except json.JSONDecodeError:
        parsed = np.fromstring(value.strip().strip("[]").replace(",", " "), sep=" ")
        if parsed.size == 0 or not np.isfinite(parsed).all():
            raise ValueError(f"Cannot parse residual vector: {value[:80]!r}")
        return parsed.astype(float).tolist()


def normalize_vector_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize native JSON and older NumPy-style vectors without changing sources."""
    output = frame.copy()
    for column in ("sensor_score_vec", "value_residual_vec", "delta_residual_vec"):
        if column in output:
            output[column] = [json.dumps(_parse_vector(value)) for value in output[column]]
    return output


def write_and_evaluate_real(
    run_dir: Path,
    model: LoadedSourceModel,
    validation_residuals: pd.DataFrame,
    failure_residuals: pd.DataFrame,
    aggregator: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    run_dir = Path(run_dir)
    infer = run_dir / "infer_tcngatre_failure"
    infer.mkdir(parents=True, exist_ok=True)
    validation = aggregate_dataframe(normalize_vector_columns(validation_residuals), aggregator, EMA_ALPHA)
    failure = aggregate_dataframe(normalize_vector_columns(failure_residuals), aggregator, EMA_ALPHA)
    validation[["flight", "t_start", "t_end", "t_mid", "raw_total_score", "total_score"]].to_csv(
        run_dir / "val_normal_scores.csv", index=False, encoding="utf-8"
    )
    failure.to_csv(
        infer / "all_failure_window_forecast_residual.csv", index=False, encoding="utf-8-sig"
    )
    failure[[
        "flight", "current_index", "t_start", "t_end", "t_mid",
        "raw_total_score", "total_score", "valid_dim_count", "aggregation_method",
    ]].to_csv(infer / "sequence_scores.csv", index=False, encoding="utf-8")
    primary = evaluate_scores(run_dir, primary_proxy(), model.cfg)
    scored = pd.read_csv(
        infer / "score_threshold_analysis" / "sequence_scores_with_labels.csv"
    )
    return primary, scored


def detection_delay_rows(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for flight, group in scored.groupby("flight", sort=False):
        current = group.sort_values("t_start", kind="mergesort")
        anomaly = current.loc[pd.to_numeric(current["label_any"], errors="coerce") > 0]
        if anomaly.empty:
            continue
        onset = float(anomaly["t_start"].min())
        end = float(anomaly["t_end"].max())
        detected = current.loc[
            (pd.to_numeric(current["pred_spot"], errors="coerce") > 0)
            & (pd.to_numeric(current["t_start"], errors="coerce") >= onset)
        ]
        first = float(detected["t_start"].min()) if not detected.empty else float("nan")
        rows.append({
            "flight": str(flight),
            "anomaly_onset": onset,
            "anomaly_end": end,
            "detected": bool(not detected.empty),
            "detection_delay": max(first - onset, 0.0) if np.isfinite(first) else float("nan"),
        })
    return pd.DataFrame(rows)


def delay_summary(scored: pd.DataFrame) -> dict[str, Any]:
    frame = detection_delay_rows(scored)
    if frame.empty:
        return {
            "flights": 0, "detected_flights": 0, "miss_rate": 0.0,
            "mean_delay": 0.0, "median_delay": 0.0,
            "mean_detected_only_delay": None,
        }
    detected = frame.loc[frame["detected"]]
    capped = frame["detection_delay"].where(
        frame["detected"],
        (frame["anomaly_end"] - frame["anomaly_onset"]).clip(lower=0.0),
    )
    return {
        "flights": int(len(frame)),
        "detected_flights": int(len(detected)),
        "miss_rate": float(1.0 - len(detected) / len(frame)),
        "mean_delay": float(capped.mean()),
        "median_delay": float(capped.median()),
        "mean_detected_only_delay": (
            float(detected["detection_delay"].mean()) if len(detected) else None
        ),
    }
