from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


METHOD = "atssd_with_label_point_adjustment"


def atssd(scores: np.ndarray, window_size: int = 96, alpha: float = 0.01) -> np.ndarray:
    """Apply the released ATSSD rule independently to one flight."""
    values = np.asarray(scores, dtype=np.float64)
    prediction = np.zeros(len(values), dtype=np.int8)
    previous_mean = 0.0
    previous_std = 0.0
    percentile = 100.0 * (1.0 - float(alpha))
    for index in range(len(values)):
        start = max(0, index - int(window_size) + 1)
        history = values[start:index + 1]
        current_mean = float(np.mean(history))
        current_std = float(np.std(history, ddof=0))
        base = float(np.percentile(history, percentile))
        if index == 0 or previous_mean == 0.0 or previous_std == 0.0:
            trend = 0.0
        else:
            trend = max(0.0, (current_mean - previous_mean) / previous_mean)
            trend += max(0.0, (current_std - previous_std) / previous_std)
        prediction[index] = int(values[index] > base * (1.0 + trend))
        previous_mean, previous_std = current_mean, current_std
    return prediction


def point_adjust(labels: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Expand a detected point to its complete labeled anomaly segment."""
    labels = np.asarray(labels, dtype=np.int8)
    adjusted = np.asarray(prediction, dtype=np.int8).copy()
    start = None
    for index, value in enumerate(np.r_[labels, 0]):
        if value == 1 and start is None:
            start = index
        elif value == 0 and start is not None:
            if np.any(adjusted[start:index] == 1):
                adjusted[start:index] = 1
            start = None
    return adjusted


def confusion_metrics(labels: np.ndarray, prediction: np.ndarray, scores: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    tp = int(np.sum((labels == 1) & (prediction == 1)))
    fp = int(np.sum((labels == 0) & (prediction == 1)))
    tn = int(np.sum((labels == 0) & (prediction == 0)))
    fn = int(np.sum((labels == 1) & (prediction == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, np.finfo(float).eps)
    result = {
        "num_samples": int(len(labels)),
        "positives": int(labels.sum()),
        "negatives": int(len(labels) - labels.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / max(len(labels), 1)),
        "fpr": float(fp / max(fp + tn, 1)),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    if not all(math.isfinite(float(result[name])) for name in (
        "precision", "recall", "f1", "accuracy", "fpr", "auroc", "average_precision"
    )):
        raise RuntimeError("MSTGCNet native evaluation produced a non-finite metric")
    return result


def evaluate_native_run(
    run_dir: Path,
    *,
    window_size: int = 96,
    alpha: float = 0.01,
) -> dict:
    run_dir = Path(run_dir)
    source = (
        run_dir / "infer_tcngatre_failure" / "score_threshold_analysis"
        / "sequence_scores_with_labels.csv"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Labeled MSTGCNet score file is missing: {source}")
    frame = pd.read_csv(source)
    required = {"flight", "t_start", "label_any", "raw_total_score"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing MSTGCNet native-evaluation columns: {missing}")
    frame = frame.loc[pd.to_numeric(frame["label_any"], errors="coerce").notna()].copy()
    frame["label_any"] = frame["label_any"].astype(np.int8)

    prediction_parts: list[pd.DataFrame] = []
    per_flight_rows: list[dict] = []
    for flight, group in frame.groupby("flight", sort=False):
        current = group.sort_values("t_start", kind="mergesort").copy()
        labels = current["label_any"].to_numpy(dtype=np.int8)
        scores = current["raw_total_score"].to_numpy(dtype=np.float64)
        native = atssd(scores, window_size=window_size, alpha=alpha)
        adjusted = point_adjust(labels, native)
        current["pred_atssd"] = native
        current["pred_atssd_point_adjusted"] = adjusted
        prediction_parts.append(current)
        per_flight_rows.append({
            "flight": str(flight),
            "threshold_method": METHOD,
            "label_col": "label_any",
            **confusion_metrics(labels, adjusted, scores),
        })

    predictions = pd.concat(prediction_parts, ignore_index=True)
    labels = predictions["label_any"].to_numpy(dtype=np.int8)
    adjusted = predictions["pred_atssd_point_adjusted"].to_numpy(dtype=np.int8)
    scores = predictions["raw_total_score"].to_numpy(dtype=np.float64)
    primary = {
        "threshold_method": METHOD,
        "label_col": "label_any",
        "aggregation": "micro_over_all_scored_windows",
        "score_source": "raw_total_score",
        **confusion_metrics(labels, adjusted, scores),
    }

    output = run_dir / "native_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        output / "sequence_predictions.csv.gz", index=False, compression="gzip",
        encoding="utf-8",
    )
    pd.DataFrame(per_flight_rows).to_csv(
        output / "per_flight_metrics.csv", index=False, encoding="utf-8-sig"
    )
    (output / "primary_metrics.json").write_text(
        json.dumps(primary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config = {
        "method": METHOD,
        "atssd_window_size": int(window_size),
        "atssd_alpha": float(alpha),
        "flightwise_independent_thresholding": True,
        "point_adjustment": "complete labeled segment after any within-segment detection",
        "failure_labels_used_for_point_adjustment": True,
        "failure_labels_used_for_training": False,
        "failure_labels_used_for_parameter_selection": False,
        "threshold_metrics": "micro confusion counts over all scored windows",
        "ranking_metrics": "AUROC and AP from raw continuous anomaly scores",
        "source_scores": str(source.resolve()),
    }
    (output / "evaluation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"primary_metrics": primary, "evaluation_config": config, "output_dir": str(output)}


__all__ = ["METHOD", "atssd", "point_adjust", "confusion_metrics", "evaluate_native_run"]
