from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_ranking_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int32)
    score = np.asarray(score, dtype=np.float64)
    positives = int(y_true.sum())
    negatives = int((1 - y_true).sum())
    out = {
        "num_samples": int(len(y_true)),
        "positives": positives,
        "negatives": negatives,
        "auroc": float("nan"),
        "average_precision": float("nan"),
    }
    if len(y_true) <= 0:
        return out
    if positives > 0:
        out["average_precision"] = float(average_precision_score(y_true, score))
    if positives > 0 and negatives > 0:
        out["auroc"] = float(roc_auc_score(y_true, score))
    return out


def compute_binary_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int32)
    pred = np.asarray(pred, dtype=np.int32)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    accuracy = float((tp + tn) / max(tp + tn + fp + fn, 1))
    tpr = recall
    fpr = float(fp / max(fp + tn, 1))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tpr": tpr,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def point_adjust_predictions(y_true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Classic segment-level point adjustment on binary predictions."""
    y_true = np.asarray(y_true, dtype=np.int32)
    pred = np.asarray(pred, dtype=np.int32).copy()
    if len(y_true) != len(pred):
        raise ValueError("y_true and pred length mismatch")
    in_run = False
    start = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_run:
            start = i
            in_run = True
        if in_run and (i == len(y_true) - 1 or y_true[i + 1] == 0):
            end = i + 1
            if bool((pred[start:end] == 1).any()):
                pred[start:end] = 1
            in_run = False
    return pred


def summarize_threshold_metrics(
    scored_df: pd.DataFrame,
    score_col: str,
    label_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for label_col in label_cols:
        if label_col not in scored_df.columns:
            continue
        valid = pd.to_numeric(scored_df[label_col], errors="coerce").notna().to_numpy()
        if not bool(valid.any()):
            continue
        y = scored_df.loc[valid, label_col].to_numpy(dtype=np.int32)
        s = scored_df.loc[valid, score_col].to_numpy(dtype=np.float64)
        p = scored_df.loc[valid, "pred_label"].to_numpy(dtype=np.int32)
        row = {"score_col": score_col, "label_col": label_col}
        row.update(compute_ranking_metrics(y_true=y, score=s))
        row.update(compute_binary_metrics(y_true=y, pred=p))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_per_flight(
    scored_df: pd.DataFrame,
    score_col: str,
    label_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for flight, g in scored_df.groupby("flight", sort=True):
        for label_col in label_cols:
            if label_col not in g.columns:
                continue
            valid = pd.to_numeric(g[label_col], errors="coerce").notna().to_numpy()
            if not bool(valid.any()):
                continue
            y = g.loc[valid, label_col].to_numpy(dtype=np.int32)
            s = g.loc[valid, score_col].to_numpy(dtype=np.float64)
            p = g.loc[valid, "pred_label"].to_numpy(dtype=np.int32)
            row = {"flight": str(flight), "label_col": label_col}
            row.update(compute_ranking_metrics(y_true=y, score=s))
            row.update(compute_binary_metrics(y_true=y, pred=p))
            rows.append(row)
    return pd.DataFrame(rows)
