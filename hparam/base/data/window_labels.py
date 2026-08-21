from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TIME_COLUMN_CANDIDATES = (
    "t", "time", "timestamp", "time_sec", "time_s", "seconds",
    "sec", "elapsed_sec", "relative_time", "relative_time_sec",
)
LABEL_COLUMN_CANDIDATES = (
    "anomaly_label", "label", "fault_flag", "is_anomaly", "anomaly",
    "target", "y", "attack", "failure", "is_failure",
)
START_COLUMN_CANDIDATES = (
    "start_sec", "t_start", "start", "start_time", "start_time_sec",
    "begin", "begin_sec", "onset", "onset_sec",
)
END_COLUMN_CANDIDATES = (
    "end_sec", "t_end", "end", "end_time", "end_time_sec",
    "stop", "stop_sec", "offset", "offset_sec",
)
def _norm_col(name: object) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lookup = {_norm_col(col): str(col) for col in df.columns}
    for cand in candidates:
        col = lookup.get(_norm_col(cand))
        if col is not None:
            return col
    return None


def _coerce_label(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        text = series.astype(str).str.strip().str.lower()
        positive = text.isin({"true", "yes", "y", "anomaly", "fault", "failure", "attack", "abnormal"})
        negative = text.isin({"false", "no", "n", "normal", "benign", "none", "ok"})
        numeric = numeric.mask(numeric.isna() & positive, 1.0)
        numeric = numeric.mask(numeric.isna() & negative, 0.0)
        return numeric.fillna(0.0)
    text = series.astype(str).str.strip().str.lower()
    positive = text.isin({"1", "true", "yes", "y", "anomaly", "fault", "failure", "attack", "abnormal"})
    return positive.astype(float)


def _from_point_labels(df: pd.DataFrame) -> pd.DataFrame | None:
    time_col = _find_column(df, TIME_COLUMN_CANDIDATES)
    label_col = _find_column(df, LABEL_COLUMN_CANDIDATES)
    if time_col is None or label_col is None:
        return None
    out = pd.DataFrame({
        "t": pd.to_numeric(df[time_col], errors="coerce"),
        "anomaly_label": _coerce_label(df[label_col]),
    }).dropna(subset=["t"])
    if out.empty:
        return None
    return out


def _from_interval_labels(df: pd.DataFrame) -> pd.DataFrame | None:
    start_col = _find_column(df, START_COLUMN_CANDIDATES)
    end_col = _find_column(df, END_COLUMN_CANDIDATES)
    if start_col is None or end_col is None:
        return None
    label_col = _find_column(df, LABEL_COLUMN_CANDIDATES)
    starts = pd.to_numeric(df[start_col], errors="coerce")
    ends = pd.to_numeric(df[end_col], errors="coerce")
    labels = _coerce_label(df[label_col]) if label_col is not None else pd.Series(np.ones(len(df)), index=df.index)
    rows: list[dict[str, float]] = []
    for start, end, label in zip(starts, ends, labels):
        if not np.isfinite(start) or not np.isfinite(end):
            continue
        s = float(min(start, end))
        e = float(max(start, end))
        v = float(label)
        rows.append({"t": np.nextafter(s, -np.inf), "anomaly_label": 0.0})
        rows.append({"t": s, "anomaly_label": v})
        rows.append({"t": e, "anomaly_label": v})
        rows.append({"t": np.nextafter(e, np.inf), "anomaly_label": 0.0})
    if not rows:
        return None
    return pd.DataFrame(rows)


def load_failure_labels(
    labels_root: Path,
    flight: str,
    time_offset_sec: float = 0.0,
) -> pd.DataFrame | None:
    path = Path(labels_root) / f"{flight}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    out = _from_point_labels(df)
    if out is None:
        out = _from_interval_labels(df)
    if out is None or out.empty:
        return None
    out["t"] = pd.to_numeric(out["t"], errors="coerce")
    out["anomaly_label"] = pd.to_numeric(out["anomaly_label"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["t"])
    if out.empty:
        return None
    if float(time_offset_sec) != 0.0:
        out["t"] = out["t"] + float(time_offset_sec)
    return out.sort_values("t", kind="mergesort").reset_index(drop=True)


def label_mid(labels_df: pd.DataFrame, t_mid: float) -> int:
    t = labels_df["t"].to_numpy(dtype=float)
    if len(t) <= 0 or float(t_mid) < float(t[0]):
        return 0
    idx = int(np.searchsorted(t, float(t_mid), side="right") - 1)
    idx = min(max(idx, 0), len(t) - 1)
    return int(float(labels_df.iloc[idx]["anomaly_label"]) > 0.5)


def label_any(labels_df: pd.DataFrame, t_start: float, t_end: float) -> int:
    t0 = float(min(t_start, t_end))
    t1 = float(max(t_start, t_end))
    t = labels_df["t"].to_numpy(dtype=float)
    y = labels_df["anomaly_label"].to_numpy(dtype=float)
    mask = (t >= t0) & (t <= t1)
    if bool(mask.any()) and bool((y[mask] > 0.5).any()):
        return 1
    return label_mid(labels_df, 0.5 * (t0 + t1))


def attach_window_labels(
    scored_df: pd.DataFrame,
    labels_root: Path,
    time_offset_sec: float = 0.0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for flight, g in scored_df.groupby("flight", sort=True):
        labels_df = load_failure_labels(
            labels_root=labels_root,
            flight=str(flight),
            time_offset_sec=time_offset_sec,
        )
        for row in g.to_dict(orient="records"):
            payload = dict(row)
            if labels_df is None:
                payload["label_mid"] = np.nan
                payload["label_any"] = np.nan
            else:
                payload["label_mid"] = label_mid(labels_df, float(payload["t_mid"]))
                payload["label_any"] = label_any(
                    labels_df,
                    float(payload["t_start"]),
                    float(payload["t_end"]),
                )
            rows.append(payload)
    return pd.DataFrame(rows)


def add_anomaly_background(ax, labels_df: pd.DataFrame | None, t_min: float, t_max: float):
    if labels_df is None or labels_df.empty:
        return
    t = labels_df["t"].to_numpy(dtype=float)
    y = (labels_df["anomaly_label"].to_numpy(dtype=float) > 0.5).astype(np.int32)
    in_run = False
    start = 0.0
    for i in range(len(t)):
        if y[i] == 1 and not in_run:
            start = float(t[i])
            in_run = True
        if in_run and (i == len(t) - 1 or y[i + 1] == 0):
            end = float(t[i] if i == len(t) - 1 else t[i + 1])
            ax.axvspan(max(start, t_min), min(end, t_max), color="tab:red", alpha=0.12, linewidth=0.0)
            in_run = False


__all__ = [
    "add_anomaly_background",
    "attach_window_labels",
    "label_any",
    "label_mid",
    "load_failure_labels",
]
