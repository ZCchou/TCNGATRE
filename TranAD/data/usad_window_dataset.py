from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_failure_labels(labels_root: Path, flight: str, time_offset_sec: float = 0.0) -> pd.DataFrame | None:
    path = Path(labels_root) / f"{flight}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "t" not in df.columns or "anomaly_label" not in df.columns:
        return None
    out = pd.DataFrame(
        {
            "t": pd.to_numeric(df["t"], errors="coerce"),
            "anomaly_label": pd.to_numeric(df["anomaly_label"], errors="coerce").fillna(0.0),
        }
    ).dropna(subset=["t"])
    if len(out) <= 0:
        return None
    if float(time_offset_sec) != 0.0:
        out["t"] = pd.to_numeric(out["t"], errors="coerce") + float(time_offset_sec)
    return out.sort_values("t", kind="mergesort").reset_index(drop=True)


def label_mid(labels_df: pd.DataFrame, t_mid: float) -> int:
    t = labels_df["t"].to_numpy(dtype=float)
    idx = int(np.searchsorted(t, float(t_mid), side="right") - 1)
    idx = min(max(idx, 0), len(t) - 1)
    return int(float(labels_df.iloc[idx]["anomaly_label"]) > 0.5)


def label_any(labels_df: pd.DataFrame, t_start: float, t_end: float) -> int:
    t = labels_df["t"].to_numpy(dtype=float)
    y = labels_df["anomaly_label"].to_numpy(dtype=float)
    mask = (t >= float(t_start)) & (t <= float(t_end))
    if bool(mask.any()):
        return int(bool((y[mask] > 0.5).any()))
    return label_mid(labels_df, 0.5 * (float(t_start) + float(t_end)))


def attach_window_labels(scored_df: pd.DataFrame, labels_root: Path, time_offset_sec: float = 0.0) -> pd.DataFrame:
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
                payload["label_any"] = label_any(labels_df, float(payload["t_start"]), float(payload["t_end"]))
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


def plot_scores_by_flight(
    scored_df: pd.DataFrame,
    out_dir: Path,
    labels_root: Path | None = None,
    threshold: float | None = None,
    max_flights: int = 0,
    time_offset_sec: float = 0.0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    flights = list(scored_df["flight"].drop_duplicates())
    if int(max_flights) > 0:
        flights = flights[: int(max_flights)]

    for flight in flights:
        g = scored_df.loc[scored_df["flight"] == flight].sort_values("t_start", kind="mergesort")
        if len(g) <= 0:
            continue
        labels_df = None
        if labels_root is not None:
            labels_df = load_failure_labels(
                labels_root=Path(labels_root),
                flight=str(flight),
                time_offset_sec=time_offset_sec,
            )

        t_mid = g["t_mid"].to_numpy(dtype=float)
        score = g["total_score"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(14, 4.8))
        ax.plot(t_mid, score, color="tab:blue", linewidth=1.5, label="total_score")
        if threshold is not None and np.isfinite(float(threshold)):
            ax.axhline(float(threshold), color="tab:orange", linestyle="--", linewidth=1.2, label="global threshold")
        add_anomaly_background(ax, labels_df=labels_df, t_min=float(np.min(t_mid)), t_max=float(np.max(t_mid)))
        ax.set_title(f"{flight} | score")
        ax.set_xlabel("t (sec)")
        ax.set_ylabel("score")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{flight}__score.png", dpi=140)
        plt.close(fig)
