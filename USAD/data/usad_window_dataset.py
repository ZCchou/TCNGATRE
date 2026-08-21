from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.normalization import apply_train_minmax, load_wide_flight_frame


TIME_COLUMN_CANDIDATES = ("t", "time", "timestamp", "time_sec", "time_s", "seconds", "sec", "elapsed_sec")
LABEL_COLUMN_CANDIDATES = ("anomaly_label", "label", "fault_flag", "is_anomaly", "anomaly", "target", "attack", "failure")
START_COLUMN_CANDIDATES = ("start_sec", "t_start", "start", "start_time", "begin", "onset")
END_COLUMN_CANDIDATES = ("end_sec", "t_end", "end", "end_time", "stop", "offset")
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
    text = series.astype(str).str.strip().str.lower()
    positive = text.isin({"1", "true", "yes", "y", "anomaly", "fault", "failure", "attack", "abnormal"})
    negative = text.isin({"0", "false", "no", "n", "normal", "benign", "none", "ok"})
    numeric = numeric.mask(numeric.isna() & positive, 1.0)
    numeric = numeric.mask(numeric.isna() & negative, 0.0)
    return numeric.fillna(0.0)


class USADWindowDataset(Dataset):
    def __init__(
        self,
        csv_root: Path | None,
        flights: list[str] | None,
        normalization_stats: dict,
        window_size: int,
        stride: int = 1,
        trim_leading_sec: float = 0.0,
        use_replication_padding: bool = True,
        flight_paths: dict[str, Path] | None = None,
    ):
        self.csv_root = None if csv_root is None else Path(csv_root)
        self.window_size = max(int(window_size), 1)
        self.stride = max(int(stride), 1)
        self.trim_leading_sec = float(trim_leading_sec)
        self.use_replication_padding = bool(use_replication_padding)
        self.normalization_stats = dict(normalization_stats)

        self.feature_names = [str(x) for x in self.normalization_stats["feature_names"]]
        self.num_features = int(len(self.feature_names))
        self.feature_dim = int(self.window_size * self.num_features)

        if flight_paths is not None:
            items = [(str(flight), Path(path)) for flight, path in flight_paths.items()]
        else:
            if self.csv_root is None or flights is None:
                raise ValueError("USADWindowDataset requires flight_paths or (csv_root, flights)")
            items = [(str(flight), self.csv_root / f"{flight}.csv") for flight in flights]

        self.flight_data: dict[str, dict[str, np.ndarray]] = {}
        self.refs: list[tuple[str, int]] = []
        for flight, csv_path in items:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                continue
            t, x_raw, cols = load_wide_flight_frame(csv_path=csv_path, trim_leading_sec=self.trim_leading_sec)
            if list(cols) != list(self.feature_names):
                raise ValueError(f"Feature names mismatch in {csv_path}")
            if x_raw.shape[0] <= 0:
                continue
            x = apply_train_minmax(x_raw, stats=self.normalization_stats)
            self.flight_data[str(flight)] = {
                "t": t.astype(np.float32, copy=False),
                "x": x.astype(np.float32, copy=False),
            }
            for current_idx in range(0, int(x.shape[0]), self.stride):
                self.refs.append((str(flight), int(current_idx)))

        self.flights = list(self.flight_data.keys())

    def __len__(self) -> int:
        return len(self.refs)

    def _build_window(self, x: np.ndarray, current_idx: int) -> np.ndarray:
        start = max(0, int(current_idx) - self.window_size + 1)
        window = x[start:int(current_idx) + 1]
        if window.shape[0] == self.window_size:
            return window.astype(np.float32, copy=False)
        if not self.use_replication_padding:
            pad = np.zeros((self.window_size - window.shape[0], self.num_features), dtype=np.float32)
            return np.concatenate([pad, window], axis=0).astype(np.float32, copy=False)
        pad_len = self.window_size - window.shape[0]
        pad_value = np.repeat(x[int(current_idx):int(current_idx) + 1], pad_len, axis=0)
        return np.concatenate([pad_value, window], axis=0).astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        flight, current_idx = self.refs[idx]
        seq = self.flight_data[str(flight)]
        t = seq["t"]
        x = seq["x"]
        window = self._build_window(x=x, current_idx=int(current_idx))
        start_idx = max(0, int(current_idx) - self.window_size + 1)
        t_start = float(t[start_idx])
        t_end = float(t[int(current_idx)])
        t_mid = 0.5 * (t_start + t_end)
        return {
            "x": torch.from_numpy(window.reshape(-1).astype(np.float32, copy=False)).float(),
            "window": torch.from_numpy(window).float(),
            "flight": str(flight),
            "sample_index": int(current_idx),
            "current_index": int(current_idx),
            "t_start": float(t_start),
            "t_end": float(t_end),
            "t_mid": float(t_mid),
        }


def collate_usad_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "window": torch.stack([item["window"] for item in batch], dim=0),
        "flight": [str(item["flight"]) for item in batch],
        "sample_index": [int(item["sample_index"]) for item in batch],
        "current_index": torch.tensor([int(item["current_index"]) for item in batch], dtype=torch.long),
        "t_start": torch.tensor([float(item["t_start"]) for item in batch], dtype=torch.float32),
        "t_end": torch.tensor([float(item["t_end"]) for item in batch], dtype=torch.float32),
        "t_mid": torch.tensor([float(item["t_mid"]) for item in batch], dtype=torch.float32),
    }


def load_failure_labels(labels_root: Path, flight: str, time_offset_sec: float = 0.0) -> pd.DataFrame | None:
    path = Path(labels_root) / f"{flight}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    time_col = _find_column(df, TIME_COLUMN_CANDIDATES)
    label_col = _find_column(df, LABEL_COLUMN_CANDIDATES)
    if time_col is not None and label_col is not None:
        out = pd.DataFrame(
            {
                "t": pd.to_numeric(df[time_col], errors="coerce"),
                "anomaly_label": _coerce_label(df[label_col]),
            }
        ).dropna(subset=["t"])
    else:
        start_col = _find_column(df, START_COLUMN_CANDIDATES)
        end_col = _find_column(df, END_COLUMN_CANDIDATES)
        if start_col is None or end_col is None:
            return None
        else:
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
                rows.extend(
                    [
                        {"t": np.nextafter(s, -np.inf), "anomaly_label": 0.0},
                        {"t": s, "anomaly_label": v},
                        {"t": e, "anomaly_label": v},
                        {"t": np.nextafter(e, np.inf), "anomaly_label": 0.0},
                    ]
                )
            out = pd.DataFrame(rows)
    if len(out) <= 0:
        return None
    if float(time_offset_sec) != 0.0:
        out["t"] = pd.to_numeric(out["t"], errors="coerce") + float(time_offset_sec)
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
        score_col = "scores_smooth" if "scores_smooth" in g.columns else "total_score"
        score = g[score_col].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(14, 4.8))
        ax.plot(t_mid, score, color="tab:blue", linewidth=1.5, label=score_col)
        if threshold is not None and np.isfinite(float(threshold)):
            ax.axhline(float(threshold), color="tab:orange", linestyle="--", linewidth=1.2, label="global threshold")
        add_anomaly_background(ax, labels_df=labels_df, t_min=float(np.min(t_mid)), t_max=float(np.max(t_mid)))
        ax.set_title(f"{flight} | USAD {score_col}")
        ax.set_xlabel("t (sec)")
        ax.set_ylabel("score")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{flight}__usad_total_score.png", dpi=140)
        plt.close(fig)
