from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.stgtcn_window_dataset import (
    STGTCNWindowDataset,
    collate_stgtcn_windows,
    resolve_flight_splits,
)
from data.window_labels import load_failure_labels as _load_labels_compat
from base_config import TCNGATREConfig
from train_impl import (
    build_model,
    ensure_dir,
    load_graph,
    save_json,
)
from utils.normalization import load_minmax_stats


def _normalize_role(role: str) -> str:
    r = str(role).strip().lower()
    if r in {"train", "train_flights"}:
        return "train_flights"
    if r in {"val", "valid", "validation", "val_flights"}:
        return "val_flights"
    return r


def load_failure_labels(labels_root: Path, flight: str, time_offset_sec: float = 0.0) -> pd.DataFrame | None:
    return _load_labels_compat(labels_root, flight, time_offset_sec=time_offset_sec)


def load_flight_filter(run_dir: Path, role: str | None) -> set[str] | None:
    role = _normalize_role("" if role is None else role)
    if not role:
        return None
    split_path = run_dir / "split_flights.json"
    if not split_path.exists():
        raise FileNotFoundError(f"split_flights.json not found: {split_path}")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if role not in payload:
        raise ValueError(f"Unknown role={role}; available={list(payload.keys())}")
    return {str(x) for x in payload[role]}


def resolve_flight_paths(cfg: TCNGATREConfig, split_name: str, flight_role: str | None) -> dict[str, Path]:
    train_paths, val_paths, failure_paths = resolve_flight_splits(
        dataset_root=Path(cfg.data_root),
        split_info_path=Path(cfg.split_info_path),
    )
    s = str(split_name).strip().lower()
    if s in {"failure", "fail", "abnormal"}:
        paths = dict(failure_paths)
    elif s in {"no_failure", "nofailure", "normal"}:
        role = _normalize_role("" if flight_role is None else flight_role)
        if role == "val_flights":
            paths = dict(val_paths)
        elif role == "train_flights":
            paths = dict(train_paths)
        else:
            paths = {**train_paths, **val_paths}
    else:
        raise ValueError(f"Unsupported split: {split_name}")
    filt = load_flight_filter(Path(cfg.run_root), role=flight_role)
    if filt is not None:
        paths = {n: p for n, p in paths.items() if str(n) in filt}
    if not paths:
        raise ValueError(f"No flights for split={split_name}, role={flight_role}")
    return paths


def load_checkpoint(cfg: TCNGATREConfig, ckpt_path: Path, num_nodes: int, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = build_model(cfg, num_nodes, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _safe_json_list(x: np.ndarray) -> str:
    arr = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return json.dumps([float(v) for v in arr.tolist()], ensure_ascii=False)


def _stack_json_vectors(series: pd.Series) -> np.ndarray:
    parsed = []
    for item in series.to_list():
        arr = np.asarray(json.loads(item) if isinstance(item, str) else item, dtype=np.float32)
        parsed.append(arr)
    arr = np.stack(parsed, axis=0).astype(np.float32)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    return arr


def materialize_vecs(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = [
                x if isinstance(x, str) else _safe_json_list(np.asarray(x, dtype=np.float32))
                for x in out[col].tolist()
            ]
    return out


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    prev = np.nan
    for i, v in enumerate(values):
        prev = float(v) if not np.isfinite(prev) else alpha * float(v) + (1.0 - alpha) * prev
        out[i] = prev
    return out


def aggregate_scores(df: pd.DataFrame, alpha: float, ema_enabled: bool) -> pd.DataFrame:
    parts = []
    for _, group in df.groupby("flight", sort=False):
        g = group.sort_values("t_start", kind="mergesort").reset_index(drop=True).copy()
        val_resid = _stack_json_vectors(g["value_residual_vec"])
        delta_resid = _stack_json_vectors(g["delta_residual_vec"])
        sensor_score = (val_resid + 0.20 * delta_resid).astype(np.float32)
        raw_total = np.mean(sensor_score, axis=1).astype(np.float64)
        total = _ema(raw_total, alpha) if ema_enabled else raw_total.copy()
        g["sensor_score_vec"] = [x.astype(np.float32, copy=True) for x in sensor_score]
        g["raw_total_score"] = raw_total
        g["total_score"] = total
        g["valid_dim_count"] = sensor_score.shape[1]
        g["score_is_valid"] = np.ones(len(g), dtype=np.int8)
        g["valid_sensor_count"] = sensor_score.shape[1]
        parts.append(g)
    return pd.concat(parts, axis=0, ignore_index=True)


def add_anomaly_background(ax, labels_df: pd.DataFrame | None, t_min: float, t_max: float) -> None:
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
            end = float(t[i if i == len(t) - 1 else i + 1])
            ax.axvspan(max(start, t_min), min(end, t_max), color="tab:red", alpha=0.12, linewidth=0.0)
            in_run = False


def _bfill_ffill(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    nxt = np.nan
    for i in range(len(arr) - 1, -1, -1):
        if np.isfinite(arr[i]):
            nxt = arr[i]
        elif np.isfinite(nxt):
            arr[i] = nxt
    prv = np.nan
    for i in range(len(arr)):
        if np.isfinite(arr[i]):
            prv = arr[i]
        elif np.isfinite(prv):
            arr[i] = prv
    return arr


def _display_norm(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    ok = np.isfinite(arr)
    if not ok.any():
        return np.zeros_like(arr)
    vmin, vmax = float(np.min(arr[ok])), float(np.max(arr[ok]))
    if vmax <= vmin + 1e-12:
        out = np.zeros_like(arr)
        out[ok] = 0.5
        return out
    out = np.zeros_like(arr)
    out[ok] = (arr[ok] - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0)


def _set_ylim(ax, *series, lower_q=0.01, upper_q=0.99, pad_ratio=0.08):
    vals = np.concatenate([np.asarray(s, dtype=np.float64) for s in series])
    ok = vals[np.isfinite(vals)]
    if ok.size == 0:
        return
    lo, hi = float(np.quantile(ok, lower_q)), float(np.quantile(ok, upper_q))
    if not np.isfinite(lo) or hi <= lo + 1e-12:
        return
    pad = max((hi - lo) * pad_ratio, 1e-6)
    ax.set_ylim(lo - pad, hi + pad)


def plot_flight_outputs(
    flight_df: pd.DataFrame,
    out_dir: Path,
    flight: str,
    sensor_names: List[str],
    cfg: TCNGATREConfig,
    top_n: int = 20,
    page_size: int = 24,
    dpi: int = 160,
) -> None:
    labels_root = Path(cfg.labels_root)
    offset = float(cfg.failure_label_time_offset_sec or 0.0)
    labels_df = load_failure_labels(labels_root, flight, time_offset_sec=offset)

    t_mid = flight_df["t_mid"].to_numpy(dtype=float)
    t_min, t_max = float(np.min(t_mid)), float(np.max(t_mid))

    score = _display_norm(flight_df["total_score"].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(14, 4))
    add_anomaly_background(ax, labels_df, t_min, t_max)
    ax.plot(t_mid, score, color="tab:blue", linewidth=1.4)
    ax.set_title(f"{flight} | TCNGATRE total score (display normalized)")
    ax.set_xlabel("t (sec)")
    ax.set_ylabel("score (display)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{flight}__total_score_timeline.png", dpi=dpi)
    plt.close(fig)

    sensor_score_mat = _stack_json_vectors(flight_df["sensor_score_vec"])
    future_value_mat = _stack_json_vectors(flight_df["future_value_vec"])
    pred_value_mat = _stack_json_vectors(flight_df["pred_value_vec"])
    resid_col = next(
        (c for c in ["smoothed_resid_abs_vec", "resid_abs_vec", "value_residual_vec"] if c in flight_df.columns),
        "value_residual_vec",
    )
    resid_mat = _stack_json_vectors(flight_df[resid_col])
    sensor_rank = np.argsort(-sensor_score_mat.mean(axis=0))

    def _plot_page(indices: List[int], save_path: Path) -> None:
        if not indices:
            return
        fig, axes = plt.subplots(len(indices), 2, figsize=(16, 4 * len(indices)), squeeze=False)
        for row_i, si in enumerate(indices):
            name = sensor_names[si] if si < len(sensor_names) else f"sensor_{si}"
            tv = _bfill_ffill(future_value_mat[:, si].astype(float))
            pv = _bfill_ffill(pred_value_mat[:, si].astype(float))
            rv = _bfill_ffill(resid_mat[:, si].astype(float))
            sv = _bfill_ffill(sensor_score_mat[:, si].astype(float))

            ax0 = axes[row_i, 0]
            add_anomaly_background(ax0, labels_df, t_min, t_max)
            ax0.plot(t_mid, tv, label="true", linewidth=1.2)
            ax0.plot(t_mid, pv, label="pred", linewidth=1.2, linestyle="--")
            ax0.set_title(f"{flight} | {name} | true vs pred")
            ax0.set_xlabel("t (sec)")
            _set_ylim(ax0, tv, pv)
            ax0.grid(True, linewidth=0.3)
            ax0.legend(loc="upper right", fontsize=8)

            ax1 = axes[row_i, 1]
            add_anomaly_background(ax1, labels_df, t_min, t_max)
            ax1.plot(t_mid, _display_norm(rv), label="|resid| (display)", linewidth=1.2)
            ax1.plot(t_mid, _display_norm(sv), label="sensor_score (display)", linewidth=1.2)
            ax1.set_title(f"{flight} | {name} | residual vs score")
            ax1.set_xlabel("t (sec)")
            ax1.set_ylim(0.0, 1.0)
            ax1.grid(True, linewidth=0.3)
            ax1.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)

    top_indices = sensor_rank[:min(top_n, len(sensor_rank))].tolist()
    _plot_page(top_indices, out_dir / f"{flight}__top{top_n}_pred_true_timeline.png")

    all_indices = sensor_rank.tolist()
    num_pages = int(np.ceil(len(all_indices) / page_size))
    for pi in range(num_pages):
        indices = all_indices[pi * page_size: (pi + 1) * page_size]
        save = out_dir / (f"{flight}__compare.png" if pi == 0 else f"{flight}__compare_all_p{pi+1:02d}.png")
        _plot_page(indices, save)


def main(runtime_cfg: TCNGATREConfig | None = None) -> None:
    cfg = TCNGATREConfig() if runtime_cfg is None else runtime_cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else torch.device(cfg.device)

    node_names, a, m = load_graph(Path(cfg.graph_dir))
    a = a.to(device)
    m = m.to(device)

    flight_paths = resolve_flight_paths(
        cfg=cfg,
        split_name=cfg.infer_source_split,
        flight_role=cfg.infer_flight_filter_role or None,
    )
    norm_stats = load_minmax_stats(Path(cfg.normalization_stats_path))
    dataset = STGTCNWindowDataset(
        csv_root=None, flights=None,
        normalization_stats=norm_stats,
        node_names=node_names,
        history_len=cfg.lookback,
        horizon=cfg.horizon_out,
        stride=cfg.sample_stride,
        trim_leading_sec=cfg.trim_leading_sec,
        use_replication_padding=False,
        flight_paths=flight_paths,
    )
    dl = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_stgtcn_windows,
        pin_memory=torch.cuda.is_available(),
    )

    ckpt_path = Path(cfg.run_root) / cfg.infer_checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = load_checkpoint(cfg, ckpt_path, len(node_names), device)

    out_dir = Path(cfg.run_root) / cfg.infer_output_name
    per_flight_dir = out_dir / "per_flight_residual_csv"
    fig_dir = out_dir / "figures"
    ensure_dir(out_dir)
    ensure_dir(per_flight_dir)
    ensure_dir(fig_dir)

    rows = []
    with torch.no_grad():
        for batch in tqdm(dl, desc="[TCNGATRE Infer]", unit="batch", dynamic_ncols=True):
            x = batch["x"].to(device).float()
            y = batch["y"].to(device).float()
            y_hat, aux = model(x, a, m, short_patch=cfg.short_patch)

            pred = y_hat[:, :, :, 0].detach().cpu().numpy().astype(np.float32)
            truth = y[:, :, :, 0].detach().cpu().numpy().astype(np.float32)

            value_resid = np.mean(np.abs(pred - truth), axis=1)
            pred_value = np.mean(pred, axis=1)
            future_value = np.mean(truth, axis=1)
            if pred.shape[1] >= 2:
                delta_resid = np.mean(np.abs(np.diff(pred, axis=1) - np.diff(truth, axis=1)), axis=1)
            else:
                delta_resid = np.zeros_like(value_resid)
            sq_err = np.mean(np.square(pred - truth), axis=1)
            abs_err = value_resid

            for i, flight in enumerate(batch["flight"]):
                t_s = float(batch["t_future_start"][i].item())
                t_e = float(batch["t_future_end"][i].item())
                rows.append({
                    "flight": str(flight),
                    "current_index": int(batch["current_index"][i].item()),
                    "t_start": t_s,
                    "t_end": t_e,
                    "t_mid": 0.5 * (t_s + t_e),
                    "mse": float(np.mean(sq_err[i])),
                    "mae": float(np.mean(abs_err[i])),
                    "rmse": float(np.sqrt(np.mean(sq_err[i]))),
                    "future_value_vec": future_value[i].astype(np.float32, copy=True),
                    "pred_value_vec": pred_value[i].astype(np.float32, copy=True),
                    "value_residual_vec": value_resid[i].astype(np.float32, copy=True),
                    "delta_residual_vec": delta_resid[i].astype(np.float32, copy=True),
                    "resid_abs_vec": value_resid[i].astype(np.float32, copy=True),
                    "sq_err_vec": sq_err[i].astype(np.float32, copy=True),
                    "short_window_len": int(aux["short_window_len"].item()),
                })

    if not rows:
        raise ValueError("Inference produced no rows.")

    all_df = pd.DataFrame(rows).sort_values(["flight", "t_start"], kind="mergesort").reset_index(drop=True)
    all_df = aggregate_scores(
        df=all_df,
        alpha=float(cfg.score_temporal_smooth_alpha),
        ema_enabled=bool(cfg.score_ema_enabled),
    )
    all_df["resid_norm_vec"] = all_df["sensor_score_vec"]

    vector_cols = [
        "future_value_vec", "pred_value_vec", "value_residual_vec",
        "delta_residual_vec", "resid_abs_vec", "sq_err_vec",
        "sensor_score_vec", "resid_norm_vec",
    ]
    export_df = materialize_vecs(all_df, vector_cols)
    csv_name = f"all_{cfg.infer_source_split.lower()}_window_forecast_residual.csv"
    export_df.to_csv(out_dir / csv_name, index=False, encoding="utf-8-sig")

    seq_cols = ["flight", "current_index", "t_start", "t_end", "t_mid", "raw_total_score", "total_score", "valid_dim_count"]
    all_df[seq_cols].to_csv(out_dir / "sequence_scores.csv", index=False, encoding="utf-8")

    for flight, g in export_df.groupby("flight", sort=False):
        g.sort_values("t_start", kind="mergesort").to_csv(
            per_flight_dir / f"{flight}.csv", index=False, encoding="utf-8-sig"
        )

    if bool(cfg.plot_scores):
        for flight, g in tqdm(
            list(all_df.groupby("flight", sort=False)),
            desc="[TCNGATRE] plotting", unit="flight", dynamic_ncols=True,
        ):
            plot_flight_outputs(
                flight_df=g.sort_values("t_start", kind="mergesort").reset_index(drop=True),
                out_dir=fig_dir,
                flight=str(flight),
                sensor_names=node_names,
                cfg=cfg,
                top_n=20,
                page_size=24,
                dpi=160,
            )

    save_json(
        {
            "model": "TCNGATRE",
            "source_split": cfg.infer_source_split,
            "checkpoint": str(ckpt_path),
            "num_nodes": len(node_names),
            "output_csv": str(out_dir / csv_name),
            "sequence_scores_csv": str(out_dir / "sequence_scores.csv"),
            "score_rule": "sensor_score = mean_h(|pred-true|) + 0.20*mean_h(|Δpred-Δtrue|); total = mean(sensor_score)",
            "ema_enabled": bool(cfg.score_ema_enabled),
            "ema_alpha": float(cfg.score_temporal_smooth_alpha),
            "num_rows": int(len(all_df)),
            "num_flights": int(all_df["flight"].nunique()),
        },
        out_dir / "infer_summary.json",
    )
    print(f"[OK] inference saved to: {out_dir}")


if __name__ == "__main__":
    main()
