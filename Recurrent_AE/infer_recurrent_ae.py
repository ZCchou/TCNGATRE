from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

from common.pred_true_resid_compare import plot_pred_true_resid_score_timelines
from data.tranad_dataset import TranADWindowDataset, collate_tranad_windows, resolve_flight_splits
from data.usad_window_dataset import add_anomaly_background, load_failure_labels
from models.classic_ts_ae import build_classic_ts_autoencoder
from recurrentaeconfig import RecurrentAEConfig
from utils.io import ensure_dir
from utils.normalization import load_minmax_stats


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run Recurrent AE inference on a bundled dataset.")
    parser.add_argument("--dataset", choices=["alfa", "alfa4hz", "simulate", "gpsdata"], default="alfa")
    return parser.parse_args(argv)


def resolve_device(cfg: RecurrentAEConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_label(cfg: RecurrentAEConfig) -> str:
    name_map = {
        "lstm": "LSTM-AE",
        "gru": "GRU-AE",
        "conv": "Conv1D-AE",
        "tcn": "TCN-AE",
    }
    return name_map.get(str(cfg.model_type).strip().lower(), f"{cfg.model_type.upper()}-AE")


def ema_by_flight(scored_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for flight, g in scored_df.groupby("flight", sort=True):
        g = g.sort_values("current_index", kind="mergesort").copy()
        values = g["raw_total_score"].to_numpy(dtype=float)
        if len(values) <= 0:
            rows.append(g)
            continue
        smoothed = np.zeros_like(values, dtype=np.float64)
        smoothed[0] = values[0]
        for i in range(1, len(values)):
            smoothed[i] = float(alpha) * values[i] + (1.0 - float(alpha)) * smoothed[i - 1]
        g["total_score"] = smoothed.astype(np.float64)
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else scored_df.copy()


def plot_score_curves(
    scored_df: pd.DataFrame,
    out_dir: Path,
    title: str,
    labels_root: Path | None = None,
    max_flights: int = 0,
    time_offset_sec: float = 0.0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    flights = list(scored_df["flight"].drop_duplicates())
    if int(max_flights) > 0:
        flights = flights[: int(max_flights)]
    for flight in flights:
        g = scored_df.loc[scored_df["flight"] == flight].sort_values("current_index", kind="mergesort")
        if len(g) <= 0:
            continue
        labels_df = (
            None
            if labels_root is None
            else load_failure_labels(
                labels_root=labels_root,
                flight=str(flight),
                time_offset_sec=time_offset_sec,
            )
        )
        t = g["t_mid"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(14, 4.8))
        ax.plot(t, g["total_score"].to_numpy(dtype=float), color="tab:blue", linewidth=1.4, label="total_score")
        if labels_df is not None:
            add_anomaly_background(ax, labels_df=labels_df, t_min=float(np.min(t)), t_max=float(np.max(t)))
        ax.set_title(f"{flight} | {title} total_score")
        ax.set_xlabel("t (sec)")
        ax.set_ylabel("score")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{flight}__recurrent_ae_total_score.png", dpi=140)
        plt.close(fig)


def plot_residual_heatmaps(
    scored_df: pd.DataFrame,
    feature_names: list[str],
    out_dir: Path,
    title: str,
    max_flights: int = 0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    dim_cols = [f"last_err__{name}" for name in feature_names if f"last_err__{name}" in scored_df.columns]
    if not dim_cols:
        return
    flights = list(scored_df["flight"].drop_duplicates())
    if int(max_flights) > 0:
        flights = flights[: int(max_flights)]
    for flight in flights:
        g = scored_df.loc[scored_df["flight"] == flight].sort_values("current_index", kind="mergesort")
        if len(g) <= 0:
            continue
        mat = g[dim_cols].to_numpy(dtype=float).T
        fig, ax = plt.subplots(figsize=(14, 5.5))
        im = ax.imshow(mat, aspect="auto", origin="lower", interpolation="nearest", cmap="magma")
        ax.set_title(f"{flight} | {title} last-step residual heatmap")
        ax.set_xlabel("time index")
        ax.set_ylabel("feature")
        ax.set_yticks(np.arange(len(dim_cols)))
        ax.set_yticklabels([x.replace("last_err__", "") for x in dim_cols], fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
        fig.tight_layout()
        fig.savefig(out_dir / f"{flight}__recurrent_ae_residual_heatmap.png", dpi=140)
        plt.close(fig)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    cfg = RecurrentAEConfig(dataset_name=args.dataset)
    device = resolve_device(cfg)
    run_root = cfg.run_root
    infer_root = run_root / cfg.infer_output_name
    ensure_dir(infer_root)

    norm_stats = load_minmax_stats(cfg.normalization_stats_path)
    train_flights, val_flights, failure_flights = resolve_flight_splits(
        dataset_root=cfg.data_root,
    )
    split_name = str(cfg.infer_split).strip().lower()
    if split_name in {"failure", "test"}:
        flights = failure_flights
        stride = cfg.stride_test
    elif split_name in {"val", "val_normal", "validation"}:
        flights = val_flights
        stride = cfg.stride_val
    elif split_name in {"train", "train_normal"}:
        flights = train_flights
        stride = cfg.stride_val
    else:
        raise ValueError(f"Unsupported infer split: {cfg.infer_split}")

    dataset = TranADWindowDataset(
        csv_root=None,
        flights=None,
        normalization_stats=norm_stats,
        window_size=cfg.window_size,
        stride=stride,
        trim_leading_sec=cfg.trim_leading_sec,
        use_replication_padding=cfg.use_replication_padding,
        context_mode=cfg.context_mode,
        flight_paths=flights,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_tranad_windows,
    )

    ckpt = torch.load(run_root / "best.pt", map_location=device)
    model = build_classic_ts_autoencoder(
        model_type=cfg.model_type,
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        conv_kernel_size=cfg.conv_kernel_size,
        tcn_kernel_size=cfg.tcn_kernel_size,
        tcn_num_levels=cfg.tcn_num_levels,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    feature_names = list(dataset.feature_names)
    rows: list[dict] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"infer {cfg.model_type} ae {split_name}", leave=False):
            window = batch["window"].to(device)
            comps = model.anomaly_components(window)
            true_value = window.detach().cpu().numpy()
            pred_value = comps["reconstruction"].detach().cpu().numpy()
            err = comps["error"].detach().cpu().numpy()
            step_score = comps["step_score"].detach().cpu().numpy()
            final_score = comps["final_score"].detach().cpu().numpy()

            for i in range(len(final_score)):
                payload = {
                    "flight": str(batch["flight"][i]),
                    "current_index": int(batch["current_index"][i].item()),
                    "t_start": float(batch["t"][i].item()),
                    "t_end": float(batch["t"][i].item()),
                    "t_mid": float(batch["t"][i].item()),
                    "raw_total_score": float(final_score[i]),
                    "total_score": float(final_score[i]),
                    "last_step_score": float(step_score[i, -1]),
                }
                for step_idx in range(step_score.shape[1]):
                    payload[f"window_step_score_{step_idx:02d}"] = float(step_score[i, step_idx])
                if cfg.save_per_dim_error:
                    for dim_idx, name in enumerate(feature_names):
                        payload[f"last_err__{name}"] = float(err[i, -1, dim_idx])
                        payload[f"last_true__{name}"] = float(true_value[i, -1, dim_idx])
                        payload[f"last_pred__{name}"] = float(pred_value[i, -1, dim_idx])
                        payload[f"last_sensor_score__{name}"] = float(err[i, -1, dim_idx])
                rows.append(payload)

    scored_df = pd.DataFrame(rows).sort_values(["flight", "current_index"], kind="mergesort").reset_index(drop=True)
    if cfg.use_ema and cfg.ema_alpha > 0.0:
        scored_df = ema_by_flight(scored_df=scored_df, alpha=cfg.ema_alpha)

    sequence_csv = infer_root / "sequence_scores.csv"
    scored_df.to_csv(sequence_csv, index=False, encoding="utf-8")

    title = model_label(cfg)
    if cfg.plot_scores:
        plot_score_curves(
            scored_df=scored_df,
            out_dir=infer_root / "flight_score_curves",
            title=title,
            labels_root=cfg.labels_root if cfg.labels_root.exists() and split_name in {"failure", "test"} else None,
            max_flights=cfg.plot_max_flights,
            time_offset_sec=cfg.failure_label_time_offset_sec,
        )
        if cfg.save_per_dim_error:
            plot_residual_heatmaps(
                scored_df=scored_df,
                feature_names=feature_names,
                out_dir=infer_root / "flight_residual_heatmaps",
                title=title,
                max_flights=cfg.plot_max_flights,
            )
            if cfg.plot_compare_timelines:
                written = plot_pred_true_resid_score_timelines(
                    scored_df=scored_df,
                    feature_names=feature_names,
                    out_dir=infer_root / "figures",
                    model_label=title,
                    labels_root=cfg.labels_root if cfg.labels_root.exists() and split_name in {"failure", "test"} else None,
                    max_flights=cfg.plot_max_flights,
                    time_offset_sec=cfg.failure_label_time_offset_sec,
                )
                if written <= 0:
                    print(f"[WARN] {title} compare timelines were skipped; required per-dim compare columns were missing.")

    (infer_root / "infer_config.json").write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {title} {split_name} inference outputs saved to: {infer_root}")
    print(f"[OK] sequence_scores.csv: {sequence_csv}")


if __name__ == "__main__":
    main()
