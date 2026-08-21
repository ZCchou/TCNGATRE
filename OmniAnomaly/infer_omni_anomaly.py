"""Infer OmniAnomaly on failure (or val) flights.

Usage:
  python infer_omni_anomaly.py

Produces standard score CSVs for later evaluation:
  - sequence_scores.csv
  - all_failure_future_forecast_residual.csv (legacy-compatible duplicate)
"""
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
from data.tranad_dataset import (
    TranADWindowDataset,
    collate_tranad_windows,
    resolve_flight_splits,
)
from data.usad_window_dataset import add_anomaly_background, load_failure_labels
from models.omni_anomaly import build_omni_anomaly
from omnianomalyconfig import OmniAnomalyConfig
from utils.io import ensure_dir
from utils.normalization import load_minmax_stats


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run OmniAnomaly inference on a bundled dataset.")
    parser.add_argument("--dataset", choices=["alfa", "simulate", "gpsdata"], default="alfa")
    return parser.parse_args(argv)


def resolve_device(cfg: OmniAnomalyConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ema_by_flight(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    for flight, g in df.groupby("flight", sort=True):
        g = g.sort_values("current_index", kind="mergesort").copy()
        vals = g["raw_total_score"].to_numpy(dtype=float)
        if len(vals) <= 0:
            rows.append(g)
            continue
        sm = np.zeros_like(vals, dtype=np.float64)
        sm[0] = vals[0]
        for i in range(1, len(vals)):
            sm[i] = alpha * vals[i] + (1 - alpha) * sm[i - 1]
        g["total_score"] = sm
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else df.copy()


def plot_score_curves(
    df: pd.DataFrame,
    out_dir: Path,
    labels_root: Path | None = None,
    max_flights: int = 0,
    time_offset_sec: float = 0.0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    flights = list(df["flight"].drop_duplicates())
    if max_flights > 0:
        flights = flights[:max_flights]
    for flight in flights:
        g = df.loc[df["flight"] == flight].sort_values("current_index")
        if len(g) <= 0:
            continue
        t = g["t_mid"].to_numpy(dtype=float)
        labels_df = None
        if labels_root is not None:
            labels_df = load_failure_labels(
                labels_root=labels_root,
                flight=str(flight),
                time_offset_sec=time_offset_sec,
            )
        fig, ax = plt.subplots(figsize=(14, 4.8))
        ax.plot(t, g["total_score"].to_numpy(dtype=float), color="tab:blue", lw=1.4, label="total_score")
        if labels_df is not None:
            add_anomaly_background(ax, labels_df=labels_df, t_min=float(np.min(t)), t_max=float(np.max(t)))
        ax.set_title(f"{flight} | OmniAnomaly total_score")
        ax.set_xlabel("t (sec)")
        ax.set_ylabel("score")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{flight}__omni_anomaly_score.png", dpi=140)
        plt.close(fig)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    cfg = OmniAnomalyConfig(dataset_name=args.dataset)
    device = resolve_device(cfg)
    run_root = cfg.run_root
    infer_root = run_root / cfg.infer_output_name
    ensure_dir(infer_root)

    # Load normalization stats
    norm_stats = load_minmax_stats(cfg.normalization_stats_path)

    # Flight splits
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
    else:
        flights = train_flights
        stride = cfg.stride_val

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
    print(f"[INFER] split={split_name}  flights={len(flights)}  samples={len(dataset)}")

    # Load model
    ckpt = torch.load(run_root / "best.pt", map_location=device)
    model = build_omni_anomaly(
        input_dim=int(ckpt["input_dim"]),
        rnn_hidden=cfg.rnn_hidden_dim,
        latent_dim=cfg.latent_dim,
        rnn_layers=cfg.rnn_layers,
        dropout=0.0,  # no dropout at inference
        num_flows=cfg.num_planar_flows,
        decoder_rnn_hidden=cfg.decoder_rnn_hidden_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    feature_names = list(dataset.feature_names)

    # Inference
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="OmniAnomaly infer"):
            window = batch["window"].to(device)
            scores = model.anomaly_score(window, mc_samples=cfg.mc_samples)
            final_score = scores["final_score"].cpu().numpy()
            step_score = scores["step_score"].cpu().numpy()
            error = scores["error"].cpu().numpy()  # (B, T, D)
            true_value = window.detach().cpu().numpy()
            pred_value = scores["reconstruction"].cpu().numpy()

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
                if cfg.save_per_dim_error:
                    for dim_idx, name in enumerate(feature_names):
                        payload[f"last_err__{name}"] = float(error[i, -1, dim_idx])
                        payload[f"last_true__{name}"] = float(true_value[i, -1, dim_idx])
                        payload[f"last_pred__{name}"] = float(pred_value[i, -1, dim_idx])
                        payload[f"last_sensor_score__{name}"] = float(error[i, -1, dim_idx])
                rows.append(payload)

    scored_df = pd.DataFrame(rows).sort_values(
        ["flight", "current_index"], kind="mergesort"
    ).reset_index(drop=True)

    if cfg.use_ema and cfg.ema_alpha > 0:
        scored_df = ema_by_flight(scored_df, cfg.ema_alpha)

    sequence_csv = infer_root / "sequence_scores.csv"
    legacy_csv = infer_root / "all_failure_future_forecast_residual.csv"
    scored_df.to_csv(sequence_csv, index=False, encoding="utf-8")
    scored_df.to_csv(legacy_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] saved {len(scored_df)} rows -> {sequence_csv}")
    print(f"[OK] legacy score csv -> {legacy_csv}")

    # Plot score curves
    if cfg.plot_scores:
        plot_score_curves(
            df=scored_df,
            out_dir=infer_root / "figures",
            labels_root=cfg.labels_root if cfg.labels_root.exists() and split_name in {"failure", "test"} else None,
            max_flights=cfg.plot_max_flights,
            time_offset_sec=cfg.failure_label_time_offset_sec,
        )
        if cfg.save_per_dim_error and cfg.plot_compare_timelines:
            written = plot_pred_true_resid_score_timelines(
                scored_df=scored_df,
                feature_names=feature_names,
                out_dir=infer_root / "figures",
                model_label="OmniAnomaly",
                labels_root=cfg.labels_root if cfg.labels_root.exists() and split_name in {"failure", "test"} else None,
                max_flights=cfg.plot_max_flights,
                time_offset_sec=cfg.failure_label_time_offset_sec,
            )
            if written <= 0:
                print("[WARN] OmniAnomaly compare timelines were skipped; required per-dim compare columns were missing.")

    # Save infer config
    (infer_root / "infer_config.json").write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[DONE] OmniAnomaly inference outputs: {infer_root}")


if __name__ == "__main__":
    main()
