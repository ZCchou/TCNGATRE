from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

from common.threshold_methods import (  # noqa: E402
    apply_threshold_methods,
    summarize_per_flight_threshold_methods,
    summarize_threshold_methods,
)
from common.pred_true_resid_compare import plot_pred_true_resid_score_timelines  # noqa: E402
from data.alfa_shared import build_flight_path_maps
from data.usad_window_dataset import (
    USADWindowDataset,
    attach_window_labels,
    collate_usad_windows,
    plot_scores_by_flight,
)
from models.usad import USAD
from usadconfig import USADConfig
from utils.global_threshold import load_global_threshold, save_global_threshold
from utils.io import ensure_dir
from utils.normalization import load_minmax_stats


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run USAD inference on a bundled dataset.")
    parser.add_argument("--dataset", choices=["alfa", "simulate", "gpsdata"], default="alfa")
    return parser.parse_args(argv)


def resolve_device(cfg: USADConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_ranking_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int32)
    score = np.asarray(score, dtype=float)
    positives = int(y_true.sum())
    negatives = int((1 - y_true).sum())
    item = {
        "num_samples": int(len(y_true)),
        "positives": positives,
        "negatives": negatives,
        "auroc": float("nan"),
        "average_precision": float("nan"),
    }
    if len(y_true) <= 0:
        return item
    if positives > 0:
        item["average_precision"] = float(average_precision_score(y_true, score))
    if positives > 0 and negatives > 0:
        item["auroc"] = float(roc_auc_score(y_true, score))
    return item


def compute_metrics_at_threshold(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int32)
    score = np.asarray(score, dtype=float)
    pred = (score >= float(threshold)).astype(np.int32)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    accuracy = float((tp + tn) / max(tp + tn + fp + fn, 1))
    fpr = float(fp / max(fp + tn, 1))
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def build_per_flight_threshold_summary(scored_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict] = []
    for flight, g in scored_df.groupby("flight", sort=True):
        for label_col in ["label_mid", "label_any"]:
            if label_col not in g.columns:
                continue
            valid = pd.to_numeric(g[label_col], errors="coerce").notna().to_numpy()
            if not bool(valid.any()):
                continue
            y = g.loc[valid, label_col].to_numpy(dtype=np.int32)
            s = g.loc[valid, "total_score"].to_numpy(dtype=float)
            row = {"flight": str(flight), "label_col": label_col}
            row.update(compute_ranking_metrics(y_true=y, score=s))
            row.update(compute_metrics_at_threshold(y_true=y, score=s, threshold=threshold))
            rows.append(row)
    return pd.DataFrame(rows)


def save_summary(scored_df: pd.DataFrame, threshold: float, out_dir):
    rows: list[dict] = []
    for label_col in ["label_mid", "label_any"]:
        if label_col not in scored_df.columns:
            continue
        valid = pd.to_numeric(scored_df[label_col], errors="coerce").notna().to_numpy()
        if not bool(valid.any()):
            continue
        y = scored_df.loc[valid, label_col].to_numpy(dtype=np.int32)
        s = scored_df.loc[valid, "total_score"].to_numpy(dtype=float)
        row = {"score_col": "total_score", "label_col": label_col}
        row.update(compute_ranking_metrics(y_true=y, score=s))
        row.update(compute_metrics_at_threshold(y_true=y, score=s, threshold=threshold))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "summary_metrics.csv", index=False, encoding="utf-8")

    per_flight = build_per_flight_threshold_summary(scored_df=scored_df, threshold=threshold)
    per_flight.to_csv(out_dir / "per_flight_total_score_single_global_threshold.csv", index=False, encoding="utf-8")
    if len(per_flight) > 0:
        wide = per_flight.pivot(index="flight", columns="label_col")
        wide.columns = [f"{metric}_{label}" for metric, label in wide.columns]
        wide = wide.reset_index()
        wide.to_csv(
            out_dir / "per_flight_total_score_single_global_threshold_summary.csv",
            index=False,
            encoding="utf-8",
        )


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    cfg = USADConfig(dataset_name=args.dataset)
    run_root = cfg.run_root
    infer_root = run_root / cfg.infer_output_name
    analysis_root = infer_root / "score_threshold_analysis"
    figures_root = analysis_root / "flight_threshold_timelines"
    ensure_dir(infer_root)
    ensure_dir(analysis_root)
    ensure_dir(figures_root)

    device = resolve_device(cfg)
    _, _, failure_flights, meta = build_flight_path_maps(cfg.data_root)
    norm_stats = load_minmax_stats(cfg.normalization_stats_path)
    dataset = USADWindowDataset(
        csv_root=None,
        flights=None,
        normalization_stats=norm_stats,
        window_size=cfg.window_size,
        stride=cfg.sample_stride,
        trim_leading_sec=cfg.trim_leading_sec,
        use_replication_padding=cfg.use_replication_padding,
        flight_paths=failure_flights,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_usad_windows,
    )

    print(
        "[USAD-INFER] "
        f"wide_root={meta['wide_root']} "
        f"failure_flights={len(failure_flights)} "
        f"samples={len(dataset)}"
    )

    checkpoint = torch.load(run_root / "best.pt", map_location=device)
    model = USAD(
        input_dim=int(checkpoint["input_dim"]),
        encoder_hidden_dims=cfg.encoder_hidden_dims,
        latent_dim=cfg.latent_dim,
        decoder_hidden_dims=cfg.decoder_hidden_dims,
        activation=cfg.activation,
        dropout=cfg.dropout,
        use_layernorm=cfg.use_layernorm,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    feature_names = list(dataset.feature_names)
    num_features = int(len(feature_names))
    rows: list[dict] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="infer usad failure", leave=False):
            x = batch["x"].to(device)
            w1, _, w3 = model.reconstruct(x)
            err1 = (x - w1).pow(2).mean(dim=1)
            err3 = (x - w3).pow(2).mean(dim=1)
            scores = (float(cfg.alpha) * err1 + (1.0 - float(cfg.alpha)) * err3).detach().cpu().numpy()
            true_value = x.detach().cpu().numpy().reshape(-1, cfg.window_size, num_features)
            pred_value = w3.detach().cpu().numpy().reshape(-1, cfg.window_size, num_features)
            last_err = np.square(pred_value[:, -1, :] - true_value[:, -1, :])
            for i in range(len(scores)):
                payload = {
                    "flight": str(batch["flight"][i]),
                    "sample_index": int(batch["sample_index"][i]),
                    "current_index": int(batch["current_index"][i].item()),
                    "t_start": float(batch["t_start"][i].item()),
                    "t_end": float(batch["t_end"][i].item()),
                    "t_mid": float(batch["t_mid"][i].item()),
                    "total_score": float(scores[i]),
                    "raw_total_score": float(scores[i]),
                }
                for dim_idx, name in enumerate(feature_names):
                    payload[f"last_err__{name}"] = float(last_err[i, dim_idx])
                    payload[f"last_true__{name}"] = float(true_value[i, -1, dim_idx])
                    payload[f"last_pred__{name}"] = float(pred_value[i, -1, dim_idx])
                    payload[f"last_sensor_score__{name}"] = float(last_err[i, dim_idx])
                rows.append(payload)

    scored_df = pd.DataFrame(rows).sort_values(["flight", "t_start"], kind="mergesort").reset_index(drop=True)
    if cfg.labels_root.exists():
        scored_df = attach_window_labels(
            scored_df=scored_df,
            labels_root=cfg.labels_root,
            time_offset_sec=cfg.failure_label_time_offset_sec,
        )

    scored_df, static_payload, dynamic_payload = apply_threshold_methods(
        scored_df=scored_df,
        alpha=cfg.threshold_smooth_alpha,
        static_p=cfg.static_threshold_p,
        static_label_col=cfg.static_threshold_label_col,
        dynamic_history=cfg.dynamic_threshold_history,
        dynamic_z_values=list(cfg.dynamic_threshold_z_values),
        dynamic_warmup_pred=cfg.dynamic_threshold_warmup_pred,
        dynamic_mad_k=getattr(cfg, "threshold_mad_k", 4.0),
        val_score_path=Path(cfg.run_root) / "val_normal_scores.csv",
        static_val_sigma_k=getattr(cfg, "threshold_sigma_k", 3.0),
    )
    threshold = float(static_payload["best_threshold"])
    static_val_payload = static_payload.get("static_val_sigma3", {})

    sequence_csv_path = infer_root / "sequence_scores.csv"
    all_csv_path = infer_root / "all_failure_usad_scores.csv"
    scored_df.to_csv(sequence_csv_path, index=False, encoding="utf-8")
    scored_df.to_csv(all_csv_path, index=False, encoding="utf-8")
    scored_df.to_csv(analysis_root / "sequence_scores_with_labels.csv", index=False, encoding="utf-8")

    (analysis_root / "threshold_static.json").write_text(
        json.dumps(static_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (analysis_root / "threshold_static_val_sigma3.json").write_text(
        json.dumps(static_val_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (analysis_root / "threshold_dynamic_config.json").write_text(
        json.dumps(dynamic_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_global_threshold(
        {
            "score_col": "scores_smooth",
            "single_global_threshold": threshold,
            "fit_mode": static_payload.get("fit_mode", "static_f1_oracle"),
        },
        analysis_root / "single_global_threshold_total_score.json",
    )

    if {"label_mid", "label_any"}.issubset(scored_df.columns):
        summary_df = summarize_threshold_methods(
            scored_df=scored_df,
            score_col="scores_smooth",
            label_cols=["label_mid", "label_any"],
        )
        summary_df.to_csv(analysis_root / "summary_metrics.csv", index=False, encoding="utf-8")
        per_flight = summarize_per_flight_threshold_methods(
            scored_df=scored_df,
            score_col="scores_smooth",
            label_cols=["label_mid", "label_any"],
        )
        per_flight.to_csv(
            analysis_root / "per_flight_total_score_threshold_methods.csv",
            index=False,
            encoding="utf-8",
        )

    if cfg.plot_scores:
        plot_scores_by_flight(
            scored_df=scored_df,
            out_dir=figures_root,
            labels_root=cfg.labels_root if cfg.labels_root.exists() else None,
            threshold=threshold,
            max_flights=cfg.plot_max_flights,
            time_offset_sec=cfg.failure_label_time_offset_sec,
        )
        if cfg.plot_compare_timelines:
            written = plot_pred_true_resid_score_timelines(
                scored_df=scored_df,
                feature_names=feature_names,
                out_dir=infer_root / "figures",
                model_label="USAD",
                labels_root=cfg.labels_root if cfg.labels_root.exists() else None,
                max_flights=cfg.plot_max_flights,
                time_offset_sec=cfg.failure_label_time_offset_sec,
            )
            if written <= 0:
                print("[WARN] USAD compare timelines were skipped; required per-dim compare columns were missing.")

    (infer_root / "infer_config.json").write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] USAD inference outputs saved to: {infer_root}")


if __name__ == "__main__":
    main()
