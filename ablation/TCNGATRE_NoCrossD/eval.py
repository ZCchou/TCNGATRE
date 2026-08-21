from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_MODEL_ROOT = Path(__file__).resolve().parent
_ABLATION_ROOT = _MODEL_ROOT.parent
_BASE_ROOT = _ABLATION_ROOT / "base"
_PROJECT_ROOT = _ABLATION_ROOT.parent
for _p in [str(_PROJECT_ROOT), str(_BASE_ROOT), str(_MODEL_ROOT)]:
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import matplotlib.pyplot as plt
import pandas as pd

from base_config import TCNGATREConfig
from common.threshold_methods import (
    apply_threshold_methods,
    summarize_per_flight_threshold_methods,
    summarize_threshold_methods,
)
from data.window_labels import add_anomaly_background, attach_window_labels, load_failure_labels
from utils.global_threshold import save_global_threshold
from utils.io import ensure_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate ablation model results.")
    parser.add_argument("--dataset", choices=["alfa", "simulate", "gpsdata"], default="alfa")
    return parser.parse_args(argv)


def _set_run_root(dataset: str) -> None:
    import os
    from runtime import MODEL_ROOT, RUN_PREFIX
    os.environ.setdefault("UAV_TCNGATRE_RUN_ROOT", str(MODEL_ROOT / "runs" / f"{RUN_PREFIX}_{dataset}"))


def plot_scores_by_flight(
    scored_df: pd.DataFrame,
    out_dir: Path,
    labels_root: Path | None,
    threshold: float | None,
    max_flights: int,
    time_offset_sec: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    flights = list(scored_df["flight"].drop_duplicates())
    if int(max_flights) > 0:
        flights = flights[:int(max_flights)]
    for flight in flights:
        g = scored_df.loc[scored_df["flight"] == flight].sort_values("t_start", kind="mergesort")
        if len(g) == 0:
            continue
        labels_df = (
            load_failure_labels(labels_root, str(flight), time_offset_sec=time_offset_sec)
            if labels_root is not None else None
        )
        t = g["t_mid"].to_numpy(dtype=float)
        score_col = "scores_smooth" if "scores_smooth" in g.columns else "total_score"
        s = g[score_col].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(14, 4.8))
        ax.plot(t, s, color="tab:blue", linewidth=1.5, label=score_col)
        if threshold is not None:
            ax.axhline(float(threshold), color="tab:orange", linestyle="--", linewidth=1.2, label="threshold")
        add_anomaly_background(ax, labels_df=labels_df, t_min=float(min(t)), t_max=float(max(t)))
        ax.set_title(f"{flight} | TCNGATRE ablation {score_col}")
        ax.set_xlabel("t (sec)")
        ax.set_ylabel("score")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{flight}__score_threshold.png", dpi=140)
        plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    _set_run_root(args.dataset)
    cfg = TCNGATREConfig(dataset_name=args.dataset)

    run_root = Path(cfg.run_root)
    infer_root = run_root / cfg.infer_output_name
    analysis_root = infer_root / "score_threshold_analysis"
    figures_root = analysis_root / "flight_threshold_timelines"
    ensure_dir(analysis_root)
    ensure_dir(figures_root)

    seq_path = infer_root / "sequence_scores.csv"
    if not seq_path.exists():
        legacy = infer_root / f"all_{cfg.infer_source_split.lower()}_window_forecast_residual.csv"
        if legacy.exists():
            seq_path = legacy
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing score file: {seq_path}")

    scored_df = pd.read_csv(seq_path)
    if "raw_total_score" not in scored_df.columns and "total_score" in scored_df.columns:
        scored_df["raw_total_score"] = pd.to_numeric(scored_df["total_score"], errors="coerce")
    if "total_score" not in scored_df.columns and "raw_total_score" in scored_df.columns:
        scored_df["total_score"] = pd.to_numeric(scored_df["raw_total_score"], errors="coerce")

    if Path(cfg.labels_root).exists():
        scored_df = attach_window_labels(
            scored_df=scored_df,
            labels_root=cfg.labels_root,
            time_offset_sec=cfg.failure_label_time_offset_sec,
        )

    score_valid = pd.to_numeric(scored_df["total_score"], errors="coerce").notna()
    scored_df = scored_df.loc[score_valid].copy().reset_index(drop=True)
    if scored_df.empty:
        raise ValueError("No valid score rows for threshold analysis.")

    scored_df, static_payload, dynamic_payload = apply_threshold_methods(
        scored_df=scored_df,
        alpha=cfg.threshold_smooth_alpha,
        static_p=cfg.static_threshold_p,
        static_label_col=cfg.static_threshold_label_col,
        dynamic_history=cfg.dynamic_threshold_history,
        dynamic_z_values=list(cfg.dynamic_threshold_z_values),
        dynamic_warmup_pred=cfg.dynamic_threshold_warmup_pred,
        dynamic_mad_k=cfg.threshold_mad_k,
        val_score_path=run_root / "val_normal_scores.csv",
        static_val_sigma_k=cfg.threshold_sigma_k,
    )
    threshold = float(static_payload["best_threshold"])

    (analysis_root / "threshold_static.json").write_text(
        json.dumps(static_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (analysis_root / "threshold_static_val_sigma3.json").write_text(
        json.dumps(static_payload.get("static_val_sigma3", {}), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (analysis_root / "threshold_dynamic_config.json").write_text(
        json.dumps(dynamic_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_global_threshold(
        {
            "score_col": "scores_smooth",
            "single_global_threshold": threshold,
            "fit_mode": static_payload.get("fit_mode", "static_f1_oracle"),
        },
        analysis_root / "single_global_threshold_total_score.json",
    )

    scored_df.to_csv(analysis_root / "sequence_scores_with_labels.csv", index=False, encoding="utf-8")

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
        index=False, encoding="utf-8",
    )

    if bool(cfg.plot_scores):
        plot_scores_by_flight(
            scored_df=scored_df,
            out_dir=figures_root,
            labels_root=Path(cfg.labels_root) if Path(cfg.labels_root).exists() else None,
            threshold=threshold,
            max_flights=cfg.plot_max_flights,
            time_offset_sec=cfg.failure_label_time_offset_sec,
        )

    (analysis_root / "eval_config.json").write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] evaluation saved to: {analysis_root}")


if __name__ == "__main__":
    main()
