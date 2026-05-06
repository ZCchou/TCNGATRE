from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

from common.threshold_methods import (  # noqa: E402
    apply_threshold_methods,
    summarize_per_flight_threshold_methods,
    summarize_threshold_methods,
)
from data.usad_window_dataset import add_anomaly_background, attach_window_labels, load_failure_labels
from tranadconfig import TranADConfig
from utils.global_threshold import fit_single_global_threshold, save_global_threshold
from utils.io import ensure_dir
from utils.metrics import point_adjust_predictions, summarize_per_flight, summarize_threshold_metrics
from utils.pot import fit_threshold, save_threshold


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Evaluate TranAD on a bundled dataset.")
    parser.add_argument("--dataset", choices=["alfa", "alfa4hz", "simulate", "gpsdata"], default="alfa")
    return parser.parse_args(argv)


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
        score_col = "scores_smooth" if "scores_smooth" in g.columns else "total_score"
        s = g[score_col].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(14, 4.8))
        ax.plot(t, s, color="tab:blue", linewidth=1.5, label=score_col)
        if threshold is not None and np.isfinite(float(threshold)):
            ax.axhline(float(threshold), color="tab:orange", linestyle="--", linewidth=1.2, label="threshold")
        add_anomaly_background(ax, labels_df=labels_df, t_min=float(np.min(t)), t_max=float(np.max(t)))
        ax.set_title(f"{flight} | TranAD {score_col}")
        ax.set_xlabel("t (sec)")
        ax.set_ylabel("score")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{flight}__tranad_total_score_threshold.png", dpi=140)
        plt.close(fig)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    cfg = TranADConfig(dataset_name=args.dataset)
    run_root = cfg.run_root
    infer_root = run_root / cfg.infer_output_name
    analysis_root = infer_root / "score_threshold_analysis"
    figures_root = analysis_root / "flight_threshold_timelines"
    ensure_dir(analysis_root)
    ensure_dir(figures_root)

    failure_scores_path = infer_root / "sequence_scores.csv"
    if not failure_scores_path.exists():
        raise FileNotFoundError(f"Missing failure score file: {failure_scores_path}")

    failure_df = pd.read_csv(failure_scores_path)

    scored_df = (
        attach_window_labels(
            scored_df=failure_df,
            labels_root=cfg.labels_root,
            time_offset_sec=cfg.failure_label_time_offset_sec,
        )
        if cfg.labels_root.exists()
        else failure_df.copy()
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
    per_flight.to_csv(analysis_root / "per_flight_total_score_threshold_methods.csv", index=False, encoding="utf-8")

    if cfg.plot_scores:
        plot_scores_by_flight(
            scored_df=scored_df,
            out_dir=figures_root,
            labels_root=cfg.labels_root if cfg.labels_root.exists() else None,
            threshold=threshold,
            max_flights=cfg.plot_max_flights,
            time_offset_sec=cfg.failure_label_time_offset_sec,
        )

    (analysis_root / "eval_config.json").write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] TranAD evaluation saved to: {analysis_root}")


if __name__ == "__main__":
    main()
