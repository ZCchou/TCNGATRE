from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .paths import ensure_import_paths

ensure_import_paths()

from common.threshold_methods import (  # noqa: E402
    apply_threshold_methods,
    summarize_per_flight_threshold_methods,
    summarize_threshold_methods,
)
from data.window_labels import attach_window_labels  # noqa: E402


def evaluate_scores(run_dir: Path, cfg, legacy_cfg) -> dict:
    infer_dir = run_dir / "infer_tcngatre_failure"
    analysis_dir = infer_dir / "score_threshold_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    score_path = infer_dir / "sequence_scores.csv"
    scored = pd.read_csv(score_path)
    scored = attach_window_labels(
        scored_df=scored,
        labels_root=legacy_cfg.labels_root,
        time_offset_sec=legacy_cfg.failure_label_time_offset_sec,
    )
    scored = scored.loc[pd.to_numeric(scored["total_score"], errors="coerce").notna()].copy()
    scored, static_payload, dynamic_payload = apply_threshold_methods(
        scored_df=scored,
        alpha=legacy_cfg.threshold_smooth_alpha,
        static_p=legacy_cfg.static_threshold_p,
        static_label_col="label_any",
        dynamic_history=legacy_cfg.dynamic_threshold_history,
        dynamic_z_values=list(legacy_cfg.dynamic_threshold_z_values),
        dynamic_warmup_pred=legacy_cfg.dynamic_threshold_warmup_pred,
        dynamic_mad_k=legacy_cfg.threshold_mad_k,
        val_score_path=run_dir / "val_normal_scores.csv",
        static_val_sigma_k=legacy_cfg.threshold_sigma_k,
    )
    scored.to_csv(analysis_dir / "sequence_scores_with_labels.csv", index=False, encoding="utf-8")
    summary = summarize_threshold_methods(
        scored_df=scored,
        label_cols=["label_mid", "label_any"],
        score_col="scores_smooth",
    )
    per_flight = summarize_per_flight_threshold_methods(
        scored_df=scored,
        label_cols=["label_mid", "label_any"],
        score_col="scores_smooth",
    )
    summary.to_csv(analysis_dir / "summary_metrics.csv", index=False, encoding="utf-8")
    per_flight.to_csv(
        analysis_dir / "per_flight_total_score_threshold_methods.csv", index=False, encoding="utf-8"
    )
    (analysis_dir / "threshold_static.json").write_text(
        json.dumps(static_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (analysis_dir / "threshold_dynamic_config.json").write_text(
        json.dumps(dynamic_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    primary_rows = summary.loc[
        (summary["threshold_method"] == cfg.threshold_method)
        & (summary["label_col"] == "label_any")
    ]
    if primary_rows.empty:
        raise RuntimeError("Primary SPOT/label_any result is missing")
    primary = primary_rows.iloc[0].to_dict()
    primary["policy"] = "flightwise causal EMA + SPOT; no failure labels in calibration"
    primary["diagnostic_oracle_excluded"] = True
    (analysis_dir / "primary_metrics.json").write_text(
        json.dumps(primary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return primary
