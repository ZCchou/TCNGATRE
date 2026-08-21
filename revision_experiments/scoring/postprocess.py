from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from revision_experiments.core.evaluation import evaluate_scores
from revision_experiments.core.provenance import write_json
from revision_experiments.scoring.aggregators import AGGREGATORS, aggregate_dataframe


def run_aggregation_suite(source_run: Path, revision_cfg, legacy_cfg) -> list[dict]:
    source = Path(source_run) / "infer_tcngatre_failure" / "all_failure_window_forecast_residual.csv"
    frame = pd.read_csv(source)
    # Reconstruct channel scores from the two canonical JSON residual columns.
    # Older smoke outputs may contain NumPy's space-separated rendering in
    # sensor_score_vec, which is intentionally ignored here.
    frame = frame.drop(columns=["sensor_score_vec"], errors="ignore")
    rows = []
    for method in AGGREGATORS:
        target_run = Path(source_run) / "ex05_channel_aggregation" / method
        infer_dir = target_run / "infer_tcngatre_failure"
        infer_dir.mkdir(parents=True, exist_ok=True)
        aggregated = aggregate_dataframe(frame, method, revision_cfg.ema_alpha)
        aggregated.to_csv(infer_dir / "all_failure_window_forecast_residual.csv", index=False, encoding="utf-8-sig")
        aggregated[[
            "flight", "current_index", "t_start", "t_end", "t_mid",
            "raw_total_score", "total_score", "valid_dim_count", "aggregation_method",
        ]].to_csv(infer_dir / "sequence_scores.csv", index=False, encoding="utf-8")
        calibration_residuals = pd.read_csv(Path(source_run) / "val_normal_residuals.csv")
        calibration = aggregate_dataframe(calibration_residuals, method, revision_cfg.ema_alpha)
        calibration[["flight", "t_start", "t_end", "t_mid", "raw_total_score", "total_score"]].to_csv(
            target_run / "val_normal_scores.csv", index=False, encoding="utf-8"
        )
        method_cfg = replace(revision_cfg, aggregator=method)
        primary = evaluate_scores(target_run, method_cfg, legacy_cfg)
        rows.append({"aggregation_method": method, **primary, "run_dir": str(target_run)})
    output = Path(source_run) / "ex05_channel_aggregation"
    pd.DataFrame(rows).to_csv(output / "aggregation_comparison.csv", index=False, encoding="utf-8")
    write_json(output / "summary.json", {"methods": len(rows), "source_run": str(source_run)})
    return rows
