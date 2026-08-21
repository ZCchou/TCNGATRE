from __future__ import annotations

import json
import traceback
import importlib.metadata
from pathlib import Path

import pandas as pd

from revision_experiments.core.evaluation import evaluate_scores
from revision_experiments.core.integrity import verify_snapshot
from revision_experiments.core.paths import REPO_ROOT
from revision_experiments.core.provenance import environment_payload, write_json

from .common_data import apply_flightwise_ema


def _dependency_versions() -> dict[str, str]:
    names = [
        "torch", "numpy", "pandas", "scipy", "scikit-learn", "einops",
        "reformer-pytorch", "yacs", "torch-geometric", "minepy",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def finalize_run(
    *,
    cfg,
    legacy_cfg,
    baseline: str,
    source: dict,
    source_commit: str,
    adapter_hash: str,
    resolved_config: dict,
    history: list[dict],
    validation_raw: pd.DataFrame,
    failure_raw: pd.DataFrame,
    extra_provenance: dict | None = None,
) -> dict:
    run_dir = cfg.run_dir
    validation = apply_flightwise_ema(validation_raw, cfg.ema_alpha, f"{baseline}_native")
    failure = apply_flightwise_ema(failure_raw, cfg.ema_alpha, f"{baseline}_native")
    validation[["flight", "t_start", "t_end", "t_mid", "raw_total_score", "total_score"]].to_csv(
        run_dir / "val_normal_scores.csv", index=False, encoding="utf-8"
    )
    infer_dir = run_dir / "infer_tcngatre_failure"
    infer_dir.mkdir(parents=True, exist_ok=True)
    failure.to_csv(infer_dir / "all_failure_window_forecast_residual.csv", index=False, encoding="utf-8-sig")
    failure[[
        "flight", "current_index", "t_start", "t_end", "t_mid", "raw_total_score",
        "total_score", "valid_dim_count", "aggregation_method",
    ]].to_csv(infer_dir / "sequence_scores.csv", index=False, encoding="utf-8")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8")
    write_json(run_dir / "history.json", history)
    write_json(run_dir / "config_resolved.json", resolved_config)
    write_json(run_dir / "provenance.json", {
        **environment_payload(REPO_ROOT),
        "baseline": baseline,
        "official_source": source,
        "official_commit": source_commit,
        "adapter_config_hash": adapter_hash,
        "adapter": "isolated add-only adapter",
        "failure_labels_available_to_training_or_calibration": False,
        "baseline_dependency_versions": _dependency_versions(),
        **(extra_provenance or {}),
    })
    primary = evaluate_scores(run_dir, cfg, legacy_cfg)
    done = {
        "status": "complete",
        "baseline": baseline,
        "config_hash": cfg.config_hash,
        "adapter_config_hash": adapter_hash,
        "source_commit": source_commit,
        "run_dir": str(run_dir),
        "primary_metrics": primary,
        "legacy_integrity": verify_snapshot(),
    }
    write_json(run_dir / "DONE.json", done)
    for stale in ("FAILED.json", "PENDING_ADAPTER.json", "NOT_REPRODUCIBLE.json"):
        path = run_dir / stale
        if path.exists():
            path.unlink()
    return done


def record_failure(run_dir: Path, baseline: str, cfg, exc: Exception) -> None:
    write_json(Path(run_dir) / "FAILED.json", {
        "status": "failed",
        "baseline": baseline,
        "config_hash": cfg.config_hash,
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    })
