from __future__ import annotations

import hashlib
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from tqdm.auto import tqdm

from revision_experiments.core.paths import ensure_import_paths
from revision_experiments.scoring.aggregators import aggregate_dataframe

from .constants import AGGREGATORS, ANALYSIS_SEED, EMA_ALPHA
from .data import FlightArray, inject_events
from .evaluation import write_and_evaluate_real
from .inference import LoadedSourceModel
from .io import environment_payload, sha256_file, stable_seed, write_json
from .parity import compare_scores
from .source import SourceRun, audit_source

ensure_import_paths()

from common.threshold_methods import apply_threshold_methods  # noqa: E402


DESIGN_VERSION = "ex05_local_v1"


def _config_hash(source: SourceRun) -> str:
    payload = {
        "design": DESIGN_VERSION,
        "source_signature": source.source_signature,
        "aggregators": AGGREGATORS,
        "channels": [1, 2, 3],
        "kinds": ["bias", "drift", "freeze", "noise"],
        "severities": [3.0, 5.0],
        "placements": [0.30, 0.50, 0.70],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def scenario_specs(stride: int, horizon: int, smoke: bool = False) -> list[dict[str, Any]]:
    short = max(int(stride), 2 * int(horizon))
    durations = [("short", short), ("long", 4 * short)]
    specs: list[dict[str, Any]] = []
    for channel_count in (1, 2, 3):
        for duration_name, duration in durations:
            for kind in ("bias", "drift", "noise"):
                for severity in (3.0, 5.0):
                    specs.append({
                        "channels": channel_count, "kind": kind, "severity": severity,
                        "duration_name": duration_name, "duration": duration,
                    })
            specs.append({
                "channels": channel_count, "kind": "freeze", "severity": None,
                "duration_name": duration_name, "duration": duration,
            })
    if len(specs) != 42:
        raise AssertionError(f"Expected 42 scenarios, found {len(specs)}")
    return specs[:2] if smoke else specs


def _scenario_id(spec: dict[str, Any]) -> str:
    severity = "na" if spec["severity"] is None else f"{spec['severity']:g}x"
    return (
        f"c{spec['channels']}__{spec['kind']}__{severity}__"
        f"{spec['duration_name']}"
    )


def _window_labels(
    scored: pd.DataFrame,
    labels_by_flight: dict[str, np.ndarray],
    horizon: int,
) -> np.ndarray:
    labels = []
    for row in scored.itertuples(index=False):
        channel_labels = labels_by_flight[str(row.flight)]
        start = int(row.current_index) + 1
        end = min(start + int(horizon), len(channel_labels))
        labels.append(int(channel_labels[start:end].any()))
    return np.asarray(labels, dtype=np.int8)


def _vectors(series: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(json.loads(item), dtype=np.float64) for item in series], axis=0)


def _finite_primary(primary: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in (
        "threshold_method", "label_col", "num_samples", "positives", "negatives",
        "precision", "recall", "f1", "fpr", "auroc", "average_precision",
        "accuracy", "tp", "fp", "tn", "fn", "threshold_mean",
    ):
        if key in primary:
            value = primary[key]
            if not isinstance(value, (int, float, np.number)) or np.isfinite(float(value)):
                output[key] = value
    return output


def _evaluate_synthetic(
    residuals: pd.DataFrame,
    validation_by_method: dict[str, Path],
    labels_by_flight: dict[str, np.ndarray],
    events: list[dict[str, Any]],
    model: LoadedSourceModel,
    scenario_id: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    sensor_scores = _vectors(residuals["sensor_score_vec"])
    residuals = residuals.copy()
    residuals["label_any"] = _window_labels(residuals, labels_by_flight, model.cfg.horizon_out)
    residuals["label_mid"] = residuals["label_any"]
    residuals["_sensor_index"] = np.arange(len(residuals))
    for method in AGGREGATORS:
        aggregated = aggregate_dataframe(residuals, method, EMA_ALPHA)
        evaluated, _, _ = apply_threshold_methods(
            scored_df=aggregated,
            alpha=model.cfg.threshold_smooth_alpha,
            static_p=model.cfg.static_threshold_p,
            static_label_col="label_any",
            dynamic_history=model.cfg.dynamic_threshold_history,
            dynamic_z_values=list(model.cfg.dynamic_threshold_z_values),
            dynamic_warmup_pred=model.cfg.dynamic_threshold_warmup_pred,
            dynamic_mad_k=model.cfg.threshold_mad_k,
            val_score_path=validation_by_method[method],
            static_val_sigma_k=model.cfg.threshold_sigma_k,
        )
        y_true = evaluated["label_any"].to_numpy(dtype=int)
        y_pred = evaluated["pred_spot"].to_numpy(dtype=int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        auprc = (
            float(average_precision_score(y_true, evaluated["scores_smooth"]))
            if len(np.unique(y_true)) > 1 else 0.0
        )
        event_rows = []
        for event in events:
            current = evaluated.loc[evaluated["flight"] == event["flight"]]
            overlap = current.loc[
                (current["t_end"] >= float(event["t_start"]))
                & (current["t_start"] <= float(event["t_end"]))
            ]
            detected = overlap.loc[overlap["pred_spot"] > 0]
            first = float(detected["t_start"].min()) if not detected.empty else float("nan")
            indexes = overlap["_sensor_index"].to_numpy(dtype=int)
            affected = set(int(value) for value in event["channels"])
            hit = float("nan")
            if len(indexes):
                mean_channel = sensor_scores[indexes].mean(axis=0)
                k = min(len(affected), len(mean_channel))
                top = set(np.argsort(-mean_channel)[:k].astype(int).tolist())
                hit = len(top & affected) / max(len(affected), 1)
            event_rows.append({
                "detected": bool(not detected.empty),
                "delay": max(first - float(event["t_start"]), 0.0) if np.isfinite(first) else float("nan"),
                "hit_at_k": hit,
            })
        detected_events = [row for row in event_rows if row["detected"]]
        output.append({
            "scenario_id": scenario_id,
            "aggregation_method": method,
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "average_precision": auprc,
            "event_count": len(event_rows),
            "event_recall": float(len(detected_events) / max(len(event_rows), 1)),
            "event_miss_rate": float(1.0 - len(detected_events) / max(len(event_rows), 1)),
            "mean_detection_delay": float(np.mean([
                row["delay"] if row["detected"] else max(
                    float(events[index]["t_end"]) - float(events[index]["t_start"]), 0.0
                )
                for index, row in enumerate(event_rows)
            ])),
            "channel_hit_at_k": float(np.nan_to_num(
                np.nanmean([row["hit_at_k"] for row in event_rows]), nan=0.0
            )),
        })
    return output


def run_ex05(
    source: SourceRun,
    output_root: Path,
    *,
    force: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    run_dir = Path(output_root) / "ex05" / source.dataset / f"seed_{source.seed}"
    done_path = run_dir / "DONE.json"
    config_hash = _config_hash(source) + ("_smoke" if smoke else "")
    if done_path.is_file() and not force:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("config_hash") == config_hash and done.get("source_signature") == source.source_signature:
            return {"status": "skipped_complete", **done}
    run_dir.mkdir(parents=True, exist_ok=True)
    if done_path.exists():
        done_path.unlink()
    checkpoint_hash_before = sha256_file(source.checkpoint)
    provenance = {
        "experiment": "ex05", "design_version": DESIGN_VERSION,
        "dataset": source.dataset, "model_seed": source.seed,
        "config_hash": config_hash, "source": source.to_dict(),
        "source_audit": audit_source(source), "environment": environment_payload(),
    }
    write_json(run_dir / "provenance.json", provenance)
    write_json(run_dir / "resolved_analysis_config.json", {
        "aggregators": AGGREGATORS,
        "scenario_count": len(scenario_specs(1, 1, smoke=smoke)) if smoke else 42,
        "channels": [1, 2, 3], "kinds": ["bias", "drift", "freeze", "noise"],
        "severities": [3.0, 5.0], "placements": [0.30, 0.50, 0.70],
        "analysis_seed": ANALYSIS_SEED, "requires_training": False,
    })
    try:
        loaded = LoadedSourceModel.load(source)
        validation_residuals, _ = loaded.score(
            loaded.validation_flights, description=f"EX-05 {source.dataset} seed={source.seed} validation"
        )
        validation_residuals.to_csv(run_dir / "val_channel_scores.csv.gz", index=False, compression="gzip")
        failure_residuals = pd.read_csv(source.failure_residuals)

        validation_paths: dict[str, Path] = {}
        real_rows = []
        for method in AGGREGATORS:
            method_dir = run_dir / "real_failure" / method
            primary, _ = write_and_evaluate_real(
                method_dir, loaded, validation_residuals, failure_residuals, method
            )
            val_path = method_dir / "val_normal_scores.csv"
            validation_paths[method] = val_path
            real_rows.append({
                "dataset": source.dataset, "seed": source.seed,
                "aggregation_method": method, **_finite_primary(primary),
            })
        real_frame = pd.DataFrame(real_rows)
        real_frame.to_csv(run_dir / "real_failure_aggregation.csv", index=False, encoding="utf-8-sig")

        original_sequence = pd.read_csv(source.sequence_scores)
        mean_sequence = pd.read_csv(run_dir / "real_failure" / "mean" / "infer_tcngatre_failure" / "sequence_scores.csv")
        parity = original_sequence.merge(
            mean_sequence, on=["flight", "current_index"], suffixes=("_source", "_posthoc"), validate="one_to_one"
        )
        if len(parity) != len(original_sequence):
            raise RuntimeError("Mean aggregation parity merge is incomplete")
        parity_result = compare_scores(
            parity["total_score_source"].to_numpy(),
            parity["total_score_posthoc"].to_numpy(),
            atol=1e-6,
        )
        parity_error = parity_result.max_abs_error
        if not parity_result.passed:
            raise RuntimeError(
                "Mean aggregation parity failed: "
                f"max_abs={parity_error}, max_rel={parity_result.max_rel_error}"
            )

        specs = scenario_specs(loaded.cfg.sample_stride, loaded.cfg.horizon_out, smoke=smoke)
        manifest_rows: list[dict[str, Any]] = []
        synthetic_rows: list[dict[str, Any]] = []
        for spec in tqdm(specs, desc=f"EX-05 {source.dataset} seed={source.seed} scenarios", unit="scenario"):
            scenario_id = _scenario_id(spec)
            synthetic: dict[str, FlightArray] = {}
            labels_by_flight: dict[str, np.ndarray] = {}
            events: list[dict[str, Any]] = []
            for flight, item in loaded.validation_flights.items():
                seed = stable_seed(ANALYSIS_SEED, source.dataset, scenario_id, flight)
                changed, labels, current_events = inject_events(
                    item,
                    channels=int(spec["channels"]),
                    kind=str(spec["kind"]),
                    severity=spec["severity"],
                    duration=int(spec["duration"]),
                    robust_scale=loaded.channel_statistics.robust_scale,
                    scenario_seed=seed,
                )
                synthetic_name = f"synthetic__{flight}__{scenario_id}"
                synthetic[synthetic_name] = FlightArray(synthetic_name, changed.time, changed.values)
                labels_by_flight[synthetic_name] = labels
                for event in current_events:
                    events.append({"flight": synthetic_name, "source_flight": flight, **event})
                manifest_rows.append({
                    "scenario_id": scenario_id, "source_flight": flight, "seed": int(seed),
                    **spec, "events": current_events,
                })
            residuals, _ = loaded.score(
                synthetic, description=f"EX-05 {source.dataset} s{source.seed} {scenario_id}"
            )
            synthetic_rows.extend(
                _evaluate_synthetic(
                    residuals, validation_paths, labels_by_flight, events, loaded, scenario_id
                )
            )
        synthetic_frame = pd.DataFrame(synthetic_rows)
        synthetic_frame.to_csv(run_dir / "synthetic_event_metrics.csv", index=False, encoding="utf-8-sig")
        write_json(run_dir / "scenario_manifest.json", {
            "design_version": DESIGN_VERSION,
            "scenario_count": len(specs),
            "placements_per_flight": 3,
            "scenarios": manifest_rows,
        })
        done = {
            "status": "complete", "experiment": "ex05", "dataset": source.dataset,
            "seed": source.seed, "config_hash": config_hash,
            "source_signature": source.source_signature,
            "source_checkpoint_sha256": source.checkpoint_sha256,
            "mean_parity_max_abs_error": parity_error,
            "mean_parity_max_rel_error": parity_result.max_rel_error,
            "aggregators": len(AGGREGATORS), "synthetic_scenarios": len(specs),
            "environment": environment_payload(),
        }
        checkpoint_hash_after = sha256_file(source.checkpoint)
        if checkpoint_hash_after != checkpoint_hash_before:
            raise RuntimeError(f"Source checkpoint changed during EX-05: {source.checkpoint}")
        write_json(done_path, done)
        failed = run_dir / "FAILED.json"
        if failed.exists():
            failed.unlink()
        return done
    except Exception as exc:
        write_json(run_dir / "FAILED.json", {
            "status": "failed", "experiment": "ex05", "dataset": source.dataset,
            "seed": source.seed, "config_hash": config_hash, "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        raise
