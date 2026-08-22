from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revision_experiments.analysis.statistics import (  # noqa: E402
    hierarchical_paired_bootstrap,
    holm_adjust,
    paired_sign_permutation,
    rank_biserial,
)
from revision_experiments.core.config import make_config  # noqa: E402
from revision_experiments.core.engine import data_protocol_payload  # noqa: E402
from revision_experiments.summarize_main_comparison import (  # noqa: E402
    _tcngatre_data_protocol_signature,
)


BASELINES = ("mstgcnet", "tsae_uav")
METRICS = ("precision", "recall", "f1", "fpr", "auroc", "average_precision")
ANALYSIS_RELATIVE = Path("infer_tcngatre_failure") / "score_threshold_analysis"
DEFAULT_RESULTS = REPO_ROOT / "revision_results" / "protocol_v1"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _canonical_split(payload: dict) -> dict:
    def values(*keys: str) -> list[str]:
        for key in keys:
            if key in payload:
                return sorted(str(value) for value in payload[key])
        return []

    return {
        "train": values("train_flights"),
        "validation": values("validation_flights"),
        "failure": values("failure_flights", "failure_flights_scored_only"),
    }


def _split_hash(split: dict) -> str:
    raw = json.dumps(split, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _primary_row(path: Path) -> dict:
    frame = pd.read_csv(path)
    required = {"threshold_method", "label_col", *METRICS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing primary metric columns in {path}: {missing}")
    selected = frame.loc[
        (frame["threshold_method"].astype(str).str.lower() == "spot")
        & (frame["label_col"].astype(str) == "label_any")
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one SPOT + label_any row in {path}; found {len(selected)}")
    row = selected.iloc[0].to_dict()
    for metric in METRICS:
        value = float(row[metric])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {metric} in {path}")
        row[metric] = value
    return row


def _per_flight(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"flight", "threshold_method", "label_col", "f1", "average_precision"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing per-flight columns in {path}: {missing}")
    selected = frame.loc[
        (frame["threshold_method"].astype(str).str.lower() == "spot")
        & (frame["label_col"].astype(str) == "label_any")
    ].copy()
    if selected["flight"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate SPOT + label_any flights in {path}")
    for metric in ("f1", "average_precision"):
        selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
        if not np.isfinite(selected[metric].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite per-flight {metric} in {path}")
    return selected


def _read_baseline_run(results_root: Path, dataset: str, baseline: str, seed: int) -> tuple[dict, pd.DataFrame]:
    run_dir = results_root / "ex03" / dataset / baseline / f"seed_{seed}"
    done = json.loads((run_dir / "DONE.json").read_text(encoding="utf-8"))
    if done.get("status") != "complete":
        raise ValueError(f"Incomplete DONE marker: {run_dir}")
    expected_cfg = make_config("ex03", dataset, baseline, seed)
    if done.get("config_hash") != expected_cfg.config_hash:
        raise ValueError(f"EX-03 formal configuration hash mismatch: {run_dir}")
    expected_classification = {
        "mstgcnet": "released_scaffold_engineering_reimplementation",
        "tsae_uav": "paper_based_protocol_compatible_reimplementation",
    }[baseline]
    if done.get("reproduction_classification") != expected_classification:
        raise ValueError(f"EX-03 reproduction classification mismatch: {run_dir}")
    expected_protocol = data_protocol_payload(expected_cfg.to_legacy())
    if done.get("data_protocol_hash") != expected_protocol["data_protocol_hash"]:
        raise ValueError(f"EX-03 data protocol hash mismatch: {run_dir}")
    split = _canonical_split(expected_protocol)
    if tuple(map(len, (split["train"], split["validation"], split["failure"]))) != (29, 1, 16):
        raise ValueError(f"EX-03 ALFA split is not 29/1/16: {run_dir}")
    metric_path = run_dir / ANALYSIS_RELATIVE / "summary_metrics.csv"
    flight_path = run_dir / ANALYSIS_RELATIVE / "per_flight_total_score_threshold_methods.csv"
    primary = _primary_row(metric_path)
    primary.update({
        "run_id": f"{dataset}/{baseline}/seed_{seed}", "dataset": dataset,
        "model": baseline, "seed": seed, "aggregation": "micro_over_all_windows",
        "protocol_split_hash": _split_hash(split), "source": str(metric_path),
    })
    flights = _per_flight(flight_path)
    flights.insert(0, "seed", seed)
    flights.insert(0, "model", baseline)
    flights.insert(0, "dataset", dataset)
    return primary, flights


def _read_tcngatre_run(root: Path, dataset: str, seed: int, expected_split: dict) -> tuple[dict, pd.DataFrame]:
    run_dir = root / f"seed_{seed}"
    done = json.loads((run_dir / "DONE.json").read_text(encoding="utf-8"))
    if done.get("status") != "complete":
        raise ValueError(f"Incomplete TCNGATRE reference: {run_dir}")
    expected_signature = _tcngatre_data_protocol_signature(dataset)
    if done.get("data_protocol_signature") != expected_signature:
        raise ValueError(f"TCNGATRE reference data protocol signature mismatch: {run_dir}")
    if int(done.get("sample_stride", -1)) != 16:
        raise ValueError(f"TCNGATRE ALFA reference stride is not 16: {run_dir}")
    if int(done.get("batch_size", -1)) != 128:
        raise ValueError(f"TCNGATRE ALFA reference batch size is not 128: {run_dir}")
    split_path = run_dir / "split_flights.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"TCNGATRE reference split metadata missing: {split_path}")
    actual_split = _canonical_split(json.loads(split_path.read_text(encoding="utf-8")))
    if actual_split != expected_split:
        raise ValueError(f"TCNGATRE reference is not the same ALFA 29/1/16 split: {run_dir}")
    metric_path = run_dir / ANALYSIS_RELATIVE / "summary_metrics.csv"
    flight_path = run_dir / ANALYSIS_RELATIVE / "per_flight_total_score_threshold_methods.csv"
    primary = _primary_row(metric_path)
    primary.update({
        "run_id": f"{dataset}/TCNGATRE/seed_{seed}", "dataset": dataset,
        "model": "TCNGATRE", "seed": seed, "aggregation": "micro_over_all_windows",
        "protocol_split_hash": _split_hash(actual_split), "source": str(metric_path),
    })
    flights = _per_flight(flight_path)
    flights.insert(0, "seed", seed)
    flights.insert(0, "model", "TCNGATRE")
    flights.insert(0, "dataset", dataset)
    return primary, flights


def _seed_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (dataset, model), group in frame.groupby(["dataset", "model"], sort=True):
        row: dict[str, Any] = {
            "dataset": dataset, "model": model, "seed_count": int(group["seed"].nunique()),
            "seeds": ",".join(str(int(value)) for value in sorted(group["seed"].unique())),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _significance(per_flight: pd.DataFrame, n_resamples: int) -> pd.DataFrame:
    rows: list[dict] = []
    reference = per_flight.loc[per_flight["model"] == "TCNGATRE"]
    for model in BASELINES:
        candidate = per_flight.loc[per_flight["model"] == model]
        for metric in ("f1", "average_precision"):
            pairs = candidate[["dataset", "flight", "seed", metric]].merge(
                reference[["dataset", "flight", "seed", metric]],
                on=["dataset", "flight", "seed"], suffixes=("_baseline", "_tcngatre"),
                how="inner", validate="one_to_one",
            )
            expected_pairs = int(candidate[["dataset", "flight", "seed"]].drop_duplicates().shape[0])
            if len(pairs) != expected_pairs:
                raise ValueError(
                    f"Incomplete paired observations for {model}/{metric}: "
                    f"expected={expected_pairs}, actual={len(pairs)}"
                )
            bootstrap = hierarchical_paired_bootstrap(
                pairs, f"{metric}_baseline", f"{metric}_tcngatre",
                n_resamples=n_resamples,
            )
            flight_mean = pairs.assign(
                difference=pairs[f"{metric}_baseline"] - pairs[f"{metric}_tcngatre"]
            ).groupby("flight", sort=False)["difference"].mean().to_numpy(dtype=float)
            permutation = paired_sign_permutation(flight_mean, n_resamples=n_resamples)
            rows.append({
                "dataset": "alfa", "baseline": model, "reference": "TCNGATRE",
                "metric": metric, "pairing": "same flight and model seed",
                "n_pairs": len(pairs), "n_flights": int(pairs["flight"].nunique()),
                "mean_difference_baseline_minus_tcngatre": bootstrap["mean_difference"],
                "ci95_low": bootstrap["ci95_low"], "ci95_high": bootstrap["ci95_high"],
                "p_value": permutation["p_value"], "permutation_exact": permutation["exact"],
                "rank_biserial": rank_biserial(flight_mean),
            })
    adjusted = holm_adjust([float(row["p_value"]) for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = value
        row["significant_0_05"] = bool(value < 0.05)
    return pd.DataFrame(rows)


def summarize(
    datasets: list[str], seeds: list[int], tcngatre_root: Path,
    output_root: Path, n_resamples: int, results_root: Path = DEFAULT_RESULTS,
) -> dict:
    if datasets != ["alfa"]:
        raise ValueError("TSAE-UAV and MSTGCNet protocol-compatible reproductions currently support --datasets alfa only")
    results_root = Path(results_root)
    output_root.mkdir(parents=True, exist_ok=True)
    expected_protocol = data_protocol_payload(make_config("ex03", "alfa", BASELINES[0], seeds[0]).to_legacy())
    expected_split = _canonical_split(expected_protocol)
    statuses: list[dict] = []
    primary_rows: list[dict] = []
    flight_rows: list[pd.DataFrame] = []
    missing: list[dict] = []
    for model in (*BASELINES, "TCNGATRE"):
        for seed in seeds:
            run_id = f"alfa/{model}/seed_{seed}"
            try:
                if model == "TCNGATRE":
                    primary, flights = _read_tcngatre_run(tcngatre_root, "alfa", seed, expected_split)
                else:
                    primary, flights = _read_baseline_run(results_root, "alfa", model, seed)
                statuses.append({"run_id": run_id, "dataset": "alfa", "model": model, "seed": seed, "status": "complete", "reason": ""})
                primary_rows.append(primary)
                flight_rows.append(flights)
            except Exception as exc:
                reason = repr(exc)
                statuses.append({"run_id": run_id, "dataset": "alfa", "model": model, "seed": seed, "status": "invalid_or_missing", "reason": reason})
                missing.append({"run_id": run_id, "dataset": "alfa", "model": model, "seed": seed, "reason": reason})
    status_frame = pd.DataFrame(statuses)
    primary = pd.DataFrame(primary_rows)
    per_flight = pd.concat(flight_rows, ignore_index=True) if flight_rows else pd.DataFrame()
    summary = _seed_summary(primary) if not primary.empty else pd.DataFrame()
    complete = not missing and len(primary_rows) == 3 * len(seeds)
    significance = _significance(per_flight, n_resamples) if complete else pd.DataFrame()
    status_frame.to_csv(output_root / "run_status.csv", index=False, encoding="utf-8-sig")
    primary.to_csv(output_root / "primary_metrics_all_runs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_root / "primary_metrics_seed_summary.csv", index=False, encoding="utf-8-sig")
    per_flight.to_csv(output_root / "per_flight_primary_all_runs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(missing, columns=["run_id", "dataset", "model", "seed", "reason"]).to_csv(
        output_root / "missing_experiment_cells.csv", index=False, encoding="utf-8-sig"
    )
    significance.to_csv(output_root / "paired_significance.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "complete" if complete else "incomplete",
        "expected_runs": 3 * len(seeds), "complete_runs": len(primary_rows),
        "missing_runs": len(missing), "datasets": datasets, "seeds": seeds,
        "reference": "original main_comparison/alfa/TCNGATRE (never ex01/full)",
        "primary_protocol": "label_any + causal EMA + flightwise SPOT + Micro",
        "reproduction_labels": {
            "tsae_uav": "paper_based_protocol_compatible_reimplementation",
            "mstgcnet": "released_scaffold_engineering_reimplementation",
        },
        "canonical_split": {**expected_split, "hash": _split_hash(expected_split)},
        "output_root": str(output_root),
    }
    _write_json(output_root / "summary.json", payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine reviewer-requested UAV reproductions with the original TCNGATRE main comparison."
    )
    parser.add_argument("--datasets", nargs="+", default=["alfa"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--tcngatre-root",
        default=str(DEFAULT_RESULTS / "main_comparison" / "alfa" / "TCNGATRE"),
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_RESULTS / "summary" / "reviewer_uav_baselines"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)) or any(seed < 0 for seed in args.seeds):
        raise SystemExit("Seeds must be unique non-negative integers")
    report = summarize(
        datasets=list(args.datasets), seeds=list(args.seeds),
        tcngatre_root=Path(args.tcngatre_root).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        n_resamples=int(args.bootstrap_resamples),
        results_root=Path(args.results_root).expanduser().resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 2 if args.require_complete and report["status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
