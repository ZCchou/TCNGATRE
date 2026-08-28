from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from revision_experiments.core.engine import data_protocol_payload
from revision_experiments.core.provenance import write_json


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(path),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


DATASET_PROTOCOLS = {
    "alfa": {"train": 29, "validation": 1, "failure": 16, "channels": 12},
    "gpsdata": {"train": 1, "validation": 1, "failure": 2, "channels": 45},
    "simulate": {"train": 8, "validation": 2, "failure": 2, "channels": 7},
}


def accumulation_groups(loader, steps: int):
    """Yield physical batches grouped for one optimizer update.

    The returned sample count lets callers weight a short final group by its
    actual number of samples instead of assuming that every group is full.
    """
    iterator = iter(loader)
    while True:
        batches = list(itertools.islice(iterator, max(int(steps), 1)))
        if not batches:
            return
        yield batches, sum(int(batch.shape[0]) for batch in batches)


def validate_dataset_protocol(bundle) -> dict:
    if bundle.dataset not in DATASET_PROTOCOLS:
        raise ValueError(f"Unsupported reviewer-baseline dataset: {bundle.dataset}")
    specification = DATASET_PROTOCOLS[bundle.dataset]
    counts = {name: len(bundle.splits[name]) for name in ("train", "validation", "failure")}
    expected = {name: int(specification[name]) for name in ("train", "validation", "failure")}
    if counts != expected:
        raise RuntimeError(
            f"{bundle.dataset} common-data protocol mismatch: expected={expected}, actual={counts}"
        )
    if len(bundle.nodes) != int(specification["channels"]):
        raise RuntimeError(
            f"{bundle.dataset} common-data channel mismatch: "
            f"expected={specification['channels']}, actual={len(bundle.nodes)}"
        )
    train = {row.flight for row in bundle.splits["train"]}
    validation = {row.flight for row in bundle.splits["validation"]}
    failure = {row.flight for row in bundle.splits["failure"]}
    overlaps = {
        "train_validation": sorted(train & validation),
        "train_failure": sorted(train & failure),
        "validation_failure": sorted(validation & failure),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"{bundle.dataset} common-data split overlap: {overlaps}")

    array_rows = 0
    for split in ("train", "validation", "failure"):
        for record in bundle.splits[split]:
            time, values = bundle.load(record)
            if values.shape != (record.rows, len(bundle.nodes)):
                raise RuntimeError(
                    f"{bundle.dataset}/{split}/{record.flight} array shape mismatch: {values.shape}"
                )
            if not np.isfinite(time).all() or not np.isfinite(values).all():
                raise RuntimeError(
                    f"{bundle.dataset}/{split}/{record.flight} contains NaN or Inf"
                )
            if len(time) > 1 and np.any(np.diff(time) < 0):
                raise RuntimeError(
                    f"{bundle.dataset}/{split}/{record.flight} time is not monotonic"
                )
            array_rows += len(time)

    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    graph_profile = manifest.get("graph_profile", {})
    graph_flights = sorted(str(value) for value in graph_profile.get("include_flights") or [])
    if graph_flights != sorted(train):
        raise RuntimeError(
            f"{bundle.dataset} MIC provenance is not train-only: "
            f"expected={sorted(train)}, actual={graph_flights}"
        )
    if manifest.get("labels_exported") is not False:
        raise RuntimeError(f"{bundle.dataset} common export must remain label-free")
    return {
        "dataset": bundle.dataset,
        "counts": counts,
        "channels": len(bundle.nodes),
        "array_rows_checked": int(array_rows),
        "graph_flights": graph_flights,
        "split_overlap": overlaps,
    }


def validate_failure_labels(cfg, bundle) -> dict:
    """Validate source labels without exporting them into common baseline data."""
    labels_root = Path(cfg.to_legacy().labels_root)
    rows = []
    for record in bundle.splits["failure"]:
        label_path = labels_root / f"{record.flight}.csv"
        if not label_path.is_file():
            raise FileNotFoundError(f"Failure label file is missing: {label_path}")
        labels = pd.read_csv(label_path)
        label_col = "anomaly_label" if "anomaly_label" in labels else None
        if label_col is None:
            raise RuntimeError(f"Failure label column is missing: {label_path}")
        numeric = pd.to_numeric(labels[label_col], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise RuntimeError(f"Failure labels are not finite: {label_path}")
        label_values = numeric.to_numpy(dtype=float)
        positives = int((label_values > 0.5).sum())
        alignment = "row_exact"
        positive_label_rows_in_data_time_range = positives
        if len(labels) != record.rows:
            time_col = next(
                (name for name in ("t", "time", "timestamp") if name in labels.columns),
                None,
            )
            if time_col is None:
                raise RuntimeError(
                    f"Failure label row mismatch without a time axis for {record.flight}: "
                    f"data_rows={record.rows}, label_rows={len(labels)}"
                )
            label_time = pd.to_numeric(labels[time_col], errors="coerce").to_numpy(dtype=float)
            data_time, _ = bundle.load(record)
            if (
                not np.isfinite(label_time).all()
                or np.any(np.diff(label_time) < 0)
                or float(label_time[0]) > float(data_time[0])
                or float(label_time[-1]) < float(data_time[-1])
            ):
                raise RuntimeError(
                    f"Failure label time axis does not cover common data for {record.flight}"
                )
            in_range = (label_time >= float(data_time[0])) & (label_time <= float(data_time[-1]))
            positives_in_range = int(((label_values > 0.5) & in_range).sum())
            positive_label_rows_in_data_time_range = positives_in_range
            alignment = "time_axis_coverage"
        rows.append({
            "flight": record.flight,
            "data_rows": record.rows,
            "label_rows": len(labels),
            "positives": positives,
            "positive_label_rows_in_data_time_range": positive_label_rows_in_data_time_range,
            "alignment": alignment,
        })
    total_time_range_positives = sum(
        row["positive_label_rows_in_data_time_range"] for row in rows
    )
    if total_time_range_positives <= 0:
        raise RuntimeError(f"No positive failure labels overlap the data time range in {labels_root}")
    return {
        "labels_root": str(labels_root),
        "failure_labels": rows,
        "positive_label_rows_in_data_time_range": int(total_time_range_positives),
    }


def validate_alfa_protocol(bundle) -> None:
    """Backward-compatible alias for older tests and external scripts."""
    if bundle.dataset != "alfa":
        raise ValueError("validate_alfa_protocol is restricted to ALFA")
    validate_dataset_protocol(bundle)


def write_split_metadata(run_dir: Path, cfg, bundle, standardizer) -> dict:
    protocol = data_protocol_payload(cfg.to_legacy())
    write_json(Path(run_dir) / "normalization_stats.json", standardizer.to_dict())
    write_json(Path(run_dir) / "split_flights.json", {
        "data_split_seed": cfg.data_split_seed,
        "model_seed": cfg.model_seed,
        "data_protocol_hash": protocol["data_protocol_hash"],
        "train_flights": [row.flight for row in bundle.splits["train"]],
        "validation_flights": [row.flight for row in bundle.splits["validation"]],
        "failure_flights_scored_only": [row.flight for row in bundle.splits["failure"]],
    })
    return protocol


def augment_done(run_dir: Path, protocol: dict, classification: str) -> dict:
    done_path = Path(run_dir) / "DONE.json"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done.update({
        "data_protocol_hash": protocol["data_protocol_hash"],
        "data_protocol": protocol,
        "reproduction_classification": classification,
    })
    write_json(done_path, done)
    return done
