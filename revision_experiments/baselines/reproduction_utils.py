from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

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


def validate_alfa_protocol(bundle) -> None:
    if bundle.dataset != "alfa":
        raise ValueError("Reviewer-requested UAV baselines currently support ALFA only")
    counts = {name: len(bundle.splits[name]) for name in ("train", "validation", "failure")}
    expected = {"train": 29, "validation": 1, "failure": 16}
    if counts != expected:
        raise RuntimeError(f"ALFA common-data protocol mismatch: expected={expected}, actual={counts}")
    if len(bundle.nodes) != 12:
        raise RuntimeError(f"ALFA common-data channel mismatch: expected=12, actual={len(bundle.nodes)}")
    train = {row.flight for row in bundle.splits["train"]}
    validation = {row.flight for row in bundle.splits["validation"]}
    failure = {row.flight for row in bundle.splits["failure"]}
    overlaps = {
        "train_validation": sorted(train & validation),
        "train_failure": sorted(train & failure),
        "validation_failure": sorted(validation & failure),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"ALFA common-data split overlap: {overlaps}")


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
