from __future__ import annotations

import csv
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from revision_experiments.core.integrity import verify_snapshot
from revision_experiments.core.paths import BASELINE_SOURCE_PATH, EXTERNAL_ROOT, RESULTS_ROOT


def load_sources() -> dict:
    return json.loads(BASELINE_SOURCE_PATH.read_text(encoding="utf-8"))


def _run(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(
        command,
        cwd=None if cwd is None else str(cwd),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        timeout=120,
    ).stdout.strip()


def audit_repository(path: Path) -> dict:
    required = ["README.md"]
    present = {name: (path / name).exists() for name in required}
    python_files = list(path.rglob("*.py"))
    status_lines = _run(["git", "status", "--porcelain"], cwd=path).splitlines()
    missing_tracked = [line for line in status_lines if line[:2] in {" D", "D ", "DD"}]
    return {
        "path": str(path),
        "commit": _run(["git", "rev-parse", "HEAD"], cwd=path),
        "branch": _run(["git", "branch", "--show-current"], cwd=path),
        "required_files": present,
        "python_file_count": len(python_files),
        "has_requirements": (path / "requirements.txt").exists(),
        "has_executable_python": any(p.name in {"main.py", "run.py"} for p in python_files),
        "checkout_clean": not status_lines,
        "missing_tracked_file_count": len(missing_tracked),
    }


def fetch_and_audit(names: list[str] | None = None) -> dict:
    sources = load_sources()
    selected = list(sources) if names is None else names
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baselines": {},
    }
    for name in selected:
        source = sources[name]
        url = source.get("repository_url")
        entry = {**source, "name": name}
        if not url:
            entry["audit_status"] = "not_reproducible"
            entry["reason"] = "No verified official repository was found. No numeric result will be fabricated."
            report["baselines"][name] = entry
            continue
        target = EXTERNAL_ROOT / name
        try:
            if not target.exists():
                _run([
                    "git", "clone", "--depth", "1", "--filter=blob:none",
                    "--branch", source["ref"], url, str(target),
                ])
            entry.update(audit_repository(target))
            if not entry["required_files"]["README.md"] or entry["python_file_count"] == 0:
                entry["audit_status"] = "official_checkout_incomplete"
                entry["reason"] = "The pinned commit exists, but its executable source tree is incomplete."
            elif source["code_status"] == "official_incomplete_pending_audit":
                entry["audit_status"] = "incomplete_pending_clean_room_review"
            elif entry["missing_tracked_file_count"]:
                entry["audit_status"] = "official_code_fetched_bundled_data_incomplete"
                entry["reason"] = (
                    "Executable source is present at the pinned commit; some author-bundled dataset files "
                    "were not materialized and are not used by the isolated common-data adapter."
                )
            else:
                entry["audit_status"] = "official_code_fetched"
        except Exception as exc:
            entry["audit_status"] = "fetch_or_audit_failed"
            entry["reason"] = repr(exc)
        report["baselines"][name] = entry
    output = EXTERNAL_ROOT / "baseline_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _finite_score_columns(path: Path) -> dict:
    columns = ("raw_total_score", "total_score", "scores_smooth")
    counts = {name: 0 for name in columns}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        available = [name for name in columns if name in (reader.fieldnames or [])]
        if not available:
            raise ValueError(f"No recognized score column in {path}")
        row_count = 0
        for row in reader:
            row_count += 1
            for name in available:
                value = row.get(name, "")
                if value == "" or not math.isfinite(float(value)):
                    raise ValueError(f"Non-finite {name} in {path} at data row {row_count}")
                counts[name] += 1
    if row_count == 0:
        raise ValueError(f"Empty score file: {path}")
    return {"rows": row_count, "finite_columns": {key: value for key, value in counts.items() if value}}


def audit_adapter_runs(seed: int = 0) -> dict:
    """Validate the six EX-04 adapter smoke artifacts without reading failure labels for calibration."""
    sources = load_sources()
    source_audit_path = EXTERNAL_ROOT / "baseline_audit.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))["baselines"]
    required = (
        "DONE.json",
        "best.pt",
        "last.pt",
        "history.csv",
        "config_resolved.json",
        "provenance.json",
        "val_normal_scores.csv",
        "infer_tcngatre_failure/sequence_scores.csv",
        "infer_tcngatre_failure/score_threshold_analysis/primary_metrics.json",
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "ex04",
        "seed": seed,
        "expected_runs": 6,
        "runs": {},
        "legacy_integrity": verify_snapshot(),
    }
    for dataset in ("alfa", "gpsdata", "simulate"):
        for baseline in ("catch", "carots"):
            key = f"{baseline}/{dataset}/seed_{seed}"
            run_dir = RESULTS_ROOT / "protocol_v1" / "ex04" / dataset / baseline / f"seed_{seed}"
            entry = {"run_dir": str(run_dir)}
            try:
                missing = [name for name in required if not (run_dir / name).is_file()]
                forbidden = [name for name in ("FAILED.json", "PENDING_ADAPTER.json") if (run_dir / name).exists()]
                if missing or forbidden:
                    raise FileNotFoundError(f"missing={missing}; forbidden={forbidden}")
                done = json.loads((run_dir / "DONE.json").read_text(encoding="utf-8"))
                provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
                metrics = json.loads(
                    (run_dir / "infer_tcngatre_failure" / "score_threshold_analysis" / "primary_metrics.json")
                    .read_text(encoding="utf-8")
                )
                expected_commit = source_audit[baseline]["commit"]
                if done.get("source_commit") != expected_commit or provenance.get("official_commit") != expected_commit:
                    raise ValueError(f"Pinned commit mismatch for {key}")
                if not done.get("adapter_config_hash"):
                    raise ValueError(f"Missing adapter_config_hash for {key}")
                if not math.isfinite(float(metrics.get("threshold_mean", math.nan))):
                    raise ValueError(f"Non-finite flightwise SPOT threshold_mean for {key}")
                dependencies = provenance.get("baseline_dependency_versions", {})
                if not dependencies.get("torch") or not dependencies.get("numpy"):
                    raise ValueError(f"Missing dependency versions for {key}")
                score_audit = _finite_score_columns(run_dir / "infer_tcngatre_failure" / "sequence_scores.csv")
                entry.update({
                    "status": "complete",
                    "source_commit": expected_commit,
                    "adapter_config_hash": done["adapter_config_hash"],
                    "threshold_mean": metrics["threshold_mean"],
                    "num_samples": metrics["num_samples"],
                    "score_audit": score_audit,
                    "dependency_versions": dependencies,
                })
            except Exception as exc:
                entry.update({"status": "failed", "reason": repr(exc)})
            report["runs"][key] = entry
    report["completed_runs"] = sum(row["status"] == "complete" for row in report["runs"].values())
    report["failed_runs"] = report["expected_runs"] - report["completed_runs"]
    report["status"] = (
        "passed"
        if report["completed_runs"] == report["expected_runs"] and report["legacy_integrity"]["ok"]
        else "failed"
    )
    output = EXTERNAL_ROOT / "baseline_adapter_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["status"] != "passed":
        raise RuntimeError(f"Adapter audit failed; see {output}")
    return report
