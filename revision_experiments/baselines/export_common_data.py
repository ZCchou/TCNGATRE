from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from revision_experiments.core.paths import REPO_ROOT, RESULTS_ROOT, ensure_import_paths

ensure_import_paths()

from data.stgtcn_window_dataset import project_node_features, resolve_flight_splits  # noqa: E402
from tcngatre_runtime import ensure_graph_ready  # noqa: E402
from tcngatre_train_impl import fit_normalization_stats, load_graph  # noqa: E402
from utils.normalization import apply_train_minmax, load_wide_flight_frame  # noqa: E402


COMMON_DATA_ROOT = RESULTS_ROOT / "protocol_v1" / "_baseline_common_data"
EXPORT_SCHEMA_VERSION = 4
EXPORT_PROFILE = "canonical_fixed_manifest_protocol_v2"
SPLITS = ("train", "validation", "failure")


def split_fingerprint(split_names: dict[str, list[str]]) -> str:
    canonical = {
        split: sorted(str(name) for name in split_names.get(split, []))
        for split in SPLITS
    }
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _current_expected_split_names(dataset: str) -> dict[str, list[str]] | None:
    data_root = REPO_ROOT / "dataset" / str(dataset)
    if not (data_root / "dataset_manifest.json").is_file():
        return None
    train, validation, failure = resolve_flight_splits(dataset_root=data_root)
    return {
        "train": sorted(str(name) for name in train),
        "validation": sorted(str(name) for name in validation),
        "failure": sorted(str(name) for name in failure),
    }


def _portable_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_record_path(raw: str) -> Path:
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _export_split(paths: dict[str, Path], nodes: list[str], stats: dict, output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for flight, csv_path in paths.items():
        time, raw, columns = load_wide_flight_frame(csv_path)
        values = project_node_features(raw, list(columns), nodes)
        normalized = apply_train_minmax(values, stats)
        target = output / f"{flight}.npz"
        np.savez_compressed(target, time=time.astype(np.float32), values=normalized.astype(np.float32))
        manifest.append({
            "flight": str(flight),
            "path": _portable_path(target),
            "rows": int(normalized.shape[0]),
            "channels": int(normalized.shape[1]),
        })
    return manifest


def validate_common_data(
    dataset: str,
    output_root: Path = COMMON_DATA_ROOT,
    *,
    verify_arrays: bool = False,
) -> dict:
    """Validate a portable, label-free baseline data export."""
    dataset = str(dataset)
    dataset_root = (Path(output_root) / dataset).resolve()
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Common baseline data is missing: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("schema_version") != EXPORT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {EXPORT_SCHEMA_VERSION}")
    if payload.get("export_profile") != EXPORT_PROFILE:
        errors.append(f"export_profile must be {EXPORT_PROFILE}")
    if payload.get("dataset") != dataset:
        errors.append(f"dataset mismatch: {payload.get('dataset')!r}")
    if payload.get("labels_exported") is not False:
        errors.append("labels_exported must be false")
    try:
        int(payload["data_split_seed"])
    except (KeyError, TypeError, ValueError):
        errors.append("data_split_seed must be recorded")
    graph_profile = payload.get("graph_profile")
    try:
        graph_points = int(graph_profile.get("max_points_per_pair", 0))
    except (AttributeError, TypeError, ValueError):
        graph_points = 0
    if (
        not isinstance(graph_profile, dict)
        or graph_profile.get("source") != "normal training flights only"
        or graph_points <= 0
    ):
        errors.append("graph_profile must describe a canonical train-normal graph")

    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or not nodes or len(nodes) != len(set(nodes)):
        errors.append("nodes must be a non-empty unique list")
        nodes = []

    actual_split_names: dict[str, list[str]] = {}
    for split in SPLITS:
        records = payload.get(split)
        if not isinstance(records, list) or not records:
            errors.append(f"{split} must contain at least one flight")
            actual_split_names[split] = []
            continue
        names = [str(record.get("flight", "")) for record in records if isinstance(record, dict)]
        actual_split_names[split] = names
        if any(not name for name in names) or len(names) != len(set(names)):
            errors.append(f"{split} flight names must be non-empty and unique")
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                errors.append(f"{split}[{index}] is not an object")
                continue
            required = {"flight", "path", "rows", "channels"}
            missing = sorted(required.difference(record))
            if missing:
                errors.append(f"{split}[{index}] missing {missing}")
                continue
            path = _resolve_record_path(str(record["path"])).resolve()
            try:
                path.relative_to(dataset_root)
            except ValueError:
                errors.append(f"{split}[{index}] escapes dataset root: {path}")
                continue
            if not path.is_file():
                errors.append(f"{split}[{index}] file is missing: {path}")
                continue
            rows = int(record["rows"])
            channels = int(record["channels"])
            if rows <= 0 or channels != len(nodes):
                errors.append(
                    f"{split}[{index}] invalid shape metadata: rows={rows}, channels={channels}"
                )
                continue
            if verify_arrays:
                try:
                    with np.load(path, allow_pickle=False) as data:
                        time = np.asarray(data["time"])
                        values = np.asarray(data["values"])
                    if time.shape != (rows,) or values.shape != (rows, channels):
                        errors.append(
                            f"{split}[{index}] array shape mismatch: "
                            f"time={time.shape}, values={values.shape}"
                        )
                    elif not np.isfinite(time).all() or not np.isfinite(values).all():
                        errors.append(f"{split}[{index}] contains NaN or Inf")
                except Exception as exc:
                    errors.append(f"{split}[{index}] cannot be read: {exc!r}")

    split_sets = {split: set(actual_split_names.get(split, [])) for split in SPLITS}
    for left, right in (("train", "validation"), ("train", "failure"), ("validation", "failure")):
        overlap = sorted(split_sets[left].intersection(split_sets[right]))
        if overlap:
            errors.append(f"{left}/{right} flight overlap: {overlap}")

    actual_fingerprint = split_fingerprint(actual_split_names)
    if payload.get("split_fingerprint") != actual_fingerprint:
        errors.append("split_fingerprint does not match manifest flight lists")

    expected_split_names = _current_expected_split_names(dataset)
    if expected_split_names is not None:
        for split in SPLITS:
            if sorted(actual_split_names.get(split, [])) != expected_split_names[split]:
                errors.append(
                    f"{split} flights do not match the current fixed dataset manifest protocol"
                )

    if isinstance(graph_profile, dict):
        graph_flights = sorted(str(name) for name in graph_profile.get("include_flights") or [])
        if graph_flights != sorted(actual_split_names.get("train", [])):
            errors.append("graph_profile.include_flights must equal the normal training split")
        try:
            graph_count = int(graph_profile.get("num_input_files", -1))
        except (TypeError, ValueError):
            graph_count = -1
        if graph_count != len(actual_split_names.get("train", [])):
            errors.append("graph_profile.num_input_files must equal the normal training count")

    if errors:
        raise RuntimeError("Invalid common baseline data:\n- " + "\n- ".join(errors))
    return payload


def export_dataset(legacy_cfg, output_root: Path = COMMON_DATA_ROOT) -> dict:
    """Export a label-free common numerical format for isolated baselines."""
    dataset_root = Path(output_root) / legacy_cfg.dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)
    legacy_cfg.normalization_stats_path = dataset_root / "train_minmax_stats.json"

    nodes, _, _ = load_graph(Path(legacy_cfg.graph_dir))
    stats = fit_normalization_stats(legacy_cfg, nodes)
    train, validation, failure = resolve_flight_splits(
        dataset_root=Path(legacy_cfg.data_root),
        split_info_path=Path(legacy_cfg.split_info_path),
    )
    split_names = {
        "train": sorted(str(name) for name in train),
        "validation": sorted(str(name) for name in validation),
        "failure": sorted(str(name) for name in failure),
    }
    graph_metadata_path = Path(legacy_cfg.graph_dir) / "build_metadata.json"
    if not graph_metadata_path.is_file():
        raise FileNotFoundError(f"MIC graph metadata is missing: {graph_metadata_path}")
    graph_metadata = json.loads(graph_metadata_path.read_text(encoding="utf-8"))
    graph_flights = sorted(str(name) for name in graph_metadata.get("include_flights") or [])
    if graph_flights != split_names["train"]:
        raise RuntimeError(
            "MIC graph was not built exclusively from the fixed normal training split"
        )
    manifest = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "export_profile": EXPORT_PROFILE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": legacy_cfg.dataset_name,
        "data_split_seed": int(legacy_cfg.split_seed),
        "split_fingerprint": split_fingerprint(split_names),
        "split_policy": {
            "source": "dataset_manifest fixed flight lists",
            "prefail_normal_policy": "train_only",
            "counts": {split: len(names) for split, names in split_names.items()},
        },
        "graph_profile": {
            "source": "normal training flights only",
            "include_flights": graph_flights,
            "num_input_files": int(graph_metadata.get("num_input_files", -1)),
            "max_points_per_pair": int(legacy_cfg.graph_max_points_per_pair),
            "mic_alpha": float(legacy_cfg.graph_mic_alpha),
            "mic_c": int(legacy_cfg.graph_mic_c),
            "mic_threshold": float(legacy_cfg.graph_mic_threshold),
        },
        "nodes": nodes,
        "normalization_source": "normal training flights only",
        "labels_exported": False,
        "train": _export_split(train, nodes, stats, dataset_root / "train_normal"),
        "validation": _export_split(validation, nodes, stats, dataset_root / "validation_normal"),
        "failure": _export_split(failure, nodes, stats, dataset_root / "failure_unlabeled"),
    }
    manifest_path = dataset_root / "manifest.json"
    temporary_path = dataset_root / "manifest.json.tmp"
    temporary_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(manifest_path)
    return validate_common_data(legacy_cfg.dataset_name, output_root, verify_arrays=True)


def _canonical_legacy_config(dataset: str):
    # Always use the full protocol graph settings so a smoke run cannot seed a
    # reduced MIC graph that would later leak into formal experiments.
    from revision_experiments.core.config import make_config

    legacy_cfg = make_config("ex01", dataset, "full", seed=0, smoke=False).to_legacy()
    legacy_cfg.graph_overwrite = True
    return legacy_cfg


def ensure_common_data(
    dataset: str,
    output_root: Path = COMMON_DATA_ROOT,
    *,
    force: bool = False,
) -> dict:
    """Return a valid export, creating it once when it is absent or stale."""
    if not force:
        try:
            return validate_common_data(dataset, output_root)
        except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError):
            pass

    legacy_cfg = _canonical_legacy_config(dataset)
    ensure_graph_ready(legacy_cfg)
    return export_dataset(legacy_cfg, output_root)
