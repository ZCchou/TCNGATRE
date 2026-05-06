from __future__ import annotations

import json
from pathlib import Path


MANIFEST_NAME = "dataset_manifest.json"
DATASET_DIR_MAP = {
    "alfa": "alfa": "alfa4hz": "alfa4HZ",
    "simulate": "simulate",
    "gpsdata": "gpsdata",
}
SUPPORTED_DATASETS = tuple(DATASET_DIR_MAP.keys())


def _is_wide_flight_csv(path: Path) -> bool:
    return str(Path(path).stem).strip().lower().endswith("_label") is False


def normalize_dataset_name(dataset_name: str) -> str:
    name = str(dataset_name).strip().lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}. Expected one of {SUPPORTED_DATASETS}")
    return name


def dataset_root_from_name(portable_root: Path, dataset_name: str) -> Path:
    normalized = normalize_dataset_name(dataset_name)
    return Path(portable_root) / "dataset" / DATASET_DIR_MAP[normalized]


def load_dataset_manifest(dataset_root: Path) -> dict:
    manifest_path = Path(dataset_root) / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def discover_wide_root(dataset_root: Path) -> Path:
    dataset_root = Path(dataset_root)
    manifest = load_dataset_manifest(dataset_root)
    manifest_wide_root = str(
        manifest.get("wide_root_dirname", manifest.get("wide_root_name", ""))
    ).strip()
    if manifest_wide_root:
        explicit = dataset_root / manifest_wide_root
        if not explicit.exists():
            raise FileNotFoundError(f"Manifest-declared wide root does not exist: {explicit}")
        if not (explicit / "No_Failure").is_dir() or not (explicit / "Failure").is_dir():
            raise ValueError(f"Manifest-declared wide root is missing split directories: {explicit}")
        return explicit
    candidates = [
        path
        for path in sorted(dataset_root.iterdir())
        if path.is_dir()
        and (path / "No_Failure").is_dir()
        and (path / "Failure").is_dir()
        and any((path / "No_Failure").glob("*.csv"))
        and any((path / "Failure").glob("*.csv"))
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one wide root under {dataset_root}, found {len(candidates)}"
        )
    return candidates[0]


def discover_labels_root(dataset_root: Path, manifest: dict | None = None) -> Path:
    manifest = load_dataset_manifest(dataset_root) if manifest is None else manifest
    labels_root = Path(dataset_root) / str(manifest.get("labels_dirname", "wide_flights_failure_labels"))
    if not labels_root.exists():
        raise FileNotFoundError(f"Missing labels root: {labels_root}")
    return labels_root


def _validate_expected_counts(
    expected_counts: dict,
    no_failure_paths: dict[str, Path],
    classic_no_failure: dict[str, Path],
    prefail_normal: dict[str, Path],
    failure_paths: dict[str, Path],
    train_paths: dict[str, Path],
    val_paths: dict[str, Path],
) -> dict:
    actual_counts = {
        "no_failure_total": len(no_failure_paths),
        "classic_no_failure": len(classic_no_failure),
        "prefail_normal": len(prefail_normal),
        "failure_total": len(failure_paths),
        "train_normal": len(train_paths),
        "val_normal": len(val_paths),
    }
    mismatches = {
        key: {"expected": int(expected_counts[key]), "actual": int(actual_counts[key])}
        for key in expected_counts
        if key in actual_counts and int(expected_counts[key]) != int(actual_counts[key])
    }
    return mismatches


def build_flight_path_maps(dataset_root: Path) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path], dict]:
    dataset_root = Path(dataset_root)
    manifest = load_dataset_manifest(dataset_root)
    wide_root = discover_wide_root(dataset_root)
    labels_root = discover_labels_root(dataset_root, manifest=manifest)
    prefail_suffix = str(manifest.get("prefail_normal_suffix", "__prefail_normal"))

    no_failure_paths = {
        path.stem: path
        for path in sorted((wide_root / "No_Failure").glob("*.csv"))
        if _is_wide_flight_csv(path)
    }
    failure_paths = {
        path.stem: path
        for path in sorted((wide_root / "Failure").glob("*.csv"))
        if _is_wide_flight_csv(path)
    }

    classic_no_failure = {
        name: path for name, path in no_failure_paths.items() if not name.endswith(prefail_suffix)
    }
    prefail_normal = {
        name: path for name, path in no_failure_paths.items() if name.endswith(prefail_suffix)
    }

    legacy_train = [str(x) for x in manifest.get("legacy_train_flights", [])]
    legacy_val = [str(x) for x in manifest.get("legacy_val_flights", [])]
    missing_train = [name for name in legacy_train if name not in classic_no_failure]
    missing_val = [name for name in legacy_val if name not in classic_no_failure]
    if missing_train or missing_val:
        raise ValueError(
            "Legacy split flights missing from shared dataset: "
            f"train={missing_train} val={missing_val}"
        )

    train_paths = {name: classic_no_failure[name] for name in legacy_train}
    train_paths.update(prefail_normal)
    val_paths = {name: classic_no_failure[name] for name in legacy_val}

    expected_count_mismatches = {}
    expected_counts = manifest.get("expected_counts", {})
    if isinstance(expected_counts, dict) and expected_counts:
        expected_count_mismatches = _validate_expected_counts(
            expected_counts=expected_counts,
            no_failure_paths=no_failure_paths,
            classic_no_failure=classic_no_failure,
            prefail_normal=prefail_normal,
            failure_paths=failure_paths,
            train_paths=train_paths,
            val_paths=val_paths,
        )

    return train_paths, val_paths, failure_paths, {
        "manifest": manifest,
        "wide_root": wide_root,
        "labels_root": labels_root,
        "classic_no_failure": classic_no_failure,
        "prefail_normal": prefail_normal,
        "all_no_failure": no_failure_paths,
        "expected_count_mismatches": expected_count_mismatches,
    }
