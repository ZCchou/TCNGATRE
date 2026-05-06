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


def _build_split_map(split_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not split_dir.exists():
        return out
    for csv_path in sorted(split_dir.glob("*.csv")):
        if not _is_wide_flight_csv(csv_path):
            continue
        out[str(csv_path.stem)] = csv_path
    return out


def build_flight_path_maps(dataset_root: Path) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path], dict]:
    dataset_root = Path(dataset_root)
    manifest = load_dataset_manifest(dataset_root)
    wide_root = discover_wide_root(dataset_root)
    labels_root = discover_labels_root(dataset_root, manifest=manifest)

    no_failure_paths = _build_split_map(wide_root / "No_Failure")
    failure_paths = _build_split_map(wide_root / "Failure")

    legacy_train = [str(x) for x in manifest.get("legacy_train_flights", [])]
    legacy_val = [str(x) for x in manifest.get("legacy_val_flights", [])]

    if legacy_train or legacy_val:
        missing = [name for name in [*legacy_train, *legacy_val] if name not in no_failure_paths]
        if missing:
            raise ValueError(
                "Legacy split flights missing from shared dataset: "
                + ", ".join(sorted(missing))
            )
        train_paths = {name: no_failure_paths[name] for name in legacy_train}
        val_paths = {name: no_failure_paths[name] for name in legacy_val}
    else:
        names = sorted(no_failure_paths.keys())
        if len(names) <= 1:
            train_paths = {name: no_failure_paths[name] for name in names}
            val_paths = {name: no_failure_paths[name] for name in names}
        else:
            split_idx = max(1, len(names) - 1)
            train_names = names[:split_idx]
            val_names = names[split_idx:]
            train_paths = {name: no_failure_paths[name] for name in train_names}
            val_paths = {name: no_failure_paths[name] for name in val_names}

    metadata = {
        "manifest": manifest,
        "wide_root": wide_root,
        "labels_root": labels_root,
        "no_failure_paths": no_failure_paths,
        "failure_paths": failure_paths,
    }
    return train_paths, val_paths, failure_paths, metadata
