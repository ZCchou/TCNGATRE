from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from revision_experiments.core.paths import ensure_import_paths

ensure_import_paths()

from data.stgtcn_window_dataset import project_node_features, resolve_flight_splits  # noqa: E402
from tcngatre_train_impl import fit_normalization_stats, load_graph  # noqa: E402
from utils.normalization import apply_train_minmax, load_wide_flight_frame  # noqa: E402


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
            "path": str(target),
            "rows": int(normalized.shape[0]),
            "channels": int(normalized.shape[1]),
        })
    return manifest

def export_dataset(legacy_cfg, output_root: Path) -> dict:
    """Export a label-free common numerical format for isolated baselines."""
    nodes, _, _ = load_graph(Path(legacy_cfg.graph_dir))
    stats = fit_normalization_stats(legacy_cfg, nodes)
    train, validation, failure = resolve_flight_splits(dataset_root=Path(legacy_cfg.data_root))
    dataset_root = Path(output_root) / legacy_cfg.dataset_name
    manifest = {
        "dataset": legacy_cfg.dataset_name,
        "nodes": nodes,
        "normalization_source": "normal training flights only",
        "labels_exported": False,
        "train": _export_split(train, nodes, stats, dataset_root / "train_normal"),
        "validation": _export_split(validation, nodes, stats, dataset_root / "validation_normal"),
        "failure": _export_split(failure, nodes, stats, dataset_root / "failure_unlabeled"),
    }
    (dataset_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
