from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from data.stgtcn_window_dataset import resolve_flight_splits
from tcngatreconfig import BUNDLE_ROOT, TCNGATREConfig
from utils.io import ensure_dir


def parse_args(argv: list[str] | None = None, description: str = "Run TCNGATRE.") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", choices=["alfa", "simulate", "gpsdata"], default="alfa")
    return parser.parse_args(argv)


def resolve_config_from_argv(argv: list[str] | None = None, description: str = "Run TCNGATRE.") -> TCNGATREConfig:
    args = parse_args(argv=argv, description=description)
    return TCNGATREConfig(dataset_name=args.dataset)


def _run_subprocess(args: list[str]) -> None:
    subprocess.run(args, cwd=str(BUNDLE_ROOT), check=True)


def expected_graph_flights(cfg: TCNGATREConfig) -> list[str]:
    train_paths, _, _ = resolve_flight_splits(
        dataset_root=Path(cfg.data_root),
        split_info_path=Path(cfg.split_info_path),
    )
    flights = sorted(str(name) for name in train_paths)
    if not flights:
        raise RuntimeError(f"No normal training flights resolved for {cfg.dataset_name}")
    return flights


def graph_cache_matches(graph_dir: Path, expected_flights: list[str]) -> bool:
    graph_dir = Path(graph_dir)
    required = (
        graph_dir / "keep_columns.json",
        graph_dir / "adjacency_dense.csv",
        graph_dir / "build_metadata.json",
        graph_dir / "edges_mic.csv",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        metadata = json.loads((graph_dir / "build_metadata.json").read_text(encoding="utf-8"))
        included = sorted(str(name) for name in metadata.get("include_flights") or [])
        count = int(metadata.get("num_input_files", -1))
        nodes = json.loads((graph_dir / "keep_columns.json").read_text(encoding="utf-8"))
        expected_pairs = len(nodes) * (len(nodes) - 1) // 2
        pair_results = int(metadata.get("num_pair_results", -1))
        with (graph_dir / "edges_mic.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            edge_rows = sum(1 for _ in csv.DictReader(handle))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    expected = sorted(str(name) for name in expected_flights)
    return (
        included == expected
        and count == len(expected)
        and pair_results == expected_pairs
        and edge_rows == expected_pairs
    )


def ensure_graph_ready(cfg: TCNGATREConfig) -> None:
    graph_dir = Path(cfg.graph_dir)
    train_flights = expected_graph_flights(cfg)
    if graph_cache_matches(graph_dir, train_flights) and not bool(cfg.graph_overwrite):
        return
    ensure_dir(graph_dir)
    # top_k_per_node=0 → keep all edges passing MIC threshold (no topk filtering)
    cmd = [
        sys.executable,
        str(BUNDLE_ROOT / "util" / "build_set_a_graph.py"),
        "--in_dir", str(cfg.graph_input_dir),
        "--out_dir", str(cfg.graph_dir),
        "--dataset_mode", str(cfg.dataset_mode),
        "--grid_sec", str(cfg.graph_grid_sec),
        "--mic_alpha", str(cfg.graph_mic_alpha),
        "--mic_c", str(cfg.graph_mic_c),
        "--min_overlap", str(cfg.graph_min_overlap),
        "--mic_threshold", str(cfg.graph_mic_threshold),
        "--top_k_per_node", "0",
        "--max_points_per_pair", str(cfg.graph_max_points_per_pair),
        "--num_workers", str(cfg.graph_num_workers),
        "--include_flights", *train_flights,
        "--overwrite",
    ]
    _run_subprocess(cmd)
    if not graph_cache_matches(graph_dir, train_flights):
        raise RuntimeError(
            f"MIC graph provenance mismatch after build: {graph_dir}; "
            f"expected {len(train_flights)} training flights"
        )


def _set_env(name: str, value) -> None:
    if value is not None:
        os.environ[str(name)] = str(value)


def apply_env(cfg: TCNGATREConfig) -> None:
    _set_env("UAV_TCNGATRE_DATASET", cfg.dataset_name)
    _set_env("UAV_TCNGATRE_DATA_ROOT", cfg.data_root)
    _set_env("UAV_TCNGATRE_LABELS_ROOT", cfg.labels_root)
    _set_env("UAV_TCNGATRE_RUN_ROOT", cfg.run_root)
    _set_env("UAV_TCNGATRE_SPLIT_INFO_PATH", cfg.split_info_path)
    _set_env("UAV_TCNGATRE_GRAPH_DIR", cfg.graph_dir)
    _set_env("UAV_TCNGATRE_INFER_OUTPUT_NAME", cfg.infer_output_name)
    _set_env("UAV_TCNGATRE_INFER_SOURCE_SPLIT", cfg.infer_source_split)


def prepare_and_apply(cfg: TCNGATREConfig) -> TCNGATREConfig:
    ensure_dir(Path(cfg.run_root))
    ensure_graph_ready(cfg)
    apply_env(cfg)
    return cfg
