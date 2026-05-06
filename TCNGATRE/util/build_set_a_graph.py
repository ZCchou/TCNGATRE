# -*- coding: utf-8 -*-
"""
build_set_a_graph.py

Build a real MIC-based graph for set_a-style model nodes.

Default behavior:
- `set_a_causal` mode:
  - read causal set_a CSVs from `dataset/alfa/wide_flights_set_a_causal/No_Failure`
  - use `{feature}_med_mm` columns
- `alfanew` mode:
  - read dense interpolated CSVs from `dataset/alfanew/.../No_Failure`
  - use the raw 14 feature columns directly
- resample each flight onto a shared time grid
- linearly interpolate each feature on that grid
- compute pairwise MIC on the resampled aligned observations
- write:
    out_static_graph_set_a_causal_mic/
        keep_columns.json
        adjacency_dense.csv
        edges_mic.csv
        build_metadata.json

The graph is fit on No_Failure only to avoid leakage from the failure split.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

SET_A_CAUSAL_FEATS = [
    "mavros-nav_info-roll:field.measured__sin",
    "mavros-nav_info-roll:field.measured__cos",
    "mavros-nav_info-pitch:field.measured__sin",
    "mavros-nav_info-pitch:field.measured__cos",
    "mavros-nav_info-yaw:field.measured__sin",
    "mavros-nav_info-yaw:field.measured__cos",
    "mavros-vfr_hud:field.groundspeed",
    "mavros-vfr_hud:field.heading__sin",
    "mavros-vfr_hud:field.heading__cos",
    "mavros-vfr_hud:field.throttle",
    "mavros-vfr_hud:field.climb",
    "mavros-vfr_hud:field.altitude",
]

ALFANEW_FEATS = [
    "MNI/roll",
    "MNI/pitch",
    "MNI/yaw",
    "MVH/groundspeed",
    "MVH/heading",
    "MVH/throttle",
    "MVH/climb",
    "MVH/altitude",
    "MLP/linear_velocity_x",
    "MLP/linear_velocity_y",
    "MLP/linear_velocity_z",
    "MLP/angular_velocity_x",
    "MLP/angular_velocity_y",
    "MLP/angular_velocity_z",
]

SIMULATE_FEATS = [
    "u_cmd",
    "position",
    "velocity",
    "accel",
    "torque",
    "current",
    "voltage",
]
LEGACY_ALFA4HZ_SAMPLE_RATE_HZ = 4.0


def _is_legacy_alfa4hz_path(path: Path) -> bool:
    return any(str(part).strip().lower() == "alfa4hz" for part in Path(path).parts)


def infer_feature_columns(in_dir: Path) -> List[str]:
    candidate_files = sorted(in_dir.glob("*.csv"))
    if not candidate_files:
        raise FileNotFoundError(f"Could not infer feature columns under {in_dir}")
    header = pd.read_csv(candidate_files[0], nrows=0, low_memory=False)
    if "t" not in header.columns and _is_legacy_alfa4hz_path(candidate_files[0]):
        raw = pd.read_csv(candidate_files[0], header=None, nrows=1, low_memory=False)
        return [f"f{idx:02d}" for idx in range(raw.shape[1])]
    return [str(col) for col in header.columns if str(col) != "t"]


def resolve_feature_columns(dataset_mode: str, in_dir: Path) -> tuple[List[str], str]:
    mode = str(dataset_mode).strip().lower()
    if mode == "alfanew":
        return list(ALFANEW_FEATS), ""
    if mode == "simulate":
        return list(SIMULATE_FEATS), ""
    if mode == "gpsdata":
        return infer_feature_columns(in_dir), ""
    return list(SET_A_CAUSAL_FEATS), "_med_mm"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a MIC graph for set_a-style features from No_Failure CSVs."
    )
    ap.add_argument(
        "--in_dir",
        default="dataset/alfa/wide_flights_set_a_causal/No_Failure",
        help="Directory containing No_Failure causal CSV files.",
    )
    ap.add_argument(
        "--out_dir",
        default="out_static_graph_set_a_causal_mic",
        help="Output directory for the MIC graph artifacts.",
    )
    ap.add_argument(
        "--feature_suffix",
        default="_med_mm",
        help="Column suffix used for MIC computation.",
    )
    ap.add_argument(
        "--dataset_mode",
        default="set_a_causal",
        choices=["set_a_causal", "alfanew", "simulate", "gpsdata"],
        help="Feature schema to use when reading CSV files.",
    )
    ap.add_argument(
        "--grid_sec",
        type=float,
        default=0.1,
        help="Uniform resampling grid width in seconds before MIC fitting.",
    )
    ap.add_argument(
        "--mic_alpha",
        type=float,
        default=0.6,
        help="minepy MINE alpha parameter.",
    )
    ap.add_argument(
        "--mic_c",
        type=int,
        default=15,
        help="minepy MINE c parameter.",
    )
    ap.add_argument(
        "--min_overlap",
        type=int,
        default=128,
        help="Minimum pairwise valid points required to keep an edge.",
    )
    ap.add_argument(
        "--mic_threshold",
        type=float,
        default=0.0,
        help="Drop edges with MIC strictly below this threshold.",
    )
    ap.add_argument(
        "--top_k_per_node",
        type=int,
        default=0,
        help=(
            "Keep only each node's top-k MIC neighbors after thresholding. "
            "0 disables top-k sparsification."
        ),
    )
    ap.add_argument(
        "--max_points_per_pair",
        type=int,
        default=200000,
        help="Upper bound on aligned points used for one pairwise MIC fit.",
    )
    ap.add_argument(
        "--include_flights",
        nargs="*",
        default=None,
        help="Optional CSV stems to include when fitting the graph.",
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes for pairwise MIC fitting. 1 runs serially.",
    )
    ap.add_argument(
        "--progress_every",
        type=int,
        default=10,
        help="Fallback progress print interval when tqdm is unavailable.",
    )
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def interpolate_feature_to_grid(
    t_src: np.ndarray,
    x_src: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(t_src) & np.isfinite(x_src)
    if int(valid.sum()) < 2:
        return np.full((len(t_grid),), np.nan, dtype=np.float64)

    t_valid = t_src[valid]
    x_valid = x_src[valid]
    order = np.argsort(t_valid, kind="mergesort")
    t_valid = t_valid[order]
    x_valid = x_valid[order]

    unique_t, unique_idx = np.unique(t_valid, return_index=True)
    x_unique = x_valid[unique_idx]
    if len(unique_t) < 2:
        return np.full((len(t_grid),), np.nan, dtype=np.float64)

    x_grid = np.interp(t_grid, unique_t, x_unique)
    in_range = (t_grid >= unique_t[0]) & (t_grid <= unique_t[-1])
    x_grid = np.where(in_range, x_grid, np.nan)
    return x_grid.astype(np.float64, copy=False)


def load_feature_frames(
    in_dir: Path,
    feat_cols: List[str],
    feature_suffix: str,
    grid_sec: float,
    include_flights: List[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    csv_files = sorted(in_dir.glob("*.csv"))
    include_set = (
        {str(item).strip() for item in include_flights if str(item).strip()}
        if include_flights is not None
        else None
    )
    if include_set is not None:
        csv_files = [path for path in csv_files if path.stem in include_set]
        found = {path.stem for path in csv_files}
        missing = sorted(include_set - found)
        if missing:
            raise FileNotFoundError(
                "Requested graph input flights missing under "
                f"{in_dir}: {', '.join(missing)}"
            )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {in_dir}")

    pieces: List[pd.DataFrame] = []
    file_rows: Dict[str, Dict[str, int]] = {}
    required_cols = ["t"] + [f"{feat}{feature_suffix}" for feat in feat_cols]

    for fp in csv_files:
        try:
            df = pd.read_csv(fp, usecols=required_cols, low_memory=False)
            t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=np.float64)
            rename_map = {f"{feat}{feature_suffix}": feat for feat in feat_cols}
            df = df.rename(columns=rename_map)
        except ValueError:
            if not _is_legacy_alfa4hz_path(fp):
                raise
            raw = pd.read_csv(fp, header=None, low_memory=False).apply(pd.to_numeric, errors="coerce")
            if raw.shape[1] != len(feat_cols):
                raise ValueError(f"Legacy alfa4hz feature width mismatch in {fp}: {raw.shape[1]} vs {len(feat_cols)}")
            raw.columns = list(feat_cols)
            raw.insert(0, "t", np.arange(len(raw), dtype=np.float64) / float(LEGACY_ALFA4HZ_SAMPLE_RATE_HZ))
            df = raw
            t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=np.float64)
        t_valid = t[np.isfinite(t)]
        if len(t_valid) < 2:
            continue

        t_min = float(np.min(t_valid))
        t_max = float(np.max(t_valid))
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
            continue

        t_grid = np.arange(t_min, t_max + 0.5 * float(grid_sec), float(grid_sec), dtype=np.float64)
        resampled: Dict[str, np.ndarray] = {}
        for feat in feat_cols:
            x = pd.to_numeric(df[feat], errors="coerce").to_numpy(dtype=np.float64)
            resampled[feat] = interpolate_feature_to_grid(t_src=t, x_src=x, t_grid=t_grid)
        pieces.append(pd.DataFrame(resampled))
        file_rows[fp.name] = {
            "raw_rows": int(len(df)),
            "grid_rows": int(len(t_grid)),
        }

    if not pieces:
        raise ValueError(f"No valid files produced resampled grids under: {in_dir}")
    merged = pd.concat(pieces, axis=0, ignore_index=True)
    return merged, file_rows


def maybe_subsample_pair(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, num=max_points, dtype=np.int64)
    return x[idx], y[idx]


def iter_feature_pairs(feat_cols: List[str]) -> Iterable[tuple[int, int, str, str]]:
    for i, src in enumerate(feat_cols):
        for j in range(i + 1, len(feat_cols)):
            yield i, j, src, feat_cols[j]


def compute_one_mic_pair(
    payload: tuple[int, int, str, str, np.ndarray, np.ndarray, float, int, int, float, int],
) -> Dict[str, object]:
    from minepy import MINE

    (
        i,
        j,
        src,
        dst,
        x,
        y,
        alpha,
        c,
        min_overlap,
        mic_threshold,
        max_points_per_pair,
    ) = payload
    valid = np.isfinite(x) & np.isfinite(y)
    overlap = int(valid.sum())
    mic = 0.0
    kept = False
    if overlap >= int(min_overlap):
        x_pair = x[valid]
        y_pair = y[valid]
        x_pair, y_pair = maybe_subsample_pair(
            x_pair,
            y_pair,
            max_points=int(max_points_per_pair),
        )
        mine = MINE(alpha=float(alpha), c=int(c))
        mine.compute_score(x_pair, y_pair)
        mic = float(mine.mic())
        kept = mic >= float(mic_threshold)
    return {
        "i": int(i),
        "j": int(j),
        "src": str(src),
        "dst": str(dst),
        "mic": float(mic),
        "overlap": int(overlap),
        "kept": int(kept),
    }


def progress_iter(items, total: int, desc: str, progress_every: int):
    try:
        from tqdm import tqdm

        yield from tqdm(items, total=total, desc=desc, unit="pair", dynamic_ncols=True)
        return
    except Exception:
        pass

    start = time.time()
    every = max(int(progress_every), 1)
    for idx, item in enumerate(items, start=1):
        if idx == 1 or idx == total or idx % every == 0:
            elapsed = max(time.time() - start, 1e-9)
            rate = idx / elapsed
            print(f"[GRAPH] {desc}: {idx}/{total} pairs done ({rate:.2f} pair/s)", flush=True)
        yield item


def compute_mic_adjacency(
    df: pd.DataFrame,
    feat_cols: List[str],
    alpha: float,
    c: int,
    min_overlap: int,
    mic_threshold: float,
    max_points_per_pair: int,
    num_workers: int = 1,
    progress_every: int = 10,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    num_nodes = len(feat_cols)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    values = {
        feat: pd.to_numeric(df[feat], errors="coerce").to_numpy(dtype=np.float64)
        for feat in feat_cols
    }
    pair_payloads = [
        (
            i,
            j,
            src,
            dst,
            values[src],
            values[dst],
            float(alpha),
            int(c),
            int(min_overlap),
            float(mic_threshold),
            int(max_points_per_pair),
        )
        for i, j, src, dst in iter_feature_pairs(feat_cols)
    ]
    total_pairs = len(pair_payloads)
    workers = max(int(num_workers), 1)
    if workers > 1:
        workers = min(workers, total_pairs, max(os.cpu_count() or 1, 1))
        print(f"[GRAPH] Fitting MIC with {workers} worker processes over {total_pairs} pairs.", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(compute_one_mic_pair, payload) for payload in pair_payloads]
            iterator = (future.result() for future in as_completed(futures))
            edge_rows = list(
                progress_iter(
                    iterator,
                    total=total_pairs,
                    desc="MIC pairs",
                    progress_every=progress_every,
                )
            )
    else:
        print(f"[GRAPH] Fitting MIC serially over {total_pairs} pairs.", flush=True)
        edge_rows = [
            compute_one_mic_pair(payload)
            for payload in progress_iter(
                pair_payloads,
                total=total_pairs,
                desc="MIC pairs",
                progress_every=progress_every,
            )
        ]

    edge_rows = sorted(edge_rows, key=lambda row: (int(row["i"]), int(row["j"])))
    for row in edge_rows:
        if int(row["kept"]):
            i = int(row["i"])
            j = int(row["j"])
            adjacency[i, j] = np.float32(float(row["mic"]))
            adjacency[j, i] = np.float32(float(row["mic"]))
        row.pop("i", None)
        row.pop("j", None)
    return adjacency, edge_rows


def sparsify_adjacency_top_k(
    adjacency: np.ndarray,
    edge_rows: List[Dict[str, object]],
    feat_cols: List[str],
    top_k_per_node: int,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    top_k = max(int(top_k_per_node), 0)
    if top_k <= 0:
        for row in edge_rows:
            row["kept_by_threshold"] = int(row["kept"])
            row["kept_by_top_k"] = int(row["kept"])
        return adjacency, edge_rows

    adj = np.asarray(adjacency, dtype=np.float32)
    keep_directed = np.zeros(adj.shape, dtype=bool)
    for i in range(adj.shape[0]):
        candidates = np.flatnonzero(adj[i] > 0)
        if candidates.size <= 0:
            continue
        weights = adj[i, candidates]
        order = np.argsort(-weights)
        chosen = candidates[order[: min(top_k, len(order))]]
        keep_directed[i, chosen] = True

    # Use the symmetric union so either endpoint can keep a strong neighbor.
    keep = keep_directed | keep_directed.T
    sparse = np.where(keep, adj, 0.0).astype(np.float32)

    name_to_idx = {name: idx for idx, name in enumerate(feat_cols)}
    for row in edge_rows:
        i = name_to_idx[str(row["src"])]
        j = name_to_idx[str(row["dst"])]
        kept_threshold = int(row["kept"])
        kept_top_k = int(sparse[i, j] > 0)
        row["kept_by_threshold"] = kept_threshold
        row["kept_by_top_k"] = kept_top_k
        row["kept"] = kept_top_k
    return sparse, edge_rows


def main() -> int:
    args = parse_args()
    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_cols, default_feature_suffix = resolve_feature_columns(args.dataset_mode, in_dir=in_dir)

    keep_fp = out_dir / "keep_columns.json"
    adj_fp = out_dir / "adjacency_dense.csv"
    edges_fp = out_dir / "edges_mic.csv"
    meta_fp = out_dir / "build_metadata.json"

    if keep_fp.exists() and adj_fp.exists() and not args.overwrite:
        print(f"[SKIP] graph already exists at {out_dir}")
        return 0

    merged, file_rows = load_feature_frames(
        in_dir=in_dir,
        feat_cols=feat_cols,
        feature_suffix=(
            default_feature_suffix
            if args.dataset_mode in {"alfanew", "simulate", "gpsdata"}
            else str(args.feature_suffix)
        ),
        grid_sec=float(args.grid_sec),
        include_flights=[str(x) for x in args.include_flights] if args.include_flights is not None else None,
    )
    print(
        f"[GRAPH] Loaded {len(file_rows)} files, rows={len(merged)}, "
        f"features={len(feat_cols)}, grid_sec={float(args.grid_sec):g}.",
        flush=True,
    )
    adjacency, edge_rows = compute_mic_adjacency(
        df=merged,
        feat_cols=feat_cols,
        alpha=float(args.mic_alpha),
        c=int(args.mic_c),
        min_overlap=int(args.min_overlap),
        mic_threshold=float(args.mic_threshold),
        max_points_per_pair=int(args.max_points_per_pair),
        num_workers=int(args.num_workers),
        progress_every=int(args.progress_every),
    )
    threshold_kept_edges = int(sum(int(row["kept"]) for row in edge_rows))
    adjacency, edge_rows = sparsify_adjacency_top_k(
        adjacency=adjacency,
        edge_rows=edge_rows,
        feat_cols=feat_cols,
        top_k_per_node=int(args.top_k_per_node),
    )

    with keep_fp.open("w", encoding="utf-8") as fh:
        json.dump(feat_cols, fh, ensure_ascii=False, indent=2)

    df_adj = pd.DataFrame(adjacency, index=feat_cols, columns=feat_cols)
    df_adj.to_csv(adj_fp, encoding="utf-8")

    edge_columns = ["src", "dst", "mic", "overlap", "kept", "kept_by_threshold", "kept_by_top_k"]
    edges_df = pd.DataFrame(edge_rows, columns=edge_columns)
    if len(edges_df) > 0:
        edges_df = edges_df.sort_values(
            ["kept", "mic", "overlap"],
            ascending=[False, False, False],
            kind="mergesort",
        )
    edges_df.to_csv(edges_fp, index=False, encoding="utf-8-sig")

    metadata = {
        "graph_type": "mic",
        "input_dir": str(in_dir),
        "dataset_mode": str(args.dataset_mode),
        "feature_suffix": (
            default_feature_suffix
            if args.dataset_mode in {"alfanew", "simulate", "gpsdata"}
            else str(args.feature_suffix)
        ),
        "grid_sec": float(args.grid_sec),
        "num_nodes": len(feat_cols),
        "num_input_files": len(file_rows),
        "include_flights": [str(x) for x in args.include_flights] if args.include_flights is not None else None,
        "num_total_rows": int(len(merged)),
        "mic_alpha": float(args.mic_alpha),
        "mic_c": int(args.mic_c),
        "min_overlap": int(args.min_overlap),
        "mic_threshold": float(args.mic_threshold),
        "top_k_per_node": max(int(args.top_k_per_node), 0),
        "max_points_per_pair": int(args.max_points_per_pair),
        "num_workers": max(int(args.num_workers), 1),
        "num_threshold_kept_edges": threshold_kept_edges,
        "num_kept_edges": int(sum(int(row["kept"]) for row in edge_rows)),
        "max_mic": float(adjacency.max()) if adjacency.size else 0.0,
        "mean_nonzero_mic": (
            float(adjacency[adjacency > 0].mean())
            if np.any(adjacency > 0)
            else 0.0
        ),
        "num_total_raw_rows": int(sum(int(v["raw_rows"]) for v in file_rows.values())),
        "num_total_grid_rows": int(sum(int(v["grid_rows"]) for v in file_rows.values())),
        "files": file_rows,
    }
    with meta_fp.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    print(f"[OK] MIC graph written to: {out_dir}")
    print(f"  nodes={len(feat_cols)}")
    print(f"  kept_edges={metadata['num_kept_edges']}")
    print(f"  top_k_per_node={metadata['top_k_per_node']}")
    print(f"  max_mic={metadata['max_mic']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
