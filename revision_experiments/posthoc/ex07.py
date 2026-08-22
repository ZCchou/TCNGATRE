from __future__ import annotations

import hashlib
import json
import traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from revision_experiments.core.paths import ensure_import_paths

from .inference import LoadedSourceModel
from .io import environment_payload, sha256_file, write_json
from .source import SourceRun, audit_source

ensure_import_paths()

from data.window_labels import attach_window_labels  # noqa: E402


DESIGN_VERSION = "ex07_graph_v1"
GRAPH_NAMES = ("A_static", "A_dyn", "A_fuse")
PERIODS = ("before", "during", "after")


def _config_hash(source: SourceRun) -> str:
    payload = {
        "design": DESIGN_VERSION,
        "source_signature": source.source_signature,
        "graphs": GRAPH_NAMES,
        "top_k": 20,
        "periods": PERIODS,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _entropy(matrix: np.ndarray) -> float:
    values = np.maximum(np.asarray(matrix, dtype=np.float64), 1e-12)
    values = values / values.sum(axis=-1, keepdims=True).clip(min=1e-12)
    return float(-np.mean(np.sum(values * np.log(values), axis=-1)))


def _top_edges(matrix: np.ndarray, node_names: list[str], k: int = 20) -> list[dict[str, Any]]:
    candidates: list[tuple[float, int, int]] = []
    for source in range(matrix.shape[0]):
        for target in range(matrix.shape[1]):
            if source == target:
                continue
            candidates.append((float(matrix[source, target]), source, target))
    candidates.sort(reverse=True)
    return [
        {
            "rank": rank + 1,
            "source": node_names[source],
            "target": node_names[target],
            "weight": weight,
        }
        for rank, (weight, source, target) in enumerate(candidates[:k])
    ]


def _jaccard(left: np.ndarray, right: np.ndarray, nodes: list[str], k: int = 20) -> float:
    left_edges = {(row["source"], row["target"]) for row in _top_edges(left, nodes, k)}
    right_edges = {(row["source"], row["target"]) for row in _top_edges(right, nodes, k)}
    union = left_edges | right_edges
    return 1.0 if not union else len(left_edges & right_edges) / len(union)


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return 0.0
    value = float(frame["left"].corr(frame["right"], method="spearman"))
    return value if np.isfinite(value) else 0.0


def _period_indices(group: pd.DataFrame) -> dict[str, np.ndarray] | None:
    ordered = group.sort_values("t_start", kind="mergesort")
    during_positions = np.flatnonzero(ordered["label_any"].to_numpy(dtype=float) > 0)
    if len(during_positions) == 0:
        return None
    first, last = int(during_positions[0]), int(during_positions[-1])
    count = int(len(during_positions))
    before_positions = np.arange(max(0, first - count), first, dtype=int)
    after_positions = np.arange(last + 1, min(len(ordered), last + 1 + count), dtype=int)
    if len(during_positions) == 0:
        return None
    rows = ordered["graph_row"].to_numpy(dtype=int)
    output = {"during": rows[during_positions]}
    if len(before_positions):
        output["before"] = rows[before_positions]
    if len(after_positions):
        output["after"] = rows[after_positions]
    return output


def _select_cases(frame: pd.DataFrame) -> list[str]:
    durations = []
    for flight, group in frame.groupby("flight", sort=False):
        during = group.loc[group["label_any"] > 0]
        if during.empty:
            continue
        durations.append((str(flight), float(during["t_end"].max() - during["t_start"].min())))
    if not durations:
        return []
    values = np.asarray([duration for _, duration in durations], dtype=float)
    selected = []
    for quantile in (0.25, 0.50, 0.75):
        target = float(np.quantile(values, quantile))
        flight = min(durations, key=lambda row: (abs(row[1] - target), row[0]))[0]
        if flight not in selected:
            selected.append(flight)
    return selected


def _plot_case(
    output: Path,
    flight: str,
    graph_name: str,
    period_means: dict[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    matrices = [
        period_means["before"], period_means["during"], period_means["after"],
        period_means["during"] - period_means["before"],
    ]
    titles = ["Before", "During", "After", "During - Before"]
    for axis, matrix, title in zip(axes, matrices, titles):
        image = axis.imshow(matrix, aspect="auto", cmap="coolwarm" if "-" in title else "viridis")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"{flight} | {graph_name}")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_ex07(
    source: SourceRun,
    output_root: Path,
    *,
    force: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    run_dir = Path(output_root) / "ex07" / source.dataset / f"seed_{source.seed}"
    done_path = run_dir / "DONE.json"
    config_hash = _config_hash(source) + ("_smoke" if smoke else "")
    if done_path.is_file() and not force:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("config_hash") == config_hash and done.get("source_signature") == source.source_signature:
            return {"status": "skipped_complete", **done}
    run_dir.mkdir(parents=True, exist_ok=True)
    if done_path.exists():
        done_path.unlink()
    checkpoint_hash_before = sha256_file(source.checkpoint)
    write_json(run_dir / "provenance.json", {
        "experiment": "ex07", "design_version": DESIGN_VERSION,
        "dataset": source.dataset, "model_seed": source.seed,
        "config_hash": config_hash, "source": source.to_dict(),
        "source_audit": audit_source(source), "environment": environment_payload(),
    })
    write_json(run_dir / "resolved_analysis_config.json", {
        "graphs": GRAPH_NAMES, "periods": PERIODS, "top_k_edges": 20,
        "case_quantiles": [0.25, 0.50, 0.75], "figure_dpi": 300,
        "requires_training": False,
    })
    try:
        loaded = LoadedSourceModel.load(source)
        residuals, graphs = loaded.score(
            loaded.failure_flights,
            capture_graph=True,
            description=f"EX-07 {source.dataset} seed={source.seed} graph capture",
        )
        if graphs is None:
            raise RuntimeError("Graph capture returned no arrays")
        scored = loaded.aggregate(residuals, "mean")
        original = pd.read_csv(source.sequence_scores)
        parity = original.merge(
            scored, on=["flight", "current_index"], suffixes=("_source", "_capture"), validate="one_to_one"
        )
        if len(parity) != len(original):
            raise RuntimeError("Graph capture parity merge is incomplete")
        parity_error = float(np.max(np.abs(parity["total_score_source"] - parity["total_score_capture"])))
        if parity_error > 5e-5:
            raise RuntimeError(f"Graph capture score parity failed: {parity_error}")

        labeled = attach_window_labels(
            scored_df=scored,
            labels_root=loaded.cfg.labels_root,
            time_offset_sec=loaded.cfg.failure_label_time_offset_sec,
        )
        labeled.to_csv(run_dir / "graph_windows_index_with_labels.csv", index=False, encoding="utf-8-sig")
        np.savez_compressed(run_dir / "graph_windows.npz", **graphs)

        metric_rows: list[dict[str, Any]] = []
        edge_rows: list[dict[str, Any]] = []
        coverage_rows: list[dict[str, Any]] = []
        period_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        for flight, group in labeled.groupby("flight", sort=False):
            indices = _period_indices(group)
            if indices is None:
                coverage_rows.append({
                    "dataset": source.dataset, "seed": source.seed, "flight": str(flight),
                    "before_windows": 0, "during_windows": 0, "after_windows": 0,
                    "status": "no_labeled_anomaly_windows",
                })
                continue
            coverage_rows.append({
                "dataset": source.dataset, "seed": source.seed, "flight": str(flight),
                "before_windows": len(indices.get("before", [])),
                "during_windows": len(indices["during"]),
                "after_windows": len(indices.get("after", [])),
                "status": "complete_before_during_after" if set(PERIODS) <= set(indices) else (
                    "after_unavailable_failure_persists_to_flight_end"
                    if "after" not in indices else "before_unavailable"
                ),
            })
            score_by_graph_row = group.set_index("graph_row")["total_score"]
            for graph_name in GRAPH_NAMES:
                graph = np.asarray(graphs[graph_name], dtype=np.float64)
                means = {period: graph[index].mean(axis=0) for period, index in indices.items()}
                period_cache[(str(flight), graph_name)] = means
                reference_period = "before" if "before" in means else "during"
                during_distance = np.linalg.norm(
                    graph[indices["during"]] - means[reference_period][None, :, :], axis=(1, 2)
                )
                during_scores = score_by_graph_row.reindex(indices["during"]).to_numpy(dtype=float)
                for comparison, left, right in (
                    ("during_minus_before", "during", "before"),
                    ("after_minus_during", "after", "during"),
                    ("after_minus_before", "after", "before"),
                ):
                    if left not in means or right not in means:
                        continue
                    difference = means[left] - means[right]
                    metric_rows.append({
                        "dataset": source.dataset, "seed": source.seed, "flight": str(flight),
                        "graph": graph_name, "comparison": comparison,
                        "before_windows": len(indices.get("before", [])),
                        "during_windows": len(indices["during"]),
                        "after_windows": len(indices.get("after", [])),
                        "frobenius_change": float(np.linalg.norm(difference)),
                        "mean_absolute_change": float(np.mean(np.abs(difference))),
                        "top20_jaccard": float(_jaccard(means[left], means[right], loaded.nodes, 20)),
                        "left_entropy": _entropy(means[left]),
                        "right_entropy": _entropy(means[right]),
                        "entropy_change": _entropy(means[left]) - _entropy(means[right]),
                        "graph_score_spearman_during": _safe_spearman(during_distance, during_scores),
                    })
                for period in PERIODS:
                    if period not in means:
                        continue
                    for row in _top_edges(means[period], loaded.nodes, 20):
                        edge_rows.append({
                            "dataset": source.dataset, "seed": source.seed, "flight": str(flight),
                            "graph": graph_name, "period": period, **row,
                            "mechanism_status": "待领域核验", "mechanism_evidence": "",
                        })

        metrics = pd.DataFrame(metric_rows)
        edges = pd.DataFrame(edge_rows)
        coverage = pd.DataFrame(coverage_rows)
        if coverage.empty or coverage["during_windows"].sum() == 0 or edges.empty:
            raise RuntimeError("Graph analysis produced no labeled flight results")
        metrics.to_csv(run_dir / "per_flight_graph_metrics.csv", index=False, encoding="utf-8-sig")
        edges.to_csv(run_dir / "top_edges_for_physics_review.csv", index=False, encoding="utf-8-sig")
        coverage.to_csv(run_dir / "flight_period_coverage.csv", index=False, encoding="utf-8-sig")

        selected = _select_cases(labeled)
        if smoke:
            selected = selected[:1]
        for flight in selected:
            for graph_name in GRAPH_NAMES:
                means = period_cache.get((flight, graph_name))
                if means is not None and set(PERIODS) <= set(means):
                    _plot_case(
                        run_dir / "figures" / f"{flight}__{graph_name}.png",
                        flight, graph_name, means,
                    )
        write_json(run_dir / "case_selection.json", {
            "rule": "nearest flights to failure-duration quantiles 0.25, 0.50, 0.75",
            "selected_flights": selected,
        })
        done = {
            "status": "complete", "experiment": "ex07", "dataset": source.dataset,
            "seed": source.seed, "config_hash": config_hash,
            "source_signature": source.source_signature,
            "source_checkpoint_sha256": source.checkpoint_sha256,
            "score_parity_max_abs_error": parity_error,
            "graph_rows": len(labeled), "metric_rows": len(metrics), "edge_rows": len(edges),
            "period_complete_flights": int((coverage["status"] == "complete_before_during_after").sum()),
            "period_incomplete_flights": int((coverage["status"] != "complete_before_during_after").sum()),
            "selected_cases": selected, "environment": environment_payload(),
        }
        checkpoint_hash_after = sha256_file(source.checkpoint)
        if checkpoint_hash_after != checkpoint_hash_before:
            raise RuntimeError(f"Source checkpoint changed during EX-07: {source.checkpoint}")
        write_json(done_path, done)
        failed = run_dir / "FAILED.json"
        if failed.exists():
            failed.unlink()
        return done
    except Exception as exc:
        write_json(run_dir / "FAILED.json", {
            "status": "failed", "experiment": "ex07", "dataset": source.dataset,
            "seed": source.seed, "config_hash": config_hash, "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        raise
