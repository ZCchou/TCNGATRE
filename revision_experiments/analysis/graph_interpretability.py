from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _entropy(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    values = np.maximum(values, 1e-12)
    return float(-np.mean(np.sum(values * np.log(values), axis=-1)))


def _top_edges(matrix: np.ndarray, node_names: list[str], k: int = 20) -> list[dict]:
    n = matrix.shape[0]
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            candidates.append((float(matrix[i, j]), i, j))
    candidates.sort(reverse=True)
    return [
        {"rank": rank + 1, "source": node_names[i], "target": node_names[j], "weight": weight}
        for rank, (weight, i, j) in enumerate(candidates[:k])
    ]


def analyze_graph_run(run_dir: Path, node_names: list[str]) -> dict:
    infer_dir = Path(run_dir) / "infer_tcngatre_failure"
    arrays = np.load(infer_dir / "graph_windows.npz")
    index = pd.read_csv(infer_dir / "graph_windows_index.csv")
    labeled = pd.read_csv(infer_dir / "score_threshold_analysis" / "sequence_scores_with_labels.csv")
    labels = labeled[["flight", "t_start", "label_any", "scores_smooth"]].copy()
    merged = index.merge(labels, on=["flight", "t_start"], how="left")
    merged["period"] = np.where(merged["label_any"].fillna(0).to_numpy() > 0, "during", "normal")

    output_dir = infer_dir / "graph_interpretability"
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict] = []
    edge_rows: list[dict] = []
    for graph_name in ("A_static", "A_dyn", "A_fuse"):
        graph = np.asarray(arrays[graph_name], dtype=np.float64)
        normal_idx = merged.index[merged["period"] == "normal"].to_numpy()
        anomaly_idx = merged.index[merged["period"] == "during"].to_numpy()
        if len(normal_idx) == 0 or len(anomaly_idx) == 0:
            continue
        normal_mean = graph[normal_idx].mean(axis=0)
        anomaly_mean = graph[anomaly_idx].mean(axis=0)
        difference = anomaly_mean - normal_mean
        normal_top = {(row["source"], row["target"]) for row in _top_edges(normal_mean, node_names, 20)}
        anomaly_top = {(row["source"], row["target"]) for row in _top_edges(anomaly_mean, node_names, 20)}
        union = normal_top | anomaly_top
        jaccard = 1.0 if not union else len(normal_top & anomaly_top) / len(union)
        metric_rows.append({
            "graph": graph_name,
            "normal_windows": len(normal_idx),
            "anomaly_windows": len(anomaly_idx),
            "frobenius_change": float(np.linalg.norm(difference)),
            "mean_absolute_change": float(np.mean(np.abs(difference))),
            "top20_jaccard": float(jaccard),
            "normal_entropy": _entropy(normal_mean),
            "anomaly_entropy": _entropy(anomaly_mean),
        })
        for period, matrix in (("normal", normal_mean), ("during", anomaly_mean)):
            for row in _top_edges(matrix, node_names, 20):
                edge_rows.append({"graph": graph_name, "period": period, **row, "mechanism_status": "待领域核验"})
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, matrix, title in zip(axes, (normal_mean, anomaly_mean, difference), ("Normal", "Anomaly", "Change")):
            image = ax.imshow(matrix, aspect="auto", cmap="coolwarm" if title == "Change" else "viridis")
            ax.set_title(f"{graph_name} | {title}")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / f"{graph_name}_normal_anomaly_change.png", dpi=300)
        plt.close(fig)

    metrics = pd.DataFrame(metric_rows)
    edges = pd.DataFrame(edge_rows)
    metrics.to_csv(output_dir / "graph_period_metrics.csv", index=False, encoding="utf-8")
    edges.to_csv(output_dir / "top_edges_for_physics_review.csv", index=False, encoding="utf-8-sig")
    summary = {"metric_rows": len(metrics), "edge_rows": len(edges), "output_dir": str(output_dir)}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
