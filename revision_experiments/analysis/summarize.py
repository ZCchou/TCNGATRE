from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from revision_experiments.core.paths import RESULTS_ROOT


SEED_PATTERN = re.compile(r"seed_(\d+)$")


def collect_primary_metrics(protocol: str, results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    rows: list[dict] = []
    root = Path(results_root) / protocol
    for path in root.glob("*/*/*/seed_*/infer_tcngatre_failure/score_threshold_analysis/primary_metrics.json"):
        seed_dir = path.parents[2]
        match = SEED_PATTERN.match(seed_dir.name)
        if match is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "experiment": path.parents[5].name,
            "dataset": path.parents[4].name,
            "variant": path.parents[3].name,
            "seed": int(match.group(1)),
            **payload,
            "source": str(path),
        })
    frame = pd.DataFrame(rows)
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "primary_metrics_all_runs.csv", index=False, encoding="utf-8")
    if not frame.empty and "f1" in frame:
        summary = frame.groupby(["experiment", "dataset", "variant"])[
            [column for column in ["precision", "recall", "f1", "fpr", "auroc", "average_precision"] if column in frame]
        ].agg(["mean", "std", "count"])
        summary.to_csv(output / "primary_metrics_seed_summary.csv", encoding="utf-8")
    return frame
