from __future__ import annotations

from pathlib import Path


DATASETS = ("alfa", "gpsdata", "simulate")
MODEL_SEEDS = (0, 1, 2, 3, 4)
EXPERIMENTS = ("ex05", "ex07", "ex08")
AGGREGATORS = ("mean", "max", "topk_1", "topk_3", "topk_5", "quantile_90", "quantile_95")
ROBUSTNESS_CONDITIONS = (
    "gaussian_0.01", "gaussian_0.05", "gaussian_0.10",
    "missing_0.10", "missing_0.20", "missing_0.30",
    "channel_dropout_1", "channel_dropout_3",
    "downsample_2", "downsample_4",
)
POSTHOC_PROTOCOL = "posthoc_v1"
PRIMARY_THRESHOLD = "spot"
PRIMARY_LABEL = "label_any"
EMA_ALPHA = 0.25
ANALYSIS_SEED = 20260821


def default_output_root(repo_root: Path) -> Path:
    return Path(repo_root) / "revision_results" / "protocol_v1" / POSTHOC_PROTOCOL
