from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .paths import PROTOCOL_PATH, RESULTS_ROOT, ensure_import_paths


DATASETS = ("alfa", "gpsdata", "simulate")


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RevisionConfig:
    protocol: str
    experiment_id: str
    dataset: str
    variant: str
    data_split_seed: int = 64
    model_seed: int = 0
    epochs: int = 100
    lookback: int = 128
    horizon: int = 4
    stride: int = 4
    batch_size: int = 128
    d_model: int = 64
    tcn_layers: int = 5
    tcn_blocks: int = 4
    short_kernel: int = 5
    short_patch: int = 8
    dropout: float = 0.2
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-4
    graph_eta: float = 2.0
    graph_beta: float = 0.5
    graph_gate_init: float = 0.15
    graph_num_hops: int = 2
    interleave_every: int = 2
    cross_dim_loss_enabled: bool = True
    aggregator: str = "mean"
    ema_alpha: float = 0.25
    threshold_method: str = "spot"
    graph_max_points_per_pair: int = 200000
    smoke: bool = False
    corruption_kind: str = "none"
    corruption_level: float = 0.0
    output_root: str = str(RESULTS_ROOT)

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        if self.model_seed < 0:
            raise ValueError("model_seed must be non-negative")
        if self.aggregator not in {
            "mean", "max", "topk_1", "topk_3", "topk_5", "quantile_90", "quantile_95"
        }:
            raise ValueError(f"Unsupported aggregator: {self.aggregator}")

    @property
    def run_dir(self) -> Path:
        return (
            Path(self.output_root) / self.protocol / self.experiment_id / self.dataset
            / self.variant / f"seed_{self.model_seed}"
        )

    @property
    def shared_graph_dir(self) -> Path:
        return Path(self.output_root) / self.protocol / "_cache" / self.dataset / "graph"

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_dir"] = str(self.run_dir)
        payload["shared_graph_dir"] = str(self.shared_graph_dir)
        payload["config_hash"] = self.config_hash
        return payload

    def to_legacy(self):
        ensure_import_paths()
        from tcngatreconfig import TCNGATREConfig

        cfg = TCNGATREConfig(
            dataset_name=self.dataset,
            run_root=self.run_dir,
            graph_dir=self.shared_graph_dir,
            normalization_stats_path=self.run_dir / "normalization_stats.json",
            lookback=self.lookback,
            horizon_out=self.horizon,
            sample_stride=self.stride,
            batch_size=self.batch_size,
            split_seed=self.data_split_seed,
            num_epochs=self.epochs,
            d_model=self.d_model,
            tcn_layers=self.tcn_layers,
            tcn_blocks=self.tcn_blocks,
            short_kernel=self.short_kernel,
            short_patch=self.short_patch,
            dropout=self.dropout,
            early_stop_patience=self.early_stop_patience,
            early_stop_min_delta=self.early_stop_min_delta,
            graph_eta=self.graph_eta,
            graph_beta=self.graph_beta,
            graph_gate_init=self.graph_gate_init,
            graph_num_hops=self.graph_num_hops,
            interleave_every=self.interleave_every,
            cross_dim_loss_enabled=self.cross_dim_loss_enabled,
            score_temporal_smooth_alpha=self.ema_alpha,
            graph_max_points_per_pair=self.graph_max_points_per_pair,
            graph_num_workers=1 if self.smoke else 4,
            plot_scores=False,
            num_workers=0,
            device="auto",
        )
        return cfg


def make_config(
    experiment_id: str,
    dataset: str,
    variant: str,
    seed: int,
    smoke: bool = False,
    protocol: dict[str, Any] | None = None,
) -> RevisionConfig:
    spec = load_protocol() if protocol is None else protocol
    training = spec["smoke" if smoke else "training"]
    aggregator = variant if experiment_id == "ex05" else "mean"
    corruption_kind = "none"
    corruption_level = 0.0
    if experiment_id == "ex08":
        prefix, raw = variant.rsplit("_", 1)
        corruption_kind = prefix
        corruption_level = float(raw)
    return RevisionConfig(
        protocol=str(spec["protocol"]),
        experiment_id=experiment_id,
        dataset=dataset,
        variant=variant,
        data_split_seed=int(spec["data_split_seed"]),
        model_seed=int(seed),
        epochs=int(training["epochs"]),
        lookback=int(training["lookback"]),
        horizon=int(training["horizon"]),
        stride=int(training["stride"]),
        batch_size=int(training["batch_size"]),
        d_model=int(training["d_model"]),
        tcn_layers=int(training["tcn_layers"]),
        tcn_blocks=int(training["tcn_blocks"]),
        short_kernel=int(training["short_kernel"]),
        short_patch=int(training["short_patch"]),
        dropout=float(training["dropout"]),
        early_stop_patience=int(training.get("early_stop_patience", 5)),
        early_stop_min_delta=float(training.get("early_stop_min_delta", 1e-4)),
        graph_max_points_per_pair=int(training.get("graph_max_points_per_pair", 200000)),
        cross_dim_loss_enabled=variant not in {"no_cross_dim"},
        aggregator=aggregator,
        smoke=bool(smoke),
        corruption_kind=corruption_kind,
        corruption_level=corruption_level,
    )
