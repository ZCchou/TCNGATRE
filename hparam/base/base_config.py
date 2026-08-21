from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from data.alfa_shared import (
    MANIFEST_NAME,
    dataset_root_from_name,
    discover_labels_root,
    discover_wide_root,
    load_dataset_manifest,
    normalize_dataset_name,
)


BUNDLE_ROOT = Path(__file__).resolve().parent        # ablation/base/
PORTABLE_ROOT = BUNDLE_ROOT.parent.parent             # F:/uavdetection/


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None else str(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_optional_float(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _env_int_list(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return tuple(int(item) for item in default)
    return tuple(int(p.strip()) for p in str(value).split(",") if p.strip())


@dataclass
class TCNGATREConfig:
    dataset_name: str = _env_str("UAV_TCNGATRE_DATASET", "alfa")
    data_root: Path | None = None
    labels_root: Path | None = None
    run_root: Path | None = None
    split_info_path: Path | None = None
    graph_dir: Path | None = None
    normalization_stats_path: Path | None = None

    # Window params
    lookback: int = _env_int("UAV_TCNGATRE_LOOKBACK", 128)
    horizon_out: int = _env_int("UAV_TCNGATRE_HORIZON_OUT", 4)
    sample_stride: int = _env_int("UAV_TCNGATRE_SAMPLE_STRIDE", 4)
    trim_leading_sec: float = 0.0
    failure_label_time_offset_sec: float | None = None
    use_replication_padding: bool = _env_bool("UAV_TCNGATRE_USE_REPLICATION_PADDING", False)

    # Training
    batch_size: int = _env_int("UAV_TCNGATRE_BATCH_SIZE", 128)
    val_ratio: float = _env_float("UAV_TCNGATRE_VAL_RATIO", 0.2)
    split_seed: int = _env_int("UAV_TCNGATRE_SPLIT_SEED", 64)
    lr: float = _env_float("UAV_TCNGATRE_LR", 5e-4)
    weight_decay: float = _env_float("UAV_TCNGATRE_WEIGHT_DECAY", 3e-4)
    num_epochs: int = _env_int("UAV_TCNGATRE_NUM_EPOCHS", 100)
    grad_clip: float = _env_float("UAV_TCNGATRE_GRAD_CLIP", 5.0)
    early_stop_patience: int = _env_int("UAV_TCNGATRE_EARLY_STOP_PATIENCE", 5)
    early_stop_min_delta: float = _env_float("UAV_TCNGATRE_EARLY_STOP_MIN_DELTA", 1e-4)
    loss_type: str = _env_str("UAV_TCNGATRE_LOSS_TYPE", "huber")
    huber_beta: float = _env_float("UAV_TCNGATRE_HUBER_BETA", 1.0)

    # Model architecture
    d_model: int = _env_int("UAV_TCNGATRE_D_MODEL", 64)
    tcn_layers: int = _env_int("UAV_TCNGATRE_TCN_LAYERS", 5)
    tcn_blocks: int = _env_int("UAV_TCNGATRE_TCN_BLOCKS", 4)
    short_kernel: int = _env_int("UAV_TCNGATRE_SHORT_KERNEL", 5)
    short_patch: int = _env_int("UAV_TCNGATRE_SHORT_PATCH", 8)
    dropout: float = _env_float("UAV_TCNGATRE_DROPOUT", 0.20)

    # Graph correction params
    graph_eta: float = _env_float("UAV_TCNGATRE_GRAPH_ETA", 2.0)
    graph_beta: float = _env_float("UAV_TCNGATRE_GRAPH_BETA", 0.5)
    graph_gate_init: float = _env_float("UAV_TCNGATRE_GRAPH_GATE_INIT", 0.15)
    interleave_every: int = _env_int("UAV_TCNGATRE_INTERLEAVE_EVERY", 2)
    graph_num_hops: int = _env_int("UAV_TCNGATRE_GRAPH_NUM_HOPS", 2)

    # Cross-dimension auxiliary loss
    cross_dim_loss_enabled: bool = _env_bool("UAV_TCNGATRE_CROSS_DIM_LOSS_ENABLED", True)
    cross_dim_lambda: float = _env_float("UAV_TCNGATRE_CROSS_DIM_LAMBDA", 1.0)
    cross_dim_dropout_prob: float = _env_float("UAV_TCNGATRE_CROSS_DIM_DROPOUT_PROB", 0.35)
    cross_dim_max_mask_ratio: float = _env_float("UAV_TCNGATRE_CROSS_DIM_MAX_MASK_RATIO", 0.50)

    # Graph building params
    graph_grid_sec: float = _env_float("UAV_TCNGATRE_GRAPH_GRID_SEC", 0.1)
    graph_mic_alpha: float = _env_float("UAV_TCNGATRE_GRAPH_MIC_ALPHA", 0.6)
    graph_mic_c: int = _env_int("UAV_TCNGATRE_GRAPH_MIC_C", 15)
    graph_min_overlap: int = _env_int("UAV_TCNGATRE_GRAPH_MIN_OVERLAP", 128)
    graph_mic_threshold: float = _env_float("UAV_TCNGATRE_GRAPH_MIC_THRESHOLD", 0.0)
    graph_max_points_per_pair: int = _env_int("UAV_TCNGATRE_GRAPH_MAX_POINTS_PER_PAIR", 200000)
    graph_num_workers: int = _env_int("UAV_TCNGATRE_GRAPH_NUM_WORKERS", 4)
    graph_overwrite: bool = _env_bool("UAV_TCNGATRE_GRAPH_OVERWRITE", False)

    # Inference
    num_workers: int = _env_int("UAV_TCNGATRE_NUM_WORKERS", 0)
    device: str = _env_str("UAV_TCNGATRE_DEVICE", "auto")
    infer_output_name: str = _env_str("UAV_TCNGATRE_INFER_OUTPUT_NAME", "infer_tcngatre_failure")
    infer_source_split: str = _env_str("UAV_TCNGATRE_INFER_SOURCE_SPLIT", "Failure")
    infer_checkpoint_name: str = _env_str("UAV_TCNGATRE_INFER_CHECKPOINT_NAME", "best.pt")
    infer_flight_filter_role: str = _env_str("UAV_TCNGATRE_INFER_FLIGHT_FILTER_ROLE", "")
    plot_scores: bool = _env_bool("UAV_TCNGATRE_PLOT_SCORES", True)
    plot_max_flights: int = _env_int("UAV_TCNGATRE_PLOT_MAX_FLIGHTS", 0)

    # Input EMA
    use_input_ema: bool = _env_bool("UAV_TCNGATRE_USE_INPUT_EMA", False)
    input_ema_alpha: float = _env_float("UAV_TCNGATRE_INPUT_EMA_ALPHA", 0.25)

    # Score EMA
    score_ema_enabled: bool = _env_bool("UAV_TCNGATRE_SCORE_EMA_ENABLED", True)
    score_temporal_smooth_alpha: float = _env_float("UAV_TCNGATRE_SCORE_TEMPORAL_SMOOTH_ALPHA", 0.25)

    # Threshold
    threshold_sigma_k: float = _env_float("UAV_TCNGATRE_THRESHOLD_SIGMA_K", 3.0)
    threshold_mad_k: float = _env_float("UAV_TCNGATRE_THRESHOLD_MAD_K", 4.0)
    threshold_quantile: float = _env_float("UAV_TCNGATRE_THRESHOLD_QUANTILE", 0.995)
    threshold_smooth_alpha: float = _env_float("UAV_TCNGATRE_THRESHOLD_SMOOTH_ALPHA", 0.20)
    static_threshold_p: int = _env_int("UAV_TCNGATRE_STATIC_THRESHOLD_P", 1000)
    static_threshold_label_col: str = _env_str("UAV_TCNGATRE_STATIC_THRESHOLD_LABEL_COL", "label_any")
    dynamic_threshold_history: int = _env_int("UAV_TCNGATRE_DYNAMIC_THRESHOLD_HISTORY", 128)
    dynamic_threshold_z_values: tuple[int, ...] = _env_int_list(
        "UAV_TCNGATRE_DYNAMIC_THRESHOLD_Z_VALUES",
        (2, 3, 4, 5, 6, 7, 8, 9, 10),
    )
    dynamic_threshold_warmup_pred: int = _env_int("UAV_TCNGATRE_DYNAMIC_THRESHOLD_WARMUP_PRED", 0)

    def __post_init__(self):
        # Dataclass field defaults with _env_*() are evaluated once at class-definition time
        # (import time), so any os.environ.setdefault() called after import has no effect on
        # those defaults. Re-read all overridable scalar params here so resolve_config()
        # setdefault() calls are picked up correctly at instance-creation time.
        _ri = [
            ("UAV_TCNGATRE_LOOKBACK", "lookback", int),
            ("UAV_TCNGATRE_HORIZON_OUT", "horizon_out", int),
            ("UAV_TCNGATRE_SAMPLE_STRIDE", "sample_stride", int),
            ("UAV_TCNGATRE_BATCH_SIZE", "batch_size", int),
            ("UAV_TCNGATRE_NUM_EPOCHS", "num_epochs", int),
            ("UAV_TCNGATRE_EARLY_STOP_PATIENCE", "early_stop_patience", int),
            ("UAV_TCNGATRE_D_MODEL", "d_model", int),
            ("UAV_TCNGATRE_TCN_LAYERS", "tcn_layers", int),
            ("UAV_TCNGATRE_TCN_BLOCKS", "tcn_blocks", int),
            ("UAV_TCNGATRE_INTERLEAVE_EVERY", "interleave_every", int),
            ("UAV_TCNGATRE_GRAPH_NUM_HOPS", "graph_num_hops", int),
            ("UAV_TCNGATRE_SHORT_KERNEL", "short_kernel", int),
            ("UAV_TCNGATRE_SHORT_PATCH", "short_patch", int),
            ("UAV_TCNGATRE_LR", "lr", float),
            ("UAV_TCNGATRE_WEIGHT_DECAY", "weight_decay", float),
            ("UAV_TCNGATRE_DROPOUT", "dropout", float),
            ("UAV_TCNGATRE_GRAD_CLIP", "grad_clip", float),
            ("UAV_TCNGATRE_GRAPH_ETA", "graph_eta", float),
            ("UAV_TCNGATRE_GRAPH_BETA", "graph_beta", float),
            ("UAV_TCNGATRE_GRAPH_GATE_INIT", "graph_gate_init", float),
            ("UAV_TCNGATRE_CROSS_DIM_LAMBDA", "cross_dim_lambda", float),
            ("UAV_TCNGATRE_CROSS_DIM_DROPOUT_PROB", "cross_dim_dropout_prob", float),
            ("UAV_TCNGATRE_CROSS_DIM_MAX_MASK_RATIO", "cross_dim_max_mask_ratio", float),
        ]
        for _env, _attr, _cast in _ri:
            _v = os.getenv(_env)
            if _v is not None:
                setattr(self, _attr, _cast(_v))
        _v = os.getenv("UAV_TCNGATRE_CROSS_DIM_LOSS_ENABLED")
        if _v is not None:
            self.cross_dim_loss_enabled = _v.strip().lower() in {"1", "true", "yes", "y", "on"}

        self.dataset_name = normalize_dataset_name(self.dataset_name)

        env_data_root = os.getenv("UAV_TCNGATRE_DATA_ROOT")
        self.data_root = Path(env_data_root) if env_data_root else (
            dataset_root_from_name(PORTABLE_ROOT, self.dataset_name)
            if self.data_root is None else Path(self.data_root)
        )
        manifest = load_dataset_manifest(self.data_root)

        env_labels_root = os.getenv("UAV_TCNGATRE_LABELS_ROOT")
        self.labels_root = (
            Path(env_labels_root) if env_labels_root
            else (discover_labels_root(self.data_root, manifest=manifest) if self.labels_root is None
                  else Path(self.labels_root))
        )

        env_run_root = os.getenv("UAV_TCNGATRE_RUN_ROOT")
        self.run_root = (
            Path(env_run_root) if env_run_root
            else (BUNDLE_ROOT / "runs" / f"tcngatre_{self.dataset_name}" if self.run_root is None
                  else Path(self.run_root))
        )

        env_split_info = os.getenv("UAV_TCNGATRE_SPLIT_INFO_PATH")
        self.split_info_path = (
            Path(env_split_info) if env_split_info
            else (self.data_root / MANIFEST_NAME if self.split_info_path is None
                  else Path(self.split_info_path))
        )

        env_graph_dir = os.getenv("UAV_TCNGATRE_GRAPH_DIR")
        self.graph_dir = (
            Path(env_graph_dir) if env_graph_dir
            else (self.run_root / "graph" if self.graph_dir is None
                  else Path(self.graph_dir))
        )

        env_norm = os.getenv("UAV_TCNGATRE_NORMALIZATION_STATS_PATH")
        self.normalization_stats_path = (
            Path(env_norm) if env_norm
            else (self.run_root / "normalization_stats.json" if self.normalization_stats_path is None
                  else Path(self.normalization_stats_path))
        )

        self.failure_label_time_offset_sec = float(
            manifest.get("failure_label_time_offset_sec", 0.0)
            if self.failure_label_time_offset_sec is None
            else self.failure_label_time_offset_sec
        )

        # Clamps
        self.lookback = max(int(self.lookback), 1)
        self.horizon_out = max(int(self.horizon_out), 1)
        self.sample_stride = max(int(self.sample_stride), 1)
        self.batch_size = max(int(self.batch_size), 1)
        self.num_epochs = max(int(self.num_epochs), 1)
        self.num_workers = max(int(self.num_workers), 0)
        self.early_stop_patience = max(int(self.early_stop_patience), 1)
        self.static_threshold_p = max(int(self.static_threshold_p), 1)
        self.dynamic_threshold_history = max(int(self.dynamic_threshold_history), 1)
        self.dynamic_threshold_warmup_pred = int(1 if int(self.dynamic_threshold_warmup_pred) else 0)
        self.dynamic_threshold_z_values = tuple(int(z) for z in self.dynamic_threshold_z_values) or (2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.score_temporal_smooth_alpha = float(min(max(self.score_temporal_smooth_alpha, 0.0), 1.0))
        self.input_ema_alpha = float(min(max(self.input_ema_alpha, 0.0), 1.0))
        self.threshold_quantile = float(min(max(self.threshold_quantile, 0.5), 0.999999))
        self.threshold_smooth_alpha = float(min(max(self.threshold_smooth_alpha, 0.0), 1.0))
    @property
    def dataset_manifest_path(self) -> Path:
        return self.data_root / MANIFEST_NAME

    @property
    def wide_root(self) -> Path:
        return discover_wide_root(self.data_root)

    @property
    def graph_input_dir(self) -> Path:
        return self.wide_root / "No_Failure"

    @property
    def dataset_mode(self) -> str:
        if self.dataset_name == "simulate":
            return "simulate"
        if self.dataset_name == "gpsdata":
            return "gpsdata"
        return "set_a_causal"

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        payload["data_root"] = str(self.data_root)
        payload["labels_root"] = str(self.labels_root)
        payload["run_root"] = str(self.run_root)
        payload["split_info_path"] = str(self.split_info_path)
        payload["graph_dir"] = str(self.graph_dir)
        payload["normalization_stats_path"] = str(self.normalization_stats_path)
        payload["wide_root"] = str(self.wide_root)
        payload["graph_input_dir"] = str(self.graph_input_dir)
        payload["dataset_manifest_path"] = str(self.dataset_manifest_path)
        payload["dataset_mode"] = self.dataset_mode
        return payload

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
