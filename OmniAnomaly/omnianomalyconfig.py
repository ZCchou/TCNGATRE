"""OmniAnomaly configuration."""
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

BUNDLE_ROOT = Path(__file__).resolve().parent
PORTABLE_ROOT = BUNDLE_ROOT.parent


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None else str(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_int_list(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return tuple(int(item) for item in default)
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip() != "")


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


@dataclass
class OmniAnomalyConfig:
    dataset_name: str = _env_str("UAV_OA_DATASET", "alfa")
    data_root: Path | None = None
    split_info_path: Path | None = None
    labels_root: Path | None = None
    run_root: Path | None = None
    trim_leading_sec: float | None = None
    failure_label_time_offset_sec: float | None = None

    window_size: int = _env_int("UAV_OA_WINDOW_SIZE", 32)
    stride_train: int = _env_int("UAV_OA_STRIDE_TRAIN", 32)
    stride_val: int = _env_int("UAV_OA_STRIDE_VAL", 32)
    stride_test: int = _env_int("UAV_OA_STRIDE_TEST", 32)
    use_replication_padding: bool = _env_bool("UAV_OA_USE_REPLICATION_PADDING", True)
    context_mode: str = _env_str("UAV_OA_CONTEXT_MODE", "window")
    num_features: int = _env_int("UAV_OA_NUM_FEATURES", 0)

    rnn_hidden_dim: int = _env_int("UAV_OA_RNN_HIDDEN_DIM", 128)
    latent_dim: int = _env_int("UAV_OA_LATENT_DIM", 16)
    rnn_layers: int = _env_int("UAV_OA_RNN_LAYERS", 1)
    dropout: float = _env_float("UAV_OA_DROPOUT", 0.1)
    num_planar_flows: int = _env_int("UAV_OA_NUM_PLANAR_FLOWS", 10)
    decoder_rnn_hidden_dim: int = _env_int("UAV_OA_DECODER_RNN_HIDDEN_DIM", 128)

    batch_size: int = _env_int("UAV_OA_BATCH_SIZE", 1024)
    num_epochs: int = _env_int("UAV_OA_NUM_EPOCHS", 30)
    lr: float = _env_float("UAV_OA_LR", 1e-3)
    weight_decay: float = _env_float("UAV_OA_WEIGHT_DECAY", 1e-4)
    scheduler_step_size: int = _env_int("UAV_OA_SCHEDULER_STEP_SIZE", 5)
    scheduler_gamma: float = _env_float("UAV_OA_SCHEDULER_GAMMA", 0.5)
    early_stop_patience: int = _env_int("UAV_OA_EARLY_STOP_PATIENCE", 8)
    early_stop_min_delta: float = _env_float("UAV_OA_EARLY_STOP_MIN_DELTA", 1e-5)
    grad_clip: float = _env_float("UAV_OA_GRAD_CLIP", 5.0)
    num_workers: int = _env_int("UAV_OA_NUM_WORKERS", 0)
    seed: int = _env_int("UAV_OA_SEED", 42)
    device: str = _env_str("UAV_OA_DEVICE", "auto")
    kl_warmup_epochs: int = _env_int("UAV_OA_KL_WARMUP_EPOCHS", 5)
    mc_samples: int = _env_int("UAV_OA_MC_SAMPLES", 1)

    threshold_fit_mode: str = _env_str("UAV_OA_THRESHOLD_FIT_MODE", "val_normal_sigma3")
    threshold_sigma_k: float = _env_float("UAV_OA_THRESHOLD_SIGMA_K", 3.0)
    threshold_smooth_alpha: float = _env_float("UAV_OA_THRESHOLD_SMOOTH_ALPHA", 0.20)
    static_threshold_p: int = _env_int("UAV_OA_STATIC_THRESHOLD_P", 1000)
    static_threshold_label_col: str = _env_str("UAV_OA_STATIC_THRESHOLD_LABEL_COL", "label_any")
    dynamic_threshold_history: int = _env_int("UAV_OA_DYNAMIC_THRESHOLD_HISTORY", 128)
    dynamic_threshold_z_values: tuple[int, ...] = _env_int_list(
        "UAV_OA_DYNAMIC_THRESHOLD_Z_VALUES",
        (2, 3, 4, 5, 6, 7, 8, 9, 10),
    )
    dynamic_threshold_warmup_pred: int = _env_int("UAV_OA_DYNAMIC_THRESHOLD_WARMUP_PRED", 0)

    use_ema: bool = _env_bool("UAV_OA_USE_EMA", False)
    ema_alpha: float = _env_float("UAV_OA_EMA_ALPHA", 0.20)
    infer_output_name: str = _env_str("UAV_OA_INFER_OUTPUT_NAME", "infer_future_window_failure")
    infer_split: str = _env_str("UAV_OA_INFER_SPLIT", "failure")
    plot_scores: bool = _env_bool("UAV_OA_PLOT_SCORES", True)
    plot_compare_timelines: bool = _env_bool("UAV_OA_PLOT_COMPARE_TIMELINES", True)
    plot_max_flights: int = _env_int("UAV_OA_PLOT_MAX_FLIGHTS", 0)
    save_per_dim_error: bool = _env_bool("UAV_OA_SAVE_PER_DIM_ERROR", True)

    normalization_stats_path: Path | None = None

    def __post_init__(self):
        self.dataset_name = normalize_dataset_name(self.dataset_name)

        env_data_root = os.getenv("UAV_OA_DATA_ROOT")
        self.data_root = Path(env_data_root) if env_data_root else (
            dataset_root_from_name(PORTABLE_ROOT, self.dataset_name)
            if self.data_root is None
            else Path(self.data_root)
        )
        manifest = load_dataset_manifest(self.data_root)

        env_split_info_path = os.getenv("UAV_OA_SPLIT_INFO_PATH")
        self.split_info_path = (
            Path(env_split_info_path)
            if env_split_info_path
            else (self.data_root / MANIFEST_NAME if self.split_info_path is None else Path(self.split_info_path))
        )

        env_labels_root = os.getenv("UAV_OA_LABELS_ROOT")
        if env_labels_root:
            self.labels_root = Path(env_labels_root)
        else:
            self.labels_root = discover_labels_root(self.data_root, manifest=manifest) if self.labels_root is None else Path(self.labels_root)

        env_run_root = os.getenv("UAV_OA_RUN_ROOT")
        self.run_root = (
            Path(env_run_root)
            if env_run_root
            else (BUNDLE_ROOT / "runs" / f"omni_anomaly_{self.dataset_name}" if self.run_root is None else Path(self.run_root))
        )

        env_trim = os.getenv("UAV_OA_TRIM_LEADING_SEC")
        self.trim_leading_sec = float(env_trim) if env_trim is not None else float(
            manifest.get("trim_leading_sec", 0.0) if self.trim_leading_sec is None else self.trim_leading_sec
        )

        env_offset = os.getenv("UAV_OA_FAILURE_LABEL_TIME_OFFSET_SEC")
        self.failure_label_time_offset_sec = float(env_offset) if env_offset is not None else float(
            manifest.get("failure_label_time_offset_sec", 0.0)
            if self.failure_label_time_offset_sec is None
            else self.failure_label_time_offset_sec
        )

        env_norm_stats_path = os.getenv("UAV_OA_NORMALIZATION_STATS_PATH")
        if env_norm_stats_path:
            self.normalization_stats_path = Path(env_norm_stats_path)
        elif self.normalization_stats_path is None:
            self.normalization_stats_path = self.run_root / "normalization_stats.json"
        else:
            self.normalization_stats_path = Path(self.normalization_stats_path)

        self.window_size = max(int(self.window_size), 2)
        self.stride_train = max(int(self.stride_train), 1)
        self.stride_val = max(int(self.stride_val), 1)
        self.stride_test = max(int(self.stride_test), 1)
        self.batch_size = max(int(self.batch_size), 1)
        self.num_epochs = max(int(self.num_epochs), 1)
        self.rnn_layers = max(int(self.rnn_layers), 1)
        self.rnn_hidden_dim = max(int(self.rnn_hidden_dim), 4)
        self.latent_dim = max(int(self.latent_dim), 2)
        self.num_planar_flows = max(int(self.num_planar_flows), 0)
        self.mc_samples = max(int(self.mc_samples), 1)
        self.kl_warmup_epochs = max(int(self.kl_warmup_epochs), 0)
        self.grad_clip = max(float(self.grad_clip), 0.0)
        self.threshold_smooth_alpha = float(min(max(self.threshold_smooth_alpha, 0.0), 1.0))
        self.static_threshold_p = max(int(self.static_threshold_p), 1)
        self.dynamic_threshold_history = max(int(self.dynamic_threshold_history), 1)
        self.dynamic_threshold_z_values = tuple(int(z) for z in self.dynamic_threshold_z_values)
        if len(self.dynamic_threshold_z_values) <= 0:
            self.dynamic_threshold_z_values = (2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.dynamic_threshold_warmup_pred = int(1 if int(self.dynamic_threshold_warmup_pred) else 0)

    @property
    def csv_root(self) -> Path:
        return discover_wide_root(self.data_root)

    @property
    def dataset_manifest_path(self) -> Path:
        return self.data_root / MANIFEST_NAME

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        payload["data_root"] = str(self.data_root)
        payload["split_info_path"] = str(self.split_info_path)
        payload["labels_root"] = str(self.labels_root)
        payload["run_root"] = str(self.run_root)
        payload["normalization_stats_path"] = str(self.normalization_stats_path)
        payload["csv_root"] = str(self.csv_root)
        payload["dataset_manifest_path"] = str(self.dataset_manifest_path)
        return payload

    def save(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
