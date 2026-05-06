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
    value_norm = str(value).strip().lower()
    if value_norm in {"1", "true", "yes", "y", "on"}:
        return True
    if value_norm in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _default_rate_groups() -> dict[str, list[str]]:
    return {
        "high_rate": [
            "roll_",
            "pitch_",
            "yaw_",
            "angular_velocity",
            "linear_velocity",
            "throttle",
            "climb",
        ],
        "mid_rate": [
            "heading_",
            "groundspeed",
        ],
        "low_rate": [
            "altitude",
        ],
    }


def _env_rate_groups(name: str, default: dict[str, list[str]]) -> dict[str, list[str]]:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return {str(key): [str(item) for item in items] for key, items in default.items()}
    payload = json.loads(str(value))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object mapping group name to pattern list")
    out: dict[str, list[str]] = {}
    for key, items in payload.items():
        if not isinstance(items, (list, tuple)):
            raise ValueError(f"{name}[{key}] must be a list")
        out[str(key)] = [str(item) for item in items if str(item).strip() != ""]
    return out


@dataclass
class USADConfig:
    dataset_name: str = _env_str("UAV_USAD_DATASET", "alfa")
    data_root: Path | None = None
    run_root: Path | None = None
    labels_root: Path | None = None
    history_steps: int = _env_int("UAV_USAD_HISTORY_STEPS", _env_int("UAV_HISTORY_STEPS", 256))
    future_steps: int = _env_int("UAV_USAD_FUTURE_STEPS", _env_int("UAV_FUTURE_STEPS", 8))
    sample_stride: int = _env_int("UAV_USAD_SAMPLE_STRIDE", 8)
    trim_leading_sec: float | None = None
    use_replication_padding: bool = _env_bool("UAV_USAD_USE_REPLICATION_PADDING", True)
    failure_label_time_offset_sec: float | None = None

    encoder_hidden_dims: tuple[int, int] = (
        _env_int("UAV_USAD_ENCODER_H1", 512),
        _env_int("UAV_USAD_ENCODER_H2", 256),
    )
    latent_dim: int = _env_int("UAV_USAD_LATENT_DIM", 96)
    decoder_hidden_dims: tuple[int, int] = (
        _env_int("UAV_USAD_DECODER_H1", 256),
        _env_int("UAV_USAD_DECODER_H2", 512),
    )
    activation: str = _env_str("UAV_USAD_ACTIVATION", "gelu")
    dropout: float = _env_float("UAV_USAD_DROPOUT", 0.10)
    use_layernorm: bool = _env_bool("UAV_USAD_USE_LAYERNORM", False)

    batch_size: int = _env_int("UAV_USAD_BATCH_SIZE", 1024)
    num_workers: int = _env_int("UAV_USAD_NUM_WORKERS", 0)
    lr: float = _env_float("UAV_USAD_LR", 1e-3)
    weight_decay: float = _env_float("UAV_USAD_WEIGHT_DECAY", 1e-4)
    num_epochs: int = _env_int("UAV_USAD_NUM_EPOCHS", 30)
    early_stop_patience: int = _env_int("UAV_USAD_EARLY_STOP_PATIENCE", 8)
    early_stop_min_delta: float = _env_float("UAV_USAD_EARLY_STOP_MIN_DELTA", 1e-4)
    grad_clip: float = _env_float("UAV_USAD_GRAD_CLIP", 1.0)
    seed: int = _env_int("UAV_USAD_SEED", 42)

    alpha: float = _env_float("UAV_USAD_ALPHA", 0.5)
    threshold_fit_mode: str = _env_str("UAV_USAD_THRESHOLD_FIT_MODE", "val_normal_sigma3")
    threshold_sigma_k: float = _env_float("UAV_USAD_THRESHOLD_SIGMA_K", 3.0)
    threshold_mad_k: float = _env_float("UAV_USAD_THRESHOLD_MAD_K", 4.0)
    threshold_quantile: float = _env_float("UAV_USAD_THRESHOLD_QUANTILE", 0.995)
    threshold_smooth_alpha: float = _env_float("UAV_USAD_THRESHOLD_SMOOTH_ALPHA", 0.20)
    static_threshold_p: int = _env_int("UAV_USAD_STATIC_THRESHOLD_P", 1000)
    static_threshold_label_col: str = _env_str("UAV_USAD_STATIC_THRESHOLD_LABEL_COL", "label_any")
    dynamic_threshold_history: int = _env_int("UAV_USAD_DYNAMIC_THRESHOLD_HISTORY", 128)
    dynamic_threshold_z_values: tuple[int, ...] = _env_int_list(
        "UAV_USAD_DYNAMIC_THRESHOLD_Z_VALUES",
        (2, 3, 4, 5, 6, 7, 8, 9, 10),
    )
    dynamic_threshold_warmup_pred: int = _env_int("UAV_USAD_DYNAMIC_THRESHOLD_WARMUP_PRED", 0)

    infer_output_name: str = _env_str("UAV_USAD_INFER_OUTPUT_NAME", "infer_usad_global_threshold")
    plot_scores: bool = _env_bool("UAV_USAD_PLOT_SCORES", True)
    plot_compare_timelines: bool = _env_bool("UAV_USAD_PLOT_COMPARE_TIMELINES", True)
    plot_max_flights: int = _env_int("UAV_USAD_PLOT_MAX_FLIGHTS", 0)
    device: str = _env_str("UAV_USAD_DEVICE", "auto")
    normalization_stats_path: Path | None = None
    rate_groups: dict[str, list[str]] = None

    def __post_init__(self):
        self.dataset_name = normalize_dataset_name(self.dataset_name)

        env_data_root = os.getenv("UAV_USAD_DATA_ROOT")
        self.data_root = Path(env_data_root) if env_data_root else (
            dataset_root_from_name(PORTABLE_ROOT, self.dataset_name)
            if self.data_root is None
            else Path(self.data_root)
        )
        manifest = load_dataset_manifest(self.data_root)

        env_labels_root = os.getenv("UAV_USAD_LABELS_ROOT")
        if env_labels_root:
            self.labels_root = Path(env_labels_root)
        else:
            self.labels_root = discover_labels_root(self.data_root, manifest=manifest) if self.labels_root is None else Path(self.labels_root)

        env_run_root = os.getenv("UAV_USAD_RUN_ROOT")
        self.run_root = (
            Path(env_run_root)
            if env_run_root
            else (BUNDLE_ROOT / "runs" / f"usad_{self.dataset_name}" if self.run_root is None else Path(self.run_root))
        )

        env_trim = os.getenv("UAV_USAD_TRIM_LEADING_SEC")
        self.trim_leading_sec = float(env_trim) if env_trim is not None else float(
            manifest.get("trim_leading_sec", 0.0) if self.trim_leading_sec is None else self.trim_leading_sec
        )

        env_offset = os.getenv("UAV_USAD_FAILURE_LABEL_TIME_OFFSET_SEC")
        self.failure_label_time_offset_sec = float(env_offset) if env_offset is not None else float(
            manifest.get("failure_label_time_offset_sec", 0.0)
            if self.failure_label_time_offset_sec is None
            else self.failure_label_time_offset_sec
        )

        env_norm_stats_path = os.getenv("UAV_USAD_NORMALIZATION_STATS_PATH")
        if env_norm_stats_path:
            self.normalization_stats_path = Path(env_norm_stats_path)
        elif self.normalization_stats_path is None:
            self.normalization_stats_path = self.run_root / "normalization_stats.json"
        else:
            self.normalization_stats_path = Path(self.normalization_stats_path)

        self.history_steps = max(int(self.history_steps), 1)
        self.future_steps = max(int(self.future_steps), 1)
        self.sample_stride = max(int(self.sample_stride), 1)
        self.batch_size = max(int(self.batch_size), 1)
        self.num_workers = max(int(self.num_workers), 0)
        self.num_epochs = max(int(self.num_epochs), 1)
        self.early_stop_patience = max(int(self.early_stop_patience), 1)
        self.alpha = float(min(max(self.alpha, 0.0), 1.0))
        self.threshold_quantile = float(min(max(self.threshold_quantile, 0.5), 0.999999))
        self.threshold_smooth_alpha = float(min(max(self.threshold_smooth_alpha, 0.0), 1.0))
        self.static_threshold_p = max(int(self.static_threshold_p), 1)
        self.dynamic_threshold_history = max(int(self.dynamic_threshold_history), 1)
        self.dynamic_threshold_z_values = tuple(int(z) for z in self.dynamic_threshold_z_values)
        if len(self.dynamic_threshold_z_values) <= 0:
            self.dynamic_threshold_z_values = (2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.dynamic_threshold_warmup_pred = int(1 if int(self.dynamic_threshold_warmup_pred) else 0)
        self.rate_groups = _env_rate_groups("UAV_USAD_RATE_GROUPS", _default_rate_groups()) if self.rate_groups is None else self.rate_groups

    @property
    def window_size(self) -> int:
        return int(self.history_steps + self.future_steps)

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
        payload["labels_root"] = str(self.labels_root)
        payload["run_root"] = str(self.run_root)
        payload["normalization_stats_path"] = str(self.normalization_stats_path)
        payload["csv_root"] = str(self.csv_root)
        payload["dataset_manifest_path"] = str(self.dataset_manifest_path)
        payload["window_size"] = int(self.window_size)
        return payload

    def save(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
