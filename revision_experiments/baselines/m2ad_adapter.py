from __future__ import annotations

import importlib.util
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from revision_experiments.core.integrity import verify_snapshot
from revision_experiments.core.paths import BASELINE_SOURCE_PATH, EXTERNAL_ROOT
from revision_experiments.core.provenance import write_json

from .common_data import CommonDataBundle, adapter_config_hash, seed_everything, window_starts
from .finalize import finalize_run, record_failure
from .reproduction_utils import (
    augment_done,
    git_commit,
    validate_dataset_protocol,
    validate_failure_labels,
    write_split_metadata,
)


BASELINE = "m2ad"
CLASSIFICATION = "official_components_with_protocol_adapter"
SOURCE_ROOT = EXTERNAL_ROOT / BASELINE


def _load_symbol(path: Path, module_name: str, symbol: str):
    if not path.is_file():
        raise FileNotFoundError(
            f"M2AD official source is missing: {path}. Run fetch-baselines --baselines m2ad first."
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official M2AD module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)


def _official_components():
    model = _load_symbol(SOURCE_ROOT / "src" / "models" / "lstm.py", "m2ad_official_lstm", "Model")
    gmm = _load_symbol(SOURCE_ROOT / "src" / "models" / "gmm.py", "m2ad_official_gmm", "GMM")
    point_errors = _load_symbol(SOURCE_ROOT / "src" / "errors.py", "m2ad_official_errors", "point_errors")
    return model, gmm, point_errors


@dataclass(frozen=True)
class TrainMinMaxScaler:
    minimum: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        result = (np.asarray(values, dtype=np.float32) - self.minimum) / self.scale
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "minimum": self.minimum.tolist(),
            "scale": self.scale.tolist(),
            "feature_range": [0.0, 1.0],
            "source": "train_normal only",
        }


def _fit_minmax(bundle: CommonDataBundle) -> TrainMinMaxScaler:
    minimum = np.full(len(bundle.nodes), np.inf, dtype=np.float64)
    maximum = np.full(len(bundle.nodes), -np.inf, dtype=np.float64)
    for record in bundle.splits["train"]:
        _, values = bundle.load(record)
        minimum = np.minimum(minimum, np.min(values, axis=0))
        maximum = np.maximum(maximum, np.max(values, axis=0))
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise RuntimeError("M2AD could not fit finite train-only min/max statistics")
    scale = maximum - minimum
    scale[scale < 1e-8] = 1.0
    return TrainMinMaxScaler(minimum.astype(np.float32), scale.astype(np.float32))


class PredictiveDataset(Dataset):
    def __init__(self, bundle, split, scaler, window, stride, max_windows_per_flight):
        self.window = int(window)
        self.entries: list[tuple[np.ndarray, int]] = []
        span = self.window + 1
        for record in bundle.splits[split]:
            _, values = bundle.load(record)
            values = scaler.transform(values)
            starts = window_starts(len(values), span, stride, max_windows_per_flight)
            self.entries.extend((values, int(start)) for start in starts)
        if not self.entries:
            raise RuntimeError(f"No M2AD predictive windows for {bundle.dataset}/{split}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        values, start = self.entries[index]
        split = start + self.window
        return torch.from_numpy(values[start:split].copy()), torch.from_numpy(values[split].copy())


def _loader(bundle, split, scaler, params, shuffle, seed):
    maximum = params["max_train_windows_per_flight"] if split == "train" else params["max_val_windows_per_flight"]
    dataset = PredictiveDataset(
        bundle, split, scaler, params["window"],
        params["train_stride"] if split == "train" else params["score_stride"], maximum,
    )
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset, batch_size=params["batch_size"], shuffle=shuffle, num_workers=0,
        pin_memory=torch.cuda.is_available(), drop_last=False, generator=generator,
    )


def _parameters(cfg, channels: int) -> dict:
    # Align the input sampling protocol with the corresponding TCNGATRE run.
    # M2AD remains a native one-step predictor because changing its output to
    # four steps would alter the released model and its per-sensor GMM scoring.
    return {
        "window": int(cfg.lookback),
        "target_size": 1,
        "train_stride": int(cfg.stride),
        "score_stride": int(cfg.stride),
        "batch_size": int(cfg.batch_size),
        "input_protocol": "tcngatre_aligned_lookback_stride_batch",
        "tcngatre_reference_horizon": int(cfg.horizon),
        "forecast_horizon": 1,
        "forecast_horizon_policy": "method_native_one_step",
        "epochs": 1 if cfg.smoke else 35,
        "early_stop_patience": 5,
        "early_stop_min_delta": 1e-6,
        "max_train_windows_per_flight": 32 if cfg.smoke else None,
        "max_val_windows_per_flight": 64 if cfg.smoke else None,
        "max_score_windows_per_flight": 256 if cfg.smoke else None,
        "lr": 1e-3,
        "lstm_units": 80,
        "n_layer": 2,
        "dropout": 0.2,
        "gmm_components": 1,
        "gmm_covariance_type": "spherical",
        "sensor_weights": "uniform",
        "error": "causally_smoothed_absolute_point_error",
        "gamma_p_threshold": 0.001,
        "channels": int(channels),
    }


@torch.no_grad()
def _predict(model, windows: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    predictions = []
    for offset in range(0, len(windows), int(batch_size)):
        batch = torch.from_numpy(windows[offset:offset + int(batch_size)]).to(device).float()
        predictions.append(model(batch).detach().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def _flight_errors(model, bundle, split, scaler, params, device, point_errors):
    flights = []
    span = params["window"] + 1
    records = tqdm(
        bundle.splits[split], desc=f"M2AD score {bundle.dataset}/{split}",
        unit="flight", dynamic_ncols=True,
    )
    for record in records:
        time, values = bundle.load(record)
        values = scaler.transform(values)
        starts = window_starts(
            len(values), span, params["score_stride"], params["max_score_windows_per_flight"]
        )
        if not len(starts):
            continue
        windows = np.stack([values[int(start):int(start) + params["window"]] for start in starts])
        targets = np.stack([values[int(start) + params["window"]] for start in starts])
        predictions = _predict(model, windows.astype(np.float32), params["batch_size"], device)
        errors = np.asarray(point_errors(targets, predictions, smooth=True), dtype=np.float64)
        if errors.shape != targets.shape or not np.isfinite(errors).all():
            raise RuntimeError(f"M2AD produced invalid errors for {record.flight}: {errors.shape}")
        flights.append((record, time, starts, targets, predictions, errors))
    if not flights:
        raise RuntimeError(f"M2AD produced no scores for {bundle.dataset}/{split}")
    return flights


def _score_flights(flights, gmm, params) -> pd.DataFrame:
    rows = []
    for record, time, starts, targets, predictions, errors in flights:
        gamma_p, sensor_p, combined, fisher_values = gmm.p_values(errors)
        gamma_p = np.asarray(gamma_p, dtype=np.float64)
        combined = np.asarray(combined, dtype=np.float64)
        fisher_values = np.asarray(fisher_values, dtype=np.float64)
        native_prediction = gamma_p < float(params["gamma_p_threshold"])
        prediction_mse = np.square(predictions - targets).mean(axis=1)
        for index, start in enumerate(starts):
            target_index = int(start) + int(params["window"])
            rows.append({
                "flight": record.flight,
                "current_index": target_index,
                "t_start": float(time[target_index]),
                "t_end": float(time[target_index]),
                "t_mid": float(time[target_index]),
                "raw_total_score": float(combined[index]),
                "sensor_score_vec": json.dumps(fisher_values[index].astype(np.float32).tolist()),
                "valid_dim_count": int(fisher_values.shape[1]),
                "prediction_mse": float(prediction_mse[index]),
                "native_gamma_p_value": float(gamma_p[index]),
                "native_prediction": int(native_prediction[index]),
            })
    return pd.DataFrame(rows)


def _confusion_metrics(labels: np.ndarray, prediction: np.ndarray, score: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    tp = int(((labels == 1) & (prediction == 1)).sum())
    fp = int(((labels == 0) & (prediction == 1)).sum())
    tn = int(((labels == 0) & (prediction == 0)).sum())
    fn = int(((labels == 1) & (prediction == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, np.finfo(float).eps),
        "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "auroc": float(roc_auc_score(labels, score)) if len(np.unique(labels)) == 2 else math.nan,
        "average_precision": float(average_precision_score(labels, score)) if labels.sum() else math.nan,
        "num_samples": int(len(labels)),
    }


def _write_native_evaluation(run_dir: Path, failure_raw: pd.DataFrame, params: dict) -> dict:
    labeled_path = run_dir / "infer_tcngatre_failure" / "score_threshold_analysis" / "sequence_scores_with_labels.csv"
    labeled = pd.read_csv(labeled_path)
    native = failure_raw[[
        "flight", "current_index", "native_gamma_p_value", "native_prediction"
    ]].copy()
    frame = labeled.merge(native, on=["flight", "current_index"], how="left", validate="one_to_one")
    frame = frame.loc[pd.to_numeric(frame["label_any"], errors="coerce").notna()].copy()
    if frame[["native_gamma_p_value", "native_prediction"]].isna().any().any():
        raise RuntimeError("M2AD native evaluation could not align all scored windows")
    frame["label_any"] = frame["label_any"].astype(np.int8)
    frame["native_anomaly_score"] = -np.log(
        np.clip(frame["native_gamma_p_value"].to_numpy(dtype=np.float64), 1e-300, 1.0)
    )
    per_flight = []
    for flight, current in frame.groupby("flight", sort=False):
        per_flight.append({
            "flight": flight,
            **_confusion_metrics(
                current["label_any"].to_numpy(), current["native_prediction"].to_numpy(),
                current["native_anomaly_score"].to_numpy(),
            ),
        })
    primary = {
        "threshold_method": "official_gamma_calibration",
        "gamma_p_threshold": float(params["gamma_p_threshold"]),
        "label_col": "label_any",
        "aggregation": "micro_over_all_scored_windows",
        **_confusion_metrics(
            frame["label_any"].to_numpy(), frame["native_prediction"].to_numpy(),
            frame["native_anomaly_score"].to_numpy(),
        ),
    }
    output = run_dir / "native_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "sequence_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(per_flight).to_csv(output / "per_flight_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(output / "primary_metrics.json", primary)
    write_json(output / "evaluation_config.json", {
        "method": "official M2AD GMM and Gamma calibration",
        "gamma_p_threshold": float(params["gamma_p_threshold"]),
        "point_adjustment": False,
        "failure_label_calibration": False,
        "paper_table_role": "diagnostic; common-protocol table uses causal EMA plus flight-wise SPOT",
    })
    return primary


def run(cfg, force: bool = False) -> dict:
    verify_snapshot()
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle = CommonDataBundle(cfg.dataset)
    data_validation = validate_dataset_protocol(bundle)
    label_validation = validate_failure_labels(cfg, bundle)
    params = _parameters(cfg, len(bundle.nodes))
    source_commit = git_commit(SOURCE_ROOT)
    adapter_hash = adapter_config_hash(cfg, BASELINE, params, source_commit, Path(__file__))
    done_path = run_dir / "DONE.json"
    if done_path.exists() and not force:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("config_hash") == cfg.config_hash and done.get("adapter_config_hash") == adapter_hash:
            return {"status": "skipped_complete", **done}
    try:
        seed_everything(cfg.model_seed)
        scaler = _fit_minmax(bundle)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader = _loader(bundle, "train", scaler, params, True, cfg.model_seed)
        val_loader = _loader(bundle, "validation", scaler, params, False, cfg.model_seed)
        OfficialModel, OfficialGMM, point_errors = _official_components()
        model = OfficialModel(
            seq_len=params["window"], in_channels=params["channels"],
            out_channels=params["channels"], lstm_units=params["lstm_units"],
            n_layer=params["n_layer"], dropout=params["dropout"],
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        criterion = nn.MSELoss()
        history = []
        best_val, best_epoch, stale = math.inf, 0, 0
        label = f"ex09/{cfg.dataset}/{BASELINE}/seed_{cfg.model_seed}"
        epoch_bar = tqdm(range(params["epochs"]), desc=label, unit="epoch", dynamic_ncols=True)
        for epoch in epoch_bar:
            model.train()
            train_sum = train_count = 0
            for x, y in tqdm(
                train_loader, desc=f"M2AD train e{epoch + 1:03d}/{params['epochs']:03d}",
                leave=False, unit="batch", dynamic_ncols=True,
            ):
                x, y = x.to(device).float(), y.to(device).float()
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError("M2AD produced a non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_sum += float(loss.detach().cpu()) * len(x)
                train_count += len(x)
            model.eval()
            val_sum = val_count = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device).float(), y.to(device).float()
                    loss = criterion(model(x), y)
                    val_sum += float(loss.detach().cpu()) * len(x)
                    val_count += len(x)
            train_loss = train_sum / max(train_count, 1)
            val_loss = val_sum / max(val_count, 1)
            improved = val_loss < best_val - params["early_stop_min_delta"]
            if improved:
                best_val, best_epoch, stale = val_loss, epoch + 1, 0
            else:
                stale += 1
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
            checkpoint = {
                "epoch": epoch + 1, "best_epoch": best_epoch, "best_val": best_val,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "adapter_config": params, "nodes": bundle.nodes, "config_hash": cfg.config_hash,
            }
            torch.save(checkpoint, run_dir / "last.pt")
            if improved:
                torch.save(checkpoint, run_dir / "best.pt")
            epoch_bar.set_postfix(
                train=f"{train_loss:.6f}", val=f"{val_loss:.6f}",
                patience=f"{stale}/{params['early_stop_patience']}", refresh=True,
            )
            if stale >= params["early_stop_patience"]:
                print(f"[EARLY STOP] epoch={epoch + 1} best_epoch={best_epoch} best_val={best_val:.8f}", flush=True)
                break

        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        train_flights = _flight_errors(model, bundle, "train", scaler, params, device, point_errors)
        train_errors = np.concatenate([item[-1] for item in train_flights], axis=0)
        gmm = OfficialGMM(
            sensors=list(bundle.nodes), n_components=params["gmm_components"],
            covariance_type=params["gmm_covariance_type"], one_sided=True,
            weights=[1.0] * len(bundle.nodes),
        )
        gmm.fit(train_errors)
        validation_raw = _score_flights(
            _flight_errors(model, bundle, "validation", scaler, params, device, point_errors), gmm, params
        )
        failure_raw = _score_flights(
            _flight_errors(model, bundle, "failure", scaler, params, device, point_errors), gmm, params
        )
        protocol = write_split_metadata(run_dir, cfg, bundle, scaler)
        source = json.loads(BASELINE_SOURCE_PATH.read_text(encoding="utf-8"))[BASELINE]
        resolved = {
            **cfg.to_dict(), "baseline_parameters": params, "data_protocol": protocol,
            "data_validation": data_validation, "failure_label_validation": label_validation,
        }
        finalize_run(
            cfg=cfg, legacy_cfg=cfg.to_legacy(), baseline=BASELINE, source=source,
            source_commit=source_commit, adapter_hash=adapter_hash, resolved_config=resolved,
            history=history, validation_raw=validation_raw, failure_raw=failure_raw,
            extra_provenance={
                "reproduction_classification": CLASSIFICATION,
                "official_components": ["models.lstm.Model", "models.gmm.GMM", "errors.point_errors"],
                "official_native_decision": "Gamma calibrated p-value < 0.001",
                "paper_primary_decision": "causal EMA plus flight-wise SPOT",
                "common_data_manifest_sha256": bundle.manifest_sha256,
                "point_adjustment_used": False,
                "failure_labels_available_to_training_or_calibration": False,
            },
        )
        native = _write_native_evaluation(run_dir, failure_raw, params)
        done = augment_done(run_dir, protocol, CLASSIFICATION)
        done["native_metrics"] = native
        write_json(done_path, done)
        return done
    except Exception as exc:
        record_failure(run_dir, BASELINE, cfg, exc)
        raise
