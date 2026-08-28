from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from revision_experiments.core.integrity import verify_snapshot
from revision_experiments.core.paths import BASELINE_SOURCE_PATH, EXTERNAL_ROOT

from .common_data import CommonDataBundle, adapter_config_hash, seed_everything, window_starts
from .finalize import finalize_run, record_failure
from .reproduction_utils import (
    augment_done,
    git_commit,
    validate_dataset_protocol,
    validate_failure_labels,
    write_split_metadata,
)


BASELINE = "gcad"
CLASSIFICATION = "official_model_with_protocol_adapter"
SOURCE_ROOT = EXTERNAL_ROOT / BASELINE


def _official_model():
    source = str(SOURCE_ROOT.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            del sys.modules[name]
    from models.tsmixer import TSMixerRevIN

    return TSMixerRevIN


class PredictiveWindowDataset(Dataset):
    def __init__(self, bundle, split, standardizer, seq_len, pred_len, stride, max_windows):
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.entries: list[tuple[np.ndarray, int]] = []
        span = self.seq_len + self.pred_len
        for record in bundle.splits[split]:
            _, values = bundle.load(record)
            values = standardizer.transform(values)
            starts = window_starts(len(values), span, stride, max_windows)
            self.entries.extend((values, int(start)) for start in starts)
        if not self.entries:
            raise RuntimeError(f"No predictive windows for {bundle.dataset}/{split}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        values, start = self.entries[index]
        split = start + self.seq_len
        x = values[start:split].copy()
        y = values[split:split + self.pred_len].copy()
        return torch.from_numpy(x), torch.from_numpy(y)


def _loader(bundle, split, standardizer, params, shuffle, seed):
    max_windows = params["max_train_windows_per_flight"] if split == "train" else params["max_val_windows_per_flight"]
    dataset = PredictiveWindowDataset(
        bundle, split, standardizer, params["seq_len"], params["pred_len"],
        params["train_stride"] if split == "train" else params["score_stride"], max_windows,
    )
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset, batch_size=params["batch_size"], shuffle=shuffle, num_workers=0,
        pin_memory=torch.cuda.is_available(), drop_last=False, generator=generator,
    )


def _parameters(cfg, channels: int) -> dict:
    return {
        "seq_len": 30,
        "pred_len": 1,
        "train_stride": 128 if cfg.smoke else 16,
        "score_stride": 16,
        "batch_size": min(int(cfg.batch_size), 128),
        "gradient_batch_size": 8 if channels >= 32 else 16,
        "epochs": 1 if cfg.smoke else 100,
        "early_stop_patience": 2,
        "early_stop_min_delta": 1e-5,
        "max_train_windows_per_flight": 32 if cfg.smoke else None,
        "max_val_windows_per_flight": 64 if cfg.smoke else None,
        "max_score_windows_per_flight": 256 if cfg.smoke else None,
        "reference_max_windows": 32 if cfg.smoke else 2048,
        "lr": 1e-4,
        "n_block": 3,
        "ff_dim": 1024,
        "dropout": 0.0,
        "sample_p": 0.2,
        "sparse_threshold": 0.005,
        "channels": int(channels),
        "score": "relative_deviation_from_train_normal_gradient_causality_graph",
    }


def _directional_sparse(causal: torch.Tensor, threshold: float) -> torch.Tensor:
    upper = torch.triu(causal, diagonal=0)
    lower_t = torch.tril(causal, diagonal=-1).transpose(1, 2)
    difference = torch.triu(upper - lower_t, diagonal=0)
    upper_out = torch.where(difference < 0, torch.zeros_like(difference), difference)
    lower_out = torch.where(difference < 0, difference.abs(), torch.zeros_like(difference)).transpose(1, 2)
    result = upper_out + lower_out
    return torch.where(result < float(threshold), torch.zeros_like(result), result)


def _causal_batch(model, x: torch.Tensor, y: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    x = x.detach().requires_grad_(True)
    prediction = model(x)
    gradients = []
    for feature in range(prediction.shape[-1]):
        loss = torch.square(prediction[:, :, feature] - y[:, :, feature]).sum()
        grad = torch.autograd.grad(
            loss, x, retain_graph=feature + 1 < prediction.shape[-1], create_graph=False
        )[0]
        gradients.append(grad.abs())
    causal = torch.stack(gradients, dim=-1).mean(dim=1)
    return _directional_sparse(causal, threshold), prediction.detach()


def _reference_graph(model, loader, device, params) -> torch.Tensor:
    selected = []
    seen = 0
    bar = tqdm(loader, desc="GCAD train-normal causal reference", unit="batch", dynamic_ncols=True)
    for batch_index, (x, y) in enumerate(bar):
        if batch_index and np.random.random() > params["sample_p"]:
            continue
        for offset in range(0, len(x), params["gradient_batch_size"]):
            current_x = x[offset:offset + params["gradient_batch_size"]].to(device).float()
            current_y = y[offset:offset + params["gradient_batch_size"]].to(device).float()
            causal, _ = _causal_batch(model, current_x, current_y, params["sparse_threshold"])
            remaining = params["reference_max_windows"] - seen
            selected.append(causal[:remaining].detach().cpu())
            seen += min(len(causal), remaining)
            if seen >= params["reference_max_windows"]:
                break
        if seen >= params["reference_max_windows"]:
            break
    if not selected:
        raise RuntimeError("GCAD did not sample any normal windows for its causal reference")
    return torch.cat(selected, dim=0).mean(dim=0).to(device)


def _score_split(model, bundle, split, standardizer, reference, device, params, label):
    rows = []
    span = params["seq_len"] + params["pred_len"]
    records = tqdm(bundle.splits[split], desc=label, unit="flight", dynamic_ncols=True)
    for record in records:
        time, values = bundle.load(record)
        values = standardizer.transform(values)
        starts = window_starts(
            len(values), span, params["score_stride"], params["max_score_windows_per_flight"]
        )
        batches = range(0, len(starts), params["gradient_batch_size"])
        for offset in tqdm(batches, desc=f"GCAD score {record.flight[:28]}", leave=False, unit="batch", dynamic_ncols=True):
            current = starts[offset:offset + params["gradient_batch_size"]]
            x_np = np.stack([values[int(s):int(s) + params["seq_len"]] for s in current])
            y_np = np.stack([
                values[int(s) + params["seq_len"]:int(s) + span] for s in current
            ])
            x = torch.from_numpy(x_np).to(device).float()
            y = torch.from_numpy(y_np).to(device).float()
            causal, prediction = _causal_batch(model, x, y, params["sparse_threshold"])
            deviation = (causal - reference.unsqueeze(0)).abs() / (reference.abs().unsqueeze(0) + 1e-4)
            channel = deviation.mean(dim=1)
            total = channel.mean(dim=1)
            pred_mse = torch.square(prediction - y).mean(dim=(1, 2))
            for index, start in enumerate(current):
                end = int(start) + span - 1
                rows.append({
                    "flight": record.flight,
                    "current_index": end,
                    "t_start": float(time[end]),
                    "t_end": float(time[end]),
                    "t_mid": float(time[end]),
                    "raw_total_score": float(total[index].detach().cpu()),
                    "sensor_score_vec": json.dumps(channel[index].detach().cpu().numpy().astype(np.float32).tolist()),
                    "valid_dim_count": int(channel.shape[1]),
                    "prediction_mse": float(pred_mse[index].detach().cpu()),
                })
    if not rows:
        raise RuntimeError(f"GCAD scoring produced no rows for {bundle.dataset}/{split}")
    return pd.DataFrame(rows)


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
        standardizer = bundle.fit_standardizer()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader = _loader(bundle, "train", standardizer, params, True, cfg.model_seed)
        val_loader = _loader(bundle, "validation", standardizer, params, False, cfg.model_seed)
        Model = _official_model()
        model = Model(
            input_shape=(params["seq_len"], params["channels"]), pred_len=params["pred_len"],
            n_block=params["n_block"], dropout=params["dropout"], ff_dim=params["ff_dim"],
            target_slice=slice(0, None),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        criterion = nn.MSELoss()
        history = []
        best_val = math.inf
        best_epoch = 0
        stale = 0
        label = f"ex09/{cfg.dataset}/{BASELINE}/seed_{cfg.model_seed}"
        epoch_bar = tqdm(range(params["epochs"]), desc=label, unit="epoch", dynamic_ncols=True)
        for epoch in epoch_bar:
            model.train()
            train_sum = 0.0
            train_count = 0
            for x, y in tqdm(train_loader, desc=f"GCAD train e{epoch + 1:03d}", leave=False, unit="batch", dynamic_ncols=True):
                x = x.to(device, non_blocking=True).float()
                y = y.to(device, non_blocking=True).float()
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError("GCAD produced a non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_sum += float(loss.detach().cpu()) * len(x)
                train_count += len(x)
            model.eval()
            val_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True).float()
                    y = y.to(device, non_blocking=True).float()
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
            epoch_bar.set_postfix(train=f"{train_loss:.6f}", val=f"{val_loss:.6f}", patience=f"{stale}/{params['early_stop_patience']}")
            if stale >= params["early_stop_patience"]:
                break
        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        reference_loader = _loader(bundle, "train", standardizer, params, False, cfg.model_seed)
        reference = _reference_graph(model, reference_loader, device, params)
        np.save(run_dir / "train_normal_causal_reference.npy", reference.detach().cpu().numpy())
        validation_raw = _score_split(model, bundle, "validation", standardizer, reference, device, params, f"{label} score validation")
        failure_raw = _score_split(model, bundle, "failure", standardizer, reference, device, params, f"{label} score failure")
        protocol = write_split_metadata(run_dir, cfg, bundle, standardizer)
        source = json.loads(BASELINE_SOURCE_PATH.read_text(encoding="utf-8"))[BASELINE]
        resolved = {
            **cfg.to_dict(), "baseline_parameters": params, "data_protocol": protocol,
            "data_validation": data_validation, "failure_label_validation": label_validation,
        }
        outcome = finalize_run(
            cfg=cfg, legacy_cfg=cfg.to_legacy(), baseline=BASELINE, source=source,
            source_commit=source_commit, adapter_hash=adapter_hash, resolved_config=resolved,
            history=history, validation_raw=validation_raw, failure_raw=failure_raw,
            extra_provenance={
                "reproduction_classification": CLASSIFICATION,
                "official_components": ["TSMixerRevIN", "ResBlock", "RevIN"],
                "official_oracle_threshold_replaced": True,
                "official_point_adjustment_replaced": True,
                "point_adjustment_used": False,
                "common_data_manifest_sha256": bundle.manifest_sha256,
            },
        )
        threshold_mean = float(outcome["primary_metrics"].get("threshold_mean", math.nan))
        if not math.isfinite(threshold_mean):
            done_path.unlink(missing_ok=True)
            raise RuntimeError("GCAD SPOT evaluation did not produce a finite threshold")
        return augment_done(run_dir, protocol, CLASSIFICATION)
    except Exception as exc:
        record_failure(run_dir, BASELINE, cfg, exc)
        raise
