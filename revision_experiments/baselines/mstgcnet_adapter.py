from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from revision_experiments.core.integrity import verify_snapshot
from revision_experiments.core.paths import BASELINE_SOURCE_PATH, EXTERNAL_ROOT

from .common_data import adapter_config_hash, CommonDataBundle, make_loader, score_split, seed_everything
from .finalize import finalize_run, record_failure
from .mstgcnet_model import MSTGCNetApprox
from .mstgcnet_native_evaluation import evaluate_native_run
from .reproduction_utils import (
    accumulation_groups,
    augment_done,
    file_sha256,
    git_commit,
    validate_dataset_protocol,
    validate_failure_labels,
    write_split_metadata,
)


BASELINE = "mstgcnet"
CLASSIFICATION = "released_scaffold_engineering_reimplementation"
EXPECTED_COMMIT = "d5087989b0d016fe6b04e4cc8a7c6074d673b7ac"
PAPER_AUDIT = Path(__file__).with_name("paper_audits") / "mstgcnet.json"


def _parameters(cfg, channels: int) -> dict:
    smoke = bool(cfg.smoke)
    profile = os.environ.get("UAV_MSTGCNET_PROFILE", "paper_faithful").strip().lower()
    if profile not in {"paper_faithful", "paper_default", "alfa_sensitivity"}:
        raise ValueError(
            "UAV_MSTGCNET_PROFILE must be 'paper_faithful', 'paper_default', "
            "or 'alfa_sensitivity'"
        )
    faithful = profile == "paper_faithful"
    window = 64 if profile == "alfa_sensitivity" else 96
    d_model = 8 if profile == "alfa_sensitivity" else 64
    patches = (
        [[8, 12, 16, 32], [6, 8, 12, 16], [2, 6, 8, 12]]
        if faithful
        else [[16, 12, 8, 32], [12, 8, 6, 4], [8, 6, 4, 2]]
    )
    effective_batch_size = min(int(cfg.batch_size), 128)
    default_physical_batch = 8 if faithful else 32
    physical_batch_size = min(
        effective_batch_size,
        int(os.environ.get("UAV_MSTGCNET_PHYSICAL_BATCH_SIZE", default_physical_batch)),
    )
    formal_epochs = int(os.environ.get("UAV_MSTGCNET_EPOCHS", 50 if faithful else 10))
    formal_patience = int(os.environ.get("UAV_MSTGCNET_PATIENCE", 10 if faithful else 3))
    formal_train_stride = int(os.environ.get("UAV_MSTGCNET_TRAIN_STRIDE", 16))
    formal_score_stride = int(os.environ.get("UAV_MSTGCNET_SCORE_STRIDE", 16))
    return {
        "adapter_revision": "paper_equation_aligned_v3",
        "parameter_profile": profile,
        "parameter_selection_source": (
            "paper Table IV and Eqs. (1)-(26)" if faithful
            else "paper Table IV" if profile == "paper_default"
            else "paper Figure 3 ALFA sensitivity analysis"
        ),
        "failure_labels_used_for_parameter_selection": False,
        "window": window,
        "train_stride": 64 if smoke else formal_train_stride,
        "validation_stride": 16 if smoke else formal_score_stride,
        "score_stride": 16 if smoke else formal_score_stride,
        "window_construction": "per-flight sliding windows; no cross-flight windows",
        "pointwise_score_coverage": formal_score_stride == 1,
        "stride_rationale": (
            "stride 16 retains the paper window length while avoiding excessive overlap "
            "in the denser fixed-flight export"
        ),
        "batch_size": effective_batch_size,
        "effective_batch_size": effective_batch_size,
        "physical_batch_size": physical_batch_size,
        "gradient_accumulation_steps": int(math.ceil(effective_batch_size / physical_batch_size)),
        "epochs": 1 if smoke else formal_epochs,
        "early_stop_patience": 1 if smoke else formal_patience,
        "early_stop_min_delta": 0.0,
        "max_train_windows_per_flight": 32 if smoke else None,
        "max_val_windows_per_flight": 64 if smoke else None,
        "max_score_windows_per_flight": 512 if smoke else None,
        "lr": 1e-4,
        "balance_loss_weight": 0.01,
        "d_model": d_model,
        "layers": 3,
        "top_k": 3,
        "knn_k": 5,
        "attention_heads": 2,
        "experts_per_layer": [4, 4, 4],
        "paper_patch_pool": [2, 6, 8, 12, 16, 32],
        "patch_size_list": patches,
        "trend_kernels": [4, 8, 12],
        "seasonality": 3,
        "noisy_gating": True,
        "dropout": 0.1,
        "revin": True,
        "residual": True,
        "channels": int(channels),
        "score": "pointwise_channel_squared_error_then_channel_mean",
        "completion_scope": (
            "FFT seasonal/trend router + sparse top-k GMoE + patch-level causal "
            "CC-STGCN + residual reconstruction"
            if faithful else "legacy engineering approximation"
        ),
        "model_sha256": file_sha256(Path(__file__).with_name("mstgcnet_model.py")),
    }


def run(cfg, force: bool = False) -> dict:
    verify_snapshot()
    source_root = EXTERNAL_ROOT / BASELINE
    if not source_root.is_dir():
        raise FileNotFoundError(
            "MSTGCNet official scaffold is missing; run fetch-baselines --baselines mstgcnet"
        )
    source_commit = git_commit(source_root)
    if source_commit != EXPECTED_COMMIT:
        raise RuntimeError(
            f"MSTGCNet source commit mismatch: expected={EXPECTED_COMMIT}, actual={source_commit}"
        )
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle = CommonDataBundle(cfg.dataset)
    data_validation = validate_dataset_protocol(bundle)
    label_validation = validate_failure_labels(cfg, bundle)
    params = _parameters(cfg, len(bundle.nodes))
    audit = json.loads(PAPER_AUDIT.read_text(encoding="utf-8"))
    adapter_hash = adapter_config_hash(cfg, BASELINE, params, source_commit, Path(__file__))
    done_path = run_dir / "DONE.json"
    if done_path.exists() and not force:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if (
            done.get("config_hash") == cfg.config_hash
            and done.get("adapter_config_hash") == adapter_hash
            and done.get("data_protocol_hash")
            and done.get("reproduction_classification") == CLASSIFICATION
        ):
            return {"status": "skipped_complete", **done}
    try:
        seed_everything(cfg.model_seed)
        standardizer = bundle.fit_standardizer()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader = make_loader(
            bundle, "train", standardizer, params["window"], params["train_stride"],
            params["physical_batch_size"], params["max_train_windows_per_flight"], True, cfg.model_seed,
        )
        val_loader = make_loader(
            bundle, "validation", standardizer, params["window"], params["validation_stride"],
            params["physical_batch_size"], params["max_val_windows_per_flight"], False, cfg.model_seed,
        )
        model = MSTGCNetApprox(
            nodes=params["channels"], window=params["window"], d_model=params["d_model"],
            patch_size_list=params["patch_size_list"], top_k=params["top_k"],
            heads=params["attention_heads"], knn_k=params["knn_k"],
            seasonal_top_k=params["seasonality"], trend_kernels=params["trend_kernels"],
            dropout=params["dropout"], revin=params["revin"],
            noisy_gating=params["noisy_gating"],
        ).to(device)
        reconstruction = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )
        history: list[dict] = []
        best_val = math.inf
        best_epoch = 0
        stale_epochs = 0
        label = f"ex03/{cfg.dataset}/{BASELINE}/seed_{cfg.model_seed}"
        epoch_bar = tqdm(range(params["epochs"]), desc=label, unit="epoch", dynamic_ncols=True)
        for epoch in epoch_bar:
            model.train()
            train_total, train_rec, train_balance = [], [], []
            train_bar = tqdm(
                accumulation_groups(train_loader, params["gradient_accumulation_steps"]),
                total=math.ceil(len(train_loader) / params["gradient_accumulation_steps"]),
                desc=f"MSTGCNet train e{epoch + 1:03d}/{params['epochs']:03d}",
                unit="update", leave=False, dynamic_ncols=True,
            )
            weighted_total = weighted_rec = weighted_balance = 0.0
            train_sample_count = 0
            for batches, group_samples in train_bar:
                optimizer.zero_grad(set_to_none=True)
                group_loss = 0.0
                for batch in batches:
                    batch = batch.to(device, non_blocking=True).float()
                    reconstructed, balance, _ = model(batch)
                    rec_loss = reconstruction(reconstructed, batch)
                    loss = rec_loss + params["balance_loss_weight"] * balance
                    if not torch.isfinite(loss):
                        raise RuntimeError("MSTGCNet approximation produced a non-finite training loss")
                    batch_samples = int(batch.shape[0])
                    (loss * (batch_samples / group_samples)).backward()
                    loss_value = float(loss.detach().cpu())
                    rec_value = float(rec_loss.detach().cpu())
                    balance_value = float(balance.detach().cpu())
                    train_total.append(loss_value)
                    train_rec.append(rec_value)
                    train_balance.append(balance_value)
                    weighted_total += loss_value * batch_samples
                    weighted_rec += rec_value * batch_samples
                    weighted_balance += balance_value * batch_samples
                    train_sample_count += batch_samples
                    group_loss += loss_value * batch_samples
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_bar.set_postfix(loss=f"{group_loss / group_samples:.6f}", refresh=False)
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in tqdm(
                    val_loader, desc=f"MSTGCNet valid e{epoch + 1:03d}/{params['epochs']:03d}",
                    unit="batch", leave=False, dynamic_ncols=True,
                ):
                    batch = batch.to(device, non_blocking=True).float()
                    reconstructed, _, _ = model(batch)
                    val_losses.append(float(reconstruction(reconstructed, batch).detach().cpu()))
            train_loss = float(weighted_total / max(train_sample_count, 1))
            val_loss = float(np.mean(val_losses))
            scheduler.step(val_loss)
            improved = val_loss < best_val - params["early_stop_min_delta"]
            if improved:
                best_val = val_loss
                best_epoch = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
            history.append({
                "epoch": epoch + 1, "train_loss": train_loss,
                "train_reconstruction_loss": float(weighted_rec / max(train_sample_count, 1)),
                "train_balance_loss": float(weighted_balance / max(train_sample_count, 1)),
                "val_loss": val_loss, "lr": float(optimizer.param_groups[0]["lr"]),
            })
            checkpoint = {
                "epoch": epoch + 1, "best_epoch": best_epoch, "best_val": best_val,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "adapter_config": params, "nodes": bundle.nodes, "config_hash": cfg.config_hash,
                "official_scaffold_commit": source_commit,
            }
            torch.save(checkpoint, run_dir / "last.pt")
            if improved:
                torch.save(checkpoint, run_dir / "best.pt")
            epoch_bar.set_postfix(
                train=f"{train_loss:.6f}", val=f"{val_loss:.6f}", best=f"{best_val:.6f}",
                patience=f"{stale_epochs}/{params['early_stop_patience']}", refresh=True,
            )
            if stale_epochs >= params["early_stop_patience"]:
                print(f"[EARLY STOP] epoch={epoch + 1} best_epoch={best_epoch} best_val={best_val:.8f}", flush=True)
                break

        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()

        @torch.no_grad()
        def score_batch(batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
            batch = batch.float()
            reconstructed, _, _ = model(batch)
            channel = torch.square(reconstructed[:, -1, :] - batch[:, -1, :])
            return channel.mean(dim=1).cpu().numpy(), channel.cpu().numpy()

        validation_raw = score_split(
            bundle, "validation", standardizer, params["window"], params["score_stride"],
            params["physical_batch_size"], params["max_score_windows_per_flight"], score_batch, device,
            progress_desc=f"{label} score validation",
        )
        failure_raw = score_split(
            bundle, "failure", standardizer, params["window"], params["score_stride"],
            params["physical_batch_size"], params["max_score_windows_per_flight"], score_batch, device,
            progress_desc=f"{label} score failure",
        )
        protocol = write_split_metadata(run_dir, cfg, bundle, standardizer)
        source = json.loads(BASELINE_SOURCE_PATH.read_text(encoding="utf-8"))[BASELINE]
        resolved = {
            **cfg.to_dict(), "baseline_parameters": params, "source_audit": audit,
            "data_protocol": protocol, "data_validation": data_validation,
            "failure_label_validation": label_validation,
        }
        outcome = finalize_run(
            cfg=cfg, legacy_cfg=cfg.to_legacy(), baseline=BASELINE, source=source,
            source_commit=source_commit, adapter_hash=adapter_hash, resolved_config=resolved,
            history=history, validation_raw=validation_raw, failure_raw=failure_raw,
            extra_provenance={
                "reproduction_classification": CLASSIFICATION,
                "official_components_executed": [],
                "official_scaffold_commit": source_commit,
                "official_scaffold_core_status": "GMoE/GraphGCN/Model support classes contain pass",
                "engineering_model_sha256": params["model_sha256"],
                "common_data_manifest_sha256": bundle.manifest_sha256,
                "original_test_each_epoch_replaced": True,
                "spot_evaluation_retained_for_audit": True,
                "paper_evaluation": "flightwise ATSSD plus label-dependent point adjustment",
                "point_adjustment_used_for_paper_metrics": True,
                "failure_labels_available_to_training_or_calibration": False,
            },
        )
        threshold_mean = float(outcome["primary_metrics"].get("threshold_mean", math.nan))
        if not math.isfinite(threshold_mean):
            (run_dir / "DONE.json").unlink(missing_ok=True)
            raise RuntimeError("MSTGCNet SPOT evaluation did not produce a finite threshold")
        native = evaluate_native_run(run_dir, window_size=params["window"], alpha=0.01)
        done = augment_done(run_dir, protocol, CLASSIFICATION)
        done.update({
            "paper_metric_source": "native_evaluation/primary_metrics.json",
            "paper_evaluation_method": "ATSSD + label-dependent point adjustment",
            "paper_primary_metrics": native["primary_metrics"],
        })
        (run_dir / "DONE.json").write_text(
            json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return done
    except Exception as exc:
        record_failure(run_dir, BASELINE, cfg, exc)
        raise
