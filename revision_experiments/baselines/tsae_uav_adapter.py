from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from revision_experiments.core.integrity import verify_snapshot
from revision_experiments.core.paths import BASELINE_SOURCE_PATH
from revision_experiments.core.provenance import write_json

from .common_data import adapter_config_hash, CommonDataBundle, make_loader, score_split, seed_everything
from .finalize import finalize_run, record_failure
from .reproduction_utils import augment_done, file_sha256, validate_alfa_protocol, write_split_metadata
from .tsae_uav_model import TSAEUAV


BASELINE = "tsae_uav"
CLASSIFICATION = "paper_based_protocol_compatible_reimplementation"
PAPER_AUDIT = Path(__file__).with_name("paper_audits") / "tsae_uav.json"


def _parameters(cfg, channels: int) -> dict:
    smoke = bool(cfg.smoke)
    return {
        "window": 16,
        "train_stride": 64 if smoke else 16,
        # Smoke training is sparse, but threshold calibration remains dense enough
        # for the common SPOT initializer to fit a finite tail threshold.
        "score_stride": 16,
        "batch_size": min(int(cfg.batch_size), 32 if smoke else 128),
        "epochs": 1 if smoke else 100,
        "early_stop_patience": 5,
        "early_stop_min_delta": 1e-4,
        "max_train_windows_per_flight": 32 if smoke else None,
        "max_val_windows_per_flight": 64 if smoke else None,
        "max_score_windows_per_flight": 512 if smoke else None,
        "lr": 1e-4,
        "lr_gamma": 0.5,
        "d_model": 64,
        "top_k": 3,
        "layers": 2,
        "dropout": 0.0,
        "channels": int(channels),
        "score": "last_time_point_channel_squared_error_then_channel_mean",
        "model_sha256": file_sha256(Path(__file__).with_name("tsae_uav_model.py")),
    }


def run(cfg, force: bool = False) -> dict:
    verify_snapshot()
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle = CommonDataBundle(cfg.dataset)
    validate_alfa_protocol(bundle)
    params = _parameters(cfg, len(bundle.nodes))
    paper = json.loads(PAPER_AUDIT.read_text(encoding="utf-8"))
    source_commit = f"pdf-sha256:{paper['pdf_sha256']}"
    adapter_hash = adapter_config_hash(cfg, BASELINE, params, source_commit, Path(__file__))
    done_path = run_dir / "DONE.json"
    if done_path.exists() and not force:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if (
            done.get("config_hash") == cfg.config_hash
            and done.get("adapter_config_hash") == adapter_hash
            and done.get("reproduction_classification") == CLASSIFICATION
        ):
            return {"status": "skipped_complete", **done}
    try:
        seed_everything(cfg.model_seed)
        standardizer = bundle.fit_standardizer()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader = make_loader(
            bundle, "train", standardizer, params["window"], params["train_stride"],
            params["batch_size"], params["max_train_windows_per_flight"], True, cfg.model_seed,
        )
        val_loader = make_loader(
            bundle, "validation", standardizer, params["window"], params["score_stride"],
            params["batch_size"], params["max_val_windows_per_flight"], False, cfg.model_seed,
        )
        model = TSAEUAV(
            channels=params["channels"], d_model=params["d_model"], top_k=params["top_k"],
            layers=params["layers"], dropout=params["dropout"],
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["lr_gamma"])
        history: list[dict] = []
        best_val = math.inf
        best_epoch = 0
        stale_epochs = 0
        label = f"ex03/{cfg.dataset}/{BASELINE}/seed_{cfg.model_seed}"
        epoch_bar = tqdm(range(params["epochs"]), desc=label, unit="epoch", dynamic_ncols=True)
        for epoch in epoch_bar:
            model.train()
            train_losses = []
            train_bar = tqdm(
                train_loader, desc=f"TSAE-UAV train e{epoch + 1:03d}/{params['epochs']:03d}",
                unit="batch", leave=False, dynamic_ncols=True,
            )
            for batch in train_bar:
                batch = batch.to(device, non_blocking=True).float()
                optimizer.zero_grad(set_to_none=True)
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                if not torch.isfinite(loss):
                    raise RuntimeError("TSAE-UAV produced a non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
                train_bar.set_postfix(loss=f"{train_losses[-1]:.6f}", refresh=False)
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in tqdm(
                    val_loader, desc=f"TSAE-UAV valid e{epoch + 1:03d}/{params['epochs']:03d}",
                    unit="batch", leave=False, dynamic_ncols=True,
                ):
                    batch = batch.to(device, non_blocking=True).float()
                    val_losses.append(float(criterion(model(batch), batch).detach().cpu()))
            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            learning_rate = float(optimizer.param_groups[0]["lr"])
            history.append({
                "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                "learning_rate": learning_rate,
            })
            improved = val_loss < best_val - params["early_stop_min_delta"]
            if improved:
                best_val = val_loss
                best_epoch = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
            checkpoint = {
                "epoch": epoch + 1, "best_epoch": best_epoch, "best_val": best_val,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "adapter_config": params,
                "nodes": bundle.nodes, "config_hash": cfg.config_hash,
            }
            torch.save(checkpoint, run_dir / "last.pt")
            if improved:
                torch.save(checkpoint, run_dir / "best.pt")
            scheduler.step()
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
            channel = torch.square(model(batch)[:, -1, :] - batch[:, -1, :])
            return channel.mean(dim=1).cpu().numpy(), channel.cpu().numpy()

        validation_raw = score_split(
            bundle, "validation", standardizer, params["window"], params["score_stride"],
            params["batch_size"], params["max_score_windows_per_flight"], score_batch, device,
            progress_desc=f"{label} score validation",
        )
        failure_raw = score_split(
            bundle, "failure", standardizer, params["window"], params["score_stride"],
            params["batch_size"], params["max_score_windows_per_flight"], score_batch, device,
            progress_desc=f"{label} score failure",
        )
        protocol = write_split_metadata(run_dir, cfg, bundle, standardizer)
        source = json.loads(BASELINE_SOURCE_PATH.read_text(encoding="utf-8"))[BASELINE]
        resolved = {
            **cfg.to_dict(), "baseline_parameters": params, "paper_audit": paper,
            "data_protocol": protocol,
        }
        outcome = finalize_run(
            cfg=cfg, legacy_cfg=cfg.to_legacy(), baseline=BASELINE, source=source,
            source_commit=source_commit, adapter_hash=adapter_hash, resolved_config=resolved,
            history=history, validation_raw=validation_raw, failure_raw=failure_raw,
            extra_provenance={
                "reproduction_classification": CLASSIFICATION,
                "paper_pdf_sha256": paper["pdf_sha256"],
                "paper_pdf_repository_path": paper["local_reference_path"],
                "common_data_manifest_sha256": bundle.manifest_sha256,
                "original_paper_threshold_replaced": True,
                "point_adjustment_used": False,
                "failure_labels_available_to_training_or_calibration": False,
            },
        )
        threshold_mean = float(outcome["primary_metrics"].get("threshold_mean", math.nan))
        if not math.isfinite(threshold_mean):
            (run_dir / "DONE.json").unlink(missing_ok=True)
            raise RuntimeError("TSAE-UAV SPOT evaluation did not produce a finite threshold")
        return augment_done(run_dir, protocol, CLASSIFICATION)
    except Exception as exc:
        record_failure(run_dir, BASELINE, cfg, exc)
        raise
