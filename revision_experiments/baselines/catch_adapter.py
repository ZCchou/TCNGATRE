from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from revision_experiments.core.integrity import verify_snapshot
from revision_experiments.core.paths import BASELINE_SOURCE_PATH, EXTERNAL_ROOT
from revision_experiments.core.provenance import write_json

from .common_data import adapter_config_hash, CommonDataBundle, make_loader, score_split, seed_everything
from .finalize import finalize_run, record_failure


BASELINE = "catch"


def _official_imports():
    source_root = EXTERNAL_ROOT / BASELINE
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from ts_benchmark.baselines.catch.CATCH import TransformerConfig
    from ts_benchmark.baselines.catch.models.CATCH_model import CATCHModel
    from ts_benchmark.baselines.catch.utils.fre_rec_loss import frequency_criterion, frequency_loss
    return TransformerConfig, CATCHModel, frequency_loss, frequency_criterion


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=EXTERNAL_ROOT / BASELINE, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
    ).stdout.strip()


def _parameters(cfg, channels: int) -> dict:
    smoke = bool(cfg.smoke)
    window = 32 if smoke else 192
    return {
        "window": window,
        "train_stride": 64 if smoke else 4,
        # Keep at least 32 normal calibration windows on the shortest GPSData
        # validation flight so the shared causal SPOT initializer is defined.
        "score_stride": 16 if smoke else 4,
        "batch_size": min(int(cfg.batch_size), 32 if smoke else 128),
        "epochs": 1 if smoke else 3,
        "max_train_windows_per_flight": 64 if smoke else None,
        "max_val_windows_per_flight": 64 if smoke else None,
        "max_score_windows_per_flight": 512 if smoke else None,
        "lr": 1e-4,
        "mask_lr": 1e-5,
        "e_layers": 1 if smoke else 3,
        "n_heads": 1 if smoke else 2,
        "cf_dim": 16 if smoke else 64,
        "d_ff": 32 if smoke else 256,
        "d_model": 16 if smoke else 128,
        "head_dim": 16 if smoke else 64,
        "patch_size": 8 if smoke else 16,
        "patch_stride": 4 if smoke else 8,
        "inference_patch_size": 8 if smoke else 32,
        "inference_patch_stride": 2 if smoke else 1,
        "dropout": 0.1 if smoke else 0.2,
        "head_dropout": 0.1,
        "auxi_lambda": 0.005,
        "score_lambda": 0.05,
        "dc_lambda": 0.005,
        "channels": int(channels),
    }


def run(cfg, force: bool = False) -> dict:
    verify_snapshot()
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle = CommonDataBundle(cfg.dataset)
    params = _parameters(cfg, len(bundle.nodes))
    source_commit = _commit()
    adapter_hash = adapter_config_hash(cfg, BASELINE, params, source_commit, Path(__file__))
    done_path = run_dir / "DONE.json"
    if done_path.exists() and not force:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("config_hash") == cfg.config_hash and done.get("adapter_config_hash") == adapter_hash:
            return {"status": "skipped_complete", **done}
    legacy_cfg = cfg.to_legacy()
    try:
        seed_everything(cfg.model_seed)
        TransformerConfig, CATCHModel, FrequencyLoss, FrequencyCriterion = _official_imports()
        standardizer = bundle.fit_standardizer()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        official_cfg = TransformerConfig(
            lr=params["lr"], Mlr=params["mask_lr"], e_layers=params["e_layers"],
            n_heads=params["n_heads"], cf_dim=params["cf_dim"], d_ff=params["d_ff"],
            d_model=params["d_model"], head_dim=params["head_dim"], dropout=params["dropout"],
            head_dropout=params["head_dropout"], auxi_lambda=params["auxi_lambda"],
            score_lambda=params["score_lambda"], dc_lambda=params["dc_lambda"],
            patch_stride=params["patch_stride"], patch_size=params["patch_size"],
            inference_patch_stride=params["inference_patch_stride"],
            inference_patch_size=params["inference_patch_size"], seq_len=params["window"],
            num_epochs=params["epochs"], batch_size=params["batch_size"],
            c_in=len(bundle.nodes), enc_in=len(bundle.nodes), dec_in=len(bundle.nodes),
            c_out=len(bundle.nodes), affine=0, subtract_last=0, mask=False,
        )
        train_loader = make_loader(
            bundle, "train", standardizer, params["window"], params["train_stride"],
            params["batch_size"], params["max_train_windows_per_flight"], True, cfg.model_seed,
        )
        val_loader = make_loader(
            bundle, "validation", standardizer, params["window"], params["score_stride"],
            params["batch_size"], params["max_val_windows_per_flight"], False, cfg.model_seed,
        )
        model = CATCHModel(official_cfg).to(device)
        reconstruction = nn.MSELoss()
        auxiliary = FrequencyLoss(official_cfg)
        main_params = [p for name, p in model.named_parameters() if "mask_generator" not in name]
        optimizer = torch.optim.Adam(main_params, lr=params["lr"])
        mask_optimizer = torch.optim.Adam(model.mask_generator.parameters(), lr=params["mask_lr"])
        history: list[dict] = []
        best_val = math.inf
        run_label = f"ex04/{cfg.dataset}/catch/seed_{cfg.model_seed}"
        epoch_bar = tqdm(
            range(params["epochs"]), desc=run_label, unit="epoch",
            dynamic_ncols=True, mininterval=0.5,
        )
        for epoch in epoch_bar:
            model.train()
            train_losses = []
            train_bar = tqdm(
                train_loader, desc=f"CATCH train e{epoch + 1:03d}/{params['epochs']:03d}",
                unit="batch", leave=False, dynamic_ncols=True, mininterval=0.5,
            )
            for batch in train_bar:
                batch = batch.to(device, non_blocking=True).float()
                optimizer.zero_grad(set_to_none=True)
                mask_optimizer.zero_grad(set_to_none=True)
                output, output_complex, dc_loss = model(batch)
                rec_loss = reconstruction(output, batch)
                normalized = model.revin_layer(batch, "transform")
                aux_loss = auxiliary(output_complex, normalized)
                loss = rec_loss + params["dc_lambda"] * dc_loss + params["auxi_lambda"] * aux_loss
                if not torch.isfinite(loss):
                    raise RuntimeError("CATCH produced a non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                mask_optimizer.step()
                current_loss = float(loss.detach().cpu())
                train_losses.append(current_loss)
                train_bar.set_postfix(loss=f"{current_loss:.6f}", refresh=False)
            model.eval()
            val_losses = []
            with torch.no_grad():
                val_bar = tqdm(
                    val_loader, desc=f"CATCH valid e{epoch + 1:03d}/{params['epochs']:03d}",
                    unit="batch", leave=False, dynamic_ncols=True, mininterval=0.5,
                )
                for batch in val_bar:
                    batch = batch.to(device, non_blocking=True).float()
                    output, _, _ = model(batch)
                    current_val = float(reconstruction(output, batch).detach().cpu())
                    val_losses.append(current_val)
                    val_bar.set_postfix(loss=f"{current_val:.6f}", refresh=False)
            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
            checkpoint = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "mask_optimizer_state": mask_optimizer.state_dict(),
                "official_config": vars(official_cfg),
                "adapter_config": params,
                "nodes": bundle.nodes,
                "config_hash": cfg.config_hash,
            }
            torch.save(checkpoint, run_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(checkpoint, run_dir / "best.pt")
            epoch_bar.set_postfix(
                train=f"{train_loss:.6f}", val=f"{val_loss:.6f}", best=f"{best_val:.6f}",
                refresh=True,
            )

        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        temp_criterion = nn.MSELoss(reduction="none")
        freq_criterion = FrequencyCriterion(official_cfg)

        @torch.no_grad()
        def score_batch(batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
            output, _, _ = model(batch.float())
            temp = temp_criterion(batch, output)
            freq = freq_criterion(batch, output)
            channel = (temp + params["score_lambda"] * freq)[:, -1, :]
            return (
                channel.mean(dim=1).detach().cpu().numpy(),
                channel.detach().cpu().numpy(),
            )

        validation_raw = score_split(
            bundle, "validation", standardizer, params["window"], params["score_stride"],
            params["batch_size"], params["max_score_windows_per_flight"], score_batch, device,
            progress_desc=f"{run_label} score validation",
        )
        failure_raw = score_split(
            bundle, "failure", standardizer, params["window"], params["score_stride"],
            params["batch_size"], params["max_score_windows_per_flight"], score_batch, device,
            progress_desc=f"{run_label} score failure",
        )
        write_json(run_dir / "normalization_stats.json", standardizer.to_dict())
        write_json(run_dir / "split_flights.json", {
            "data_split_seed": cfg.data_split_seed,
            "model_seed": cfg.model_seed,
            "train_flights": [row.flight for row in bundle.splits["train"]],
            "validation_flights": [row.flight for row in bundle.splits["validation"]],
            "failure_flights_scored_only": [row.flight for row in bundle.splits["failure"]],
        })
        source = json.loads(BASELINE_SOURCE_PATH.read_text(encoding="utf-8"))[BASELINE]
        resolved = {**cfg.to_dict(), "baseline_parameters": params, "official_config": vars(official_cfg)}
        return finalize_run(
            cfg=cfg, legacy_cfg=legacy_cfg, baseline=BASELINE, source=source,
            source_commit=source_commit, adapter_hash=adapter_hash,
            resolved_config=resolved, history=history,
            validation_raw=validation_raw, failure_raw=failure_raw,
            extra_provenance={
                "official_components": ["CATCHModel", "frequency_loss", "frequency_criterion"],
                "common_data_manifest_sha256": bundle.manifest_sha256,
            },
        )
    except Exception as exc:
        record_failure(run_dir, BASELINE, cfg, exc)
        raise
