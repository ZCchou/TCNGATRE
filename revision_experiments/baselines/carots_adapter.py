from __future__ import annotations

import json
import importlib
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from revision_experiments.core.integrity import verify_snapshot
from revision_experiments.core.paths import BASELINE_SOURCE_PATH, EXTERNAL_ROOT
from revision_experiments.core.provenance import write_json

from .common_data import adapter_config_hash, CommonDataBundle, make_loader, score_split, seed_everything
from .finalize import finalize_run, record_failure


BASELINE = "carots"


def _clear_official_namespace_conflicts() -> None:
    # CAROTS uses repository-top-level packages named `models`, `layers`, and
    # `utils`. TCNGATRE common-data preparation imports its own `utils` first;
    # without clearing that cached package, `utils.masking` resolves against the
    # wrong repository even when the CAROTS source root is first on sys.path.
    prefixes = ("models", "layers", "utils")
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            del sys.modules[name]
    importlib.invalidate_caches()


def _official_imports():
    source_root = EXTERNAL_ROOT / BASELINE
    source_text = str(source_root)
    while source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)
    _clear_official_namespace_conflicts()
    from config import get_cfg_defaults
    from models.carots.loss import loss_fn
    from models.carots.modeling_carots import CAROTS
    return get_cfg_defaults, CAROTS, loss_fn


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=EXTERNAL_ROOT / BASELINE, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
    ).stdout.strip()


def _parameters(cfg, channels: int) -> dict:
    smoke = bool(cfg.smoke)
    return {
        "window": 10,
        "input_step": 9,
        "pred_step": 1,
        "train_stride": 64 if smoke else 1,
        # The common evaluator requires at least 32 causal calibration points.
        "score_stride": 16 if smoke else 1,
        "batch_size": min(int(cfg.batch_size), 16 if smoke else 256),
        "causal_epochs": 1 if smoke else 50,
        "contrastive_epochs": 1 if smoke else 30,
        "max_train_windows_per_flight": 32 if smoke else None,
        "max_val_windows_per_flight": 32 if smoke else None,
        "max_score_windows_per_flight": 512 if smoke else None,
        "hidden_dim": 64 if smoke else 512,
        "projector_hidden": 128 if smoke else 1024,
        "projector_output": 64 if smoke else 512,
        "causal_hidden": 16 if smoke else 32,
        "lr": 1e-4,
        "causal_lr": 1e-2,
        "graph_lr": 1e-3,
        "sparsity_lambda": 0.1,
        "channels": int(channels),
    }


def _score_statistics(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    return mean, max(std, 1e-8)


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
        if not torch.cuda.is_available():
            raise RuntimeError("The official CAROTS modules hard-code CUDA; a CUDA environment is required")
        seed_everything(cfg.model_seed)
        get_cfg_defaults, CAROTS, loss_fn = _official_imports()
        standardizer = bundle.fit_standardizer()
        device = torch.device("cuda")
        official_cfg = get_cfg_defaults()
        official_cfg.SEED = int(cfg.model_seed)
        official_cfg.NUM_GPUS = 1
        official_cfg.VISIBLE_DEVICES = 0
        official_cfg.DATA.NAME = f"TCNGATRE_{cfg.dataset}"
        official_cfg.DATA.N_VAR = len(bundle.nodes)
        official_cfg.DATA.WIN_SIZE = params["window"]
        official_cfg.DATA.TRAIN_STEP = params["train_stride"]
        official_cfg.DATA.TEST_STEP = params["score_stride"]
        official_cfg.TRAIN.BATCH_SIZE = params["batch_size"]
        official_cfg.VAL.BATCH_SIZE = params["batch_size"]
        official_cfg.TEST.BATCH_SIZE = params["batch_size"]
        official_cfg.DATA_LOADER.NUM_WORKERS = 0
        official_cfg.SOLVER.MAX_EPOCH = params["contrastive_epochs"]
        official_cfg.SOLVER.BASE_LR = params["lr"]
        official_cfg.LSTM.HIDDEN_DIM = params["hidden_dim"]
        official_cfg.CAROTS.PROJECTOR.INPUT_DIM = params["hidden_dim"]
        official_cfg.CAROTS.PROJECTOR.HIDDEN_DIM = params["projector_hidden"]
        official_cfg.CAROTS.PROJECTOR.OUTPUT_DIM = params["projector_output"]
        official_cfg.CAROTS.SIM_THRESHOLD = 0.0 if cfg.smoke else 0.5
        official_cfg.CAROTS.SIM_THRESHOLD_SCHEDULE = False if cfg.smoke else True
        official_cfg.CUTS_PLUS.N_NODES = len(bundle.nodes)
        official_cfg.CUTS_PLUS.N_GROUPS = len(bundle.nodes)
        official_cfg.CUTS_PLUS.INPUT_STEP = params["input_step"]
        official_cfg.CUTS_PLUS.DATA_PRED.PRED_STEP = params["pred_step"]
        official_cfg.CUTS_PLUS.DATA_PRED.MLP_HID = params["causal_hidden"]
        official_cfg.CUTS_PLUS.DATA_PRED.LR_DATA_START = params["causal_lr"]
        official_cfg.CUTS_PLUS.DATA_PRED.LR_DATA_END = params["causal_lr"]
        official_cfg.CUTS_PLUS.GRAPH_DISCOV.LR_GRAPH_START = params["graph_lr"]
        official_cfg.CUTS_PLUS.GRAPH_DISCOV.LR_GRAPH_END = params["graph_lr"]
        official_cfg.CUTS_PLUS.GRAPH_DISCOV.LAMBDA_S_START = params["sparsity_lambda"]
        official_cfg.CUTS_PLUS.GRAPH_DISCOV.LAMBDA_S_END = params["sparsity_lambda"]
        official_cfg.CUTS_PLUS.SOLVER.MAX_EPOCH = params["causal_epochs"]

        train_loader = make_loader(
            bundle, "train", standardizer, params["window"], params["train_stride"],
            params["batch_size"], params["max_train_windows_per_flight"], True, cfg.model_seed,
        )
        train_score_loader = make_loader(
            bundle, "train", standardizer, params["window"], params["train_stride"],
            params["batch_size"], params["max_train_windows_per_flight"], False, cfg.model_seed,
        )
        val_loader = make_loader(
            bundle, "validation", standardizer, params["window"], params["score_stride"],
            params["batch_size"], params["max_val_windows_per_flight"], False, cfg.model_seed,
        )
        model = CAROTS(official_cfg).to(device)
        causal = model.causal_discoverer
        data_params = [parameter for name, parameter in causal.named_parameters() if name != "GT"]
        data_optimizer = torch.optim.Adam(data_params, lr=params["causal_lr"])
        graph_optimizer = torch.optim.Adam([causal.GT], lr=params["graph_lr"])
        history: list[dict] = []

        # Official CUTS+ network, trained in its two phases on boundary-safe normal windows.
        for epoch in range(params["causal_epochs"]):
            causal.train()
            prediction_losses = []
            graph_losses = []
            for batch in train_loader:
                batch = batch.to(device, non_blocking=True).float()
                x, y = batch[:, :params["input_step"]], batch[:, params["input_step"]:]
                probability = torch.sigmoid(causal.GT)

                data_optimizer.zero_grad(set_to_none=True)
                sampled = torch.bernoulli(probability.detach()).unsqueeze(0).expand(len(batch), -1, -1)
                prediction = causal(x, sampled).transpose(1, 2)
                prediction_loss = F.mse_loss(prediction, y)
                if not torch.isfinite(prediction_loss):
                    raise RuntimeError("CAROTS CUTS+ data-prediction loss is non-finite")
                prediction_loss.backward()
                torch.nn.utils.clip_grad_norm_(data_params, 1.0)
                data_optimizer.step()
                prediction_losses.append(float(prediction_loss.detach().cpu()))

                causal.zero_grad(set_to_none=True)
                graph_optimizer.zero_grad(set_to_none=True)
                probability = torch.sigmoid(causal.GT)
                graph = probability.unsqueeze(0).expand(len(batch), -1, -1)
                prediction = causal(x, graph).transpose(1, 2)
                graph_loss = F.mse_loss(prediction, y) + params["sparsity_lambda"] * probability.mean()
                if not torch.isfinite(graph_loss):
                    raise RuntimeError("CAROTS CUTS+ graph loss is non-finite")
                graph_loss.backward()
                torch.nn.utils.clip_grad_norm_([causal.GT], 1.0)
                graph_optimizer.step()
                graph_losses.append(float(graph_loss.detach().cpu()))
            history.append({
                "stage": "cuts_plus", "epoch": epoch + 1,
                "train_loss": float(np.mean(prediction_losses)),
                "graph_loss": float(np.mean(graph_losses)),
            })

        causal.eval()
        model.positive_augmentor.set_causal_discoverer(causal)
        for parameter in causal.parameters():
            parameter.requires_grad = False
        for parameter in model.positive_augmentor.parameters():
            parameter.requires_grad = False
        contrastive_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(contrastive_params, lr=params["lr"], weight_decay=1e-4)
        best_val = math.inf
        for epoch in range(params["contrastive_epochs"]):
            model.train()
            train_losses = []
            for batch in train_loader:
                batch = batch.to(device, non_blocking=True).float()
                optimizer.zero_grad(set_to_none=True)
                output = model(batch, positive_augment=True, negative_augment=True)
                loss = loss_fn(output, official_cfg)
                if not torch.isfinite(loss):
                    raise RuntimeError("CAROTS contrastive loss is non-finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(contrastive_params, 0.5)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device, non_blocking=True).float()
                    output = model(batch, positive_augment=True, negative_augment=True)
                    val_loss = loss_fn(output, official_cfg)
                    if torch.isfinite(val_loss):
                        val_losses.append(float(val_loss.detach().cpu()))
            mean_train = float(np.mean(train_losses))
            mean_val = float(np.mean(val_losses)) if val_losses else mean_train
            history.append({"stage": "carots", "epoch": epoch + 1, "train_loss": mean_train, "val_loss": mean_val})
            checkpoint = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "official_config": official_cfg.dump(),
                "adapter_config": params,
                "nodes": bundle.nodes,
                "config_hash": cfg.config_hash,
            }
            torch.save(checkpoint, run_dir / "last.pt")
            if mean_val < best_val:
                best_val = mean_val
                torch.save(checkpoint, run_dir / "best.pt")

        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()

        # Match the official Predictor: centroid L2 plus CUTS+ prediction score,
        # each normalized using normal training scores only.
        centroid_parts = []
        with torch.no_grad():
            for batch in train_score_loader:
                batch = batch.to(device, non_blocking=True).float()
                output = model(batch, positive_augment=True, negative_augment=True)
                centroid_parts.append(output[:len(output) // 2])
        centroid_values = torch.cat(centroid_parts, dim=0)
        centroid = centroid_values.mean(dim=0, keepdim=True)

        @torch.no_grad()
        def components(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            embedding = model(batch.float(), positive_augment=False, negative_augment=False)
            l2 = torch.cdist(embedding, centroid).reshape(-1)
            x, y = batch[:, :params["input_step"]], batch[:, params["input_step"]:]
            graph = (model.causal_discoverer.causality_mtx > 0.5).float()
            graph = graph.unsqueeze(0).expand(len(batch), -1, -1)
            prediction = model.causal_discoverer(x, graph).transpose(1, 2)
            causal_score = F.mse_loss(prediction, y, reduction="none").mean(dim=(1, 2))
            return l2, causal_score

        train_l2 = []
        train_causal = []
        with torch.no_grad():
            for batch in train_score_loader:
                l2, causal_score = components(batch.to(device, non_blocking=True).float())
                train_l2.append(l2.detach().cpu().numpy())
                train_causal.append(causal_score.detach().cpu().numpy())
        l2_mean, l2_std = _score_statistics(np.concatenate(train_l2))
        causal_mean, causal_std = _score_statistics(np.concatenate(train_causal))

        @torch.no_grad()
        def score_batch(batch: torch.Tensor) -> tuple[np.ndarray, None]:
            l2, causal_score = components(batch)
            score = (l2 - l2_mean) / l2_std + (causal_score - causal_mean) / causal_std
            return score.detach().cpu().numpy(), None

        validation_raw = score_split(
            bundle, "validation", standardizer, params["window"], params["score_stride"],
            params["batch_size"], params["max_score_windows_per_flight"], score_batch, device,
        )
        failure_raw = score_split(
            bundle, "failure", standardizer, params["window"], params["score_stride"],
            params["batch_size"], params["max_score_windows_per_flight"], score_batch, device,
        )
        np.save(run_dir / "causality_matrix.npy", model.causal_discoverer.causality_mtx.detach().cpu().numpy())
        write_json(run_dir / "normalization_stats.json", standardizer.to_dict())
        write_json(run_dir / "native_score_normalization.json", {
            "source": "train_normal only",
            "l2_mean": l2_mean, "l2_std": l2_std,
            "causal_mean": causal_mean, "causal_std": causal_std,
        })
        write_json(run_dir / "split_flights.json", {
            "data_split_seed": cfg.data_split_seed,
            "model_seed": cfg.model_seed,
            "train_flights": [row.flight for row in bundle.splits["train"]],
            "validation_flights": [row.flight for row in bundle.splits["validation"]],
            "failure_flights_scored_only": [row.flight for row in bundle.splits["failure"]],
        })
        source = json.loads(BASELINE_SOURCE_PATH.read_text(encoding="utf-8"))[BASELINE]
        resolved = {**cfg.to_dict(), "baseline_parameters": params, "official_config": official_cfg.dump()}
        return finalize_run(
            cfg=cfg, legacy_cfg=legacy_cfg, baseline=BASELINE, source=source,
            source_commit=source_commit, adapter_hash=adapter_hash,
            resolved_config=resolved, history=history,
            validation_raw=validation_raw, failure_raw=failure_raw,
            extra_provenance={
                "official_components": ["CAROTS", "CUTS_Plus_Net", "loss_fn"],
                "common_data_manifest_sha256": bundle.manifest_sha256,
                "native_score": "normalized L2 centroid score + normalized CUTS+ prediction score",
            },
        )
    except Exception as exc:
        record_failure(run_dir, BASELINE, cfg, exc)
        raise
