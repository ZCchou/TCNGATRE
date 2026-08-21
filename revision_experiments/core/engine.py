from __future__ import annotations

import json
import math
import random
import traceback
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from revision_experiments.analysis.robustness import corrupt_tensor
from revision_experiments.models.variants import build_revision_model, transform_graph_prior
from revision_experiments.scoring.aggregators import aggregate_dataframe

from .config import RevisionConfig
from .evaluation import evaluate_scores
from .integrity import verify_snapshot
from .paths import REPO_ROOT, ensure_import_paths
from .provenance import environment_payload, write_json

ensure_import_paths()

from data.stgtcn_window_dataset import (  # noqa: E402
    STGTCNWindowDataset,
    collate_stgtcn_windows,
    resolve_flight_splits,
)
from tcngatre_runtime import ensure_graph_ready  # noqa: E402
from tcngatre_train_impl import (  # noqa: E402
    compute_cross_dim_loss,
    compute_loss,
    fit_normalization_stats,
    load_graph,
    make_dataloaders,
)
from utils.normalization import load_minmax_stats  # noqa: E402


def set_model_seed(seed: int) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Match the native single-seed trainer: seed every RNG without forcing
    # PyTorch onto substantially slower deterministic convolution kernels.
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    return {
        "model_seed": int(seed),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _device(legacy_cfg) -> torch.device:
    if legacy_cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(legacy_cfg.device)


def _run_epoch(
    model, loader, a, m, legacy_cfg, device, optimizer=None, progress_desc: str | None = None
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "value": 0.0, "delta": 0.0, "cross": 0.0, "batches": 0}
    batches = tqdm(
        loader,
        desc=progress_desc,
        unit="batch",
        leave=False,
        dynamic_ncols=False,
        mininterval=5.0,
        disable=progress_desc is None,
    )
    for batch in batches:
        x = batch["x"].to(device).float()
        y = batch["y"].to(device).float()
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(x, a, m, short_patch=legacy_cfg.short_patch)
        losses = compute_loss(prediction[..., 0], y[..., 0], legacy_cfg.huber_beta)
        objective = losses["total_loss"]
        cross = torch.zeros((), device=device)
        if training and legacy_cfg.cross_dim_loss_enabled:
            cross = compute_cross_dim_loss(model, x, y, a, m, legacy_cfg)
            objective = objective + float(legacy_cfg.cross_dim_lambda) * cross
        if training:
            if not torch.isfinite(objective):
                raise RuntimeError("Non-finite training objective")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), legacy_cfg.grad_clip)
            optimizer.step()
        totals["loss"] += float(losses["total_loss"].detach().cpu())
        totals["value"] += float(losses["value_loss"].detach().cpu())
        totals["delta"] += float(losses["delta_loss"].detach().cpu())
        totals["cross"] += float(cross.detach().cpu())
        totals["batches"] += 1
        batches.set_postfix(
            loss=f"{totals['loss'] / totals['batches']:.6g}",
            cross=f"{totals['cross'] / totals['batches']:.6g}",
            refresh=False,
        )
    denom = max(totals.pop("batches"), 1)
    return {key: value / denom for key, value in totals.items()}


def _json_vector(values: np.ndarray) -> str:
    clean = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return json.dumps(clean.tolist(), ensure_ascii=False)


@torch.no_grad()
def _score_loader(
    model,
    loader,
    a,
    m,
    cfg: RevisionConfig,
    legacy_cfg,
    device,
    capture_graph: bool,
) -> tuple[pd.DataFrame, dict[str, np.ndarray] | None, list[dict]]:
    model.eval()
    rows: list[dict] = []
    graph_store: dict[str, list[np.ndarray]] = {"A_static": [], "A_dyn": [], "A_fuse": []}
    corruption_log: list[dict] = []
    global_row = 0
    for batch in loader:
        x = batch["x"].to(device).float()
        y = batch["y"].to(device).float()
        if cfg.corruption_kind != "none":
            x, metadata = corrupt_tensor(
                x, cfg.corruption_kind, cfg.corruption_level, cfg.model_seed * 100000 + global_row
            )
            corruption_log.append(metadata)
        prediction, aux = model(x, a, m, short_patch=legacy_cfg.short_patch)
        pred = prediction[..., 0].detach().cpu().numpy().astype(np.float32)
        truth = y[..., 0].detach().cpu().numpy().astype(np.float32)
        value_residual = np.mean(np.abs(pred - truth), axis=1)
        delta_residual = (
            np.mean(np.abs(np.diff(pred, axis=1) - np.diff(truth, axis=1)), axis=1)
            if pred.shape[1] >= 2 else np.zeros_like(value_residual)
        )
        pred_value = np.mean(pred, axis=1)
        future_value = np.mean(truth, axis=1)
        if capture_graph:
            for name in graph_store:
                graph = aux.get(name)
                if graph is not None and graph.ndim == 3 and graph.shape[0] == len(batch["flight"]):
                    graph_store[name].append(graph.detach().cpu().numpy().astype(np.float32))
        for idx, flight in enumerate(batch["flight"]):
            t_start = float(batch["t_future_start"][idx].item())
            t_end = float(batch["t_future_end"][idx].item())
            rows.append({
                "flight": str(flight),
                "current_index": int(batch["current_index"][idx].item()),
                "t_start": t_start,
                "t_end": t_end,
                "t_mid": 0.5 * (t_start + t_end),
                "future_value_vec": _json_vector(future_value[idx]),
                "pred_value_vec": _json_vector(pred_value[idx]),
                "value_residual_vec": _json_vector(value_residual[idx]),
                "delta_residual_vec": _json_vector(delta_residual[idx]),
                "graph_row": global_row + idx if capture_graph else -1,
            })
        global_row += len(batch["flight"])
    if not rows:
        raise RuntimeError("Scoring produced no rows")
    graph_arrays = None
    if capture_graph and all(graph_store[name] for name in graph_store):
        graph_arrays = {name: np.concatenate(parts, axis=0) for name, parts in graph_store.items()}
    return pd.DataFrame(rows), graph_arrays, corruption_log


def _loader_for_paths(paths, norm, nodes, cfg: RevisionConfig, legacy_cfg) -> DataLoader:
    dataset = STGTCNWindowDataset(
        csv_root=None,
        flights=None,
        normalization_stats=norm,
        node_names=nodes,
        history_len=cfg.lookback,
        horizon=cfg.horizon,
        stride=cfg.stride,
        trim_leading_sec=legacy_cfg.trim_leading_sec,
        use_replication_padding=False,
        flight_paths=paths,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_stgtcn_windows,
        pin_memory=torch.cuda.is_available(),
    )


def _save_loss_plot(history: list[dict], path: Path) -> None:
    if not history:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
    ax.plot([row["epoch"] for row in history], [row["val_loss"] for row in history], label="validation")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Revision experiment training loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def execute_training_run(cfg: RevisionConfig, force: bool = False) -> dict:
    verify_snapshot()
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    done_path = run_dir / "DONE.json"
    if done_path.exists() and not force:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("config_hash") == cfg.config_hash:
            return {"status": "skipped_complete", "run_dir": str(run_dir), **done}

    write_json(run_dir / "config_resolved.json", cfg.to_dict())
    legacy_cfg = cfg.to_legacy()
    try:
        randomness = set_model_seed(cfg.model_seed)
        write_json(
            run_dir / "provenance.json",
            {**environment_payload(REPO_ROOT), "randomness": randomness},
        )
        ensure_graph_ready(legacy_cfg)
        device = _device(legacy_cfg)
        nodes, a, m = load_graph(Path(legacy_cfg.graph_dir))
        a, m, prior_manifest = transform_graph_prior(a.to(device), m.to(device), cfg.variant, cfg.model_seed)
        write_json(run_dir / "graph_prior_manifest.json", prior_manifest)
        norm = fit_normalization_stats(legacy_cfg, nodes)
        train_ds, val_ds, train_loader, val_loader, _, split_source = make_dataloaders(
            legacy_cfg, nodes, norm
        )
        model = build_revision_model(cfg, legacy_cfg, len(nodes), device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=legacy_cfg.lr, weight_decay=legacy_cfg.weight_decay)

        history: list[dict] = []
        best = math.inf
        best_epoch = 0
        patience = 0
        start_epoch = 0
        last_path = run_dir / "last.pt"
        if last_path.exists() and not force:
            checkpoint = torch.load(last_path, map_location=device)
            if checkpoint.get("config_hash") == cfg.config_hash:
                model.load_state_dict(checkpoint["model"], strict=True)
                optimizer.load_state_dict(checkpoint["optimizer"])
                history = list(checkpoint.get("history", []))
                start_epoch = int(checkpoint.get("epoch", 0))
                best = float(checkpoint.get("best_val", math.inf))
                best_epoch = int(checkpoint.get("best_epoch", 0))
                patience = int(checkpoint.get("patience", 0))

        print(
            "[TRAIN CONFIG] "
            f"device={device} train_windows={len(train_ds)} val_windows={len(val_ds)} "
            f"train_batches={len(train_loader)} val_batches={len(val_loader)} "
            f"batch_size={cfg.batch_size} epochs={cfg.epochs} "
            f"early_stop_patience={legacy_cfg.early_stop_patience} "
            f"cross_dim_loss={legacy_cfg.cross_dim_loss_enabled} randomness={randomness}",
            flush=True,
        )

        patience_limit = int(legacy_cfg.early_stop_patience)
        end_epoch = start_epoch if start_epoch > 0 and patience >= patience_limit else cfg.epochs
        if end_epoch == start_epoch:
            print(
                f"[RESUME] early-stop state already reached at epoch={start_epoch}; "
                "continuing with best-checkpoint evaluation",
                flush=True,
            )
        epoch_progress = tqdm(
            range(start_epoch, end_epoch),
            total=cfg.epochs,
            initial=start_epoch,
            desc=f"{cfg.experiment_id}/{cfg.dataset}/{cfg.variant}/seed_{cfg.model_seed}",
            unit="epoch",
            dynamic_ncols=False,
        )
        for epoch in epoch_progress:
            train_metrics = _run_epoch(
                model,
                train_loader,
                a,
                m,
                legacy_cfg,
                device,
                optimizer,
                progress_desc=f"train e{epoch + 1:03d}/{cfg.epochs}",
            )
            with torch.no_grad():
                val_metrics = _run_epoch(
                    model,
                    val_loader,
                    a,
                    m,
                    legacy_cfg,
                    device,
                    None,
                    progress_desc=f"val   e{epoch + 1:03d}/{cfg.epochs}",
                )
            record = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_cross_dim_loss": train_metrics["cross"],
            }
            history.append(record)
            improved = val_metrics["loss"] < (best - float(legacy_cfg.early_stop_min_delta))
            if improved:
                best = float(val_metrics["loss"])
                best_epoch = epoch + 1
                patience = 0
            else:
                patience += 1
            epoch_progress.set_postfix(
                train_loss=f"{train_metrics['loss']:.6g}",
                val_loss=f"{val_metrics['loss']:.6g}",
                best_val=f"{best:.6g}",
                patience=f"{patience}/{legacy_cfg.early_stop_patience}",
            )
            checkpoint = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "best_val": best,
                "best_epoch": best_epoch,
                "patience": patience,
                "config_hash": cfg.config_hash,
                "revision_config": cfg.to_dict(),
                "sensor_names": nodes,
            }
            torch.save(checkpoint, last_path)
            if improved:
                torch.save(checkpoint, run_dir / "best.pt")
            if patience >= patience_limit:
                tqdm.write(
                    f"[EARLY STOP] epoch={epoch + 1} best_epoch={best_epoch} "
                    f"best_val={best:.6g}"
                )
                break
        epoch_progress.close()

        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8")
        write_json(run_dir / "history.json", history)
        _save_loss_plot(history, run_dir / "loss_curve.png")
        split_payload = {
            "data_split_seed": cfg.data_split_seed,
            "model_seed": cfg.model_seed,
            "split_source": split_source,
            "train_flights": sorted(train_ds.flights),
            "val_flights": sorted(val_ds.flights),
        }
        write_json(run_dir / "split_flights.json", split_payload)

        best_checkpoint = torch.load(run_dir / "best.pt", map_location=device)
        model.load_state_dict(best_checkpoint["model"], strict=True)
        _, val_paths, failure_paths = resolve_flight_splits(dataset_root=Path(legacy_cfg.data_root))
        val_scores, _, _ = _score_loader(
            model, _loader_for_paths(val_paths, norm, nodes, cfg, legacy_cfg), a, m,
            cfg, legacy_cfg, device, capture_graph=False,
        )
        val_scores.to_csv(run_dir / "val_normal_residuals.csv", index=False, encoding="utf-8")
        val_scores = aggregate_dataframe(val_scores, cfg.aggregator, cfg.ema_alpha)
        val_scores[["flight", "t_start", "t_end", "t_mid", "raw_total_score", "total_score"]].to_csv(
            run_dir / "val_normal_scores.csv", index=False, encoding="utf-8"
        )

        failure_scores, graph_arrays, corruption_log = _score_loader(
            model, _loader_for_paths(failure_paths, norm, nodes, cfg, legacy_cfg), a, m,
            cfg, legacy_cfg, device, capture_graph=True,
        )
        failure_scores = aggregate_dataframe(failure_scores, cfg.aggregator, cfg.ema_alpha)
        infer_dir = run_dir / "infer_tcngatre_failure"
        infer_dir.mkdir(parents=True, exist_ok=True)
        failure_scores.to_csv(
            infer_dir / "all_failure_window_forecast_residual.csv", index=False, encoding="utf-8-sig"
        )
        failure_scores[[
            "flight", "current_index", "t_start", "t_end", "t_mid",
            "raw_total_score", "total_score", "valid_dim_count", "aggregation_method",
        ]].to_csv(infer_dir / "sequence_scores.csv", index=False, encoding="utf-8")
        if graph_arrays is not None:
            np.savez_compressed(infer_dir / "graph_windows.npz", **graph_arrays)
            failure_scores[["flight", "t_start", "t_end", "t_mid", "graph_row"]].to_csv(
                infer_dir / "graph_windows_index.csv", index=False, encoding="utf-8"
            )
        write_json(infer_dir / "corruption_manifest.json", corruption_log)
        primary = evaluate_scores(run_dir, cfg, legacy_cfg)
        integrity = verify_snapshot()
        done = {
            "status": "complete",
            "config_hash": cfg.config_hash,
            "run_dir": str(run_dir),
            "best_val_loss": float(best),
            "best_epoch": int(best_epoch),
            "stopped_epoch": int(history[-1]["epoch"]),
            "early_stopped": bool(history[-1]["epoch"] < cfg.epochs),
            "primary_metrics": primary,
            "legacy_integrity": integrity,
        }
        write_json(done_path, done)
        failed = run_dir / "FAILED.json"
        if failed.exists():
            failed.unlink()
        return done
    except Exception as exc:
        failure = {
            "status": "failed",
            "config_hash": cfg.config_hash,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "FAILED.json", failure)
        raise


def execute_robustness_inference(cfg: RevisionConfig, source_run: Path, force: bool = False) -> dict:
    """Evaluate a clean full checkpoint under a deterministic test-time corruption."""
    verify_snapshot()
    source_run = Path(source_run)
    if not (source_run / "DONE.json").exists():
        raise FileNotFoundError(f"Clean source run is incomplete: {source_run}")
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    done_path = run_dir / "DONE.json"
    if done_path.exists() and not force:
        payload = json.loads(done_path.read_text(encoding="utf-8"))
        if payload.get("config_hash") == cfg.config_hash:
            return {"status": "skipped_complete", **payload}

    legacy_cfg = cfg.to_legacy()
    clean_cfg = replace(cfg, experiment_id="ex01", variant="full", corruption_kind="none", corruption_level=0.0)
    clean_legacy = clean_cfg.to_legacy()
    device = _device(legacy_cfg)
    nodes, a, m = load_graph(Path(clean_legacy.graph_dir))
    a, m = a.to(device), m.to(device)
    norm = load_minmax_stats(source_run / "normalization_stats.json")
    model = build_revision_model(clean_cfg, clean_legacy, len(nodes), device)
    checkpoint = torch.load(source_run / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    _, _, failure_paths = resolve_flight_splits(dataset_root=Path(clean_legacy.data_root))
    scores, graphs, corruption_log = _score_loader(
        model,
        _loader_for_paths(failure_paths, norm, nodes, cfg, legacy_cfg),
        a,
        m,
        cfg,
        legacy_cfg,
        device,
        capture_graph=True,
    )
    scores = aggregate_dataframe(scores, cfg.aggregator, cfg.ema_alpha)
    infer_dir = run_dir / "infer_tcngatre_failure"
    infer_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(infer_dir / "all_failure_window_forecast_residual.csv", index=False, encoding="utf-8-sig")
    scores[[
        "flight", "current_index", "t_start", "t_end", "t_mid",
        "raw_total_score", "total_score", "valid_dim_count", "aggregation_method",
    ]].to_csv(infer_dir / "sequence_scores.csv", index=False, encoding="utf-8")
    if graphs is not None:
        np.savez_compressed(infer_dir / "graph_windows.npz", **graphs)
    source_calibration = source_run / "val_normal_scores.csv"
    pd.read_csv(source_calibration).to_csv(run_dir / "val_normal_scores.csv", index=False, encoding="utf-8")
    write_json(run_dir / "config_resolved.json", cfg.to_dict())
    write_json(run_dir / "provenance.json", {
        **environment_payload(REPO_ROOT),
        "clean_checkpoint_source": str(source_run / "best.pt"),
    })
    write_json(infer_dir / "corruption_manifest.json", corruption_log)
    primary = evaluate_scores(run_dir, cfg, legacy_cfg)
    done = {
        "status": "complete",
        "config_hash": cfg.config_hash,
        "source_run": str(source_run),
        "primary_metrics": primary,
        "legacy_integrity": verify_snapshot(),
    }
    write_json(done_path, done)
    return done
