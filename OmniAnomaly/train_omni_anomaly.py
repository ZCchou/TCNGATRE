"""Train OmniAnomaly on ALFA dataset.

Usage:
  python train_omni_anomaly.py

Follows the same pattern as train_recurrent_ae.py:
  1. Load normal flights, fit MinMax normalization
  2. Create sliding-window datasets (reuses TranADWindowDataset)
  3. Train with KL warmup and early stopping
  4. Save val_normal_scores.csv and global_threshold.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.tranad_dataset import (
    TranADWindowDataset,
    collate_tranad_windows,
    resolve_flight_splits,
)
from models.omni_anomaly import build_omni_anomaly
from omnianomalyconfig import OmniAnomalyConfig
from utils.io import ensure_dir
from utils.normalization import fit_train_minmax, save_minmax_stats
from utils.seed import set_seed


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train OmniAnomaly on a bundled dataset.")
    parser.add_argument("--dataset", choices=["alfa", "simulate", "gpsdata"], default="alfa")
    return parser.parse_args(argv)


def resolve_device(cfg: OmniAnomalyConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_history(history: list[dict], run_root: Path):
    (run_root / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(run_root / "history.csv", index=False, encoding="utf-8")

    df = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("OmniAnomaly training history")

    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_total"], label="train_total", lw=1.4)
    ax.plot(df["epoch"], df["val_total"], label="val_total", lw=1.4)
    ax.set_title("Total loss")
    ax.legend()
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.plot(df["epoch"], df["train_recon"], label="train_recon", lw=1.4)
    ax.plot(df["epoch"], df["val_recon"], label="val_recon", lw=1.4)
    ax.set_title("Reconstruction NLL")
    ax.legend()
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.plot(df["epoch"], df["train_kl"], label="train_kl", lw=1.4)
    ax.plot(df["epoch"], df["val_kl"], label="val_kl", lw=1.4)
    ax.set_title("KL divergence")
    ax.legend()
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.plot(df["epoch"], df["train_flow"], label="train_flow", lw=1.4)
    ax.plot(df["epoch"], df["val_flow"], label="val_flow", lw=1.4)
    ax.set_title("Flow log|det| (neg)")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(run_root / "loss_curve.png", dpi=150)
    plt.close(fig)


def kl_annealing_weight(epoch: int, warmup_epochs: int) -> float:
    """Linear KL warmup: 0 at epoch 1, 1.0 at warmup_epochs+1."""
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch) / float(warmup_epochs))


def build_loader(dataset: TranADWindowDataset, cfg: OmniAnomalyConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=bool(shuffle),
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_tranad_windows,
    )


def train_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    kl_weight: float,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_flow = 0.0
    count = 0

    for batch in loader:
        window = batch["window"].to(device)
        B = window.shape[0]

        optimizer.zero_grad(set_to_none=True)
        out = model(window, kl_weight=kl_weight)
        out.total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += float(out.total_loss.item()) * B
        total_recon += float(out.recon_loss.item()) * B
        total_kl += float(out.kl_loss.item()) * B
        total_flow += float(out.flow_loss.item()) * B
        count += B

    d = max(count, 1)
    return {
        "total": total_loss / d,
        "recon": total_recon / d,
        "kl": total_kl / d,
        "flow": total_flow / d,
    }


@torch.no_grad()
def eval_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
    kl_weight: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_flow = 0.0
    total_score = 0.0
    count = 0

    for batch in loader:
        window = batch["window"].to(device)
        B = window.shape[0]

        out = model(window, kl_weight=kl_weight)
        score = model.anomaly_score(window, mc_samples=1)

        total_loss += float(out.total_loss.item()) * B
        total_recon += float(out.recon_loss.item()) * B
        total_kl += float(out.kl_loss.item()) * B
        total_flow += float(out.flow_loss.item()) * B
        total_score += float(score["final_score"].sum().item())
        count += B

    d = max(count, 1)
    return {
        "total": total_loss / d,
        "recon": total_recon / d,
        "kl": total_kl / d,
        "flow": total_flow / d,
        "score_mean": total_score / d,
    }


@torch.no_grad()
def collect_val_scores(model, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        window = batch["window"].to(device)
        scores = model.anomaly_score(window, mc_samples=1)
        fs = scores["final_score"].cpu().numpy()
        for i in range(len(fs)):
            rows.append({
                "flight": str(batch["flight"][i]),
                "current_index": int(batch["current_index"][i].item()),
                "t_start": float(batch["t"][i].item()),
                "t_end": float(batch["t"][i].item()),
                "t_mid": float(batch["t"][i].item()),
                "total_score": float(fs[i]),
            })
    return pd.DataFrame(rows).sort_values(["flight", "current_index"]).reset_index(drop=True)


def fit_threshold(val_scores: np.ndarray, mode: str, sigma_k: float) -> dict:
    """Fit anomaly threshold from validation normal scores."""
    mu = float(np.mean(val_scores))
    sigma = float(np.std(val_scores, ddof=1))
    threshold = mu + sigma_k * sigma
    return {
        "fit_mode": mode,
        "threshold": threshold,
        "mean": mu,
        "std": sigma,
        "sigma_k": sigma_k,
        "num_samples": int(len(val_scores)),
        "score_col": "total_score",
        "fit_source": "val_normal",
    }


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    cfg = OmniAnomalyConfig(dataset_name=args.dataset)
    ensure_dir(cfg.run_root)
    set_seed(cfg.seed)
    device = resolve_device(cfg)
    print(f"[OmniAnomaly] device={device}  run_root={cfg.run_root}")

    # Data splits
    train_flights, val_flights, _ = resolve_flight_splits(
        dataset_root=cfg.data_root,
    )
    print(f"[DATA] train={len(train_flights)} val={len(val_flights)} flights")

    # Normalization
    norm_stats = fit_train_minmax(
        flight_paths=train_flights,
        trim_leading_sec=cfg.trim_leading_sec,
    )
    save_minmax_stats(norm_stats, cfg.normalization_stats_path)

    # Datasets
    train_ds = TranADWindowDataset(
        csv_root=None,
        flights=None,
        normalization_stats=norm_stats,
        window_size=cfg.window_size,
        stride=cfg.stride_train,
        trim_leading_sec=cfg.trim_leading_sec,
        use_replication_padding=cfg.use_replication_padding,
        context_mode=cfg.context_mode,
        flight_paths=train_flights,
    )
    val_ds = TranADWindowDataset(
        csv_root=None,
        flights=None,
        normalization_stats=norm_stats,
        window_size=cfg.window_size,
        stride=cfg.stride_val,
        trim_leading_sec=cfg.trim_leading_sec,
        use_replication_padding=cfg.use_replication_padding,
        context_mode=cfg.context_mode,
        flight_paths=val_flights,
    )
    cfg.num_features = int(train_ds.num_features)
    cfg.save(cfg.run_root / "config.json")
    print(f"[DATA] train={len(train_ds)} val={len(val_ds)} samples  features={cfg.num_features}")

    train_loader = build_loader(train_ds, cfg, shuffle=True)
    val_loader = build_loader(val_ds, cfg, shuffle=False)

    # Model
    model = build_omni_anomaly(
        input_dim=cfg.num_features,
        rnn_hidden=cfg.rnn_hidden_dim,
        latent_dim=cfg.latent_dim,
        rnn_layers=cfg.rnn_layers,
        dropout=cfg.dropout,
        num_flows=cfg.num_planar_flows,
        decoder_rnn_hidden=cfg.decoder_rnn_hidden_dim,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] OmniAnomaly  params={num_params:,}  latent={cfg.latent_dim}  flows={cfg.num_planar_flows}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma,
    )

    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict] = []

    for epoch in range(1, cfg.num_epochs + 1):
        kl_w = kl_annealing_weight(epoch, cfg.kl_warmup_epochs)

        train_stats = train_epoch(model, train_loader, optimizer, device, kl_w, cfg.grad_clip)
        val_stats = eval_epoch(model, val_loader, device, kl_w)
        scheduler.step()

        row = {
            "epoch": epoch,
            "kl_weight": kl_w,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_total": train_stats["total"],
            "train_recon": train_stats["recon"],
            "train_kl": train_stats["kl"],
            "train_flow": train_stats["flow"],
            "val_total": val_stats["total"],
            "val_recon": val_stats["recon"],
            "val_kl": val_stats["kl"],
            "val_flow": val_stats["flow"],
            "val_score_mean": val_stats["score_mean"],
        }
        history.append(row)
        print(
            f"[epoch {epoch:03d}] kl_w={kl_w:.2f}  "
            f"train={train_stats['total']:.4f} (r={train_stats['recon']:.4f} k={train_stats['kl']:.4f} f={train_stats['flow']:.4f})  "
            f"val={val_stats['total']:.4f} score_mean={val_stats['score_mean']:.6f}"
        )

        # Early stopping on val_score_mean
        current_val = float(val_stats["score_mean"])
        if current_val < (best_val - cfg.early_stop_min_delta):
            best_val = current_val
            best_epoch = epoch
            stale = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": cfg.num_features,
                "feature_names": train_ds.feature_names,
                "epoch": epoch,
                "best_val_score_mean": best_val,
                "config": cfg.to_dict(),
            }, cfg.run_root / "best.pt")
        else:
            stale += 1
            if stale >= cfg.early_stop_patience:
                print(f"[early-stop] patience reached at epoch {epoch}")
                break

    # Save last checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": cfg.num_features,
        "feature_names": train_ds.feature_names,
        "epoch": history[-1]["epoch"] if history else 0,
        "best_val_score_mean": best_val,
        "config": cfg.to_dict(),
    }, cfg.run_root / "last.pt")
    save_history(history, cfg.run_root)

    # Collect val scores from best checkpoint
    ckpt = torch.load(cfg.run_root / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    val_df = collect_val_scores(model, val_loader, device)
    val_df.to_csv(cfg.run_root / "val_normal_scores.csv", index=False, encoding="utf-8")

    # Fit threshold
    val_scores = val_df["total_score"].to_numpy(dtype=float)
    thr_info = fit_threshold(val_scores, cfg.threshold_fit_mode, cfg.threshold_sigma_k)
    (cfg.run_root / "global_threshold.json").write_text(
        json.dumps(thr_info, indent=2), encoding="utf-8"
    )

    print(f"\n[OK] OmniAnomaly best epoch={best_epoch}  val_score_mean={best_val:.6f}")
    print(f"[OK] threshold={thr_info['threshold']:.6g} (mu={thr_info['mean']:.6g} + {cfg.threshold_sigma_k}*sigma={thr_info['std']:.6g})")
    print(f"[OK] saved to: {cfg.run_root}")


if __name__ == "__main__":
    main()
