from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.alfa_shared import build_flight_path_maps
from data.usad_window_dataset import USADWindowDataset, collate_usad_windows
from models.usad import USAD
from usadconfig import USADConfig
from utils.global_threshold import fit_single_global_threshold, save_global_threshold
from utils.io import ensure_dir
from utils.normalization import fit_train_minmax, save_minmax_stats
from utils.seed import set_seed


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train USAD on a bundled dataset.")
    parser.add_argument("--dataset", choices=["alfa", "alfa4hz", "simulate", "gpsdata"], default="alfa")
    return parser.parse_args(argv)


def resolve_device(cfg: USADConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loader(
    flight_paths: dict[str, Path],
    norm_stats: dict,
    cfg: USADConfig,
    shuffle: bool,
) -> tuple[USADWindowDataset, DataLoader]:
    dataset = USADWindowDataset(
        csv_root=None,
        flights=None,
        normalization_stats=norm_stats,
        window_size=cfg.window_size,
        stride=cfg.sample_stride,
        trim_leading_sec=cfg.trim_leading_sec,
        use_replication_padding=cfg.use_replication_padding,
        flight_paths=flight_paths,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=bool(shuffle),
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_usad_windows,
    )
    return dataset, loader


def evaluate(model: USAD, loader: DataLoader, device: torch.device, alpha: float, epoch_index: int) -> dict[str, float]:
    model.eval()
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_score = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            out = model.compute_losses(x=x, epoch_index=epoch_index)
            score = model.anomaly_score(x=x, alpha=alpha)
            batch_size = int(x.shape[0])
            total_loss1 += float(out["loss1"].item()) * batch_size
            total_loss2 += float(out["loss2"].item()) * batch_size
            total_score += float(score.mean().item()) * batch_size
            count += batch_size
    denom = max(count, 1)
    return {
        "val_loss1": total_loss1 / denom,
        "val_loss2": total_loss2 / denom,
        "val_score_mean": total_score / denom,
    }


def collect_scores(model: USAD, loader: DataLoader, device: torch.device, alpha: float) -> pd.DataFrame:
    model.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            scores = model.anomaly_score(x=x, alpha=alpha).detach().cpu().numpy()
            for i in range(len(scores)):
                rows.append(
                    {
                        "flight": str(batch["flight"][i]),
                        "sample_index": int(batch["sample_index"][i]),
                        "current_index": int(batch["current_index"][i].item()),
                        "t_start": float(batch["t_start"][i].item()),
                        "t_end": float(batch["t_end"][i].item()),
                        "t_mid": float(batch["t_mid"][i].item()),
                        "total_score": float(scores[i]),
                        "raw_total_score": float(scores[i]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["flight", "t_start"], kind="mergesort").reset_index(drop=True)


def save_history(history: list[dict], run_root: Path):
    history_path = run_root / "history.json"
    history_csv_path = run_root / "history.csv"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(history_csv_path, index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5))
    df = pd.DataFrame(history)
    ax.plot(df["epoch"], df["train_loss1"], label="train_loss1", linewidth=1.4)
    ax.plot(df["epoch"], df["train_loss2"], label="train_loss2", linewidth=1.4)
    ax.plot(df["epoch"], df["val_score_mean"], label="val_score_mean", linewidth=1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / score")
    ax.set_title("USAD training history")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run_root / "loss_curve.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    cfg = USADConfig(dataset_name=args.dataset)
    ensure_dir(cfg.run_root)
    set_seed(cfg.seed)
    device = resolve_device(cfg)

    train_flights, val_flights, _, meta = build_flight_path_maps(cfg.data_root)
    norm_stats = fit_train_minmax(
        flight_paths=train_flights,
        trim_leading_sec=cfg.trim_leading_sec,
    )
    save_minmax_stats(norm_stats, cfg.normalization_stats_path)

    train_dataset, train_loader = build_loader(train_flights, norm_stats=norm_stats, cfg=cfg, shuffle=True)
    _, val_loader = build_loader(val_flights, norm_stats=norm_stats, cfg=cfg, shuffle=False)
    cfg.save(cfg.run_root / "config.json")

    print(
        "[USAD] "
        f"device={device} "
        f"wide_root={meta['wide_root']} "
        f"train_flights={len(train_flights)} "
        f"val_flights={len(val_flights)} "
        f"window_size={cfg.window_size}"
    )

    model = USAD(
        input_dim=train_dataset.feature_dim,
        encoder_hidden_dims=cfg.encoder_hidden_dims,
        latent_dim=cfg.latent_dim,
        decoder_hidden_dims=cfg.decoder_hidden_dims,
        activation=cfg.activation,
        dropout=cfg.dropout,
        use_layernorm=cfg.use_layernorm,
    ).to(device)

    optimizer1 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder1.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    optimizer2 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder2.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val = float("inf")
    best_epoch = -1
    stale_epochs = 0
    history: list[dict] = []

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total_loss1 = 0.0
        total_loss2 = 0.0
        count = 0
        progress = tqdm(train_loader, desc=f"train usad epoch {epoch:03d}", leave=False)
        for batch in progress:
            x = batch["x"].to(device)
            batch_size = int(x.shape[0])

            optimizer1.zero_grad(set_to_none=True)
            out1 = model.compute_losses(x=x, epoch_index=epoch)
            loss1 = out1["loss1"]
            loss1.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.encoder.parameters()) + list(model.decoder1.parameters()),
                    max_norm=cfg.grad_clip,
                )
            optimizer1.step()

            optimizer2.zero_grad(set_to_none=True)
            out2 = model.compute_losses(x=x, epoch_index=epoch)
            loss2 = out2["loss2"]
            loss2.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.encoder.parameters()) + list(model.decoder2.parameters()),
                    max_norm=cfg.grad_clip,
                )
            optimizer2.step()

            total_loss1 += float(loss1.item()) * batch_size
            total_loss2 += float(loss2.item()) * batch_size
            count += batch_size
            progress.set_postfix(
                train_loss1=f"{total_loss1 / max(count, 1):.6f}",
                train_loss2=f"{total_loss2 / max(count, 1):.6f}",
            )

        train_loss1 = total_loss1 / max(count, 1)
        train_loss2 = total_loss2 / max(count, 1)
        val_stats = evaluate(model=model, loader=val_loader, device=device, alpha=cfg.alpha, epoch_index=epoch)

        row = {
            "epoch": epoch,
            "train_loss1": train_loss1,
            "train_loss2": train_loss2,
            **val_stats,
        }
        history.append(row)
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss1={train_loss1:.6f} "
            f"train_loss2={train_loss2:.6f} "
            f"val_score_mean={val_stats['val_score_mean']:.6f}"
        )

        current_val = float(val_stats["val_score_mean"])
        if current_val < (best_val - cfg.early_stop_min_delta):
            best_val = current_val
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": int(train_dataset.feature_dim),
                    "feature_names": train_dataset.feature_names,
                    "epoch": int(epoch),
                    "best_val_score_mean": float(best_val),
                    "config": cfg.to_dict(),
                },
                cfg.run_root / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.early_stop_patience:
                print(f"[early-stop] patience reached at epoch {epoch}")
                break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(train_dataset.feature_dim),
            "feature_names": train_dataset.feature_names,
            "epoch": int(history[-1]["epoch"]) if history else 0,
            "best_val_score_mean": float(best_val),
            "config": cfg.to_dict(),
        },
        cfg.run_root / "last.pt",
    )
    save_history(history=history, run_root=cfg.run_root)

    best_ckpt = torch.load(cfg.run_root / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    val_scores_df = collect_scores(model=model, loader=val_loader, device=device, alpha=cfg.alpha)
    val_scores_df.to_csv(cfg.run_root / "val_normal_scores.csv", index=False, encoding="utf-8")

    threshold_payload = fit_single_global_threshold(
        scores=val_scores_df["total_score"].to_numpy(dtype=float),
        mode=cfg.threshold_fit_mode,
        sigma_k=cfg.threshold_sigma_k,
        mad_k=cfg.threshold_mad_k,
        quantile=cfg.threshold_quantile,
    )
    threshold_payload["score_col"] = "total_score"
    threshold_payload["fit_source"] = "val_normal"
    save_global_threshold(threshold_payload, cfg.run_root / "global_threshold.json")

    print(f"[OK] best epoch={best_epoch} val_score_mean={best_val:.6f}")
    print(f"[OK] global threshold saved to: {cfg.run_root / 'global_threshold.json'}")


if __name__ == "__main__":
    main()
