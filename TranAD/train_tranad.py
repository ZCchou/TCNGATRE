from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.tranad_dataset import TranADWindowDataset, collate_tranad_windows, resolve_flight_splits
from models.tranad import TranAD
from tranadconfig import TranADConfig
from utils.io import ensure_dir
from utils.normalization import fit_train_minmax, save_minmax_stats
from utils.seed import set_seed


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train TranAD on a bundled dataset.")
    parser.add_argument("--dataset", choices=["alfa", "simulate", "gpsdata"], default="alfa")
    return parser.parse_args(argv)


def resolve_device(cfg: TranADConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_history(history: list[dict], run_root: Path):
    (run_root / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(run_root / "history.csv", index=False, encoding="utf-8")

    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["train_loss1"], label="train_loss1", linewidth=1.4)
    ax.plot(df["epoch"], df["train_loss2"], label="train_loss2", linewidth=1.4)
    ax.plot(df["epoch"], df["val_score_mean"], label="val_score_mean", linewidth=1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / score")
    ax.set_title("TranAD training history")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run_root / "loss_curve.png", dpi=150)
    plt.close(fig)


def build_loader(dataset: TranADWindowDataset, cfg: TranADConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=bool(shuffle),
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_tranad_windows,
    )


def evaluate(model: TranAD, loader: DataLoader, device: torch.device, epoch_index: int) -> dict[str, float]:
    model.eval()
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_score = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            window = batch["window"].to(device)
            context = batch["context"].to(device)
            out = model.compute_losses(window=window, context=context, epoch_index=epoch_index)
            score = model.anomaly_components(window=window, context=context)["final_score"]
            batch_size = int(window.shape[0])
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


def collect_scores(model: TranAD, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            window = batch["window"].to(device)
            context = batch["context"].to(device)
            score = model.anomaly_components(window=window, context=context)["final_score"].detach().cpu().numpy()
            for i in range(len(score)):
                rows.append(
                    {
                        "flight": str(batch["flight"][i]),
                        "current_index": int(batch["current_index"][i].item()),
                        "t_start": float(batch["t"][i].item()),
                        "t_end": float(batch["t"][i].item()),
                        "t_mid": float(batch["t"][i].item()),
                        "raw_total_score": float(score[i]),
                        "total_score": float(score[i]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["flight", "current_index"], kind="mergesort").reset_index(drop=True)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    cfg = TranADConfig(dataset_name=args.dataset)
    if cfg.use_maml:
        raise NotImplementedError("The minimal engineering TranAD baseline keeps use_maml=false.")

    ensure_dir(cfg.run_root)
    set_seed(cfg.seed)
    device = resolve_device(cfg)

    train_flights, val_flights, _ = resolve_flight_splits(
        dataset_root=cfg.data_root,
    )
    norm_stats = fit_train_minmax(
        flight_paths=train_flights,
        trim_leading_sec=cfg.trim_leading_sec,
    )
    save_minmax_stats(norm_stats, cfg.normalization_stats_path)

    train_dataset = TranADWindowDataset(
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
    val_dataset = TranADWindowDataset(
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
    cfg.num_features = int(train_dataset.num_features)
    cfg.save(cfg.run_root / "config.json")

    train_loader = build_loader(train_dataset, cfg=cfg, shuffle=True)
    val_loader = build_loader(val_dataset, cfg=cfg, shuffle=False)

    model = TranAD(
        input_dim=train_dataset.num_features,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        use_positional_encoding=cfg.use_positional_encoding,
        eps_adv=cfg.eps_adv,
    ).to(device)

    optimizer1 = torch.optim.AdamW(model.params_for_loss1(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    optimizer2 = torch.optim.AdamW(model.params_for_loss2(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma)

    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict] = []

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total_loss1 = 0.0
        total_loss2 = 0.0
        count = 0
        progress = tqdm(train_loader, desc=f"train tranad epoch {epoch:03d}", leave=False)
        for batch in progress:
            window = batch["window"].to(device)
            context = batch["context"].to(device)
            batch_size = int(window.shape[0])

            optimizer1.zero_grad(set_to_none=True)
            out1 = model.compute_losses(window=window, context=context, epoch_index=epoch)
            loss1 = out1["loss1"]
            loss1.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(model.params_for_loss1()), max_norm=cfg.grad_clip)
            optimizer1.step()

            optimizer2.zero_grad(set_to_none=True)
            out2 = model.compute_losses(window=window, context=context, epoch_index=epoch)
            loss2 = out2["loss2"]
            loss2.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(model.params_for_loss2()), max_norm=cfg.grad_clip)
            optimizer2.step()

            total_loss1 += float(loss1.item()) * batch_size
            total_loss2 += float(loss2.item()) * batch_size
            count += batch_size
            progress.set_postfix(
                train_loss1=f"{total_loss1 / max(count, 1):.6f}",
                train_loss2=f"{total_loss2 / max(count, 1):.6f}",
            )

        scheduler1.step()
        scheduler2.step()

        train_loss1 = total_loss1 / max(count, 1)
        train_loss2 = total_loss2 / max(count, 1)
        val_stats = evaluate(model=model, loader=val_loader, device=device, epoch_index=epoch)
        row = {
            "epoch": epoch,
            "train_loss1": train_loss1,
            "train_loss2": train_loss2,
            **val_stats,
            "lr1": float(optimizer1.param_groups[0]["lr"]),
            "lr2": float(optimizer2.param_groups[0]["lr"]),
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
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": int(train_dataset.num_features),
                    "feature_names": train_dataset.feature_names,
                    "epoch": int(epoch),
                    "best_val_score_mean": float(best_val),
                    "config": cfg.to_dict(),
                },
                cfg.run_root / "best.pt",
            )
        else:
            stale += 1
            if stale >= cfg.early_stop_patience:
                print(f"[early-stop] patience reached at epoch {epoch}")
                break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(train_dataset.num_features),
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
    val_scores_df = collect_scores(model=model, loader=val_loader, device=device)
    val_scores_df.to_csv(cfg.run_root / "val_normal_scores.csv", index=False, encoding="utf-8")

    print(f"[OK] best epoch={best_epoch} val_score_mean={best_val:.6f}")
    print(f"[OK] val scores saved to: {cfg.run_root / 'val_normal_scores.csv'}")


if __name__ == "__main__":
    main()
