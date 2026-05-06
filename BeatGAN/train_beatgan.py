from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


BUNDLE_ROOT = Path(__file__).resolve().parent.parent
RECURRENT_AE_ROOT = BUNDLE_ROOT / "Recurrent_AE"
if str(RECURRENT_AE_ROOT) not in sys.path:
    sys.path.insert(0, str(RECURRENT_AE_ROOT))

from data.tranad_dataset import TranADWindowDataset, collate_tranad_windows, resolve_flight_splits  # noqa: E402
from utils.io import ensure_dir  # noqa: E402
from utils.normalization import fit_train_minmax, save_minmax_stats  # noqa: E402
from utils.seed import set_seed  # noqa: E402

from beatgan_model import BeatGAN, BeatGANConfig as BeatGANModelConfig, window_batch_to_beatgan_input
from beatganconfig import BeatGANRunConfig


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train BeatGAN on a bundled dataset.")
    parser.add_argument("--dataset", choices=["alfa", "alfa4hz", "simulate", "gpsdata"], default="alfa")
    return parser.parse_args(argv)


def resolve_device(cfg: BeatGANRunConfig) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_history(history: list[dict], run_root: Path):
    (run_root / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(run_root / "history.csv", index=False, encoding="utf-8")

    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in ["d_loss", "g_loss", "g_rec", "g_adv", "val_score_mean"]:
        if col in df.columns:
            ax.plot(df["epoch"], df[col], label=col, linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / score")
    ax.set_title("BeatGAN training history")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run_root / "loss_curve.png", dpi=150)
    plt.close(fig)


def build_loader(dataset: TranADWindowDataset, cfg: BeatGANRunConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=bool(shuffle),
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_tranad_windows,
    )


def evaluate(model: BeatGAN, loader: DataLoader, score_mode: str) -> dict[str, float]:
    total_score = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            window = window_batch_to_beatgan_input(batch["window"])
            score = model.anomaly_components(window, mode=score_mode)["final_score"]
            batch_size = int(score.shape[0])
            total_score += float(score.mean().item()) * batch_size
            count += batch_size
    return {"val_score_mean": total_score / max(count, 1)}


def collect_scores(model: BeatGAN, loader: DataLoader, score_mode: str) -> pd.DataFrame:
    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            window = window_batch_to_beatgan_input(batch["window"])
            score = model.anomaly_components(window, mode=score_mode)["final_score"].detach().cpu().numpy()
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
    cfg = BeatGANRunConfig(dataset_name=args.dataset)
    ensure_dir(cfg.run_root)
    set_seed(cfg.seed)
    device = resolve_device(cfg)

    train_flights, val_flights, _ = resolve_flight_splits(dataset_root=cfg.data_root)
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

    model_cfg = BeatGANModelConfig(
        in_channels=int(train_dataset.num_features),
        seq_len=int(cfg.window_size),
        nz=int(cfg.latent_dim),
        ndf=int(cfg.base_channels),
        ngf=int(cfg.base_channels),
        lambda_adv=float(cfg.lambda_adv),
        lr=float(cfg.lr),
        beta1=float(cfg.beta1),
        beta2=float(cfg.beta2),
        device=str(device),
    )
    model = BeatGAN(model_cfg)

    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict] = []

    for epoch in range(1, cfg.num_epochs + 1):
        model.G.train()
        model.D.train()
        meter = {
            "d_loss": 0.0,
            "d_real": 0.0,
            "d_fake": 0.0,
            "g_loss": 0.0,
            "g_rec": 0.0,
            "g_adv": 0.0,
        }
        steps = 0

        progress = tqdm(train_loader, desc=f"train beatgan epoch {epoch:03d}", leave=False)
        for batch in progress:
            window = window_batch_to_beatgan_input(batch["window"])
            losses = model.train_step(window)
            for key in meter:
                meter[key] += float(losses[key])
            steps += 1
            progress.set_postfix(g_loss=f"{meter['g_loss'] / max(steps, 1):.6f}")

        for key in meter:
            meter[key] /= max(steps, 1)

        val_stats = evaluate(model=model, loader=val_loader, score_mode=cfg.score_mode)
        row = {
            "epoch": epoch,
            **meter,
            **val_stats,
        }
        history.append(row)
        print(
            f"[epoch {epoch:03d}] "
            f"d_loss={meter['d_loss']:.6f} "
            f"g_loss={meter['g_loss']:.6f} "
            f"g_rec={meter['g_rec']:.6f} "
            f"g_adv={meter['g_adv']:.6f} "
            f"val_score_mean={val_stats['val_score_mean']:.6f}"
        )

        current_val = float(val_stats["val_score_mean"])
        if current_val < (best_val - cfg.early_stop_min_delta):
            best_val = current_val
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "generator_state_dict": model.G.state_dict(),
                    "discriminator_state_dict": model.D.state_dict(),
                    "input_dim": int(train_dataset.num_features),
                    "feature_names": train_dataset.feature_names,
                    "epoch": int(epoch),
                    "best_val_score_mean": float(best_val),
                    "model_config": model_cfg.__dict__,
                    "run_config": cfg.to_dict(),
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
            "generator_state_dict": model.G.state_dict(),
            "discriminator_state_dict": model.D.state_dict(),
            "input_dim": int(train_dataset.num_features),
            "feature_names": train_dataset.feature_names,
            "epoch": int(history[-1]["epoch"]) if history else 0,
            "best_val_score_mean": float(best_val),
            "model_config": model_cfg.__dict__,
            "run_config": cfg.to_dict(),
        },
        cfg.run_root / "last.pt",
    )
    save_history(history=history, run_root=cfg.run_root)

    best_ckpt = torch.load(cfg.run_root / "best.pt", map_location=device)
    model.G.load_state_dict(best_ckpt["generator_state_dict"])
    model.D.load_state_dict(best_ckpt["discriminator_state_dict"])
    val_scores_df = collect_scores(model=model, loader=val_loader, score_mode=cfg.score_mode)
    val_scores_df.to_csv(cfg.run_root / "val_normal_scores.csv", index=False, encoding="utf-8")

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_score_mean": float(best_val),
        "device": str(device),
        "num_features": int(train_dataset.num_features),
        "num_train_windows": int(len(train_dataset)),
        "num_val_windows": int(len(val_dataset)),
    }
    (cfg.run_root / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] BeatGAN training finished. Outputs saved to: {cfg.run_root}")


if __name__ == "__main__":
    main()
