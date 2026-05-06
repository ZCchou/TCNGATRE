from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.stgtcn_window_dataset import (
    STGTCNWindowDataset,
    collate_stgtcn_windows,
    project_node_features,
    resolve_flight_splits,
)
from model.tcngatre import STGraphTCN
from base_config import TCNGATREConfig
from utils.normalization import load_wide_flight_frame, save_minmax_stats


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None
    return obj


def save_json(payload: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)


def save_loss_curve(history: Dict[str, List[float]], path: Path) -> None:
    epochs = [int(x) for x in history.get("epoch", [])]
    train_loss = [float(x) for x in history.get("train_loss", [])]
    val_loss = [float(x) for x in history.get("val_loss", [])]
    if not epochs or len(train_loss) != len(epochs):
        return
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_loss, label="train_loss", linewidth=1.6)
    ax.plot(epochs, val_loss, label="val_loss", linewidth=1.6)
    ax.set_title("TCNGATRE Training Loss Curve")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_graph(graph_dir: Path) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    keep_fp = graph_dir / "keep_columns.json"
    adj_fp = graph_dir / "adjacency_dense.csv"
    with keep_fp.open("r", encoding="utf-8") as f:
        node_names = [str(x) for x in json.load(f)]
    if not node_names:
        raise ValueError(f"No graph nodes found in {keep_fp}")
    df_a = pd.read_csv(adj_fp, index_col=0, low_memory=False)
    df_a.index = df_a.index.astype(str)
    df_a.columns = df_a.columns.astype(str)
    for c in node_names:
        if c not in df_a.index:
            df_a.loc[c] = 0.0
        if c not in df_a.columns:
            df_a[c] = 0.0
    df_a = df_a.reindex(index=node_names, columns=node_names).fillna(0.0)
    a = df_a.to_numpy(dtype=np.float32)
    np.fill_diagonal(a, 0.0)
    if float(a.max()) > 0.0:
        a = a / (float(a.max()) + 1e-6)
    m = (a > 0).astype(np.float32)
    isolated = np.where(m.sum(axis=1) <= 0.0)[0].tolist()
    for idx in isolated:
        m[idx, :] = 1.0
        m[:, idx] = 1.0
        m[idx, idx] = 1.0
    return node_names, torch.from_numpy(a), torch.from_numpy(m)


def build_model(cfg: TCNGATREConfig, num_nodes: int, device: torch.device) -> STGraphTCN:
    return STGraphTCN(
        num_nodes_hint=num_nodes,
        in_feat=1,
        d_model=cfg.d_model,
        short_kernel=cfg.short_kernel,
        tcn_layers=cfg.tcn_layers,
        tcn_blocks=cfg.tcn_blocks,
        dropout=cfg.dropout,
        eta=cfg.graph_eta,
        beta=cfg.graph_beta,
        out_feat=1,
        horizon=cfg.horizon_out,
        predict_logvar=False,
        graph_topk=None,
        graph_gate_init=cfg.graph_gate_init,
        interleave_every=cfg.interleave_every,
        num_hops=cfg.graph_num_hops,
    ).to(device)


def make_dataloaders(
    cfg: TCNGATREConfig,
    node_names: Sequence[str],
    normalization_stats: dict,
) -> tuple[STGTCNWindowDataset, STGTCNWindowDataset, DataLoader, DataLoader, DataLoader, str]:
    train_paths, val_paths, failure_paths = resolve_flight_splits(
        dataset_root=Path(cfg.data_root),
        split_info_path=Path(cfg.split_info_path),
    )
    ds_kwargs = dict(
        csv_root=None,
        flights=None,
        normalization_stats=normalization_stats,
        node_names=list(node_names),
        history_len=cfg.lookback,
        horizon=cfg.horizon_out,
        trim_leading_sec=cfg.trim_leading_sec,
        use_replication_padding=False,
    )
    train_ds = STGTCNWindowDataset(**ds_kwargs, stride=cfg.sample_stride, flight_paths=train_paths)
    val_ds = STGTCNWindowDataset(**ds_kwargs, stride=cfg.sample_stride, flight_paths=val_paths)
    failure_ds = STGTCNWindowDataset(**ds_kwargs, stride=cfg.sample_stride, flight_paths=failure_paths)

    dl_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_stgtcn_windows,
        pin_memory=torch.cuda.is_available(),
    )
    train_dl = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    failure_dl = DataLoader(failure_ds, shuffle=False, **dl_kwargs)
    split_source = str(Path(cfg.split_info_path).resolve())
    return train_ds, val_ds, train_dl, val_dl, failure_dl, split_source


def fit_normalization_stats(cfg: TCNGATREConfig, node_names: Sequence[str]) -> dict:
    train_paths, _, _ = resolve_flight_splits(
        dataset_root=Path(cfg.data_root),
        split_info_path=Path(cfg.split_info_path),
    )
    train_min = None
    train_max = None
    total_rows = 0
    for _, csv_path in train_paths.items():
        _, x_raw, cols = load_wide_flight_frame(csv_path=csv_path, trim_leading_sec=cfg.trim_leading_sec)
        if x_raw.size <= 0:
            continue
        x_nodes = project_node_features(
            x_raw=x_raw, feature_names_all=list(cols), node_names=list(node_names)
        )
        cur_min = np.nanmin(x_nodes, axis=0)
        cur_max = np.nanmax(x_nodes, axis=0)
        train_min = cur_min if train_min is None else np.minimum(train_min, cur_min)
        train_max = cur_max if train_max is None else np.maximum(train_max, cur_max)
        total_rows += int(x_nodes.shape[0])
    if train_min is None:
        raise ValueError("No valid training rows for normalization")
    all_nan = ~np.isfinite(train_min) | ~np.isfinite(train_max)
    train_min = np.where(all_nan, 0.0, train_min)
    train_max = np.where(all_nan, 1.0, train_max)
    scale = np.maximum(train_max - train_min, 1e-8)
    stats = {
        "feature_names": [str(x) for x in node_names],
        "train_min": train_min.astype(np.float64).tolist(),
        "train_max": train_max.astype(np.float64).tolist(),
        "scale": scale.astype(np.float64).tolist(),
        "all_nan_mask": all_nan.astype(np.int32).tolist(),
        "trim_leading_sec": float(cfg.trim_leading_sec),
        "num_train_flights": int(len(train_paths)),
        "num_train_rows": int(total_rows),
        "eps": float(1e-8),
    }
    save_minmax_stats(stats, Path(cfg.normalization_stats_path))
    return stats


def _masked_huber(diff: torch.Tensor, huber_beta: float) -> torch.Tensor:
    return torch.nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff), beta=float(huber_beta))


def compute_cross_dim_loss(
    model: STGraphTCN,
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    m: torch.Tensor,
    cfg: TCNGATREConfig,
) -> torch.Tensor:
    num_nodes = x.shape[2]
    device = x.device

    mask = torch.rand(num_nodes, device=device) < float(cfg.cross_dim_dropout_prob)
    max_mask = max(1, int(num_nodes * float(cfg.cross_dim_max_mask_ratio)))
    if int(mask.sum()) > max_mask:
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        keep = idx[torch.randperm(len(idx), device=device)[:max_mask]]
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        mask[keep] = True

    if int(mask.sum()) == 0:
        return torch.zeros((), device=device)

    x_masked = x.clone()
    x_masked[:, :, mask, :] = 0.0
    y_hat, _ = model(x_masked, a, m, short_patch=cfg.short_patch)
    return _masked_huber(y_hat[:, :, mask, 0] - y[:, :, mask, 0], cfg.huber_beta)


def compute_loss(pred: torch.Tensor, target: torch.Tensor, huber_beta: float) -> Dict[str, torch.Tensor]:
    diff = pred - target
    value_loss = _masked_huber(diff, huber_beta)
    if target.shape[1] >= 2:
        delta_loss = _masked_huber(
            (pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1]),
            huber_beta,
        )
    else:
        delta_loss = torch.zeros((), device=target.device, dtype=value_loss.dtype)
    return {"total_loss": value_loss + 0.20 * delta_loss, "value_loss": value_loss, "delta_loss": delta_loss}


def run_epoch(
    model: STGraphTCN,
    dl: DataLoader,
    a: torch.Tensor,
    m: torch.Tensor,
    cfg: TCNGATREConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    desc: str = "",
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "value_loss": 0.0, "delta_loss": 0.0, "cross_dim_loss": 0.0, "n": 0}
    for batch in tqdm(dl, desc=desc, unit="batch", dynamic_ncols=True, leave=False):
        x = batch["x"].to(device).float()
        y = batch["y"].to(device).float()
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        y_hat, _ = model(x, a, m, short_patch=cfg.short_patch)
        losses = compute_loss(y_hat[..., 0], y[..., 0], cfg.huber_beta)
        opt_loss = losses["total_loss"]

        cd_loss_val = 0.0
        if is_train and cfg.cross_dim_loss_enabled:
            cd_loss = compute_cross_dim_loss(model, x, y, a, m, cfg)
            opt_loss = losses["total_loss"] + float(cfg.cross_dim_lambda) * cd_loss
            cd_loss_val = float(cd_loss.detach().cpu())

        if is_train:
            if not torch.isfinite(opt_loss):
                raise RuntimeError("Non-finite loss during training")
            opt_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        totals["loss"] += float(losses["total_loss"].detach().cpu())
        totals["value_loss"] += float(losses["value_loss"].detach().cpu())
        totals["delta_loss"] += float(losses["delta_loss"].detach().cpu())
        totals["cross_dim_loss"] += cd_loss_val
        totals["n"] += 1
    n = max(totals["n"], 1)
    return {k: v / n for k, v in totals.items() if k != "n"}


@torch.no_grad()
def compute_val_normal_scores(
    model: STGraphTCN,
    cfg: TCNGATREConfig,
    a: torch.Tensor,
    m: torch.Tensor,
    device: torch.device,
    node_names: list[str],
    normalization_stats: dict,
    out_path: Path,
) -> None:
    _, val_paths, _ = resolve_flight_splits(
        dataset_root=Path(cfg.data_root),
        split_info_path=Path(cfg.split_info_path),
    )
    if not val_paths:
        return
    val_ds = STGTCNWindowDataset(
        csv_root=None, flights=None,
        normalization_stats=normalization_stats,
        node_names=node_names,
        history_len=cfg.lookback,
        horizon=cfg.horizon_out,
        stride=cfg.sample_stride,
        trim_leading_sec=cfg.trim_leading_sec,
        use_replication_padding=False,
        flight_paths=val_paths,
    )
    dl = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_stgtcn_windows,
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    rows = []
    alpha = float(cfg.score_temporal_smooth_alpha)
    for batch in tqdm(dl, desc="[ValNormal]", unit="batch", dynamic_ncols=True, leave=False):
        x = batch["x"].to(device).float()
        y = batch["y"].to(device).float()
        y_hat, _ = model(x, a, m, short_patch=cfg.short_patch)
        pred = y_hat[:, :, :, 0].detach().cpu().numpy().astype(np.float32)
        truth = y[:, :, :, 0].detach().cpu().numpy().astype(np.float32)
        value_resid = np.mean(np.abs(pred - truth), axis=1)
        raw_score = np.mean(value_resid, axis=1)
        for i, flight in enumerate(batch["flight"]):
            t_start = float(batch["t_future_start"][i].item())
            t_end = float(batch["t_future_end"][i].item())
            rows.append({
                "flight": str(flight),
                "t_start": t_start,
                "t_end": t_end,
                "t_mid": 0.5 * (t_start + t_end),
                "raw_total_score": float(raw_score[i]),
            })
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["flight", "t_start"], kind="mergesort").reset_index(drop=True)
    parts = []
    for _, g in df.groupby("flight", sort=False):
        g = g.sort_values("t_start").reset_index(drop=True).copy()
        scores = g["raw_total_score"].to_numpy(dtype=np.float64)
        smoothed = np.empty_like(scores)
        prev = np.nan
        for i, v in enumerate(scores):
            prev = v if not np.isfinite(prev) else alpha * v + (1.0 - alpha) * prev
            smoothed[i] = prev
        g["total_score"] = smoothed
        parts.append(g)
    result = pd.concat(parts, axis=0, ignore_index=True)
    ensure_dir(out_path.parent)
    result.to_csv(out_path, index=False, encoding="utf-8")


def main(runtime_cfg: TCNGATREConfig | None = None) -> None:
    cfg = TCNGATREConfig() if runtime_cfg is None else runtime_cfg
    set_seed(cfg.split_seed)
    out_dir = Path(cfg.run_root)
    ensure_dir(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else torch.device(cfg.device)

    node_names, a, m = load_graph(Path(cfg.graph_dir))
    a = a.to(device)
    m = m.to(device)

    norm_stats = fit_normalization_stats(cfg, node_names)
    train_ds, val_ds, train_dl, val_dl, _, split_source = make_dataloaders(cfg, node_names, norm_stats)

    num_nodes = len(node_names)
    model = build_model(cfg, num_nodes, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: Dict[str, list] = {k: [] for k in ["epoch", "train_loss", "val_loss", "train_value_loss", "train_delta_loss", "val_value_loss", "val_delta_loss", "train_cross_dim_loss"]}
    best_val = float("inf")
    best_epoch = -1
    patience = 0

    epoch_iter = tqdm(range(cfg.num_epochs), desc="[Train] epochs", unit="epoch", dynamic_ncols=True)
    for epoch in epoch_iter:
        train_m = run_epoch(model, train_dl, a, m, cfg, device, optimizer, desc=f"[Train] epoch {epoch+1}")
        with torch.no_grad():
            val_m = run_epoch(model, val_dl, a, m, cfg, device, None, desc=f"[Val] epoch {epoch+1}")

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(train_m["loss"]))
        history["val_loss"].append(float(val_m["loss"]))
        history["train_value_loss"].append(float(train_m["value_loss"]))
        history["train_delta_loss"].append(float(train_m["delta_loss"]))
        history["val_value_loss"].append(float(val_m["value_loss"]))
        history["val_delta_loss"].append(float(val_m["delta_loss"]))
        history["train_cross_dim_loss"].append(float(train_m.get("cross_dim_loss", 0.0)))

        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.to_dict(),
            "history": history,
            "graph_num_nodes": num_nodes,
            "sensor_names": list(node_names),
        }
        torch.save(ckpt, out_dir / "last.pt")

        improved = float(val_m["loss"]) < (best_val - cfg.early_stop_min_delta)
        if improved:
            best_val = float(val_m["loss"])
            best_epoch = epoch + 1
            patience = 0
            torch.save(ckpt, out_dir / "best.pt")
        else:
            patience += 1

        print(
            f"[INFO] epoch={epoch+1} "
            f"train_loss={train_m['loss']:.6f} val_loss={val_m['loss']:.6f} "
            f"cd_loss={train_m.get('cross_dim_loss', 0.0):.6f} "
            f"best_val={best_val:.6f}"
        )
        epoch_iter.set_postfix(train=f"{train_m['loss']:.4f}", val=f"{val_m['loss']:.4f}", best=f"{best_val:.4f}")
        if patience >= cfg.early_stop_patience:
            print(f"[EARLY STOP] epoch={epoch+1}, best_epoch={best_epoch}")
            break

    compute_val_normal_scores(
        model=model, cfg=cfg, a=a, m=m, device=device,
        node_names=node_names, normalization_stats=norm_stats,
        out_path=out_dir / "val_normal_scores.csv",
    )

    payload = {
        "mode": "trained",
        "config": cfg.to_dict(),
        "graph": {"num_nodes": num_nodes, "node_names_head": node_names[:10]},
        "dataset": {
            "train_windows": len(train_ds), "val_windows": len(val_ds),
            "lookback": cfg.lookback, "horizon": cfg.horizon_out, "num_nodes": num_nodes,
        },
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "split": {
            "val_ratio": float(cfg.val_ratio), "split_seed": int(cfg.split_seed),
            "split_source": str(split_source),
            "train_flights": sorted(train_ds.flights),
            "val_flights": sorted(val_ds.flights),
        },
    }
    save_json(payload, out_dir / "setup_summary.json")
    save_json(payload["split"], out_dir / "split_flights.json")
    save_json(cfg.to_dict(), out_dir / "config.json")
    save_json(history, out_dir / "history.json")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, encoding="utf-8")
    save_loss_curve(history, out_dir / "loss_curve.png")
    print(f"[OK] training complete: {out_dir}")


if __name__ == "__main__":
    main()
