from __future__ import annotations

import numpy as np
import torch


def corrupt_array(values: np.ndarray, kind: str, level: float, seed: int) -> tuple[np.ndarray, dict]:
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"values must be [batch,time,channel], got {x.shape}")
    if kind == "none" or float(level) <= 0:
        return x.copy(), {"kind": "none", "level": 0.0, "seed": int(seed)}
    rng = np.random.default_rng(int(seed))
    out = x.copy()
    metadata: dict = {"kind": kind, "level": float(level), "seed": int(seed)}

    if kind == "gaussian":
        out += rng.normal(0.0, float(level), size=out.shape).astype(np.float32)
    elif kind == "missing":
        mask = rng.random(out.shape) < float(level)
        out[mask] = np.nan
        for b in range(out.shape[0]):
            for c in range(out.shape[2]):
                previous = 0.5
                for t in range(out.shape[1]):
                    if np.isfinite(out[b, t, c]):
                        previous = float(out[b, t, c])
                    else:
                        out[b, t, c] = previous
        metadata["masked_fraction"] = float(mask.mean())
    elif kind == "channel_dropout":
        count = min(max(int(round(level)), 1), out.shape[2])
        channels = np.sort(rng.choice(out.shape[2], size=count, replace=False))
        out[:, :, channels] = 0.5
        metadata["channels"] = channels.tolist()
    elif kind == "downsample":
        factor = min(max(int(round(level)), 1), out.shape[1])
        for t in range(out.shape[1]):
            source = (t // factor) * factor
            out[:, t, :] = x[:, source, :]
        metadata["factor"] = factor
    else:
        raise ValueError(f"Unsupported corruption kind: {kind}")
    return out, metadata


def corrupt_tensor(x: torch.Tensor, kind: str, level: float, seed: int) -> tuple[torch.Tensor, dict]:
    if x.ndim != 4 or x.shape[-1] != 1:
        raise ValueError("x must be [batch,time,channel,1]")
    values, metadata = corrupt_array(x[..., 0].detach().cpu().numpy(), kind, level, seed)
    return torch.as_tensor(values, dtype=x.dtype, device=x.device).unsqueeze(-1), metadata
