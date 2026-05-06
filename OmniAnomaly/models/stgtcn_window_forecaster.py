from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.st_graph_tcn import build_stgtcn


class STGTCNWindowForecaster(nn.Module):
    """Dense sliding-window STGTCN forecaster used by the fast pipeline."""

    def __init__(
        self,
        *,
        a_stat: torch.Tensor,
        m_mask: torch.Tensor,
        num_nodes: int,
        horizon: int,
        in_feat: int = 1,
        model_name: str = "parallel_cross_attn",
        d_model: int = 64,
        short_kernel: int = 9,
        nhead: int = 8,
        tcn_layers: int = 5,
        dropout: float = 0.20,
        eta: float = 2.0,
        beta: float = 0.5,
        temporal_encoder_type: str = "tcn_only",
        short_patch: int = 25,
        loss_type: str = "huber",
        huber_beta: float = 1.0,
        score_mode: str = "mae",
    ) -> None:
        super().__init__()
        self.horizon = max(int(horizon), 1)
        self.short_patch = max(int(short_patch), 1)
        self.loss_type = str(loss_type).strip().lower()
        self.huber_beta = float(huber_beta)
        self.score_mode = str(score_mode).strip().lower()

        self.core = build_stgtcn(
            model_name=model_name,
            num_nodes=int(num_nodes),
            in_feat=int(in_feat),
            d_model=int(d_model),
            short_kernel=int(short_kernel),
            nhead=int(nhead),
            tcn_layers=int(tcn_layers),
            dropout=float(dropout),
            eta=float(eta),
            beta=float(beta),
            out_feat=int(in_feat),
            horizon=self.horizon,
            temporal_encoder_type=temporal_encoder_type,
        )
        self.register_buffer("a_stat", a_stat.float(), persistent=False)
        self.register_buffer("m_mask", m_mask.float(), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.a_stat.device

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device, non_blocking=True)

    def _loss_map(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return (pred - target) ** 2
        if self.loss_type == "mae":
            return (pred - target).abs()
        return F.smooth_l1_loss(
            pred,
            target,
            reduction="none",
            beta=float(self.huber_beta),
        )

    def _score_error(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.score_mode == "mse":
            return (pred - target) ** 2
        if self.score_mode == "huber":
            return F.smooth_l1_loss(
                pred,
                target,
                reduction="none",
                beta=float(self.huber_beta),
            )
        return (pred - target).abs()

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = mask.float()
        numerator = (value * weight).sum()
        denominator = weight.sum().clamp_min(1.0)
        return numerator / denominator, numerator, denominator

    def predict(self, x: torch.Tensor, zmask: torch.Tensor | None = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        y_hat, aux = self.core(
            x,
            self.a_stat,
            self.m_mask,
            short_patch=self.short_patch,
            zmask=zmask,
        )
        return y_hat.squeeze(-1), aux

    def compute_losses(self, batch: Dict[str, torch.Tensor | object]) -> Dict[str, torch.Tensor]:
        x = self._to_device(batch["x"]).float()
        y = self._to_device(batch["y"]).float().squeeze(-1)
        zmask = self._to_device(batch["zmask"]).float()
        y_mask = self._to_device(batch["y_mask"]).float()

        y_hat, aux = self.predict(x=x, zmask=zmask)
        if y_hat.shape != y.shape:
            raise ValueError(
                "Prediction shape mismatch: "
                f"pred={tuple(y_hat.shape)} target={tuple(y.shape)}"
            )

        loss_map = self._loss_map(pred=y_hat, target=y)
        loss, loss_num, loss_den = self._masked_mean(loss_map, y_mask)

        error = self._score_error(pred=y_hat, target=y)
        step_den = y_mask.sum(dim=-1).clamp_min(1.0)
        dim_den = y_mask.sum(dim=1).clamp_min(1.0)
        final_den = y_mask.sum(dim=(1, 2)).clamp_min(1.0)

        step_score = (error * y_mask).sum(dim=-1) / step_den
        dim_score = (error * y_mask).sum(dim=1) / dim_den
        final_score = (error * y_mask).sum(dim=(1, 2)) / final_den
        valid_future_count = y_mask.sum(dim=(1, 2))

        return {
            "loss": loss,
            "loss_numerator": loss_num.detach(),
            "loss_denominator": loss_den.detach(),
            "prediction": y_hat,
            "target": y,
            "target_mask": y_mask,
            "error": error,
            "step_score": step_score,
            "dim_score": dim_score,
            "final_score": final_score,
            "valid_future_count": valid_future_count,
            "valid_future_any": valid_future_count > 0.0,
            "aux": aux,
        }

    def anomaly_components(self, batch: Dict[str, torch.Tensor | object]) -> Dict[str, torch.Tensor]:
        return self.compute_losses(batch)
