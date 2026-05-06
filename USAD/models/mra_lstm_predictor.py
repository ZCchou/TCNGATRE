from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_future_forecast import build_control_sensor_mask


def _prob_to_logit(prob: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = prob.clamp(min=eps, max=1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


class StraightThroughBoundaryGate(nn.Module):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = float(threshold)

    def forward(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.sigmoid(logits)
        hard = (probs >= self.threshold).to(dtype=probs.dtype)
        ste = hard.detach() - probs.detach() + probs
        return ste, probs


class GroupInputProjection(nn.Module):
    def __init__(
        self,
        num_groups: int,
        group_dim: int,
        d_model: int,
        dropout: float,
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        self.group_dim = int(group_dim)
        self.d_model = int(d_model)
        in_dim = self.group_dim * 2 + 1
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_model),
                )
                for _ in range(self.num_groups)
            ]
        )

    def forward(
        self,
        x_t_k: torch.Tensor,
        m_t_k: torch.Tensor,
        v_t_k: torch.Tensor,
        group_idx: int,
    ) -> torch.Tensor:
        v = v_t_k.view(-1, 1).to(dtype=x_t_k.dtype)
        masked = x_t_k * m_t_k
        proj_in = torch.cat([masked, m_t_k, v], dim=-1)
        return self.projectors[int(group_idx)](proj_in)


class MRAUpdateBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        gate_hidden: int,
        dropout: float,
        use_s2: bool,
        boundary_threshold: float,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.use_s2 = bool(use_s2)
        self.boundary_gate = StraightThroughBoundaryGate(threshold=boundary_threshold)
        gate_in_dim = self.hidden_dim * 3 + self.d_model + 1
        self.boundary_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1),
        )
        self.s1_cell = nn.LSTMCell(self.d_model + self.hidden_dim * 2 + 1, self.hidden_dim)
        self.full_cell = nn.LSTMCell(self.d_model + self.hidden_dim * 2 + 1, self.hidden_dim)
        self.local_cell = nn.LSTMCell(self.d_model + 1, self.hidden_dim)
        self.cross_cell = nn.LSTMCell(self.hidden_dim * 2 + 1, self.hidden_dim)

    def full_update(
        self,
        x_t: torch.Tensor,
        z_feat: torch.Tensor,
        prev_h: torch.Tensor,
        prev_c: torch.Tensor,
        high_freq_cur: torch.Tensor,
        low_freq_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([x_t, high_freq_cur, low_freq_prev, z_feat], dim=-1)
        return self.full_cell(inp, (prev_h, prev_c))

    def local_update(
        self,
        x_t: torch.Tensor,
        z_feat: torch.Tensor,
        prev_h: torch.Tensor,
        prev_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([x_t, z_feat], dim=-1)
        return self.local_cell(inp, (prev_h, prev_c))

    def cross_update(
        self,
        z_feat: torch.Tensor,
        prev_h: torch.Tensor,
        prev_c: torch.Tensor,
        high_freq_cur: torch.Tensor,
        low_freq_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([high_freq_cur, low_freq_prev, z_feat], dim=-1)
        return self.cross_cell(inp, (prev_h, prev_c))

    def carry(
        self,
        prev_h: torch.Tensor,
        prev_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return prev_h, prev_c

    def forward(
        self,
        x_t: torch.Tensor,
        z_t: torch.Tensor,
        prev_h: torch.Tensor,
        prev_c: torch.Tensor,
        high_freq_cur: torch.Tensor,
        low_freq_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_feat = z_t.view(-1, 1).to(dtype=x_t.dtype)
        if not self.use_s2:
            s1_in = torch.cat([x_t, high_freq_cur, low_freq_prev, z_feat], dim=-1)
            h_upd, c_upd = self.s1_cell(s1_in, (prev_h, prev_c))
            has_context = (
                (z_t > 0.5)
                | (high_freq_cur.abs().sum(dim=-1) > 1e-8)
                | (low_freq_prev.abs().sum(dim=-1) > 1e-8)
            ).view(-1, 1).to(dtype=x_t.dtype)
            h_new = has_context * h_upd + (1.0 - has_context) * prev_h
            c_new = has_context * c_upd + (1.0 - has_context) * prev_c
            return h_new, c_new, has_context, has_context

        gate_in = torch.cat([prev_h, high_freq_cur, low_freq_prev, x_t, z_feat], dim=-1)
        boundary_logit = self.boundary_mlp(gate_in)
        boundary_hard, boundary_prob = self.boundary_gate(boundary_logit)
        has_input = (z_t > 0.5).view(-1, 1).to(dtype=x_t.dtype)

        h_full, c_full = self.full_update(
            x_t=x_t,
            z_feat=z_feat,
            prev_h=prev_h,
            prev_c=prev_c,
            high_freq_cur=high_freq_cur,
            low_freq_prev=low_freq_prev,
        )
        h_local, c_local = self.local_update(
            x_t=x_t,
            z_feat=z_feat,
            prev_h=prev_h,
            prev_c=prev_c,
        )
        h_cross, c_cross = self.cross_update(
            z_feat=z_feat,
            prev_h=prev_h,
            prev_c=prev_c,
            high_freq_cur=high_freq_cur,
            low_freq_prev=low_freq_prev,
        )
        h_keep, c_keep = self.carry(prev_h=prev_h, prev_c=prev_c)

        case_full = has_input * boundary_hard
        case_local = has_input * (1.0 - boundary_hard)
        case_cross = (1.0 - has_input) * boundary_hard
        case_carry = (1.0 - has_input) * (1.0 - boundary_hard)

        h_new = case_full * h_full + case_local * h_local + case_cross * h_cross + case_carry * h_keep
        c_new = case_full * c_full + case_local * c_local + case_cross * c_cross + case_carry * c_keep
        return h_new, c_new, boundary_hard, boundary_prob


class HSEAggregation(nn.Module):
    def __init__(self, num_groups: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.num_groups = int(num_groups)
        self.hidden_dim = int(hidden_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.num_groups * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_groups * self.hidden_dim),
        )
        self.out_ln = nn.LayerNorm(self.hidden_dim)

    def forward(self, h_stack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(h_stack.shape[0])
        summary = h_stack.reshape(batch_size, self.num_groups * self.hidden_dim)
        gate = torch.sigmoid(self.gate_mlp(summary)).view(batch_size, self.num_groups, self.hidden_dim)
        weighted = gate * h_stack
        denom = gate.sum(dim=1).clamp_min(1e-6)
        fused = weighted.sum(dim=1) / denom
        return self.out_ln(fused), gate


class MRAMultirateEncoder(nn.Module):
    def __init__(
        self,
        num_groups: int,
        group_dim: int,
        d_model: int,
        hidden_dim: int,
        gate_hidden: int,
        dropout: float,
        use_s2: bool,
        use_hse: bool,
        boundary_threshold: float,
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        self.hidden_dim = int(hidden_dim)
        self.use_hse = bool(use_hse)
        self.group_projection = GroupInputProjection(
            num_groups=num_groups,
            group_dim=group_dim,
            d_model=d_model,
            dropout=dropout,
        )
        self.update_blocks = nn.ModuleList(
            [
                MRAUpdateBlock(
                    d_model=d_model,
                    hidden_dim=hidden_dim,
                    gate_hidden=gate_hidden,
                    dropout=dropout,
                    use_s2=use_s2,
                    boundary_threshold=boundary_threshold,
                )
                for _ in range(self.num_groups)
            ]
        )
        self.hse = HSEAggregation(num_groups=num_groups, hidden_dim=hidden_dim, dropout=dropout)
        self.mean_ln = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        group_value_seq: torch.Tensor,
        group_mask_seq: torch.Tensor,
        group_valid_seq: torch.Tensor,
        win_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_groups, _ = group_value_seq.shape
        if int(num_groups) != self.num_groups:
            raise ValueError(f"MRA encoder group mismatch: expected {self.num_groups}, got {num_groups}")

        prev_h = [
            torch.zeros((batch_size, self.hidden_dim), device=group_value_seq.device, dtype=group_value_seq.dtype)
            for _ in range(self.num_groups)
        ]
        prev_c = [
            torch.zeros((batch_size, self.hidden_dim), device=group_value_seq.device, dtype=group_value_seq.dtype)
            for _ in range(self.num_groups)
        ]
        e_seq = []
        boundary_seq = []

        for t in range(seq_len):
            cur_h: list[torch.Tensor] = []
            cur_c: list[torch.Tensor] = []
            cur_boundary: list[torch.Tensor] = []
            for k in range(self.num_groups):
                x_t = group_value_seq[:, t, k]
                m_t = group_mask_seq[:, t, k]
                z_t = group_valid_seq[:, t, k]
                g_t = self.group_projection(
                    x_t_k=x_t,
                    m_t_k=m_t,
                    v_t_k=z_t,
                    group_idx=k,
                )
                high_freq_cur = cur_h[k - 1] if k > 0 else torch.zeros_like(prev_h[k])
                low_freq_prev = prev_h[k + 1] if (k + 1) < self.num_groups else torch.zeros_like(prev_h[k])
                h_new, c_new, boundary_hard, _boundary_prob = self.update_blocks[k](
                    x_t=g_t,
                    z_t=z_t,
                    prev_h=prev_h[k],
                    prev_c=prev_c[k],
                    high_freq_cur=high_freq_cur,
                    low_freq_prev=low_freq_prev,
                )
                active = win_mask[:, t].view(-1, 1).to(dtype=h_new.dtype)
                h_new = active * h_new + (1.0 - active) * prev_h[k]
                c_new = active * c_new + (1.0 - active) * prev_c[k]
                cur_h.append(h_new)
                cur_c.append(c_new)
                cur_boundary.append(boundary_hard)

            prev_h = cur_h
            prev_c = cur_c
            h_stack = torch.stack(cur_h, dim=1)
            if self.use_hse:
                e_t, _gate = self.hse(h_stack)
            else:
                e_t = self.mean_ln(h_stack.mean(dim=1))
            e_seq.append(e_t)
            boundary_seq.append(torch.cat(cur_boundary, dim=1))

        e_seq_t = torch.stack(e_seq, dim=1)
        boundary_t = torch.stack(boundary_seq, dim=1)
        lengths = win_mask.long().sum(dim=1).clamp(min=1) - 1
        gather_index = lengths.view(-1, 1, 1).expand(-1, 1, self.hidden_dim)
        e_last = e_seq_t.gather(dim=1, index=gather_index).squeeze(1)
        return e_last, e_seq_t, boundary_t


class MRALSTMPredictor(nn.Module):
    def __init__(
        self,
        num_sensors: int,
        rate_group_names: Sequence[str],
        max_group_dim: int,
        hidden_dim: int,
        group_proj_dim: int,
        gate_hidden: int,
        decoder_hidden: int,
        dropout: float,
        model_type: str,
        use_onset_head: bool = True,
        sensor_names: Sequence[str] | None = None,
        control_sensor_patterns: Sequence[str] = (),
        future_len: int = 1,
        use_s2: bool | None = None,
        use_hse: bool | None = None,
        boundary_threshold: float = 0.5,
        use_huber_loss: bool = False,
        huber_beta: float = 1.0,
        loss_weight_mask: float = 0.0,
        loss_weight_value: float = 1.0,
        loss_weight_delta: float = 0.0,
        loss_weight_mean: float = 0.0,
        loss_weight_range: float = 0.0,
        loss_weight_var: float = 0.0,
        loss_weight_count: float = 0.0,
        loss_weight_constant: float = 0.0,
        loss_weight_abs_diff: float = 0.0,
        loss_weight_zero_diff: float = 0.0,
        loss_weight_slope: float = 0.0,
        loss_weight_control_change: float = 0.0,
        loss_weight_onset_mask: float = 0.0,
        loss_weight_onset_mean: float = 0.0,
        loss_weight_onset_delta: float = 0.0,
        score_weight_mask: float = 0.0,
        score_weight_value: float = 1.0,
        score_weight_delta: float = 0.0,
        score_weight_mean: float = 0.0,
        score_weight_var: float = 0.0,
        score_weight_range: float = 0.0,
        score_weight_count: float = 0.0,
        score_weight_constant: float = 0.0,
        score_weight_abs_diff: float = 0.0,
        score_weight_zero_diff: float = 0.0,
        score_weight_slope: float = 0.0,
        score_weight_control_change: float = 0.0,
        score_weight_onset_mean: float = 0.0,
        score_weight_onset_delta: float = 0.0,
    ):
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.rate_group_names = [str(x) for x in rate_group_names]
        self.max_group_dim = int(max_group_dim)
        self.hidden_dim = int(hidden_dim)
        self.group_proj_dim = int(group_proj_dim)
        self.future_len = int(max(future_len, 1))
        self.use_onset_head = bool(use_onset_head)
        self.model_type = str(model_type).strip().lower()
        default_use_s2 = self.model_type in {"mra_s1_s2", "mra_full"}
        default_use_hse = self.model_type in {"mra_s1_s3", "mra_full"}
        self.use_s2 = default_use_s2 if use_s2 is None else bool(use_s2)
        self.use_hse = default_use_hse if use_hse is None else bool(use_hse)
        self.use_huber_loss = bool(use_huber_loss)
        self.huber_beta = float(huber_beta)
        self.score_weight_mask = float(score_weight_mask)
        self.score_weight_value = float(score_weight_value)
        self.score_weight_delta = float(score_weight_delta)
        self.score_weight_mean = float(score_weight_mean)
        self.score_weight_var = float(score_weight_var)
        self.score_weight_range = float(score_weight_range)
        self.score_weight_count = float(score_weight_count)
        self.score_weight_constant = float(score_weight_constant)
        self.score_weight_abs_diff = float(score_weight_abs_diff)
        self.score_weight_zero_diff = float(score_weight_zero_diff)
        self.score_weight_slope = float(score_weight_slope)
        self.score_weight_control_change = float(score_weight_control_change)
        self.score_weight_onset_mean = float(score_weight_onset_mean)
        self.score_weight_onset_delta = float(score_weight_onset_delta)

        self.encoder = MRAMultirateEncoder(
            num_groups=len(self.rate_group_names),
            group_dim=self.max_group_dim,
            d_model=self.group_proj_dim,
            hidden_dim=self.hidden_dim,
            gate_hidden=gate_hidden,
            dropout=dropout,
            use_s2=self.use_s2,
            use_hse=self.use_hse,
            boundary_threshold=boundary_threshold,
        )
        self.future_head = nn.Sequential(
            nn.Linear(self.hidden_dim, decoder_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, self.future_len * self.num_sensors),
        )
        self.register_buffer(
            "control_sensor_mask",
            build_control_sensor_mask(
                sensor_names=sensor_names,
                sensor_patterns=control_sensor_patterns,
                num_sensors=num_sensors,
            ),
        )
        self.register_buffer("value_mean", torch.zeros((num_sensors,), dtype=torch.float32))
        self.register_buffer("value_std", torch.ones((num_sensors,), dtype=torch.float32))

    def set_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor):
        mean = mean.detach().float().view(-1)
        std = std.detach().float().view(-1)
        if mean.numel() == self.num_sensors and std.numel() == self.num_sensors:
            self.value_mean.copy_(mean)
            self.value_std.copy_(std.clamp_min(1e-6))

    def _pointwise_regression_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_huber_loss:
            return F.smooth_l1_loss(pred, target, beta=self.huber_beta, reduction="none")
        return (pred - target).pow(2)

    def _safe_masked_mean(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        weight = valid_mask.to(dtype=x.dtype)
        denom = weight.sum()
        if float(denom.item()) <= 0.0:
            return x.new_zeros(())
        return (x * weight).sum() / denom

    def _sequence_last(self, seq: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        last = seq.new_zeros((seq.shape[0], seq.shape[2]))
        seen = torch.zeros_like(last, dtype=torch.bool)
        for t in range(seq.shape[1]):
            cur_valid = valid[:, t]
            last = torch.where(cur_valid, seq[:, t], last)
            seen = seen | cur_valid
        return last, seen.to(dtype=seq.dtype)

    def _sequence_first(self, seq: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first = seq.new_zeros((seq.shape[0], seq.shape[2]))
        seen = torch.zeros_like(first, dtype=torch.bool)
        for t in range(seq.shape[1]):
            cur_valid = valid[:, t] & (~seen)
            first = torch.where(cur_valid, seq[:, t], first)
            seen = seen | valid[:, t]
        return first, seen.to(dtype=seq.dtype)

    def _derive_sequence_statistics(
        self,
        pred_seq: torch.Tensor,
        seq_mask: torch.Tensor,
        hist_last_value: torch.Tensor,
        hist_has_value: torch.Tensor,
        future_mask_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        valid = seq_mask > 0.5
        weight = valid.to(dtype=pred_seq.dtype)
        count = weight.sum(dim=1)
        future_mask = (count > 0.0).to(dtype=pred_seq.dtype)
        seq_sum = (pred_seq * weight).sum(dim=1)
        future_mean_hat = torch.where(
            count > 0.0,
            seq_sum / count.clamp_min(1.0),
            torch.zeros_like(seq_sum),
        )
        future_last_hat, future_last_seen = self._sequence_last(pred_seq, valid)
        future_last_hat = future_last_hat * future_last_seen
        future_delta_mask = future_mask * (hist_has_value > 0.5).to(dtype=pred_seq.dtype)
        future_delta_hat = torch.where(
            future_delta_mask > 0.5,
            future_last_hat - hist_last_value,
            torch.zeros_like(future_last_hat),
        )

        centered = pred_seq - future_mean_hat.unsqueeze(1)
        future_var_hat = torch.where(
            count > 0.0,
            (centered.pow(2) * weight).sum(dim=1) / count.clamp_min(1.0),
            torch.zeros_like(future_mean_hat),
        )

        masked_max = pred_seq.masked_fill(~valid, float("-inf")).amax(dim=1)
        masked_min = pred_seq.masked_fill(~valid, float("inf")).amin(dim=1)
        future_range_hat = torch.where(
            future_mask > 0.5,
            masked_max - masked_min,
            torch.zeros_like(future_mean_hat),
        )
        future_log_count_hat = torch.log1p(count.clamp_min(0.0))
        future_constant_prob = (future_range_hat <= 1e-3).to(dtype=pred_seq.dtype)
        future_constant_logit = _prob_to_logit(future_constant_prob)

        if pred_seq.shape[1] >= 2:
            diffs = pred_seq[:, 1:] - pred_seq[:, :-1]
            pair_valid = valid[:, 1:] & valid[:, :-1]
            pair_weight = pair_valid.to(dtype=pred_seq.dtype)
            pair_count = pair_weight.sum(dim=1)
            future_abs_diff_hat = (diffs.abs() * pair_weight).sum(dim=1)
            zero_ratio = ((diffs.abs() <= 1e-3).to(dtype=pred_seq.dtype) * pair_weight).sum(dim=1)
            future_zero_diff_ratio = torch.where(
                pair_count > 0.0,
                zero_ratio / pair_count.clamp_min(1.0),
                torch.zeros_like(future_mean_hat),
            )

            time_axis = torch.linspace(0.0, 1.0, steps=pred_seq.shape[1], device=pred_seq.device, dtype=pred_seq.dtype)
            time_axis = time_axis.view(1, pred_seq.shape[1], 1)
            valid_sum = weight.sum(dim=1).clamp_min(1.0)
            time_mean = (time_axis * weight).sum(dim=1) / valid_sum
            xc = time_axis - time_mean.unsqueeze(1)
            yc = pred_seq - future_mean_hat.unsqueeze(1)
            denom = (xc.pow(2) * weight).sum(dim=1)
            numer = (xc * yc * weight).sum(dim=1)
            future_slope_hat = torch.where(
                count > 1.0,
                numer / denom.clamp_min(1e-8),
                torch.zeros_like(future_mean_hat),
            )
        else:
            pair_count = torch.zeros_like(count)
            future_abs_diff_hat = torch.zeros_like(future_mean_hat)
            future_zero_diff_ratio = torch.zeros_like(future_mean_hat)
            future_slope_hat = torch.zeros_like(future_mean_hat)

        future_diff_mask = (pair_count > 0.0).to(dtype=pred_seq.dtype)
        future_zero_diff_logit = _prob_to_logit(future_zero_diff_ratio.clamp(min=1e-4, max=1.0 - 1e-4))

        onset_value_hat, onset_seen = self._sequence_first(pred_seq, valid)
        future_onset_mask = onset_seen
        future_onset_mean_hat = onset_value_hat * onset_seen
        future_onset_delta_hat = torch.where(
            (future_onset_mask > 0.5) & (hist_has_value > 0.5),
            future_onset_mean_hat - hist_last_value,
            torch.zeros_like(future_onset_mean_hat),
        )
        future_onset_mask_logit = _prob_to_logit(future_onset_mask.clamp(min=1e-4, max=1.0 - 1e-4))

        future_control_change_hat = (
            torch.abs(future_delta_hat) * future_delta_mask
            + 0.50 * future_abs_diff_hat * future_diff_mask
            + 0.35 * torch.abs(future_slope_hat) * future_diff_mask
            + 0.25 * future_range_hat * future_diff_mask
        ) * self.control_sensor_mask.view(1, -1).to(device=pred_seq.device, dtype=pred_seq.dtype)

        future_mask = future_mask_target.to(dtype=pred_seq.dtype)
        future_mask_logit = _prob_to_logit(future_mask.clamp(min=1e-4, max=1.0 - 1e-4))
        return {
            "future_mask_logit": future_mask_logit,
            "future_value_hat": future_last_hat,
            "future_delta_hat": future_delta_hat,
            "future_mean_hat": future_mean_hat,
            "future_var_hat": future_var_hat,
            "future_range_hat": future_range_hat,
            "future_log_count_hat": future_log_count_hat,
            "future_constant_logit": future_constant_logit,
            "future_abs_diff_hat": future_abs_diff_hat,
            "future_zero_diff_logit": future_zero_diff_logit,
            "future_slope_hat": future_slope_hat,
            "future_control_change_hat": future_control_change_hat,
            "future_onset_mask_logit": future_onset_mask_logit,
            "future_onset_mean_hat": future_onset_mean_hat,
            "future_onset_delta_hat": future_onset_delta_hat,
        }

    def compute_losses_and_scores(
        self,
        pred_seq: torch.Tensor,
        pred_stats: Dict[str, torch.Tensor],
        future_value_seq: torch.Tensor,
        future_value_mask_seq: torch.Tensor,
        future_mask: torch.Tensor,
        future_last_value: torch.Tensor,
        future_delta_value: torch.Tensor,
        future_delta_mask: torch.Tensor,
        future_diff_mask: torch.Tensor,
        future_mean_value: torch.Tensor,
        future_var_value: torch.Tensor,
        future_range_value: torch.Tensor,
        future_log_count: torch.Tensor,
        future_is_constant: torch.Tensor,
        future_abs_diff_sum: torch.Tensor,
        future_zero_diff_ratio: torch.Tensor,
        future_slope: torch.Tensor,
        future_control_change: torch.Tensor,
        future_onset_mask: torch.Tensor,
        future_onset_mean_value: torch.Tensor,
        future_onset_delta_value: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        seq_valid = future_value_mask_seq > 0.5
        seq_loss_map = self._pointwise_regression_loss(pred_seq, future_value_seq)
        loss_value = self._safe_masked_mean(seq_loss_map, seq_valid)
        zero = loss_value.new_zeros(())
        loss = loss_value

        pred_mask_prob = torch.sigmoid(pred_stats["future_mask_logit"])
        pred_constant_prob = torch.sigmoid(pred_stats["future_constant_logit"])
        pred_zero_diff_ratio = torch.sigmoid(pred_stats["future_zero_diff_logit"])
        pred_count = torch.expm1(pred_stats["future_log_count_hat"]).clamp_min(0.0)
        pred_onset_mask_prob = torch.sigmoid(pred_stats["future_onset_mask_logit"])

        valid = future_mask > 0.5
        delta_valid = future_delta_mask > 0.5
        diff_valid = future_diff_mask > 0.5
        control_valid = valid & (self.control_sensor_mask.view(1, -1).to(device=future_mask.device) > 0.5)
        onset_valid = future_onset_mask > 0.5
        onset_delta_valid = onset_valid & valid

        resid_mask = torch.abs(pred_mask_prob - future_mask)
        resid_value = torch.abs(pred_stats["future_value_hat"] - future_last_value) * valid.to(dtype=pred_seq.dtype)
        resid_delta = torch.abs(pred_stats["future_delta_hat"] - future_delta_value) * delta_valid.to(dtype=pred_seq.dtype)
        resid_mean = torch.abs(pred_stats["future_mean_hat"] - future_mean_value) * valid.to(dtype=pred_seq.dtype)
        resid_var = torch.abs(pred_stats["future_var_hat"] - future_var_value) * diff_valid.to(dtype=pred_seq.dtype)
        resid_range = torch.abs(pred_stats["future_range_hat"] - future_range_value) * diff_valid.to(dtype=pred_seq.dtype)
        resid_count = torch.abs(pred_stats["future_log_count_hat"] - future_log_count) * valid.to(dtype=pred_seq.dtype)
        resid_constant = torch.abs(pred_constant_prob - future_is_constant) * diff_valid.to(dtype=pred_seq.dtype)
        resid_abs_diff = torch.abs(pred_stats["future_abs_diff_hat"] - future_abs_diff_sum) * diff_valid.to(dtype=pred_seq.dtype)
        resid_zero_diff = torch.abs(pred_zero_diff_ratio - future_zero_diff_ratio) * diff_valid.to(dtype=pred_seq.dtype)
        resid_slope = torch.abs(pred_stats["future_slope_hat"] - future_slope) * diff_valid.to(dtype=pred_seq.dtype)
        resid_control_change = torch.abs(pred_stats["future_control_change_hat"] - future_control_change) * control_valid.to(dtype=pred_seq.dtype)
        resid_onset_mask = torch.abs(pred_onset_mask_prob - future_onset_mask)
        resid_onset_mean = torch.abs(pred_stats["future_onset_mean_hat"] - future_onset_mean_value) * onset_valid.to(dtype=pred_seq.dtype)
        resid_onset_delta = torch.abs(pred_stats["future_onset_delta_hat"] - future_onset_delta_value) * onset_delta_valid.to(dtype=pred_seq.dtype)

        value_sensor_score = resid_value
        control_change_sensor_score = resid_control_change
        sensor_score = (
            self.score_weight_mask * resid_mask
            + self.score_weight_value * resid_value
            + self.score_weight_delta * resid_delta
            + self.score_weight_mean * resid_mean
            + self.score_weight_var * resid_var
            + self.score_weight_range * resid_range
            + self.score_weight_count * resid_count
            + self.score_weight_constant * resid_constant
            + self.score_weight_abs_diff * resid_abs_diff
            + self.score_weight_zero_diff * resid_zero_diff
            + self.score_weight_slope * resid_slope
            + self.score_weight_control_change * resid_control_change
            + self.score_weight_onset_mean * resid_onset_mean
            + self.score_weight_onset_delta * resid_onset_delta
        )
        denom = valid.to(dtype=sensor_score.dtype).sum(dim=1)
        total_score = torch.where(
            denom > 0,
            (sensor_score * valid.to(dtype=sensor_score.dtype)).sum(dim=1) / denom.clamp_min(1.0),
            torch.zeros_like(denom),
        )
        value_total_score = torch.where(
            denom > 0,
            (value_sensor_score * valid.to(dtype=value_sensor_score.dtype)).sum(dim=1) / denom.clamp_min(1.0),
            torch.zeros_like(denom),
        )
        control_denom = control_valid.to(dtype=control_change_sensor_score.dtype).sum(dim=1)
        control_change_total_score = torch.where(
            control_denom > 0,
            (control_change_sensor_score * control_valid.to(dtype=control_change_sensor_score.dtype)).sum(dim=1)
            / control_denom.clamp_min(1.0),
            torch.zeros_like(control_denom),
        )
        zeros_bin = future_mask.new_zeros((future_mask.shape[0], future_mask.shape[1], 0))
        return {
            "loss": loss,
            "loss_mask": zero,
            "loss_value": loss_value,
            "loss_delta": zero,
            "loss_mean": zero,
            "loss_var": zero,
            "loss_range": zero,
            "loss_count": zero,
            "loss_constant": zero,
            "loss_abs_diff": zero,
            "loss_zero_diff": zero,
            "loss_slope": zero,
            "loss_control_change": zero,
            "loss_bin_mask": zero,
            "loss_bin_last": zero,
            "loss_bin_mean": zero,
            "loss_bin_count": zero,
            "loss_onset_mask": zero,
            "loss_onset_mean": zero,
            "loss_onset_delta": zero,
            "pred_mask_prob": pred_mask_prob,
            "pred_constant_prob": pred_constant_prob,
            "pred_zero_diff_ratio": pred_zero_diff_ratio,
            "pred_count": pred_count,
            "pred_bin_mask_prob": zeros_bin,
            "pred_bin_count": zeros_bin,
            "pred_onset_mask_prob": pred_onset_mask_prob,
            "resid_mask": resid_mask,
            "resid_value": resid_value,
            "resid_delta": resid_delta,
            "resid_mean": resid_mean,
            "resid_var": resid_var,
            "resid_range": resid_range,
            "resid_count": resid_count,
            "resid_constant": resid_constant,
            "resid_abs_diff": resid_abs_diff,
            "resid_zero_diff": resid_zero_diff,
            "resid_slope": resid_slope,
            "resid_control_change": resid_control_change,
            "resid_bin_mask": zeros_bin,
            "resid_bin_last": zeros_bin,
            "resid_bin_mean": zeros_bin,
            "resid_bin_count": zeros_bin,
            "resid_onset_mask": resid_onset_mask,
            "resid_onset_mean": resid_onset_mean,
            "resid_onset_delta": resid_onset_delta,
            "value_sensor_score": value_sensor_score,
            "control_change_sensor_score": control_change_sensor_score,
            "sensor_score": sensor_score,
            "value_total_score": value_total_score,
            "control_change_total_score": control_change_total_score,
            "total_score": total_score,
        }

    def forward(self, batch: Dict[str, torch.Tensor], return_details: bool = False) -> Dict[str, torch.Tensor]:
        del return_details
        device = next(self.parameters()).device
        group_value_seq = batch["group_value_seq"].to(device)
        group_mask_seq = batch["group_mask_seq"].to(device)
        group_available_seq = batch["group_available_seq"].to(device)
        win_mask = batch["win_mask"].to(device)
        hist_last_value = batch["hist_last_value"].to(device)
        hist_has_value = batch["hist_has_value"].to(device)
        future_value_seq = batch["future_value_seq"].to(device)
        future_value_mask_seq = batch["future_value_mask_seq"].to(device)

        future_mask = batch["future_mask"].to(device)
        future_last_value = batch["future_last_value"].to(device)
        future_delta_value = batch["future_delta_value"].to(device)
        future_delta_mask = batch["future_delta_mask"].to(device)
        future_diff_mask = batch["future_diff_mask"].to(device)
        future_mean_value = batch["future_mean_value"].to(device)
        future_var_value = batch["future_var_value"].to(device)
        future_range_value = batch["future_range_value"].to(device)
        future_log_count = batch["future_log_count"].to(device)
        future_is_constant = batch["future_is_constant"].to(device)
        future_abs_diff_sum = batch["future_abs_diff_sum"].to(device)
        future_zero_diff_ratio = batch["future_zero_diff_ratio"].to(device)
        future_slope = batch["future_slope"].to(device)
        future_onset_mask = batch["future_onset_mask"].to(device)
        future_onset_mean_value = batch["future_onset_mean_value"].to(device)
        future_onset_delta_value = batch["future_onset_delta_value"].to(device)

        future_control_change = (
            torch.abs(future_delta_value) * future_delta_mask
            + 0.50 * future_abs_diff_sum * future_diff_mask
            + 0.35 * torch.abs(future_slope) * future_diff_mask
            + 0.25 * future_range_value * future_diff_mask
        ) * self.control_sensor_mask.view(1, -1).to(device=device, dtype=future_mask.dtype)

        e_last, e_seq, boundary_seq = self.encoder(
            group_value_seq=group_value_seq,
            group_mask_seq=group_mask_seq,
            group_valid_seq=group_available_seq,
            win_mask=win_mask,
        )
        pred_seq = self.future_head(e_last).view(-1, self.future_len, self.num_sensors)
        pred_stats = self._derive_sequence_statistics(
            pred_seq=pred_seq,
            seq_mask=future_value_mask_seq,
            hist_last_value=hist_last_value,
            hist_has_value=hist_has_value,
            future_mask_target=future_mask,
        )
        pred_stats["future_bin_mask_logit"] = future_mask.new_zeros((future_mask.shape[0], future_mask.shape[1], 0))
        pred_stats["future_bin_last_hat"] = future_mask.new_zeros((future_mask.shape[0], future_mask.shape[1], 0))
        pred_stats["future_bin_mean_hat"] = future_mask.new_zeros((future_mask.shape[0], future_mask.shape[1], 0))
        pred_stats["future_bin_log_count_hat"] = future_mask.new_zeros((future_mask.shape[0], future_mask.shape[1], 0))

        out = self.compute_losses_and_scores(
            pred_seq=pred_seq,
            pred_stats=pred_stats,
            future_value_seq=future_value_seq,
            future_value_mask_seq=future_value_mask_seq,
            future_mask=future_mask,
            future_last_value=future_last_value,
            future_delta_value=future_delta_value,
            future_delta_mask=future_delta_mask,
            future_diff_mask=future_diff_mask,
            future_mean_value=future_mean_value,
            future_var_value=future_var_value,
            future_range_value=future_range_value,
            future_log_count=future_log_count,
            future_is_constant=future_is_constant,
            future_abs_diff_sum=future_abs_diff_sum,
            future_zero_diff_ratio=future_zero_diff_ratio,
            future_slope=future_slope,
            future_control_change=future_control_change,
            future_onset_mask=future_onset_mask,
            future_onset_mean_value=future_onset_mean_value,
            future_onset_delta_value=future_onset_delta_value,
        )
        out.update(pred_stats)
        out["future_value_seq_hat"] = pred_seq
        out["future_value_seq_target"] = future_value_seq
        out["future_value_seq_mask"] = future_value_mask_seq
        out["future_control_change_target"] = future_control_change
        out["mra_e_last"] = e_last
        out["mra_e_seq"] = e_seq
        out["mra_boundary_seq"] = boundary_seq
        return out
