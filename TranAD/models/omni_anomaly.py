"""OmniAnomaly: Robust Anomaly Detection for Multivariate Time Series.

Faithful implementation of the OmniAnomaly architecture:
  - qnet (inference network): GRU encoder + autoregressive latent posterior + planar normalizing flow
  - pnet (generative network): latent transition prior p(z_t|z_{t-1}) + decoder GRU + Gaussian observation model
  - Planar normalizing flow with numerically-stable log|det J|
  - Negative ELBO loss with reconstruction NLL, KL divergence, and flow correction

Reference:
  Su et al., "Robust Anomaly Detection for Multivariate Time Series through
  Stochastic Recurrent Neural Network", KDD 2019.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Planar Flow
# ---------------------------------------------------------------------------

class PlanarFlowLayer(nn.Module):
    """Single planar flow layer: z' = z + u * tanh(w^T z + b).

    Implements the invertibility constraint on u (Rezende & Mohamed 2015, Appendix A)
    and numerically-stable log|det dz'/dz|.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim) * 0.01)
        self.u = nn.Parameter(torch.randn(dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def _constrained_u(self) -> torch.Tensor:
        """Ensure w^T u >= -1 for invertibility."""
        wtu = (self.w * self.u).sum()
        m_wtu = -1.0 + F.softplus(wtu)  # m(w^T u) = -1 + log(1+exp(w^T u))
        # u_hat = u + (m(w^T u) - w^T u) * w / ||w||^2
        w_norm_sq = (self.w * self.w).sum().clamp(min=1e-12)
        u_hat = self.u + (m_wtu - wtu) * self.w / w_norm_sq
        return u_hat

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            z: (batch, dim)

        Returns:
            z_prime: (batch, dim)
            log_det: (batch,)
        """
        u_hat = self._constrained_u()
        # linear part: w^T z + b   -> (batch,)
        linear = z @ self.w + self.b
        tanh_val = torch.tanh(linear)  # (batch,)
        z_prime = z + u_hat.unsqueeze(0) * tanh_val.unsqueeze(1)  # (batch, dim)

        # log|det J| = log|1 + u_hat^T * dtanh/dlinear * w|
        dtanh = 1.0 - tanh_val * tanh_val  # (batch,)
        psi = dtanh.unsqueeze(1) * self.w.unsqueeze(0)  # (batch, dim)
        det = 1.0 + (psi * u_hat.unsqueeze(0)).sum(dim=1)  # (batch,)
        log_det = torch.log(det.abs().clamp(min=1e-8))  # (batch,)
        return z_prime, log_det


class PlanarFlow(nn.Module):
    """Stack of K planar flow layers."""

    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([PlanarFlowLayer(dim) for _ in range(num_layers)])

    def forward(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply K planar flow transformations.

        Args:
            z0: (batch, dim) initial sample from q0

        Returns:
            zK: (batch, dim) transformed sample
            sum_log_det: (batch,) cumulative log|det J|
        """
        z = z0
        sum_log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in self.layers:
            z, ld = layer(z)
            sum_log_det = sum_log_det + ld
        return z, sum_log_det


# ---------------------------------------------------------------------------
#  qnet (Inference / Recognition Network)
# ---------------------------------------------------------------------------

class QNet(nn.Module):
    """Inference network q(z_t | x_{<=t}, z_{<t}).

    1. GRU encodes observations into deterministic states e_t.
    2. [z_{t-1}, e_t] -> MLP -> mu_z, logvar_z  (autoregressive posterior)
    3. Reparameterization trick -> z0_t
    4. Planar flow K layers -> z_t
    """

    def __init__(
        self,
        input_dim: int,
        rnn_hidden: int,
        latent_dim: int,
        rnn_layers: int = 1,
        dropout: float = 0.0,
        num_flows: int = 10,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.rnn_hidden = rnn_hidden

        # Encoder GRU
        self.encoder_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        # Posterior: [z_{t-1}, e_t] -> mu, logvar
        self.posterior_net = nn.Sequential(
            nn.Linear(latent_dim + rnn_hidden, rnn_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(rnn_hidden, 2 * latent_dim),  # mu + logvar
        )

        # Planar flow
        self.flow = PlanarFlow(dim=latent_dim, num_layers=num_flows) if num_flows > 0 else None

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass over a sequence.

        Args:
            x: (batch, T, input_dim)

        Returns dict with:
            z_seq:       (batch, T, latent_dim) — final z after flow
            z0_seq:      (batch, T, latent_dim) — pre-flow z
            mu_seq:      (batch, T, latent_dim)
            logvar_seq:  (batch, T, latent_dim)
            log_det_seq: (batch, T) — cumulative flow log|det J| per step
            e_seq:       (batch, T, rnn_hidden)
        """
        B, T, _ = x.shape

        # Encode full sequence
        e_seq, _ = self.encoder_gru(x)  # (B, T, rnn_hidden)

        z_seq = []
        z0_seq = []
        mu_seq = []
        logvar_seq = []
        log_det_seq = []

        z_prev = torch.zeros(B, self.latent_dim, device=x.device, dtype=x.dtype)

        for t in range(T):
            e_t = e_seq[:, t, :]  # (B, rnn_hidden)
            # Posterior parameters
            cat_input = torch.cat([z_prev, e_t], dim=1)  # (B, latent_dim + rnn_hidden)
            params = self.posterior_net(cat_input)
            mu = params[:, :self.latent_dim]
            logvar = params[:, self.latent_dim:]

            # Reparameterization
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z0 = mu + std * eps

            # Planar flow
            if self.flow is not None:
                z, log_det = self.flow(z0)
            else:
                z = z0
                log_det = torch.zeros(B, device=x.device, dtype=x.dtype)

            z_seq.append(z)
            z0_seq.append(z0)
            mu_seq.append(mu)
            logvar_seq.append(logvar)
            log_det_seq.append(log_det)

            z_prev = z.detach()  # stop gradient to previous step (per OmniAnomaly)

        return {
            "z_seq": torch.stack(z_seq, dim=1),        # (B, T, D_z)
            "z0_seq": torch.stack(z0_seq, dim=1),
            "mu_seq": torch.stack(mu_seq, dim=1),
            "logvar_seq": torch.stack(logvar_seq, dim=1),
            "log_det_seq": torch.stack(log_det_seq, dim=1),  # (B, T)
            "e_seq": e_seq,
        }


# ---------------------------------------------------------------------------
#  pnet (Generative / Decoder Network)
# ---------------------------------------------------------------------------

class PNet(nn.Module):
    """Generative network.

    1. Prior transition: p(z_t | z_{t-1}) parameterized by MLP -> mu_prior, logvar_prior
    2. Decoder GRU combines z_t into hidden state d_t
    3. Observation model: d_t -> mu_x, logvar_x  (diagonal Gaussian)
    """

    def __init__(
        self,
        output_dim: int,
        rnn_hidden: int,
        latent_dim: int,
        rnn_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Prior transition: p(z_t | z_{t-1})
        self.prior_net = nn.Sequential(
            nn.Linear(latent_dim, rnn_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(rnn_hidden, 2 * latent_dim),  # mu_prior + logvar_prior
        )

        # Decoder GRU: input = z_t
        self.decoder_gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        # Observation model: d_t -> mu_x, logvar_x
        self.output_net = nn.Sequential(
            nn.Linear(rnn_hidden, rnn_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(rnn_hidden, 2 * output_dim),
        )

    def forward(
        self,
        z_seq: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            z_seq: (batch, T, latent_dim) — sampled latent variables

        Returns dict with:
            mu_x:        (batch, T, output_dim)
            logvar_x:    (batch, T, output_dim)
            mu_prior:    (batch, T, latent_dim)
            logvar_prior:(batch, T, latent_dim)
        """
        B, T, D_z = z_seq.shape

        # Prior: p(z_t | z_{t-1})
        z_prev = torch.cat([
            torch.zeros(B, 1, D_z, device=z_seq.device, dtype=z_seq.dtype),
            z_seq[:, :-1, :],
        ], dim=1)  # shift right, first step uses z=0

        prior_params = self.prior_net(z_prev)  # (B, T, 2*D_z)
        mu_prior = prior_params[:, :, :D_z]
        logvar_prior = prior_params[:, :, D_z:]

        # Decoder GRU
        d_seq, _ = self.decoder_gru(z_seq)  # (B, T, rnn_hidden)

        # Observation distribution
        out_params = self.output_net(d_seq)  # (B, T, 2*output_dim)
        mu_x = out_params[:, :, :self.output_dim]
        logvar_x = out_params[:, :, self.output_dim:]
        # Clamp logvar for numerical stability
        logvar_x = logvar_x.clamp(-6.0, 6.0)

        return {
            "mu_x": mu_x,
            "logvar_x": logvar_x,
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
        }


# ---------------------------------------------------------------------------
#  Full OmniAnomaly model
# ---------------------------------------------------------------------------

class OmniAnomalyOutput(NamedTuple):
    total_loss: torch.Tensor
    recon_loss: torch.Tensor
    kl_loss: torch.Tensor
    flow_loss: torch.Tensor
    mu_x: torch.Tensor
    logvar_x: torch.Tensor


class OmniAnomaly(nn.Module):
    """OmniAnomaly: stochastic RNN with planar normalizing flows.

    Combines qnet + pnet into a unified VAE-like model with temporal latent dynamics.
    """

    def __init__(
        self,
        input_dim: int,
        rnn_hidden: int = 128,
        latent_dim: int = 16,
        rnn_layers: int = 1,
        dropout: float = 0.0,
        num_flows: int = 10,
        decoder_rnn_hidden: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.qnet = QNet(
            input_dim=input_dim,
            rnn_hidden=rnn_hidden,
            latent_dim=latent_dim,
            rnn_layers=rnn_layers,
            dropout=dropout,
            num_flows=num_flows,
        )
        self.pnet = PNet(
            output_dim=input_dim,
            rnn_hidden=decoder_rnn_hidden,
            latent_dim=latent_dim,
            rnn_layers=rnn_layers,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> OmniAnomalyOutput:
        """Full forward pass and loss computation.

        Args:
            x:          (batch, T, input_dim)
            kl_weight:  weight on KL term (for KL annealing / warmup)

        Returns:
            OmniAnomalyOutput with total_loss, recon_loss, kl_loss, flow_loss, mu_x, logvar_x
        """
        B, T, D = x.shape

        # 1. Inference
        q_out = self.qnet(x)
        z_seq = q_out["z_seq"]           # (B, T, D_z)
        mu_q = q_out["mu_seq"]           # (B, T, D_z)
        logvar_q = q_out["logvar_seq"]   # (B, T, D_z)
        log_det = q_out["log_det_seq"]   # (B, T)

        # 2. Generation
        p_out = self.pnet(z_seq)
        mu_x = p_out["mu_x"]            # (B, T, D)
        logvar_x = p_out["logvar_x"]    # (B, T, D)
        mu_prior = p_out["mu_prior"]    # (B, T, D_z)
        logvar_prior = p_out["logvar_prior"]  # (B, T, D_z)

        # 3. Reconstruction loss: -log p(x_t | z_t)
        # Gaussian NLL per dimension per timestep
        var_x = torch.exp(logvar_x).clamp(min=1e-6)
        recon_nll = 0.5 * (
            logvar_x + (x - mu_x).pow(2) / var_x + math.log(2 * math.pi)
        )  # (B, T, D)
        recon_loss = recon_nll.sum(dim=2).sum(dim=1).mean()  # sum over D,T; mean over B

        # 4. KL divergence: KL(q(z_t|...) || p(z_t|z_{t-1}))
        # Both are Gaussian: q = N(mu_q, exp(logvar_q)), p = N(mu_prior, exp(logvar_prior))
        var_q = torch.exp(logvar_q).clamp(min=1e-8)
        var_p = torch.exp(logvar_prior).clamp(min=1e-8)
        kl_per_dim = 0.5 * (
            logvar_prior - logvar_q
            + var_q / var_p
            + (mu_q - mu_prior).pow(2) / var_p
            - 1.0
        )  # (B, T, D_z)
        kl_loss = kl_per_dim.sum(dim=2).sum(dim=1).mean()  # sum over D_z,T; mean over B

        # 5. Flow correction: -sum_t log|det J_t|
        # The ELBO includes +log_det (since flow makes posterior tighter),
        # so in the negative ELBO (loss) it appears as -log_det.
        flow_loss = -log_det.sum(dim=1).mean()  # sum over T, mean over B

        # 6. Total loss = recon + kl_weight * kl - flow_correction
        # = recon + kl_weight * kl + flow_loss (since flow_loss = -log_det)
        total_loss = recon_loss + kl_weight * kl_loss + flow_loss

        return OmniAnomalyOutput(
            total_loss=total_loss,
            recon_loss=recon_loss,
            kl_loss=kl_loss,
            flow_loss=flow_loss,
            mu_x=mu_x,
            logvar_x=logvar_x,
        )

    @torch.no_grad()
    def anomaly_score(
        self,
        x: torch.Tensor,
        mc_samples: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Compute anomaly scores via reconstruction probability.

        For each MC sample, computes per-timestep-per-dimension reconstruction error,
        then aggregates.

        Args:
            x:           (batch, T, D)
            mc_samples:  number of MC forward passes

        Returns dict:
            step_score:   (batch, T) — anomaly score per timestep
            final_score:  (batch,)   — aggregated score per sample
            error:        (batch, T, D) — per-dim reconstruction error (last MC sample)
        """
        B, T, D = x.shape
        all_scores = []
        last_error = None

        for _ in range(mc_samples):
            q_out = self.qnet(x)
            p_out = self.pnet(q_out["z_seq"])
            mu_x = p_out["mu_x"]
            logvar_x = p_out["logvar_x"]

            # Reconstruction NLL per dim
            var_x = torch.exp(logvar_x).clamp(min=1e-6)
            nll = 0.5 * (logvar_x + (x - mu_x).pow(2) / var_x + math.log(2 * math.pi))
            # Also track simple squared error for interpretable per-dim error
            error = (x - mu_x).pow(2)
            last_error = error

            step_score = nll.sum(dim=2)  # (B, T)
            all_scores.append(step_score)

        # Average over MC samples
        step_score = torch.stack(all_scores, dim=0).mean(dim=0)  # (B, T)
        # Final score: mean over timesteps
        final_score = step_score.mean(dim=1)  # (B,)

        return {
            "step_score": step_score,
            "final_score": final_score,
            "error": last_error,
        }


def build_omni_anomaly(
    input_dim: int,
    rnn_hidden: int = 128,
    latent_dim: int = 16,
    rnn_layers: int = 1,
    dropout: float = 0.0,
    num_flows: int = 10,
    decoder_rnn_hidden: int = 128,
) -> OmniAnomaly:
    """Factory function."""
    return OmniAnomaly(
        input_dim=input_dim,
        rnn_hidden=rnn_hidden,
        latent_dim=latent_dim,
        rnn_layers=rnn_layers,
        dropout=dropout,
        num_flows=num_flows,
        decoder_rnn_hidden=decoder_rnn_hidden,
    )
