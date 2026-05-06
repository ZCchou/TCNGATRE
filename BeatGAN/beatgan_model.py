from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def conv1d_out_len(length: int, kernel_size: int = 4, stride: int = 2, padding: int = 1) -> int:
    return (length + 2 * padding - kernel_size) // stride + 1


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(flag)


def weights_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.01)


@dataclass
class BeatGANConfig:
    in_channels: int
    seq_len: int
    nz: int = 48
    ndf: int = 32
    ngf: int = 32
    lambda_adv: float = 1.0
    lr: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    device: str = "cpu"

    def reduced_len(self) -> int:
        length = int(self.seq_len)
        for _ in range(5):
            length = conv1d_out_len(length, 4, 2, 1)
        if length < 1:
            raise ValueError(f"seq_len={self.seq_len} is too short for 5 downsampling layers.")
        if int(self.seq_len) % 32 != 0:
            raise ValueError(
                f"seq_len={self.seq_len} is not divisible by 32. "
                "Use lengths like 32, 64, 128, 256, 320 for this architecture."
            )
        return int(length)


class Encoder1D(nn.Module):
    def __init__(self, cfg: BeatGANConfig, out_channels: int):
        super().__init__()
        self.final_len = cfg.reduced_len()
        width = int(cfg.ndf)
        self.features = nn.Sequential(
            nn.Conv1d(cfg.in_channels, width, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(width, width * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(width * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(width * 2, width * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(width * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(width * 4, width * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(width * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(width * 8, width * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(width * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.to_latent = nn.Conv1d(width * 16, out_channels, kernel_size=self.final_len, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.features(x)
        latent = self.to_latent(features)
        if return_features:
            return latent, features
        return latent


class Decoder1D(nn.Module):
    def __init__(self, cfg: BeatGANConfig):
        super().__init__()
        final_len = cfg.reduced_len()
        width = int(cfg.ngf)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(cfg.nz, width * 16, kernel_size=final_len, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(width * 16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(width * 16, width * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(width * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(width * 8, width * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(width * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(width * 4, width * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(width * 2, width, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(width, cfg.in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Generator(nn.Module):
    def __init__(self, cfg: BeatGANConfig):
        super().__init__()
        self.encoder = Encoder1D(cfg, out_channels=cfg.nz)
        self.decoder = Decoder1D(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class Discriminator(nn.Module):
    def __init__(self, cfg: BeatGANConfig):
        super().__init__()
        self.encoder = Encoder1D(cfg, out_channels=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logit, feat = self.encoder(x, return_features=True)
        prob = torch.sigmoid(logit.view(logit.size(0), -1).squeeze(1))
        return prob, feat


class BeatGAN:
    def __init__(self, cfg: BeatGANConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.G = Generator(cfg).to(self.device)
        self.D = Discriminator(cfg).to(self.device)
        self.G.apply(weights_init)
        self.D.apply(weights_init)

        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

    def _unpack_batch(self, batch) -> torch.Tensor:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        return batch.to(self.device, non_blocking=True)

    def train_step(self, x: torch.Tensor) -> dict[str, float]:
        x = x.to(self.device)
        batch_size = x.size(0)
        ones = torch.ones(batch_size, device=self.device)
        zeros = torch.zeros(batch_size, device=self.device)

        set_requires_grad(self.D, True)
        self.opt_d.zero_grad(set_to_none=True)

        real_prob, _ = self.D(x)
        x_hat, _ = self.G(x)
        fake_prob, _ = self.D(x_hat.detach())

        d_loss_real = self.bce(real_prob, ones)
        d_loss_fake = self.bce(fake_prob, zeros)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.opt_d.step()

        set_requires_grad(self.D, False)
        self.opt_g.zero_grad(set_to_none=True)

        x_hat, _ = self.G(x)
        _, feat_fake = self.D(x_hat)
        with torch.no_grad():
            _, feat_real = self.D(x)

        adv_loss = self.mse(feat_fake, feat_real)
        rec_loss = self.mse(x_hat, x)
        g_loss = rec_loss + self.cfg.lambda_adv * adv_loss
        g_loss.backward()
        self.opt_g.step()
        set_requires_grad(self.D, True)

        return {
            "d_loss": float(d_loss.item()),
            "d_real": float(d_loss_real.item()),
            "d_fake": float(d_loss_fake.item()),
            "g_loss": float(g_loss.item()),
            "g_rec": float(rec_loss.item()),
            "g_adv": float(adv_loss.item()),
        }

    def fit(self, train_loader: DataLoader, epochs: int = 50, verbose: bool = True) -> dict[str, list[float]]:
        history: dict[str, list[float]] = defaultdict(list)
        self.G.train()
        self.D.train()
        for epoch in range(1, int(epochs) + 1):
            meter: dict[str, float] = defaultdict(float)
            steps = 0
            for batch in train_loader:
                losses = self.train_step(self._unpack_batch(batch))
                for key, value in losses.items():
                    meter[key] += float(value)
                steps += 1
            for key in list(meter):
                meter[key] /= max(steps, 1)
                history[key].append(meter[key])
            if verbose:
                print(
                    f"[{epoch:03d}/{epochs:03d}] "
                    f"D={meter['d_loss']:.4f} "
                    f"G={meter['g_loss']:.4f} "
                    f"G_rec={meter['g_rec']:.4f} "
                    f"G_adv={meter['g_adv']:.4f}"
                )
        return history

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        self.G.eval()
        return self.G(x.to(self.device))[0]

    @torch.no_grad()
    def anomaly_components(self, x: torch.Tensor, mode: str = "rec") -> dict[str, torch.Tensor]:
        self.G.eval()
        self.D.eval()

        x = x.to(self.device)
        x_hat, _ = self.G(x)
        error = (x_hat - x) ** 2
        rec = torch.mean(error, dim=(1, 2))

        if mode == "rec":
            final_score = rec
            feature_error = torch.zeros_like(rec)
        elif mode == "combined":
            _, feat_real = self.D(x)
            _, feat_fake = self.D(x_hat)
            feature_error = torch.mean((feat_fake - feat_real) ** 2, dim=(1, 2))
            final_score = rec + self.cfg.lambda_adv * feature_error
        else:
            raise ValueError("mode must be 'rec' or 'combined'")

        return {
            "reconstruction": x_hat,
            "error": error,
            "step_score": torch.mean(error, dim=1),
            "rec_score": rec,
            "feature_score": feature_error,
            "final_score": final_score,
        }

    @torch.no_grad()
    def predict_loader_scores(self, loader: DataLoader, mode: str = "rec") -> torch.Tensor:
        scores = []
        for batch in loader:
            x = self._unpack_batch(batch)
            scores.append(self.anomaly_components(x, mode=mode)["final_score"].detach().cpu())
        return torch.cat(scores, dim=0) if scores else torch.zeros(0, dtype=torch.float32)


def window_batch_to_beatgan_input(window: torch.Tensor) -> torch.Tensor:
    x = window.transpose(1, 2).contiguous()
    return x * 2.0 - 1.0
