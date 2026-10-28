"""
models.py — Encoder, Decoder, GRL, auxiliary heads, and full FCVAE.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.autograd import Function

from .config import VAEConfig


# ── Gradient Reversal Layer ───────────────────────────────────────────────────


class _GRLFunction(Function):
    """Gradient reversal: forward = identity, backward = −λ · grad."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lam * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = 0.0

    def set_lambda(self, lam: float):
        self.lam = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRLFunction.apply(x, self.lam)


# ── Encoder ───────────────────────────────────────────────────────────────────


class Encoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        dims = [cfg.input_dim] + cfg.encoder_dims
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.LeakyReLU(cfg.leaky_relu_slope))
            if i < len(dims) - 2:          # no dropout on last hidden
                layers.append(nn.Dropout(cfg.dropout_main))
        self.backbone = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(cfg.encoder_dims[-1], cfg.latent_dim)
        self.fc_logvar = nn.Linear(cfg.encoder_dims[-1], cfg.latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.fc_mu(h), self.fc_logvar(h)


# ── Decoder ───────────────────────────────────────────────────────────────────


class Decoder(nn.Module):
    """MLP decoder with the encoder's hidden widths in reverse order.  It is not an exact mirror of
    the encoder: it uses ReLU (the encoder uses LeakyReLU) and applies dropout after every hidden
    layer (the encoder skips dropout on its last hidden layer)."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        dims = [cfg.latent_dim] + cfg.decoder_dims
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.dropout_main))
        layers.append(nn.Linear(dims[-1], cfg.input_dim))  # no activation
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ── Auxiliary heads ───────────────────────────────────────────────────────────


class SiteHead(nn.Module):
    """Site classifier on GRL-reversed z → 3-class logits."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.net = nn.Sequential(
            nn.Dropout(cfg.dropout_head),
            nn.Linear(cfg.latent_dim, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, cfg.n_sites),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.grl(z))


class BioHead(nn.Module):
    """Diagnosis preservation head — z → sigmoid probability."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(cfg.dropout_head),
            nn.Linear(cfg.latent_dim, cfg.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)       # (B,)


# ── Full model ────────────────────────────────────────────────────────────────


class FCVAE(nn.Module):
    """
    Full FC-VAE with adversarial site removal and bio preservation.

    Forward returns:
        x_hat, mu, logvar, site_logits, diag_logits
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.use_site_head = cfg.lambda_site > 0.0
        self.site_head = SiteHead(cfg) if self.use_site_head else None
        self.bio_head = BioHead(cfg)

    # ── reparameterise ────────────────────────────────────────────────────

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    # ── GRL λ schedule (Ganin et al. 2016) ────────────────────────────────

    def update_grl_lambda(self, epoch: int, max_epochs: int, gamma: float):
        if not self.use_site_head:
            return 0.0
        p = epoch / max_epochs
        lam = 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0
        self.site_head.grl.set_lambda(lam)
        return lam

    # ── forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decoder(z)
        site_logits = self.site_head(z) if self.site_head is not None else None
        diag_logits = self.bio_head(z)
        return x_hat, mu, logvar, site_logits, diag_logits

    # ── Inference helpers ─────────────────────────────────────────────────

    @torch.no_grad()
    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        """Return deterministic μ vectors (no sampling)."""
        mu, _ = self.encoder(x)
        return mu
