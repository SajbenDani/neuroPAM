"""
losses.py — Loss functions for the FC-VAE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VAEConfig


def kl_divergence_free_bits(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float = 0.125,
) -> torch.Tensor:
    """
    KL(q(z|x) || p(z)) with a per-sample, per-dimension free-bits floor.

    KL_ij = 0.5 * (mu_ij^2 + exp(logvar_ij) - logvar_ij - 1)      (sample i, dimension j)
    L_KL  = mean_over_batch_i( sum_j max(KL_ij, free_bits) )

    Note: this clamps each sample's per-dimension KL before averaging over the batch.  The
    original free-bits formulation (Kingma et al., 2016) clamps the *batch-averaged* KL of each
    dimension.  The per-sample variant is what the frozen checkpoints were trained with, so it is
    kept unchanged; the paper describes it as such.
    """
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)   # (B, d)
    kl_clamped = kl_per_dim.clamp(min=free_bits)                     # (B, d)
    return kl_clamped.sum(dim=1).mean()                              # scalar


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """MSE reconstruction loss, mean over batch and features."""
    return F.mse_loss(x_hat, x, reduction="mean")


def combined_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    site_logits: torch.Tensor | None,
    site_labels: torch.Tensor,
    diag_logits: torch.Tensor,
    diag_labels: torch.Tensor,
    beta: float,
    cfg: VAEConfig,
) -> dict[str, torch.Tensor]:
    """
    Compute all loss terms and the combined scalar.

    Returns a dict with keys:
        total, recon, kl, site, bio, beta
    """
    l_recon = reconstruction_loss(x, x_hat)
    l_kl = kl_divergence_free_bits(mu, logvar, free_bits=cfg.free_bits)
    if site_logits is None:
        l_site = torch.tensor(0.0, device=x.device)
    else:
        l_site = F.cross_entropy(site_logits, site_labels)
    l_bio = F.binary_cross_entropy_with_logits(diag_logits, diag_labels)

    total = (
        cfg.lambda_recon * l_recon
        + beta * l_kl
        + cfg.lambda_site * l_site
        + cfg.lambda_bio * l_bio
    )

    return {
        "total": total,
        "recon": l_recon.detach(),
        "kl": l_kl.detach(),
        "site": l_site.detach(),
        "bio": l_bio.detach(),
        "beta": torch.tensor(beta),
    }
