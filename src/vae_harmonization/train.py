"""
train.py — Training loop for the FC-VAE with KL annealing and GRL scheduling.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .config import VAEConfig
from .logging_utils import get_logger
from .losses import combined_loss
from .models import FCVAE

log = get_logger(__name__)


def _beta_schedule(epoch: int, cfg: VAEConfig) -> float:
    """Linear warmup: 0 → β_final over beta_warmup_epochs, then hold."""
    if epoch >= cfg.beta_warmup_epochs:
        return cfg.beta_final
    return cfg.beta_final * (epoch / cfg.beta_warmup_epochs)


def _active_dims(mu: torch.Tensor, logvar: torch.Tensor, threshold: float = 0.2) -> int:
    """Count latent dimensions with KL > threshold (averaged over batch)."""
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)     # (B, d)
    mean_kl = kl_per_dim.mean(dim=0)                                   # (d,)
    return int((mean_kl > threshold).sum().item())


def _run_epoch(
    model: FCVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    beta: float,
    cfg: VAEConfig,
    device: torch.device,
) -> dict[str, float]:
    """Run one training or validation epoch. Pass optimizer=None for val."""
    is_train = optimizer is not None
    model.train(is_train)

    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0, "site": 0.0, "bio": 0.0}
    n_samples = 0
    all_mu, all_logvar = [], []

    for x, site, diag in loader:
        x = x.to(device)
        site = site.to(device)
        diag = diag.to(device)

        x_hat, mu, logvar, site_logits, diag_logits = model(x)
        losses = combined_loss(
            x, x_hat, mu, logvar,
            site_logits, site,
            diag_logits, diag,
            beta=beta, cfg=cfg,
        )

        if is_train:
            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        bs = x.size(0)
        for k in totals:
            totals[k] += losses[k].item() * bs
        n_samples += bs

        all_mu.append(mu.detach())
        all_logvar.append(logvar.detach())

    # Average over samples
    metrics = {k: v / n_samples for k, v in totals.items()}
    metrics["beta"] = beta
    metrics["active_dims"] = _active_dims(
        torch.cat(all_mu), torch.cat(all_logvar)
    )
    return metrics


def train_vae(
    cfg: VAEConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device | None = None,
) -> tuple[FCVAE, list[dict]]:
    """
    Full training loop.

    Returns
    -------
    model : FCVAE with best weights restored
    history : list of per-epoch metric dicts
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FCVAE(cfg).to(device)
    optimizer = AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=cfg.lr_patience,
        factor=cfg.lr_factor,
        min_lr=cfg.lr_min,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history: list[dict] = []

    log.info("training on %s | %s params | lambda_site=%.2f lambda_bio=%.2f",
             device, f"{sum(p.numel() for p in model.parameters()):,}", cfg.lambda_site, cfg.lambda_bio)
    log.info("%4s %8s %8s %8s %8s %5s %5s %9s", "ep", "trn_rec", "trn_kl", "val_rec", "val_kl", "beta", "act_d", "lr")

    for epoch in range(cfg.max_epochs):
        beta = _beta_schedule(epoch, cfg)
        lam_grl = model.update_grl_lambda(epoch, cfg.max_epochs, cfg.gamma_grl)

        trn = _run_epoch(model, train_loader, optimizer, beta, cfg, device)
        val = _run_epoch(model, val_loader, None, beta, cfg, device)

        # Monitor val recon + β·KL (no auxiliary losses for early stopping)
        val_monitor = val["recon"] + beta * val["kl"]
        scheduler.step(val_monitor)

        lr_now = optimizer.param_groups[0]["lr"]
        if epoch % 10 == 0 or epoch == cfg.max_epochs - 1:
            log.info("%4d %8.4f %8.4f %8.4f %8.4f %5.3f %5d %9.2e", epoch, trn["recon"], trn["kl"],
                     val["recon"], val["kl"], beta, val["active_dims"], lr_now)

        row = {"epoch": epoch, "lr": lr_now, "lam_grl": lam_grl}
        for prefix, m in [("trn_", trn), ("val_", val)]:
            for k, v in m.items():
                row[prefix + k] = v
        history.append(row)

        # Early stopping on val_monitor
        if val_monitor < best_val_loss:
            best_val_loss = val_monitor
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                log.info("early stopping at epoch %d (patience=%d)", epoch, cfg.patience)
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def save_checkpoint(model: FCVAE, cfg: VAEConfig, history: list[dict]):
    """Save model weights and training history."""
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.checkpoint_dir / f"fcvae_d{cfg.latent_dim}.pt"
    torch.save(model.state_dict(), ckpt_path)
    hist_path = cfg.checkpoint_dir / f"history_d{cfg.latent_dim}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2, default=float)
    log.info("checkpoint + history saved -> %s", cfg.checkpoint_dir)


def load_checkpoint(cfg: VAEConfig, device: torch.device | None = None) -> FCVAE:
    """Load a saved model checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCVAE(cfg).to(device)
    ckpt_path = cfg.checkpoint_dir / f"fcvae_d{cfg.latent_dim}.pt"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Backward compatibility: legacy checkpoints may contain site_head weights
    # even when current run disables site head (lambda_site=0.0).
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)

    unexpected_non_site = [k for k in unexpected if not k.startswith("site_head.")]
    if unexpected_non_site:
        raise RuntimeError(
            f"Unexpected checkpoint keys not related to site_head: {unexpected_non_site}"
        )

    if unexpected:
        log.debug("checkpoint: ignored unexpected keys %s", unexpected)
    if missing:
        log.debug("checkpoint: missing keys initialised from defaults %s", missing)

    model.eval()
    return model
