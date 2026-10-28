"""
insample.py -- In-sample diagnostics of the trained VAE and of a single
harmonization map fitted on the whole cohort.

Everything in here is *descriptive* (reconstruction quality, Gaussianity of
the latent, figures).  Numbers that appear in the main tables of the paper
come exclusively from :mod:`vae_harmonization.evaluation.protocol`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import VAEConfig
from ..analyses.gaussianity import full_gaussianity_report
from ..harmonization import ProtectedAffineHarmonizer, decompose, make_variant
from ..logging_utils import get_logger
from .latents import extract_latents
from .plots import (
    plot_complement_site,
    plot_covariance_heatmaps,
    plot_protected_axis,
    plot_training_curves,
    plot_tsne,
)

log = get_logger(__name__)


def reconstruction_metrics(X_orig, X_hat, y_site, site_names: dict) -> dict:
    """Per-site and overall MSE and mean per-subject Pearson r."""
    def _r(a, b):
        r = np.corrcoef(a, b)[0, 1]
        return 0.0 if np.isnan(r) else float(r)
    out = {}
    for sid, name in site_names.items():
        m = y_site == sid
        if not m.any():
            continue
        out[name] = {"mse": float(np.mean((X_orig[m] - X_hat[m]) ** 2)),
                     "pearson_r": float(np.mean([_r(X_orig[m][i], X_hat[m][i]) for i in range(m.sum())])),
                     "n": int(m.sum())}
    out["ALL"] = {"mse": float(np.mean((X_orig - X_hat) ** 2)),
                  "pearson_r": float(np.mean([_r(X_orig[i], X_hat[i]) for i in range(len(X_orig))])),
                  "n": int(len(X_orig))}
    return out


def run_insample_diagnostics(model, dataset, cfg: VAEConfig, history: list[dict], device,
                             make_plots: bool = True) -> dict:
    """
    Latent extraction, Gaussianity report, reconstruction quality and the
    figures for a whole-cohort fit of the proposed map (labelled in-sample).
    """
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    Z_mu, Z_logvar, y_site, y_diag, X_hat = extract_latents(model, dataset, device)
    y_site = y_site.astype(int)
    log.info("latents: %s", Z_mu.shape)

    gauss = full_gaussianity_report(Z_mu, Z_mu, Z_logvar, label="All sites", out_dir=fig_dir,
                                    free_bits=cfg.free_bits)

    X_pca = np.asarray(dataset.X)
    recon_pca = reconstruction_metrics(X_pca, X_hat, y_site, cfg.site_names)
    recon_raw = reconstruction_metrics(dataset.X_raw, dataset.inverse_to_raw(X_hat), y_site, cfg.site_names)
    log.info("reconstruction (raw FC space): MSE=%.5f r=%.4f", recon_raw["ALL"]["mse"], recon_raw["ALL"]["pearson_r"])

    h: ProtectedAffineHarmonizer = make_variant("ours", target_site=int(cfg.target_site),
                                                min_cov_samples=cfg.min_cov_samples, svm_C=cfg.bio_svm_C,
                                                seed=cfg.seed)
    h.fit(Z_mu, y_site, y_diag=y_diag)
    Z_h = h.transform(Z_mu, y_site)

    if make_plots:
        if history:
            plot_training_curves(history, fig_dir / "training_curves.png")
        plot_tsne(Z_mu, y_site, cfg.site_names, cfg.site_colors, "t-SNE of z by site (pre)",
                  fig_dir / "tsne_site_pre.png", cfg.seed)
        plot_tsne(Z_h, y_site, cfg.site_names, cfg.site_colors, "t-SNE of z by site (post, in-sample)",
                  fig_dir / "tsne_site_post.png", cfg.seed)
        plot_tsne(Z_mu, y_diag, cfg.diag_names, cfg.diag_colors, "t-SNE of z by diagnosis (pre)",
                  fig_dir / "tsne_diag_pre.png", cfg.seed)
        plot_tsne(Z_h, y_diag, cfg.diag_names, cfg.diag_colors, "t-SNE of z by diagnosis (post, in-sample)",
                  fig_dir / "tsne_diag_post.png", cfg.seed)
        Z_by = {s: Z_mu[y_site == s] for s in cfg.site_names if (y_site == s).any()}
        Zh_by = {s: Z_h[y_site == s] for s in cfg.site_names if (y_site == s).any()}
        plot_covariance_heatmaps(Z_by, Zh_by, int(cfg.target_site), cfg.site_names, fig_dir)
        a_pre, P_pre = decompose(Z_mu, h.W_bio_, h.W_perp_)
        a_post, P_post = decompose(Z_h, h.W_bio_, h.W_perp_)
        plot_protected_axis(a_pre[:, 0], a_post[:, 0], y_diag, cfg.diag_names, cfg.diag_colors,
                            fig_dir / "protected_axis.png")
        plot_complement_site(P_pre, P_post, y_site, cfg.site_names, cfg.site_colors,
                             fig_dir / "complement_site.png", cfg.seed)

    return {
        "gaussianity": gauss,
        "reconstruction_raw": recon_raw,
        "reconstruction_pca": recon_pca,
        "insample_map": h.describe(),
        "Z_mu": Z_mu,
        "Z_logvar": Z_logvar,
        "y_site": y_site,
        "y_diag": y_diag,
        "Z_harm_insample": Z_h,
        "harmonizer_insample": h,
    }
