"""
plots.py -- Diagnostic figures (t-SNE, training curves, covariance heatmaps,
protected-axis histograms, complement scatter).  ASCII-only labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_tsne(Z, labels, label_names: dict, colors: dict, title: str, save_path: Path, seed: int = 42):
    tsne = TSNE(n_components=2, random_state=seed, perplexity=min(30, len(Z) - 1))
    Z2 = tsne.fit_transform(Z)
    fig, ax = plt.subplots(figsize=(7, 6))
    for lid in sorted(label_names):
        m = labels == lid
        if m.any():
            ax.scatter(Z2[m, 0], Z2[m, 1], alpha=0.5, s=14, label=label_names[lid],
                       color=colors.get(lid, None), edgecolors="none")
    ax.set_title(title)
    ax.legend(markerscale=2.5, fontsize=7, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_training_curves(history: list[dict], save_path: Path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [h["trn_recon"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_recon"] for h in history], label="val")
    axes[0].set_title("Reconstruction loss (MSE)")
    axes[1].plot(epochs, [h["trn_kl"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_kl"] for h in history], label="val")
    axes[1].set_title("KL divergence")
    axes[2].plot(epochs, [h["val_active_dims"] for h in history], color="k")
    axes[2].set_title("Active latent dims (val)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_covariance_heatmaps(Z_by_site: dict, Z_harm_by_site: dict, target_site: int,
                             site_names: dict, out_dir: Path, max_sites: int = 6):
    """Before / target / after covariance heatmaps for the largest non-target sites."""
    C_t = np.cov(Z_by_site[target_site], rowvar=False)
    others = sorted((s for s in Z_by_site if s != target_site), key=lambda s: -len(Z_by_site[s]))[:max_sites]
    for sid in others:
        if len(Z_by_site[sid]) < 3:
            continue
        C_b = np.cov(Z_by_site[sid], rowvar=False)
        C_a = np.cov(Z_harm_by_site[sid], rowvar=False)
        vmin, vmax = min(C_b.min(), C_t.min(), C_a.min()), max(C_b.max(), C_t.max(), C_a.max())
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, C, lab in zip(axes, [C_b, C_t, C_a],
                              [f"{site_names[sid]} (before)", f"{site_names[target_site]} (hub)",
                               f"{site_names[sid]} (after)"]):
            im = ax.imshow(C, vmin=vmin, vmax=vmax, cmap="RdBu_r", aspect="equal")
            ax.set_title(lab, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.8)
        fig.savefig(out_dir / f"latent_cov_{site_names[sid]}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_protected_axis(alpha_pre, alpha_post, y_diag, diag_names: dict, diag_colors: dict, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, alpha, title in zip(axes, [alpha_pre, alpha_post], ["Pre-harmonization", "Post-harmonization"]):
        for lid in sorted(diag_names):
            m = y_diag == lid
            ax.hist(alpha[m], bins=30, alpha=0.55, label=diag_names[lid], color=diag_colors.get(lid), density=True)
        ax.set_title(title)
        ax.set_xlabel("Protected coordinate (alpha)")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Protected axis: diagnosis separation")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_complement_site(P_pre, P_post, y_site, site_names: dict, site_colors: dict, save_path: Path, seed=42):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, P, title in zip(axes, [P_pre, P_post], ["Pre-harmonization", "Post-harmonization"]):
        Z2 = PCA(n_components=2, random_state=seed).fit_transform(P)
        for sid in sorted(site_names):
            m = y_site == sid
            if m.any():
                ax.scatter(Z2[m, 0], Z2[m, 1], s=14, alpha=0.5, label=site_names[sid],
                           color=site_colors.get(sid), edgecolors="none")
        ax.set_title(title)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=6, markerscale=2, ncol=2)
    fig.suptitle("Complement subspace: site overlap")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_method_bars(results: dict, save_path: Path, classifier: str = "lr"):
    """Site accuracy (with majority-rate line) and diagnosis accuracy per method."""
    names = list(results["methods"].keys())
    site = [results["methods"][n]["summary"]["site"][classifier]["acc"] for n in names]
    diag = [results["methods"][n]["summary"]["diag"][classifier]["acc"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 1.2 * len(names)), 4))
    x = np.arange(len(names))
    for ax, vals, title, chance in (
        (axes[0], site, "Site accuracy (lower is better)", results["chance"]["site"]["majority_rate"]),
        (axes[1], diag, "Diagnosis accuracy (higher is better)", results["chance"]["diag"]["majority_rate"]),
    ):
        means = [v["mean"] for v in vals]
        err = [[v["mean"] - v["ci_low"] if v["mean"] == v["mean"] else 0 for v in vals],
               [v["ci_high"] - v["mean"] if v["mean"] == v["mean"] else 0 for v in vals]]
        ax.bar(x, means, yerr=err, capsize=3, color="#4e79a7")
        ax.axhline(chance, color="gray", ls="--", lw=0.9, label=f"majority rate {chance:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
