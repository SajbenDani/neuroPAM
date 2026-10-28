"""
gaussianity.py — Latent-space Gaussianity diagnostics.

Tests whether the VAE latent vectors z (per-site) are approximately Gaussian,
which is the prerequisite for the closed-form affine harmonization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def shapiro_wilk_per_dim(
    Z: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Run Shapiro-Wilk test on each latent dimension.

    Parameters
    ----------
    Z : (N, d) array of latent vectors
    alpha : significance level

    Returns
    -------
    dict with keys:
        p_values : (d,) array
        pass_rate : fraction of dimensions with p > alpha
        n_pass : count of passing dimensions
        n_total : d
    """
    d = Z.shape[1]
    p_values = np.zeros(d)
    for j in range(d):
        _, p = stats.shapiro(Z[:, j])
        p_values[j] = p

    n_pass = int((p_values > alpha).sum())
    return {
        "p_values": p_values,
        "pass_rate": n_pass / d,
        "n_pass": n_pass,
        "n_total": d,
    }


def active_dims_count(
    Z: np.ndarray,
    mu: np.ndarray | None = None,
    logvar: np.ndarray | None = None,
    threshold: float = 0.2,
) -> int:
    """
    Count latent dimensions with mean KL > threshold.

    If mu and logvar are provided, compute KL analytically.
    Otherwise approximate from the variance of Z columns.
    """
    if mu is not None and logvar is not None:
        kl_per_dim = 0.5 * (mu ** 2 + np.exp(logvar) - logvar - 1.0)
        mean_kl = kl_per_dim.mean(axis=0)
    else:
        # Approximate: KL ≈ 0.5*(var − 1 − log(var) + mean²)
        var = Z.var(axis=0)
        mean = Z.mean(axis=0)
        mean_kl = 0.5 * (var - 1.0 - np.log(np.clip(var, 1e-8, None)) + mean ** 2)
    return int((mean_kl > threshold).sum())


def qq_plots(
    Z: np.ndarray,
    n_dims: int = 6,
    save_path: Path | None = None,
    title_prefix: str = "",
) -> plt.Figure:
    """
    QQ plots for the first `n_dims` latent dimensions against N(0,1).
    """
    n_dims = min(n_dims, Z.shape[1])
    ncols = min(3, n_dims)
    nrows = (n_dims + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.atleast_2d(axes)

    for i in range(n_dims):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        stats.probplot(Z[:, i], dist="norm", plot=ax)
        ax.set_title(f"{title_prefix}Dim {i}")
        ax.get_lines()[0].set_markersize(3)

    # Hide unused axes
    for i in range(n_dims, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def kl_histogram(
    mu: np.ndarray,
    logvar: np.ndarray,
    free_bits: float = 0.125,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Histogram of per-dimension mean KL values.
    Shows which dimensions are informative vs. collapsed to prior.
    """
    kl_per_dim = 0.5 * (mu ** 2 + np.exp(logvar) - logvar - 1.0)  # (N, d)
    mean_kl = kl_per_dim.mean(axis=0)  # (d,)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(mean_kl)), sorted(mean_kl, reverse=True), color="#4e79a7")
    ax.axhline(free_bits, color="red", ls="--", lw=1, label=f"free_bits = {free_bits}")
    ax.set_xlabel("Latent dimension (sorted by KL)")
    ax.set_ylabel("Mean KL (nats)")
    ax.set_title("Per-dimension KL divergence")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def full_gaussianity_report(
    Z: np.ndarray,
    mu: np.ndarray | None = None,
    logvar: np.ndarray | None = None,
    label: str = "",
    out_dir: Path | None = None,
    free_bits: float = 0.125,
) -> dict:
    """
    Run all Gaussianity diagnostics and save plots.

    Returns summary dict.
    """
    sw = shapiro_wilk_per_dim(Z)
    n_active = active_dims_count(Z, mu, logvar)

    result = {
        "label": label,
        "shapiro_wilk_pass_rate": sw["pass_rate"],
        "shapiro_wilk_n_pass": sw["n_pass"],
        "shapiro_wilk_n_total": sw["n_total"],
        "active_dims": n_active,
    }

    suffix = label.lower().replace(" ", "_") if label else "all"

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        qq_plots(Z, n_dims=6, save_path=out_dir / f"qq_{suffix}.png",
                 title_prefix=f"{label} " if label else "")
        if mu is not None and logvar is not None:
            kl_histogram(mu, logvar, free_bits=free_bits,
                         save_path=out_dir / f"kl_hist_{suffix}.png")

    from ..logging_utils import get_logger
    get_logger(__name__).info(f"Gaussianity [{label}]: Shapiro-Wilk pass {sw['n_pass']}/{sw['n_total']} "
          f"({sw['pass_rate']:.0%}), active dims = {n_active}")

    return result


def mardia_test(Z: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Mardia's multivariate normality test (skewness + kurtosis).

    Parameters
    ----------
    Z : (N, d) array of latent vectors
    alpha : significance level

    Returns
    -------
    dict with 'skewness_stat', 'skewness_p', 'kurtosis_stat', 'kurtosis_p',
    'mv_normal' (both tests pass at alpha)
    """
    N, d = Z.shape
    Z_centered = Z - Z.mean(axis=0)
    S_inv = np.linalg.pinv(np.cov(Z_centered, rowvar=False))

    # D_ij = (z_i - mu)^T S^{-1} (z_j - mu)
    D = Z_centered @ S_inv @ Z_centered.T  # (N, N)

    # Mardia's skewness: b_{1,d} = (1/N^2) * sum_ij D_ij^3
    b1d = (D ** 3).sum() / (N * N)
    # Under H0: (N/6) * b1d ~ chi^2(d*(d+1)*(d+2)/6)
    skew_stat = N * b1d / 6.0
    skew_df = d * (d + 1) * (d + 2) / 6.0
    skew_p = 1.0 - stats.chi2.cdf(skew_stat, skew_df)

    # Mardia's kurtosis: b_{2,d} = (1/N) * sum_i D_ii^2
    b2d = (np.diag(D) ** 2).sum() / N
    # Under H0: b2d ~ N(d*(d+2), 8*d*(d+2)/N)
    kurt_mean = d * (d + 2)
    kurt_var = 8.0 * d * (d + 2) / N
    kurt_stat = (b2d - kurt_mean) / np.sqrt(kurt_var) if kurt_var > 0 else 0.0
    kurt_p = 2.0 * (1.0 - stats.norm.cdf(abs(kurt_stat)))

    mv_normal = (skew_p > alpha) and (kurt_p > alpha)

    return {
        "skewness_stat": float(skew_stat),
        "skewness_p": float(skew_p),
        "kurtosis_stat": float(kurt_stat),
        "kurtosis_p": float(kurt_p),
        "mv_normal": bool(mv_normal),
    }
