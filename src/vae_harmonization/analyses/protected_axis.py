"""
Where does the site effect live relative to the protected (diagnosis) axis?

Reviewers asked whether leaving the protected coordinate out of the
covariance-matching step "leaves the site effect along the diagnosis axis
untreated".  This analysis quantifies, on the fitted latent space:

* the share of total between-site variance that falls inside the protected
  subspace versus its orthogonal complement (before harmonization);
* eta^2 / epsilon^2 of site for the protected coordinate before and after the
  mean shift (in-sample; the leakage-free per-fold values are in
  ``protocol_results.json`` under ``protected_eps2``);
* the diagnosis effect size (Cohen's d) along the protected coordinate before
  and after harmonization -- it must be unchanged by construction;
* per-latent-dimension site eta^2 before/after (for the figure).
"""

from __future__ import annotations

import numpy as np

from ..harmonization.affine import ProtectedAffineHarmonizer
from ..metrics import compute_epsilon_squared, compute_eta_squared, compute_eta_squared_per_dim, eta_squared_null_expectation


def _between_site_ss(X: np.ndarray, y_site: np.ndarray) -> float:
    grand = X.mean(axis=0)
    return float(sum((y_site == s).sum() * np.sum((X[y_site == s].mean(axis=0) - grand) ** 2)
                     for s in np.unique(y_site)))


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    a, b = x[y == 1], x[y == 0]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / (sp + 1e-12))


def protected_axis_report(Z: np.ndarray, y_site: np.ndarray, y_diag: np.ndarray,
                          harmonizer: ProtectedAffineHarmonizer, severity=None, covariates=None) -> dict:
    """Fit ``harmonizer`` on the full latent matrix and decompose the site variance."""
    Z = np.asarray(Z, dtype=np.float64)
    y_site = np.asarray(y_site).astype(int)
    y_diag = np.asarray(y_diag).astype(int)
    labelled = y_diag >= 0
    harmonizer.fit(Z, y_site, y_diag=y_diag, severity=severity, covariates=covariates)
    Zh = harmonizer.transform(Z, y_site)
    W_bio, W_perp = harmonizer.W_bio_, harmonizer.W_perp_
    A, P = Z @ W_bio, Z @ W_perp
    Ah, Ph = Zh @ W_bio, Zh @ W_perp

    ss_bio, ss_perp = _between_site_ss(A, y_site), _between_site_ss(P, y_site)
    ss_bio_h, ss_perp_h = _between_site_ss(Ah, y_site), _between_site_ss(Ph, y_site)
    G, N = np.unique(y_site).size, len(y_site)

    out = {
        "k_protected": int(harmonizer.k_),
        "protected_names": list(harmonizer.protected_names_),
        "latent_dim": int(Z.shape[1]),
        "between_site_ss": {
            "protected_pre": ss_bio, "complement_pre": ss_perp,
            "protected_share_pre": ss_bio / max(ss_bio + ss_perp, 1e-12),
            "protected_post": ss_bio_h, "complement_post": ss_perp_h,
            "note": "expected share under a site-agnostic axis = k/d = %.3f" % (harmonizer.k_ / Z.shape[1]),
        },
        "eta2_null_expectation": eta_squared_null_expectation(N, G),
        "per_dim_site_eta2_pre": compute_eta_squared_per_dim(Z, y_site).tolist(),
        "per_dim_site_eta2_post": compute_eta_squared_per_dim(Zh, y_site).tolist(),
        "axes": [],
    }
    for j in range(harmonizer.k_):
        a_pre, a_post = A[:, j], Ah[:, j]
        out["axes"].append({
            "name": harmonizer.protected_names_[j] if j < len(harmonizer.protected_names_) else f"axis{j}",
            "site_eta2_pre": compute_eta_squared(a_pre, y_site),
            "site_eta2_post": compute_eta_squared(a_post, y_site),
            "site_eps2_pre": compute_epsilon_squared(a_pre, y_site),
            "site_eps2_post": compute_epsilon_squared(a_post, y_site),
            "diag_d_pre": _cohens_d(a_pre[labelled], y_diag[labelled]),
            "diag_d_post": _cohens_d(a_post[labelled], y_diag[labelled]),
            "diag_eta2_pre": compute_eta_squared(a_pre[labelled], y_diag[labelled]),
            "diag_eta2_post": compute_eta_squared(a_post[labelled], y_diag[labelled]),
            "per_site_mean_pre": {int(s): float(a_pre[y_site == s].mean()) for s in np.unique(y_site)},
            "per_site_mean_post": {int(s): float(a_post[y_site == s].mean()) for s in np.unique(y_site)},
        })
    # site effect on the complement, summarised by the mean per-dim eta^2
    out["complement_mean_eta2_pre"] = float(np.mean(compute_eta_squared_per_dim(P, y_site)))
    out["complement_mean_eta2_post"] = float(np.mean(compute_eta_squared_per_dim(Ph, y_site)))
    return out
