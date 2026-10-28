"""
metrics.py -- Distribution-shift metrics for multi-site harmonization.

Kernel MMD (median-heuristic bandwidth, biased / unbiased estimators,
bootstrap CIs, bandwidth sweep), one-way ICC, eta-squared site variance,
per-dimension Gaussian KL, sliced W2, variance diagnostics and a
site/biology Pareto helper.

Statistical hypothesis tests live in :mod:`vae_harmonization.stats`; the
two most-used names are re-exported here for backward compatibility.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _sps
from scipy.spatial.distance import cdist, pdist

from .stats import paired_test, pairwise_significance_matrix  # noqa: F401

# Kernel values below this level mean the RBF kernel has effectively vanished
# between the two samples: the biased MMD^2 then degenerates to 1/n + 1/m
# irrespective of the data.  This is exactly the failure mode of using
# gamma = 1/d on 64-D latents, which an earlier version of this code did.
KERNEL_FLOOR = 1e-6


# -- Bandwidth selection -------------------------------------------------------

def median_heuristic_bandwidth(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    max_samples: int = 2000,
    seed: int = 0,
) -> float:
    """
    Median pairwise Euclidean distance of the pooled sample.

    Returns ``sigma`` such that the RBF kernel is
    ``k(x, y) = exp(-||x - y||^2 / (2 sigma^2))``, i.e. ``gamma = 1 / (2 sigma^2)``.
    Following Gretton et al. (2012) the median is computed on the pooled
    sample so that the same bandwidth can be shared across conditions.
    """
    Z = np.asarray(X, dtype=float) if Y is None else np.vstack([X, Y]).astype(float)
    if Z.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        Z = Z[rng.choice(Z.shape[0], size=max_samples, replace=False)]
    d = pdist(Z, metric="euclidean")
    d = d[d > 0]
    if d.size == 0:
        return 1.0
    return float(np.median(d))


def gamma_from_sigma(sigma: float) -> float:
    return 1.0 / (2.0 * sigma * sigma)


# -- MMD -----------------------------------------------------------------------

def _rbf(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    D2 = cdist(A, B, metric="sqeuclidean")
    return np.exp(-gamma * D2)


def _mmd2_from_blocks(K_XX: np.ndarray, K_YY: np.ndarray, K_XY: np.ndarray, unbiased: bool = True) -> float:
    n, m = K_XX.shape[0], K_YY.shape[0]
    if unbiased:
        sum_xx = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1))
        sum_yy = (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1))
    else:
        sum_xx = K_XX.sum() / (n * n)
        sum_yy = K_YY.sum() / (m * m)
    return float(sum_xx + sum_yy - 2.0 * K_XY.mean())


def compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "rbf",
    gamma: float | None = None,
    unbiased: bool = True,
    return_details: bool = False,
):
    """
    Squared Maximum Mean Discrepancy between two samples.

    Parameters
    ----------
    X, Y      : (n, d), (m, d) samples
    kernel    : 'rbf' or 'linear'
    gamma     : RBF precision.  ``None`` -> median heuristic on the pooled
                sample.  To compare *conditions* (pre vs post harmonization)
                pass the same ``gamma`` to every call; see
                :func:`compute_site_pairwise_mmd`.
    unbiased  : U-statistic (Gretton et al. 2012, Eq. 3) when True; the
                biased V-statistic otherwise.  The unbiased estimate can be
                slightly negative under H0.
    return_details : also return a dict with the bandwidth used and an
                underflow flag.

    Returns
    -------
    float (MMD^2) or (float, dict)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        val = float("nan")
        return (val, {"gamma": gamma, "kernel_floor": True}) if return_details else val

    if kernel == "rbf":
        if gamma is None:
            gamma = gamma_from_sigma(median_heuristic_bandwidth(X, Y))
        K_XX, K_YY, K_XY = _rbf(X, X, gamma), _rbf(Y, Y, gamma), _rbf(X, Y, gamma)
    elif kernel == "linear":
        K_XX, K_YY, K_XY = X @ X.T, Y @ Y.T, X @ Y.T
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    mmd2 = _mmd2_from_blocks(K_XX, K_YY, K_XY, unbiased)

    if not return_details:
        return mmd2
    details = {
        "gamma": float(gamma) if gamma is not None else None,
        "sigma": float(np.sqrt(0.5 / gamma)) if (kernel == "rbf" and gamma) else None,
        "kernel_floor": bool(kernel == "rbf" and K_XY.mean() < KERNEL_FLOOR),
        "mean_cross_kernel": float(K_XY.mean()),
        "unbiased": bool(unbiased),
        "n": int(n),
        "m": int(m),
    }
    return mmd2, details


def compute_mmd_ci(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float,
    n_boot: int = 200,
    alpha: float = 0.05,
    unbiased: bool = True,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI for MMD^2 (resampling each sample with replacement)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    point = compute_mmd(X, Y, gamma=gamma, unbiased=unbiased)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        xi = rng.integers(0, X.shape[0], X.shape[0])
        yi = rng.integers(0, Y.shape[0], Y.shape[0])
        boots[b] = compute_mmd(X[xi], Y[yi], gamma=gamma, unbiased=unbiased)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mmd2": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "n_boot": int(n_boot), "gamma": float(gamma)}


def compute_site_pairwise_mmd(
    X: np.ndarray,
    y_site: np.ndarray,
    kernel: str = "rbf",
    gamma: float | None = None,
    unbiased: bool = True,
    min_site_size: int = 5,
    n_boot: int = 0,
    seed: int = 0,
) -> dict:
    """
    MMD^2 between every pair of sites with one shared RBF bandwidth.

    ``gamma`` should be fixed once (median heuristic on the *pre-harmonization*
    matrix) and reused for every harmonized condition so that MMD values are
    comparable across conditions.  When ``gamma`` is None the median
    heuristic is applied to ``X`` itself and reported in ``gamma_used``.

    Sites with fewer than ``min_site_size`` subjects are excluded from the
    pairwise average (their MMD estimates are too noisy to interpret) and
    listed under ``excluded_sites``.
    """
    X = np.asarray(X, dtype=float)
    y_site = np.asarray(y_site)
    sites, counts = np.unique(y_site, return_counts=True)
    keep = sites[counts >= min_site_size]
    excluded = [int(s) for s in sites[counts < min_site_size]]

    if kernel == "rbf" and gamma is None:
        gamma = gamma_from_sigma(median_heuristic_bandwidth(X, seed=seed))

    pair: dict[str, float] = {}
    pair_ci: dict[str, dict] = {}
    vals: list[float] = []
    floor_hits = 0
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            si, sj = keep[i], keep[j]
            Xi, Xj = X[y_site == si], X[y_site == sj]
            v, det = compute_mmd(Xi, Xj, kernel=kernel, gamma=gamma,
                                 unbiased=unbiased, return_details=True)
            key = f"site_{int(si)}_vs_site_{int(sj)}"
            pair[key] = v
            vals.append(v)
            floor_hits += int(det["kernel_floor"])
            if n_boot > 0:
                pair_ci[key] = compute_mmd_ci(Xi, Xj, gamma=gamma, n_boot=n_boot,
                                              unbiased=unbiased, seed=seed)

    out: dict = dict(pair)
    out["mean_pairwise_mmd"] = float(np.mean(vals)) if vals else float("nan")
    out["median_pairwise_mmd"] = float(np.median(vals)) if vals else float("nan")
    out["gamma_used"] = float(gamma) if gamma is not None else None
    out["sigma_used"] = float(np.sqrt(0.5 / gamma)) if gamma else None
    out["unbiased"] = bool(unbiased)
    out["n_pairs"] = len(vals)
    out["kernel_floor_pairs"] = int(floor_hits)
    out["excluded_sites"] = excluded
    if n_boot > 0:
        out["pairwise_ci"] = pair_ci
    return out


def mmd_bandwidth_sweep(
    conditions: dict[str, np.ndarray],
    y_site: np.ndarray,
    gamma_ref: float,
    factors: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
    unbiased: bool = True,
    min_site_size: int = 5,
) -> dict:
    """
    Mean pairwise MMD^2 for each condition at several multiples of the
    reference bandwidth ``sigma_ref`` (``gamma = gamma_ref / factor^2``).
    Shows that the ranking of conditions is not an artefact of one bandwidth.
    """
    sweep: dict[str, dict[str, float]] = {}
    for f in factors:
        g = gamma_ref / (f * f)
        row = {}
        for name, Xc in conditions.items():
            row[name] = compute_site_pairwise_mmd(
                Xc, y_site, gamma=g, unbiased=unbiased, min_site_size=min_site_size,
            )["mean_pairwise_mmd"]
        sweep[f"sigma_x{f:g}"] = row
    return {"gamma_ref": float(gamma_ref), "factors": list(factors), "mean_pairwise_mmd": sweep}


# -- Variance decomposition ----------------------------------------------------

def compute_eta_squared(x: np.ndarray, groups: np.ndarray) -> float:
    """
    eta^2 = SS_between / SS_total for a 1-D variable, i.e. the fraction of
    variance in ``x`` explained by group (site) membership.
    """
    x = np.asarray(x, dtype=float).ravel()
    groups = np.asarray(groups)
    grand = x.mean()
    ss_tot = float(((x - grand) ** 2).sum())
    if ss_tot <= 1e-300:
        return 0.0
    ss_b = 0.0
    for g in np.unique(groups):
        m = groups == g
        ss_b += m.sum() * (x[m].mean() - grand) ** 2
    return float(ss_b / ss_tot)


def compute_epsilon_squared(x: np.ndarray, groups: np.ndarray) -> float:
    """
    Bias-corrected effect size epsilon^2 = (SS_between - (G-1) MS_within) / SS_total.

    Under the null of no group effect E[eta^2] ~ (G-1)/(N-1), which is far from
    zero when many small sites fall into one CV fold; epsilon^2 is centred on
    zero under the null (Kelley 1935; Okada 2013) and can be slightly negative.
    """
    x = np.asarray(x, dtype=float).ravel()
    groups = np.asarray(groups)
    N = x.size
    uniq = np.unique(groups)
    G = uniq.size
    if N - G < 1:
        return float("nan")
    grand = x.mean()
    ss_tot = float(((x - grand) ** 2).sum())
    if ss_tot <= 1e-300:
        return 0.0
    ss_b = sum((groups == g).sum() * (x[groups == g].mean() - grand) ** 2 for g in uniq)
    ms_w = (ss_tot - ss_b) / (N - G)
    return float((ss_b - (G - 1) * ms_w) / ss_tot)


def eta_squared_null_expectation(n: int, n_groups: int) -> float:
    """E[eta^2] under H0 for a one-way layout: (G-1)/(N-1)."""
    return float((n_groups - 1) / max(n - 1, 1))


def compute_eta_squared_per_dim(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.array([compute_eta_squared(X[:, j], groups) for j in range(X.shape[1])])


def compute_icc(X: np.ndarray, y_group: np.ndarray) -> dict:
    """
    One-way random-effects ICC(1) per feature dimension::

        ICC = (MS_between - MS_within) / (MS_between + (k - 1) MS_within)

    with ``k`` the harmonic-mean group size, clipped to [0, 1].
    """
    X = np.asarray(X, dtype=float)
    y_group = np.asarray(y_group)
    N, D = X.shape
    groups = np.unique(y_group)
    G = len(groups)
    if G < 2 or N - G < 1:
        return {"icc_mean": float("nan"), "icc_per_dim": np.full(D, np.nan)}

    grand = X.mean(axis=0)
    ss_b = np.zeros(D)
    ss_w = np.zeros(D)
    n_g = np.zeros(G)
    for i, g in enumerate(groups):
        m = y_group == g
        n_g[i] = m.sum()
        gm = X[m].mean(axis=0)
        ss_b += n_g[i] * (gm - grand) ** 2
        ss_w += ((X[m] - gm) ** 2).sum(axis=0)
    ms_b = ss_b / (G - 1)
    ms_w = ss_w / (N - G)
    k = G / np.sum(1.0 / n_g)
    denom = ms_b + (k - 1) * ms_w
    icc = np.where(denom > 1e-300, (ms_b - ms_w) / np.maximum(denom, 1e-300), 0.0)
    icc = np.clip(icc, 0.0, 1.0)
    return {"icc_mean": float(icc.mean()), "icc_per_dim": icc}


# -- Other divergences ---------------------------------------------------------

def compute_wasserstein2(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean of per-dimension 1-D Wasserstein distances (a sliced surrogate)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    return float(np.mean([_sps.wasserstein_distance(X[:, j], Y[:, j]) for j in range(X.shape[1])]))


def compute_per_dim_kl(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """KL( N(mu_x, s_x^2) || N(mu_y, s_y^2) ) per dimension."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    mu_x, var_x = X.mean(axis=0), X.var(axis=0) + 1e-10
    mu_y, var_y = Y.mean(axis=0), Y.var(axis=0) + 1e-10
    return 0.5 * (np.log(var_y / var_x) + (var_x + (mu_x - mu_y) ** 2) / var_y - 1.0)


# -- Protocol-matched permutation null -----------------------------------------

def _mmd2_from_K(K: np.ndarray, a: np.ndarray, b: np.ndarray, unbiased: bool = True) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    return _mmd2_from_blocks(K[np.ix_(a, a)], K[np.ix_(b, b)], K[np.ix_(a, b)], unbiased)


def _mean_pairwise_mmd_from_K(K: np.ndarray, y_site: np.ndarray, min_site_size: int = 5) -> float:
    sites, counts = np.unique(y_site, return_counts=True)
    keep = sites[counts >= min_site_size]
    idx = {int(s): np.flatnonzero(y_site == s) for s in keep}
    vals = []
    for i, si in enumerate(keep):
        for sj in keep[i + 1:]:
            vals.append(_mmd2_from_K(K, idx[int(si)], idx[int(sj)]))
    return float(np.mean(vals)) if vals else float("nan")


def permutation_null_oos(
    X: np.ndarray,
    y_site: np.ndarray,
    *,
    gamma: float,
    min_site_size: int = 5,
    n_perm: int = 200,
    seed: int = 42,
) -> dict:
    """
    Null distribution of ICC and pairwise MMD^2 obtained by permuting site
    labels on a *fixed* feature matrix (the pooled held-out matrix after
    harmonization).

    Mean-matching methods pin site means, so the unbiased MMD^2 of the
    harmonized matrix is systematically negative even when the sites already
    share a distribution.  Comparing the observed value to this permutation
    null (same matrix, shuffled labels) is the honest test of residual site
    structure.  Returns the observed values, the null mean/sd, a z-score and
    the one-sided percentile (fraction of permutations as or more extreme
    toward *larger* site structure).
    """
    X = np.asarray(X, dtype=float)
    y_site = np.asarray(y_site).astype(int)
    obs_icc = compute_icc(X, y_site)["icc_mean"]
    K = _rbf(X, X, float(gamma))
    obs_mmd = _mean_pairwise_mmd_from_K(K, y_site, min_site_size=min_site_size)
    rng = np.random.default_rng(seed)
    null_icc, null_mmd = [], []
    for _ in range(int(n_perm)):
        yp = rng.permutation(y_site)
        null_icc.append(compute_icc(X, yp)["icc_mean"])
        null_mmd.append(_mean_pairwise_mmd_from_K(K, yp, min_site_size=min_site_size))
    null_icc = np.asarray(null_icc, float)
    null_mmd = np.asarray(null_mmd, float)

    def _pack(obs, null, larger_is_more_structure):
        mu, sd = float(null.mean()), float(null.std(ddof=1)) if null.size > 1 else float("nan")
        z = (obs - mu) / sd if sd and sd == sd and sd > 1e-12 else float("nan")
        if larger_is_more_structure:
            pct = float(np.mean(null >= obs - 1e-15))
        else:
            pct = float(np.mean(null <= obs + 1e-15))
        return {"observed": float(obs), "null_mean": mu, "null_sd": sd, "z": float(z),
                "percentile": pct, "n_perm": int(null.size)}

    return {
        "icc": _pack(obs_icc, null_icc, True),
        "mmd2_unbiased": _pack(obs_mmd, null_mmd, True),
    }


# -- Pareto --------------------------------------------------------------------

def site_biology_pareto(methods_results: dict[str, tuple[float, float]]) -> list[dict]:
    """Pareto set for (site_acc lower-is-better, diag_acc higher-is-better)."""
    rows = [{"method": k, "site_acc": sa, "diag_acc": da} for k, (sa, da) in methods_results.items()]
    for r in rows:
        r["pareto_optimal"] = True
        for o in rows:
            if o is r:
                continue
            if (o["site_acc"] <= r["site_acc"] and o["diag_acc"] >= r["diag_acc"]
                    and (o["site_acc"] < r["site_acc"] or o["diag_acc"] > r["diag_acc"])):
                r["pareto_optimal"] = False
                break
    return rows


# -- Variance diagnostics ------------------------------------------------------

def compute_variance_stats(X: np.ndarray, y_site: np.ndarray) -> dict:
    X = np.asarray(X, dtype=float)
    y_site = np.asarray(y_site)
    per_site, means = {}, []
    for s in np.unique(y_site):
        Xs = X[y_site == s]
        per_site[int(s)] = float(Xs.var(axis=0).mean())
        means.append(Xs.mean(axis=0))
    pooled = float(X.var(axis=0).mean())
    between = float(np.stack(means).var(axis=0).mean())
    return {
        "per_site_variance": per_site,
        "pooled_variance": pooled,
        "between_site_variance": between,
        "variance_ratio": between / pooled if pooled > 1e-12 else 0.0,
    }


def compare_variance_before_after(X_before, X_after, y_site) -> dict:
    b = compute_variance_stats(X_before, y_site)
    a = compute_variance_stats(X_after, y_site)
    return {
        "before": b,
        "after": a,
        "pooled_variance_ratio": a["pooled_variance"] / b["pooled_variance"]
        if b["pooled_variance"] > 1e-12 else float("nan"),
        "between_variance_ratio": a["between_site_variance"] / b["between_site_variance"]
        if b["between_site_variance"] > 1e-12 else float("nan"),
    }
