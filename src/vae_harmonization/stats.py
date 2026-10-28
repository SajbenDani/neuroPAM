"""
stats.py -- Paired hypothesis tests, effect sizes, equivalence tests,
confidence intervals, chance levels and classification summaries.

Conventions
-----------
* Fold-level scores are treated as *paired* observations: the same CV split
  is evaluated under every method, so effect sizes are Cohen's d_z
  (mean of paired differences / SD of paired differences), not the
  independent-samples pooled-SD version.
* Wilcoxon signed-rank is the default test (10 folds, no normality
  assumption).  ``method='ttest'`` gives the paired t-test.
* Equivalence is assessed with TOST (two one-sided t-tests) on the paired
  differences against a symmetric margin.
* Multiple comparisons use Holm's step-down procedure.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
)


# -- Effect sizes and intervals ------------------------------------------------

def paired_cohens_dz(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d_z for paired samples: mean(a-b) / sd(a-b, ddof=1)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if d.size < 2:
        return float("nan")
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 1e-12 else (0.0 if abs(d.mean()) < 1e-12 else float("inf"))


def mean_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """t-based (1-alpha) CI of the mean. Returns (mean, low, high)."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    n = v.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = v.mean()
    if n == 1:
        return float(m), float(m), float(m)
    se = v.std(ddof=1) / np.sqrt(n)
    h = sps.t.ppf(1 - alpha / 2, n - 1) * se
    return float(m), float(m - h), float(m + h)


def bootstrap_ci(
    values: np.ndarray,
    stat=np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of ``stat`` over 1-D ``values``."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([stat(v[rng.integers(0, v.size, v.size)]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(v)), float(lo), float(hi)


def summarize_folds(values, alpha: float = 0.05) -> dict:
    """mean / std / n / t-CI for a list of fold scores (NaNs dropped)."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    m, lo, hi = mean_ci(v, alpha)
    return {
        "mean": m,
        "std": float(v.std(ddof=0)) if v.size else float("nan"),
        "n": int(v.size),
        "ci_low": lo,
        "ci_high": hi,
    }


# -- Paired tests --------------------------------------------------------------

def paired_test(a, b, method: str = "wilcoxon", alpha: float = 0.05) -> dict:
    """
    Paired comparison of fold-level scores ``a`` vs ``b``.

    Returns statistic, two-sided p-value, Cohen's d_z, mean difference with a
    t-CI, and significance flags.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = a - b
    n = d.size
    dz = paired_cohens_dz(a, b)
    m, lo, hi = mean_ci(d, alpha)

    if n < 2:
        stat, p = float("nan"), float("nan")
    elif method == "wilcoxon":
        if np.all(d == 0):
            stat, p = 0.0, 1.0
        else:
            try:
                res = sps.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
                stat, p = float(res.statistic), float(res.pvalue)
            except ValueError:
                stat, p = 0.0, 1.0
    elif method == "ttest":
        res = sps.ttest_rel(a, b)
        stat, p = float(res.statistic), float(res.pvalue)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'wilcoxon' or 'ttest'.")

    return {
        "method": method,
        "n": int(n),
        "statistic": stat,
        "p_value": p,
        "effect_size_dz": dz,
        # legacy key kept for older table generators
        "effect_size_cohens_d": dz,
        "mean_diff": m,
        "mean_diff_ci_low": lo,
        "mean_diff_ci_high": hi,
        "significant_005": bool(p < 0.05) if np.isfinite(p) else False,
        "significant_001": bool(p < 0.01) if np.isfinite(p) else False,
    }


def tost_paired(a, b, margin: float, alpha: float = 0.05, variance_correction: float | None = None) -> dict:
    """
    Two one-sided tests (TOST) for equivalence of paired scores.

    H0: |mean(a-b)| >= margin.  Equivalence is declared when both one-sided
    tests reject at ``alpha`` (equivalently the (1-2 alpha) CI of the mean
    difference lies inside (-margin, +margin)).

    For repeated cross-validation pass the Nadeau-Bengio factor
    ``1/n + n_test/n_train`` as ``variance_correction``. The naive ``1/n``
    ignores the overlap between training folds and makes equivalence too easy.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    n = d.size
    if n < 2:
        return {"margin": margin, "n": n, "p_tost": float("nan"), "equivalent": False}
    m = d.mean()
    factor = 1.0 / n if variance_correction is None else float(variance_correction)
    se = float(np.sqrt(factor * d.var(ddof=1)))
    if se < 1e-12:
        p_lower = 0.0 if m > -margin else 1.0
        p_upper = 0.0 if m < margin else 1.0
    else:
        t_lower = (m + margin) / se          # H0: m <= -margin
        t_upper = (m - margin) / se          # H0: m >=  margin
        p_lower = float(1.0 - sps.t.cdf(t_lower, n - 1))
        p_upper = float(sps.t.cdf(t_upper, n - 1))
    p_tost = max(p_lower, p_upper)
    half = float(sps.t.ppf(1.0 - alpha, n - 1)) * se
    return {
        "margin": float(margin),
        "n": int(n),
        "mean_diff": float(m),
        "ci90_low": float(m - half),
        "ci90_high": float(m + half),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": float(p_tost),
        "equivalent": bool(p_tost < alpha),
        "variance_correction": factor,
    }


def nadeau_bengio_corrected_ttest(a, b, n_train: int, n_test: int, alpha: float = 0.05) -> dict:
    """
    Corrected resampled t-test of Nadeau & Bengio (2003) for repeated CV.

    The naive paired t-test underestimates variance because the training
    folds overlap.  The correction uses ``(1/n + n_test/n_train) s^2`` as
    the variance of the mean paired difference, with ``n`` the number of
    resampled scores (folds x repeats).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    n = int(d.size)
    if n < 2 or n_train <= 0:
        return {"n": n, "statistic": float("nan"), "p_value": float("nan"), "mean_diff": float("nan")}
    m = float(d.mean())
    s2 = float(d.var(ddof=1))
    corr = 1.0 / n + float(n_test) / float(n_train)
    se = float(np.sqrt(max(corr * s2, 0.0)))
    if se < 1e-15:
        t = 0.0 if abs(m) < 1e-15 else float("inf")
        p = 1.0 if abs(m) < 1e-15 else 0.0
    else:
        t = m / se
        p = float(2.0 * sps.t.sf(abs(t), n - 1))
    return {
        "method": "nadeau_bengio",
        "n": n,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "statistic": float(t),
        "p_value": p,
        "mean_diff": m,
        "variance_correction": float(corr),
        "significant_005": bool(p < alpha) if np.isfinite(p) else False,
    }


def holm_bonferroni(p_values) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p_values, float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (n - rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def pairwise_significance_matrix(
    all_fold_scores: dict[str, np.ndarray],
    metric_name: str = "accuracy",
    method: str = "wilcoxon",
    correction: str | None = "holm",
) -> dict:
    """
    Pairwise paired tests between all methods.

    Returns the method list, raw and (optionally) Holm-adjusted p-value
    matrices, a d_z matrix (row minus column) and mean-difference matrix.
    """
    names = list(all_fold_scores.keys())
    n = len(names)
    p_raw = np.ones((n, n))
    dz = np.zeros((n, n))
    md = np.zeros((n, n))
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            r = paired_test(all_fold_scores[names[i]], all_fold_scores[names[j]], method=method)
            p_raw[i, j] = p_raw[j, i] = r["p_value"]
            dz[i, j], dz[j, i] = r["effect_size_dz"], -r["effect_size_dz"]
            md[i, j], md[j, i] = r["mean_diff"], -r["mean_diff"]
            pairs.append((i, j))
    p_adj = p_raw.copy()
    if correction == "holm" and pairs:
        flat = np.array([p_raw[i, j] for i, j in pairs])
        finite = np.isfinite(flat)
        adj = flat.copy()
        adj[finite] = holm_bonferroni(flat[finite])
        for (i, j), v in zip(pairs, adj):
            p_adj[i, j] = p_adj[j, i] = v
    return {
        "methods": names,
        "metric": metric_name,
        "test": method,
        "correction": correction,
        "p_values": p_raw,
        "p_values_adjusted": p_adj,
        "effect_sizes": dz,
        "mean_diffs": md,
    }


# -- Chance levels and classification summaries --------------------------------

def majority_class_rate(y) -> float:
    """Accuracy of always predicting the most frequent class."""
    y = np.asarray(y)
    if y.size == 0:
        return float("nan")
    _, c = np.unique(y, return_counts=True)
    return float(c.max() / y.size)


def chance_levels(y) -> dict:
    """Both reference points for a K-class label vector."""
    y = np.asarray(y)
    k = int(np.unique(y).size)
    return {
        "n_classes": k,
        "uniform_chance": 1.0 / k if k else float("nan"),
        "majority_rate": majority_class_rate(y),
    }


def classification_summary(y_true, y_pred, y_score=None, task: str = "generic") -> dict:
    """
    Accuracy, balanced accuracy, macro-F1, and (binary) AUC or (multi-class)
    NMI between true and predicted labels.  ``task='site'`` adds NMI, which
    is the label-permutation-invariant residual-site-information measure.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "acc": float((y_true == y_pred).mean()),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    classes = np.unique(y_true)
    if y_score is not None and classes.size == 2:
        try:
            out["auc"] = float(roc_auc_score(y_true, np.asarray(y_score)))
        except ValueError:
            out["auc"] = float("nan")
    if task == "site" or classes.size > 2:
        out["nmi"] = float(normalized_mutual_info_score(y_true, y_pred))
    return out
