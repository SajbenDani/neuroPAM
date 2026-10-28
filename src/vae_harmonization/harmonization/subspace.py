"""
subspace.py -- Fitting the protected (label-defined) subspace.

The protected subspace ``span(W_bio)`` collects the directions of the latent
space along which supervised signal lives (diagnosis, symptom severity,
demographic covariates).  Harmonization only mean-shifts these coordinates
and applies the covariance-matching transport on the orthogonal complement.

All fitting functions accept *training-fold* data only; the caller is
responsible for never passing held-out subjects (see evaluation/protocol.py).
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("degenerate (zero-norm) axis")
    return v / n


# -- Individual axes -----------------------------------------------------------

def fit_diagnosis_axis(
    Z: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    seed: int = 42,
    max_iter: int = 20_000,
) -> np.ndarray:
    """
    Unit normal of a linear SVM separating ``y==0`` from ``y==1``, oriented so
    that it points from the class-0 mean toward the class-1 mean.  Samples with
    ``y < 0`` are treated as unlabelled and ignored (leave-one-site-out
    harmonization fits site statistics on unlabelled held-out sites).
    """
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y).astype(int)
    labelled = y >= 0
    Z, y = Z[labelled], y[labelled]
    if np.unique(y).size < 2:
        raise ValueError("fit_diagnosis_axis needs both classes in the training data")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        svm = LinearSVC(C=C, max_iter=max_iter, random_state=seed)
        svm.fit(Z, y)
    w = _unit(svm.coef_[0])
    gap = Z[y == 1].mean(axis=0) - Z[y == 0].mean(axis=0)
    if np.dot(w, gap) < 0:
        w = -w
    return w


def fit_severity_axis(
    Z: np.ndarray,
    scores: np.ndarray,
    min_valid: int = 20,
    seed: int = 42,
) -> np.ndarray | None:
    """
    First PLS component predicting a continuous score (NaNs ignored).
    Returns ``None`` when fewer than ``min_valid`` scores are available.
    """
    Z = np.asarray(Z, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    ok = np.isfinite(scores)
    if ok.sum() < min_valid or np.std(scores[ok]) < 1e-12:
        return None
    pls = PLSRegression(n_components=1, scale=True)
    pls.fit(Z[ok], scores[ok])
    w = _unit(pls.x_weights_[:, 0])
    # orient toward increasing score
    if np.corrcoef(Z[ok] @ w, scores[ok])[0, 1] < 0:
        w = -w
    return w


def fit_covariate_axis(
    Z: np.ndarray,
    cov: np.ndarray,
    kind: str = "auto",
    alpha: float = 1.0,
    seed: int = 42,
    min_valid: int = 20,
) -> np.ndarray | None:
    """
    Direction most associated with a covariate.

    ``kind='binary'``     -> linear-SVM normal (e.g. sex)
    ``kind='continuous'`` -> ridge-regression coefficient direction (e.g. age)
    ``kind='auto'``       -> binary when the covariate has exactly two values
    """
    Z = np.asarray(Z, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    ok = np.isfinite(cov)
    if ok.sum() < min_valid:
        return None
    vals = np.unique(cov[ok])
    if kind == "auto":
        kind = "binary" if vals.size == 2 else "continuous"
    if kind == "binary":
        if vals.size != 2:
            return None
        yb = (cov[ok] == vals.max()).astype(int)
        return fit_diagnosis_axis(Z[ok], yb, C=1.0, seed=seed)
    if np.std(cov[ok]) < 1e-12:
        return None
    Zc = Z[ok] - Z[ok].mean(axis=0)
    yc = cov[ok] - cov[ok].mean()
    reg = Ridge(alpha=alpha, fit_intercept=False).fit(Zc, yc)
    return _unit(reg.coef_)


# -- Combining axes ------------------------------------------------------------

def orthonormal_protected_basis(
    axes: list[np.ndarray],
    collinearity_threshold: float = 0.8,
) -> tuple[np.ndarray, dict]:
    """
    Gram-Schmidt (QR) the given unit axes into an orthonormal ``W_bio`` (d, k).

    Axes whose absolute cosine with the span of the already-accepted axes
    exceeds ``collinearity_threshold`` are dropped (they carry no new
    protected direction).  Returns ``(W_bio, info)`` with the kept indices.
    """
    kept: list[np.ndarray] = []
    kept_idx: list[int] = []
    dropped: list[int] = []
    for i, a in enumerate(axes):
        if a is None:
            dropped.append(i)
            continue
        a = _unit(a)
        if kept:
            Q = np.stack(kept, axis=1)
            proj = Q @ (Q.T @ a)
            cos_to_span = np.linalg.norm(proj)
            if cos_to_span > collinearity_threshold:
                dropped.append(i)
                continue
            a = a - proj
            a = _unit(a)
        kept.append(a)
        kept_idx.append(i)
    if not kept:
        d = len(next(x for x in axes if x is not None)) if any(x is not None for x in axes) else 0
        W = np.zeros((d, 0))
    else:
        W = np.stack(kept, axis=1)
    return W, {"k": W.shape[1], "kept": kept_idx, "dropped": dropped}


def complement_basis(W_bio: np.ndarray, d: int | None = None) -> np.ndarray:
    """Orthonormal basis of the orthogonal complement of ``span(W_bio)``."""
    if W_bio.shape[1] == 0:
        return np.eye(W_bio.shape[0] if d is None else d)
    return scipy.linalg.null_space(W_bio.T)


def decompose(Z: np.ndarray, W_bio: np.ndarray, W_perp: np.ndarray):
    """Protected coordinates ``(N, k)`` and complement coordinates ``(N, d-k)``."""
    return Z @ W_bio, Z @ W_perp


def recompose(A: np.ndarray, P: np.ndarray, W_bio: np.ndarray, W_perp: np.ndarray) -> np.ndarray:
    return A @ W_bio.T + P @ W_perp.T
