"""
transport.py -- Gaussian optimal-transport primitives for covariance matching.

All maps here act on zero-mean coordinates.  For source covariance ``C_s``
and target covariance ``C_t`` (both SPD):

* :func:`bures_wasserstein_map`  -- the unique symmetric PD ``B`` with
  ``B C_s B = C_t``; it is the Brenier map between the two centred Gaussians
  and the minimum-displacement linear map among all ``A`` with
  ``A C_s A^T = C_t`` (Theorem 3.1 of the paper).
* :func:`cholesky_map`           -- ``L_t L_s^{-1}``; also satisfies the
  covariance constraint but is not symmetric and not displacement-minimal.
* :func:`zca_map`                -- ``C_t^{1/2} C_s^{-1/2}`` (whiten-then-colour);
  coincides with the BW map only when ``C_s`` and ``C_t`` commute.
* :func:`geodesic_map`           -- ``(1 - lam) I + lam B``, the McCann
  interpolant between identity and the BW map; ``lam`` in [0, 1] trades
  covariance alignment against displacement.
* :func:`bures_wasserstein_barycenter` -- fixed-point iteration for the
  Wasserstein barycenter of several centred Gaussians.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from sklearn.covariance import LedoitWolf


# -- Covariance estimation -----------------------------------------------------

COVARIANCE_METHODS = ("lw", "empirical", "diagonal", "pooled")


def shrinkage_intensity(X: np.ndarray, target: np.ndarray) -> float:
    """
    Analytic shrinkage intensity ``delta*`` for the estimator
    ``(1 - delta) S + delta T`` with ``S`` the sample covariance of the rows of
    ``X`` and ``T`` a fixed target (Ledoit & Wolf 2003; Schaefer & Strimmer 2005,
    eq. for a target that does not depend on ``S``)::

        delta* = sum_ij Var(s_ij) / sum_ij (s_ij - t_ij)^2 ,   clipped to [0, 1]

    where ``Var(s_ij) = n / (n-1)^3 * sum_k (w_kij - mean_k w_kij)^2`` and
    ``w_kij = (x_ki - xbar_i)(x_kj - xbar_j)``.  Entries whose target equals the
    sample value (e.g. the diagonal of a diagonal target) contribute nothing to
    either sum, so they are left untouched.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if n < 3:
        return 1.0
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / (n - 1)
    W = Xc[:, :, None] * Xc[:, None, :]                      # (n, d, d)
    var_s = n / (n - 1.0) ** 3 * np.sum((W - W.mean(axis=0)) ** 2, axis=0)
    diff2 = (S - np.asarray(target, dtype=np.float64)) ** 2
    denom = float(diff2.sum())
    if denom <= 0.0:
        return 1.0
    return float(np.clip(var_s.sum() / denom, 0.0, 1.0))


def shrink_to_target(X: np.ndarray, target: np.ndarray, ridge: float = 1e-6) -> tuple[np.ndarray, float]:
    """``(1 - delta*) S + delta* T`` with the analytic intensity of :func:`shrinkage_intensity`."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    T = np.asarray(target, dtype=np.float64)
    if n < 2:
        return 0.5 * (T + T.T) + ridge * np.eye(d), 1.0
    S = np.atleast_2d(np.cov(X, rowvar=False))
    delta = shrinkage_intensity(X, T)
    C = (1.0 - delta) * S + delta * T
    return 0.5 * (C + C.T) + ridge * np.eye(d), delta


def estimate_covariance(
    X: np.ndarray,
    method: str = "lw",
    ridge: float = 1e-6,
    target: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    Covariance of the rows of ``X`` plus a tiny ridge for numerical safety.

    ``method``
        'lw'        Ledoit-Wolf shrinkage toward the scaled identity ``(tr S / d) I``
                    (sklearn).  Flattens the spectrum, which is a poor target when the
                    coordinates have very different variances (e.g. PCA scores).
        'empirical' sample covariance.
        'diagonal'  shrinkage toward ``diag(S)`` (Schaefer & Strimmer target D): keeps
                    every variance, shrinks only the correlations.
        'pooled'    shrinkage toward the matrix passed as ``target`` (in the harmonizer:
                    the pooled within-site covariance of the *other* sites).  When the
                    sites share a covariance this target is the right one, and the
                    resulting transport map is close to the identity for small sites.

    Returns ``(C, shrinkage)`` where ``shrinkage`` is the intensity ``delta`` in
    [0, 1] with which the estimate was pulled toward its target (0 for the
    empirical estimator).
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if method == "lw":
        lw = LedoitWolf().fit(X)
        C, delta = lw.covariance_, float(lw.shrinkage_)
    elif method == "empirical":
        C = np.cov(X, rowvar=False) if n > 1 else np.zeros((d, d))
        C = np.atleast_2d(C)
        delta = 0.0
    elif method == "diagonal":
        S = np.atleast_2d(np.cov(X, rowvar=False)) if n > 1 else np.zeros((d, d))
        return shrink_to_target(X, np.diag(np.diag(S)), ridge)
    elif method == "pooled":
        if target is None:
            raise ValueError("method='pooled' needs the pooled target covariance")
        return shrink_to_target(X, target, ridge)
    else:
        raise ValueError(f"Unknown covariance method '{method}' (use one of {COVARIANCE_METHODS}).")
    C = 0.5 * (C + C.T) + ridge * np.eye(d)
    return C, delta


def pooled_within_covariance(X: np.ndarray, y_group: np.ndarray, exclude: int | None = None) -> np.ndarray | None:
    """
    Pooled within-group sample covariance ``sum_g (n_g - 1) S_g / sum_g (n_g - 1)`` over the
    groups with at least two members, optionally leaving one group out.  ``None`` when no
    group qualifies.  Only per-group means and scatter matrices enter, so the quantity can be
    assembled from per-site summaries.
    """
    X = np.asarray(X, dtype=np.float64)
    y_group = np.asarray(y_group)
    d = X.shape[1]
    scatter, dof = np.zeros((d, d)), 0
    for g in np.unique(y_group):
        if exclude is not None and g == exclude:
            continue
        m = y_group == g
        n_g = int(m.sum())
        if n_g < 2:
            continue
        Xc = X[m] - X[m].mean(axis=0)
        scatter += Xc.T @ Xc
        dof += n_g - 1
    if dof == 0:
        return None
    return scatter / dof


# -- Matrix functions ----------------------------------------------------------

def _eigh_psd(C: np.ndarray, floor: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    return np.clip(w, floor, None), V


def sqrtm_psd(C: np.ndarray) -> np.ndarray:
    w, V = _eigh_psd(C)
    return (V * np.sqrt(w)) @ V.T


def invsqrtm_psd(C: np.ndarray) -> np.ndarray:
    w, V = _eigh_psd(C)
    return (V / np.sqrt(w)) @ V.T


# -- Maps ----------------------------------------------------------------------

def bures_wasserstein_map(C_s: np.ndarray, C_t: np.ndarray) -> np.ndarray:
    """
    ``B = C_s^{-1/2} (C_s^{1/2} C_t C_s^{1/2})^{1/2} C_s^{-1/2}``.

    Symmetric positive definite; ``B C_s B = C_t``.
    """
    Cs_h = sqrtm_psd(C_s)
    Cs_ih = invsqrtm_psd(C_s)
    M = sqrtm_psd(Cs_h @ C_t @ Cs_h)
    B = Cs_ih @ M @ Cs_ih
    return 0.5 * (B + B.T)


def cholesky_map(C_s: np.ndarray, C_t: np.ndarray) -> np.ndarray:
    """``A = L_t L_s^{-1}`` with ``C = L L^T`` (lower Cholesky factors)."""
    L_s = np.linalg.cholesky(C_s)
    L_t = np.linalg.cholesky(C_t)
    return L_t @ np.linalg.inv(L_s)


def zca_map(C_s: np.ndarray, C_t: np.ndarray) -> np.ndarray:
    """``A = C_t^{1/2} C_s^{-1/2}`` (whitening followed by colouring)."""
    return sqrtm_psd(C_t) @ invsqrtm_psd(C_s)


def geodesic_map(B: np.ndarray, lam: float) -> np.ndarray:
    """McCann interpolant ``(1 - lam) I + lam B`` between identity and ``B``."""
    lam = float(np.clip(lam, 0.0, 1.0))
    return (1.0 - lam) * np.eye(B.shape[0]) + lam * B


# -- Distances and displacement ------------------------------------------------

def bures_wasserstein_distance2(C_1: np.ndarray, C_2: np.ndarray) -> float:
    """Squared Bures-Wasserstein distance ``tr C_1 + tr C_2 - 2 tr (C_1^{1/2} C_2 C_1^{1/2})^{1/2}``."""
    C1_h = sqrtm_psd(C_1)
    cross = sqrtm_psd(C1_h @ C_2 @ C1_h)
    return float(max(np.trace(C_1) + np.trace(C_2) - 2.0 * np.trace(cross), 0.0))


def gaussian_w2_distance2(mu_1, C_1, mu_2, C_2) -> float:
    """Squared 2-Wasserstein distance between two Gaussians."""
    return float(np.sum((np.asarray(mu_1) - np.asarray(mu_2)) ** 2)) + bures_wasserstein_distance2(C_1, C_2)


def expected_squared_displacement(A: np.ndarray, C_s: np.ndarray) -> float:
    """
    ``E ||A z - z||^2`` for ``z ~ N(0, C_s)``, i.e. ``tr((A - I) C_s (A - I)^T)``.
    Used to verify that the BW map minimises displacement among covariance-
    matching linear maps.
    """
    D = A - np.eye(A.shape[0])
    return float(np.trace(D @ C_s @ D.T))


# -- Barycenter ----------------------------------------------------------------

def bures_wasserstein_barycenter(
    covs: list[np.ndarray],
    weights: np.ndarray | None = None,
    n_iter: int = 100,
    tol: float = 1e-8,
    init: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Wasserstein barycenter of centred Gaussians ``N(0, C_i)`` with weights
    ``w_i`` (Agueh & Carlier 2011) via the fixed-point iteration of
    Alvarez-Esteban et al. (2016)::

        S_{k+1} = S_k^{-1/2} ( sum_i w_i (S_k^{1/2} C_i S_k^{1/2})^{1/2} )^2 S_k^{-1/2}

    Returns ``(S, info)`` with ``info['n_iter']`` and ``info['converged']``.
    """
    covs = [np.asarray(C, dtype=np.float64) for C in covs]
    m = len(covs)
    if weights is None:
        weights = np.full(m, 1.0 / m)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    S = np.asarray(init, dtype=np.float64) if init is not None else sum(w * C for w, C in zip(weights, covs))
    converged = False
    it = 0
    for it in range(1, n_iter + 1):
        S_h = sqrtm_psd(S)
        S_ih = invsqrtm_psd(S)
        T = sum(w * sqrtm_psd(S_h @ C @ S_h) for w, C in zip(weights, covs))
        S_new = S_ih @ (T @ T) @ S_ih
        S_new = 0.5 * (S_new + S_new.T)
        delta = np.linalg.norm(S_new - S, "fro") / max(np.linalg.norm(S, "fro"), 1e-12)
        S = S_new
        if delta < tol:
            converged = True
            break
    return S, {"n_iter": it, "converged": converged}
