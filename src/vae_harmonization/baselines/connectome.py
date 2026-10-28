"""
connectome.py -- Rebuild SPD correlation matrices from Fisher-z upper-triangle vectors.

The stored FC of every cohort is the upper triangle of a Fisher-z correlation
matrix (diagonal omitted).  Riemannian / tangent-space baselines need the
square SPD matrices; this module is the shared conversion, with a tiny ridge
so that a rank-deficient empirical correlation stays positive definite.
"""

from __future__ import annotations

import numpy as np


def n_rois_from_n_edges(n_edges: int) -> int:
    disc = 1 + 8 * int(n_edges)
    n = int(round((1 + np.sqrt(disc)) / 2.0))
    if n * (n - 1) // 2 != int(n_edges):
        raise ValueError(f"feature count {n_edges} is not an upper triangle of a square matrix")
    return n


def vec_to_corr(z: np.ndarray, n_rois: int | None = None) -> np.ndarray:
    """One Fisher-z vector -> correlation matrix with unit diagonal."""
    z = np.asarray(z, dtype=np.float64)
    n = n_rois if n_rois is not None else n_rois_from_n_edges(z.size)
    C = np.eye(n)
    iu = np.triu_indices(n, 1)
    r = np.tanh(np.clip(z, -20.0, 20.0))
    C[iu] = r
    C[(iu[1], iu[0])] = r
    return C


def corr_to_vec(C: np.ndarray) -> np.ndarray:
    """Correlation matrix -> Fisher-z upper triangle."""
    C = np.asarray(C, dtype=np.float64)
    iu = np.triu_indices(C.shape[0], 1)
    r = np.clip(C[iu], -0.9999, 0.9999)
    return np.arctanh(r)


def vecs_to_corrs(Z: np.ndarray, n_rois: int | None = None) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    n = n_rois if n_rois is not None else n_rois_from_n_edges(Z.shape[1])
    out = np.empty((Z.shape[0], n, n), dtype=np.float64)
    for i in range(Z.shape[0]):
        out[i] = vec_to_corr(Z[i], n)
    return out


def corrs_to_vecs(Cs: np.ndarray) -> np.ndarray:
    return np.stack([corr_to_vec(C) for C in Cs])


def _eigh_sym(C: np.ndarray):
    return np.linalg.eigh(0.5 * (C + C.T))


def spd_floor(C: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    w, V = _eigh_sym(C)
    return (V * np.clip(w, floor, None)) @ V.T


def sqrtm_spd(C: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    w, V = _eigh_sym(C)
    return (V * np.sqrt(np.clip(w, floor, None))) @ V.T


def invsqrtm_spd(C: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    w, V = _eigh_sym(C)
    return (V / np.sqrt(np.clip(w, floor, None))) @ V.T


def logm_spd(C: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    w, V = _eigh_sym(C)
    return (V * np.log(np.clip(w, floor, None))) @ V.T


def expm_spd(S: np.ndarray) -> np.ndarray:
    w, V = _eigh_sym(S)
    return (V * np.exp(w)) @ V.T


def log_euclidean_mean(covs: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Log-Euclidean Frechet mean of a stack of SPD matrices."""
    acc = np.zeros(covs.shape[1:], dtype=np.float64)
    for C in covs:
        acc += logm_spd(spd_floor(C, floor))
    return expm_spd(acc / max(len(covs), 1))


def vech(S: np.ndarray) -> np.ndarray:
    """Independent entries of a symmetric matrix (upper triangle including the diagonal)."""
    return S[np.triu_indices(S.shape[0], 0)]


def unvech(v: np.ndarray, n: int) -> np.ndarray:
    S = np.zeros((n, n), dtype=np.float64)
    iu = np.triu_indices(n, 0)
    S[iu] = v
    S[(iu[1], iu[0])] = v
    return S
