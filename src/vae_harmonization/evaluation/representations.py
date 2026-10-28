"""
representations.py -- The linear (PCA) representation used by the main pipeline.

Every FC vector (Fisher-z upper triangle, ~20k edges) is standardized edge-wise and
projected on its first ``n_components`` principal components.  Inside the
leakage-free protocol the scaler and the PCA are fitted on the training fold only
(:func:`make_pca_representation_fn`), so no held-out subject touches any fitted
quantity, the representation included.

Because the principal axes differ slightly from fold to fold, harmonized held-out
scores of different folds cannot be stacked directly.  :meth:`PCARepresentation.embed`
maps fold scores back into the standardized edge space (``scores @ components + mean``).
This map is an isometry (the components are orthonormal), so pairwise distances,
and hence the RBF-MMD, are exactly those of the score space, while all folds now
live in one common space and can be pooled into a single out-of-sample matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCARepresentation:
    scaler: StandardScaler
    pca: PCA

    @classmethod
    def fit(cls, X_raw: np.ndarray, n_components: int, seed: int = 42, whiten: bool = False) -> "PCARepresentation":
        X_raw = np.asarray(X_raw, dtype=np.float64)
        scaler = StandardScaler().fit(X_raw)
        # the exact SVD is deterministic and cheap for a few hundred subjects (about a second)
        pca = PCA(n_components=min(n_components, X_raw.shape[0] - 1, X_raw.shape[1]), random_state=seed,
                  whiten=whiten, svd_solver="full").fit(scaler.transform(X_raw))
        return cls(scaler, pca)

    @property
    def n_components(self) -> int:
        return int(self.pca.n_components_)

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        """Raw FC -> PCA scores."""
        return self.pca.transform(self.scaler.transform(np.asarray(X_raw, dtype=np.float64)))

    def embed(self, scores: np.ndarray) -> np.ndarray:
        """PCA scores -> standardized edge space (isometric for an un-whitened PCA)."""
        return self.pca.inverse_transform(np.asarray(scores, dtype=np.float64))

    def project(self, X_std: np.ndarray) -> np.ndarray:
        """Standardized edge space -> this PCA's scores (used to draw pooled outputs in one basis)."""
        return self.pca.transform(np.asarray(X_std, dtype=np.float64))

    def inverse(self, scores: np.ndarray) -> np.ndarray:
        """PCA scores -> raw FC (Fisher-z) space."""
        return self.scaler.inverse_transform(self.embed(scores))

    def summary(self) -> dict:
        evr = self.pca.explained_variance_ratio_
        return {"n_components": self.n_components, "whiten": bool(self.pca.whiten),
                "explained_variance_ratio_total": float(evr.sum()),
                "explained_variance_ratio": [float(v) for v in evr],
                "eigenvalue_ratio_first_last": float(self.pca.explained_variance_[0] / self.pca.explained_variance_[-1])}


def make_pca_representation_fn(X_raw: np.ndarray, n_components: int, seed: int = 42, whiten: bool = False,
                               store: list | None = None):
    """
    Return ``f(fold, tr, te) -> (reps, embed)`` for :func:`run_protocol`: scaler + PCA fitted on
    ``tr`` only; ``reps = {"raw": X_raw, "pca": scores}``, ``embed = {"pca": rep.embed}``.
    Fitted representations are appended to ``store`` when given.
    """
    X_raw = np.asarray(X_raw, dtype=np.float64)

    def f(fold, tr, te):
        rep = PCARepresentation.fit(X_raw[tr], n_components, seed=seed, whiten=whiten)
        if store is not None:
            store.append(rep)
        return {"raw": X_raw, "pca": rep.transform(X_raw)}, {"pca": rep.embed}

    return f
