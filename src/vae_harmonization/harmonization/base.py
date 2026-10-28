"""
base.py -- The ``Harmonizer`` interface shared by our method and all baselines.

Every method is a *fit / transform* estimator so that the evaluation
protocol can fit it on the training folds only and apply the frozen
transform to held-out subjects.  ``fit`` receives the training features, the
site labels and (optionally) the biological labels a method is allowed to
protect; ``transform`` receives features and site labels only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Harmonizer(ABC):
    """Site-harmonization estimator with scikit-learn-like semantics."""

    name: str = "harmonizer"

    @abstractmethod
    def fit(self, X: np.ndarray, y_site: np.ndarray, y_diag: np.ndarray | None = None, **aux) -> "Harmonizer":
        """Estimate all site-specific parameters from training data only."""

    @abstractmethod
    def transform(self, X: np.ndarray, y_site: np.ndarray) -> np.ndarray:
        """Apply the frozen transform to (possibly unseen) subjects."""

    def fit_transform(self, X, y_site, y_diag=None, **aux) -> np.ndarray:
        return self.fit(X, y_site, y_diag=y_diag, **aux).transform(X, y_site)

    def describe(self) -> dict:
        """Small JSON-serialisable description used in result files."""
        return {"name": self.name, "class": type(self).__name__}


class IdentityHarmonizer(Harmonizer):
    """No harmonization (the 'Raw' reference)."""

    name = "Raw"

    def fit(self, X, y_site, y_diag=None, **aux):
        return self

    def transform(self, X, y_site):
        return np.asarray(X, dtype=np.float64).copy()
