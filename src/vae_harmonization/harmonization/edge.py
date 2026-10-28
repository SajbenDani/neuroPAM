"""
edge.py -- Carry a score-space harmonizer back to the full connectome as a residual update.

A harmonizer on the PCA scores changes only the part of a connectome that the components span. Instead
of reconstructing the connectome from the harmonized scores, which discards everything outside the
components (individual detail included), the change is added to the original edges:

    x' = x + R(T(z)) - R(z),        z = P(x)

with ``P`` the scaler + PCA fitted on the training rows and ``R`` its reconstruction. Because ``R`` is
affine, ``R(T(z)) - R(z) = ((T(z) - z) V) * sd``: the edges outside the span of the components are left
exactly as they were. The output is a full harmonized connectome, as edge-level analyses need.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..evaluation.representations import PCARepresentation
from .base import Harmonizer


class EdgeResidualHarmonizer(Harmonizer):
    name = "edge residual"

    def __init__(self, inner_factory: Callable[[], Harmonizer], pca_dim: int = 100, seed: int = 42,
                 representation: PCARepresentation | None = None, name: str | None = None):
        self.inner_factory = inner_factory
        self.pca_dim = int(pca_dim)
        self.seed = int(seed)
        self.representation = representation       # a fixed, already fitted basis (e.g. from other people)
        if name is not None:
            self.name = name

    def fit(self, X, y_site, y_diag=None, **aux):
        X = np.asarray(X, dtype=np.float64)
        self.rep_ = self.representation or PCARepresentation.fit(X, self.pca_dim, seed=self.seed)
        self.inner_ = self.inner_factory()
        if y_diag is not None:
            aux = {"y_diag": y_diag, **aux}
        self.inner_.fit(self.rep_.transform(X), y_site, **aux)
        return self

    def transform(self, X, y_site, **aux):
        X = np.asarray(X, dtype=np.float64)
        Z = self.rep_.transform(X)
        Z_h = self.inner_.transform(Z, y_site, **aux)
        return X + self.rep_.inverse(Z_h) - self.rep_.inverse(Z)

    def plugin_displacement(self) -> float:
        f = getattr(self.inner_, "plugin_displacement", None)
        return f() if callable(f) else float("nan")

    def describe(self):
        d = super().describe()
        inner = getattr(self, "inner_", None)
        d.update({"pca_dim": self.pca_dim, "inner": inner.describe() if inner is not None else None,
                  "uses_labels_at_transform": False})
        return d
