"""
coral.py -- CORAL (Sun, Feng & Saenko, 2016) with every site as a source domain and the hub as target.

Algorithm 1 of the paper, applied per site:

    normalise source and target features to zero mean and unit standard deviation
    C_S = cov(D_S) + eye(p),  C_T = cov(D_T) + eye(p)          # lambda = 1, as in the paper
    D_S* = D_S C_S^{-1/2} C_T^{1/2}

The adapted source then lives in the hub's normalised coordinates. To pool it with the hub we map it back
with the hub's feature means and standard deviations, which leaves the hub unchanged. Standard
deviations and covariances use the n - 1 denominator (MATLAB ``std`` and ``cov``, the paper's code). A site
with fewer than two training scans has no standard deviation and passes through unchanged.
"""

from __future__ import annotations

import numpy as np

from ..harmonization.base import Harmonizer
from ..harmonization.transport import sqrtm_psd
from ..logging_utils import get_logger

log = get_logger(__name__)


def _inv_sqrtm(C: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    return (V / np.sqrt(w)) @ V.T


class CORALHarmonizer(Harmonizer):
    name = "CORAL"

    def __init__(self, target_site: int, lam: float = 1.0, name: str | None = None):
        self.target_site = int(target_site)
        self.lam = float(lam)
        if name is not None:
            self.name = name

    def fit(self, X, y_site, y_diag=None, **aux):
        X = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        if self.target_site not in np.unique(y_site):
            raise ValueError("target site absent from training data")
        p = X.shape[1]
        hub = X[y_site == self.target_site]
        self.mu_t_ = hub.mean(axis=0)
        self.sd_t_ = np.maximum(hub.std(axis=0, ddof=1), 1e-12)
        Zt = (hub - self.mu_t_) / self.sd_t_
        Ct_sqrt = sqrtm_psd(np.cov(Zt, rowvar=False, ddof=1) + self.lam * np.eye(p))
        self.maps_: dict[int, tuple | None] = {}
        for s in np.unique(y_site):
            s = int(s)
            if s == self.target_site:
                self.maps_[s] = None
                continue
            Xs = X[y_site == s]
            if len(Xs) < 2:
                log.debug("CORAL: site %s has %d training scan(s); left unchanged", s, len(Xs))
                self.maps_[s] = "identity"
                continue
            mu = Xs.mean(axis=0)
            sd = np.maximum(Xs.std(axis=0, ddof=1), 1e-12)
            Zs = (Xs - mu) / sd
            Cs = np.cov(Zs, rowvar=False, ddof=1) + self.lam * np.eye(p)
            self.maps_[s] = (mu, sd, _inv_sqrtm(Cs) @ Ct_sqrt)
        return self

    def transform(self, X, y_site, **aux):
        X = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        out = X.copy()
        unknown = []
        for s in np.unique(y_site):
            s = int(s)
            if s not in self.maps_:
                unknown.append(s)
                continue
            mp = self.maps_[s]
            if mp is None or isinstance(mp, str):
                continue
            mu, sd, A = mp
            m = y_site == s
            out[m] = (((X[m] - mu) / sd) @ A) * self.sd_t_ + self.mu_t_
        if unknown:
            log.warning("transform: sites %s were not present at fit time; left unchanged", unknown)
        return out

    def describe(self):
        d = super().describe()
        d.update({"target_site": self.target_site, "lambda": self.lam, "uses_labels_at_transform": False})
        return d
