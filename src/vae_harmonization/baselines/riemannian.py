"""
riemannian.py -- Riemannian baselines that act on the connectivity matrices as SPD matrices.

Representation
--------------
The stored connectomes are Fisher-z vectors of correlation matrices. Most of those matrices are rank
deficient: band-pass filtering and short scans leave fewer independent time points than regions, and the
number of zero eigenvalues differs between sites because scan length and repetition time do. A
Riemannian method needs strictly positive definite input, so every correlation matrix is first shrunk
toward the identity, ``C <- (1 - alpha) C + alpha I`` (it stays a correlation matrix), with one fixed
``alpha`` for all subjects (:func:`log_euclidean_vectors`, default 0.1). The papers below work with
full-rank covariance matrices and do not prescribe this step.

Every method reads and writes the vector ``upper(logm(C))``: the upper triangle of the matrix logarithm
with off-diagonal entries multiplied by sqrt(2), so the vector norm equals the Frobenius norm
(``pyriemann.utils.tangentspace.upper``). For the re-centred methods this is the tangent vector at the
identity, which is how pyRiemann's transfer-learning pipelines hand SPD matrices to a vector classifier.
The un-harmonized counterpart is the same map applied to the raw matrices.

Methods (statistics from the training fold; a held-out matrix is transformed with its site's statistics)
--------------------------------------------------------------------------------------------------------
* :class:`RiemannianRecenteringHarmonizer` -- Zanini et al. (2018): ``C -> M_s^{-1/2} C M_s^{-1/2}`` with
  ``M_s`` the affine-invariant (Riemannian) mean of site ``s``. This is what ``pyriemann.transfer.TLCenter``
  (metric "riemann") computes for each domain, applied per site.
* :class:`EuclideanAlignmentHarmonizer` -- He & Wu (2020): the same with the arithmetic mean of the site's
  matrices (``TLCenter`` with metric "euclid").
* :class:`RiemannianProcrustesHarmonizer` -- the first two steps of Riemannian Procrustes analysis
  (Rodrigues et al. 2019): re-centre as above, then stretch every site to the hub's dispersion,
  ``C -> C^{sqrt(d_hub / d_s)}``, where ``d`` is the mean squared affine-invariant distance of a site's
  matrices to its mean (``pyriemann.transfer.TLStretch``). The third step rotates each site so that its
  class means match the hub's; at 200 regions that manifold optimisation takes about four minutes per site
  and fold on a desktop CPU, so it is not run.
* :class:`RLETHarmonizer` -- rigid log-Euclidean translation of Simeon et al. (2022):
  ``log C -> log C - L_s + L_bar`` with ``L_s`` the mean matrix logarithm of site ``s`` and ``L_bar`` the
  unweighted mean of the site means.

A site with a single training matrix has no dispersion and is re-centred without stretching. Sites that
are absent at fit time pass through unchanged.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np
from pyriemann.utils.base import expm, invsqrtm, logm, powm
from pyriemann.utils.distance import distance
from pyriemann.utils.mean import mean_euclid, mean_riemann
from pyriemann.utils.tangentspace import unupper, upper

from ..harmonization.base import Harmonizer
from ..logging_utils import get_logger
from .connectome import vecs_to_corrs

log = get_logger(__name__)

DEFAULT_SHRINKAGE = 0.1

# Re-centering and Riemannian Procrustes compute the same affine-invariant site means on the same
# training rows; the iterative mean is the slow step at 200 regions, so the last results are reused.
_MEAN_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_MEAN_CACHE_SIZE = 64


def _riemann_mean_cached(C: np.ndarray) -> np.ndarray:
    key = hashlib.sha1(np.ascontiguousarray(C).tobytes()).hexdigest()
    if key in _MEAN_CACHE:
        _MEAN_CACHE.move_to_end(key)
        return _MEAN_CACHE[key]
    M = mean_riemann(C)
    _MEAN_CACHE[key] = M
    if len(_MEAN_CACHE) > _MEAN_CACHE_SIZE:
        _MEAN_CACHE.popitem(last=False)
    return M


def n_from_upper(p: int) -> int:
    """Matrix size n from the length n(n+1)/2 of an ``upper`` vector."""
    n = int(round((-1 + np.sqrt(1 + 8 * int(p))) / 2))
    if n * (n + 1) // 2 != int(p):
        raise ValueError(f"{p} is not the length of an upper-triangular vector with diagonal")
    return n


def regularized_correlations(X_fisher: np.ndarray, alpha: float = DEFAULT_SHRINKAGE) -> np.ndarray:
    """Fisher-z vectors -> correlation matrices shrunk toward the identity (strictly positive definite)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    C = vecs_to_corrs(np.asarray(X_fisher, dtype=np.float64))
    n = C.shape[-1]
    return (1.0 - alpha) * C + alpha * np.eye(n)[None, :, :]


def log_euclidean_vectors(X_fisher: np.ndarray, alpha: float = DEFAULT_SHRINKAGE) -> np.ndarray:
    """The SPD representation: ``upper(logm(C_alpha))`` for every subject (label-free, no fitted quantity)."""
    return upper(logm(regularized_correlations(X_fisher, alpha)))


def to_spd(V: np.ndarray) -> np.ndarray:
    """Inverse of :func:`log_euclidean_vectors` up to the shrinkage: ``expm(unupper(v))``."""
    return expm(unupper(np.asarray(V, dtype=np.float64)))


class _SPDSiteHarmonizer(Harmonizer):
    """Fit per-site statistics on SPD matrices rebuilt from the vectors; transform site by site."""

    def __init__(self, target_site: int | None = None, name: str | None = None):
        self.target_site = None if target_site is None else int(target_site)
        if name is not None:
            self.name = name

    # subclasses fill these two
    def _fit_site(self, C: np.ndarray, s: int) -> dict:
        raise NotImplementedError

    def _apply_site(self, C: np.ndarray, st: dict) -> np.ndarray:
        raise NotImplementedError

    def _finish_fit(self) -> None:
        """Hook run after every site is fitted (e.g. to read the hub's statistics)."""

    def fit(self, X, y_site, y_diag=None, **aux):
        X = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        self.n_ = n_from_upper(X.shape[1])
        C = to_spd(X)
        self.site_stats_ = {int(s): self._fit_site(C[y_site == s], int(s)) for s in np.unique(y_site)}
        self._finish_fit()
        return self

    def transform(self, X, y_site, **aux):
        X = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        out = X.copy()
        unknown = []
        for s in np.unique(y_site):
            st = self.site_stats_.get(int(s))
            if st is None:
                unknown.append(int(s))
                continue
            m = y_site == s
            out[m] = upper(logm(self._apply_site(to_spd(X[m]), st)))
        if unknown:
            log.warning("transform: sites %s were not present at fit time; left unchanged", unknown)
        return out

    def describe(self):
        d = super().describe()
        d.update({"target_site": self.target_site, "n_regions": getattr(self, "n_", None),
                  "uses_labels_at_transform": False})
        return d


class RiemannianRecenteringHarmonizer(_SPDSiteHarmonizer):
    """Zanini et al. (2018): whiten every site by its affine-invariant mean."""

    name = "Riemannian re-centering"

    def _fit_site(self, C, s):
        return {"isqrt": invsqrtm(_riemann_mean_cached(C)), "n": len(C)}

    def _apply_site(self, C, st):
        return st["isqrt"] @ C @ st["isqrt"]


class EuclideanAlignmentHarmonizer(_SPDSiteHarmonizer):
    """He & Wu (2020): whiten every site by the arithmetic mean of its matrices."""

    name = "Euclidean alignment"

    def _fit_site(self, C, s):
        return {"isqrt": invsqrtm(mean_euclid(C)), "n": len(C)}

    def _apply_site(self, C, st):
        return st["isqrt"] @ C @ st["isqrt"]


class RiemannianProcrustesHarmonizer(_SPDSiteHarmonizer):
    """Rodrigues et al. (2019), steps 1-2: re-centre each site, then stretch it to the hub's dispersion."""

    name = "Riemannian Procrustes (re-centre + stretch)"

    def __init__(self, target_site: int, name: str | None = None):
        super().__init__(target_site=target_site, name=name)

    def _fit_site(self, C, s):
        M = _riemann_mean_cached(C)
        isqrt = invsqrtm(M)
        # dispersion around the site mean, as in pyriemann.transfer.TLStretch (uniform weights)
        disp = float(np.mean(distance(C, M, metric="riemann", squared=True))) if len(C) > 1 else 0.0
        return {"isqrt": isqrt, "dispersion": disp, "n": len(C)}

    def _finish_fit(self):
        if self.target_site not in self.site_stats_:
            raise ValueError("target site absent from training data")
        d_hub = self.site_stats_[self.target_site]["dispersion"]
        for s, st in self.site_stats_.items():
            st["power"] = float(np.sqrt(d_hub / st["dispersion"])) if st["dispersion"] > 0 and d_hub > 0 else 1.0

    def _apply_site(self, C, st):
        Cc = st["isqrt"] @ C @ st["isqrt"]
        return powm(Cc, st["power"]) if st["power"] != 1.0 else Cc


class RLETHarmonizer(Harmonizer):
    """Simeon et al. (2022): rigid translation of the matrix logarithms onto the mean of the site means.

    The input vectors are already ``upper(log C)`` and ``upper`` is linear, so the translation is applied to
    the vectors directly; mapping back with the exponential and forward with the logarithm would return
    the same vector.
    """

    name = "RLET"

    def __init__(self, name: str | None = None):
        if name is not None:
            self.name = name

    def fit(self, X, y_site, y_diag=None, **aux):
        X = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        self.site_means_ = {int(s): X[y_site == s].mean(axis=0) for s in np.unique(y_site)}
        self.global_mean_ = np.mean(np.stack(list(self.site_means_.values())), axis=0)
        return self

    def transform(self, X, y_site, **aux):
        X = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        out = X.copy()
        unknown = []
        for s in np.unique(y_site):
            L = self.site_means_.get(int(s))
            if L is None:
                unknown.append(int(s))
                continue
            m = y_site == s
            out[m] = X[m] - L + self.global_mean_
        if unknown:
            log.warning("transform: sites %s were not present at fit time; left unchanged", unknown)
        return out

    def describe(self):
        d = super().describe()
        d.update({"uses_labels_at_transform": False})
        return d
