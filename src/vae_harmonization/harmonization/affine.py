"""
affine.py -- Protected-subspace affine harmonization (the proposed method).

For every source site ``s`` the transform is

    T_s(z) = W_bio (W_bio^T z + shift_s)                         # protected: mean shift
           + W_perp ( B_s (W_perp^T z - mu_perp_s) + mu_perp_hub ) # complement: transport

with ``B_s`` the Bures-Wasserstein (Gaussian OT) map between the complement
covariance of site ``s`` and that of the hub (Theorem 3.1), optionally
relaxed along the McCann geodesic ``B_s(lam) = (1-lam) I + lam B_s``.

Options cover every ablation reported in the paper so that all variants are
fitted and evaluated by one code path:

``protected``        tuple of axis names: 'diagnosis', 'severity', or any key
                     of the ``covariates`` dict passed to ``fit`` (e.g. 'age',
                     'sex').  Empty tuple -> full-space affine (no protection).
``complement_map``   'bw' (default), 'cholesky', 'zca', 'identity' (mean shift only)
``covariance``       per-site covariance estimator on the complement:
                     'pooled' (default): sample covariance shrunk toward the pooled
                     within-site covariance of the other sites, with the analytic
                     Ledoit-Wolf / Schaefer-Strimmer intensity; 'lw': Ledoit-Wolf
                     shrinkage toward a scaled identity; 'diagonal': shrinkage
                     toward the site's own diagonal; 'empirical': no shrinkage.
``lam``              float in [0, 1] or 'auto' (= 1 - shrinkage intensity of the site)
``hub``              'target' (hub-and-spoke to ``target_site``) or 'barycenter'
``min_cov_samples``  sites with fewer training subjects get a mean shift only
``mean_shrinkage``   'none' (default) or 'james-stein': shrink each site's offset from the
                     hub mean, ``delta_s``, by the positive-part James-Stein factor
                     ``max(0, 1 - tau_s / ||delta_s||^2)`` with ``tau_s = tr(S) (1/n_s + 1/n_hub)``
                     and ``S`` the pooled within-site covariance of the scores.  ``tau_s`` is
                     the expected squared length of the offset's sampling error (James & Stein
                     1961; Efron & Morris 1973), so the factor removes about that much length.
                     A diagnostic variant.
``mean_model``       'conditional' (default): site means are the intercepts of a
                     linear model with site dummies and the protected covariates
                     (diagnosis, severity, age, ...), so that the part of a
                     between-site mean difference explained by an unbalanced
                     covariate is *not* removed (the location model of ComBat);
                     'marginal': plain site means (removes all between-site
                     differences, including those due to e.g. different age
                     ranges).  Either way no label is needed at transform time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..logging_utils import get_logger
from .base import Harmonizer
from .subspace import (
    complement_basis,
    decompose,
    fit_covariate_axis,
    fit_diagnosis_axis,
    fit_severity_axis,
    orthonormal_protected_basis,
    recompose,
)
from .transport import (
    COVARIANCE_METHODS,
    bures_wasserstein_barycenter,
    bures_wasserstein_map,
    cholesky_map,
    estimate_covariance,
    expected_squared_displacement,
    geodesic_map,
    pooled_within_covariance,
    zca_map,
)

log = get_logger(__name__)

_MAPS = {"bw": bures_wasserstein_map, "cholesky": cholesky_map, "zca": zca_map}


def conditional_site_means(Z: np.ndarray, y_site: np.ndarray, C: np.ndarray, min_complete: int = 3):
    """
    Site intercepts of the linear model  Z = D_site M + (C - mean(C)) B + E  (OLS).

    ``C`` holds the protected covariates; NaN marks a missing value.  Every subject
    informs its site intercept; only observed values inform the covariate slopes.
    Missing entries are imputed before the fit: sporadic gaps (the covariate is
    observed for at least 80% of the subject's site) get the site mean of the
    observed values, so a site's age distribution is not distorted; structural
    gaps (e.g. a symptom-severity score that exists for cases only, i.e. for
    roughly half of every site) get the grand mean, which is a constant coding
    that the diagnosis intercept absorbs -- a site-specific constant would not be.  A
    covariate observed on fewer than ``min_complete`` subjects is dropped from the
    model.  The intercepts are evaluated at the covariate grand mean, so a site
    whose subjects are older / more often cases is not shifted for that reason.
    Returns ({site: (d,) mean}, B (p, d), covariate mean).
    """
    Z = np.asarray(Z, dtype=np.float64)
    y_site = np.asarray(y_site).astype(int)
    C = np.asarray(C, dtype=np.float64)
    sites = np.unique(y_site)
    marginal = {int(s): Z[y_site == s].mean(axis=0) for s in sites}
    if C.ndim != 2 or C.shape[1] == 0:
        return marginal, np.zeros((0, Z.shape[1])), np.zeros(0)
    finite = np.isfinite(C)
    usable = finite.sum(axis=0) >= max(min_complete, 3)
    c_mean = np.where(usable, np.nanmean(np.where(finite, C, np.nan), axis=0), 0.0)
    C_imp = np.where(finite, C, c_mean)                             # structural gaps: grand mean
    for s in sites:
        m = y_site == s
        obs_frac = finite[m].mean(axis=0)
        for j in np.flatnonzero(usable & (obs_frac >= 0.8) & (obs_frac < 1.0)):
            C_imp[m & ~finite[:, j], j] = C[m & finite[:, j], j].mean()   # sporadic gaps: site mean
    Cc = (C_imp - c_mean)[:, usable]                                 # centred
    if Cc.shape[1] == 0 or Z.shape[0] < sites.size + Cc.shape[1] + 2:
        return marginal, np.zeros((C.shape[1], Z.shape[1])), c_mean
    D = (y_site[:, None] == sites[None, :]).astype(float)
    coef, *_ = np.linalg.lstsq(np.hstack([D, Cc]), Z, rcond=None)
    M = coef[: sites.size]
    B = np.zeros((C.shape[1], Z.shape[1]))
    B[usable] = coef[sites.size:]
    means = {int(s): M[i] for i, s in enumerate(sites)}
    return means, B, c_mean


@dataclass
class SiteTransform:
    site: int
    n: int
    shift_bio: np.ndarray            # (k,)
    mu_perp_s: np.ndarray            # (d-k,)
    B: np.ndarray | None             # (d-k, d-k) or None for mean-shift only
    lam: float
    shrinkage: float
    identity: bool = False           # hub site under hub='target'
    displacement_train: float = float("nan")
    extra: dict = field(default_factory=dict)


class ProtectedAffineHarmonizer(Harmonizer):
    name = "Ours"

    def __init__(
        self,
        target_site: int | None = None,
        protected: tuple[str, ...] = ("diagnosis",),
        complement_map: str = "bw",
        covariance: str = "pooled",
        lam: float | str = 1.0,
        hub: str = "target",
        ridge: float = 1e-6,
        min_cov_samples: int = 10,
        svm_C: float = 1.0,
        collinearity_threshold: float = 0.8,
        seed: int = 42,
        name: str | None = None,
        mean_model: str = "conditional",
        mean_shrinkage: str = "none",
    ):
        if complement_map not in (*_MAPS, "identity"):
            raise ValueError(f"complement_map must be one of {list(_MAPS) + ['identity']}")
        if mean_model not in ("conditional", "marginal"):
            raise ValueError("mean_model must be 'conditional' or 'marginal'")
        if mean_shrinkage not in ("none", "james-stein"):
            raise ValueError("mean_shrinkage must be 'none' or 'james-stein'")
        if covariance not in COVARIANCE_METHODS:
            raise ValueError(f"covariance must be one of {COVARIANCE_METHODS}")
        if hub not in ("target", "barycenter"):
            raise ValueError("hub must be 'target' or 'barycenter'")
        if hub == "target" and target_site is None:
            raise ValueError("target_site is required for hub='target'")
        if not (lam == "auto" or 0.0 <= float(lam) <= 1.0):
            raise ValueError("lam must be in [0, 1] or 'auto'")
        self.target_site = target_site
        self.protected = tuple(protected)
        self.complement_map = complement_map
        self.covariance = covariance
        self.lam = lam
        self.hub = hub
        self.ridge = ridge
        self.min_cov_samples = min_cov_samples
        self.svm_C = svm_C
        self.collinearity_threshold = collinearity_threshold
        self.seed = seed
        self.mean_model = mean_model
        self.mean_shrinkage = mean_shrinkage
        if name is not None:
            self.name = name

    # -- fitting -------------------------------------------------------------

    def _fit_protected_axes(self, Z, y_diag, severity, covariates) -> tuple[list, list[str]]:
        axes, names = [], []
        for spec in self.protected:
            if spec == "diagnosis":
                if y_diag is None:
                    raise ValueError("protected='diagnosis' requires y_diag")
                axes.append(fit_diagnosis_axis(Z, y_diag, C=self.svm_C, seed=self.seed))
            elif spec == "severity":
                axes.append(None if severity is None else fit_severity_axis(Z, severity, seed=self.seed))
            else:
                if covariates is None or spec not in covariates:
                    raise ValueError(f"protected axis '{spec}' not found in covariates")
                axes.append(fit_covariate_axis(Z, covariates[spec], kind="auto", seed=self.seed))
            names.append(spec)
        return axes, names

    def _covariate_matrix(self, n, y_diag, severity, covariates) -> np.ndarray:
        """Protected covariates as an (n, p) float matrix (NaN = missing); labels < 0 are missing."""
        cols = []
        for spec in self.protected:
            if spec == "diagnosis" and y_diag is not None:
                y = np.asarray(y_diag, dtype=float)
                cols.append(np.where(y >= 0, y, np.nan))
            elif spec == "severity" and severity is not None:
                cols.append(np.asarray(severity, dtype=float))
            elif covariates is not None and spec in covariates:
                cols.append(np.asarray(covariates[spec], dtype=float))
        return np.column_stack(cols) if cols else np.zeros((n, 0))

    def _site_covariance(self, P, y_site, s) -> tuple[np.ndarray, float]:
        """Complement covariance of site ``s`` with the configured estimator (see module docstring)."""
        m = y_site == s
        if self.covariance == "pooled":
            target = pooled_within_covariance(P, y_site, exclude=s)
            if target is not None:
                return estimate_covariance(P[m], "pooled", self.ridge, target=target)
            return estimate_covariance(P[m], "lw", self.ridge)      # single-site data: no pool to shrink to
        return estimate_covariance(P[m], self.covariance, self.ridge)

    def fit(self, X, y_site, y_diag=None, severity=None, covariates=None, **aux):
        Z = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        n, d = Z.shape
        sites, counts = np.unique(y_site, return_counts=True)
        self.d_ = d

        # 1. protected subspace ------------------------------------------------
        axes, names = self._fit_protected_axes(Z, y_diag, severity, covariates)
        W_bio, info = orthonormal_protected_basis(axes, self.collinearity_threshold)
        if W_bio.shape[0] == 0:
            W_bio = np.zeros((d, 0))
        self.W_bio_ = W_bio
        self.W_perp_ = complement_basis(W_bio, d)
        self.protected_names_ = [names[i] for i in info["kept"]]
        self.k_ = W_bio.shape[1]

        A, P = decompose(Z, self.W_bio_, self.W_perp_)

        # 2. per-site statistics ------------------------------------------------
        if self.mean_model == "conditional":
            Cmat = self._covariate_matrix(n, y_diag, severity, covariates)
            site_means, self.covariate_slopes_, self.covariate_means_ = conditional_site_means(Z, y_site, Cmat)
        else:
            site_means = {int(s): Z[y_site == s].mean(axis=0) for s in sites}
            self.covariate_slopes_, self.covariate_means_ = None, None
        self.mean_shrinkage_factors_ = {}
        if self.mean_shrinkage == "james-stein":
            site_means = self._james_stein(Z, y_site, site_means)
        stats: dict[int, dict] = {}
        for s, c in zip(sites, counts):
            m = y_site == s
            mu = site_means[int(s)]
            st = {"n": int(c), "mu_bio": mu @ self.W_bio_, "mu_perp": mu @ self.W_perp_, "C": None, "delta": 0.0}
            if c >= self.min_cov_samples and self.complement_map != "identity":
                st["C"], st["delta"] = self._site_covariance(P, y_site, int(s))
            stats[int(s)] = st
        self.site_stats_ = stats

        # 3. hub -----------------------------------------------------------------
        if self.hub == "target":
            t = int(self.target_site)
            if t not in stats:
                raise ValueError(f"target site {t} absent from training data")
            hub_mu_bio, hub_mu_perp, hub_C = stats[t]["mu_bio"], stats[t]["mu_perp"], stats[t]["C"]
            if hub_C is None and self.complement_map != "identity":
                hub_C, _ = self._site_covariance(P, y_site, t)
            bary_info = None
        else:
            with_cov = [s for s in stats if stats[s]["C"] is not None]
            w_all = np.array([stats[s]["n"] for s in stats], float)
            hub_mu_bio = sum(stats[s]["n"] * stats[s]["mu_bio"] for s in stats) / w_all.sum()
            hub_mu_perp = sum(stats[s]["n"] * stats[s]["mu_perp"] for s in stats) / w_all.sum()
            if self.complement_map == "identity" or not with_cov:
                hub_C, bary_info = None, None
            else:
                w = np.array([stats[s]["n"] for s in with_cov], float)
                hub_C, bary_info = bures_wasserstein_barycenter([stats[s]["C"] for s in with_cov], w)
        self.hub_mu_bio_, self.hub_mu_perp_, self.hub_C_ = hub_mu_bio, hub_mu_perp, hub_C
        self.barycenter_info_ = bary_info

        # 4. per-site maps -------------------------------------------------------
        transforms: dict[int, SiteTransform] = {}
        for s, st in stats.items():
            is_hub_site = self.hub == "target" and s == int(self.target_site)
            shift = hub_mu_bio - st["mu_bio"]
            B, lam_s = None, 0.0
            if not is_hub_site and st["C"] is not None and hub_C is not None:
                B_full = _MAPS[self.complement_map](st["C"], hub_C)
                lam_s = float(1.0 - st["delta"]) if self.lam == "auto" else float(self.lam)
                B = geodesic_map(B_full, lam_s)
            tr = SiteTransform(
                site=s, n=st["n"], shift_bio=shift, mu_perp_s=st["mu_perp"], B=B,
                lam=lam_s, shrinkage=float(st["delta"]), identity=is_hub_site,
            )
            if B is not None:
                tr.displacement_train = expected_squared_displacement(B, st["C"]) + float(
                    np.sum((hub_mu_perp - st["mu_perp"]) ** 2)) + float(np.sum(shift ** 2))
            elif not is_hub_site:
                tr.displacement_train = float(np.sum((hub_mu_perp - st["mu_perp"]) ** 2)) + float(np.sum(shift ** 2))
            else:
                tr.displacement_train = 0.0
            transforms[s] = tr
        self.site_transforms_ = transforms
        self.mean_shift_only_sites_ = [s for s, tr in transforms.items()
                                       if tr.B is None and not tr.identity and self.complement_map != "identity"]
        if self.mean_shift_only_sites_:
            log.debug("sites with n < %d received a mean shift only: %s",
                      self.min_cov_samples, self.mean_shift_only_sites_)
        return self

    def _james_stein(self, Z, y_site, site_means):
        """Positive-part James-Stein shrinkage of every site's offset from the hub (see module docstring)."""
        sites, counts = np.unique(y_site, return_counts=True)
        n_by_site = dict(zip(sites.astype(int).tolist(), counts.tolist()))
        # sampling variance of a site mean: pooled within-site covariance of the scores around each
        # site's own sample mean (one variance for all sites, as in ComBat's pooled variance)
        S = pooled_within_covariance(Z, y_site)
        tr_S = float(np.trace(S)) if S is not None else 0.0
        if self.hub == "target":
            t = int(self.target_site)
            hub_mu, n_hub = site_means[t], n_by_site[t]
        else:
            w = np.array([n_by_site[int(s)] for s in sites], float)
            hub_mu = sum(n_by_site[int(s)] * site_means[int(s)] for s in sites) / w.sum()
            n_hub = float(w.sum())
        out = {}
        for s in sites:
            s = int(s)
            delta = site_means[s] - hub_mu
            norm2 = float(delta @ delta)
            if self.hub == "target" and s == int(self.target_site):
                factor = 1.0
            else:
                tau = tr_S * (1.0 / n_by_site[s] + 1.0 / n_hub)
                factor = max(0.0, 1.0 - tau / norm2) if norm2 > 0 else 1.0
            self.mean_shrinkage_factors_[s] = factor
            out[s] = hub_mu + factor * delta
        return out

    def plugin_displacement(self) -> float:
        """Expected squared displacement per training subject under the fitted (plug-in) moments."""
        n = np.array([tr.n for tr in self.site_transforms_.values()], float)
        d = np.array([tr.displacement_train for tr in self.site_transforms_.values()], float)
        ok = np.isfinite(d)
        return float(np.average(d[ok], weights=n[ok])) if ok.any() else float("nan")

    # -- applying --------------------------------------------------------------

    def transform(self, X, y_site, **_ignored):
        """Apply the fitted map. Covariates are used at fit time only, so extra keywords are ignored."""
        Z = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        A, P = decompose(Z, self.W_bio_, self.W_perp_)
        A_h, P_h = A.copy(), P.copy()
        unknown = []
        for s in np.unique(y_site):
            m = y_site == s
            tr = self.site_transforms_.get(int(s))
            if tr is None:
                unknown.append(int(s))
                continue
            if tr.identity:
                continue
            A_h[m] = A[m] + tr.shift_bio
            Pc = P[m] - tr.mu_perp_s
            if tr.B is not None:
                Pc = Pc @ tr.B.T
            P_h[m] = Pc + self.hub_mu_perp_
        if unknown:
            log.warning("transform: sites %s were not seen during fit; left unchanged", unknown)
        return recompose(A_h, P_h, self.W_bio_, self.W_perp_)

    # -- introspection ---------------------------------------------------------

    def protected_coordinates(self, X) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.W_bio_

    def displacement(self, X, y_site) -> np.ndarray:
        """Per-subject squared displacement ``||T(z) - z||^2``."""
        Z = np.asarray(X, dtype=np.float64)
        return np.sum((self.transform(Z, y_site) - Z) ** 2, axis=1)

    def describe(self) -> dict:
        out = super().describe()
        out.update({
            "protected": list(self.protected),
            "protected_kept": getattr(self, "protected_names_", None),
            "k": getattr(self, "k_", None),
            "complement_map": self.complement_map,
            "covariance": self.covariance,
            "lam": self.lam,
            "hub": self.hub,
            "target_site": self.target_site,
            "min_cov_samples": self.min_cov_samples,
            "ridge": self.ridge,
            "mean_model": self.mean_model,
            "mean_shrinkage": self.mean_shrinkage,
        })
        if getattr(self, "mean_shrinkage_factors_", None):
            out["mean_shrinkage_factors"] = {str(k): v for k, v in self.mean_shrinkage_factors_.items()}
        if hasattr(self, "site_transforms_"):
            out["sites"] = {
                str(s): {"n": tr.n, "lam": tr.lam, "shrinkage": tr.shrinkage,
                         "mean_shift_only": tr.B is None and not tr.identity,
                         "displacement_train": tr.displacement_train}
                for s, tr in self.site_transforms_.items()
            }
            if self.barycenter_info_ is not None:
                out["barycenter"] = self.barycenter_info_
        return out


# -- convenience constructors used by the CLI / protocol --------------------------

def make_variant(variant: str, target_site: int, **kw) -> ProtectedAffineHarmonizer:
    """
    Named variants used in the paper's tables.

    'ours'          protected diagnosis + BW complement (main method)
    'ours-2d'       protected diagnosis + severity
    'full-affine'   no protected subspace, BW in full space
    'mean-shift'    protected diagnosis + mean shift only on complement
    'cholesky'      protected diagnosis + Cholesky map (non-minimal displacement)
    'zca'           protected diagnosis + whiten/colour map
    'ours-auto'     protected diagnosis + geodesic lam = 1 - shrinkage intensity
    'barycenter'    protected diagnosis + BW barycenter hub
    'ours-marginal' as 'ours' but with plain (marginal) site means
    'ours-lw'       as 'ours' with Ledoit-Wolf (identity-target) covariances
    'ours-diag'     as 'ours' with diagonal-target shrinkage
    'ours-emp'      as 'ours' with empirical covariances
    'ours-js'       as 'ours' with James-Stein shrinkage of the site offsets
    """
    table = {
        "ours": dict(protected=("diagnosis",), complement_map="bw"),
        "ours-marginal": dict(protected=("diagnosis",), complement_map="bw", mean_model="marginal"),
        "ours-2d": dict(protected=("diagnosis", "severity"), complement_map="bw"),
        "full-affine": dict(protected=(), complement_map="bw"),
        "mean-shift": dict(protected=("diagnosis",), complement_map="identity"),
        "cholesky": dict(protected=("diagnosis",), complement_map="cholesky"),
        "zca": dict(protected=("diagnosis",), complement_map="zca"),
        "ours-auto": dict(protected=("diagnosis",), complement_map="bw", lam="auto"),
        "barycenter": dict(protected=("diagnosis",), complement_map="bw", hub="barycenter"),
        "ours-lw": dict(protected=("diagnosis",), complement_map="bw", covariance="lw"),
        "ours-diag": dict(protected=("diagnosis",), complement_map="bw", covariance="diagonal"),
        "ours-emp": dict(protected=("diagnosis",), complement_map="bw", covariance="empirical"),
        "ours-js": dict(protected=("diagnosis",), complement_map="bw", mean_shrinkage="james-stein"),
    }
    if variant not in table:
        raise KeyError(f"unknown variant '{variant}'; choose from {sorted(table)}")
    params = {**table[variant], **kw}
    # Callers pass the run's default covariance. The estimator variants exist to change it.
    if "covariance" in table[variant]:
        params["covariance"] = table[variant]["covariance"]
    params.setdefault("name", variant)
    return ProtectedAffineHarmonizer(target_site=target_site, **params)
