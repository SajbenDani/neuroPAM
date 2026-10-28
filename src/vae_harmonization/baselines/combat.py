"""
combat.py -- ComBat (Johnson et al. 2007; Fortin et al. 2018) and CovBat
(Chen et al. 2022) as fit/transform estimators.

Fitting uses the reference ``neuroCombat`` implementation.  Applying the
fitted location/scale estimates to held-out subjects is implemented here
(``_apply_estimates``) because ``neuroCombatFromTraining`` ignores biological
covariates.  On the training data the apply step reproduces
``neuroCombat(...)['data']`` exactly (see tests/test_baselines.py).

Note on labels: ComBat with a diagnosis covariate needs the diagnosis of
*every* subject it adjusts, including held-out ones -- that is how the
method is defined and used in practice.  The protocol therefore passes
``y_diag`` to ``transform`` for these two baselines only; the proposed
method never sees held-out labels.  ``covariate_keys=("age", "sex")`` with
``protect_diagnosis=False`` is label-free ComBat: it needs the age and sex of
every adjusted subject but no diagnosis.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ..harmonization.base import Harmonizer
from ..logging_utils import get_logger

log = get_logger(__name__)


def _quiet_neurocombat(**kw):
    """Run neuroCombat silently; shim the removed ``np.int`` alias its ref_batch path uses."""
    from neuroCombat import neuroCombat
    patched = False
    if kw.get("ref_batch") is not None and not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
        patched = True
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return neuroCombat(**kw)
    finally:
        if patched:
            del np.int  # type: ignore[attr-defined]


def _one_hot_drop_first(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Dummy coding identical to neuroCombat's make_design_matrix (drops first level)."""
    idx = np.searchsorted(levels, values)
    oh = np.zeros((len(values), len(levels)))
    oh[np.arange(len(values)), idx] = 1.0
    return oh[:, 1:]


def _apply_estimates(
    dat: np.ndarray,            # (D, N) features x subjects
    batch_idx: np.ndarray,      # (N,) index into estimates' batch levels
    grand_mean: np.ndarray,     # (D,)
    var_pooled: np.ndarray,     # (D,)
    gamma_star: np.ndarray,     # (n_batch, D)
    delta_star: np.ndarray,     # (n_batch, D)
    mod_mean: np.ndarray,       # (D, N) covariate effects (zeros when none)
) -> np.ndarray:
    sd = np.sqrt(var_pooled)[:, None]
    stand = grand_mean[:, None] + mod_mean
    s = (dat - stand) / sd
    s = (s - gamma_star[batch_idx].T) / np.sqrt(delta_star[batch_idx]).T
    return s * sd + stand


class ComBatHarmonizer(Harmonizer):
    name = "ComBat"

    def __init__(
        self,
        protect_diagnosis: bool = True,
        ref_batch: int | None = None,
        eb: bool = True,
        parametric: bool = True,
        mean_only: bool = False,
        name: str | None = None,
        covariate_keys: tuple[str, ...] = (),
    ):
        self.protect_diagnosis = protect_diagnosis
        self.ref_batch = ref_batch
        self.eb = eb
        self.parametric = parametric
        self.mean_only = mean_only
        # continuous covariates kept in the model, read from ``covariates`` at fit and transform.
        # Sex is coded 1 = male, 0 = female, so as a continuous column it equals neuroCombat's dummy coding.
        self.covariate_keys = tuple(covariate_keys)
        if name is not None:
            self.name = name

    uses_labels_at_transform = True

    def _covariate_columns(self, covariates):
        """(n, k) matrix of ``covariate_keys``; a missing value gets the training-fold mean of that covariate."""
        if covariates is None or any(k not in covariates for k in self.covariate_keys):
            raise ValueError(f"ComBat with covariates {list(self.covariate_keys)} needs them at fit and transform")
        C = np.column_stack([np.asarray(covariates[k], dtype=np.float64) for k in self.covariate_keys])
        return np.where(np.isfinite(C), C, self.covariate_fill_)

    def _covariate_design(self, y_diag, covariates=None, n=0):
        """Covariate columns in neuroCombat's order: diagnosis dummies, then the continuous covariates."""
        cols = [np.zeros((n if y_diag is None else len(y_diag), 0))]
        if self.protect_diagnosis:
            if y_diag is None:
                raise ValueError("ComBat with protect_diagnosis=True needs y_diag")
            cols.append(_one_hot_drop_first(np.asarray(y_diag), self.diag_levels_))
        if self.covariate_keys:
            cols.append(self._covariate_columns(covariates))
        return np.hstack(cols)

    min_site_n = 2      # a location/scale model cannot be estimated from a single scan

    def fit(self, X, y_site, y_diag=None, covariates=None, **aux):
        X = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        N = X.shape[0]
        sites, counts = np.unique(y_site, return_counts=True)
        tiny = sites[counts < self.min_site_n]
        if tiny.size:
            log.warning("fit: sites %s have < %d training scans; ComBat leaves them unchanged",
                        tiny.tolist(), self.min_site_n)
        self.unseen_sites_ = sorted(tiny.tolist())
        keep = ~np.isin(y_site, tiny)
        Xk, sk = X[keep], y_site[keep]

        covars = pd.DataFrame({"batch": sk})
        cat_cols: list[str] = []
        if self.protect_diagnosis:
            if y_diag is None:
                raise ValueError("ComBat with protect_diagnosis=True needs y_diag")
            y_diag = np.asarray(y_diag).astype(int)
            self.diag_levels_ = np.unique(y_diag)
            covars["diagnosis"] = y_diag[keep]
            cat_cols = ["diagnosis"]
        else:
            self.diag_levels_ = None
        cov_keep = None
        if self.covariate_keys:
            if covariates is None or any(k not in covariates for k in self.covariate_keys):
                raise ValueError(f"ComBat with covariates {list(self.covariate_keys)} needs them at fit and transform")
            cov_keep = {k: np.asarray(covariates[k], dtype=np.float64)[keep] for k in self.covariate_keys}
            self.covariate_fill_ = np.array([np.nanmean(v) if np.isfinite(v).any() else 0.0
                                             for v in cov_keep.values()])
            C = self._covariate_columns(cov_keep)
            for j, k in enumerate(self.covariate_keys):
                covars[k] = C[:, j]

        res = _quiet_neurocombat(
            dat=Xk.T.copy(), covars=covars, batch_col="batch", categorical_cols=cat_cols,
            continuous_cols=list(self.covariate_keys) or None,
            eb=self.eb, parametric=self.parametric, mean_only=self.mean_only,
            ref_batch=self.ref_batch,
        )
        est = res["estimates"]
        self.batch_levels_ = np.asarray(est["batches"]).astype(int)
        self.var_pooled_ = np.asarray(est["var.pooled"], dtype=np.float64).ravel()
        self.grand_mean_ = np.asarray(est["stand.mean"], dtype=np.float64)[:, 0]
        self.gamma_star_ = np.asarray(est["gamma.star"], dtype=np.float64)
        self.delta_star_ = np.asarray(est["delta.star"], dtype=np.float64)

        # recover covariate coefficients B_cov (p, D) from mod.mean = (design_cov B_cov)^T
        design_cov = self._covariate_design(None if y_diag is None else y_diag[keep], cov_keep, int(keep.sum()))
        if design_cov.shape[1] > 0:
            mod_mean = np.asarray(est["mod.mean"], dtype=np.float64)   # (D, N_keep)
            self.B_cov_, *_ = np.linalg.lstsq(design_cov, mod_mean.T, rcond=None)
        else:
            self.B_cov_ = np.zeros((0, X.shape[1]))
        Xh = X.copy()
        Xh[keep] = np.asarray(res["data"], dtype=np.float64).T
        self.X_train_harmonized_ = Xh
        self.n_train_ = N
        return self

    def _mod_mean(self, y_diag, N, covariates=None):
        if self.B_cov_.shape[0] == 0:
            return np.zeros((self.grand_mean_.size, N))
        return (self._covariate_design(y_diag, covariates, N) @ self.B_cov_).T

    def transform(self, X, y_site, y_diag=None, covariates=None, **aux):
        X = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        unknown = np.setdiff1d(np.unique(y_site), self.batch_levels_)
        seen = ~np.isin(y_site, unknown)
        new_unknown = np.setdiff1d(unknown, np.asarray(getattr(self, "unseen_sites_", []), dtype=int))
        if new_unknown.size:
            # no batch estimates exist for a site absent from the training fold: pass it through
            # unchanged (same policy as the affine harmonizer) and record it
            log.warning("transform: sites %s were not present when ComBat was fitted; left unchanged",
                        new_unknown.tolist())
            self.unseen_sites_ = sorted(set(getattr(self, "unseen_sites_", [])) | set(unknown.tolist()))
        out = X.copy()
        if seen.any():
            batch_idx = np.searchsorted(self.batch_levels_, y_site[seen])
            cov_seen = None if not self.covariate_keys or covariates is None else \
                {k: np.asarray(covariates[k], dtype=np.float64)[seen] for k in self.covariate_keys if k in covariates}
            mod = self._mod_mean(None if y_diag is None else np.asarray(y_diag)[seen], int(seen.sum()), cov_seen)
            adj = _apply_estimates(X[seen].T, batch_idx, self.grand_mean_, self.var_pooled_,
                                   self.gamma_star_, self.delta_star_, mod)
            out[seen] = adj.T
        if self.ref_batch is not None:
            ref_mask = y_site == int(self.ref_batch)
            out[ref_mask] = X[ref_mask]
        return out

    def describe(self):
        d = super().describe()
        d.update({"protect_diagnosis": self.protect_diagnosis, "eb": self.eb, "parametric": self.parametric,
                  "ref_batch": self.ref_batch, "uses_labels_at_transform": self.protect_diagnosis,
                  "covariate_keys": list(self.covariate_keys)})
        return d


class CovBatHarmonizer(Harmonizer):
    """
    CovBat (Chen et al. 2022), following the authors' reference code
    (github.com/andy1764/CovBat_Harmonization, R function ``covbat`` with its defaults):

    1. ComBat with empirical Bayes and the covariates (here neuroCombat);
    2. the ComBat residuals (adjusted data minus grand mean and covariate effects) are centred and
       scaled to unit variance per feature (``std.var = TRUE``) and decomposed by PCA;
    3. the first scores that explain more than ``percent.var = 0.95`` of the variance are harmonized by
       ComBat without covariates and without empirical Bayes (``score.eb = FALSE``);
    4. the scores are projected back, unscaled and the ComBat mean structure is added again.

    The reference code fits and adjusts one data set. For held-out subjects the same four steps are
    applied with every quantity (ComBat estimates, scaling, principal axes, score estimates) taken from
    the training fold. The scaling uses the n - 1 denominator, as R's ``prcomp(scale. = TRUE)``.
    """

    name = "CovBat"
    uses_labels_at_transform = True

    def __init__(
        self,
        protect_diagnosis: bool = True,
        var_threshold: float = 0.95,
        n_pcs: int | None = None,
        seed: int = 42,
        name: str | None = None,
    ):
        self.protect_diagnosis = protect_diagnosis
        self.var_threshold = var_threshold
        self.n_pcs = n_pcs
        self.seed = seed
        if name is not None:
            self.name = name

    def _residuals(self, X_c, y_diag, n):
        return X_c - (self.combat_.grand_mean_[None, :] + self.combat_._mod_mean(y_diag, n).T)

    def fit(self, X, y_site, y_diag=None, **aux):
        X = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        self.combat_ = ComBatHarmonizer(protect_diagnosis=self.protect_diagnosis).fit(X, y_site, y_diag)
        X_c = self.combat_.X_train_harmonized_
        resid = self._residuals(X_c, y_diag, X.shape[0])

        # std.var = TRUE: centre and scale every residual feature before the PCA
        self.center_ = resid.mean(axis=0)
        self.scale_ = resid.std(axis=0, ddof=1)
        self.scale_[self.scale_ < 1e-12] = 1.0
        R = (resid - self.center_) / self.scale_

        max_comp = min(R.shape[0] - 1, R.shape[1])
        n_comp = self.n_pcs
        if n_comp is None:
            probe = PCA(n_components=max_comp, svd_solver="full", random_state=self.seed).fit(R)
            cum = np.cumsum(probe.explained_variance_ratio_)
            # first component count whose cumulative share exceeds percent.var (R: which(cum > pv)[1])
            n_comp = int(np.argmax(cum > self.var_threshold) + 1) if np.any(cum > self.var_threshold) else max_comp
        n_comp = int(min(max(n_comp, 1), max_comp))
        self.pca_ = PCA(n_components=n_comp, svd_solver="full", random_state=self.seed).fit(R)
        scores = self.pca_.transform(R)

        # ComBat on the scores: batch only, no covariates, no empirical Bayes (score.eb = FALSE)
        self.score_combat_ = ComBatHarmonizer(protect_diagnosis=False, eb=False).fit(scores, y_site)
        scores_h = self.score_combat_.transform(scores, y_site)
        self.X_train_harmonized_ = X_c + ((scores_h - scores) @ self.pca_.components_) * self.scale_
        self.n_pcs_ = n_comp
        return self

    def transform(self, X, y_site, y_diag=None, **aux):
        X = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        X_c = self.combat_.transform(X, y_site, y_diag=y_diag)
        R = (self._residuals(X_c, y_diag, X.shape[0]) - self.center_) / self.scale_
        scores = self.pca_.transform(R)
        scores_h = self.score_combat_.transform(scores, y_site)
        return X_c + ((scores_h - scores) @ self.pca_.components_) * self.scale_

    def describe(self):
        d = super().describe()
        d.update({"protect_diagnosis": self.protect_diagnosis, "var_threshold": self.var_threshold,
                  "std_var": True, "n_pcs": getattr(self, "n_pcs_", self.n_pcs),
                  "uses_labels_at_transform": self.protect_diagnosis})
        return d


class ComBatGAMHarmonizer(Harmonizer):
    """
    ComBat-GAM (Pomponio et al. 2020) via ``neuroHarmonize``.

    Diagnosis is a covariate (same role as in ComBat). Age is a smooth term when
    it is passed through ``covariates``. Held-out subjects keep the fold-wise fit.

    Three details make the package usable inside cross-validation:

    * ``harmonizationApply`` rebuilds its site one-hot from the rows it is given,
      so a held-out fold that lacks a training site shifts every later site onto
      its neighbour's estimates. One stored training row per site is appended
      before the call and removed after it. The apply step is row-wise, so the
      anchors do not change the other rows.
    * The age spline cannot evaluate outside its training knots, so held-out ages
      are clipped to the training range.
    * Sites with fewer than two training scans are left unchanged, as in ComBat.
    """

    name = "ComBat-GAM"
    uses_labels_at_transform = True
    min_site_n = 2

    def __init__(self, protect_diagnosis: bool = True, name: str | None = None):
        self.protect_diagnosis = protect_diagnosis
        if name is not None:
            self.name = name

    def _covars(self, y_site, y_diag, age) -> pd.DataFrame:
        covars = pd.DataFrame({"SITE": np.asarray(y_site).astype(int)})
        if self.protect_diagnosis:
            if y_diag is None:
                raise ValueError("ComBat-GAM with protect_diagnosis=True needs y_diag")
            covars["diagnosis"] = np.asarray(y_diag).astype(int)
        if self.smooth_terms_:
            age = np.asarray(age, dtype=float)
            age = np.where(np.isfinite(age), age, self.age_fill_)
            covars["age"] = np.clip(age, *self.age_range_)
        return covars

    def fit(self, X, y_site, y_diag=None, covariates=None, **aux):
        try:
            from neuroHarmonize import harmonizationLearn
        except ImportError as exc:
            raise ImportError("ComBat-GAM needs the neuroHarmonize package (`pip install neuroHarmonize`)") from exc
        X = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        sites, counts = np.unique(y_site, return_counts=True)
        tiny = sites[counts < self.min_site_n]
        if tiny.size:
            log.warning("fit: sites %s have < %d training scans; ComBat-GAM leaves them unchanged",
                        tiny.tolist(), self.min_site_n)
        keep = ~np.isin(y_site, tiny)
        diag = None if y_diag is None else np.asarray(y_diag).astype(int)[keep]
        age = np.asarray((covariates or {}).get("age", np.full(len(y_site), np.nan)), dtype=float)[keep]
        self.smooth_terms_ = ["age"] if np.isfinite(age).sum() >= 10 else []
        if self.smooth_terms_:
            self.age_fill_ = float(np.nanmedian(age))
            self.age_range_ = (float(np.nanmin(age)), float(np.nanmax(age)))
        covars = self._covars(y_site[keep], diag, age)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_, _ = harmonizationLearn(X[keep].copy(), covars, smooth_terms=self.smooth_terms_)
        self.site_levels_ = np.unique(y_site[keep])
        first = np.array([np.flatnonzero(y_site[keep] == s)[0] for s in self.site_levels_])
        self.anchor_X_ = X[keep][first]
        self.anchor_covars_ = covars.iloc[first].reset_index(drop=True)
        return self

    def transform(self, X, y_site, y_diag=None, covariates=None, **aux):
        from neuroHarmonize import harmonizationApply
        X = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        seen = np.isin(y_site, self.site_levels_)
        out = X.copy()
        if not seen.any():
            return out
        age = np.asarray((covariates or {}).get("age", np.full(len(y_site), np.nan)), dtype=float)
        diag = None if y_diag is None else np.asarray(y_diag)[seen]
        covars = pd.concat([self._covars(y_site[seen], diag, age[seen]), self.anchor_covars_], ignore_index=True)
        rows = np.vstack([X[seen], self.anchor_X_])
        adj = harmonizationApply(rows, covars, self.model_)
        out[seen] = np.asarray(adj, dtype=np.float64)[: int(seen.sum())]
        return out

    def describe(self):
        d = super().describe()
        d.update({"protect_diagnosis": self.protect_diagnosis,
                  "smooth_terms": getattr(self, "smooth_terms_", []),
                  "uses_labels_at_transform": self.protect_diagnosis})
        return d

