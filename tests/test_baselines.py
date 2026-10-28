"""Tests for the fit/transform baselines."""

import numpy as np
import pytest

from vae_harmonization.baselines import (
    ComBatHarmonizer,
    CovBatHarmonizer,
)


def test_combat_apply_step_reproduces_reference_output(cohort):
    h = ComBatHarmonizer(protect_diagnosis=True).fit(cohort.Z, cohort.y_site, cohort.y_diag)
    Xt = h.transform(cohort.Z, cohort.y_site, y_diag=cohort.y_diag)
    assert np.allclose(Xt, h.X_train_harmonized_, atol=1e-6)
    h0 = ComBatHarmonizer(protect_diagnosis=False).fit(cohort.Z, cohort.y_site)
    assert np.allclose(h0.transform(cohort.Z, cohort.y_site), h0.X_train_harmonized_, atol=1e-6)


def test_combat_reduces_site_mean_differences_out_of_sample(cohort):
    idx = np.arange(len(cohort.y_site))
    tr, te = idx[::2], idx[1::2]
    h = ComBatHarmonizer().fit(cohort.Z[tr], cohort.y_site[tr], cohort.y_diag[tr])
    Zte = h.transform(cohort.Z[te], cohort.y_site[te], y_diag=cohort.y_diag[te])
    def spread(Z, s):
        means = np.stack([Z[s == k].mean(0) for k in np.unique(s)])
        return means.var(0).mean()
    assert spread(Zte, cohort.y_site[te]) < 0.2 * spread(cohort.Z[te], cohort.y_site[te])
    # a site absent from the training fold has no estimates: passed through unchanged and recorded
    Zu = h.transform(cohort.Z[te], np.full(len(te), 42), y_diag=cohort.y_diag[te])
    assert np.allclose(Zu, cohort.Z[te]) and h.unseen_sites_ == [42]


def test_combat_ref_batch_keeps_reference_untouched(cohort):
    h = ComBatHarmonizer(ref_batch=0).fit(cohort.Z, cohort.y_site, cohort.y_diag)
    Zt = h.transform(cohort.Z, cohort.y_site, y_diag=cohort.y_diag)
    m = cohort.y_site == 0
    assert np.allclose(Zt[m], cohort.Z[m])
    assert np.allclose(Zt, h.X_train_harmonized_, atol=1e-6)


def _age_confounded_with_site(cohort, beta_scale=0.8, seed=3):
    """Ages that differ by site, sexes at random, and a planted linear age effect on the features."""
    rng = np.random.default_rng(seed)
    n, d = cohort.Z.shape
    age = 10.0 + 5.0 * cohort.y_site + rng.normal(scale=2.0, size=n)
    sex = rng.integers(0, 2, size=n).astype(float)
    beta = beta_scale * rng.normal(size=d)
    Z = cohort.Z + np.outer(age - age.mean(), beta) / 5.0
    return Z, {"age": age, "sex": sex}, beta / 5.0


def test_label_free_combat_reproduces_reference_output_without_diagnosis(cohort):
    Z, cov, _ = _age_confounded_with_site(cohort)
    h = ComBatHarmonizer(protect_diagnosis=False, covariate_keys=("age", "sex")).fit(Z, cohort.y_site, covariates=cov)
    assert np.allclose(h.transform(Z, cohort.y_site, covariates=cov), h.X_train_harmonized_, atol=1e-6)
    assert not h.describe()["uses_labels_at_transform"]
    # the apply step is row-wise: a subset is adjusted exactly as in the full call
    sub = np.arange(len(cohort.y_site))[::3]
    part = h.transform(Z[sub], cohort.y_site[sub], covariates={k: v[sub] for k, v in cov.items()})
    assert np.allclose(part, h.X_train_harmonized_[sub], atol=1e-6)
    with pytest.raises(ValueError):
        h.transform(Z, cohort.y_site)                       # the covariates are needed at transform


def test_label_free_combat_keeps_an_age_effect_that_is_confounded_with_site(cohort):
    Z, cov, beta = _age_confounded_with_site(cohort)
    age = cov["age"]

    def slope(X):
        a = age - age.mean()
        return (a @ (X - X.mean(0))) / (a @ a)

    with_age = ComBatHarmonizer(protect_diagnosis=False, covariate_keys=("age", "sex")).fit(
        Z, cohort.y_site, covariates=cov).X_train_harmonized_
    without = ComBatHarmonizer(protect_diagnosis=False).fit(Z, cohort.y_site).X_train_harmonized_
    err_with = np.linalg.norm(slope(with_age) - beta) / np.linalg.norm(beta)
    err_without = np.linalg.norm(slope(without) - beta) / np.linalg.norm(beta)
    assert err_with < 0.5 * err_without


def test_label_free_combat_fills_missing_covariates_with_the_training_mean(cohort):
    Z, cov, _ = _age_confounded_with_site(cohort)
    gaps = {k: v.copy() for k, v in cov.items()}
    gaps["age"][:7] = np.nan
    gaps["sex"][:7] = np.nan
    h = ComBatHarmonizer(protect_diagnosis=False, covariate_keys=("age", "sex")).fit(Z, cohort.y_site, covariates=gaps)
    assert np.all(np.isfinite(h.X_train_harmonized_))
    assert np.allclose(h.covariate_fill_, [np.nanmean(gaps["age"]), np.nanmean(gaps["sex"])])
    assert np.allclose(h.transform(Z, cohort.y_site, covariates=gaps), h.X_train_harmonized_, atol=1e-6)


def test_covbat_in_sample_consistency_and_covariance_alignment(cohort):
    h = CovBatHarmonizer(var_threshold=0.95).fit(cohort.Z, cohort.y_site, cohort.y_diag)
    Zt = h.transform(cohort.Z, cohort.y_site, y_diag=cohort.y_diag)
    assert np.allclose(Zt, h.X_train_harmonized_, atol=1e-6)
    assert 1 <= h.n_pcs_ <= cohort.Z.shape[1]
    # per-site covariance spread should shrink relative to raw
    def cov_spread(Z, s):
        covs = np.stack([np.cov(Z[s == k], rowvar=False) for k in np.unique(s)])
        return covs.var(0).mean()
    assert cov_spread(Zt, cohort.y_site) < cov_spread(cohort.Z, cohort.y_site)
    # out-of-sample path runs
    idx = np.arange(len(cohort.y_site))
    h2 = CovBatHarmonizer().fit(cohort.Z[idx[::2]], cohort.y_site[idx[::2]], cohort.y_diag[idx[::2]])
    out = h2.transform(cohort.Z[idx[1::2]], cohort.y_site[idx[1::2]], y_diag=cohort.y_diag[idx[1::2]])
    assert out.shape == cohort.Z[idx[1::2]].shape and np.all(np.isfinite(out))
