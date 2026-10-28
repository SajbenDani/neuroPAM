"""Tests for the new baselines, permutation null, federated check and Nadeau-Bengio."""

import numpy as np
import pytest

from vae_harmonization.analyses.federated import federated_equality
from vae_harmonization.analyses.overlap import covariate_overlap
from vae_harmonization.baselines.coral import CORALHarmonizer
from vae_harmonization.baselines.connectome import (
    corr_to_vec, n_rois_from_n_edges, spd_floor, vec_to_corr, vecs_to_corrs,
)
from vae_harmonization.baselines.mmd_affine import MMDAffineHarmonizer
from vae_harmonization.metrics import permutation_null_oos
from vae_harmonization.stats import nadeau_bengio_corrected_ttest


def test_connectome_roundtrip():
    rng = np.random.default_rng(0)
    n = 8
    A = rng.normal(size=(n, n))
    C = spd_floor(A @ A.T / n + np.eye(n))
    # rescale to a correlation
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    z = corr_to_vec(C)
    assert n_rois_from_n_edges(z.size) == n
    C2 = vec_to_corr(z, n)
    assert np.allclose(C, C2, atol=1e-6)


def test_coral_aligns_means_and_runs_oos(cohort):
    h = CORALHarmonizer(target_site=0).fit(cohort.Z, cohort.y_site)
    Zh = h.transform(cohort.Z, cohort.y_site)
    assert np.allclose(Zh[cohort.y_site == 0], cohort.Z[cohort.y_site == 0])
    mu0 = Zh[cohort.y_site == 0].mean(0)
    for s in (1, 2, 3):
        assert np.allclose(Zh[cohort.y_site == s].mean(0), mu0, atol=1e-6)
    idx = np.arange(len(cohort.y_site))
    h2 = CORALHarmonizer(target_site=0).fit(cohort.Z[idx[::2]], cohort.y_site[idx[::2]])
    out = h2.transform(cohort.Z[idx[1::2]], cohort.y_site[idx[1::2]])
    assert out.shape == cohort.Z[idx[1::2]].shape and np.all(np.isfinite(out))


def test_mmd_affine_smoke(cohort):
    h = MMDAffineHarmonizer(target_site=0, steps=8, seed=0).fit(cohort.Z, cohort.y_site)
    out = h.transform(cohort.Z, cohort.y_site)
    assert out.shape == cohort.Z.shape and np.all(np.isfinite(out))
    assert np.allclose(out[cohort.y_site == 0], cohort.Z[cohort.y_site == 0])


def test_transform_ignores_fit_time_covariates(cohort):
    from vae_harmonization.harmonization import ProtectedAffineHarmonizer
    age = np.linspace(20, 40, len(cohort.y_site))
    cov = {"age": age}
    h = ProtectedAffineHarmonizer(target_site=0, protected=("diagnosis", "age"))
    h.fit(cohort.Z, cohort.y_site, y_diag=cohort.y_diag, covariates=cov)
    out = h.transform(cohort.Z, cohort.y_site, covariates=cov)
    assert out.shape == cohort.Z.shape
    assert np.isfinite(out).all()


def test_tost_uses_corrected_variance():
    from vae_harmonization.stats import tost_paired
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.66, 0.05, 50), rng.normal(0.66, 0.05, 50)
    naive = tost_paired(a, b, margin=0.02)
    nb = tost_paired(a, b, margin=0.02, variance_correction=1 / 50 + 77 / 700)
    assert nb["ci90_high"] - nb["ci90_low"] > 2 * (naive["ci90_high"] - naive["ci90_low"])
    assert nb["p_tost"] > naive["p_tost"]


def test_combat_gam_rows_do_not_depend_on_other_sites():
    pytest.importorskip("neuroHarmonize")
    from vae_harmonization.baselines.combat import ComBatGAMHarmonizer
    rng = np.random.default_rng(1)
    n, p = 120, 4
    site = np.repeat([0, 1, 2], n // 3)
    X = rng.normal(size=(n, p)) + site[:, None] * 0.8
    y = rng.integers(0, 2, n)
    age = rng.uniform(10, 40, n)
    h = ComBatGAMHarmonizer().fit(X, site, y, covariates={"age": age})
    full = h.transform(X, site, y_diag=y, covariates={"age": age})
    keep = site != 1                                   # a held-out fold without site 1
    part = h.transform(X[keep], site[keep], y_diag=y[keep], covariates={"age": age[keep] + 100})
    ref = h.transform(X[keep], site[keep], y_diag=y[keep], covariates={"age": np.clip(age[keep] + 100, 0, age.max())})
    assert np.isfinite(part).all()
    assert np.allclose(part, ref)                      # ages outside the training range are clipped
    same = h.transform(X[keep], site[keep], y_diag=y[keep], covariates={"age": age[keep]})
    assert np.allclose(same, full[keep])               # site columns stay aligned


def test_estimator_variants_keep_their_covariance():
    from vae_harmonization.harmonization import make_variant
    for variant, expected in (("ours-lw", "lw"), ("ours-diag", "diagonal"), ("ours-emp", "empirical"),
                              ("ours", "pooled")):
        h = make_variant(variant, 0, covariance="pooled", min_cov_samples=10)
        assert h.covariance == expected, variant


def test_federated_equality(cohort):
    rep = federated_equality(cohort.Z, cohort.y_site, cohort.y_diag, target_site=0, covariance="empirical")
    assert rep["agree"] and rep["max_abs_diff"] < 1e-8


def test_permutation_null_contains_observed():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 6))
    y = np.repeat(np.arange(4), 20)
    from vae_harmonization.metrics import gamma_from_sigma, median_heuristic_bandwidth
    g = gamma_from_sigma(median_heuristic_bandwidth(X, seed=0))
    null = permutation_null_oos(X, y, gamma=g, min_site_size=5, n_perm=40, seed=0)
    assert "icc" in null and "mmd2_unbiased" in null
    assert 0 <= null["icc"]["percentile"] <= 1
    assert null["icc"]["n_perm"] == 40


def test_nadeau_bengio_identical_scores_not_significant():
    a = np.linspace(0.6, 0.7, 20)
    r = nadeau_bengio_corrected_ttest(a, a, n_train=80, n_test=20)
    assert r["p_value"] == 1.0 and abs(r["mean_diff"]) < 1e-15


def test_covariate_overlap_binary_and_continuous():
    y = np.repeat(np.arange(3), 30)
    age = np.concatenate([np.random.default_rng(0).normal(10 + 8 * s, 2, 30) for s in range(3)])
    sex = np.concatenate([np.random.default_rng(1).binomial(1, 0.3 + 0.2 * s, 30) for s in range(3)])
    rep = covariate_overlap({"age": age, "sex": sex}, y, site_names={0: "A", 1: "B", 2: "C"})
    assert 0 <= rep["covariates"]["age"]["mean_overlap"] <= 1
    assert "rate" in rep["covariates"]["sex"]["per_site"]["0"]
