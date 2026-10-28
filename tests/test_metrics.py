"""Tests for distribution-shift metrics (MMD, ICC, eta^2)."""

import numpy as np
import pytest

from vae_harmonization.metrics import (
    KERNEL_FLOOR,
    compute_eta_squared,
    compute_icc,
    compute_mmd,
    compute_mmd_ci,
    compute_site_pairwise_mmd,
    gamma_from_sigma,
    median_heuristic_bandwidth,
    mmd_bandwidth_sweep,
)


@pytest.fixture
def two_samples(rng):
    X = rng.normal(size=(120, 16))
    Y = rng.normal(size=(100, 16)) + 1.0
    return X, Y


def test_median_heuristic_is_finite_and_scales(two_samples):
    X, Y = two_samples
    s = median_heuristic_bandwidth(X, Y)
    assert np.isfinite(s) and s > 0
    # scaling the data by c scales sigma by c
    s2 = median_heuristic_bandwidth(3.0 * X, 3.0 * Y)
    assert np.isclose(s2, 3.0 * s, rtol=1e-6)
    assert np.isclose(gamma_from_sigma(s), 1.0 / (2 * s * s))


def test_mmd_zero_under_h0_and_positive_under_h1(rng):
    X = rng.normal(size=(200, 8))
    Y = rng.normal(size=(200, 8))
    g = gamma_from_sigma(median_heuristic_bandwidth(X, Y))
    m0 = compute_mmd(X, Y, gamma=g, unbiased=True)
    assert abs(m0) < 0.01  # unbiased estimate is centred on zero
    m1 = compute_mmd(X, Y + 1.5, gamma=g, unbiased=True)
    assert m1 > 0.1 and m1 > m0
    # biased estimator is always >= 0 and >= unbiased for the same data
    mb = compute_mmd(X, Y, gamma=g, unbiased=False)
    assert mb >= 0.0


def test_fixed_bandwidth_is_reproducible_across_conditions(two_samples):
    X, Y = two_samples
    g = 0.03
    a, det_a = compute_mmd(X, Y, gamma=g, return_details=True)
    b, det_b = compute_mmd(X, Y, gamma=g, return_details=True)
    assert a == b and det_a["gamma"] == det_b["gamma"] == g
    assert not det_a["kernel_floor"]


def test_gamma_one_over_d_underflows_in_high_dimension(rng):
    """Reproduce the failure of an earlier version: gamma=1/d on wide latents."""
    d = 64
    X = rng.normal(size=(150, d)) * 3.0
    Y = rng.normal(size=(140, d)) * 3.0 + 0.2
    m, det = compute_mmd(X, Y, gamma=1.0 / d, unbiased=False, return_details=True)
    # With gamma=1/d and pairwise sq-distances ~ 2*9*64 = 1152, k ~ exp(-18): vanished
    assert det["kernel_floor"]
    assert det["mean_cross_kernel"] < KERNEL_FLOOR
    # the biased value collapses to 1/n + 1/m regardless of the shift
    assert np.isclose(m, 1 / 150 + 1 / 140, atol=1e-4)
    # the median heuristic does not underflow on the same data
    m2, det2 = compute_mmd(X, Y, gamma=None, unbiased=True, return_details=True)
    assert not det2["kernel_floor"]


def test_mmd_ci_contains_point(two_samples):
    X, Y = two_samples
    g = gamma_from_sigma(median_heuristic_bandwidth(X, Y))
    ci = compute_mmd_ci(X, Y, gamma=g, n_boot=50, seed=0)
    assert ci["ci_low"] <= ci["mmd2"] <= ci["ci_high"] or np.isclose(ci["ci_low"], ci["mmd2"])


def test_site_pairwise_mmd_shared_bandwidth_and_exclusion(cohort):
    res_pre = compute_site_pairwise_mmd(cohort.Z, cohort.y_site, min_site_size=5)
    g = res_pre["gamma_used"]
    assert g is not None and res_pre["n_pairs"] == 6
    assert res_pre["kernel_floor_pairs"] == 0
    # perfectly harmonized data (all sites identical distribution) -> ~0 with SAME gamma
    Z_h = np.random.default_rng(0).normal(size=cohort.Z.shape)
    res_post = compute_site_pairwise_mmd(Z_h, cohort.y_site, gamma=g, min_site_size=5)
    assert res_post["gamma_used"] == g
    assert res_post["mean_pairwise_mmd"] < res_pre["mean_pairwise_mmd"]
    # tiny sites are excluded
    y = cohort.y_site.copy()
    y[:3] = 99
    res_ex = compute_site_pairwise_mmd(cohort.Z, y, gamma=g, min_site_size=5)
    assert res_ex["excluded_sites"] == [99]


def test_bandwidth_sweep_preserves_ranking(cohort):
    g = compute_site_pairwise_mmd(cohort.Z, cohort.y_site)["gamma_used"]
    Z_h = cohort.Z.copy()
    for s in np.unique(cohort.y_site):
        m = cohort.y_site == s
        Z_h[m] -= Z_h[m].mean(axis=0)  # mean-centre each site: reduces shift
    sweep = mmd_bandwidth_sweep({"pre": cohort.Z, "post": Z_h}, cohort.y_site, g)
    for row in sweep["mean_pairwise_mmd"].values():
        assert row["post"] < row["pre"]


def test_eta_squared_bounds_and_behaviour(rng):
    groups = np.repeat([0, 1, 2], 50)
    x_same = rng.normal(size=150)
    assert 0.0 <= compute_eta_squared(x_same, groups) < 0.1
    x_sep = groups * 10.0 + rng.normal(scale=0.1, size=150)
    assert compute_eta_squared(x_sep, groups) > 0.99


def test_epsilon_squared_is_centred_under_null(rng):
    from vae_harmonization.metrics import compute_epsilon_squared, eta_squared_null_expectation
    # many small groups in a small sample: eta^2 is badly biased, epsilon^2 is not
    groups = np.repeat(np.arange(17), 5)          # G=17, N=85
    etas, epss = [], []
    for _ in range(300):
        x = rng.normal(size=85)
        etas.append(compute_eta_squared(x, groups))
        epss.append(compute_epsilon_squared(x, groups))
    assert abs(np.mean(etas) - eta_squared_null_expectation(85, 17)) < 0.03
    assert abs(np.mean(epss)) < 0.03
    # real effect still detected
    x = groups * 1.0 + rng.normal(size=85)
    assert compute_epsilon_squared(x, groups) > 0.8


def test_icc_behaviour(rng):
    groups = np.repeat([0, 1, 2, 3], 40)
    X_same = rng.normal(size=(160, 5))
    assert compute_icc(X_same, groups)["icc_mean"] < 0.1
    X_shift = X_same + groups[:, None] * 5.0
    assert compute_icc(X_shift, groups)["icc_mean"] > 0.9
    assert np.isnan(compute_icc(X_same[:10], np.zeros(10))["icc_mean"])
