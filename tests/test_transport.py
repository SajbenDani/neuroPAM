"""Tests for Gaussian-OT primitives and the protected affine harmonizer."""

import numpy as np

from vae_harmonization.harmonization.affine import conditional_site_means


def test_conditional_site_means_recover_site_offsets_under_confounding():
    rng = np.random.default_rng(3)
    d, sizes = 5, (60, 60, 60)
    offsets = rng.normal(scale=2.0, size=(3, d))          # true site effects
    slope = rng.normal(size=d)                             # shared age effect
    Z, ys, age = [], [], []
    for s, n in enumerate(sizes):
        a = rng.normal(loc=10 + 10 * s, scale=2.0, size=n)  # age differs strongly by site
        Z.append(offsets[s] + np.outer(a, slope) + 0.1 * rng.normal(size=(n, d)))
        ys += [s] * n
        age.append(a)
    Z, ys, age = np.vstack(Z), np.array(ys), np.concatenate(age)
    means, B, c_mean = conditional_site_means(Z, ys, age[:, None])
    marg = {s: Z[ys == s].mean(0) for s in range(3)}
    # marginal differences contain the age-driven part, conditional ones only the site offsets
    true_diff = offsets[1] - offsets[0]
    assert np.allclose(means[1] - means[0], true_diff, atol=0.15)
    assert not np.allclose(marg[1] - marg[0], true_diff, atol=1.0)
    assert np.allclose(B[0], slope, atol=0.05)
    # sporadic missing covariates are imputed with the site mean, so intercepts stay unbiased
    age_nan = age.copy(); age_nan[::7] = np.nan
    means2, _, _ = conditional_site_means(Z, ys, age_nan[:, None])
    assert np.allclose(means2[2] - means2[0], offsets[2] - offsets[0], atol=0.2)


def test_conditional_site_means_structural_missingness_uses_all_subjects():
    """A severity score that exists for cases only must not make the intercepts case-only estimates."""
    rng = np.random.default_rng(5)
    d, n = 4, 80
    offsets = rng.normal(scale=2.0, size=(2, d))
    b_dx, b_sev = rng.normal(size=d), rng.normal(size=d)
    Z, ys, dx, sev = [], [], [], []
    for s in range(2):
        y = rng.integers(0, 2, n)
        sv = np.where(y == 1, rng.normal(12 + 4 * s, 2, n), np.nan)       # cases only, site-dependent level
        Z.append(offsets[s] + np.outer(y, b_dx) + np.outer(np.nan_to_num(sv), b_sev) + 0.1 * rng.normal(size=(n, d)))
        ys += [s] * n; dx.append(y); sev.append(sv)
    Z, ys, dx, sev = np.vstack(Z), np.array(ys), np.concatenate(dx), np.concatenate(sev)
    means, B, _ = conditional_site_means(Z, ys, np.column_stack([dx, sev]))
    # the site difference must not contain the severity-level difference (4 * b_sev) nor the case-mix difference
    assert np.allclose(means[1] - means[0], offsets[1] - offsets[0], atol=0.35)
    assert np.allclose(B[1], b_sev, atol=0.1)
import pytest

from vae_harmonization.harmonization import (
    IdentityHarmonizer,
    ProtectedAffineHarmonizer,
    bures_wasserstein_barycenter,
    bures_wasserstein_distance2,
    bures_wasserstein_map,
    cholesky_map,
    estimate_covariance,
    expected_squared_displacement,
    fit_diagnosis_axis,
    geodesic_map,
    make_variant,
    orthonormal_protected_basis,
    zca_map,
)


def _spd(rng, d):
    A = rng.normal(size=(d, d))
    return A @ A.T / d + 0.3 * np.eye(d)


@pytest.fixture
def cov_pair(rng):
    return _spd(rng, 6), _spd(rng, 6)


def test_all_maps_match_covariance(cov_pair):
    Cs, Ct = cov_pair
    for f in (bures_wasserstein_map, cholesky_map, zca_map):
        A = f(Cs, Ct)
        assert np.allclose(A @ Cs @ A.T, Ct, atol=1e-8), f.__name__


def test_bw_map_is_symmetric_pd_and_minimal_displacement(cov_pair):
    """Theorem 3.1: among A with A Cs A^T = Ct, the BW map minimises tr((A-I) Cs (A-I)^T)."""
    Cs, Ct = cov_pair
    B = bures_wasserstein_map(Cs, Ct)
    assert np.allclose(B, B.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(B) > 0)
    disp_bw = expected_squared_displacement(B, Cs)
    for f in (cholesky_map, zca_map):
        assert disp_bw <= expected_squared_displacement(f(Cs, Ct), Cs) + 1e-9
    # random orthogonal perturbations A = B_any @ Q-type maps also satisfy the constraint:
    rng = np.random.default_rng(0)
    L_s = np.linalg.cholesky(Cs)
    L_t = np.linalg.cholesky(Ct)
    for _ in range(20):
        Q, _ = np.linalg.qr(rng.normal(size=Cs.shape))
        A = L_t @ Q @ np.linalg.inv(L_s)
        assert np.allclose(A @ Cs @ A.T, Ct, atol=1e-8)
        assert disp_bw <= expected_squared_displacement(A, Cs) + 1e-9
    # displacement equals the BW distance^2 between the two centred Gaussians
    assert np.isclose(disp_bw, bures_wasserstein_distance2(Cs, Ct), atol=1e-8)


def test_geodesic_endpoints_and_monotone_alignment(cov_pair):
    Cs, Ct = cov_pair
    B = bures_wasserstein_map(Cs, Ct)
    assert np.allclose(geodesic_map(B, 0.0), np.eye(6))
    assert np.allclose(geodesic_map(B, 1.0), B)
    d_prev = np.inf
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        Bl = geodesic_map(B, lam)
        d = bures_wasserstein_distance2(Bl @ Cs @ Bl.T, Ct)
        assert d <= d_prev + 1e-9
        d_prev = d
    assert d_prev < 1e-8


def test_barycenter_fixed_point(rng):
    covs = [_spd(rng, 4) for _ in range(3)]
    S, info = bures_wasserstein_barycenter(covs, n_iter=200, tol=1e-10)
    assert info["converged"]
    # first-order optimality: sum_i w_i (S^{1/2} C_i S^{1/2})^{1/2} = S
    from vae_harmonization.harmonization.transport import sqrtm_psd
    S_h = sqrtm_psd(S)
    lhs = sum(sqrtm_psd(S_h @ C @ S_h) for C in covs) / 3
    assert np.allclose(lhs, S, atol=1e-6)
    # the barycenter of identical matrices is that matrix
    S2, _ = bures_wasserstein_barycenter([covs[0]] * 3)
    assert np.allclose(S2, covs[0], atol=1e-8)


def test_estimate_covariance_returns_spd_and_shrinkage(rng):
    X = rng.normal(size=(12, 8))
    C, delta = estimate_covariance(X, "lw")
    assert 0 <= delta <= 1 and np.all(np.linalg.eigvalsh(C) > 0)
    C2, d2 = estimate_covariance(X, "empirical", ridge=1e-3)
    assert d2 == 0.0 and np.all(np.linalg.eigvalsh(C2) > 0)


def test_diagnosis_axis_recovers_planted_direction(cohort):
    w = fit_diagnosis_axis(cohort.Z, cohort.y_diag)
    assert abs(np.dot(w, cohort.w_true)) > 0.7


def test_orthonormal_basis_drops_collinear():
    a = np.array([1.0, 0, 0])
    b = np.array([0.95, 0.31, 0]) / np.linalg.norm([0.95, 0.31, 0])
    c = np.array([0, 0, 1.0])
    W, info = orthonormal_protected_basis([a, b, c, None], collinearity_threshold=0.8)
    assert info["k"] == 2 and info["dropped"] == [1, 3]
    assert np.allclose(W.T @ W, np.eye(2))


@pytest.mark.parametrize("covariance,tol", [("empirical", 1e-4), ("lw", 0.35), ("pooled", 0.4), ("diagonal", 0.4)])
def test_harmonizer_aligns_moments_and_protects_axis(cohort, covariance, tol):
    # marginal site means -> exact first/second-moment alignment on the complement
    h = make_variant("ours-marginal", target_site=cohort.target_site, covariance=covariance)
    h.fit(cohort.Z, cohort.y_site, y_diag=cohort.y_diag)
    Zh = h.transform(cohort.Z, cohort.y_site)
    W_perp = h.W_perp_
    Pt = Zh[cohort.y_site == 0] @ W_perp
    for s in (1, 2, 3):
        Ps = Zh[cohort.y_site == s] @ W_perp
        assert np.allclose(Ps.mean(0), Pt.mean(0), atol=1e-8)
        # exact identity on the fitted covariances: B C_s B = C_hub
        tr = h.site_transforms_[s]
        assert np.allclose(tr.B @ h.site_stats_[s]["C"] @ tr.B.T, h.hub_C_, atol=1e-8)
        # the empirical covariance after transport is close to the hub's empirical one (exactly equal
        # for the empirical estimator; the synthetic sites have unrelated covariances, so shrinkage toward
        # any target leaves a residual)
        Cs_emp = np.cov(Ps, rowvar=False)
        Ct_emp = np.cov(Pt, rowvar=False)
        assert np.linalg.norm(Cs_emp - Ct_emp) / np.linalg.norm(Ct_emp) < tol
    # protected coordinate: within-site diagnosis gap is unchanged by the transform
    a_pre = h.protected_coordinates(cohort.Z)[:, 0]
    a_post = h.protected_coordinates(Zh)[:, 0]
    for s in range(4):
        m = cohort.y_site == s
        gap_pre = a_pre[m & (cohort.y_diag == 1)].mean() - a_pre[m & (cohort.y_diag == 0)].mean()
        gap_post = a_post[m & (cohort.y_diag == 1)].mean() - a_post[m & (cohort.y_diag == 0)].mean()
        assert np.isclose(gap_pre, gap_post, atol=1e-10)
    # target site untouched
    assert np.allclose(Zh[cohort.y_site == 0], cohort.Z[cohort.y_site == 0])


def test_pooled_shrinkage_is_near_identity_map_under_shared_covariance():
    """When all sites share one covariance, the pooled target makes small-site maps close to identity,
    while Ledoit-Wolf (identity target) flattens the spectrum and moves points further."""
    from vae_harmonization.harmonization.transport import pooled_within_covariance, shrinkage_intensity

    rng = np.random.default_rng(0)
    d = 20
    ev = np.geomspace(10.0, 0.5, d)                       # strongly anisotropic spectrum (as PCA scores)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    Sigma = (Q * ev) @ Q.T
    sizes = (150, 60, 25, 20)
    Z = np.vstack([rng.multivariate_normal(np.zeros(d), Sigma, size=n) for n in sizes])
    ys = np.repeat(np.arange(4), sizes)
    T = pooled_within_covariance(Z, ys, exclude=3)
    assert T.shape == (d, d) and np.allclose(T, T.T)
    # intensity is in [0, 1] and grows as the site shrinks
    d_small = shrinkage_intensity(Z[ys == 3], T)
    d_large = shrinkage_intensity(Z[ys == 0], pooled_within_covariance(Z, ys, exclude=0))
    assert 0 <= d_large < d_small <= 1
    disp = {}
    for cov in ("pooled", "lw"):
        h = ProtectedAffineHarmonizer(target_site=0, protected=(), covariance=cov, min_cov_samples=10)
        h.fit(Z, ys)
        B = h.site_transforms_[3].B
        disp[cov] = expected_squared_displacement(B, Sigma)
    assert disp["pooled"] < 0.5 * disp["lw"]


def test_estimate_covariance_methods():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 6)) @ np.diag([3, 2, 1, 1, 0.5, 0.2])
    S = np.cov(X, rowvar=False)
    C_emp, d0 = estimate_covariance(X, "empirical")
    assert d0 == 0.0 and np.allclose(C_emp, S, atol=1e-5)
    C_diag, dd = estimate_covariance(X, "diagonal")
    assert 0 < dd <= 1 and np.allclose(np.diag(C_diag), np.diag(S), atol=1e-5)
    assert np.all(np.abs(C_diag - np.diag(np.diag(C_diag))) <= np.abs(S - np.diag(np.diag(S))) + 1e-9)
    T = np.eye(6)
    C_pool, dp = estimate_covariance(X, "pooled", target=T)
    assert 0 <= dp <= 1 and np.allclose(C_pool, (1 - dp) * S + dp * T, atol=1e-5)
    with pytest.raises(ValueError):
        estimate_covariance(X, "pooled")
    with pytest.raises(ValueError):
        estimate_covariance(X, "nope")


def test_transform_is_out_of_sample_capable(cohort):
    idx = np.arange(len(cohort.y_site))
    tr, te = idx[::2], idx[1::2]
    h = make_variant("ours", target_site=0).fit(cohort.Z[tr], cohort.y_site[tr], y_diag=cohort.y_diag[tr])
    Zte = h.transform(cohort.Z[te], cohort.y_site[te])
    assert Zte.shape == cohort.Z[te].shape and np.all(np.isfinite(Zte))
    # applying the fitted transform twice is NOT identity (it is affine, not a projection),
    # but the same input always gives the same output
    assert np.allclose(Zte, h.transform(cohort.Z[te], cohort.y_site[te]))


def test_small_site_gets_mean_shift_only(small_site_cohort):
    c = small_site_cohort
    h = make_variant("ours-marginal", target_site=0, min_cov_samples=10).fit(c.Z, c.y_site, y_diag=c.y_diag)
    assert h.mean_shift_only_sites_ == [2]
    assert h.site_transforms_[2].B is None
    Zh = h.transform(c.Z, c.y_site)
    m = c.y_site == 2
    # mean on complement matches target
    assert np.allclose((Zh[m] @ h.W_perp_).mean(0), (Zh[c.y_site == 0] @ h.W_perp_).mean(0), atol=1e-8)


def test_conditional_means_keep_site_confounded_covariate(cohort):
    """Age differs by site and has a latent slope: marginal means erase it, conditional means keep it."""
    rng = np.random.default_rng(11)
    Z, ys = cohort.Z.copy(), cohort.y_site
    age_full = np.empty(len(ys))
    for s in np.unique(ys):
        age_full[ys == s] = rng.normal(10 + 8 * s, 1.5, size=(ys == s).sum())
    slope = np.zeros(Z.shape[1]); slope[-1] = 0.3
    Z = Z + np.outer(age_full - age_full.mean(), slope)
    cov = {"age": age_full}
    r = {}
    for mm in ("conditional", "marginal"):
        h = make_variant("ours", target_site=cohort.target_site, protected=("diagnosis", "age"), mean_model=mm)
        Zh = h.fit(Z, ys, y_diag=cohort.y_diag, covariates=cov).transform(Z, ys)
        r[mm] = abs(np.corrcoef(Zh[:, -1], age_full)[0, 1])
        # per-site constant shifts never change within-site diagnosis gaps on the protected axis
        a_pre, a_post = h.protected_coordinates(Z)[:, 0], h.protected_coordinates(Zh)[:, 0]
        for s in np.unique(ys):
            m = ys == s
            g0 = a_pre[m & (cohort.y_diag == 1)].mean() - a_pre[m & (cohort.y_diag == 0)].mean()
            g1 = a_post[m & (cohort.y_diag == 1)].mean() - a_post[m & (cohort.y_diag == 0)].mean()
            assert np.isclose(g0, g1, atol=1e-8)
    assert abs(np.corrcoef(Z[:, -1], age_full)[0, 1]) > 0.8
    assert r["conditional"] > 0.6
    assert r["marginal"] < r["conditional"] - 0.2
    assert h.describe()["mean_model"] == "marginal"


def test_variants_construct_and_run(cohort):
    for v in ("ours", "full-affine", "mean-shift", "cholesky", "zca", "ours-auto", "barycenter"):
        h = make_variant(v, target_site=0).fit(cohort.Z, cohort.y_site, y_diag=cohort.y_diag)
        Zh = h.transform(cohort.Z, cohort.y_site)
        assert Zh.shape == cohort.Z.shape and np.all(np.isfinite(Zh)), v
    h2 = make_variant("ours-2d", target_site=0).fit(cohort.Z, cohort.y_site, y_diag=cohort.y_diag,
                                                    severity=cohort.severity)
    assert h2.k_ in (1, 2)
    h_auto = make_variant("ours-auto", target_site=0).fit(cohort.Z, cohort.y_site, y_diag=cohort.y_diag)
    for s, tr in h_auto.site_transforms_.items():
        if not tr.identity:
            assert 0 <= tr.lam <= 1 and np.isclose(tr.lam, 1 - tr.shrinkage)
    assert np.allclose(IdentityHarmonizer().fit_transform(cohort.Z, cohort.y_site), cohort.Z)
