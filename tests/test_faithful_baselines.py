"""The baselines against their published definitions, and the new analysis pieces."""

import numpy as np
import pytest
from pyriemann.utils.distance import distance
from pyriemann.utils.mean import mean_euclid, mean_riemann

from vae_harmonization.analyses.downstream import effect_size_reproducibility_crossfit
from vae_harmonization.analyses.traveling import interaction_anova, per_person_scanner_left
from vae_harmonization.analyses.unseen_site import unseen_site_curve
from vae_harmonization.baselines import (
    CORALHarmonizer,
    DeepComBatHarmonizer,
    EuclideanAlignmentHarmonizer,
    RiemannianProcrustesHarmonizer,
    RiemannianRecenteringHarmonizer,
    RLETHarmonizer,
    deepcombat_available,
    log_euclidean_vectors,
)
from vae_harmonization.baselines.riemannian import to_spd
from vae_harmonization.evaluation.protocol import MethodSpec
from vae_harmonization.harmonization import IdentityHarmonizer, ProtectedAffineHarmonizer, make_variant
from vae_harmonization.harmonization.edge import EdgeResidualHarmonizer


def _inv_sqrt(C):
    w, V = np.linalg.eigh(C)
    return (V / np.sqrt(w)) @ V.T


def _sqrt(C):
    w, V = np.linalg.eigh(C)
    return (V * np.sqrt(w)) @ V.T


# -- CORAL ---------------------------------------------------------------------------------

def test_coral_is_algorithm_1_of_the_paper(cohort):
    """Per site: z-score, C_S = cov + I, C_T = cov + I, D_S C_S^{-1/2} C_T^{1/2}, back to hub units."""
    h = CORALHarmonizer(target_site=0).fit(cohort.Z, cohort.y_site)
    out = h.transform(cohort.Z, cohort.y_site)
    hub = cohort.Z[cohort.y_site == 0]
    Zt = (hub - hub.mean(0)) / hub.std(0, ddof=1)
    Ct = np.cov(Zt, rowvar=False) + np.eye(hub.shape[1])
    for s in (1, 2, 3):
        Xs = cohort.Z[cohort.y_site == s]
        Ds = (Xs - Xs.mean(0)) / Xs.std(0, ddof=1)
        Cs = np.cov(Ds, rowvar=False) + np.eye(Xs.shape[1])
        expected = (Ds @ _inv_sqrt(Cs) @ _sqrt(Ct)) * hub.std(0, ddof=1) + hub.mean(0)
        assert np.allclose(out[cohort.y_site == s], expected, atol=1e-8)
    assert np.allclose(out[cohort.y_site == 0], hub)


def test_coral_leaves_a_one_scan_site_unchanged(cohort):
    y = cohort.y_site.copy()
    y[0] = 9
    h = CORALHarmonizer(target_site=1).fit(cohort.Z, y)
    assert np.allclose(h.transform(cohort.Z[:1], y[:1]), cohort.Z[:1])


# -- Riemannian family ------------------------------------------------------------------------

def _spd_vectors(cohort, n_rois=6, n_time=40, seed=0):
    """Fisher-z vectors of real correlation matrices: time series mixed by a site-specific matrix."""
    rng = np.random.default_rng(seed)
    mix = {int(s): np.eye(n_rois) + 0.4 * rng.normal(size=(n_rois, n_rois)) for s in np.unique(cohort.y_site)}
    iu = np.triu_indices(n_rois, 1)
    rows = []
    for s in cohort.y_site:
        ts = rng.normal(size=(n_time, n_rois)) @ mix[int(s)]
        rows.append(np.arctanh(np.clip(np.corrcoef(ts, rowvar=False)[iu], -0.999, 0.999)))
    return log_euclidean_vectors(np.array(rows), alpha=0.1)


def test_recentering_and_ea_put_every_site_mean_at_the_identity(cohort):
    V = _spd_vectors(cohort)
    n = to_spd(V[:1]).shape[-1]
    out = RiemannianRecenteringHarmonizer(target_site=0).fit(V, cohort.y_site).transform(V, cohort.y_site)
    out_ea = EuclideanAlignmentHarmonizer(target_site=0).fit(V, cohort.y_site).transform(V, cohort.y_site)
    for s in np.unique(cohort.y_site):
        m = cohort.y_site == s
        assert np.allclose(mean_riemann(to_spd(out[m])), np.eye(n), atol=1e-5)
        assert np.allclose(mean_euclid(to_spd(out_ea[m])), np.eye(n), atol=1e-8)


def test_rpa_stretches_every_site_to_the_hub_dispersion(cohort):
    V = _spd_vectors(cohort)
    out = RiemannianProcrustesHarmonizer(target_site=0).fit(V, cohort.y_site).transform(V, cohort.y_site)
    disp = {}
    for s in np.unique(cohort.y_site):
        C = to_spd(out[cohort.y_site == s])
        M = mean_riemann(C)
        disp[int(s)] = float(np.mean(distance(C, M, metric="riemann", squared=True)))
    for s in (1, 2, 3):
        assert disp[s] == pytest.approx(disp[0], rel=1e-3)


def test_rlet_moves_every_site_mean_to_the_mean_of_site_means(cohort):
    V = _spd_vectors(cohort)
    h = RLETHarmonizer().fit(V, cohort.y_site)
    out = h.transform(V, cohort.y_site)
    target = np.mean([V[cohort.y_site == s].mean(0) for s in np.unique(cohort.y_site)], axis=0)
    for s in np.unique(cohort.y_site):
        assert np.allclose(out[cohort.y_site == s].mean(0), target, atol=1e-10)


# -- CovBat against the authors' R package -----------------------------------------------------

def _r_package_available(name):
    import shutil
    import subprocess
    if shutil.which("Rscript") is None:
        return False
    proc = subprocess.run(["Rscript", "--vanilla", "-e", f"quit(status = as.integer(!requireNamespace('{name}', quietly = TRUE)))"],
                          capture_output=True, text=True)
    return proc.returncode == 0


@pytest.mark.skipif(not _r_package_available("CovBat"), reason="the CovBat R package is not installed")
def test_covbat_matches_the_reference_r_implementation(tmp_path):
    import subprocess
    from vae_harmonization.baselines import CovBatHarmonizer
    rng = np.random.default_rng(0)
    n, p = 240, 30
    site = np.repeat([0, 1, 2, 3], 60)
    dx = rng.integers(0, 2, n)
    A = {s: np.eye(p) + 0.3 * rng.normal(size=(p, p)) for s in range(4)}
    X = np.stack([rng.normal(size=p) @ A[s] + 0.5 * s + 0.8 * d * np.linspace(0, 1, p) for s, d in zip(site, dx)])
    ours = CovBatHarmonizer(protect_diagnosis=True).fit(X, site, dx)
    np.savetxt(tmp_path / "X.csv", X, delimiter=",")
    np.savetxt(tmp_path / "s.csv", site, fmt="%d")
    np.savetxt(tmp_path / "d.csv", dx, fmt="%d")
    script = tmp_path / "run.R"
    script.write_text(
        "suppressPackageStartupMessages(library(CovBat))\n"
        f"X <- as.matrix(read.csv('{(tmp_path / 'X.csv').as_posix()}', header = FALSE))\n"
        f"s <- scan('{(tmp_path / 's.csv').as_posix()}', quiet = TRUE)\n"
        f"d <- scan('{(tmp_path / 'd.csv').as_posix()}', quiet = TRUE)\n"
        "out <- covbat(t(X), s, mod = model.matrix(~ d)[, -1, drop = FALSE])\n"
        f"write.table(t(out$dat.covbat), '{(tmp_path / 'out.csv').as_posix()}', sep = ',', row.names = FALSE, col.names = FALSE)\n")
    subprocess.run(["Rscript", "--vanilla", str(script)], check=True, capture_output=True)
    ref = np.loadtxt(tmp_path / "out.csv", delimiter=",")
    assert np.abs(ours.X_train_harmonized_ - ref).max() < 1e-8 * np.abs(ref).max()


# -- DeepComBat (authors' R package) ---------------------------------------------------------

@pytest.mark.skipif(not deepcombat_available(), reason="the DeepComBat R package is not installed")
def test_deepcombat_bridge_runs_and_keeps_the_reference_site(cohort):
    idx = np.arange(len(cohort.y_site))
    tr, te = idx[idx % 4 != 0], idx[idx % 4 == 0]
    h = DeepComBatHarmonizer(target_site=0, seed=1).fit(cohort.Z[tr], cohort.y_site[tr], cohort.y_diag[tr])
    out = h.transform(cohort.Z[te], cohort.y_site[te], y_diag=cohort.y_diag[te])
    assert out.shape == cohort.Z[te].shape and np.all(np.isfinite(out))
    ref = cohort.y_site[te] == 0
    assert np.allclose(out[ref], cohort.Z[te][ref], atol=1e-4)     # ComBat with a reference batch
    # a held-out set that lacks some training sites gives the same rows (anchors keep the estimates aligned)
    only = te[cohort.y_site[te] == 2]
    out_only = h.transform(cohort.Z[only], cohort.y_site[only], y_diag=cohort.y_diag[only])
    assert np.allclose(out_only, out[cohort.y_site[te] == 2], atol=1e-5)


# -- RELIEF (authors' R package) -------------------------------------------------------------

@pytest.mark.skipif(not _r_package_available("RELIEF"), reason="the RELIEF R package is not installed")
def test_relief_bridge_runs_without_labels_by_default(cohort):
    from vae_harmonization.baselines import RELIEFHarmonizer
    idx = np.arange(len(cohort.y_site))
    tr, te = idx[idx % 4 != 0], idx[idx % 4 == 0]
    Z = cohort.Z
    h = RELIEFHarmonizer()
    assert not h.uses_labels_at_transform
    h.fit(Z[tr], cohort.y_site[tr])
    # the training rows come back as the fit produced them, not refitted together with a copy of themselves
    assert np.array_equal(h.transform(Z[tr], cohort.y_site[tr]), h.X_train_harmonized_)
    out = h.transform(Z[te], cohort.y_site[te])          # no diagnosis needed
    assert out.shape == Z[te].shape and np.all(np.isfinite(out))
    gap = lambda X, s: np.ptp([X[s == k].mean(0) for k in np.unique(s)], axis=0).mean()
    assert gap(out, cohort.y_site[te]) < gap(Z[te], cohort.y_site[te])
    labelled = RELIEFHarmonizer(use_covariates=True)
    assert labelled.uses_labels_at_transform
    with pytest.raises(ValueError):
        labelled.fit(Z[tr], cohort.y_site[tr], cohort.y_diag[tr]).transform(Z[te], cohort.y_site[te])


# -- PAM extensions -----------------------------------------------------------------------------

def test_james_stein_shrinks_site_offsets_and_keeps_the_hub(cohort):
    plain = ProtectedAffineHarmonizer(target_site=0, protected=(), complement_map="identity").fit(cohort.Z, cohort.y_site)
    js = ProtectedAffineHarmonizer(target_site=0, protected=(), complement_map="identity",
                                   mean_shrinkage="james-stein").fit(cohort.Z, cohort.y_site)
    f = js.mean_shrinkage_factors_
    assert f[0] == 1.0 and all(0.0 <= v <= 1.0 for v in f.values())
    shift_plain = plain.transform(cohort.Z, cohort.y_site) - cohort.Z
    shift_js = js.transform(cohort.Z, cohort.y_site) - cohort.Z
    for s in (1, 2, 3):
        m = cohort.y_site == s
        assert np.allclose(shift_js[m].mean(0), f[s] * shift_plain[m].mean(0), atol=1e-8)


def test_edge_residual_only_changes_the_component_span(cohort, rng):
    X = cohort.Z @ rng.normal(size=(cohort.Z.shape[1], 40)) + 0.3 * rng.normal(size=(len(cohort.Z), 40))
    ident = EdgeResidualHarmonizer(IdentityHarmonizer, pca_dim=5).fit(X, cohort.y_site)
    assert np.allclose(ident.transform(X, cohort.y_site), X)
    h = EdgeResidualHarmonizer(lambda: make_variant("full-affine", 0), pca_dim=5).fit(X, cohort.y_site)
    delta = (h.transform(X, cohort.y_site) - X) / h.rep_.scaler.scale_
    V = h.rep_.pca.components_
    assert np.allclose(delta - (delta @ V.T) @ V, 0.0, atol=1e-8)     # no change outside the 5 components
    assert np.isfinite(h.plugin_displacement())


def test_plugin_displacement_orders_the_maps_as_the_theorem_says(cohort):
    disp = {}
    for v in ("ours", "zca", "cholesky"):
        disp[v] = make_variant(v, 0).fit(cohort.Z, cohort.y_site, y_diag=cohort.y_diag).plugin_displacement()
    assert disp["ours"] <= disp["zca"] + 1e-9 and disp["ours"] <= disp["cholesky"] + 1e-9


# -- traveling-subject metrics --------------------------------------------------------------------

def _travelers(rng, n_trav=6, n_site=4, reps=3, p=30, shared=1.0, specific=0.0):
    b = shared * rng.normal(size=(n_site, p))
    ab = specific * rng.normal(size=(n_trav, n_site, p))
    a = 2.0 * rng.normal(size=(n_trav, p))
    site, trav, X = [], [], []
    for s in range(n_site):
        for t in range(n_trav):
            for _ in range(reps):
                site.append(s), trav.append(t)
                X.append(a[t] + b[s] + ab[t, s] + 0.5 * rng.normal(size=p))
    return np.array(X), np.array(site), np.array(trav)


def _leave_one_out_shift(X, site, trav):
    out = np.empty_like(X)
    for t in np.unique(trav):
        tr, te = trav != t, trav == t
        h = ProtectedAffineHarmonizer(target_site=0, protected=(), complement_map="identity").fit(X[tr], site[tr])
        out[te] = h.transform(X[te], site[te])
    return out


def test_per_person_metric_sees_what_the_pooled_metric_cannot(rng):
    from vae_harmonization.analyses.traveling import variance_shares
    for specific, expect_low in ((0.0, True), (3.0, False)):
        X, site, trav = _travelers(rng, shared=1.0, specific=specific)
        Xh = _leave_one_out_shift(X, site, trav)
        pooled = variance_shares(Xh, site, trav)["ss_site"] / variance_shares(X, site, trav)["ss_site"]
        assert pooled < 0.05                                   # cancels by construction either way
        left = per_person_scanner_left(X, Xh, site, trav)["left"]
        assert (left < 0.3) if expect_low else (left > 0.6)
    a = interaction_anova(X, site, trav)
    shares = a["site_share"] + a["traveler_share"] + a["interaction_share"] + a["within_share"]
    assert shares == pytest.approx(1.0, abs=0.05) and a["F_interaction"] > 1.0


# -- leakage-free effect sizes and unseen site -------------------------------------------------------

def test_crossfit_effect_sizes_hide_the_scored_half(cohort):
    spec = MethodSpec("Ours", "pca", lambda: make_variant("ours", 0))
    r = effect_size_reproducibility_crossfit(spec, cohort.Z, cohort.y_site, cohort.y_diag, n_splits=4, seed=0)
    assert r["labels"] == "hidden for the scored half" and r["n_splits"] >= 1 and -1 <= r["r"]["mean"] <= 1
    free = MethodSpec("Full", "pca", lambda: make_variant("full-affine", 0), fit_aux=())
    assert effect_size_reproducibility_crossfit(free, cohort.Z, cohort.y_site, cohort.y_diag, n_splits=4)["labels"] == "none"


def test_unseen_site_curve_scores_unfitted_scans(cohort):
    rep = unseen_site_curve(cohort.Z, cohort.y_site, cohort.y_diag, lambda: make_variant("ours", 0),
                            sizes=(10, 0.75), n_draws=2, seed=0, max_test=20)
    for key in ("10", "0.75"):
        r = rep["by_size"][key]
        assert r["n"] >= 1
        assert 0.0 <= r["heldout_vs_rest_acc"]["mean"] <= 1.0
        assert r["heldout_vs_rest_acc_unharmonized"]["mean"] >= 0.5     # the synthetic sites differ strongly
