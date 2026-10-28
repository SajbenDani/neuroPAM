import numpy as np
import pytest

from vae_harmonization.analyses.downstream import (
    cohens_d_per_feature,
    effect_size_reproducibility,
    loso_all_methods,
    motion_leakage,
)
from vae_harmonization.analyses.fingerprinting import (
    fingerprint_metrics,
    run_fingerprint_analysis,
    split_timeseries_fc,
    within_site_fingerprint,
)
from vae_harmonization.analyses.networks import (
    N_NETWORKS,
    compare_network_matrices,
    compute_network_matrix,
    label_table_path,
    load_network_labels,
    network_edge_means,
    run_network_analysis,
)
from vae_harmonization.analyses.protected_axis import protected_axis_report
from vae_harmonization.analyses.simulation import run_star_vs_chain
from vae_harmonization.evaluation.protocol import MethodSpec
from vae_harmonization.harmonization import IdentityHarmonizer, make_variant


# -- fingerprinting -----------------------------------------------------------------

def test_split_timeseries_fc_shapes():
    ts = np.random.default_rng(0).normal(size=(60, 5))
    a, b = split_timeseries_fc(ts, 5)
    assert a.shape == b.shape == (10,)
    assert np.all(np.abs(a) <= 4.0)


def test_fingerprint_metrics_perfect_and_chance(rng):
    Z1 = rng.normal(size=(30, 8))
    r = fingerprint_metrics(Z1, Z1 + 1e-6 * rng.normal(size=Z1.shape))
    assert r["rank1"] == 1.0 and r["median_rank"] == 1.0 and r["idiff"] > 0
    r2 = fingerprint_metrics(Z1, rng.normal(size=(30, 8)))
    assert r2["rank1"] < 0.3


def test_single_map_fingerprint_uses_one_fit(cohort):
    rng = np.random.default_rng(1)
    Z1 = cohort.Z + 0.3 * rng.normal(size=cohort.Z.shape)
    Z2 = cohort.Z + 0.3 * rng.normal(size=cohort.Z.shape)
    h = make_variant("ours", cohort.target_site, min_cov_samples=10)
    rep = run_fingerprint_analysis(cohort.Z, Z1, Z2, cohort.y_site, cohort.y_diag, {"Ours": h})
    pre, post = rep["conditions"]["Unharmonized"], rep["conditions"]["Ours"]
    # the same fitted site maps were applied to both halves -> within-site identifiability unchanged
    # up to numerical noise (an affine map with a per-site constant does not change within-site ranks
    # under euclidean metric; cosine can move slightly)
    assert abs(post["within_site"]["rank1_weighted"] - pre["within_site"]["rank1_weighted"]) < 0.15
    assert "delta_rank1_within_pp" in post
    ws = within_site_fingerprint(Z1, Z2, cohort.y_site)
    assert ws["n_sites"] == 4 and 0 < ws["chance_weighted"] < 0.1


# -- networks -------------------------------------------------------------------------

def test_network_matrix_recovers_block_structure(rng):
    n_rois = 14
    labels = np.array([0] * 4 + [1] * 4 + [2] * 4 + [-1] * 2)
    r, c = np.triu_indices(n_rois, k=1)
    edge = np.where(labels[r] == labels[c], 0.8, 0.1)
    fc = edge[None, :] + 0.01 * rng.normal(size=(20, edge.size))
    M = compute_network_matrix(fc, labels)
    assert M.shape == (N_NETWORKS, N_NETWORKS)
    assert np.allclose(np.diag(M)[:3], 0.8, atol=0.02)
    assert np.allclose(M[0, 1], 0.1, atol=0.02)
    assert np.isnan(M[3, 3])                          # no ROI in network 3
    sub = network_edge_means(fc, labels)
    assert sub.shape == (20, N_NETWORKS, N_NETWORKS)
    cmp = compare_network_matrices(M, M * 0.9 + 0.01)
    assert cmp["pearson_r"] > 0.999 and cmp["within_vs_between_preserved"]
    rep = run_network_analysis({"Raw FC": fc, "Other": fc * 0.5}, labels, np.repeat([0, 1], 10))
    assert rep["conditions"]["Other"]["per_subject_r"]["mean"] > 0.99
    assert rep["n_rois_cortical"] == 12


@pytest.mark.parametrize("cohort_name,n_rois", [("abide1_full", 200), ("adhd200", 190), ("abide2_10site", 195)])
def test_committed_label_tables(cohort_name, n_rois):
    if not label_table_path(cohort_name).exists():
        pytest.skip("label table not built")
    labels = load_network_labels(cohort_name, n_rois=n_rois)
    assert labels.shape == (n_rois,)
    assert set(np.unique(labels)) <= set(range(-1, N_NETWORKS))
    assert (labels >= 0).mean() > 0.7                  # most CC200 ROIs are cortical
    assert all((labels == k).sum() >= 5 for k in range(N_NETWORKS))


# -- protected axis -------------------------------------------------------------------

def test_protected_axis_report_preserves_diagnosis_effect(cohort):
    h = make_variant("ours", cohort.target_site, min_cov_samples=10)
    rep = protected_axis_report(cohort.Z, cohort.y_site, cohort.y_diag, h)
    ax = rep["axes"][0]
    assert ax["site_eta2_post"] < ax["site_eta2_pre"]
    assert abs(ax["diag_d_post"] - ax["diag_d_pre"]) < 0.25   # mean shift is per site, d changes only via site mix
    assert rep["complement_mean_eta2_post"] < rep["complement_mean_eta2_pre"]
    assert 0.0 <= rep["between_site_ss"]["protected_share_pre"] <= 1.0


# -- downstream -----------------------------------------------------------------------

def test_cohens_d_and_effect_size_reproducibility(cohort):
    d = cohens_d_per_feature(cohort.Z, cohort.y_diag)
    assert d.shape == (cohort.Z.shape[1],)
    # the diagnosis effect in the fixture is a mean shift along w_true: per-feature d must follow it
    assert abs(np.corrcoef(d, cohort.w_true)[0, 1]) > 0.5
    rep = effect_size_reproducibility(cohort.Z, cohort.y_site, cohort.y_diag, n_splits=5, seed=0)
    assert rep["n_splits"] >= 1 and -1 <= rep["r"]["mean"] <= 1


def test_loso_all_methods_runs_and_masks_labels(cohort):
    specs = [
        MethodSpec("VAE latent", "latent", IdentityHarmonizer, fit_aux=(), is_reference=True),
        MethodSpec("Ours", "latent", lambda: make_variant("ours", cohort.target_site, min_cov_samples=10),
                   is_proposed=True),
    ]
    rep = loso_all_methods({"latent": cohort.Z}, cohort.y_site, cohort.y_diag, specs,
                           target_site=cohort.target_site, classifiers=("lr",), min_test_size=10)
    assert len(rep["sites"]) == 4
    for m in ("VAE latent", "Ours"):
        assert rep["methods"][m]["uses_test_labels"] is False
        assert len(rep["methods"][m]["per_site"]["lr"]["acc"]) == 4
    assert "Ours" in rep["comparisons"]["lr"]


def test_motion_leakage_detects_signal(cohort, rng):
    fd = cohort.Z[:, 1] * 0.5 + 0.05 * rng.normal(size=len(cohort.Z))
    rep = motion_leakage(cohort.Z, fd, cohort.y_site, n_folds=3)
    assert rep["r2"]["mean"] > 0.5
    rep_none = motion_leakage(rng.normal(size=cohort.Z.shape), fd, cohort.y_site, n_folds=3)
    assert rep_none["r2"]["mean"] < 0.3


# -- simulation ------------------------------------------------------------------------

def test_star_vs_chain_simulation_small():
    rep = run_star_vs_chain(n_sites=4, d=6, n_per_site=40, n_rep=3, seed=0)
    r = rep["results"]
    assert set(r) == {"none", "star_target", "star_barycenter", "chain"}
    assert r["star_target"]["w2_to_hub_pop"]["mean"] < r["none"]["w2_to_hub_pop"]["mean"]
    assert r["star_target"]["probe_acc"]["mean"] < r["none"]["probe_acc"]["mean"]
