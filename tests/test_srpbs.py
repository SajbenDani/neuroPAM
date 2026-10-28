"""SRPBS helpers and the traveling-subject analysis on synthetic data. No real scans are read."""

import numpy as np
import pytest

from vae_harmonization.analyses.traveling import (
    TSMethod,
    cross_site_identification,
    offset_cosine,
    run_transfer,
    run_within_travelers,
    variance_shares,
)
from vae_harmonization.evaluation.representations import PCARepresentation
from vae_harmonization.harmonization import IdentityHarmonizer
from vae_harmonization.harmonization.affine import ProtectedAffineHarmonizer
from vae_harmonization.srpbs import decnef_site, n_rois_from_edges, r_to_fisher_z, sex_male


def test_decnef_site_mapping():
    assert decnef_site("ATR_Trio_30_999_00") == "ATT"
    assert decnef_site("ATR_Verio_30_998_00") == "ATV"
    assert decnef_site("ATN_Trio_30_997_00") == "ATT"
    assert decnef_site("UHI_Verio_30_999_00") == "COI"
    assert decnef_site("UHI_Spectra_30_999_00") == "HKH"
    assert decnef_site("UHI_SignaHDxt1_30_999_00") == "HUH"
    assert decnef_site("UKY_TimTrio_30_996_00") == "KUT"
    assert decnef_site("UTO_GEMR750W_30_999_00") == "UTO"
    with pytest.raises(ValueError):
        decnef_site("XYZ_001")


def test_fisher_z_and_codes():
    z = r_to_fisher_z(np.array([0.0, 0.5, 1.0, -1.0]))
    assert z.dtype == np.float32 and np.isfinite(z).all()
    assert np.isclose(z[1], np.arctanh(0.5))
    assert np.array_equal(np.nan_to_num(sex_male([1, 2, 9]), nan=-1), [1.0, 0.0, -1.0])
    assert n_rois_from_edges(9730) == 140 and n_rois_from_edges(35778) == 268
    with pytest.raises(ValueError):
        n_rois_from_edges(11)


def _travelers(seed=0, n_trav=6, n_site=4, reps=3, p=60, scanner=1.0, person=2.0, noise=0.5):
    rng = np.random.default_rng(seed)
    bias = scanner * rng.normal(size=(n_site, p))
    who = person * rng.normal(size=(n_trav, p))
    site, trav, X = [], [], []
    for s in range(n_site):
        for t in range(n_trav):
            for _ in range(reps):
                site.append(s)
                trav.append(t)
                X.append(bias[s] + who[t] + noise * rng.normal(size=p))
    return np.array(X), np.array(site), np.array(trav)


def test_variance_shares_recover_the_design():
    X, site, trav = _travelers(scanner=1.0, person=2.0, noise=0.3)
    sh = variance_shares(X, site, trav)
    assert sh["traveler_share"] > sh["site_share"] > sh["residual_share"]
    assert np.isclose(sh["site_share"] + sh["traveler_share"] + sh["residual_share"], 1.0, atol=0.05)
    ident = cross_site_identification(X, site, trav)
    assert ident["rank1"] > 0.9 and ident["chance"] < 0.25


def test_offset_cosine_is_one_for_a_perfect_correction():
    X, site, trav = _travelers()
    hub = 0
    means = {s: X[site == s].mean(axis=0) for s in np.unique(site)}
    corrected = X - np.array([means[s] - means[hub] for s in site])
    off = offset_cosine(X, corrected, site, trav, hub)
    assert off["mean"] > 0.999
    assert off["offset_left_mean"] < 1e-8


def test_leave_one_traveler_out_with_a_fixed_basis_removes_the_scanner_effect():
    X, site, trav = _travelers(seed=1)
    rep = PCARepresentation.fit(X + np.random.default_rng(2).normal(size=X.shape), 20)   # independent people
    methods = [TSMethod("PCA scores", "pca", IdentityHarmonizer, reference=True),
               TSMethod("Mean shift", "pca",
                        lambda: ProtectedAffineHarmonizer(target_site=0, protected=(), complement_map="identity"))]
    res = run_within_travelers(X, site, trav, methods, hub=0, pca_dim=20, rep=rep, probes=False)
    assert res["representation"].startswith("fixed")
    # shared scanner offsets only: the other travelers' means remove the new person's offset too
    assert res["methods"]["Mean shift"]["per_person"]["left"] < 0.2
    assert res["methods"]["PCA scores"]["per_person"]["left"] == pytest.approx(1.0)
    assert "anova" in res and res["anova"]["pca"]["site_share"] > 0


def test_transfer_runs_and_reports_every_metric():
    X, site, trav = _travelers(seed=3)
    rng = np.random.default_rng(4)
    X_clin = np.vstack([X[site == s].mean(axis=0) + rng.normal(size=(15, X.shape[1])) for s in range(4)])
    s_clin = np.repeat(np.arange(4), 15)
    methods = [TSMethod("PCA scores", "pca", IdentityHarmonizer, reference=True),
               TSMethod("PAM", "pca", lambda: ProtectedAffineHarmonizer(target_site=0, protected=()))]
    res = run_transfer(X_clin, s_clin, X, site, trav, methods, hub=0, pca_dim=20, probes=True)
    e = res["methods"]["PAM"]
    for key in ("shares", "identification", "scanner_ss_left", "offset", "per_person", "site_probe",
                "traveler_probe"):
        assert key in e
    assert "offset_left_sq_mean" in e["offset"]
    assert 0.0 <= e["site_probe"]["bacc"] <= 1.0
