"""Tests for paired statistics helpers."""

import numpy as np

from vae_harmonization.stats import (
    chance_levels,
    classification_summary,
    holm_bonferroni,
    majority_class_rate,
    paired_cohens_dz,
    paired_test,
    pairwise_significance_matrix,
    summarize_folds,
    tost_paired,
)


def test_paired_dz_uses_difference_sd():
    a = np.array([0.70, 0.72, 0.71, 0.73, 0.69])
    b = a - 0.02  # constant offset -> zero SD of differences -> inf
    assert np.isinf(paired_cohens_dz(a, b))
    b2 = a - np.array([0.01, 0.03, 0.02, 0.04, 0.00])
    d = a - b2
    assert np.isclose(paired_cohens_dz(a, b2), d.mean() / d.std(ddof=1))


def test_paired_test_reports_ci_and_flags(rng):
    a = rng.normal(0.70, 0.02, size=10)
    b = a - 0.05 + rng.normal(0, 0.005, size=10)
    r = paired_test(a, b)
    assert r["p_value"] < 0.01 and r["significant_001"]
    assert r["mean_diff_ci_low"] <= r["mean_diff"] <= r["mean_diff_ci_high"]
    assert r["effect_size_dz"] == r["effect_size_cohens_d"]
    r_same = paired_test(a, a)
    assert r_same["p_value"] == 1.0


def test_tost_declares_equivalence_only_when_close(rng):
    a = rng.normal(0.70, 0.01, size=10)
    b = a + rng.normal(0, 0.003, size=10)
    res = tost_paired(a, b, margin=0.02)
    assert res["equivalent"]
    c = a - 0.10
    assert not tost_paired(a, c, margin=0.02)["equivalent"]


def test_holm_is_monotone_and_capped():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    adj = holm_bonferroni(p)
    assert np.all(adj <= 1.0) and np.all(adj >= p)
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12)


def test_pairwise_matrix_symmetry(rng):
    scores = {
        "A": rng.normal(0.7, 0.02, 10),
        "B": rng.normal(0.65, 0.02, 10),
        "C": rng.normal(0.7, 0.02, 10),
    }
    m = pairwise_significance_matrix(scores, "acc")
    P = m["p_values"]
    assert np.allclose(P, P.T)
    assert np.allclose(m["effect_sizes"], -m["effect_sizes"].T)
    assert np.all(m["p_values_adjusted"] >= P - 1e-12)


def test_chance_levels_and_summary():
    y = np.array([0] * 70 + [1] * 20 + [2] * 10)
    assert majority_class_rate(y) == 0.7
    ch = chance_levels(y)
    assert ch["n_classes"] == 3 and np.isclose(ch["uniform_chance"], 1 / 3)
    pred = np.zeros_like(y)
    s = classification_summary(y, pred, task="site")
    assert np.isclose(s["acc"], 0.7)
    assert np.isclose(s["bacc"], 1 / 3)
    assert "nmi" in s and s["nmi"] == 0.0
    yb = np.array([0, 0, 1, 1])
    sb = classification_summary(yb, yb, y_score=[0.1, 0.2, 0.8, 0.9])
    assert sb["auc"] == 1.0
    fs = summarize_folds([0.7, 0.8, float("nan"), 0.75])
    assert fs["n"] == 3 and fs["ci_low"] <= fs["mean"] <= fs["ci_high"]
