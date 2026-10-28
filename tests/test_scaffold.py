"""Smoke tests for the test scaffold itself."""

import numpy as np

from conftest import make_synthetic_cohort, write_synthetic_hdf5


def test_synthetic_cohort_shapes(cohort):
    N, d = cohort.Z.shape
    assert N == 240 and d == 8
    assert cohort.y_site.shape == (N,)
    assert set(np.unique(cohort.y_site)) == {0, 1, 2, 3}
    assert np.isclose(np.linalg.norm(cohort.w_true), 1.0)
    # severity is defined only for cases
    assert np.all(np.isnan(cohort.severity[cohort.y_diag == 0]))
    assert np.all(~np.isnan(cohort.severity[cohort.y_diag == 1]))


def test_planted_diagnosis_effect(cohort):
    alpha = cohort.Z @ cohort.w_true
    gap = alpha[cohort.y_diag == 1].mean() - alpha[cohort.y_diag == 0].mean()
    assert gap > 0.8  # planted effect is 1.5 along w_true, minus site noise


def test_small_site_fixture(small_site_cohort):
    counts = np.bincount(small_site_cohort.y_site)
    assert tuple(counts) == (80, 40, 4)


def test_hdf5_writer(tmp_path):
    import h5py

    c = make_synthetic_cohort(n_sites=2, sizes=(20, 20), d=4, seed=3)
    p = write_synthetic_hdf5(tmp_path / "toy.h5", c, n_rois=6)
    with h5py.File(p, "r") as f:
        assert f["X"].shape == (40, 15)
        assert f["fc_half1"].shape == f["X"].shape
        assert list(f.attrs["site_names"].astype(str)) == ["S0", "S1"]
