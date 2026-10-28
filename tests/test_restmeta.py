"""REST-meta-MDD inclusion rules. No real scans are read."""

import numpy as np
import pytest

from vae_harmonization.restmeta import (
    N_EDGES,
    diagnosis_from_yan,
    fisher_z_edges,
    is_excluded_site,
    site_number,
)


def test_site_4_is_excluded_and_site_14_is_not():
    assert site_number("S14-1-0001") == 14
    assert is_excluded_site("S4-1-0001")
    assert is_excluded_site("S4-2-0048")
    assert not is_excluded_site("S14-2-0001")


def test_yan_diagnosis_codes():
    assert diagnosis_from_yan(1) == 1
    assert diagnosis_from_yan(-1) == 0
    with pytest.raises(ValueError):
        diagnosis_from_yan(0)


def test_fisher_z_upper_triangle():
    series = np.random.default_rng(0).normal(size=(200, 160))
    edges = fisher_z_edges(series)
    assert edges.shape == (N_EDGES,)
    assert edges.dtype == np.float32
    assert np.isfinite(edges).all()


def test_constant_parcel_is_rejected():
    series = np.random.default_rng(1).normal(size=(200, 160))
    series[:, 3] = 0.0
    with pytest.raises(ValueError):
        fisher_z_edges(series)
