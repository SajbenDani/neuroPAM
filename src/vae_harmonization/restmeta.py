"""
Helpers for the REST-meta-MDD cohort.

The consortium states that site 4 duplicates site 14, so every analysis drops
site 4. Connectivity uses the Dosenbach-160 block of the shared ROI file
(1-indexed columns 1409-1568), Fisher-z of the upper triangle. The default
sample is the subject list of Yan et al. (PNAS, 2019) with site 4 removed.
"""

from __future__ import annotations

import re

import numpy as np

# 1-indexed columns 1409-1568. Full files have 1833 columns. Some stop at 1568.
DOSENBACH_160 = slice(1408, 1568)
N_ROIS = 160
N_EDGES = N_ROIS * (N_ROIS - 1) // 2
EXCLUDE_SITES = frozenset({4})
ROI_COLUMNS = 1833
_ID = re.compile(r"S(\d+)-\d+-\d+")
_TRI = np.triu_indices(N_ROIS, k=1)


def site_number(subject_id: str) -> int:
    match = _ID.fullmatch(str(subject_id).strip())
    if match is None:
        raise ValueError(f"unrecognized REST-meta-MDD subject id: {subject_id!r}")
    return int(match.group(1))


def is_excluded_site(subject_id: str) -> bool:
    return site_number(subject_id) in EXCLUDE_SITES


def fisher_z_edges(time_series: np.ndarray, clip: float = 0.999999) -> np.ndarray:
    """
    Fisher-z upper triangle of a (time, ROI) series.

    Raises ValueError when a parcel is constant or the correlation is not finite.
    """
    series = np.asarray(time_series, dtype=np.float64)
    if series.ndim != 2 or series.shape[1] != N_ROIS:
        raise ValueError(f"expected (time, {N_ROIS}) series, got {series.shape}")
    if np.any(np.std(series, axis=0) < 1e-8):
        raise ValueError("constant parcel")
    corr = np.corrcoef(series, rowvar=False)
    if not np.isfinite(corr).all():
        raise ValueError("non-finite correlation")
    corr = np.clip(corr, -clip, clip)
    return np.arctanh(corr)[_TRI].astype(np.float32)


def replace_missing(series):
    """Phenotype codes -9999 and blank mean missing (consortium convention)."""
    return series.replace({-9999: np.nan, -9999.0: np.nan, "-9999": np.nan, "": np.nan})


def diagnosis_from_yan(dx: int) -> int:
    """Yan 2019 Dx codes: 1 = MDD, -1 = control. Returned labels are 1 and 0."""
    code = int(dx)
    if code == 1:
        return 1
    if code == -1:
        return 0
    raise ValueError(f"unexpected Yan 2019 Dx code: {code}")
