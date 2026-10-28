"""
Helpers for the two SRPBS releases of ATR (no real data are read here).

* SRPBS Multidisorder Connectivity Dataset (``srpbsfc``, Tanaka et al. 2021): one file per
  institute and diagnosis, Pearson correlations of the 140-region BAL atlas (Brainvisa Sulci
  Atlas plus three cerebellar regions), lower triangle vectorized column-wise. Some institutes
  used several scanners, so the scanner, not the institute, is the site.
* DecNef Harmonization Dataset (Yamashita et al. 2019): Fisher-z connectivity of a 268-region
  atlas for the SRPBS clinical sample and for nine traveling subjects scanned at twelve sites.

Data-use terms: the data are not redistributed, and traveling subjects are referred to by an
index of our own (``T1``..``T9``), never by the initials used in the release.
"""

from __future__ import annotations

import numpy as np

# Scanner sites of the traveling-subject study, in the 1-12 order of the release.
TS_SITES = ("ATT", "ATV", "COI", "HKH", "HUH", "KPM", "KUS", "KUT", "SWA", "UTO", "YC1", "YC2")

# Site of a clinical subject in the DecNef release, from the prefix of its ID. This is the
# mapping of MakeConnectivity_SubtractBias.m in the release (KPM data are not distributed).
_DECNEF_PREFIX = {"ATN": "ATT", "UKY": "KUT", "SWA": "SWA", "UTO": "UTO"}
_DECNEF_UHI = {"Verio": "COI", "Spect": "HKH", "Signa": "HUH"}

# Diagnosis codes shared by the SRPBS releases (UTO README and DecNef DATA_SUB).
DX_CODES = {0: "HC", 1: "ASD", 2: "MDD", 4: "SCZ", 7: "BPD", 8: "DYS", 99: "OTHER"}


def decnef_site(subject_id: str) -> str:
    """Scanner site of a DecNef clinical subject, e.g. 'UHI_Spectra_30_999_00' -> 'HKH'."""
    sid = str(subject_id).strip()
    prefix = sid[:3]
    if prefix == "ATR":
        return "ATT" if sid[4:8] == "Trio" else "ATV"
    if prefix == "UHI":
        for tag, site in _DECNEF_UHI.items():
            if sid[4:4 + len(tag)] == tag:
                return site
        raise ValueError(f"unknown Hiroshima scanner in {sid!r}")
    if prefix in _DECNEF_PREFIX:
        return _DECNEF_PREFIX[prefix]
    raise ValueError(f"unrecognized SRPBS subject id: {sid!r}")


def r_to_fisher_z(r: np.ndarray, clip: float = 0.999999) -> np.ndarray:
    """Fisher z of Pearson correlations, clipped so that |r| = 1 stays finite."""
    r = np.asarray(r, dtype=np.float64)
    return np.arctanh(np.clip(r, -clip, clip)).astype(np.float32)


def sex_male(code) -> np.ndarray:
    """Release coding M=1, F=2 -> 1 male, 0 female, NaN otherwise."""
    c = np.asarray(code, dtype=float)
    return np.where(c == 1, 1.0, np.where(c == 2, 0.0, np.nan))


def n_rois_from_edges(n_edges: int) -> int:
    n = int(round((1 + np.sqrt(1 + 8 * n_edges)) / 2))
    if n * (n - 1) // 2 != n_edges:
        raise ValueError(f"{n_edges} is not a triangular number of edges")
    return n
