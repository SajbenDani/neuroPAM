#!/usr/bin/env python
"""
Build the REST-meta-MDD connectivity HDF5.

Reads the Phase-1 ROI signals (ROISignals_FunImgARCWF), keeps the subjects in
the Yan et al. (PNAS, 2019) list, and drops site 4. The consortium states that
site 4 duplicates site 14. Edges are the Fisher-z upper triangle of Dosenbach-160.

The output HDF5 and the downloaded subject list stay under data/, which is
gitignored. Do not copy either into a path that git tracks.

Usage:
    python scripts/preprocessing/preprocess_rest_meta_mdd.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vae_harmonization.restmeta import (  # noqa: E402
    DOSENBACH_160,
    EXCLUDE_SITES,
    N_EDGES,
    diagnosis_from_yan,
    fisher_z_edges,
    replace_missing,
    site_number,
)

DATA = ROOT / "data" / "REST-meta-MDD"
ROI_DIR = DATA / "Results" / "ROISignals_FunImgARCWF"
PHENO = DATA / "REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx"
MOTION = DATA / "RealignParameter"
YAN_MAT = DATA / "Stat_Sub_Info_848MDDvs794NC.mat"
YAN_URL = (
    "https://raw.githubusercontent.com/Chaogan-Yan/PaperScripts/master/"
    "Yan_2019_PNAS/StatsSubInfo/Stat_Sub_Info_848MDDvs794NC.mat"
)
OUT = ROOT / "data" / "processed" / "rest_meta_mdd_fc.h5"


def _ensure_yan_list(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading published subject list -> {path}")
    urllib.request.urlretrieve(YAN_URL, path)


def _inclusion(path: Path) -> dict[str, int]:
    mat = scipy.io.loadmat(path, squeeze_me=True)
    kept: dict[str, int] = {}
    n_site4 = 0
    for sid, dx in zip(mat["SubID"], mat["Dx"]):
        subject = str(sid).strip()
        if site_number(subject) in EXCLUDE_SITES:
            n_site4 += 1
            continue
        kept[subject] = diagnosis_from_yan(dx)
    print(f"Yan 2019 list: kept {len(kept)}, dropped site 4: {n_site4}")
    return kept


def _hamd(path: Path, subject_ids: list[str]) -> np.ndarray:
    mdd = pd.read_excel(path, sheet_name="MDD")
    mdd["ID"] = mdd["ID"].astype(str).str.strip()
    hamd = replace_missing(mdd.set_index("ID")["HAMD"])
    values = pd.to_numeric(hamd.reindex(subject_ids), errors="coerce").to_numpy(dtype=np.float32)
    print(f"HAMD: {int(np.isfinite(values).sum())}/{len(subject_ids)} finite (controls are missing by design)")
    return values


def _mean_power_fd(subject_id: str) -> float:
    path = MOTION / subject_id / f"FD_Power_{subject_id}.txt"
    if not path.exists():
        return float("nan")
    values = np.loadtxt(path)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def main() -> None:
    if not ROI_DIR.is_dir():
        raise SystemExit(f"ROI directory not found: {ROI_DIR}")
    if not PHENO.exists():
        raise SystemExit(f"phenotype workbook not found: {PHENO}")
    _ensure_yan_list(YAN_MAT)
    inclusion = _inclusion(YAN_MAT)

    rows: list[np.ndarray] = []
    sites: list[int] = []
    diagnoses: list[int] = []
    ids: list[str] = []
    fd: list[float] = []
    n_missing_roi = 0
    n_bad_series = 0
    for subject, diagnosis in inclusion.items():
        roi_path = ROI_DIR / f"ROISignals_{subject}.mat"
        if not roi_path.exists():
            n_missing_roi += 1
            continue
        signals = scipy.io.loadmat(roi_path, variable_names=["ROISignals"])["ROISignals"]
        # Full files have 1833 columns (through Power-264). A minority stop at column 1568,
        # which is the last Dosenbach parcel, so the same slice is valid.
        if signals.shape[1] < DOSENBACH_160.stop:
            n_bad_series += 1
            continue
        try:
            edges = fisher_z_edges(signals[:, DOSENBACH_160])
        except ValueError:
            n_bad_series += 1
            continue
        if edges.shape != (N_EDGES,):
            raise SystemExit(f"unexpected edge count {edges.shape}")
        rows.append(edges)
        sites.append(site_number(subject))
        diagnoses.append(diagnosis)
        ids.append(subject)
        fd.append(_mean_power_fd(subject))
        if len(rows) % 200 == 0:
            print(f"  {len(rows)} subjects")

    if not rows:
        raise SystemExit("no subjects written")
    site_numbers = sorted(set(sites))
    site_names = [f"S{n}" for n in site_numbers]
    site_index = {n: i for i, n in enumerate(site_numbers)}
    y_site = np.array([site_index[s] for s in sites], dtype=np.int32)
    X = np.stack(rows).astype(np.float32)
    y_dx = np.array(diagnoses, dtype=np.int32)
    mean_fd = np.array(fd, dtype=np.float32)
    hamd = _hamd(PHENO, ids)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUT, "w") as handle:
        handle.create_dataset("X", data=X, compression="gzip", compression_opts=4)
        handle.create_dataset("y_diagnosis", data=y_dx)
        handle.create_dataset("y_site", data=y_site)
        handle.create_dataset("subject_ids", data=ids, dtype=h5py.string_dtype(encoding="utf-8"))
        handle.create_dataset("mean_fd", data=mean_fd)
        handle.create_dataset("hamd", data=hamd)
        handle.attrs["site_names"] = site_names
        handle.attrs["n_sites"] = len(site_names)
        handle.attrs["atlas"] = "Dosenbach160"
        handle.attrs["pipeline"] = "FunImgARCWF"
        handle.attrs["excluded_sites"] = "4"
        handle.attrs["inclusion"] = "Yan2019_848MDDvs794NC_minus_site4"
        handle.attrs["n_missing_roi"] = n_missing_roi
        handle.attrs["n_bad_series"] = n_bad_series

    counts = np.bincount(y_site)
    hub = site_names[int(np.argmax(counts))]
    print(
        f"wrote {OUT.name}: N={X.shape[0]}, sites={len(site_names)}, edges={X.shape[1]}, "
        f"MDD={int((y_dx == 1).sum())}, control={int((y_dx == 0).sum())}, hub={hub} (n={int(counts.max())})"
    )
    print(f"missing ROI files: {n_missing_roi}; dropped non-finite series: {n_bad_series}")
    print(f"mean Power FD finite: {int(np.isfinite(mean_fd).sum())}/{len(mean_fd)}")
    if 4 in site_numbers:
        raise SystemExit("site 4 was written; that is a bug")


if __name__ == "__main__":
    main()
