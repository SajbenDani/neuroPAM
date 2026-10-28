#!/usr/bin/env python
"""
Build the SRPBS connectivity HDF5 files from the two ATR releases.

1. ``data/processed/srpbs_fc.h5`` -- SRPBS Multidisorder Connectivity Dataset (Tanaka et al. 2021),
   major depressive disorder (MDD) against healthy controls at the six scanners that contributed
   both groups: Hiroshima GE Signa HDxt 1 and 2 (HUH1, HUH2), Siemens Spectra (HKH) and Verio (COI),
   Kyoto TimTrio (KUT) and Tokyo GE MR750W (UTO). 140-region BAL atlas, Pearson r -> Fisher z.
   Scans with missing parcels (NaN edges, 32 Hiroshima scans) are dropped. Age and sex come from
   the release's demographic tables and are stored in the file.
2. ``data/processed/srpbs_decnef_fc.h5`` -- the clinical SRPBS sample of the DecNef Harmonization
   Dataset (Yamashita et al. 2019), 268-region atlas, Fisher z as released, eight scanners,
   controls and patients with MDD, schizophrenia (SCZ) or autism (ASD).
3. ``data/processed/srpbs_ts_fc.h5`` -- the nine traveling subjects of the same release, one row
   per session at twelve scanners, minus the 14 high-motion sessions the release lists
   (DATA_NOTUSE_TS.mat). ``traveler_id`` is our own index 0-8.

All outputs stay under data/, which is gitignored. The data-use terms forbid redistribution:
never copy these files, or any subject-level derivative, into a tracked path.

Usage:
    python scripts/preprocessing/preprocess_srpbs.py [--fc-root data/SRPBS/SRPBS_FC]
                                                     [--decnef-root data/SourceCodeAndDATA/data]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vae_harmonization.srpbs import (  # noqa: E402
    DX_CODES,
    TS_SITES,
    decnef_site,
    n_rois_from_edges,
    r_to_fisher_z,
    sex_male,
)

OUT = ROOT / "data" / "processed"


def _tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding_errors="replace")
    df.columns = [c.strip() for c in df.columns]
    return df


def _col(df: pd.DataFrame, start: str) -> str:
    for c in df.columns:
        if c.lower().replace(" ", "").startswith(start):
            return c
    raise KeyError(f"no column starting with {start!r} in {list(df.columns)}")


def _mat_X(path: Path) -> np.ndarray:
    m = scipy.io.loadmat(path, squeeze_me=True, variable_names=["X"])
    return np.asarray(m["X"], dtype=np.float64)


def _write(path: Path, X, y_site, y_dx, ids, site_names, attrs: dict, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.create_dataset("X", data=np.asarray(X, np.float32), compression="gzip", compression_opts=4)
        h.create_dataset("y_site", data=np.asarray(y_site, np.int32))
        h.create_dataset("y_diagnosis", data=np.asarray(y_dx, np.int32))
        h.create_dataset("subject_ids", data=[str(s) for s in ids], dtype=h5py.string_dtype(encoding="utf-8"))
        for k, v in extra.items():
            h.create_dataset(k, data=np.asarray(v))
        h.attrs["site_names"] = list(site_names)
        h.attrs["n_sites"] = len(site_names)
        for k, v in attrs.items():
            h.attrs[k] = v
    counts = np.bincount(np.asarray(y_site), minlength=len(site_names))
    parts = [f"{n} {c}/{int(((np.asarray(y_site) == i) & (np.asarray(y_dx) == 1)).sum())}"
             for i, (n, c) in enumerate(zip(site_names, counts))]
    print(f"wrote {path.name}: N={len(y_site)}, edges={X.shape[1]}, sites (n/cases): {', '.join(parts)}")


# -- 1. SRPBS Multidisorder Connectivity Dataset, MDD vs controls -------------------------------

def build_multidisorder(fc: Path) -> None:
    blocks = []      # (X_r, site names, dx 0/1, ids, age, sex)

    hir = _tsv(fc / "HiroshimaU" / "SUBINFO_HiroshimaU.tsv")
    X = _mat_X(fc / "HiroshimaU" / "DEP_BAL-UHIBP39_COR.mat")
    scanner = {1: "HUH1", 2: "HUH2", 3: "HKH", 4: "COI"}
    dx = hir[_col(hir, "diagnosis")].astype(int).map({1: 0, 2: 1}).to_numpy()
    blocks.append((X, hir["Scanner"].astype(int).map(scanner).to_numpy(), dx, hir["participant_id"].to_numpy(),
                   hir[_col(hir, "age")].to_numpy(float), sex_male(hir[_col(hir, "sex")])))

    for sub, label in (("HC", 0), ("DEP", 1)):
        d = fc / "KyotoU" / sub
        tab = _tsv(next(d.glob("SUBINFO_*.tsv")))
        X = _mat_X(next(d.glob("*_COR.mat")))
        site = tab[_col(tab, "site")].astype(int).to_numpy()
        keep = site == 1                       # TimTrio; the Trio scanner has no MDD patients
        ids = ("UKY_" + tab["participant_id"].astype(str)).to_numpy()
        blocks.append((X[keep], np.full(keep.sum(), "KUT", dtype=object), np.full(keep.sum(), label),
                       ids[keep], tab[_col(tab, "age")].to_numpy(float)[keep],
                       sex_male(tab[_col(tab, "sex")])[keep]))

    uto = fc / "UTO" / "UTO_GEMR750W_30_BAL"
    for code, label in (("00", 0), ("02", 1)):
        tab = _tsv(uto / f"UTO_GEMR750W_30_{code}_BAL_SUB.tsv")
        X = _mat_X(uto / f"UTO_GEMR750W_30_{code}_BAL_COR.mat")
        blocks.append((X, np.full(len(tab), "UTO", dtype=object), np.full(len(tab), label),
                       tab["participant_id"].to_numpy(), tab[_col(tab, "age")].to_numpy(float),
                       sex_male(tab[_col(tab, "sex")])))

    for X, *rest in blocks:
        if X.shape[0] != len(rest[0]):
            raise SystemExit(f"matrix/table size mismatch: {X.shape[0]} vs {len(rest[0])}")
    X = np.vstack([b[0] for b in blocks])
    site = np.concatenate([b[1] for b in blocks]).astype(str)
    dx = np.concatenate([b[2] for b in blocks]).astype(int)
    ids = np.concatenate([b[3] for b in blocks]).astype(str)
    age = np.concatenate([b[4] for b in blocks]).astype(float)
    sex = np.concatenate([b[5] for b in blocks]).astype(float)

    finite = np.isfinite(X).all(axis=1)
    print(f"multidisorder: dropping {int((~finite).sum())} scans with missing parcels "
          f"({pd.Series(site[~finite]).value_counts().to_dict()})")
    if np.abs(X[finite]).max() > 1.0:
        raise SystemExit("expected Pearson correlations in the multidisorder release")
    if len(set(ids[finite])) != finite.sum():
        raise SystemExit("duplicate participant ids")
    X, site, dx, ids, age, sex = X[finite], site[finite], dx[finite], ids[finite], age[finite], sex[finite]

    names = ["COI", "HKH", "HUH1", "HUH2", "KUT", "UTO"]
    y_site = np.array([names.index(s) for s in site])
    Z = r_to_fisher_z(X)
    _write(OUT / "srpbs_fc.h5", Z, y_site, dx, ids, names,
           {"atlas": "BAL140", "release": "SRPBS Multidisorder Connectivity Dataset (srpbsfc)",
            "contrast": "MDD vs HC, scanners with both groups", "n_rois": n_rois_from_edges(Z.shape[1]),
            "n_dropped_missing_parcels": int((~finite).sum())},
           {"age": age.astype(np.float32), "sex": sex.astype(np.float32)})


# -- 2. and 3. DecNef Harmonization Dataset -------------------------------------------------------

def _decnef_clinical(dec: Path, mat_name: str, out_name: str, dm_rows: np.ndarray, labels: list[str],
                     contrast: str) -> None:
    """One clinical sample of the DecNef release, checked against the authors' design matrix."""
    m = scipy.io.loadmat(dec / mat_name, squeeze_me=True)
    ids = [str(s) for s in m["IDLIST"]]
    X = np.asarray(m["X"], dtype=np.float64)
    data_sub = np.asarray(m["DATA_SUB"], dtype=float)
    if dm_rows.shape[0] != len(ids):
        raise SystemExit(f"design matrix has {dm_rows.shape[0]} clinical rows, {mat_name} has {len(ids)}")
    site = [decnef_site(s) for s in ids]
    s0 = labels.index("ATT")
    dm_site = [labels[s0 + j].replace("UTA", "UTO") for j in dm_rows[:, s0:s0 + 12].argmax(axis=1)]
    if site != dm_site:
        raise SystemExit(f"{mat_name}: site from subject id disagrees with the design matrix")
    code = data_sub[:, 0].astype(int)
    inv = {v: k for k, v in DX_CODES.items()}
    for name in ("ASD", "MDD", "SCZ"):
        if not np.array_equal(code == inv[name], dm_rows[:, labels.index(name)] == 1):
            raise SystemExit(f"{mat_name}: diagnosis code of {name} disagrees with the design matrix")
    names = sorted(set(site))
    y_site = np.array([names.index(s) for s in site])
    # DATA_SUB: diagnosis code, site code, age, sex (1 = female; checked against the ATR tables)
    age = data_sub[:, 2]
    sex = np.where(data_sub[:, 3] == 1, 0.0, 1.0)
    _write(OUT / out_name, X, y_site, (code > 0).astype(int), ids, names,
           {"atlas": "268-region (Yamashita et al. 2019)", "release": "DecNef Harmonization Dataset",
            "contrast": contrast, "n_rois": n_rois_from_edges(X.shape[1])},
           {"dx_code": code.astype(np.int32), "age": age.astype(np.float32), "sex": sex.astype(np.float32)})
    print("  diagnoses:", {DX_CODES[c]: int((code == c).sum()) for c in np.unique(code)})


def build_decnef(dec: Path, fc: Path | None) -> None:
    dm = scipy.io.loadmat(dec / "DesignMatrix_SRPBS_ALL.mat", squeeze_me=True)
    labels = [str(s) for s in dm["LABEL"]]
    DM = np.asarray(dm["DM"], dtype=float)
    n_ts = int((DM[:, :9].sum(axis=1) > 0).sum())
    # all clinical scans, and the subset acquired with the unified DecNef protocol that the
    # traveling subjects followed (the sample Yamashita et al. used to estimate measurement bias)
    _decnef_clinical(dec, "COR_SRPBS_ALL_ORG.mat", "srpbs_decnef_fc.h5", DM[n_ts:], labels,
                     "any patient vs HC; dx_code holds the disorder")
    du = scipy.io.loadmat(dec / "DesignMatrix.mat", squeeze_me=True, struct_as_record=False)
    u_labels = [str(s) for s in du["LABEL"].SITE]
    u_DM = np.asarray(du["DM"].SITE, dtype=float)
    n_ts_u = int((u_DM[:, :9].sum(axis=1) > 0).sum())
    _decnef_clinical(dec, "COR_SRPBS_UnifiedProtocol.mat", "srpbs_decnef_unified_fc.h5", u_DM[n_ts_u:], u_labels,
                     "unified-protocol scans only; any patient vs HC; dx_code holds the disorder")

    # traveling subjects
    Xts = np.asarray(scipy.io.loadmat(dec / "COR_TravelingSubject.mat", squeeze_me=True)["X"], dtype=np.float64)
    out = np.asarray(scipy.io.loadmat(dec / "DATA_NOTUSE_TS.mat", squeeze_me=True)["OUT"]).astype(bool)
    if out.size != Xts.shape[0]:
        raise SystemExit("motion-exclusion list does not match the traveling-subject matrix")
    Xts = Xts[~out]
    dm_ts = DM[:n_ts]
    if dm_ts.shape[0] != Xts.shape[0]:
        raise SystemExit(f"{dm_ts.shape[0]} traveling rows in the design matrix, {Xts.shape[0]} sessions kept")
    traveler = dm_ts[:, :9].argmax(axis=1)
    ts_site_label = [labels[9 + j] for j in dm_ts[:, 9:21].argmax(axis=1)]
    rename = {"KFU": "KPM", "UTA": "UTO"}
    ts_site = np.array([TS_SITES.index(rename.get(s, s)) for s in ts_site_label])
    extra = {"traveler_id": traveler.astype(np.int32)}
    if fc is not None:
        # the multidisorder release lists site (1-12) and subject (1-9) of the same 411 sessions
        tab = _tsv(fc / "TravelingSubjects" / "SUBINFO_TravelingSubjects.tsv")
        if len(tab) == out.size:
            tab = tab[~out].reset_index(drop=True)
            if not np.array_equal(tab["site"].to_numpy() - 1, ts_site):
                raise SystemExit("traveling-subject site order differs between the two releases")
            pairs = pd.crosstab(tab["subject"].to_numpy(), traveler)
            if not ((pairs > 0).sum(axis=1) == 1).all():
                raise SystemExit("traveling-subject identity differs between the two releases")
            extra["age"] = tab[_col(tab, "age")].to_numpy(np.float32)
    ids_ts = [f"T{t + 1}_{TS_SITES[s]}_{k:03d}" for k, (t, s) in enumerate(zip(traveler, ts_site))]
    _write(OUT / "srpbs_ts_fc.h5", Xts, ts_site, np.zeros(len(ts_site), int), ids_ts, list(TS_SITES),
           {"atlas": "268-region (Yamashita et al. 2019)", "release": "DecNef Harmonization Dataset",
            "contrast": "traveling subjects (healthy), one row per session",
            "n_excluded_motion": int(out.sum()), "n_rois": n_rois_from_edges(Xts.shape[1])}, extra)
    print("  sessions per traveler:", np.bincount(traveler).tolist())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fc-root", default=str(ROOT / "data" / "SRPBS" / "SRPBS_FC"))
    p.add_argument("--decnef-root", default=str(ROOT / "data" / "SourceCodeAndDATA" / "data"))
    p.add_argument("--only", choices=("multidisorder", "decnef"), default=None, help="build one release only")
    a = p.parse_args()
    fc, dec = Path(a.fc_root), Path(a.decnef_root)
    if a.only == "decnef":
        build_decnef(dec, fc if fc.is_dir() else None)
        return
    if a.only == "multidisorder":
        build_multidisorder(fc)
        return
    if fc.is_dir():
        build_multidisorder(fc)
    else:
        print(f"multidisorder release not found at {fc}; skipped")
    if dec.is_dir():
        build_decnef(dec, fc if fc.is_dir() else None)
    else:
        print(f"DecNef release not found at {dec}; skipped")


if __name__ == "__main__":
    main()
