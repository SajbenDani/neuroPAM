"""
Phenotypic covariates (age, sex, mean framewise displacement) for the cohorts.

The HDF5 feature files carry only diagnosis, site and (where available) a
symptom score.  Demographic and motion covariates are read from the original
phenotypic tables and aligned to the HDF5 ``subject_ids``:

* ABIDE-I   ``Phenotypic_V1_0b_preprocessed1.csv``: FILE_ID (== subject id),
            AGE_AT_SCAN, SEX (1 = male, 2 = female), func_mean_fd
* ABIDE-II  ``ABIDEII_Composite_Usable20_Phenotypic.csv``: SUB_ID (subject id is
            ``sub-<SUB_ID>``), ``AGE_AT_SCAN `` (sic, trailing space), SEX (1/2);
            no motion summary is distributed with the phenotypes
* ADHD-200  per-site ``<SITE>_phenotypic.csv``: ScanDir ID, Age, Gender (1 = male,
            0 = female); no motion summary
* REST-meta-MDD  phenotype workbook, sheets ``MDD`` and ``Controls``: ID, Sex
            (1 = male, 2 = female), Age. Codes -9999 and blank are missing.
            Mean Power FD is read from the processed HDF5, not from the workbook.
* SRPBS     age and sex are written into the processed HDF5 by the preprocessor
            (``phenotype_kind = 'hdf5'``), because the release spreads them over
            one table per site and diagnosis.

Returned covariates: ``age`` (years), ``sex`` (1 = male, 0 = female), and
``mean_fd`` (mm) when available; missing values are NaN.  All arrays are
aligned with ``subject_ids`` and have length N.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .logging_utils import get_logger

log = get_logger(__name__)


def _to_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _align(table: pd.DataFrame, key_col: str, subject_ids: list[str], columns: dict[str, str]) -> dict[str, np.ndarray]:
    table = table.drop_duplicates(subset=key_col).set_index(key_col)
    idx = pd.Index([str(s) for s in subject_ids])
    out = {}
    for name, col in columns.items():
        if col not in table.columns:
            log.warning("phenotype column '%s' not found; covariate '%s' skipped", col, name)
            continue
        out[name] = _to_float(table[col].reindex(idx))
    return out


def load_abide1(csv_path: Path, subject_ids: list[str]) -> dict[str, np.ndarray]:
    df = pd.read_csv(csv_path)
    df["FILE_ID"] = df["FILE_ID"].astype(str)
    cov = _align(df, "FILE_ID", subject_ids, {"age": "AGE_AT_SCAN", "sex": "SEX", "mean_fd": "func_mean_fd"})
    if "sex" in cov:
        cov["sex"] = np.where(np.isfinite(cov["sex"]), (cov["sex"] == 1).astype(float), np.nan)
    return cov


def load_abide2(csv_path: Path, subject_ids: list[str]) -> dict[str, np.ndarray]:
    df = pd.read_csv(csv_path, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    df = df.assign(key="sub-" + df["SUB_ID"].astype(int).astype(str)).copy()
    cov = _align(df, "key", subject_ids, {"age": "AGE_AT_SCAN", "sex": "SEX"})
    if "sex" in cov:
        cov["sex"] = np.where(np.isfinite(cov["sex"]), (cov["sex"] == 1).astype(float), np.nan)
    return cov


def load_adhd200(root: Path, subject_ids: list[str]) -> dict[str, np.ndarray]:
    frames = []
    for csv in sorted(Path(root).rglob("*_phenotypic.csv")):
        try:
            d = pd.read_csv(csv)
        except Exception as exc:  # pragma: no cover - malformed site file
            log.warning("could not read %s: %s", csv, exc)
            continue
        d.columns = [c.strip() for c in d.columns]
        if "ScanDir ID" not in d.columns:
            continue
        d["key"] = d["ScanDir ID"].astype(int).astype(str)
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no *_phenotypic.csv under {root}")
    df = pd.concat(frames, ignore_index=True)
    ids = [str(int(s)) if str(s).isdigit() else str(s) for s in subject_ids]
    cov = _align(df, "key", ids, {"age": "Age", "sex": "Gender"})
    if "sex" in cov:
        cov["sex"] = np.where(np.isfinite(cov["sex"]), (cov["sex"] == 1).astype(float), np.nan)
    return cov


def load_restmeta(path: Path, subject_ids: list[str]) -> dict[str, np.ndarray]:
    """REST-meta-MDD workbook (sheets MDD and Controls). Sex becomes 1 = male, 0 = female."""
    from .restmeta import replace_missing
    frames = []
    for sheet in ("MDD", "Controls"):
        part = pd.read_excel(path, sheet_name=sheet)[["ID", "Sex", "Age"]].copy()
        part["ID"] = part["ID"].astype(str).str.strip()
        part[["Sex", "Age"]] = part[["Sex", "Age"]].apply(replace_missing)
        frames.append(part)
    cov = _align(pd.concat(frames, ignore_index=True), "ID", subject_ids, {"age": "Age", "sex": "Sex"})
    if "sex" in cov:
        cov["sex"] = np.where(np.isfinite(cov["sex"]), (cov["sex"] == 1).astype(float), np.nan)
    return cov


def load_hdf5(path: Path, subject_ids: list[str]) -> dict[str, np.ndarray]:
    """Covariates stored in the cohort HDF5 itself (SRPBS): ``age`` and ``sex`` (1 = male, 0 = female)."""
    import h5py
    with h5py.File(path, "r") as handle:
        stored = [s.decode() if isinstance(s, bytes) else str(s) for s in handle["subject_ids"][:]]
        index = {s: i for i, s in enumerate(stored)}
        rows = np.array([index.get(str(s), -1) for s in subject_ids])
        out = {}
        for name in ("age", "sex", "mean_fd"):
            if name in handle:
                values = handle[name][:].astype(float)
                out[name] = np.where(rows >= 0, values[np.maximum(rows, 0)], np.nan)
    return out


def _mean_fd_from_hdf5(hdf5_path, subject_ids: list[str]) -> np.ndarray | None:
    """Mean FD stored by the preprocessor (REST-meta-MDD), aligned to ``subject_ids``."""
    if hdf5_path is None or not Path(hdf5_path).exists():
        return None
    import h5py
    with h5py.File(hdf5_path, "r") as handle:
        if "mean_fd" not in handle or "subject_ids" not in handle:
            return None
        stored = [s.decode() if isinstance(s, bytes) else str(s) for s in handle["subject_ids"][:]]
        lookup = dict(zip(stored, handle["mean_fd"][:].astype(float)))
    return np.array([lookup.get(str(s), np.nan) for s in subject_ids], dtype=float)


_LOADERS = {"abide1": load_abide1, "abide2": load_abide2, "adhd200": load_adhd200, "restmeta": load_restmeta,
            "hdf5": load_hdf5}


def load_covariates(ds_entry: dict, subject_ids: list[str]) -> dict[str, np.ndarray]:
    """Dispatch on ``ds_entry['phenotype_kind']``; returns {} when no phenotype file is configured."""
    kind = ds_entry.get("phenotype_kind")
    path = ds_entry.get("phenotype")
    if not kind or path is None:
        return {}
    path = Path(path)
    if not path.exists():
        log.warning("phenotype file %s not found; covariates unavailable", path)
        return {}
    cov = _LOADERS[kind](path, list(subject_ids))
    if "mean_fd" not in cov:
        fd = _mean_fd_from_hdf5(ds_entry.get("hdf5_path"), list(subject_ids))
        if fd is not None:
            cov["mean_fd"] = fd
    # drop covariates that are essentially missing
    return {k: v for k, v in cov.items() if np.isfinite(v).sum() >= 0.5 * len(v)}
