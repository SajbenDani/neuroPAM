"""
add_split_half_fc_adhd200.py -- Augment ADHD-200 HDF5 with split-half FC.

Reads subject_ids and y_site from the HDF5, finds the FCON1000-format
.1D timeseries for each subject, splits at the temporal midpoint,
computes FC for each half (Pearson r -> Fisher z, upper triangle),
and writes fc_half1 / fc_half2 datasets back to the HDF5.

Usage:
    python scripts/preprocessing/add_split_half_fc_adhd200.py
    python scripts/preprocessing/add_split_half_fc_adhd200.py --hdf5 data/processed/adhd200_fc.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[2]
ADHD200_DIR = _REPO / "data" / "ADHD-200"
N_ROIS = 190
N_FEATURES = N_ROIS * (N_ROIS - 1) // 2  # 17 955

# Maps canonical site name (as stored in HDF5) to raw directories in ADHD200_DIR
CANONICAL_TO_RAW_DIRS = {
    "KKI": ["KKI"],
    "NYU": ["NYU"],
    "NeuroIMAGE": ["NeuroIMAGE"],
    "OHSU": ["OHSU"],
    "Peking": ["Peking_1", "Peking_2", "Peking_3"],
    "Pittsburgh": ["Pittsburgh"],
    "WashU": ["WashU"],
}


def load_timeseries_1d_fcon(ts_path: Path) -> np.ndarray | None:
    """Load a FCON1000 AFNI ROIstats .1D file -> (T, N_ROIs) array."""
    try:
        df = pd.read_csv(ts_path, sep="\t")
        if df.shape[1] < 3:
            return None
        data = df.iloc[:, 2:].values.astype(np.float64)
        if data.ndim != 2 or data.shape[0] < 4 or data.shape[1] != N_ROIS:
            return None
        return data
    except Exception:
        return None


def find_timeseries_fcon(site_dir: Path, subject_id: str) -> Path | None:
    """Find CC200 timeseries for FCON1000 layout (handles NYU dual naming)."""
    dir_ids = list(dict.fromkeys([subject_id, subject_id.zfill(7)]))
    for dir_id in dir_ids:
        subj_dir = site_dir / dir_id
        if not subj_dir.exists():
            continue
        for prefix in ("sfnwmrda", "snwmrda"):
            candidate = subj_dir / f"{prefix}{dir_id}_session_1_rest_1_cc200_TCs.1D"
            if candidate.exists():
                return candidate
    return None


def compute_fc_vector(ts: np.ndarray) -> np.ndarray:
    """Pearson r -> Fisher z, clipped, upper-triangle vector."""
    corr = np.corrcoef(ts.T)
    np.nan_to_num(corr, copy=False, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    z = np.arctanh(np.clip(corr, -0.9999, 0.9999))
    np.fill_diagonal(z, 0.0)
    return z[np.triu_indices(N_ROIS, k=1)].astype(np.float32)


def augment_hdf5(hdf5_path: Path, adhd200_dir: Path) -> None:
    with h5py.File(hdf5_path, "r") as f:
        raw_ids = f["subject_ids"][:]
        subject_ids = [s.decode() if isinstance(s, bytes) else s for s in raw_ids]
        y_site = f["y_site"][:]
        site_names = list(f.attrs.get("site_names", []))
        n_subjects = len(subject_ids)

    print(f"Augmenting {hdf5_path.name} with split-half FC ({n_subjects} subjects)...")

    fc_half1 = np.full((n_subjects, N_FEATURES), np.nan, dtype=np.float32)
    fc_half2 = np.full((n_subjects, N_FEATURES), np.nan, dtype=np.float32)

    stats = {"found": 0, "missing": 0, "too_short": 0, "wrong_rois": 0}

    for i, (subj_id, site_idx) in enumerate(tqdm(
        zip(subject_ids, y_site), total=n_subjects, desc="Split-half FC"
    )):
        canonical = site_names[site_idx] if site_idx < len(site_names) else "unknown"
        raw_dirs = CANONICAL_TO_RAW_DIRS.get(canonical, [canonical])

        ts_path = None
        for raw_dir_name in raw_dirs:
            site_dir = adhd200_dir / raw_dir_name
            ts_path = find_timeseries_fcon(site_dir, subj_id)
            if ts_path is not None:
                break

        if ts_path is None:
            stats["missing"] += 1
            continue

        ts = load_timeseries_1d_fcon(ts_path)
        if ts is None:
            stats["missing"] += 1
            continue
        if ts.shape[1] != N_ROIS:
            stats["wrong_rois"] += 1
            continue
        if ts.shape[0] < 4:
            stats["too_short"] += 1
            continue

        mid = ts.shape[0] // 2
        fc_half1[i] = compute_fc_vector(ts[:mid])
        fc_half2[i] = compute_fc_vector(ts[mid:])
        stats["found"] += 1

    print(f"\nSummary: found={stats['found']}, missing={stats['missing']}, "
          f"too_short={stats['too_short']}, wrong_rois={stats['wrong_rois']}")
    nan_rows = np.sum(np.all(np.isnan(fc_half1), axis=1))
    print(f"NaN rows (cannot use for fingerprinting): {nan_rows}/{n_subjects}")

    with h5py.File(hdf5_path, "a") as f:
        for key, data in [("fc_half1", fc_half1), ("fc_half2", fc_half2)]:
            if key in f:
                del f[key]
            f.create_dataset(key, data=data, compression="gzip", compression_opts=4)
            print(f"  Wrote '{key}': shape {data.shape}")

    print(f"\nDone. HDF5 augmented at {hdf5_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add split-half FC to ADHD-200 HDF5")
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=_REPO / "data" / "processed" / "adhd200_fc.h5",
        help="Path to the ADHD-200 HDF5 file",
    )
    parser.add_argument(
        "--adhd200-dir",
        type=Path,
        default=ADHD200_DIR,
        help="Root directory of ADHD-200 FCON1000 data",
    )
    args = parser.parse_args()

    if not args.hdf5.exists():
        sys.exit(f"ERROR: HDF5 not found: {args.hdf5}")
    if not args.adhd200_dir.exists():
        sys.exit(f"ERROR: ADHD-200 dir not found: {args.adhd200_dir}")

    augment_hdf5(args.hdf5, args.adhd200_dir)


if __name__ == "__main__":
    main()
