"""
add_split_half_fc.py -- Augment an existing ABIDE-I HDF5 with split-half FC.

Reads the subject_ids from the HDF5, finds the raw .1D timeseries for each,
splits at the temporal midpoint, computes FC for each half (Pearson r -> Fisher z,
upper triangle), and writes fc_half1 / fc_half2 datasets back to the HDF5.

Subjects whose .1D file cannot be found or who have <4 timepoints are filled
with NaN rows so that the dataset index aligns with X.

Usage:
    python scripts/preprocessing/add_split_half_fc_abide.py
    python scripts/preprocessing/add_split_half_fc_abide.py --hdf5 data/processed/abide1_full_fc.h5
    python scripts/preprocessing/add_split_half_fc_abide.py --hdf5 data/processed/abide1_full_fc.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[2]
RAW_DIR = _REPO / "data" / "raw"
N_ROIS = 200
N_FEATURES = N_ROIS * (N_ROIS - 1) // 2  # 19 900
DERIVATIVE = "rois_cc200"


def find_timeseries_file(raw_dir: Path, file_id: str) -> Path | None:
    fname = f"{file_id}_{DERIVATIVE}.1D"
    matches = list(raw_dir.rglob(fname))
    return matches[0] if matches else None


def load_timeseries(ts_path: Path) -> np.ndarray | None:
    try:
        data = np.loadtxt(ts_path)
    except Exception:
        return None
    if data.ndim != 2 or data.shape[1] != N_ROIS:
        return None
    return data


def compute_fc_vector(ts: np.ndarray) -> np.ndarray:
    """Pearson r -> Fisher z, clipped, upper-triangle vector."""
    corr = np.corrcoef(ts.T)
    np.nan_to_num(corr, copy=False, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    z = np.arctanh(np.clip(corr, -0.9999, 0.9999))
    np.fill_diagonal(z, 0.0)
    return z[np.triu_indices(N_ROIS, k=1)].astype(np.float32)


def augment_hdf5(hdf5_path: Path, raw_dir: Path) -> None:
    """Add fc_half1 and fc_half2 datasets to an existing ABIDE-I HDF5."""
    with h5py.File(hdf5_path, "r") as f:
        subject_ids = [s.decode() if isinstance(s, bytes) else s
                       for s in f["subject_ids"][:]]
        n_subjects = len(subject_ids)
        n_features = int(f["X"].shape[1])

    if n_features != N_FEATURES:
        sys.exit(f"ERROR: expected {N_FEATURES} features, got {n_features}")

    print(f"Augmenting {hdf5_path.name} with split-half FC ({n_subjects} subjects)...")

    fc_half1 = np.full((n_subjects, N_FEATURES), np.nan, dtype=np.float32)
    fc_half2 = np.full((n_subjects, N_FEATURES), np.nan, dtype=np.float32)

    n_found = 0
    n_missing = 0
    n_too_short = 0

    for i, file_id in enumerate(tqdm(subject_ids, desc="Split-half FC")):
        ts_path = find_timeseries_file(raw_dir, file_id)
        if ts_path is None:
            n_missing += 1
            continue

        ts = load_timeseries(ts_path)
        if ts is None or ts.shape[0] < 4:
            n_too_short += 1
            continue

        mid = ts.shape[0] // 2
        fc_half1[i] = compute_fc_vector(ts[:mid])
        fc_half2[i] = compute_fc_vector(ts[mid:])
        n_found += 1

    print(f"\nSummary: found={n_found}, missing={n_missing}, too_short={n_too_short}")
    n_nan = int(np.isnan(fc_half1[:, 0]).sum())
    print(f"NaN rows (cannot use for fingerprinting): {n_nan}/{n_subjects}")

    # Write back to existing HDF5
    with h5py.File(hdf5_path, "a") as f:
        for key, arr in [("fc_half1", fc_half1), ("fc_half2", fc_half2)]:
            if key in f:
                del f[key]
                print(f"  Replaced existing '{key}' dataset.")
            f.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
            print(f"  Wrote '{key}': shape {arr.shape}")

    print(f"\nDone. HDF5 augmented at {hdf5_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add split-half FC to ABIDE-I HDF5")
    parser.add_argument(
        "--hdf5",
        type=str,
        default=str(_REPO / "data" / "processed" / "abide1_full_fc.h5"),
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(RAW_DIR),
    )
    args = parser.parse_args()

    augment_hdf5(Path(args.hdf5), Path(args.raw_dir))
