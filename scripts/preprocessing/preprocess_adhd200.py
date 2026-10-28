#!/usr/bin/env python3
"""
preprocess_adhd200.py -- ADHD-200 Preprocessing Pipeline

Processes ADHD-200 resting-state fMRI data into functional connectivity matrices
matching the format of our ABIDE preprocessing.

Data Sources:
  Option 1: nilearn.datasets.fetch_adhd() - 40 subjects, preprocessed (quick demo)
  Option 2: Full ADHD-200 from NITRC Athena CC200 - 900+ subjects (publication)
  Option 3: NITRC FCON1000 per-site layout (downloaded from NITRC site pages)

This script supports:
  - `--source nilearn`:    Use nilearn's 40-subject preprocessed data (default)
  - `--source nitrc`:      Use NITRC Athena flat format ({id}_rois_cc200.1D)
  - `--source nitrc-fcon`: Use NITRC FCON1000 per-site layout (actual download format)
                           Layout: {raw_dir}/{SITE}/{SUBJECT_ID}/sfnwmrda{id}_session_1_rest_1_cc200_TCs.1D
                           Phenotypic: {raw_dir}/{SITE}/{SITE}_phenotypic.csv

Usage:
    # Quick start with nilearn (40 subjects, 8 sites)
    python scripts/preprocessing/preprocess_adhd200.py --source nilearn --out-dir data/processed

    # Full dataset from NITRC Athena CC200 (flat format)
    python scripts/preprocessing/preprocess_adhd200.py --source nitrc --raw-dir data/raw/adhd200 --out-dir data/processed

    # Full dataset from NITRC FCON1000 per-site download (actual format)
    python scripts/preprocessing/preprocess_adhd200.py --source nitrc-fcon --raw-dir data/ADHD-200 --out-dir data/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

# -- Constants ----------------------------------------------------------------

N_ROIS = 200
N_FEATURES = N_ROIS * (N_ROIS - 1) // 2  # 19 900

# ADHD-200 Sites
SITE_NAME_MAP = {
    "Peking_1": "Peking",
    "Peking_2": "Peking", 
    "Peking_3": "Peking",
    "Peking University": "Peking",
    "KKI": "KKI",
    "Kennedy Krieger Institute": "KKI",
    "NeuroIMAGE": "NI",
    "NYU": "NYU",
    "New York University Child Study Center": "NYU",
    "OHSU": "OHSU",
    "Oregon Health and Science University": "OHSU",
    "Pittsburgh": "Pitt",
    "University of Pittsburgh": "Pitt",
    "WashU": "WashU",
    "Washington University in St. Louis": "WashU",
    "Brown": "Brown",
    "Brown University": "Brown",
}

# DX codes: 0=TDC, 1=ADHD-Combined, 2=ADHD-H/I, 3=ADHD-Inattentive
# Binary: ADHD (1,2,3) vs TDC (0)


# -- Dataset presets ----------------------------------------------------------

DATASET_PRESETS = {
    "adhd200_full": {
        "sites": ["Peking", "KKI", "NI", "NYU", "OHSU", "Pitt", "WashU", "Brown"],
        "out_name": "adhd200_full_fc.h5",
    },
    "adhd200_5site": {
        "sites": ["Peking", "KKI", "NYU", "OHSU", "Pitt"],
        "out_name": "adhd200_5site_fc.h5",
    },
    "adhd200_abide_overlap": {
        "sites": ["KKI", "NYU", "OHSU", "Pitt"],
        "out_name": "adhd200_overlap_fc.h5",
    },
}


# -- Helpers ------------------------------------------------------------------

def compute_fc_from_timeseries(ts: np.ndarray) -> np.ndarray:
    """Pearson correlation -> Fisher z-transform -> upper-triangle vector."""
    corr = np.corrcoef(ts.T)  # (n_rois, n_rois)
    np.nan_to_num(corr, copy=False, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    z = np.arctanh(np.clip(corr, -0.9999, 0.9999))
    np.fill_diagonal(z, 0.0)
    n_rois = corr.shape[0]
    return z[np.triu_indices(n_rois, k=1)]


def make_pca_plot(
    X: np.ndarray,
    y_site: np.ndarray,
    site_names: list[str],
    out_path: Path,
) -> None:
    """2-component PCA coloured by site, saved as PNG."""
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(site_names))))
    for sid, name in enumerate(site_names):
        sel = y_site == sid
        if not np.any(sel):
            continue
        ax.scatter(
            X2[sel, 0], X2[sel, 1],
            label=name, alpha=0.5, s=15, color=colors[sid],
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("PCA of Raw FC Vectors -- ADHD-200 Site Effect")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"PCA plot saved to {out_path}")


# -- Nilearn pipeline (40 subjects) -------------------------------------------

def run_nilearn_pipeline(
    out_dir: Path,
    out_name: str = "adhd200_nilearn_fc.h5",
    data_dir: Path | None = None,
) -> Path:
    """Download ADHD-200 via nilearn and build FC matrices.
    
    Uses CC200 atlas for parcellation to match our ABIDE preprocessing.
    nilearn provides 40 subjects across multiple sites.
    """
    try:
        from nilearn import datasets
        from nilearn.maskers import NiftiLabelsMasker
        from nilearn.connectome import ConnectivityMeasure
    except ImportError:
        sys.exit("ERROR: nilearn not installed. Run: pip install nilearn")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Fetching ADHD-200 data via nilearn...")
    adhd_data = datasets.fetch_adhd(
        n_subjects=None,  # Get all 40 subjects
        data_dir=str(data_dir) if data_dir else None,
    )
    
    print("Fetching CC200 atlas...")
    atlas = datasets.fetch_atlas_craddock_2012()
    # Use the 200-region parcellation
    atlas_filename = atlas.scorr_mean  # 200 regions
    
    print(f"  Found {len(adhd_data.func)} subjects")
    print(f"  Atlas: {atlas_filename}")
    
    # Extract timeseries using CC200 atlas
    masker = NiftiLabelsMasker(
        labels_img=atlas_filename,
        standardize=True,
        memory='nilearn_cache',
        verbose=0,
    )
    
    fc_vectors = []
    diagnoses = []
    sites_arr = []
    subject_ids = []
    
    # Parse phenotypic data
    pheno = adhd_data.phenotypic
    site_names_raw = pheno['Site'].unique().tolist()
    site_names = sorted(set(SITE_NAME_MAP.get(s, s) for s in site_names_raw))
    site_label_map = {name: idx for idx, name in enumerate(site_names)}
    
    print(f"  Sites: {site_names}")
    
    for i, func_file in enumerate(tqdm(adhd_data.func, desc="Extracting FC")):
        try:
            # Extract timeseries
            ts = masker.fit_transform(func_file)  # (n_timepoints, n_rois)
            
            # Build FC matrix
            fc = compute_fc_from_timeseries(ts)
            
            # Get phenotypic info
            row = pheno.iloc[i]
            site_raw = row['Site']
            site = SITE_NAME_MAP.get(site_raw, site_raw)
            
            # ADHD label (assuming column exists)
            if 'ADHD' in pheno.columns:
                dx = 1 if row['ADHD'] == 1 else 0
            elif 'DX' in pheno.columns:
                dx = 1 if row['DX'] > 0 else 0
            else:
                dx = 0  # Default TDC if not available
            
            fc_vectors.append(fc)
            diagnoses.append(dx)
            sites_arr.append(site_label_map.get(site, 0))
            subject_ids.append(str(row.get('Subject', f'sub_{i}')))
            
        except Exception as e:
            print(f"  Warning: Failed to process subject {i}: {e}")
            continue
    
    if not fc_vectors:
        sys.exit("ERROR: No valid FC matrices produced.")
    
    X = np.array(fc_vectors, dtype=np.float32)
    y_dx = np.array(diagnoses, dtype=np.int32)
    y_site = np.array(sites_arr, dtype=np.int32)
    
    # Sanity checks
    print(f"\n=== Results ===")
    print(f"  Total subjects: {X.shape[0]}")
    print(f"  FC features: {X.shape[1]}")
    for sid, name in enumerate(site_names):
        n = int((y_site == sid).sum())
        print(f"  {name}: n={n}")
    
    # Save HDF5
    h5_path = out_dir / out_name
    print(f"\nSaving HDF5 -> {h5_path}")
    
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X", data=X, compression="gzip", compression_opts=4)
        f.create_dataset("y_diagnosis", data=y_dx)
        f.create_dataset("y_site", data=y_site)
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("subject_ids", data=subject_ids, dtype=dt)
        
        f.attrs["site_names"] = site_names
        f.attrs["n_subjects"] = X.shape[0]
        f.attrs["n_sites"] = len(site_names)
        f.attrs["dataset"] = "ADHD-200"
        f.attrs["source"] = "nilearn"
        f.attrs["parcellation"] = "CC200"
    
    # PCA plot
    pca_path = out_dir / out_name.replace(".h5", "_pca.png")
    if X.shape[0] >= 3:
        make_pca_plot(X, y_site, site_names, pca_path)
    
    print(f"\nPipeline complete. {X.shape[0]} subjects, {len(site_names)} sites.")
    return h5_path


# -- NITRC FCON1000 per-site pipeline (actual download format) ----------------

# Directory names -> canonical site labels (merges Peking_1/2/3)
FCON_SITE_MERGE: dict[str, str] = {
    "Peking_1": "Peking",
    "Peking_2": "Peking",
    "Peking_3": "Peking",
    "KKI": "KKI",
    "NeuroIMAGE": "NeuroIMAGE",
    "NYU": "NYU",
    "OHSU": "OHSU",
    "Pittsburgh": "Pittsburgh",
    "WashU": "WashU",
}


def load_timeseries_1d_fcon(ts_path: Path) -> np.ndarray | None:
    """Load a FCON1000 AFNI ROIstats .1D file -> (T, N_ROIs) array.

    FCON1000 CC200 files use AFNI 3dROIstats format:
        Row 0 (header): File  Sub-brick  Mean_1  Mean_3  ...
        Rows 1+:         <filename>  <brick[?]>  <val> ...
    Skips the first 2 columns (File, Sub-brick) and returns remaining values.
    Returns None on parse failure.
    """
    try:
        import pandas as pd
        df = pd.read_csv(ts_path, sep="\t")
        if df.shape[1] < 3:
            return None
        data = df.iloc[:, 2:].values.astype(np.float64)
        if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
            return None
        return data
    except Exception:
        return None


def find_timeseries_fcon(site_dir: Path, subject_id: str) -> Path | None:
    """Find CC200 timeseries for FCON1000 per-site layout.

    Tries both the raw subject_id and a 7-digit zero-padded version as directory
    names (NYU stores some subjects as 0010001 when ScanDir ID is 10001).

    File preference order:
        sfnwmrda{dir_id}_session_1_rest_1_cc200_TCs.1D  (scrubbed+filtered)
        snwmrda{dir_id}_session_1_rest_1_cc200_TCs.1D   (filtered, no scrubbing)
    """
    # Try both raw and zero-padded (7-digit) directory names
    dir_ids = [subject_id, subject_id.zfill(7)]
    for dir_id in dict.fromkeys(dir_ids):  # deduplicate while preserving order
        subj_dir = site_dir / dir_id
        if not subj_dir.exists():
            continue
        for prefix in ("sfnwmrda", "snwmrda"):
            candidate = subj_dir / f"{prefix}{dir_id}_session_1_rest_1_cc200_TCs.1D"
            if candidate.exists():
                return candidate
    return None


def run_nitrc_fcon_pipeline(
    raw_dir: Path,
    out_dir: Path,
    out_name: str = "adhd200_fc.h5",
    test_mode: bool = False,
) -> Path:
    """Process NITRC FCON1000-style per-site ADHD-200 download into FC matrices.

    Expected layout::

        {raw_dir}/{SITE}/{SITE}_phenotypic.csv
        {raw_dir}/{SITE}/{SUBJECT_ID}/sfnwmrda{id}_session_1_rest_1_cc200_TCs.1D

    Merges Peking_1, Peking_2, Peking_3 into a single "Peking" site.
    Quality filter: QC_Rest_1 == 1 (if column present).
    DX: 0 -> TDC (0), 1/2/3 -> ADHD (1). Rows with DX<0 or non-numeric are skipped.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover site directories that have a matching phenotypic CSV
    site_dirs = sorted([
        d for d in raw_dir.iterdir()
        if d.is_dir() and (d / f"{d.name}_phenotypic.csv").exists()
    ])
    if not site_dirs:
        sys.exit(f"ERROR: No site directories with phenotypic CSVs found in {raw_dir}")

    print(f"Found {len(site_dirs)} site directories: {[d.name for d in site_dirs]}")

    all_subjects: list[tuple[Path, str, int, str]] = []  # (site_dir, subject_id, dx, canonical)

    for site_dir in site_dirs:
        raw_site = site_dir.name
        canonical = FCON_SITE_MERGE.get(raw_site, raw_site)

        pheno = pd.read_csv(site_dir / f"{raw_site}_phenotypic.csv")
        pheno.columns = [c.strip().strip('"') for c in pheno.columns]

        # Quality filter
        if "QC_Rest_1" in pheno.columns:
            pheno = pheno[pheno["QC_Rest_1"] == 1].copy()

        # Parse and validate DX
        pheno["_dx_num"] = pd.to_numeric(pheno["DX"], errors="coerce")
        pheno = pheno[pheno["_dx_num"].notna() & (pheno["_dx_num"] >= 0)].copy()
        pheno["_dx_num"] = pheno["_dx_num"].astype(int)

        # Detect subject ID column -- some sites use "ScanDir ID", others "ScanDirID"
        id_col = next(
            (c for c in pheno.columns if c.replace(" ", "").lower() == "scandirid"),
            pheno.columns[0],
        )
        for _, row in pheno.iterrows():
            subject_id = str(int(row[id_col]))
            dx = 1 if row["_dx_num"] > 0 else 0
            all_subjects.append((site_dir, subject_id, dx, canonical))

    # Sorted canonical site list (determines y_site indices)
    canonical_sites = sorted(set(s[3] for s in all_subjects))
    site_label_map = {name: idx for idx, name in enumerate(canonical_sites)}

    print(f"Canonical sites ({len(canonical_sites)}): {canonical_sites}")
    print(f"Total subjects after QC: {len(all_subjects)}")

    fc_vectors: list[np.ndarray] = []
    diagnoses: list[int] = []
    sites_arr: list[int] = []
    subject_ids: list[str] = []
    site_counts: dict[str, int] = {s: 0 for s in canonical_sites}
    n_missing = 0

    for site_dir, subject_id, dx, canonical in tqdm(all_subjects, desc="Building FC"):
        if test_mode and site_counts.get(canonical, 0) >= 3:
            continue

        ts_path = find_timeseries_fcon(site_dir, subject_id)
        if ts_path is None:
            n_missing += 1
            continue

        ts = load_timeseries_1d_fcon(ts_path)
        if ts is None:
            n_missing += 1
            continue

        fc = compute_fc_from_timeseries(ts)
        # Allow any consistent feature size (FCON1000 CC200 has 190 ROIs -> 17955 features)
        if fc_vectors and fc.shape[0] != fc_vectors[0].shape[0]:
            n_missing += 1
            continue

        fc_vectors.append(fc)
        diagnoses.append(dx)
        sites_arr.append(site_label_map[canonical])
        subject_ids.append(subject_id)
        site_counts[canonical] = site_counts.get(canonical, 0) + 1

    if n_missing:
        print(f"  Warning: {n_missing} subjects skipped (missing/invalid timeseries)")

    if not fc_vectors:
        sys.exit("ERROR: No valid FC matrices produced. Check file naming and QC filters.")

    X = np.array(fc_vectors, dtype=np.float32)
    y_dx = np.array(diagnoses, dtype=np.int32)
    y_site = np.array(sites_arr, dtype=np.int32)

    print(f"\n=== Results ===")
    print(f"  Total subjects: {X.shape[0]}")
    for sid, name in enumerate(canonical_sites):
        n = int((y_site == sid).sum())
        n_adhd = int(((y_site == sid) & (y_dx == 1)).sum())
        print(f"  {name}: n={n} (ADHD={n_adhd}, TDC={n - n_adhd})")

    h5_path = out_dir / out_name
    print(f"\nSaving HDF5 -> {h5_path}")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X", data=X, compression="gzip", compression_opts=4)
        f.create_dataset("y_diagnosis", data=y_dx)
        f.create_dataset("y_site", data=y_site)
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("subject_ids", data=subject_ids, dtype=dt)
        f.attrs["site_names"] = canonical_sites
        f.attrs["n_subjects"] = X.shape[0]
        f.attrs["n_sites"] = len(canonical_sites)
        f.attrs["dataset"] = "ADHD-200"
        f.attrs["source"] = "nitrc_fcon1000"
        f.attrs["parcellation"] = "CC200"

    pca_path = out_dir / out_name.replace(".h5", "_pca.png")
    if X.shape[0] >= 3:
        make_pca_plot(X, y_site, canonical_sites, pca_path)

    print(f"\nPipeline complete. {X.shape[0]} subjects, {len(canonical_sites)} sites.")
    return h5_path


# -- NITRC Athena pipeline (full dataset) -------------------------------------

def load_timeseries_1d(ts_path: Path) -> np.ndarray | None:
    """Load a .1D timeseries file -> (T, 200) array."""
    try:
        data = np.loadtxt(ts_path)
    except Exception:
        return None
    if data.ndim != 2 or data.shape[1] != N_ROIS:
        return None
    return data


def find_timeseries_file(raw_dir: Path, subject_id: str) -> Path | None:
    """Locate the CC200 .1D timeseries file for a given subject ID."""
    fname = f"{subject_id}_rois_cc200.1D"
    
    # Try direct match
    direct = raw_dir / fname
    if direct.exists():
        return direct
    
    # Try recursive search
    matches = list(raw_dir.rglob(fname))
    if matches:
        return matches[0]
    
    return None


def run_nitrc_pipeline(
    raw_dir: Path,
    out_dir: Path,
    sites: list[str],
    out_name: str = "adhd200_fc.h5",
    test_mode: bool = False,
) -> Path:
    """Process NITRC Athena CC200 timeseries into FC matrices."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    site_names = sites
    site_label_map = {name: idx for idx, name in enumerate(site_names)}
    
    # Load phenotypic data
    pheno_path = raw_dir / "adhd200_preprocessed_phenotypics.tsv"
    if not pheno_path.exists():
        alt_path = raw_dir.parent / "adhd200_phenotypics.tsv"
        if alt_path.exists():
            pheno_path = alt_path
        else:
            sys.exit(f"ERROR: Phenotypic file not found. Expected: {pheno_path}")
    
    print("Loading phenotypic data...")
    pheno = pd.read_csv(pheno_path, sep="\t")
    pheno.columns = [c.strip() for c in pheno.columns]
    
    # Map site names
    if "Site" in pheno.columns:
        pheno["site_canonical"] = pheno["Site"].map(
            lambda s: SITE_NAME_MAP.get(str(s).strip(), str(s).strip())
        )
    
    pheno = pheno[pheno["site_canonical"].isin(sites)].copy()
    print(f"  Subjects at target sites: {len(pheno)}")
    
    # Build FC matrices
    fc_vectors = []
    diagnoses = []
    sites_arr = []
    subject_ids = []
    
    if test_mode:
        print("  [TEST MODE] Processing max 3 subjects per site")
    site_counts = {s: 0 for s in site_names}
    
    id_col = "ScanDir ID" if "ScanDir ID" in pheno.columns else pheno.columns[0]
    dx_col = "DX" if "DX" in pheno.columns else "Dx"
    
    for _, row in tqdm(pheno.iterrows(), total=len(pheno), desc="Building FC"):
        subject_id = str(row[id_col]).strip()
        site = row["site_canonical"]
        
        if test_mode and site_counts.get(site, 0) >= 3:
            continue
        
        ts_path = find_timeseries_file(raw_dir, subject_id)
        if ts_path is None:
            continue
        
        ts = load_timeseries_1d(ts_path)
        if ts is None:
            continue
        
        fc = compute_fc_from_timeseries(ts)
        if fc.shape[0] != N_FEATURES:
            continue
        
        raw_dx = int(row.get(dx_col, 0))
        dx = 1 if raw_dx > 0 else 0
        
        fc_vectors.append(fc)
        diagnoses.append(dx)
        sites_arr.append(site_label_map[site])
        subject_ids.append(subject_id)
        site_counts[site] = site_counts.get(site, 0) + 1
    
    if not fc_vectors:
        sys.exit("ERROR: No valid FC matrices. Check that CC200 .1D files exist.")
    
    X = np.array(fc_vectors, dtype=np.float32)
    y_dx = np.array(diagnoses, dtype=np.int32)
    y_site = np.array(sites_arr, dtype=np.int32)
    
    print(f"\n=== Results ===")
    print(f"  Total subjects: {X.shape[0]}")
    for sid, name in enumerate(site_names):
        n = int((y_site == sid).sum())
        n_adhd = int(((y_site == sid) & (y_dx == 1)).sum())
        print(f"  {name}: n={n} (ADHD={n_adhd})")
    
    # Save HDF5
    h5_path = out_dir / out_name
    print(f"\nSaving HDF5 -> {h5_path}")
    
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X", data=X, compression="gzip", compression_opts=4)
        f.create_dataset("y_diagnosis", data=y_dx)
        f.create_dataset("y_site", data=y_site)
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("subject_ids", data=subject_ids, dtype=dt)
        
        f.attrs["site_names"] = site_names
        f.attrs["n_subjects"] = X.shape[0]
        f.attrs["n_sites"] = len(site_names)
        f.attrs["dataset"] = "ADHD-200"
        f.attrs["source"] = "nitrc_athena"
        f.attrs["parcellation"] = "CC200"
    
    # PCA plot
    pca_path = out_dir / out_name.replace(".h5", "_pca.png")
    if X.shape[0] >= 3:
        make_pca_plot(X, y_site, site_names, pca_path)
    
    print(f"\nPipeline complete. {X.shape[0]} subjects, {len(site_names)} sites.")
    return h5_path


# -- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ADHD-200 preprocessing: timeseries -> FC matrices -> HDF5"
    )
    parser.add_argument(
        "--source", type=str, default="nilearn",
        choices=["nilearn", "nitrc", "nitrc-fcon"],
        help=(
            "Data source: nilearn (40 subjects), nitrc (Athena flat format), "
            "or nitrc-fcon (FCON1000 per-site layout -- actual NITRC download)"
        ),
    )
    parser.add_argument(
        "--dataset", type=str, default="adhd200_full",
        help="Dataset preset (for nitrc source): adhd200_full, adhd200_5site",
    )
    parser.add_argument(
        "--sites", type=str, default=None,
        help="Comma-separated site names (overrides --dataset)",
    )
    parser.add_argument(
        "--raw-dir", type=str, default="data/raw/adhd200",
        help="Directory with CC200 .1D files (for nitrc source)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="data/processed",
        help="Output directory for HDF5",
    )
    parser.add_argument(
        "--out-name", type=str, default=None,
        help="Output HDF5 filename",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: process max 3 subjects per site",
    )
    args = parser.parse_args()
    
    if args.source == "nilearn":
        out_name = args.out_name or "adhd200_nilearn_fc.h5"
        run_nilearn_pipeline(
            out_dir=Path(args.out_dir),
            out_name=out_name,
        )
    elif args.source == "nitrc-fcon":
        out_name = args.out_name or "adhd200_fc.h5"
        run_nitrc_fcon_pipeline(
            raw_dir=Path(args.raw_dir),
            out_dir=Path(args.out_dir),
            out_name=out_name,
            test_mode=args.test,
        )
    else:  # nitrc (Athena flat format)
        if args.sites:
            sites = [s.strip() for s in args.sites.split(",")]
            out_name = args.out_name or "adhd200_custom_fc.h5"
        else:
            preset = DATASET_PRESETS.get(args.dataset)
            if not preset:
                sys.exit(f"Unknown dataset: {args.dataset}")
            sites = preset["sites"]
            out_name = args.out_name or preset["out_name"]

        run_nitrc_pipeline(
            raw_dir=Path(args.raw_dir),
            out_dir=Path(args.out_dir),
            sites=sites,
            out_name=out_name,
            test_mode=args.test,
        )
