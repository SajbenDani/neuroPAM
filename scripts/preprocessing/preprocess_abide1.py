#!/usr/bin/env python3
"""
scripts/preprocessing/preprocess_abide1.py -- ABIDE-I Preprocessing Pipeline

Steps:
  1. Quality control filtering using phenotypic CSV
  2. Build 200x200 functional connectivity matrices from CC200 ROI timeseries
  3. Save HDF5 dataset + summary CSV + PCA site-effect plot

Usage:
    python scripts/preprocessing/preprocess_abide1.py --raw-dir data/raw --out-dir data/processed
    python scripts/preprocessing/preprocess_abide1.py --dataset abide1_full --raw-dir data/raw --out-dir data/processed
    python scripts/preprocessing/preprocess_abide1.py --sites NYU,USM,UCLA_1,UCLA_2 --out-name custom.h5
    python scripts/preprocessing/preprocess_abide1.py --test
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

QC_THRESHOLDS = {
    "qc_rater_1": "OK",
    "func_perc_fd_max": 20.0,    # percent (>20 % high-motion frames)
    "func_mean_fd_max": 0.25,    # mm
}

DERIVATIVE = "rois_cc200"
PIPELINE = "cpac"
STRATEGY = "filt_global"

# -- Dataset presets ----------------------------------------------------------
# Each preset: (raw_site_ids, canonical_map, output_filename)
# canonical_map merges sub-sites (e.g. UCLA_1 + UCLA_2 -> UCLA)

DATASET_PRESETS = {
    "abide1_3site": {
        "raw_sites": ["NYU", "USM", "UCLA_1", "UCLA_2"],
        "canonical_map": {
            "NYU": "NYU", "USM": "USM",
            "UCLA_1": "UCLA", "UCLA_2": "UCLA",
        },
        "out_name": "abide_3site_fc.h5",
    },
    "abide1_8site": {
        "raw_sites": [
            "NYU", "USM", "UCLA_1", "UCLA_2", "CALTECH",
            "KKI", "LEUVEN_1", "LEUVEN_2", "MAX_MUN", "TRINITY",
        ],
        "canonical_map": {
            "NYU": "NYU", "USM": "USM",
            "UCLA_1": "UCLA", "UCLA_2": "UCLA",
            "CALTECH": "CALTECH", "KKI": "KKI",
            "LEUVEN_1": "LEUVEN", "LEUVEN_2": "LEUVEN",
            "MAX_MUN": "MAX_MUN", "TRINITY": "TRINITY",
        },
        "out_name": "abide_8site_fc.h5",
    },
    "abide1_full": {
        "raw_sites": [
            "CALTECH", "CMU", "KKI", "LEUVEN_1", "LEUVEN_2",
            "MAX_MUN", "NYU", "OHSU", "OLIN", "PITT", "SBL",
            "SDSU", "STANFORD", "TRINITY", "UCLA_1", "UCLA_2",
            "UM_1", "UM_2", "USM", "YALE",
        ],
        "canonical_map": {
            "CALTECH": "CALTECH", "CMU": "CMU", "KKI": "KKI",
            "LEUVEN_1": "LEUVEN", "LEUVEN_2": "LEUVEN",
            "MAX_MUN": "MAX_MUN", "NYU": "NYU", "OHSU": "OHSU",
            "OLIN": "OLIN", "PITT": "PITT", "SBL": "SBL",
            "SDSU": "SDSU", "STANFORD": "STANFORD", "TRINITY": "TRINITY",
            "UCLA_1": "UCLA", "UCLA_2": "UCLA",
            "UM_1": "UM", "UM_2": "UM",
            "USM": "USM", "YALE": "YALE",
        },
        "out_name": "abide1_full_fc.h5",
    },
    # ABIDE-II: 19 sites, cpac/filt_global/rois_cc200 pipeline
    # Sub-sites with trailing numbers are merged to canonical names.
    "abide2": {
        "raw_sites": [
            "BNI_1", "ETH_1", "GU_1", "IP_1", "IU_1",
            "KKI_1", "NYU_1", "NYU_2", "OHSU_1", "ONRC_2",
            "OILH_1", "SBL_1", "SDSU_1", "SEO_1", "STANFORD_1",
            "TCD_1", "UCLA_1", "UCLA_2", "UCLA_3", "UCD_1",
            "UM_1", "UM_2", "UM_3", "USM_1",
        ],
        "canonical_map": {
            "BNI_1": "BNI", "ETH_1": "ETH", "GU_1": "GU",
            "IP_1": "IP", "IU_1": "IU", "KKI_1": "KKI",
            "NYU_1": "NYU", "NYU_2": "NYU", "OHSU_1": "OHSU",
            "ONRC_2": "ONRC", "OILH_1": "OILH", "SBL_1": "SBL",
            "SDSU_1": "SDSU", "SEO_1": "SEO", "STANFORD_1": "STANFORD",
            "TCD_1": "TCD", "UCLA_1": "UCLA", "UCLA_2": "UCLA",
            "UCLA_3": "UCLA", "UCD_1": "UCD", "UM_1": "UM",
            "UM_2": "UM", "UM_3": "UM", "USM_1": "USM",
        },
        "out_name": "abide2_fc.h5",
        "pheno_csv": "abide2_Phenotypic.csv",  # downloaded to data/raw/
    },
}


# -- Helpers ------------------------------------------------------------------

def load_phenotypic(pheno_path: Path) -> pd.DataFrame:
    """Load and lightly clean the ABIDE phenotypic CSV."""
    df = pd.read_csv(pheno_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def apply_qc(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows passing all three QC criteria."""
    mask = pd.Series(True, index=df.index)

    if "qc_rater_1" in df.columns:
        mask &= df["qc_rater_1"].astype(str).str.strip().str.upper() == "OK"

    if "func_perc_fd" in df.columns:
        col = pd.to_numeric(df["func_perc_fd"], errors="coerce")
        mask &= col.notna() & (col <= QC_THRESHOLDS["func_perc_fd_max"])

    if "func_mean_fd" in df.columns:
        col = pd.to_numeric(df["func_mean_fd"], errors="coerce")
        mask &= col.notna() & (col <= QC_THRESHOLDS["func_mean_fd_max"])

    return df[mask].copy()


def find_timeseries_file(raw_dir: Path, file_id: str) -> Path | None:
    """Locate the .1D timeseries file for a given FILE_ID.

    Searches raw_dir recursively; also checks raw_dir/abide2/ for ABIDE-II data.
    """
    fname = f"{file_id}_{DERIVATIVE}.1D"
    matches = list(raw_dir.rglob(fname))
    if matches:
        return matches[0]
    # Also check abide2 subdirectory
    abide2_dir = raw_dir / "abide2"
    if abide2_dir.exists():
        matches = list(abide2_dir.rglob(fname))
        if matches:
            return matches[0]
    return None


def load_timeseries(ts_path: Path) -> np.ndarray | None:
    """Load a .1D timeseries file -> (T, 200) array."""
    try:
        data = np.loadtxt(ts_path)
    except Exception:
        return None
    if data.ndim != 2 or data.shape[1] != N_ROIS:
        return None
    return data


def compute_fc(ts: np.ndarray) -> np.ndarray:
    """Pearson correlation -> Fisher z-transform -> upper-triangle vector."""
    corr = np.corrcoef(ts.T)  # (200, 200)
    np.nan_to_num(corr, copy=False, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    z = np.arctanh(np.clip(corr, -0.9999, 0.9999))
    np.fill_diagonal(z, 0.0)
    return z[np.triu_indices(N_ROIS, k=1)]


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
    ax.set_title("PCA of Raw FC Vectors -- Site Effect")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"PCA plot saved to {out_path}")


# -- Main pipeline ------------------------------------------------------------

def run_pipeline(
    raw_dir: Path,
    out_dir: Path,
    raw_sites: list[str],
    canonical_map: dict[str, str],
    out_name: str = "abide_fc.h5",
    test_mode: bool = False,
    pheno_csv: str | None = None,
) -> Path:
    """Run the full preprocessing pipeline.

    Parameters
    ----------
    raw_dir : Path to directory with .1D files and phenotypic CSV.
    out_dir : Output directory for HDF5, CSV, and PCA plot.
    raw_sites : List of raw ABIDE site IDs to include.
    canonical_map : Mapping from raw site ID -> canonical site name.
    out_name : Output HDF5 filename.
    test_mode : If True, process max 3 subjects per site.
    pheno_csv : Phenotypic CSV filename override (default: ABIDE-I name).

    Returns
    -------
    Path to the output HDF5 file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive canonical site names (ordered, deduplicated)
    seen = set()
    site_names = []
    for raw_s in raw_sites:
        canonical = canonical_map[raw_s]
        if canonical not in seen:
            site_names.append(canonical)
            seen.add(canonical)
    site_label_map = {name: idx for idx, name in enumerate(site_names)}
    target_site_ids = set(raw_sites)

    # -- Step 1: Load & QC-filter phenotypic data ---
    if pheno_csv is not None:
        pheno_path = raw_dir / pheno_csv
    else:
        pheno_path = raw_dir / "Phenotypic_V1_0b_preprocessed1.csv"
        if not pheno_path.exists():
            pheno_path = raw_dir / "ABIDE_pcp_abide_phenotypic_V1_0d.csv"
    if not pheno_path.exists():
        sys.exit(f"ERROR: Phenotypic CSV not found in {raw_dir}")

    print("Loading phenotypic data...")
    pheno = load_phenotypic(pheno_path)
    pheno["SITE_ID_NORM"] = pheno["SITE_ID"].astype(str).str.strip().str.upper()
    pheno = pheno[pheno["SITE_ID_NORM"].isin(target_site_ids)].copy()
    print(f"  Subjects at target sites (pre-QC): {len(pheno)}")

    pheno_qc = apply_qc(pheno)
    print(f"  Subjects passing QC: {len(pheno_qc)}")

    # -- Step 2: Build FC matrices ---
    fc_vectors: list[np.ndarray] = []
    diagnoses: list[int] = []
    sites: list[int] = []
    subject_ids: list[str] = []
    summary_rows: list[dict] = []

    if test_mode:
        print("  [TEST MODE] Processing only first 3 subjects per site")
    site_counts: dict[str, int] = {s: 0 for s in site_names}

    for _, row in tqdm(pheno_qc.iterrows(), total=len(pheno_qc),
                       desc="Building FC matrices"):
        file_id = str(row["FILE_ID"]).strip()
        site_id = str(row["SITE_ID_NORM"]).strip().upper()
        canonical_site = canonical_map.get(site_id)

        if canonical_site is None or file_id == "no_filename":
            continue

        if test_mode and site_counts.get(canonical_site, 0) >= 3:
            continue

        ts_path = find_timeseries_file(raw_dir, file_id)
        if ts_path is None:
            summary_rows.append({
                "subject_id": file_id,
                "site": canonical_site,
                "diagnosis": int(row.get("DX_GROUP", -1)),
                "mean_fd": row.get("func_mean_fd", np.nan),
                "n_timepoints": 0,
                "passed_qc": False,
            })
            continue

        ts = load_timeseries(ts_path)
        if ts is None:
            continue

        fc = compute_fc(ts)
        if fc.shape[0] != N_FEATURES:
            continue

        dx = 1 if int(row["DX_GROUP"]) == 1 else 0

        fc_vectors.append(fc)
        diagnoses.append(dx)
        sites.append(site_label_map[canonical_site])
        subject_ids.append(file_id)
        site_counts[canonical_site] = site_counts.get(canonical_site, 0) + 1

        summary_rows.append({
            "subject_id": file_id,
            "site": canonical_site,
            "diagnosis": dx,
            "mean_fd": float(row.get("func_mean_fd", np.nan)),
            "n_timepoints": int(ts.shape[0]),
            "passed_qc": True,
        })

    if not fc_vectors:
        sys.exit("ERROR: No valid FC matrices produced. "
                 "Check that timeseries files exist in raw_dir.")

    X = np.array(fc_vectors, dtype=np.float32)
    y_dx = np.array(diagnoses, dtype=np.int32)
    y_site = np.array(sites, dtype=np.int32)

    # -- Step 3: Sanity checks ---
    print("\n=== Sanity Checks ===")
    print(f"  Total subjects: {X.shape[0]}")
    for sid, name in enumerate(site_names):
        n = int((y_site == sid).sum())
        n_asd = int(((y_site == sid) & (y_dx == 1)).sum())
        n_tc = int(((y_site == sid) & (y_dx == 0)).sum())
        print(f"  {name}: n={n}  (ASD={n_asd}, TC={n_tc})")

    assert not np.isnan(X).any(), "Feature matrix contains NaNs!"
    assert not np.isinf(X).any(), "Feature matrix contains Infs!"
    print("  No NaN/Inf in feature matrix -- OK")

    if not test_mode:
        for sid, name in enumerate(site_names):
            n_site = int((y_site == sid).sum())
            if n_site < 5:
                print(f"  WARNING: {name} count ({n_site}) is very low (<5)")

    # -- Step 4: Save outputs ---
    h5_path = out_dir / out_name
    print(f"\nSaving HDF5 -> {h5_path}")

    # Build subject-indexed lookup for ADOS scores
    pheno_by_file_id = pheno_qc.set_index("FILE_ID") if "FILE_ID" in pheno_qc.columns else None
    ados_columns = ["ADOS_TOTAL", "ADOS_COMM", "ADOS_SOCIAL", "ADOS_GOTHAM_TOTAL"]

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X", data=X, compression="gzip", compression_opts=4)
        f.create_dataset("y_diagnosis", data=y_dx)
        f.create_dataset("y_site", data=y_site)
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("subject_ids", data=subject_ids, dtype=dt)

        # Store ADOS scores when available in phenotypic data
        if pheno_by_file_id is not None:
            for ados_col in ados_columns:
                if ados_col in pheno_by_file_id.columns:
                    ados_arr = np.full(len(subject_ids), np.nan, dtype=np.float32)
                    for i, sid in enumerate(subject_ids):
                        if sid in pheno_by_file_id.index:
                            val = pheno_by_file_id.loc[sid, ados_col]
                            if isinstance(val, pd.Series):
                                val = val.iloc[0]
                            try:
                                fval = float(val)
                                if fval != -9999.0:
                                    ados_arr[i] = fval
                            except (ValueError, TypeError):
                                pass
                    f.create_dataset(ados_col.lower(), data=ados_arr)
                    n_valid = int(np.sum(~np.isnan(ados_arr)))
                    print(f"  {ados_col}: {n_valid}/{len(subject_ids)} valid scores stored")

        f.attrs["site_names"] = site_names
        f.attrs["n_subjects"] = X.shape[0]
        f.attrs["n_sites"] = len(site_names)
        for sid, name in enumerate(site_names):
            f.attrs[f"n_{name}"] = int((y_site == sid).sum())
        f.attrs["qc_rater_1"] = QC_THRESHOLDS["qc_rater_1"]
        f.attrs["func_perc_fd_max"] = QC_THRESHOLDS["func_perc_fd_max"]
        f.attrs["func_mean_fd_max"] = QC_THRESHOLDS["func_mean_fd_max"]
        f.attrs["derivative"] = DERIVATIVE
        f.attrs["pipeline"] = PIPELINE
        f.attrs["strategy"] = STRATEGY

    csv_path = out_dir / out_name.replace(".h5", "_summary.csv")
    print(f"Saving summary CSV -> {csv_path}")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

    # -- Step 5: PCA plot ---
    pca_path = out_dir / out_name.replace(".h5", "_pca.png")
    if X.shape[0] >= 3:
        make_pca_plot(X, y_site, site_names, pca_path)
    else:
        print("Skipping PCA plot (not enough subjects).")

    print(f"\nPipeline complete. {X.shape[0]} subjects, {len(site_names)} sites.")
    return h5_path


# -- CLI ----------------------------------------------------------------------

def _resolve_preset_or_custom(args) -> tuple[list[str], dict[str, str], str, str | None]:
    """Return (raw_sites, canonical_map, out_name, pheno_csv) from CLI args."""
    if args.sites:
        raw_sites = [s.strip() for s in args.sites.split(",")]
        canonical_map = {s: s for s in raw_sites}
        out_name = args.out_name or "abide_custom_fc.h5"
        return raw_sites, canonical_map, out_name, None

    preset_name = args.dataset
    if preset_name not in DATASET_PRESETS:
        available = ", ".join(sorted(DATASET_PRESETS.keys()))
        sys.exit(f"ERROR: Unknown dataset '{preset_name}'. Available: {available}")

    preset = DATASET_PRESETS[preset_name]
    out_name = args.out_name or preset["out_name"]
    pheno_csv = preset.get("pheno_csv", None)
    return preset["raw_sites"], preset["canonical_map"], out_name, pheno_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ABIDE-I preprocessing: QC -> FC matrices -> HDF5"
    )
    parser.add_argument(
        "--dataset", type=str, default="abide1_3site",
        help="Dataset preset: abide1_3site, abide1_8site, abide1_full",
    )
    parser.add_argument(
        "--sites", type=str, default=None,
        help="Comma-separated raw site IDs (overrides --dataset)",
    )
    parser.add_argument(
        "--raw-dir", type=str, default="data/raw",
        help="Directory with downloaded .1D files and phenotypic CSV",
    )
    parser.add_argument(
        "--out-dir", type=str, default="data/processed",
        help="Output directory for HDF5, CSV, and PCA plot",
    )
    parser.add_argument(
        "--out-name", type=str, default=None,
        help="Output HDF5 filename (default: from preset)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: process max 3 subjects per site",
    )
    args = parser.parse_args()

    raw_sites, canonical_map, out_name, pheno_csv = _resolve_preset_or_custom(args)
    print(f"Dataset: {args.dataset} | Sites: {raw_sites}")

    run_pipeline(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        raw_sites=raw_sites,
        canonical_map=canonical_map,
        out_name=out_name,
        test_mode=args.test,
        pheno_csv=pheno_csv,
    )
