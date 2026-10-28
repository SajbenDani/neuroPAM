#!/usr/bin/env python3
"""
download_abide.py -- Automated ABIDE-I / ABIDE-II download.

Downloads preprocessed CC200 ROI timeseries (C-PAC pipeline, filt_global
strategy) from the Preprocessed Connectomes Project S3 bucket, plus the
phenotypic CSV.

Usage:
    python scripts/preprocessing/download_abide1.py --dataset abide1 --out-dir data/raw
    python scripts/preprocessing/download_abide1.py --dataset abide1 --out-dir data/raw --sites NYU,USM,UCLA_1,UCLA_2
    python scripts/preprocessing/download_abide1.py --dataset abide1 --out-dir data/raw --max-subjects 10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib import request as urlrequest

# ---------------------------------------------------------------------------
# S3 configuration for ABIDE-I
# ---------------------------------------------------------------------------

ABIDE1_S3_PREFIX = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"
)
ABIDE1_PHENO_URL = f"{ABIDE1_S3_PREFIX}/Phenotypic_V1_0b_preprocessed1.csv"

# ---------------------------------------------------------------------------
# S3 configuration for ABIDE-II
# ---------------------------------------------------------------------------
ABIDE2_S3_PREFIX = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE2"
)
ABIDE2_PHENO_URL = f"{ABIDE2_S3_PREFIX}/Phenotypic.csv"

DERIVATIVE = "rois_cc200"
PIPELINE = "cpac"
STRATEGY = "filt_global"

# All ABIDE-I sites
ABIDE1_ALL_SITES = [
    "CALTECH", "CMU", "KKI", "LEUVEN_1", "LEUVEN_2",
    "MAX_MUN", "NYU", "OHSU", "OLIN", "PITT", "SBL",
    "SDSU", "STANFORD", "TRINITY", "UCLA_1", "UCLA_2",
    "UM_1", "UM_2", "USM", "YALE",
]

# ABIDE-II sites with enough subjects for harmonization (n >= 20)
ABIDE2_SITES = [
    "BNI_1", "ETH_1", "GU_1", "IP_1", "IU_1",
    "KKI_1", "NYU_1", "NYU_2", "OHSU_1", "ONRC_2",
    "OILH_1", "SBL_1", "SDSU_1", "SEO_1", "STANFORD_1",
    "TCD_1", "UCLA_1", "UCLA_2", "UCLA_3", "UCD_1",
    "UM_1", "UM_2", "UM_3", "USM_1",
]


def download_file(url: str, dest: Path) -> bool:
    """Download a single file. Returns True on success."""
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlrequest.urlretrieve(url, str(dest))
        return True
    except Exception as exc:
        print(f"  WARN: Failed to download {url}: {exc}")
        return False


def download_abide2_timeseries(
    out_dir: Path,
    sites: list[str] | None = None,
    max_subjects: int | None = None,
) -> int:
    """Download ABIDE-II CC200 ROI timeseries from S3.

    ABIDE-II uses the same cpac/filt_global/rois_cc200 pipeline but lives
    under the ABIDE2 S3 prefix.  Phenotypic CSV uses SITE_ID and FILE_ID
    columns that match ABIDE-I conventions.
    """
    print("Fetching ABIDE-II phenotypic from S3...")
    try:
        pheno_resp = urlrequest.urlopen(ABIDE2_PHENO_URL)
        pheno_lines = pheno_resp.readlines()
    except Exception as exc:
        print(f"  ERROR: Could not fetch ABIDE-II phenotypic: {exc}")
        return 0

    header = pheno_lines[0].decode().strip().split(",")
    # Handle quoted headers
    header = [h.strip('"') for h in header]

    # Try both FILE_ID and SUB_ID column names
    file_col = "FILE_ID" if "FILE_ID" in header else (
        "SUB_ID" if "SUB_ID" in header else None
    )
    site_col = "SITE_ID" if "SITE_ID" in header else None
    if file_col is None or site_col is None:
        print(f"  ERROR: Expected FILE_ID/SITE_ID columns, found: {header[:10]}")
        return 0

    file_idx = header.index(file_col)
    site_idx = header.index(site_col)

    to_download = []
    for row_bytes in pheno_lines[1:]:
        row = row_bytes.decode().strip().split(",")
        row = [c.strip('"') for c in row]
        if len(row) <= max(file_idx, site_idx):
            continue
        file_id = row[file_idx].strip()
        site_id = row[site_idx].strip()

        if file_id in ("no_filename", "", "nan"):
            continue
        if sites is not None and site_id not in sites:
            continue

        filename = f"{file_id}_{DERIVATIVE}.1D"
        s3_url = "/".join([
            ABIDE2_S3_PREFIX, "Outputs", PIPELINE, STRATEGY,
            DERIVATIVE, filename,
        ])
        local_path = out_dir / "abide2" / "Outputs" / PIPELINE / STRATEGY / DERIVATIVE / filename
        to_download.append((s3_url, local_path))

    if max_subjects is not None:
        to_download = to_download[:max_subjects]

    total = len(to_download)
    print(f"ABIDE-II files to download: {total}")
    n_ok = 0
    for i, (url, dest) in enumerate(to_download):
        if dest.exists():
            n_ok += 1
            continue
        ok = download_file(url, dest)
        if ok:
            n_ok += 1
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  Progress: {i + 1}/{total} ({100 * (i + 1) / total:.0f}%)")

    print(f"ABIDE-II download complete: {n_ok}/{total} files available.")
    return n_ok


def download_phenotypic(out_dir: Path, dataset: str) -> Path:
    """Download the phenotypic CSV to out_dir. Returns path."""
    if dataset == "abide1":
        pheno_name = "Phenotypic_V1_0b_preprocessed1.csv"
        pheno_url = ABIDE1_PHENO_URL
    elif dataset == "abide2":
        pheno_name = "abide2_Phenotypic.csv"
        pheno_url = ABIDE2_PHENO_URL
    else:
        sys.exit(f"ERROR: Phenotypic download for '{dataset}' not yet implemented.")

    dest = out_dir / pheno_name
    if dest.exists():
        print(f"Phenotypic CSV already exists: {dest}")
        return dest

    print(f"Downloading phenotypic CSV -> {dest}")
    download_file(pheno_url, dest)
    if not dest.exists():
        sys.exit(f"ERROR: Failed to download phenotypic CSV from {pheno_url}")
    return dest


def download_abide1_timeseries(
    out_dir: Path,
    sites: list[str] | None = None,
    max_subjects: int | None = None,
) -> int:
    """Download ABIDE-I CC200 ROI timeseries from S3.

    Parameters
    ----------
    out_dir : Path
        Root output directory (files go into out_dir/Outputs/cpac/...)
    sites : list[str] or None
        If given, only download subjects from these sites. Use ABIDE site IDs
        like 'NYU', 'UCLA_1', etc.
    max_subjects : int or None
        Max total subjects to download (for testing).

    Returns
    -------
    int : number of files downloaded or already present
    """
    # Load phenotypic from S3 to get FILE_IDs
    print("Fetching phenotypic file index from S3...")
    pheno_resp = urlrequest.urlopen(ABIDE1_PHENO_URL)
    pheno_lines = pheno_resp.readlines()

    header = pheno_lines[0].decode().strip().split(",")
    file_idx = header.index("FILE_ID")
    site_idx = header.index("SITE_ID")

    # Build list of files to download
    to_download = []
    for row_bytes in pheno_lines[1:]:
        row = row_bytes.decode().strip().split(",")
        if len(row) <= max(file_idx, site_idx):
            continue
        file_id = row[file_idx].strip()
        site_id = row[site_idx].strip()

        if file_id == "no_filename":
            continue
        if sites is not None and site_id not in sites:
            continue

        filename = f"{file_id}_{DERIVATIVE}.1D"
        s3_url = "/".join([
            ABIDE1_S3_PREFIX, "Outputs", PIPELINE, STRATEGY,
            DERIVATIVE, filename,
        ])
        local_path = out_dir / "Outputs" / PIPELINE / STRATEGY / DERIVATIVE / filename
        to_download.append((s3_url, local_path))

    if max_subjects is not None:
        to_download = to_download[:max_subjects]

    total = len(to_download)
    print(f"Files to download: {total}")

    n_ok = 0
    for i, (url, dest) in enumerate(to_download):
        if dest.exists():
            n_ok += 1
            continue
        ok = download_file(url, dest)
        if ok:
            n_ok += 1
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  Progress: {i + 1}/{total} ({100 * (i + 1) / total:.0f}%)")

    print(f"Download complete: {n_ok}/{total} files available.")
    return n_ok


def main():
    parser = argparse.ArgumentParser(
        description="Download ABIDE preprocessed timeseries data"
    )
    parser.add_argument(
        "--dataset", type=str, default="abide1",
        choices=["abide1", "abide2"],
        help="Which dataset to download (default: abide1)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="data/raw",
        help="Output directory for raw data",
    )
    parser.add_argument(
        "--sites", type=str, default=None,
        help="Comma-separated list of site IDs to download (default: all)",
    )
    parser.add_argument(
        "--max-subjects", type=int, default=None,
        help="Max subjects to download (for testing)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sites = None
    if args.sites:
        sites = [s.strip() for s in args.sites.split(",")]

    if args.dataset == "abide1":
        download_phenotypic(out_dir, "abide1")
        download_abide1_timeseries(out_dir, sites=sites, max_subjects=args.max_subjects)
    elif args.dataset == "abide2":
        download_phenotypic(out_dir, "abide2")
        download_abide2_timeseries(out_dir, sites=sites, max_subjects=args.max_subjects)
    else:
        sys.exit(f"ERROR: Dataset '{args.dataset}' not yet supported.")


if __name__ == "__main__":
    main()
