#!/usr/bin/env python3
"""
preprocess_abide2.py -- ABIDE-II Preprocessing Pipeline (with checkpointing)

ABIDE-II does NOT have cpac/filt_global/rois_cc200 .1D files on S3 (unlike ABIDE-I).
It only has fMRIPrep and DCAN preprocessed outputs.

This script uses the fMRIPrep preprocessed BOLD images from the FCP-INDI S3 bucket
and applies CC200 atlas parcellation via nilearn to produce FC matrices in the same
format as abide_8site_fc.h5.

S3 structure (ABIDE-II fMRIPrep -- verified 2026-04-07):
  s3://fcp-indi/data/Projects/ABIDE2/Outputs/fmriprep/fmriprep/sub-{SUB_ID}/ses-{N}/func/
    sub-{SUB_ID}_ses-{N}_task-rest[_acq-{label}]_run-{N}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
    sub-{SUB_ID}_ses-{N}_task-rest[_acq-{label}]_run-{N}_desc-confounds_regressors.tsv
  Note: NO zero-padding on subject IDs; some sites (e.g. KKI) include acq-* entity.
  URL discovery: _discover_bold_urls() queries S3 listing to find the actual filename
  rather than constructing a fixed URL, handling site-specific acquisition entities.

CHECKPOINTING:
  Each subject's FC vector is saved as a .npz file in a checkpoint directory
  immediately after extraction.  On restart, already-checkpointed subjects are
  skipped — download resumes from where it left off.  At the end, all checkpoint
  files are merged into the final abide2_fc.h5.

  Checkpoint dir: <out-dir>/abide2_checkpoints/sub-{ID}.npz  (default)
  Override with --checkpoint-dir.

ADDING MORE SITES:
  The phenotypic CSV controls which subjects are processed.  The default CSV
  (ABIDEII_Composite_Phenotypic.csv) currently covers 3 sites (IP_1, UCD_1,
  UCLA_1).  To add more sites, download the full ABIDE-II composite phenotypic
  from http://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html and replace
  the default CSV.  The script will automatically process all subjects in the CSV
  (diagnosis labels are required and are only in the phenotypic file).

NOTE: Each subject's BOLD file is ~100-300 MB; it is deleted immediately after
extraction.  Only the 18,915-dimensional FC vector (~74 KB uncompressed) persists
per subject.

Usage:
    # Full run with checkpointing (safe to interrupt and resume):
    python scripts/preprocessing/preprocess_abide2.py \\
        --pheno data/ABIDE-II/ABIDEII_Composite_Phenotypic.csv \\
        --out-dir data/processed

    # Resume after interruption (automatic -- just re-run the same command):
    python scripts/preprocessing/preprocess_abide2.py \\
        --pheno data/ABIDE-II/ABIDEII_Composite_Phenotypic.csv \\
        --out-dir data/processed

    # Merge existing checkpoints to H5 without downloading anything new:
    python scripts/preprocessing/preprocess_abide2.py \\
        --pheno data/ABIDE-II/ABIDEII_Composite_Phenotypic.csv \\
        --out-dir data/processed --merge-only

    # Local files already downloaded (no S3):
    python scripts/preprocessing/preprocess_abide2.py \\
        --pheno data/ABIDE-II/ABIDEII_Composite_Phenotypic.csv \\
        --local-dir /path/to/fmriprep/root \\
        --out-dir data/processed

    # Test mode (no downloads; verifies atlas + phenotypic + checkpoint status):
    python scripts/preprocessing/preprocess_abide2.py \\
        --pheno data/ABIDE-II/ABIDEII_Composite_Phenotypic.csv --test

    # Concurrent mode (2-3x speedup on 6-core / 16GB RAM systems):
    python scripts/preprocessing/preprocess_abide2.py \\
        --pheno data/ABIDE-II/ABIDEII_Composite_Phenotypic.csv \\
        --out-dir data/processed \\
        --workers 2 --download-threads 3
"""

from __future__ import annotations

import argparse
import dataclasses
import queue
import re
import signal
import sys
import tempfile
import threading
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional heavy imports -- checked at runtime
try:
    import nilearn.datasets
    import nilearn.input_data
    NILEARN_OK = True
except ImportError:
    NILEARN_OK = False

try:
    import nibabel as nib
    NIBABEL_OK = True
except ImportError:
    NIBABEL_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ROIS = 200          # nominal CC200 target; actual ROI count determined at runtime
N_FEATURES = N_ROIS * (N_ROIS - 1) // 2  # 19 900 nominal

# FCP-INDI S3 base (HTTP, no AWS credentials needed for public bucket)
# NOTE: actual S3 structure has an extra /fmriprep/ level:
#   .../fmriprep/fmriprep/sub-{ID}/ses-1/func/
S3_BUCKET = "https://s3.amazonaws.com/fcp-indi"
S3_KEY_PREFIX = "data/Projects/ABIDE2/Outputs/fmriprep/fmriprep"
S3_BASE = f"{S3_BUCKET}/{S3_KEY_PREFIX}"

# Motion QC threshold (mean framewise displacement)
MAX_MEAN_FD = 0.5  # mm -- slightly relaxed vs ABIDE-I (0.25) since fMRIPrep uses different FD def

# ABIDE-II site presets: raw SITE_ID prefix -> canonical name
# SITE_ID in phenotypic looks like "ABIDEII-NYU_1"
SITE_CANONICAL = {
    "BNI_1": "BNI",
    "EMC_1": "EMC",
    "ETH_1": "ETH",
    "GU_1": "GU",
    "IP_1": "IP",
    "IU_1": "IU",
    "KKI_1": "KKI",
    "NYU_1": "NYU",
    "NYU_2": "NYU",
    "OHSU_1": "OHSU",
    "ONRC_2": "ONRC",
    "OILH_1": "OILH",
    "SBL_1": "SBL",
    "SDSU_1": "SDSU",
    "SEO_1": "SEO",
    "STANFORD_1": "STANFORD",
    "TCD_1": "TCD",
    "UCLA_1": "UCLA",
    "UCLA_2": "UCLA",
    "UCLA_3": "UCLA",
    "UCD_1": "UCD",
    "U_MIA": "UM",
    "U_MIA_1": "UM",
    "U_MIA_2": "UM",
    "U_MIA_3": "UM",
    "UM_1": "UM",
    "UM_2": "UM",
    "UM_3": "UM",
    "USM_1": "USM",
}

# Confound columns to regress from bold signal (standard 24-parameter model)
# ABIDE-II fMRIPrep only provides 8 params; the script uses whichever are available.
CONFOUND_COLS = [
    "trans_x", "trans_y", "trans_z",
    "rot_x", "rot_y", "rot_z",
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
    "trans_x_power2", "trans_y_power2", "trans_z_power2",
    "rot_x_power2", "rot_y_power2", "rot_z_power2",
    "trans_x_derivative1_power2", "trans_y_derivative1_power2",
    "trans_z_derivative1_power2",
    "rot_x_derivative1_power2", "rot_y_derivative1_power2",
    "rot_z_derivative1_power2",
    "white_matter", "csf",
]

# Regex to validate checkpoint filenames
_CKPT_RE = re.compile(r"^sub-(\d+)\.npz$")


@dataclasses.dataclass
class SubjectTask:
    """Work item passed through the concurrent pipeline."""
    sub_id: int
    site_raw: str
    canonical: str
    dx: int
    ados_val: float
    bold_path: Path | None = None
    conf_path: Path | None = None
    was_downloaded: bool = False


# ---------------------------------------------------------------------------
# Atlas helpers
# ---------------------------------------------------------------------------

def _extract_3d_atlas(atlas) -> tuple:
    """
    Extract a single 3D parcellation volume from the CC200 atlas NIfTI.

    ``fetch_atlas_craddock_2012()`` returns a 4D image (47x56x46x43) where
    each volume along the 4th axis corresponds to a different parcellation
    granularity.  Volume 19 (0-indexed) yields ~195 ROIs -- the closest to
    the CC200 k=200 target.

    Returns
    -------
    (atlas_3d_img, n_rois)  where atlas_3d_img is a nibabel Nifti1Image.
    """
    if not NIBABEL_OK:
        raise ImportError("nibabel is required for 4D atlas handling. "
                          "Install with: pip install nibabel")

    img = nib.load(str(atlas.maps))
    data = img.get_fdata()

    if data.ndim == 3:
        n_rois = int(np.unique(data[data > 0]).size)
        print(f"  CC200 atlas: 3D, {n_rois} ROIs")
        return img, n_rois

    # 4D: find the volume whose label count is closest to k=200
    target_k = 200
    best_vol, best_diff = 0, float("inf")
    for i in range(data.shape[3]):
        n_labels = int(np.unique(data[..., i][data[..., i] > 0]).size)
        diff = abs(n_labels - target_k)
        if diff < best_diff:
            best_diff = diff
            best_vol = i

    vol_data = np.round(data[..., best_vol]).astype(np.int32)
    n_rois = int(np.unique(vol_data[vol_data > 0]).size)

    atlas_3d = nib.Nifti1Image(vol_data, img.affine, img.header)
    print(f"  CC200 atlas: extracted volume {best_vol} with {n_rois} ROIs "
          f"(target k=200, diff={best_diff})")
    return atlas_3d, n_rois


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_path(checkpoint_dir: Path, sub_id: int) -> Path:
    """Return path for a subject's FC checkpoint file."""
    return checkpoint_dir / f"sub-{sub_id}.npz"


def _save_checkpoint(
    checkpoint_dir: Path,
    sub_id: int,
    fc: np.ndarray,
    dx: int,
    site_canonical: str,
    ados_val: float,
    fc_half1: "np.ndarray | None" = None,
    fc_half2: "np.ndarray | None" = None,
) -> None:
    """Save a subject's FC vector + metadata as a compressed .npz checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    extra = {}
    if fc_half1 is not None:
        extra["fc_half1"] = fc_half1
        extra["fc_half2"] = fc_half2
    np.savez_compressed(
        _checkpoint_path(checkpoint_dir, sub_id),
        fc=fc.astype(np.float32),
        dx=np.int32(dx),
        site=np.array(site_canonical, dtype=object),
        sub_id=np.int32(sub_id),
        # Store NaN as sentinel -9999 since npz doesn't natively handle NaN metadata
        ados=np.float32(ados_val if not np.isnan(ados_val) else -9999.0),
        **extra,
    )


def _load_existing_checkpoints(checkpoint_dir: Path) -> set[int]:
    """Return the set of sub_ids that already have a valid checkpoint file."""
    if not checkpoint_dir.exists():
        return set()
    done = set()
    for p in checkpoint_dir.glob("sub-*.npz"):
        m = _CKPT_RE.match(p.name)
        if m:
            done.add(int(m.group(1)))
    return done


def _merge_checkpoints_to_h5(
    checkpoint_dir: Path,
    h5_path: Path,
    n_rois_actual: int,
    n_features_actual: int,
) -> Path:
    """
    Load all per-subject .npz checkpoints and merge into a single HDF5.

    Site ordering is determined by alphabetical sort of all canonical site names
    found across checkpoints — reproducible regardless of processing order.
    """
    npz_paths = sorted(checkpoint_dir.glob("sub-*.npz"))
    if not npz_paths:
        sys.exit(f"ERROR: No checkpoint files found in {checkpoint_dir}")

    fc_vectors: list[np.ndarray] = []
    fc_half1_vectors: list[np.ndarray] = []
    fc_half2_vectors: list[np.ndarray] = []
    has_split_half = False
    diagnoses: list[int] = []
    sites_str: list[str] = []
    subject_ids: list[str] = []
    ados_scores: list[float] = []
    skipped = 0

    for p in npz_paths:
        try:
            d = np.load(p, allow_pickle=True)
            fc = d["fc"]
            if fc.shape[0] != n_features_actual:
                print(f"  [merge] Skipping {p.name}: dim {fc.shape[0]} != {n_features_actual}")
                skipped += 1
                continue
            fc_vectors.append(fc)
            diagnoses.append(int(d["dx"]))
            sites_str.append(str(d["site"]))
            subject_ids.append(f"sub-{int(d['sub_id'])}")
            ados_raw = float(d["ados"])
            ados_scores.append(np.nan if ados_raw == -9999.0 else ados_raw)
            if "fc_half1" in d:
                has_split_half = True
                fc_half1_vectors.append(d["fc_half1"].astype(np.float32))
                fc_half2_vectors.append(d["fc_half2"].astype(np.float32))
            elif has_split_half:
                fc_half1_vectors.append(np.full(n_features_actual, np.nan, dtype=np.float32))
                fc_half2_vectors.append(np.full(n_features_actual, np.nan, dtype=np.float32))
        except Exception as exc:
            print(f"  [merge] Could not load {p.name}: {exc}")
            skipped += 1
            continue

    if not fc_vectors:
        sys.exit("ERROR: No valid FC vectors found in checkpoints after filtering.")

    if skipped:
        print(f"  [merge] Skipped {skipped} corrupt/mismatched checkpoints.")

    # Alphabetical site ordering for consistent y_site encoding across runs
    site_names_ordered = sorted(set(sites_str))
    site_label_map = {name: idx for idx, name in enumerate(site_names_ordered)}
    site_indices = [site_label_map[s] for s in sites_str]

    X = np.array(fc_vectors, dtype=np.float32)
    y_dx = np.array(diagnoses, dtype=np.int32)
    y_site = np.array(site_indices, dtype=np.int32)

    print("\n=== Merge Summary ===")
    print(f"  Total subjects: {X.shape[0]}  |  Sites: {len(site_names_ordered)}")
    for sid, name in enumerate(site_names_ordered):
        n = int((y_site == sid).sum())
        if n == 0:
            continue
        n_asd = int(((y_site == sid) & (y_dx == 1)).sum())
        print(f"  {name}: n={n} (ASD={n_asd}, TC={n - n_asd})")

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("X", data=X, compression="gzip", compression_opts=4)
        f.create_dataset("y_diagnosis", data=y_dx)
        f.create_dataset("y_site", data=y_site)
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("subject_ids", data=subject_ids, dtype=dt)
        f.create_dataset("ados_g_total", data=np.array(ados_scores, dtype=np.float32))
        if has_split_half and fc_half1_vectors:
            f.create_dataset("fc_half1", data=np.array(fc_half1_vectors, dtype=np.float32),
                             compression="gzip", compression_opts=4)
            f.create_dataset("fc_half2", data=np.array(fc_half2_vectors, dtype=np.float32),
                             compression="gzip", compression_opts=4)

        f.attrs["site_names"] = site_names_ordered
        f.attrs["n_subjects"] = X.shape[0]
        f.attrs["n_sites"] = len(site_names_ordered)
        for sid, name in enumerate(site_names_ordered):
            f.attrs[f"n_{name}"] = int((y_site == sid).sum())
        f.attrs["pipeline"] = "fmriprep"
        f.attrs["atlas"] = "cc200_craddock2012"
        f.attrs["n_rois"] = n_rois_actual
        f.attrs["n_features"] = n_features_actual
        f.attrs["max_mean_fd"] = MAX_MEAN_FD
        f.attrs["confounds"] = "8-param motion + WM + CSF (older fMRIPrep format)"

    print(f"Saved -> {h5_path}")
    return h5_path


# ---------------------------------------------------------------------------
# Phenotypic helpers
# ---------------------------------------------------------------------------

def load_pheno(pheno_path: Path) -> pd.DataFrame:
    """Load ABIDE-II composite phenotypic CSV and normalise columns."""
    df = pd.read_csv(pheno_path)
    df.columns = [c.strip() for c in df.columns]

    # SITE_ID looks like "ABIDEII-NYU_1" -- extract raw site suffix
    if "SITE_ID" in df.columns:
        df["SITE_RAW"] = (
            df["SITE_ID"].astype(str)
            .str.strip()
            .str.replace("ABIDEII-", "", regex=False)
            .str.upper()
        )
    else:
        sys.exit("ERROR: SITE_ID column not found in phenotypic CSV")

    # DX_GROUP: 1=ASD, 2=TC (ABIDE-II convention -- opposite to ABIDE-I where 2=TC)
    # Normalise: 1=ASD, 0=TC
    if "DX_GROUP" in df.columns:
        df["DX_NORM"] = df["DX_GROUP"].apply(lambda x: 1 if int(x) == 1 else 0)

    return df


def parse_site_filter(sites_arg: str | None) -> set[str] | None:
    """Parse --sites CLI argument into a set of raw site IDs."""
    if sites_arg is None:
        return None
    return {s.strip().upper() for s in sites_arg.split(",")}


# ---------------------------------------------------------------------------
# S3 download helpers
# ---------------------------------------------------------------------------

def _bold_url(sub_id: int, run: str = "run-1") -> str:
    # S3 uses no zero-padding (sub-29580, not sub-0029580) and has ses-1 subdir
    sub_str = f"sub-{sub_id}"
    return (
        f"{S3_BASE}/{sub_str}/ses-1/func/"
        f"{sub_str}_ses-1_task-rest_{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )


def _confounds_url(sub_id: int, run: str = "run-1") -> str:
    # S3 uses older fMRIPrep convention: desc-confounds_regressors.tsv (not timeseries)
    sub_str = f"sub-{sub_id}"
    return (
        f"{S3_BASE}/{sub_str}/ses-1/func/"
        f"{sub_str}_ses-1_task-rest_{run}_desc-confounds_regressors.tsv"
    )


def _discover_bold_urls(sub_id: int) -> tuple[str | None, str | None]:
    """
    Query S3 listing to find the actual preproc BOLD and confounds URLs for a subject.

    Tries ses-1 then ses-2. Some sites (e.g. KKI) encode acquisition parameters in
    the filename (acq-rc8chan), making hardcoded URL templates fail. This function
    discovers the actual filename from the S3 bucket listing instead.

    Returns (bold_url, confounds_url) or (None, None) if no func data found.
    """
    _NS = "http://s3.amazonaws.com/doc/2006-03-01/"
    sub_str = f"sub-{sub_id}"
    for ses in ("ses-1", "ses-2"):
        prefix = f"{S3_KEY_PREFIX}/{sub_str}/{ses}/func/"
        list_url = (
            f"{S3_BUCKET}/?list-type=2"
            f"&prefix={urllib.parse.quote(prefix, safe='/')}"
            f"&max-keys=40"
        )
        try:
            with urllib.request.urlopen(list_url, timeout=15) as resp:
                root = ET.fromstring(resp.read())
        except Exception:
            continue
        keys = [el.text for el in root.iter(f"{{{_NS}}}Key") if el.text]
        bold_key = next(
            (k for k in keys if k.endswith("_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")),
            None,
        )
        if bold_key is None:
            continue
        conf_key = next(
            (k for k in keys if k.endswith("_desc-confounds_regressors.tsv")),
            None,
        )
        bold_url = f"{S3_BUCKET}/{bold_key}"
        conf_url = f"{S3_BUCKET}/{conf_key}" if conf_key else None
        return bold_url, conf_url
    return None, None


def _try_download(url: str, dest: Path) -> bool:
    """Download url to dest. Return True on success."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as exc:
        print(f"    Download failed ({url}): {exc}")
        return False


# ---------------------------------------------------------------------------
# fMRIPrep bold -> timeseries extraction
# ---------------------------------------------------------------------------

def _load_confounds(confounds_path: Path) -> np.ndarray | None:
    """Load motion + WM/CSF confounds from fMRIPrep tsv."""
    try:
        df = pd.read_csv(confounds_path, sep="\t")
    except Exception:
        return None

    available = [c for c in CONFOUND_COLS if c in df.columns]
    if not available:
        return None

    if len(available) < len(CONFOUND_COLS):
        print(f"    [confounds] Using {len(available)}/{len(CONFOUND_COLS)}-param model "
              f"(older fMRIPrep format — {len(available)} cols)")

    conf = df[available].fillna(0.0).to_numpy()
    return conf


def extract_timeseries_from_bold(
    bold_path: Path,
    confounds_path: Path | None,
    atlas_3d_img,
    n_rois: int,
    masker_kwargs: dict | None = None,
) -> np.ndarray | None:
    """
    Use nilearn NiftiLabelsMasker to extract CC200 parcellated timeseries.

    Parameters
    ----------
    bold_path     : Path to preprocessed bold .nii.gz
    confounds_path: Path to fMRIPrep confounds_regressors.tsv (optional)
    atlas_3d_img  : 3D nibabel Nifti1Image (single parcellation volume)
    n_rois        : Expected number of ROIs (for shape validation)
    masker_kwargs : extra kwargs for NiftiLabelsMasker

    Returns
    -------
    (T, n_rois) array or None on failure
    """
    if not NILEARN_OK:
        print("ERROR: nilearn not installed. Run: pip install nilearn")
        return None

    masker = nilearn.input_data.NiftiLabelsMasker(
        labels_img=atlas_3d_img,   # 3D NIfTI -- NOT atlas.maps (which is 4D)
        standardize=True,          # z-score timeseries per ROI
        detrend=True,              # linear + constant detrend
        t_r=2.0,                   # most ABIDE-II sites use TR=2s
        low_pass=0.1,              # band-pass: match filt_global pipeline
        high_pass=0.01,
        memory_level=0,
        verbose=0,
        **(masker_kwargs or {}),
    )

    confounds_arr = None
    if confounds_path is not None and confounds_path.exists():
        confounds_arr = _load_confounds(confounds_path)

    try:
        ts = masker.fit_transform(str(bold_path), confounds=confounds_arr)
    except Exception as exc:
        print(f"    NiftiLabelsMasker failed: {exc}")
        return None

    if ts.shape[1] != n_rois:
        print(f"    Unexpected ROI count: got {ts.shape[1]}, expected {n_rois}")
        return None

    return ts  # (T, n_rois)


def compute_fc(ts: np.ndarray) -> np.ndarray:
    """Pearson correlation -> Fisher-z -> upper-triangle vector.

    The output length is n_rois*(n_rois-1)//2 where n_rois = ts.shape[1].
    For 200 ROIs this is 19,900; for 195 ROIs it is 18,915.
    """
    n_rois = ts.shape[1]
    corr = np.corrcoef(ts.T)
    np.nan_to_num(corr, copy=False, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    z = np.arctanh(np.clip(corr, -0.9999, 0.9999))
    np.fill_diagonal(z, 0.0)
    return z[np.triu_indices(n_rois, k=1)].astype(np.float32)


# ---------------------------------------------------------------------------
# Concurrent pipeline helpers
# ---------------------------------------------------------------------------

def _download_subject(
    task: SubjectTask,
    tmp_path: Path,
    local_dir: Path | None,
    download: bool,
    disk_semaphore: threading.BoundedSemaphore,
    shutdown_event: threading.Event,
) -> SubjectTask:
    """
    Locate or download BOLD + confounds for a single subject.

    Acquires ``disk_semaphore`` before writing a downloaded file to disk.
    The semaphore is released by ``_process_subject`` after the file is deleted.
    If no file ends up on disk (not found / local), the semaphore is not held.

    Mutates and returns *task* with bold_path / conf_path / was_downloaded set.
    """
    if shutdown_event.is_set():
        return task

    sub_id = task.sub_id

    # Try local files first
    if local_dir is not None:
        bold_path, conf_path = _find_local_bold(local_dir, sub_id)
        if bold_path is not None:
            task.bold_path = bold_path
            task.conf_path = conf_path
            task.was_downloaded = False
            return task

    if not download:
        return task

    # Acquire disk slot BEFORE writing -- blocks if too many files on disk
    disk_semaphore.acquire()
    if shutdown_event.is_set():
        disk_semaphore.release()
        return task

    sub_str = f"sub-{sub_id}"
    bold_dest = tmp_path / sub_str / "bold.nii.gz"
    conf_dest = tmp_path / sub_str / "confounds.tsv"

    bold_url, conf_url = _discover_bold_urls(sub_id)
    downloaded = False
    if bold_url is not None:
        if _try_download(bold_url, bold_dest):
            if conf_url:
                _try_download(conf_url, conf_dest)
            downloaded = True

    if downloaded:
        task.bold_path = bold_dest
        task.conf_path = conf_dest if conf_dest.exists() else None
        task.was_downloaded = True
    else:
        # No file on disk -- release the semaphore immediately
        disk_semaphore.release()

    return task


def _process_subject(
    task: SubjectTask,
    atlas_3d_img,
    n_rois_actual: int,
    n_features_actual: int,
    checkpoint_dir: Path,
    disk_semaphore: threading.BoundedSemaphore | None,
    shutdown_event: threading.Event,
    split_half: bool = False,
) -> dict:
    """
    Extract timeseries, compute FC, save checkpoint, and clean up.

    Releases ``disk_semaphore`` after deleting the downloaded BOLD file
    (only when ``task.was_downloaded`` is True).

    Returns a summary dict compatible with summary_rows.
    """
    sub_id = task.sub_id
    canonical = task.canonical
    dx = task.dx

    if shutdown_event.is_set() or task.bold_path is None or not task.bold_path.exists():
        if task.was_downloaded and disk_semaphore is not None:
            disk_semaphore.release()
        return {"sub_id": sub_id, "site": canonical, "dx": dx, "status": "bold_not_found"}

    try:
        ts = extract_timeseries_from_bold(
            task.bold_path, task.conf_path, atlas_3d_img, n_rois_actual,
        )
    except Exception as exc:
        print(f"    [sub-{sub_id}] extraction error: {exc}")
        ts = None
    finally:
        # Always clean up downloaded files and release disk slot
        if task.was_downloaded:
            if task.bold_path is not None and task.bold_path.exists():
                task.bold_path.unlink(missing_ok=True)
            if task.conf_path is not None and task.conf_path.exists():
                task.conf_path.unlink(missing_ok=True)
            if disk_semaphore is not None:
                disk_semaphore.release()

    if ts is None or ts.shape[0] < 50:
        return {"sub_id": sub_id, "site": canonical, "dx": dx, "status": "extraction_failed"}

    fc = compute_fc(ts)
    if fc.shape[0] != n_features_actual:
        print(f"    [sub-{sub_id}] FC shape mismatch: got {fc.shape[0]}, expected {n_features_actual}")
        return {"sub_id": sub_id, "site": canonical, "dx": dx, "status": "fc_shape_mismatch"}

    fc_h1 = fc_h2 = None
    if split_half and ts.shape[0] >= 8:
        mid = ts.shape[0] // 2
        fc_h1 = compute_fc(ts[:mid])
        fc_h2 = compute_fc(ts[mid:])

    _save_checkpoint(checkpoint_dir, sub_id, fc, dx, canonical, task.ados_val,
                     fc_half1=fc_h1, fc_half2=fc_h2)
    return {"sub_id": sub_id, "site": canonical, "dx": dx, "status": "ok"}


def _run_concurrent_pipeline(
    subjects: list[SubjectTask],
    atlas_3d_img,
    n_rois_actual: int,
    n_features_actual: int,
    checkpoint_dir: Path,
    tmp_path: Path,
    local_dir: Path | None,
    download: bool,
    n_workers: int,
    n_download_threads: int,
    max_on_disk: int = 4,
    split_half: bool = False,
) -> tuple[list[dict], int]:
    """
    Producer-consumer concurrent pipeline.

    Download threads fetch BOLD files and put ready tasks on a processing queue.
    Processing threads extract timeseries, compute FC, checkpoint, and delete.

    Returns (summary_rows, n_new_ok).
    """
    shutdown_event = threading.Event()
    disk_semaphore = threading.BoundedSemaphore(max_on_disk)
    download_q: queue.Queue[SubjectTask] = queue.Queue()
    process_q: queue.Queue[SubjectTask | None] = queue.Queue(maxsize=max_on_disk)

    summary_results: list[dict] = []
    summary_lock = threading.Lock()
    n_new = [0]  # mutable counter shared across threads
    progress = tqdm(total=len(subjects), desc="ABIDE-II FC (concurrent)")

    # Populate download queue
    for task in subjects:
        download_q.put(task)

    download_done_event = threading.Event()

    # --- Install SIGINT handler ---
    original_handler = signal.getsignal(signal.SIGINT)

    def _shutdown_handler(signum, frame):
        print("\n[shutdown] Ctrl+C received -- finishing current subjects...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, _shutdown_handler)

    def _download_worker():
        while not shutdown_event.is_set():
            try:
                task = download_q.get(timeout=1.0)
            except queue.Empty:
                return
            task = _download_subject(
                task, tmp_path, local_dir, download,
                disk_semaphore, shutdown_event,
            )
            if task.bold_path is None or not task.bold_path.exists():
                with summary_lock:
                    summary_results.append({
                        "sub_id": task.sub_id, "site": task.canonical,
                        "dx": task.dx, "status": "bold_not_found",
                    })
                    progress.update(1)
            else:
                process_q.put(task)

    def _process_worker():
        while True:
            if shutdown_event.is_set():
                return
            try:
                task = process_q.get(timeout=1.0)
            except queue.Empty:
                if download_done_event.is_set() and process_q.empty():
                    return
                continue
            if task is None:
                return
            result = _process_subject(
                task, atlas_3d_img, n_rois_actual, n_features_actual,
                checkpoint_dir, disk_semaphore, shutdown_event,
                split_half=split_half,
            )
            with summary_lock:
                summary_results.append(result)
                if result["status"] == "ok":
                    n_new[0] += 1
                progress.update(1)

    # Launch thread pools
    dl_futures = []
    proc_futures = []

    with ThreadPoolExecutor(max_workers=n_download_threads, thread_name_prefix="dl") as dl_pool:
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="proc") as proc_pool:
            for _ in range(n_download_threads):
                dl_futures.append(dl_pool.submit(_download_worker))
            for _ in range(n_workers):
                proc_futures.append(proc_pool.submit(_process_worker))

            # Wait for all downloads to complete
            for f in dl_futures:
                f.result()
            download_done_event.set()

            # Send sentinel values to wake process workers
            for _ in range(n_workers):
                process_q.put(None)

            for f in proc_futures:
                f.result()

    progress.close()

    # Restore original signal handler
    signal.signal(signal.SIGINT, original_handler)

    return summary_results, n_new[0]


# ---------------------------------------------------------------------------
# Local file lookup (for pre-downloaded datasets)
# ---------------------------------------------------------------------------

def _n_timepoints_from_nifti(path: Path) -> int:
    """Read number of timepoints from a 4D NIfTI file; return -1 on failure."""
    if not NIBABEL_OK:
        return -1
    try:
        shape = nib.load(str(path)).shape
    except Exception:
        return -1
    if len(shape) < 4:
        return -1
    return int(shape[3])


def _rest_run_key(filename: str) -> str:
    """Build a task/run key for matching a bold file to confounds."""
    match = re.search(r"(sub-\d+_task-rest(?:_run-\d+)?)", filename)
    return match.group(1) if match else filename


def _match_confounds_for_bold(conf_hits: list[Path], bold_path: Path) -> Path | None:
    """Pick the best confounds TSV candidate for a selected bold file."""
    if not conf_hits:
        return None

    key = _rest_run_key(bold_path.name)
    keyed = [c for c in conf_hits if key in c.name]
    if keyed:
        return sorted(keyed)[0]

    same_dir = [c for c in conf_hits if c.parent == bold_path.parent]
    if same_dir:
        return sorted(same_dir)[0]

    return sorted(conf_hits)[0]


def _find_local_bold(local_dir: Path, sub_id: int) -> tuple[Path | None, Path | None]:
    """
    Search local_dir for pre-downloaded bold + confounds for sub_id.

    Expects one of these layouts (fMRIPrep BIDS output):
      local_dir/sub-{sub_id}/ses-1/func/sub-{sub_id}_ses-1_task-rest_run-1_..._bold.nii.gz
      local_dir/fmriprep/sub-{sub_id}/...
    """
    sub_str = f"sub-{sub_id}"
    bold_pattern = f"{sub_str}_*task-rest*desc-preproc_bold.nii.gz"
    # Support both newer (confounds_timeseries) and older (confounds_regressors) fMRIPrep
    conf_pattern_new = f"{sub_str}_*task-rest*confounds_timeseries.tsv"
    conf_pattern_old = f"{sub_str}_*task-rest*confounds_regressors.tsv"

    bold_hits = sorted(local_dir.rglob(bold_pattern))
    conf_hits = sorted(local_dir.rglob(conf_pattern_new)) or sorted(local_dir.rglob(conf_pattern_old))

    if not bold_hits:
        return None, None

    # Multiple runs may exist; prefer the longest run to maximize signal.
    ranked_bold = sorted(
        bold_hits,
        key=lambda p: (-_n_timepoints_from_nifti(p), str(p)),
    )
    bold = ranked_bold[0]
    conf = _match_confounds_for_bold(conf_hits, bold)
    return bold, conf


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    pheno_path: Path,
    out_dir: Path,
    checkpoint_dir: Path | None = None,
    sites_filter: set[str] | None = None,
    max_per_site: int | None = None,
    local_dir: Path | None = None,
    download: bool = True,
    test_mode: bool = False,
    merge_only: bool = False,
    n_workers: int = 1,
    n_download_threads: int = 3,
    split_half: bool = False,
) -> Path:
    """
    Full ABIDE-II preprocessing pipeline with checkpoint support.

    Parameters
    ----------
    pheno_path         : Path to ABIDEII_Composite_Phenotypic.csv
    out_dir            : Output directory (HDF5, summary CSV)
    checkpoint_dir     : Directory for per-subject .npz checkpoint files.
                         Defaults to <out_dir>/abide2_checkpoints/
    sites_filter       : Set of raw SITE_RAW values to include (None = all)
    max_per_site       : Max subjects per canonical site (counts already-checkpointed ones)
    local_dir          : If provided, look for bold files here instead of downloading
    download           : If True, download from S3 when not found locally
    test_mode          : If True, cap at 2 subjects, no downloads, verify setup only
    merge_only         : If True, skip downloading and just merge existing checkpoints to H5
    n_workers          : Number of concurrent processing workers (default=1 for serial)
    n_download_threads : Number of concurrent download threads (default=3)

    Returns
    -------
    Path to the output HDF5 file.
    """
    if not NILEARN_OK:
        sys.exit(
            "nilearn is required. Install with: pip install nilearn\n"
            "Also install nibabel: pip install nibabel"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = out_dir / "abide2_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Fetch CC200 atlas once and extract the single 3D volume closest to k=200
    print("Fetching CC200 atlas via nilearn...")
    atlas = nilearn.datasets.fetch_atlas_craddock_2012()
    print(f"  Atlas maps file: {atlas.maps}")
    atlas_3d, n_rois_actual = _extract_3d_atlas(atlas)
    n_features_actual = n_rois_actual * (n_rois_actual - 1) // 2
    print(f"  N_ROIS={n_rois_actual}, N_FEATURES={n_features_actual}")

    # --merge-only: skip download loop entirely
    if merge_only:
        print(f"\n[merge-only] Merging existing checkpoints from {checkpoint_dir}...")
        existing = _load_existing_checkpoints(checkpoint_dir)
        print(f"  Found {len(existing)} checkpoint files.")
        h5_path = _merge_checkpoints_to_h5(
            checkpoint_dir, out_dir / "abide2_fc.h5",
            n_rois_actual, n_features_actual,
        )
        return h5_path

    # Load phenotypic
    print(f"\nLoading phenotypic from {pheno_path}...")
    pheno = load_pheno(pheno_path)

    # Surface unexpected site labels early to prevent silent canonical drift.
    site_raw_values = {
        str(s).strip().upper()
        for s in pheno["SITE_RAW"].dropna().unique().tolist()
    }
    unmapped_sites = sorted(site_raw_values.difference(SITE_CANONICAL.keys()))
    if unmapped_sites:
        print(
            "  [warn] Unmapped SITE_RAW values detected; using raw labels as canonical: "
            + ", ".join(unmapped_sites)
        )

    # Filter to requested sites
    if sites_filter is not None:
        sites_filter_upper = {s.upper() for s in sites_filter}
        pheno = pheno[pheno["SITE_RAW"].isin(sites_filter_upper)].copy()
        print(f"  After site filter: {len(pheno)} subjects")
    else:
        print(f"  Total subjects in phenotypic: {len(pheno)}")

    # Load existing checkpoints for resume support
    already_done = _load_existing_checkpoints(checkpoint_dir)
    if already_done:
        print(f"  [resume] {len(already_done)} subjects already checkpointed — will skip.")
    else:
        print(f"  [resume] No existing checkpoints — starting fresh.")

    # Per-site count tracking (includes already-checkpointed subjects)
    site_counts: dict[str, int] = {}

    # Pre-populate site_counts from existing checkpoints so max_per_site is respected
    if max_per_site is not None and already_done:
        for npz in checkpoint_dir.glob("sub-*.npz"):
            m = _CKPT_RE.match(npz.name)
            if not m:
                continue
            try:
                d = np.load(npz, allow_pickle=True)
                site_str = str(d["site"])
                site_counts[site_str] = site_counts.get(site_str, 0) + 1
            except Exception:
                pass

    if test_mode:
        print("  [TEST MODE] Will verify setup only (max 2 subjects, no downloads)")
        pheno = pheno.head(2)

    # --- Build subject task list (pre-filter checkpointed + per-site limits) ---
    subject_tasks: list[SubjectTask] = []
    summary_rows: list[dict] = []

    for _, row in pheno.iterrows():
        sub_id = int(row["SUB_ID"])
        site_raw = str(row["SITE_RAW"]).strip().upper()
        canonical = SITE_CANONICAL.get(site_raw, site_raw)
        dx = int(row.get("DX_NORM", -1))
        if dx not in (0, 1):
            continue

        # Resume: skip subjects already checkpointed
        if sub_id in already_done:
            summary_rows.append({
                "sub_id": sub_id, "site": canonical, "dx": dx,
                "status": "resumed_from_checkpoint",
            })
            continue

        # Enforce per-site limit (counting both already-done and newly-done)
        if max_per_site is not None and site_counts.get(canonical, 0) >= max_per_site:
            summary_rows.append({
                "sub_id": sub_id, "site": canonical, "dx": dx,
                "status": "skipped_max_per_site",
            })
            continue

        # ADOS score (use ADOS_G_TOTAL for ABIDE-II)
        ados_val = np.nan
        for col in ("ADOS_G_TOTAL", "ADOS_2_TOTAL"):
            if col in row.index:
                try:
                    v = float(row[col])
                    if v not in (-9999.0, -9.0):
                        ados_val = v
                        break
                except (ValueError, TypeError):
                    pass

        site_counts[canonical] = site_counts.get(canonical, 0) + 1
        subject_tasks.append(SubjectTask(
            sub_id=sub_id,
            site_raw=site_raw,
            canonical=canonical,
            dx=dx,
            ados_val=ados_val,
        ))

    print(f"  Subjects to process: {len(subject_tasks)}")

    n_new = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        if n_workers > 1 and len(subject_tasks) > 0:
            # --- Concurrent producer-consumer pipeline ---
            print(f"  [concurrent] {n_workers} processing workers, "
                  f"{n_download_threads} download threads")
            concurrent_results, n_new = _run_concurrent_pipeline(
                subjects=subject_tasks,
                atlas_3d_img=atlas_3d,
                n_rois_actual=n_rois_actual,
                n_features_actual=n_features_actual,
                checkpoint_dir=checkpoint_dir,
                tmp_path=tmp_path,
                local_dir=local_dir,
                download=download and not test_mode,
                n_workers=n_workers,
                n_download_threads=n_download_threads,
                split_half=split_half,
            )
            summary_rows.extend(concurrent_results)
        else:
            # --- Original serial loop (backward compatible) ---
            for task in tqdm(subject_tasks, desc="ABIDE-II FC"):
                sub_str = f"sub-{task.sub_id}"

                # --- Locate bold and confounds ---
                bold_path = None
                conf_path = None

                if local_dir is not None:
                    bold_path, conf_path = _find_local_bold(local_dir, task.sub_id)

                _was_downloaded = False
                if bold_path is None and download and not test_mode:
                    bold_dest = tmp_path / sub_str / "bold.nii.gz"
                    conf_dest = tmp_path / sub_str / "confounds.tsv"
                    downloaded = False
                    bold_url, conf_url = _discover_bold_urls(task.sub_id)
                    if bold_url is not None:
                        if _try_download(bold_url, bold_dest):
                            if conf_url:
                                _try_download(conf_url, conf_dest)
                            downloaded = True
                    if downloaded:
                        bold_path = bold_dest
                        conf_path = conf_dest if conf_dest.exists() else None
                        _was_downloaded = True

                if bold_path is None or not bold_path.exists():
                    summary_rows.append({
                        "sub_id": task.sub_id, "site": task.canonical,
                        "dx": task.dx, "status": "bold_not_found",
                    })
                    continue

                # --- Extract timeseries ---
                ts = extract_timeseries_from_bold(
                    bold_path, conf_path, atlas_3d, n_rois_actual,
                )

                # Free disk space: delete downloaded BOLD immediately
                if _was_downloaded and bold_path.exists():
                    bold_path.unlink(missing_ok=True)
                    if conf_path is not None and conf_path.exists():
                        conf_path.unlink(missing_ok=True)

                if ts is None or ts.shape[0] < 50:
                    summary_rows.append({
                        "sub_id": task.sub_id, "site": task.canonical,
                        "dx": task.dx, "status": "extraction_failed",
                    })
                    continue

                fc = compute_fc(ts)
                if fc.shape[0] != n_features_actual:
                    print(f"    FC shape mismatch: got {fc.shape[0]}, "
                          f"expected {n_features_actual}")
                    summary_rows.append({
                        "sub_id": task.sub_id, "site": task.canonical,
                        "dx": task.dx, "status": "fc_shape_mismatch",
                    })
                    continue

                fc_h1 = fc_h2 = None
                if split_half and ts.shape[0] >= 8:
                    mid = ts.shape[0] // 2
                    fc_h1 = compute_fc(ts[:mid])
                    fc_h2 = compute_fc(ts[mid:])

                _save_checkpoint(
                    checkpoint_dir, task.sub_id, fc, task.dx,
                    task.canonical, task.ados_val,
                    fc_half1=fc_h1, fc_half2=fc_h2,
                )
                n_new += 1
                summary_rows.append({
                    "sub_id": task.sub_id, "site": task.canonical,
                    "dx": task.dx, "status": "ok",
                })

    print(f"\n  Newly processed this run: {n_new}")
    print(f"  Total checkpoints: {len(_load_existing_checkpoints(checkpoint_dir))}")

    # Save summary CSV regardless of whether H5 is written
    pd.DataFrame(summary_rows).to_csv(out_dir / "abide2_fc_summary.csv", index=False)

    if test_mode:
        print("\n[TEST MODE] Skipping H5 merge. Re-run without --test to produce H5.")
        return out_dir / "abide2_fc.h5"

    # Merge all checkpoints -> final HDF5
    total_checkpoints = len(_load_existing_checkpoints(checkpoint_dir))
    if total_checkpoints == 0:
        sys.exit(
            "ERROR: No FC matrices produced.\n"
            "If downloading, check S3 access. If using --local-dir, verify path structure."
        )

    print(f"\nMerging {total_checkpoints} checkpoints -> HDF5...")
    h5_path = _merge_checkpoints_to_h5(
        checkpoint_dir, out_dir / "abide2_fc.h5",
        n_rois_actual, n_features_actual,
    )

    print(f"\nPipeline complete.")
    return h5_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ABIDE-II: fMRIPrep + CC200 -> FC matrices -> HDF5 (resumable)"
    )
    parser.add_argument(
        "--pheno", type=str,
        default="data/ABIDE-II/ABIDEII_Composite_Phenotypic.csv",
        help="Path to ABIDEII_Composite_Phenotypic.csv",
    )
    parser.add_argument(
        "--sites", type=str, default=None,
        help=(
            "Comma-separated raw SITE_RAW IDs to include "
            "(e.g. IP_1,UCD_1,UCLA_1). Default: all sites in phenotypic CSV."
        ),
    )
    parser.add_argument(
        "--max-per-site", type=int, default=None,
        help="Maximum subjects per canonical site (default: no cap)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="data/processed",
        help="Output directory (default: data/processed)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help=(
            "Directory for per-subject .npz checkpoint files. "
            "Defaults to <out-dir>/abide2_checkpoints/. "
            "Resume is automatic: re-run the same command after interruption."
        ),
    )
    parser.add_argument(
        "--local-dir", type=str, default=None,
        help=(
            "Root directory of locally downloaded fMRIPrep outputs. "
            "If set, skips S3 download. Expected BIDS layout: "
            "{local_dir}/sub-XXXXXXX/func/sub-XXXXXXX_task-rest_*_bold.nii.gz"
        ),
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Disable S3 download (only use --local-dir files)",
    )
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Skip download loop; just merge existing checkpoints to HDF5",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: verify setup (atlas, phenotypic, checkpoints) without downloading",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Number of concurrent processing workers (default: 1 = serial). "
            "Each worker uses 3-6 GB RAM during nilearn extraction. "
            "Recommended: 2-3 for 16 GB RAM systems."
        ),
    )
    parser.add_argument(
        "--download-threads", type=int, default=3,
        help=(
            "Number of concurrent download threads (default: 3). "
            "Downloads run in the background while processing happens."
        ),
    )
    parser.add_argument(
        "--split-half", action="store_true", default=False,
        help=(
            "Compute and save fc_half1 / fc_half2 (temporal midpoint split) "
            "for fingerprinting. Adds ~1%% overhead; required for --fingerprint."
        ),
    )
    args = parser.parse_args()

    pheno_path = Path(args.pheno)
    if not pheno_path.exists():
        sys.exit(f"ERROR: Phenotypic CSV not found: {pheno_path}")

    sites_filter = parse_site_filter(args.sites)
    local_dir = Path(args.local_dir) if args.local_dir else None
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    run_pipeline(
        pheno_path=pheno_path,
        out_dir=Path(args.out_dir),
        checkpoint_dir=checkpoint_dir,
        sites_filter=sites_filter,
        max_per_site=args.max_per_site,
        local_dir=local_dir,
        download=not args.no_download,
        test_mode=args.test,
        merge_only=args.merge_only,
        n_workers=args.workers,
        n_download_threads=args.download_threads,
        split_half=args.split_half,
    )
