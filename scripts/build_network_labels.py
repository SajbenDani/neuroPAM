"""
Build the CC200 -> Yeo-7 ROI label tables used by the network-preservation analysis.

Writes data/atlases/yeo7_labels_<cohort>.csv for the three cohorts:

  abide1_full   PCP cc200_roi_atlas.nii.gz (ABIDE-I C-PAC rois_cc200; header #1..#200)
  adhd200       Athena ADHD200_parcellate_200.nii.gz (190 non-empty ROIs; ids from a
                cc200_TCs.1D header, "Mean_<id>" columns)
  abide2_10site nilearn fetch_atlas_craddock_2012() maps, volume with label count
                closest to 200 (same rule as preprocess_abide2.py) -> 195 ROIs

Requires nibabel + nilearn (``pip install -e .[preprocess]``).  The Yeo 2011 atlas is
fetched through nilearn on first use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vae_harmonization.analyses.networks import ATLAS_DIR, build_yeo7_labels, label_table_path, roi_ids_from_1d_header  # noqa: E402


def craddock_volume_closest_to_200():
    import nibabel as nib
    from nilearn import datasets

    atlas = datasets.fetch_atlas_craddock_2012()
    img = nib.load(str(atlas["maps"]))
    data = np.asarray(img.get_fdata())
    if data.ndim == 3:
        return img, 0
    best, best_diff = 0, np.inf
    for i in range(data.shape[3]):
        n = np.unique(data[..., i][data[..., i] > 0]).size
        if abs(n - 200) < best_diff:
            best, best_diff = i, abs(n - 200)
    vol = np.round(data[..., best]).astype(np.int32)
    return nib.Nifti1Image(vol, img.affine), best


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--abide1-atlas", default=str(ATLAS_DIR / "cc200_roi_atlas_abide1.nii.gz"))
    p.add_argument("--adhd200-atlas", default=str(REPO / "data" / "ADHD-200" / "templates" / "ADHD200_parcellate_200.nii.gz"))
    p.add_argument("--adhd200-header", default=None,
                   help="any *_cc200_TCs.1D file (ROI id order); default: first one found under data/ADHD-200")
    p.add_argument("--min-overlap", type=float, default=0.3)
    p.add_argument("--cohorts", default="abide1_full,adhd200,abide2_10site")
    args = p.parse_args(argv)
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    cohorts = [c.strip() for c in args.cohorts.split(",")]

    if "abide1_full" in cohorts:
        df = build_yeo7_labels(args.abide1_atlas, roi_ids=np.arange(1, 201), min_overlap=args.min_overlap)
        df.to_csv(label_table_path("abide1_full"), index=False)
        _report("abide1_full", df)

    if "adhd200" in cohorts:
        hdr = args.adhd200_header
        if hdr is None:
            found = sorted((REPO / "data" / "ADHD-200").rglob("*cc200_TCs.1D"))
            if not found:
                raise SystemExit("no *_cc200_TCs.1D file found; pass --adhd200-header")
            hdr = found[0]
        roi_ids = roi_ids_from_1d_header(hdr)
        df = build_yeo7_labels(args.adhd200_atlas, roi_ids=roi_ids, min_overlap=args.min_overlap)
        df.to_csv(label_table_path("adhd200"), index=False)
        _report("adhd200", df)

    if "abide2_10site" in cohorts:
        img, vol = craddock_volume_closest_to_200()
        df = build_yeo7_labels(img, roi_ids=None, min_overlap=args.min_overlap)
        df.to_csv(label_table_path("abide2_10site"), index=False)
        _report(f"abide2_10site (craddock volume {vol})", df)


def _report(name, df):
    counts = df["yeo7_name"].value_counts().to_dict()
    print(f"{name}: {len(df)} ROIs, {int((df.yeo7 >= 0).sum())} cortical -> {counts}")


if __name__ == "__main__":
    main()
