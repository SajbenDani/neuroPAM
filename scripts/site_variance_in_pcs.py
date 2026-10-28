#!/usr/bin/env python3
"""
How much of each cohort's between-site variance lies inside the first 100 principal components?

The PCA is the label-free full-cohort fit used by the auxiliary analyses (standardized edges). The
between-site sum of squares is that of the site means around the weighted grand mean; its share inside the
components is the part a score-space harmonizer can reach, and the rest is what the edge-residual version
(``x' = x + R(T(z)) - R(z)``) leaves untouched.

    python scripts/site_variance_in_pcs.py        # -> results/<cohort>/site_variance_pcs.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vae_harmonization.data_registry import DATASET_REGISTRY  # noqa: E402
from vae_harmonization.evaluation.representations import PCARepresentation  # noqa: E402

COHORTS = ("abide1_full", "adhd200", "abide2_10site", "rest_meta_mdd", "srpbs")


def main():
    for key in COHORTS:
        path = Path(DATASET_REGISTRY[key]["hdf5_path"])
        if not path.exists():
            print("missing", path)
            continue
        with h5py.File(path, "r") as h:
            X = h["X"][:].astype(np.float64)
            s = h["y_site"][:].astype(int)
        rep = PCARepresentation.fit(X, 100, seed=42)
        Xs = rep.scaler.transform(X)
        sites, n = np.unique(s, return_counts=True)
        M = np.stack([Xs[s == k].mean(axis=0) for k in sites])
        G = M - np.average(M, axis=0, weights=n)
        ss_all = float(np.sum(n[:, None] * G ** 2))
        ss_in = float(np.sum(n[:, None] * (G @ rep.pca.components_.T) ** 2))
        out = {"between_site_share_in_pcs": ss_in / ss_all,
               "total_variance_share_in_pcs": float(rep.pca.explained_variance_ratio_.sum()), "n_components": 100}
        dst = REPO / "results" / key / "site_variance_pcs.json"
        dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"{key}: between-site share {100 * out['between_site_share_in_pcs']:.0f}%, "
              f"total variance {100 * out['total_variance_share_in_pcs']:.0f}%")


if __name__ == "__main__":
    main()
