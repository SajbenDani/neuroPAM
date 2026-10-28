"""
PCA-dimension sweep (CPU): the leakage-free protocol for the un-harmonized PCA scores, Ours and a
few ablations at several numbers of principal components.

Example:
    python scripts/sensitivity_grid.py --dataset abide1_full
    python scripts/sensitivity_grid.py --dataset abide1_full --pca-dims 25,50 --n-folds 3   # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vae_harmonization.analyses.sensitivity import (  # noqa: E402
    DEFAULT_PCA_DIMS, DEFAULT_VARIANTS, run_pca_dim_sweep, sweep_table,
)
from vae_harmonization.config import VAEConfig  # noqa: E402
from vae_harmonization.dataset import ABIDEDataset  # noqa: E402
from vae_harmonization.evaluation.protocol import to_jsonable  # noqa: E402
from vae_harmonization.logging_utils import configure_logging  # noqa: E402
from vae_harmonization.run_vae_harmonization import _configure_sites_from_hdf5, _load_covariates, _load_severity  # noqa: E402
from vae_harmonization.data_registry import get_dataset_config  # noqa: E402


def _ints(s):
    return tuple(int(x) for x in s.split(",") if x.strip())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="abide1_full")
    p.add_argument("--config-yaml", default=None)
    p.add_argument("--pca-dims", default=",".join(map(str, DEFAULT_PCA_DIMS)))
    p.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    p.add_argument("--covariance", default=None, choices=("pooled", "lw", "diagonal", "empirical"))
    p.add_argument("--n-folds", type=int, default=None)
    p.add_argument("--classifier", default="lr")
    p.add_argument("--out", default=None, help="default results/<dataset>_sensitivity/pca_sweep.json")
    a = p.parse_args(argv)

    cfg = VAEConfig.from_yaml(a.config_yaml) if a.config_yaml else VAEConfig.from_dataset(a.dataset)
    if a.covariance:
        cfg.covariance = a.covariance
    _configure_sites_from_hdf5(cfg)
    try:
        ds_entry = get_dataset_config(cfg.dataset_name)
    except KeyError:
        ds_entry = {}
    out = Path(a.out) if a.out else REPO / "results" / f"{cfg.dataset_name}_sensitivity" / "pca_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=out.parent / "pca_sweep.log")
    ds = ABIDEDataset(cfg.hdf5_path)
    severity = _load_severity(cfg, ds_entry)
    covariates = _load_covariates(cfg, ds_entry, ds.subject_ids) or None

    def save(cells):
        with open(out, "w", encoding="utf-8") as f:
            json.dump(to_jsonable({"partial": True, "cells": cells}), f, indent=2)

    grid = run_pca_dim_sweep(cfg, ds.X_raw.astype(float), ds.y_site.astype(int), ds.y_diag.astype(int),
                             pca_dims=_ints(a.pca_dims), variants=tuple(v for v in a.variants.split(",") if v),
                             classifiers=(a.classifier,), n_folds=a.n_folds, severity=severity,
                             covariates=covariates, progress_cb=save)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(grid), f, indent=2)
    for key in ("diag_acc", "site_acc", "site_nmi", "mmd2", "displacement"):
        print(sweep_table(grid, key))
        print()
    print("written", out)


if __name__ == "__main__":
    main()
