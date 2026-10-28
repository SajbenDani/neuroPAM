"""
Sensitivity to the representation: PCA-dimension sweep (x covariance estimator).

For every number of principal components the leakage-free protocol is run with the
per-fold PCA representation for the un-harmonized scores (reference), Ours and a few
ablations, and only the summary numbers are kept.  Every cell is deterministic and
runs on the CPU in a few minutes; ``scripts/sensitivity_grid.py`` exposes it on the
command line.
"""

from __future__ import annotations

import copy
import time

from ..config import VAEConfig
from ..evaluation.protocol import MethodSpec, run_protocol
from ..evaluation.representations import make_pca_representation_fn
from ..harmonization import IdentityHarmonizer, make_variant
from ..logging_utils import get_logger

log = get_logger(__name__)

DEFAULT_PCA_DIMS = (25, 50, 75, 100, 150, 200)
DEFAULT_VARIANTS = ("ours", "ours-lw", "mean-shift", "full-affine")


def _cell_specs(cfg: VAEConfig, variants) -> list[MethodSpec]:
    t = int(cfg.target_site)
    common = dict(min_cov_samples=cfg.min_cov_samples, svm_C=cfg.bio_svm_C, ridge=cfg.ridge, seed=cfg.seed,
                  covariance=cfg.covariance)
    specs = [MethodSpec("PCA scores", "pca", IdentityHarmonizer, fit_aux=(), is_reference=True)]
    for v in variants:
        params = dict(common)
        specs.append(MethodSpec(v, "pca", (lambda v=v, params=params: make_variant(v, t, name=v, **params)),
                                fit_aux=() if v == "full-affine" else ("y_diag",), is_proposed=(v == "ours")))
    return specs


def _summ(results: dict, name: str, clf: str) -> dict:
    e = results["methods"][name]
    s = e["summary"]
    d = e.get("oos_distribution", {})
    return {
        "diag_acc": s["diag"][clf]["acc"]["mean"], "diag_bacc": s["diag"][clf]["bacc"]["mean"],
        "diag_auc": s["diag"][clf]["auc"]["mean"],
        "site_acc": s["site"][clf]["acc"]["mean"], "site_bacc": s["site"][clf]["bacc"]["mean"],
        "site_nmi": s["site"][clf]["nmi"]["mean"],
        "icc": d.get("icc_mean"), "mmd2": d.get("mmd2_unbiased_mean"),
        "displacement": e["displacement"]["mean"],
        "protected_eps2_post": e["protected_eps2"]["mean"], "protected_eps2_pre": e["protected_eps2_pre"]["mean"],
    }


def run_pca_dim_sweep(cfg_base: VAEConfig, X_raw, y_site, y_diag, *, pca_dims=DEFAULT_PCA_DIMS,
                      variants=DEFAULT_VARIANTS, classifiers=("lr",), n_folds: int | None = None,
                      severity=None, covariates=None, progress_cb=None) -> dict:
    cells = []
    for k, pca_dim in enumerate(pca_dims, 1):
        cfg = copy.deepcopy(cfg_base)
        cfg.pca_dim = cfg.input_dim = int(pca_dim)
        if n_folds is not None:
            cfg.n_cv_folds = int(n_folds)
        if pca_dim > min(X_raw.shape) - 1:
            log.warning("pca_dim %d exceeds data rank; skipped", pca_dim)
            continue
        t0 = time.perf_counter()
        rep_fn = make_pca_representation_fn(X_raw, int(pca_dim), seed=cfg.seed)
        specs = _cell_specs(cfg, variants)
        res = run_protocol(rep_fn, y_site, y_diag, specs, n_folds=cfg.n_cv_folds, seed=cfg.seed,
                           classifiers=tuple(classifiers), severity=severity, covariates=covariates,
                           return_oos=False)
        clf = classifiers[0]
        cell = {"pca_dim": int(pca_dim), "seconds": time.perf_counter() - t0,
                "explained_variance_ratio_total": None,
                "raw": _summ(res, "PCA scores", clf), **{v: _summ(res, v, clf) for v in variants}}
        cells.append(cell)
        log.info("sweep %d/%d pca=%d: diag %.3f->%.3f site %.3f->%.3f NMI %.3f->%.3f (%.0fs)", k, len(pca_dims),
                 pca_dim, cell["raw"]["diag_acc"], cell["ours"]["diag_acc"], cell["raw"]["site_acc"],
                 cell["ours"]["site_acc"], cell["raw"]["site_nmi"], cell["ours"]["site_nmi"], cell["seconds"])
        if progress_cb:
            progress_cb(cells)
    return {"grid": {"pca_dims": list(pca_dims), "variants": list(variants), "covariance": cfg_base.covariance},
            "dataset": cfg_base.dataset_name, "classifier": classifiers[0], "cells": cells}


def sweep_table(grid: dict, key: str = "diag_acc") -> str:
    """Text table pca_dim x variant of ``key`` (reference in the first column)."""
    variants = grid["grid"]["variants"]
    lines = [f"{key}", f"{'pca':<6}{'PCA scores':>14}" + "".join(f"{v:>14}" for v in variants)]
    for c in grid["cells"]:
        vals = [c["raw"][key]] + [c[v][key] for v in variants]
        lines.append(f"{c['pca_dim']:<6}" + "".join(f"{'-' if v is None else f'{v:.3f}':>14}" for v in vals))
    return "\n".join(lines)
