"""
Optional analyses that complement the main leakage-free protocol.

* :mod:`protected_axis`  -- where the site effect lives relative to the protected axis
* :mod:`fingerprinting`  -- split-half identifiability (single-map protocol)
* :mod:`networks`        -- Yeo-7 network preservation on reconstructed FC
* :mod:`downstream`      -- LOSO for all methods, effect-size reproducibility, motion leakage
* :mod:`simulation`      -- star vs chain vs barycenter hub with estimated maps
* :mod:`gaussianity`     -- per-dimension normality diagnostics of a representation

All of them work on a representation fitted once on the whole cohort (label-free): they
are transductive (LOSO fits the harmonizer on every site's scans), descriptive (protected
axis, network preservation) or need one map applied to two half-scans (fingerprinting).
The main protocol numbers never use this representation.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from ..evaluation.protocol import to_jsonable
from ..harmonization import make_variant
from ..logging_utils import get_logger

log = get_logger(__name__)


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def run_optional_analyses(args, cfg, ds_entry, X_raw, Z, y_site, y_diag, severity, covariates, specs, *,
                          space: str, encode, decode, recon_label: str, extra_reps: dict | None = None) -> dict:
    """
    Dispatch the analyses requested on the command line; each writes its own JSON.

    ``Z`` are the full-cohort coordinates of representation ``space`` ('pca' or 'latent'),
    ``encode`` maps raw FC into that representation and ``decode`` maps it back to FC.
    ``extra_reps`` holds further label-free full-cohort representations (e.g. the log-Euclidean 'spd'
    vectors), used by the effect-size analysis.
    """
    out_dir = Path(cfg.out_dir)
    t = int(cfg.target_site)
    common = dict(min_cov_samples=cfg.min_cov_samples, svm_C=cfg.bio_svm_C, ridge=cfg.ridge, seed=cfg.seed,
                  covariance=cfg.covariance)
    # deep baselines re-train a network per call; keep them out of the (already slow) auxiliary analyses
    specs = [s for s in specs if not s.name.startswith(("DeepComBat", "RELIEF", "SLACC"))]
    rep_specs = [s for s in specs if s.representation == space]
    X_raw = np.asarray(X_raw, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    written = {}

    if getattr(args, "protected_axis", False):
        from .protected_axis import protected_axis_report
        rep = protected_axis_report(Z, y_site, y_diag, make_variant("ours", t, **common), severity=severity,
                                    covariates=covariates)
        _dump(out_dir / "protected_axis.json", rep)
        ax = rep["axes"][0] if rep["axes"] else {}
        log.info("protected axis: between-site SS share %.3f (k/d=%.3f); eps2 %.3f -> %.3f; diag d %.2f -> %.2f",
                 rep["between_site_ss"]["protected_share_pre"], rep["k_protected"] / rep["latent_dim"],
                 ax.get("site_eps2_pre", np.nan), ax.get("site_eps2_post", np.nan),
                 ax.get("diag_d_pre", np.nan), ax.get("diag_d_post", np.nan))
        written["protected_axis"] = rep

    if getattr(args, "fingerprint", False):
        from .fingerprinting import run_fingerprint_analysis
        halves = None
        with h5py.File(cfg.hdf5_path, "r") as f:
            if "fc_half1" in f and "fc_half2" in f:
                halves = (f["fc_half1"][:].astype(np.float64), f["fc_half2"][:].astype(np.float64))
        if halves is None:
            log.warning("fingerprinting skipped: HDF5 has no fc_half1/fc_half2")
        else:
            Z1, Z2 = encode(halves[0]), encode(halves[1])
            harmonizers = {s.name: s.factory() for s in rep_specs if not s.is_reference and not s.transform_uses_labels}
            fit_aux = {k: v for k, v in (("severity", severity), ("covariates", covariates)) if v is not None}
            rep = None
            if harmonizers:
                rep = run_fingerprint_analysis(Z, Z1, Z2, y_site, y_diag, harmonizers, site_names=cfg.site_names,
                                               fit_aux=fit_aux)
                rep["representation"] = space
                _dump(out_dir / "fingerprinting.json", rep)
            # edge-residual harmonizers act on the raw connectome: fingerprint the raw halves
            edge = {s.name: s.factory() for s in specs if s.representation == "raw" and "edge residual" in s.name}
            if edge:
                rep_e = run_fingerprint_analysis(X_raw, halves[0], halves[1], y_site, y_diag, edge,
                                                 site_names=cfg.site_names)
                rep_e["representation"] = "raw"
                _dump(out_dir / "fingerprinting_edge.json", rep_e)
                for k, v in rep_e["conditions"].items():
                    log.info("fingerprint (raw edges) %-24s rank-1 all %.3f | within-site %.3f", k,
                             v["all"]["rank1"], v["within_site"]["rank1_weighted"])
                rep = rep_e if rep is None else rep
            if rep is not None:
                for k, v in rep["conditions"].items():
                    log.info("fingerprint %-24s rank-1 all %.3f | within-site %.3f | Idiff %.2f", k, v["all"]["rank1"],
                             v["within_site"]["rank1_weighted"], v["within_site"]["idiff_weighted"])
                written["fingerprinting"] = rep

    if getattr(args, "network_analysis", False):
        from .networks import load_network_labels, run_network_analysis
        try:
            labels = load_network_labels(cfg.dataset_name, n_rois=ds_entry.get("n_rois"))
        except (FileNotFoundError, ValueError) as exc:
            log.warning("network analysis skipped: %s", exc)
            labels = None
        if labels is not None:
            fc = {"Raw FC": X_raw, recon_label: np.asarray(decode(Z), dtype=np.float64)}
            for s in rep_specs:
                if s.is_reference or s.transform_uses_labels:
                    continue
                h = s.factory()
                kw = {"y_diag": y_diag} if "y_diag" in s.fit_aux else {}
                if "severity" in s.fit_aux and severity is not None:
                    kw["severity"] = severity
                if "covariates" in s.fit_aux:
                    if covariates is None:
                        continue
                    kw["covariates"] = covariates
                h.fit(Z, y_site, **kw)
                t_kw = {"covariates": kw["covariates"]} if "covariates" in kw else {}
                fc[f"Decoded {s.name}"] = np.asarray(decode(h.transform(Z, y_site, **t_kw)), dtype=np.float64)
            for s in specs:
                if s.representation == "raw" and s.transform_uses_labels:      # ComBat / CovBat on FC
                    h = s.factory()
                    h.fit(X_raw, y_site, y_diag=y_diag)
                    fc[s.name] = h.transform(X_raw, y_site, y_diag=y_diag)
            rep = run_network_analysis(fc, labels, y_site)
            rep["representation"] = space
            rep["reconstruction_label"] = recon_label
            _dump(out_dir / "network_preservation.json", rep)
            for k, v in rep["conditions"].items():
                if "vs_reference" in v:
                    log.info("network %-28s r=%.4f rmse=%.4f per-subject r=%.3f site eta2 %.3f", k,
                             v["vs_reference"]["pearson_r"], v["vs_reference"]["rmse"], v["per_subject_r"]["mean"],
                             v["site_eta2"]["mean"])
            written["network_preservation"] = rep

    if getattr(args, "downstream", False):
        from .downstream import run_downstream
        reps = {"raw": X_raw, space: Z}
        fd = (covariates or {}).get("mean_fd")
        rep = run_downstream(reps, y_site, y_diag, specs, target_site=t, site_names=cfg.site_names,
                             classifiers=tuple(args.classifiers.split(",")) if getattr(args, "classifiers", None) else ("lr",),
                             seed=cfg.seed, severity=severity, covariates=covariates, fd=fd)
        rep["representation"] = space
        _dump(out_dir / "downstream.json", rep)
        for k, v in rep["loso"]["methods"].items():
            c = list(v["summary"])[0]
            log.info("LOSO %-24s acc %.3f (weighted %.3f)%s", k, v["summary"][c]["acc"]["mean"],
                     v["summary"][c]["acc_weighted"], "  [uses test labels]" if v["uses_test_labels"] else "")
        for k, v in rep["effect_size_reproducibility"].items():
            log.info("effect-size reproducibility %-24s r=%.3f sign agreement %.3f", k, v["r"]["mean"],
                     v["sign_agreement"]["mean"])
        written["downstream"] = rep

    if getattr(args, "effect_sizes", False):
        from ..baselines import ComBatHarmonizer
        from ..evaluation.protocol import MethodSpec
        from .downstream import effect_size_reproducibility_crossfit
        reps = {"raw": X_raw, space: Z, **(extra_reps or {})}
        es_specs = [s for s in specs if s.representation in reps]
        # label-free ComBat in both spaces, for a comparison that does not use the labels of the scored half
        for rep_name, label in (("raw", "ComBat (no covariate)"), (space, f"ComBat ({'PCA' if space == 'pca' else space}, no covariate)")):
            es_specs.append(MethodSpec(label, rep_name, lambda: ComBatHarmonizer(protect_diagnosis=False), fit_aux=()))
        rep = {"protocol": "site halves; labels of the scored half hidden from label-using harmonizers",
               "representation": space, "methods": {}}
        for s in es_specs:
            try:
                rep["methods"][s.name] = effect_size_reproducibility_crossfit(
                    s, reps[s.representation], y_site, y_diag, seed=cfg.seed, severity=severity,
                    covariates=covariates)
            except Exception as exc:
                log.warning("effect sizes: %s failed: %s", s.name, exc)
                continue
            v = rep["methods"][s.name]
            log.info("effect-size r %-30s %.3f (labels: %s)", s.name, v["r"]["mean"], v["labels"])
        _dump(out_dir / "effect_sizes.json", rep)
        written["effect_sizes"] = rep

    if getattr(args, "unseen_site", False):
        from .unseen_site import unseen_site_curve
        factory = lambda: make_variant("ours", t, **common)
        rep = unseen_site_curve(Z, y_site, y_diag, factory, seed=cfg.seed)
        _dump(out_dir / "unseen_site.json", rep)
        written["unseen_site"] = rep

    if getattr(args, "federated", False):
        from .federated import federated_equality
        rep = federated_equality(Z, y_site, y_diag, target_site=t, covariance=cfg.covariance,
                                 ridge=cfg.ridge, seed=cfg.seed)
        _dump(out_dir / "federated.json", rep)
        written["federated"] = rep

    if getattr(args, "overlap", False) and covariates:
        from .overlap import covariate_overlap
        rep = covariate_overlap(covariates, y_site, site_names=cfg.site_names)
        _dump(out_dir / "overlap.json", rep)
        written["overlap"] = rep
    return written
