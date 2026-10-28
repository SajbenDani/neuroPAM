#!/usr/bin/env python3
"""
Traveling-subject evaluation on the DecNef Harmonization Dataset (Yamashita et al. 2019).

    python scripts/run_traveling.py                       # -> results/srpbs_ts/traveling.json

Setting 'within': leave one of the nine travelers out, fit the harmonizer on the other eight
(twelve scanners, site labels only), transform the held-out person. Scaler and PCA are learned from
the clinical sample (independent people); 'within_refit' re-fits them on the eight travelers.
Setting 'transfer': fit scaler, PCA and harmonizer on the clinical SRPBS scans acquired with the
travelers' unified protocol (six scanners, controls and patients), transform the traveling-subject
sessions recorded at those scanners. 'transfer_all' uses every clinical scan of the release,
including the older site-specific protocols of two Hiroshima scanners.

Inputs are written by scripts/preprocessing/preprocess_srpbs.py. Data and results stay local
(data-use terms of ATR); results/srpbs* is gitignored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vae_harmonization.analyses.traveling import TSMethod, run_transfer, run_within_travelers  # noqa: E402
from vae_harmonization.baselines import ComBatHarmonizer, CORALHarmonizer, CovBatHarmonizer  # noqa: E402
from vae_harmonization.evaluation.protocol import to_jsonable  # noqa: E402
from vae_harmonization.evaluation.representations import PCARepresentation  # noqa: E402
from vae_harmonization.harmonization import IdentityHarmonizer  # noqa: E402
from vae_harmonization.harmonization.affine import ProtectedAffineHarmonizer  # noqa: E402
from vae_harmonization.harmonization.edge import EdgeResidualHarmonizer  # noqa: E402
from vae_harmonization.logging_utils import configure_logging, get_logger  # noqa: E402

log = get_logger("run_traveling")
DX = {"asd": 1, "mdd": 2, "scz": 4}


def _read(path: Path) -> dict:
    with h5py.File(path, "r") as h:
        out = {k: h[k][:] for k in h.keys() if k != "subject_ids"}
        out["site_names"] = [s.decode() if isinstance(s, bytes) else str(s) for s in h.attrs["site_names"]]
    return out


def pam(hub, name, **kw):
    return lambda: ProtectedAffineHarmonizer(target_site=hub, name=name, **kw)


def within_methods(hub: int, rep=None) -> list[TSMethod]:
    """No diagnosis exists among healthy travelers, so nothing is protected."""
    edge = []
    if rep is not None:
        # PAM / mean shift on the clinical-basis scores, change added back to the full connectome
        edge = [
            TSMethod("PAM (edge residual)", "raw",
                     lambda: EdgeResidualHarmonizer(pam(hub, "PAM", protected=()), representation=rep)),
            TSMethod("Mean shift (edge residual)", "raw",
                     lambda: EdgeResidualHarmonizer(pam(hub, "Mean shift", protected=(), complement_map="identity"),
                                                    representation=rep)),
        ]
    return edge + [
        TSMethod("Raw FC", "raw", IdentityHarmonizer, reference=True),
        TSMethod("ComBat", "raw", lambda: ComBatHarmonizer(protect_diagnosis=False)),
        TSMethod("CovBat", "raw", lambda: CovBatHarmonizer(protect_diagnosis=False)),
        TSMethod("PCA scores", "pca", IdentityHarmonizer, reference=True),
        TSMethod("ComBat (PCA)", "pca", lambda: ComBatHarmonizer(protect_diagnosis=False)),
        TSMethod("CORAL", "pca", lambda: CORALHarmonizer(target_site=hub)),
        TSMethod("Mean shift", "pca", pam(hub, "Mean shift", protected=(), complement_map="identity")),
        TSMethod("PAM", "pca", pam(hub, "PAM", protected=())),
        TSMethod("PAM (LW covariance)", "pca", pam(hub, "PAM (LW covariance)", protected=(), covariance="lw")),
        TSMethod("PAM (empirical covariance)", "pca",
                 pam(hub, "PAM (empirical covariance)", protected=(), covariance="empirical")),
    ]


def transfer_methods(hub: int, clin: dict, pca_dim: int = 100) -> list[TSMethod]:
    code = clin["dx_code"].astype(int)
    cov = {k: (code == v).astype(float) for k, v in DX.items()}
    cov["age"] = clin["age"].astype(float)
    cov["sex"] = clin["sex"].astype(float)
    dx_axes = ("mdd", "scz", "asd")

    def with_cov(idx):
        return {"covariates": {k: v[idx] for k, v in cov.items()}}

    def with_dx(idx):
        return {"y_diag": code[idx]}

    hc = {"y_diag": None}           # filled per call below (the travelers are healthy controls)
    return [
        TSMethod("Raw FC", "raw", IdentityHarmonizer, reference=True),
        TSMethod("ComBat (diagnosis covariate)", "raw", lambda: ComBatHarmonizer(protect_diagnosis=True), with_dx, hc),
        TSMethod("ComBat (no covariate)", "raw", lambda: ComBatHarmonizer(protect_diagnosis=False)),
        TSMethod("PCA scores", "pca", IdentityHarmonizer, reference=True),
        TSMethod("ComBat (PCA, diagnosis covariate)", "pca", lambda: ComBatHarmonizer(protect_diagnosis=True),
                 with_dx, hc),
        TSMethod("ComBat (PCA, no covariate)", "pca", lambda: ComBatHarmonizer(protect_diagnosis=False)),
        TSMethod("CORAL", "pca", lambda: CORALHarmonizer(target_site=hub)),
        TSMethod("Mean shift, marginal means", "pca",
                 pam(hub, "Mean shift, marginal means", protected=(), complement_map="identity")),
        TSMethod("Full-space BW, marginal means", "pca", pam(hub, "Full-space BW, marginal means", protected=())),
        TSMethod("PAM, marginal means", "pca",
                 pam(hub, "PAM, marginal means", protected=dx_axes, mean_model="marginal"), with_cov),
        TSMethod("Mean shift (diagnosis axes)", "pca",
                 pam(hub, "Mean shift (diagnosis axes)", protected=dx_axes, complement_map="identity"), with_cov),
        TSMethod("PAM", "pca", pam(hub, "PAM", protected=dx_axes), with_cov),
        TSMethod("PAM (diagnosis + age + sex)", "pca",
                 pam(hub, "PAM (diagnosis + age + sex)", protected=dx_axes + ("age", "sex")), with_cov),
        # positive-part James-Stein shrinkage of the site offsets (a diagnostic of estimation noise)
        TSMethod("Mean shift, marginal means, James-Stein", "pca",
                 pam(hub, "Mean shift, marginal means, James-Stein", protected=(), complement_map="identity",
                     mean_shrinkage="james-stein")),
        TSMethod("PAM, marginal means, James-Stein", "pca",
                 pam(hub, "PAM, marginal means, James-Stein", protected=dx_axes, mean_model="marginal",
                     mean_shrinkage="james-stein"), with_cov),
        TSMethod("PAM, James-Stein", "pca",
                 pam(hub, "PAM, James-Stein", protected=dx_axes, mean_shrinkage="james-stein"), with_cov),
        # PAM fitted on the clinical PCA scores, change added back to the full connectome
        TSMethod("PAM (edge residual)", "raw",
                 lambda: EdgeResidualHarmonizer(pam(hub, "PAM", protected=dx_axes), pca_dim=pca_dim, seed=42),
                 with_cov),
    ]


def _transfer(clin: dict, X, site, traveler, names, pca_dim, probes) -> dict:
    cnames = clin["site_names"]
    keep = np.isin([names[s] for s in site], cnames)
    site_t = np.array([cnames.index(names[s]) for s in site[keep]])
    s_clin = clin["y_site"].astype(int)
    hub = int(np.argmax(np.bincount(s_clin)))
    methods = transfer_methods(hub, clin, pca_dim)
    for m in methods:
        if "y_diag" in m.transform_kw:
            m.transform_kw = {"y_diag": np.zeros(int(keep.sum()), dtype=int)}
    log.info("transfer: %d clinical subjects at %s; %d traveling sessions at those scanners; hub %s",
             len(s_clin), cnames, int(keep.sum()), cnames[hub])
    res = run_transfer(clin["X"].astype(np.float64), s_clin, X[keep], site_t, traveler[keep], methods,
                       hub=hub, pca_dim=pca_dim, probes=probes)
    res["hub_name"] = cnames[hub]
    res["clinical_sites"] = cnames
    res["clinical_site_counts"] = np.bincount(s_clin).tolist()
    code = clin["dx_code"].astype(int)
    res["clinical_diagnoses"] = {"HC": int((code == 0).sum()), **{k.upper(): int((code == v).sum())
                                                                    for k, v in DX.items()}}
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ts", default=str(REPO / "data" / "processed" / "srpbs_ts_fc.h5"))
    p.add_argument("--clinical", default=str(REPO / "data" / "processed" / "srpbs_decnef_unified_fc.h5"),
                   help="clinical sample acquired with the travelers' (unified) protocol")
    p.add_argument("--clinical-all", default=str(REPO / "data" / "processed" / "srpbs_decnef_fc.h5"),
                   help="all clinical scans, including site-specific protocols (setting 'transfer_all')")
    p.add_argument("--out", default=str(REPO / "results" / "srpbs_ts" / "traveling.json"))
    p.add_argument("--pca-dim", type=int, default=100)
    p.add_argument("--no-probes", action="store_true", help="skip the probes (smoke test)")
    p.add_argument("--settings", default="within,transfer,transfer_all")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=out.parent / "traveling.log")
    t0 = time.perf_counter()

    ts = _read(Path(a.ts))
    site, traveler, X = ts["y_site"].astype(int), ts["traveler_id"].astype(int), ts["X"].astype(np.float64)
    names = ts["site_names"]
    result = {"ts_sites": names, "sessions_per_site": np.bincount(site, minlength=len(names)).tolist(),
              "sessions_per_traveler": np.bincount(traveler).tolist()}
    settings = a.settings.split(",")
    clin = _read(Path(a.clinical))
    probes = not a.no_probes

    hub = int(np.argmax(np.bincount(site)))
    for key in ("within", "within_refit"):
        if key not in settings:
            continue
        # 'within': scaler + PCA learned from the clinical sample (independent people), so no
        # quantity is estimated from a held-out traveler; 'within_refit': re-fitted on 8 travelers
        rep = PCARepresentation.fit(clin["X"].astype(np.float64), a.pca_dim) if key == "within" else None
        log.info("%s: %d sessions, %d sites, hub %s", key, len(site), len(names), names[hub])
        res = run_within_travelers(X, site, traveler, within_methods(hub, rep), hub=hub, pca_dim=a.pca_dim,
                                   probes=probes, rep=rep)
        res["hub_name"] = names[hub]
        result[key] = res

    if "transfer" in settings:
        result["transfer"] = _transfer(clin, X, site, traveler, names, a.pca_dim, probes)
    if "transfer_all" in settings:
        result["transfer_all"] = _transfer(_read(Path(a.clinical_all)), X, site, traveler, names, a.pca_dim, probes)

    result["seconds"] = time.perf_counter() - t0
    with open(out, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(result), f, indent=2)
    log.info("wrote %s in %.1f min", out, result["seconds"] / 60)


if __name__ == "__main__":
    main()
