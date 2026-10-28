#!/usr/bin/env python3
"""
How much of the error of clinical site-mean corrections is plain estimation noise?

For each non-hub scanner shared by the clinical sample (unified protocol) and the traveling subjects:

* the correction a marginal mean shift applies, ``c_s = mean(clinical, s) - mean(clinical, hub)``;
* the travelers' own offset, ``o_s``: the site effect of an additive site + person model, minus the hub's;
* the squared discrepancy ``||c_s - o_s||^2`` and the part of it that noise alone would produce in
  expectation: the sampling variance of the two clinical means, ``tr(S_s)/n_s + tr(S_hub)/n_hub``, plus the
  noise of the travelers' offsets, ``tr(Sigma_e) (1/n_s + 1/n_hub)`` for session noise and
  ``2 tr(Sigma_int) / P`` for the person x scanner interaction averaged over ``P`` people.

Everything is computed on the 100 PCA scores of the clinical basis, as in the transfer setting.

    python scripts/traveling_noise_budget.py      # -> results/srpbs_ts/noise_budget.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vae_harmonization.analyses.traveling import interaction_anova, site_effects  # noqa: E402
from vae_harmonization.evaluation.protocol import to_jsonable  # noqa: E402
from vae_harmonization.evaluation.representations import PCARepresentation  # noqa: E402


def _read(path: Path) -> dict:
    with h5py.File(path, "r") as h:
        out = {k: h[k][:] for k in h.keys() if k != "subject_ids"}
        out["site_names"] = [s.decode() if isinstance(s, bytes) else str(s) for s in h.attrs["site_names"]]
    return out


def _cell_variance_components(Z, site, trav):
    """Within-cell variance trace and interaction variance trace (method of moments, per session)."""
    a = interaction_anova(Z, site, trav)
    tot = float(np.sum((Z - Z.mean(axis=0)) ** 2))
    n = len(site)
    cells = np.unique(trav.astype(np.int64) * 10_000 + site.astype(np.int64), return_counts=True)[1]
    ms_w = a["within_share"] * tot / a["df_within"]
    ms_int = a["interaction_share"] * tot / a["df_interaction"]
    n0 = (n - np.sum(cells.astype(float) ** 2) / n) / (len(cells) - 1)
    return ms_w, max(ms_int - ms_w, 0.0) / n0


def main():
    ts = _read(REPO / "data" / "processed" / "srpbs_ts_fc.h5")
    clin = _read(REPO / "data" / "processed" / "srpbs_decnef_unified_fc.h5")
    names = ts["site_names"]
    site, trav = ts["y_site"].astype(int), ts["traveler_id"].astype(int)
    cn = clin["site_names"]
    keep = np.isin([names[s] for s in site], cn)
    site_t = np.array([cn.index(names[s]) for s in site[keep]])
    trav_t = trav[keep]
    rep = PCARepresentation.fit(clin["X"].astype(np.float64), 100)
    Zc = rep.transform(clin["X"].astype(np.float64))
    Zt = rep.transform(ts["X"][keep].astype(np.float64))
    sc = clin["y_site"].astype(int)
    hub = int(np.argmax(np.bincount(sc)))
    eff = site_effects(Zt, site_t, trav_t)
    tr_e, tr_int = _cell_variance_components(Zt, site_t, trav_t)
    P = np.unique(trav_t).size
    by_site, tot = {}, {"true": 0.0, "discrepancy": 0.0, "noise_clinical": 0.0, "noise_travelers": 0.0}
    for s in np.unique(sc):
        if s == hub:
            continue
        o = eff[s] - eff[hub]
        c = Zc[sc == s].mean(0) - Zc[sc == hub].mean(0)
        disc = float(np.sum((c - o) ** 2))
        n_cl = (np.trace(np.cov(Zc[sc == s].T)) / np.sum(sc == s)
                + np.trace(np.cov(Zc[sc == hub].T)) / np.sum(sc == hub))
        n_ts = tr_e * (1 / np.sum(site_t == s) + 1 / np.sum(site_t == hub)) + 2 * tr_int / P
        by_site[cn[s]] = {"n_clinical": int(np.sum(sc == s)), "n_sessions": int(np.sum(site_t == s)),
                          "true_offset2": float(np.sum(o ** 2)), "discrepancy2": disc,
                          "noise_clinical": float(n_cl), "noise_travelers": float(n_ts)}
        tot["true"] += float(np.sum(o ** 2))
        tot["discrepancy"] += disc
        tot["noise_clinical"] += float(n_cl)
        tot["noise_travelers"] += float(n_ts)
    out = {"hub": cn[hub], "by_site": by_site, "total": tot,
           "noise_share_of_discrepancy": (tot["noise_clinical"] + tot["noise_travelers"]) / tot["discrepancy"],
           "clinical_noise_share_of_discrepancy": tot["noise_clinical"] / tot["discrepancy"]}
    path = REPO / "results" / "srpbs_ts" / "noise_budget.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(out), indent=2), encoding="utf-8")
    print(f"noise explains {100 * out['noise_share_of_discrepancy']:.0f}% of the squared discrepancy "
          f"(clinical sampling {100 * out['clinical_noise_share_of_discrepancy']:.0f}%); wrote {path}")


if __name__ == "__main__":
    main()
