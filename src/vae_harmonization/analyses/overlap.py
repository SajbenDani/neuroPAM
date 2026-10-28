"""
Covariate overlap diagnostics per site (age, sex, motion).

When sites do not overlap on a covariate, a residual 'site effect' can be
confounded shift of that covariate (Bridgeford et al. 2025).  The numbers
here are the motivation for covariate-conditional site intercepts: a site
whose subjects are older should not be shifted for that reason.
"""

from __future__ import annotations

import numpy as np

from ..logging_utils import get_logger

log = get_logger(__name__)


def _overlap_1d(a, b, bins=20):
    """Histogram intersection of two 1-D samples, in [0, 1]."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 3 or b.size < 3:
        return float("nan")
    lo, hi = np.nanmin([a.min(), b.min()]), np.nanmax([a.max(), b.max()])
    if hi <= lo:
        return 1.0
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    ha = ha / (ha.sum() + 1e-12)
    hb = hb / (hb.sum() + 1e-12)
    return float(np.minimum(ha, hb).sum())


def covariate_overlap(covariates: dict[str, np.ndarray], y_site: np.ndarray,
                      site_names: dict | None = None) -> dict:
    y_site = np.asarray(y_site).astype(int)
    sites = [int(s) for s in np.unique(y_site)]
    names = {s: (site_names or {}).get(s, str(s)) for s in sites}
    out = {"sites": {str(s): {"name": names[s], "n": int((y_site == s).sum())} for s in sites},
           "covariates": {}}
    for cname, v in covariates.items():
        v = np.asarray(v, dtype=float)
        per = {}
        for s in sites:
            vs = v[y_site == s]
            ok = vs[np.isfinite(vs)]
            entry = {"n": int(ok.size)}
            if ok.size == 0:
                per[str(s)] = entry
                continue
            uniq = np.unique(ok)
            if uniq.size <= 2 and set(np.round(uniq, 6)).issubset({0.0, 1.0}):
                entry["rate"] = float(ok.mean())
            else:
                entry.update({"mean": float(ok.mean()), "std": float(ok.std(ddof=0)),
                              "min": float(ok.min()), "max": float(ok.max())})
            per[str(s)] = entry
        pairs = {}
        for i, si in enumerate(sites):
            for sj in sites[i + 1:]:
                pairs[f"{si}_{sj}"] = _overlap_1d(v[y_site == si], v[y_site == sj])
        vals = [p for p in pairs.values() if p == p]
        out["covariates"][cname] = {
            "per_site": per,
            "pairwise_overlap": pairs,
            "mean_overlap": float(np.mean(vals)) if vals else float("nan"),
            "min_overlap": float(np.min(vals)) if vals else float("nan"),
        }
        log.info("overlap %-10s mean %.2f  min %.2f", cname,
                 out["covariates"][cname]["mean_overlap"], out["covariates"][cname]["min_overlap"])
    return out
