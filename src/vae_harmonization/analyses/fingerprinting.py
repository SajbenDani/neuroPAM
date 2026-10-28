"""
Split-half subject identifiability ("fingerprinting") under harmonization.

Protocol (single-map)
---------------------
Each subject's ROI time series is split at its temporal midpoint and an FC
vector is computed for each half (``fc_half1``/``fc_half2`` in the HDF5).  Both
halves are pushed through the *frozen* scaler -> PCA -> encoder to obtain Z1, Z2.
A harmonizer is fitted **once** on the full-scan latents Z (the same fit used
elsewhere in the paper) and the *same* fitted site maps are applied to Z1 and
Z2.  This is the operationally meaningful setting -- a deployed harmonizer maps
any new scan of a site with one fixed transform -- and it cannot create
identifiability by fitting separate maps to the two halves.

Metrics
-------
* rank-1 identification accuracy across the whole cohort (pool = all subjects)
* within-site rank-1 accuracy (pool = subjects of the same site; removes the
  site-signature shortcut that inflates pre-harmonization identifiability)
* differential identifiability Idiff = mean(self-similarity) - mean(other-
  similarity), Amico & Goni (2018), in percent
* chance levels 1/N and mean_s 1/n_s
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from ..harmonization.base import Harmonizer


# ---------------------------------------------------------------------------
# FC from split time series (used by the preprocessing scripts)
# ---------------------------------------------------------------------------

def split_timeseries_fc(timeseries: np.ndarray, n_rois: int) -> tuple[np.ndarray, np.ndarray]:
    """Split (T, R) time series at T//2 and return the two Fisher-z upper-triangle FC vectors."""
    ts = np.asarray(timeseries, dtype=np.float64)
    if ts.ndim != 2 or ts.shape[1] != n_rois:
        raise ValueError(f"expected (T, {n_rois}) time series, got {ts.shape}")
    if ts.shape[0] < 4:
        raise ValueError("need at least 4 time points to split in half")
    mid = ts.shape[0] // 2
    triu = np.triu_indices(n_rois, k=1)

    def fc(x):
        c = np.clip(np.corrcoef(x.T), -1 + 1e-7, 1 - 1e-7)
        return np.clip(np.arctanh(c), -4.0, 4.0)[triu]

    return fc(ts[:mid]), fc(ts[mid:])


# ---------------------------------------------------------------------------
# Identifiability metrics
# ---------------------------------------------------------------------------

def _similarity(Z1: np.ndarray, Z2: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        return 1.0 - cdist(Z1, Z2, metric="cosine")
    if metric == "euclidean":
        return -cdist(Z1, Z2, metric="euclidean")
    if metric == "correlation":
        return 1.0 - cdist(Z1, Z2, metric="correlation")
    raise ValueError(f"unknown metric '{metric}'")


def fingerprint_metrics(Z1: np.ndarray, Z2: np.ndarray, metric: str = "cosine") -> dict:
    """Rank-1 accuracy, median rank and Idiff for paired representations Z1[i] <-> Z2[i]."""
    Z1 = np.atleast_2d(np.asarray(Z1, dtype=float))
    Z2 = np.atleast_2d(np.asarray(Z2, dtype=float))
    n = Z1.shape[0]
    if n <= 1:
        return {"rank1": float("nan"), "median_rank": float("nan"), "idiff": float("nan"), "n": int(n)}
    S = _similarity(Z1, Z2, metric)
    order = np.argsort(-S, axis=1)
    ranks = np.where(order == np.arange(n)[:, None])[1] + 1
    self_sim = np.diag(S)
    other = S[~np.eye(n, dtype=bool)]
    # symmetric rank-1: half1 -> half2 and half2 -> half1
    order_t = np.argsort(-S.T, axis=1)
    ranks_t = np.where(order_t == np.arange(n)[:, None])[1] + 1
    rank1 = 0.5 * (np.mean(ranks == 1) + np.mean(ranks_t == 1))
    return {
        "rank1": float(rank1),
        "median_rank": float(np.median(np.concatenate([ranks, ranks_t]))),
        "idiff": float(100.0 * (self_sim.mean() - other.mean())),
        "n": int(n),
    }


def within_site_fingerprint(Z1: np.ndarray, Z2: np.ndarray, y_site: np.ndarray, metric: str = "cosine",
                            min_site_size: int = 5, site_names: dict | None = None) -> dict:
    """Per-site identifiability with the candidate pool restricted to the same site."""
    y_site = np.asarray(y_site)
    per_site, weights, accs, idiffs, chances = {}, [], [], [], []
    for s in np.unique(y_site):
        m = y_site == s
        if m.sum() < min_site_size:
            continue
        r = fingerprint_metrics(Z1[m], Z2[m], metric)
        name = (site_names or {}).get(int(s), str(int(s)))
        per_site[name] = r
        weights.append(int(m.sum()))
        accs.append(r["rank1"])
        idiffs.append(r["idiff"])
        chances.append(1.0 / m.sum())
    w = np.asarray(weights, float)
    return {
        "rank1_weighted": float(np.average(accs, weights=w)) if accs else float("nan"),
        "rank1_unweighted": float(np.mean(accs)) if accs else float("nan"),
        "idiff_weighted": float(np.average(idiffs, weights=w)) if idiffs else float("nan"),
        "chance_weighted": float(np.average(chances, weights=w)) if chances else float("nan"),
        "n_sites": len(accs),
        "per_site": per_site,
    }


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def run_fingerprint_analysis(Z: np.ndarray, Z1: np.ndarray, Z2: np.ndarray, y_site: np.ndarray,
                             y_diag: np.ndarray, harmonizers: dict[str, Harmonizer], *,
                             site_names: dict | None = None, metric: str = "cosine",
                             min_site_size: int = 5, fit_aux: dict | None = None) -> dict:
    """
    Identifiability before harmonization and after each harmonizer, using the
    single-map protocol (fit on Z, apply to Z1 and Z2).

    Parameters
    ----------
    Z, Z1, Z2   : (N, d) full-scan and split-half latents (same subject order)
    harmonizers : {name: unfitted Harmonizer}; each is fitted on ``Z``
    fit_aux     : extra keyword arrays forwarded to ``fit`` (e.g. severity)
    """
    y_site = np.asarray(y_site)
    n = Z.shape[0]
    out = {
        "protocol": "single-map: harmonizer fitted on full-scan latents, same site maps applied to both halves",
        "metric": metric,
        "n_subjects": int(n),
        "chance_all": 1.0 / n,
        "conditions": {},
    }
    pre_all = fingerprint_metrics(Z1, Z2, metric)
    pre_within = within_site_fingerprint(Z1, Z2, y_site, metric, min_site_size, site_names)
    out["conditions"]["Unharmonized"] = {"all": pre_all, "within_site": pre_within}
    for name, h in harmonizers.items():
        h.fit(Z, y_site, y_diag, **(fit_aux or {}))
        Z1h = h.transform(Z1, y_site)
        Z2h = h.transform(Z2, y_site)
        post_all = fingerprint_metrics(Z1h, Z2h, metric)
        post_within = within_site_fingerprint(Z1h, Z2h, y_site, metric, min_site_size, site_names)
        out["conditions"][name] = {
            "all": post_all,
            "within_site": post_within,
            "delta_rank1_all_pp": 100.0 * (post_all["rank1"] - pre_all["rank1"]),
            "delta_rank1_within_pp": 100.0 * (post_within["rank1_weighted"] - pre_within["rank1_weighted"]),
            "delta_idiff_within": post_within["idiff_weighted"] - pre_within["idiff_weighted"],
        }
    return out
