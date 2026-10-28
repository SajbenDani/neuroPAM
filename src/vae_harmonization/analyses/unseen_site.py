"""
Unseen-site curve: how well the map of a new site, estimated from a few of its unlabelled scans,
works on that site's *other* scans.

For every eligible site ``s`` and every draw:

* ``n`` scans of ``s`` (labels masked) and 75% of every other site (with labels) fit the harmonizer;
  the remaining 25% of every other site is held out of the fit;
* every scan is transformed;
* site detectability: a cross-validated binary probe separates the *unfitted* scans of ``s`` from an
  equal number of *held-out* scans of the other sites (chance 50%). Both groups were transformed by
  maps that never saw them, so a residual site effect of the new site shows up as accuracy above 50%;
* diagnosis: a probe trained on the fitted scans of the other sites is tested on the unfitted scans of
  ``s``.

The same two probes on the un-harmonized scores give the reference. Sizes are absolute (10, 20, 30, 50
scans) or fractions of the site (0.75). A site is eligible for a size when it has both diagnoses and at
least ``min_test`` scans left after taking ``n``.
"""

from __future__ import annotations

import numpy as np

from ..evaluation.classifiers import fit_and_score
from ..logging_utils import get_logger
from ..stats import summarize_folds

log = get_logger(__name__)


def _cv_accuracy(classifier: str, X: np.ndarray, y: np.ndarray, seed: int, n_splits: int = 5) -> float:
    """Stratified k-fold accuracy of a binary probe (chance 0.5 for balanced groups)."""
    from sklearn.model_selection import StratifiedKFold
    k = int(min(n_splits, np.bincount(y).min()))
    if k < 2:
        return float("nan")
    correct = 0
    for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y):
        sc = fit_and_score(classifier, X[tr], y[tr], X[te], y[te], task="diag", seed=seed)
        correct += sc["acc"] * len(te)
    return float(correct / len(y))


def unseen_site_curve(X, y_site, y_diag, factory, *, sizes=(10, 20, 30, 50, 0.75), n_draws: int = 8,
                      seed: int = 42, holdout_frac: float = 0.25, max_test: int = 40, min_test: int = 10,
                      classifier: str = "lr") -> dict:
    X = np.asarray(X, dtype=np.float64)
    y_site = np.asarray(y_site).astype(int)
    y_diag = np.asarray(y_diag).astype(int)
    rng = np.random.default_rng(seed)
    sites, counts = np.unique(y_site, return_counts=True)
    both = [int(s) for s in sites if np.unique(y_diag[y_site == s]).size == 2]
    out = {"sizes": [str(z) for z in sizes], "n_draws": n_draws, "holdout_frac": holdout_frac,
           "max_test": max_test, "min_test": min_test, "by_size": {}}
    for size in sizes:
        rec = {k: [] for k in ("site_h", "site_raw", "diag_h", "diag_raw")}
        used = []
        for s in both:
            idx = np.flatnonzero(y_site == s)
            n_fit = int(np.floor(size * len(idx))) if isinstance(size, float) else int(size)
            if n_fit < 2 or len(idx) - n_fit < min_test:
                continue
            used.append(s)
            for _ in range(n_draws):
                perm = rng.permutation(idx)
                take, rest_new = perm[:n_fit], perm[n_fit:]
                hold = []
                for o in sites:
                    if int(o) == s:
                        continue
                    io = np.flatnonzero(y_site == o)
                    k = int(round(holdout_frac * len(io)))
                    if k >= 1 and len(io) - k >= 1:
                        hold.extend(rng.choice(io, size=k, replace=False).tolist())
                hold = np.asarray(hold, dtype=int)
                others_fit = np.setdiff1d(np.flatnonzero(y_site != s), hold)
                fit_rows = np.concatenate([others_fit, take])
                y_fit = y_diag.copy()
                y_fit[y_site == s] = -1
                try:
                    h = factory()
                    if getattr(h, "hub", None) == "target" and int(getattr(h, "target_site", -1)) == s:
                        so, co = np.unique(y_site[others_fit], return_counts=True)
                        h.target_site = int(so[np.argmax(co)])
                    Xh = h.fit(X[fit_rows], y_site[fit_rows], y_diag=y_fit[fit_rows]).transform(X, y_site)
                except Exception as exc:
                    log.warning("unseen-site size=%s site=%s failed: %s", size, s, exc)
                    continue
                m = int(min(len(rest_new), len(hold), max_test))
                a = rng.choice(rest_new, size=m, replace=False)
                b = rng.choice(hold, size=m, replace=False)
                pair = np.concatenate([a, b])
                y_bin = (y_site[pair] == s).astype(int)
                rec["site_h"].append(_cv_accuracy(classifier, Xh[pair], y_bin, seed))
                rec["site_raw"].append(_cv_accuracy(classifier, X[pair], y_bin, seed))
                if np.unique(y_diag[rest_new]).size == 2:
                    for key, M in (("diag_h", Xh), ("diag_raw", X)):
                        sc = fit_and_score(classifier, M[others_fit], y_diag[others_fit], M[rest_new],
                                           y_diag[rest_new], task="diag", seed=seed)
                        rec[key].append(float(sc.get("acc", np.nan)))
        key = str(size)
        out["by_size"][key] = {
            "heldout_vs_rest_acc": summarize_folds(rec["site_h"]),
            "heldout_vs_rest_acc_unharmonized": summarize_folds(rec["site_raw"]),
            "diag_acc": summarize_folds(rec["diag_h"]),
            "diag_acc_unharmonized": summarize_folds(rec["diag_raw"]),
            "sites": used, "n": len(rec["site_h"]),
        }
        r = out["by_size"][key]
        log.info("unseen-site size=%s: new-vs-rest acc %.3f (un-harmonized %.3f), diag acc %.3f (%d draws, %d sites)",
                 key, r["heldout_vs_rest_acc"]["mean"], r["heldout_vs_rest_acc_unharmonized"]["mean"],
                 r["diag_acc"]["mean"], r["n"], len(used))
    return out
