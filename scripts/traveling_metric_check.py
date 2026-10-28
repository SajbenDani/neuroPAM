#!/usr/bin/env python3
"""
Why the pooled scanner metric cannot judge the leave-one-traveler-out setting.

Synthetic sessions on the real visit pattern of the traveling subjects (which person was scanned how often
at which scanner; no connectivity values are read): person effects, scanner offsets that are either shared
by everyone or partly person-specific, and session noise. Each person is corrected with the scanner means of
the other people (a leave-one-traveler-out mean shift). The pooled scanner sum of squares of an additive
scanner + person model, relative to before, is close to zero in every case; the per-person metric (each
person's between-scanner spread, noise floor subtracted) shows how much of a new person's scanner effect the
other people's means actually remove.

    python scripts/traveling_metric_check.py     # -> results/srpbs_ts/metric_check.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vae_harmonization.analyses.traveling import per_person_scanner_left, variance_shares  # noqa: E402
from vae_harmonization.harmonization.affine import ProtectedAffineHarmonizer  # noqa: E402


def main(d: int = 100, n_rep: int = 5, seed: int = 0):
    with h5py.File(REPO / "data" / "processed" / "srpbs_ts_fc.h5", "r") as h:
        site, trav = h["y_site"][:].astype(int), h["traveler_id"][:].astype(int)
    hub = int(np.argmax(np.bincount(site)))
    S, P = np.unique(site).size, np.unique(trav).size
    rng = np.random.default_rng(seed)
    out = {"n_sessions": int(len(site)), "d": d, "n_rep": n_rep, "settings": {}}
    for ratio in (0.0, 1.0, 4.0):           # variance of person-specific offsets relative to shared ones
        pooled, per = [], []
        for _ in range(n_rep):
            b = rng.normal(size=(S, d))
            ab = rng.normal(scale=np.sqrt(ratio), size=(P, S, d))
            a = rng.normal(scale=2.0, size=(P, d))
            Z = a[trav] + b[site] + ab[trav, site] + rng.normal(size=(len(site), d))
            Zh = np.empty_like(Z)
            for t in np.unique(trav):
                tr, te = trav != t, trav == t
                h = ProtectedAffineHarmonizer(target_site=hub, protected=(), complement_map="identity").fit(Z[tr], site[tr])
                Zh[te] = h.transform(Z[te], site[te])
            pooled.append(variance_shares(Zh, site, trav)["ss_site"] / variance_shares(Z, site, trav)["ss_site"])
            per.append(per_person_scanner_left(Z, Zh, site, trav)["left"])
        out["settings"][str(ratio)] = {"pooled_left": float(np.mean(pooled)), "per_person_left": float(np.mean(per))}
        print(f"person-specific / shared = {ratio}: pooled left {100 * np.mean(pooled):.1f}%, "
              f"per person {100 * np.mean(per):.1f}%")
    path = REPO / "results" / "srpbs_ts" / "metric_check.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
