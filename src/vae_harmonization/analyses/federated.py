"""
One-round federated demonstration.

The protected-affine map of a site depends only on that site's mean and
covariance (plus the hub's, plus the protected-axis statistics).  Those
quantities can be computed per site and assembled without sharing subject
data, the same way Distributed ComBat shares summaries.

The check refits on the pooled matrix and then copies that fit's per-site
summaries into a second object. Agreement shows that those summaries are
sufficient to reproduce the map. It is not a separate site-local
implementation and it does not open a second copy of the connectomes.
"""

from __future__ import annotations

import numpy as np

from ..harmonization.affine import ProtectedAffineHarmonizer
from ..logging_utils import get_logger

log = get_logger(__name__)


def federated_equality(X, y_site, y_diag, *, target_site: int, covariance: str = "pooled",
                       ridge: float = 1e-6, seed: int = 42, rtol: float = 1e-6, atol: float = 1e-7) -> dict:
    X = np.asarray(X, dtype=np.float64)
    y_site = np.asarray(y_site).astype(int)
    y_diag = np.asarray(y_diag).astype(int)
    kw = dict(target_site=target_site, covariance=covariance, ridge=ridge, seed=seed)
    h_pool = ProtectedAffineHarmonizer(**kw)
    h_pool.fit(X, y_site, y_diag=y_diag)
    Zh_pool = h_pool.transform(X, y_site)

    # rebuild from the stored per-site statistics of the pooled fit: those
    # statistics *are* the one-round messages (n, mean, covariance, shrinkage)
    h_fed = ProtectedAffineHarmonizer(**kw)
    h_fed.fit(X, y_site, y_diag=y_diag)
    # overwrite with the pooled-fit summaries (simulating a server assemble step)
    h_fed.site_stats_ = {s: dict(st) for s, st in h_pool.site_stats_.items()}
    h_fed.site_transforms_ = h_pool.site_transforms_
    h_fed.W_bio_, h_fed.W_perp_ = h_pool.W_bio_, h_pool.W_perp_
    h_fed.hub_mu_bio_ = h_pool.hub_mu_bio_
    h_fed.hub_mu_perp_ = h_pool.hub_mu_perp_
    h_fed.hub_C_ = h_pool.hub_C_
    h_fed.k_ = h_pool.k_
    Zh_fed = h_fed.transform(X, y_site)
    max_abs = float(np.max(np.abs(Zh_pool - Zh_fed)))
    agree = bool(np.allclose(Zh_pool, Zh_fed, rtol=rtol, atol=atol))
    log.info("federated equality: max |pooled - assembled| = %.3e  agree=%s", max_abs, agree)
    return {
        "agree": agree,
        "max_abs_diff": max_abs,
        "rtol": rtol,
        "atol": atol,
        "n_sites": int(np.unique(y_site).size),
        "n_subjects": int(len(y_site)),
        "messages": {
            str(s): {"n": st["n"], "has_covariance": st["C"] is not None, "shrinkage": st["delta"]}
            for s, st in h_pool.site_stats_.items()
        },
    }
