"""
Star-vs-chain simulation for the hub-and-spoke design.

Question (reviewer): why map every site directly to one hub ("star") instead of
aligning sites sequentially ("chain": site_1 -> site_2 -> ... -> hub) or to a
consensus?  With population Gaussians the compositions agree, but every map is
*estimated* from finite samples and the chain composes S-1 noisy maps.

For each replicate we draw S site Gaussians N(mu_s, C_s) in d dimensions
(random SPD covariances with controlled condition number and a common shared
structure), sample n_s subjects per site, and harmonize with

* star/target      : BW map of each site to the (largest) hub site
* star/barycenter  : BW map of each site to the BW barycenter of all sites
* chain            : site_1 -> site_2 -> ... -> site_S (hub) by composing BW maps
                     fitted on the *transformed* previous site

and measure, against the population target, the mean squared Bures-Wasserstein
distance of the harmonized site distributions to the hub, the total expected
squared displacement, and an out-of-sample site-probe accuracy (LR).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..harmonization.transport import (
    bures_wasserstein_barycenter,
    bures_wasserstein_map,
    estimate_covariance,
    gaussian_w2_distance2,
)
from ..stats import summarize_folds


def _random_spd(d: int, rng: np.random.Generator, cond: float = 10.0) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    eig = np.exp(rng.uniform(0.0, np.log(cond), size=d))
    return Q @ np.diag(eig) @ Q.T


def make_sites(n_sites: int, d: int, rng: np.random.Generator, shared: float = 0.7, mean_shift: float = 1.0,
               cond: float = 10.0):
    base = _random_spd(d, rng, cond)
    covs, mus = [], []
    for _ in range(n_sites):
        covs.append(shared * base + (1.0 - shared) * _random_spd(d, rng, cond))
        mus.append(rng.normal(scale=mean_shift, size=d))
    return mus, covs


def _affine_fit(Xs, Xt, method="lw"):
    """Return (A, b) with x -> A(x - mu_s) + mu_t matching means/covariances (BW map)."""
    Cs, _ = estimate_covariance(Xs, method)
    Ct, _ = estimate_covariance(Xt, method)
    return bures_wasserstein_map(Cs, Ct), Xs.mean(axis=0), Xt.mean(axis=0)


def _apply(X, A, mu_s, mu_t):
    return (X - mu_s) @ A.T + mu_t


def _probe_acc(X, y, rng, frac=0.3):
    idx = rng.permutation(len(y))
    n_te = int(frac * len(y))
    te, tr = idx[:n_te], idx[n_te:]
    clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    return float(clf.score(X[te], y[te]))


def run_star_vs_chain(n_sites: int = 6, d: int = 16, n_per_site: int = 40, n_rep: int = 50, seed: int = 0,
                      shared: float = 0.7, mean_shift: float = 1.0) -> dict:
    rng = np.random.default_rng(seed)
    res = {k: {"w2_to_hub_pop": [], "probe_acc": [], "displacement": []}
           for k in ("none", "star_target", "star_barycenter", "chain")}
    for _ in range(n_rep):
        mus, covs = make_sites(n_sites, d, rng, shared, mean_shift)
        sizes = [n_per_site] * n_sites
        sizes[-1] = 3 * n_per_site                       # the hub is the largest site
        X = [rng.multivariate_normal(mus[s], covs[s], size=sizes[s]) for s in range(n_sites)]
        Xte = [rng.multivariate_normal(mus[s], covs[s], size=sizes[s]) for s in range(n_sites)]
        y = np.concatenate([np.full(sizes[s], s) for s in range(n_sites)])
        hub = n_sites - 1

        def evaluate(name, X_h_te, X_h_tr, disp):
            Xall = np.vstack(X_h_te)
            # population-level quality: W2^2 between each harmonized test site (empirical Gaussian) and the hub
            mu_hub, C_hub = X_h_te[hub].mean(axis=0), estimate_covariance(X_h_te[hub])[0]
            w2 = [gaussian_w2_distance2(Xs.mean(axis=0), estimate_covariance(Xs)[0], mu_hub, C_hub)
                  for s, Xs in enumerate(X_h_te) if s != hub]
            res[name]["w2_to_hub_pop"].append(float(np.mean(w2)))
            res[name]["probe_acc"].append(_probe_acc(Xall, y, rng))
            res[name]["displacement"].append(float(disp))

        evaluate("none", Xte, X, 0.0)

        # star / target hub
        out_tr, out_te, disp = [], [], 0.0
        for s in range(n_sites):
            if s == hub:
                out_tr.append(X[s]); out_te.append(Xte[s]); continue
            A, mu_s, mu_t = _affine_fit(X[s], X[hub])
            out_tr.append(_apply(X[s], A, mu_s, mu_t)); out_te.append(_apply(Xte[s], A, mu_s, mu_t))
            disp += np.mean(np.sum((out_te[-1] - Xte[s]) ** 2, axis=1))
        evaluate("star_target", out_te, out_tr, disp / (n_sites - 1))

        # star / barycenter hub
        covs_hat = [estimate_covariance(X[s])[0] for s in range(n_sites)]
        w = np.array(sizes, float)
        C_bar, _ = bures_wasserstein_barycenter(covs_hat, w)
        mu_bar = sum(w[s] * X[s].mean(axis=0) for s in range(n_sites)) / w.sum()
        out_tr, out_te, disp = [], [], 0.0
        for s in range(n_sites):
            A = bures_wasserstein_map(covs_hat[s], C_bar)
            out_tr.append(_apply(X[s], A, X[s].mean(axis=0), mu_bar)); out_te.append(_apply(Xte[s], A, X[s].mean(axis=0), mu_bar))
            disp += np.mean(np.sum((out_te[-1] - Xte[s]) ** 2, axis=1))
        # for comparability measure W2 to the *hub site* after everyone (incl. hub) moved to the barycenter
        evaluate("star_barycenter", out_te, out_tr, disp / n_sites)

        # chain: site_0 -> site_1 -> ... -> hub; at step s the pooled, already-moved sites 0..s-1
        # are mapped onto site s with a map estimated from the *transformed* data
        pos_tr = [X[s].copy() for s in range(n_sites)]
        pos_te = [Xte[s].copy() for s in range(n_sites)]
        for s in range(1, n_sites):
            pooled = np.vstack([pos_tr[r] for r in range(s)])
            A, mu_s, mu_t = _affine_fit(pooled, X[s])
            for r in range(s):
                pos_tr[r] = _apply(pos_tr[r], A, mu_s, mu_t)
                pos_te[r] = _apply(pos_te[r], A, mu_s, mu_t)
        disp = sum(np.mean(np.sum((pos_te[s] - Xte[s]) ** 2, axis=1)) for s in range(n_sites - 1))
        evaluate("chain", pos_te, pos_tr, disp / (n_sites - 1))

    return {
        "settings": dict(n_sites=n_sites, d=d, n_per_site=n_per_site, hub_size=3 * n_per_site, n_rep=n_rep,
                         shared=shared, mean_shift=mean_shift, seed=seed),
        "chance_probe": float(max(sizes) / sum(sizes)),
        "results": {k: {m: summarize_folds(v) for m, v in d_.items()} for k, d_ in res.items()},
    }
