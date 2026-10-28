"""
traveling.py -- Traveling subjects as ground truth for scanner effects.

The same nine healthy people were scanned at every site (Yamashita et al. 2019), so any
systematic difference between the sites of one person is a scanner effect, and differences
between people are biology. A harmonizer should shrink the first and keep the second.

Two settings, both leakage-free with respect to the people being evaluated:

* ``run_within_travelers`` -- leave one traveler out: the harmonizer is fitted on the sessions of
  the other eight travelers (site labels only), and the held-out person's sessions are
  transformed. The representation is a scaler + PCA fitted on independent people (the clinical
  sample), or re-fitted on the eight training travelers. The held-out sessions of all nine folds
  are pooled in the Fisher-z edge space.
* ``run_transfer`` -- the realistic case: scaler, PCA and harmonizer are fitted on a clinical
  sample scanned at the same scanners (different people, different diagnoses and ages), and
  applied to the traveling subjects. Site means of a clinical sample mix the scanner effect with
  the make-up of each site's sample (``sampling bias``); only the scanner effect should go.

Metrics on the pooled traveling-subject matrix (every row transformed by a map that never saw
that person):

* variance shares of an additive two-way model edge by edge: site given traveler, traveler given
  site, and the remainder (session-to-session noise), summed over edges;
* ``scanner_ss_left``: the site sum of squares relative to the un-harmonized input of the same
  representation (raw edges for raw-space methods, the PCA reconstruction for PCA-space methods).
  In the leave-one-traveler-out setting this pooled site effect cancels by construction for any
  method that removes the site means of the other travelers: with one session per person and site,
  every person's correction is P/(P-1) times the deviation from the full site mean, and these
  deviations sum to zero over people, whether or not the scanner effect is shared. It is therefore
  reported only for the transfer setting, where the maps come from other people;
* ``per_person``: the between-scanner sum of squares within each person (scanner cell means around
  the person's mean), after harmonization relative to before, with the part that session noise alone
  produces subtracted from both ((S_p - 1) times the pooled within-cell variance of the same matrix).
  This is the scanner variance a new person keeps;
* cross-scanner identification: a session's most similar session recorded at a *different*
  scanner belongs to the same person (cosine of mean-centred edge vectors);
* a site probe trained on eight travelers and tested on the ninth (balanced accuracy, chance
  1 / number of sites) and a traveler probe trained on all scanners but one and tested on the
  held-out scanner (accuracy, chance about 1/9);
* ``offset_cosine``: cosine between the true scanner offset of each non-hub site (additive-model
  site effect minus the hub's, before harmonization) and the correction the method applied to
  that site's sessions; 1 means the correction points exactly against the scanner effect.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..evaluation.representations import PCARepresentation
from ..logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class TSMethod:
    """One harmonizer in the traveling-subject analysis."""
    name: str
    space: str                                   # 'raw' (Fisher-z edges) or 'pca'
    factory: Callable[[], object]
    fit_kw: Callable[[np.ndarray], dict] = field(default=lambda idx: {})   # training rows -> fit keywords
    transform_kw: dict = field(default_factory=dict)                      # keywords for the travelers
    reference: bool = False


# -- metrics -------------------------------------------------------------------------------------

def _dummies(labels: np.ndarray, drop_first: bool) -> np.ndarray:
    levels = np.unique(labels)
    D = (labels[:, None] == levels[None, :]).astype(float)
    return D[:, 1:] if drop_first else D


def _rss(design: np.ndarray, X: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(design, X, rcond=None)
    return np.sum((X - design @ coef) ** 2, axis=0)


def variance_shares(X: np.ndarray, site: np.ndarray, traveler: np.ndarray) -> dict:
    """Type-II sums of squares of the additive model edge ~ site + traveler, summed over edges."""
    X = np.asarray(X, dtype=np.float64)
    one = np.ones((X.shape[0], 1))
    S, T = _dummies(site, True), _dummies(traveler, True)
    rss_full = _rss(np.hstack([one, S, T]), X)
    ss_site = float(np.sum(_rss(np.hstack([one, T]), X) - rss_full))
    ss_trav = float(np.sum(_rss(np.hstack([one, S]), X) - rss_full))
    ss_tot = float(np.sum((X - X.mean(axis=0)) ** 2))
    return {"ss_total": ss_tot, "ss_site": ss_site, "ss_traveler": ss_trav, "ss_residual": float(np.sum(rss_full)),
            "site_share": ss_site / ss_tot, "traveler_share": ss_trav / ss_tot,
            "residual_share": float(np.sum(rss_full)) / ss_tot}


def _within_cell_trace(X: np.ndarray, site: np.ndarray, traveler: np.ndarray) -> float:
    """Pooled within (person x scanner) cell variance, summed over features."""
    tot, df = 0.0, 0
    for p in np.unique(traveler):
        for sv in np.unique(site):
            m = (traveler == p) & (site == sv)
            if m.sum() >= 2:
                D = X[m] - X[m].mean(axis=0)
                tot += float(np.sum(D * D))
                df += int(m.sum() - 1)
    return tot / df if df else float("nan")


def per_person_scanner_ss(X: np.ndarray, site: np.ndarray, traveler: np.ndarray) -> tuple[float, float]:
    """Sum over people of the between-scanner SS of their cell means, and the part session noise explains."""
    X = np.asarray(X, dtype=np.float64)
    tr = _within_cell_trace(X, site, traveler)
    ss = floor = 0.0
    for p in np.unique(traveler):
        mp = traveler == p
        sp = np.unique(site[mp])
        n = np.array([np.sum(mp & (site == sv)) for sv in sp], float)
        mu = np.stack([X[mp & (site == sv)].mean(axis=0) for sv in sp])
        g = np.average(mu, axis=0, weights=n)
        ss += float(np.sum(n[:, None] * (mu - g) ** 2))
        floor += (len(sp) - 1) * tr
    return ss, floor


def per_person_scanner_left(X_before: np.ndarray, X_after: np.ndarray, site: np.ndarray,
                            traveler: np.ndarray) -> dict:
    """Between-scanner variance a person keeps after harmonization, beyond session noise."""
    b, fb = per_person_scanner_ss(X_before, site, traveler)
    a, fa = per_person_scanner_ss(X_after, site, traveler)
    return {"left": (a - fa) / (b - fb) if b - fb > 0 else float("nan"),
            "raw_ratio": a / b if b > 0 else float("nan"),
            "noise_share_before": fb / b if b > 0 else float("nan")}


def interaction_anova(X: np.ndarray, site: np.ndarray, traveler: np.ndarray) -> dict:
    """
    Two-way ANOVA with interaction, summed over features: shares of the total sum of squares for the
    scanner (type II), the person (type II), the person x scanner interaction and the within-cell
    remainder; the interaction F statistic; and method-of-moments variance components, from which
    ``interaction_vs_site`` is the share of a new person's expected between-scanner variance that is
    not shared with other people.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    one = np.ones((n, 1))
    S, T = _dummies(site, True), _dummies(traveler, True)
    cell = traveler.astype(np.int64) * 10_000 + site.astype(np.int64)
    cu, cnt = np.unique(cell, return_counts=True)
    Cd = (cell[:, None] == cu[None, :]).astype(float)
    tot = float(np.sum((X - X.mean(axis=0)) ** 2))
    r_add = float(np.sum(_rss(np.hstack([one, S, T]), X)))
    r_cell = float(np.sum(_rss(Cd, X)))
    ss_site = float(np.sum(_rss(np.hstack([one, T]), X))) - r_add
    ss_trav = float(np.sum(_rss(np.hstack([one, S]), X))) - r_add
    df_int = Cd.shape[1] - (1 + S.shape[1] + T.shape[1])
    df_w = n - Cd.shape[1]
    ms_int, ms_w = (r_add - r_cell) / df_int, r_cell / df_w
    ms_site = ss_site / S.shape[1]
    n0 = (n - np.sum(cnt.astype(float) ** 2) / n) / (len(cu) - 1)     # effective cell size
    var_int = max(ms_int - ms_w, 0.0) / n0
    var_site = max(ms_site - ms_int, 0.0) / (n / (S.shape[1] + 1))
    return {"site_share": ss_site / tot, "traveler_share": ss_trav / tot,
            "interaction_share": (r_add - r_cell) / tot, "within_share": r_cell / tot,
            "F_interaction": ms_int / ms_w, "df_interaction": int(df_int), "df_within": int(df_w),
            "interaction_vs_site": var_int / (var_int + var_site) if var_int + var_site > 0 else float("nan")}


def site_effects(X: np.ndarray, site: np.ndarray, traveler: np.ndarray) -> dict[int, np.ndarray]:
    """Site coefficients of the additive model (traveler-adjusted site means up to a common constant)."""
    X = np.asarray(X, dtype=np.float64)
    sites = np.unique(site)
    design = np.hstack([_dummies(site, False), _dummies(traveler, True)])
    coef, *_ = np.linalg.lstsq(design, X, rcond=None)
    return {int(s): coef[i] for i, s in enumerate(sites)}


def cross_site_identification(X: np.ndarray, site: np.ndarray, traveler: np.ndarray) -> dict:
    """Rank-1 identification of a person across scanners (cosine of mean-centred vectors)."""
    Xc = np.asarray(X, dtype=np.float64) - np.mean(X, axis=0)
    sim = 1.0 - cdist(Xc, Xc, metric="cosine")
    other = site[:, None] != site[None, :]
    sim = np.where(other, sim, -np.inf)
    nn = np.argmax(sim, axis=1)
    hit = traveler[nn] == traveler
    same = (traveler[:, None] == traveler[None, :]) & other
    chance = same.sum(axis=1) / other.sum(axis=1)
    return {"rank1": float(hit.mean()), "chance": float(chance.mean())}


def _lr() -> Pipeline:
    """
    Linear probe on standardized edges. A ridge classifier has a closed-form solution, which matters
    here: with 35,778 edges and a few hundred sessions an iterative logistic regression can need
    thousands of iterations per fit on some harmonized outputs.
    """
    return Pipeline([("scaler", StandardScaler()), ("clf", RidgeClassifier(alpha=1.0))])


def site_probe(X: np.ndarray, site: np.ndarray, traveler: np.ndarray) -> dict:
    """Train on eight travelers, predict the scanner of the ninth's sessions."""
    pred = np.empty_like(site)
    for t in np.unique(traveler):
        te = traveler == t
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pred[te] = _lr().fit(X[~te], site[~te]).predict(X[te])
    return {"bacc": float(balanced_accuracy_score(site, pred)), "acc": float(np.mean(pred == site)),
            "chance_bacc": 1.0 / np.unique(site).size}


def traveler_probe(X: np.ndarray, site: np.ndarray, traveler: np.ndarray) -> dict:
    """Train on all scanners but one, predict the person at the held-out scanner."""
    pred = np.empty_like(traveler)
    for s in np.unique(site):
        te = site == s
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pred[te] = _lr().fit(X[~te], traveler[~te]).predict(X[te])
    return {"acc": float(np.mean(pred == traveler)), "chance": 1.0 / np.unique(traveler).size}


def offset_cosine(X_before: np.ndarray, X_after: np.ndarray, site: np.ndarray, traveler: np.ndarray,
                  hub: int) -> dict:
    """Cosine between each site's true scanner offset and the correction applied to it."""
    pre, post = site_effects(X_before, site, traveler), site_effects(X_after, site, traveler)
    cos, left, size = {}, {}, {}
    for s in pre:
        if s == hub:
            continue
        bias = pre[s] - pre[hub]
        moved = (post[s] - post[hub]) - bias                  # correction applied, relative to the hub
        nb, nm = np.linalg.norm(bias), np.linalg.norm(moved)
        cos[s] = float(-moved @ bias / (nb * nm)) if nb > 0 and nm > 0 else float("nan")
        left[s] = float(np.linalg.norm(post[s] - post[hub]) / nb) if nb > 0 else float("nan")
        size[s] = float(nm / nb) if nb > 0 else float("nan")   # length of the correction / true offset

    def _mean(d):
        v = np.array(list(d.values()), float)
        return float(np.nanmean(v)) if v.size and np.isfinite(v).any() else float("nan")

    left_sq = {k: v * v for k, v in left.items()}
    return {"by_site": cos, "mean": _mean(cos), "offset_left_by_site": left, "offset_left_mean": _mean(left),
            "offset_left_sq_by_site": left_sq, "offset_left_sq_mean": _mean(left_sq),
            "correction_size_by_site": size, "correction_size_mean": _mean(size)}


def evaluate_matrix(X: np.ndarray, site: np.ndarray, traveler: np.ndarray, *, X_reference: np.ndarray | None,
                    hub: int, probes: bool = True) -> dict:
    t0 = time.perf_counter()
    out = {"shares": variance_shares(X, site, traveler),
           "identification": cross_site_identification(X, site, traveler)}
    if X_reference is not None:
        ref = variance_shares(X_reference, site, traveler)
        out["scanner_ss_left"] = out["shares"]["ss_site"] / ref["ss_site"]
        out["traveler_ss_kept"] = out["shares"]["ss_traveler"] / ref["ss_traveler"]
        out["offset"] = offset_cosine(X_reference, X, site, traveler, hub)
        out["per_person"] = per_person_scanner_left(X_reference, X, site, traveler)
    if probes:
        out["site_probe"] = site_probe(X, site, traveler)
        out["traveler_probe"] = traveler_probe(X, site, traveler)
    out["seconds"] = time.perf_counter() - t0
    return out


# -- the two settings ----------------------------------------------------------------------------

def _apply(method: TSMethod, Z_fit, s_fit, fit_idx, Z_new, s_new):
    h = method.factory()
    h.fit(Z_fit, s_fit, **method.fit_kw(fit_idx))
    out = h.transform(Z_new, s_new, **method.transform_kw)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError(f"{method.name}: non-finite output")
    return out


def run_within_travelers(X: np.ndarray, site: np.ndarray, traveler: np.ndarray, methods: list[TSMethod], *,
                         hub: int, pca_dim: int = 100, seed: int = 42, probes: bool = True,
                         rep: PCARepresentation | None = None) -> dict:
    """
    Leave one traveler out; harmonizers see site labels of the other travelers only.

    ``rep``: a representation fitted on other people (the clinical sample of the same release).
    Without it, scaler and PCA are re-fitted on the eight training travelers in every fold. With
    so few people, the held-out person's projection then differs systematically from the training
    people's, and every PCA-space correction over-corrects by the training people's own
    person-by-scanner effects (about a third of the scanner variance is left by a plain mean shift,
    against under 1% with a fixed basis). A basis learned from independent people avoids this.
    """
    X = np.asarray(X, dtype=np.float64)
    pooled = {m.name: np.full_like(X, np.nan) for m in methods}
    recon = np.full_like(X, np.nan)
    fixed = rep
    for t in np.unique(traveler):
        tr, te = traveler != t, traveler == t
        rep = fixed if fixed is not None else PCARepresentation.fit(X[tr], pca_dim, seed=seed)
        Z = rep.transform(X)
        recon[te] = rep.inverse(Z[te])
        for m in methods:
            if m.reference:
                pooled[m.name][te] = X[te] if m.space == "raw" else rep.inverse(Z[te])
                continue
            if m.space == "raw":
                pooled[m.name][te] = _apply(m, X[tr], site[tr], np.flatnonzero(tr), X[te], site[te])
            else:
                pooled[m.name][te] = rep.inverse(_apply(m, Z[tr], site[tr], np.flatnonzero(tr), Z[te], site[te]))
        log.info("within travelers: fold %d/%d done", int(t) + 1, np.unique(traveler).size)
    results = {}
    anova = {"raw": interaction_anova(X, site, traveler), "pca": interaction_anova(recon, site, traveler)}
    log.info("within: interaction share %.3f vs scanner share %.3f (PCA reconstruction), F = %.2f",
             anova["pca"]["interaction_share"], anova["pca"]["site_share"], anova["pca"]["F_interaction"])
    for m in methods:
        ref = X if m.space == "raw" else recon
        results[m.name] = {"space": m.space, "reference": m.reference,
                           **evaluate_matrix(pooled[m.name], site, traveler, X_reference=ref, hub=hub, probes=probes)}
        log.info("within %-30s site share %.3f traveler share %.3f ident %.3f scanner SS left %.2f", m.name,
                 results[m.name]["shares"]["site_share"], results[m.name]["shares"]["traveler_share"],
                 results[m.name]["identification"]["rank1"], results[m.name].get("scanner_ss_left", np.nan))
    return {"setting": "leave-one-traveler-out", "n_sessions": int(len(site)), "n_sites": int(np.unique(site).size),
            "n_travelers": int(np.unique(traveler).size), "hub": int(hub), "pca_dim": pca_dim,
            "representation": "fixed (fitted on independent people)" if fixed is not None else "re-fitted per fold",
            "anova": anova, "methods": results}


def run_transfer(X_clin: np.ndarray, site_clin: np.ndarray, X_ts: np.ndarray, site_ts: np.ndarray,
                 traveler: np.ndarray, methods: list[TSMethod], *, hub: int, pca_dim: int = 100, seed: int = 42,
                 probes: bool = True) -> dict:
    """Fit on the clinical sample; transform the traveling subjects (sites already mapped to clinical ids)."""
    X_clin = np.asarray(X_clin, dtype=np.float64)
    X_ts = np.asarray(X_ts, dtype=np.float64)
    rep = PCARepresentation.fit(X_clin, pca_dim, seed=seed)
    Z_clin, Z_ts = rep.transform(X_clin), rep.transform(X_ts)
    recon = rep.inverse(Z_ts)
    idx = np.arange(len(site_clin))
    results = {}
    anova = {"raw": interaction_anova(X_ts, site_ts, traveler), "pca": interaction_anova(recon, site_ts, traveler)}
    for m in methods:
        if m.reference:
            out = X_ts if m.space == "raw" else recon
        elif m.space == "raw":
            out = _apply(m, X_clin, site_clin, idx, X_ts, site_ts)
        else:
            out = rep.inverse(_apply(m, Z_clin, site_clin, idx, Z_ts, site_ts))
        ref = X_ts if m.space == "raw" else recon
        results[m.name] = {"space": m.space, "reference": m.reference,
                           **evaluate_matrix(out, site_ts, traveler, X_reference=ref, hub=hub, probes=probes)}
        log.info("transfer %-30s site share %.3f traveler share %.3f ident %.3f scanner SS left %.2f cos %.2f",
                 m.name, results[m.name]["shares"]["site_share"], results[m.name]["shares"]["traveler_share"],
                 results[m.name]["identification"]["rank1"], results[m.name].get("scanner_ss_left", np.nan),
                 results[m.name].get("offset", {}).get("mean", np.nan))
    return {"setting": "fit on clinical sample, apply to traveling subjects", "n_clinical": int(len(site_clin)),
            "n_sessions": int(len(site_ts)), "n_sites": int(np.unique(site_ts).size),
            "n_travelers": int(np.unique(traveler).size), "hub": int(hub), "pca_dim": pca_dim,
            "pca_variance": float(rep.pca.explained_variance_ratio_.sum()), "anova": anova, "methods": results}
