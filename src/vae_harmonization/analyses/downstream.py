"""
Downstream pay-off analyses: does harmonization help anything beyond the probes?

1. Leave-one-site-out (LOSO) diagnosis classification for every method
   ------------------------------------------------------------------------
   For each held-out site the harmonizer is fitted *transductively*: it sees the
   FC/latent vectors of all sites (a deployed harmonizer needs the new site's
   unlabelled scans to estimate its moments) but the diagnosis labels of the
   held-out site are masked (``-1``).  The probe classifier is trained on the
   remaining sites and tested on the held-out site.  Methods whose transform
   needs diagnosis labels of the adjusted subjects (ComBat/CovBat with covariate
   protection) receive the true labels and are flagged ``uses_test_labels``.

2. Case-control effect-size reproducibility across disjoint site halves
   ---------------------------------------------------------------------
   Sites are randomly split into two disjoint groups; per-feature Cohen's d for
   diagnosis is computed in each half; the Pearson correlation and sign
   agreement of the two d-vectors measures how reproducible the case-control
   pattern is.  Site effects that are confounded with diagnosis lower this
   number; harmonization should raise it without inflating |d|.

3. Motion (mean FD) leakage
   ------------------------
   Cross-validated R^2 of predicting mean framewise displacement from the
   harmonized features, plus the site eta^2 of FD itself.  Harmonization that
   removes site effects should not *increase* the motion signal, and a
   diagnosis probe should not gain accuracy through motion.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..evaluation.classifiers import fit_and_score
from ..logging_utils import get_logger
from ..metrics import compute_eta_squared
from ..stats import paired_test, summarize_folds

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. LOSO
# ---------------------------------------------------------------------------

def _fit_transform_transductive(spec, X, y_site, y_diag, test_mask, target_site, severity=None, covariates=None):
    """Fit ``spec.factory()`` on all X with test-site labels masked; return harmonized X and flags."""
    y_fit = np.asarray(y_diag).astype(int).copy()
    uses_test_labels = bool(getattr(spec, "transform_uses_labels", False))
    if not uses_test_labels:
        y_fit[test_mask] = -1
    h = spec.factory()
    aux = {}
    if "severity" in spec.fit_aux and severity is not None:
        aux["severity"] = severity
    if "covariates" in spec.fit_aux and covariates:
        aux["covariates"] = covariates
    # our harmonizer needs its hub inside the data: if the held-out site is the hub, use the largest other site
    held_out = int(np.unique(y_site[test_mask])[0])
    if getattr(h, "hub", None) == "target" and int(getattr(h, "target_site", -1)) == held_out:
        sites, counts = np.unique(y_site[~test_mask], return_counts=True)
        h.target_site = int(sites[np.argmax(counts)])
    kw = {"y_diag": y_fit} if "y_diag" in spec.fit_aux else {}
    # covariates go to transform as well: label-free ComBat adds the age and sex effects back there
    t_aux = {k: v for k, v in aux.items() if k == "covariates"}
    Xh = h.fit(X, y_site, **kw, **aux).transform(X, y_site, **t_aux) if not uses_test_labels else \
        h.fit(X, y_site, y_diag=y_fit, **aux).transform(X, y_site, y_diag=y_fit, **t_aux)
    return Xh, uses_test_labels


def loso_all_methods(representations: dict[str, np.ndarray], y_site: np.ndarray, y_diag: np.ndarray, methods,
                     *, target_site: int, classifiers=("lr",), seed: int = 42, min_test_size: int = 10,
                     site_names: dict | None = None, severity=None, covariates=None) -> dict:
    """Leave-one-site-out diagnosis accuracy for every MethodSpec (see module docstring)."""
    y_site = np.asarray(y_site).astype(int)
    y_diag = np.asarray(y_diag).astype(int)
    sites, counts = np.unique(y_site, return_counts=True)
    eligible = [int(s) for s, c in zip(sites, counts)
                if c >= min_test_size and np.unique(y_diag[y_site == s]).size == 2]
    out = {"protocol": "transductive harmonizer fit (held-out site unlabelled), probe trained on other sites",
           "min_test_size": min_test_size, "sites": [], "methods": {}, "comparisons": {}}
    per = {m.name: {c: {"acc": [], "bacc": [], "auc": []} for c in classifiers} for m in methods}
    flags = {m.name: False for m in methods}
    for s in eligible:
        test = y_site == s
        out["sites"].append({"site": s, "name": (site_names or {}).get(s, str(s)), "n_test": int(test.sum()),
                             "case_rate": float(y_diag[test].mean())})
        for m in methods:
            X = np.asarray(representations[m.representation], dtype=np.float64)
            try:
                Xh, used = _fit_transform_transductive(m, X, y_site, y_diag, test, target_site, severity, covariates)
            except Exception as exc:  # small held-out sites can break individual baselines
                log.warning("LOSO site %s method %s failed: %s", s, m.name, exc)
                for c in classifiers:
                    for k in per[m.name][c]:
                        per[m.name][c][k].append(float("nan"))
                continue
            flags[m.name] |= used
            for c in classifiers:
                r = fit_and_score(c, Xh[~test], y_diag[~test], Xh[test], y_diag[test], task="diag", seed=seed)
                for k in per[m.name][c]:
                    per[m.name][c][k].append(float(r.get(k, np.nan)))
    # every method is compared with the un-harmonized reference of its own representation
    # (Raw FC for raw-space methods, PCA scores / VAE latent for methods in that space); when a
    # representation has no reference in the method list, the first reference method is used
    ref_by_rep = {m.representation: m.name for m in methods if getattr(m, "is_reference", False)}
    fallback = next(iter(ref_by_rep.values()), methods[0].name)
    for m in methods:
        out["methods"][m.name] = {
            "representation": m.representation,
            "reference": ref_by_rep.get(m.representation, fallback),
            "uses_test_labels": flags[m.name],
            "per_site": {c: per[m.name][c] for c in classifiers},
            "summary": {c: {k: summarize_folds(v) for k, v in per[m.name][c].items()} for c in classifiers},
        }
    weights = np.array([d["n_test"] for d in out["sites"]], float)
    for m in methods:
        for c in classifiers:
            a = np.array(per[m.name][c]["acc"], float)
            ok = np.isfinite(a)
            out["methods"][m.name]["summary"][c]["acc_weighted"] = float(np.average(a[ok], weights=weights[ok])) if ok.any() else float("nan")
    for c in classifiers:
        comps = {}
        for m in methods:
            ref = ref_by_rep.get(m.representation, fallback)
            if m.name == ref:
                continue
            a = np.array(per[m.name][c]["acc"], float)
            b = np.array(per[ref][c]["acc"], float)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 3:
                comps[m.name] = {"vs": ref, **paired_test(a[ok], b[ok])}
        out["comparisons"][c] = comps
    return out


# ---------------------------------------------------------------------------
# 2. Effect-size reproducibility
# ---------------------------------------------------------------------------

def cohens_d_per_feature(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    a, b = X[y == 1], X[y == 0]
    n1, n0 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(axis=0, ddof=1) + (n0 - 1) * b.var(axis=0, ddof=1)) / max(n1 + n0 - 2, 1))
    return (a.mean(axis=0) - b.mean(axis=0)) / (sp + 1e-12)


def _balanced_site_split(sites, counts, rng):
    """Greedy split of sites into two groups with similar total n."""
    order = rng.permutation(len(sites))
    g = [[], []]
    tot = [0, 0]
    for i in order:
        k = int(tot[1] < tot[0])
        g[k].append(int(sites[i]))
        tot[k] += counts[i]
    return g[0], g[1]


def effect_size_reproducibility(X: np.ndarray, y_site: np.ndarray, y_diag: np.ndarray, n_splits: int = 20,
                                seed: int = 42, min_group_cases: int = 10) -> dict:
    """Correlation of per-feature Cohen's d between two disjoint site halves (mean over random splits)."""
    rng = np.random.default_rng(seed)
    y_site = np.asarray(y_site).astype(int)
    y_diag = np.asarray(y_diag).astype(int)
    sites, counts = np.unique(y_site, return_counts=True)
    rs, signs, mags = [], [], []
    for _ in range(n_splits):
        A, B = _balanced_site_split(sites, counts, rng)
        mA, mB = np.isin(y_site, A), np.isin(y_site, B)
        if min((y_diag[mA] == 1).sum(), (y_diag[mA] == 0).sum(), (y_diag[mB] == 1).sum(), (y_diag[mB] == 0).sum()) < min_group_cases:
            continue
        dA, dB = cohens_d_per_feature(X[mA], y_diag[mA]), cohens_d_per_feature(X[mB], y_diag[mB])
        rs.append(float(np.corrcoef(dA, dB)[0, 1]))
        signs.append(float(np.mean(np.sign(dA) == np.sign(dB))))
        mags.append(float(0.5 * (np.abs(dA).mean() + np.abs(dB).mean())))
    return {"r": summarize_folds(rs), "sign_agreement": summarize_folds(signs), "mean_abs_d": summarize_folds(mags),
            "n_splits": len(rs)}


def effect_size_reproducibility_crossfit(spec, X: np.ndarray, y_site: np.ndarray, y_diag: np.ndarray, *,
                                         n_splits: int = 20, seed: int = 42, min_group_cases: int = 10,
                                         severity=None, covariates=None) -> dict:
    """
    Effect-size reproducibility in which the harmonizer never sees the diagnoses of the half it is scored on.

    The sites are split as in :func:`effect_size_reproducibility` (same random sequence). For a method that
    uses diagnoses (or severity scores) only when it is fitted, the harmonizer is fitted twice per split:
    once with the labels of half A hidden (``-1``; severity set to NaN), and its output on half A gives
    ``d_A``; once with half B hidden, giving ``d_B``. A label-free method is fitted once. A method whose
    transform needs every subject's diagnosis (ComBat and its relatives) cannot hide them; it is fitted
    once with all labels and marked ``labels: "all"``.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    y_site = np.asarray(y_site).astype(int)
    y_diag = np.asarray(y_diag).astype(int)
    sites, counts = np.unique(y_site, return_counts=True)
    needs_labels_at_transform = bool(getattr(spec, "transform_uses_labels", False))
    fit_labels = ("y_diag" in spec.fit_aux) or ("severity" in spec.fit_aux)
    mode = "all" if needs_labels_at_transform else ("hidden for the scored half" if fit_labels else "none")

    def fit_transform(hide=None):
        y_fit = y_diag.copy()
        sev = None if severity is None else np.asarray(severity, dtype=float).copy()
        if hide is not None and not needs_labels_at_transform:
            y_fit[hide] = -1
            if sev is not None:
                sev[hide] = np.nan
        h = spec.factory()
        aux = {}
        if "severity" in spec.fit_aux and sev is not None:
            aux["severity"] = sev
        if "covariates" in spec.fit_aux and covariates:
            aux["covariates"] = covariates
        kw = {"y_diag": y_fit} if "y_diag" in spec.fit_aux else {}
        h.fit(X, y_site, **kw, **aux)
        t_aux = {k: v for k, v in aux.items() if k == "covariates"}
        if needs_labels_at_transform:
            return h.transform(X, y_site, y_diag=y_diag, **t_aux)
        return h.transform(X, y_site, **t_aux)

    single = None if fit_labels and not needs_labels_at_transform else fit_transform()
    rs, signs, mags = [], [], []
    for _ in range(n_splits):
        A, B = _balanced_site_split(sites, counts, rng)
        mA, mB = np.isin(y_site, A), np.isin(y_site, B)
        if min((y_diag[mA] == 1).sum(), (y_diag[mA] == 0).sum(), (y_diag[mB] == 1).sum(), (y_diag[mB] == 0).sum()) < min_group_cases:
            continue
        if single is not None:
            XA = XB = single
        else:
            XA, XB = fit_transform(hide=mA), fit_transform(hide=mB)
        dA, dB = cohens_d_per_feature(XA[mA], y_diag[mA]), cohens_d_per_feature(XB[mB], y_diag[mB])
        rs.append(float(np.corrcoef(dA, dB)[0, 1]))
        signs.append(float(np.mean(np.sign(dA) == np.sign(dB))))
        mags.append(float(0.5 * (np.abs(dA).mean() + np.abs(dB).mean())))
    return {"r": summarize_folds(rs), "sign_agreement": summarize_folds(signs), "mean_abs_d": summarize_folds(mags),
            "n_splits": len(rs), "labels": mode, "representation": spec.representation}


# ---------------------------------------------------------------------------
# 3. Motion leakage
# ---------------------------------------------------------------------------

def motion_leakage(X: np.ndarray, fd: np.ndarray, y_site: np.ndarray, n_folds: int = 5, seed: int = 42) -> dict:
    """CV R^2 of ridge-predicting mean FD from features; site eta^2 of FD for context."""
    fd = np.asarray(fd, dtype=np.float64)
    ok = np.isfinite(fd)
    if ok.sum() < 30:
        return {"r2": float("nan"), "n": int(ok.sum())}
    X, fd, y_site = np.asarray(X, dtype=np.float64)[ok], fd[ok], np.asarray(y_site)[ok]
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-2, 4, 13)))])
    r2s = []
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X[tr], fd[tr])
        pred = pipe.predict(X[te])
        ss_res = np.sum((fd[te] - pred) ** 2)
        ss_tot = np.sum((fd[te] - fd[te].mean()) ** 2)
        r2s.append(1.0 - ss_res / max(ss_tot, 1e-12))
    return {"r2": summarize_folds(r2s), "fd_site_eta2": compute_eta_squared(fd, y_site), "n": int(ok.sum())}


def run_downstream(representations, y_site, y_diag, methods, *, target_site, site_names=None, classifiers=("lr",),
                   seed=42, severity=None, covariates=None, fd=None, n_es_splits=20) -> dict:
    """LOSO for all methods + effect-size reproducibility + motion leakage for full-data harmonized outputs."""
    out = {"loso": loso_all_methods(representations, y_site, y_diag, methods, target_site=target_site,
                                    classifiers=classifiers, seed=seed, site_names=site_names,
                                    severity=severity, covariates=covariates),
           "effect_size_reproducibility": {}, "motion_leakage": {}}
    y_diag = np.asarray(y_diag).astype(int)
    for m in methods:
        X = np.asarray(representations[m.representation], dtype=np.float64)
        h = m.factory()
        aux = {}
        if "severity" in m.fit_aux and severity is not None:
            aux["severity"] = severity
        if "covariates" in m.fit_aux and covariates:
            aux["covariates"] = covariates
        kw = {"y_diag": y_diag} if "y_diag" in m.fit_aux else {}
        t_aux = {k: v for k, v in aux.items() if k == "covariates"}
        try:
            h.fit(X, y_site, **kw, **aux)
            Xh = h.transform(X, y_site, y_diag=y_diag, **t_aux) if getattr(m, "transform_uses_labels", False) \
                else h.transform(X, y_site, **t_aux)
        except Exception as exc:
            log.warning("downstream: %s failed on full data: %s", m.name, exc)
            continue
        out["effect_size_reproducibility"][m.name] = effect_size_reproducibility(Xh, y_site, y_diag, n_es_splits, seed)
        if fd is not None:
            out["motion_leakage"][m.name] = motion_leakage(Xh, fd, y_site, seed=seed)
    return out
