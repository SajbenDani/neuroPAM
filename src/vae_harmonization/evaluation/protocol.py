"""
protocol.py -- The single leakage-free evaluation protocol.

For each of ``n_folds`` stratified folds and for every method:

1. ``fit`` the harmonizer on the training fold only (labels of training
   subjects only, except ComBat/CovBat which by definition also consume the
   diagnosis covariate of the subjects they adjust);
2. ``transform`` the training and the held-out fold with the frozen map;
3. fit the probe classifiers (diagnosis, site) on the harmonized training
   fold and score them on the harmonized held-out fold;
4. keep the held-out harmonized rows -> an out-of-sample (OOS) matrix in
   which every subject was transformed by a model that never saw it.

Distribution metrics (ICC, median-heuristic MMD with one bandwidth per
representation, eta^2 of the protected coordinate) are computed on the OOS
matrices / held-out folds, so they are not inflated by in-sample moment
matching.  Paired statistics across folds (Wilcoxon, d_z, TOST, Holm) are
attached under ``comparisons``.

The result is one JSON-serialisable dict per cohort; tables in the paper are
generated from these files only.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ..harmonization.base import Harmonizer
from ..logging_utils import get_logger
from ..metrics import (
    compute_epsilon_squared,
    compute_eta_squared,
    compute_icc,
    eta_squared_null_expectation,
    compute_site_pairwise_mmd,
    gamma_from_sigma,
    median_heuristic_bandwidth,
    permutation_null_oos,
)
from ..stats import (
    chance_levels,
    holm_bonferroni,
    nadeau_bengio_corrected_ttest,
    paired_test,
    summarize_folds,
    tost_paired,
)
from .classifiers import covariate_probe, fit_and_score

log = get_logger(__name__)

TASKS = ("diag", "site")
METRICS_BY_TASK = {"diag": ("acc", "bacc", "f1_macro", "auc"), "site": ("acc", "bacc", "f1_macro", "nmi")}


@dataclass
class MethodSpec:
    name: str
    representation: str
    factory: Callable[[], Harmonizer]
    fit_aux: tuple[str, ...] = ("y_diag",)          # subset of {'y_diag','severity','covariates'}
    transform_uses_labels: bool = False              # ComBat/CovBat: pass y_diag to transform
    is_reference: bool = False                       # the 'Raw' method of its representation
    is_proposed: bool = False                        # the method other methods are compared to
    notes: str = ""
    tags: dict = field(default_factory=dict)


def _make_folds(y_site, y_diag, n_folds, seed):
    """
    Stratify on site x diagnosis when every cell has >= n_folds members; otherwise on site
    (keeps every site represented in every training fold as long as n_s >= 2, so that
    per-site harmonizers can be fitted); otherwise on diagnosis alone.
    """
    y_site = y_site.astype(int)
    y_diag = y_diag.astype(int)
    combined = y_site * 2 + y_diag
    _, ccounts = np.unique(combined, return_counts=True)
    _, scounts = np.unique(y_site, return_counts=True)
    _, dcounts = np.unique(y_diag, return_counts=True)
    n_folds = int(min(n_folds, dcounts.min()))
    if n_folds < 2:
        raise ValueError("not enough samples per diagnosis class for cross-validation")
    if ccounts.min() >= n_folds:
        strat, mode = combined, "site_x_diag"
    elif scounts.min() >= 2:
        strat, mode = y_site, "site"
    else:
        strat, mode = y_diag, "diag"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # "least populated class has fewer members than n_splits"
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = list(kf.split(np.zeros(len(y_diag)), strat))
    return folds, n_folds, mode


def _nan_summary(n):
    return [float("nan")] * n


def _call_representations(fn, fold, tr, te):
    """``fn`` returns either ``{name: X}`` or ``({name: X}, {name: embed})``."""
    out = fn(fold, tr, te)
    if isinstance(out, tuple):
        reps, embed = out
    else:
        reps, embed = out, {}
    return reps, dict(embed or {})


def run_protocol(
    representations,
    y_site: np.ndarray,
    y_diag: np.ndarray,
    methods: list[MethodSpec],
    *,
    n_folds: int = 10,
    seed: int = 42,
    classifiers: tuple[str, ...] = ("lr", "rf"),
    severity: np.ndarray | None = None,
    covariates: dict[str, np.ndarray] | None = None,
    site_names: dict[int, str] | None = None,
    mmd_min_site: int = 5,
    mmd_bootstrap: int = 0,
    n_perm: int = 0,
    n_repeats: int = 1,
    tost_margin: float = 0.02,
    return_oos: bool = True,
    oos_hook: Callable[[str, str, np.ndarray], None] | None = None,
) -> dict:
    """
    Parameters
    ----------
    return_oos : keep every method's pooled out-of-sample matrix under ``results['_oos']``
                 (memory: N x embedded dimension per method; use ``oos_hook`` for large cohorts).
    oos_hook   : ``f(method_name, representation, X_oos)`` called once per method with the pooled
                 out-of-sample matrix before it is discarded.
    representations : {name: (N, p) array}  -- fixed representations (raw FC, or latents of a
                      model fitted once, label-free, on all subjects), or a callable
                      ``f(fold_index, train_idx, test_idx) -> (reps, embed)`` that re-fits the
                      representation (scaler + PCA) on the training fold and returns
                      ``reps = {name: (N, p) array}`` whose rows were produced by the fold's
                      model, and ``embed = {name: g}`` with ``g`` an isometric map from the fold's
                      coordinates into a space shared by all folds (for PCA scores: the
                      reconstruction ``scores @ components + mean`` in the standardized edge space).
                      Held-out rows are pooled after embedding, so ICC / MMD are computed on one
                      out-of-sample matrix exactly as for fixed representations.  Representations
                      without an ``embed`` entry are assumed to live in a common space already.
    """
    y_site = np.asarray(y_site).astype(int)
    y_diag = np.asarray(y_diag).astype(int)
    N = len(y_site)
    per_fold_reps = callable(representations)
    if not per_fold_reps:
        for k, X in representations.items():
            if X.shape[0] != N:
                raise ValueError(f"representation '{k}' has {X.shape[0]} rows, expected {N}")
    names = [m.name for m in methods]
    if len(set(names)) != len(names):
        raise ValueError("method names must be unique")

    n_repeats = max(1, int(n_repeats))
    folds_per_repeat = []
    for r in range(n_repeats):
        fr, n_folds, strat_mode = _make_folds(y_site, y_diag, n_folds, seed + 1000 * r)
        folds_per_repeat.append(fr)
    folds = folds_per_repeat[0]          # first repeat: the pooled OOS matrix and MMD bandwidth
    log.info("protocol: %d folds x %d repeats (stratified on %s), %d methods, %d subjects, %d sites%s",
             n_folds, n_repeats, strat_mode, len(methods), N, np.unique(y_site).size,
             " [representation re-fitted per fold]" if per_fold_reps else "")
    fold_sizes = [int(len(te)) for _, te in folds]
    n_train_typ = int(np.mean([len(tr) for tr, _ in folds]))
    n_test_typ = int(np.mean(fold_sizes))
    rep_names = {m.representation for m in methods}

    # -- representations per fold (fitted on the training rows only) ---------------------
    # Held-out rows are pooled after ``embed`` into one out-of-sample matrix per method.  Methods
    # are the outer loop below so that only one such matrix (N x embedded dim) is alive at a time.
    fold_reps: list[tuple[dict, dict]] = []
    rep_info = []
    pre_oos: dict[str, np.ndarray] = {}
    for fi, (tr, te) in enumerate(folds):
        if per_fold_reps:
            t_rep = time.perf_counter()
            reps, embed = _call_representations(representations, fi, tr, te)
            rep_info.append({"fold": fi, "seconds": time.perf_counter() - t_rep,
                             **{k: list(v.shape) for k, v in reps.items()}})
        else:
            reps, embed = representations, {}
        fold_reps.append((reps, embed))
        for r in rep_names:
            if r in embed:
                if r not in pre_oos:
                    pre_oos[r] = np.full((N, embed[r](reps[r][:1]).shape[1]), np.nan)
                pre_oos[r][te] = embed[r](reps[r][te])
            else:                                  # already in a common space: the matrix itself
                pre_oos[r] = reps[r]
    # shared MMD bandwidth per representation: median heuristic on the pooled pre-harmonization
    # matrix (label-free; for fixed representations this is the full matrix itself)
    gamma_by_rep = {k: gamma_from_sigma(median_heuristic_bandwidth(X, seed=seed)) for k, X in pre_oos.items()}

    # -- per method: all folds, then the pooled distribution metrics ------------------------
    cov_names = list(covariates) if covariates else []
    results: dict = {
        "protocol": {
            "n_folds": n_folds, "seed": seed, "stratification": strat_mode, "fold_sizes": fold_sizes,
            "classifiers": list(classifiers), "n_subjects": int(N),
            "n_sites": int(np.unique(y_site).size), "mmd_min_site": mmd_min_site,
            # MMD^2 is the mean over all pairs of sites with >= mmd_min_site held-out rows (no hub involved);
            # the RBF bandwidth is the median heuristic on the pooled pre-harmonization matrix.
            "mmd_scope": "all_site_pairs",
            "mmd_bandwidth_source": "pooled_pre_harmonization_matrix",
            "mmd_gamma_by_representation": gamma_by_rep,
            "mmd_sigma_by_representation": {k: float(np.sqrt(0.5 / g)) for k, g in gamma_by_rep.items()},
            "tost_margin": tost_margin,
            "n_perm": int(n_perm),
            "n_repeats": int(n_repeats),
            "n_train_typical": n_train_typ,
            "n_test_typical": n_test_typ,
            "nested_representation": per_fold_reps,
            "representation_folds": rep_info,
            "distribution_scope": "pooled_out_of_sample",
        },
        "chance": {"site": chance_levels(y_site), "diag": chance_levels(y_diag)},
        "site_counts": {str(int(s)): int(c) for s, c in zip(*np.unique(y_site, return_counts=True))},
        "site_names": {str(k): v for k, v in (site_names or {}).items()},
        "methods": {},
        "comparisons": {},
    }
    fold_scores_all: dict[str, dict] = {}
    oos_keep: dict[str, np.ndarray] = {}

    for mi, m in enumerate(methods):
        log.info("method %d/%d %s ...", mi + 1, len(methods), m.name)
        fold_scores = {t: {c: {met: [] for met in METRICS_BY_TASK[t]} for c in classifiers} for t in TASKS}
        eta2_folds, eps2_folds, eps2_pre_folds, disp_folds = [], [], [], []
        disp_plugin_folds = []
        cov_folds = {c: [] for c in cov_names}
        fit_seconds, failures = [], []
        describe = None
        X_oos: np.ndarray | None = None
        t_method = time.perf_counter()

        all_splits = [(ri, fi, tr, te)
                      for ri, fr in enumerate(folds_per_repeat) for fi, (tr, te) in enumerate(fr)]
        for ri, fi, tr, te in all_splits:
            s_tr, s_te = y_site[tr], y_site[te]
            d_tr, d_te = y_diag[tr], y_diag[te]
            if ri == 0:
                reps, embed = fold_reps[fi]
            elif per_fold_reps:
                reps, embed = _call_representations(representations, fi, tr, te)
            else:
                reps, embed = representations, {}
            X = reps[m.representation]
            X_tr, X_te = X[tr], X[te]
            aux_fit = {}
            if "y_diag" in m.fit_aux:
                aux_fit["y_diag"] = d_tr
            if "severity" in m.fit_aux and severity is not None:
                aux_fit["severity"] = np.asarray(severity, float)[tr]
            if "covariates" in m.fit_aux and covariates is not None:
                aux_fit["covariates"] = {k: np.asarray(v, float)[tr] for k, v in covariates.items()}
            aux_tr = {"y_diag": d_tr} if m.transform_uses_labels else {}
            aux_te = {"y_diag": d_te} if m.transform_uses_labels else {}
            if "covariates" in m.fit_aux and covariates is not None:
                aux_tr["covariates"] = {k: np.asarray(v, float)[tr] for k, v in covariates.items()}
                aux_te["covariates"] = {k: np.asarray(v, float)[te] for k, v in covariates.items()}
            t0 = time.perf_counter()
            try:
                h = m.factory()
                h.fit(X_tr, s_tr, **aux_fit)
                Xtr_h = h.transform(X_tr, s_tr, **aux_tr)
                Xte_h = h.transform(X_te, s_te, **aux_te)
                if not (np.all(np.isfinite(Xtr_h)) and np.all(np.isfinite(Xte_h))):
                    raise FloatingPointError("non-finite harmonized values")
            except Exception as exc:  # record and keep going; NaNs propagate to summaries
                failures.append({"fold": fi, "error": f"{type(exc).__name__}: {exc}"})
                log.warning("fold %d, %s failed: %s", fi, m.name, exc)
                for t in TASKS:
                    for c in classifiers:
                        for met in METRICS_BY_TASK[t]:
                            fold_scores[t][c][met].append(float("nan"))
                for lst in (eta2_folds, eps2_folds, eps2_pre_folds, disp_folds, disp_plugin_folds):
                    lst.append(float("nan"))
                for cname in cov_names:
                    cov_folds[cname].append({"kind": "failed"})
                fit_seconds.append(time.perf_counter() - t0)
                continue
            fit_seconds.append(time.perf_counter() - t0)
            if describe is None:
                try:
                    describe = h.describe()
                except Exception:
                    describe = {"name": m.name}
            if ri == 0:
                emb = embed.get(m.representation, lambda A: A)(Xte_h)
                if X_oos is None:
                    X_oos = np.full((N, emb.shape[1]), np.nan)
                X_oos[te] = emb

            # probes
            for t, y_tr_t, y_te_t in (("diag", d_tr, d_te), ("site", s_tr, s_te)):
                for c in classifiers:
                    sc = fit_and_score(c, Xtr_h, y_tr_t, Xte_h, y_te_t, task=t, seed=seed)
                    for met in METRICS_BY_TASK[t]:
                        fold_scores[t][c][met].append(float(sc.get(met, float("nan"))))

            # protected-coordinate site variance on the held-out fold (our method only):
            # eta^2 (biased upward by ~(G-1)/(N-1)), bias-corrected epsilon^2, and the
            # pre-harmonization value along the same fold-specific axis
            if hasattr(h, "protected_coordinates") and getattr(h, "k_", 0) > 0:
                a_te = h.protected_coordinates(Xte_h)[:, 0]
                a_pre = h.protected_coordinates(X_te)[:, 0]
                eta2_folds.append(compute_eta_squared(a_te, s_te))
                eps2_folds.append(compute_epsilon_squared(a_te, s_te))
                eps2_pre_folds.append(compute_epsilon_squared(a_pre, s_te))
            else:
                for lst in (eta2_folds, eps2_folds, eps2_pre_folds):
                    lst.append(float("nan"))
            # mean squared displacement of held-out subjects, and (for the affine maps) the expected
            # displacement under the fitted moments, which is what Theorem 1 is about
            disp_folds.append(float(np.mean(np.sum((Xte_h - X_te) ** 2, axis=1))))
            plug = getattr(h, "plugin_displacement", None)
            disp_plugin_folds.append(float(plug()) if callable(plug) else float("nan"))
            # covariate preservation (age / sex / motion decodability on the held-out fold)
            for cname in cov_names:
                cv = np.asarray(covariates[cname], dtype=float)
                cov_folds[cname].append(covariate_probe(Xtr_h, cv[tr], Xte_h, cv[te], seed=seed))

        # pooled out-of-sample distribution metrics
        dist = {}
        ok = np.zeros(N, dtype=bool) if X_oos is None else np.all(np.isfinite(X_oos), axis=1)
        if ok.sum() >= 10 and np.unique(y_site[ok]).size >= 2:
            icc = compute_icc(X_oos[ok], y_site[ok])
            mmd = compute_site_pairwise_mmd(X_oos[ok], y_site[ok], gamma=gamma_by_rep[m.representation],
                                            min_site_size=mmd_min_site, n_boot=mmd_bootstrap, seed=seed)
            mmd_b = compute_site_pairwise_mmd(X_oos[ok], y_site[ok], gamma=gamma_by_rep[m.representation],
                                              min_site_size=mmd_min_site, unbiased=False)
            if mmd["excluded_sites"] and mi == 0:
                log.info("MMD excludes sites with < %d held-out rows: %s", mmd_min_site,
                         [(site_names or {}).get(int(s), s) for s in mmd["excluded_sites"]])
            dist = {
                "icc_mean": icc["icc_mean"],
                "mmd2_unbiased_mean": mmd["mean_pairwise_mmd"],
                "mmd2_unbiased_median": mmd["median_pairwise_mmd"],
                "mmd2_biased_mean": mmd_b["mean_pairwise_mmd"],
                "mmd_kernel_floor_pairs": mmd["kernel_floor_pairs"],
                "mmd_excluded_sites": mmd["excluded_sites"],
                "n_oos_rows": int(ok.sum()),
            }
            if mmd_bootstrap > 0:
                cis = np.array([[v["ci_low"], v["ci_high"]] for v in mmd["pairwise_ci"].values()])
                dist["mmd2_pair_ci_mean"] = [float(cis[:, 0].mean()), float(cis[:, 1].mean())]
            if n_perm > 0:
                try:
                    dist["permutation"] = permutation_null_oos(
                        X_oos[ok], y_site[ok], gamma=gamma_by_rep[m.representation],
                        min_site_size=mmd_min_site, n_perm=n_perm, seed=seed)
                except Exception as exc:
                    log.warning("permutation null failed for %s: %s", m.name, exc)
        if X_oos is not None:
            if oos_hook is not None:
                oos_hook(m.name, m.representation, X_oos)
            if return_oos:
                oos_keep[m.name] = X_oos
        del X_oos

        fold_scores_all[m.name] = fold_scores
        results["methods"][m.name] = {
            "representation": m.representation,
            "is_reference": m.is_reference,
            "is_proposed": m.is_proposed,
            "notes": m.notes,
            "describe": describe,
            "folds": fold_scores,
            "summary": {t: {c: {met: summarize_folds(fold_scores[t][c][met]) for met in METRICS_BY_TASK[t]}
                            for c in classifiers} for t in TASKS},
            "protected_eta2_folds": eta2_folds,
            "protected_eta2": summarize_folds(eta2_folds),
            "protected_eta2_null": float(np.mean([eta_squared_null_expectation(n, np.unique(y_site).size)
                                                  for n in fold_sizes])),
            "protected_eps2_folds": eps2_folds,
            "protected_eps2": summarize_folds(eps2_folds),
            "protected_eps2_pre_folds": eps2_pre_folds,
            "protected_eps2_pre": summarize_folds(eps2_pre_folds),
            "displacement_folds": disp_folds,
            "displacement": summarize_folds(disp_folds),
            "displacement_plugin_folds": disp_plugin_folds,
            "displacement_plugin": summarize_folds(disp_plugin_folds),
            "covariates": {
                cname: {
                    "folds": cov_folds[cname],
                    **{k: summarize_folds([f.get(k, float("nan")) for f in cov_folds[cname]])
                       for k in ("r", "r2", "auc", "acc")
                       if any(k in f for f in cov_folds[cname])},
                } for cname in cov_names
            },
            "oos_distribution": dist,
            "fit_seconds_mean": float(np.mean(fit_seconds)) if fit_seconds else float("nan"),
            "failures": failures,
        }
        s = results["methods"][m.name]["summary"]
        log.info("%-28s done in %5.1fs | diag acc %.3f | site acc %.3f | NMI %.3f | MMD2 %s",
                 m.name, time.perf_counter() - t_method, s["diag"][classifiers[0]]["acc"]["mean"],
                 s["site"][classifiers[0]]["acc"]["mean"], s["site"][classifiers[0]]["nmi"]["mean"],
                 f"{dist['mmd2_unbiased_mean']:.4f}" if dist else "n/a")

    # -- comparisons ------------------------------------------------------------------
    results["comparisons"] = compute_comparisons(
        fold_scores_all,
        {m.name: {"representation": m.representation, "is_reference": m.is_reference, "is_proposed": m.is_proposed}
         for m in methods},
        classifiers, tost_margin=tost_margin,
        n_train=n_train_typ if n_repeats > 1 else None,
        n_test=n_test_typ if n_repeats > 1 else None)
    if return_oos:
        results["_oos"] = oos_keep
    return results


COMPARED_METRICS = {"diag": ("acc", "bacc", "auc"), "site": ("acc", "bacc", "nmi")}


def compute_comparisons(fold_scores: dict, meta: dict, classifiers, tost_margin: float = 0.02,
                       n_train: int | None = None, n_test: int | None = None) -> dict:
    """
    Paired fold-level tests of every method against (a) the un-harmonized reference of its representation and
    (b) every proposed method.  Holm correction is applied within a family = (task, classifier, metric, kind), i.e.
    the 'each method vs. its reference' tests are corrected together and the 'vs. proposed' tests separately.
    Works from the ``folds`` entries stored in protocol_results.json, so it can be recomputed offline.

    With repeated CV (``n_train`` and ``n_test`` given) the folds overlap, so the Nadeau-Bengio test is the
    primary test: Holm uses its p-value and TOST uses its variance factor.
    """
    by_rep_ref = {v["representation"]: k for k, v in meta.items() if v["is_reference"]}
    proposed = [k for k, v in meta.items() if v["is_proposed"]]
    repeated = n_train is not None and n_test is not None
    comps: dict = {}
    for t in TASKS:
        for c in classifiers:
            for metric in COMPARED_METRICS[t]:
                entries: dict = {}
                fam: dict[str, list] = {"vs_reference": [], "vs_proposed": []}
                for name in meta:
                    if c not in fold_scores[name][t]:        # merged run probed with fewer classifiers
                        continue
                    a = np.array(fold_scores[name][t][c][metric], float)
                    targets = []
                    ref = by_rep_ref.get(meta[name]["representation"])
                    if ref and ref != name:
                        targets.append(("vs_reference", ref))
                    for p in proposed:
                        if p != name:
                            targets.append(("vs_proposed", p))
                    for kind, other in targets:
                        if c not in fold_scores[other][t]:
                            continue
                        b = np.array(fold_scores[other][t][c][metric], float)
                        ok = np.isfinite(a) & np.isfinite(b)
                        if ok.sum() < 3:
                            continue
                        r = paired_test(a[ok], b[ok])
                        e = {"other": other, **r}
                        primary_p, factor = r["p_value"], None
                        if repeated:
                            nb = nadeau_bengio_corrected_ttest(a[ok], b[ok], n_train, n_test)
                            e["nadeau_bengio"] = nb
                            primary_p, factor = nb["p_value"], nb.get("variance_correction")
                        e["primary_test"] = "nadeau_bengio" if repeated else "wilcoxon"
                        if t == "diag" and kind == "vs_reference":
                            e["tost"] = tost_paired(a[ok], b[ok], margin=tost_margin, variance_correction=factor)
                        entries.setdefault(name, {})[kind] = e
                        fam[kind].append((name, primary_p))
                for kind, items in fam.items():
                    if not items:
                        continue
                    pv = np.array([p for _, p in items], float)
                    adj = pv.copy()
                    finite = np.isfinite(pv)
                    adj[finite] = holm_bonferroni(pv[finite])
                    for (name, _), p_adj in zip(items, adj):
                        entries[name][kind]["p_value_holm"] = float(p_adj)
                        entries[name][kind]["holm_family_size"] = int(finite.sum())
                comps[f"{t}/{c}/{metric}"] = entries
    return comps


def recompute_comparisons(results: dict, tost_margin: float = 0.02) -> dict:
    """Recompute ``results['comparisons']`` from the stored per-fold scores (e.g. after merging runs on the same folds)."""
    meta = {k: {"representation": v["representation"], "is_reference": v["is_reference"], "is_proposed": v["is_proposed"]}
            for k, v in results["methods"].items()}
    folds = {k: v["folds"] for k, v in results["methods"].items()}
    classifiers = list(next(iter(folds.values()))["diag"].keys())
    proto = results.get("protocol", {})
    n_train = n_test = None
    if int(proto.get("n_repeats", 1)) > 1:
        n_test = int(proto.get("n_test_typical") or np.mean(proto["fold_sizes"]))
        n_train = int(proto.get("n_train_typical") or proto["n_subjects"] - np.mean(proto["fold_sizes"]))
    results["comparisons"] = compute_comparisons(folds, meta, classifiers, tost_margin=tost_margin,
                                                 n_train=n_train, n_test=n_test)
    return results


def to_jsonable(obj):
    """Recursively convert numpy types / NaN so that json.dump works and NaN -> null."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (f != f or f in (float("inf"), float("-inf"))) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def summary_table(results: dict, classifier: str = "lr") -> str:
    """Console-friendly table of the main numbers."""
    ch = results["chance"]
    lines = [
        f"{'Method':<24}{'Rep':<8}{'Diag acc':>12}{'Diag bAcc':>11}{'Diag AUC':>10}"
        f"{'Site acc':>11}{'Site bAcc':>11}{'Site NMI':>10}{'ICC':>8}{'MMD2':>9}{'eps2 pre>post':>15}",
        "-" * 129,
    ]
    def _f(v):   # JSON round-trips NaN as None
        return float("nan") if v is None else float(v)

    for name, e in results["methods"].items():
        s = e["summary"]
        d = e["oos_distribution"]
        def g(task, met):
            v = _f(s[task].get(classifier, {}).get(met, {}).get("mean"))
            return f"{v:.3f}" if v == v else "  nan"
        icc = _f(d.get("icc_mean"))
        mmd = _f(d.get("mmd2_unbiased_mean"))
        e_pre = _f(e.get("protected_eps2_pre", {}).get("mean"))
        e_post = _f(e.get("protected_eps2", {}).get("mean"))
        eps = f"{e_pre:.3f}>{e_post:.3f}" if e_post == e_post else "-"
        lines.append(
            f"{name:<24}{e['representation']:<8}{g('diag','acc'):>12}{g('diag','bacc'):>11}{g('diag','auc'):>10}"
            f"{g('site','acc'):>11}{g('site','bacc'):>11}{g('site','nmi'):>10}"
            f"{icc:>8.3f}{mmd:>9.4f}{eps:>15}"
        )
    lines.append("-" * 129)
    lines.append(f"chance: site majority={ch['site']['majority_rate']:.3f}, uniform={ch['site']['uniform_chance']:.3f}; "
                 f"diag majority={ch['diag']['majority_rate']:.3f}")
    return "\n".join(lines)
