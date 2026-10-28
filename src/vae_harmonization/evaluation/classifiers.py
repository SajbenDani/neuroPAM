"""
classifiers.py -- Probe classifiers used to score harmonized representations.

Two probes are used throughout: an L2 logistic regression ('lr') and a
100-tree random forest ('rf'), both preceded by feature standardization fitted
on the training fold.  :func:`fit_and_score` returns accuracy, balanced
accuracy, macro-F1 and AUC (binary) / NMI (multi-class) on the test fold.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..stats import classification_summary

CLASSIFIER_LABELS = {"lr": "Logistic Reg.", "rf": "Random Forest"}

# Trees of the random forest are grown in parallel; the result is identical for any n_jobs
# because the per-tree random streams are derived from random_state.  VAEH_RF_JOBS=1 restores
# single-threaded behaviour (e.g. when several evaluations share one machine).
RF_N_JOBS = int(os.environ.get("VAEH_RF_JOBS", "-1"))


def make_classifier(kind: str, seed: int = 42) -> Pipeline:
    if kind == "lr":
        clf = LogisticRegression(random_state=seed, max_iter=2000)
    elif kind == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=RF_N_JOBS)
    else:
        raise ValueError(f"unknown classifier '{kind}'")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def fit_and_score(kind: str, X_tr, y_tr, X_te, y_te, task: str, seed: int = 42) -> dict:
    """Fit a probe on the training fold, score on the test fold."""
    y_tr = np.asarray(y_tr)
    y_te = np.asarray(y_te)
    if np.unique(y_tr).size < 2:
        return {"acc": float("nan"), "bacc": float("nan"), "f1_macro": float("nan")}
    pipe = make_classifier(kind, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    score = None
    if task == "diag" and np.unique(y_tr).size == 2 and hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_te)
        pos = list(pipe.classes_).index(pipe.classes_.max())
        score = proba[:, pos]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # "y_pred contains classes not in y_true"
        return classification_summary(y_te, pred, y_score=score, task=task)


def covariate_probe(X_tr, c_tr, X_te, c_te, seed: int = 42, min_n: int = 20) -> dict:
    """
    How much of a covariate (age, sex, motion...) is still linearly decodable from the
    harmonized features?  Binary covariates -> logistic-regression AUC; continuous ->
    ridge (CV-alpha) Pearson r and out-of-sample R^2.  NaN covariate entries are dropped.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import roc_auc_score

    c_tr = np.asarray(c_tr, dtype=float)
    c_te = np.asarray(c_te, dtype=float)
    ok_tr, ok_te = np.isfinite(c_tr), np.isfinite(c_te)
    if ok_tr.sum() < min_n or ok_te.sum() < 5:
        return {"kind": "unavailable", "r": float("nan"), "r2": float("nan"), "auc": float("nan")}
    X_tr, c_tr, X_te, c_te = X_tr[ok_tr], c_tr[ok_tr], X_te[ok_te], c_te[ok_te]
    vals = np.unique(c_tr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if vals.size == 2:
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=seed))])
            pipe.fit(X_tr, c_tr.astype(int))
            if np.unique(c_te).size < 2:
                return {"kind": "binary", "auc": float("nan"), "acc": float(pipe.score(X_te, c_te.astype(int)))}
            proba = pipe.predict_proba(X_te)[:, list(pipe.classes_).index(int(vals.max()))]
            return {"kind": "binary", "auc": float(roc_auc_score(c_te.astype(int), proba)),
                    "acc": float(pipe.score(X_te, c_te.astype(int)))}
        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-2, 4, 13)))])
        pipe.fit(X_tr, c_tr)
        pred = pipe.predict(X_te)
    ss_res = float(np.sum((c_te - pred) ** 2))
    ss_tot = float(np.sum((c_te - c_te.mean()) ** 2))
    r = float(np.corrcoef(c_te, pred)[0, 1]) if np.std(pred) > 1e-12 else 0.0
    return {"kind": "continuous", "r": r, "r2": 1.0 - ss_res / max(ss_tot, 1e-12)}


# -- legacy global-fit CV (kept for in-sample diagnostics only) ------------------------

def cv_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    tag: str,
    n_folds: int = 10,
    seed: int = 42,
    methods: tuple[str, ...] = ("lr", "rf"),
) -> list[dict]:
    """
    Stratified k-fold CV of the probes on an already-harmonized matrix.

    This does NOT re-fit the harmonization per fold and is therefore only
    appropriate for in-sample diagnostics (Appendix); the primary numbers
    come from :mod:`vae_harmonization.evaluation.protocol`.
    """
    _, counts = np.unique(y, return_counts=True)
    n_folds = min(n_folds, int(counts.min())) if counts.min() >= 2 else 2
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    out = []
    for kind in methods:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            scores = cross_validate(
                make_classifier(kind, seed), X, y, cv=cv,
                scoring=["accuracy", "balanced_accuracy", "f1_macro"],
            )
        out.append({
            "method": CLASSIFIER_LABELS[kind],
            "tag": tag,
            "acc_mean": float(scores["test_accuracy"].mean()),
            "acc_std": float(scores["test_accuracy"].std()),
            "bacc_mean": float(scores["test_balanced_accuracy"].mean()),
            "f1_mean": float(scores["test_f1_macro"].mean()),
            "fold_acc": scores["test_accuracy"].tolist(),
        })
    return out


def format_cv(results: list[dict], header: str) -> str:
    lines = [f"  {header}", f"  {'Probe':<16} {'Acc':>13} {'bAcc':>7} {'F1':>7}", "  " + "-" * 46]
    for r in results:
        lines.append(f"  {r['method']:<16} {r['acc_mean']:.3f}+/-{r['acc_std']:.3f} "
                     f"{r['bacc_mean']:7.3f} {r['f1_mean']:7.3f}")
    return "\n".join(lines)
