#!/usr/bin/env python3
"""
Generate every LaTeX table of the paper, and the number macros quoted in the prose,
from the JSON files written by the pipeline.

    python scripts/generate_tables.py                 # results/ -> paper/tables/

Inputs per cohort (a missing file gives a stub row that says so):
    results/<cohort>/protocol_results.json            main protocol
    results/<cohort>_{geometric,r,gam,deep}/...        extra methods on the same folds
    results/<cohort>_repeated[_est]/...                5x10 repeated CV
    results/<cohort>/{downstream,fingerprinting,network_preservation,protected_axis,
                      unseen_site,federated,overlap}.json
    results/<cohort>_sensitivity/pca_sweep.json        PCA-dimension sweep
    results/<cohort>_vae/, results/abide1_full_whiten/ appendix ablations
    results/simulation/star_vs_chain.json, results/theorem/*.json

Runs on the same folds are merged into one results dict and their paired tests are
recomputed together, so every method of a cohort shares one Holm family.
No number in the paper is typed by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vae_harmonization.evaluation.protocol import recompute_comparisons  # noqa: E402

COHORTS = [("abide1_full", "ABIDE-I", "A"), ("adhd200", "ADHD-200", "B"),
           ("abide2_10site", "ABIDE-II", "C"), ("rest_meta_mdd", "REST-meta-MDD", "D"), ("srpbs", "SRPBS", "E")]
CASE_LABEL = {"abide1_full": "ASD", "adhd200": "ADHD", "abide2_10site": "ASD", "rest_meta_mdd": "MDD", "srpbs": "MDD"}

DISPLAY = {
    "Raw": "Raw FC", "ComBat": "ComBat", "CovBat": "CovBat", "DeepComBat": "DeepComBat",
    "Log-Euclidean FC": "Log-Euclidean FC (no harm.)",
    "Riemannian re-centering": "Riemannian re-centering", "Euclidean alignment": "Euclidean alignment",
    "Riemannian Procrustes": "Riemannian Procrustes, steps 1--2", "RLET": "RLET",
    "Ours (edge residual)": "\\ours, edge residual", "ComBat (no covariate)": "ComBat, no covariate",
    "ComBat (PCA, no covariate)": "ComBat, no covariate",
    "ComBat (age+sex)": "ComBat, age + sex", "ComBat (PCA, age+sex)": "ComBat, age + sex",
    "PCA scores": "PCA scores (no harm.)", "VAE latent": "VAE latent (no harm.)",
    "ComBat (PCA)": "ComBat", "CovBat (PCA)": "CovBat", "ComBat-GAM": "ComBat-GAM", "RELIEF": "RELIEF",
    "CORAL": "CORAL", "MMD affine": "MMD-fitted affine",
    "Ours": r"\textbf{\ours}", "Full affine": "Full-space BW (no protected axis)",
    "Mean shift": "Protected axis + mean shift", "Cholesky map": "Protected axis + Cholesky map",
    "ZCA map": "Protected axis + ZCA map", "Ours (marginal means)": "\\ours, marginal site means",
    "Ours (lambda=1-delta)": r"\ours, geodesic $\lambda_s=1-\delta_s$", "Ours (barycenter hub)": "\\ours, barycenter hub",
    "Ours (diag+severity)": "\\ours, diagnosis + severity axes", "Ours (diag+age+sex)": "\\ours, diagnosis + age + sex axes",
    "Ours (covariates only)": "\\ours, age + sex axes only", "Ours (LW covariance)": "Ledoit--Wolf (identity target)",
    "Ours (diagonal target)": "Diagonal target", "Ours (empirical covariance)": "Empirical (no shrinkage)",
}

# macro tag of every method quoted in the text
MTAG = {
    "Raw": "Raw", "ComBat": "Combat", "CovBat": "Covbat", "DeepComBat": "Deep",
    "Log-Euclidean FC": "Spd", "Riemannian re-centering": "Riem", "Euclidean alignment": "Ea",
    "Riemannian Procrustes": "Rpa", "RLET": "Rlet", "Ours (edge residual)": "Edge",
    "ComBat (no covariate)": "CombatNoCov", "ComBat (PCA, no covariate)": "CombatPcaNoCov",
    "ComBat (age+sex)": "CombatAs", "ComBat (PCA, age+sex)": "CombatPcaAs",
    "PCA scores": "Pca", "ComBat (PCA)": "CombatPca", "CovBat (PCA)": "CovbatPca",
    "ComBat-GAM": "Gam", "RELIEF": "Relief", "CORAL": "Coral", "MMD affine": "MmdAff", "Ours": "Ours",
    "Full affine": "Full", "Mean shift": "Shift", "Cholesky map": "Chol", "ZCA map": "Zca",
    "Ours (marginal means)": "Marg", "Ours (lambda=1-delta)": "Geo", "Ours (barycenter hub)": "Bary",
    "Ours (diag+severity)": "Sev", "Ours (diag+age+sex)": "AgeSex", "Ours (covariates only)": "CovOnly",
    "Ours (LW covariance)": "Lw", "Ours (diagonal target)": "DiagT", "Ours (empirical covariance)": "Emp",
    "VAE latent": "Vae",
}
ESTIMATORS = ["Ours", "Ours (LW covariance)", "Ours (diagonal target)", "Ours (empirical covariance)"]
AGE_SEX_NOTE = ("ComBat, age + sex: label-free ComBat, with age and sex as the only covariates and no diagnosis. It "
                "needs the age and sex of every subject it adjusts. ")


# -- helpers ---------------------------------------------------------------------------------

def load(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def f3(x, nd=3, dash="--"):
    if x is None:
        return dash
    try:
        return dash if x != x else f"{x:.{nd}f}"
    except TypeError:
        return dash


def pct(x, nd=1, dash="--"):
    return dash if x is None or x != x else f"{100 * x:.{nd}f}"


def ms(d, nd=3):
    if not d or d.get("mean") is None:
        return "--"
    sd = d.get("std")
    return f"{d['mean']:.{nd}f}" + (f" $\\pm$ {sd:.{nd}f}" if sd is not None and sd == sd else "")


def pms(d, nd=1):
    if not d or d.get("mean") is None:
        return "--"
    sd = d.get("std")
    return f"{100 * d['mean']:.{nd}f}" + (f" $\\pm$ {100 * sd:.{nd}f}" if sd is not None and sd == sd else "")


def pfmt(p):
    if p is None or p != p:
        return "--"
    return "$<$0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def sci(x, nd=1):
    """LaTeX scientific notation for math mode, e.g. 5.0\\times10^{-5}."""
    if not x:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    return f"{x / 10 ** exp:.{nd}f}\\times10^{{{exp}}}"


def entry(res, name):
    return None if res is None else res["methods"].get(name)


def summ(e, task, clf, met):
    try:
        return e["summary"][task][clf][met]
    except (KeyError, TypeError):
        return None


def mean_of(e, task, clf, met):
    s = summ(e, task, clf, met)
    return None if s is None else s.get("mean")


def comp(res, key, name):
    try:
        return res["comparisons"][key][name]["vs_reference"]
    except (KeyError, TypeError):
        return None


def mmd_z(e):
    perm = ((e or {}).get("oos_distribution") or {}).get("permutation") or {}
    return (perm.get("mmd2_unbiased") or {}).get("z")


def uses_labels(e):
    return bool(((e or {}).get("describe") or {}).get("uses_labels_at_transform"))


def label(name, e=None):
    return DISPLAY.get(name, name) + ("$^\\dagger$" if uses_labels(e) else "")


def table(caption, lab, colspec, header, rows, notes=None, size=r"\small", star=False):
    env = "table*" if star else "table"
    out = [f"\\begin{{{env}}}[!htbp]", r"\centering", size, f"\\caption{{{caption}}}", f"\\label{{{lab}}}",
           r"\fittable{%", f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule"]
    for r in rows:
        is_rule = r.lstrip().startswith(("\\midrule", "\\addlinespace"))
        out.append(r if is_rule or r.rstrip().endswith("\\\\") else r + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}"]
    if notes:
        out.append(r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize " + notes + r"\end{minipage}")
    out.append(f"\\end{{{env}}}")
    return "\n".join(out) + "\n"


def stub(caption, lab, why):
    return table(caption, lab, "l", "Status", [why.replace("_", r"\_")])


def group_row(text, ncol):
    return f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{{text}}}}}"


# -- merging runs on the same folds ----------------------------------------------------------

def _same_folds(a, b) -> bool:
    pa, pb = a["protocol"], b["protocol"]
    return all(pa.get(k) == pb.get(k) for k in ("seed", "fold_sizes", "stratification", "n_subjects", "n_repeats"))


def merge(dst, extra, override=False, source=""):
    """Add the methods of ``extra`` to ``dst``. Only runs on identical folds are merged."""
    if dst is None or not extra or not extra.get("methods"):
        return
    if not _same_folds(dst, extra):
        print(f"warning: {source} is not on the same folds; not merged")
        return
    for name, e in extra["methods"].items():
        if override or name not in dst["methods"]:
            dst["methods"][name] = e


def load_cohort(R: Path, key: str):
    res = load(R / key / "protocol_results.json")
    if res is not None:
        for tag in ("geometric", "r", "gam", "deep", "edge", "combatas"):
            merge(res, load(R / f"{key}_{tag}" / "protocol_results.json"), source=f"{key}_{tag}")
        # CovBat was rerun after aligning it with the authors' code (scaling before the PCA step)
        merge(res, load(R / f"{key}_covbat" / "protocol_results.json"), override=True, source=f"{key}_covbat")
        recompute_comparisons(res)
    rep = load(R / f"{key}_repeated" / "protocol_results.json")
    if rep is not None:
        merge(rep, load(R / f"{key}_repeated_est" / "protocol_results.json"), override=True,
              source=f"{key}_repeated_est")
        recompute_comparisons(rep)
    for r, name in ((res, key), (rep, f"{key}_repeated")):
        for m, e in (r or {}).get("methods", {}).items():
            if e.get("failures"):
                print(f"warning: {name} / {m}: {len(e['failures'])} failed folds")
    return res, rep


# -- tables ----------------------------------------------------------------------------------

def t_cohorts():
    import h5py
    from vae_harmonization.data_registry import DATASET_REGISTRY as REG
    rows = []
    for key, lab, _ in COHORTS:
        e = REG[key]
        path = Path(e["hdf5_path"])
        if not path.exists():
            rows.append(f"{lab} & \\multicolumn{{4}}{{l}}{{\\emph{{{path.name} not found}}}}")
            continue
        with h5py.File(path, "r") as f:
            n_edges = f["X"].shape[1]
            s, d = f["y_site"][:], f["y_diagnosis"][:]
            names = [str(x) for x in f.attrs["site_names"]]
        u, c = np.unique(s, return_counts=True)
        us = "\\_"
        parts = [f"{names[u[i]].replace('_', us)} {c[i]}/{int(((s == u[i]) & (d == 1)).sum())}" for i in np.argsort(-c)]
        rows.append(" & ".join([lab, str(len(s)), f"{e['n_rois']} / {n_edges:,}", f"{int((d == 1).sum())} {CASE_LABEL[key]}",
                                "\\parbox[t]{0.55\\linewidth}{\\raggedright " + ", ".join(parts) + "}"]))
    ts_path = REPO / "data" / "processed" / "srpbs_ts_fc.h5"
    if ts_path.exists():
        with h5py.File(ts_path, "r") as f:
            s, n_edges = f["y_site"][:], f["X"].shape[1]
            t = f["traveler_id"][:]
            names = [str(x) for x in f.attrs["site_names"]]
        u, c = np.unique(s, return_counts=True)
        parts = [f"{names[u[i]]} {c[i]}" for i in np.argsort(-c, kind="stable")]
        rows.append(" & ".join(["SRPBS travelers", f"{np.unique(t).size} ({len(s)} sessions)",
                                f"268 / {n_edges:,}", "0",
                                "\\parbox[t]{0.55\\linewidth}{\\raggedright " + ", ".join(parts) + " (sessions)}"]))
    notes = ("A site whose case count is 0 or equals $n$ contains a single diagnostic group. Site identity then predicts "
             "diagnosis, which matters for harmonization (\\S\\ref{sec:confounding}). SRPBS: the six scanners of the "
             "Multidisorder Connectivity Dataset that recorded both patients and controls. SRPBS travelers: healthy "
             "people scanned at every scanner, used only as ground truth (\\S\\ref{sec:traveling}).")
    return table("Cohorts, with the sizes and case counts of every site.", "tab:cohorts-full", "lcccl", "Cohort & $N$ & ROIs / edges & Cases & Sites ($n$ / cases), largest first",
                 rows, notes, size=r"\footnotesize", star=True)


ATLAS = {"abide1_full": "CC200", "adhd200": "CC200", "abide2_10site": "CC200", "rest_meta_mdd": "Dosenbach-160",
         "srpbs": "BAL-140"}


def t_cohorts_compact():
    import h5py
    from vae_harmonization.data_registry import DATASET_REGISTRY as REG
    rows = []
    for key, lab, _ in COHORTS:
        path = Path(REG[key]["hdf5_path"])
        if not path.exists():
            rows.append(f"{lab} & \\multicolumn{{6}}{{l}}{{\\emph{{{path.name} not found}}}}")
            continue
        with h5py.File(path, "r") as f:
            n_edges = f["X"].shape[1]
            s, d = f["y_site"][:], f["y_diagnosis"][:]
            names = [str(x) for x in f.attrs["site_names"]]
        u, c = np.unique(s, return_counts=True)
        single = [names[i].replace("_", "\\_") for i in u if np.unique(d[s == i]).size == 1]
        rows.append(" & ".join([lab, f"{len(s):,}", str(len(u)), f"{ATLAS[key]} ({n_edges:,})",
                                f"{int((d == 1).sum())} {CASE_LABEL[key]}", f"{c.min()}--{c.max()}",
                                ", ".join(single) if single else "none"]))
    ts_path = REPO / "data" / "processed" / "srpbs_ts_fc.h5"
    if ts_path.exists():
        with h5py.File(ts_path, "r") as f:
            s, n_edges, t = f["y_site"][:], f["X"].shape[1], f["traveler_id"][:]
        c = np.bincount(s)
        rows.append(" & ".join(["SRPBS travelers", f"{np.unique(t).size} people, {len(s)} sessions", str(len(c)),
                                f"Shen-268 ({n_edges:,})", "0", f"{c.min()}--{c.max()}", "all healthy"]))
    notes = ("Site sizes: smallest and largest site. A single-group site contains only cases or only controls, so its "
             "identity predicts diagnosis (\\S\\ref{sec:confounding}). SRPBS: the six scanners of the Multidisorder "
             "Connectivity Dataset that recorded both patients and controls. SRPBS travelers: the same healthy people "
             "scanned at every scanner, used only as ground truth (\\S\\ref{sec:traveling}). Per-site counts are in "
             "Table~\\ref{tab:cohorts-full}.")
    return table("Cohorts.", "tab:cohorts", "lcclccl",
                 "Cohort & $N$ & Sites & Atlas (edges) & Cases & Site sizes & Single-group sites", rows, notes,
                 size=r"\footnotesize", star=True)


MAIN_ROWS = [("Raw connectome (Fisher-$z$ edges)", ["Raw", "ComBat", "ComBat (age+sex)", "CovBat",
                                                     "Ours (edge residual)"]),
             ("Matrix logarithms of the shrunk correlation matrices", ["Log-Euclidean FC", "Riemannian re-centering",
                                                                       "Euclidean alignment", "Riemannian Procrustes",
                                                                       "RLET"]),
             ("100 PCA scores (re-fitted in every fold)", ["PCA scores", "ComBat (PCA)", "ComBat (PCA, age+sex)",
                                                          "CovBat (PCA)", "ComBat-GAM", "RELIEF", "DeepComBat", "CORAL",
                                                          "MMD affine", "Ours"])]


def t_main(res):
    if res is None:
        return stub("Main results.", "tab:main", "results/abide1_full/protocol_results.json not found")
    rows = []
    for title, names in MAIN_ROWS:
        rows.append(group_row(title, 9))
        for name in names:
            e = entry(res, name)
            if e is None:
                continue
            c = comp(res, "diag/lr/acc", name)
            rows.append(" & ".join([
                "\\quad " + label(name, e), pms(summ(e, "diag", "lr", "acc")), pms(summ(e, "diag", "lr", "auc")),
                pms(summ(e, "site", "lr", "acc")), pms(summ(e, "site", "rf", "acc")),
                f3(mean_of(e, "site", "lr", "nmi"), 2), f3(e["oos_distribution"].get("icc_mean"), 3),
                f3(mmd_z(e), 1), "--" if c is None else pfmt(c.get("p_value_holm"))]))
    ch, pr = res["chance"], res["protocol"]
    header = ("Method & Diag.\\ acc.\\ (\\%) & Diag.\\ AUC (\\%) & Site acc.\\ LR (\\%) & Site acc.\\ RF (\\%) & "
              "Site NMI & ICC & MMD $z$ & $p$ (Holm)")
    notes = (f"$N={pr['n_subjects']}$, {pr['n_folds']}-fold cross-validation. Everything, including scaler and PCA, is fitted on "
             f"the training fold. Chance levels: site majority {pct(ch['site']['majority_rate'])}\\%, uniform "
             f"{pct(ch['site']['uniform_chance'])}\\%, diagnosis majority {pct(ch['diag']['majority_rate'])}\\%. "
             "Site NMI: normalized mutual information between true and predicted site (LR probe). "
             "ICC and MMD are computed on the pooled held-out subjects. MMD $z$: observed MMD$^2$ against 200 permutations "
             "of the site labels on the same matrix. For a method that equalizes site means it tests what differs "
             "beyond the means (Section~\\ref{sec:protocol}). The matrix-logarithm block uses correlation matrices shrunk "
             "toward the identity ($C\\leftarrow0.9\\,C+0.1\\,I$). "
             "$p$: paired Wilcoxon on fold-wise diagnosis accuracy against the un-harmonized input of the same block, "
             "Holm-corrected over all methods of the cohort. $^\\dagger$Uses the diagnosis of held-out subjects. "
             + AGE_SEX_NOTE + "Mean $\\pm$ s.d.\\ over folds.")
    return table("Harmonization results on ABIDE-I (17 sites).", "tab:main", "lcccccccc", header, rows, notes, star=True)


CROSS_ROWS = ["Raw", "ComBat", "ComBat (age+sex)", "CovBat", "Ours (edge residual)", "Log-Euclidean FC",
              "Riemannian re-centering", "Euclidean alignment", "Riemannian Procrustes", "RLET", "PCA scores",
              "ComBat (PCA, age+sex)", "ComBat-GAM", "RELIEF", "DeepComBat", "CORAL", "MMD affine", "Ours",
              "Ours (marginal means)", "Full affine"]


def t_cross(results):
    rows = []
    for key, lab, _ in COHORTS[1:]:
        res = results.get(key)
        if res is None:
            rows.append(group_row(f"{lab}: results/{key}/protocol_results.json not found", 9))
            continue
        ch = res["chance"]
        rows.append(group_row(f"{lab}, $N={res['protocol']['n_subjects']}$, {res['protocol']['n_sites']} sites, "
                              f"site majority {pct(ch['site']['majority_rate'])}\\%, diagnosis majority "
                              f"{pct(ch['diag']['majority_rate'])}\\%", 9))
        for name in CROSS_ROWS:
            e = entry(res, name)
            if e is None:
                continue
            c = comp(res, "diag/lr/auc", name)
            eps = e.get("protected_eps2", {}).get("mean")
            eps_s = "--" if eps is None or eps != eps else f"{e['protected_eps2_pre']['mean']:.3f}$\\to${eps:.3f}"
            rows.append(" & ".join([
                "\\quad " + label(name, e), pms(summ(e, "diag", "lr", "acc")), pms(summ(e, "diag", "lr", "auc")),
                pms(summ(e, "site", "lr", "acc")), pms(summ(e, "site", "rf", "acc")),
                f3(mean_of(e, "site", "lr", "nmi"), 2), f3(mmd_z(e), 1), eps_s,
                "--" if c is None else pfmt(c.get("p_value_holm"))]))
    header = ("Cohort / method & Diag.\\ acc.\\ (\\%) & Diag.\\ AUC (\\%) & Site acc.\\ LR (\\%) & Site acc.\\ RF (\\%) & "
              "Site NMI & MMD $z$ & $\\epsilon^2$ pre$\\to$post & $p$ AUC (Holm)")
    notes = ("Same configuration as ABIDE-I, no re-tuning. $\\epsilon^2$: bias-corrected share of the variance of the held-out "
             "protected coordinate that site explains, before and after harmonization. $p$: paired Wilcoxon on fold-wise "
             "diagnosis AUC against the un-harmonized input, Holm-corrected. AUC is used here because the diagnosis classes "
             "are unbalanced in ADHD-200. $^\\dagger$Uses the diagnosis of held-out subjects. " + AGE_SEX_NOTE.rstrip())
    return table("Full results on the other four cohorts.", "tab:cross-full", "lcccccccc", header, rows, notes, star=True)


ABLATION_ROWS = ["PCA scores", "Ours", "Full affine", "Mean shift", "Cholesky map", "ZCA map", "CORAL",
                 "Ours (marginal means)", "Ours (lambda=1-delta)", "Ours (barycenter hub)", "Ours (diag+severity)",
                 "Ours (diag+age+sex)", "Ours (covariates only)"]


def t_ablation(res, whiten, disp=None):
    if res is None:
        return stub("Ablations.", "tab:ablation", "results/abide1_full/protocol_results.json not found")
    rows = []

    def fitted(name):
        e = entry(disp, name)
        return "--" if e is None else f3((e.get("displacement_plugin") or {}).get("mean"), 0)

    def row(name, e, src, show_disp=True, prefix=""):
        c = comp(src, "diag/lr/acc", name)
        return " & ".join([
            prefix + label(name, e), pms(summ(e, "diag", "lr", "acc")), pms(summ(e, "diag", "lr", "auc")),
            pms(summ(e, "site", "lr", "acc")), pms(summ(e, "site", "rf", "acc")), f3(mean_of(e, "site", "lr", "nmi"), 2),
            f3(mmd_z(e), 1), f3(e.get("displacement", {}).get("mean"), 0) if show_disp else "n/a",
            fitted(name) if show_disp else "n/a", "--" if c is None else pfmt(c.get("p_value_holm"))])

    for name in ABLATION_ROWS:
        e = entry(res, name)
        if e is not None:
            rows.append(row(name, e, res))
    if whiten is not None:
        rows.append(group_row("Whitened PCA scores (separate run, same folds)", 10))
        for name in ("PCA scores", "Ours", "Full affine"):
            e = entry(whiten, name)
            if e is not None:
                rows.append(row(name, e, whiten, show_disp=False, prefix="\\quad "))
    header = ("Variant & Diag.\\ acc.\\ (\\%) & Diag.\\ AUC (\\%) & Site acc.\\ LR (\\%) & Site acc.\\ RF (\\%) & Site NMI & "
              "MMD $z$ & Disp.\\ held-out & Disp.\\ fitted & $p$ (Holm)")
    notes = ("ABIDE-I, same folds as Table~\\ref{tab:main}. Disp.\\ held-out: mean squared distance each held-out subject "
             "moves in PCA space. Disp.\\ fitted: the expected squared displacement per subject under the fitted (shrunk) "
             "moments, averaged over training sites and folds. This is the quantity Theorem~\\ref{thm:main} minimizes, so "
             "among maps that match the same covariances and fix the same subspace the Bures--Wasserstein map has the "
             "smallest value. Held-out subjects do not follow the fitted covariance exactly, so their displacement can rank "
             "two near-optimal maps differently. $p$: paired Wilcoxon on fold-wise diagnosis accuracy against the PCA "
             "scores, Holm-corrected.")
    return table("Ablations on ABIDE-I.", "tab:ablation", "lccccccccc", header, rows, notes, star=True)


def disp_fmt(x):
    """Displacement for a table cell; a diverging map is printed as an order of magnitude."""
    if x is None or x != x:
        return "--"
    if x >= 1e6:
        return f"$>$10$^{{{int(np.floor(np.log10(x)))}}}$"
    return f"{x:.0f}"


def t_estimator(results):
    rows = []
    for key, lab, _ in COHORTS:
        res = results.get(key)
        if res is None or entry(res, "Ours") is None:
            continue
        rows.append(group_row(lab, 7))
        for name in ESTIMATORS:
            e = entry(res, name)
            if e is None:
                continue
            nm = "Pooled within-site target (default)" if name == "Ours" else DISPLAY[name]
            rows.append(" & ".join([
                "\\quad " + nm, pms(summ(e, "diag", "lr", "acc")), pms(summ(e, "diag", "lr", "auc")),
                pms(summ(e, "site", "lr", "acc")), f3(mean_of(e, "site", "lr", "nmi"), 2), f3(mmd_z(e), 1),
                disp_fmt(e.get("displacement", {}).get("mean"))]))
    if not rows:
        return stub("Covariance estimators.", "tab:estimator-full", "no protocol results found")
    header = ("Cohort / covariance estimator & Diag.\\ acc.\\ (\\%) & Diag.\\ AUC (\\%) & Site acc.\\ LR (\\%) & Site NMI & "
              "MMD $z$ & Displacement")
    notes = ("Only the plug-in covariance of the complement changes between rows. Shrinkage intensities are analytic "
             "\\citep{ledoit2004shrinkage,schafer2005shrinkage}. Displacement is on the PCA scale of each cohort.")
    return table("Choice of the covariance estimator (full).", "tab:estimator-full", "lcccccc", header, rows, notes)


REPEATED_ROWS = ["ComBat", "CovBat", "Ours", "Full affine", "Mean shift", "Ours (covariates only)",
                 "Ours (LW covariance)", "ComBat (PCA)"]


def t_repeated(reps):
    rows = []
    for key, lab, _ in COHORTS:
        rep = reps.get(key)
        if rep is None:
            rows.append(group_row(f"{lab}: results/{key}_repeated not found", 8))
            continue
        pr = rep["protocol"]
        rows.append(group_row(f"{lab}, {pr['n_repeats']}$\\times${pr['n_folds']} folds", 8))
        for name in REPEATED_ROWS:
            e = entry(rep, name)
            c = comp(rep, "diag/lr/acc", name)
            if e is None or c is None:
                continue
            t = c.get("tost") or {}
            ref = "Raw" if e["representation"] == "raw" else "PCA"
            rows.append(" & ".join([
                "\\quad " + label(name, e) + f" (vs.\\ {ref})", pms(summ(e, "diag", "lr", "acc")),
                f"{100 * c['mean_diff']:+.1f}", f"[{100 * t.get('ci90_low', float('nan')):+.1f}, "
                                                 f"{100 * t.get('ci90_high', float('nan')):+.1f}]",
                pfmt(c.get("p_value_holm")), "yes" if t.get("equivalent") else "no",
                pms(summ(e, "site", "lr", "acc")), f3(mean_of(e, "site", "lr", "nmi"), 2)]))
    header = ("Cohort / method & Diag.\\ acc.\\ (\\%) & $\\Delta$ (pp) & 90\\% CI (pp) & $p$ (NB, Holm) & "
              "Equiv.\\ $\\pm$2\\,pp & Site acc.\\ LR (\\%) & Site NMI")
    notes = ("Repeated cross-validation with 5 different fold assignments. $\\Delta$: mean difference in diagnosis accuracy "
             "against the un-harmonized input. The confidence interval and the $p$-value use the Nadeau--Bengio variance "
             "correction for overlapping training sets \\citep{nadeau2003inference}. Equivalence: two one-sided tests "
             "with a $\\pm$2\\,pp margin and the same corrected variance \\citep{lakens2017tost}. "
             "$^\\dagger$Uses the diagnosis of held-out subjects.")
    return table("Repeated 5$\\times$10 cross-validation with corrected tests (full).", "tab:repeated-full", "lccccccc", header, rows,
                 notes, star=True)


PERM_ROWS = ["Raw", "ComBat", "Log-Euclidean FC", "Riemannian re-centering", "PCA scores", "Mean shift", "Ours"]


def t_permutation(results):
    rows = []
    for key, lab, _ in COHORTS:
        res = results.get(key)
        if res is None:
            continue
        rows.append(group_row(lab, 7))
        for name in PERM_ROWS:
            e = entry(res, name)
            perm = ((e or {}).get("oos_distribution") or {}).get("permutation") or {}
            mmd, icc = perm.get("mmd2_unbiased") or {}, perm.get("icc") or {}
            if not mmd:
                continue
            rows.append(" & ".join(["\\quad " + label(name, e), f3(mmd.get("observed"), 4), f3(mmd.get("null_mean"), 4),
                                    f3(mmd.get("z"), 1), f3(mmd.get("percentile"), 2), f3(icc.get("z"), 1),
                                    f3(icc.get("percentile"), 2)]))
    if not rows:
        return stub("Permutation nulls.", "tab:permutation", "no protocol results found")
    header = "Cohort / method & MMD$^2$ & MMD$^2$ null & MMD $z$ & MMD pct. & ICC $z$ & ICC pct."
    notes = ("Site labels are permuted 200 times on the fixed pooled held-out matrix of each method. Percentile: share of "
             "permutations with at least as much site structure as observed. With folds stratified by site, the "
             "corrections of a method that equalizes training-fold site means nearly cancel in the pooled held-out "
             "site means, so these sites are more alike in their means than random subsets are. The MMD$^2$ then falls "
             "below its null, and the $z$-score tests what differs beyond the means.")
    return table("Permutation nulls for MMD$^2$ and ICC.", "tab:permutation", "lcccccc", header, rows, notes,
                 size=r"\footnotesize", star=True)


def t_protected(pa_by, results):
    rows = []
    for key, lab, _ in COHORTS:
        pa, res = pa_by.get(key), results.get(key)
        if pa is None:
            continue
        ax = pa["axes"][0]
        e = entry(res, "Ours")
        eps_fold = "--" if e is None else f"{f3(e['protected_eps2_pre']['mean'])}$\\to${f3(e['protected_eps2']['mean'])}"
        rows.append(" & ".join([lab, pct(pa["between_site_ss"]["protected_share_pre"]),
                                pct(pa["k_protected"] / pa["latent_dim"]),
                                f"{f3(ax['site_eps2_pre'])}$\\to${f3(ax['site_eps2_post'])}", eps_fold,
                                f"{f3(ax['diag_d_pre'], 2)}$\\to${f3(ax['diag_d_post'], 2)}",
                                f"{f3(pa['complement_mean_eta2_pre'])}$\\to${f3(pa['complement_mean_eta2_post'])}"]))
    if not rows:
        return stub("Protected axis.", "tab:protected", "no protected_axis.json found")
    header = ("Cohort & Site SS on axis (\\%) & $k/d$ (\\%) & $\\epsilon^2$ on axis (in-sample) & $\\epsilon^2$ (held-out) & "
              "Diag.\\ $d$ on axis & Complement $\\eta^2$")
    notes = ("Site SS on axis: share of the between-site sum of squares that lies on the protected diagnosis axis. A "
             "direction unrelated to site would carry $k/d$. $\\epsilon^2$: bias-corrected site effect along the axis, "
             "before$\\to$after harmonization. Diag.\\ $d$: Cohen's $d$ of diagnosis along the axis. Complement $\\eta^2$: "
             "mean per-dimension site $\\eta^2$ on the complement, before$\\to$after.")
    return table("How much site variance lies on the protected axis.", "tab:protected", "lcccccc", header, rows, notes,
                 star=True)


def t_covariates(res):
    if res is None:
        return stub("Covariate preservation.", "tab:covariates", "results/abide1_full/protocol_results.json not found")
    rows = []
    for name in ("PCA scores", "Ours (marginal means)", "Ours", "Ours (diag+age+sex)", "Ours (covariates only)",
                 "Mean shift", "ComBat (PCA)"):
        e = entry(res, name)
        if e is None or not e.get("covariates"):
            continue
        cv = e["covariates"]
        age, sex, fd = cv.get("age", {}), cv.get("sex", {}), cv.get("mean_fd", {})
        rows.append(" & ".join([label(name, e), ms(age.get("r"), nd=2), ms(sex.get("auc"), nd=2), ms(fd.get("r"), nd=2),
                                pms(summ(e, "site", "lr", "acc")), f3(mean_of(e, "site", "lr", "nmi"), 2)]))
    header = "Method & Age $r$ & Sex AUC & Motion $r$ & Site acc.\\ LR (\\%) & Site NMI"
    notes = ("ABIDE-I. Held-out probes: ridge regression for age and mean framewise displacement (motion), logistic "
             "regression for sex. $^\\dagger$Uses the diagnosis of held-out subjects.")
    return table("Age, sex and motion after harmonization (ABIDE-I).", "tab:covariates", "lccccc", header, rows, notes)


def t_overlap(ov_by):
    rows = []
    for key, lab, _ in COHORTS:
        ov = ov_by.get(key)
        if ov is None:
            continue
        for cname, rec in (ov.get("covariates") or {}).items():
            rows.append(" & ".join([lab, cname.replace("mean_fd", "motion"), f3(rec.get("mean_overlap"), 2),
                                    f3(rec.get("min_overlap"), 2)]))
    if not rows:
        return stub("Covariate overlap.", "tab:overlap", "no overlap.json found")
    header = "Cohort & Covariate & Mean pairwise overlap & Min pairwise overlap"
    notes = ("Histogram intersection between the covariate distributions of every pair of sites (1 = identical). "
             "Low age overlap is why the site intercepts are estimated conditionally on the protected covariates.")
    return table("Covariate overlap across sites.", "tab:overlap", "llcc", header, rows, notes)


def t_fingerprint(fp_by, fp_edge_by=None):
    rows = []
    for key, lab, _ in COHORTS:
        fp = fp_by.get(key)
        if fp is None:
            continue
        rows.append(group_row(f"{lab}, $N={fp['n_subjects']}$", 5))
        for name in ("Unharmonized", "Mean shift", "Ours (lambda=1-delta)", "Ours", "Cholesky map", "Full affine"):
            c = fp["conditions"].get(name)
            if c is None:
                continue
            w = c["within_site"]
            nm = "PCA scores (no harm.)" if name == "Unharmonized" else DISPLAY[name]
            rows.append(" & ".join(["\\quad " + nm, pct(c["all"]["rank1"]), pct(w["rank1_weighted"]),
                                    pct(w["chance_weighted"]), f3(w["idiff_weighted"], 1)]))
        fe = (fp_edge_by or {}).get(key)
        if fe:
            for name, nm in (("Unharmonized", "Raw FC (no harm.)"), ("Ours (edge residual)", "\\ours, edge residual")):
                c = fe["conditions"].get(name)
                if c is None:
                    continue
                w = c["within_site"]
                rows.append(" & ".join(["\\quad " + nm, pct(c["all"]["rank1"]), pct(w["rank1_weighted"]),
                                        pct(w["chance_weighted"]), f3(w["idiff_weighted"], 1)]))
    if not rows:
        return stub("Fingerprinting.", "tab:fingerprint", "no fingerprinting.json found")
    header = "Cohort / condition & Rank-1 all (\\%) & Rank-1 within site (\\%) & Within-site chance (\\%) & $I_\\mathrm{diff}$"
    notes = ("Split-half identifiability. One harmonizer is fitted on the full-scan data and applied to both half-scans. "
             "`All': the candidate pool is the whole cohort, where a shared site signature helps. `Within site': the pool "
             "is the subject's own site. $I_\\mathrm{diff}$: mean self-similarity minus mean other-similarity "
             "(cosine $\\times$100). The last two rows of each cohort compare whole connectomes, the others PCA "
             "scores. REST-meta-MDD and SRPBS have no split-half data.")
    return table("Subject identifiability before and after harmonization.", "tab:fingerprint", "lcccc", header, rows, notes)


def t_network(net_by):
    rows = []
    for key, lab, _ in COHORTS:
        n = net_by.get(key)
        if n is None:
            continue
        rows.append(group_row(f"{lab}, {n['n_rois_cortical']} of {n['n_rois']} parcels cortical", 5))
        ref = n["conditions"]["Raw FC"]
        rows.append(" & ".join(["\\quad Raw FC (reference)", "--", "--", "--", f3(ref["site_eta2"]["mean"])]))
        for name, nm in (("PCA reconstruction", "PCA reconstruction (ceiling)"), ("Decoded Mean shift", "Mean shift"),
                         ("Decoded Ours", "\\ours"), ("ComBat", "ComBat on raw FC$^\\dagger$")):
            c = n["conditions"].get(name)
            if c is None or "vs_reference" not in c:
                continue
            rows.append(" & ".join(["\\quad " + nm, f3(c["vs_reference"]["pearson_r"], 3), f3(c["vs_reference"]["rmse"], 3),
                                    f3(c["per_subject_r"]["mean"], 3), f3(c["site_eta2"]["mean"])]))
    if not rows:
        return stub("Network preservation.", "tab:network", "no network_preservation.json found")
    header = "Cohort / condition & Group $r$ & RMSE & Per-subject $r$ & Site $\\eta^2$"
    notes = ("Scores are mapped back to connectivity and averaged into a $7\\times7$ Yeo-network matrix. Group $r$ and RMSE "
             "compare with the raw group matrix. Per-subject $r$ compares each subject with their own raw matrix. Site "
             "$\\eta^2$: mean over the 28 network-level edges. The PCA reconstruction without harmonization is the ceiling "
             "set by the 100-component inversion.")
    return table("Functional-network structure after harmonization.", "tab:network", "lcccc", header, rows, notes)


DOWN_FULL = ["Raw", "ComBat", "ComBat (no covariate)", "ComBat (age+sex)", "PCA scores", "ComBat (PCA)",
             "ComBat (PCA, no covariate)", "ComBat (PCA, age+sex)", "Full affine", "Mean shift",
             "Ours"]      # the Riemannian effect sizes are in table_es.tex


def _es(es_by, key, name):
    return (((es_by or {}).get(key) or {}).get("methods") or {}).get(name)


def t_downstream(down_by, es_by=None):
    rows = []
    for key, lab, _ in COHORTS:
        d = down_by.get(key)
        if d is None and not _es(es_by, key, "Ours"):
            continue
        loso = (d or {}).get("loso") or {"methods": {}, "sites": [], "comparisons": {}}
        rows.append(group_row(f"{lab}, {len(loso['sites'])} held-out sites", 7))
        for name in DOWN_FULL:
            m = loso["methods"].get(name)
            es = _es(es_by, key, name)
            if m is None and es is None:
                continue
            labels_all = (es or {}).get("labels") == "all" or (m or {}).get("uses_test_labels")
            dag = "$^\\dagger$" if labels_all else ""
            if m is not None:
                clf = list(m["summary"])[0]
                sm = m["summary"][clf]
                cmpv = loso["comparisons"].get(clf, {}).get(name)
                loso_cells = [pms(sm["acc"]), pms(sm["auc"]), "--" if cmpv is None else pfmt(cmpv["p_value"])]
            else:
                loso_cells = ["--", "--", "--"]
            es_cells = [ms((es or {}).get("r"), nd=2), ms((es or {}).get("sign_agreement"), nd=2)]
            hidden = {"hidden for the scored half": "hidden", "none": "none", "all": "all"}.get((es or {}).get("labels"), "--")
            rows.append(" & ".join(["\\quad " + DISPLAY.get(name, name) + dag] + loso_cells + es_cells + [hidden]))
    if not rows:
        return stub("Downstream analyses.", "tab:downstream-full", "no downstream.json found")
    header = ("Cohort / method & LOSO acc.\\ (\\%) & LOSO AUC (\\%) & $p$ vs.\\ ref. & Effect-size $r$ & Sign agreement & "
              "Labels")
    notes = ("Leave-one-site-out (LOSO): the harmonizer sees the held-out site's scans, and the LR probe is trained on the "
             "other sites. $p$: paired Wilcoxon over held-out sites against the un-harmonized input. Effect-size "
             "reproducibility: correlation (and sign agreement) of per-feature Cohen's $d$ for diagnosis between two "
             "disjoint halves of the sites, 20 random splits, computed on the edges or the PCA scores of each row. "
             "Labels: what the harmonizer was given. With `hidden', the diagnoses of the half being scored were hidden "
             "from it (it is fitted twice per split). With `none', it uses no diagnosis. With `all', its model needs the "
             "diagnosis of every subject it adjusts ($^\\dagger$), so the scored half's labels shape its output. "
             + AGE_SEX_NOTE.rstrip())
    return table("Leave-one-site-out generalization and effect-size reproducibility (full).", "tab:downstream-full",
                 "lcccccc", header, rows, notes, size=r"\footnotesize", star=True)


UNSEEN_SIZES = (("10", "$n=10$"), ("20", "$n=20$"), ("30", "$n=30$"), ("50", "$n=50$"), ("0.75", "75\\% of the site"))


def t_unseen(unseen_by):
    rows = []
    for key, lab, _ in COHORTS:
        u = unseen_by.get(key)
        if u is None or "heldout_vs_rest_acc_unharmonized" not in next(iter((u.get("by_size") or {}).values()), {}):
            continue
        rows.append(group_row(lab, 6))
        for sz, nm in UNSEEN_SIZES:
            cell = (u.get("by_size") or {}).get(sz)
            if not cell or not cell.get("n"):
                continue
            rows.append(" & ".join([f"\\quad {nm}", pms(cell["heldout_vs_rest_acc"]),
                                    pms(cell["heldout_vs_rest_acc_unharmonized"]), pms(cell["diag_acc"]),
                                    pms(cell["diag_acc_unharmonized"]), f"{len(cell['sites'])} / {cell['n']}"]))
    if not rows:
        return stub("Unseen-site curve.", "tab:unseen", "no unseen_site.json found")
    header = ("Cohort / scans that fit the new site & New-vs-rest, \\ours\\ (\\%) & New-vs-rest, none (\\%) & "
              "Diag.\\ acc., \\ours\\ (\\%) & Diag.\\ acc., none (\\%) & Sites / draws")
    notes = ("One site at a time is new. $n$ of its scans, without labels, and 75\\% of every other site fit the map. The "
             "other 25\\% of every other site is held out of the fit. New-vs-rest: 5-fold cross-validated accuracy of a "
             "binary LR probe that separates the new site's \\emph{other} scans from an equal number (at most 40) of "
             "held-out scans of the other sites, so both groups were transformed by maps that never saw them (chance "
             "50\\%). None: the same probe on the un-harmonized scores. Diag.\\ acc.: a probe trained on the other sites, "
             "tested on the new site's other scans.")
    return table("Harmonizing a new site from a few unlabelled scans.", "tab:unseen", "lccccc", header, rows, notes,
                 star=True)


def t_simulation(sim):
    if sim is None:
        return stub("Hub simulation.", "tab:simulation", "results/simulation/star_vs_chain.json not found")
    r, s = sim["results"], sim["settings"]
    names = {"none": "No harmonization", "star_target": "Star, largest site as hub",
             "star_barycenter": "Star, BW barycenter hub", "chain": "Chain (sequential pairwise)"}
    rows = [" & ".join([names[k], ms(r[k]["w2_to_hub_pop"], nd=2), pms(r[k]["probe_acc"]), ms(r[k]["displacement"], nd=1)])
            for k in ("none", "star_target", "star_barycenter", "chain")]
    notes = (f"{s['n_sites']} Gaussian sites in $d={s['d']}$ with $n={s['n_per_site']}$ each (hub {s['hub_size']}), "
             f"{s['n_rep']} replicates. Maps are estimated from samples and evaluated on new samples. $W_2^2$: "
             f"Wasserstein distance from each harmonized site to the hub. Probe majority rate {pct(sim['chance_probe'])}\\%.")
    return table("Hub design in simulation.", "tab:simulation", "lccc",
                 "Design & $W_2^2$ to hub & Site probe acc.\\ (\\%) & Displacement", rows, notes)


def t_sensitivity(grids):
    keys = [(k, lab) for k, lab, _ in COHORTS if grids.get(k)]
    if not keys:
        return stub("PCA-dimension sweep.", "tab:sensitivity", "no pca_sweep.json found")
    dims = sorted({c["pca_dim"] for k, _ in keys for c in grids[k]["cells"]})
    rows = []
    for dim in dims:
        cells = []
        for k, _ in keys:
            c = next((c for c in grids[k]["cells"] if c["pca_dim"] == dim), None)
            cells += (["--"] * 3 if c is None else
                      [pct(c["raw"]["diag_acc"]), pct(c["ours"]["diag_acc"]), f3(c["ours"]["site_nmi"], 2)])
        rows.append(" & ".join([str(dim)] + cells))
    header = (" & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{lab}}}" for _, lab in keys) + r" \\" + "\n" +
              "PCA dim. & " + " & ".join(["PCA acc. & \\ours\\ acc. & \\ours\\ NMI"] * len(keys)))
    notes = ("Leakage-free protocol at each number of components (LR probes). Acc.: diagnosis accuracy (\\%). NMI: site NMI "
             "of \\ours. The site accuracy of \\ours\\ stays at the majority rate in every cell on ABIDE-I.")
    return table("Sensitivity to the number of principal components.", "tab:sensitivity", "l" + "ccc" * len(keys),
                 header, rows, notes, star=True)


def t_vae(vaes, results):
    rows = []
    for key, lab, _ in COHORTS:
        v = vaes.get(key)
        if v is None:
            continue
        rows.append(group_row(lab, 7))
        for name in ("VAE latent", "Ours", "Mean shift", "Full affine"):
            e = entry(v, name)
            if e is not None:
                rows.append(" & ".join(["\\quad " + label(name, e), pms(summ(e, "diag", "lr", "acc")),
                                        pms(summ(e, "diag", "lr", "auc")), pms(summ(e, "site", "lr", "acc")),
                                        pms(summ(e, "site", "rf", "acc")), f3(mean_of(e, "site", "lr", "nmi"), 2),
                                        f3(mmd_z(e), 1)]))
        e = entry(results.get(key), "Ours")
        if e is not None:
            rows.append(" & ".join(["\\quad \\ours\\ on PCA scores (main run)", pms(summ(e, "diag", "lr", "acc")),
                                    pms(summ(e, "diag", "lr", "auc")), pms(summ(e, "site", "lr", "acc")),
                                    pms(summ(e, "site", "rf", "acc")), f3(mean_of(e, "site", "lr", "nmi"), 2),
                                    f3(mmd_z(e), 1)]))
    if not rows:
        return stub("VAE ablation.", "tab:vae", "no results/<cohort>_vae found")
    header = "Cohort / method & Diag.\\ acc.\\ (\\%) & Diag.\\ AUC (\\%) & Site acc.\\ LR (\\%) & Site acc.\\ RF (\\%) & Site NMI & MMD $z$"
    notes = ("Frozen 64-D FC-VAE latent (encoder trained once on 85\\% of each cohort, not re-fitted per fold) against the "
             "per-fold PCA scores of the main run. Same folds and probes.")
    return table("The harmonizer on a frozen VAE latent.", "tab:vae", "lcccccc", header, rows, notes, star=True)


# -- compact cohort-by-method matrices for the main text ---------------------------------------

def _cohort_header(keys, sub, ncol):
    top = " & " + " & ".join(f"\\multicolumn{{{ncol}}}{{c}}{{{lab}}}" for _, lab in keys)
    rules = "".join(f"\\cmidrule(lr){{{2 + i * ncol}-{1 + (i + 1) * ncol}}}" for i in range(len(keys)))
    return top + r" \\" + rules + "\nMethod & " + " & ".join([sub] * len(keys))


CROSS_COMPACT = ["Raw", "ComBat", "ComBat (age+sex)", "Ours (edge residual)", "Riemannian re-centering",
                 "Euclidean alignment", "RLET", "PCA scores", "ComBat-GAM", "RELIEF", "DeepComBat", "CORAL", "Ours",
                 "Ours (marginal means)"]


def t_cross_compact(results):
    keys = [(k, lab) for k, lab, _ in COHORTS[1:] if results.get(k) is not None]
    if not keys:
        return stub("Other cohorts.", "tab:cross", "no protocol results found")
    rows = []
    for name in CROSS_COMPACT:
        cells, e_any = [], None
        for k, _ in keys:
            e = entry(results[k], name)
            e_any = e_any or e
            cells += ["--"] * 3 if e is None else [pct(mean_of(e, "diag", "lr", "auc")), pct(mean_of(e, "site", "lr", "acc")),
                                                   f3(mean_of(e, "site", "lr", "nmi"), 2)]
        if e_any is not None:
            rows.append(" & ".join([label(name, e_any)] + cells))
    rows.append(r"\midrule")
    rows.append(" & ".join(["Chance (site: majority rate)"] + sum((["50.0", pct(results[k]["chance"]["site"]["majority_rate"]),
                                                                    "0"] for k, _ in keys), [])))
    header = _cohort_header(keys, "AUC & Site & NMI", 3)
    notes = ("Held-out diagnosis AUC (\\%), site accuracy of the LR probe (\\%) and site NMI, 10-fold cross-validation with "
             "the same configuration as ABIDE-I and no re-tuning. A site accuracy below the majority rate with a positive "
             "NMI is anti-prediction, not removal. The full table with random-forest probes, MMD $z$ and paired tests is "
             "Table~\\ref{tab:cross-full}. $^\\dagger$Uses the diagnosis of held-out subjects. " + AGE_SEX_NOTE.rstrip())
    return table("Site removal and diagnosis on the other four cohorts.", "tab:cross", "l" + "ccc" * len(keys), header,
                 rows, notes, size=r"\footnotesize", star=True)


REPEATED_COMPACT = ["ComBat", "Ours", "Full affine", "Mean shift", "Ours (covariates only)", "Ours (LW covariance)",
                    "Ours (empirical covariance)"]


def t_repeated_compact(reps):
    keys = [(k, lab) for k, lab, _ in COHORTS if reps.get(k) is not None]
    if not keys:
        return stub("Repeated cross-validation.", "tab:repeated", "no repeated results found")
    rows = [" & ".join(["Input accuracy, PCA scores (\\%)"] + [pms(summ(entry(reps[k], "PCA scores"), "diag", "lr", "acc"))
                                                               for k, _ in keys])]
    rows.append(r"\midrule")
    for name in REPEATED_COMPACT:
        cells, e_any = [], None
        for k, _ in keys:
            e, c = entry(reps[k], name), comp(reps[k], "diag/lr/acc", name)
            e_any = e_any or e
            if c is None:
                cells.append("--")
                continue
            t = c.get("tost") or {}
            star = "$^*$" if (c.get("p_value_holm") is not None and c["p_value_holm"] < 0.05) else ""
            cells.append(f"{100 * c['mean_diff']:+.1f} [{100 * t.get('ci90_low', float('nan')):+.1f}, "
                         f"{100 * t.get('ci90_high', float('nan')):+.1f}]{star}")
        if e_any is not None:
            ref = " (vs.\\ raw)" if e_any["representation"] == "raw" else ""
            rows.append(" & ".join([label(name, e_any) + ref] + cells))
    header = "Method & " + " & ".join(lab for _, lab in keys)
    notes = ("Change in held-out diagnosis accuracy against the un-harmonized input (percentage points) with the 90\\% "
             "interval, 5$\\times$10-fold cross-validation. Intervals and tests use the Nadeau--Bengio correction for "
             "overlapping training sets \\citep{nadeau2003inference}. $^*$Holm-corrected $p<0.05$. An interval inside "
             "$\\pm$2\\,pp would certify equivalence \\citep{lakens2017tost}. $^\\dagger$Uses the diagnosis of held-out "
             "subjects. Table~\\ref{tab:repeated-full} gives accuracies, site metrics and the equivalence tests.")
    return table("Does harmonization change diagnosis accuracy? Repeated cross-validation.", "tab:repeated",
                 "l" + "c" * len(keys), header, rows, notes, size=r"\footnotesize", star=True)


def t_estimator_compact(results):
    keys = [(k, lab) for k, lab, _ in COHORTS if results.get(k) is not None and entry(results[k], "Ours") is not None]
    if not keys:
        return stub("Covariance estimators.", "tab:estimator", "no protocol results found")
    rows = []
    for name in ESTIMATORS:
        cells = []
        for k, _ in keys:
            e = entry(results[k], name)
            cells += ["--", "--"] if e is None else [f3(mean_of(e, "site", "lr", "nmi"), 2),
                                                     disp_fmt(e.get("displacement", {}).get("mean"))]
        nm = "Pooled within-site target (default)" if name == "Ours" else DISPLAY[name]
        rows.append(" & ".join([nm] + cells))
    header = _cohort_header(keys, "NMI & Disp.", 2)
    notes = ("Only the plug-in covariance of the complement changes between rows. NMI: site NMI of the LR probe. Disp.: mean "
             "squared distance a held-out subject moves (PCA scale of each cohort). The unshrunk sample covariance "
             "makes the map diverge, shown as an order of magnitude. Diagnosis accuracy and AUC are in "
             "Table~\\ref{tab:estimator-full}.")
    return table("Choice of the covariance estimator.", "tab:estimator", "l" + "cc" * len(keys), header, rows, notes,
                 size=r"\footnotesize", star=True)


DOWN_COMPACT = ["Raw", "ComBat", "PCA scores", "Mean shift", "Ours"]


LOSO_COMPACT = ["Raw", "ComBat", "ComBat (age+sex)", "PCA scores", "ComBat (PCA)", "ComBat (PCA, age+sex)",
                "Mean shift", "Ours"]
ES_COMPACT = ["PCA scores", "ComBat (PCA)", "ComBat (PCA, no covariate)", "ComBat (PCA, age+sex)", "Full affine",
              "Mean shift", "Ours"]
ES_COMPACT_SPD = ["Log-Euclidean FC", "Riemannian re-centering", "RLET"]


def t_downstream_compact(down, es_by=None):
    keys = [(k, lab) for k, lab, _ in COHORTS if down.get(k)]
    if not keys:
        return stub("Leave-one-site-out diagnosis.", "tab:downstream", "no downstream.json found")
    rows = []
    for name in LOSO_COMPACT:
        cells, flag = [], ""
        for k, _ in keys:
            m = down[k]["loso"]["methods"].get(name)
            if m is None:
                cells += ["--", "--"]
                continue
            flag = "$^\\dagger$" if m.get("uses_test_labels") else flag
            clf = list(m["summary"])[0]
            cells += [pct(m["summary"][clf]["acc"]["mean"]), pct(m["summary"][clf]["auc"]["mean"])]
        ref = " (raw)" if name in ("Raw", "ComBat", "ComBat (age+sex)") else ""
        rows.append(" & ".join([DISPLAY.get(name, name) + ref + flag] + cells))
    header = _cohort_header(keys, "Acc. & AUC", 2)
    notes = ("Leave-one-site-out diagnosis accuracy and AUC (\\%), averaged over held-out sites. The harmonizer sees the "
             "held-out site's scans, and the LR probe is trained on the other sites. $^\\dagger$ComBat receives the "
             "held-out site's diagnoses and writes each subject's fitted diagnosis effect back into its features. "
             + AGE_SEX_NOTE + "Tests and the raw-edge rows are in Table~\\ref{tab:downstream-full}.")
    return table("Leave-one-site-out diagnosis.", "tab:downstream", "l" + "cc" * len(keys), header, rows, notes,
                 size=r"\footnotesize", star=True)


def t_es_compact(es_by):
    keys = [(k, lab) for k, lab, _ in COHORTS if _es(es_by, k, "Ours")]
    if not keys:
        return stub("Effect-size reproducibility.", "tab:es", "no effect_sizes.json found")
    rows = []
    spd = [n for n in ES_COMPACT_SPD if any(_es(es_by, k, n) for k, _ in keys)]
    blocks = [("100 PCA scores", ES_COMPACT)] + ([("Matrix logarithms", spd)] if spd else [])
    for title, names in blocks:
        if spd:
            rows.append(group_row(title, 2 + len(keys)))
        for name in names:
            cells, flag, mode = [], "", None
            for k, _ in keys:
                es = _es(es_by, k, name)
                cells.append("--" if es is None else f3(es["r"]["mean"], 2))
                if es is not None:
                    mode = es.get("labels")
                    flag = "$^\\dagger$" if mode == "all" else flag
            how = {"hidden for the scored half": "hidden", "none": "none", "all": "all"}.get(mode, "--")
            rows.append(" & ".join([("\\quad " if spd else "") + DISPLAY.get(name, name) + flag, how] + cells))
    header = "Method & Labels & " + " & ".join(lab for _, lab in keys)
    where = ("on the 100 PCA scores of the full cohort, or on the matrix logarithms for the Riemannian rows, so values are "
             "comparable within a block" if spd else "on the 100 PCA scores of the full cohort")
    notes = ("Correlation of the per-feature Cohen's $d$ for diagnosis between two disjoint halves of the sites, mean over "
             f"20 random splits, {where}. Labels: with `hidden', the harmonizer never saw the diagnoses of the half it is "
             "scored on. With `none', it uses no diagnosis. With `all', its model needs every subject's diagnosis "
             "($^\\dagger$). " + AGE_SEX_NOTE + "Sign agreement and the raw-edge rows are in "
             "Table~\\ref{tab:downstream-full}.")
    return table("Effect-size reproducibility across disjoint halves of the sites.", "tab:es", "ll" + "c" * len(keys),
                 header, rows, notes, size=r"\footnotesize", star=True)


TS_ROWS = {
    "within": [("Fisher-$z$ edges", ["Raw FC", "ComBat", "CovBat", "PAM (edge residual)", "Mean shift (edge residual)"]),
               ("100 PCA scores", ["PCA scores", "ComBat (PCA)", "CORAL", "Mean shift", "PAM",
                                   "PAM (LW covariance)", "PAM (empirical covariance)"])],
    "transfer": [("Fisher-$z$ edges", ["Raw FC", "ComBat (no covariate)", "ComBat (diagnosis covariate)",
                                       "PAM (edge residual)"]),
                 ("100 PCA scores", ["PCA scores", "ComBat (PCA, no covariate)", "ComBat (PCA, diagnosis covariate)",
                                     "CORAL", "Full-space BW, marginal means", "PAM, marginal means", "PAM",
                                     "PAM (diagnosis + age + sex)", "Mean shift, marginal means, James-Stein",
                                     "PAM, marginal means, James-Stein", "PAM, James-Stein"])],
}
TS_DISPLAY = {
    "Raw FC": "No harmonization", "PCA scores": "No harmonization", "ComBat": "ComBat", "CovBat": "CovBat",
    "ComBat (PCA)": "ComBat", "CORAL": "CORAL", "Mean shift": "Mean shift", "PAM": r"\textbf{\ours}",
    "PAM (LW covariance)": r"\ours, Ledoit--Wolf covariance", "PAM (empirical covariance)": r"\ours, sample covariance",
    "ComBat (diagnosis covariate)": "ComBat$^\\ddagger$", "ComBat (PCA, diagnosis covariate)": "ComBat$^\\ddagger$",
    "ComBat (no covariate)": "ComBat, no covariate", "ComBat (PCA, no covariate)": "ComBat, no covariate",
    "Full-space BW, marginal means": "Full-space BW, marginal means", "PAM, marginal means": r"\ours, marginal means",
    "PAM (diagnosis + age + sex)": r"\ours, diagnosis + age + sex",
    "PAM (edge residual)": r"\ours, edge residual", "Mean shift (edge residual)": "Mean shift, edge residual",
    "Mean shift, marginal means, James-Stein": "Mean shift, marginal, James--Stein",
    "PAM, marginal means, James-Stein": r"\ours, marginal, James--Stein", "PAM, James-Stein": r"\ours, James--Stein",
}
TS_TAG = {
    "Raw FC": "Raw", "ComBat": "Combat", "CovBat": "Covbat", "PCA scores": "Pca", "ComBat (PCA)": "CombatPca",
    "CORAL": "Coral", "Mean shift": "Shift", "PAM": "Ours", "PAM (LW covariance)": "Lw",
    "PAM (empirical covariance)": "Emp", "ComBat (diagnosis covariate)": "Combat",
    "ComBat (no covariate)": "CombatNoCov", "ComBat (PCA, diagnosis covariate)": "CombatPca",
    "ComBat (PCA, no covariate)": "CombatPcaNoCov", "Mean shift, marginal means": "ShiftMarg",
    "Full-space BW, marginal means": "Full", "PAM, marginal means": "Marg", "Mean shift (diagnosis axes)": "Shift",
    "PAM (diagnosis + age + sex)": "AgeSex", "PAM (edge residual)": "Edge", "Mean shift (edge residual)": "ShiftEdge",
    "Mean shift, marginal means, James-Stein": "ShiftMargJs", "PAM, marginal means, James-Stein": "MargJs",
    "PAM, James-Stein": "OursJs",
}


def _pct_fine(x):
    """Percent with one decimal below 10% (0.5% must not print as 0%), whole numbers otherwise."""
    return pct(x, 1) if x is not None and x < 0.1 else pct(x, 0)


def _left(x):
    return "--" if x is None or x != x else ("$>$1000" if x > 10 else _pct_fine(x))


def _ts_cells(e, key):
    ref = e.get("reference")
    pp = (e.get("per_person") or {}).get("left")
    off = (e.get("offset") or {}).get("offset_left_sq_mean")
    pooled = e.get("scanner_ss_left")
    within = key == "within"
    return ["--" if ref else _left(pp),
            "--" if ref or within else _left(pooled),
            "--" if ref or within else _left(off),
            pct(e["identification"]["rank1"], 1),
            "--" if within else pct((e.get("site_probe") or {}).get("bacc"), 1),
            pct((e.get("traveler_probe") or {}).get("acc"), 1)]


def t_traveling(ts):
    if ts is None or "within" not in ts:
        return stub("Traveling subjects.", "tab:traveling", "results/srpbs_ts/traveling.json not found")
    rows = []
    for key, title in (("within", "(a) Fitted on eight travelers, applied to the ninth"),
                       ("transfer", "(b) Fitted on the clinical sample, applied to the travelers")):
        res = ts.get(key)
        if res is None:
            continue
        rows.append(group_row(f"{title} ({res['n_sites']} scanners, {res['n_sessions']} sessions)", 7))
        for block, names in TS_ROWS[key]:
            rows.append(group_row("\\quad " + block, 7))
            for name in names:
                e = res["methods"].get(name)
                if e is not None:
                    rows.append(" & ".join(["\\qquad " + TS_DISPLAY.get(name, name)] + _ts_cells(e, key)))
    tr = ts.get("transfer") or {}
    chance = "--"
    for e in (tr.get("methods") or {}).values():
        if e.get("site_probe"):
            chance = pct(e["site_probe"]["chance_bacc"], 1)
            break
    header = ("Method & Per person (\\%) & Pooled (\\%) & Per site (\\%) & Identification (\\%) & "
              "Site probe (\\%) & Traveler probe (\\%)")
    notes = ("Nine healthy travelers scanned at up to twelve scanners \\citep{yamashita2019harmonization}. Every session is "
             "transformed by a map that never saw that person. Per person: between-scanner variance left within each "
             "traveler (cell means around the person's mean), after harmonization relative to before, with the part "
             "that session noise produces subtracted from both. Above 100\\% the map adds variance. Pooled: scanner sum "
             "of squares of an additive scanner + person model, relative to the un-harmonized input of the same block. "
             "Per site: squared length of each scanner's remaining offset from the hub, relative to its true offset, "
             "averaged over scanners. In (a) every person is corrected with the other travelers' scanner means, which "
             "makes the pooled scanner effect and the site probe vanish by construction (Section~\\ref{sec:traveling-protocol}), "
             "so only per-person quantities are shown. Identification: the most similar session from a different "
             "scanner belongs to the same person (chance about 11\\%). Site probe: balanced accuracy of a probe trained "
             f"on eight travelers and tested on the ninth (chance {chance}\\%). Traveler probe: trained on all scanners "
             "but one, tested on the held-out scanner (chance 11\\%). In (a) the PCA basis is learned from the clinical "
             "sample. In (b) the PCA and the harmonizers are fitted on the clinical scans acquired with the travelers' "
             "protocol. $^\\ddagger$Diagnosis as a covariate (the travelers are entered as controls).")
    return table("Traveling subjects: scanner effect removed and individual differences kept.", "tab:traveling",
                 "lcccccc", header, rows, notes, star=True)


# -- number macros ---------------------------------------------------------------------------

def deepcombat_size(p: int, n_batch: int = 16, n_cov: int = 1) -> tuple[list[int], int]:
    """Default DeepComBat architecture (package rule: latent = nearest power of 2 to p/4, three hidden layers)."""
    import math
    lat = 2 ** round(math.log2(p / 4))
    dims = [p] + [lat + math.floor((p - lat) * (3 - i + 1) / 4) for i in range(1, 4)] + [lat]
    n = len(dims)
    enc = ((dims[0] + n_batch + n_cov) * dims[1] + dims[1] + sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(1, n - 2))
           + 2 * (dims[n - 2] * dims[n - 1] + dims[n - 1]))
    dec = ((dims[n - 1] + n_batch + n_cov) * dims[n - 2] + dims[n - 2]
           + sum(dims[i] * dims[i - 1] + dims[i - 1] for i in range(n - 2, 0, -1)))
    return dims, enc + dec


def _count_words(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f} billion"
    if n >= 1e6:
        return f"{n / 1e6:.0f} million"
    return f"{n / 1e3:.0f} thousand"


def numbers_tex(results, reps, down, fp, net, pa, unseen, federated, overlap, grids, vaes, whiten, sim, thm, est,
                ts=None, es_by=None, disp_by=None, fp_edge_by=None, sitevar_by=None, spd_sens=None, noise=None,
                metric_check=None, minutes=None):
    M: dict[str, str] = {}

    def put(name, val):
        if val is not None:
            M[name] = str(val)

    put("DeepParamsRaw", _count_words(deepcombat_size(19900)[1]))
    put("DeepParamsPca", _count_words(deepcombat_size(100)[1]))
    put("DeepDimsPca", "--".join(str(d) for d in deepcombat_size(100)[0]))

    if metric_check:
        for ratio, slug in (("0.0", "Shared"), ("1.0", "Equal"), ("4.0", "Specific")):
            c = (metric_check.get("settings") or {}).get(ratio)
            if c:
                put(f"TsSimPooled{slug}", pct(c["pooled_left"], 1))
                put(f"TsSimPerPerson{slug}", pct(c["per_person_left"], 0))
    for key, val in (minutes or {}).items():
        put(key, val)
    if ts and ts.get("seconds"):
        put("TsMinutes", f"{ts['seconds'] / 60:.0f}")
    if noise:
        put("TsNoiseShare", pct(noise["noise_share_of_discrepancy"], 0))
        put("TsNoiseClin", pct(noise["clinical_noise_share_of_discrepancy"], 0))
        put("TsNoiseTrav", pct(noise["noise_share_of_discrepancy"] - noise["clinical_noise_share_of_discrepancy"], 0))
        put("TsNClinAtt", (noise.get("by_site") or {}).get("ATT", {}).get("n_clinical"))
    for key, ktag in (("within", "Within"), ("transfer", "Transfer"), ("transfer_all", "TransferAll"),
                      ("within_refit", "Refit")):
        res = (ts or {}).get(key)
        if not res:
            continue
        put(f"Ts{ktag}NSessions", res["n_sessions"])
        put(f"Ts{ktag}NSites", res["n_sites"])
        if "n_clinical" in res:
            put(f"Ts{ktag}NClinical", res["n_clinical"])
        for name, e in res["methods"].items():
            mt = TS_TAG.get(name)
            if mt is None:
                continue
            if name == "Mean shift (diagnosis axes)":
                mt = "ShiftDx"
            left = e.get("scanner_ss_left")
            if left is not None and not e.get("reference"):
                put(f"Ts{ktag}{mt}Left", "$>$1000" if left > 10 else _pct_fine(left))
                put(f"Ts{ktag}{mt}Removed", _pct_fine(1 - left) if left <= 1 else "0")
            put(f"Ts{ktag}{mt}TravShare", pct(e["shares"]["traveler_share"], 1))
            put(f"Ts{ktag}{mt}SiteShare", pct(e["shares"]["site_share"], 1))
            put(f"Ts{ktag}{mt}Ident", pct(e["identification"]["rank1"], 1))
            if e.get("site_probe"):
                put(f"Ts{ktag}{mt}SiteProbe", pct(e["site_probe"]["bacc"], 1))
                put(f"Ts{ktag}SiteChance", pct(e["site_probe"]["chance_bacc"], 1))
            if e.get("traveler_probe"):
                put(f"Ts{ktag}{mt}TravProbe", pct(e["traveler_probe"]["acc"], 1))
            off = e.get("offset") or {}
            pp = (e.get("per_person") or {}).get("left")
            if pp is not None and pp == pp and not e.get("reference"):
                put(f"Ts{ktag}{mt}PerPerson", _left(pp))
                put(f"Ts{ktag}{mt}PerPersonRemoved", _pct_fine(1 - pp) if pp <= 1 else "0")
            osq = off.get("offset_left_sq_mean")
            if osq is not None and osq == osq and not e.get("reference"):
                put(f"Ts{ktag}{mt}OffLeft", _left(osq))
            if off.get("mean") is not None and off.get("mean") == off.get("mean"):
                put(f"Ts{ktag}{mt}Cos", f3(off["mean"], 2))
            if off.get("correction_size_mean") is not None and off["correction_size_mean"] == off["correction_size_mean"]:
                put(f"Ts{ktag}{mt}CorrSize", pct(off["correction_size_mean"], 0))
        if key == "within":
            chance = next((e["identification"]["chance"] for e in res["methods"].values()), None)
            put("TsIdentChance", pct(chance, 0))
        lefts = [e["scanner_ss_left"] for e in res["methods"].values()
                 if not e.get("reference") and e.get("scanner_ss_left") is not None]
        if lefts:
            put(f"Ts{ktag}BestRemoved", pct(1 - min(lefts), 0))
        pps = [(e.get("per_person") or {}).get("left") for e in res["methods"].values() if not e.get("reference")]
        pps = [v for v in pps if v is not None and v == v]
        if pps:
            put(f"Ts{ktag}BestPerPersonRemoved", pct(1 - min(pps), 0))
            put(f"Ts{ktag}BestPerPersonLeft", pct(min(pps), 0))
        offs = [(e.get("offset") or {}).get("offset_left_sq_mean") for e in res["methods"].values() if not e.get("reference")]
        offs = [v for v in offs if v is not None and v == v]
        if offs:
            put(f"Ts{ktag}BestOffRemoved", pct(1 - min(offs), 0))
        an = res.get("anova") or {}
        for space, stag in (("pca", ""), ("raw", "Raw")):
            a = an.get(space)
            if a:
                put(f"Ts{ktag}IntShare{stag}", pct(a["interaction_share"], 0))
                put(f"Ts{ktag}ScanShare{stag}", pct(a["site_share"], 0) if a["site_share"] >= 0.1 else pct(a["site_share"], 1))
                put(f"Ts{ktag}PersonShare{stag}", pct(a["traveler_share"], 0))
                put(f"Ts{ktag}FInt{stag}", f3(a["F_interaction"], 2))
                put(f"Ts{ktag}DfInt{stag}", a["df_interaction"])
                put(f"Ts{ktag}DfWithin{stag}", a["df_within"])
                put(f"Ts{ktag}IntVsSite{stag}", pct(a["interaction_vs_site"], 0))

    def put_method(prefix, e, tag, res=None, name=None):
        d = e.get("oos_distribution", {})
        put(f"{prefix}DiagAcc{tag}", pct(mean_of(e, "diag", "lr", "acc")))
        put(f"{prefix}DiagAuc{tag}", pct(mean_of(e, "diag", "lr", "auc")))
        put(f"{prefix}SiteAcc{tag}", pct(mean_of(e, "site", "lr", "acc")))
        put(f"{prefix}SiteAccRf{tag}", pct(mean_of(e, "site", "rf", "acc")))
        put(f"{prefix}SiteNmi{tag}", f3(mean_of(e, "site", "lr", "nmi"), 2))
        put(f"{prefix}SiteNmiRf{tag}", f3(mean_of(e, "site", "rf", "nmi"), 2))
        put(f"{prefix}Icc{tag}", f3(d.get("icc_mean"), 3))
        put(f"{prefix}Mmd{tag}", f3(d.get("mmd2_unbiased_mean"), 4))
        put(f"{prefix}MmdZ{tag}", f3(mmd_z(e), 1))
        put(f"{prefix}Disp{tag}", f3(e.get("displacement", {}).get("mean"), 0))
        if e.get("protected_eps2"):
            put(f"{prefix}EpsPre{tag}", f3(e["protected_eps2_pre"]["mean"]))
            put(f"{prefix}EpsPost{tag}", f3(e["protected_eps2"]["mean"]))
        for cname, ctag in (("age", "Age"), ("sex", "Sex"), ("mean_fd", "Fd")):
            cv = (e.get("covariates") or {}).get(cname)
            if cv:
                v = cv.get("r") or cv.get("auc")
                put(f"{prefix}{ctag}{tag}", f3(v.get("mean"), 2))
        if res is not None:
            c = comp(res, "diag/lr/acc", name)
            if c:
                put(f"{prefix}P{tag}", pfmt(c.get("p_value_holm")))
                put(f"{prefix}PRaw{tag}", pfmt(c.get("p_value")))
                put(f"{prefix}Dz{tag}", f3(c["effect_size_dz"], 2))
            c = comp(res, "diag/lr/auc", name)
            if c:
                put(f"{prefix}PAuc{tag}", pfmt(c.get("p_value_holm")))

    for key, _, tag in COHORTS:
        r = results.get(key)
        if r is not None:
            pr = r["protocol"]
            put(f"n{tag}", pr["n_subjects"])
            put(f"nSites{tag}", pr["n_sites"])
            put(f"siteMaj{tag}", pct(r["chance"]["site"]["majority_rate"]))
            put(f"siteUnif{tag}", pct(r["chance"]["site"]["uniform_chance"]))
            put(f"diagMaj{tag}", pct(r["chance"]["diag"]["majority_rate"]))
            put(f"hub{tag}", str(r.get("run", {}).get("hub_site", "")).replace("_", r"\_"))
            put(f"eps2Null{tag}", f"{(pr['n_sites'] - 1) / max(pr['n_subjects'] - 1, 1):.2f}")
            put(f"mmdMinSite{tag}", pr.get("mmd_min_site", 5))
            folds = r.get("run", {}).get("fold_pca")
            if folds:
                put(f"pcaVar{tag}", f"{100 * np.mean([f['explained_variance_ratio_total'] for f in folds]):.0f}")
            for name, mtag in MTAG.items():
                e = entry(r, name)
                if e is not None:
                    put_method(mtag, e, tag, r, name)
            fam = next((c["vs_reference"].get("holm_family_size") for c in r["comparisons"]["diag/lr/acc"].values()
                        if "vs_reference" in c), None)
            put(f"holmFamily{tag}", fam)
        rep = reps.get(key)
        if rep is not None:
            for name, mtag in MTAG.items():
                c = comp(rep, "diag/lr/acc", name)
                e = entry(rep, name)
                if c is None or e is None:
                    continue
                t = c.get("tost") or {}
                put(f"Rep{mtag}DiagAcc{tag}", pct(mean_of(e, "diag", "lr", "acc")))
                put(f"Rep{mtag}Diff{tag}", f"{100 * c['mean_diff']:+.1f}")
                put(f"Rep{mtag}Lo{tag}", f"{100 * t.get('ci90_low', float('nan')):+.1f}")
                put(f"Rep{mtag}Hi{tag}", f"{100 * t.get('ci90_high', float('nan')):+.1f}")
                put(f"Rep{mtag}P{tag}", pfmt(c.get("p_value_holm")))
                put(f"Rep{mtag}SiteAcc{tag}", pct(mean_of(e, "site", "lr", "acc")))
                put(f"Rep{mtag}SiteNmi{tag}", f3(mean_of(e, "site", "lr", "nmi"), 2))
        dd = down.get(key)
        if dd:
            put(f"nLoso{tag}", len(dd["loso"]["sites"]))
            for name, mtag in MTAG.items():
                m = dd["loso"]["methods"].get(name)
                if m:
                    clf = list(m["summary"])[0]
                    put(f"{mtag}Loso{tag}", pct(m["summary"][clf]["acc"]["mean"]))
                    put(f"{mtag}LosoAuc{tag}", pct(m["summary"][clf]["auc"]["mean"]))
                es = dd["effect_size_reproducibility"].get(name)
                if es:
                    put(f"{mtag}EsInSample{tag}", f3(es["r"]["mean"], 2))
        for name, e in ((((es_by or {}).get(key) or {}).get("methods")) or {}).items():
            mtag = MTAG.get(name)
            if mtag is not None and e.get("r"):
                put(f"{mtag}Es{tag}", f3(e["r"]["mean"], 2))
                put(f"{mtag}EsSign{tag}", f3((e.get("sign_agreement") or {}).get("mean"), 2))
        ff = fp.get(key)
        if ff:
            for name, mtag in [("Unharmonized", "Pca")] + list(MTAG.items()):
                c = ff["conditions"].get(name)
                if c:
                    put(f"{mtag}FpAll{tag}", pct(c["all"]["rank1"]))
                    put(f"{mtag}FpWithin{tag}", pct(c["within_site"]["rank1_weighted"]))
        nn = net.get(key)
        if nn:
            put(f"NetRawEta{tag}", f3(nn["conditions"]["Raw FC"]["site_eta2"]["mean"]))
            for cname, ctag in (("PCA reconstruction", "Recon"), ("Decoded Ours", "Ours"), ("ComBat", "Combat")):
                c = nn["conditions"].get(cname)
                if c and "vs_reference" in c:
                    put(f"Net{ctag}R{tag}", f3(c["vs_reference"]["pearson_r"], 3))
                    put(f"Net{ctag}SubjR{tag}", f3(c["per_subject_r"]["mean"], 3))
                    put(f"Net{ctag}Eta{tag}", f3(c["site_eta2"]["mean"]))
        pp = pa.get(key)
        if pp:
            put(f"ProtShare{tag}", pct(pp["between_site_ss"]["protected_share_pre"]))
            put(f"ProtKd{tag}", pct(pp["k_protected"] / pp["latent_dim"]))
            put(f"ProtEpsPre{tag}", f3(pp["axes"][0]["site_eps2_pre"]))
            put(f"ProtEpsPost{tag}", f3(pp["axes"][0]["site_eps2_post"]))
        uu = unseen.get(key)
        if uu:
            for n, slug in (("10", "Ten"), ("20", "Twenty"), ("30", "Thirty"), ("50", "Fifty"), ("0.75", "ThreeQ")):
                cell = (uu.get("by_size") or {}).get(n)
                if cell and cell.get("n"):
                    put(f"Unseen{slug}Diag{tag}", pct(cell["diag_acc"]["mean"]))
                    put(f"Unseen{slug}Site{tag}", pct(cell["heldout_vs_rest_acc"]["mean"]))
                    if cell.get("heldout_vs_rest_acc_unharmonized"):
                        put(f"Unseen{slug}SiteNone{tag}", pct(cell["heldout_vs_rest_acc_unharmonized"]["mean"]))
                        put(f"Unseen{slug}DiagNone{tag}", pct(cell["diag_acc_unharmonized"]["mean"]))
        dsp = (disp_by or {}).get(key)
        if dsp:
            for name, mtag in MTAG.items():
                e = entry(dsp, name)
                if e is not None and e.get("displacement_plugin"):
                    put(f"{mtag}DispFit{tag}", f3(e["displacement_plugin"]["mean"], 0))
                    put(f"{mtag}DispHeld{tag}", f3(e["displacement"]["mean"], 0))
        fe = (fp_edge_by or {}).get(key)
        if fe:
            for name, mtag in (("Unharmonized", "Raw"), ("Ours (edge residual)", "Edge")):
                c = fe["conditions"].get(name)
                if c:
                    put(f"{mtag}FpAll{tag}", pct(c["all"]["rank1"]))
                    put(f"{mtag}FpWithin{tag}", pct(c["within_site"]["rank1_weighted"]))
        sv = (sitevar_by or {}).get(key)
        if sv:
            put(f"SiteVarPcs{tag}", pct(sv["between_site_share_in_pcs"], 0))
            put(f"TotVarPcs{tag}", pct(sv["total_variance_share_in_pcs"], 0))
        fe = federated.get(key)
        if fe:
            put(f"FedAgree{tag}", "yes" if fe.get("agree") else "no")
            put(f"FedMax{tag}", sci(fe.get("max_abs_diff", 0.0), nd=0))
        ov = overlap.get(key)
        if ov:
            for cname, ctag in (("age", "Age"), ("sex", "Sex"), ("mean_fd", "Fd")):
                rec = (ov.get("covariates") or {}).get(cname)
                if rec:
                    put(f"{ctag}Overlap{tag}", f3(rec.get("mean_overlap"), 2))
                    put(f"{ctag}OverlapMin{tag}", f3(rec.get("min_overlap"), 2))
        g = grids.get(key)
        if g and g.get("cells"):
            cells = g["cells"]
            put(f"SensDims{tag}", ", ".join(str(c["pca_dim"]) for c in cells))
            put(f"SensOursDiagMin{tag}", pct(min(c["ours"]["diag_acc"] for c in cells)))
            put(f"SensOursDiagMax{tag}", pct(max(c["ours"]["diag_acc"] for c in cells)))
            put(f"SensOursNmiMin{tag}", f3(min(c["ours"]["site_nmi"] for c in cells), 2))
            put(f"SensOursNmiMax{tag}", f3(max(c["ours"]["site_nmi"] for c in cells), 2))
            put(f"SensOursSiteMax{tag}", pct(max(c["ours"]["site_acc"] for c in cells)))
        v = vaes.get(key)
        if v:
            for name, mtag in (("VAE latent", "Lat"), ("Ours", "Ours"), ("Mean shift", "Shift"), ("Full affine", "Full")):
                e = entry(v, name)
                if e is not None:
                    put_method(f"Vae{mtag}", e, tag)
    svs = [v for v in (sitevar_by or {}).values() if v]
    if svs:
        put("SiteVarPcsMin", pct(min(v["between_site_share_in_pcs"] for v in svs), 0))
        put("SiteVarPcsMax", pct(max(v["between_site_share_in_pcs"] for v in svs), 0))
        put("TotVarPcsMin", pct(min(v["total_variance_share_in_pcs"] for v in svs), 0))
        put("TotVarPcsMax", pct(max(v["total_variance_share_in_pcs"] for v in svs), 0))
    for slug, r in (spd_sens or {}).items():
        if r is None:
            continue
        for name, mtag in (("Log-Euclidean FC", "Ref"), ("Riemannian re-centering", "Riem"), ("RLET", "Rlet")):
            e = entry(r, name)
            if e is not None:
                put(f"Spd{slug}{mtag}SiteAcc", pct(mean_of(e, "site", "lr", "acc")))
                put(f"Spd{slug}{mtag}SiteNmi", f3(mean_of(e, "site", "lr", "nmi"), 2))
                put(f"Spd{slug}{mtag}DiagAcc", pct(mean_of(e, "diag", "lr", "acc")))
    present = [results[k]["protocol"] for k, _, _ in COHORTS if results.get(k) is not None]
    if present:
        put("nTotal", f"{sum(p['n_subjects'] for p in present):,}")
        put("nSitesTotal", sum(p["n_sites"] for p in present))
        put("nCohorts", {3: "three", 4: "four", 5: "five", 6: "six"}.get(len(present), len(present)))
    if whiten is not None:
        for name, mtag in (("PCA scores", "Pca"), ("Ours", "Ours"), ("Full affine", "Full"), ("Mean shift", "Shift")):
            e = entry(whiten, name)
            if e is not None:
                put_method(f"Wh{mtag}", e, "A")

    # cohort composition from the HDF5 files
    try:
        import h5py
        from vae_harmonization.data_registry import DATASET_REGISTRY as REG
        for key, wanted in (("adhd200", ("Pittsburgh", "WashU")), ("abide2_10site", ("KUL_3", "NYU")),
                            ("srpbs", ("KUT", "HUH2"))):
            path = Path(REG[key]["hdf5_path"])
            if not path.exists():
                continue
            with h5py.File(path, "r") as f:
                s, d = f["y_site"][:], f["y_diagnosis"][:]
                names = [str(x) for x in f.attrs["site_names"]]
            if key == "srpbs":
                with h5py.File(path, "r") as f:
                    if "n_dropped_missing_parcels" in f.attrs:
                        put("SrpbsNanN", int(f.attrs["n_dropped_missing_parcels"]))
            for name in wanted:
                if name in names:
                    sid = names.index(name)
                    n, n_case = int((s == sid).sum()), int(((s == sid) & (d == 1)).sum())
                    slug = "".join(ch for ch in name if ch.isalpha())
                    put(f"{slug}N", n)
                    put(f"{slug}CasePct", f"{100 * n_case / max(n, 1):.0f}")
    except Exception as exc:  # pragma: no cover
        print("cohort macros skipped:", exc)

    if sim:
        s = sim["settings"]
        put("SimNSites", s["n_sites"])
        put("SimDim", s["d"])
        put("SimNPerSite", s["n_per_site"])
        put("SimNHub", s["hub_size"])
        put("SimNRep", s["n_rep"])
        for k, t in (("star_target", "Star"), ("star_barycenter", "Bary"), ("chain", "Chain"), ("none", "None")):
            put(f"Sim{t}W", f"{sim['results'][k]['w2_to_hub_pop']['mean']:.1f}")
            put(f"Sim{t}Disp", f"{sim['results'][k]['displacement']['mean']:.1f}")
    if thm:
        put("ThmTrials", thm["n_trials"])
        put("ThmComp", thm["n_competitors_per_trial"])
        put("ThmDim", thm["d"])
        put("ThmNeg", thm["n_negative_gaps"])
        put("ThmMinGap", sci(thm["gap_min"]))
        put("ThmResid", sci(thm.get("competitor_constraint_residual_max", 0.0), nd=0))
    if est:
        summary = est["summary"]
        sizes = est["settings"]["sizes"][1:]
        for regime, rtag in (("shared", "Shared"), ("distinct", "Distinct")):
            for e_name, etag in (("pooled", "Pooled"), ("lw", "Lw"), ("diagonal", "Diag")):
                vals = [summary[regime][e_name][str(n)]["w2_post"] for n in sizes]
                put(f"Est{rtag}{etag}", f"{np.mean(vals):.1f}")
            put(f"Est{rtag}Pre", f"{np.mean([summary[regime]['oracle'][str(n)]['w2_pre'] for n in sizes]):.1f}")
        put("EstMinN", min(sizes))
        put("EstDim", est["settings"]["d"])

    lines = ["% generated by scripts/generate_tables.py -- do not edit"]
    for k in sorted(M):
        name = "".join(ch for ch in k if ch.isalpha())
        lines.append(f"\\newcommand{{\\{name}}}{{{M[k]}}}")
    return "\n".join(lines) + "\n"


# -- main ------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default=str(REPO / "results"))
    p.add_argument("--out-dir", default=str(REPO / "paper" / "tables"))
    a = p.parse_args(argv)
    R, out = Path(a.results_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results, reps = {}, {}
    for key, _, _ in COHORTS:
        results[key], reps[key] = load_cohort(R, key)

    def per_cohort(fname, suffix=""):
        return {k: load(R / f"{k}{suffix}" / fname) for k, _, _ in COHORTS}

    down, fp = per_cohort("downstream.json"), per_cohort("fingerprinting.json")
    # label-free ComBat (age and sex as covariates) was run later on the same folds (tag combatas)
    for k, extra in per_cohort("downstream.json", "_combatas").items():
        d = down.get(k)
        if not (d and extra):
            continue
        if [s["site"] for s in d["loso"]["sites"]] != [s["site"] for s in extra["loso"]["sites"]]:
            print(f"warning: {k}_combatas LOSO used other held-out sites; not merged")
            continue
        for name, v in extra["loso"]["methods"].items():
            d["loso"]["methods"].setdefault(name, v)
        for clf, comps in (extra["loso"].get("comparisons") or {}).items():
            for name, v in comps.items():
                d["loso"]["comparisons"].setdefault(clf, {}).setdefault(name, v)
        for name, v in (extra.get("effect_size_reproducibility") or {}).items():
            d.setdefault("effect_size_reproducibility", {}).setdefault(name, v)
    net, pa = per_cohort("network_preservation.json"), per_cohort("protected_axis.json")
    unseen, federated, overlap = per_cohort("unseen_site.json"), per_cohort("federated.json"), per_cohort("overlap.json")
    grids = per_cohort("pca_sweep.json", "_sensitivity")
    vaes = per_cohort("protocol_results.json", "_vae")
    whiten = load(R / "abide1_full_whiten" / "protocol_results.json")
    sim = load(R / "simulation" / "star_vs_chain.json")
    thm = load(R / "theorem" / "theorem_validation.json")
    est = load(R / "theorem" / "estimator_check.json")
    ts = load(R / "srpbs_ts" / "traveling.json")
    noise = load(R / "srpbs_ts" / "noise_budget.json")
    metric_check = load(R / "srpbs_ts" / "metric_check.json")
    minutes = {}
    import re as _re
    for qlog in (R / "queue_review.log", R / "queue_geo.log"):     # wall times written by the run queues
        if not qlog.exists():
            continue
        for line in qlog.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
            m = _re.search(r"done (geometric|deep|edge)_abide1_full in ([0-9.]+) min", line)
            if m:
                minutes[{"geometric": "GeoMinutesA", "deep": "DeepMinutesA", "edge": "EdgeMinutesA"}[m.group(1)]] = \
                    f"{float(m.group(2)):.0f}"
    es_by = per_cohort("effect_sizes.json", "_es")
    # the Riemannian baselines on the log-Euclidean vectors come from a separate run (tag es_spd)
    for k, e in per_cohort("effect_sizes.json", "_es_spd").items():
        if e and es_by.get(k):
            for name, v in e["methods"].items():
                if v.get("representation") == "spd":
                    es_by[k]["methods"].setdefault(name, v)
    for k, e in per_cohort("effect_sizes.json", "_combatas").items():
        if e and es_by.get(k):
            for name, v in e["methods"].items():
                es_by[k]["methods"].setdefault(name, v)
    unseen_new = per_cohort("unseen_site.json", "_unseen")
    unseen = {k: unseen_new.get(k) or unseen.get(k) for k in unseen}
    disp_by = {k: load(R / f"{k}_disp" / "protocol_results.json") for k, _, _ in COHORTS}
    fp_edge_by = per_cohort("fingerprinting_edge.json", "_edge")
    sitevar_by = per_cohort("site_variance_pcs.json")
    spd_sens = {"Low": load(R / "abide1_full_spd001" / "protocol_results.json"),
                "High": load(R / "abide1_full_spd05" / "protocol_results.json")}

    tables = {
        "table_traveling.tex": t_traveling(ts),
        "table_cohorts.tex": t_cohorts_compact(),
        "table_cohorts_full.tex": t_cohorts(),
        "table_main.tex": t_main(results["abide1_full"]),
        "table_cross_dataset.tex": t_cross_compact(results),
        "table_cross_dataset_full.tex": t_cross(results),
        "table_ablation.tex": t_ablation(results["abide1_full"], whiten, disp_by.get("abide1_full")),
        "table_estimator.tex": t_estimator_compact(results),
        "table_estimator_full.tex": t_estimator(results),
        "table_repeated.tex": t_repeated_compact(reps),
        "table_repeated_full.tex": t_repeated(reps),
        "table_downstream_full.tex": t_downstream(down, es_by),
        "table_permutation.tex": t_permutation(results),
        "table_protected_axis.tex": t_protected(pa, results),
        "table_covariates.tex": t_covariates(results["abide1_full"]),
        "table_overlap.tex": t_overlap(overlap),
        "table_fingerprint.tex": t_fingerprint(fp, fp_edge_by),
        "table_network.tex": t_network(net),
        "table_downstream.tex": t_downstream_compact(down, es_by),
        "table_es.tex": t_es_compact(es_by),
        "table_unseen.tex": t_unseen(unseen),
        "table_simulation.tex": t_simulation(sim),
        "table_sensitivity.tex": t_sensitivity(grids),
        "table_vae.tex": t_vae(vaes, results),
        "numbers.tex": numbers_tex(results, reps, down, fp, net, pa, unseen, federated, overlap, grids, vaes, whiten,
                                   sim, thm, est, ts, es_by=es_by, disp_by=disp_by, fp_edge_by=fp_edge_by,
                                   sitevar_by=sitevar_by, spd_sens=spd_sens, noise=noise,
                                   metric_check=metric_check, minutes=minutes),
    }
    for name, tex in tables.items():
        (out / name).write_text(tex, encoding="utf-8")
        print("wrote", out / name)


if __name__ == "__main__":
    main()
