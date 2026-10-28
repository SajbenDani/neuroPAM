"""
run_vae_harmonization.py -- Command-line entry point.

Representation
--------------
``--representation pca`` (default)
    Standardized FC edges -> first ``pca_dim`` principal components.  Inside the
    protocol the scaler and PCA are re-fitted on every training fold, so no held-out
    subject touches any fitted quantity.  Deterministic, CPU only, seconds per fold.
``--representation vae``
    The frozen FC-VAE checkpoint of the cohort (appendix ablation).  Scaler, PCA and
    VAE are those of the original training run (fitted on the 85% training split);
    ``--train-vae`` re-trains instead of loading.

Stages
------
0. data      load the cohort HDF5 (+ severity and covariates when available)
1. model     PCA: fit the label-free full-cohort PCA used by the auxiliary analyses;
             VAE: load / train the checkpoint and run the in-sample diagnostics
2. protocol  the leakage-free per-fold evaluation of every method  <- primary numbers
3. analyses  optional: --downstream, --fingerprint, --network-analysis, --protected-axis

Examples
--------
    python -m vae_harmonization.run_vae_harmonization --dataset abide1_full
    python -m vae_harmonization.run_vae_harmonization --dataset adhd200 --methods all --deep-epochs 200
    python -m vae_harmonization.run_vae_harmonization --dataset abide1_full --representation vae --run-tag vae
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent.parent
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from vae_harmonization.baselines import (  # noqa: E402
    ComBatGAMHarmonizer, ComBatHarmonizer, CORALHarmonizer, CovBatHarmonizer,
    DeepComBatHarmonizer, EuclideanAlignmentHarmonizer, MMDAffineHarmonizer,
    RELIEFHarmonizer, RiemannianProcrustesHarmonizer, RiemannianRecenteringHarmonizer,
    RLETHarmonizer, SLACCHarmonizer, deepcombat_available, log_euclidean_vectors, rscript_available,
)
from vae_harmonization.harmonization.edge import EdgeResidualHarmonizer  # noqa: E402
from vae_harmonization.config import VAEConfig, site_palette  # noqa: E402
from vae_harmonization.data_registry import get_dataset_config, list_datasets  # noqa: E402
from vae_harmonization.evaluation import MethodSpec, run_protocol, summary_table, to_jsonable  # noqa: E402
from vae_harmonization.evaluation.plots import plot_method_bars  # noqa: E402
from vae_harmonization.evaluation.representations import PCARepresentation, make_pca_representation_fn  # noqa: E402
from vae_harmonization.harmonization import IdentityHarmonizer, make_variant  # noqa: E402
from vae_harmonization.logging_utils import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)

# -- method catalogue ------------------------------------------------------------------
# Raw-space methods act on the Fisher-z FC vectors; "representation-space" methods act on the
# PCA scores (default) or the VAE latent (--representation vae).

BASELINES = ["raw", "combat", "covbat"]
DEEP = ["deepcombat"]
REFERENCE = ["rep"]
OURS = ["ours"]
ABLATIONS = ["full-affine", "mean-shift", "cholesky", "zca", "ours-marginal", "ours-lw", "ours-diag", "ours-emp",
             "ours-cov", "combat-rep", "covbat-rep"]
EDGE = ["pam-edge"]
# label-free ComBat: age and sex as the only covariates (no diagnosis), on the raw edges and on the PCA scores
COMBAT_AGE_SEX = ["combat-as", "combat-rep-as"]
EXTENSIONS = ["ours-auto", "barycenter", "ours-2d", "ours-demo"]
GEOMETRIC = ["spd-ref", "riemannian", "ea", "rpa", "rlet", "coral", "mmd-affine"]
SPD_METHODS = {"spd-ref", "riemannian", "ea", "rpa", "rlet"}   # act on the log-Euclidean SPD representation
# ComBat-GAM is not in the geometric group. neuroHarmonize fits a GAM per feature,
# and on the raw edges that does not finish on CPU. Run it on the PCA scores with
# --methods combat-gam.
# SLACC is requestable on its own (--methods slacc). A full-atlas fit takes about an
# hour on a desktop CPU and the protocol calls it twice per fold, so it is not in a group.
R_BASELINES = ["relief"]
GROUPS = {
    "default": BASELINES + REFERENCE + OURS + ABLATIONS,
    "baselines": BASELINES,
    "deep": DEEP,
    "ablations": ABLATIONS,
    "extensions": EXTENSIONS,
    "geometric": GEOMETRIC,
    "r-baselines": R_BASELINES,
    "edge": EDGE,
    "combat-age-sex": COMBAT_AGE_SEX,
    "all": BASELINES + COMBAT_AGE_SEX + DEEP + REFERENCE + OURS + ABLATIONS + EXTENSIONS + GEOMETRIC + R_BASELINES,
}


def _expand_methods(spec: str) -> list[str]:
    out: list[str] = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        items = GROUPS.get(tok, [tok])
        for it in items:
            if it not in out:
                out.append(it)
    return out


def build_method_specs(names: list[str], cfg: VAEConfig, *, have_severity: bool, covariate_keys: list[str],
                       deep_epochs: int, device: str, space: str = "pca") -> list[MethodSpec]:
    """
    ``space`` is the representation key of the proposed method and its ablations:
    'pca' (PCA scores, default) or 'latent' (VAE posterior means).
    """
    t = int(cfg.target_site)
    common = dict(min_cov_samples=cfg.min_cov_samples, svm_C=cfg.bio_svm_C, ridge=cfg.ridge, seed=cfg.seed,
                  covariance=cfg.covariance)
    S = "PCA" if space == "pca" else "latent"
    ref_name = "PCA scores" if space == "pca" else "VAE latent"
    ref_note = (f"{cfg.pca_dim}-D PCA scores, no harmonization" if space == "pca"
                else f"{cfg.latent_dim}-D posterior means, no harmonization")

    def ours(variant, name, **kw):
        params = {**common, **kw}
        return lambda: make_variant(variant, t, name=name, **params)

    specs: list[MethodSpec] = []
    for n in names:
        if n == "raw":
            specs.append(MethodSpec("Raw", "raw", IdentityHarmonizer, fit_aux=(), is_reference=True,
                                    notes="Fisher-z FC vectors, no harmonization"))
        elif n == "combat":
            specs.append(MethodSpec("ComBat", "raw", lambda: ComBatHarmonizer(protect_diagnosis=True),
                                    transform_uses_labels=True, notes="neuroCombat, diagnosis covariate"))
        elif n == "covbat":
            specs.append(MethodSpec("CovBat", "raw", lambda: CovBatHarmonizer(protect_diagnosis=True, seed=cfg.seed),
                                    transform_uses_labels=True, notes="ComBat + ComBat on PC scores of residuals"))
        elif n == "deepcombat":
            if not deepcombat_available():
                log.warning("DeepComBat requested but the R package is not installed (third_party/DeepComBat); skipped")
                continue
            specs.append(MethodSpec("DeepComBat", space,
                                    lambda: DeepComBatHarmonizer(target_site=t, seed=cfg.seed),
                                    fit_aux=("y_diag",), transform_uses_labels=True,
                                    notes="authors' R package (vendored, commit bf0eda1) on the PCA scores, "
                                          "diagnosis covariate, hub as reference batch"))
        elif n == "rep":
            specs.append(MethodSpec(ref_name, space, IdentityHarmonizer, fit_aux=(), is_reference=True, notes=ref_note))
        elif n == "ours":
            specs.append(MethodSpec("Ours", space, ours("ours", "Ours"), is_proposed=True,
                                    notes=f"protected diagnosis axis + BW map on complement, {cfg.covariance} covariances"))
        elif n == "full-affine":
            specs.append(MethodSpec("Full affine", space, ours("full-affine", "Full affine"), fit_aux=(),
                                    notes="BW map in the full space, no protected axis"))
        elif n == "mean-shift":
            specs.append(MethodSpec("Mean shift", space, ours("mean-shift", "Mean shift"),
                                    notes="protected axis + mean shift only on complement"))
        elif n == "cholesky":
            specs.append(MethodSpec("Cholesky map", space, ours("cholesky", "Cholesky map"),
                                    notes="covariance-matching but not displacement-minimal"))
        elif n == "zca":
            specs.append(MethodSpec("ZCA map", space, ours("zca", "ZCA map"), notes="whiten-then-colour map"))
        elif n == "ours-marginal":
            specs.append(MethodSpec("Ours (marginal means)", space, ours("ours-marginal", "Ours (marginal means)"),
                                    notes="site means not adjusted for the protected covariates"))
        elif n == "ours-lw":
            specs.append(MethodSpec("Ours (LW covariance)", space, ours("ours-lw", "Ours (LW covariance)"),
                                    notes="Ledoit-Wolf shrinkage toward a scaled identity"))
        elif n == "ours-diag":
            specs.append(MethodSpec("Ours (diagonal target)", space, ours("ours-diag", "Ours (diagonal target)"),
                                    notes="shrinkage toward the site's own diagonal"))
        elif n == "ours-emp":
            specs.append(MethodSpec("Ours (empirical covariance)", space,
                                    ours("ours-emp", "Ours (empirical covariance)"), notes="no shrinkage"))
        elif n == "combat-rep":
            specs.append(MethodSpec(f"ComBat ({S})", space, lambda: ComBatHarmonizer(protect_diagnosis=True),
                                    transform_uses_labels=True, notes=f"ComBat applied to the {S} coordinates"))
        elif n == "covbat-rep":
            specs.append(MethodSpec(f"CovBat ({S})", space,
                                    lambda: CovBatHarmonizer(protect_diagnosis=True, seed=cfg.seed),
                                    transform_uses_labels=True, notes=f"CovBat applied to the {S} coordinates"))
        elif n in ("combat-as", "combat-rep-as"):
            keys = tuple(k for k in ("age", "sex") if k in covariate_keys)
            if not keys:
                log.warning("%s requested but no age/sex covariates available; skipped", n)
                continue
            on_raw = n == "combat-as"
            specs.append(MethodSpec("ComBat (age+sex)" if on_raw else f"ComBat ({S}, age+sex)", "raw" if on_raw else space,
                                    lambda keys=keys: ComBatHarmonizer(protect_diagnosis=False, covariate_keys=keys),
                                    fit_aux=("covariates",),
                                    notes=f"label-free ComBat: {' and '.join(keys)} as the only covariates, no diagnosis; "
                                          "needs the age and sex of every adjusted subject"))
        elif n == "ours-auto":
            specs.append(MethodSpec("Ours (lambda=1-delta)", space, ours("ours-auto", "Ours (lambda=1-delta)"),
                                    notes="geodesic relaxation, lambda_s = 1 - shrinkage intensity"))
        elif n == "barycenter":
            specs.append(MethodSpec("Ours (barycenter hub)", space, ours("barycenter", "Ours (barycenter hub)"),
                                    notes="BW barycenter of site covariances as hub"))
        elif n == "ours-2d":
            if not have_severity:
                log.warning("ours-2d requested but the cohort has no severity score; skipped")
                continue
            specs.append(MethodSpec("Ours (diag+severity)", space, ours("ours-2d", "Ours (diag+severity)"),
                                    fit_aux=("y_diag", "severity"), notes="2-D protected subspace"))
        elif n == "ours-demo":
            keys = [k for k in ("age", "sex") if k in covariate_keys]
            if not keys:
                log.warning("ours-demo requested but no age/sex covariates available; skipped")
                continue
            specs.append(MethodSpec("Ours (diag+age+sex)", space,
                                    ours("ours", "Ours (diag+age+sex)", protected=("diagnosis", *keys)),
                                    fit_aux=("y_diag", "covariates"), notes="protected diagnosis + demographic axes"))
        elif n == "ours-cov":
            keys = [k for k in ("age", "sex") if k in covariate_keys]
            if not keys:
                log.warning("ours-cov requested but no age/sex covariates available; skipped")
                continue
            specs.append(MethodSpec("Ours (covariates only)", space,
                                    ours("ours", "Ours (covariates only)", protected=tuple(keys)),
                                    fit_aux=("covariates",), notes="protected demographic axes, no diagnosis label"))
        elif n == "spd-ref":
            specs.append(MethodSpec("Log-Euclidean FC", "spd", IdentityHarmonizer, fit_aux=(), is_reference=True,
                                    notes="upper(logm(C)) of the shrunk correlation matrices, no harmonization"))
        elif n == "riemannian":
            specs.append(MethodSpec("Riemannian re-centering", "spd",
                                    lambda: RiemannianRecenteringHarmonizer(target_site=t),
                                    fit_aux=(), notes="Zanini et al. 2018: whiten each site by its Riemannian mean"))
        elif n == "ea":
            specs.append(MethodSpec("Euclidean alignment", "spd",
                                    lambda: EuclideanAlignmentHarmonizer(target_site=t),
                                    fit_aux=(), notes="He & Wu 2020: whiten each site by its arithmetic mean"))
        elif n == "rpa":
            specs.append(MethodSpec("Riemannian Procrustes", "spd",
                                    lambda: RiemannianProcrustesHarmonizer(target_site=t),
                                    fit_aux=(), notes="Rodrigues et al. 2019, steps 1-2: re-centre and stretch to the hub"))
        elif n == "rlet":
            specs.append(MethodSpec("RLET", "spd", RLETHarmonizer, fit_aux=(),
                                    notes="Simeon et al. 2022: rigid log-Euclidean translation to the mean of site means"))
        elif n == "coral":
            specs.append(MethodSpec("CORAL", space, lambda: CORALHarmonizer(target_site=t),
                                    fit_aux=(), notes="Sun et al. 2016, Algorithm 1 per site against the hub"))
        elif n == "combat-gam":
            try:
                import neuroHarmonize  # noqa: F401
            except ImportError:
                log.warning("ComBat-GAM requested but neuroHarmonize is not installed; skipped")
                continue
            specs.append(MethodSpec("ComBat-GAM", space,
                                    lambda: ComBatGAMHarmonizer(protect_diagnosis=True),
                                    transform_uses_labels=True, fit_aux=("y_diag", "covariates"),
                                    notes="neuroHarmonize ComBat-GAM on the PCA scores; a per-edge GAM on the raw connectome does not finish on CPU"))
        elif n == "mmd-affine":
            specs.append(MethodSpec("MMD affine", space, lambda: MMDAffineHarmonizer(target_site=t, seed=cfg.seed),
                                    fit_aux=(), notes="our own baseline: per-site affine map fitted by MMD to the hub"))
        elif n == "relief":
            if not rscript_available():
                log.warning("RELIEF requested but Rscript is not on PATH; skipped")
                continue
            try:
                RELIEFHarmonizer()
            except Exception as exc:
                log.warning("RELIEF unavailable (%s); skipped", exc)
                continue
            specs.append(MethodSpec("RELIEF", space, RELIEFHarmonizer, fit_aux=(),
                                    notes="R package RELIEF on the PCA scores without covariates (the README advises "
                                          "leaving the tested covariate out of mod); held-out rows are refitted with "
                                          "the training rows; the raw-edge call does not finish per fold on CPU"))
        elif n == "pam-edge":
            specs.append(MethodSpec("Ours (edge residual)", "raw",
                                    lambda: EdgeResidualHarmonizer(lambda: make_variant("ours", t, name="Ours", **common),
                                                                   pca_dim=cfg.pca_dim, seed=cfg.seed),
                                    notes="PAM on the training-fold PCA scores, change added back to the full connectome"))
        elif n == "slacc":
            if not rscript_available():
                log.warning("SLACC requested but Rscript is not on PATH; skipped")
                continue
            try:
                SLACCHarmonizer()
            except Exception as exc:
                log.warning("SLACC unavailable (%s); skipped", exc)
                continue
            specs.append(MethodSpec("SLACC", "raw", SLACCHarmonizer, fit_aux=("y_diag",),
                                    notes="R package SLACC on the edge vector; a full-atlas fit is about 50 min on this CPU"))
        else:
            raise SystemExit(f"unknown method '{n}'. Known: {sorted(set(GROUPS['all']))} or groups {sorted(GROUPS)}")
    return specs


# -- helpers -----------------------------------------------------------------------------

def _configure_sites_from_hdf5(cfg: VAEConfig) -> None:
    with h5py.File(cfg.hdf5_path, "r") as f:
        attr = f.attrs.get("site_names", None)
        y_site = f["y_site"][:].astype(int)
    names = ([s.decode() if isinstance(s, bytes) else str(s) for s in attr] if attr is not None
             else [f"SITE_{s}" for s in sorted(np.unique(y_site))])
    cfg.n_sites = len(names)
    cfg.site_names = {i: n for i, n in enumerate(names)}
    cfg.site_colors = site_palette(cfg.n_sites)
    # hub rule of the paper: the largest site (NYU in ABIDE-I, Peking in ADHD-200, KKI in ABIDE-II)
    sites, counts = np.unique(y_site, return_counts=True)
    cfg.target_site = int(sites[np.argmax(counts)])


def _resolve_site(spec: str, cfg: VAEConfig) -> int:
    tok = spec.strip()
    if tok.isdigit():
        sid = int(tok)
        if sid not in cfg.site_names:
            raise SystemExit(f"unknown site id {sid}; known: {sorted(cfg.site_names)}")
        return sid
    for i, n in cfg.site_names.items():
        if n.upper() == tok.upper():
            return i
    raise SystemExit(f"unknown site '{spec}'; known: {', '.join(f'{i}:{n}' for i, n in cfg.site_names.items())}")


def _load_severity(cfg: VAEConfig, ds_entry: dict) -> np.ndarray | None:
    keys = [ds_entry.get("severity_key")] if ds_entry.get("severity_key") else []
    keys += ["ados_total", "ados_g_total", "y_symptom", "ados"]
    with h5py.File(cfg.hdf5_path, "r") as f:
        for k in keys:
            if k and k in f:
                s = f[k][:].astype(float)
                n_ok = int(np.isfinite(s).sum())
                log.info("severity score '%s': %d/%d valid", k, n_ok, len(s))
                return s if n_ok >= 20 else None
    return None


def _load_covariates(cfg: VAEConfig, ds_entry: dict, subject_ids: list[str]) -> dict[str, np.ndarray]:
    try:
        from vae_harmonization.phenotypes import load_covariates
    except ImportError:
        return {}
    try:
        cov = load_covariates(ds_entry, subject_ids)
    except Exception as exc:  # phenotype files are optional
        log.warning("covariates unavailable (%s)", exc)
        return {}
    for k, v in cov.items():
        log.info("covariate '%s': %d/%d valid", k, int(np.isfinite(v).sum()), len(v))
    return cov


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2)


# -- main -----------------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Protected-subspace affine harmonization of multi-site FC")
    p.add_argument("--dataset", default="abide1_full", help=f"one of {list_datasets()}")
    p.add_argument("--config-yaml", default=None, help="YAML overriding VAEConfig fields")
    p.add_argument("--run-tag", default=None, help="suffix for the results directory")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--representation", choices=("pca", "vae"), default="pca",
                   help="pca: per-fold PCA scores (default); vae: frozen FC-VAE latent (ablation)")
    p.add_argument("--pca-dim", type=int, default=None)
    p.add_argument("--whiten", action="store_true", help="whiten the PCA scores before harmonization (ablation)")
    p.add_argument("--covariance", default=None, choices=("pooled", "lw", "diagonal", "empirical"),
                   help="covariance estimator of Ours (default: pooled)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--target-site", default=None, help="hub site id or name (default: the largest site)")
    p.add_argument("--methods", default="default",
                   help="comma list of methods and/or groups: " + ", ".join(GROUPS))
    p.add_argument("--n-folds", type=int, default=None)
    p.add_argument("--classifiers", default="lr,rf")
    p.add_argument("--deep-epochs", type=int, default=200, help="unused (DeepComBat follows its package schedule)")
    p.add_argument("--spd-shrinkage", type=float, default=0.1,
                   help="shrinkage of each correlation matrix toward the identity before the Riemannian baselines")
    p.add_argument("--skip-protocol", action="store_true",
                   help="run only the requested optional analyses (protocol_results.json is left untouched)")
    p.add_argument("--effect-sizes", action="store_true",
                   help="leakage-free effect-size reproducibility across site halves -> effect_sizes.json")
    p.add_argument("--mmd-bootstrap", type=int, default=0, help="bootstrap replicates for pairwise MMD CIs")
    p.add_argument("--mmd-min-site", type=int, default=5,
                   help="sites with fewer held-out rows are excluded from the pairwise MMD (default 5)")
    p.add_argument("--n-perm", type=int, default=0,
                   help="permutation replicates for the protocol-matched ICC/MMD null (0 disables)")
    p.add_argument("--n-repeats", type=int, default=1,
                   help="repeated CV repeats (1 = the usual 10-fold split; 5 = 5x10)")
    p.add_argument("--min-cov-samples", type=int, default=None)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--downstream", action="store_true", help="LOSO for all methods, effect-size reproducibility, motion")
    p.add_argument("--fingerprint", action="store_true", help="split-half identifiability (needs fc_half1/2)")
    p.add_argument("--network-analysis", action="store_true", help="Yeo-7 network preservation on reconstructed FC")
    p.add_argument("--protected-axis", action="store_true", help="protected-axis site-variance analysis")
    p.add_argument("--unseen-site", action="store_true", help="unseen-site size curve (10/20/30/50 scans)")
    p.add_argument("--federated", action="store_true", help="one-round federated equality check")
    p.add_argument("--overlap", action="store_true", help="per-site covariate overlap diagnostics")
    # VAE ablation only
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--train-vae", action="store_true", help="re-train the FC-VAE instead of loading the checkpoint")
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--skip-insample", action="store_true", help="skip the VAE in-sample diagnostics and figures")
    p.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _load_cohort(cfg: VAEConfig):
    """The cohort arrays without any fitted preprocessing."""
    from vae_harmonization.dataset import ABIDEDataset
    ds = ABIDEDataset(cfg.hdf5_path)
    return ds


def main(argv=None):
    args = parse_args(argv)

    # -- config -------------------------------------------------------------------------
    if args.config_yaml:
        cfg = VAEConfig.from_yaml(args.config_yaml)
    else:
        cfg = VAEConfig.from_dataset(args.dataset)
    try:
        ds_entry = get_dataset_config(cfg.dataset_name)
    except KeyError:
        # a YAML pointing at an arbitrary HDF5 (e.g. the synthetic CI cohort): no phenotypes, no atlas
        ds_entry = {}
    for k, v in (("latent_dim", args.latent_dim), ("pca_dim", args.pca_dim), ("max_epochs", args.max_epochs),
                 ("seed", args.seed), ("n_cv_folds", args.n_folds), ("min_cov_samples", args.min_cov_samples),
                 ("covariance", args.covariance)):
        if v is not None:
            setattr(cfg, k, v)
    cfg.input_dim = cfg.pca_dim
    cfg.lambda_site = 0.0
    cfg.lambda_bio = 0.0
    _configure_sites_from_hdf5(cfg)
    if args.target_site is not None:
        cfg.target_site = _resolve_site(args.target_site, cfg)
    use_vae = args.representation == "vae"
    space = "latent" if use_vae else "pca"

    run_name = cfg.dataset_name if not args.run_tag else f"{cfg.dataset_name}_{args.run_tag}"
    cfg.out_dir = Path(args.out_dir) if args.out_dir else _REPO_DIR / "results" / run_name
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_dir:
        cfg.checkpoint_dir = Path(args.checkpoint_dir)
    elif args.train_vae:
        cfg.checkpoint_dir = _REPO_DIR / "checkpoints" / run_name
    else:
        cfg.checkpoint_dir = _REPO_DIR / "checkpoints" / cfg.dataset_name   # frozen per-dataset checkpoint

    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO), log_file=cfg.out_dir / "run.log")
    device = None
    if use_vae or any(m in _expand_methods(args.methods) for m in DEEP):
        import torch
        device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("run '%s' | dataset=%s | representation=%s (%d-D) | covariance=%s | device=%s", run_name,
             cfg.dataset_name, space, cfg.latent_dim if use_vae else cfg.pca_dim, cfg.covariance, device or "cpu")
    log.info("sites: %s", {n: i for i, n in cfg.site_names.items()})
    log.info("hub (target) site: %s", cfg.site_names[cfg.target_site])
    t_start = time.perf_counter()

    # -- stage 0: data -------------------------------------------------------------------
    full_ds = _load_cohort(cfg)
    y_site = full_ds.y_site.astype(int)
    y_diag = full_ds.y_diag.astype(int)
    X_raw = full_ds.X_raw.astype(np.float64)
    counts = {cfg.site_names[int(s)]: int(c) for s, c in zip(*np.unique(y_site, return_counts=True))}
    log.info("subjects: %d, features %d; site counts: %s", len(y_site), X_raw.shape[1], counts)
    severity = _load_severity(cfg, ds_entry)
    covariates = _load_covariates(cfg, ds_entry, full_ds.subject_ids)

    # -- stage 1: representation ----------------------------------------------------------
    # Z_full / encode / decode are used only by the auxiliary analyses (LOSO, fingerprinting, network
    # preservation, protected axis), all of which are transductive or descriptive by design.  The main
    # protocol below never uses them.
    model = None
    if use_vae:
        import torch
        from vae_harmonization.dataset import get_dataloaders
        from vae_harmonization.evaluation import run_insample_diagnostics
        from vae_harmonization.evaluation.latents import decode_to_fc, encode_features
        from vae_harmonization.train import load_checkpoint, save_checkpoint, train_vae
        train_loader, val_loader, scaler, pca, full_ds = get_dataloaders(cfg)
        if args.train_vae:
            model, history = train_vae(cfg, train_loader, val_loader, device)
            save_checkpoint(model, cfg, history)
        else:
            model = load_checkpoint(cfg, device)
            hist_path = cfg.checkpoint_dir / f"history_d{cfg.latent_dim}.json"
            history = json.load(open(hist_path)) if hist_path.exists() else []
            log.info("loaded checkpoint from %s (%d epochs of history)", cfg.checkpoint_dir, len(history))
        diag = run_insample_diagnostics(model, full_ds, cfg, history, device,
                                        make_plots=not (args.no_plots or args.skip_insample))
        Z_full = diag["Z_mu"].astype(np.float64)
        _write_json(cfg.out_dir / "insample_diagnostics.json",
                    {k: diag[k] for k in ("gaussianity", "reconstruction_raw", "reconstruction_pca", "insample_map")})
        representations = {"raw": X_raw, "latent": Z_full}

        def encode(H_raw):
            return encode_features(model, full_ds.pca.transform(full_ds.scaler.transform(H_raw)).astype(np.float32),
                                   device).astype(np.float64)

        def decode(Z):
            return decode_to_fc(model, Z, full_ds, device).astype(np.float64)

        recon_label = "VAE reconstruction"
    else:
        rep_full = PCARepresentation.fit(X_raw, cfg.pca_dim, seed=cfg.seed, whiten=args.whiten)
        Z_full = rep_full.transform(X_raw)
        summary = rep_full.summary()
        log.info("full-cohort PCA (auxiliary analyses only): %d components explain %.1f%% of the variance",
                 summary["n_components"], 100 * summary["explained_variance_ratio_total"])
        _write_json(cfg.out_dir / "representation.json", summary)
        fold_reps: list[PCARepresentation] = []
        representations = make_pca_representation_fn(X_raw, cfg.pca_dim, seed=cfg.seed, whiten=args.whiten,
                                                     store=fold_reps)
        encode, decode, recon_label = rep_full.transform, rep_full.inverse, "PCA reconstruction"

    # -- stage 2: leakage-free protocol ----------------------------------------------------
    method_names = _expand_methods(args.methods)
    X_spd = None
    if any(m in SPD_METHODS for m in method_names) and not use_vae:
        if "spd-ref" not in method_names:
            method_names.insert(0, "spd-ref")
        t_spd = time.perf_counter()
        X_spd = log_euclidean_vectors(X_raw, args.spd_shrinkage)
        log.info("SPD representation: %d log-Euclidean features per subject (shrinkage %.2f, %.1fs)",
                 X_spd.shape[1], args.spd_shrinkage, time.perf_counter() - t_spd)
        pca_fn = representations

        def representations(fold, tr, te, _f=pca_fn, _spd=X_spd):
            reps, embed = _f(fold, tr, te)
            return {**reps, "spd": _spd}, embed
    specs = build_method_specs(method_names, cfg, have_severity=severity is not None,
                               covariate_keys=list(covariates), deep_epochs=args.deep_epochs, device=str(device),
                               space=space)
    log.info("protocol methods: %s", [s.name for s in specs])
    # pooled held-out coordinates of the representation-space methods, kept for the figures.  PCA outputs
    # arrive embedded in the standardized edge space and are drawn in the basis of the full-cohort PCA.
    oos: dict[str, np.ndarray] = {}

    def keep_oos(name, rep, X_oos):
        if rep != space:
            return
        if use_vae:
            oos[name] = np.asarray(X_oos, dtype=np.float32)
            return
        # Failed folds leave NaN rows. Project only the finite ones so one
        # broken fold does not discard the finished protocol.
        finite = np.isfinite(X_oos).all(axis=1)
        projected = np.full((X_oos.shape[0], rep_full.n_components), np.nan, dtype=np.float32)
        if finite.any():
            projected[finite] = rep_full.project(X_oos[finite]).astype(np.float32)
        oos[name] = projected

    if args.skip_protocol:
        log.info("--skip-protocol: the protocol is not run")
        results = None
    else:
        results = run_protocol(representations, y_site, y_diag, specs, n_folds=cfg.n_cv_folds, seed=cfg.seed,
                           classifiers=tuple(args.classifiers.split(",")), severity=severity,
                           covariates=covariates or None, site_names=cfg.site_names,
                           mmd_bootstrap=args.mmd_bootstrap, mmd_min_site=args.mmd_min_site,
                           n_perm=args.n_perm, n_repeats=args.n_repeats,
                               return_oos=False, oos_hook=keep_oos)
    if results is not None:
        results["run"] = {
            "run_name": run_name, "dataset": cfg.dataset_name, "hub_site": cfg.site_names[cfg.target_site],
            "representation": space, "representation_dim": int(cfg.latent_dim if use_vae else cfg.pca_dim),
            "pca_dim": cfg.pca_dim, "whiten": bool(args.whiten), "covariance": cfg.covariance, "seed": cfg.seed,
            "n_rois": ds_entry.get("n_rois"), "n_features": int(X_raw.shape[1]),
            "checkpoint": str(cfg.checkpoint_dir) if use_vae else None, "device": str(device or "cpu"),
            "spd_shrinkage": float(args.spd_shrinkage),
        }
        if not use_vae:
            results["run"]["fold_pca"] = [{"fold": i, "explained_variance_ratio_total": r.summary()["explained_variance_ratio_total"]}
                                          for i, r in enumerate(fold_reps)]
        _write_json(cfg.out_dir / "protocol_results.json", results)
        if oos:
            np.savez_compressed(cfg.out_dir / f"oos_{space}.npz", y_site=y_site, y_diag=y_diag,
                                **{k.replace(" ", "_"): v for k, v in oos.items()})
        table = summary_table(results, classifier="lr")
        (cfg.out_dir / "summary.txt").write_text(table + "\n", encoding="utf-8")
        log.info("\n%s", table)
        if not args.no_plots:
            plot_method_bars(results, cfg.out_dir / "figures" / "method_bars.png")

    # -- stage 3: optional analyses ---------------------------------------------------------
    if (args.protected_axis or args.downstream or args.fingerprint or args.network_analysis
            or args.unseen_site or args.federated or args.overlap or args.effect_sizes):
        from vae_harmonization.analyses import run_optional_analyses
        run_optional_analyses(args, cfg, ds_entry, X_raw, Z_full, y_site, y_diag, severity, covariates, specs,
                              space=space, encode=encode, decode=decode, recon_label=recon_label,
                              extra_reps=None if X_spd is None else {"spd": X_spd})

    log.info("done in %.1f min; outputs in %s", (time.perf_counter() - t_start) / 60, cfg.out_dir)


if __name__ == "__main__":
    main()
