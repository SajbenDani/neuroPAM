# Reproducing the paper

Every table of the paper is produced by `scripts/generate_tables.py` from the result files in
`results/`. No number is typed by hand. This file lists the commands that create those
result files.

The main pipeline is **CPU only**. It uses per-fold PCA scores as the representation
(scaler and PCA are fitted on the training fold). The frozen FC-VAE checkpoints are an
optional appendix ablation (`--representation vae`) and are never required for the
headline tables.

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
python -m pytest tests -q
```

On Windows, activate with `.venv\Scripts\activate`. The random-forest probe grows its
trees on all cores (`VAEH_RF_JOBS=1` forces a single thread); results do not depend on
the thread count.

Optional extras used by named baselines:

```bash
pip install neuroHarmonize          # ComBat-GAM
# RELIEF (and optionally SLACC): R 4.5 with the RELIEF package (plus MASS, Matrix;
# SLACC also needs rTensor, foreach and clue). Rscript must be on PATH.
# If Rscript or the package is missing the row is skipped.
# DeepComBat: the authors' R package, vendored with four documented patches
# (third_party/DeepComBat/PATCHES.md). In R: install.packages(c("torch", "coro", "dplyr", "remotes"));
# torch::install_torch(); remotes::install_github("Jfortin1/neuroCombat_Rpackage"). Then
R CMD INSTALL third_party/DeepComBat
```

## 1. Data

Processed inputs are HDF5 files with `X` (N x edges, Fisher-z upper triangle), `y_site`,
`y_diagnosis`, `subject_ids`, optional `fc_half1` / `fc_half2` (split-half FC for
fingerprinting) and severity scores. `attrs['site_names']` fixes the site order.
`python scripts/preprocessing/validate_fc_h5_schema.py --h5 <file>` checks a file.

```bash
# ABIDE-I: C-PAC CC200 time series (filt_global) + phenotypic CSV from the PCP S3 bucket
python scripts/preprocessing/download_abide1.py --dataset abide1 --out-dir data/raw
python scripts/preprocessing/preprocess_abide1.py --dataset abide1_full --raw-dir data/raw --out-dir data/processed
python scripts/preprocessing/add_split_half_fc_abide.py

# ADHD-200: Athena CC200 release (NITRC / FCON1000 mirror) into data/ADHD-200
python scripts/preprocessing/preprocess_adhd200.py --source nitrc-fcon --raw-dir data/ADHD-200 --out-dir data/processed
python scripts/preprocessing/add_split_half_fc_adhd200.py

# ABIDE-II: fMRIPrep derivatives from FCP-INDI S3, CC200 time-series extraction, QC, split halves
# (the 10 sites of the paper are those with >= 20 usable subjects; the script is resumable)
python scripts/preprocessing/preprocess_abide2.py --pheno data/ABIDE-II/ABIDEII_Composite_Phenotypic.csv \
    --out-dir data/processed_abide2_10site_fp --split-half

# REST-meta-MDD: Phase-1 functional ROI signals after a signed data-use agreement
# (ScienceDB DOI 10.57760/sciencedb.o00115.00013, or rfmrilab@gmail.com).
# Extract under data/REST-meta-MDD/. The script downloads the public Yan 2019
# subject list, drops site 4, and writes data/processed/rest_meta_mdd_fc.h5.
# None of those files are committed.
python scripts/preprocessing/preprocess_rest_meta_mdd.py

# SRPBS (ATR, after approved applications; never commit anything from these releases):
# Multidisorder Connectivity Dataset -> data/SRPBS/SRPBS_FC/, DecNef Harmonization Dataset ->
# data/SourceCodeAndDATA/. Writes data/processed/srpbs_fc.h5 (MDD vs controls, six scanners, BAL-140),
# srpbs_decnef_fc.h5 and srpbs_decnef_unified_fc.h5 (clinical scans, 268 regions) and srpbs_ts_fc.h5
# (traveling subjects). The script checks site and diagnosis codes against the release's design matrices.
python scripts/preprocessing/preprocess_srpbs.py
```

Paths, site names, hub and covariate sources live in `src/vae_harmonization/data_registry.py`.
The phenotypic tables of ABIDE and ADHD-200 are committed. The REST-meta-MDD workbook is not.
The HDF5 files (`data/processed/abide1_full_fc.h5`, `data/processed/adhd200_fc.h5`,
`data/processed_abide2_10site_fp/abide2_fc.h5`, `data/processed/rest_meta_mdd_fc.h5`)
have to be produced or copied in.
The Yeo-7 mapping tables in `data/atlases/` are committed; to rebuild them:
`python scripts/build_network_labels.py --cohorts abide1_full` (downloads the Yeo atlas
through nilearn).

## 2. Runs behind the tables

All runs are CPU only and use the same folds (seed 42), so `generate_tables.py` merges the
runs of one cohort and recomputes all paired tests together (one Holm family per cohort).
Wall times are for a desktop CPU (ABIDE-I / REST-meta-MDD).

```bash
DATASETS="abide1_full adhd200 abide2_10site rest_meta_mdd srpbs"
RUN="python -m vae_harmonization.run_vae_harmonization"

for DS in $DATASETS; do
  # main protocol and all analyses (40 / 65 min)
  $RUN --dataset $DS --methods default,extensions --classifiers lr,rf --mmd-min-site 5 --n-perm 200 \
       --protected-axis --fingerprint --network-analysis --downstream --unseen-site --federated --overlap
  # geometric baselines: pyRiemann methods on the shrunk correlation matrices, CORAL, MMD map
  $RUN --dataset $DS --run-tag geometric --methods geometric --classifiers lr,rf --mmd-min-site 5 --n-perm 200
  # ComBat-GAM, RELIEF (no covariates, as its README advises for a tested covariate) and DeepComBat on the PCA scores
  $RUN --dataset $DS --run-tag gam --methods combat-gam --classifiers lr,rf --n-perm 200
  $RUN --dataset $DS --run-tag r --methods relief --classifiers lr,rf --n-perm 200
  $RUN --dataset $DS --run-tag deep --methods deepcombat --classifiers lr,rf --mmd-min-site 5 --n-perm 200
  # CovBat as in the authors' code (scaled residuals), PAM's edge-residual version
  $RUN --dataset $DS --run-tag covbat --methods covbat,covbat-rep --classifiers lr,rf --mmd-min-site 5 --n-perm 200
  # label-free ComBat (age and sex as the only covariates) with its LOSO and effect sizes
  $RUN --dataset $DS --run-tag combatas --methods raw,rep,combat-as,combat-rep-as --classifiers lr,rf \
       --mmd-min-site 5 --n-perm 200 --downstream --effect-sizes --no-plots
  $RUN --dataset $DS --run-tag edge --methods pam-edge --classifiers lr,rf --mmd-min-site 5 --n-perm 200 \
       --fingerprint                                      # drop --fingerprint for REST-meta-MDD and SRPBS
  # displacement under the fitted moments (the quantity of the theorem)
  $RUN --dataset $DS --run-tag disp --methods ours,full-affine,mean-shift,cholesky,zca --classifiers lr
  # leakage-free effect sizes and the corrected new-site analysis (no protocol run)
  $RUN --dataset $DS --run-tag es --methods default,extensions --skip-protocol --effect-sizes --no-plots
  $RUN --dataset $DS --run-tag es_spd --methods spd-ref,riemannian,ea,rpa,rlet --skip-protocol --effect-sizes --no-plots
  $RUN --dataset $DS --run-tag unseen --methods ours --skip-protocol --unseen-site --no-plots
  # repeated 5 x 10-fold CV with Nadeau-Bengio corrected tests (65 / 115 min)
  $RUN --dataset $DS --run-tag repeated --methods default --n-repeats 5 --classifiers lr
  $RUN --dataset $DS --run-tag repeated_est --methods ours-lw,ours-diag,ours-emp --n-repeats 5 --classifiers lr
done

# SRPBS has no split-half data or Yeo labels: drop --fingerprint and --network-analysis for it.
python scripts/run_traveling.py           # traveling subjects -> results/srpbs_ts/traveling.json
python scripts/traveling_noise_budget.py  # -> results/srpbs_ts/noise_budget.json
python scripts/traveling_metric_check.py  # simulation on the visit pattern -> results/srpbs_ts/metric_check.json
python scripts/site_variance_in_pcs.py    # between-site variance inside 100 PCs -> results/<cohort>/site_variance_pcs.json
# sensitivity of the Riemannian baselines to the shrinkage toward the identity (ABIDE-I)
for A in 0.01 0.5; do
  $RUN --dataset abide1_full --run-tag spd${A/./} --methods spd-ref,riemannian,rlet --classifiers lr --spd-shrinkage $A
done

for DS in abide1_full adhd200 abide2_10site srpbs; do
  python scripts/sensitivity_grid.py --dataset $DS                       # PCA-dimension sweep
done
for DS in abide1_full adhd200 abide2_10site; do
  $RUN --dataset $DS --run-tag vae --representation vae --skip-insample \
       --methods rep,ours,mean-shift,full-affine --classifiers lr,rf --n-perm 200   # appendix
done
$RUN --dataset abide1_full --run-tag whiten --whiten --methods rep,ours,mean-shift,full-affine \
     --classifiers lr,rf --n-perm 200

python scripts/run_simulation.py --n-rep 100
python scripts/validate_theorem.py
python scripts/generate_tables.py
python scripts/generate_paper_figures.py
```

`default` is Raw, ComBat, CovBat, the PCA scores, PAM (Ours) with the pooled-target covariance, the
map and estimator ablations and ComBat/CovBat on the PCA scores. `extensions` adds the geodesic
relaxation, the barycenter hub, the severity axis and the demographic axes. `geometric` is the
log-Euclidean reference, Riemannian re-centering, Euclidean alignment, the re-centering and
stretching steps of Riemannian Procrustes analysis and RLET, all through pyRiemann 0.10 on
correlation matrices shrunk toward the identity (`--spd-shrinkage`, default 0.1), plus CORAL as in
Algorithm 1 of Sun et al. (2016) and the MMD-fitted affine map (our own comparison method).
`combat-age-sex` (`combat-as`, `combat-rep-as`) is label-free ComBat on the edges and on the PCA scores, with
age and sex as the only covariates. ComBat-GAM (`combat-gam`) fits one GAM per feature, so it runs on the PCA scores. `r-baselines` is
RELIEF. `deep` is DeepComBat through the authors' R package. SLACC is available as
`--methods slacc` but a full-atlas fit takes about an hour and the protocol calls it twice per
fold, so it is not in the tables.

Outputs per cohort in `results/<cohort>[_tag]/`: `protocol_results.json`, `protected_axis.json`,
`fingerprinting.json`, `fingerprinting_edge.json` (edge run), `network_preservation.json`,
`downstream.json`, `effect_sizes.json` (es run), `unseen_site.json` (unseen run), `federated.json`,
`overlap.json`, `site_variance_pcs.json`, `oos_pca.npz`, `summary.txt`, `run.log`.
`generate_tables.py` prefers the `_es`, `_unseen` and `_covbat` outputs over those of the main run and adds
the Riemannian rows of `_es_spd` and the label-free ComBat rows of `_combatas` (protocol, LOSO, effect sizes). RELIEF calls R once per fold and again for the held-out rows; one call may take
up to `VAEH_R_TIMEOUT` seconds (default 3600) before the fold is marked failed.

MMD^2 is the mean over all pairs of sites with at least `--mmd-min-site` held-out subjects
(default 5). The RBF bandwidth is the median heuristic on the pooled pre-harmonization
matrix. Both choices are recorded under `protocol` in the JSON.

## 3. Tables and figures

`scripts/generate_tables.py` writes the LaTeX tables and `numbers.tex`, which holds every number
quoted in the paper as a macro, to `paper/tables/`. `scripts/generate_paper_figures.py` writes the
figures to `paper/figures/`. The manuscript itself is not part of this repository, and the tables
that use REST-meta-MDD or SRPBS need those cohorts' result files, which are not distributed.

## 4. Tests

```bash
python -m pytest tests -q -W ignore::RuntimeWarning
```

The suite runs on synthetic cohorts only (no data download) and covers the transport
maps and theorem, the protected harmonizer (including covariate-conditional intercepts,
pooled-target shrinkage and imputation), the baselines, the metrics, the leakage-free
protocol, permutation nulls and every analysis module.

## 5. SRPBS: cohort definition and traveling subjects

Two ATR releases are used, both as connectivity matrices, both after an approved application and
under non-redistribution terms. Nothing from them (archives, tables, processed HDF5,
`results/srpbs*`) is committed.

* **SRPBS Multidisorder Connectivity Dataset** (<https://bicr-resource.atr.jp/srpbsfc/>): one file
  per institute and diagnosis, Pearson correlations of the 140-region BAL atlas. The cohort `srpbs`
  is MDD against controls at the six scanners that recorded both groups (Hiroshima Signa 1 and 2,
  Spectra and Verio; Kyoto TimTrio; Tokyo MR750W). Correlations are Fisher-z transformed, and
  32 Hiroshima scans with missing parcels are dropped. Age and sex are stored in the HDF5.
  The archive also contains the traveling subjects in the BAL atlas; they are not used.
* **DecNef Harmonization Dataset** (<https://bicr.atr.jp/dcn/en/download/harmonization/>, data of
  Yamashita et al. 2019): Fisher-z connectivity (268-region atlas of Shen et al. 2013) of nine
  traveling subjects at twelve scanners (411 sessions, 397 after the release's motion exclusions)
  and of the clinical SRPBS sample (637 scans; 496 acquired with the unified protocol that the
  travelers followed). The release's subject initials are replaced by T1 to T9.

`scripts/run_traveling.py` runs three settings (`--settings within,transfer,transfer_all`). In
`within`, each traveler is held out in turn and the harmonizer is fitted on the other eight (site
labels only, nothing protected); the PCA basis is learned from the unified-protocol clinical scans,
so nothing is estimated from the held-out person (`within_refit` re-fits the PCA on the eight
travelers and is available as a check). In `transfer`, PCA and harmonizers are fitted on the
unified-protocol clinical scans (diagnoses MDD, SCZ and ASD as protected covariates) and applied to
the travelers at those six scanners; `transfer_all` uses all 637 clinical scans.

Metrics. *Per person*: each traveler's between-scanner sum of squares of their scanner cell means,
after relative to before, with the session-noise floor ((S_p - 1) times the pooled within-cell
variance) subtracted from both; this is the main measure. *Pooled*: the scanner sum of squares of
an additive scanner + person model. In `within` the pooled measure cancels by construction for any
method that removes the other travelers' scanner means (`scripts/traveling_metric_check.py` shows
this on simulated sessions with the real visit pattern), so it is reported for the transfer
settings only. *Per site*: the squared length of each scanner's offset from the hub after
harmonization relative to before. Also: a two-way ANOVA with person x scanner interaction and its
variance components, cross-scanner identification, a site probe (leave one traveler out), a
traveler probe (leave one scanner out), and the cosine and length of each site's correction
relative to its true offset. `scripts/traveling_noise_budget.py` estimates how much of the
difference between the clinical site-mean corrections and the travelers' offsets is sampling noise.
