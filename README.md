# neuroPAM

**Protected-subspace Affine Matching (PAM)**: a closed-form harmonizer for multi-site functional
connectivity that removes site effects and keeps the labels you want to study.

This repository holds the code, the configuration and the result files of the paper

> Dániel Sajben, Luca Szegletes and the DIRECT Consortium. *Moving less takes you further:
> minimum-displacement affine harmonization of multi-site functional connectivity with a
> label-protected subspace.* Submitted to *Medical Image Analysis*, 2026.

## What PAM does

Connectomes pooled from several scanners carry a site signature that a linear classifier detects
more easily than the diagnosis. PAM works on a low-dimensional representation of the connectome
(by default 100 principal components, fitted on the training data) and fits one affine map per site:

* Directions that encode the labels to keep (diagnosis, and optionally severity, age or sex) form a
  **protected subspace**. They only receive a mean shift, and the site means are estimated given
  those labels.
* The remaining directions are mapped onto a reference site with the **Bures–Wasserstein map**
  between the site covariances, which are shrunk toward the pooled within-site covariance.

This is the unique affine map that matches the covariances of the remaining directions, leaves the
protected subspace fixed and moves subjects least (Theorem 1 of the paper). Fitting needs only
per-site means and covariances, and a new subject needs only its site label, never its diagnosis.

## Installation

```bash
git clone https://github.com/SajbenDani/neuroPAM.git
cd neuroPAM
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
python -m pytest tests -q            # synthetic data only
```

Python 3.10 to 3.12. Everything runs on a CPU. The requirements install this repository as the
package `vae_harmonization` (the name is historical). The R baselines (RELIEF, and DeepComBat
through its authors' package in `third_party/DeepComBat`) need R 4.5, see [REPRODUCE.md](REPRODUCE.md).

## Using PAM on your own data

```python
from vae_harmonization.harmonization import make_variant

# Z_*: (subjects, features) representation of the connectomes, e.g. PCA scores fitted on the
# training subjects; site_*: integer site labels; dx_train: diagnosis of the training subjects
pam = make_variant("ours", target_site=0)          # site 0 is the reference ("hub")
pam.fit(Z_train, site_train, y_diag=dx_train)      # labels are used to fit, never to transform
Z_test_harmonized = pam.transform(Z_test, site_test)
pam.describe()                                     # per-site maps, shrinkage and displacement
```

To harmonize the full connectome instead of the scores, wrap PAM in the edge-residual version,
which fits it on the principal components and adds the change back to the original edges:

```python
from vae_harmonization.harmonization.edge import EdgeResidualHarmonizer

edge = EdgeResidualHarmonizer(lambda: make_variant("ours", target_site=0), pca_dim=100)
edge.fit(X_train, site_train, y_diag=dx_train)     # X_*: (subjects, edges) Fisher-z connectivity
X_test_harmonized = edge.transform(X_test, site_test)
```

Other variants of `make_variant`:

| name | what changes |
|---|---|
| `ours` | the method of the paper: protected diagnosis axis, pooled-target covariance |
| `ours-2d` | also protects symptom severity (`severity=` in `fit`) |
| `ours-marginal` | site means without conditioning on the labels |
| `ours-js` | James–Stein shrinkage of the site offsets |
| `ours-lw`, `ours-diag`, `ours-emp` | identity-target, diagonal-target or no covariance shrinkage |
| `ours-auto`, `barycenter` | geodesic relaxation, or the Wasserstein barycenter as the reference |
| `full-affine`, `mean-shift` | no protected subspace, or a mean shift only |
| `cholesky`, `zca` | other maps that match the same covariances |

To protect age and sex as well, pass `protected=("diagnosis", "age", "sex")` to `make_variant` and
`covariates={"age": age, "sex": sex}` to `fit`.

## Reproducing the paper

1. **Get the data** (see below) and turn it into the processed HDF5 files with the scripts in
   `scripts/preprocessing/`.
2. **Run the protocol** for a cohort, for example ABIDE-I:

   ```bash
   python -m vae_harmonization.run_vae_harmonization --dataset abide1_full \
       --methods default,extensions --classifiers lr,rf --mmd-min-site 5 --n-perm 200 \
       --protected-axis --fingerprint --network-analysis --downstream --unseen-site --federated --overlap
   ```

   Results go to `results/<cohort>[_tag]/`. Every scaler, PCA and harmonizer is fitted on the
   training fold only.
3. **Build the tables and figures** from the result files:

   ```bash
   python scripts/generate_tables.py           # LaTeX tables and number macros -> paper/tables/
   python scripts/generate_paper_figures.py    # figures -> paper/figures/
   ```

[REPRODUCE.md](REPRODUCE.md) lists every run behind the tables, with run times on a desktop CPU.

A quick check that needs no data (under a minute):

```bash
python tests/make_toy_cohort.py --out /tmp/toy
python -m vae_harmonization.run_vae_harmonization --config-yaml /tmp/toy.yaml --out-dir /tmp/toy_out \
    --n-folds 3 --methods raw,rep,ours --classifiers lr --skip-insample --no-plots
```

## Data

| cohort | sites | subjects | regions / edges | preprocessing | access |
|---|---|---|---|---|---|
| ABIDE-I | 17 | 778 | 200 / 19,900 | C-PAC, CC200 | public |
| ADHD-200 | 7 | 659 | 190 / 17,955 | Athena, CC200 | public |
| ABIDE-II | 10 | 534 | 195 / 18,915 | fMRIPrep, CC200 | public |
| REST-meta-MDD | 16 | 1,586 | 160 / 12,720 | DPARSF, Dosenbach-160 | on application |
| SRPBS (depression vs controls) | 6 | 809 | 140 / 9,730 | release pipeline, BAL atlas | on application |
| SRPBS traveling subjects | 12 scanners | 9 people, 397 sessions | 268 / 35,778 | release pipeline, Shen-268 | on application |

The connectivity data are not redistributed. The phenotypic tables of ABIDE and ADHD-200 and the
CC200-to-Yeo-7 mapping tables are included in `data/`, under the terms of their original releases.

**REST-meta-MDD.** The REST-meta-MDD Consortium grants access after a signed data-use agreement.
Request the Phase-1 functional ROI release from the Psychological Science Data Bank
([10.57760/sciencedb.o00115.00013](https://doi.org/10.57760/sciencedb.o00115.00013)) or at
rfmrilab@gmail.com, place it in `data/REST-meta-MDD/` and run
`python scripts/preprocessing/preprocess_rest_meta_mdd.py`. The script keeps the subject list of
Yan et al. (2019) and drops site 4, which duplicates site 14. Work that uses these data lists the
DIRECT Consortium as an author and cites Yan et al. (2019) and Chen et al. (2022).

**SRPBS.** ATR shares two releases after an approved application: the
[SRPBS Multidisorder Connectivity Dataset](https://bicr-resource.atr.jp/srpbsfc/) (cite Tanaka et
al., 2021, and Yahata et al., 2016) and the
[DecNef Harmonization Dataset](https://bicr.atr.jp/dcn/en/download/harmonization/) with the
traveling subjects (cite Yamashita et al., 2019). Extract them to `data/SRPBS/` and
`data/SourceCodeAndDATA/` and run `python scripts/preprocessing/preprocess_srpbs.py`. The
traveling-subject analysis is `python scripts/run_traveling.py`.

Nothing from the two restricted datasets is in this repository, including their result files, so
the tables that use them can be rebuilt only by researchers who have access.

## Repository layout

```
src/vae_harmonization/
  harmonization/   PAM, the Bures–Wasserstein, Cholesky and ZCA maps, barycenter, geodesic
                   relaxation, James–Stein site offsets and the edge-residual wrapper
  baselines/       ComBat, CovBat, ComBat-GAM, Riemannian methods (pyRiemann), CORAL, MMD map,
                   and bridges to the authors' R code of DeepComBat, RELIEF and SLACC
  evaluation/      leakage-free cross-validation protocol, per-fold PCA, probes, statistics
  analyses/        protected axis, fingerprinting, Yeo-7 networks, leave-one-site-out, effect
                   sizes, unseen sites, federated fitting, traveling subjects
  metrics.py       MMD, ICC, eta-squared and permutation nulls
  data_registry.py cohort definitions (paths, sites, reference site, covariates)
scripts/           preprocessing, table and figure generation, simulation, theorem check,
                   traveling-subject analysis
configs/           per-cohort settings (identical across cohorts)
results/           result files of the three public cohorts, the simulation and the theorem check
checkpoints/       frozen VAE checkpoints for the appendix comparison (optional)
data/              phenotypic tables and atlas mapping tables
third_party/       the DeepComBat R package (authors' code, MIT) with four documented patches
tests/             pytest suite on synthetic cohorts
```

## Citation

```bibtex
@article{sajben2026pam,
  author  = {Sajben, D{\'a}niel and Szegletes, Luca and {the DIRECT Consortium}},
  title   = {Moving less takes you further: minimum-displacement affine harmonization of
             multi-site functional connectivity with a label-protected subspace},
  journal = {Medical Image Analysis},
  note    = {Submitted},
  year    = {2026}
}
```

## License

The code is released under the MIT license ([LICENSE](LICENSE)). The vendored DeepComBat package
keeps its own license, and the data files in `data/` keep the terms of the datasets they come from.
