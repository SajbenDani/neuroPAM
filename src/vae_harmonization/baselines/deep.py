"""
deep.py -- DeepComBat (Hu et al. 2024) through the authors' R package.

The package is vendored in ``third_party/DeepComBat`` (upstream commit bf0eda1, two documented
compatibility patches, see ``third_party/DeepComBat/PATCHES.md``) and must be installed in the R library
that ``Rscript`` uses. The bridge follows the package README's recipe for external harmonization:

* ``fit``: ``deepcombat_setup(~ diagnosis, ~ site, X, covariates, reference_batch = hub,
  use_default_optim = FALSE)``, ``deepcombat_trainer`` with ``optim_adamw(lr = 0.01)`` and the default
  training schedule, then ``deepcombat_harmonize`` on the training rows. The trained network
  (``torch_save``) and the standardization, covariate scaling and ComBat estimates (``saveRDS``) are kept
  in a temporary directory.
* ``transform``: ``deepcombat_setup_from_train`` and ``deepcombat_harmonize(setup, model, train_output)``,
  which applies the stored estimates to new subjects.

DeepComBat models the covariates of every subject it adjusts, so like ComBat it needs the diagnosis of
held-out subjects. The package's apply step rebuilds its batch indicator from the rows it is given, so a
held-out set that lacks a training site would shift the estimates of the later sites. One stored training
row per site is appended before the call and removed after it; harmonization is row-wise given the
estimates, so the anchors do not change the other rows. Sites with fewer than two training scans have no
ComBat variance and pass through unchanged, as in the ComBat baseline.

The default architecture (three hidden tanh layers, latent size the power of two nearest a quarter of
the number of features) is meant for tens to a few hundred features. On the 100 PCA scores it is
100-83-66-49-32. On the roughly 20,000 raw edges the default network has about 600 million parameters,
which does not fit a desktop CPU, so the method runs on the PCA scores.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import weakref
from pathlib import Path

import numpy as np

from ..harmonization.base import Harmonizer
from ..logging_utils import get_logger
from .relief_r import rscript_available

log = get_logger(__name__)

_DEEPCOMBAT_R = r"""
args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]
work <- args[2]
hub <- paste0("s", args[3])
seed <- as.integer(args[4])
suppressPackageStartupMessages({
  library(DeepComBat)
  library(torch)
})
read_mat <- function(f) as.matrix(read.csv(file.path(work, f), header = FALSE))
read_vec <- function(f) read.csv(file.path(work, f), header = FALSE)[, 1]
if (mode == "fit") {
  set.seed(seed)
  torch_manual_seed(seed)
  X <- read_mat("X_fit.csv")
  site <- read_vec("site_fit.csv")
  dx <- read_vec("dx_fit.csv")
  lev <- unique(site)
  covs <- data.frame(diagnosis = dx, site = factor(paste0("s", site), levels = paste0("s", lev)))
  setup <- deepcombat_setup(~ diagnosis, ~ site, X, covs, reference_batch = hub,
                            use_default_optim = FALSE)
  model <- deepcombat_trainer(setup, optimizer = optim_adamw(setup$cvae$parameters, lr = 0.01))
  harmonized <- deepcombat_harmonize(setup, model)
  torch_save(model, file.path(work, "model.pt"))
  saveRDS(list(input = setup$input, cvae_settings = setup$cvae_settings,
               latent_estimates = harmonized$additional_outputs$latent_estimates,
               residual_estimates = harmonized$additional_outputs$residual_estimates,
               dims = setup$cvae$dims),
          file.path(work, "fit.rds"))
  writeLines(paste(setup$cvae$dims, collapse = ","), file.path(work, "dims.txt"))
} else if (mode == "apply") {
  fit <- readRDS(file.path(work, "fit.rds"))
  model <- torch_load(file.path(work, "model.pt"))
  model$eval()
  setup_train <- structure(list(dataloader = NULL, cvae = NULL, optimizer = NULL, input = fit$input,
                                cvae_settings = fit$cvae_settings), class = "deepcombat_setup_object")
  train_out <- structure(list(additional_outputs = list(latent_estimates = fit$latent_estimates,
                                                        residual_estimates = fit$residual_estimates)),
                         class = "deepcombat_harmonize_object")
  X <- read_mat("X_new.csv")
  site <- read_vec("site_new.csv")
  dx <- read_vec("dx_new.csv")
  covs <- data.frame(diagnosis = dx, site = paste0("s", site))
  setup_new <- deepcombat_setup_from_train(setup_train, ~ diagnosis, ~ site, X, covs, reference_batch = hub)
  out <- deepcombat_harmonize(setup_new, model, train_out)
  write.table(out$harmonized, file.path(work, "X_new_h.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
} else {
  stop("unknown mode")
}
"""


def deepcombat_available() -> bool:
    if not rscript_available():
        return False
    try:
        proc = subprocess.run(["Rscript", "--vanilla", "-e",
                               "quit(status = as.integer(!requireNamespace('DeepComBat', quietly = TRUE)))"],
                              capture_output=True, text=True, timeout=120)
        return proc.returncode == 0
    except Exception:
        return False


class DeepComBatHarmonizer(Harmonizer):
    name = "DeepComBat"
    uses_labels_at_transform = True
    min_site_n = 2

    def __init__(self, target_site: int, seed: int = 42, timeout: int = 3600, name: str | None = None):
        self.target_site = int(target_site)
        self.seed = int(seed)
        self.timeout = int(timeout)
        if name is not None:
            self.name = name
        if not rscript_available():
            raise ImportError("Rscript is not on PATH")

    def _run(self, mode: str):
        script = Path(self.work_) / "deepcombat.R"
        if not script.exists():
            script.write_text(_DEEPCOMBAT_R, encoding="utf-8")
        proc = subprocess.Popen(["Rscript", "--vanilla", str(script), mode, self.work_, str(self.target_site),
                                 str(self.seed)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            _out, err = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
            else:
                proc.kill()
            proc.communicate()
            raise RuntimeError(f"DeepComBat ({mode}) exceeded {self.timeout}s and was stopped")
        if proc.returncode != 0:
            raise RuntimeError(f"DeepComBat ({mode}) failed: {(err or '')[-800:]}")

    @staticmethod
    def _save(path: Path, X, site, dx):
        np.savetxt(path.parent / f"X_{path.name}.csv", X, delimiter=",")
        np.savetxt(path.parent / f"site_{path.name}.csv", np.asarray(site).reshape(-1, 1), delimiter=",", fmt="%d")
        np.savetxt(path.parent / f"dx_{path.name}.csv", np.asarray(dx).reshape(-1, 1), delimiter=",", fmt="%d")

    def fit(self, X, y_site, y_diag=None, **aux):
        if y_diag is None:
            raise ValueError("DeepComBat with a diagnosis covariate needs y_diag")
        X = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        y_diag = np.asarray(y_diag).astype(int)
        sites, counts = np.unique(y_site, return_counts=True)
        tiny = sites[counts < self.min_site_n]
        if tiny.size:
            log.warning("fit: sites %s have < %d training scans; DeepComBat leaves them unchanged",
                        tiny.tolist(), self.min_site_n)
        keep = ~np.isin(y_site, tiny)
        if self.target_site not in y_site[keep]:
            raise ValueError("target site absent from training data")
        self.work_ = tempfile.mkdtemp(prefix="deepcombat_")
        self._cleanup = weakref.finalize(self, shutil.rmtree, self.work_, True)
        self._save(Path(self.work_) / "fit", X[keep], y_site[keep], y_diag[keep])
        self._run("fit")
        self.dims_ = (Path(self.work_) / "dims.txt").read_text().strip()
        self.site_levels_ = np.unique(y_site[keep])
        first = np.array([np.flatnonzero(y_site[keep] == s)[0] for s in self.site_levels_])
        self.anchor_X_ = X[keep][first]
        self.anchor_site_ = self.site_levels_.copy()
        self.anchor_dx_ = y_diag[keep][first]
        return self

    def transform(self, X, y_site, y_diag=None, **aux):
        if y_diag is None:
            raise ValueError("DeepComBat needs the diagnosis of the subjects it adjusts")
        X = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        y_diag = np.asarray(y_diag).astype(int)
        seen = np.isin(y_site, self.site_levels_)
        out = X.copy()
        if not seen.any():
            return out
        n_seen = int(seen.sum())
        rows = np.vstack([X[seen], self.anchor_X_])
        sites = np.concatenate([y_site[seen], self.anchor_site_])
        dx = np.concatenate([y_diag[seen], self.anchor_dx_])
        self._save(Path(self.work_) / "new", rows, sites, dx)
        self._run("apply")
        harmonized = np.loadtxt(Path(self.work_) / "X_new_h.csv", delimiter=",", ndmin=2)
        out[seen] = harmonized[:n_seen]
        return out

    def describe(self):
        d = super().describe()
        d.update({"target_site": self.target_site, "architecture": getattr(self, "dims_", None),
                  "package": "DeepComBat (R, vendored, commit bf0eda1)", "uses_labels_at_transform": True})
        return d
