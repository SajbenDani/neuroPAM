"""
relief_r.py -- RELIEF and SLACC via an Rscript bridge.

Both methods ship as R packages (Zhang, Oliver, Voineskos & Park 2023;
Zhang, Tuzhilina & Park 2026).  The Python wrapper writes a temporary CSV,
calls ``Rscript``, and reads the harmonized matrix back.  RELIEF expects
features in rows.  SLACC expects a triangular vector that includes the
diagonal, so a zero diagonal is inserted and removed around the call.
If Rscript or the package is missing the factory raises ``ImportError`` and
the CLI skips the row.  Neither package has an apply step for new subjects.
Held-out rows of a site seen at fit time are therefore adjusted in a second call
on the training rows plus those held-out rows.  RELIEF runs without covariates
by default: its README builds ``mod`` with ``model.matrix()`` and advises leaving
the covariate that will be tested out of it, and the diagnosis is what the
protocol tests.  With ``use_covariates=True`` the diagnosis enters ``mod`` as
``model.matrix(~ diagnosis)`` (intercept included) and every adjusted row needs
its diagnosis, held-out rows included.  SLACC always models the diagnosis.
Sites unseen at fit time, and sites with a single training scan, pass through
unchanged.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..harmonization.base import Harmonizer
from ..logging_utils import get_logger

log = get_logger(__name__)

_RELIEF_R = r"""
args <- commandArgs(trailingOnly = TRUE)
X <- as.matrix(read.csv(args[1], header = FALSE))
site <- as.integer(read.csv(args[2], header = FALSE)[, 1])
diag <- as.integer(read.csv(args[3], header = FALSE)[, 1])
out <- args[4]
use_cov <- length(args) >= 5 && args[5] == "1"
suppressPackageStartupMessages({
  library(Matrix)
  library(MASS)
  library(RELIEF)
})
# relief() stores features in rows and samples in columns. verbose=TRUE is required:
# the installed package only sets the covariate count inside that branch.
mod <- if (use_cov) model.matrix(~ diagnosis, data.frame(diagnosis = factor(diag))) else NULL
fit <- RELIEF::relief(t(X), batch = factor(site), mod = mod, verbose = TRUE)
write.table(t(as.matrix(fit$dat.relief)), file = out, sep = ",", row.names = FALSE, col.names = FALSE)
"""

_SLACC_R = r"""
args <- commandArgs(trailingOnly = TRUE)
X <- as.matrix(read.csv(args[1], header = FALSE))
site <- as.integer(read.csv(args[2], header = FALSE)[, 1])
diag <- as.integer(read.csv(args[3], header = FALSE)[, 1])
out <- args[4]
suppressPackageStartupMessages({
  library(Matrix)
  library(MASS)
  library(SLACC)
})
mod <- cbind(diagnosis = as.numeric(diag))
batch <- factor(site)

# Stored edges are the strict upper triangle. SLACC reads a triangle that includes
# the diagonal, in column-major order. Insert a zero diagonal, harmonize, drop it.
n_off <- (1 + sqrt(1 + 8 * ncol(X))) / 2
if (abs(n_off - round(n_off)) > 1e-6) stop("SLACC needs a strict-upper-triangle edge vector")
n <- as.integer(round(n_off))
rr <- unlist(lapply(0:(n - 2L), function(r) rep(r, n - 1L - r)))
cc <- unlist(lapply(0:(n - 2L), function(r) (r + 1L):(n - 1L)))
dest <- cc * (cc + 1) / 2 + rr + 1
Xin <- matrix(0, nrow(X), n * (n + 1) / 2)
Xin[, dest] <- X
fit <- SLACC::SLACC(Xin, batch = batch, mod = mod)
applied <- SLACC::SLACC_harmonize(Xin, fit, mod = mod, batch = batch)
Xout <- applied$dat_harmonized[, dest, drop = FALSE]
write.table(Xout, file = out, sep = ",", row.names = FALSE, col.names = FALSE)
"""


def rscript_available() -> bool:
    return shutil.which("Rscript") is not None


# Wall-time limit of one Rscript call. RELIEF on REST-meta-MDD (about 1,400 training scans) needs several
# minutes per call and more when other jobs share the CPU.
R_TIMEOUT = int(os.environ.get("VAEH_R_TIMEOUT", "3600"))


class _RBridgeHarmonizer(Harmonizer):
    script: str = ""
    package: str = ""
    use_covariates: bool = True

    def __init__(self, name: str | None = None, use_covariates: bool | None = None):
        if name is not None:
            self.name = name
        if use_covariates is not None:
            self.use_covariates = bool(use_covariates)
        if not rscript_available():
            raise ImportError("Rscript is not on PATH")

    @property
    def uses_labels_at_transform(self) -> bool:
        return self.use_covariates

    def _run(self, X, y_site, y_diag):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.savetxt(tmp / "X.csv", X, delimiter=",")
            np.savetxt(tmp / "site.csv", np.asarray(y_site).reshape(-1, 1), delimiter=",", fmt="%d")
            np.savetxt(tmp / "diag.csv", np.asarray(y_diag).reshape(-1, 1), delimiter=",", fmt="%d")
            script = tmp / "run.R"
            script.write_text(self.script, encoding="utf-8")
            out = tmp / "Xh.csv"
            proc = subprocess.Popen(
                ["Rscript", "--vanilla", str(script), str(tmp / "X.csv"), str(tmp / "site.csv"),
                 str(tmp / "diag.csv"), str(out), "1" if self.use_covariates else "0"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            try:
                _stdout, stderr = proc.communicate(timeout=R_TIMEOUT)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                   capture_output=True)
                else:
                    proc.kill()
                proc.communicate()
                raise RuntimeError(f"{self.package} exceeded {R_TIMEOUT}s and was stopped")
            if proc.returncode != 0 or not out.exists():
                raise RuntimeError(f"{self.package} Rscript failed: {(stderr or '')[-500:]}")
            return np.loadtxt(out, delimiter=",")

    min_site_n = 2      # per-site scaling is undefined for a single scan

    def fit(self, X, y_site, y_diag=None, **aux):
        X = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        if y_diag is None:
            y_diag = np.zeros(len(y_site), dtype=int)
        sites, counts = np.unique(y_site, return_counts=True)
        tiny = sites[counts < self.min_site_n]
        if tiny.size:
            log.warning("fit: sites %s have < %d training scans; %s leaves them unchanged",
                        tiny.tolist(), self.min_site_n, self.package)
        keep = ~np.isin(y_site, tiny)
        self.site_levels_ = sites[counts >= self.min_site_n]
        self.X_fit_, self.y_site_fit_ = X.copy(), y_site.copy()
        self.X_train_ = X[keep]
        self.y_site_train_ = y_site[keep]
        self.y_diag_train_ = np.asarray(y_diag).astype(int)[keep]
        self.X_train_harmonized_ = X.copy()
        self.X_train_harmonized_[keep] = self._run(self.X_train_, self.y_site_train_, self.y_diag_train_)
        return self

    def transform(self, X, y_site, y_diag=None, **aux):
        X = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        # the training rows themselves: return the fit's own output (refitting them together with a copy of
        # themselves would duplicate every subject)
        if (X.shape == self.X_fit_.shape and np.array_equal(y_site, self.y_site_fit_)
                and np.array_equal(X, self.X_fit_)):
            return self.X_train_harmonized_.copy()
        # no apply step exists: refit on the training rows plus the seen-site held-out rows (with the
        # diagnoses of all of them when the covariate model is used)
        seen = np.isin(y_site, self.site_levels_)
        out = X.copy()
        if not seen.any():
            return out
        if y_diag is None:
            if self.use_covariates:
                raise ValueError(f"{self.package} needs the diagnosis of the subjects it adjusts")
            y_diag = np.zeros(len(y_site), dtype=int)       # not read by the R script without covariates
        y_diag = np.asarray(y_diag).astype(int)
        Xall = np.vstack([self.X_train_, X[seen]])
        sall = np.concatenate([self.y_site_train_, y_site[seen]])
        dall = np.concatenate([self.y_diag_train_, y_diag[seen]])
        hall = self._run(Xall, sall, dall)
        out[seen] = hall[len(self.X_train_):]
        return out

    def describe(self):
        d = super().describe()
        d.update({"package": self.package, "use_covariates": self.use_covariates,
                  "uses_labels_at_transform": self.use_covariates})
        return d


class RELIEFHarmonizer(_RBridgeHarmonizer):
    name = "RELIEF"
    package = "RELIEF"
    script = _RELIEF_R
    use_covariates = False


class SLACCHarmonizer(_RBridgeHarmonizer):
    name = "SLACC"
    package = "SLACC"
    script = _SLACC_R
