"""
mmd_affine.py -- Per-site affine map fitted by gradient descent on MMD to the hub.

Our own comparison method, not a re-implementation of a published one: it asks what an affine map
reaches when it is fitted to a distribution distance instead of to the first two moments. The map of site
``s`` is ``T_s(x) = A_s (x - mu_s) + mu_hub``. ``A_s`` starts at the identity and minimizes

    MMD^2_u(T_s(X_s), X_hub) + rho * ||A_s - I||_F^2

(unbiased RBF MMD, bandwidth sigma from the median pairwise distance of up to 400 training rows,
gamma = 1 / (2 sigma^2), rho = 0.01) by 200 Adam steps with learning rate 0.05. The penalty keeps the
fit near a mean shift.
"""

from __future__ import annotations

import numpy as np
import torch

from ..harmonization.base import Harmonizer
from ..logging_utils import get_logger

log = get_logger(__name__)


def _mmd2(X, Y, gamma):
    """Unbiased RBF MMD^2, differentiable in X."""
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        return X.new_zeros(())
    XX = torch.cdist(X, X, p=2).pow(2)
    YY = torch.cdist(Y, Y, p=2).pow(2)
    XY = torch.cdist(X, Y, p=2).pow(2)
    kxx = torch.exp(-gamma * XX)
    kyy = torch.exp(-gamma * YY)
    kxy = torch.exp(-gamma * XY)
    kxx = kxx - torch.diag(torch.diag(kxx))
    kyy = kyy - torch.diag(torch.diag(kyy))
    return kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2.0 * kxy.mean()


class MMDAffineHarmonizer(Harmonizer):
    name = "MMD affine"

    def __init__(self, target_site: int, steps: int = 200, lr: float = 5e-2, ridge: float = 1e-2,
                 seed: int = 42, name: str | None = None):
        self.target_site = int(target_site)
        self.steps = steps
        self.lr = lr
        self.ridge = ridge
        self.seed = seed
        if name is not None:
            self.name = name

    def fit(self, X, y_site, y_diag=None, **aux):
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        if self.target_site not in np.unique(y_site):
            raise ValueError("target site absent from training data")
        hub = torch.as_tensor(X[y_site == self.target_site], dtype=torch.float64)
        mu_hub = hub.mean(dim=0)
        # median-heuristic gamma on the training matrix (label-free)
        with torch.no_grad():
            Z = torch.as_tensor(X, dtype=torch.float64)
            if Z.shape[0] > 400:
                idx = torch.randperm(Z.shape[0])[:400]
                Z = Z[idx]
            d = torch.cdist(Z, Z, p=2)
            sigma = float(d[d > 0].median().clamp_min(1e-6))
        gamma = 1.0 / (2.0 * sigma * sigma)
        self.gamma_ = gamma
        self.mu_hub_ = mu_hub.numpy()
        self.means_ = {}
        self.maps_ = {}
        for s in np.unique(y_site):
            s = int(s)
            Xs = torch.as_tensor(X[y_site == s], dtype=torch.float64)
            mu = Xs.mean(dim=0)
            self.means_[s] = mu.detach().numpy()
            if s == self.target_site:
                self.maps_[s] = np.eye(X.shape[1])
                continue
            A = torch.eye(X.shape[1], dtype=torch.float64, requires_grad=True)
            opt = torch.optim.Adam([A], lr=self.lr)
            Xc = Xs - mu
            for _ in range(self.steps):
                opt.zero_grad()
                mapped = Xc @ A.T + mu_hub
                loss = _mmd2(mapped, hub, gamma) + self.ridge * torch.sum((A - torch.eye(A.shape[0], dtype=A.dtype)) ** 2)
                loss.backward()
                opt.step()
            self.maps_[s] = A.detach().numpy()
        return self

    def transform(self, X, y_site, **aux):
        X = np.asarray(X, dtype=np.float64)
        y_site = np.asarray(y_site).astype(int)
        out = X.copy()
        unknown = []
        for s in np.unique(y_site):
            s = int(s)
            m = y_site == s
            A = self.maps_.get(s)
            mu = self.means_.get(s)
            if A is None or mu is None:
                unknown.append(s)
                continue
            out[m] = (X[m] - mu) @ A.T + self.mu_hub_
        if unknown:
            log.warning("transform: sites %s were not present at fit time; left unchanged", unknown)
        return out

    def describe(self):
        d = super().describe()
        d.update({"target_site": self.target_site, "steps": self.steps, "ridge": self.ridge,
                  "gamma": getattr(self, "gamma_", None), "uses_labels_at_transform": False})
        return d
