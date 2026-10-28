"""
dataset.py -- PyTorch Dataset and DataLoader utilities for ABIDE FC data.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from .config import VAEConfig
from .logging_utils import get_logger

log = get_logger(__name__)


class ABIDEDataset(Dataset):
    """
    Loads the full HDF5 once and serves (x, site, diagnosis) tuples.

    Parameters
    ----------
    hdf5_path : path to HDF5 file with X, y_site, y_diagnosis, subject_ids
    scaler    : optional *fitted* StandardScaler (for val/test sets)
    pca       : optional *fitted* PCA (for val/test sets)
    """

    def __init__(
        self,
        hdf5_path,
        scaler: StandardScaler | None = None,
        pca: PCA | None = None,
    ):
        with h5py.File(hdf5_path, "r") as f:
            self.X = f["X"][:].astype(np.float32)
            self.y_site = f["y_site"][:].astype(np.int64)
            self.y_diag = f["y_diagnosis"][:].astype(np.int64)
            self.subject_ids = [s.decode() if isinstance(s, bytes) else s
                                for s in f["subject_ids"][:]]

        self.X_raw = self.X.copy()

        if scaler is not None and pca is not None:
            self.X = scaler.transform(self.X).astype(np.float32)
            self.X = pca.transform(self.X).astype(np.float32)
            self.scaler = scaler
            self.pca = pca
        else:
            self.scaler = None
            self.pca = None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])
        site = torch.tensor(self.y_site[idx], dtype=torch.long)
        diag = torch.tensor(self.y_diag[idx], dtype=torch.float32)
        return x, site, diag

    def inverse_to_raw(self, X_model: np.ndarray) -> np.ndarray:
        """Map model-space vectors (PCA space) back to raw 19,900-dim FC space."""
        if self.scaler is None or self.pca is None:
            return X_model
        X_scaled = self.pca.inverse_transform(X_model)
        return self.scaler.inverse_transform(X_scaled).astype(np.float32)

    def get_site_counts(self) -> dict[int, int]:
        """Return {site_id: count} dict."""
        unique, counts = np.unique(self.y_site, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


# -- Split persistence ---------------------------------------------------------

def save_split_indices(path: str | Path, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    """Save train/val split indices to a .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, train_idx=train_idx, val_idx=val_idx)
    log.info("split indices saved to %s", path)


def load_split_indices(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load train/val split indices from a .npz file."""
    data = np.load(path)
    return data["train_idx"], data["val_idx"]


# -- DataLoader factory --------------------------------------------------------

def get_dataloaders(
    cfg: VAEConfig,
) -> tuple[DataLoader, DataLoader, StandardScaler, PCA, ABIDEDataset]:
    """
    Build train / val DataLoaders with stratified split by site x diagnosis.

    If cfg.split_path is set and the file exists, indices are loaded from disk.
    If cfg.split_path is set but missing, indices are created and saved.

    Returns
    -------
    train_loader, val_loader, fitted_scaler, fitted_pca, full_dataset
    """
    full_ds = ABIDEDataset(cfg.hdf5_path)

    # Stratify on combined site x diagnosis label
    combined_labels = full_ds.y_site * 2 + full_ds.y_diag

    # Determine split indices
    split_path = Path(cfg.split_path) if cfg.split_path else None

    if split_path is not None and split_path.exists():
        log.info("loading split indices from %s", split_path)
        train_idx, val_idx = load_split_indices(split_path)
    else:
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=cfg.val_fraction,
            random_state=cfg.seed,
        )
        try:
            train_idx, val_idx = next(sss.split(full_ds.X, combined_labels))
        except ValueError:
            # Fall back to site-only stratification when some site x diag
            # combinations have too few members (e.g. very small sites)
            log.warning("site x diag stratification failed; falling back to site-only stratification")
            from sklearn.model_selection import ShuffleSplit
            try:
                sss2 = StratifiedShuffleSplit(
                    n_splits=1, test_size=cfg.val_fraction,
                    random_state=cfg.seed,
                )
                train_idx, val_idx = next(sss2.split(full_ds.X, full_ds.y_site))
            except ValueError:
                log.warning("site-only stratification also failed; using a random split")
                ss = ShuffleSplit(
                    n_splits=1, test_size=cfg.val_fraction,
                    random_state=cfg.seed,
                )
                train_idx, val_idx = next(ss.split(full_ds.X))
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)

        if split_path is not None:
            save_split_indices(split_path, train_idx, val_idx)

    # Fit scaler and PCA on training split only
    scaler = StandardScaler().fit(full_ds.X[train_idx])
    X_scaled = scaler.transform(full_ds.X).astype(np.float32)
    pca = PCA(n_components=cfg.pca_dim, random_state=cfg.seed)
    pca.fit(X_scaled[train_idx])

    full_ds.X = pca.transform(X_scaled).astype(np.float32)
    full_ds.scaler = scaler
    full_ds.pca = pca

    train_subset = Subset(full_ds, train_idx)
    val_subset = Subset(full_ds, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, scaler, pca, full_ds
