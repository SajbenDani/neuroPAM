"""
config.py -- Centralised hyperparameters for the FC-VAE harmonization pipeline.

Defaults reproduce the paper's configuration (100 PCA dims -> [512, 256] ->
64-D latent, beta = 0.1 with 50-epoch warm-up, free bits 0.125, AdamW 1e-3,
300 epochs with patience 30, seed 42).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent   # src/vae_harmonization
_REPO_DIR = _SCRIPT_DIR.parent.parent          # src -> repo root


@dataclass
class VAEConfig:
    # -- Paths ---------------------------------------------------------
    hdf5_path: Path = _REPO_DIR / "data" / "processed" / "abide1_full_fc.h5"
    out_dir: Path = _REPO_DIR / "results"
    checkpoint_dir: Path = _REPO_DIR / "checkpoints"
    split_path: str = ""  # optional .npz with train/val indices

    # -- Data ----------------------------------------------------------
    dataset_name: str = "abide1_full"
    input_dim: int = 100
    pca_dim: int = 100
    n_sites: int = 17
    val_fraction: float = 0.15
    seed: int = 42

    # -- Architecture --------------------------------------------------
    encoder_dims: list[int] = field(default_factory=lambda: [512, 256])
    decoder_dims: list[int] = field(default_factory=lambda: [256, 512])
    latent_dim: int = 64
    head_hidden_dim: int = 16
    dropout_main: float = 0.0
    dropout_head: float = 0.0
    leaky_relu_slope: float = 0.2

    # -- Loss weights --------------------------------------------------
    lambda_recon: float = 1.0
    lambda_site: float = 0.0   # adversarial head is disabled: pure reconstruction VAE
    lambda_bio: float = 0.0    # diagnosis head is disabled: no labels during representation learning

    # -- KL annealing (beta-VAE warmup) --------------------------------
    beta_final: float = 0.1
    beta_warmup_epochs: int = 50
    free_bits: float = 0.125       # nats per latent dimension

    # -- GRL schedule (only used when lambda_site > 0) -------------------
    gamma_grl: float = 10.0

    # -- Training ------------------------------------------------------
    batch_size: int = 64
    max_epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    patience: int = 30
    lr_patience: int = 10
    lr_factor: float = 0.5
    lr_min: float = 1e-6

    # -- Evaluation ----------------------------------------------------
    n_cv_folds: int = 10
    target_site: int = 5   # NYU in abide1_full

    # -- Protected-subspace affine harmonization ------------------------
    bio_svm_C: float = 1.0        # LinearSVC regularisation for the diagnosis axis
    min_cov_samples: int = 10     # sites with fewer training subjects get a mean shift only
    ridge: float = 1e-6           # covariance ridge
    covariance: str = "pooled"    # per-site covariance estimator: pooled | lw | diagonal | empirical

    # -- Site / diagnosis labels ----------------------------------------
    site_names: dict[int, str] = field(default_factory=dict)
    diag_names: dict[int, str] = field(default_factory=lambda: {0: "TC", 1: "Case"})
    site_colors: dict[int, str] = field(default_factory=dict)
    diag_colors: dict[int, str] = field(default_factory=lambda: {0: "#4e79a7", 1: "#e15759"})

    # -- Factory methods -----------------------------------------------

    @classmethod
    def from_dataset(cls, name: str, **overrides) -> "VAEConfig":
        """Create a config pre-populated for a registered dataset."""
        from .data_registry import get_dataset_config

        ds = get_dataset_config(name)
        site_list = ds["site_names"] or []
        n_sites = ds["n_sites"] or len(site_list)
        kwargs = dict(
            dataset_name=name,
            hdf5_path=ds["hdf5_path"],
            n_sites=n_sites,
            pca_dim=ds.get("pca_dim", 100),
            input_dim=ds.get("pca_dim", 100),
            target_site=ds.get("target_site", 0) or 0,
            site_names={i: s for i, s in enumerate(site_list)},
            site_colors=site_palette(n_sites),
            diag_names={0: "TC", 1: ds.get("case_label", "Case")},
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VAEConfig":
        """Load config from a YAML file (a ``dataset_name`` key selects the preset)."""
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        from .data_registry import DATASET_REGISTRY
        name = d.pop("dataset_name", None)
        # a registered name loads the preset; any other name is an ad-hoc cohort (synthetic / new data)
        cfg = cls.from_dataset(name) if name in DATASET_REGISTRY else cls(dataset_name=name or "custom")
        for k, v in d.items():
            if hasattr(cfg, k):
                if k in ("hdf5_path", "out_dir", "checkpoint_dir"):
                    v = Path(v)
                setattr(cfg, k, v)
        return cfg


def site_palette(n_sites: int) -> dict[int, str]:
    """Deterministic tab10/tab20 colours for any number of sites."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    cmap = plt.cm.tab20 if n_sites > 10 else plt.cm.tab10
    arr = cmap(np.linspace(0, 1, max(n_sites, 2)))
    return {i: matplotlib.colors.rgb2hex(arr[i]) for i in range(n_sites)}
