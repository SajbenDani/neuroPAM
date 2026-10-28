"""
vae_harmonization -- Minimum-displacement affine harmonization of multi-site
functional connectivity in a VAE latent space.

Sub-packages
------------
harmonization   transport maps (Bures-Wasserstein, geodesic, barycenter),
                protected-subspace fitting, the ``ProtectedAffineHarmonizer``
baselines       ComBat / CovBat / DeepComBat / ImUnity as fit/transform estimators
evaluation      the leakage-free per-fold protocol, probes, figures
analyses        fingerprinting, network preservation, downstream, simulations
"""

__version__ = "1.0.0"
