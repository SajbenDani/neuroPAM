"""Protected-subspace affine harmonization: transport maps, subspace fitting, estimator."""

from .affine import ProtectedAffineHarmonizer, SiteTransform, make_variant
from .base import Harmonizer, IdentityHarmonizer
from .subspace import (
    complement_basis,
    decompose,
    fit_covariate_axis,
    fit_diagnosis_axis,
    fit_severity_axis,
    orthonormal_protected_basis,
    recompose,
)
from .transport import (
    bures_wasserstein_barycenter,
    bures_wasserstein_distance2,
    bures_wasserstein_map,
    cholesky_map,
    estimate_covariance,
    expected_squared_displacement,
    gaussian_w2_distance2,
    geodesic_map,
    zca_map,
)

__all__ = [
    "Harmonizer", "IdentityHarmonizer", "ProtectedAffineHarmonizer", "SiteTransform", "make_variant",
    "fit_diagnosis_axis", "fit_severity_axis", "fit_covariate_axis", "orthonormal_protected_basis",
    "complement_basis", "decompose", "recompose",
    "estimate_covariance", "bures_wasserstein_map", "cholesky_map", "zca_map", "geodesic_map",
    "bures_wasserstein_distance2", "gaussian_w2_distance2", "expected_squared_displacement",
    "bures_wasserstein_barycenter",
]
