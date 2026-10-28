"""Baseline harmonization methods with a common fit/transform interface."""

from ..harmonization.base import Harmonizer, IdentityHarmonizer
from .combat import ComBatGAMHarmonizer, ComBatHarmonizer, CovBatHarmonizer
from .coral import CORALHarmonizer
from .deep import DeepComBatHarmonizer, deepcombat_available
from .mmd_affine import MMDAffineHarmonizer
from .relief_r import RELIEFHarmonizer, SLACCHarmonizer, rscript_available
from .riemannian import (
    EuclideanAlignmentHarmonizer,
    RiemannianProcrustesHarmonizer,
    RiemannianRecenteringHarmonizer,
    RLETHarmonizer,
    log_euclidean_vectors,
)

__all__ = [
    "Harmonizer", "IdentityHarmonizer",
    "ComBatHarmonizer", "CovBatHarmonizer", "ComBatGAMHarmonizer",
    "DeepComBatHarmonizer", "deepcombat_available",
    "CORALHarmonizer", "MMDAffineHarmonizer",
    "RiemannianRecenteringHarmonizer", "EuclideanAlignmentHarmonizer",
    "RiemannianProcrustesHarmonizer", "RLETHarmonizer", "log_euclidean_vectors",
    "RELIEFHarmonizer", "SLACCHarmonizer", "rscript_available",
]
