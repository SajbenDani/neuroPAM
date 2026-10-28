"""Evaluation: leakage-free protocol, probes, in-sample diagnostics, figures."""

from .classifiers import cv_evaluate, fit_and_score, format_cv, make_classifier
from .insample import reconstruction_metrics, run_insample_diagnostics
from .latents import decode_latents, decode_to_fc, encode_features, extract_latents
from .protocol import MethodSpec, compute_comparisons, recompute_comparisons, run_protocol, summary_table, to_jsonable

__all__ = [
    "MethodSpec", "run_protocol", "summary_table", "to_jsonable",
    "make_classifier", "fit_and_score", "cv_evaluate", "format_cv",
    "extract_latents", "encode_features", "decode_latents", "decode_to_fc",
    "run_insample_diagnostics", "reconstruction_metrics",
]
