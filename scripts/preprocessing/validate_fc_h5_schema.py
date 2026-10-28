#!/usr/bin/env python3
"""Validate FC HDF5 schema compatibility with ABIDE-I pipeline expectations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REQUIRED_KEYS = ["X", "y_diagnosis", "y_site", "subject_ids"]


def validate_file(h5_path: Path, expected_features: int) -> tuple[bool, list[str]]:
    errors: list[str] = []

    if not h5_path.exists():
        return False, [f"HDF5 file not found: {h5_path}"]

    with h5py.File(h5_path, "r") as f:
        for key in REQUIRED_KEYS:
            if key not in f:
                errors.append(f"Missing dataset key: {key}")

        if errors:
            return False, errors

        X = f["X"][:]
        y_dx = f["y_diagnosis"][:]
        y_site = f["y_site"][:]
        sub_ids = f["subject_ids"][:]

        if X.ndim != 2:
            errors.append(f"X must be 2D, got shape {X.shape}")
        else:
            if X.shape[1] != expected_features:
                errors.append(
                    f"Feature dimension mismatch: expected {expected_features}, got {X.shape[1]}"
                )

        n = X.shape[0] if X.ndim == 2 else None
        if n is not None:
            if y_dx.shape[0] != n:
                errors.append(f"y_diagnosis length mismatch: expected {n}, got {y_dx.shape[0]}")
            if y_site.shape[0] != n:
                errors.append(f"y_site length mismatch: expected {n}, got {y_site.shape[0]}")
            if sub_ids.shape[0] != n:
                errors.append(f"subject_ids length mismatch: expected {n}, got {sub_ids.shape[0]}")

        if not np.issubdtype(X.dtype, np.floating):
            errors.append(f"X dtype must be floating, got {X.dtype}")

        finite_mask = np.isfinite(X)
        if not finite_mask.all():
            bad = np.size(X) - int(finite_mask.sum())
            errors.append(f"X contains non-finite values: {bad}")

    return len(errors) == 0, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FC HDF5 schema")
    parser.add_argument("--h5", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument(
        "--expected-features",
        type=int,
        default=19900,
        help="Expected feature dimension (default: 19900)",
    )
    args = parser.parse_args()

    ok, errors = validate_file(Path(args.h5), int(args.expected_features))
    if ok:
        print("PASS: HDF5 schema is compatible with ABIDE-I pipeline expectations")
        return

    print("FAIL: HDF5 schema validation failed")
    for err in errors:
        print(f" - {err}")
    sys.exit(1)


if __name__ == "__main__":
    main()
