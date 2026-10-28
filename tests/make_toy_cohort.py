"""
Write a small synthetic multi-site cohort (HDF5 + YAML config) for CLI smoke tests.

    python tests/make_toy_cohort.py --out /tmp/toy            # -> /tmp/toy.h5, /tmp/toy.yaml
    python -m vae_harmonization.run_vae_harmonization --config-yaml /tmp/toy.yaml --out-dir /tmp/toy_out \
        --n-folds 3 --methods raw,rep,ours --classifiers lr --no-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import make_synthetic_cohort, write_synthetic_hdf5  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="path prefix; writes <out>.h5 and <out>.yaml")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)
    out = Path(a.out)
    cohort = make_synthetic_cohort(n_sites=4, sizes=(80, 60, 50, 40), d=8, seed=a.seed)
    h5 = write_synthetic_hdf5(str(out.with_suffix(".h5")), cohort, n_rois=20)
    yaml = out.with_suffix(".yaml")
    yaml.write_text(
        f"hdf5_path: {Path(h5).resolve().as_posix()}\n"
        "dataset_name: toy\npca_dim: 6\nlatent_dim: 4\nencoder_dims: [16, 8]\ndecoder_dims: [8, 16]\n"
        "max_epochs: 3\nn_cv_folds: 3\ntarget_site: 0\nmin_cov_samples: 10\n", encoding="utf-8")
    print("wrote", h5, "and", yaml)


if __name__ == "__main__":
    main()
