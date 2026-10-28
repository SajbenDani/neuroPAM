#!/usr/bin/env python3
"""
Recompute the paired fold-level comparisons (Wilcoxon, d_z, Holm within family, TOST) stored in
results/<run>/protocol_results.json from the per-fold scores in the same file.

    python scripts/recompute_comparisons.py                       # every results/*/protocol_results.json
    python scripts/recompute_comparisons.py results/abide1_full   # one run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vae_harmonization.evaluation.protocol import recompute_comparisons, to_jsonable  # noqa: E402


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    paths = [Path(a) / "protocol_results.json" if Path(a).is_dir() else Path(a) for a in argv] or \
        sorted((REPO / "results").glob("*/protocol_results.json"))
    for p in paths:
        if not p.exists():
            print("missing", p)
            continue
        res = json.load(open(p, encoding="utf-8"))
        recompute_comparisons(res)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(res), f, indent=1)
        print("updated", p)


if __name__ == "__main__":
    main()
