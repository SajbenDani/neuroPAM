"""Star-vs-chain-vs-barycenter simulation (Appendix); CPU, ~1 min for the default settings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vae_harmonization.analyses.simulation import run_star_vs_chain  # noqa: E402
from vae_harmonization.evaluation.protocol import to_jsonable  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-sites", type=int, default=8)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--n-per-site", type=int, default=40)
    p.add_argument("--n-rep", type=int, default=100)
    p.add_argument("--shared", type=float, default=0.7)
    p.add_argument("--mean-shift", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(REPO / "results" / "simulation" / "star_vs_chain.json"))
    a = p.parse_args(argv)
    rep = run_star_vs_chain(a.n_sites, a.dim, a.n_per_site, a.n_rep, a.seed, a.shared, a.mean_shift)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(rep), f, indent=2)
    print(f"{'design':<18}{'W2^2 to hub':>14}{'site probe':>12}{'displacement':>14}")
    for k, v in rep["results"].items():
        print(f"{k:<18}{v['w2_to_hub_pop']['mean']:>14.3f}{v['probe_acc']['mean']:>12.3f}{v['displacement']['mean']:>14.3f}")
    print(f"probe chance (majority) = {rep['chance_probe']:.3f}; written {a.out}")


if __name__ == "__main__":
    main()
