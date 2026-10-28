#!/usr/bin/env python3
"""
Generate the figures of the paper from the result files.

    python scripts/generate_paper_figures.py                  # results/ -> paper/figures/
    python scripts/generate_paper_figures.py --cohort abide1_full

Figures
    tsne_site_pre.png        t-SNE of the un-harmonized PCA scores, coloured by site
    tsne_site_post.png       t-SNE of the held-out harmonized scores (PAM), same colours
    fig_tradeoff.pdf         held-out site accuracy vs diagnosis accuracy for every method
    fig_network.pdf          site eta^2 of every Yeo-7 network pair, before and after harmonization

Inputs: results/<cohort>/oos_pca.npz, results/<cohort>/network_preservation.json and the protocol
results of the cohort, merged across runs on the same folds exactly as in scripts/generate_tables.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_tables import load_cohort

REPO = Path(__file__).resolve().parents[1]

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#999933", "#000000",
           "#882255", "#44AA99", "#DDCC77", "#AA4499", "#117733", "#332288", "#88CCEE", "#CC6677", "#661100"]


def _tsne(Z, seed=0):
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(Z)


def fig_tsne(res_dir: Path, out_dir: Path, site_names: dict[int, str]):
    path = res_dir / "oos_pca.npz"
    if not path.exists():
        print(f"{path} not found; t-SNE figures skipped")
        return
    oos = np.load(path)
    y_site = oos["y_site"]
    for name, key in (("tsne_site_pre.png", "PCA_scores"), ("tsne_site_post.png", "Ours")):
        E = _tsne(oos[key])
        fig, ax = plt.subplots(figsize=(4.2, 5.3 if name.endswith("pre.png") else 4.2))
        for i, s in enumerate(np.unique(y_site)):
            m = y_site == s
            ax.scatter(E[m, 0], E[m, 1], s=7, color=PALETTE[i % len(PALETTE)], alpha=0.75,
                       label=f"{site_names.get(int(s), s)} ({m.sum()})", linewidths=0)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if name.endswith("pre.png"):
            ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.01),
                      markerscale=1.6, handletextpad=0.2, columnspacing=0.8)
        fig.tight_layout(pad=0.2)
        fig.savefig(out_dir / name, dpi=220)
        plt.close(fig)
        print("wrote", out_dir / name)


SHORT = {
    "Raw": "Raw FC", "PCA scores": "PCA scores", "ComBat": "ComBat", "CovBat": "CovBat",
    "ComBat (PCA)": "ComBat, PCA", "CovBat (PCA)": "CovBat, PCA", "ComBat-GAM": "ComBat-GAM", "RELIEF": "RELIEF",
    "ComBat (age+sex)": "ComBat, age+sex", "ComBat (PCA, age+sex)": "ComBat, PCA, age+sex",
    "DeepComBat": "DeepComBat", "CORAL": "CORAL", "MMD affine": "MMD affine", "Ours": "PAM",
    "Ours (edge residual)": "PAM, edge residual", "Log-Euclidean FC": "log-Euclidean FC",
    "Riemannian re-centering": "Riem. re-centering", "Euclidean alignment": "Eucl. alignment",
    "Riemannian Procrustes": "RPA (steps 1-2)", "RLET": "RLET",
    "Ours (empirical covariance)": "PAM, sample covariance",
}
MARKER = {"raw": "s", "spd": "o", "pca": "D"}


def _place_labels(ax, pts, labels, fontsize=8, avoid=(), big=(), regions=()):
    """Greedy placement: try offsets on growing rings until the label box hits no other label or marker.

    Markers are treated as small boxes (larger for the points in ``big``), not as their centres only.
    ``regions`` are data-space rectangles (x0, y0, x1, y1) that labels must not cover."""
    from matplotlib.transforms import Bbox
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    taken, placed = [], []
    for x0, y0, x1, y1 in regions:
        (a, b), (c, d) = ax.transData.transform([(x0, y0), (x1, y1)])
        taken.append(Bbox.from_extents(min(a, c), min(b, d), max(a, c), max(b, d)))
    px_per_pt = fig.dpi / 72.0
    markers = []
    for q in list(pts) + list(avoid):
        cx, cy = ax.transData.transform(q)
        h = (8.0 if tuple(q) in {tuple(b) for b in big} else 4.0) * px_per_pt
        markers.append(Bbox.from_extents(cx - h, cy - h, cx + h, cy + h))
    for (x, y), text, bold in labels:
        best = None
        for r in (8, 13, 19, 26, 34, 44, 56):
            for ang in (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):
                dx, dy = r * np.cos(np.radians(ang)), r * np.sin(np.radians(ang))
                t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                                ha="left" if dx >= 0 else "right", va="center",
                                fontweight="bold" if bold else None)
                bb = t.get_window_extent(renderer).expanded(1.04, 1.12)
                inside = ax.get_window_extent(renderer).contains(bb.x0, bb.y0) and \
                    ax.get_window_extent(renderer).contains(bb.x1, bb.y1)
                own = ax.transData.transform((x, y))
                clash = any(bb.overlaps(o) for o in taken) or any(
                    bb.overlaps(m) for m in markers if not m.contains(*own))
                if inside and not clash:
                    best = (t, bb, dx, dy)
                    break
                t.remove()
            if best:
                break
        if best is None:          # fall back to the first candidate
            t = ax.annotate(text, (x, y), xytext=(9, 0), textcoords="offset points", fontsize=fontsize, va="center")
            best = (t, t.get_window_extent(renderer), 9, 0)
        t, bb, dx, dy = best
        if np.hypot(dx, dy) > 12:
            ax.annotate("", (x, y), xytext=(dx, dy), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="#777777", shrinkA=0, shrinkB=2))
        taken.append(bb)
        placed.append(t)
    return placed


def fig_tradeoff(res: dict, out_dir: Path):
    ch = res["chance"]
    site_maj, diag_maj = 100 * ch["site"]["majority_rate"], 100 * ch["diag"]["majority_rate"]
    pts = []
    for name, e in res["methods"].items():
        sx, dy = e["summary"]["site"]["lr"]["acc"], e["summary"]["diag"]["lr"]["acc"]
        if sx["mean"] is None or dy["mean"] is None:
            continue
        pts.append((name, e, 100 * sx["mean"], 100 * dy["mean"],
                    100 * (sx["std"] or 0) / np.sqrt(sx["n"]), 100 * (dy["std"] or 0) / np.sqrt(dy["n"])))
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    near = [q for q in pts if abs(q[2] - site_maj) < 3.0]
    zx = (site_maj - 2.2, site_maj + 2.2)
    zy = (min(q[3] for q in near) - 1.2, max(q[3] for q in near) + 1.2) if near else (diag_maj, diag_maj + 10)
    for a, zoom in ((ax, False), (axz, True)):
        for name, e, x, y, ex, ey in sorted(pts, key=lambda q: q[1].get("is_proposed", False)):
            proposed = bool(e.get("is_proposed"))
            labelled = name in SHORT
            color = "#D55E00" if proposed else ("#888888" if e.get("is_reference") else
                                               ("#0072B2" if labelled else "#9DB9D3"))
            a.errorbar(x, y, xerr=ex, yerr=ey, fmt="*" if proposed else MARKER.get(e["representation"], "o"),
                       color=color, ms=12 if proposed else (5 if labelled else 4), elinewidth=0.6, capsize=0,
                       alpha=0.95, zorder=4 if proposed else 3)
        a.axvline(site_maj, ls="--", lw=0.7, color="k", zorder=1)
        a.axhline(diag_maj, ls="--", lw=0.7, color="k", zorder=1)
        a.tick_params(labelsize=8)
        if zoom:
            a.set_xlim(*zx)
            a.set_ylim(*zy)
            a.set_title("zoom on the site majority line", fontsize=8.5)
        else:
            a.add_patch(plt.Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False, lw=0.6,
                                      ls=":", color="#555555", zorder=1))
            a.set_ylabel("held-out diagnosis accuracy, LR (%)", fontsize=8.5)
        a.set_xlabel("held-out site accuracy, LR (%)", fontsize=8.5)
    star = [(q[2], q[3]) for q in pts if q[1].get("is_proposed")]
    far = [(q[2], q[3]) for q in pts if not (zx[0] <= q[2] <= zx[1] and zy[0] <= q[3] <= zy[1])]
    inside = [q for q in pts if zx[0] <= q[2] <= zx[1] and zy[0] <= q[3] <= zy[1]]
    _place_labels(ax, far, [((q[2], q[3]), SHORT[q[0]], bool(q[1].get("is_proposed"))) for q in pts
                            if q[0] in SHORT and (q[2], q[3]) in far], fontsize=7.5, big=star,
                  avoid=[(q[2], q[3]) for q in inside], regions=[(zx[0], zy[0], zx[1], zy[1])])
    # the proposed method first, so the other labels make room around the star
    lab_in = sorted([((q[2], q[3]), SHORT[q[0]], bool(q[1].get("is_proposed"))) for q in inside if q[0] in SHORT],
                    key=lambda t: not t[2])
    _place_labels(axz, [(q[2], q[3]) for q in inside], lab_in, fontsize=7.5, big=star)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(out_dir / "fig_tradeoff.pdf")
    plt.close(fig)
    print("wrote", out_dir / "fig_tradeoff.pdf")


NET_ABBR = {"Visual": "VIS", "Somatomotor": "SMN", "DorsalAttention": "DAN", "VentralAttention": "VAN",
            "Limbic": "LIM", "Frontoparietal": "FPN", "DefaultMode": "DMN"}
# one-hue sequential ramp; the lightest step (no site effect) stays just visible against the page
SEQ_BLUE = ["#eef3fa", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
NET_PANELS = [("Raw FC", "Raw edges"), ("PCA reconstruction", "PCA scores"), ("Decoded Ours", "PAM"),
              ("ComBat", "ComBat")]


def fig_network(res_dir: Path, out_dir: Path):
    path = res_dir / "network_preservation.json"
    if not path.exists():
        print(f"{path} not found; network figure skipped")
        return
    from matplotlib.colors import LinearSegmentedColormap
    net = json.loads(path.read_text())
    cond = net["conditions"]
    panels = [(k, t) for k, t in NET_PANELS if k in cond]
    mats = [np.asarray(cond[k]["site_eta2"]["matrix"], dtype=float) for k, _ in panels]
    vmax = max(np.nanmax(m) for m in mats)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    names = [NET_ABBR.get(n, n) for n in net["network_names"]]
    n = len(names)
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.5))
    for i, (ax, (key, title), m) in enumerate(zip(axes, panels, mats)):
        # the 28 network pairs: lower triangle and diagonal of the symmetric matrix
        im = ax.pcolormesh(np.ma.masked_where(np.triu(np.ones((n, n), bool), k=1), m), cmap=cmap,
                           vmin=0.0, vmax=vmax, edgecolors="#ffffff", linewidth=1.2)
        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_xticklabels(names, rotation=90, fontsize=7, color="#52514e")
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_yticklabels(names if i == 0 else [], fontsize=7, color="#52514e")
        ax.tick_params(length=0, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        mean = cond[key]["site_eta2"]["mean"]
        ax.set_title(f"{title}\nmean $\\eta^2$ {'< 0.001' if mean < 0.001 else f'= {mean:.3f}'}", fontsize=8)
    cb = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.015)
    cb.set_label("site $\\eta^2$", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_visible(False)
    fig.savefig(out_dir / "fig_network.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out_dir / "fig_network.pdf")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default=str(REPO / "results"))
    p.add_argument("--cohort", default="abide1_full")
    p.add_argument("--out-dir", default=str(REPO / "paper" / "figures"))
    a = p.parse_args(argv)
    R, out_dir = Path(a.results_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res, _ = load_cohort(R, a.cohort)
    if res is None:
        raise SystemExit(f"results/{a.cohort}/protocol_results.json not found")
    sn = res.get("site_names") or {}
    site_names = {int(k): v for k, v in sn.items()} if isinstance(sn, dict) else dict(enumerate(sn))
    fig_tsne(R / a.cohort, out_dir, site_names)
    fig_tradeoff(res, out_dir)
    fig_network(R / a.cohort, out_dir)


if __name__ == "__main__":
    main()
