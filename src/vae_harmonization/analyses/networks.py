"""
Functional-network preservation analysis (Yeo-7) on decoded FC.

ROI -> network assignment
-------------------------
Each CC200 ROI is assigned to the Yeo 2011 7-network label that covers the
majority of its voxels (Yeo liberal cortical mask, resampled with nearest-
neighbour interpolation onto the parcellation grid).  ROIs whose voxels
overlap the cortical mask in less than ``min_overlap`` of their volume
(subcortical, cerebellar, white-matter clusters) receive label -1 and are
excluded from the 7x7 summaries.  Assignments are pre-computed with
:func:`build_yeo7_labels` and stored as CSV files under ``data/atlases`` so the
analysis itself does not need nilearn/nibabel:

    data/atlases/yeo7_labels_abide1_full.csv     (PCP cc200_roi_atlas, 200 ROIs)
    data/atlases/yeo7_labels_adhd200.csv         (Athena ADHD200_parcellate_200, 190 ROIs)
    data/atlases/yeo7_labels_abide2_10site.csv   (nilearn Craddock-2012 volume, 195 ROIs)

Regenerate with ``scripts/build_network_labels.py``.

Metrics
-------
For every harmonization condition we compute the group-mean 7x7 network
matrix (mean Fisher-z FC over all ROI pairs in the network pair), compare it
with the raw-FC matrix (Pearson r over the 28 upper-triangular entries, RMSE,
within- vs between-network contrast), compute per-subject network-matrix
correlations with the subject's own raw matrix, and the site effect (mean
eta^2 across the 28 network-level edges) before and after harmonization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..metrics import compute_eta_squared

YEO7_NAMES = [
    "Visual",
    "Somatomotor",
    "DorsalAttention",
    "VentralAttention",
    "Limbic",
    "Frontoparietal",
    "DefaultMode",
]
N_NETWORKS = 7

_REPO_DIR = Path(__file__).resolve().parents[3]
ATLAS_DIR = _REPO_DIR / "data" / "atlases"


# ---------------------------------------------------------------------------
# ROI -> Yeo-7 assignment (requires nibabel + nilearn; run offline once)
# ---------------------------------------------------------------------------

def roi_ids_from_1d_header(path: str | Path, prefix: str = "Mean_") -> np.ndarray:
    """ROI label ids from an AFNI ``3dROIstats``-style header (``Mean_<id>`` columns)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split()
    ids = [int(tok[len(prefix):]) for tok in header if tok.startswith(prefix)]
    if not ids:
        ids = [int(tok.lstrip("#")) for tok in header if tok.startswith("#")]
    return np.asarray(ids, dtype=int)


def build_yeo7_labels(parcellation, roi_ids: np.ndarray | None = None, yeo_img=None,
                      min_overlap: float = 0.3):
    """
    Assign every ROI of a label image to its majority Yeo-7 network.

    Parameters
    ----------
    parcellation : path or nibabel image with integer ROI labels (3-D)
    roi_ids      : ROI ids in the column order of the FC features; default = all
                   non-zero labels sorted ascending (NiftiLabelsMasker order)
    yeo_img      : path/image of the Yeo 7-network atlas; default = nilearn
                   ``fetch_atlas_yeo_2011()['maps']`` (thick_7, liberal mask)
    min_overlap  : minimum fraction of ROI voxels inside the cortical mask for
                   a network assignment (else label -1 = "non-cortical")

    Returns
    -------
    pandas.DataFrame with columns roi_index, roi_id, x, y, z, n_voxels,
    cortical_fraction, yeo7, yeo7_name, majority_fraction
    """
    import nibabel as nib
    import pandas as pd
    from nilearn import datasets
    from nilearn.image import resample_to_img

    parc = nib.load(str(parcellation)) if isinstance(parcellation, (str, Path)) else parcellation
    parc_data = np.round(np.asarray(parc.get_fdata())).astype(int)
    if parc_data.ndim == 4:
        parc_data = parc_data[..., 0]
    if yeo_img is None:
        yeo_img = datasets.fetch_atlas_yeo_2011()["maps"]
    yeo = resample_to_img(nib.load(str(yeo_img)) if isinstance(yeo_img, (str, Path)) else yeo_img,
                          parc, interpolation="nearest", force_resample=True, copy_header=True)
    yeo_data = np.round(np.asarray(yeo.get_fdata())).astype(int)
    if yeo_data.ndim == 4:
        yeo_data = yeo_data[..., 0]

    if roi_ids is None:
        roi_ids = np.unique(parc_data[parc_data > 0])
    rows = []
    for idx, rid in enumerate(np.asarray(roi_ids, dtype=int)):
        mask = parc_data == rid
        n_vox = int(mask.sum())
        if n_vox == 0:
            rows.append(dict(roi_index=idx, roi_id=int(rid), x=np.nan, y=np.nan, z=np.nan, n_voxels=0,
                             cortical_fraction=0.0, yeo7=-1, yeo7_name="NonCortical", majority_fraction=0.0))
            continue
        ijk = np.argwhere(mask)
        xyz = nib.affines.apply_affine(parc.affine, ijk).mean(axis=0)
        labels = yeo_data[mask]
        cortical = labels > 0
        frac_cortical = float(cortical.mean())
        if frac_cortical < min_overlap:
            net, maj = -1, 0.0
        else:
            counts = np.bincount(labels[cortical], minlength=N_NETWORKS + 1)[1:]
            net = int(np.argmax(counts))
            maj = float(counts[net] / cortical.sum())
        rows.append(dict(roi_index=idx, roi_id=int(rid), x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
                         n_voxels=n_vox, cortical_fraction=frac_cortical, yeo7=net,
                         yeo7_name=YEO7_NAMES[net] if net >= 0 else "NonCortical", majority_fraction=maj))
    return pd.DataFrame(rows)


def label_table_path(cohort: str) -> Path:
    return ATLAS_DIR / f"yeo7_labels_{cohort}.csv"


def load_network_labels(cohort: str, n_rois: int | None = None) -> np.ndarray:
    """Load the pre-computed (n_rois,) Yeo-7 label vector for a cohort (-1 = non-cortical)."""
    import pandas as pd

    path = label_table_path(cohort)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/build_network_labels.py")
    df = pd.read_csv(path).sort_values("roi_index")
    labels = df["yeo7"].to_numpy(dtype=int)
    if n_rois is not None and labels.size != n_rois:
        raise ValueError(f"label table for {cohort} has {labels.size} ROIs, data has {n_rois}")
    return labels


# ---------------------------------------------------------------------------
# Network-level matrices
# ---------------------------------------------------------------------------

def _n_rois_from_features(n_features: int) -> int:
    n = int(round((1.0 + np.sqrt(1.0 + 8.0 * n_features)) / 2.0))
    if n * (n - 1) // 2 != n_features:
        raise ValueError(f"{n_features} is not an upper-triangle length")
    return n


def edge_network_pairs(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each upper-triangle edge return (network_a, network_b, valid) with a <= b."""
    labels = np.asarray(labels, dtype=int)
    r, c = np.triu_indices(labels.size, k=1)
    a, b = labels[r], labels[c]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    valid = (a >= 0) & (b >= 0)
    return lo, hi, valid


def network_edge_means(fc_vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Per-subject network matrices: (N, 7, 7) mean FC over ROI pairs in each network pair.
    """
    fc = np.asarray(fc_vectors, dtype=float)
    if fc.ndim == 1:
        fc = fc[None, :]
    n_rois = _n_rois_from_features(fc.shape[1])
    if labels.size != n_rois:
        raise ValueError(f"labels has {labels.size} entries, FC implies {n_rois} ROIs")
    lo, hi, valid = edge_network_pairs(labels)
    out = np.full((fc.shape[0], N_NETWORKS, N_NETWORKS), np.nan)
    for i in range(N_NETWORKS):
        for j in range(i, N_NETWORKS):
            m = valid & (lo == i) & (hi == j)
            if not m.any():
                continue
            v = fc[:, m].mean(axis=1)
            out[:, i, j] = v
            out[:, j, i] = v
    return out


def compute_network_matrix(fc_vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Group-mean 7x7 network matrix (mean over subjects of the per-subject matrices)."""
    return np.nanmean(network_edge_means(fc_vectors, labels), axis=0)


def _upper(mat: np.ndarray, k: int = 0) -> np.ndarray:
    r, c = np.triu_indices(mat.shape[0], k=k)
    return mat[r, c]


def compare_network_matrices(mat_ref: np.ndarray, mat_other: np.ndarray) -> dict:
    """Pearson r / RMSE over the 28 upper-triangular entries + within/between contrast."""
    a, b = _upper(mat_ref), _upper(mat_other)
    ok = np.isfinite(a) & np.isfinite(b)
    r = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() >= 3 else float("nan")
    rmse = float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.any() else float("nan")
    within_ref, within_other = np.diag(mat_ref), np.diag(mat_other)
    between_ref, between_other = _upper(mat_ref, 1), _upper(mat_other, 1)
    return {
        "pearson_r": r,
        "rmse": rmse,
        "within_net_ref": [float(v) for v in within_ref],
        "within_net_other": [float(v) for v in within_other],
        "between_net_ref": float(np.nanmean(between_ref)),
        "between_net_other": float(np.nanmean(between_other)),
        "within_minus_between_ref": float(np.nanmean(within_ref) - np.nanmean(between_ref)),
        "within_minus_between_other": float(np.nanmean(within_other) - np.nanmean(between_other)),
        "within_vs_between_preserved": bool(np.nanmean(within_other) > np.nanmean(between_other)),
    }


def per_subject_network_correlation(sub_ref: np.ndarray, sub_other: np.ndarray) -> dict:
    """Mean/SD over subjects of corr(subject's ref 7x7, subject's other 7x7)."""
    r, c = np.triu_indices(N_NETWORKS, k=0)
    A = sub_ref[:, r, c]
    B = sub_other[:, r, c]
    ok = np.isfinite(A).all(axis=0) & np.isfinite(B).all(axis=0)
    A, B = A[:, ok], B[:, ok]
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A ** 2).sum(axis=1) * (B ** 2).sum(axis=1)) + 1e-12
    rs = num / den
    return {"mean": float(rs.mean()), "std": float(rs.std(ddof=1)) if rs.size > 1 else 0.0,
            "min": float(rs.min()), "n": int(rs.size)}


def network_site_eta2(sub_mats: np.ndarray, y_site: np.ndarray) -> dict:
    """Site eta^2 for each of the 28 network-level edges; returns mean/max and the 7x7 map."""
    r, c = np.triu_indices(N_NETWORKS, k=0)
    eta = np.full((N_NETWORKS, N_NETWORKS), np.nan)
    vals = []
    for i, j in zip(r, c):
        x = sub_mats[:, i, j]
        if not np.isfinite(x).all():
            continue
        e = compute_eta_squared(x, y_site)
        eta[i, j] = eta[j, i] = e
        vals.append(e)
    return {"mean": float(np.mean(vals)) if vals else float("nan"),
            "max": float(np.max(vals)) if vals else float("nan"),
            "matrix": eta.tolist()}


def run_network_analysis(fc_by_condition: dict[str, np.ndarray], labels: np.ndarray, y_site: np.ndarray,
                         reference: str = "Raw FC") -> dict:
    """
    Network-preservation summary for several FC conditions.

    Parameters
    ----------
    fc_by_condition : {condition name: (N, n_features) Fisher-z FC vectors}; must
                      contain ``reference`` (the un-harmonized FC).
    labels          : (n_rois,) Yeo-7 labels (-1 = excluded)
    y_site          : (N,) site labels
    """
    if reference not in fc_by_condition:
        raise KeyError(f"reference condition '{reference}' missing")
    labels = np.asarray(labels, dtype=int)
    sub = {k: network_edge_means(v, labels) for k, v in fc_by_condition.items()}
    group = {k: np.nanmean(v, axis=0) for k, v in sub.items()}
    ref_sub, ref_group = sub[reference], group[reference]
    out = {
        "network_names": YEO7_NAMES,
        "n_rois": int(labels.size),
        "n_rois_cortical": int((labels >= 0).sum()),
        "rois_per_network": {YEO7_NAMES[i]: int((labels == i).sum()) for i in range(N_NETWORKS)},
        "reference": reference,
        "conditions": {},
    }
    for k in fc_by_condition:
        entry = {"group_matrix": np.where(np.isfinite(group[k]), group[k], None).tolist(),
                 "site_eta2": network_site_eta2(sub[k], y_site)}
        if k != reference:
            entry["vs_reference"] = compare_network_matrices(ref_group, group[k])
            entry["per_subject_r"] = per_subject_network_correlation(ref_sub, sub[k])
        out["conditions"][k] = entry
    return out
