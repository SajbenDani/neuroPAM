"""
data_registry.py -- Cohorts used in the paper.

Site order in every entry matches ``attrs['site_names']`` of the HDF5 file
(the CLI re-reads the attribute and overrides these lists at run time).
``phenotype`` points at the phenotypic CSV used for age / sex / motion
covariates; ``severity_key`` is the HDF5 dataset holding the symptom score.
"""

from __future__ import annotations

from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent.parent.parent  # src/vae_harmonization -> src -> repo root
_DATA = _REPO_DIR / "data"
_PROCESSED = _DATA / "processed"

DATASET_REGISTRY: dict[str, dict] = {
    # ABIDE-I, all 17 sites, CC200 (200 ROIs -> 19,900 edges)
    "abide1_full": {
        "hdf5_path": _PROCESSED / "abide1_full_fc.h5",
        "n_sites": 17,
        "site_names": [
            "CALTECH", "CMU", "KKI", "LEUVEN", "MAX_MUN",
            "NYU", "OHSU", "OLIN", "PITT", "SBL",
            "SDSU", "STANFORD", "TRINITY", "UCLA", "UM",
            "USM", "YALE",
        ],
        "target_site": 5,  # NYU, the largest site (166 subjects)
        "pca_dim": 100,
        "n_rois": 200,
        "case_label": "ASD",
        "severity_key": "ados_total",
        "phenotype": _DATA / "raw" / "Phenotypic_V1_0b_preprocessed1.csv",
        "phenotype_kind": "abide1",
    },
    # ADHD-200, 7 sites, Athena CC200 (190 ROIs -> 17,955 edges)
    "adhd200": {
        "hdf5_path": _PROCESSED / "adhd200_fc.h5",
        "n_sites": 7,
        "site_names": ["KKI", "NYU", "NeuroIMAGE", "OHSU", "Peking", "Pittsburgh", "WashU"],
        "target_site": 4,  # Peking, the largest site (183 subjects); Pittsburgh and WashU contain controls only
        "pca_dim": 100,
        "n_rois": 190,
        "case_label": "ADHD",
        "severity_key": None,
        "phenotype": _DATA / "ADHD-200",
        "phenotype_kind": "adhd200",
    },
    # ABIDE-II, 10 sites with >= 20 subjects, fmriprep + CC200 (195 ROIs -> 18,915 edges),
    # with split-half FC for fingerprinting.
    "abide2_10site": {
        "hdf5_path": _DATA / "processed_abide2_10site_fp" / "abide2_fc.h5",
        "n_sites": 10,
        "site_names": ["EMC", "ETH", "GU", "IP", "KKI", "KUL_3", "NYU", "OHSU", "TCD", "UCD"],
        "target_site": 4,  # KKI, the largest site (193 subjects)
        "pca_dim": 100,
        "n_rois": 195,
        "case_label": "ASD",
        "severity_key": "ados_g_total",
        "phenotype": _DATA / "ABIDE-II" / "ABIDEII_Composite_Usable20_Phenotypic.csv",
        "phenotype_kind": "abide2",
    },
    # REST-meta-MDD. Site 4 is excluded in the preprocessor (it duplicates site 14).
    # The phenotype workbook is not redistributable. See README.md.
    "rest_meta_mdd": {
        "hdf5_path": _PROCESSED / "rest_meta_mdd_fc.h5",
        "n_sites": 16,
        "site_names": [],  # filled from the HDF5 attribute at run time
        "target_site": 0,  # overridden by the largest site (S20 in the Yan 2019 sample)
        "pca_dim": 100,
        "n_rois": 160,
        "case_label": "MDD",
        "severity_key": "hamd",
        "phenotype": _DATA / "REST-meta-MDD" / "REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx",
        "phenotype_kind": "restmeta",
    },
    # SRPBS Multidisorder Connectivity Dataset (ATR, srpbsfc): MDD vs controls at the six scanners
    # with both groups, BAL atlas (140 ROIs -> 9,730 edges). Age and sex are stored in the HDF5
    # (scripts/preprocessing/preprocess_srpbs.py). Not redistributable, like REST-meta-MDD.
    "srpbs": {
        "hdf5_path": _PROCESSED / "srpbs_fc.h5",
        "n_sites": 6,
        "site_names": [],  # filled from the HDF5 attribute at run time
        "target_site": 0,  # overridden by the largest site (UTO)
        "pca_dim": 100,
        "n_rois": 140,
        "case_label": "MDD",
        "severity_key": None,
        "phenotype": _PROCESSED / "srpbs_fc.h5",
        "phenotype_kind": "hdf5",
    },
}

# backwards-compatible alias used by earlier run tags
DATASET_REGISTRY["abide2_10site_fp"] = DATASET_REGISTRY["abide2_10site"]

# The DecNef Harmonization Dataset (traveling subjects and its clinical SRPBS sample,
# data/processed/srpbs_ts_fc.h5 and srpbs_decnef_fc.h5) is not a protocol cohort. It is read by
# scripts/run_traveling.py.


def get_dataset_config(name: str) -> dict:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {', '.join(sorted(DATASET_REGISTRY))}")
    return dict(DATASET_REGISTRY[name])


def list_datasets() -> list[str]:
    return sorted(k for k in DATASET_REGISTRY if k != "abide2_10site_fp")
