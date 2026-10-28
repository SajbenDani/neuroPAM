"""Per-fold (leakage-free) PCA representation inside the protocol."""

import numpy as np

from vae_harmonization.evaluation.protocol import MethodSpec, run_protocol
from vae_harmonization.evaluation.representations import PCARepresentation, make_pca_representation_fn
from vae_harmonization.harmonization import IdentityHarmonizer, make_variant


def test_pca_embed_is_isometric(cohort):
    rng = np.random.default_rng(0)
    X_raw = np.hstack([cohort.Z, 0.1 * rng.normal(size=(len(cohort.Z), 12))])
    rep = PCARepresentation.fit(X_raw, n_components=6, seed=0)
    S = rep.transform(X_raw)
    E = rep.embed(S)
    d_scores = np.linalg.norm(S[:, None] - S[None], axis=-1)
    d_embed = np.linalg.norm(E[:, None] - E[None], axis=-1)
    assert np.allclose(d_scores, d_embed, atol=1e-8)
    assert rep.project(E).shape == S.shape and np.allclose(rep.project(E), S, atol=1e-8)
    assert 0 < rep.summary()["explained_variance_ratio_total"] <= 1


def test_per_fold_pca_protocol_pools_out_of_sample_rows(cohort):
    rng = np.random.default_rng(0)
    X_raw = np.hstack([cohort.Z, 0.1 * rng.normal(size=(len(cohort.Z), 12))])   # 20 "edges"
    store = []
    rep_fn = make_pca_representation_fn(X_raw, n_components=6, seed=0, store=store)
    specs = [
        MethodSpec("Raw", "raw", IdentityHarmonizer, fit_aux=(), is_reference=True),
        MethodSpec("PCA scores", "pca", IdentityHarmonizer, fit_aux=(), is_reference=True),
        MethodSpec("Ours", "pca", lambda: make_variant("ours", cohort.target_site, min_cov_samples=10),
                   is_proposed=True),
    ]
    res = run_protocol(rep_fn, cohort.y_site, cohort.y_diag, specs, n_folds=3, seed=0, classifiers=("lr",))
    assert res["protocol"]["nested_representation"] is True
    assert len(store) == 3                                        # one PCA per fold
    assert res["protocol"]["distribution_scope"] == "pooled_out_of_sample"
    assert set(res["_oos"]) == {"Raw", "PCA scores", "Ours"}
    assert res["_oos"]["Ours"].shape == X_raw.shape                # embedded into the edge space
    assert np.all(np.isfinite(res["_oos"]["Ours"]))
    assert "pca" in res["protocol"]["mmd_gamma_by_representation"]
    ours = res["methods"]["Ours"]
    assert np.isfinite(ours["summary"]["diag"]["lr"]["acc"]["mean"])
    assert np.isfinite(ours["oos_distribution"]["mmd2_unbiased_mean"])
    assert ours["summary"]["site"]["lr"]["acc"]["mean"] < res["methods"]["Raw"]["summary"]["site"]["lr"]["acc"]["mean"]
