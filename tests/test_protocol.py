"""End-to-end test of the leakage-free protocol on the synthetic cohort."""

import json

import numpy as np

from vae_harmonization.baselines import ComBatHarmonizer
from vae_harmonization.evaluation import MethodSpec, recompute_comparisons, run_protocol, summary_table, to_jsonable
from vae_harmonization.harmonization import IdentityHarmonizer, make_variant


def _methods():
    return [
        MethodSpec("Raw", "latent", IdentityHarmonizer, fit_aux=(), is_reference=True),
        MethodSpec("ComBat", "latent", lambda: ComBatHarmonizer(protect_diagnosis=True),
                   transform_uses_labels=True),
        MethodSpec("Ours", "latent", lambda: make_variant("ours", target_site=0, name="Ours"),
                   is_proposed=True),
        MethodSpec("Mean-shift", "latent", lambda: make_variant("mean-shift", target_site=0, name="Mean-shift")),
    ]


def test_protocol_runs_and_ranks_methods(cohort, tmp_path):
    res = run_protocol({"latent": cohort.Z}, cohort.y_site, cohort.y_diag, _methods(),
                       n_folds=4, seed=0, classifiers=("lr",), mmd_min_site=5)
    m = res["methods"]
    assert set(m) == {"Raw", "ComBat", "Ours", "Mean-shift"}
    for name in m:
        assert len(m[name]["folds"]["diag"]["lr"]["acc"]) == 4
        assert not m[name]["failures"]
    # site information is removed: Ours site accuracy far below Raw and near the majority rate
    raw_site = m["Raw"]["summary"]["site"]["lr"]["acc"]["mean"]
    ours_site = m["Ours"]["summary"]["site"]["lr"]["acc"]["mean"]
    assert raw_site > 0.9 and ours_site < 0.5
    # planted diagnosis signal survives
    assert m["Ours"]["summary"]["diag"]["lr"]["acc"]["mean"] > 0.75
    # OOS distribution metrics decrease
    assert m["Ours"]["oos_distribution"]["mmd2_unbiased_mean"] < m["Raw"]["oos_distribution"]["mmd2_unbiased_mean"]
    assert m["Ours"]["oos_distribution"]["icc_mean"] < m["Raw"]["oos_distribution"]["icc_mean"]
    # protected-axis eta^2 is recorded for our method only
    assert np.isfinite(m["Ours"]["protected_eta2"]["mean"])
    assert not np.isfinite(m["Raw"]["protected_eta2"]["mean"])
    # comparisons carry Holm-adjusted p-values and TOST vs reference
    comp = res["comparisons"]["diag/lr/acc"]["Ours"]["vs_reference"]
    assert comp["other"] == "Raw" and "p_value_holm" in comp and "tost" in comp
    assert "vs_proposed" in res["comparisons"]["site/lr/acc"]["ComBat"]
    # OOS matrices fully populated
    assert np.all(np.isfinite(res["_oos"]["Ours"]))
    # JSON round-trip drops private keys and NaN -> null
    js = to_jsonable(res)
    assert "_oos" not in js
    text = json.dumps(js)
    assert "NaN" not in text
    (tmp_path / "r.json").write_text(text)
    assert "Ours" in summary_table(res)
    # Holm families: 'vs_reference' tests are corrected together (3 non-reference methods), separately from 'vs_proposed'
    assert comp["holm_family_size"] == 3
    assert res["comparisons"]["site/lr/acc"]["ComBat"]["vs_proposed"]["holm_family_size"] == 3
    # comparisons can be recomputed offline from the JSON and agree with the in-run values
    re = recompute_comparisons(json.loads(text))
    assert re["comparisons"]["diag/lr/acc"]["Ours"]["vs_reference"]["p_value"] == comp["p_value"]
    assert re["comparisons"]["diag/lr/acc"]["Ours"]["vs_reference"]["p_value_holm"] == comp["p_value_holm"]
