from etf_momentum.analysis.sim_gbm import gbm_ab_significance  # pylint: disable=import-error


def test_gbm_ab_significance_structure_and_ranges():
    out = gbm_ab_significance(
        start="19900101",
        end="19911231",
        n_worlds=120,
        n_assets=4,
        vol_low=0.05,
        vol_high=0.30,
        seed=123,
        strategy_a={"lookback_days": 20, "top_k": 1, "trend_filter": True, "trend_sma_window": 5, "trend_ma_type": "ema"},
        strategy_b={"lookback_days": 20, "top_k": 1, "trend_filter": False},
        n_perm=1000,
        n_boot=1000,
    )
    assert out["ok"] is True
    assert int(out["meta"]["n_samples"]) > 0
    p = float(out["stats"]["pvalue_permutation_one_sided"])
    assert 0.0 <= p <= 1.0
    ci = out["stats"]["bootstrap_ci_95"]["mean"]
    assert isinstance(ci, list) and len(ci) == 2
