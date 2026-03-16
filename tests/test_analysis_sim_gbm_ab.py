from etf_momentum.analysis.sim_gbm import gbm_ab_significance  # pylint: disable=import-error


def test_gbm_ab_significance_structure_and_ranges():
    out = gbm_ab_significance(
        start="19900101",
        end="19911231",
        n_worlds=2,
        n_assets=4,
        vol_low=0.05,
        vol_high=0.30,
        seed=123,
        strategy_a={"lookback_days": 20, "top_k": 1, "trend_filter": True, "trend_sma_window": 5, "trend_ma_type": "vma"},
        strategy_b={"lookback_days": 20, "top_k": 1, "trend_filter": False},
        n_perm=1000,
        n_boot=1000,
    )
    assert out["ok"] is True
    assert int(out["meta"]["n_samples"]) > 0
    p = float(out["stats"]["pvalue_permutation_one_sided"])
    assert 0.0 <= p <= 1.0
    p_sign = float(out["stats"]["pvalue_sign_test_one_sided"])
    assert 0.0 <= p_sign <= 1.0
    p_wil = float(out["stats"]["pvalue_wilcoxon_one_sided"])
    assert 0.0 <= p_wil <= 1.0
    ci = out["stats"]["bootstrap_ci_95"]["mean"]
    assert isinstance(ci, list) and len(ci) == 2


def test_gbm_ab_significance_rotation_vs_equal_weight_mode():
    out = gbm_ab_significance(
        start="19900101",
        end="19911231",
        n_worlds=2,
        n_assets=4,
        vol_low=0.05,
        vol_high=0.30,
        seed=123,
        strategy_a={"lookback_days": 20, "top_k": 1},
        strategy_b={},
        comparison_mode="rotation_vs_equal_weight",
        n_perm=1000,
        n_boot=1000,
    )
    assert out["ok"] is True
    assert out["comparison"]["mode"] == "rotation_vs_equal_weight"
    assert out["comparison"]["label_a"] == "轮动策略A"
    assert out["comparison"]["label_b"] == "等权再平衡"
    assert int(out["meta"]["n_samples"]) > 0


def test_gbm_ab_significance_equal_weight_vs_cash_mode():
    out = gbm_ab_significance(
        start="19900101",
        end="19911231",
        n_worlds=2,
        n_assets=4,
        vol_low=0.05,
        vol_high=0.30,
        seed=123,
        strategy_a={},
        strategy_b={},
        comparison_mode="equal_weight_vs_cash",
        n_perm=1000,
        n_boot=1000,
    )
    assert out["ok"] is True
    assert out["comparison"]["mode"] == "equal_weight_vs_cash"
    assert out["comparison"]["label_a"] == "等权再平衡"
    assert out["comparison"]["label_b"] == "持有现金"
    assert int(out["meta"]["n_samples"]) > 0


def test_gbm_ab_significance_risk_parity_vs_equal_weight_mode():
    out = gbm_ab_significance(
        start="19900101",
        end="19911231",
        n_worlds=2,
        n_assets=4,
        vol_low=0.05,
        vol_high=0.30,
        seed=123,
        strategy_a={},
        strategy_b={},
        comparison_mode="risk_parity_vs_equal_weight",
        n_perm=1000,
        n_boot=1000,
    )
    assert out["ok"] is True
    assert out["comparison"]["mode"] == "risk_parity_vs_equal_weight"
    assert out["comparison"]["label_a"] == "风险平价再平衡"
    assert out["comparison"]["label_b"] == "等权再平衡"
    assert int(out["meta"]["n_samples"]) > 0


def test_gbm_ab_significance_rotation_vs_risk_parity_mode():
    out = gbm_ab_significance(
        start="19900101",
        end="19911231",
        n_worlds=2,
        n_assets=4,
        vol_low=0.05,
        vol_high=0.30,
        seed=123,
        strategy_a={"lookback_days": 20, "top_k": 1},
        strategy_b={},
        comparison_mode="rotation_vs_risk_parity",
        n_perm=1000,
        n_boot=1000,
    )
    assert out["ok"] is True
    assert out["comparison"]["mode"] == "rotation_vs_risk_parity"
    assert out["comparison"]["label_a"] == "轮动策略A"
    assert out["comparison"]["label_b"] == "风险平价再平衡"
    assert int(out["meta"]["n_samples"]) > 0


def test_gbm_ab_significance_independent_targets_risk_parity_vs_equal_weight():
    out = gbm_ab_significance(
        start="19900101",
        end="19911231",
        n_worlds=2,
        n_assets=4,
        vol_low=0.05,
        vol_high=0.30,
        seed=123,
        strategy_a={"lookback_days": 20, "top_k": 1},
        strategy_b={"lookback_days": 60, "top_k": 1},
        target_a="risk_parity",
        target_b="equal_weight",
        n_perm=1000,
        n_boot=1000,
    )
    assert out["ok"] is True
    assert out["comparison"]["target_a"] == "risk_parity"
    assert out["comparison"]["target_b"] == "equal_weight"
    assert out["comparison"]["label_a"] == "风险平价再平衡"
    assert out["comparison"]["label_b"] == "等权再平衡"
    assert int(out["meta"]["n_samples"]) > 0


def test_gbm_ab_significance_independent_targets_rotation_b_vs_cash():
    out = gbm_ab_significance(
        start="19900101",
        end="19911231",
        n_worlds=2,
        n_assets=4,
        vol_low=0.05,
        vol_high=0.30,
        seed=123,
        strategy_a={"lookback_days": 20, "top_k": 1},
        strategy_b={"lookback_days": 60, "top_k": 1},
        target_a="rotation_b",
        target_b="cash",
        n_perm=1000,
        n_boot=1000,
    )
    assert out["ok"] is True
    assert out["comparison"]["target_a"] == "rotation_b"
    assert out["comparison"]["target_b"] == "cash"
    assert out["comparison"]["label_a"] == "轮动策略B"
    assert out["comparison"]["label_b"] == "持有现金"
    assert int(out["meta"]["n_samples"]) > 0


def test_gbm_ab_significance_seed_stability_enabled():
    out = gbm_ab_significance(
        start="19900101",
        end="19911231",
        n_worlds=2,
        n_assets=4,
        vol_low=0.05,
        vol_high=0.30,
        seed=123,
        strategy_a={"lookback_days": 20, "top_k": 1},
        strategy_b={"lookback_days": 20, "top_k": 1},
        comparison_mode="custom_ab",
        stability_repeats=2,
        stability_worlds=2,
        n_perm=1000,
        n_boot=1000,
    )
    assert out["ok"] is True
    stab = ((out.get("robustness") or {}).get("seed_stability") or {})
    assert stab.get("enabled") is True
    assert int(stab.get("repeats") or 0) == 2
    assert len(stab.get("mean_diff") or []) == 2


def test_gbm_ab_significance_accepts_corr_and_mu_ranges():
    out = gbm_ab_significance(
        start="19900101",
        end="19911231",
        n_worlds=2,
        n_assets=4,
        vol_low=0.05,
        vol_high=0.30,
        corr_low=0.1,
        corr_high=0.3,
        mu_low=-0.02,
        mu_high=0.10,
        seed=123,
        strategy_a={"lookback_days": 20, "top_k": 1},
        strategy_b={"lookback_days": 20, "top_k": 1},
        n_perm=1000,
        n_boot=1000,
    )
    assert out["ok"] is True
    meta = out.get("meta") or {}
    assert meta.get("corr_low") == 0.1
    assert meta.get("corr_high") == 0.3
    assert meta.get("mu_low") == -0.02
    assert meta.get("mu_high") == 0.10
