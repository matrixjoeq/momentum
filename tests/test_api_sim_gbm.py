import pytest  # pylint: disable=import-error

from tests.helpers.rotation_case_data import post_json_ok

# Minimal ranges: phase1/2 only need simulate_gbm (>=10 sessions); MC paths need >=100 business days.
_SIM_START = "19900101"
_SIM_END_QUICK = "19900228"
_SIM_END_MC = "19900518"  # 100 business days from 19900101 (MC path minimum)
# API/schema floors
_MIN_ASSETS = 2
_MIN_SIMS = 8
_MIN_LB = 2
_MIN_PERM = 200
_MIN_BOOT = 200


@pytest.mark.parametrize("path", ["/api/analysis/sim/gbm/phase1", "/api/analysis/sim/gbm/phase2"])
def test_sim_gbm_phase1_and_phase2_ok(api_client, path):
    c = api_client
    data = post_json_ok(
        c,
        path,
        {
            "start": _SIM_START,
            "end": _SIM_END_QUICK,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 123,
            "lookback_days": _MIN_LB,
        },
    )
    assert data["ok"] is True


def test_sim_gbm_phase3_ok(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/phase3",
        {
            "start": _SIM_START,
            "end": _SIM_END_MC,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 7,
            "lookback_days": _MIN_LB,
            "n_sims": _MIN_SIMS,
            "chunk_size": 1,
            "n_jobs": 1,
        },
    )
    assert data["ok"] is True
    assert "dist" in data
    assert len(data["dist"]["rotation"]["cagr"]) == _MIN_SIMS
    assert len(data["dist"]["equal_weight"]["cagr"]) == _MIN_SIMS


def test_sim_gbm_phase4_ok(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/phase4",
        {
            "start": _SIM_START,
            "end": _SIM_END_MC,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 7,
            "lookback_days": _MIN_LB,
            "n_sims": _MIN_SIMS,
            "chunk_size": 1,
            "n_jobs": 1,
            "initial_cash": 1.0,
            "position_pct": 0.01,
        },
    )
    assert data["ok"] is True
    assert "sizing" in data
    assert "one" in data


def test_sim_gbm_phase1_supports_corr_and_mu_ranges(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/phase1",
        {
            "start": _SIM_START,
            "end": _SIM_END_QUICK,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "corr_low": 0.2,
            "corr_high": 0.4,
            "mu_low": -0.02,
            "mu_high": 0.12,
            "seed": 123,
        },
    )
    assert data["ok"] is True
    assert data["meta"]["corr_low"] == pytest.approx(0.2)
    assert data["meta"]["corr_high"] == pytest.approx(0.4)
    assert data["meta"]["mu_low"] == pytest.approx(-0.02)
    assert data["meta"]["mu_high"] == pytest.approx(0.12)
    assert isinstance((data.get("assets") or {}).get("ann_mus"), dict)


def test_sim_gbm_phase1_supports_negative_correlation_range(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/phase1",
        {
            "start": _SIM_START,
            "end": _SIM_END_QUICK,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "corr_low": -0.3,
            "corr_high": -0.1,
            "seed": 123,
        },
    )
    assert data["ok"] is True
    assert data["meta"]["corr_low"] == pytest.approx(-0.3)
    assert data["meta"]["corr_high"] == pytest.approx(-0.1)


def test_sim_gbm_phase1_pairwise_corr_not_single_value(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/phase1",
        {
            "start": _SIM_START,
            "end": "19900731",
            "n_assets": 6,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "corr_low": 0.2,
            "corr_high": 0.8,
            "seed": 123,
        },
    )
    assert data["ok"] is True
    mat = ((data.get("corr") or {}).get("matrix") or [])
    off = [float(mat[i][j]) for i in range(len(mat)) for j in range(i + 1, len(mat))]
    assert len(off) > 0
    # Pairwise correlations should not collapse to one same number.
    assert (max(off) - min(off)) > 0.05


def test_sim_gbm_phase3_uses_strategy_a_payload(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/phase3",
        {
            "start": _SIM_START,
            "end": _SIM_END_MC,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 7,
            "lookback_days": _MIN_LB,
            "n_sims": _MIN_SIMS,
            "chunk_size": 1,
            "n_jobs": 1,
            "strategy_a": {"lookback_days": 2, "top_k": 1, "trend_filter": True, "trend_sma_window": 2, "trend_ma_type": "vma"},
        },
    )
    assert data["ok"] is True
    assert bool((data.get("meta") or {}).get("strategy_a_applied")) is True
    assert len(data["dist"]["rotation"]["cagr"]) == _MIN_SIMS


def test_sim_gbm_phase4_uses_strategy_a_payload(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/phase4",
        {
            "start": _SIM_START,
            "end": _SIM_END_MC,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 7,
            "lookback_days": _MIN_LB,
            "n_sims": _MIN_SIMS,
            "chunk_size": 1,
            "n_jobs": 1,
            "initial_cash": 1.0,
            "position_pct": 0.01,
            "strategy_a": {"lookback_days": 2, "top_k": 1, "trend_filter": True, "trend_sma_window": 2, "trend_ma_type": "vma"},
        },
    )
    assert data["ok"] is True
    assert bool((((data.get("mc") or {}).get("meta") or {}).get("strategy_a_applied"))) is True


def test_sim_gbm_phase2_uses_strategy_a_payload(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/phase2",
        {
            "start": _SIM_START,
            "end": _SIM_END_QUICK,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 123,
            "lookback_days": _MIN_LB,
            "strategy_a": {"lookback_days": 2, "top_k": 1, "trend_filter": True, "trend_sma_window": 2, "trend_ma_type": "vma"},
        },
    )
    assert data["ok"] is True
    assert "rotation" in data and "equal_weight" in data


def test_sim_gbm_phase2_holding_strategy_supports_rebalance_and_cost(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/phase2",
        {
            "start": _SIM_START,
            "end": _SIM_END_QUICK,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 123,
            "lookback_days": _MIN_LB,
            "holding_strategy": {"rebalance": "monthly", "cost_bps": 7.0, "rp_vol_window": 2},
            "strategy_a": {"lookback_days": 2, "top_k": 1},
        },
    )
    assert data["ok"] is True
    ew_h = ((data.get("equal_weight") or {}).get("holding") or {})
    rp_h = ((data.get("risk_parity") or {}).get("holding") or {})
    assert ew_h.get("rebalance") == "monthly"
    assert rp_h.get("rebalance") == "monthly"
    assert ew_h.get("cost_bps") == pytest.approx(7.0)
    assert rp_h.get("cost_bps") == pytest.approx(7.0)


def test_sim_gbm_phase2_reuses_phase1_payload(api_client):
    c = api_client
    p1 = post_json_ok(
        c,
        "/api/analysis/sim/gbm/phase1",
        {
            "start": _SIM_START,
            "end": _SIM_END_QUICK,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 123,
        },
    )
    p2 = post_json_ok(
        c,
        "/api/analysis/sim/gbm/phase2",
        {
            "start": _SIM_START,
            "end": _SIM_END_QUICK,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 999,
            "lookback_days": _MIN_LB,
            "phase1_base": p1,
        },
    )
    assert p2["ok"] is True
    assert bool((p2.get("meta") or {}).get("phase1_reused")) is True
    assert (p2.get("corr") or {}).get("matrix") == (p1.get("corr") or {}).get("matrix")


def test_sim_gbm_ab_significance_ok(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/ab-significance",
        {
            "start": _SIM_START,
            "end": _SIM_END_MC,
            "n_worlds": 2,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 11,
            "n_perm": _MIN_PERM,
            "n_boot": _MIN_BOOT,
            "n_jobs": 1,
            "strategy_a": {"lookback_days": 2, "top_k": 1, "trend_filter": True, "trend_sma_window": 2, "trend_ma_type": "vma"},
            "strategy_b": {"lookback_days": 2, "top_k": 1, "trend_filter": False},
        },
    )
    assert data["ok"] is True
    assert "stats" in data
    assert "dist" in data
    p = float(data["stats"]["pvalue_permutation_one_sided"])
    assert 0.0 <= p <= 1.0
    p_sign = float(data["stats"]["pvalue_sign_test_one_sided"])
    assert 0.0 <= p_sign <= 1.0
    p_wil = float(data["stats"]["pvalue_wilcoxon_one_sided"])
    assert 0.0 <= p_wil <= 1.0
    ci = data["stats"]["bootstrap_ci_95"]["mean"]
    assert isinstance(ci, list) and len(ci) == 2


def test_sim_gbm_ab_significance_holding_strategy_params(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/ab-significance",
        {
            "start": _SIM_START,
            "end": _SIM_END_MC,
            "n_worlds": 2,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 11,
            "n_perm": _MIN_PERM,
            "n_boot": _MIN_BOOT,
            "n_jobs": 1,
            "target_a": "equal_weight",
            "target_b": "risk_parity",
            "holding_strategy_a": {"rebalance": "monthly", "cost_bps": 6.0, "rp_vol_window": 2},
            "holding_strategy_b": {"rebalance": "monthly", "cost_bps": 6.0, "rp_vol_window": 2},
            "strategy_a": {"lookback_days": 2, "top_k": 1},
            "strategy_b": {"lookback_days": 2, "top_k": 1},
        },
    )
    assert data["ok"] is True
    assert data["comparison"]["target_a"] == "equal_weight"
    assert data["comparison"]["target_b"] == "risk_parity"


def test_sim_gbm_ab_significance_rotation_vs_equal_weight(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/ab-significance",
        {
            "start": _SIM_START,
            "end": _SIM_END_MC,
            "n_worlds": 2,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 11,
            "n_perm": _MIN_PERM,
            "n_boot": _MIN_BOOT,
            "n_jobs": 1,
            "comparison_mode": "rotation_vs_equal_weight",
            "strategy_a": {"lookback_days": 2, "top_k": 1},
            "strategy_b": {},
        },
    )
    assert data["ok"] is True
    assert data["comparison"]["mode"] == "rotation_vs_equal_weight"
    assert data["comparison"]["label_a"] == "轮动策略A"
    assert data["comparison"]["label_b"] == "等权再平衡"


def test_sim_gbm_ab_significance_equal_weight_vs_cash(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/ab-significance",
        {
            "start": _SIM_START,
            "end": _SIM_END_MC,
            "n_worlds": 2,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 11,
            "n_perm": _MIN_PERM,
            "n_boot": _MIN_BOOT,
            "n_jobs": 1,
            "comparison_mode": "equal_weight_vs_cash",
            "strategy_a": {},
            "strategy_b": {},
        },
    )
    assert data["ok"] is True
    assert data["comparison"]["mode"] == "equal_weight_vs_cash"
    assert data["comparison"]["label_a"] == "等权再平衡"
    assert data["comparison"]["label_b"] == "持有现金"


def test_sim_gbm_ab_significance_risk_parity_vs_equal_weight(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/ab-significance",
        {
            "start": _SIM_START,
            "end": _SIM_END_MC,
            "n_worlds": 2,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 11,
            "n_perm": _MIN_PERM,
            "n_boot": _MIN_BOOT,
            "n_jobs": 1,
            "comparison_mode": "risk_parity_vs_equal_weight",
            "strategy_a": {},
            "strategy_b": {},
        },
    )
    assert data["ok"] is True
    assert data["comparison"]["mode"] == "risk_parity_vs_equal_weight"
    assert data["comparison"]["label_a"] == "逆波动加权(仿真,非ERC)"
    assert data["comparison"]["label_b"] == "等权再平衡"


def test_sim_gbm_ab_significance_rotation_vs_risk_parity(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/ab-significance",
        {
            "start": _SIM_START,
            "end": _SIM_END_MC,
            "n_worlds": 2,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 11,
            "n_perm": _MIN_PERM,
            "n_boot": _MIN_BOOT,
            "n_jobs": 1,
            "comparison_mode": "rotation_vs_risk_parity",
            "strategy_a": {"lookback_days": 2, "top_k": 1},
            "strategy_b": {},
        },
    )
    assert data["ok"] is True
    assert data["comparison"]["mode"] == "rotation_vs_risk_parity"
    assert data["comparison"]["label_a"] == "轮动策略A"
    assert data["comparison"]["label_b"] == "逆波动加权(仿真,非ERC)"


def test_sim_gbm_ab_significance_independent_targets_risk_parity_vs_equal_weight(
    api_client,
):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/ab-significance",
        {
            "start": _SIM_START,
            "end": _SIM_END_MC,
            "n_worlds": 2,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 11,
            "n_perm": _MIN_PERM,
            "n_boot": _MIN_BOOT,
            "n_jobs": 1,
            "target_a": "risk_parity",
            "target_b": "equal_weight",
            "strategy_a": {"lookback_days": 2, "top_k": 1},
            "strategy_b": {"lookback_days": 2, "top_k": 1},
        },
    )
    assert data["ok"] is True
    assert data["comparison"]["target_a"] == "risk_parity"
    assert data["comparison"]["target_b"] == "equal_weight"
    assert data["comparison"]["label_a"] == "逆波动加权(仿真,非ERC)"
    assert data["comparison"]["label_b"] == "等权再平衡"


def test_sim_gbm_ab_significance_independent_targets_rotation_b_vs_cash(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/ab-significance",
        {
            "start": _SIM_START,
            "end": _SIM_END_MC,
            "n_worlds": 2,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 11,
            "n_perm": _MIN_PERM,
            "n_boot": _MIN_BOOT,
            "n_jobs": 1,
            "target_a": "rotation_b",
            "target_b": "cash",
            "strategy_a": {"lookback_days": 2, "top_k": 1},
            "strategy_b": {"lookback_days": 2, "top_k": 1},
        },
    )
    assert data["ok"] is True
    assert data["comparison"]["target_a"] == "rotation_b"
    assert data["comparison"]["target_b"] == "cash"
    assert data["comparison"]["label_a"] == "轮动策略B"
    assert data["comparison"]["label_b"] == "持有现金"


def test_sim_gbm_ab_significance_with_seed_stability(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/ab-significance",
        {
            "start": _SIM_START,
            "end": _SIM_END_MC,
            "n_worlds": 2,
            "n_assets": _MIN_ASSETS,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 11,
            "n_perm": _MIN_PERM,
            "n_boot": _MIN_BOOT,
            "n_jobs": 1,
            "comparison_mode": "custom_ab",
            "stability_repeats": 1,
            "stability_worlds": 2,
            "strategy_a": {"lookback_days": 2, "top_k": 1},
            "strategy_b": {"lookback_days": 2, "top_k": 1},
        },
    )
    assert data["ok"] is True
    stab = ((data.get("robustness") or {}).get("seed_stability") or {})
    assert stab.get("enabled") is True
    assert int(stab.get("repeats") or 0) == 1
    assert len(stab.get("mean_diff") or []) == 1
