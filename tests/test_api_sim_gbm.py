import pytest  # pylint: disable=import-error

from tests.helpers.rotation_case_data import post_json_ok


@pytest.mark.parametrize("path", ["/api/analysis/sim/gbm/phase1", "/api/analysis/sim/gbm/phase2"])
def test_sim_gbm_phase1_and_phase2_ok(api_client, path):
    c = api_client
    data = post_json_ok(
        c,
        path,
        {
            "start": "19900101",
            "end": "19900330",
            "n_assets": 4,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 123,
            "lookback_days": 20,
        },
    )
    assert data["ok"] is True


def test_sim_gbm_phase3_ok(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/phase3",
        {
            "start": "19900101",
            "end": "19920331",
            "n_assets": 4,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 7,
            "lookback_days": 20,
            "n_sims": 200,
            "chunk_size": 50,
        },
    )
    assert data["ok"] is True
    assert "dist" in data
    assert len(data["dist"]["rotation"]["cagr"]) == 200
    assert len(data["dist"]["equal_weight"]["cagr"]) == 200


def test_sim_gbm_phase4_ok(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/phase4",
        {
            "start": "19900101",
            "end": "19920331",
            "n_assets": 4,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 7,
            "lookback_days": 20,
            "n_sims": 200,
            "chunk_size": 50,
            "initial_cash": 1000000,
            "position_pct": 0.10,
        },
    )
    assert data["ok"] is True
    assert "sizing" in data
    assert "one" in data


def test_sim_gbm_ab_significance_ok(api_client):
    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/sim/gbm/ab-significance",
        {
            "start": "19900101",
            "end": "19911231",
            "n_worlds": 2,
            "n_assets": 4,
            "vol_low": 0.05,
            "vol_high": 0.30,
            "seed": 11,
            "n_perm": 1200,
            "n_boot": 1200,
            "strategy_a": {"lookback_days": 20, "top_k": 1, "trend_filter": True, "trend_sma_window": 5, "trend_ma_type": "ema"},
            "strategy_b": {"lookback_days": 20, "top_k": 1, "trend_filter": False},
        },
    )
    assert data["ok"] is True
    assert "stats" in data
    assert "dist" in data
    p = float(data["stats"]["pvalue_permutation_one_sided"])
    assert 0.0 <= p <= 1.0
    ci = data["stats"]["bootstrap_ci_95"]["mean"]
    assert isinstance(ci, list) and len(ci) == 2

