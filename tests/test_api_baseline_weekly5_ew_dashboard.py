import pytest

from tests.helpers.api_test_client import FIXED_MINIPROGRAM_POOL, upsert_and_fetch_etfs
from tests.helpers.rotation_case_data import post_json_ok


def test_api_baseline_weekly5_ew_dashboard_smoke(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20240102",
        end_date="20240103",
    )

    data = post_json_ok(
        c,
        "/api/analysis/baseline/weekly5-ew-dashboard",
        {
            "start": "20240102",
            "end": "20240103",
            "risk_free_rate": 0.02,
            "rebalance_shift": "prev",
        },
    )
    assert data["meta"]["type"] == "baseline_weekly5_ew_dashboard"
    assert "by_anchor" in data
    assert set(data["by_anchor"].keys()) == {"1", "2", "3", "4", "5"}
    for k in ["1", "2", "3", "4", "5"]:
        corr = data["by_anchor"][k]["correlation"]
        assert corr["method"] == "pearson_log_return"
        assert isinstance(corr["n_obs"], int)


def test_api_baseline_weekly5_mix_combo_attribution_net_consistency(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20240102",
        end_date="20240103",
    )

    data = post_json_ok(
        c,
        "/api/analysis/baseline/weekly5-ew-dashboard-combo",
        {
            "start": "20240102",
            "end": "20240103",
            "risk_free_rate": 0.02,
            "rebalance_shift": "prev",
        },
    )
    mix = ((data.get("by_anchor") or {}).get("mix")) or {}
    metrics = (mix.get("metrics") or {}) or {}
    attr_ret = ((mix.get("attribution") or {}).get("return")) or {}
    rows = list(attr_ret.get("by_code") or [])

    cum = float(metrics.get("cumulative_return") or 0.0)
    total = float(attr_ret.get("total_return") or 0.0)
    row_sum = float(sum(float((r.get("return_contribution") or 0.0)) for r in rows))

    assert total == pytest.approx(cum, rel=0.0, abs=1e-12)
    assert row_sum == pytest.approx(cum, rel=0.0, abs=1e-12)
