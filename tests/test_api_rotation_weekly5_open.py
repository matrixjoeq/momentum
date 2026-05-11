import math

from tests.helpers.api_test_client import FIXED_MINIPROGRAM_POOL, upsert_and_fetch_etfs
from tests.helpers.rotation_case_data import (
    build_rotation_case_series,
    map_case_series_to_miniprogram_codes,
    post_json_ok,
    seed_prices,
)


def test_api_rotation_weekly5_open_sim(api_client):
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
        "/api/analysis/rotation/weekly5-open",
        {"start": "20240102", "end": "20240103", "anchor_weekday": 5},
    )
    assert data["meta"]["type"] == "rotation_weekly5_open"
    assert data["meta"]["exec_price"] == "open"
    assert data["meta"]["rebalance_shift"] == "prev"
    by = data["by_anchor"]
    assert set(by.keys()) == {"5"}
    # spot-check one result payload shape
    one = by["5"]
    assert "nav" in one and "series" in one["nav"]
    assert "ROTATION" in one["nav"]["series"]
    assert "EW_REBAL" in one["nav"]["series"]
    assert "EXCESS" in one["nav"]["series"]


def test_api_rotation_weekly5_open_combo_ignores_payload_risk_free_rate(
    api_client, engine
):
    c = api_client
    dates, src = build_rotation_case_series()
    mapped = map_case_series_to_miniprogram_codes(src)
    seed_prices(engine, code_to_series=mapped, dates=dates)

    payload = {
        "start": "20240102",
        "end": "20240731",
        "score_method": "raw_mom",
        "top_k": 1,
    }
    low_rf = post_json_ok(
        c,
        "/api/analysis/rotation/weekly5-open-combo",
        {**payload, "risk_free_rate": 0.0},
    )
    high_rf = post_json_ok(
        c,
        "/api/analysis/rotation/weekly5-open-combo",
        {**payload, "risk_free_rate": 0.20},
    )

    nav_low = low_rf["by_anchor"]["mix"]["nav"]["series"]["ROTATION"]
    nav_high = high_rf["by_anchor"]["mix"]["nav"]["series"]["ROTATION"]
    assert nav_low == nav_high

    m_low = low_rf["by_anchor"]["mix"]["metrics"]["strategy"]
    m_high = high_rf["by_anchor"]["mix"]["metrics"]["strategy"]
    assert math.isfinite(float(m_low["sharpe_ratio"]))
    assert math.isfinite(float(m_high["sharpe_ratio"]))
    assert float(m_high["sharpe_ratio"]) == float(m_low["sharpe_ratio"])
