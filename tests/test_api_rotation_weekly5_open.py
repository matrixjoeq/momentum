import math

import pytest

from tests.helpers.api_test_client import FIXED_MINIPROGRAM_POOL, upsert_and_fetch_etfs
from tests.helpers.rotation_case_data import (
    build_rotation_case_series,
    map_case_series_to_miniprogram_codes,
    post_json,
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

    cap = ((low_rf.get("by_anchor") or {}).get("mix") or {}).get(
        "capacity_estimate"
    ) or {}
    assert str(cap.get("method") or "") == "asset_participation_bottleneck_daily"
    assert str(((cap.get("meta") or {}).get("status") or "")) in {"ok", "unavailable"}

    m_low = low_rf["by_anchor"]["mix"]["metrics"]["strategy"]
    m_high = high_rf["by_anchor"]["mix"]["metrics"]["strategy"]
    assert math.isfinite(float(m_low["sharpe_ratio"]))
    assert math.isfinite(float(m_high["sharpe_ratio"]))
    assert float(m_high["sharpe_ratio"]) == float(m_low["sharpe_ratio"])


def test_api_rotation_weekly5_open_accepts_new_stop_scheme_params(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20240102",
        end_date="20240131",
    )
    data = post_json_ok(
        c,
        "/api/analysis/rotation/weekly5-open",
        {
            "start": "20240102",
            "end": "20240131",
            "anchor_weekday": 5,
            "stop_scheme": "equity_budget",
            "equity_stop_risk_pct": 0.02,
            "atr_stop_execution_mode": "next_day",
            "atr_stop_execution_time": "close",
        },
    )
    assert data["meta"]["type"] == "rotation_weekly5_open"
    assert "by_anchor" in data


def test_api_rotation_weekly5_open_rejects_equity_budget_non_close_execution(
    api_client,
):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20240102",
        end_date="20240131",
    )
    err = post_json(
        c,
        "/api/analysis/rotation/weekly5-open",
        {
            "start": "20240102",
            "end": "20240131",
            "anchor_weekday": 5,
            "stop_scheme": "equity_budget",
            "equity_stop_risk_pct": 0.02,
            "atr_stop_execution_mode": "intraday",
            "atr_stop_execution_time": "open",
        },
        expected_status=422,
    )
    assert "equity_budget" in str(err)


@pytest.mark.parametrize(
    "path",
    [
        "/api/analysis/rotation/weekly5-open",
        "/api/analysis/rotation/weekly5-open-lite",
        "/api/analysis/rotation/weekly5-open-combo-lite",
        "/api/analysis/rotation/weekly5-open-combo",
    ],
)
def test_api_rotation_weekly5_open_variant_rejects_atr_scheme_with_none_mode(
    api_client, path: str
):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20240102",
        end_date="20240131",
    )
    payload = {
        "start": "20240102",
        "end": "20240131",
        "stop_scheme": "atr",
        "atr_stop_mode": "none",
        "atr_stop_execution_mode": "intraday",
        "atr_stop_execution_time": "close",
    }
    if path == "/api/analysis/rotation/weekly5-open":
        payload["anchor_weekday"] = 5
    err = post_json(
        c,
        path,
        payload,
        expected_status=422,
    )
    assert "stop_scheme=atr requires atr_stop_mode" in str(err)


@pytest.mark.parametrize(
    "path",
    [
        "/api/analysis/rotation/weekly5-open",
        "/api/analysis/rotation/weekly5-open-lite",
        "/api/analysis/rotation/weekly5-open-combo-lite",
        "/api/analysis/rotation/weekly5-open-combo",
    ],
)
@pytest.mark.parametrize(
    "patch,err_msg",
    [
        (
            {
                "stop_scheme": "atr",
                "atr_stop_mode": "static",
                "atr_stop_atr_basis": "oops",
            },
            "atr_stop_atr_basis must be one of: entry|latest",
        ),
        (
            {
                "stop_scheme": "atr",
                "atr_stop_mode": "static",
                "atr_stop_reentry_mode": "hold",
            },
            "atr_stop_reentry_mode must be one of: reenter|wait_next_entry",
        ),
        (
            {
                "stop_scheme": "atr",
                "atr_stop_mode": "tightening",
                "atr_stop_n": 0.5,
                "atr_stop_m": 1.0,
            },
            "atr_stop_n must be > atr_stop_m when atr_stop_mode=tightening",
        ),
    ],
)
def test_api_rotation_weekly5_open_variant_rejects_invalid_atr_aux_fields(
    api_client, path: str, patch: dict[str, object], err_msg: str
):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20240102",
        end_date="20240131",
    )
    payload = {
        "start": "20240102",
        "end": "20240131",
        "atr_stop_execution_mode": "intraday",
        "atr_stop_execution_time": "close",
        **patch,
    }
    if path == "/api/analysis/rotation/weekly5-open":
        payload["anchor_weekday"] = 5
    err = post_json(c, path, payload, expected_status=422)
    assert err_msg in str(err)


@pytest.mark.parametrize(
    "path,expect_type",
    [
        ("/api/analysis/rotation/weekly5-open-lite", "rotation_weekly5_open_lite"),
        (
            "/api/analysis/rotation/weekly5-open-combo-lite",
            "rotation_weekly5_open_combo_lite",
        ),
        ("/api/analysis/rotation/weekly5-open-combo", "rotation_weekly5_open_combo"),
    ],
)
def test_api_rotation_weekly5_open_variant_accepts_new_stop_scheme_params(
    api_client, path: str, expect_type: str
):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20240102",
        end_date="20240131",
    )
    data = post_json_ok(
        c,
        path,
        {
            "start": "20240102",
            "end": "20240131",
            "stop_scheme": "equity_budget",
            "equity_stop_risk_pct": 0.02,
            "atr_stop_execution_mode": "next_day",
            "atr_stop_execution_time": "close",
        },
    )
    assert data["meta"]["type"] == expect_type
    assert "by_anchor" in data


@pytest.mark.parametrize(
    "path",
    [
        "/api/analysis/rotation/weekly5-open-lite",
        "/api/analysis/rotation/weekly5-open-combo-lite",
        "/api/analysis/rotation/weekly5-open-combo",
    ],
)
def test_api_rotation_weekly5_open_variant_rejects_equity_budget_non_close_execution(
    api_client, path: str
):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=[x[0] for x in FIXED_MINIPROGRAM_POOL],
        names={k: v for k, v in FIXED_MINIPROGRAM_POOL},
        start_date="20240102",
        end_date="20240131",
    )
    err = post_json(
        c,
        path,
        {
            "start": "20240102",
            "end": "20240131",
            "stop_scheme": "equity_budget",
            "equity_stop_risk_pct": 0.02,
            "atr_stop_execution_mode": "intraday",
            "atr_stop_execution_time": "open",
        },
        expected_status=422,
    )
    assert "equity_budget" in str(err)
