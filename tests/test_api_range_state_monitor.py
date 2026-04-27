from __future__ import annotations

from tests.helpers.api_test_client import upsert_and_fetch_etfs
from tests.helpers.rotation_case_data import post_json, post_json_ok


def test_api_range_state_monitor_contract_ok(api_client) -> None:
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=["510300"],
        names={"510300": "沪深300ETF"},
        start_date="20240101",
        end_date="20240131",
    )
    data = post_json_ok(
        c,
        "/api/analysis/range-state-monitor",
        {
            "etf_code": "510300",
            "start": "20240101",
            "end": "20240131",
            "adjust": "qfq",
            "mode": "adx",
            "window": 14,
            "enter_threshold": 20.0,
            "exit_threshold": 25.0,
        },
    )
    assert data["ok"] is True
    assert (data.get("meta") or {}).get("etf_code") == "510300"
    assert "series" in data
    assert "summary" in data
    assert isinstance((data.get("series") or {}).get("dates") or [], list)
    assert isinstance((data.get("series") or {}).get("state") or [], list)


def test_api_range_state_monitor_rejects_invalid_threshold(api_client) -> None:
    c = api_client
    err = post_json(
        c,
        "/api/analysis/range-state-monitor",
        {
            "etf_code": "510300",
            "start": "20240101",
            "end": "20240131",
            "adjust": "qfq",
            "mode": "er",
            "window": 10,
            "enter_threshold": 0.6,
            "exit_threshold": 0.5,
        },
        expected_status=422,
    )
    assert "exit_threshold must be >= enter_threshold" in str(err)


def test_api_range_state_monitor_supports_adxr_mode(api_client) -> None:
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=["510300"],
        names={"510300": "沪深300ETF"},
        start_date="20240101",
        end_date="20240131",
    )
    data = post_json_ok(
        c,
        "/api/analysis/range-state-monitor",
        {
            "etf_code": "510300",
            "start": "20240101",
            "end": "20240131",
            "adjust": "qfq",
            "mode": "adxr",
            "window": 14,
            "enter_threshold": 20.0,
            "exit_threshold": 25.0,
        },
    )
    assert data["ok"] is True
    assert str((data.get("meta") or {}).get("mode") or "").lower() == "adxr"
