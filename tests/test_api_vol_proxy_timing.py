from __future__ import annotations

from tests.helpers.api_test_client import upsert_and_fetch_etfs
from tests.helpers.rotation_case_data import post_json_ok


def test_api_vol_proxy_timing_smoke(api_client) -> None:
    # Ensure ETF prices exist (fake AkShare provides 2 days only)
    upsert_and_fetch_etfs(
        api_client,
        codes=["510300"],
        names={"510300": "沪深300ETF"},
        start_date="20240101",
        end_date="20240131",
    )

    # The series is too short for full timing, but endpoint should respond and not crash.
    data = post_json_ok(
        api_client,
        "/api/analysis/vol-proxy-timing",
        {
            "etf_code": "510300",
            "start": "20240101",
            "end": "20240131",
            "adjust": "hfq",
            "methods": [{"name": "rv20", "kind": "rv_close", "window": 20, "ann": 252}],
            "level_quantiles": [0.8, 0.9],
            "level_exposures": [1.0, 0.5, 0.2],
            "trade_cost_bps": 10,
            "walk_forward": True,
            "train_ratio": 0.6,
        },
    )
    # It may be ok=false due to insufficient samples, but must return a structured response.
    assert "ok" in data
    assert "error" in data or "methods" in data

