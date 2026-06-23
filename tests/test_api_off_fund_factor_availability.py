from __future__ import annotations

import datetime as dt

from fastapi.testclient import TestClient

from etf_momentum.db.models import EtfPrice
from etf_momentum.db.session import make_session_factory


def _seed_benchmark_prices(client: TestClient) -> tuple[str, str]:
    engine = client.app.state.engine
    sf = make_session_factory(engine)
    d0 = dt.date(2024, 1, 1)
    with sf() as db:
        for i in range(220):
            d = d0 + dt.timedelta(days=i)
            p300 = 1.0 + 0.002 * i
            db.add(
                EtfPrice(
                    code="000300",
                    trade_date=d,
                    open=p300,
                    high=p300,
                    low=p300,
                    close=p300,
                    volume=1000.0,
                    amount=1000.0 * p300,
                    source="unit_test",
                    adjust="hfq",
                )
            )
            if i < 100:
                p500 = 1.0 + 0.001 * i
                db.add(
                    EtfPrice(
                        code="000905",
                        trade_date=d,
                        open=p500,
                        high=p500,
                        low=p500,
                        close=p500,
                        volume=1000.0,
                        amount=1000.0 * p500,
                        source="unit_test",
                        adjust="hfq",
                    )
                )
        db.commit()
    return d0.strftime("%Y%m%d"), (d0 + dt.timedelta(days=219)).strftime("%Y%m%d")


def test_factor_availability_returns_per_factor_coverage(api_client: TestClient) -> None:
    start, end = _seed_benchmark_prices(api_client)
    resp = api_client.post(
        "/api/analysis/off-fund/factor-availability",
        json={
            "start": start,
            "end": end,
            "benchmark_adjust": "hfq",
            "rolling_window": 200,
            "min_samples": 120,
            "benchmark_factors": [
                {"key": "F300", "label": "大盘", "aliases": ["000300"]},
                {"key": "F500", "label": "中盘", "aliases": ["000905"]},
            ],
        },
    )
    assert resp.status_code == 200
    out = resp.json()
    assert out["ok"] is True
    items = out["items"]
    assert len(items) == 2
    by_key = {x["key"]: x for x in items}
    assert by_key["F300"]["selected_code"] == "000300"
    assert by_key["F300"]["enough"] is True
    assert by_key["F500"]["selected_code"] == "000905"
    assert by_key["F500"]["enough"] is False
    assert int(out["meta"]["enough_factor_count"]) == 1


def test_factor_availability_profile_handles_empty_benchmark_data(
    api_client: TestClient,
) -> None:
    resp = api_client.post(
        "/api/analysis/off-fund/factor-availability",
        json={
            "start": "20240101",
            "end": "20240501",
            "benchmark_adjust": "hfq",
            "benchmark_profile": "cn_stock_core",
            "rolling_window": 120,
            "min_samples": 80,
        },
    )
    assert resp.status_code == 200
    out = resp.json()
    assert out["ok"] is True
    assert int(out["meta"]["factor_count"]) == 4
    assert int(out["meta"]["enough_factor_count"]) == 0
    assert all(str(x["status"]) == "missing" for x in out["items"])
