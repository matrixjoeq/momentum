from __future__ import annotations

import datetime as dt

from fastapi.testclient import TestClient

from etf_momentum.db.models import GlobalBenchmarkPrice
from etf_momentum.db.session import make_session_factory


def test_create_list_delete_global_benchmark(api_client: TestClient) -> None:
    client = api_client
    r = client.post(
        "/api/global-benchmark",
        json={
            "code": "^GSPC",
            "name": "标普500",
            "code_format": "yahoo",
            "provider_hint": "auto",
            "start_date": "20000101",
            "end_date": "20250101",
        },
    )
    assert r.status_code == 200
    out = r.json()
    assert out["code"] == "^GSPC"
    assert out["code_format"] == "yahoo"

    rows = client.get("/api/global-benchmark?adjust=none").json()
    assert len(rows) == 1
    assert rows[0]["code"] == "^GSPC"

    d = client.delete("/api/global-benchmark/%5EGSPC")
    assert d.status_code == 200
    body = d.json()
    assert body["deleted"] is True
    assert "purged" in body
    assert "prices" in body["purged"]


def test_global_benchmark_prices_contract_and_date_range(
    api_client: TestClient,
) -> None:
    client = api_client
    _ = client.post(
        "/api/global-benchmark",
        json={
            "code": "^NDX",
            "name": "纳指100",
            "start_date": "20240101",
            "end_date": "20240131",
        },
    )
    sf = make_session_factory(client.app.state.engine)
    with sf() as db:
        db.add(
            GlobalBenchmarkPrice(
                code="^NDX",
                trade_date=dt.date(2024, 1, 2),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1.0,
                amount=100.0,
                source="unit",
                adjust="none",
            )
        )
        db.add(
            GlobalBenchmarkPrice(
                code="^NDX",
                trade_date=dt.date(2024, 1, 3),
                open=100.5,
                high=102.0,
                low=100.0,
                close=101.5,
                volume=1.2,
                amount=121.8,
                source="unit",
                adjust="none",
            )
        )
        db.commit()

    prices = client.get("/api/global-benchmark/%5ENDX/prices?adjust=none").json()
    assert len(prices) == 2
    assert prices[0]["code"] == "^NDX"
    assert prices[0]["adjust"] == "none"
    pool = client.get("/api/global-benchmark?adjust=none").json()
    by_code = {x["code"]: x for x in pool}
    assert by_code["^NDX"]["last_data_start_date"] == "20240102"
    assert by_code["^NDX"]["last_data_end_date"] == "20240103"


def test_global_benchmark_rejects_non_none_adjust(api_client: TestClient) -> None:
    client = api_client
    bad_list = client.get("/api/global-benchmark?adjust=hfq")
    assert bad_list.status_code == 400
    bad_prices = client.get("/api/global-benchmark/000300/prices?adjust=qfq")
    assert bad_prices.status_code == 400
