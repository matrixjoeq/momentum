from __future__ import annotations

import datetime as dt

from fastapi.testclient import TestClient

from etf_momentum.data.global_benchmark_defaults import (
    DEFAULT_GLOBAL_BENCHMARK_UNIVERSE,
)
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


def test_global_benchmark_fetch_one_and_batch_contract(api_client: TestClient) -> None:
    client = api_client
    _ = client.post(
        "/api/global-benchmark",
        json={
            "code": "000300",
            "name": "沪深300",
            "code_format": "cn_6",
            "provider_hint": "tencent",
            "start_date": "20240101",
            "end_date": "20240131",
        },
    )
    one = client.post("/api/global-benchmark/000300/fetch", json={})
    assert one.status_code == 200
    out = one.json()
    assert out["status"] == "success"
    assert out["final_provider"] == "tencent"
    assert len(out["provider_attempts"]) == 1
    prices = client.get("/api/global-benchmark/000300/prices?adjust=none").json()
    assert len(prices) >= 1

    all_resp = client.post("/api/global-benchmark/fetch-all", json={})
    assert all_resp.status_code == 200
    rows = all_resp.json()
    assert len(rows) == 1
    assert rows[0]["status"] == "success"

    sel = client.post(
        "/api/global-benchmark/fetch-selected",
        json={"codes": ["NOPE", "000300"]},
    )
    assert sel.status_code == 200
    out_sel = sel.json()
    by_code = {x["code"]: x for x in out_sel}
    assert by_code["NOPE"]["status"] == "failed"
    assert by_code["000300"]["status"] == "success"


def test_global_benchmark_default_universe_install_and_acceptance(
    api_client: TestClient,
) -> None:
    client = api_client
    r = client.post(
        "/api/global-benchmark/default-universe/install",
        json={"overwrite_existing": False},
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is True
    assert out["total"] == len(DEFAULT_GLOBAL_BENCHMARK_UNIVERSE)
    assert out["inserted"] == len(DEFAULT_GLOBAL_BENCHMARK_UNIVERSE)
    assert out["updated"] == 0
    assert out["skipped"] == 0

    r2 = client.post(
        "/api/global-benchmark/default-universe/install",
        json={"overwrite_existing": False},
    )
    assert r2.status_code == 200
    out2 = r2.json()
    assert out2["skipped"] == len(DEFAULT_GLOBAL_BENCHMARK_UNIVERSE)

    # no-network acceptance contract path
    chk = client.post(
        "/api/global-benchmark/default-universe/acceptance",
        json={"fetch": False, "continue_on_error": True},
    )
    assert chk.status_code == 200
    rep = chk.json()
    assert rep["total"] == len(DEFAULT_GLOBAL_BENCHMARK_UNIVERSE)
    assert rep["failed"] == 0
    assert rep["skipped"] == len(DEFAULT_GLOBAL_BENCHMARK_UNIVERSE)

    # smoke fetch with fake-ak covered code only
    chk2 = client.post(
        "/api/global-benchmark/default-universe/acceptance",
        json={"codes": ["000300"], "fetch": True, "continue_on_error": False},
    )
    assert chk2.status_code == 200
    rep2 = chk2.json()
    assert rep2["total"] == 1
    assert rep2["succeeded"] == 1
    assert rep2["failed"] == 0
    assert rep2["items"][0]["code"] == "000300"
    assert rep2["items"][0]["status"] == "success"
