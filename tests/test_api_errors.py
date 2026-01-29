from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient


def test_upsert_date_validation_errors(api_client: TestClient) -> None:
    # bad start_date length
    resp = api_client.post("/api/etf", json={"code": "510300", "name": "沪深300ETF", "start_date": "202401", "end_date": None})
    assert resp.status_code == 400

    # bad end_date length
    resp = api_client.post("/api/etf", json={"code": "510300", "name": "沪深300ETF", "start_date": None, "end_date": "2024"})
    assert resp.status_code == 400

    # start > end
    resp = api_client.post(
        "/api/etf",
        json={"code": "510300", "name": "沪深300ETF", "start_date": "20240201", "end_date": "20240101"},
    )
    assert resp.status_code == 400


def test_delete_missing_etf_404(api_client: TestClient) -> None:
    resp = api_client.delete("/api/etf/NOPE")
    assert resp.status_code == 404


def test_fetch_missing_etf_404(api_client: TestClient) -> None:
    resp = api_client.post("/api/etf/NOPE/fetch", json={})
    assert resp.status_code == 404


def test_fetch_failure_sets_status_failed(api_client: TestClient) -> None:
    # create ETF first
    resp = api_client.post("/api/etf", json={"code": "510300", "name": "沪深300ETF"})
    assert resp.status_code == 200

    class BadAk:
        def fund_etf_hist_em(self, **kwargs):
            raise RuntimeError("boom")

    # override akshare dependency to force failure
    import etf_momentum.api.routes as routes

    api_client.app.dependency_overrides[routes.get_akshare] = lambda: BadAk()

    resp = api_client.post("/api/etf/510300/fetch", json={})
    assert resp.status_code == 500

    # status should be marked failed
    resp = api_client.get("/api/etf")
    item = resp.json()[0]
    assert item["last_fetch_status"] == "failed"
    # Fetchers may swallow provider exceptions and report an aggregated fetch failure message.
    assert (item["last_fetch_message"] or "") != ""


def test_fetch_all_has_failed_and_success(api_client: TestClient) -> None:
    api_client.post("/api/etf", json={"code": "510300", "name": "沪深300ETF"})
    api_client.post("/api/etf", json={"code": "588000", "name": "科创50ETF"})

    class MixedAk:
        def fund_etf_hist_em(self, **kwargs):
            if kwargs.get("symbol") == "588000":
                raise RuntimeError("bad-symbol")
            return pd.DataFrame(
                {
                    "日期": ["2024-01-02"],
                    "开盘": [1.0],
                        "最高": [1.15],
                        "最低": [0.95],
                    "收盘": [1.1],
                }
            )

    import etf_momentum.api.routes as routes

    api_client.app.dependency_overrides[routes.get_akshare] = lambda: MixedAk()
    resp = api_client.post("/api/fetch-all", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert any(x["code"] == "510300" and x["status"] == "success" for x in data)
    assert any(x["code"] == "588000" and x["status"] == "failed" for x in data)

