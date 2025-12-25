from __future__ import annotations

from fastapi.testclient import TestClient


def test_prices_crud_via_fetch_read_delete(api_client: TestClient) -> None:
    # Create ETF and fetch (this is our "Create/Update" for prices)
    resp = api_client.post("/api/etf", json={"code": "510300", "name": "沪深300ETF", "start_date": "20240101", "end_date": "20240131"})
    assert resp.status_code == 200

    resp = api_client.post("/api/etf/510300/fetch", json={})
    assert resp.status_code == 200

    # Read prices
    resp = api_client.get("/api/etf/510300/prices")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["trade_date"] == "2024-01-02"

    # Delete one day
    resp = api_client.delete("/api/etf/510300/prices", params={"end": "20240102"})
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 1

    resp = api_client.get("/api/etf/510300/prices")
    assert [x["trade_date"] for x in resp.json()] == ["2024-01-03"]

    # Re-fetch should restore missing day via upsert
    resp = api_client.post("/api/etf/510300/fetch", json={})
    assert resp.status_code == 200

    resp = api_client.get("/api/etf/510300/prices")
    assert [x["trade_date"] for x in resp.json()] == ["2024-01-02", "2024-01-03"]

