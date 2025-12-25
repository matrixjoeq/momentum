from __future__ import annotations

from fastapi.testclient import TestClient

def test_create_list_delete_etf_and_fetch(api_client: TestClient) -> None:
    client = api_client

    resp = client.post(
        "/api/etf",
        json={"code": "510300", "name": "沪深300ETF", "start_date": "20240101", "end_date": "20240131"},
    )
    assert resp.status_code == 200
    data = resp.json()
    # auto-inferred policy should exist
    assert data["validation_policy"]["name"] == "cn_stock_etf_10"

    resp = client.get("/api/etf")
    assert resp.status_code == 200
    assert len(resp.json()) == 1

    resp = client.post("/api/etf/510300/fetch")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    # fetched data range should be updated
    resp = client.get("/api/etf")
    item = resp.json()[0]
    assert item["last_data_start_date"] == "20240102"
    assert item["last_data_end_date"] == "20240103"

    resp = client.delete("/api/etf/510300")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": True}

