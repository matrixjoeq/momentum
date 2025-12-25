from __future__ import annotations

from fastapi.testclient import TestClient


def test_fetch_selected_success(api_client: TestClient) -> None:
    api_client.post("/api/etf", json={"code": "510300", "name": "沪深300ETF"})
    api_client.post("/api/etf", json={"code": "588000", "name": "科创50ETF"})

    resp = api_client.post("/api/fetch-selected", json={"codes": ["510300"]})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["code"] == "510300"
    assert data[0]["status"] == "success"


def test_fetch_selected_missing_code(api_client: TestClient) -> None:
    api_client.post("/api/etf", json={"code": "510300", "name": "沪深300ETF"})
    resp = api_client.post("/api/fetch-selected", json={"codes": ["NOPE", "510300"]})
    assert resp.status_code == 200
    data = resp.json()
    assert {x["code"] for x in data} == {"NOPE", "510300"}
    assert any(x["code"] == "NOPE" and x["status"] == "failed" for x in data)

