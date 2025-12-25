from __future__ import annotations

from fastapi.testclient import TestClient

def test_list_validation_policies(api_client: TestClient) -> None:
    client = api_client
    resp = client.get("/api/validation-policies")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert any(p["name"] == "cn_stock_etf_10" for p in data)

