from __future__ import annotations

from fastapi.testclient import TestClient


def test_delete_off_fund_always_purge_mode(api_client: TestClient) -> None:
    client = api_client
    r = client.post(
        "/api/off-fund",
        json={"code": "000001", "name": "测试基金", "start_date": "20240101", "end_date": "20241231"},
    )
    assert r.status_code == 200

    d = client.delete("/api/off-fund/000001")
    assert d.status_code == 200
    body = d.json()
    assert body["deleted"] is True
    assert body["purged"] is not None
    assert "navs" in body["purged"]
    assert "events" in body["purged"]
