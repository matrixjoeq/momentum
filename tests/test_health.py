from fastapi.testclient import TestClient


def test_health_ok(api_client: TestClient) -> None:
    client = api_client
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_api_list_empty(api_client: TestClient) -> None:
    client = api_client
    resp = client.get("/api/etf")
    assert resp.status_code == 200
    assert resp.json() == []

