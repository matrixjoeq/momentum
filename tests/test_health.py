from fastapi.testclient import TestClient


def test_health_ok() -> None:
    # health doesn't need DB; use isolated api_client anyway for consistency
    from etf_momentum.app import create_app

    client = TestClient(create_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_api_list_empty(api_client: TestClient) -> None:
    client = api_client
    resp = client.get("/api/etf")
    assert resp.status_code == 200
    assert resp.json() == []

