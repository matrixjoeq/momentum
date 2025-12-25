from __future__ import annotations

from fastapi.testclient import TestClient

from etf_momentum.app import create_app


def test_frontend_backend_contract_smoke() -> None:
    """
    Lightweight integration test:
    - root page is served
    - frontend HTML references API paths
    - core APIs respond successfully
    """
    with TestClient(create_app()) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.text
        # Frontend uses a helper `api(path)` that prefixes `/api`
        assert "/validation-policies" in html
        assert "/etf" in html
        assert "/fetch-selected" in html

        resp = client.get("/api/validation-policies")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

        resp = client.get("/api/etf")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

