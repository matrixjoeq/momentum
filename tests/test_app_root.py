from __future__ import annotations

from fastapi.testclient import TestClient

from etf_momentum.app import create_app


def test_root_serves_html() -> None:
    # use context manager to trigger lifespan startup/shutdown
    with TestClient(create_app()) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "ETF 候选池配置" in resp.text


def test_research_serves_html() -> None:
    with TestClient(create_app()) as client:
        resp = client.get("/research")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "基准分析" in resp.text

