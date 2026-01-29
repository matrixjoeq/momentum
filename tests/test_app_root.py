from __future__ import annotations

from fastapi.testclient import TestClient


def test_root_serves_html(api_client: TestClient) -> None:
    client = api_client
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "ETF 候选池配置" in resp.text


def test_research_serves_html(api_client: TestClient) -> None:
    client = api_client
    resp = client.get("/research")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "基准分析" in resp.text

