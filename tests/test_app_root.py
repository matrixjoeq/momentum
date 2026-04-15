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


def test_futures_pool_serves_html(api_client: TestClient) -> None:
    resp = api_client.get("/futures-pool")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "期货标的池配置" in resp.text


def test_futures_research_serves_html(api_client: TestClient) -> None:
    resp = api_client.get("/research/futures")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "期货研究" in resp.text


def test_favicon_does_not_404(api_client: TestClient) -> None:
    client = api_client
    resp = client.get("/favicon.ico")
    assert resp.status_code in (200, 204)


def test_static_shared_terminal_css(api_client: TestClient) -> None:
    resp = api_client.get("/static/terminal.css")
    assert resp.status_code == 200
    assert ":root" in resp.text
    assert "Noto Sans SC" in resp.text

