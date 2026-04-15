from __future__ import annotations

from fastapi.testclient import TestClient


def test_frontend_backend_contract_smoke(api_client: TestClient) -> None:
    """
    Lightweight integration test:
    - root page is served
    - frontend HTML references API paths
    - core APIs respond successfully
    """
    client = api_client
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    # Frontend uses a helper `api(path)` that prefixes `/api`
    assert "/validation-policies" in html
    assert "/etf" in html
    assert "/fetch-selected" in html
    assert "/fetch-all" in html
    assert "fetchMode" in html
    assert "/futures-pool" in html
    assert "/research/futures" in html

    resp = client.get("/api/validation-policies")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    resp = client.get("/api/etf")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    resp = client.get("/futures-pool")
    assert resp.status_code == 200
    fut_html = resp.text
    assert "/futures/fetch-selected" in fut_html
    assert "/futures/fetch-all" in fut_html

    resp = client.get("/research/futures")
    assert resp.status_code == 200
    fut_research = resp.text
    assert "/futures/research/groups" in fut_research
    assert "/futures/research/correlation" in fut_research
    assert "/futures/research/coverage-summary" in fut_research
    assert "/futures/research/correlation-select" in fut_research
    assert "/futures/research/trend-backtest" in fut_research
    assert "corrPickBasis" in fut_research
    assert "futures-research:corr-pick-basis" in fut_research
    assert "futures-research:corr-pick-n" in fut_research
    assert "futures-research:corr-range" in fut_research
    assert "trendRuleBadge" in fut_research
    assert "rule-badge" in fut_research

