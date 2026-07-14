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
    assert "/global-benchmark-pool" in html
    assert "/futures-pool" in html
    assert "/research/futures" in html
    resp = client.get("/research")
    assert resp.status_code == 200
    research_html = resp.text
    assert "trendMaTrailingStopEnable" in research_html
    assert "trendMaTrailingStopMaType" in research_html
    assert "trendMaTrailingStopExec" in research_html
    assert "trendMaTrailingStopDelayDays" in research_html
    assert "trendMaTrailingStopReduceWin" in research_html
    assert "trendMaTrailingStopExitWin" in research_html
    assert "trendMaTrailingStopReduceFraction" in research_html
    assert "trendRScaleBreakevenStopEnable" in research_html
    assert "trendBiasVTpBreakevenStopEnable" in research_html
    assert "ma_trailing_stop_enabled" in research_html
    assert "ma_trailing_stop_execution_mode" in research_html
    assert "ma_trailing_stop_effective_delay_days" in research_html
    assert "ma_trailing_stop_reduce_fraction" in research_html
    assert "r_profit_scaleout_breakeven_stop_enabled" in research_html
    assert "bias_v_take_profit_breakeven_stop_enabled" in research_html
    assert 'option value="momentum_20_0"' in research_html
    assert "distMomentumTsTable" in research_html
    assert "distMomentumCsTable" in research_html
    assert "lookback=20，skip=0" in research_html
    assert "distFutureN&lt;5 时自动按 5 计算" in research_html
    assert "Math.max(5, Math.min(252, Math.floor(nRaw)))" in research_html
    assert '"momentum_20_0"' in research_html

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

    resp = client.get("/global-benchmark-pool")
    assert resp.status_code == 200
    gb_html = resp.text
    assert "/global-benchmark?adjust=none" in gb_html
    assert "/global-benchmark" in gb_html
    assert "/global-benchmark/fetch-all" in gb_html
    assert "/global-benchmark/fetch-selected" in gb_html
    assert 'data-kind="total_return"' in gb_html
    assert "/global-benchmark/default-universe/install" in gb_html
    assert "/global-benchmark/default-universe/acceptance" in gb_html

    resp = client.get("/research/futures")
    assert resp.status_code == 200
    fut_research = resp.text
    assert "/futures/research/groups" in fut_research
    assert "/futures/research/correlation" in fut_research
    assert "/futures/research/coverage-summary" in fut_research
    assert "/futures/research/correlation-select" in fut_research
    assert "/futures/research/trend-backtest" in fut_research
    assert "/futures/research/rotation-backtest" in fut_research
    assert "corrPickBasis" in fut_research
    assert "futures-research:corr-pick-basis" in fut_research
    assert "futures-research:corr-pick-n" in fut_research
    assert "futures-research:corr-range" in fut_research
    assert "trendRuleBadge" in fut_research
    assert "rule-badge" in fut_research
