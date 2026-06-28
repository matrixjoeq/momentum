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
    assert "/etf/research/groups" in resp.text


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


def test_trading_records_serves_html(api_client: TestClient) -> None:
    resp = api_client.get("/trading-records")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "实盘交易记录" in resp.text
    assert "资金总览（按所选维度）" in resp.text
    assert 'id="etfCodeDatalist"' in resp.text
    assert 'id="tradeCode"' in resp.text
    assert 'id="caCode"' in resp.text
    assert 'list="etfCodeDatalist"' in resp.text
    assert 'id="tradeDate" type="date"' in resp.text
    assert 'id="accountFlowDate" type="date"' in resp.text
    assert 'id="transferDate" type="date"' in resp.text
    assert 'id="tradeTime"' in resp.text
    assert 'type="time"' in resp.text
    assert 'min="09:00:00"' in resp.text
    assert 'max="15:00:00"' in resp.text
    assert 'step="1"' in resp.text
    assert 'id="tradeStrategySelect"' in resp.text
    assert 'id="tradeQty"' in resp.text and 'min="100"' in resp.text
    assert 'id="tradeQty"' in resp.text and 'step="100"' in resp.text
    assert 'id="tradeTimeQuick"' in resp.text
    assert "开盘" in resp.text and "收盘" in resp.text
    assert "策略</th>" in resp.text
    assert "股东账号</th>" in resp.text
    assert 'id="holdingsTable"' in resp.text
    assert 'id="recentTradesTable"' in resp.text
    assert 'id="closedRoundsTable"' in resp.text
    assert 'id="cancelTradeEditBtn"' in resp.text
    assert 'id="reasonModal"' in resp.text
    assert 'id="reasonModalInput"' in resp.text
    assert 'id="reasonModalConfirmBtn"' in resp.text
    assert "操作</th>" in resp.text
    assert 'id="roundCodeFilter"' in resp.text
    assert 'class="sortable-th"' in resp.text
    assert 'data-sort-key="trade_date"' in resp.text
    assert 'data-sort-key="market_value"' in resp.text
    assert 'data-sort-key="open_date"' in resp.text
    assert 'data-sort-key="realized_pnl"' in resp.text
    assert "/static/vendor/flatpickr.min.css" in resp.text
    assert "/static/vendor/flatpickr.min.js" in resp.text
    assert "/static/vendor/plotly-cartesian-2.30.0.min.js" in resp.text


def test_favicon_does_not_404(api_client: TestClient) -> None:
    client = api_client
    resp = client.get("/favicon.ico")
    assert resp.status_code in (200, 204)


def test_static_shared_terminal_css(api_client: TestClient) -> None:
    resp = api_client.get("/static/terminal.css")
    assert resp.status_code == 200
    assert ":root" in resp.text
    assert "Noto Sans SC" in resp.text


def test_static_trading_records_plotly_bundle(api_client: TestClient) -> None:
    resp = api_client.get("/static/vendor/plotly-cartesian-2.30.0.min.js")
    assert resp.status_code == 200
    assert "javascript" in resp.headers.get("content-type", "")


def test_static_trading_records_flatpickr_assets(api_client: TestClient) -> None:
    js = api_client.get("/static/vendor/flatpickr.min.js")
    css = api_client.get("/static/vendor/flatpickr.min.css")
    assert js.status_code == 200
    assert "javascript" in js.headers.get("content-type", "")
    assert css.status_code == 200
    assert "css" in css.headers.get("content-type", "")


def test_static_calendar_timing_param_search_page(api_client: TestClient) -> None:
    resp = api_client.get("/static/calendar_timing_param_search.html")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "日历效应择时参数搜索结果" in resp.text
