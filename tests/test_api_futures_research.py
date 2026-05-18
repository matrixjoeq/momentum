from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from fastapi.testclient import TestClient

from etf_momentum.analysis import futures_research as fut_research
from etf_momentum.db.futures_research_repo import FuturesGroupData
from tests.helpers.rotation_case_data import post_json_ok


def test_futures_research_groups_state_and_correlation(api_client: TestClient) -> None:
    client = api_client

    # Prepare pool and prices
    post_json_ok(
        client,
        "/api/futures",
        {
            "code": "RB0",
            "name": "螺纹钢主连",
            "contract_multiplier": 10.0,
            "min_price_tick": 1.0,
        },
    )
    post_json_ok(
        client,
        "/api/futures",
        {
            "code": "IF0",
            "name": "股指主连",
            "contract_multiplier": 300.0,
            "min_price_tick": 0.2,
        },
    )
    post_json_ok(client, "/api/futures/RB0/fetch", {})
    post_json_ok(client, "/api/futures/IF0/fetch", {})
    synth = client.post("/api/futures/synthesize-all")
    assert synth.status_code == 200
    assert synth.json().get("succeeded", 0) >= 1

    # Save group as active
    g = post_json_ok(
        client,
        "/api/futures/research/groups",
        {"name": "默认分组", "codes": ["RB0", "IF0"], "set_active": True},
    )
    assert g["name"] == "默认分组"
    assert g["is_active"] is True

    # Update shared state with quick range
    s = client.put(
        "/api/futures/research/state",
        json={
            "start_date": "20240101",
            "end_date": "20241231",
            "dynamic_universe": True,
            "quick_range_key": "1y",
        },
    )
    assert s.status_code == 200
    st = s.json()
    assert st["dynamic_universe"] is True
    assert st["active_group"] == "默认分组"

    # Correlation uses active group by default
    c = client.post("/api/futures/research/correlation", json={"range_key": "all"})
    assert c.status_code == 200
    data = c.json()
    assert data["ok"] is True
    assert len(data["aliases"]) == 2
    assert len(data["matrix"]) == 2
    assert len(data["matrix"][0]) == 2
    cm = data.get("meta") or {}
    assert cm.get("returns") == "daily_log"
    assert cm.get("close_price_basis") == "hfq"
    assert cm.get("continuous_series") == "{root}889"

    cov = client.post(
        "/api/futures/research/coverage-summary", json={"range_key": "all"}
    )
    assert cov.status_code == 200
    cs = cov.json()
    assert cs["ok"] is True
    assert cs["meta"]["union_points"] >= 1
    assert cs["meta"]["intersection_points"] >= 1
    assert cs["meta"]["effective_points"] >= 1
    assert len(cs["symbols"]) == 2

    pick = client.post(
        "/api/futures/research/correlation-select",
        json={"range_key": "all", "mode": "lowest", "score_basis": "mean_abs", "n": 1},
    )
    assert pick.status_code == 200
    ps = pick.json()
    assert ps["ok"] is True
    assert ps["mode"] == "lowest"
    assert ps["score_basis"] == "mean_abs"
    assert ps["effective_n"] == 1
    assert len(ps["items"]) == 1
    assert "avg_corr" in ps["items"][0]
    assert "avg_abs_corr" in ps["items"][0]

    # Regression: non-all ranges should not become empty when state end is in future.
    s_future = client.put(
        "/api/futures/research/state",
        json={
            "start_date": "20240101",
            "end_date": "20991231",
            "dynamic_universe": True,
            "quick_range_key": "all",
        },
    )
    assert s_future.status_code == 200
    c_1m = client.post("/api/futures/research/correlation", json={"range_key": "1m"})
    assert c_1m.status_code == 200
    d_1m = c_1m.json()
    assert d_1m["ok"] is True
    assert d_1m["meta"]["range_key"] == "1m"


def test_futures_correlation_handles_non_positive_hfq_closes(monkeypatch) -> None:
    idx = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)]

    def _fake_list_futures_prices(
        _db,
        *,
        code,
        adjust,
        start_date,
        end_date,
        limit,
    ):
        assert adjust == "hfq"
        if code == "RB889":
            closes = [100.0, 0.0, 102.0]
        elif code == "IF889":
            closes = [200.0, -1.0, 201.0]
        else:
            closes = [100.0, 101.0, 102.0]
        return [
            SimpleNamespace(trade_date=d, close=v)
            for d, v in zip(idx, closes, strict=False)
        ]

    monkeypatch.setattr(fut_research, "list_futures_prices", _fake_list_futures_prices)
    monkeypatch.setattr(
        fut_research,
        "list_futures_pool",
        lambda _db: [
            SimpleNamespace(code="RB0", name="螺纹钢主连"),
            SimpleNamespace(code="IF0", name="股指主连"),
        ],
    )

    out = fut_research.compute_futures_group_correlation(
        db=None,  # type: ignore[arg-type]
        group=FuturesGroupData(name="G", codes=["RB0", "IF0"], is_active=True),
        start="20240101",
        end="20240131",
        dynamic_universe=True,
        min_obs=2,
    )
    assert out["ok"] is True
    assert len(out["matrix"]) == 2
    assert len(out["matrix"][0]) == 2
    assert out["meta"]["returns"] == "daily_log"
    assert out["meta"]["close_price_basis"] == "hfq"


def test_futures_research_groups_import_export_overwrite(
    api_client: TestClient,
) -> None:
    client = api_client
    post_json_ok(
        client,
        "/api/futures",
        {
            "code": "RB0",
            "name": "螺纹钢主连",
            "contract_multiplier": 10.0,
            "min_price_tick": 1.0,
        },
    )
    post_json_ok(
        client,
        "/api/futures",
        {
            "code": "IF0",
            "name": "股指主连",
            "contract_multiplier": 300.0,
            "min_price_tick": 0.2,
        },
    )

    post_json_ok(
        client,
        "/api/futures/research/groups",
        {"name": "A组", "codes": ["RB0"], "set_active": True},
    )
    # same-name overwrite
    post_json_ok(
        client,
        "/api/futures/research/groups",
        {"name": "A组", "codes": ["IF0"], "set_active": True},
    )
    all_groups = client.get("/api/futures/research/groups").json()
    hit = [x for x in all_groups if x["name"] == "A组"]
    assert len(hit) == 1
    assert hit[0]["codes"] == ["IF0"]

    exported = client.get("/api/futures/research/groups-export")
    assert exported.status_code == 200
    body = exported.json()
    assert body["format"] == "etf_momentum_futures_groups_v1"
    assert "A组" in body["groups"]

    imported = client.post(
        "/api/futures/research/groups-import",
        json={
            "groups": {"A组": ["RB0", "IF0"], "B组": ["RB0"]},
            "active_group": "B组",
        },
    )
    assert imported.status_code == 200
    out = imported.json()
    assert out["ok"] is True
    assert out["active_group"] == "B组"


def test_futures_trend_backtest_api_contract(api_client: TestClient) -> None:
    client = api_client
    post_json_ok(
        client,
        "/api/futures",
        {
            "code": "RB0",
            "name": "螺纹钢主连",
            "contract_multiplier": 10.0,
            "min_price_tick": 1.0,
        },
    )
    post_json_ok(
        client,
        "/api/futures",
        {
            "code": "IF0",
            "name": "股指主连",
            "contract_multiplier": 300.0,
            "min_price_tick": 0.2,
        },
    )
    post_json_ok(client, "/api/futures/RB0/fetch", {})
    post_json_ok(client, "/api/futures/IF0/fetch", {})
    synth = client.post("/api/futures/synthesize-all")
    assert synth.status_code == 200
    assert synth.json().get("succeeded", 0) >= 1
    post_json_ok(
        client,
        "/api/futures/research/groups",
        {"name": "趋势组", "codes": ["RB0", "IF0"], "set_active": True},
    )
    st = client.put(
        "/api/futures/research/state",
        json={
            "start_date": "20240101",
            "end_date": "20241231",
            "dynamic_universe": True,
            "quick_range_key": "all",
        },
    )
    assert st.status_code == 200

    resp = client.post(
        "/api/futures/research/trend-backtest",
        json={
            "range_key": "all",
            "exec_price": "close",
            "fast_ma": 2,
            "slow_ma": 3,
            "min_points": 2,
            "cost_bps": 5.0,
            "fee_side": "two_way",
            "slippage_type": "percent",
            "slippage_value": 0.0005,
            "slippage_side": "two_way",
        },
    )
    assert resp.status_code == 200
    out = resp.json()
    assert out["ok"] is True
    assert "series" in out
    assert "strategy_nav" in out["series"]
    assert "benchmark_nav" in out["series"]
    assert "summary" in out
    assert out["meta"]["exec_price"] == "close"
    assert out["meta"]["trend_strategy"] == "ma_cross"
    assert out["meta"]["ma_type"] == "sma"
    assert out["meta"]["entry_filter_enabled"] is False
    assert out["meta"]["long_entry_filter_ma"] == 200
    assert out["meta"]["short_entry_filter_ma"] == 200
    assert out["meta"]["benchmark_price_basis"] == "close"
    assert out["meta"]["fee_side"] == "two_way"
    assert out["meta"]["slippage_type"] == "percent"
    assert out["meta"]["signal_execution_rule"] == "signal_t_execute_t_plus_1_close"
    assert out["meta"]["signal_lag_trading_days"] == 1
    tsp = out["meta"].get("trend_series_policy") or {}
    assert isinstance(tsp, dict)
    assert "benchmark" in tsp and "signals" in tsp and "execution_and_returns" in tsp
    syms = out.get("symbols") or []
    assert syms and syms[0].get("trend_resolution") == "synthetic_hfq_continuous"
    assert syms[0].get("trend_execution_adjust") == "hfq"
    assert out["meta"].get("backtest_mode") == "portfolio"
    assert out["meta"].get("position_sizing") == "equal"
    assert out["meta"].get("monthly_risk_budget_enabled") is False
    assert out["meta"].get("monthly_risk_budget_effective") is False

    resp_mon = client.post(
        "/api/futures/research/trend-backtest",
        json={
            "range_key": "all",
            "exec_price": "close",
            "fast_ma": 2,
            "slow_ma": 3,
            "min_points": 2,
            "cost_bps": 5.0,
            "fee_side": "two_way",
            "slippage_type": "percent",
            "slippage_value": 0.0005,
            "slippage_side": "two_way",
            "monthly_risk_budget_enabled": True,
            "monthly_risk_budget_pct": 0.06,
            "monthly_risk_budget_include_new_trade_risk": False,
            "atr_stop_mode": "none",
            "atr_stop_window": 14,
        },
    )
    assert resp_mon.status_code == 200
    mon_j = resp_mon.json()
    assert mon_j["ok"] is True
    assert mon_j["meta"]["monthly_risk_budget_effective"] is True
    port_m = mon_j["meta"].get("portfolio") or {}
    gate = port_m.get("monthly_risk_budget_gate") or {}
    assert gate.get("enabled") is True
    atr_gate_cfg = port_m.get("monthly_risk_budget_atr_stop") or {}
    assert atr_gate_cfg.get("fallback_position_risk") == 0.01
    assert mon_j["meta"].get("atr_stop_reentry_mode") == "reenter"

    resp_single = client.post(
        "/api/futures/research/trend-backtest",
        json={
            "range_key": "all",
            "exec_price": "close",
            "fast_ma": 2,
            "slow_ma": 3,
            "min_points": 2,
            "cost_bps": 5.0,
            "fee_side": "two_way",
            "slippage_type": "percent",
            "slippage_value": 0.0005,
            "slippage_side": "two_way",
            "backtest_mode": "single",
            "single_code": "RB0",
            "entry_filter_enabled": True,
            "long_entry_filter_ma": 150,
            "short_entry_filter_ma": 180,
        },
    )
    assert resp_single.status_code == 200
    one = resp_single.json()
    assert one["ok"] is True
    assert one["meta"]["backtest_mode"] == "single"
    assert one["meta"]["single_code"] == "RB0"
    assert one["meta"]["entry_filter_enabled"] is True
    assert one["meta"]["long_entry_filter_ma"] == 150
    assert one["meta"]["short_entry_filter_ma"] == 180
    assert one["meta"]["position_sizing"] is None
    assert len(one.get("symbols") or []) == 1
    assert one["meta"].get("monthly_risk_budget_effective") is False

    resp_open = client.post(
        "/api/futures/research/trend-backtest",
        json={
            "range_key": "all",
            "exec_price": "open",
            "fast_ma": 2,
            "slow_ma": 3,
            "min_points": 2,
            "cost_bps": 5.0,
            "fee_side": "two_way",
            "slippage_type": "percent",
            "slippage_value": 0.0005,
            "slippage_side": "two_way",
        },
    )
    assert resp_open.status_code == 200
    out_open = resp_open.json()
    assert out_open["ok"] is True
    assert out_open["meta"]["exec_price"] == "open"
    assert out_open["meta"]["benchmark_price_basis"] == "open"

    resp_filter = client.post(
        "/api/futures/research/trend-backtest",
        json={
            "range_key": "all",
            "exec_price": "close",
            "trend_strategy": "ma_filter",
            "ma_type": "kama",
            "trade_direction": "both",
            "kama_er_window": 10,
            "kama_fast_window": 2,
            "kama_slow_window": 30,
            "kama_std_window": 20,
            "kama_std_coef": 1.0,
            "min_points": 2,
        },
    )
    assert resp_filter.status_code == 200
    filter_out = resp_filter.json()
    assert filter_out["ok"] is True
    assert filter_out["meta"]["trend_strategy"] == "ma_filter"
    assert filter_out["meta"]["ma_type"] == "kama"
    assert filter_out["meta"]["trade_direction"] == "both"
    assert filter_out["meta"]["kama_er_window"] == 10
    assert filter_out["meta"]["kama_fast_window"] == 2
    assert filter_out["meta"]["kama_slow_window"] == 30
    assert filter_out["meta"]["kama_std_window"] == 20
    assert abs(float(filter_out["meta"]["kama_std_coef"]) - 1.0) < 1e-12


def test_futures_trend_backtest_requires_synthetic_hfq(api_client: TestClient) -> None:
    """Without /api/futures/synthesize-all, 889 hfq rows are absent and backtest errors."""
    client = api_client
    post_json_ok(
        client,
        "/api/futures",
        {
            "code": "RB0",
            "name": "螺纹钢主连",
            "contract_multiplier": 10.0,
            "min_price_tick": 1.0,
        },
    )
    post_json_ok(client, "/api/futures/RB0/fetch", {})
    post_json_ok(
        client,
        "/api/futures/research/groups",
        {"name": "仅主连组", "codes": ["RB0"], "set_active": True},
    )
    client.put(
        "/api/futures/research/state",
        json={
            "start_date": "20240101",
            "end_date": "20241231",
            "dynamic_universe": True,
            "quick_range_key": "all",
        },
    )
    resp = client.post(
        "/api/futures/research/trend-backtest",
        json={
            "range_key": "all",
            "exec_price": "close",
            "fast_ma": 2,
            "slow_ma": 3,
            "min_points": 2,
            "cost_bps": 5.0,
            "fee_side": "two_way",
            "slippage_type": "percent",
            "slippage_value": 0.0005,
            "slippage_side": "two_way",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["error"] == "missing_synthetic_hfq_continuous"
    errs = (body.get("meta") or {}).get("errors") or []
    assert errs and any(
        "missing synthetic hfq continuous series" in str(e) for e in errs
    )


def test_futures_trend_backtest_rejects_invalid_position_sizing(
    api_client: TestClient,
) -> None:
    client = api_client
    post_json_ok(
        client,
        "/api/futures",
        {
            "code": "RB0",
            "name": "螺纹钢主连",
            "contract_multiplier": 10.0,
            "min_price_tick": 1.0,
        },
    )
    post_json_ok(client, "/api/futures/RB0/fetch", {})
    post_json_ok(
        client,
        "/api/futures/research/groups",
        {"name": "趋势组", "codes": ["RB0"], "set_active": True},
    )
    bad_ps = client.post(
        "/api/futures/research/trend-backtest",
        json={
            "range_key": "all",
            "exec_price": "close",
            "fast_ma": 2,
            "slow_ma": 3,
            "min_points": 2,
            "backtest_mode": "portfolio",
            "position_sizing": "vol_target",
        },
    )
    assert bad_ps.status_code == 200
    assert bad_ps.json()["ok"] is False
    assert bad_ps.json()["error"] == "invalid_position_sizing"


def test_futures_trend_backtest_rejects_invalid_semantics(
    api_client: TestClient,
) -> None:
    client = api_client
    post_json_ok(
        client,
        "/api/futures",
        {
            "code": "RB0",
            "name": "螺纹钢主连",
            "contract_multiplier": 10.0,
            "min_price_tick": 1.0,
        },
    )
    post_json_ok(client, "/api/futures/RB0/fetch", {})
    post_json_ok(
        client,
        "/api/futures/research/groups",
        {"name": "趋势组", "codes": ["RB0"], "set_active": True},
    )

    bad_exec = client.post(
        "/api/futures/research/trend-backtest",
        json={
            "range_key": "all",
            "exec_price": "oc2",
            "fast_ma": 2,
            "slow_ma": 3,
            "min_points": 2,
        },
    )
    assert bad_exec.status_code == 200
    assert bad_exec.json()["ok"] is False
    assert bad_exec.json()["error"] == "invalid_exec_price"

    bad_fee = client.post(
        "/api/futures/research/trend-backtest",
        json={
            "range_key": "all",
            "exec_price": "close",
            "fast_ma": 2,
            "slow_ma": 3,
            "min_points": 2,
            "fee_side": "x",
        },
    )
    assert bad_fee.status_code == 200
    assert bad_fee.json()["ok"] is False
    assert bad_fee.json()["error"] == "invalid_fee_side"

    bad_ma = client.post(
        "/api/futures/research/trend-backtest",
        json={
            "range_key": "all",
            "exec_price": "close",
            "fast_ma": 2,
            "slow_ma": 3,
            "min_points": 2,
            "ma_type": "kama",
        },
    )
    assert bad_ma.status_code == 200
    assert bad_ma.json()["ok"] is False
    assert bad_ma.json()["error"] == "invalid_ma_type"

    bad_ts = client.post(
        "/api/futures/research/trend-backtest",
        json={
            "range_key": "all",
            "exec_price": "close",
            "fast_ma": 2,
            "slow_ma": 3,
            "min_points": 2,
            "trend_strategy": "donchian",
        },
    )
    assert bad_ts.status_code == 200
    assert bad_ts.json()["ok"] is False
    assert bad_ts.json()["error"] == "unsupported_trend_strategy"

    bad_filter_ma = client.post(
        "/api/futures/research/trend-backtest",
        json={
            "range_key": "all",
            "exec_price": "close",
            "trend_strategy": "ma_filter",
            "ma_type": "sma",
            "fast_ma": 2,
            "slow_ma": 3,
            "min_points": 2,
        },
    )
    assert bad_filter_ma.status_code == 200
    assert bad_filter_ma.json()["ok"] is False
    assert bad_filter_ma.json()["error"] == "ma_filter_requires_kama"


def test_futures_trend_backtest_risk_budget_accepts_both_direction(
    api_client: TestClient,
) -> None:
    client = api_client
    post_json_ok(
        client,
        "/api/futures",
        {
            "code": "RB0",
            "name": "螺纹钢主连",
            "contract_multiplier": 10.0,
            "min_price_tick": 1.0,
        },
    )
    post_json_ok(client, "/api/futures/RB0/fetch", {})
    synth = client.post("/api/futures/synthesize-all")
    assert synth.status_code == 200
    post_json_ok(
        client,
        "/api/futures/research/groups",
        {"name": "趋势组", "codes": ["RB0"], "set_active": True},
    )
    resp = client.post(
        "/api/futures/research/trend-backtest",
        json={
            "range_key": "all",
            "exec_price": "close",
            "fast_ma": 2,
            "slow_ma": 3,
            "min_points": 2,
            "position_sizing": "risk_budget",
            "trade_direction": "both",
            "risk_budget_pct": 0.01,
            "risk_budget_atr_window": 20,
        },
    )
    assert resp.status_code == 200
    out = resp.json()
    assert out["ok"] is True
    assert out["meta"]["position_sizing"] == "risk_budget"
    assert out["meta"]["trade_direction"] == "both"
