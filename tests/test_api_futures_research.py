from __future__ import annotations

from fastapi.testclient import TestClient

from tests.helpers.rotation_case_data import post_json_ok


def test_futures_research_groups_state_and_correlation(api_client: TestClient) -> None:
    client = api_client

    # Prepare pool and prices
    post_json_ok(client, "/api/futures", {"code": "RB0", "name": "螺纹钢主连"})
    post_json_ok(client, "/api/futures", {"code": "IF0", "name": "股指主连"})
    post_json_ok(client, "/api/futures/RB0/fetch", {})
    post_json_ok(client, "/api/futures/IF0/fetch", {})

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

    cov = client.post("/api/futures/research/coverage-summary", json={"range_key": "all"})
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


def test_futures_research_groups_import_export_overwrite(api_client: TestClient) -> None:
    client = api_client
    post_json_ok(client, "/api/futures", {"code": "RB0", "name": "螺纹钢主连"})
    post_json_ok(client, "/api/futures", {"code": "IF0", "name": "股指主连"})

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
    post_json_ok(client, "/api/futures", {"code": "RB0", "name": "螺纹钢主连"})
    post_json_ok(client, "/api/futures", {"code": "IF0", "name": "股指主连"})
    post_json_ok(client, "/api/futures/RB0/fetch", {})
    post_json_ok(client, "/api/futures/IF0/fetch", {})
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
    assert out["meta"]["benchmark_price_basis"] == "close"
    assert out["meta"]["fee_side"] == "two_way"
    assert out["meta"]["slippage_type"] == "percent"
    assert out["meta"]["signal_execution_rule"] == "signal_t_execute_t_plus_1_close"
    assert out["meta"]["signal_lag_trading_days"] == 1

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


def test_futures_trend_backtest_rejects_invalid_semantics(api_client: TestClient) -> None:
    client = api_client
    post_json_ok(client, "/api/futures", {"code": "RB0", "name": "螺纹钢主连"})
    post_json_ok(client, "/api/futures/RB0/fetch", {})
    post_json_ok(
        client,
        "/api/futures/research/groups",
        {"name": "趋势组", "codes": ["RB0"], "set_active": True},
    )

    bad_exec = client.post(
        "/api/futures/research/trend-backtest",
        json={"range_key": "all", "exec_price": "oc2", "fast_ma": 2, "slow_ma": 3, "min_points": 2},
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
