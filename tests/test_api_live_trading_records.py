from __future__ import annotations

import datetime as dt

from etf_momentum.db.models import EtfPrice, LiveHoldingSnapshot, LiveNavDaily
from tests.helpers.rotation_case_data import get_json, post_json, post_response


def _assert_scope_financial_audit(c, *, scope_type: str, scope_id: int) -> None:
    perf = get_json(
        c,
        f"/api/live/performance?scope_type={scope_type}&scope_id={scope_id}&return_basis=both",
    )
    nav = perf.get("nav", [])
    for i, row in enumerate(nav):
        daily = float(row.get("daily_return_twr") or 0.0)
        rebuild = (
            float(row.get("selection_return") or 0.0)
            + float(row.get("timing_return") or 0.0)
            + float(row.get("position_return") or 0.0)
            + float(row.get("cost_drag_return") or 0.0)
            + float(row.get("cash_drag_return") or 0.0)
            + float(row.get("repo_carry_return") or 0.0)
            + float(row.get("repo_fee_drag_return") or 0.0)
        )
        assert abs(daily - rebuild) < 1e-6
        if i > 0:
            prev = nav[i - 1]
            lhs = float(row["nav_twr"])
            rhs = float(prev["nav_twr"]) * (1.0 + daily)
            assert abs(lhs - rhs) < 5e-6

    attr = get_json(
        c, f"/api/live/attribution?scope_type={scope_type}&scope_id={scope_id}"
    )
    period = attr.get("period", {})
    assert (
        abs(
            float(period.get("rebuild_total") or 0.0)
            - float(period.get("total_return_twr_sum") or 0.0)
        )
        < 1e-6
    )

    holdings = get_json(
        c, f"/api/live/holdings?scope_type={scope_type}&scope_id={scope_id}"
    )
    if holdings and nav:
        latest_nav_day = str(nav[-1]["nav_date"])
        assert all(str(x["snapshot_date"]) == latest_nav_day for x in holdings)


def _seed_live_prices(session_factory) -> None:
    dates = [
        d.date()
        for d in __import__("pandas").date_range("2024-06-20", periods=8, freq="B")
    ]
    series = {
        "159915": [4.00, 4.10, 4.20, 4.30, 4.40, 4.50, 4.60, 4.70],
        "159916": [8.00, 8.05, 8.10, 8.20, 8.30, 8.40, 8.50, 8.60],
    }
    with session_factory() as db:
        for code, vals in series.items():
            for d, px in zip(dates, vals):
                for adj in ("none", "hfq", "qfq"):
                    db.add(
                        EtfPrice(
                            code=code,
                            trade_date=d,
                            open=float(px),
                            high=float(px),
                            low=float(px),
                            close=float(px),
                            volume=1000.0,
                            amount=float(px) * 1000.0,
                            source="seed",
                            adjust=adj,
                        )
                    )
        db.commit()


def test_live_trading_records_contract(api_client, session_factory):
    _seed_live_prices(session_factory)
    c = api_client

    # account / strategy / shareholder
    acc = post_json(c, "/api/live/accounts", {"name": "实盘A", "initial_cash": 0})
    aid = int(acc["id"])
    st = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略1"})
    sid = int(st["id"])
    holder = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "A123456789"},
    )
    hid = int(holder["id"])

    # external account cashflow + internal transfer (must not pollute account external flow)
    post_json(
        c,
        f"/api/live/accounts/{aid}/cashflows",
        {"flow_date": "20240620", "amount": 100000, "flow_type": "deposit"},
    )
    tf = post_json(
        c,
        f"/api/live/accounts/{aid}/strategy-transfer",
        {
            "strategy_id": sid,
            "flow_date": "20240621",
            "amount": 60000,
            "direction": "to_strategy",
        },
    )
    assert tf["ok"] is True

    # quantity required
    r_missing_qty = post_response(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240621",
            "trade_time": "09:31:00",
            "side": "BUY",
            "price": 4.10,
            "fee": 3.0,
        },
        expected_status=422,
    )
    assert r_missing_qty.status_code == 422

    # quantity must be a 100-lot multiple
    r_bad_qty = post_response(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240621",
            "trade_time": "09:31:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 150,
            "fee": 0.2,
            "idempotency_key": "bad_qty_guard",
        },
        expected_status=400,
    )
    assert "multiple of 100" in r_bad_qty.json().get("detail", "")

    # trade time guardrail
    r_bad_time = post_response(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240621",
            "trade_time": "08:59:59",
            "side": "BUY",
            "price": 4.10,
            "quantity": 100,
            "fee": 0.2,
            "idempotency_key": "bad_time_guard",
        },
        expected_status=400,
    )
    assert "between 09:00:00 and 14:59:59" in r_bad_time.json().get("detail", "")

    # fifo trades (2 buys + 2 sells -> 1 closed round)
    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915.SZ",
            "name": "创业板ETF",
            "trade_date": "20240621",
            "trade_time": "09:31:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 1000,
            "fee": 10,
            "idempotency_key": "k1",
        },
    )
    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240624",
            "trade_time": "09:32:00",
            "side": "BUY",
            "price": 4.20,
            "quantity": 500,
            "fee": 5,
            "idempotency_key": "k2",
        },
    )
    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240625",
            "trade_time": "10:01:00",
            "side": "SELL",
            "price": 4.50,
            "quantity": 1200,
            "fee": 12,
            "idempotency_key": "k3",
        },
    )
    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240626",
            "trade_time": "10:15:00",
            "side": "SELL",
            "price": 4.60,
            "quantity": 300,
            "fee": 3,
            "idempotency_key": "k4",
        },
    )

    # oversell must fail under long-only rule
    r_oversell = post_response(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240627",
            "trade_time": "10:20:00",
            "side": "SELL",
            "price": 4.60,
            "quantity": 100,
            "fee": 0,
            "idempotency_key": "k5",
        },
        expected_status=400,
    )
    assert "exceeds position" in r_oversell.json().get("detail", "")

    # idempotency duplicate must fail
    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240627",
            "trade_time": "09:40:00",
            "side": "BUY",
            "price": 4.60,
            "quantity": 100,
            "fee": 1,
            "idempotency_key": "dup_key",
        },
    )
    r_dup = post_response(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240627",
            "trade_time": "09:41:00",
            "side": "BUY",
            "price": 4.60,
            "quantity": 100,
            "fee": 1,
            "idempotency_key": "dup_key",
        },
        expected_status=400,
    )
    assert "duplicate" in r_dup.json().get("detail", "")

    # corporate actions: split then code change
    post_json(
        c,
        "/api/live/corporate-actions",
        {
            "account_id": aid,
            "strategy_id": sid,
            "event_type": "split",
            "code": "159915",
            "event_date": "20240627",
            "effective_date": "20240627",
            "ratio_factor": 2.0,
        },
    )
    post_json(
        c,
        "/api/live/corporate-actions",
        {
            "account_id": aid,
            "strategy_id": sid,
            "event_type": "code_change",
            "code": "159915",
            "new_code": "159916",
            "event_date": "20240628",
            "effective_date": "20240628",
        },
    )

    holdings = get_json(c, f"/api/live/holdings?scope_type=strategy&scope_id={sid}")
    assert len(holdings) >= 1
    assert any(x["code"] == "159916" for x in holdings)

    rounds = get_json(
        c,
        f"/api/live/closed-rounds?scope_type=strategy&scope_id={sid}&page=1&page_size=50",
    )
    assert rounds["total"] >= 1
    first = rounds["items"][0]
    assert first["buy_count"] >= 2
    assert first["sell_count"] >= 2

    fee_stats = get_json(c, f"/api/live/stats/fees?scope_type=strategy&scope_id={sid}")
    assert fee_stats["total_fee"] > 0
    assert fee_stats["buy_fee"] > 0
    assert fee_stats["sell_fee"] > 0
    closed_fee_stats = get_json(
        c, f"/api/live/stats/fees/closed-rounds?scope_type=strategy&scope_id={sid}"
    )
    rounds_total_fee = sum(float(x.get("total_fee") or 0.0) for x in rounds["items"])
    assert abs(float(closed_fee_stats["total_fee"]) - rounds_total_fee) < 1e-9
    # This scenario keeps one extra open BUY trade fee outside closed rounds.
    assert float(closed_fee_stats["total_fee"]) < float(fee_stats["total_fee"])

    perf = get_json(
        c, f"/api/live/performance?scope_type=strategy&scope_id={sid}&return_basis=both"
    )
    assert len(perf["nav"]) > 0
    assert "annualized_return" in perf["dietz_basis_metrics"]
    assert "annualized_return" in perf["twr_basis_metrics"]

    attr = get_json(c, f"/api/live/attribution?scope_type=strategy&scope_id={sid}")
    assert "period" in attr
    period = attr["period"]
    assert (
        abs(float(period["rebuild_total"]) - float(period["total_return_twr_sum"]))
        < 1e-6
    )

    # account-level external flow should not be polluted by transfer rows.
    perf_acc = get_json(
        c, f"/api/live/performance?scope_type=account&scope_id={aid}&return_basis=both"
    )
    nav_rows = perf_acc["nav"]
    row_20240621 = [
        x for x in nav_rows if x["nav_date"] == dt.date(2024, 6, 21).isoformat()
    ]
    if row_20240621:
        assert abs(float(row_20240621[0]["external_flow"])) < 1e-9


def test_live_replay_account_refreshes_strategy_scope(api_client, session_factory):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(c, "/api/live/accounts", {"name": "实盘Replay", "initial_cash": 0})
    aid = int(acc["id"])
    st = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略Replay"})
    sid = int(st["id"])
    holder = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "REPLAY001"},
    )
    hid = int(holder["id"])

    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240621",
            "trade_time": "10:00:00",
            "side": "BUY",
            "price": 4.00,
            "quantity": 100,
            "fee": 1.0,
            "idempotency_key": "replay-seed-1",
        },
    )
    hs_before = get_json(c, f"/api/live/holdings?scope_type=strategy&scope_id={sid}")
    assert len(hs_before) > 0

    with session_factory() as db:
        (
            db.query(LiveHoldingSnapshot)
            .filter(
                LiveHoldingSnapshot.scope_type == "strategy",
                LiveHoldingSnapshot.scope_id == sid,
            )
            .delete(synchronize_session=False)
        )
        (
            db.query(LiveNavDaily)
            .filter(LiveNavDaily.scope_type == "strategy", LiveNavDaily.scope_id == sid)
            .delete(synchronize_session=False)
        )
        db.commit()

    out = post_json(c, "/api/live/replay", {"account_id": aid})
    assert int(out.get("strategies_replayed", 0)) >= 1

    hs_after = get_json(c, f"/api/live/holdings?scope_type=strategy&scope_id={sid}")
    assert len(hs_after) > 0


def test_live_trading_shareholder_isolated_fifo(api_client, session_factory):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(c, "/api/live/accounts", {"name": "实盘B", "initial_cash": 0})
    aid = int(acc["id"])
    st = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略H"})
    sid = int(st["id"])
    h1 = int(
        post_json(
            c,
            f"/api/live/accounts/{aid}/shareholders",
            {"shareholder_account": "HOLDER_A"},
        )["id"]
    )
    h2 = int(
        post_json(
            c,
            f"/api/live/accounts/{aid}/shareholders",
            {"shareholder_account": "HOLDER_B"},
        )["id"]
    )

    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": h1,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240621",
            "trade_time": "09:31:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 100,
            "fee": 1,
            "idempotency_key": "h1-buy-1",
        },
    )
    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": h2,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240621",
            "trade_time": "09:32:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 200,
            "fee": 2,
            "idempotency_key": "h2-buy-1",
        },
    )

    # Holder-level matching: H1 cannot sell beyond H1's own position,
    # even when other holders in the same strategy still hold quantity.
    r_cross_holder_oversell = post_response(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": h1,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240624",
            "trade_time": "10:01:00",
            "side": "SELL",
            "price": 4.20,
            "quantity": 200,
            "fee": 1,
            "idempotency_key": "h1-sell-oversize",
        },
        expected_status=400,
    )
    assert "shareholder_account_id" in r_cross_holder_oversell.json().get("detail", "")

    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": h1,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240624",
            "trade_time": "10:05:00",
            "side": "SELL",
            "price": 4.20,
            "quantity": 100,
            "fee": 1,
            "idempotency_key": "h1-sell-1",
        },
    )

    rounds1 = get_json(
        c,
        f"/api/live/closed-rounds?scope_type=strategy&scope_id={sid}&page=1&page_size=50",
    )
    assert rounds1["total"] == 1
    assert rounds1["items"][0]["buy_count"] == 1
    assert rounds1["items"][0]["sell_count"] == 1

    holdings = get_json(c, f"/api/live/holdings?scope_type=strategy&scope_id={sid}")
    row = next(x for x in holdings if x["code"] == "159915")
    assert abs(float(row["quantity"]) - 200.0) < 1e-9

    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": h2,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240625",
            "trade_time": "10:10:00",
            "side": "SELL",
            "price": 4.30,
            "quantity": 200,
            "fee": 2,
            "idempotency_key": "h2-sell-1",
        },
    )

    rounds2 = get_json(
        c,
        f"/api/live/closed-rounds?scope_type=strategy&scope_id={sid}&page=1&page_size=50",
    )
    assert rounds2["total"] == 2


def test_live_trade_fee_default_rule(api_client, session_factory):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(c, "/api/live/accounts", {"name": "实盘C", "initial_cash": 0})
    aid = int(acc["id"])
    st = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略Fee"})
    sid = int(st["id"])
    holder = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "FEE001"},
    )
    hid = int(holder["id"])

    # fee omitted -> payload default 0 -> server should apply max(amount*1e-4, 0.2)
    # amount = 4.10 * 100 = 410 => 0.041 < 0.2, so default fee should be 0.2
    t1 = post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240621",
            "trade_time": "14:56:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 100,
            "idempotency_key": "fee-default-min",
        },
    )
    assert abs(float(t1["fee"]) - 0.2) < 1e-9

    # default fee uses round-half-up to 2 decimals
    # amount = 3.005 * 10000 = 30050 => fee_raw = 3.005 -> 3.01
    t1b = post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240624",
            "trade_time": "14:56:00",
            "side": "BUY",
            "price": 3.005,
            "quantity": 10000,
            "idempotency_key": "fee-default-round-half-up",
        },
    )
    assert abs(float(t1b["fee"]) - 3.01) < 1e-9

    # explicit positive fee should be respected, not overwritten by default rule
    t2 = post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240624",
            "trade_time": "14:56:00",
            "side": "BUY",
            "price": 4.20,
            "quantity": 100,
            "fee": 9.999,
            "idempotency_key": "fee-explicit-keep",
        },
    )
    assert abs(float(t2["fee"]) - 10.0) < 1e-9


def test_live_trade_update_delete_require_reason(api_client, session_factory):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(c, "/api/live/accounts", {"name": "实盘D", "initial_cash": 0})
    aid = int(acc["id"])
    st = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略Edit"})
    sid = int(st["id"])
    holder = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "EDIT001"},
    )
    hid = int(holder["id"])

    trade = post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240621",
            "trade_time": "14:56:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 100,
            "fee": 0.2,
            "idempotency_key": "edit-case-1",
        },
    )
    tid = int(trade["id"])

    update_payload = {
        "account_id": aid,
        "strategy_id": sid,
        "shareholder_account_id": hid,
        "code": "159915",
        "name": "创业板ETF",
        "trade_date": "20240621",
        "trade_time": "14:56:00",
        "side": "BUY",
        "price": 4.10,
        "quantity": 200,
        "fee": 0.5,
        "broker_trade_no": "BRK-EDIT-1",
    }
    r_update_no_reason = c.put(f"/api/live/trades/{tid}", json=update_payload)
    assert r_update_no_reason.status_code == 422

    r_update_blank_reason = c.put(
        f"/api/live/trades/{tid}",
        json={**update_payload, "reason": "   "},
    )
    assert r_update_blank_reason.status_code == 400
    assert "reason is required" in r_update_blank_reason.json().get("detail", "")

    r_update_ok = c.put(
        f"/api/live/trades/{tid}",
        json={**update_payload, "reason": "修正数量录入错误"},
    )
    assert r_update_ok.status_code == 200
    updated = r_update_ok.json()
    assert abs(float(updated["quantity"]) - 200.0) < 1e-9
    assert abs(float(updated["fee"]) - 0.5) < 1e-9
    assert updated["broker_trade_no"] == "BRK-EDIT-1"

    holdings_after_update = get_json(
        c, f"/api/live/holdings?scope_type=strategy&scope_id={sid}"
    )
    row = next(x for x in holdings_after_update if x["code"] == "159915")
    assert abs(float(row["quantity"]) - 200.0) < 1e-9

    r_delete_no_reason = c.request("DELETE", f"/api/live/trades/{tid}", json={})
    assert r_delete_no_reason.status_code == 422

    r_delete_blank_reason = c.request(
        "DELETE",
        f"/api/live/trades/{tid}",
        json={"reason": "   "},
    )
    assert r_delete_blank_reason.status_code == 400
    assert "reason is required" in r_delete_blank_reason.json().get("detail", "")

    r_delete_ok = c.request(
        "DELETE",
        f"/api/live/trades/{tid}",
        json={"reason": "误录，整单撤销"},
    )
    assert r_delete_ok.status_code == 200
    assert r_delete_ok.json().get("ok") is True

    trades_after_delete = get_json(
        c, f"/api/live/trades?strategy_id={sid}&page=1&page_size=10"
    )
    assert int(trades_after_delete["total"]) == 0


def test_live_trade_pnl_includes_fees_without_cost_price_allocation(
    api_client, session_factory
):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(c, "/api/live/accounts", {"name": "实盘E", "initial_cash": 0})
    aid = int(acc["id"])
    st = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略PnL"})
    sid = int(st["id"])
    holder = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "PNL001"},
    )
    hid = int(holder["id"])

    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240621",
            "trade_time": "10:00:00",
            "side": "BUY",
            "price": 4.00,
            "quantity": 100,
            "fee": 10.0,
            "idempotency_key": "pnl-fee-buy",
        },
    )

    holdings = get_json(c, f"/api/live/holdings?scope_type=strategy&scope_id={sid}")
    row = next(x for x in holdings if x["code"] == "159915")
    # Cost basis keeps execution price (fees are not allocated into cost_price).
    assert abs(float(row["cost_price"]) - 4.00) < 1e-9
    assert abs(float(row["cost_value"]) - 400.0) < 1e-9
    # Holding endpoint returns latest snapshot (seed latest close=4.70):
    # 4.70*100 - 4.00*100 - 10 = 60.
    assert abs(float(row["pnl_amount"]) - 60.0) < 1e-9

    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240624",
            "trade_time": "10:05:00",
            "side": "SELL",
            "price": 4.50,
            "quantity": 100,
            "fee": 10.0,
            "idempotency_key": "pnl-fee-sell",
        },
    )

    rounds = get_json(
        c,
        f"/api/live/closed-rounds?scope_type=strategy&scope_id={sid}&page=1&page_size=20",
    )
    assert int(rounds["total"]) == 1
    r0 = rounds["items"][0]
    # Closed-round PnL includes both buy and sell fees:
    # (4.50 - 4.00) * 100 - 10 - 10 = 30.
    assert abs(float(r0["realized_pnl"]) - 30.0) < 1e-9
    # Fee is still tracked separately.
    assert abs(float(r0["total_fee"]) - 20.0) < 1e-9


def test_live_bond_repo_round_contract(api_client, session_factory):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(
        c, "/api/live/accounts", {"name": "实盘Repo", "initial_cash": 100000}
    )
    aid = int(acc["id"])
    st = post_json(
        c,
        f"/api/live/accounts/{aid}/strategies",
        {"name": "国债逆回购", "strategy_type": "bond_repo"},
    )
    sid = int(st["id"])
    assert st["strategy_type"] == "bond_repo"
    assert st["capital_mode"] == "shared_account_cash"
    holder = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "REPO001"},
    )
    hid = int(holder["id"])

    bad_repo_qty = post_response(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "204001",
            "name": "GC001",
            "trade_date": "20240621",
            "trade_time": "10:00:00",
            "side": "BUY",
            "price": 2.5,
            "quantity": 1.5,
            "repo_action": "LEND",
            "repo_principal_amount": 1500.0,
            "repo_interest_days": 3,
            "idempotency_key": "repo-bad-qty",
        },
        expected_status=400,
    )
    assert "positive integer" in str(bad_repo_qty.json().get("detail", ""))

    bad_repo_lend_time = post_response(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "204001",
            "name": "GC001",
            "trade_date": "20240621",
            "trade_time": "00:00:00",
            "side": "BUY",
            "price": 1.46,
            "quantity": 90,
            "amount": 90000.0,
            "repo_action": "LEND",
            "repo_principal_amount": 90000.0,
            "repo_interest_days": 3,
            "idempotency_key": "repo-bad-lend-time",
        },
        expected_status=400,
    )
    assert "09:30:00 and 15:30:00" in str(
        bad_repo_lend_time.json().get("detail", ""),
    )

    bad_repo_lend_amount = post_response(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "204001",
            "name": "GC001",
            "trade_date": "20240621",
            "trade_time": "10:00:00",
            "side": "BUY",
            "price": 1.46,
            "quantity": 90,
            "amount": 90010.8,
            "repo_action": "LEND",
            "repo_principal_amount": 90000.0,
            "repo_interest_days": 3,
            "idempotency_key": "repo-bad-lend-amount",
        },
        expected_status=400,
    )
    assert "LEND amount must equal quantity * 1000" in str(
        bad_repo_lend_amount.json().get("detail", ""),
    )

    open_trade = post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "204001",
            "name": "国债逆回购",
            "trade_date": "20240621",
            "trade_time": "15:30:00",
            "side": "BUY",
            "price": 1.46,
            "quantity": 90,
            "amount": 90000.0,
            "repo_action": "LEND",
            "repo_principal_amount": 90000.0,
            "repo_interest_days": 3,
            "idempotency_key": "repo-open-1",
        },
    )
    assert open_trade["repo_action"] == "LEND"
    assert abs(float(open_trade["repo_principal_amount"]) - 90000.0) < 1e-9
    assert int(open_trade["repo_interest_days"]) == 3
    assert open_trade["name"] == "GC001"
    assert abs(float(open_trade["fee"]) - 0.9) < 1e-9

    holdings = get_json(c, f"/api/live/holdings?scope_type=strategy&scope_id={sid}")
    h0 = next(x for x in holdings if str(x.get("code")) == "204001")
    # 90000 * 1.46% * 3 / 365 = 10.8; minus fee 0.9 => 9.9
    assert abs(float(h0["pnl_amount"]) - 9.9) < 1e-6

    close_trade = post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "204001",
            "name": "国债逆回购",
            "trade_date": "20240624",
            "trade_time": "00:00:00",
            "side": "SELL",
            "price": 1.46,
            "quantity": 90,
            "amount": 90010.8,
            "repo_action": "BUYBACK",
            "repo_principal_amount": 90000.0,
            "idempotency_key": "repo-close-1",
        },
    )
    assert close_trade["repo_action"] == "BUYBACK"
    assert close_trade["repo_open_trade_id"] is None
    assert abs(float(close_trade["fee"]) - 0.0) < 1e-9
    assert abs(float(close_trade["amount"]) - 90010.8) < 1e-9

    rounds = get_json(
        c,
        f"/api/live/closed-rounds?scope_type=strategy&scope_id={sid}&page=1&page_size=20",
    )
    assert int(rounds["total"]) == 1
    r0 = rounds["items"][0]
    assert abs(float(r0["realized_pnl"]) - 9.9) < 1e-6
    assert abs(float(r0["total_fee"]) - 0.9) < 1e-9

    attr = get_json(c, f"/api/live/attribution?scope_type=strategy&scope_id={sid}")
    period = attr["period"]
    assert "repo_carry_return" in period
    assert "repo_fee_drag_return" in period
    assert isinstance(float(period["repo_carry_return"]), float)
    assert isinstance(float(period["repo_fee_drag_return"]), float)
    assert (
        abs(float(period["rebuild_total"]) - float(period["total_return_twr_sum"]))
        < 1e-6
    )

    trades = get_json(c, f"/api/live/trades?strategy_id={sid}&page=1&page_size=20")
    assert int(trades["total"]) == 2
    assert any(x.get("repo_action") == "LEND" for x in trades["items"])
    assert any(x.get("repo_action") == "BUYBACK" for x in trades["items"])
    _assert_scope_financial_audit(c, scope_type="strategy", scope_id=sid)


def test_live_holdings_align_to_latest_nav_day(api_client, session_factory):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(
        c, "/api/live/accounts", {"name": "实盘持仓对齐", "initial_cash": 100000}
    )
    aid = int(acc["id"])
    st = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略持仓对齐"})
    sid = int(st["id"])
    holder = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "ALIGN001"},
    )
    hid = int(holder["id"])

    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240621",
            "trade_time": "10:00:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 100,
            "fee": 0.2,
            "idempotency_key": "align-buy",
        },
    )
    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240624",
            "trade_time": "10:00:00",
            "side": "SELL",
            "price": 4.20,
            "quantity": 100,
            "fee": 0.2,
            "idempotency_key": "align-sell",
        },
    )

    perf = get_json(
        c, f"/api/live/performance?scope_type=strategy&scope_id={sid}&return_basis=both"
    )
    assert len(perf.get("nav", [])) > 0
    holdings = get_json(c, f"/api/live/holdings?scope_type=strategy&scope_id={sid}")
    # Latest nav day has no open position, so holdings must not return stale rows.
    assert holdings == []
    _assert_scope_financial_audit(c, scope_type="strategy", scope_id=sid)


def test_live_trade_allows_when_strategy_allocation_insufficient_but_account_ok(
    api_client, session_factory
):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(
        c, "/api/live/accounts", {"name": "实盘资金校验S", "initial_cash": 100000}
    )
    aid = int(acc["id"])
    st = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略资金校验S"})
    sid = int(st["id"])
    holder = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "CASHCHK-S"},
    )
    hid = int(holder["id"])

    post_json(
        c,
        f"/api/live/accounts/{aid}/strategy-transfer",
        {
            "strategy_id": sid,
            "flow_date": "20240620",
            "amount": 1000,
            "direction": "to_strategy",
        },
    )

    t = post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240621",
            "trade_time": "10:00:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 300,  # amount=1230 > transferred 1000, but account cash is enough
            "fee": 0.2,
            "idempotency_key": "cash-check-strategy-insufficient",
        },
    )
    assert int(t["strategy_id"]) == sid


def test_live_trade_allows_update_when_switching_to_insufficient_strategy(
    api_client, session_factory
):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(
        c, "/api/live/accounts", {"name": "实盘资金校验U", "initial_cash": 100000}
    )
    aid = int(acc["id"])
    st_a = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略A"})
    st_b = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略B"})
    sid_a = int(st_a["id"])
    sid_b = int(st_b["id"])
    holder = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "CASHCHK-U"},
    )
    hid = int(holder["id"])

    post_json(
        c,
        f"/api/live/accounts/{aid}/strategy-transfer",
        {
            "strategy_id": sid_a,
            "flow_date": "20240620",
            "amount": 3000,
            "direction": "to_strategy",
        },
    )
    post_json(
        c,
        f"/api/live/accounts/{aid}/strategy-transfer",
        {
            "strategy_id": sid_b,
            "flow_date": "20240620",
            "amount": 500,
            "direction": "to_strategy",
        },
    )

    t = post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid_a,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240621",
            "trade_time": "10:00:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 200,  # amount=820
            "fee": 0.2,
            "idempotency_key": "cash-check-update-base",
        },
    )
    tid = int(t["id"])

    r = c.put(
        f"/api/live/trades/{tid}",
        json={
            "account_id": aid,
            "strategy_id": sid_b,  # switch to low-budget strategy
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240621",
            "trade_time": "10:00:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 200,
            "fee": 0.2,
            "reason": "改策略测试资金校验",
        },
    )
    assert r.status_code == 200

    trades = get_json(c, f"/api/live/trades?account_id={aid}&page=1&page_size=20")
    only = next(x for x in trades["items"] if int(x["id"]) == tid)
    assert int(only["strategy_id"]) == sid_b


def test_live_trade_rejects_when_account_cash_insufficient_without_financing(
    api_client, session_factory
):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(
        c, "/api/live/accounts", {"name": "实盘资金校验A", "initial_cash": 1000}
    )
    aid = int(acc["id"])
    st = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略资金校验A"})
    sid = int(st["id"])
    holder = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "CASHCHK-A"},
    )
    hid = int(holder["id"])

    r = post_response(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240621",
            "trade_time": "10:00:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 300,  # amount=1230 > account initial cash 1000
            "fee": 0.2,
            "idempotency_key": "cash-check-account-insufficient",
        },
        expected_status=400,
    )
    assert "insufficient account cash" in str(r.json().get("detail", ""))
