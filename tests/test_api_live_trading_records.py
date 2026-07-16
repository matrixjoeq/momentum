from __future__ import annotations

import datetime as dt
import hashlib
import json
from decimal import ROUND_HALF_UP, Decimal

from etf_momentum.db.models import (
    EtfPrice,
    LiveHoldingSnapshot,
    LiveNavDaily,
    LiveTrade,
    LiveTradeAuditLog,
)
from etf_momentum.calendar.trading_calendar import trading_days
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


def _snapshot_sha256(payload_doc: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload_doc,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


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
    assert "between 09:00:00 and 15:00:00" in r_bad_time.json().get("detail", "")

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

    # Holdings are grouped by (code, shareholder_account_id), not merged by code only.
    hs_split = get_json(c, f"/api/live/holdings?scope_type=strategy&scope_id={sid}")
    hs_159915 = [x for x in hs_split if str(x.get("code")) == "159915"]
    assert len(hs_159915) == 2
    qty_by_holder = {
        int(x["shareholder_account_id"]): float(x["quantity"]) for x in hs_159915
    }
    assert abs(qty_by_holder[h1] - 100.0) < 1e-9
    assert abs(qty_by_holder[h2] - 200.0) < 1e-9

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
    # With cumulative-fee-inclusive basis:
    # cost_value = market_value - cumulative_pnl = 470 - 60 = 410
    # cost_price = cost_value / qty = 4.10
    assert abs(float(row["cost_price"]) - 4.10) < 1e-9
    assert abs(float(row["cost_value"]) - 410.0) < 1e-9
    # Holdings cumulative PnL includes symbol-level fees:
    # 4.70*100 - 4.00*100 - 10 = 60.
    assert abs(float(row["pnl_amount"]) - 60.0) < 1e-9
    # cumulative_return = cumulative_pnl / (market_value - cumulative_pnl)
    assert abs(float(row["pnl_rate"]) - (60.0 / 410.0)) < 1e-9

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
    # Holdings cumulative PnL includes symbol-level fees.
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


def test_live_holdings_duration_days_weighted_for_account_scope(
    api_client, session_factory
):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(
        c, "/api/live/accounts", {"name": "实盘持仓时长", "initial_cash": 100000}
    )
    aid = int(acc["id"])
    s1 = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略时长A"})
    s2 = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略时长B"})
    sid1 = int(s1["id"])
    sid2 = int(s2["id"])
    holder = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "DUR001"},
    )
    hid = int(holder["id"])

    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid1,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240621",
            "trade_time": "10:00:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 100,
            "fee": 0.2,
            "idempotency_key": "duration-s1-buy",
        },
    )
    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid2,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240624",
            "trade_time": "10:05:00",
            "side": "BUY",
            "price": 4.20,
            "quantity": 100,
            "fee": 0.2,
            "idempotency_key": "duration-s2-buy",
        },
    )

    hs_s1 = get_json(c, f"/api/live/holdings?scope_type=strategy&scope_id={sid1}")
    row_s1 = next(x for x in hs_s1 if str(x.get("code")) == "159915")
    asof = dt.date.fromisoformat(str(row_s1["snapshot_date"]))
    d1 = max(len(trading_days(dt.date(2024, 6, 21), asof)) - 1, 0)
    assert int(row_s1["holding_duration_days"]) == d1

    hs_acc = get_json(c, f"/api/live/holdings?scope_type=account&scope_id={aid}")
    row_acc = next(x for x in hs_acc if str(x.get("code")) == "159915")
    d2 = max(len(trading_days(dt.date(2024, 6, 24), asof)) - 1, 0)
    expected_weighted = int(
        Decimal(str((100.0 * d1 + 100.0 * d2) / 200.0)).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )
    assert abs(float(row_acc["quantity"]) - 200.0) < 1e-9
    assert int(row_acc["holding_duration_days"]) == expected_weighted


def test_live_closed_rounds_duration_and_stats(api_client, session_factory):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(
        c, "/api/live/accounts", {"name": "实盘平仓时长", "initial_cash": 100000}
    )
    aid = int(acc["id"])
    st = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略平仓时长"})
    sid = int(st["id"])
    holder = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "CLOSE001"},
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
            "idempotency_key": "close-duration-round1-buy",
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
            "trade_time": "10:01:00",
            "side": "SELL",
            "price": 4.50,
            "quantity": 100,
            "fee": 0.2,
            "idempotency_key": "close-duration-round1-sell",
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
            "trade_date": "20240627",
            "trade_time": "10:02:00",
            "side": "BUY",
            "price": 4.60,
            "quantity": 100,
            "fee": 0.2,
            "idempotency_key": "close-duration-round2-buy",
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
            "trade_date": "20240628",
            "trade_time": "10:03:00",
            "side": "SELL",
            "price": 4.70,
            "quantity": 100,
            "fee": 0.2,
            "idempotency_key": "close-duration-round2-sell",
        },
    )

    rounds = get_json(
        c,
        f"/api/live/closed-rounds?scope_type=strategy&scope_id={sid}&page=1&page_size=50",
    )
    assert int(rounds["total"]) == 2
    duration_days = sorted(
        int(x["holding_duration_days"])
        for x in rounds["items"]
        if x.get("holding_duration_days") is not None
    )
    expected_d1 = max(
        len(trading_days(dt.date(2024, 6, 21), dt.date(2024, 6, 26))) - 1, 0
    )
    expected_d2 = max(
        len(trading_days(dt.date(2024, 6, 27), dt.date(2024, 6, 28))) - 1, 0
    )
    assert duration_days == sorted([expected_d1, expected_d2])

    stats = get_json(
        c, f"/api/live/stats/closed-rounds?scope_type=strategy&scope_id={sid}"
    )
    assert int(stats["min_holding_duration_days"]) == min(expected_d1, expected_d2)
    assert int(stats["max_holding_duration_days"]) == max(expected_d1, expected_d2)
    expected_avg = (float(expected_d1) + float(expected_d2)) / 2.0
    expected_avg_days = int(
        Decimal(str(expected_avg)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
    assert isinstance(stats["avg_holding_duration_days"], int)
    assert int(stats["avg_holding_duration_days"]) == expected_avg_days


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


def test_live_account_snapshot_export_import_overwrite_contract(
    api_client, session_factory
):
    _seed_live_prices(session_factory)
    c = api_client
    acc = post_json(
        c, "/api/live/accounts", {"name": "导入导出账户A", "initial_cash": 100000}
    )
    aid = int(acc["id"])
    st = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略快照A"})
    sid = int(st["id"])
    holder = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "SNAP-A"},
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
            "trade_time": "09:31:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 1000,
            "fee": 2.0,
            "idempotency_key": "snapshot-base-k1",
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
            "quantity": 300,
            "fee": 1.0,
            "idempotency_key": "snapshot-base-k2",
        },
    )
    exported = get_json(c, f"/api/live/accounts/{aid}/export")
    assert exported["format"] == "etf_momentum_live_account_snapshot"
    assert int(exported["version"]) == 1
    original_total = int(
        get_json(c, f"/api/live/trades?account_id={aid}&page=1&page_size=200")["total"]
    )

    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159916",
            "name": "深证ETF",
            "trade_date": "20240625",
            "trade_time": "10:30:00",
            "side": "BUY",
            "price": 8.20,
            "quantity": 100,
            "fee": 0.3,
            "idempotency_key": "snapshot-extra-k3",
        },
    )
    assert (
        int(
            get_json(c, f"/api/live/trades?account_id={aid}&page=1&page_size=200")[
                "total"
            ]
        )
        == original_total + 1
    )

    req = {
        "replace_all": True,
        "payload": exported,
        "payload_sha256": _snapshot_sha256(exported),
        "import_request_id": "snapshot-import-req-1",
    }
    imp = post_json(c, f"/api/live/accounts/{aid}/import", req)
    assert imp["ok"] is True
    assert imp["dry_run"] is False
    assert int(imp["deleted_counts"]["trades"]) == original_total + 1
    assert int(imp["inserted_counts"]["trades"]) == original_total
    after = get_json(c, f"/api/live/trades?account_id={aid}&page=1&page_size=200")
    assert int(after["total"]) == original_total
    assert all(
        str(x.get("idempotency_key") or "") != "snapshot-extra-k3"
        for x in after["items"]
    )

    # same import_request_id must be idempotent and return previous result snapshot
    imp_retry = post_json(c, f"/api/live/accounts/{aid}/import", req)
    assert imp_retry["import_request_id"] == "snapshot-import-req-1"
    assert imp_retry["payload_sha256"] == imp["payload_sha256"]
    assert imp_retry["imported_at"] == imp["imported_at"]

    _assert_scope_financial_audit(c, scope_type="account", scope_id=aid)
    strategies_now = get_json(c, f"/api/live/accounts/{aid}/strategies")
    for row in strategies_now:
        _assert_scope_financial_audit(c, scope_type="strategy", scope_id=int(row["id"]))


def test_live_account_snapshot_import_dry_run_reports_unique_conflict(
    api_client, session_factory
):
    _seed_live_prices(session_factory)
    c = api_client

    acc_a = post_json(
        c, "/api/live/accounts", {"name": "导入冲突账户A", "initial_cash": 50000}
    )
    aid_a = int(acc_a["id"])
    st_a = post_json(c, f"/api/live/accounts/{aid_a}/strategies", {"name": "策略A"})
    sid_a = int(st_a["id"])
    h_a = post_json(
        c,
        f"/api/live/accounts/{aid_a}/shareholders",
        {"shareholder_account": "SNAP-CA"},
    )
    hid_a = int(h_a["id"])
    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid_a,
            "strategy_id": sid_a,
            "shareholder_account_id": hid_a,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240621",
            "trade_time": "09:31:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 100,
            "fee": 0.2,
            "idempotency_key": "dryrun-source-idem",
        },
    )
    exported = get_json(c, f"/api/live/accounts/{aid_a}/export")

    acc_b = post_json(
        c, "/api/live/accounts", {"name": "导入冲突账户B", "initial_cash": 50000}
    )
    aid_b = int(acc_b["id"])
    st_b = post_json(c, f"/api/live/accounts/{aid_b}/strategies", {"name": "策略B"})
    sid_b = int(st_b["id"])
    h_b = post_json(
        c,
        f"/api/live/accounts/{aid_b}/shareholders",
        {"shareholder_account": "SNAP-CB"},
    )
    hid_b = int(h_b["id"])
    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid_b,
            "strategy_id": sid_b,
            "shareholder_account_id": hid_b,
            "code": "159916",
            "name": "深证ETF",
            "trade_date": "20240621",
            "trade_time": "09:31:00",
            "side": "BUY",
            "price": 8.10,
            "quantity": 100,
            "fee": 0.2,
            "idempotency_key": "dryrun-conflict-key",
        },
    )

    assert exported["payload"]["trades"]
    exported["payload"]["trades"][0]["idempotency_key"] = "dryrun-conflict-key"
    before_total = int(
        get_json(c, f"/api/live/trades?account_id={aid_a}&page=1&page_size=100")[
            "total"
        ]
    )
    r = c.post(
        f"/api/live/accounts/{aid_a}/import",
        json={
            "replace_all": True,
            "dry_run": True,
            "payload": exported,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["dry_run"] is True
    assert any(
        str(x.get("field")) == "idempotency_key" for x in body.get("conflicts", [])
    )
    after_total = int(
        get_json(c, f"/api/live/trades?account_id={aid_a}&page=1&page_size=100")[
            "total"
        ]
    )
    assert after_total == before_total


def test_live_account_snapshot_import_contract_errors(api_client, session_factory):
    _seed_live_prices(session_factory)
    c = api_client
    acc = post_json(
        c, "/api/live/accounts", {"name": "导入校验账户A", "initial_cash": 30000}
    )
    aid = int(acc["id"])
    st = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略校验A"})
    sid = int(st["id"])
    h = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "SNAP-VA"},
    )
    hid = int(h["id"])
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
            "trade_time": "09:31:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 100,
            "fee": 0.2,
            "idempotency_key": "snapshot-validate-k1",
        },
    )
    exported = get_json(c, f"/api/live/accounts/{aid}/export")
    r_bad_sha = post_response(
        c,
        f"/api/live/accounts/{aid}/import",
        {
            "replace_all": True,
            "payload": exported,
            "payload_sha256": "0" * 64,
        },
        expected_status=400,
    )
    assert "payload_sha256 mismatch" in str(r_bad_sha.json().get("detail", ""))

    r_extra = post_response(
        c,
        f"/api/live/accounts/{aid}/import",
        {
            "replace_all": True,
            "dry_run": True,
            "payload": exported,
            "unknown_field": "x",
        },
        expected_status=422,
    )
    assert r_extra.status_code == 422


def test_live_account_snapshot_import_allows_deleted_trade_audit_logs(
    api_client, session_factory
):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(
        c, "/api/live/accounts", {"name": "导入审计账户A", "initial_cash": 20000}
    )
    aid = int(acc["id"])
    st = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略审计A"})
    sid = int(st["id"])
    h = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "SNAP-AD"},
    )
    hid = int(h["id"])
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
            "trade_time": "09:31:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 100,
            "fee": 0.2,
            "idempotency_key": "snapshot-audit-delete-k1",
        },
    )
    tid = int(t["id"])
    r_del = c.request(
        "DELETE",
        f"/api/live/trades/{tid}",
        json={"reason": "测试删除成交审计导入"},
    )
    assert r_del.status_code == 200

    snapshot = get_json(c, f"/api/live/accounts/{aid}/export")
    assert int(len(snapshot["payload"]["trades"])) == 0
    assert int(len(snapshot["payload"]["trade_audit_logs"])) >= 1
    assert any(
        int(x.get("trade_id", 0)) == tid
        for x in snapshot["payload"]["trade_audit_logs"]
    )

    imported = post_json(
        c,
        f"/api/live/accounts/{aid}/import",
        {
            "replace_all": True,
            "payload": snapshot,
        },
    )
    assert imported["ok"] is True
    assert int(imported["inserted_counts"]["trade_audit_logs"]) >= 1
    assert any(
        "remapped deleted trade_id" in str(x) for x in imported.get("warnings", [])
    )


def test_live_account_snapshot_import_deleted_audit_trade_id_no_collision(
    api_client, session_factory
):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(
        c, "/api/live/accounts", {"name": "导入审计账户B", "initial_cash": 30000}
    )
    aid = int(acc["id"])
    st = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略审计B"})
    sid = int(st["id"])
    h = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "SNAP-AD-B"},
    )
    hid = int(h["id"])
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
            "trade_time": "09:31:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 100,
            "fee": 0.2,
            "idempotency_key": "snapshot-audit-b-k1",
        },
    )
    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159916",
            "name": "深证ETF",
            "trade_date": "20240624",
            "trade_time": "10:00:00",
            "side": "BUY",
            "price": 8.20,
            "quantity": 100,
            "fee": 0.3,
            "idempotency_key": "snapshot-audit-b-k2",
        },
    )
    tid_deleted = int(t1["id"])
    r_del = c.request(
        "DELETE",
        f"/api/live/trades/{tid_deleted}",
        json={"reason": "测试删除成交审计ID冲突"},
    )
    assert r_del.status_code == 200

    snapshot = get_json(c, f"/api/live/accounts/{aid}/export")
    assert int(len(snapshot["payload"]["trades"])) == 1
    imported = post_json(
        c,
        f"/api/live/accounts/{aid}/import",
        {
            "replace_all": True,
            "payload": snapshot,
        },
    )
    assert imported["ok"] is True

    with session_factory() as db:
        trade_ids = {
            int(x.id)
            for x in db.query(LiveTrade).filter(LiveTrade.account_id == int(aid)).all()
        }
        logs = (
            db.query(LiveTradeAuditLog)
            .filter(
                LiveTradeAuditLog.account_id == int(aid),
                LiveTradeAuditLog.action == "delete",
            )
            .all()
        )
        assert logs
        target = next(x for x in logs if str(x.reason) == "测试删除成交审计ID冲突")
        assert int(target.trade_id) not in trade_ids
        assert int(target.trade_id) < 0


def test_live_account_snapshot_export_preserves_repo_buyback_interest_days_zero(
    api_client, session_factory
):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(
        c, "/api/live/accounts", {"name": "导入Repo账户A", "initial_cash": 100000}
    )
    aid = int(acc["id"])
    st = post_json(
        c,
        f"/api/live/accounts/{aid}/strategies",
        {"name": "逆回购策略A", "strategy_type": "bond_repo"},
    )
    sid = int(st["id"])
    h = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "SNAP-REPO"},
    )
    hid = int(h["id"])

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
            "quantity": 10,
            "amount": 10000.0,
            "repo_action": "LEND",
            "repo_principal_amount": 10000.0,
            "repo_interest_days": 3,
            "idempotency_key": "snapshot-repo-open-k1",
        },
    )
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
            "quantity": 10,
            "amount": 10001.2,
            "repo_action": "BUYBACK",
            "repo_principal_amount": 10000.0,
            "repo_open_trade_id": int(open_trade["id"]),
            "idempotency_key": "snapshot-repo-close-k1",
        },
    )
    assert close_trade["repo_action"] == "BUYBACK"
    assert close_trade["repo_interest_days"] is None

    snapshot = get_json(c, f"/api/live/accounts/{aid}/export")
    details = {int(x["trade_id"]): x for x in snapshot["payload"]["repo_trade_details"]}
    close_detail = details[int(close_trade["id"])]
    assert int(close_detail["interest_days"]) == 0

    imported = post_json(
        c,
        f"/api/live/accounts/{aid}/import",
        {
            "replace_all": True,
            "payload": snapshot,
            "payload_sha256": _snapshot_sha256(snapshot),
        },
    )
    assert imported["ok"] is True
    snapshot2 = get_json(c, f"/api/live/accounts/{aid}/export")
    details2 = {
        int(x["trade_id"]): x for x in snapshot2["payload"]["repo_trade_details"]
    }
    assert any(int(x.get("interest_days", -1)) == 0 for x in details2.values())


def test_live_account_snapshot_import_request_id_dry_run_not_cached(
    api_client, session_factory
):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(
        c, "/api/live/accounts", {"name": "导入幂等账户A", "initial_cash": 50000}
    )
    aid = int(acc["id"])
    st = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略幂等A"})
    sid = int(st["id"])
    h = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "SNAP-IDEMP"},
    )
    hid = int(h["id"])
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
            "trade_time": "09:31:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 100,
            "fee": 0.2,
            "idempotency_key": "snapshot-idemp-base-k1",
        },
    )
    snapshot = get_json(c, f"/api/live/accounts/{aid}/export")
    rid = "snapshot-request-dryrun-1"
    dry_run = post_json(
        c,
        f"/api/live/accounts/{aid}/import",
        {
            "replace_all": True,
            "dry_run": True,
            "import_request_id": rid,
            "payload": snapshot,
        },
    )
    assert dry_run["dry_run"] is True

    post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid,
            "shareholder_account_id": hid,
            "code": "159916",
            "name": "深证ETF",
            "trade_date": "20240624",
            "trade_time": "10:00:00",
            "side": "BUY",
            "price": 8.20,
            "quantity": 100,
            "fee": 0.3,
            "idempotency_key": "snapshot-idemp-extra-k2",
        },
    )
    before_total = int(
        get_json(c, f"/api/live/trades?account_id={aid}&page=1&page_size=100")["total"]
    )
    assert before_total == 2

    imported = post_json(
        c,
        f"/api/live/accounts/{aid}/import",
        {
            "replace_all": True,
            "dry_run": False,
            "import_request_id": rid,
            "payload": snapshot,
            "payload_sha256": _snapshot_sha256(snapshot),
        },
    )
    assert imported["ok"] is True
    assert imported["dry_run"] is False
    after_total = int(
        get_json(c, f"/api/live/trades?account_id={aid}&page=1&page_size=100")["total"]
    )
    assert after_total == 1


def test_live_account_snapshot_import_request_id_conflict_on_payload_change(
    api_client, session_factory
):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(
        c, "/api/live/accounts", {"name": "导入幂等账户B", "initial_cash": 50000}
    )
    aid = int(acc["id"])
    st = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略幂等B"})
    sid = int(st["id"])
    h = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "SNAP-IDEMP-B"},
    )
    hid = int(h["id"])
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
            "trade_time": "09:31:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 100,
            "fee": 0.2,
            "idempotency_key": "snapshot-idemp-b-k1",
        },
    )
    snapshot = get_json(c, f"/api/live/accounts/{aid}/export")
    rid = "snapshot-request-payload-check-1"
    ok = post_json(
        c,
        f"/api/live/accounts/{aid}/import",
        {
            "replace_all": True,
            "import_request_id": rid,
            "payload": snapshot,
            "payload_sha256": _snapshot_sha256(snapshot),
        },
    )
    assert ok["ok"] is True
    changed = json.loads(json.dumps(snapshot))
    changed["payload"]["account"]["notes"] = "changed payload"
    r = post_response(
        c,
        f"/api/live/accounts/{aid}/import",
        {
            "replace_all": True,
            "import_request_id": rid,
            "payload": changed,
            "payload_sha256": _snapshot_sha256(changed),
        },
        expected_status=409,
    )
    assert "different payload" in str(r.json().get("detail", ""))


def test_live_account_snapshot_import_rejects_repo_detail_on_non_repo_trade(
    api_client, session_factory
):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(
        c, "/api/live/accounts", {"name": "导入Repo校验账户A", "initial_cash": 50000}
    )
    aid = int(acc["id"])
    st = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略普通A"})
    sid = int(st["id"])
    h = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "SNAP-REPO-CHECK"},
    )
    hid = int(h["id"])
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
            "trade_time": "09:31:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 100,
            "fee": 0.2,
            "idempotency_key": "snapshot-repo-illegal-k1",
        },
    )
    snapshot = get_json(c, f"/api/live/accounts/{aid}/export")
    trade_id = int(snapshot["payload"]["trades"][0]["id"])
    snapshot["payload"]["repo_trade_details"] = [
        {
            "trade_id": trade_id,
            "repo_action": "LEND",
            "principal_amount": 1000.0,
            "annual_rate_pct": 1.46,
            "interest_days": 3,
            "day_count_basis": 365,
            "open_trade_id": None,
        }
    ]
    r = post_response(
        c,
        f"/api/live/accounts/{aid}/import",
        {
            "replace_all": True,
            "payload": snapshot,
            "payload_sha256": _snapshot_sha256(snapshot),
        },
        expected_status=400,
    )
    assert "only allowed for bond_repo trades" in str(r.json().get("detail", ""))


def test_live_account_snapshot_import_rejects_repo_open_trade_id_non_repo_reference(
    api_client, session_factory
):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(
        c, "/api/live/accounts", {"name": "导入Repo校验账户B", "initial_cash": 120000}
    )
    aid = int(acc["id"])
    st_etf = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略普通B"})
    sid_etf = int(st_etf["id"])
    st_repo = post_json(
        c,
        f"/api/live/accounts/{aid}/strategies",
        {"name": "策略逆回购B", "strategy_type": "bond_repo"},
    )
    sid_repo = int(st_repo["id"])
    h = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "SNAP-REPO-OPEN-CHECK"},
    )
    hid = int(h["id"])

    etf_trade = post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid_etf,
            "shareholder_account_id": hid,
            "code": "159915",
            "name": "创业板ETF",
            "trade_date": "20240621",
            "trade_time": "09:31:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 100,
            "fee": 0.2,
            "idempotency_key": "snapshot-repo-open-illegal-etf-k1",
        },
    )
    open_repo = post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid_repo,
            "shareholder_account_id": hid,
            "code": "204001",
            "name": "国债逆回购",
            "trade_date": "20240621",
            "trade_time": "15:30:00",
            "side": "BUY",
            "price": 1.46,
            "quantity": 10,
            "amount": 10000.0,
            "repo_action": "LEND",
            "repo_principal_amount": 10000.0,
            "repo_interest_days": 3,
            "idempotency_key": "snapshot-repo-open-illegal-open-k2",
        },
    )
    close_repo = post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid_repo,
            "shareholder_account_id": hid,
            "code": "204001",
            "name": "国债逆回购",
            "trade_date": "20240624",
            "trade_time": "00:00:00",
            "side": "SELL",
            "price": 1.46,
            "quantity": 10,
            "amount": 10001.2,
            "repo_action": "BUYBACK",
            "repo_principal_amount": 10000.0,
            "repo_open_trade_id": int(open_repo["id"]),
            "idempotency_key": "snapshot-repo-open-illegal-close-k3",
        },
    )

    snapshot = get_json(c, f"/api/live/accounts/{aid}/export")
    details = snapshot["payload"]["repo_trade_details"]
    close_trade_id = int(close_repo["id"])
    for d in details:
        if int(d["trade_id"]) == close_trade_id:
            d["open_trade_id"] = int(etf_trade["id"])
            break
    r = post_response(
        c,
        f"/api/live/accounts/{aid}/import",
        {
            "replace_all": True,
            "payload": snapshot,
            "payload_sha256": _snapshot_sha256(snapshot),
        },
        expected_status=400,
    )
    assert "must reference a bond_repo trade" in str(r.json().get("detail", ""))


def test_live_account_snapshot_import_rejects_repo_action_side_mismatch(
    api_client, session_factory
):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(
        c, "/api/live/accounts", {"name": "导入Repo校验账户C", "initial_cash": 120000}
    )
    aid = int(acc["id"])
    st_repo = post_json(
        c,
        f"/api/live/accounts/{aid}/strategies",
        {"name": "策略逆回购C", "strategy_type": "bond_repo"},
    )
    sid_repo = int(st_repo["id"])
    h = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "SNAP-REPO-SIDE-CHECK"},
    )
    hid = int(h["id"])

    open_repo = post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid_repo,
            "shareholder_account_id": hid,
            "code": "204001",
            "name": "国债逆回购",
            "trade_date": "20240621",
            "trade_time": "15:30:00",
            "side": "BUY",
            "price": 1.46,
            "quantity": 10,
            "amount": 10000.0,
            "repo_action": "LEND",
            "repo_principal_amount": 10000.0,
            "repo_interest_days": 3,
            "idempotency_key": "snapshot-repo-side-open-k1",
        },
    )
    close_repo = post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid_repo,
            "shareholder_account_id": hid,
            "code": "204001",
            "name": "国债逆回购",
            "trade_date": "20240624",
            "trade_time": "00:00:00",
            "side": "SELL",
            "price": 1.46,
            "quantity": 10,
            "amount": 10001.2,
            "repo_action": "BUYBACK",
            "repo_principal_amount": 10000.0,
            "repo_open_trade_id": int(open_repo["id"]),
            "idempotency_key": "snapshot-repo-side-close-k2",
        },
    )

    snapshot = get_json(c, f"/api/live/accounts/{aid}/export")
    details = snapshot["payload"]["repo_trade_details"]
    close_trade_id = int(close_repo["id"])
    for d in details:
        if int(d["trade_id"]) == close_trade_id:
            d["repo_action"] = "LEND"
            break
    r = post_response(
        c,
        f"/api/live/accounts/{aid}/import",
        {
            "replace_all": True,
            "payload": snapshot,
            "payload_sha256": _snapshot_sha256(snapshot),
        },
        expected_status=400,
    )
    assert "LEND detail must map to BUY side trade" in str(r.json().get("detail", ""))


def test_live_account_snapshot_import_rejects_duplicate_trade_id(
    api_client, session_factory
):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(
        c, "/api/live/accounts", {"name": "导入重复ID账户A", "initial_cash": 50000}
    )
    aid = int(acc["id"])
    st = post_json(c, f"/api/live/accounts/{aid}/strategies", {"name": "策略重复IDA"})
    sid = int(st["id"])
    h = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "SNAP-DUP-ID-A"},
    )
    hid = int(h["id"])
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
            "trade_time": "09:31:00",
            "side": "BUY",
            "price": 4.10,
            "quantity": 100,
            "fee": 0.2,
            "idempotency_key": "snapshot-dup-id-k1",
        },
    )
    snapshot = get_json(c, f"/api/live/accounts/{aid}/export")
    assert len(snapshot["payload"]["trades"]) == 1
    dup = json.loads(json.dumps(snapshot["payload"]["trades"][0]))
    dup["idempotency_key"] = "snapshot-dup-id-k2"
    snapshot["payload"]["trades"].append(dup)
    r = post_response(
        c,
        f"/api/live/accounts/{aid}/import",
        {
            "replace_all": True,
            "payload": snapshot,
            "payload_sha256": _snapshot_sha256(snapshot),
        },
        expected_status=400,
    )
    assert "duplicate trade id in payload" in str(r.json().get("detail", ""))


def test_live_account_snapshot_import_rejects_repo_open_trade_id_self_reference(
    api_client, session_factory
):
    _seed_live_prices(session_factory)
    c = api_client

    acc = post_json(
        c, "/api/live/accounts", {"name": "导入Repo校验账户D", "initial_cash": 120000}
    )
    aid = int(acc["id"])
    st_repo = post_json(
        c,
        f"/api/live/accounts/{aid}/strategies",
        {"name": "策略逆回购D", "strategy_type": "bond_repo"},
    )
    sid_repo = int(st_repo["id"])
    h = post_json(
        c,
        f"/api/live/accounts/{aid}/shareholders",
        {"shareholder_account": "SNAP-REPO-SELF-CHECK"},
    )
    hid = int(h["id"])

    open_repo = post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid_repo,
            "shareholder_account_id": hid,
            "code": "204001",
            "name": "国债逆回购",
            "trade_date": "20240621",
            "trade_time": "15:30:00",
            "side": "BUY",
            "price": 1.46,
            "quantity": 10,
            "amount": 10000.0,
            "repo_action": "LEND",
            "repo_principal_amount": 10000.0,
            "repo_interest_days": 3,
            "idempotency_key": "snapshot-repo-self-open-k1",
        },
    )
    close_repo = post_json(
        c,
        "/api/live/trades",
        {
            "account_id": aid,
            "strategy_id": sid_repo,
            "shareholder_account_id": hid,
            "code": "204001",
            "name": "国债逆回购",
            "trade_date": "20240624",
            "trade_time": "00:00:00",
            "side": "SELL",
            "price": 1.46,
            "quantity": 10,
            "amount": 10001.2,
            "repo_action": "BUYBACK",
            "repo_principal_amount": 10000.0,
            "repo_open_trade_id": int(open_repo["id"]),
            "idempotency_key": "snapshot-repo-self-close-k2",
        },
    )

    snapshot = get_json(c, f"/api/live/accounts/{aid}/export")
    details = snapshot["payload"]["repo_trade_details"]
    close_trade_id = int(close_repo["id"])
    for d in details:
        if int(d["trade_id"]) == close_trade_id:
            d["open_trade_id"] = close_trade_id
            break
    r = post_response(
        c,
        f"/api/live/accounts/{aid}/import",
        {
            "replace_all": True,
            "payload": snapshot,
            "payload_sha256": _snapshot_sha256(snapshot),
        },
        expected_status=400,
    )
    assert "must not reference itself" in str(r.json().get("detail", ""))
