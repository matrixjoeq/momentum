from __future__ import annotations

import datetime as dt
import json
import math
import uuid
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .deps import get_session
from .schemas import (
    LiveAccountCashflowCreateRequest,
    LiveAccountCreateRequest,
    LiveAccountOut,
    LiveAccountUpdateRequest,
    LiveAttributionOut,
    LiveCashflowOut,
    LiveClosedRoundOut,
    LiveCorporateActionCreateRequest,
    LiveCorporateActionOut,
    LiveFeeStatsOut,
    LiveHoldingOut,
    LivePerformanceOut,
    LiveReplayRequest,
    LiveShareholderAccountCreateRequest,
    LiveShareholderAccountOut,
    LiveStrategyCashflowCreateRequest,
    LiveStrategyCreateRequest,
    LiveStrategyOut,
    LiveStrategyTransferRequest,
    LiveStrategyUpdateRequest,
    LiveTradeBatchCreateRequest,
    LiveTradeCreateRequest,
    LiveTradeDeleteRequest,
    LiveTradeOut,
    LiveTradeUpdateRequest,
)
from ..analysis.baseline import (
    _annualized_return,
    _annualized_vol,
    _max_drawdown,
    _max_drawdown_duration_days,
    _sharpe,
    _sortino,
    _ulcer_index,
)
from ..calendar.trading_calendar import shift_to_trading_day, trading_days
from ..db.models import (
    EtfPrice,
    LiveAccount,
    LiveAccountCashflow,
    LiveClosedRound,
    LiveClosedRoundLeg,
    LiveCorporateActionEvent,
    LiveHoldingSnapshot,
    LiveNavDaily,
    LiveShareholderAccount,
    LiveStrategy,
    LiveStrategyCashflow,
    LiveStrategyProfile,
    LiveSymbolAlias,
    LiveTrade,
    LiveTradeAuditLog,
    LiveRepoTradeDetail,
)

router = APIRouter()
TRADE_FEE_RATE = 1e-4
TRADE_FEE_MIN = 0.2
REPO_TRADE_FEE_RATE = 1e-5
REPO_LOT_AMOUNT = 1000.0
REPO_SYMBOL_NAME_MAP: dict[str, str] = {
    "204001": "GC001",
    "131810": "R-001",
}


@dataclass
class _Scope:
    scope_type: str  # account|strategy
    scope_id: int
    account_id: int
    strategy_id: int | None


@dataclass
class _RoundState:
    round_no: int
    open_date: dt.date
    code: str
    name: str
    buy_count: int = 0
    sell_count: int = 0
    buy_qty: float = 0.0
    sell_qty: float = 0.0
    buy_amount: float = 0.0
    sell_amount: float = 0.0
    realized_pnl: float = 0.0
    total_fee: float = 0.0
    legs: list[dict[str, Any]] | None = None

    def ensure_legs(self) -> list[dict[str, Any]]:
        if self.legs is None:
            self.legs = []
        return self.legs


def _to_date(x: str) -> dt.date:
    s = str(x or "").strip()
    if len(s) == 8 and s.isdigit():
        return dt.datetime.strptime(s, "%Y%m%d").date()
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def _date_s(d: dt.date | None) -> str | None:
    return d.isoformat() if d is not None else None


def _dt_s(x: dt.datetime | None) -> str:
    if x is None:
        return ""
    return x.isoformat()


def _norm_code(code: str) -> str:
    s = str(code or "").strip().upper()
    if "." in s:
        s = s.split(".", 1)[0]
    if s.startswith(("SH", "SZ")) and len(s) > 2:
        s = s[2:]
    return s


def _norm_side(side: str) -> str:
    s = str(side or "").strip().upper()
    if s in {"BUY", "买入"}:
        return "BUY"
    if s in {"SELL", "卖出"}:
        return "SELL"
    raise HTTPException(status_code=400, detail="side must be BUY or SELL")


def _norm_time(x: str) -> str:
    s = str(x or "").strip()
    if len(s) == 5:
        s = f"{s}:00"
    try:
        dt.datetime.strptime(s, "%H:%M:%S")
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="trade_time must be HH:MM[:SS]"
        ) from exc
    return s


def _norm_strategy_type(raw: str | None) -> str:
    s = str(raw or "").strip().lower()
    if s in {"", "etf", "etf_spot"}:
        return "etf_spot"
    if s in {"bond_repo", "repo", "reverse_repo"}:
        return "bond_repo"
    raise HTTPException(
        status_code=400, detail="strategy_type must be etf_spot|bond_repo"
    )


def _default_capital_mode(strategy_type: str) -> str:
    return "shared_account_cash" if strategy_type == "bond_repo" else "segregated"


def _norm_capital_mode(raw: str | None, *, strategy_type: str) -> str:
    if raw is None or str(raw).strip() == "":
        return _default_capital_mode(strategy_type)
    s = str(raw).strip().lower()
    if s not in {"segregated", "shared_account_cash"}:
        raise HTTPException(
            status_code=400,
            detail="capital_mode must be segregated|shared_account_cash",
        )
    return s


def _strategy_type_map(db: Session, strategy_ids: list[int]) -> dict[int, str]:
    if not strategy_ids:
        return {}
    out: dict[int, str] = {int(sid): "etf_spot" for sid in strategy_ids}
    rows = (
        db.query(LiveStrategyProfile)
        .filter(LiveStrategyProfile.strategy_id.in_(strategy_ids))
        .all()
    )
    for x in rows:
        out[int(x.strategy_id)] = _norm_strategy_type(x.strategy_type)
    return out


def _strategy_profile_map(
    db: Session, strategy_ids: list[int]
) -> dict[int, tuple[str, str]]:
    if not strategy_ids:
        return {}
    out: dict[int, tuple[str, str]] = {
        int(sid): ("etf_spot", "segregated") for sid in strategy_ids
    }
    rows = (
        db.query(LiveStrategyProfile)
        .filter(LiveStrategyProfile.strategy_id.in_(strategy_ids))
        .all()
    )
    for x in rows:
        stype = _norm_strategy_type(x.strategy_type)
        cmode = _norm_capital_mode(x.capital_mode, strategy_type=stype)
        out[int(x.strategy_id)] = (stype, cmode)
    return out


def _strategy_type_for(db: Session, strategy_id: int) -> str:
    row = (
        db.query(LiveStrategyProfile)
        .filter(LiveStrategyProfile.strategy_id == int(strategy_id))
        .one_or_none()
    )
    if row is None:
        return "etf_spot"
    return _norm_strategy_type(row.strategy_type)


def _strategy_profile_for(db: Session, strategy_id: int) -> tuple[str, str]:
    row = (
        db.query(LiveStrategyProfile)
        .filter(LiveStrategyProfile.strategy_id == int(strategy_id))
        .one_or_none()
    )
    if row is None:
        return ("etf_spot", "segregated")
    stype = _norm_strategy_type(row.strategy_type)
    cmode = _norm_capital_mode(row.capital_mode, strategy_type=stype)
    return (stype, cmode)


def _is_trade_time_allowed(x: str) -> bool:
    t = dt.datetime.strptime(x, "%H:%M:%S").time()
    return dt.time(9, 0, 0) <= t <= dt.time(15, 0, 0)


def _is_repo_lend_time_allowed(x: str) -> bool:
    t = dt.datetime.strptime(x, "%H:%M:%S").time()
    return dt.time(9, 30, 0) <= t <= dt.time(15, 30, 0)


def _round_fee_2(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _default_trade_fee(price: float, quantity: float) -> float:
    amount = float(price) * float(quantity)
    return _round_fee_2(max(amount * TRADE_FEE_RATE, TRADE_FEE_MIN))


def _default_repo_trade_fee(amount: float) -> float:
    return _round_fee_2(float(amount) * REPO_TRADE_FEE_RATE)


def _norm_repo_symbol(code_raw: str) -> tuple[str, str]:
    code = _norm_code(code_raw)
    name = REPO_SYMBOL_NAME_MAP.get(code)
    if not name:
        raise HTTPException(
            status_code=400,
            detail="bond_repo code must be one of: 204001(GC001), 131810(R-001)",
        )
    return code, name


def _scope_from_ids(
    db: Session, *, account_id: int | None, strategy_id: int | None
) -> _Scope:
    if strategy_id is not None:
        strategy = (
            db.query(LiveStrategy).filter(LiveStrategy.id == strategy_id).one_or_none()
        )
        if strategy is None:
            raise HTTPException(status_code=404, detail="strategy not found")
        return _Scope(
            scope_type="strategy",
            scope_id=int(strategy.id),
            account_id=int(strategy.account_id),
            strategy_id=int(strategy.id),
        )
    if account_id is None:
        raise HTTPException(
            status_code=400, detail="account_id or strategy_id is required"
        )
    account = db.query(LiveAccount).filter(LiveAccount.id == account_id).one_or_none()
    if account is None:
        raise HTTPException(status_code=404, detail="account not found")
    return _Scope(
        scope_type="account",
        scope_id=int(account.id),
        account_id=int(account.id),
        strategy_id=None,
    )


def _latest_price_map(
    db: Session, *, codes: list[str], start: dt.date, end: dt.date
) -> dict[str, tuple[list[dt.date], list[float]]]:
    out: dict[str, tuple[list[dt.date], list[float]]] = {}
    if not codes:
        return out
    rows = (
        db.query(EtfPrice)
        .filter(
            EtfPrice.code.in_(codes),
            EtfPrice.adjust == "none",
            EtfPrice.trade_date >= start,
            EtfPrice.trade_date <= end,
        )
        .order_by(EtfPrice.code.asc(), EtfPrice.trade_date.asc())
        .all()
    )
    grouped_dates: dict[str, list[dt.date]] = defaultdict(list)
    grouped_close: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.close is None:
            continue
        c = _norm_code(str(row.code))
        grouped_dates[c].append(row.trade_date)
        grouped_close[c].append(float(row.close))
    for c in codes:
        out[c] = (grouped_dates.get(c, []), grouped_close.get(c, []))
    return out


def _price_on_or_before(
    price_map: dict[str, tuple[list[dt.date], list[float]]], code: str, day: dt.date
) -> tuple[float | None, dt.date | None]:
    dates, closes = price_map.get(code, ([], []))
    if not dates:
        return None, None
    idx = bisect_right(dates, day) - 1
    if idx < 0:
        return None, None
    return float(closes[idx]), dates[idx]


def _scope_rows(
    db: Session, scope: _Scope
) -> tuple[
    list[LiveTrade],
    list[LiveAccountCashflow],
    list[LiveStrategyCashflow],
    list[LiveCorporateActionEvent],
]:
    if scope.scope_type == "strategy":
        trades = (
            db.query(LiveTrade)
            .filter(LiveTrade.strategy_id == scope.strategy_id)
            .order_by(
                LiveTrade.trade_date.asc(),
                LiveTrade.trade_time.asc(),
                LiveTrade.id.asc(),
            )
            .all()
        )
        a_cfs = []
        s_cfs = (
            db.query(LiveStrategyCashflow)
            .filter(LiveStrategyCashflow.strategy_id == scope.strategy_id)
            .order_by(
                LiveStrategyCashflow.flow_date.asc(), LiveStrategyCashflow.id.asc()
            )
            .all()
        )
        events = (
            db.query(LiveCorporateActionEvent)
            .filter(
                (LiveCorporateActionEvent.account_id.is_(None))
                | (LiveCorporateActionEvent.account_id == scope.account_id),
                (LiveCorporateActionEvent.strategy_id.is_(None))
                | (LiveCorporateActionEvent.strategy_id == scope.strategy_id),
            )
            .order_by(
                LiveCorporateActionEvent.effective_date.asc(),
                LiveCorporateActionEvent.id.asc(),
            )
            .all()
        )
        return trades, a_cfs, s_cfs, events

    trades = (
        db.query(LiveTrade)
        .filter(LiveTrade.account_id == scope.account_id)
        .order_by(
            LiveTrade.trade_date.asc(), LiveTrade.trade_time.asc(), LiveTrade.id.asc()
        )
        .all()
    )
    a_cfs = (
        db.query(LiveAccountCashflow)
        .filter(LiveAccountCashflow.account_id == scope.account_id)
        .order_by(LiveAccountCashflow.flow_date.asc(), LiveAccountCashflow.id.asc())
        .all()
    )
    s_cfs = []
    events = (
        db.query(LiveCorporateActionEvent)
        .filter(
            (LiveCorporateActionEvent.account_id.is_(None))
            | (LiveCorporateActionEvent.account_id == scope.account_id)
        )
        .order_by(
            LiveCorporateActionEvent.effective_date.asc(),
            LiveCorporateActionEvent.id.asc(),
        )
        .all()
    )
    return trades, a_cfs, s_cfs, events


def _trade_order_key(
    trade_date: dt.date, trade_time: str, trade_id: int
) -> tuple[dt.date, str, int]:
    return trade_date, _norm_time(trade_time), int(trade_id)


def _trade_cash_delta(
    *,
    side: str,
    amount: float,
    fee: float,
    strategy_type: str,
    repo_detail: LiveRepoTradeDetail | dict[str, Any] | None,
) -> float:
    side_norm = _norm_side(side)
    amt = float(amount)
    fee_v = float(fee or 0.0)
    if strategy_type == "bond_repo":
        if repo_detail is None:
            raise HTTPException(status_code=400, detail="missing repo detail")
        if isinstance(repo_detail, LiveRepoTradeDetail):
            action = _norm_repo_action(repo_detail.repo_action, side=side_norm)
            principal = float(repo_detail.principal_amount or amt)
        else:
            action = _norm_repo_action(
                str(repo_detail.get("repo_action") or ""), side=side_norm
            )
            principal = float(repo_detail.get("principal_amount") or amt)
        if action == "LEND":
            return -(principal + fee_v)
        return amt - fee_v
    if side_norm == "BUY":
        return -(amt + fee_v)
    return amt - fee_v


def _account_cash_before_order(
    db: Session,
    *,
    account_id: int,
    order_date: dt.date,
    order_time: str,
    order_id: int,
    exclude_trade_id: int | None = None,
) -> float:
    account = (
        db.query(LiveAccount).filter(LiveAccount.id == int(account_id)).one_or_none()
    )
    cash = float(account.initial_cash) if account is not None else 0.0
    flows = (
        db.query(LiveAccountCashflow)
        .filter(LiveAccountCashflow.account_id == int(account_id))
        .order_by(LiveAccountCashflow.flow_date.asc(), LiveAccountCashflow.id.asc())
        .all()
    )
    for f in flows:
        if f.flow_date > order_date:
            break
        flow_type = str(f.flow_type or "").strip().lower()
        if flow_type.startswith("transfer_"):
            continue
        if str(f.notes or "").strip().lower() == "initial_cash":
            # Avoid double-counting the synthetic initial cashflow row:
            # baseline cash already starts from account.initial_cash.
            continue
        cash += float(f.amount)

    trades = (
        db.query(LiveTrade)
        .filter(
            LiveTrade.account_id == int(account_id),
            LiveTrade.trade_date <= order_date,
        )
        .order_by(
            LiveTrade.trade_date.asc(), LiveTrade.trade_time.asc(), LiveTrade.id.asc()
        )
        .all()
    )
    if not trades:
        return float(cash)
    strategy_type_by_id = _strategy_type_map(
        db, sorted({int(t.strategy_id) for t in trades})
    )
    repo_detail_by_trade_id = _repo_detail_map(db, [int(t.id) for t in trades])
    target_key = _trade_order_key(order_date, order_time, int(order_id))
    for t in trades:
        tid = int(t.id)
        if exclude_trade_id is not None and tid == int(exclude_trade_id):
            continue
        if _trade_order_key(t.trade_date, str(t.trade_time), tid) >= target_key:
            break
        delta = _trade_cash_delta(
            side=str(t.side),
            amount=float(t.amount or (float(t.quantity) * float(t.price))),
            fee=float(t.fee or 0.0),
            strategy_type=strategy_type_by_id.get(int(t.strategy_id), "etf_spot"),
            repo_detail=repo_detail_by_trade_id.get(tid),
        )
        cash += delta
    return float(cash)


def _strategy_cash_before_order(
    db: Session,
    *,
    strategy_id: int,
    order_date: dt.date,
    order_time: str,
    order_id: int,
    exclude_trade_id: int | None = None,
) -> float:
    cash = 0.0
    flows = (
        db.query(LiveStrategyCashflow)
        .filter(LiveStrategyCashflow.strategy_id == int(strategy_id))
        .order_by(LiveStrategyCashflow.flow_date.asc(), LiveStrategyCashflow.id.asc())
        .all()
    )
    for f in flows:
        if f.flow_date > order_date:
            break
        cash += float(f.amount)

    trades = (
        db.query(LiveTrade)
        .filter(
            LiveTrade.strategy_id == int(strategy_id),
            LiveTrade.trade_date <= order_date,
        )
        .order_by(
            LiveTrade.trade_date.asc(), LiveTrade.trade_time.asc(), LiveTrade.id.asc()
        )
        .all()
    )
    if not trades:
        return float(cash)
    strategy_type_by_id = _strategy_type_map(
        db, sorted({int(t.strategy_id) for t in trades})
    )
    repo_detail_by_trade_id = _repo_detail_map(db, [int(t.id) for t in trades])
    target_key = _trade_order_key(order_date, order_time, int(order_id))
    for t in trades:
        tid = int(t.id)
        if exclude_trade_id is not None and tid == int(exclude_trade_id):
            continue
        if _trade_order_key(t.trade_date, str(t.trade_time), tid) >= target_key:
            break
        delta = _trade_cash_delta(
            side=str(t.side),
            amount=float(t.amount or (float(t.quantity) * float(t.price))),
            fee=float(t.fee or 0.0),
            strategy_type=strategy_type_by_id.get(int(t.strategy_id), "etf_spot"),
            repo_detail=repo_detail_by_trade_id.get(tid),
        )
        cash += delta
    return float(cash)


def _validate_trade_funding_constraints(
    db: Session,
    *,
    account_id: int,
    strategy_id: int,
    strategy_type: str,
    trade_date: dt.date,
    trade_time: str,
    side: str,
    amount: float,
    fee: float,
    repo_detail: dict[str, Any] | None,
    exclude_trade_id: int | None = None,
    order_trade_id: int | None = None,
) -> None:
    _ = strategy_id
    # Financing/leverage is not enabled yet: account-level cash must stay
    # non-negative at order time (account total position <= 100%).
    # Strategy-level transferred budget is treated as a soft allocation target:
    # strategies within the same account can share the account cash pool.
    order_id = int(order_trade_id) if order_trade_id is not None else 10**18
    order_time_norm = _norm_time(trade_time)
    delta = _trade_cash_delta(
        side=side,
        amount=amount,
        fee=fee,
        strategy_type=strategy_type,
        repo_detail=repo_detail,
    )
    account = (
        db.query(LiveAccount).filter(LiveAccount.id == int(account_id)).one_or_none()
    )
    account_has_external_budget = bool(
        account is not None and float(account.initial_cash or 0.0) > 1e-12
    ) or (
        db.query(LiveAccountCashflow)
        .filter(LiveAccountCashflow.account_id == int(account_id))
        .filter(~LiveAccountCashflow.flow_type.like("transfer_%"))
        .count()
        > 0
    )
    if account_has_external_budget:
        account_cash_before = _account_cash_before_order(
            db,
            account_id=int(account_id),
            order_date=trade_date,
            order_time=order_time_norm,
            order_id=order_id,
            exclude_trade_id=exclude_trade_id,
        )
        account_cash_after = account_cash_before + delta
        if account_cash_after < -1e-6:
            raise HTTPException(
                status_code=400,
                detail=(
                    "insufficient account cash at order time; "
                    "without financing account total position cannot exceed 100%"
                ),
            )


def _apply_symbol_alias(code: str, alias_map: dict[str, str]) -> str:
    cur = code
    # avoid loops from broken data
    for _ in range(6):
        nxt = alias_map.get(cur)
        if not nxt or nxt == cur:
            return cur
        cur = nxt
    return cur


def _safe_json_float(x: float | int | None) -> float | None:
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def _calc_metrics(nav: pd.Series, ret: pd.Series) -> dict[str, Any]:
    if nav.empty:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_recovery_days": 0,
            "sharpe_ratio": None,
            "calmar_ratio": None,
            "sortino_ratio": None,
            "ulcer_index": None,
            "ulcer_performance_index": None,
        }
    # Cumulative return should include every daily return in the window,
    # including the first replay day return.
    cum_ret = float((1.0 + ret.fillna(0.0)).prod() - 1.0) if len(ret) > 0 else 0.0
    ann_ret = float(_annualized_return(nav))
    ann_vol = float(_annualized_vol(ret))
    mdd = float(_max_drawdown(nav))
    mdd_days = int(_max_drawdown_duration_days(nav))
    sharpe = float(_sharpe(ret))
    sortino = float(_sortino(ret))
    calmar = float(ann_ret / abs(mdd)) if mdd < 0 else float("nan")
    ui = float(_ulcer_index(nav, in_percent=True))
    upi = float(ann_ret / (ui / 100.0)) if ui > 0 else float("nan")
    return {
        "cumulative_return": _safe_json_float(cum_ret),
        "annualized_return": _safe_json_float(ann_ret),
        "annualized_volatility": _safe_json_float(ann_vol),
        "max_drawdown": _safe_json_float(mdd),
        "max_drawdown_recovery_days": mdd_days,
        "sharpe_ratio": _safe_json_float(sharpe),
        "calmar_ratio": _safe_json_float(calmar),
        "sortino_ratio": _safe_json_float(sortino),
        "ulcer_index": _safe_json_float(ui),
        "ulcer_performance_index": _safe_json_float(upi),
    }


def _delete_existing_scope_rows(db: Session, scope: _Scope) -> None:
    old_round_ids = [
        int(x[0])
        for x in db.query(LiveClosedRound.id)
        .filter(
            LiveClosedRound.scope_type == scope.scope_type,
            LiveClosedRound.scope_id == scope.scope_id,
        )
        .all()
    ]
    if old_round_ids:
        (
            db.query(LiveClosedRoundLeg)
            .filter(LiveClosedRoundLeg.round_id.in_(old_round_ids))
            .delete(synchronize_session=False)
        )
    (
        db.query(LiveClosedRound)
        .filter(
            LiveClosedRound.scope_type == scope.scope_type,
            LiveClosedRound.scope_id == scope.scope_id,
        )
        .delete(synchronize_session=False)
    )
    (
        db.query(LiveHoldingSnapshot)
        .filter(
            LiveHoldingSnapshot.scope_type == scope.scope_type,
            LiveHoldingSnapshot.scope_id == scope.scope_id,
        )
        .delete(synchronize_session=False)
    )
    (
        db.query(LiveNavDaily)
        .filter(
            LiveNavDaily.scope_type == scope.scope_type,
            LiveNavDaily.scope_id == scope.scope_id,
        )
        .delete(synchronize_session=False)
    )


def _replay_scope(db: Session, scope: _Scope) -> dict[str, Any]:
    trades, account_flows, strategy_flows, events = _scope_rows(db, scope)
    if not trades and not account_flows and not strategy_flows and not events:
        _delete_existing_scope_rows(db, scope)
        return {"scope_type": scope.scope_type, "scope_id": scope.scope_id, "days": 0}

    min_dates: list[dt.date] = []
    max_dates: list[dt.date] = []
    for t in trades:
        min_dates.append(t.trade_date)
        max_dates.append(t.trade_date)
    for f in account_flows:
        min_dates.append(f.flow_date)
        max_dates.append(f.flow_date)
    for f in strategy_flows:
        min_dates.append(f.flow_date)
        max_dates.append(f.flow_date)
    for ev in events:
        min_dates.append(ev.effective_date)
        max_dates.append(ev.effective_date)
    if not min_dates:
        return {"scope_type": scope.scope_type, "scope_id": scope.scope_id, "days": 0}

    start = min(min_dates)
    end = max(max_dates)
    end = max(end, shift_to_trading_day(dt.date.today(), shift="prev"))

    codes = sorted({_norm_code(t.code) for t in trades})
    strategy_type_by_id = _strategy_type_map(
        db, sorted({int(t.strategy_id) for t in trades})
    )
    repo_detail_by_trade_id = _repo_detail_map(db, [int(t.id) for t in trades])
    price_map = _latest_price_map(
        db, codes=codes, start=start - dt.timedelta(days=3650), end=end
    )

    trade_by_day: dict[dt.date, list[LiveTrade]] = defaultdict(list)
    for t in trades:
        trade_by_day[t.trade_date].append(t)

    account_flow_by_day: dict[dt.date, list[LiveAccountCashflow]] = defaultdict(list)
    for f in account_flows:
        account_flow_by_day[f.flow_date].append(f)

    strategy_flow_by_day: dict[dt.date, list[LiveStrategyCashflow]] = defaultdict(list)
    for f in strategy_flows:
        strategy_flow_by_day[f.flow_date].append(f)

    events_by_day: dict[dt.date, list[LiveCorporateActionEvent]] = defaultdict(list)
    for ev in events:
        events_by_day[ev.effective_date].append(ev)

    _delete_existing_scope_rows(db, scope)

    # Position/FIFO lot matching must be isolated by shareholder account.
    # key: (code, shareholder_account_id)
    lots: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    repo_lots: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    rounds_open: dict[tuple[str, int], _RoundState] = {}
    round_no_by_code: dict[str, int] = defaultdict(int)
    closed_round_rows: list[dict[str, Any]] = []
    closed_round_legs: list[dict[str, Any]] = []

    cash = 0.0
    initial_equity_base = 0.0
    if scope.scope_type == "strategy" and trades:
        # Reconstruct a stable strategy capital base:
        # - no strategy cashflow: infer from buy-side gross notional (for segregated books)
        # - with strategy cashflow: top up only the minimum cash deficit so pre-flow trades
        #   won't collapse equity to (near) zero and break NAV/return math.
        inferred_initial_cash = 0.0
        for t in trades:
            side = _norm_side(t.side)
            strategy_type = strategy_type_by_id.get(int(t.strategy_id), "etf_spot")
            amount = float(t.amount or (float(t.quantity) * float(t.price)))
            fee = float(t.fee or 0.0)
            if strategy_type == "bond_repo":
                detail = repo_detail_by_trade_id.get(int(t.id))
                if detail is None:
                    continue
                action = _norm_repo_action(detail.repo_action, side=side)
                if action == "LEND":
                    principal = float(detail.principal_amount or amount)
                    inferred_initial_cash += principal + fee
            elif side == "BUY":
                inferred_initial_cash += amount + fee

        cash_probe = 0.0
        min_cash_probe = 0.0
        probe_days = sorted(
            set([t.trade_date for t in trades] + [f.flow_date for f in strategy_flows])
        )
        for pday in probe_days:
            for f in strategy_flow_by_day.get(pday, []):
                cash_probe += float(f.amount)
            day_trades_probe = sorted(
                trade_by_day.get(pday, []),
                key=lambda x: (str(x.trade_time or ""), int(x.id)),
            )
            for t in day_trades_probe:
                side = _norm_side(t.side)
                strategy_type = strategy_type_by_id.get(int(t.strategy_id), "etf_spot")
                qty = float(t.quantity)
                amount = float(t.amount or (qty * float(t.price)))
                fee = float(t.fee or 0.0)
                if strategy_type == "bond_repo":
                    detail = repo_detail_by_trade_id.get(int(t.id))
                    if detail is None:
                        continue
                    action = _norm_repo_action(detail.repo_action, side=side)
                    principal = float(detail.principal_amount or amount)
                    if action == "LEND":
                        cash_probe -= principal + fee
                    else:
                        cash_probe += amount - fee
                else:
                    if side == "BUY":
                        cash_probe -= amount + fee
                    else:
                        cash_probe += amount - fee
                min_cash_probe = min(min_cash_probe, cash_probe)

        deficit_topup = max(0.0, -min_cash_probe)
        if strategy_flows:
            cash = float(deficit_topup)
        else:
            cash = float(max(inferred_initial_cash, deficit_topup))
        initial_equity_base = float(cash)
    nav_twr = 1.0
    nav_dietz = 1.0
    prev_equity: float | None = None
    prev_cash: float | None = None
    prev_day: dt.date | None = None
    alias_map: dict[str, str] = {}

    for alias in db.query(LiveSymbolAlias).all():
        alias_map[_norm_code(alias.old_code)] = _norm_code(alias.new_code)

    def _keys_for_code(target_code: str) -> list[tuple[str, int]]:
        return [k for k in lots.keys() if k[0] == target_code]

    def _round_keys_for_code(target_code: str) -> list[tuple[str, int]]:
        return [k for k in rounds_open.keys() if k[0] == target_code]

    def _repo_keys_for_code(target_code: str) -> list[tuple[str, int]]:
        return [k for k in repo_lots.keys() if k[0] == target_code]

    def _qty_for_code(target_code: str) -> float:
        return float(
            sum(
                float(lot["qty"])
                for k in _keys_for_code(target_code)
                for lot in lots.get(k, [])
            )
        )

    def _lots_for_code(target_code: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for k in _keys_for_code(target_code):
            out.extend(lots.get(k, []))
        return out

    def _repo_lots_for_code(target_code: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for k in _repo_keys_for_code(target_code):
            out.extend(repo_lots.get(k, []))
        return out

    days = trading_days(start, end)
    for day in days:
        fees_day = 0.0
        repo_fee_day = 0.0
        repo_carry_day = 0.0
        external_flow = 0.0

        for ev in events_by_day.get(day, []):
            code = _apply_symbol_alias(_norm_code(ev.code), alias_map)
            event_type = str(ev.event_type or "").strip().lower()
            if event_type == "cash_dividend":
                cps = float(ev.cash_per_share or 0.0)
                if cps > 0.0:
                    qty = _qty_for_code(code)
                    cash += qty * cps
            elif event_type in {"split", "share_conversion"}:
                fac = float(ev.ratio_factor or 0.0)
                if fac > 0.0:
                    for key in _keys_for_code(code):
                        for lot in lots.get(key, []):
                            lot_qty = float(lot["qty"])
                            lot["qty"] = lot_qty * fac
                            lot["unit_cost"] = float(lot["unit_cost"]) / fac
            elif event_type == "code_change":
                new_code = _norm_code(ev.new_code or "")
                if new_code:
                    alias_map[code] = new_code
                    for old_key in list(_keys_for_code(code)):
                        new_key = (new_code, old_key[1])
                        if new_key == old_key:
                            continue
                        lots[new_key].extend(lots.pop(old_key))
                    for old_key in list(_repo_keys_for_code(code)):
                        new_key = (new_code, old_key[1])
                        if new_key == old_key:
                            continue
                        repo_lots[new_key].extend(repo_lots.pop(old_key))
                    for old_key in list(_round_keys_for_code(code)):
                        st_old = rounds_open.pop(old_key)
                        new_key = (new_code, old_key[1])
                        if new_key == old_key:
                            rounds_open[old_key] = st_old
                            continue
                        st_old.code = new_code
                        if new_key not in rounds_open:
                            rounds_open[new_key] = st_old
                        else:
                            st_new = rounds_open[new_key]
                            st_new.open_date = min(st_new.open_date, st_old.open_date)
                            st_new.buy_count += st_old.buy_count
                            st_new.sell_count += st_old.sell_count
                            st_new.buy_qty += st_old.buy_qty
                            st_new.sell_qty += st_old.sell_qty
                            st_new.buy_amount += st_old.buy_amount
                            st_new.sell_amount += st_old.sell_amount
                            st_new.realized_pnl += st_old.realized_pnl
                            st_new.total_fee += st_old.total_fee
                            if st_old.legs:
                                st_new.ensure_legs().extend(st_old.legs)
                    db.add(
                        LiveSymbolAlias(
                            old_code=code,
                            new_code=new_code,
                            effective_date=day,
                            notes=ev.notes,
                        )
                    )

        if scope.scope_type == "strategy":
            for f in strategy_flow_by_day.get(day, []):
                amt = float(f.amount)
                cash += amt
                external_flow += amt
        else:
            for f in account_flow_by_day.get(day, []):
                flow_type = str(f.flow_type or "").strip().lower()
                amt = float(f.amount)
                if flow_type.startswith("transfer_"):
                    # internal movement between account and strategy, excluded from account equity.
                    continue
                cash += amt
                external_flow += amt

        day_trades = sorted(
            trade_by_day.get(day, []),
            key=lambda x: (str(x.trade_time or ""), int(x.id)),
        )
        for t in day_trades:
            code = _apply_symbol_alias(_norm_code(t.code), alias_map)
            holder_id = int(t.shareholder_account_id)
            key = (code, holder_id)
            side = _norm_side(t.side)
            strategy_type = strategy_type_by_id.get(int(t.strategy_id), "etf_spot")
            qty = float(t.quantity)
            px = float(t.price)
            fee = float(t.fee or 0.0)
            amount = float(t.amount or (qty * px))
            if qty <= 0:
                raise HTTPException(
                    status_code=400, detail=f"invalid quantity in trade {t.id}"
                )
            if strategy_type == "bond_repo":
                detail = repo_detail_by_trade_id.get(int(t.id))
                if detail is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"missing repo detail for trade {t.id}",
                    )
                action = _norm_repo_action(detail.repo_action, side=side)
                principal = float(detail.principal_amount or amount)
                annual_rate_pct = float(detail.annual_rate_pct)
                interest_days = int(detail.interest_days)
                day_count_basis = int(detail.day_count_basis or 365)
                open_trade_id = (
                    int(detail.open_trade_id)
                    if detail.open_trade_id is not None
                    else None
                )
                if principal <= 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"invalid repo principal for trade {t.id}",
                    )
                if action == "LEND":
                    cash -= principal + fee
                    repo_lots[key].append(
                        {
                            "principal": principal,
                            "fee_open_remain": fee,
                            "annual_rate_pct": annual_rate_pct,
                            "interest_days": interest_days,
                            "day_count_basis": day_count_basis,
                            "open_trade_id": int(t.id),
                            "open_date": t.trade_date,
                            "name": t.name,
                            "shareholder_account_id": holder_id,
                        }
                    )
                    fees_day += fee
                    repo_fee_day += fee
                    if key not in rounds_open:
                        round_no_by_code[code] += 1
                        rounds_open[key] = _RoundState(
                            round_no=round_no_by_code[code],
                            open_date=t.trade_date,
                            code=code,
                            name=t.name,
                        )
                    st = rounds_open[key]
                    st.buy_count += 1
                    st.buy_qty += principal
                    st.buy_amount += principal
                    st.total_fee += fee
                    st.ensure_legs().append(
                        {
                            "trade_id": int(t.id),
                            "side": "BUY",
                            "quantity": principal,
                            "price": annual_rate_pct,
                            "fee": fee,
                            "trade_date": t.trade_date,
                            "trade_time": _norm_time(t.trade_time),
                        }
                    )
                else:
                    available = float(
                        sum(float(x["principal"]) for x in repo_lots.get(key, []))
                    )
                    if available + 1e-9 < principal:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                "repo close principal exceeds position for "
                                f"code={code}, shareholder_account_id={holder_id}, trade_id={t.id}"
                            ),
                        )
                    remain = principal
                    interest_total = 0.0
                    buy_fee_out = 0.0
                    fifo = repo_lots.get(key, [])
                    close_income_total = amount - principal
                    close_income_scale = (
                        close_income_total / principal if principal > 1e-12 else 0.0
                    )

                    def _next_lot() -> dict[str, Any] | None:
                        for lot in fifo:
                            if float(lot["principal"]) <= 1e-12:
                                continue
                            if open_trade_id is not None and int(
                                lot["open_trade_id"]
                            ) != int(open_trade_id):
                                continue
                            return lot
                        return None

                    while remain > 1e-12:
                        top = _next_lot()
                        if top is None:
                            raise HTTPException(
                                status_code=400,
                                detail=(
                                    "repo close cannot match open lot for "
                                    f"code={code}, shareholder_account_id={holder_id}, trade_id={t.id}"
                                ),
                            )
                        top_principal = float(top["principal"])
                        take = min(remain, top_principal)
                        lot_rate = float(top["annual_rate_pct"])
                        lot_days = int(top["interest_days"])
                        lot_basis = int(top["day_count_basis"])
                        if abs(close_income_total) > 1e-12:
                            interest_take = take * close_income_scale
                        else:
                            effective_days = (
                                lot_days
                                if lot_days > 0
                                else max((day - top["open_date"]).days + 1, 1)
                            )
                            interest_take = (
                                take * lot_rate / 100.0 * effective_days / lot_basis
                            )
                        elapsed_prev = (
                            max(
                                0, min((prev_day - top["open_date"]).days + 1, lot_days)
                            )
                            if lot_days > 0
                            else max((prev_day - top["open_date"]).days + 1, 0)
                            if prev_day is not None
                            else 0
                        )
                        accrued_prev_take = (
                            take * lot_rate / 100.0 * float(elapsed_prev) / lot_basis
                        )
                        top_fee = float(top.get("fee_open_remain", 0.0) or 0.0)
                        fee_take = (
                            top_fee * (take / top_principal)
                            if top_principal > 1e-12 and top_fee > 0
                            else 0.0
                        )
                        interest_total += interest_take
                        repo_carry_day += interest_take - accrued_prev_take
                        buy_fee_out += fee_take
                        top["fee_open_remain"] = max(0.0, top_fee - fee_take)
                        top["principal"] = top_principal - take
                        remain -= take
                        if float(top["principal"]) <= 1e-12:
                            fifo.remove(top)
                    realized = interest_total - buy_fee_out - fee
                    cash += amount - fee
                    fees_day += fee
                    repo_fee_day += fee
                    if key not in rounds_open:
                        round_no_by_code[code] += 1
                        rounds_open[key] = _RoundState(
                            round_no=round_no_by_code[code],
                            open_date=t.trade_date,
                            code=code,
                            name=t.name,
                        )
                    st = rounds_open[key]
                    st.sell_count += 1
                    st.sell_qty += principal
                    st.sell_amount += principal
                    st.realized_pnl += realized
                    st.total_fee += fee
                    st.ensure_legs().append(
                        {
                            "trade_id": int(t.id),
                            "side": "SELL",
                            "quantity": principal,
                            "price": annual_rate_pct,
                            "fee": fee,
                            "trade_date": t.trade_date,
                            "trade_time": _norm_time(t.trade_time),
                        }
                    )
                    if (
                        float(
                            sum(float(x["principal"]) for x in repo_lots.get(key, []))
                        )
                        <= 1e-12
                    ):
                        buy_amt = float(st.buy_amount)
                        round_row = {
                            "scope_type": scope.scope_type,
                            "scope_id": scope.scope_id,
                            "account_id": scope.account_id,
                            "strategy_id": scope.strategy_id,
                            "round_no": int(st.round_no),
                            "code": code,
                            "name": st.name or "",
                            "open_date": st.open_date,
                            "close_date": day,
                            "buy_count": int(st.buy_count),
                            "sell_count": int(st.sell_count),
                            "buy_qty": float(st.buy_qty),
                            "sell_qty": float(st.sell_qty),
                            "avg_buy_price": None,
                            "avg_sell_price": None,
                            "realized_pnl": float(st.realized_pnl),
                            "return_rate": (
                                float(st.realized_pnl / buy_amt)
                                if buy_amt > 1e-12
                                else None
                            ),
                            "total_fee": float(st.total_fee),
                            "legs": st.legs or [],
                        }
                        closed_round_rows.append(round_row)
                        rounds_open.pop(key, None)
                continue
            if side == "BUY":
                gross = amount + fee
                cash -= gross
                # Keep holding cost_price on pure execution price (no fee allocation),
                # while tracking remaining buy-side fee per lot for PnL attribution.
                unit_cost = amount / qty
                lots[key].append(
                    {
                        "qty": qty,
                        "unit_cost": unit_cost,
                        "fee_remain": fee,
                        "trade_id": int(t.id),
                        "trade_date": t.trade_date,
                        "trade_time": t.trade_time,
                        "name": t.name,
                        "shareholder_account_id": holder_id,
                    }
                )
                fees_day += fee
                if key not in rounds_open:
                    round_no_by_code[code] += 1
                    rounds_open[key] = _RoundState(
                        round_no=round_no_by_code[code],
                        open_date=t.trade_date,
                        code=code,
                        name=t.name,
                    )
                st = rounds_open[key]
                st.buy_count += 1
                st.buy_qty += qty
                st.buy_amount += amount
                st.total_fee += fee
                st.ensure_legs().append(
                    {
                        "trade_id": int(t.id),
                        "side": side,
                        "quantity": qty,
                        "price": px,
                        "fee": fee,
                        "trade_date": t.trade_date,
                        "trade_time": _norm_time(t.trade_time),
                    }
                )
            else:
                available = float(sum(float(x["qty"]) for x in lots.get(key, [])))
                if available + 1e-9 < qty:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "sell quantity exceeds position for "
                            f"code={code}, shareholder_account_id={holder_id}, trade_id={t.id}"
                        ),
                    )
                remain = qty
                cost_out = 0.0
                buy_fee_out = 0.0
                fifo = lots.get(key, [])
                while remain > 1e-12 and fifo:
                    top = fifo[0]
                    top_qty = float(top["qty"])
                    take = min(remain, top_qty)
                    cost_out += take * float(top["unit_cost"])
                    top_fee = float(top.get("fee_remain", 0.0) or 0.0)
                    fee_take = (
                        top_fee * (take / top_qty)
                        if top_qty > 1e-12 and top_fee > 0
                        else 0.0
                    )
                    buy_fee_out += fee_take
                    top["fee_remain"] = max(0.0, top_fee - fee_take)
                    top["qty"] = top_qty - take
                    remain -= take
                    if float(top["qty"]) <= 1e-12:
                        fifo.pop(0)
                proceeds = amount - fee
                # Realized trade PnL includes both allocated buy fees and sell fees.
                realized = amount - cost_out - buy_fee_out - fee
                cash += proceeds
                fees_day += fee
                if key not in rounds_open:
                    round_no_by_code[code] += 1
                    rounds_open[key] = _RoundState(
                        round_no=round_no_by_code[code],
                        open_date=t.trade_date,
                        code=code,
                        name=t.name,
                    )
                st = rounds_open[key]
                st.sell_count += 1
                st.sell_qty += qty
                st.sell_amount += amount
                st.realized_pnl += realized
                st.total_fee += fee
                st.ensure_legs().append(
                    {
                        "trade_id": int(t.id),
                        "side": side,
                        "quantity": qty,
                        "price": px,
                        "fee": fee,
                        "trade_date": t.trade_date,
                        "trade_time": _norm_time(t.trade_time),
                    }
                )
                if float(sum(float(x["qty"]) for x in lots.get(key, []))) <= 1e-12:
                    buy_amt = float(st.buy_amount)
                    round_row = {
                        "scope_type": scope.scope_type,
                        "scope_id": scope.scope_id,
                        "account_id": scope.account_id,
                        "strategy_id": scope.strategy_id,
                        "round_no": int(st.round_no),
                        "code": code,
                        "name": st.name or "",
                        "open_date": st.open_date,
                        "close_date": day,
                        "buy_count": int(st.buy_count),
                        "sell_count": int(st.sell_count),
                        "buy_qty": float(st.buy_qty),
                        "sell_qty": float(st.sell_qty),
                        "avg_buy_price": (
                            float(st.buy_amount / st.buy_qty)
                            if st.buy_qty > 0
                            else None
                        ),
                        "avg_sell_price": (
                            float(st.sell_amount / st.sell_qty)
                            if st.sell_qty > 0
                            else None
                        ),
                        "realized_pnl": float(st.realized_pnl),
                        "return_rate": (
                            float(st.realized_pnl / buy_amt)
                            if buy_amt > 1e-12
                            else None
                        ),
                        "total_fee": float(st.total_fee),
                        "legs": st.legs or [],
                    }
                    closed_round_rows.append(round_row)
                    rounds_open.pop(key, None)

        for code, holder_id in list(repo_lots.keys()):
            for lot in repo_lots.get((code, holder_id), []):
                principal = float(lot["principal"])
                if principal <= 1e-12:
                    continue
                lot_days = int(lot["interest_days"])
                lot_basis = int(lot["day_count_basis"])
                lot_rate = float(lot["annual_rate_pct"])
                if lot_days > 0:
                    elapsed_today = lot_days
                else:
                    elapsed_today = max((day - lot["open_date"]).days + 1, 0)
                if prev_day is None:
                    elapsed_prev = 0
                else:
                    if lot_days > 0:
                        elapsed_prev = lot_days if prev_day >= lot["open_date"] else 0
                    else:
                        elapsed_prev = max((prev_day - lot["open_date"]).days + 1, 0)
                if elapsed_today <= elapsed_prev:
                    continue
                repo_carry_day += (
                    principal
                    * lot_rate
                    / 100.0
                    * float(elapsed_today - elapsed_prev)
                    / float(lot_basis)
                )

        account_id = scope.account_id
        strategy_id = scope.strategy_id
        market_value = 0.0
        today_holdings: list[LiveHoldingSnapshot] = []
        stock_active_codes = {
            code
            for code, holder_id in lots.keys()
            if sum(float(x["qty"]) for x in lots.get((code, holder_id), [])) > 1e-12
        }
        repo_active_codes = {
            code
            for code, holder_id in repo_lots.keys()
            if sum(float(x["principal"]) for x in repo_lots.get((code, holder_id), []))
            > 1e-12
        }
        active_codes = sorted(stock_active_codes | repo_active_codes)
        for code in active_codes:
            stock_code_lots = _lots_for_code(code)
            repo_code_lots = _repo_lots_for_code(code)
            if repo_code_lots and not stock_code_lots:
                qty_total = float(sum(float(x["principal"]) for x in repo_code_lots))
                cost_value = qty_total
                cost_price = None
                mpx = None
                stale_days = None
                price_missing = False
                fee_open = float(
                    sum(
                        float(x.get("fee_open_remain", 0.0) or 0.0)
                        for x in repo_code_lots
                    )
                )
                accrued_interest = 0.0
                for lot in repo_code_lots:
                    principal = float(lot["principal"])
                    lot_days = int(lot["interest_days"])
                    elapsed = (
                        lot_days
                        if lot_days > 0
                        else max((day - lot["open_date"]).days + 1, 0)
                    )
                    accrued_interest += (
                        principal
                        * float(lot["annual_rate_pct"])
                        / 100.0
                        * float(elapsed)
                        / float(lot["day_count_basis"])
                    )
                mv = float(cost_value + accrued_interest)
                pnl = float(accrued_interest - fee_open)
                pnl_rate = (pnl / cost_value) if cost_value > 1e-12 else None
                hold_name = str(repo_code_lots[-1].get("name", "") or "")
            else:
                code_lots = stock_code_lots
                qty_total = float(sum(float(x["qty"]) for x in code_lots))
                cost_value = float(
                    sum(float(x["qty"]) * float(x["unit_cost"]) for x in code_lots)
                )
                fee_open = float(
                    sum(float(x.get("fee_remain", 0.0) or 0.0) for x in code_lots)
                )
                cost_price = (
                    float(cost_value / qty_total) if qty_total > 1e-12 else None
                )
                mpx, mpx_day = _price_on_or_before(price_map, code, day)
                stale_days = (day - mpx_day).days if mpx_day is not None else None
                price_missing = mpx is None
                mv = float(qty_total * mpx) if mpx is not None else None
                pnl = (mv - cost_value - fee_open) if mv is not None else None
                pnl_rate = (
                    (pnl / cost_value)
                    if (pnl is not None and cost_value > 1e-12)
                    else None
                )
                hold_name = str(code_lots[-1].get("name", "") if code_lots else "")
            if mv is not None:
                market_value += mv
            today_holdings.append(
                LiveHoldingSnapshot(
                    snapshot_date=day,
                    scope_type=scope.scope_type,
                    scope_id=scope.scope_id,
                    account_id=account_id,
                    strategy_id=strategy_id,
                    code=code,
                    name=hold_name,
                    quantity=qty_total,
                    cost_price=cost_price,
                    market_price=mpx,
                    cost_value=cost_value,
                    market_value=mv,
                    pnl_amount=pnl,
                    pnl_rate=pnl_rate,
                    price_missing=bool(price_missing),
                    stale_days=stale_days,
                )
            )
        for hs in today_holdings:
            db.add(hs)

        equity = float(cash + market_value)
        daily_twr: float | None = None
        daily_dietz: float | None = None
        selection = 0.0
        timing = 0.0
        position = 0.0
        cost_drag = 0.0
        cash_drag = 0.0
        repo_carry = 0.0
        repo_fee_drag = 0.0
        if prev_equity is not None and prev_equity > 1e-12:
            daily_twr = float((equity - prev_equity - external_flow) / prev_equity)
            denom = float(prev_equity + 0.5 * external_flow)
            if abs(denom) <= 1e-12:
                daily_dietz = daily_twr
            else:
                daily_dietz = float((equity - prev_equity - external_flow) / denom)
            nav_twr *= 1.0 + daily_twr
            nav_dietz *= 1.0 + daily_dietz

            if prev_day is not None and codes:
                rets_universe: list[float] = []
                rets_held: list[float] = []
                held_val_weights: list[tuple[float, float]] = []
                prev_market = 0.0
                for code in codes:
                    p0, _ = _price_on_or_before(price_map, code, prev_day)
                    p1, _ = _price_on_or_before(price_map, code, day)
                    if p0 is None or p1 is None or p0 <= 0:
                        continue
                    r = float(p1 / p0 - 1.0)
                    rets_universe.append(r)
                    qty_prev = _qty_for_code(code)
                    if qty_prev <= 1e-12:
                        continue
                    val_prev = qty_prev * p0
                    prev_market += val_prev
                    rets_held.append(r)
                    held_val_weights.append((val_prev, r))

                u_ret = float(np.mean(rets_universe)) if rets_universe else 0.0
                invested_weight = (
                    float(prev_market / prev_equity) if prev_equity > 0 else 0.0
                )
                cash_weight = (
                    float(max(0.0, (prev_cash or 0.0) / prev_equity))
                    if prev_equity > 0
                    else 0.0
                )
                if rets_held and invested_weight > 0:
                    eq_hold_ret = float(np.mean(rets_held))
                    weighted_hold_ret = float(
                        sum((v / prev_equity) * r for v, r in held_val_weights)
                    )
                    selection = float(invested_weight * (eq_hold_ret - u_ret))
                    position = float(weighted_hold_ret - invested_weight * eq_hold_ret)
                else:
                    selection = 0.0
                    position = 0.0
                repo_carry = float(repo_carry_day / prev_equity)
                repo_fee_drag = float(-repo_fee_day / prev_equity)
                cost_drag = float(-(fees_day - repo_fee_day) / prev_equity)
                cash_drag = float(-cash_weight * u_ret)
                timing = float(
                    (daily_twr or 0.0)
                    - (
                        selection
                        + position
                        + cost_drag
                        + cash_drag
                        + repo_carry
                        + repo_fee_drag
                    )
                )
        elif prev_equity is None:
            # First-day return should be measured against reconstructed starting capital
            # when strategy replay has an inferred/top-up base.
            if initial_equity_base > 1e-12:
                daily_twr = float(
                    (equity - initial_equity_base - external_flow) / initial_equity_base
                )
                denom = float(initial_equity_base + 0.5 * external_flow)
                if abs(denom) <= 1e-12:
                    daily_dietz = daily_twr
                else:
                    daily_dietz = float(
                        (equity - initial_equity_base - external_flow) / denom
                    )
                nav_twr *= 1.0 + daily_twr
                nav_dietz *= 1.0 + daily_dietz
                # No previous-day decomposition context on day 1; assign to timing
                # so attribution period rebuild remains consistent with TWR sum.
                timing = float(daily_twr)
            else:
                # No usable baseline (e.g., pure account scope without prior capital).
                daily_twr = 0.0
                daily_dietz = 0.0

        db.add(
            LiveNavDaily(
                nav_date=day,
                scope_type=scope.scope_type,
                scope_id=scope.scope_id,
                account_id=account_id,
                strategy_id=strategy_id,
                equity=equity,
                cash=float(cash),
                market_value=float(market_value),
                external_flow=float(external_flow),
                trading_fee=float(fees_day),
                daily_return_twr=daily_twr,
                daily_return_dietz=daily_dietz,
                nav_twr=float(nav_twr),
                nav_dietz=float(nav_dietz),
                selection_return=float(selection),
                timing_return=float(timing),
                position_return=float(position),
                cost_drag_return=float(cost_drag),
                cash_drag_return=float(cash_drag),
                repo_carry_return=float(repo_carry),
                repo_fee_drag_return=float(repo_fee_drag),
            )
        )
        prev_equity = float(equity)
        prev_cash = float(cash)
        prev_day = day

    for row in closed_round_rows:
        rr = LiveClosedRound(
            scope_type=row["scope_type"],
            scope_id=row["scope_id"],
            account_id=row["account_id"],
            strategy_id=row["strategy_id"],
            round_no=row["round_no"],
            code=row["code"],
            name=row["name"],
            open_date=row["open_date"],
            close_date=row["close_date"],
            buy_count=row["buy_count"],
            sell_count=row["sell_count"],
            buy_qty=row["buy_qty"],
            sell_qty=row["sell_qty"],
            avg_buy_price=row["avg_buy_price"],
            avg_sell_price=row["avg_sell_price"],
            realized_pnl=row["realized_pnl"],
            return_rate=row["return_rate"],
            total_fee=row["total_fee"],
        )
        db.add(rr)
        db.flush()
        for idx, leg in enumerate(row["legs"], start=1):
            closed_round_legs.append(
                {
                    "round_id": int(rr.id),
                    "sort_order": idx,
                    "trade_id": int(leg["trade_id"]),
                    "side": str(leg["side"]),
                    "quantity": float(leg["quantity"]),
                    "price": float(leg["price"]),
                    "fee": float(leg["fee"]),
                    "trade_date": leg["trade_date"],
                    "trade_time": str(leg["trade_time"]),
                }
            )
    for x in closed_round_legs:
        db.add(
            LiveClosedRoundLeg(
                round_id=x["round_id"],
                sort_order=x["sort_order"],
                trade_id=x["trade_id"],
                side=x["side"],
                quantity=x["quantity"],
                price=x["price"],
                fee=x["fee"],
                trade_date=x["trade_date"],
                trade_time=x["trade_time"],
            )
        )
    return {
        "scope_type": scope.scope_type,
        "scope_id": scope.scope_id,
        "days": len(days),
        "closed_rounds": len(closed_round_rows),
    }


def _serialize_account(x: LiveAccount) -> LiveAccountOut:
    return LiveAccountOut(
        id=int(x.id),
        name=str(x.name),
        base_ccy=str(x.base_ccy),
        initial_cash=float(x.initial_cash),
        notes=x.notes,
        created_at=_dt_s(x.created_at),
    )


def _serialize_strategy(
    x: LiveStrategy,
    *,
    strategy_type: str = "etf_spot",
    capital_mode: str = "segregated",
) -> LiveStrategyOut:
    norm_type = _norm_strategy_type(strategy_type)
    return LiveStrategyOut(
        id=int(x.id),
        account_id=int(x.account_id),
        name=str(x.name),
        strategy_type=norm_type,
        capital_mode=_norm_capital_mode(capital_mode, strategy_type=norm_type),
        notes=x.notes,
        created_at=_dt_s(x.created_at),
    )


def _serialize_shareholder(x: LiveShareholderAccount) -> LiveShareholderAccountOut:
    return LiveShareholderAccountOut(
        id=int(x.id),
        account_id=int(x.account_id),
        shareholder_account=str(x.shareholder_account),
        notes=x.notes,
        created_at=_dt_s(x.created_at),
    )


def _repo_detail_map(
    db: Session, trade_ids: list[int]
) -> dict[int, LiveRepoTradeDetail]:
    if not trade_ids:
        return {}
    rows = (
        db.query(LiveRepoTradeDetail)
        .filter(LiveRepoTradeDetail.trade_id.in_(trade_ids))
        .all()
    )
    return {int(x.trade_id): x for x in rows}


def _serialize_trade(
    x: LiveTrade, *, repo_detail: LiveRepoTradeDetail | None = None
) -> LiveTradeOut:
    return LiveTradeOut(
        id=int(x.id),
        account_id=int(x.account_id),
        strategy_id=int(x.strategy_id),
        shareholder_account_id=int(x.shareholder_account_id),
        code=str(x.code),
        name=str(x.name or ""),
        trade_date=x.trade_date.isoformat(),
        trade_time=str(x.trade_time),
        side=str(x.side),
        price=float(x.price),
        quantity=float(x.quantity),
        fee=float(x.fee),
        amount=float(x.amount),
        repo_action=repo_detail.repo_action if repo_detail else None,
        repo_principal_amount=(
            float(repo_detail.principal_amount) if repo_detail is not None else None
        ),
        repo_annual_rate_pct=(
            float(repo_detail.annual_rate_pct) if repo_detail is not None else None
        ),
        repo_interest_days=(
            (
                int(repo_detail.interest_days)
                if int(repo_detail.interest_days) > 0
                else None
            )
            if repo_detail is not None
            else None
        ),
        repo_day_count_basis=(
            int(repo_detail.day_count_basis) if repo_detail is not None else None
        ),
        repo_open_trade_id=(
            int(repo_detail.open_trade_id)
            if (repo_detail is not None and repo_detail.open_trade_id is not None)
            else None
        ),
        idempotency_key=x.idempotency_key,
        broker_trade_no=x.broker_trade_no,
        notes=x.notes,
        created_at=_dt_s(x.created_at),
    )


@router.post("/accounts", response_model=LiveAccountOut)
def live_create_account(
    payload: LiveAccountCreateRequest, db: Session = Depends(get_session)
):
    row = LiveAccount(
        name=payload.name.strip(),
        base_ccy=payload.base_ccy.strip().upper(),
        initial_cash=float(payload.initial_cash),
        notes=payload.notes,
    )
    db.add(row)
    try:
        db.flush()
    except IntegrityError as exc:
        raise HTTPException(
            status_code=400, detail="account name already exists"
        ) from exc
    if float(payload.initial_cash) > 0.0:
        db.add(
            LiveAccountCashflow(
                account_id=int(row.id),
                flow_date=shift_to_trading_day(dt.date.today(), shift="prev"),
                amount=float(payload.initial_cash),
                flow_type="deposit",
                notes="initial_cash",
            )
        )
    db.flush()
    _replay_scope(
        db,
        _Scope(
            scope_type="account",
            scope_id=int(row.id),
            account_id=int(row.id),
            strategy_id=None,
        ),
    )
    return _serialize_account(row)


@router.get("/accounts", response_model=list[LiveAccountOut])
def live_list_accounts(db: Session = Depends(get_session)):
    rows = db.query(LiveAccount).order_by(LiveAccount.id.asc()).all()
    return [_serialize_account(x) for x in rows]


@router.patch("/accounts/{account_id}", response_model=LiveAccountOut)
def live_update_account(
    account_id: int,
    payload: LiveAccountUpdateRequest,
    db: Session = Depends(get_session),
):
    row = db.query(LiveAccount).filter(LiveAccount.id == account_id).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="account not found")
    if payload.name is not None:
        row.name = payload.name.strip()
    if payload.notes is not None:
        row.notes = payload.notes
    try:
        db.flush()
    except IntegrityError as exc:
        raise HTTPException(
            status_code=400, detail="account name already exists"
        ) from exc
    return _serialize_account(row)


@router.delete("/accounts/{account_id}")
def live_delete_account(account_id: int, db: Session = Depends(get_session)):
    row = db.query(LiveAccount).filter(LiveAccount.id == account_id).one_or_none()
    if row is None:
        return {"ok": True}
    db.delete(row)
    return {"ok": True}


@router.post(
    "/accounts/{account_id}/shareholders", response_model=LiveShareholderAccountOut
)
def live_add_shareholder(
    account_id: int,
    payload: LiveShareholderAccountCreateRequest,
    db: Session = Depends(get_session),
):
    account = db.query(LiveAccount).filter(LiveAccount.id == account_id).one_or_none()
    if account is None:
        raise HTTPException(status_code=404, detail="account not found")
    row = LiveShareholderAccount(
        account_id=account_id,
        shareholder_account=payload.shareholder_account.strip(),
        notes=payload.notes,
    )
    db.add(row)
    try:
        db.flush()
    except IntegrityError as exc:
        raise HTTPException(
            status_code=400, detail="shareholder account already exists"
        ) from exc
    return _serialize_shareholder(row)


@router.get(
    "/accounts/{account_id}/shareholders",
    response_model=list[LiveShareholderAccountOut],
)
def live_list_shareholders(account_id: int, db: Session = Depends(get_session)):
    rows = (
        db.query(LiveShareholderAccount)
        .filter(LiveShareholderAccount.account_id == account_id)
        .order_by(LiveShareholderAccount.id.asc())
        .all()
    )
    return [_serialize_shareholder(x) for x in rows]


@router.post("/accounts/{account_id}/strategies", response_model=LiveStrategyOut)
def live_create_strategy(
    account_id: int,
    payload: LiveStrategyCreateRequest,
    db: Session = Depends(get_session),
):
    account = db.query(LiveAccount).filter(LiveAccount.id == account_id).one_or_none()
    if account is None:
        raise HTTPException(status_code=404, detail="account not found")
    strategy_type = _norm_strategy_type(payload.strategy_type)
    capital_mode = _norm_capital_mode(payload.capital_mode, strategy_type=strategy_type)
    row = LiveStrategy(
        account_id=account_id, name=payload.name.strip(), notes=payload.notes
    )
    db.add(row)
    try:
        db.flush()
    except IntegrityError as exc:
        raise HTTPException(
            status_code=400, detail="strategy name already exists in account"
        ) from exc
    db.add(
        LiveStrategyProfile(
            strategy_id=int(row.id),
            strategy_type=strategy_type,
            capital_mode=capital_mode,
        )
    )
    db.flush()
    return _serialize_strategy(
        row, strategy_type=strategy_type, capital_mode=capital_mode
    )


@router.get("/accounts/{account_id}/strategies", response_model=list[LiveStrategyOut])
def live_list_strategies(account_id: int, db: Session = Depends(get_session)):
    rows = (
        db.query(LiveStrategy)
        .filter(LiveStrategy.account_id == account_id)
        .order_by(LiveStrategy.id.asc())
        .all()
    )
    profile_map = _strategy_profile_map(db, [int(x.id) for x in rows])
    out: list[LiveStrategyOut] = []
    for x in rows:
        stype, cmode = profile_map.get(int(x.id), ("etf_spot", "segregated"))
        out.append(_serialize_strategy(x, strategy_type=stype, capital_mode=cmode))
    return out


@router.patch("/strategies/{strategy_id}", response_model=LiveStrategyOut)
def live_update_strategy(
    strategy_id: int,
    payload: LiveStrategyUpdateRequest,
    db: Session = Depends(get_session),
):
    row = db.query(LiveStrategy).filter(LiveStrategy.id == strategy_id).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="strategy not found")
    if payload.name is not None:
        row.name = payload.name.strip()
    if payload.notes is not None:
        row.notes = payload.notes
    profile = (
        db.query(LiveStrategyProfile)
        .filter(LiveStrategyProfile.strategy_id == int(row.id))
        .one_or_none()
    )
    stype = (
        _norm_strategy_type(payload.strategy_type) if payload.strategy_type else None
    )
    if stype is not None or payload.capital_mode is not None:
        if profile is None:
            base_type = stype or "etf_spot"
            base_mode = _norm_capital_mode(
                payload.capital_mode, strategy_type=base_type
            )
            profile = LiveStrategyProfile(
                strategy_id=int(row.id),
                strategy_type=base_type,
                capital_mode=base_mode,
            )
            db.add(profile)
        if stype is not None:
            profile.strategy_type = stype
        profile.strategy_type = _norm_strategy_type(profile.strategy_type)
        profile.capital_mode = _norm_capital_mode(
            payload.capital_mode
            if payload.capital_mode is not None
            else profile.capital_mode,
            strategy_type=profile.strategy_type,
        )
    try:
        db.flush()
    except IntegrityError as exc:
        raise HTTPException(
            status_code=400, detail="strategy name already exists in account"
        ) from exc
    out_type, out_mode = _strategy_profile_for(db, int(row.id))
    return _serialize_strategy(row, strategy_type=out_type, capital_mode=out_mode)


@router.delete("/strategies/{strategy_id}")
def live_delete_strategy(strategy_id: int, db: Session = Depends(get_session)):
    row = db.query(LiveStrategy).filter(LiveStrategy.id == strategy_id).one_or_none()
    if row is None:
        return {"ok": True}
    (
        db.query(LiveStrategyProfile)
        .filter(LiveStrategyProfile.strategy_id == int(strategy_id))
        .delete(synchronize_session=False)
    )
    db.delete(row)
    return {"ok": True}


@router.post("/accounts/{account_id}/cashflows", response_model=LiveCashflowOut)
def live_add_account_cashflow(
    account_id: int,
    payload: LiveAccountCashflowCreateRequest,
    db: Session = Depends(get_session),
):
    if db.query(LiveAccount).filter(LiveAccount.id == account_id).one_or_none() is None:
        raise HTTPException(status_code=404, detail="account not found")
    row = LiveAccountCashflow(
        account_id=account_id,
        flow_date=_to_date(payload.flow_date),
        amount=float(payload.amount),
        flow_type=str(payload.flow_type).strip().lower(),
        transfer_id=payload.transfer_id,
        notes=payload.notes,
    )
    db.add(row)
    db.flush()
    _replay_scope(
        db,
        _Scope(
            scope_type="account",
            scope_id=account_id,
            account_id=account_id,
            strategy_id=None,
        ),
    )
    return LiveCashflowOut(
        id=int(row.id),
        account_id=account_id,
        strategy_id=None,
        flow_date=row.flow_date.isoformat(),
        amount=float(row.amount),
        flow_type=str(row.flow_type),
        transfer_id=row.transfer_id,
        notes=row.notes,
        created_at=_dt_s(row.created_at),
    )


@router.get("/accounts/{account_id}/cashflows", response_model=list[LiveCashflowOut])
def live_list_account_cashflows(account_id: int, db: Session = Depends(get_session)):
    rows = (
        db.query(LiveAccountCashflow)
        .filter(LiveAccountCashflow.account_id == account_id)
        .order_by(LiveAccountCashflow.flow_date.asc(), LiveAccountCashflow.id.asc())
        .all()
    )
    return [
        LiveCashflowOut(
            id=int(x.id),
            account_id=int(x.account_id),
            strategy_id=None,
            flow_date=x.flow_date.isoformat(),
            amount=float(x.amount),
            flow_type=str(x.flow_type),
            transfer_id=x.transfer_id,
            notes=x.notes,
            created_at=_dt_s(x.created_at),
        )
        for x in rows
    ]


@router.post("/strategies/{strategy_id}/cashflows", response_model=LiveCashflowOut)
def live_add_strategy_cashflow(
    strategy_id: int,
    payload: LiveStrategyCashflowCreateRequest,
    db: Session = Depends(get_session),
):
    strategy = (
        db.query(LiveStrategy).filter(LiveStrategy.id == strategy_id).one_or_none()
    )
    if strategy is None:
        raise HTTPException(status_code=404, detail="strategy not found")
    row = LiveStrategyCashflow(
        strategy_id=strategy_id,
        flow_date=_to_date(payload.flow_date),
        amount=float(payload.amount),
        flow_type=str(payload.flow_type).strip().lower(),
        transfer_id=payload.transfer_id,
        notes=payload.notes,
    )
    db.add(row)
    db.flush()
    _replay_scope(
        db,
        _Scope(
            scope_type="strategy",
            scope_id=int(strategy_id),
            account_id=int(strategy.account_id),
            strategy_id=int(strategy_id),
        ),
    )
    _replay_scope(
        db,
        _Scope(
            scope_type="account",
            scope_id=int(strategy.account_id),
            account_id=int(strategy.account_id),
            strategy_id=None,
        ),
    )
    return LiveCashflowOut(
        id=int(row.id),
        account_id=int(strategy.account_id),
        strategy_id=int(strategy_id),
        flow_date=row.flow_date.isoformat(),
        amount=float(row.amount),
        flow_type=str(row.flow_type),
        transfer_id=row.transfer_id,
        notes=row.notes,
        created_at=_dt_s(row.created_at),
    )


@router.get("/strategies/{strategy_id}/cashflows", response_model=list[LiveCashflowOut])
def live_list_strategy_cashflows(strategy_id: int, db: Session = Depends(get_session)):
    strategy = (
        db.query(LiveStrategy).filter(LiveStrategy.id == strategy_id).one_or_none()
    )
    if strategy is None:
        raise HTTPException(status_code=404, detail="strategy not found")
    rows = (
        db.query(LiveStrategyCashflow)
        .filter(LiveStrategyCashflow.strategy_id == strategy_id)
        .order_by(LiveStrategyCashflow.flow_date.asc(), LiveStrategyCashflow.id.asc())
        .all()
    )
    return [
        LiveCashflowOut(
            id=int(x.id),
            account_id=int(strategy.account_id),
            strategy_id=int(x.strategy_id),
            flow_date=x.flow_date.isoformat(),
            amount=float(x.amount),
            flow_type=str(x.flow_type),
            transfer_id=x.transfer_id,
            notes=x.notes,
            created_at=_dt_s(x.created_at),
        )
        for x in rows
    ]


@router.post("/accounts/{account_id}/strategy-transfer")
def live_strategy_transfer(
    account_id: int,
    payload: LiveStrategyTransferRequest,
    db: Session = Depends(get_session),
):
    strategy = (
        db.query(LiveStrategy)
        .filter(LiveStrategy.id == payload.strategy_id)
        .one_or_none()
    )
    if strategy is None or int(strategy.account_id) != int(account_id):
        raise HTTPException(status_code=404, detail="strategy not found in account")
    flow_day = _to_date(payload.flow_date)
    transfer_id = payload.transfer_id or f"tf_{uuid.uuid4().hex[:16]}"
    amt = float(payload.amount)
    direction = str(payload.direction or "to_strategy").strip().lower()
    if direction not in {"to_strategy", "from_strategy"}:
        raise HTTPException(
            status_code=400, detail="direction must be to_strategy|from_strategy"
        )
    if direction == "to_strategy":
        acc_amt = -amt
        str_amt = amt
        acc_type = "transfer_to_strategy"
        str_type = "transfer_in"
    else:
        acc_amt = amt
        str_amt = -amt
        acc_type = "transfer_from_strategy"
        str_type = "transfer_out"
    db.add(
        LiveAccountCashflow(
            account_id=account_id,
            flow_date=flow_day,
            amount=acc_amt,
            flow_type=acc_type,
            transfer_id=transfer_id,
            notes=payload.notes,
        )
    )
    db.add(
        LiveStrategyCashflow(
            strategy_id=int(strategy.id),
            flow_date=flow_day,
            amount=str_amt,
            flow_type=str_type,
            transfer_id=transfer_id,
            notes=payload.notes,
        )
    )
    db.flush()
    _replay_scope(
        db,
        _Scope(
            scope_type="strategy",
            scope_id=int(strategy.id),
            account_id=int(account_id),
            strategy_id=int(strategy.id),
        ),
    )
    _replay_scope(
        db,
        _Scope(
            scope_type="account",
            scope_id=account_id,
            account_id=account_id,
            strategy_id=None,
        ),
    )
    return {"ok": True, "transfer_id": transfer_id}


def _validate_trade_payload(
    payload: LiveTradeCreateRequest | LiveTradeUpdateRequest, db: Session
) -> tuple[int, int]:
    account = (
        db.query(LiveAccount).filter(LiveAccount.id == payload.account_id).one_or_none()
    )
    if account is None:
        raise HTTPException(status_code=404, detail="account not found")
    strategy = (
        db.query(LiveStrategy)
        .filter(LiveStrategy.id == payload.strategy_id)
        .one_or_none()
    )
    if strategy is None:
        raise HTTPException(status_code=404, detail="strategy not found")
    if int(strategy.account_id) != int(payload.account_id):
        raise HTTPException(
            status_code=400, detail="strategy does not belong to account"
        )
    holder = (
        db.query(LiveShareholderAccount)
        .filter(LiveShareholderAccount.id == payload.shareholder_account_id)
        .one_or_none()
    )
    if holder is None or int(holder.account_id) != int(payload.account_id):
        raise HTTPException(
            status_code=400, detail="shareholder account is invalid for account"
        )
    return int(account.id), int(strategy.id)


def _require_change_reason(raw_reason: str) -> str:
    reason = str(raw_reason or "").strip()
    if not reason:
        raise HTTPException(status_code=400, detail="reason is required")
    return reason


def _norm_repo_action(raw: str | None, *, side: str) -> str:
    if raw is None or str(raw).strip() == "":
        return "LEND" if side == "BUY" else "BUYBACK"
    s = str(raw).strip()
    su = s.upper()
    if su in {"OPEN", "LEND"} or s in {"融券"}:
        return "LEND"
    if su in {"CLOSE", "BUYBACK"} or s in {"融券购回"}:
        return "BUYBACK"
    raise HTTPException(
        status_code=400,
        detail="repo_action must be LEND|BUYBACK (or 融券|融券购回)",
    )


def _build_repo_detail_payload(
    payload: LiveTradeCreateRequest | LiveTradeUpdateRequest, *, side: str
) -> dict[str, Any]:
    action = _norm_repo_action(payload.repo_action, side=side)
    if action == "LEND" and side != "BUY":
        raise HTTPException(status_code=400, detail="repo LEND must use BUY side")
    if action == "BUYBACK" and side != "SELL":
        raise HTTPException(status_code=400, detail="repo BUYBACK must use SELL side")
    qty_lots = float(payload.quantity)
    if qty_lots <= 0:
        raise HTTPException(status_code=400, detail="repo quantity must be > 0")
    qty_int = int(round(qty_lots))
    if not math.isclose(qty_lots, float(qty_int), rel_tol=0.0, abs_tol=1e-9):
        raise HTTPException(
            status_code=400, detail="repo quantity must be a positive integer"
        )
    principal_from_qty = float(qty_int) * REPO_LOT_AMOUNT
    principal = (
        float(payload.repo_principal_amount)
        if payload.repo_principal_amount is not None
        else principal_from_qty
    )
    if principal <= 0:
        raise HTTPException(status_code=400, detail="repo principal_amount must be > 0")
    principal_lot_ratio = principal / REPO_LOT_AMOUNT
    if not math.isclose(
        principal_lot_ratio, round(principal_lot_ratio), rel_tol=0.0, abs_tol=1e-9
    ):
        raise HTTPException(
            status_code=400,
            detail="repo principal_amount must be a positive multiple of 1000",
        )
    if not math.isclose(principal, principal_from_qty, rel_tol=0.0, abs_tol=1e-9):
        raise HTTPException(
            status_code=400,
            detail="repo principal_amount must equal quantity * 1000",
        )
    annual_rate_pct = float(payload.price or 0.0)
    if annual_rate_pct <= 0:
        raise HTTPException(
            status_code=400, detail="repo price(annual rate) must be > 0"
        )
    raw_interest_days = payload.repo_interest_days
    if action == "LEND":
        if raw_interest_days is None:
            raise HTTPException(
                status_code=400, detail="repo interest_days is required for LEND"
            )
        interest_days = int(raw_interest_days)
        if interest_days <= 0:
            raise HTTPException(
                status_code=400, detail="repo interest_days must be >= 1"
            )
    else:
        if raw_interest_days is None:
            interest_days = 0
        else:
            interest_days = int(raw_interest_days)
            if interest_days <= 0:
                raise HTTPException(
                    status_code=400, detail="repo interest_days must be >= 1"
                )
    day_count_basis = 365
    open_trade_id = (
        int(payload.repo_open_trade_id)
        if payload.repo_open_trade_id is not None
        else None
    )
    return {
        "repo_action": action,
        "principal_amount": principal,
        "lot_quantity": float(qty_int),
        "annual_rate_pct": annual_rate_pct,
        "interest_days": interest_days,
        "day_count_basis": day_count_basis,
        "open_trade_id": open_trade_id,
    }


def _upsert_repo_trade_detail(
    db: Session, *, trade_id: int, detail_payload: dict[str, Any] | None
) -> LiveRepoTradeDetail | None:
    row = (
        db.query(LiveRepoTradeDetail)
        .filter(LiveRepoTradeDetail.trade_id == int(trade_id))
        .one_or_none()
    )
    if detail_payload is None:
        if row is not None:
            db.delete(row)
        return None
    if row is None:
        row = LiveRepoTradeDetail(trade_id=int(trade_id))
        db.add(row)
    row.repo_action = str(detail_payload["repo_action"])
    row.principal_amount = float(detail_payload["principal_amount"])
    row.annual_rate_pct = float(detail_payload["annual_rate_pct"])
    row.interest_days = int(detail_payload["interest_days"])
    row.day_count_basis = int(detail_payload["day_count_basis"])
    row.open_trade_id = detail_payload["open_trade_id"]
    return row


def _apply_trade_values(
    row: LiveTrade,
    payload: LiveTradeCreateRequest | LiveTradeUpdateRequest,
    *,
    db: Session,
    account_id: int,
    strategy_id: int,
) -> tuple[str, dict[str, Any] | None]:
    strategy_type = _strategy_type_for(db, int(strategy_id))
    side = _norm_side(payload.side)
    code = _norm_code(payload.code)
    trade_date = _to_date(payload.trade_date)
    trade_time = _norm_time(payload.trade_time)
    qty = float(payload.quantity)
    if qty <= 0:
        raise HTTPException(status_code=400, detail="quantity must be > 0")
    repo_detail: dict[str, Any] | None = None
    if strategy_type == "bond_repo":
        repo_detail = _build_repo_detail_payload(payload, side=side)
        if repo_detail["repo_action"] == "LEND" and not _is_repo_lend_time_allowed(
            trade_time
        ):
            raise HTTPException(
                status_code=400,
                detail="repo LEND trade_time must be between 09:30:00 and 15:30:00",
            )
        code, repo_name = _norm_repo_symbol(payload.code)
        qty = float(repo_detail["lot_quantity"])
        px = float(repo_detail["annual_rate_pct"])
        amount = float(
            payload.amount
            if payload.amount is not None
            else repo_detail["principal_amount"]
        )
        if amount <= 0:
            raise HTTPException(status_code=400, detail="repo amount must be > 0")
        if (
            repo_detail["repo_action"] == "LEND"
            and payload.amount is not None
            and not math.isclose(
                amount,
                float(repo_detail["principal_amount"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise HTTPException(
                status_code=400,
                detail="repo LEND amount must equal quantity * 1000",
            )
    else:
        if not _is_trade_time_allowed(trade_time):
            raise HTTPException(
                status_code=400,
                detail="trade_time must be between 09:00:00 and 15:00:00",
            )
        if (
            payload.repo_action is not None
            or payload.repo_principal_amount is not None
            or payload.repo_annual_rate_pct is not None
            or payload.repo_interest_days is not None
            or payload.repo_day_count_basis is not None
            or payload.repo_open_trade_id is not None
        ):
            raise HTTPException(
                status_code=400,
                detail="repo fields are only allowed for bond_repo strategy",
            )
        lot_ratio = qty / 100.0
        if not math.isclose(lot_ratio, round(lot_ratio), rel_tol=0.0, abs_tol=1e-9):
            raise HTTPException(
                status_code=400, detail="quantity must be a multiple of 100"
            )
        px = float(payload.price)
        amount = float(payload.amount if payload.amount is not None else (qty * px))
    if strategy_type == "bond_repo":
        fee = (
            0.0
            if repo_detail and repo_detail["repo_action"] == "BUYBACK"
            else _default_repo_trade_fee(amount)
        )
    else:
        raw_fee = float(payload.fee or 0.0)
        fee_default = _default_trade_fee(px, qty)
        fee = _round_fee_2(raw_fee if raw_fee > 0.0 else fee_default)

    row.account_id = account_id
    row.strategy_id = strategy_id
    row.shareholder_account_id = int(payload.shareholder_account_id)
    row.code = code
    row.name = repo_name if strategy_type == "bond_repo" else str(payload.name or "")
    row.trade_date = trade_date
    row.trade_time = trade_time
    row.side = side
    row.price = px
    row.quantity = qty
    row.fee = fee
    row.amount = amount
    row.broker_trade_no = payload.broker_trade_no
    row.notes = payload.notes
    return strategy_type, repo_detail


def _replay_touched_scopes(
    db: Session, *, strategy_ids: set[int], account_ids: set[int]
) -> None:
    for sid in sorted(strategy_ids):
        st = db.query(LiveStrategy).filter(LiveStrategy.id == sid).one_or_none()
        if st is None:
            continue
        _replay_scope(
            db,
            _Scope(
                scope_type="strategy",
                scope_id=int(st.id),
                account_id=int(st.account_id),
                strategy_id=int(st.id),
            ),
        )
    for aid in sorted(account_ids):
        _replay_scope(
            db,
            _Scope(
                scope_type="account", scope_id=aid, account_id=aid, strategy_id=None
            ),
        )


def _log_trade_audit(
    db: Session,
    *,
    trade_id: int,
    account_id: int,
    strategy_id: int,
    action: str,
    reason: str,
    snapshot: dict[str, Any] | None = None,
) -> None:
    snapshot_json = (
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True) if snapshot else None
    )
    db.add(
        LiveTradeAuditLog(
            trade_id=trade_id,
            account_id=account_id,
            strategy_id=strategy_id,
            action=action,
            reason=reason,
            snapshot_json=snapshot_json,
        )
    )


def _insert_trade(payload: LiveTradeCreateRequest, db: Session) -> LiveTrade:
    account_id, strategy_id = _validate_trade_payload(payload, db)
    row = LiveTrade(
        account_id=account_id,
        strategy_id=strategy_id,
        shareholder_account_id=int(payload.shareholder_account_id),
        code="",
        name="",
        trade_date=_to_date(payload.trade_date),
        trade_time="09:30:00",
        side="BUY",
        price=0.0,
        quantity=0.0,
        fee=0.0,
        amount=0.0,
        idempotency_key=payload.idempotency_key,
        broker_trade_no=None,
        notes=None,
    )
    _, repo_detail = _apply_trade_values(
        row,
        payload,
        db=db,
        account_id=account_id,
        strategy_id=strategy_id,
    )
    _validate_trade_funding_constraints(
        db,
        account_id=int(row.account_id),
        strategy_id=int(row.strategy_id),
        strategy_type=_strategy_type_for(db, int(row.strategy_id)),
        trade_date=row.trade_date,
        trade_time=str(row.trade_time),
        side=str(row.side),
        amount=float(row.amount),
        fee=float(row.fee or 0.0),
        repo_detail=repo_detail,
    )
    db.add(row)
    try:
        db.flush()
    except IntegrityError as exc:
        raise HTTPException(
            status_code=400, detail="duplicate idempotency_key or broker_trade_no"
        ) from exc
    _upsert_repo_trade_detail(db, trade_id=int(row.id), detail_payload=repo_detail)
    db.flush()
    return row


@router.post("/trades", response_model=LiveTradeOut)
def live_add_trade(payload: LiveTradeCreateRequest, db: Session = Depends(get_session)):
    row = _insert_trade(payload, db)
    _replay_touched_scopes(
        db,
        strategy_ids={int(row.strategy_id)},
        account_ids={int(row.account_id)},
    )
    repo_detail = (
        db.query(LiveRepoTradeDetail)
        .filter(LiveRepoTradeDetail.trade_id == int(row.id))
        .one_or_none()
    )
    return _serialize_trade(row, repo_detail=repo_detail)


@router.put("/trades/{trade_id}", response_model=LiveTradeOut)
def live_update_trade(
    trade_id: int, payload: LiveTradeUpdateRequest, db: Session = Depends(get_session)
):
    row = db.query(LiveTrade).filter(LiveTrade.id == trade_id).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="trade not found")
    reason = _require_change_reason(payload.reason)
    old_account_id = int(row.account_id)
    old_strategy_id = int(row.strategy_id)
    old_repo_detail = (
        db.query(LiveRepoTradeDetail)
        .filter(LiveRepoTradeDetail.trade_id == int(row.id))
        .one_or_none()
    )
    old_snapshot = _serialize_trade(row, repo_detail=old_repo_detail).model_dump()
    account_id, strategy_id = _validate_trade_payload(payload, db)
    _, repo_detail_payload = _apply_trade_values(
        row,
        payload,
        db=db,
        account_id=account_id,
        strategy_id=strategy_id,
    )
    _validate_trade_funding_constraints(
        db,
        account_id=int(row.account_id),
        strategy_id=int(row.strategy_id),
        strategy_type=_strategy_type_for(db, int(row.strategy_id)),
        trade_date=row.trade_date,
        trade_time=str(row.trade_time),
        side=str(row.side),
        amount=float(row.amount),
        fee=float(row.fee or 0.0),
        repo_detail=repo_detail_payload,
        exclude_trade_id=int(row.id),
        order_trade_id=int(row.id),
    )
    try:
        db.flush()
    except IntegrityError as exc:
        raise HTTPException(
            status_code=400, detail="duplicate broker_trade_no"
        ) from exc
    repo_detail = _upsert_repo_trade_detail(
        db, trade_id=int(row.id), detail_payload=repo_detail_payload
    )
    db.flush()
    _log_trade_audit(
        db,
        trade_id=int(row.id),
        account_id=int(row.account_id),
        strategy_id=int(row.strategy_id),
        action="update",
        reason=reason,
        snapshot={
            "before": old_snapshot,
            "after": _serialize_trade(row, repo_detail=repo_detail).model_dump(),
        },
    )
    _replay_touched_scopes(
        db,
        strategy_ids={old_strategy_id, int(row.strategy_id)},
        account_ids={old_account_id, int(row.account_id)},
    )
    return _serialize_trade(row, repo_detail=repo_detail)


@router.post("/trades/batch")
def live_add_trades_batch(
    payload: LiveTradeBatchCreateRequest, db: Session = Depends(get_session)
):
    inserted: list[LiveTrade] = []
    touched_strategy_ids: set[int] = set()
    touched_account_ids: set[int] = set()
    for one in payload.trades:
        row = _insert_trade(one, db)
        inserted.append(row)
        touched_strategy_ids.add(int(row.strategy_id))
        touched_account_ids.add(int(row.account_id))
    _replay_touched_scopes(
        db, strategy_ids=touched_strategy_ids, account_ids=touched_account_ids
    )
    return {"ok": True, "inserted": len(inserted)}


@router.get("/trades")
def live_list_trades(
    account_id: int | None = Query(default=None),
    strategy_id: int | None = Query(default=None),
    code: str | None = Query(default=None),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_session),
):
    q = db.query(LiveTrade)
    if account_id is not None:
        q = q.filter(LiveTrade.account_id == account_id)
    if strategy_id is not None:
        q = q.filter(LiveTrade.strategy_id == strategy_id)
    if code:
        q = q.filter(LiveTrade.code == _norm_code(code))
    if start:
        q = q.filter(LiveTrade.trade_date >= _to_date(start))
    if end:
        q = q.filter(LiveTrade.trade_date <= _to_date(end))
    total = int(q.count())
    rows = (
        q.order_by(
            LiveTrade.trade_date.desc(),
            LiveTrade.trade_time.desc(),
            LiveTrade.id.desc(),
        )
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    repo_map = _repo_detail_map(db, [int(x.id) for x in rows])
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [
            _serialize_trade(x, repo_detail=repo_map.get(int(x.id))).model_dump()
            for x in rows
        ],
    }


@router.delete("/trades/{trade_id}")
def live_delete_trade(
    trade_id: int, payload: LiveTradeDeleteRequest, db: Session = Depends(get_session)
):
    reason = _require_change_reason(payload.reason)
    row = db.query(LiveTrade).filter(LiveTrade.id == trade_id).one_or_none()
    if row is None:
        return {"ok": True}
    account_id = int(row.account_id)
    strategy_id = int(row.strategy_id)
    repo_detail = (
        db.query(LiveRepoTradeDetail)
        .filter(LiveRepoTradeDetail.trade_id == int(row.id))
        .one_or_none()
    )
    snapshot = _serialize_trade(row, repo_detail=repo_detail).model_dump()
    (
        db.query(LiveRepoTradeDetail)
        .filter(LiveRepoTradeDetail.trade_id == int(trade_id))
        .delete(synchronize_session=False)
    )
    db.delete(row)
    db.flush()
    _log_trade_audit(
        db,
        trade_id=trade_id,
        account_id=account_id,
        strategy_id=strategy_id,
        action="delete",
        reason=reason,
        snapshot={"before": snapshot},
    )
    _replay_touched_scopes(db, strategy_ids={strategy_id}, account_ids={account_id})
    return {"ok": True}


@router.post("/corporate-actions", response_model=LiveCorporateActionOut)
def live_add_corporate_action(
    payload: LiveCorporateActionCreateRequest, db: Session = Depends(get_session)
):
    event_type = str(payload.event_type or "").strip().lower()
    if event_type not in {"cash_dividend", "split", "share_conversion", "code_change"}:
        raise HTTPException(status_code=400, detail="invalid event_type")
    code = _norm_code(payload.code)
    new_code = _norm_code(payload.new_code or "") if payload.new_code else None
    row = LiveCorporateActionEvent(
        account_id=payload.account_id,
        strategy_id=payload.strategy_id,
        event_type=event_type,
        code=code,
        new_code=new_code,
        event_date=_to_date(payload.event_date),
        effective_date=_to_date(payload.effective_date),
        ratio_factor=payload.ratio_factor,
        cash_per_share=payload.cash_per_share,
        notes=payload.notes,
    )
    db.add(row)
    db.flush()
    target = _scope_from_ids(
        db, account_id=payload.account_id, strategy_id=payload.strategy_id
    )
    _replay_scope(db, target)
    if target.scope_type == "strategy":
        _replay_scope(
            db,
            _Scope(
                scope_type="account",
                scope_id=target.account_id,
                account_id=target.account_id,
                strategy_id=None,
            ),
        )
    return LiveCorporateActionOut(
        id=int(row.id),
        account_id=row.account_id,
        strategy_id=row.strategy_id,
        event_type=row.event_type,
        code=row.code,
        new_code=row.new_code,
        event_date=row.event_date.isoformat(),
        effective_date=row.effective_date.isoformat(),
        ratio_factor=row.ratio_factor,
        cash_per_share=row.cash_per_share,
        notes=row.notes,
        created_at=_dt_s(row.created_at),
    )


@router.get("/corporate-actions", response_model=list[LiveCorporateActionOut])
def live_list_corporate_actions(
    account_id: int | None = Query(default=None),
    strategy_id: int | None = Query(default=None),
    db: Session = Depends(get_session),
):
    q = db.query(LiveCorporateActionEvent)
    if account_id is not None:
        q = q.filter(LiveCorporateActionEvent.account_id == account_id)
    if strategy_id is not None:
        q = q.filter(LiveCorporateActionEvent.strategy_id == strategy_id)
    rows = q.order_by(
        LiveCorporateActionEvent.effective_date.desc(),
        LiveCorporateActionEvent.id.desc(),
    ).all()
    return [
        LiveCorporateActionOut(
            id=int(x.id),
            account_id=x.account_id,
            strategy_id=x.strategy_id,
            event_type=str(x.event_type),
            code=str(x.code),
            new_code=x.new_code,
            event_date=x.event_date.isoformat(),
            effective_date=x.effective_date.isoformat(),
            ratio_factor=x.ratio_factor,
            cash_per_share=x.cash_per_share,
            notes=x.notes,
            created_at=_dt_s(x.created_at),
        )
        for x in rows
    ]


@router.post("/replay")
def live_replay(payload: LiveReplayRequest, db: Session = Depends(get_session)):
    scope = _scope_from_ids(
        db, account_id=payload.account_id, strategy_id=payload.strategy_id
    )
    account_id = int(scope.account_id)
    strategy_ids = [
        int(x.id)
        for x in db.query(LiveStrategy.id)
        .filter(LiveStrategy.account_id == account_id)
        .all()
    ]
    strategy_out_by_id: dict[int, dict[str, Any]] = {}
    for sid in sorted(strategy_ids):
        strategy_out_by_id[sid] = _replay_scope(
            db,
            _Scope(
                scope_type="strategy",
                scope_id=sid,
                account_id=account_id,
                strategy_id=sid,
            ),
        )
    account_out = _replay_scope(
        db,
        _Scope(
            scope_type="account",
            scope_id=account_id,
            account_id=account_id,
            strategy_id=None,
        ),
    )
    if scope.scope_type == "strategy":
        target_out = strategy_out_by_id.get(int(scope.strategy_id or 0), account_out)
    else:
        target_out = account_out
    return {"ok": True, "strategies_replayed": len(strategy_ids), **target_out}


@router.get("/holdings", response_model=list[LiveHoldingOut])
def live_holdings(
    scope_type: str = Query(description="account|strategy"),
    scope_id: int = Query(ge=1),
    db: Session = Depends(get_session),
):
    st = str(scope_type or "").strip().lower()
    if st not in {"account", "strategy"}:
        raise HTTPException(
            status_code=400, detail="scope_type must be account|strategy"
        )
    # Holdings must be aligned to the latest replayed NAV day for the scope.
    # Otherwise callers can observe stale positions from an older day
    # when the latest day has no positions (e.g. fully closed book).
    max_nav_day = (
        db.query(LiveNavDaily.nav_date)
        .filter(
            LiveNavDaily.scope_type == st,
            LiveNavDaily.scope_id == scope_id,
        )
        .order_by(LiveNavDaily.nav_date.desc())
        .first()
    )
    if max_nav_day is None:
        return []
    day = max_nav_day[0]
    if st == "strategy":
        strategy = (
            db.query(LiveStrategy).filter(LiveStrategy.id == scope_id).one_or_none()
        )
        if strategy is None:
            return []
        scope = _Scope(
            scope_type="strategy",
            scope_id=int(scope_id),
            account_id=int(strategy.account_id),
            strategy_id=int(scope_id),
        )
    else:
        account = db.query(LiveAccount).filter(LiveAccount.id == scope_id).one_or_none()
        if account is None:
            return []
        scope = _Scope(
            scope_type="account",
            scope_id=int(scope_id),
            account_id=int(scope_id),
            strategy_id=None,
        )

    trades_all, _, _, events_all = _scope_rows(db, scope)
    trades = [t for t in trades_all if t.trade_date <= day]
    events = [ev for ev in events_all if ev.effective_date <= day]
    if not trades and not events:
        return []

    strategy_type_by_id = _strategy_type_map(
        db, sorted({int(t.strategy_id) for t in trades})
    )
    repo_detail_by_trade_id = _repo_detail_map(db, [int(t.id) for t in trades])

    start = min(
        [t.trade_date for t in trades] + [ev.effective_date for ev in events] or [day]
    )
    price_map = _latest_price_map(
        db,
        codes=sorted({_norm_code(t.code) for t in trades}),
        start=start - dt.timedelta(days=3650),
        end=day,
    )

    trade_by_day: dict[dt.date, list[LiveTrade]] = defaultdict(list)
    for t in trades:
        trade_by_day[t.trade_date].append(t)
    events_by_day: dict[dt.date, list[LiveCorporateActionEvent]] = defaultdict(list)
    for ev in events:
        events_by_day[ev.effective_date].append(ev)

    alias_map: dict[str, str] = {}
    for alias in db.query(LiveSymbolAlias).all():
        alias_map[_norm_code(alias.old_code)] = _norm_code(alias.new_code)

    lots: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    repo_lots: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    open_rounds: dict[tuple[str, int], dict[str, Any]] = {}

    def _ensure_round(key: tuple[str, int], code: str, name: str) -> dict[str, Any]:
        if key not in open_rounds:
            open_rounds[key] = {
                "code": code,
                "name": name or "",
                "buy_amount": 0.0,
                "sell_amount": 0.0,
                "total_fee": 0.0,
            }
        st_round = open_rounds[key]
        if not st_round.get("name") and name:
            st_round["name"] = name
        return st_round

    def _is_key_closed(key: tuple[str, int]) -> bool:
        stock_qty = float(sum(float(x["qty"]) for x in lots.get(key, [])))
        repo_qty = float(sum(float(x["principal"]) for x in repo_lots.get(key, [])))
        return stock_qty <= 1e-12 and repo_qty <= 1e-12

    for d in trading_days(start, day):
        for ev in events_by_day.get(d, []):
            code = _apply_symbol_alias(_norm_code(ev.code), alias_map)
            event_type = str(ev.event_type or "").strip().lower()
            if event_type in {"split", "share_conversion"}:
                fac = float(ev.ratio_factor or 0.0)
                if fac > 0.0:
                    for key in [k for k in lots.keys() if k[0] == code]:
                        for lot in lots.get(key, []):
                            q = float(lot["qty"])
                            lot["qty"] = q * fac
                            lot["unit_cost"] = float(lot["unit_cost"]) / fac
            elif event_type == "code_change":
                new_code = _norm_code(ev.new_code or "")
                if new_code:
                    alias_map[code] = new_code
                    for old_key in [k for k in list(lots.keys()) if k[0] == code]:
                        new_key = (new_code, old_key[1])
                        if new_key == old_key:
                            continue
                        lots[new_key].extend(lots.pop(old_key))
                    for old_key in [k for k in list(repo_lots.keys()) if k[0] == code]:
                        new_key = (new_code, old_key[1])
                        if new_key == old_key:
                            continue
                        repo_lots[new_key].extend(repo_lots.pop(old_key))
                    for old_key in [
                        k for k in list(open_rounds.keys()) if k[0] == code
                    ]:
                        st_old = open_rounds.pop(old_key)
                        new_key = (new_code, old_key[1])
                        if new_key == old_key:
                            open_rounds[old_key] = st_old
                            continue
                        st_new = _ensure_round(
                            new_key, new_code, st_old.get("name", "")
                        )
                        st_new["buy_amount"] = float(st_new["buy_amount"]) + float(
                            st_old.get("buy_amount", 0.0)
                        )
                        st_new["sell_amount"] = float(st_new["sell_amount"]) + float(
                            st_old.get("sell_amount", 0.0)
                        )
                        st_new["total_fee"] = float(st_new["total_fee"]) + float(
                            st_old.get("total_fee", 0.0)
                        )

        day_trades = sorted(
            trade_by_day.get(d, []),
            key=lambda x: (str(x.trade_time or ""), int(x.id)),
        )
        for t in day_trades:
            code = _apply_symbol_alias(_norm_code(t.code), alias_map)
            holder_id = int(t.shareholder_account_id)
            key = (code, holder_id)
            side = _norm_side(t.side)
            qty = float(t.quantity)
            px = float(t.price)
            fee = float(t.fee or 0.0)
            amount = float(t.amount or (qty * px))
            strategy_type = strategy_type_by_id.get(int(t.strategy_id), "etf_spot")

            if strategy_type == "bond_repo":
                detail = repo_detail_by_trade_id.get(int(t.id))
                if detail is None:
                    continue
                action = _norm_repo_action(detail.repo_action, side=side)
                principal = float(detail.principal_amount or amount)
                st_round = _ensure_round(key, code, t.name or "")
                st_round["total_fee"] = float(st_round["total_fee"]) + fee
                if action == "LEND":
                    repo_lots[key].append(
                        {
                            "principal": principal,
                            "fee_open_remain": fee,
                            "annual_rate_pct": float(detail.annual_rate_pct or px),
                            "open_date": t.trade_date,
                            "interest_days": int(detail.interest_days or 0),
                            "day_count_basis": int(detail.day_count_basis or 365),
                            "name": t.name,
                        }
                    )
                    st_round["buy_amount"] = float(st_round["buy_amount"]) + principal
                else:
                    remain = principal
                    fifo = repo_lots.get(key, [])
                    while remain > 1e-12 and fifo:
                        top = fifo[0]
                        top_principal = float(top["principal"])
                        take = min(remain, top_principal)
                        top_fee = float(top.get("fee_open_remain", 0.0) or 0.0)
                        fee_take = (
                            top_fee * (take / top_principal)
                            if top_principal > 1e-12 and top_fee > 0
                            else 0.0
                        )
                        top["fee_open_remain"] = max(0.0, top_fee - fee_take)
                        top["principal"] = top_principal - take
                        remain -= take
                        if float(top["principal"]) <= 1e-12:
                            fifo.pop(0)
                    st_round["sell_amount"] = float(st_round["sell_amount"]) + amount
                    if _is_key_closed(key):
                        open_rounds.pop(key, None)
                continue

            st_round = _ensure_round(key, code, t.name or "")
            st_round["total_fee"] = float(st_round["total_fee"]) + fee
            if side == "BUY":
                lots[key].append(
                    {
                        "qty": qty,
                        "unit_cost": amount / qty if qty > 1e-12 else 0.0,
                        "fee_remain": fee,
                        "name": t.name,
                    }
                )
                st_round["buy_amount"] = float(st_round["buy_amount"]) + amount
            else:
                remain = qty
                fifo = lots.get(key, [])
                while remain > 1e-12 and fifo:
                    top = fifo[0]
                    top_qty = float(top["qty"])
                    take = min(remain, top_qty)
                    top_fee = float(top.get("fee_remain", 0.0) or 0.0)
                    fee_take = (
                        top_fee * (take / top_qty)
                        if top_qty > 1e-12 and top_fee > 0
                        else 0.0
                    )
                    top["fee_remain"] = max(0.0, top_fee - fee_take)
                    top["qty"] = top_qty - take
                    remain -= take
                    if float(top["qty"]) <= 1e-12:
                        fifo.pop(0)
                st_round["sell_amount"] = float(st_round["sell_amount"]) + amount
                if _is_key_closed(key):
                    open_rounds.pop(key, None)

    holder_text_by_id = {
        int(x.id): str(x.shareholder_account or "")
        for x in db.query(LiveShareholderAccount)
        .filter(LiveShareholderAccount.account_id == scope.account_id)
        .all()
    }

    active_keys = sorted(
        {
            key
            for key in set(lots.keys()) | set(repo_lots.keys())
            if (
                sum(float(x["qty"]) for x in lots.get(key, [])) > 1e-12
                or sum(float(x["principal"]) for x in repo_lots.get(key, [])) > 1e-12
            )
        },
        key=lambda k: (k[0], int(k[1])),
    )
    out_rows: list[LiveHoldingOut] = []
    for code, holder_id in active_keys:
        key = (code, holder_id)
        stock_lots = lots.get(key, [])
        repo_open_lots = repo_lots.get(key, [])
        st_round = open_rounds.get(key)
        buy_amount = float(st_round.get("buy_amount", 0.0)) if st_round else 0.0
        sell_amount = float(st_round.get("sell_amount", 0.0)) if st_round else 0.0
        total_fee = float(st_round.get("total_fee", 0.0)) if st_round else 0.0

        market_price: float | None = None
        stale_days: int | None = None
        price_missing = False
        name = ""

        if stock_lots:
            qty_total = float(sum(float(x["qty"]) for x in stock_lots))
            mpx, mpx_day = _price_on_or_before(price_map, code, day)
            market_price = mpx
            stale_days = (day - mpx_day).days if mpx_day is not None else None
            price_missing = mpx is None
            market_value = float(qty_total * mpx) if mpx is not None else None
            name = str(stock_lots[-1].get("name", "") or "")
        else:
            qty_total = float(sum(float(x["principal"]) for x in repo_open_lots))
            accrued_interest = 0.0
            for lot in repo_open_lots:
                principal = float(lot["principal"])
                lot_days = int(lot["interest_days"])
                elapsed = (
                    lot_days
                    if lot_days > 0
                    else max((day - lot["open_date"]).days + 1, 0)
                )
                accrued_interest += (
                    principal
                    * float(lot["annual_rate_pct"])
                    / 100.0
                    * float(elapsed)
                    / float(lot["day_count_basis"])
                )
            market_value = float(qty_total + accrued_interest)
            name = str(repo_open_lots[-1].get("name", "") or "")

        if market_value is not None:
            cumulative_pnl = float(market_value + sell_amount - buy_amount - total_fee)
            cost_value = float(market_value - cumulative_pnl)
        else:
            cumulative_pnl = None
            cost_value = float(buy_amount + total_fee - sell_amount)
        cost_price = float(cost_value / qty_total) if qty_total > 1e-12 else None
        pnl_rate = (
            float(cumulative_pnl / cost_value)
            if cumulative_pnl is not None and abs(cost_value) > 1e-12
            else None
        )

        out_rows.append(
            LiveHoldingOut(
                snapshot_date=day.isoformat(),
                scope_type=st,
                scope_id=int(scope_id),
                account_id=int(scope.account_id),
                strategy_id=scope.strategy_id,
                shareholder_account_id=int(holder_id),
                shareholder_account=holder_text_by_id.get(int(holder_id)) or None,
                code=code,
                name=name,
                quantity=qty_total,
                cost_price=cost_price,
                market_price=market_price,
                cost_value=float(cost_value),
                market_value=market_value,
                pnl_amount=cumulative_pnl,
                pnl_rate=pnl_rate,
                price_missing=bool(price_missing),
                stale_days=stale_days,
            )
        )
    return out_rows


@router.get("/closed-rounds")
def live_closed_rounds(
    scope_type: str = Query(description="account|strategy"),
    scope_id: int = Query(ge=1),
    code: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_session),
):
    st = str(scope_type or "").strip().lower()
    if st not in {"account", "strategy"}:
        raise HTTPException(
            status_code=400, detail="scope_type must be account|strategy"
        )
    q = db.query(LiveClosedRound).filter(
        LiveClosedRound.scope_type == st, LiveClosedRound.scope_id == scope_id
    )
    if code:
        q = q.filter(LiveClosedRound.code == _norm_code(code))
    total = int(q.count())
    rows = (
        q.order_by(LiveClosedRound.close_date.desc(), LiveClosedRound.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    items = [
        LiveClosedRoundOut(
            id=int(x.id),
            scope_type=x.scope_type,
            scope_id=int(x.scope_id),
            account_id=int(x.account_id),
            strategy_id=x.strategy_id,
            round_no=int(x.round_no),
            code=x.code,
            name=x.name,
            open_date=x.open_date.isoformat(),
            close_date=x.close_date.isoformat(),
            buy_count=int(x.buy_count),
            sell_count=int(x.sell_count),
            buy_qty=float(x.buy_qty),
            sell_qty=float(x.sell_qty),
            avg_buy_price=x.avg_buy_price,
            avg_sell_price=x.avg_sell_price,
            realized_pnl=float(x.realized_pnl),
            return_rate=x.return_rate,
            total_fee=float(x.total_fee),
        ).model_dump()
        for x in rows
    ]
    return {"total": total, "page": page, "page_size": page_size, "items": items}


@router.get("/stats/closed-rounds")
def live_closed_round_stats(
    scope_type: str = Query(description="account|strategy"),
    scope_id: int = Query(ge=1),
    db: Session = Depends(get_session),
):
    st = str(scope_type or "").strip().lower()
    rows = (
        db.query(LiveClosedRound)
        .filter(LiveClosedRound.scope_type == st, LiveClosedRound.scope_id == scope_id)
        .all()
    )
    n = len(rows)
    pnl = (
        np.asarray([float(x.realized_pnl) for x in rows], dtype=float)
        if rows
        else np.asarray([], dtype=float)
    )
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    win_count = int(len(wins))
    loss_count = int(len(losses))
    denom = win_count + loss_count
    win_rate = float(win_count / denom) if denom > 0 else float("nan")
    avg_win = float(np.mean(wins)) if win_count > 0 else float("nan")
    avg_loss = float(np.mean(losses)) if loss_count > 0 else float("nan")
    payoff = (
        float(avg_win / abs(avg_loss))
        if loss_count > 0 and avg_loss < 0
        else float("nan")
    )
    return {
        "scope_type": st,
        "scope_id": scope_id,
        "total_trades": n,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": _safe_json_float(win_rate),
        "avg_profit_amount": _safe_json_float(avg_win),
        "avg_loss_amount": _safe_json_float(avg_loss),
        "max_profit_amount": _safe_json_float(
            float(np.max(wins)) if win_count > 0 else float("nan")
        ),
        "max_loss_amount": _safe_json_float(
            float(np.min(losses)) if loss_count > 0 else float("nan")
        ),
        "payoff_ratio": _safe_json_float(payoff),
    }


@router.get("/stats/fees", response_model=LiveFeeStatsOut)
def live_fee_stats(
    scope_type: str = Query(description="account|strategy"),
    scope_id: int = Query(ge=1),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    db: Session = Depends(get_session),
):
    st = str(scope_type or "").strip().lower()
    if st not in {"account", "strategy"}:
        raise HTTPException(
            status_code=400, detail="scope_type must be account|strategy"
        )
    q = db.query(LiveTrade)
    if st == "account":
        q = q.filter(LiveTrade.account_id == scope_id)
    else:
        q = q.filter(LiveTrade.strategy_id == scope_id)
    if start:
        q = q.filter(LiveTrade.trade_date >= _to_date(start))
    if end:
        q = q.filter(LiveTrade.trade_date <= _to_date(end))
    rows = q.order_by(LiveTrade.trade_date.asc(), LiveTrade.id.asc()).all()
    total_fee = float(sum(float(x.fee or 0.0) for x in rows))
    buy_fee = float(
        sum(float(x.fee or 0.0) for x in rows if _norm_side(x.side) == "BUY")
    )
    sell_fee = float(
        sum(float(x.fee or 0.0) for x in rows if _norm_side(x.side) == "SELL")
    )
    avg_fee = float(total_fee / len(rows)) if rows else 0.0
    by_day_map: dict[str, float] = defaultdict(float)
    for x in rows:
        by_day_map[x.trade_date.isoformat()] += float(x.fee or 0.0)
    by_day = [{"date": k, "fee": float(v)} for k, v in sorted(by_day_map.items())]
    return LiveFeeStatsOut(
        scope_type=st,
        scope_id=scope_id,
        total_fee=total_fee,
        buy_fee=buy_fee,
        sell_fee=sell_fee,
        avg_fee_per_trade=avg_fee,
        by_day=by_day,
    )


@router.get("/stats/fees/closed-rounds", response_model=LiveFeeStatsOut)
def live_closed_round_fee_stats(
    scope_type: str = Query(description="account|strategy"),
    scope_id: int = Query(ge=1),
    code: str | None = Query(default=None),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    db: Session = Depends(get_session),
):
    st = str(scope_type or "").strip().lower()
    if st not in {"account", "strategy"}:
        raise HTTPException(
            status_code=400, detail="scope_type must be account|strategy"
        )
    q = db.query(LiveClosedRoundLeg, LiveClosedRound).join(
        LiveClosedRound, LiveClosedRound.id == LiveClosedRoundLeg.round_id
    )
    q = q.filter(LiveClosedRound.scope_type == st, LiveClosedRound.scope_id == scope_id)
    if code:
        q = q.filter(LiveClosedRound.code == _norm_code(code))
    if start:
        q = q.filter(LiveClosedRoundLeg.trade_date >= _to_date(start))
    if end:
        q = q.filter(LiveClosedRoundLeg.trade_date <= _to_date(end))
    rows = q.order_by(
        LiveClosedRoundLeg.trade_date.asc(), LiveClosedRoundLeg.id.asc()
    ).all()
    total_fee = float(sum(float(leg.fee or 0.0) for leg, _ in rows))
    buy_fee = float(
        sum(
            float(leg.fee or 0.0)
            for leg, _ in rows
            if _norm_side(getattr(leg, "side", "")) == "BUY"
        )
    )
    sell_fee = float(
        sum(
            float(leg.fee or 0.0)
            for leg, _ in rows
            if _norm_side(getattr(leg, "side", "")) == "SELL"
        )
    )
    avg_fee = float(total_fee / len(rows)) if rows else 0.0
    by_day_map: dict[str, float] = defaultdict(float)
    for leg, _ in rows:
        by_day_map[leg.trade_date.isoformat()] += float(leg.fee or 0.0)
    by_day = [{"date": k, "fee": float(v)} for k, v in sorted(by_day_map.items())]
    return LiveFeeStatsOut(
        scope_type=st,
        scope_id=scope_id,
        total_fee=total_fee,
        buy_fee=buy_fee,
        sell_fee=sell_fee,
        avg_fee_per_trade=avg_fee,
        by_day=by_day,
    )


@router.get("/performance", response_model=LivePerformanceOut)
def live_performance(
    scope_type: str = Query(description="account|strategy"),
    scope_id: int = Query(ge=1),
    return_basis: str = Query(default="both", description="dietz|twr|both"),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    db: Session = Depends(get_session),
):
    st = str(scope_type or "").strip().lower()
    if st not in {"account", "strategy"}:
        raise HTTPException(
            status_code=400, detail="scope_type must be account|strategy"
        )
    rb = str(return_basis or "both").strip().lower()
    if rb not in {"dietz", "twr", "both"}:
        raise HTTPException(
            status_code=400, detail="return_basis must be dietz|twr|both"
        )
    q = db.query(LiveNavDaily).filter(
        LiveNavDaily.scope_type == st, LiveNavDaily.scope_id == scope_id
    )
    if start:
        q = q.filter(LiveNavDaily.nav_date >= _to_date(start))
    if end:
        q = q.filter(LiveNavDaily.nav_date <= _to_date(end))
    rows = q.order_by(LiveNavDaily.nav_date.asc()).all()
    if not rows:
        return LivePerformanceOut(
            scope_type=st,
            scope_id=scope_id,
            return_basis=rb,
            nav=[],
            dietz_basis_metrics=_calc_metrics(
                pd.Series(dtype=float), pd.Series(dtype=float)
            ),
            twr_basis_metrics=_calc_metrics(
                pd.Series(dtype=float), pd.Series(dtype=float)
            ),
        )
    idx = pd.DatetimeIndex([pd.Timestamp(x.nav_date) for x in rows])
    nav_d = pd.Series([float(x.nav_dietz) for x in rows], index=idx, dtype=float)
    nav_t = pd.Series([float(x.nav_twr) for x in rows], index=idx, dtype=float)
    ret_d = pd.Series(
        [float(x.daily_return_dietz or 0.0) for x in rows], index=idx, dtype=float
    )
    ret_t = pd.Series(
        [float(x.daily_return_twr or 0.0) for x in rows], index=idx, dtype=float
    )
    metrics_d = _calc_metrics(nav_d, ret_d)
    metrics_t = _calc_metrics(nav_t, ret_t)
    nav_points = [
        {
            "nav_date": x.nav_date.isoformat(),
            "equity": float(x.equity),
            "cash": float(x.cash),
            "market_value": float(x.market_value),
            "external_flow": float(x.external_flow),
            "trading_fee": float(x.trading_fee),
            "nav_twr": float(x.nav_twr),
            "nav_dietz": float(x.nav_dietz),
            "daily_return_twr": (
                float(x.daily_return_twr) if x.daily_return_twr is not None else None
            ),
            "daily_return_dietz": (
                float(x.daily_return_dietz)
                if x.daily_return_dietz is not None
                else None
            ),
            "selection_return": (
                float(x.selection_return) if x.selection_return is not None else None
            ),
            "timing_return": float(x.timing_return)
            if x.timing_return is not None
            else None,
            "position_return": (
                float(x.position_return) if x.position_return is not None else None
            ),
            "cost_drag_return": (
                float(x.cost_drag_return) if x.cost_drag_return is not None else None
            ),
            "cash_drag_return": (
                float(x.cash_drag_return) if x.cash_drag_return is not None else None
            ),
            "repo_carry_return": (
                float(x.repo_carry_return) if x.repo_carry_return is not None else None
            ),
            "repo_fee_drag_return": (
                float(x.repo_fee_drag_return)
                if x.repo_fee_drag_return is not None
                else None
            ),
        }
        for x in rows
    ]
    return LivePerformanceOut(
        scope_type=st,
        scope_id=scope_id,
        return_basis=rb,
        nav=nav_points,
        dietz_basis_metrics=metrics_d,
        twr_basis_metrics=metrics_t,
    )


@router.get("/attribution", response_model=LiveAttributionOut)
def live_attribution(
    scope_type: str = Query(description="account|strategy"),
    scope_id: int = Query(ge=1),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    db: Session = Depends(get_session),
):
    st = str(scope_type or "").strip().lower()
    if st not in {"account", "strategy"}:
        raise HTTPException(
            status_code=400, detail="scope_type must be account|strategy"
        )
    q = db.query(LiveNavDaily).filter(
        LiveNavDaily.scope_type == st, LiveNavDaily.scope_id == scope_id
    )
    if start:
        q = q.filter(LiveNavDaily.nav_date >= _to_date(start))
    if end:
        q = q.filter(LiveNavDaily.nav_date <= _to_date(end))
    rows = q.order_by(LiveNavDaily.nav_date.asc()).all()
    daily: list[dict[str, Any]] = []
    sums = {
        "selection_return": 0.0,
        "timing_return": 0.0,
        "position_return": 0.0,
        "cost_drag_return": 0.0,
        "cash_drag_return": 0.0,
        "repo_carry_return": 0.0,
        "repo_fee_drag_return": 0.0,
    }
    for x in rows:
        item = {
            "date": x.nav_date.isoformat(),
            "selection_return": float(x.selection_return or 0.0),
            "timing_return": float(x.timing_return or 0.0),
            "position_return": float(x.position_return or 0.0),
            "cost_drag_return": float(x.cost_drag_return or 0.0),
            "cash_drag_return": float(x.cash_drag_return or 0.0),
            "repo_carry_return": float(x.repo_carry_return or 0.0),
            "repo_fee_drag_return": float(x.repo_fee_drag_return or 0.0),
            "daily_return_twr": float(x.daily_return_twr or 0.0),
        }
        daily.append(item)
        for k in sums:
            sums[k] += float(item[k])
    total = float(sum(float(x.daily_return_twr or 0.0) for x in rows))
    sums["total_return_twr_sum"] = total
    sums["rebuild_total"] = float(
        sums["selection_return"]
        + sums["timing_return"]
        + sums["position_return"]
        + sums["cost_drag_return"]
        + sums["cash_drag_return"]
        + sums["repo_carry_return"]
        + sums["repo_fee_drag_return"]
    )
    return LiveAttributionOut(
        scope_type=st, scope_id=scope_id, daily=daily, period=sums
    )
