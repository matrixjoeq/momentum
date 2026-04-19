from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from ..db.futures_repo import (
    FuturesPriceRow,
    get_futures_last_trade_date,
    get_futures_pool_by_code,
    mark_futures_fetch_status,
    update_futures_pool_data_range,
    upsert_futures_prices,
)
from ..settings import get_settings


@dataclass(frozen=True)
class IngestFuturesResult:
    code: str
    upserted: int
    status: str
    message: str | None = None


def _parse_yyyymmdd(x: str) -> dt.date:
    return dt.datetime.strptime(x, "%Y%m%d").date()


def _pick_col(columns: list[str], keywords: tuple[str, ...]) -> str | None:
    for c in columns:
        cs = str(c)
        if any(k in cs for k in keywords):
            return c
    return None


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(x):
        return None
    return x


def _normalize_futures_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["trade_date", "open", "high", "low", "close", "settle", "volume", "hold", "amount"])
    cols = [str(c) for c in list(df.columns)]
    date_col = _pick_col(cols, ("日期", "交易日期", "date"))
    open_col = _pick_col(cols, ("开盘", "open"))
    high_col = _pick_col(cols, ("最高", "high"))
    low_col = _pick_col(cols, ("最低", "low"))
    close_col = _pick_col(cols, ("收盘", "close"))
    settle_col = _pick_col(cols, ("结算", "结算价", "settle", "settlement"))
    volume_col = _pick_col(cols, ("成交量", "volume"))
    amount_col = _pick_col(cols, ("成交额", "amount"))
    hold_col = _pick_col(cols, ("持仓量", "持仓", "hold", "open_interest", "oi"))

    if date_col is None or close_col is None:
        return pd.DataFrame(columns=["trade_date", "open", "high", "low", "close", "settle", "volume", "hold", "amount"])

    out = pd.DataFrame()
    out["trade_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    out["open"] = pd.to_numeric(df[open_col], errors="coerce") if open_col is not None else None
    out["high"] = pd.to_numeric(df[high_col], errors="coerce") if high_col is not None else None
    out["low"] = pd.to_numeric(df[low_col], errors="coerce") if low_col is not None else None
    out["close"] = pd.to_numeric(df[close_col], errors="coerce")
    out["settle"] = pd.to_numeric(df[settle_col], errors="coerce") if settle_col is not None else None
    out["volume"] = pd.to_numeric(df[volume_col], errors="coerce") if volume_col is not None else None
    out["amount"] = pd.to_numeric(df[amount_col], errors="coerce") if amount_col is not None else None
    out["hold"] = pd.to_numeric(df[hold_col], errors="coerce") if hold_col is not None else None
    out = out.dropna(subset=["trade_date"]).sort_values("trade_date", ascending=True)
    return out


def _fetch_futures_daily_sina_df(*, ak: Any, symbol: str) -> pd.DataFrame:
    fn = getattr(ak, "futures_zh_daily_sina", None)
    if fn is None:
        raise ValueError("akshare.futures_zh_daily_sina unavailable")
    try:
        return fn(symbol=symbol)
    except TypeError:
        return fn(symbol)


def ingest_one_futures(
    db: Session,
    *,
    ak: Any,
    code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    fetch_type: str = "incremental",
) -> IngestFuturesResult:
    pool = get_futures_pool_by_code(db, code)
    if pool is None:
        raise ValueError(f"futures {code} not found in pool")

    settings = get_settings()
    base_start = start_date or pool.start_date or settings.default_futures_start_date
    end = end_date or pool.end_date or settings.default_end_date
    mode = str(fetch_type or "incremental").strip().lower()
    if mode not in {"incremental", "full"}:
        raise ValueError("fetch_type must be incremental or full")

    start = base_start
    fallback_to_full = False
    if mode == "incremental":
        last_trade_date = get_futures_last_trade_date(db, code=code, adjust="none")
        if last_trade_date is not None:
            next_start_d = last_trade_date + dt.timedelta(days=1)
            next_start = next_start_d.strftime("%Y%m%d")
            start = max(base_start, next_start)
        else:
            fallback_to_full = True
            start = base_start

    start_d = _parse_yyyymmdd(start)
    end_d = _parse_yyyymmdd(end)
    if start_d > end_d:
        msg = f"no new futures data to fetch (mode={mode}, start={start}, end={end})"
        mark_futures_fetch_status(db, code=code, status="success", message=msg)
        db.commit()
        return IngestFuturesResult(code=code, upserted=0, status="success", message=msg)

    try:
        raw_df = _fetch_futures_daily_sina_df(ak=ak, symbol=code)
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as e:
        msg = f"fetch sina futures failed: {e}"
        mark_futures_fetch_status(db, code=code, status="failed", message=msg)
        db.commit()
        return IngestFuturesResult(code=code, upserted=0, status="failed", message=msg)

    norm = _normalize_futures_df(raw_df)
    if norm.empty:
        msg = "sina futures data is empty"
        mark_futures_fetch_status(db, code=code, status="failed", message=msg)
        db.commit()
        return IngestFuturesResult(code=code, upserted=0, status="failed", message=msg)

    norm = norm[(norm["trade_date"] >= start_d) & (norm["trade_date"] <= end_d)].copy()
    if norm.empty:
        msg = f"no futures data in requested range (mode={mode}, start={start}, end={end})"
        mark_futures_fetch_status(db, code=code, status="success", message=msg)
        db.commit()
        return IngestFuturesResult(code=code, upserted=0, status="success", message=msg)

    rows = [
        FuturesPriceRow(
            code=code,
            trade_date=row.trade_date,
            open=_to_float(row.open),
            high=_to_float(row.high),
            low=_to_float(row.low),
            close=_to_float(row.close),
            settle=_to_float(row.settle),
            volume=_to_float(row.volume),
            amount=_to_float(row.amount),
            hold=_to_float(row.hold),
            source="sina",
            adjust="none",
        )
        for row in norm.itertuples(index=False)
    ]
    n = upsert_futures_prices(db, rows)
    update_futures_pool_data_range(db, code=code, adjust="none")
    mode_note = mode
    if fallback_to_full:
        mode_note = "incremental->full"
    msg = f"none={len(rows)} source=sina mode={mode_note} range={start}~{end}"
    mark_futures_fetch_status(db, code=code, status="success", message=msg)
    db.commit()
    return IngestFuturesResult(code=code, upserted=int(n), status="success", message=msg)
