from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from ..db.repo import PriceRow


class AkshareLike:
    def fund_etf_hist_em(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class FetchRequest:
    code: str
    start_date: str  # YYYYMMDD
    end_date: str  # YYYYMMDD
    adjust: str = "qfq"  # 前复权


def _parse_trade_date(x) -> dt.date:
    if isinstance(x, dt.date) and not isinstance(x, dt.datetime):
        return x
    if isinstance(x, dt.datetime):
        return x.date()
    s = str(x)
    # common: "2024-01-02"
    return dt.date.fromisoformat(s[:10])


def fetch_etf_daily_qfq(
    ak: AkshareLike,
    req: FetchRequest,
) -> list[PriceRow]:
    """
    Fetch ETF daily prices from akshare (eastmoney) and return standardized rows.

    We prefer calling akshare with start/end if supported; otherwise fetch all and filter.
    """
    symbol = req.code
    kwargs = {"symbol": symbol, "period": "daily", "adjust": req.adjust}
    df: pd.DataFrame
    try:
        df = ak.fund_etf_hist_em(**kwargs, start_date=req.start_date, end_date=req.end_date)
    except TypeError:
        df = ak.fund_etf_hist_em(**kwargs)

    if df is None or df.empty:
        return []

    # Normalize column names (akshare uses Chinese column names).
    col_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    renamed = {c: col_map[c] for c in df.columns if c in col_map}
    df2 = df.rename(columns=renamed).copy()

    # Minimal required fields for our strategy backtests are open/close.
    # high/low/volume/amount are optional and will be stored when available.
    required = ["date", "open", "close"]
    missing = [c for c in required if c not in df2.columns]
    if missing:
        raise ValueError(f"akshare result missing columns: {missing}. columns={list(df.columns)}")

    df2["trade_date"] = df2["date"].apply(_parse_trade_date)

    # Filter by requested range (trade_date in [start,end])
    start = dt.datetime.strptime(req.start_date, "%Y%m%d").date()
    end = dt.datetime.strptime(req.end_date, "%Y%m%d").date()
    df2 = df2[(df2["trade_date"] >= start) & (df2["trade_date"] <= end)]

    rows: list[PriceRow] = []
    for _, r in df2.iterrows():
        rows.append(
            PriceRow(
                code=req.code,
                trade_date=r["trade_date"],
                open=float(r["open"]) if pd.notna(r["open"]) else None,
                high=float(r["high"]) if ("high" in df2.columns and pd.notna(r.get("high"))) else None,
                low=float(r["low"]) if ("low" in df2.columns and pd.notna(r.get("low"))) else None,
                close=float(r["close"]) if pd.notna(r["close"]) else None,
                volume=float(r["volume"]) if "volume" in df2.columns and pd.notna(r.get("volume")) else None,
                amount=float(r["amount"]) if "amount" in df2.columns and pd.notna(r.get("amount")) else None,
                source="eastmoney",
                adjust=req.adjust,
            )
        )

    # Ensure ascending and unique by trade_date
    rows.sort(key=lambda x: x.trade_date)
    dedup: dict[dt.date, PriceRow] = {}
    for row in rows:
        dedup[row.trade_date] = row
    return [dedup[d] for d in sorted(dedup.keys())]

