from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from ..db.repo import PriceRow


class AkshareTencentLike:
    def stock_zh_a_hist_tx(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class FetchRequest:
    code: str  # 6-digit
    start_date: str  # YYYYMMDD
    end_date: str  # YYYYMMDD
    adjust: str = "qfq"  # qfq/hfq/none


def _with_ex(code6: str) -> str:
    c = str(code6).strip()
    if c.lower().startswith(("sh", "sz")):
        return c.lower()
    exch = "sh" if c and c[0] in {"5", "6", "9"} else "sz"
    return f"{exch}{c}"


def fetch_etf_daily_tencent(
    ak: AkshareTencentLike,
    req: FetchRequest,
) -> list[PriceRow]:
    """
    Fetch daily OHLC+amount from Tencent via akshare.stock_zh_a_hist_tx.

    Notes:
    - For many ETFs, tencent's `adjust='qfq'` appears equivalent to `adjust=''` (none).
      We still pass through the requested adjust; if it errors, fallback to '' for qfq/none.
    - Volume is not provided by this API; we keep volume=None and store amount.
    """
    sym = _with_ex(req.code)
    adj = str(req.adjust or "none").strip().lower()
    tx_adj = "" if adj in {"none", ""} else adj

    df: pd.DataFrame
    try:
        df = ak.stock_zh_a_hist_tx(symbol=sym, start_date=req.start_date, end_date=req.end_date, adjust=tx_adj)
    except Exception:
        # fallback: qfq may fail or behave like none; retry with ""
        df = ak.stock_zh_a_hist_tx(symbol=sym, start_date=req.start_date, end_date=req.end_date, adjust="")

    if df is None or df.empty:
        return []

    # expected cols: date/open/close/high/low/amount
    if "date" not in df.columns or "open" not in df.columns or "close" not in df.columns:
        return []

    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce").dt.date
    df2 = df2.dropna(subset=["date"]).sort_values("date", ascending=True)
    for c in ["open", "close", "high", "low", "amount"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    rows: list[PriceRow] = []
    for _, r in df2.iterrows():
        rows.append(
            PriceRow(
                code=req.code,
                trade_date=r["date"],
                open=float(r["open"]) if pd.notna(r.get("open")) else None,
                high=float(r["high"]) if pd.notna(r.get("high")) else None,
                low=float(r["low"]) if pd.notna(r.get("low")) else None,
                close=float(r["close"]) if pd.notna(r.get("close")) else None,
                volume=None,
                amount=float(r["amount"]) if pd.notna(r.get("amount")) else None,
                source="tencent",
                adjust=adj,
            )
        )

    dedup: dict[dt.date, PriceRow] = {rr.trade_date: rr for rr in rows}
    return [dedup[d] for d in sorted(dedup)]

