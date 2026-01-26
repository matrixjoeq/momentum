from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..db.repo import PriceRow


class AkshareSinaLike:
    def fund_etf_hist_sina(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def fund_etf_dividend_sina(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class FetchRequest:
    code: str  # 6-digit
    start_date: str  # YYYYMMDD
    end_date: str  # YYYYMMDD


def _with_ex(code6: str) -> str:
    c = str(code6).strip()
    if c.lower().startswith(("sh", "sz")):
        return c.lower()
    exch = "sh" if c and c[0] in {"5", "6", "9"} else "sz"
    return f"{exch}{c}"


def _date_ymd(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y%m%d").date()


def _clip_range(df: pd.DataFrame, *, start: dt.date, end: dt.date) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date
        d = d.dropna(subset=["date"])
        d = d[(d["date"] >= start) & (d["date"] <= end)]
        d = d.sort_values("date", ascending=True)
    return d


def _scale_prices(df: pd.DataFrame, *, scale: float) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for c in ["open", "high", "low", "close"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") * float(scale)
    return out


def _add_cash_dividend(df: pd.DataFrame, *, cash_div: pd.Series) -> pd.DataFrame:
    """
    Approximate total-return (hfq-like) by adding cumulative cash dividends to price.

    Sina's `fund_etf_dividend_sina` currently exposes "累计分红" (cash), but not splits.
    This is a best-effort conversion and may deviate from Eastmoney's adjusted series.
    """
    if df.empty:
        return df
    out = df.copy()
    cd = cash_div.reindex(out["date"]).astype(float)
    for c in ["open", "high", "low", "close"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") + cd.to_numpy(dtype=float)
    return out


def fetch_etf_daily_sina_none_and_adjusted(
    ak: AkshareSinaLike,
    req: FetchRequest,
) -> dict[str, list[PriceRow]]:
    """
    Fetch Sina ETF daily series:
    - none: raw OHLCV
    - hfq: best-effort total-return by adding cumulative dividends
    - qfq: scaled hfq so that last close matches raw last close (common qfq convention)

    Returns dict keys: none/qfq/hfq, each a list[PriceRow].
    """
    sym = _with_ex(req.code)
    start = _date_ymd(req.start_date)
    end = _date_ymd(req.end_date)

    df0 = ak.fund_etf_hist_sina(symbol=sym)
    df0 = _clip_range(df0, start=start, end=end)
    if df0.empty:
        return {"none": [], "qfq": [], "hfq": []}

    # cash dividend factor (cumulative), may be missing; treat missing as 0.
    div = None
    try:
        div = ak.fund_etf_dividend_sina(symbol=sym)
    except Exception:  # pylint: disable=broad-exception-caught
        div = None
    cash = pd.Series(0.0, index=df0["date"])
    if div is not None and (not div.empty) and ("日期" in div.columns) and ("累计分红" in div.columns):
        d2 = div.copy()
        d2["日期"] = pd.to_datetime(d2["日期"], errors="coerce").dt.date
        d2 = d2.dropna(subset=["日期"])
        # IMPORTANT: build by raw arrays to avoid pandas aligning Series indices (which would yield NaNs).
        vals = pd.to_numeric(d2["累计分红"], errors="coerce").to_numpy(dtype=float)
        idx = d2["日期"].to_list()
        s = pd.Series(vals, index=idx, dtype=float)
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if not s.empty:
            cash = s.reindex(df0["date"]).ffill().fillna(0.0).astype(float)

    hfq_df = _add_cash_dividend(df0, cash_div=cash)

    # qfq: scale hfq to match last close of raw (so latest is tradable)
    raw_last = float(pd.to_numeric(df0["close"], errors="coerce").dropna().iloc[-1])
    hfq_last = float(pd.to_numeric(hfq_df["close"], errors="coerce").dropna().iloc[-1])
    scale = (raw_last / hfq_last) if (np.isfinite(raw_last) and np.isfinite(hfq_last) and hfq_last != 0) else 1.0
    qfq_df = _scale_prices(hfq_df, scale=scale)

    def _to_rows(df: pd.DataFrame, *, adjust: str) -> list[PriceRow]:
        rows: list[PriceRow] = []
        for _, r in df.iterrows():
            td = r.get("date")
            if td is None:
                continue
            rows.append(
                PriceRow(
                    code=req.code,
                    trade_date=td,
                    open=float(r["open"]) if pd.notna(r.get("open")) else None,
                    high=float(r["high"]) if pd.notna(r.get("high")) else None,
                    low=float(r["low"]) if pd.notna(r.get("low")) else None,
                    close=float(r["close"]) if pd.notna(r.get("close")) else None,
                    volume=float(r["volume"]) if pd.notna(r.get("volume")) else None,
                    amount=None,
                    source="sina",
                    adjust=adjust,
                )
            )
        rows.sort(key=lambda x: x.trade_date)
        dedup: dict[dt.date, PriceRow] = {rr.trade_date: rr for rr in rows}
        return [dedup[d] for d in sorted(dedup)]

    return {
        "none": _to_rows(df0, adjust="none"),
        "hfq": _to_rows(hfq_df, adjust="hfq"),
        "qfq": _to_rows(qfq_df, adjust="qfq"),
    }

