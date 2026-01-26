from __future__ import annotations

from dataclasses import dataclass

from ..db.repo import PriceRow
from .akshare_fetcher import AkshareLike, FetchRequest as AkFetchRequest, fetch_etf_daily_qfq


@dataclass(frozen=True)
class FetchRequest:
    code: str  # 6-digit
    start_date: str  # YYYYMMDD
    end_date: str  # YYYYMMDD
    adjust: str = "qfq"  # qfq/hfq/none


def fetch_etf_daily_eastmoney(
    ak: AkshareLike,
    req: FetchRequest,
) -> list[PriceRow]:
    """
    Fetch ETF daily prices from Eastmoney via akshare.fund_etf_hist_em.

    This is a thin wrapper around `akshare_fetcher.fetch_etf_daily_qfq` with:
    - consistent adjust normalization (none -> "")
    - consistent PriceRow.adjust values (qfq/hfq/none)
    """
    adj = str(req.adjust or "qfq").strip().lower()
    ak_adj = "" if adj == "none" else adj

    rows = fetch_etf_daily_qfq(
        ak,
        AkFetchRequest(code=req.code, start_date=req.start_date, end_date=req.end_date, adjust=ak_adj),
    )

    if not rows:
        return []

    # Ensure adjust normalization in returned rows (akshare_fetcher uses "" for none).
    out: list[PriceRow] = []
    for r in rows:
        out.append(
            PriceRow(
                code=r.code,
                trade_date=r.trade_date,
                open=r.open,
                high=r.high,
                low=r.low,
                close=r.close,
                volume=r.volume,
                amount=r.amount,
                source=r.source or "eastmoney",
                adjust=adj,
            )
        )
    return out

