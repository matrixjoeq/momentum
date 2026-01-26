from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..db.repo import PriceRow
from .akshare_eastmoney_fetcher import FetchRequest as EastmoneyFetchRequest
from .akshare_eastmoney_fetcher import fetch_etf_daily_eastmoney
from .akshare_sina_fetcher import FetchRequest as SinaFetchRequest
from .akshare_sina_fetcher import fetch_etf_daily_sina_none_and_adjusted
from .akshare_tencent_fetcher import FetchRequest as TencentFetchRequest
from .akshare_tencent_fetcher import fetch_etf_daily_tencent


@dataclass(frozen=True)
class FetchRequest:
    code: str
    start_date: str  # YYYYMMDD
    end_date: str  # YYYYMMDD
    adjust: str = "qfq"  # qfq/hfq/none (akshare uses "" for none; we normalize outside)


def _try_tencent(ak: Any, *, req: FetchRequest) -> list[PriceRow]:
    try:
        return fetch_etf_daily_tencent(
            ak,
            TencentFetchRequest(
                code=req.code,
                start_date=req.start_date,
                end_date=req.end_date,
                adjust=req.adjust,
            ),
        )
    except Exception:  # pylint: disable=broad-exception-caught
        return []


def _try_sina(ak: Any, *, req: FetchRequest) -> list[PriceRow]:
    try:
        packs = fetch_etf_daily_sina_none_and_adjusted(
            ak,
            SinaFetchRequest(code=req.code, start_date=req.start_date, end_date=req.end_date),
        )
    except Exception:  # pylint: disable=broad-exception-caught
        return []
    adj = str(req.adjust or "qfq").strip().lower()
    key = "none" if adj == "none" else ("hfq" if adj == "hfq" else "qfq")
    return packs.get(key) or []


def _try_eastmoney(ak: Any, *, req: FetchRequest) -> list[PriceRow]:
    try:
        return fetch_etf_daily_eastmoney(
            ak,
            EastmoneyFetchRequest(
                code=req.code,
                start_date=req.start_date,
                end_date=req.end_date,
                adjust=req.adjust,
            ),
        )
    except Exception:  # pylint: disable=broad-exception-caught
        return []


def fetch_etf_daily_with_fallback(
    *,
    ak: Any,
    req: FetchRequest,
) -> tuple[list[PriceRow], dict[str, Any]]:
    """
    Fetch prices in priority order:
      1) Tencent (qfq/hfq/none)
      2) Sina: none + factor -> derive qfq/hfq
      3) Eastmoney (akshare)

    Returns (rows, meta):
    - rows: standardized PriceRow list (may be empty)
    - meta: {primary_source, fallback_used, errors}
    """
    meta: dict[str, Any] = {
        "primary_source": "tencent",
        "fallback_used": False,
        "errors": {},
    }

    rows = _try_tencent(ak, req=req)
    if rows:
        return rows, meta
    meta["errors"]["tencent"] = "empty_or_failed"

    rows2 = _try_sina(ak, req=req)
    if rows2:
        meta["fallback_used"] = True
        meta["secondary_source"] = "sina"
        return rows2, meta
    meta["errors"]["sina"] = "empty_or_failed"

    rows3 = _try_eastmoney(ak, req=req)
    if rows3:
        meta["fallback_used"] = True
        meta["secondary_source"] = "eastmoney"
        return rows3, meta
    meta["errors"]["eastmoney"] = "empty_or_failed"
    return [], meta

