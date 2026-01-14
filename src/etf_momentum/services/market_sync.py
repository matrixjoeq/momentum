from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Iterable

from sqlalchemy.orm import Session

from etf_momentum.data.ingestion import ingest_one_etf
from etf_momentum.db.repo import get_etf_pool_by_code, get_price_date_range, upsert_etf_pool, update_etf_pool_data_range
from etf_momentum.settings import get_settings

logger = logging.getLogger(__name__)

_DEFAULT_ADJUSTS = ("qfq", "hfq", "none")

# fixed mini-program pool
FIXED_CODES = ["159915", "511010", "513100", "518880"]
FIXED_NAMES = {"159915": "创业板ETF", "511010": "国债ETF", "513100": "纳指ETF", "518880": "黄金ETF"}


def _ymd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def _date_from_ymd(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y%m%d").date()


def _next_day_ymd(s: str) -> str:
    d = _date_from_ymd(s)
    return _ymd(d + dt.timedelta(days=1))


def sync_fixed_pool_prices(
    *,
    db: Session,
    ak: Any,
    run_date: dt.date,
    adjusts: Iterable[str] | None = None,
    full_refresh: bool | None = None,
) -> dict[str, object]:
    """
    Sync the fixed 4-ETF pool prices into SQLite for the given date (inclusive).

    Notes:
    - Ensures pool entries exist.
    - For each adjust in (qfq/hfq/none), refreshes either full history (default) or incrementally.
    - Returns a structured summary for logging / cloud trigger diagnostics.
    """
    settings = get_settings()
    adj_list = list(adjusts) if adjusts is not None else list(_DEFAULT_ADJUSTS)
    adj_list = [str(x).strip().lower() for x in adj_list if str(x).strip()]
    if not adj_list:
        adj_list = list(_DEFAULT_ADJUSTS)
    if any(x not in set(_DEFAULT_ADJUSTS) for x in adj_list):
        raise ValueError(f"adjusts must be subset of {_DEFAULT_ADJUSTS}; got {adj_list}")

    do_full = bool(settings.auto_sync_full_refresh if full_refresh is None else full_refresh)

    out: dict[str, object] = {
        "date": _ymd(run_date),
        "ok": True,
        "full_refresh": do_full,
        "adjusts": adj_list,
        "codes": {},
    }

    # ensure pool entries exist
    for code in FIXED_CODES:
        if get_etf_pool_by_code(db, code) is None:
            upsert_etf_pool(db, code=code, name=FIXED_NAMES.get(code, code), start_date=None, end_date=None)
    db.flush()

    for code in FIXED_CODES:
        code_out: dict[str, object] = {"ok": True, "adjusts": {}}
        for adj in adj_list:
            try:
                pool = get_etf_pool_by_code(db, code)
                pool_start = (pool.start_date if pool is not None and pool.start_date else None) or settings.default_start_date

                if do_full:
                    start = pool_start
                else:
                    _, last = get_price_date_range(db, code=code, adjust=adj)
                    start = _next_day_ymd(last) if last else pool_start
                end = _ymd(run_date)
                if start > end:
                    code_out["adjusts"][adj] = {"skipped": True, "reason": "up_to_date", "start": start, "end": end}
                    continue

                res = ingest_one_etf(db, ak=ak, code=code, start_date=start, end_date=end, adjust=adj)
                code_out["adjusts"][adj] = {
                    "skipped": False,
                    "status": res.status,
                    "upserted": int(res.upserted),
                    "start": start,
                    "end": end,
                }
                if res.status != "success":
                    code_out["ok"] = False
                    out["ok"] = False
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.exception("sync_fixed_pool_prices failed: code=%s adj=%s", code, adj)
                code_out["adjusts"][adj] = {"skipped": False, "status": "failed", "error": str(e)}
                code_out["ok"] = False
                out["ok"] = False

        # update overall pool coverage range (any adjust)
        try:
            update_etf_pool_data_range(db, code=code)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        out["codes"][code] = code_out

    return out

