from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
from zoneinfo import ZoneInfo

from fastapi import FastAPI

from etf_momentum.api.deps import get_akshare
from etf_momentum.calendar.trading_calendar import is_trading_day
from etf_momentum.data.ingestion import ingest_one_etf
from etf_momentum.db.repo import get_etf_pool_by_code, get_price_date_range, upsert_etf_pool, update_etf_pool_data_range
from etf_momentum.db.session import session_scope
from etf_momentum.settings import get_settings

logger = logging.getLogger(__name__)

_ALL_ADJUSTS = ("qfq", "hfq", "none")

# fixed mini-program pool
_FIXED_CODES = ["159915", "511010", "513100", "518880"]
_FIXED_NAMES = {"159915": "创业板ETF", "511010": "国债ETF", "513100": "纳指ETF", "518880": "黄金ETF"}


def _ymd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def _date_from_ymd(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y%m%d").date()


def _next_day_ymd(s: str) -> str:
    d = _date_from_ymd(s)
    return _ymd(d + dt.timedelta(days=1))


def _compute_next_run(now: dt.datetime, *, hour: int, minute: int, tz: ZoneInfo) -> dt.datetime:
    local_now = now.astimezone(tz)
    target = local_now.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
    if target <= local_now:
        target = target + dt.timedelta(days=1)
    return target


def _sync_fixed_pool_prices(*, app: FastAPI, run_date: dt.date) -> dict[str, object]:
    """
    Run one sync pass:
    - If pool doesn't contain the fixed codes, auto-create them.
    - For qfq/hfq/none: refresh full range for consistency (history can be revised).
    """
    settings = get_settings()
    sf = app.state.session_factory
    ak = get_akshare()

    out: dict[str, object] = {"date": _ymd(run_date), "codes": {}, "ok": True}

    with session_scope(sf) as db:
        # ensure pool entries exist
        for code in _FIXED_CODES:
            if get_etf_pool_by_code(db, code) is None:
                upsert_etf_pool(db, code=code, name=_FIXED_NAMES.get(code, code), start_date=None, end_date=None)

        for code in _FIXED_CODES:
            code_out: dict[str, object] = {"adjusts": {}, "ok": True}
            for adj in _ALL_ADJUSTS:
                try:
                    pool = get_etf_pool_by_code(db, code)
                    pool_start = (pool.start_date if pool is not None and pool.start_date else None) or settings.default_start_date

                    if bool(settings.auto_sync_full_refresh):
                        start = pool_start
                    else:
                        _, last = get_price_date_range(db, code=code, adjust=adj)
                        start = _next_day_ymd(last) if last else pool_start
                    end = _ymd(run_date)
                    if start > end:
                        code_out["adjusts"][adj] = {"skipped": True, "reason": "up_to_date", "start": start, "end": end}
                        continue
                    res = ingest_one_etf(db, ak=ak, code=code, start_date=start, end_date=end, adjust=adj)
                    code_out["adjusts"][adj] = {"skipped": False, "status": res.status, "upserted": int(res.upserted), "start": start, "end": end}
                    if res.status != "success":
                        code_out["ok"] = False
                        out["ok"] = False
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.exception("auto_sync failed: code=%s adj=%s", code, adj)
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


async def _market_sync_loop(app: FastAPI) -> None:
    settings = get_settings()
    tz = ZoneInfo(settings.auto_sync_tz)
    cal = settings.auto_sync_calendar
    hour = int(settings.auto_sync_hour)
    minute = int(settings.auto_sync_minute)

    logger.info("auto_sync scheduler started (tz=%s time=%02d:%02d cal=%s enabled=%s)", settings.auto_sync_tz, hour, minute, cal, settings.auto_sync_enabled)

    # serialize runs in-process
    lock = asyncio.Lock()

    while True:
        now = dt.datetime.now(tz=tz)
        next_run = _compute_next_run(now, hour=hour, minute=minute, tz=tz)
        sleep_s = max(1.0, (next_run - now).total_seconds())
        await asyncio.sleep(sleep_s)

        run_date = dt.datetime.now(tz=tz).date()
        if not is_trading_day(run_date, cal=cal):
            logger.info("auto_sync: %s is not trading day; skip", run_date.isoformat())
            continue

        async with lock:
            # run blocking ingestion in a thread (avoid blocking event loop)
            logger.info("auto_sync: start sync for %s", run_date.isoformat())
            res = await asyncio.to_thread(_sync_fixed_pool_prices, app=app, run_date=run_date)
            ok = bool(res.get("ok"))
            logger.info("auto_sync: done sync for %s ok=%s", run_date.isoformat(), ok)


def start_auto_sync(app: FastAPI) -> None:
    """
    Start background market sync task and store it on app.state.
    """
    # Avoid running scheduler under pytest to keep tests deterministic & offline.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        logger.info("auto_sync disabled under pytest")
        return

    settings = get_settings()
    if not bool(settings.auto_sync_enabled):
        logger.info("auto_sync disabled by settings")
        return

    if getattr(app.state, "auto_sync_task", None) is not None:
        return

    app.state.auto_sync_task = asyncio.create_task(_market_sync_loop(app))


async def stop_auto_sync(app: FastAPI) -> None:
    t = getattr(app.state, "auto_sync_task", None)
    if t is None:
        return
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass
    finally:
        app.state.auto_sync_task = None

