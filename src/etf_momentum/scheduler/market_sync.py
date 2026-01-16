from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
from contextlib import contextmanager
from zoneinfo import ZoneInfo

from fastapi import FastAPI

from etf_momentum.calendar.trading_calendar import is_trading_day
from etf_momentum.db.repo import create_sync_job, update_sync_job
from etf_momentum.db.session import session_scope
from etf_momentum.settings import get_settings
from etf_momentum.services.market_sync import sync_fixed_pool_prices

logger = logging.getLogger(__name__)

_ALL_ADJUSTS = ("qfq", "hfq", "none")


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


def _dedupe_key(run_date: dt.date, *, full_refresh: bool, adjusts: list[str]) -> str:
    ad = ",".join([str(x).strip().lower() for x in (adjusts or []) if str(x).strip()])
    return f"auto_fixed_pool:{run_date.strftime('%Y%m%d')}:full={int(bool(full_refresh))}:adj={ad}"


@contextmanager
def _session_cm(sf):
    # session_scope is a generator; wrap it as a proper context manager for this module.
    yield from session_scope(sf)


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
            full_refresh = bool(settings.auto_sync_full_refresh)
            adjusts = list(_ALL_ADJUSTS)
            dedupe = _dedupe_key(run_date, full_refresh=full_refresh, adjusts=adjusts)

            # Create/lookup job row (unique on dedupe_key).
            sf = app.state.session_factory
            with _session_cm(sf) as db:
                job = create_sync_job(db, dedupe_key=dedupe, run_date=run_date, full_refresh=full_refresh, adjusts=adjusts)
                # If already finished, skip (avoid duplicate runs if instance restarts within the window)
                if job.finished_at is not None:
                    logger.info("auto_sync: job already finished (job_id=%s status=%s); skip", int(job.id), job.status)
                    continue
                update_sync_job(
                    db,
                    job_id=int(job.id),
                    status="running",
                    started_at=dt.datetime.now(dt.timezone.utc),
                    progress={"phase": "starting"},
                )
                db.commit()

            # Run sync in a thread (blocking), update progress via DB.
            def _progress(p: dict) -> None:
                with _session_cm(sf) as db2:
                    update_sync_job(db2, job_id=int(job.id), progress=p)
                    db2.commit()

            def _run() -> dict[str, object]:
                # Use the same service implementation as the admin API.
                with _session_cm(sf) as db3:
                    ak = __import__("akshare")
                    res = sync_fixed_pool_prices(
                        db=db3,
                        ak=ak,
                        run_date=run_date,
                        adjusts=adjusts,
                        full_refresh=full_refresh,
                        progress_cb=_progress,
                    )
                    db3.commit()
                    return res

            res = await asyncio.to_thread(_run)
            ok = bool(res.get("ok"))
            with _session_cm(sf) as db4:
                update_sync_job(
                    db4,
                    job_id=int(job.id),
                    status=("success" if ok else "failed"),
                    finished_at=dt.datetime.now(dt.timezone.utc),
                    result=dict(res),
                    error_message=(None if ok else "auto_sync failed; see result.codes for details"),
                    progress={"phase": "done", "ok": ok},
                )
                db4.commit()
            logger.info("auto_sync: done sync for %s ok=%s job_id=%s", run_date.isoformat(), ok, int(job.id))


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

