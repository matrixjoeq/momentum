from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from .models import OffFundEvent, OffFundNav, OffFundPool


@dataclass(frozen=True)
class OffFundNavRow:
    code: str
    trade_date: dt.date
    nav: float | None
    accum_nav: float | None = None
    source: str = "eastmoney"
    adjust: str = "none"  # none/qfq/hfq


@dataclass(frozen=True)
class OffFundEventRow:
    code: str
    effective_date: dt.date
    event_type: str  # dividend|split
    event_key: str
    cash_dividend: float | None = None
    split_ratio: float | None = None
    raw_payload: str | None = None
    source: str = "eastmoney"


def normalize_adjust(adjust: str | None) -> str:
    a = str(adjust or "none").strip().lower()
    if a in {"", "raw", "nfq"}:
        a = "none"
    if a not in {"none", "qfq", "hfq"}:
        raise ValueError(f"invalid adjust={adjust}")
    return a


def upsert_off_fund_pool(
    db: Session,
    *,
    code: str,
    name: str,
    start_date: str | None,
    end_date: str | None,
) -> OffFundPool:
    existing = db.execute(select(OffFundPool).where(OffFundPool.code == code)).scalar_one_or_none()
    if existing is None:
        obj = OffFundPool(code=code, name=name, start_date=start_date, end_date=end_date)
        db.add(obj)
        db.flush()
        return obj
    existing.name = name
    existing.start_date = start_date
    existing.end_date = end_date
    db.flush()
    return existing


def list_off_fund_pool(db: Session) -> list[OffFundPool]:
    return list(db.execute(select(OffFundPool).order_by(OffFundPool.code.asc())).scalars().all())


def get_off_fund_pool_by_code(db: Session, code: str) -> OffFundPool | None:
    return db.execute(select(OffFundPool).where(OffFundPool.code == code)).scalar_one_or_none()


def delete_off_fund_pool(db: Session, code: str) -> bool:
    obj = get_off_fund_pool_by_code(db, code)
    if obj is None:
        return False
    db.delete(obj)
    db.flush()
    return True


def purge_off_fund_data(db: Session, *, code: str) -> dict[str, int]:
    """
    Permanently delete all persisted off-fund data for one code:
    - nav series (none/qfq/hfq)
    - event records (dividend/split)
    """
    r_navs = db.execute(delete(OffFundNav).where(OffFundNav.code == code))
    r_events = db.execute(delete(OffFundEvent).where(OffFundEvent.code == code))
    return {
        "navs": int(getattr(r_navs, "rowcount", 0) or 0),
        "events": int(getattr(r_events, "rowcount", 0) or 0),
    }


def list_off_fund_navs(
    db: Session,
    *,
    code: str,
    adjust: str = "none",
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    limit: int = 5000,
) -> list[OffFundNav]:
    adj = normalize_adjust(adjust)
    stmt = select(OffFundNav).where(OffFundNav.code == code, OffFundNav.adjust == adj)
    if start_date is not None:
        stmt = stmt.where(OffFundNav.trade_date >= start_date)
    if end_date is not None:
        stmt = stmt.where(OffFundNav.trade_date <= end_date)
    stmt = stmt.order_by(OffFundNav.trade_date.asc()).limit(limit)
    return list(db.execute(stmt).scalars().all())


def delete_off_fund_navs(
    db: Session,
    *,
    code: str,
    adjust: str | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
) -> int:
    stmt = delete(OffFundNav).where(OffFundNav.code == code)
    if adjust is not None:
        stmt = stmt.where(OffFundNav.adjust == normalize_adjust(adjust))
    if start_date is not None:
        stmt = stmt.where(OffFundNav.trade_date >= start_date)
    if end_date is not None:
        stmt = stmt.where(OffFundNav.trade_date <= end_date)
    res = db.execute(stmt)
    return int(getattr(res, "rowcount", 0) or 0)


def upsert_off_fund_navs(db: Session, rows: list[OffFundNavRow]) -> int:
    if not rows:
        return 0
    values = [
        {
            "code": r.code,
            "trade_date": r.trade_date,
            "nav": r.nav,
            "accum_nav": r.accum_nav,
            "source": r.source,
            "adjust": normalize_adjust(r.adjust),
        }
        for r in rows
    ]
    dialect = (db.get_bind().dialect.name if db.get_bind() is not None else "").lower()
    if dialect == "mysql":
        stmt = mysql_insert(OffFundNav).values(values)
        stmt = stmt.on_duplicate_key_update(
            {
                "nav": stmt.inserted.nav,
                "accum_nav": stmt.inserted.accum_nav,
                "source": stmt.inserted.source,
                "adjust": stmt.inserted.adjust,
                "ingested_at": dt.datetime.now(dt.timezone.utc),
            }
        )
    else:
        stmt = sqlite_insert(OffFundNav).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[OffFundNav.code, OffFundNav.trade_date, OffFundNav.adjust],
            set_={
                "nav": stmt.excluded.nav,
                "accum_nav": stmt.excluded.accum_nav,
                "source": stmt.excluded.source,
                "adjust": stmt.excluded.adjust,
                "ingested_at": dt.datetime.now(dt.timezone.utc),
            },
        )
    res = db.execute(stmt)
    return int(getattr(res, "rowcount", 0) or 0)


def replace_off_fund_events(db: Session, *, code: str, events: list[OffFundEventRow]) -> int:
    db.execute(delete(OffFundEvent).where(OffFundEvent.code == code))
    if not events:
        return 0
    values = [
        {
            "code": e.code,
            "effective_date": e.effective_date,
            "event_type": e.event_type,
            "event_key": e.event_key,
            "cash_dividend": e.cash_dividend,
            "split_ratio": e.split_ratio,
            "raw_payload": e.raw_payload,
            "source": e.source,
        }
        for e in events
    ]
    dialect = (db.get_bind().dialect.name if db.get_bind() is not None else "").lower()
    if dialect == "mysql":
        stmt = mysql_insert(OffFundEvent).values(values)
        stmt = stmt.on_duplicate_key_update(
            {
                "cash_dividend": stmt.inserted.cash_dividend,
                "split_ratio": stmt.inserted.split_ratio,
                "raw_payload": stmt.inserted.raw_payload,
                "source": stmt.inserted.source,
                "ingested_at": dt.datetime.now(dt.timezone.utc),
            }
        )
    else:
        stmt = sqlite_insert(OffFundEvent).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[OffFundEvent.code, OffFundEvent.effective_date, OffFundEvent.event_type, OffFundEvent.event_key],
            set_={
                "cash_dividend": stmt.excluded.cash_dividend,
                "split_ratio": stmt.excluded.split_ratio,
                "raw_payload": stmt.excluded.raw_payload,
                "source": stmt.excluded.source,
                "ingested_at": dt.datetime.now(dt.timezone.utc),
            },
        )
    res = db.execute(stmt)
    return int(getattr(res, "rowcount", 0) or 0)


def get_off_fund_date_range(db: Session, *, code: str, adjust: str = "none") -> tuple[str | None, str | None]:
    adj = normalize_adjust(adjust)
    start_d, end_d = db.execute(
        select(func.min(OffFundNav.trade_date), func.max(OffFundNav.trade_date)).where(
            OffFundNav.code == code,
            OffFundNav.adjust == adj,
        )
    ).one()
    if start_d is None or end_d is None:
        return (None, None)
    return (start_d.strftime("%Y%m%d"), end_d.strftime("%Y%m%d"))


def update_off_fund_pool_data_range(db: Session, *, code: str, adjust: str = "hfq") -> tuple[str | None, str | None]:
    obj = get_off_fund_pool_by_code(db, code)
    if obj is None:
        return (None, None)
    start, end = get_off_fund_date_range(db, code=code, adjust=adjust)
    obj.last_data_start_date = start
    obj.last_data_end_date = end
    db.flush()
    return (start, end)


def mark_off_fund_fetch_status(
    db: Session,
    *,
    code: str,
    status: str,
    message: str | None = None,
    when: dt.datetime | None = None,
) -> None:
    obj = get_off_fund_pool_by_code(db, code)
    if obj is None:
        return
    msg = None if message is None else str(message)
    if msg is not None and len(msg) > 512:
        msg = msg[:498] + "...(truncated)"
    obj.last_fetch_at = when or dt.datetime.now(dt.timezone.utc)
    obj.last_fetch_status = status
    obj.last_fetch_message = msg
    db.flush()

