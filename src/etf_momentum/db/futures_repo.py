from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from .models import FuturesPool, FuturesPrice


@dataclass(frozen=True)
class FuturesPriceRow:
    code: str
    trade_date: dt.date
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None = None
    amount: float | None = None
    open_interest: float | None = None
    source: str = "sina"
    adjust: str = "none"


def normalize_futures_adjust(adjust: str | None) -> str:
    a = str(adjust or "none").strip().lower()
    if a in {"", "raw", "nfq"}:
        a = "none"
    if a != "none":
        raise ValueError(f"invalid adjust={adjust}; futures only support none")
    return a


def upsert_futures_pool(
    db: Session,
    *,
    code: str,
    name: str,
    start_date: str | None,
    end_date: str | None,
) -> FuturesPool:
    existing = db.execute(select(FuturesPool).where(FuturesPool.code == code)).scalar_one_or_none()
    if existing is None:
        obj = FuturesPool(code=code, name=name, start_date=start_date, end_date=end_date)
        db.add(obj)
        db.flush()
        return obj
    existing.name = name
    existing.start_date = start_date
    existing.end_date = end_date
    db.flush()
    return existing


def list_futures_pool(db: Session) -> list[FuturesPool]:
    return list(db.execute(select(FuturesPool).order_by(FuturesPool.code.asc())).scalars().all())


def get_futures_pool_by_code(db: Session, code: str) -> FuturesPool | None:
    return db.execute(select(FuturesPool).where(FuturesPool.code == code)).scalar_one_or_none()


def delete_futures_pool(db: Session, code: str) -> bool:
    obj = get_futures_pool_by_code(db, code)
    if obj is None:
        return False
    db.delete(obj)
    db.flush()
    return True


def upsert_futures_prices(db: Session, rows: list[FuturesPriceRow]) -> int:
    if not rows:
        return 0
    values = [
        {
            "code": r.code,
            "trade_date": r.trade_date,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
            "amount": r.amount,
            "open_interest": r.open_interest,
            "source": r.source,
            "adjust": normalize_futures_adjust(r.adjust),
        }
        for r in rows
    ]
    dialect = (db.get_bind().dialect.name if db.get_bind() is not None else "").lower()
    if dialect == "mysql":
        stmt = mysql_insert(FuturesPrice).values(values)
        stmt = stmt.on_duplicate_key_update(
            {
                "open": stmt.inserted.open,
                "high": stmt.inserted.high,
                "low": stmt.inserted.low,
                "close": stmt.inserted.close,
                "volume": stmt.inserted.volume,
                "amount": stmt.inserted.amount,
                "open_interest": stmt.inserted.open_interest,
                "source": stmt.inserted.source,
                "adjust": stmt.inserted.adjust,
                "ingested_at": dt.datetime.now(dt.timezone.utc),
            }
        )
    else:
        stmt = sqlite_insert(FuturesPrice).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[FuturesPrice.code, FuturesPrice.trade_date, FuturesPrice.adjust],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
                "amount": stmt.excluded.amount,
                "open_interest": stmt.excluded.open_interest,
                "source": stmt.excluded.source,
                "adjust": stmt.excluded.adjust,
                "ingested_at": dt.datetime.now(dt.timezone.utc),
            },
        )
    res = db.execute(stmt)
    return int(getattr(res, "rowcount", 0) or 0)


def list_futures_prices(
    db: Session,
    *,
    code: str,
    adjust: str = "none",
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    limit: int = 5000,
) -> list[FuturesPrice]:
    adj = normalize_futures_adjust(adjust)
    stmt = select(FuturesPrice).where(FuturesPrice.code == code, FuturesPrice.adjust == adj)
    if start_date is not None:
        stmt = stmt.where(FuturesPrice.trade_date >= start_date)
    if end_date is not None:
        stmt = stmt.where(FuturesPrice.trade_date <= end_date)
    stmt = stmt.order_by(FuturesPrice.trade_date.asc()).limit(limit)
    return list(db.execute(stmt).scalars().all())


def delete_futures_prices(
    db: Session,
    *,
    code: str,
    adjust: str | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
) -> int:
    stmt = delete(FuturesPrice).where(FuturesPrice.code == code)
    if adjust is not None:
        stmt = stmt.where(FuturesPrice.adjust == normalize_futures_adjust(adjust))
    if start_date is not None:
        stmt = stmt.where(FuturesPrice.trade_date >= start_date)
    if end_date is not None:
        stmt = stmt.where(FuturesPrice.trade_date <= end_date)
    res = db.execute(stmt)
    return int(getattr(res, "rowcount", 0) or 0)


def get_futures_date_range(db: Session, *, code: str, adjust: str = "none") -> tuple[str | None, str | None]:
    adj = normalize_futures_adjust(adjust)
    start_d, end_d = db.execute(
        select(func.min(FuturesPrice.trade_date), func.max(FuturesPrice.trade_date)).where(
            FuturesPrice.code == code,
            FuturesPrice.adjust == adj,
        )
    ).one()
    if start_d is None or end_d is None:
        return (None, None)
    return (start_d.strftime("%Y%m%d"), end_d.strftime("%Y%m%d"))


def update_futures_pool_data_range(db: Session, *, code: str, adjust: str = "none") -> tuple[str | None, str | None]:
    obj = get_futures_pool_by_code(db, code)
    if obj is None:
        return (None, None)
    start, end = get_futures_date_range(db, code=code, adjust=adjust)
    obj.last_data_start_date = start
    obj.last_data_end_date = end
    db.flush()
    return (start, end)


def mark_futures_fetch_status(
    db: Session,
    *,
    code: str,
    status: str,
    message: str | None = None,
    when: dt.datetime | None = None,
) -> None:
    obj = get_futures_pool_by_code(db, code)
    if obj is None:
        return
    msg = None if message is None else str(message)
    if msg is not None and len(msg) > 512:
        msg = msg[:498] + "...(truncated)"
    obj.last_fetch_at = when or dt.datetime.now(dt.timezone.utc)
    obj.last_fetch_status = status
    obj.last_fetch_message = msg
    db.flush()
