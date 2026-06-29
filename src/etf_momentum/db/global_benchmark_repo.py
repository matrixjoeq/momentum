from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from .models import GlobalBenchmarkPool, GlobalBenchmarkPrice


def normalize_adjust(adjust: str | None) -> str:
    a = str(adjust or "none").strip().lower()
    if a in {"", "raw", "nfq"}:
        a = "none"
    if a != "none":
        raise ValueError(f"global benchmark only supports adjust=none, got {adjust}")
    return a


@dataclass(frozen=True)
class GlobalBenchmarkPriceRow:
    code: str
    trade_date: dt.date
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    amount: float | None = None
    source: str = "unknown"
    adjust: str = "none"


def upsert_global_benchmark_pool(
    db: Session,
    *,
    code: str,
    name: str,
    code_format: str | None,
    provider_hint: str | None,
    start_date: str | None,
    end_date: str | None,
) -> GlobalBenchmarkPool:
    existing = db.execute(
        select(GlobalBenchmarkPool).where(GlobalBenchmarkPool.code == code)
    ).scalar_one_or_none()
    if existing is None:
        obj = GlobalBenchmarkPool(
            code=code,
            name=name,
            code_format=code_format,
            provider_hint=provider_hint,
            start_date=start_date,
            end_date=end_date,
        )
        db.add(obj)
        db.flush()
        return obj
    existing.name = name
    existing.code_format = code_format
    existing.provider_hint = provider_hint
    existing.start_date = start_date
    existing.end_date = end_date
    db.flush()
    return existing


def list_global_benchmark_pool(db: Session) -> list[GlobalBenchmarkPool]:
    return list(
        db.execute(
            select(GlobalBenchmarkPool).order_by(GlobalBenchmarkPool.code.asc())
        ).scalars()
    )


def get_global_benchmark_pool_by_code(
    db: Session, code: str
) -> GlobalBenchmarkPool | None:
    return db.execute(
        select(GlobalBenchmarkPool).where(GlobalBenchmarkPool.code == code)
    ).scalar_one_or_none()


def delete_global_benchmark_pool(db: Session, code: str) -> bool:
    obj = get_global_benchmark_pool_by_code(db, code)
    if obj is None:
        return False
    db.delete(obj)
    db.flush()
    return True


def purge_global_benchmark_data(db: Session, *, code: str) -> dict[str, int]:
    r_prices = db.execute(
        delete(GlobalBenchmarkPrice).where(GlobalBenchmarkPrice.code == code)
    )
    return {"prices": int(getattr(r_prices, "rowcount", 0) or 0)}


def mark_global_benchmark_fetch_status(
    db: Session,
    *,
    code: str,
    status: str,
    message: str | None = None,
    when: dt.datetime | None = None,
) -> None:
    obj = get_global_benchmark_pool_by_code(db, code)
    if obj is None:
        return
    msg = None if message is None else str(message)
    if msg is not None and len(msg) > 512:
        msg = msg[:498] + "...(truncated)"
    obj.last_fetch_at = when or dt.datetime.now(dt.timezone.utc)
    obj.last_fetch_status = status
    obj.last_fetch_message = msg
    db.flush()


def upsert_global_benchmark_prices(
    db: Session, rows: list[GlobalBenchmarkPriceRow]
) -> int:
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
            "source": r.source,
            "adjust": normalize_adjust(r.adjust),
        }
        for r in rows
    ]
    dialect = (db.get_bind().dialect.name if db.get_bind() is not None else "").lower()
    if dialect == "mysql":
        stmt = mysql_insert(GlobalBenchmarkPrice).values(values)
        stmt = stmt.on_duplicate_key_update(
            {
                "open": stmt.inserted.open,
                "high": stmt.inserted.high,
                "low": stmt.inserted.low,
                "close": stmt.inserted.close,
                "volume": stmt.inserted.volume,
                "amount": stmt.inserted.amount,
                "source": stmt.inserted.source,
                "adjust": stmt.inserted.adjust,
                "ingested_at": dt.datetime.now(dt.timezone.utc),
            }
        )
    else:
        stmt = sqlite_insert(GlobalBenchmarkPrice).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                GlobalBenchmarkPrice.code,
                GlobalBenchmarkPrice.trade_date,
                GlobalBenchmarkPrice.adjust,
            ],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
                "amount": stmt.excluded.amount,
                "source": stmt.excluded.source,
                "adjust": stmt.excluded.adjust,
                "ingested_at": dt.datetime.now(dt.timezone.utc),
            },
        )
    res = db.execute(stmt)
    return int(getattr(res, "rowcount", 0) or 0)


def list_global_benchmark_prices(
    db: Session,
    *,
    code: str,
    adjust: str = "none",
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    limit: int = 5000,
) -> list[GlobalBenchmarkPrice]:
    adj = normalize_adjust(adjust)
    stmt = select(GlobalBenchmarkPrice).where(
        GlobalBenchmarkPrice.code == code,
        GlobalBenchmarkPrice.adjust == adj,
    )
    if start_date is not None:
        stmt = stmt.where(GlobalBenchmarkPrice.trade_date >= start_date)
    if end_date is not None:
        stmt = stmt.where(GlobalBenchmarkPrice.trade_date <= end_date)
    stmt = stmt.order_by(GlobalBenchmarkPrice.trade_date.asc()).limit(limit)
    return list(db.execute(stmt).scalars())


def get_global_benchmark_date_range(
    db: Session, *, code: str, adjust: str = "none"
) -> tuple[str | None, str | None]:
    adj = normalize_adjust(adjust)
    start_d, end_d = db.execute(
        select(
            func.min(GlobalBenchmarkPrice.trade_date),
            func.max(GlobalBenchmarkPrice.trade_date),
        ).where(
            GlobalBenchmarkPrice.code == code,
            GlobalBenchmarkPrice.adjust == adj,
        )
    ).one()
    if start_d is None or end_d is None:
        return (None, None)
    return (start_d.strftime("%Y%m%d"), end_d.strftime("%Y%m%d"))


def update_global_benchmark_pool_data_range(
    db: Session, *, code: str, adjust: str = "none"
) -> tuple[str | None, str | None]:
    obj = get_global_benchmark_pool_by_code(db, code)
    if obj is None:
        return (None, None)
    start, end = get_global_benchmark_date_range(db, code=code, adjust=adjust)
    obj.last_data_start_date = start
    obj.last_data_end_date = end
    db.flush()
    return (start, end)
