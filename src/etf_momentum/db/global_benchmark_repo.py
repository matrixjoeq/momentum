from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

from sqlalchemy import delete, func as sa_func, select
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


def normalize_series_kind(series_kind: str | None) -> str:
    k = str(series_kind or "price").strip().lower()
    if k not in {"price", "total_return"}:
        raise ValueError(f"unsupported series_kind: {series_kind}")
    return k


@dataclass(frozen=True)
class GlobalBenchmarkPriceRow:
    code: str
    series_kind: str
    trade_date: dt.date
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    amount: float | None = None
    source: str = "unknown"
    adjust: str = "none"


def _encode_fallback_sources(v: list[dict[str, str]] | None) -> str | None:
    if not v:
        return None
    return json.dumps(v, ensure_ascii=False)


def _decode_fallback_sources(v: str | None) -> list[dict[str, str]]:
    if not v:
        return []
    try:
        x = json.loads(v)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(x, list):
        return []
    out: list[dict[str, str]] = []
    for item in x:
        if not isinstance(item, dict):
            continue
        p = str(item.get("provider") or "").strip()
        s = str(item.get("symbol") or "").strip()
        if not p or not s:
            continue
        out.append({"provider": p, "symbol": s})
    return out


def upsert_global_benchmark_pool(
    db: Session,
    *,
    code: str,
    series_kind: str,
    name: str,
    code_format: str | None,
    provider_hint: str | None,
    provider_symbol: str | None,
    source_locked: bool | None,
    fallback_sources: list[dict[str, str]] | None,
    start_date: str | None,
    end_date: str | None,
) -> GlobalBenchmarkPool:
    kind = normalize_series_kind(series_kind)
    existing = db.execute(
        select(GlobalBenchmarkPool).where(
            GlobalBenchmarkPool.code == code,
            GlobalBenchmarkPool.series_kind == kind,
        )
    ).scalar_one_or_none()
    if existing is None:
        obj = GlobalBenchmarkPool(
            code=code,
            series_kind=kind,
            name=name,
            code_format=code_format,
            provider_hint=provider_hint,
            provider_symbol=provider_symbol,
            source_locked=bool(source_locked) if source_locked is not None else False,
            fallback_sources_json=_encode_fallback_sources(fallback_sources),
            start_date=start_date,
            end_date=end_date,
        )
        db.add(obj)
        db.flush()
        return obj
    existing.name = name
    existing.code_format = code_format
    existing.provider_hint = provider_hint
    existing.provider_symbol = provider_symbol
    if source_locked is not None:
        existing.source_locked = bool(source_locked)
    existing.fallback_sources_json = _encode_fallback_sources(fallback_sources)
    existing.start_date = start_date
    existing.end_date = end_date
    db.flush()
    return existing


def list_global_benchmark_pool(
    db: Session,
    *,
    code: str | None = None,
    series_kind: str | None = None,
) -> list[GlobalBenchmarkPool]:
    stmt = select(GlobalBenchmarkPool)
    if code is not None:
        stmt = stmt.where(GlobalBenchmarkPool.code == str(code))
    if series_kind is not None:
        stmt = stmt.where(
            GlobalBenchmarkPool.series_kind == normalize_series_kind(series_kind)
        )
    stmt = stmt.order_by(
        GlobalBenchmarkPool.code.asc(), GlobalBenchmarkPool.series_kind.asc()
    )
    return list(db.execute(stmt).scalars())


def get_global_benchmark_pool_by_code(
    db: Session, code: str, *, series_kind: str = "price"
) -> GlobalBenchmarkPool | None:
    kind = normalize_series_kind(series_kind)
    return db.execute(
        select(GlobalBenchmarkPool).where(
            GlobalBenchmarkPool.code == code,
            GlobalBenchmarkPool.series_kind == kind,
        )
    ).scalar_one_or_none()


def get_global_benchmark_pool_series(
    db: Session, code: str
) -> dict[str, GlobalBenchmarkPool]:
    rows = list_global_benchmark_pool(db, code=code)
    return {str(x.series_kind): x for x in rows}


def delete_global_benchmark_pool(
    db: Session, code: str, *, series_kind: str | None = None
) -> bool:
    rows = list_global_benchmark_pool(db, code=code, series_kind=series_kind)
    if not rows:
        return False
    for x in rows:
        db.delete(x)
    db.flush()
    return True


def purge_global_benchmark_data(
    db: Session, *, code: str, series_kind: str | None = None
) -> dict[str, int]:
    stmt = delete(GlobalBenchmarkPrice).where(GlobalBenchmarkPrice.code == code)
    if series_kind is not None:
        stmt = stmt.where(
            GlobalBenchmarkPrice.series_kind == normalize_series_kind(series_kind)
        )
    r_prices = db.execute(stmt)
    return {"prices": int(getattr(r_prices, "rowcount", 0) or 0)}


def mark_global_benchmark_fetch_status(
    db: Session,
    *,
    code: str,
    series_kind: str,
    status: str,
    message: str | None = None,
    when: dt.datetime | None = None,
) -> None:
    obj = get_global_benchmark_pool_by_code(db, code, series_kind=series_kind)
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
            "series_kind": normalize_series_kind(r.series_kind),
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
                GlobalBenchmarkPrice.series_kind,
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
    series_kind: str = "price",
    adjust: str = "none",
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    limit: int = 5000,
) -> list[GlobalBenchmarkPrice]:
    adj = normalize_adjust(adjust)
    kind = normalize_series_kind(series_kind)
    stmt = select(GlobalBenchmarkPrice).where(
        GlobalBenchmarkPrice.code == code,
        GlobalBenchmarkPrice.series_kind == kind,
        GlobalBenchmarkPrice.adjust == adj,
    )
    if start_date is not None:
        stmt = stmt.where(GlobalBenchmarkPrice.trade_date >= start_date)
    if end_date is not None:
        stmt = stmt.where(GlobalBenchmarkPrice.trade_date <= end_date)
    stmt = stmt.order_by(GlobalBenchmarkPrice.trade_date.asc()).limit(limit)
    return list(db.execute(stmt).scalars())


def get_global_benchmark_date_range(
    db: Session,
    *,
    code: str,
    series_kind: str = "price",
    adjust: str = "none",
) -> tuple[str | None, str | None]:
    adj = normalize_adjust(adjust)
    kind = normalize_series_kind(series_kind)
    start_d, end_d = db.execute(
        select(
            sa_func.min(GlobalBenchmarkPrice.trade_date),
            sa_func.max(GlobalBenchmarkPrice.trade_date),
        ).where(
            GlobalBenchmarkPrice.code == code,
            GlobalBenchmarkPrice.series_kind == kind,
            GlobalBenchmarkPrice.adjust == adj,
        )
    ).one()
    if start_d is None or end_d is None:
        return (None, None)
    return (start_d.strftime("%Y%m%d"), end_d.strftime("%Y%m%d"))


def count_global_benchmark_prices(
    db: Session,
    *,
    code: str,
    series_kind: str = "price",
    adjust: str = "none",
) -> int:
    adj = normalize_adjust(adjust)
    kind = normalize_series_kind(series_kind)
    ids = db.execute(
        select(GlobalBenchmarkPrice.id).where(
            GlobalBenchmarkPrice.code == code,
            GlobalBenchmarkPrice.series_kind == kind,
            GlobalBenchmarkPrice.adjust == adj,
        )
    ).scalars()
    return int(len(list(ids)))


def list_global_benchmark_trade_dates(
    db: Session,
    *,
    code: str,
    series_kind: str = "price",
    adjust: str = "none",
) -> list[dt.date]:
    adj = normalize_adjust(adjust)
    kind = normalize_series_kind(series_kind)
    dates = db.execute(
        select(GlobalBenchmarkPrice.trade_date).where(
            GlobalBenchmarkPrice.code == code,
            GlobalBenchmarkPrice.series_kind == kind,
            GlobalBenchmarkPrice.adjust == adj,
        )
    ).scalars()
    return sorted({d for d in dates if d is not None})


def update_global_benchmark_pool_data_range(
    db: Session,
    *,
    code: str,
    series_kind: str,
    adjust: str = "none",
) -> tuple[str | None, str | None]:
    obj = get_global_benchmark_pool_by_code(db, code, series_kind=series_kind)
    if obj is None:
        return (None, None)
    start, end = get_global_benchmark_date_range(
        db, code=code, series_kind=series_kind, adjust=adjust
    )
    obj.last_data_start_date = start
    obj.last_data_end_date = end
    db.flush()
    return (start, end)


def get_fallback_sources_for_pool_item(x: GlobalBenchmarkPool) -> list[dict[str, str]]:
    return _decode_fallback_sources(getattr(x, "fallback_sources_json", None))
