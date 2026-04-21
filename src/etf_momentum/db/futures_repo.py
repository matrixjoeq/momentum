from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from .models import FuturesContractFetchStatus, FuturesPool, FuturesPrice


@dataclass(frozen=True)
class FuturesPriceRow:
    code: str
    trade_date: dt.date
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    settle: float | None = None
    volume: float | None = None
    amount: float | None = None
    hold: float | None = None
    source: str = "sina"
    adjust: str = "none"
    pool_id: int | None = None


def normalize_futures_adjust(adjust: str | None) -> str:
    a = str(adjust or "none").strip().lower()
    if a in {"", "raw", "nfq"}:
        a = "none"
    if a != "none":
        raise ValueError(f"invalid adjust={adjust}; futures only support none")
    return a


def _futures_symbol_root(code: str) -> str:
    c = str(code or "").strip().upper()
    m = re.match(r"^([A-Z]+)", c)
    return m.group(1) if m else c


def infer_futures_default_tag(code: str, name: str | None = None) -> str:
    root = _futures_symbol_root(code)
    if root in {"IF", "IH", "IC", "IM"}:
        return "股指期货"
    if root in {"T", "TF", "TS", "TL"}:
        return "国债期货"
    if root in {"AU", "AG"}:
        return "贵金属"
    if root in {"CU", "AL", "ZN", "PB", "NI", "SN", "SS", "BC"}:
        return "有色金属"
    if root in {"RB", "HC", "I", "J", "JM", "SF", "SM"}:
        return "黑色系"
    if root in {"SC", "LU", "FU", "BU", "PG"}:
        return "能源化工"
    if root in {"RU", "NR", "SP", "L", "V", "PP", "EB", "EG", "TA", "MA", "SA", "FG", "UR"}:
        return "化工建材"
    if root in {"A", "B", "M", "Y", "P", "OI", "RM", "C", "CS", "JD", "LH", "AP", "CJ", "CF", "SR", "PK"}:
        return "农产品"
    if root in {"SI", "LC"}:
        return "新能源金属"
    n = str(name or "")
    if "国债" in n:
        return "国债期货"
    if "股指" in n:
        return "股指期货"
    return "其他期货"


def _normalize_futures_tags(code: str, name: str, tags: list[str] | None) -> list[str]:
    cleaned: list[str] = []
    for t in (tags or []):
        s = str(t or "").strip()
        if not s:
            continue
        cleaned.extend([x.strip() for x in s.split(",") if str(x).strip()])
    if not cleaned:
        cleaned = [infer_futures_default_tag(code, name)]
    seen: set[str] = set()
    out: list[str] = []
    for t in cleaned:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def serialize_futures_tags(code: str, name: str, tags: list[str] | None) -> str:
    vals = _normalize_futures_tags(code=code, name=name, tags=tags)
    return json.dumps(vals, ensure_ascii=False)


def deserialize_futures_tags(raw: str | None, *, code: str, name: str) -> list[str]:
    txt = str(raw or "").strip()
    if not txt:
        return [infer_futures_default_tag(code, name)]
    try:
        v = json.loads(txt)
        if isinstance(v, list):
            return _normalize_futures_tags(code=code, name=name, tags=[str(x) for x in v])
    except (TypeError, ValueError):
        pass
    return _normalize_futures_tags(code=code, name=name, tags=[txt])


def upsert_futures_pool(
    db: Session,
    *,
    code: str,
    name: str,
    start_date: str | None,
    end_date: str | None,
    min_margin_ratio: float | None = None,
    contract_multiplier: float | None = None,
    price_unit: str | None = None,
    min_price_tick: float | None = None,
    tags: list[str] | None = None,
    contract_extend_calendar_days: int | None = None,
    contract_parallel: int | None = None,
) -> FuturesPool:
    ext_days = int(contract_extend_calendar_days) if contract_extend_calendar_days is not None else 366
    if contract_parallel is not None:
        int(contract_parallel)
    par = 1  # deliverable-month fetch uses AkShare serial-only policy (see futures_contract_ingestion)
    existing = db.execute(select(FuturesPool).where(FuturesPool.code == code)).scalar_one_or_none()
    if existing is None:
        obj = FuturesPool(
            code=code,
            name=name,
            start_date=start_date,
            end_date=end_date,
            min_margin_ratio=min_margin_ratio,
            contract_multiplier=contract_multiplier,
            price_unit=price_unit,
            min_price_tick=min_price_tick,
            tags_json=serialize_futures_tags(code=code, name=name, tags=tags),
            contract_extend_calendar_days=ext_days,
            contract_parallel=par,
        )
        db.add(obj)
        db.flush()
        return obj
    existing.name = name
    existing.start_date = start_date
    existing.end_date = end_date
    existing.min_margin_ratio = min_margin_ratio
    existing.contract_multiplier = contract_multiplier
    existing.price_unit = price_unit
    existing.min_price_tick = min_price_tick
    existing.tags_json = serialize_futures_tags(code=code, name=name, tags=tags)
    if contract_extend_calendar_days is not None:
        existing.contract_extend_calendar_days = ext_days
    existing.contract_parallel = par
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
    pid = int(obj.id)
    db.execute(delete(FuturesContractFetchStatus).where(FuturesContractFetchStatus.pool_id == pid))
    db.execute(delete(FuturesPrice).where(FuturesPrice.pool_id == pid))
    delete_futures_prices(db, code=code)
    db.delete(obj)
    db.flush()
    return True


def upsert_futures_prices(db: Session, rows: list[FuturesPriceRow]) -> int:
    if not rows:
        return 0
    values = [
        {
            "pool_id": r.pool_id,
            "code": r.code,
            "trade_date": r.trade_date,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "settle": r.settle,
            "volume": r.volume,
            "amount": r.amount,
            "hold": r.hold,
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
                "pool_id": stmt.inserted.pool_id,
                "open": stmt.inserted.open,
                "high": stmt.inserted.high,
                "low": stmt.inserted.low,
                "close": stmt.inserted.close,
                "settle": stmt.inserted.settle,
                "volume": stmt.inserted.volume,
                "amount": stmt.inserted.amount,
                "hold": stmt.inserted.hold,
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
                "pool_id": stmt.excluded.pool_id,
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "settle": stmt.excluded.settle,
                "volume": stmt.excluded.volume,
                "amount": stmt.excluded.amount,
                "hold": stmt.excluded.hold,
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


def get_futures_last_trade_date(db: Session, *, code: str, adjust: str = "none") -> dt.date | None:
    adj = normalize_futures_adjust(adjust)
    end_d = db.execute(
        select(func.max(FuturesPrice.trade_date)).where(
            FuturesPrice.code == code,
            FuturesPrice.adjust == adj,
        )
    ).scalar_one_or_none()
    return end_d


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


def mark_futures_contract_pool_fetch(
    db: Session,
    *,
    code: str,
    status: str,
    message: str | None = None,
) -> None:
    obj = get_futures_pool_by_code(db, code)
    if obj is None:
        return
    msg = None if message is None else str(message)
    if msg is not None and len(msg) > 512:
        msg = msg[:498] + "...(truncated)"
    obj.last_contract_fetch_at = dt.datetime.now(dt.timezone.utc)
    obj.last_contract_fetch_status = status
    obj.last_contract_fetch_message = msg
    db.flush()


def record_contract_fetch_status(
    db: Session,
    *,
    pool_id: int,
    contract_code: str,
    status: str,
    message: str | None,
    rows_upserted: int,
) -> None:
    code_u = str(contract_code).strip().upper()
    existing = db.execute(
        select(FuturesContractFetchStatus).where(
            FuturesContractFetchStatus.pool_id == pool_id,
            FuturesContractFetchStatus.contract_code == code_u,
        )
    ).scalar_one_or_none()
    msg = None if message is None else str(message)
    if msg is not None and len(msg) > 512:
        msg = msg[:498] + "...(truncated)"
    end_d = get_futures_last_trade_date(db, code=code_u, adjust="none")
    end_str = end_d.strftime("%Y%m%d") if end_d is not None else None
    if existing is None:
        db.add(
            FuturesContractFetchStatus(
                pool_id=pool_id,
                contract_code=code_u,
                last_fetch_status=status,
                last_fetch_message=msg,
                rows_upserted=int(rows_upserted),
                last_data_end_date=end_str,
            )
        )
    else:
        existing.last_fetch_status = status
        existing.last_fetch_message = msg
        existing.rows_upserted = int(rows_upserted)
        existing.last_data_end_date = end_str
    db.flush()


def delete_contract_fetch_status(db: Session, *, pool_id: int, contract_code: str) -> None:
    """Remove per-contract status when there is no price data to surface (see contract ingestion)."""
    code_u = str(contract_code).strip().upper()
    obj = db.execute(
        select(FuturesContractFetchStatus).where(
            FuturesContractFetchStatus.pool_id == pool_id,
            FuturesContractFetchStatus.contract_code == code_u,
        )
    ).scalar_one_or_none()
    if obj is None:
        return
    db.delete(obj)
    db.flush()


def list_contract_fetch_statuses(db: Session, *, pool_id: int) -> list[FuturesContractFetchStatus]:
    return list(
        db.execute(
            select(FuturesContractFetchStatus)
            .where(FuturesContractFetchStatus.pool_id == pool_id)
            .order_by(FuturesContractFetchStatus.contract_code.asc())
        )
        .scalars()
        .all()
    )
