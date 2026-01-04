from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from .models import EtfPool, EtfPrice, EtfPriceAudit, IngestionBatch, IngestionItem, ValidationPolicy


@dataclass(frozen=True)
class PriceRow:
    code: str
    trade_date: dt.date
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None
    amount: float | None
    source: str = "eastmoney"
    adjust: str = "qfq"  # qfq/hfq/none


def normalize_adjust(adjust: str | None) -> str:
    """
    Normalize adjustment type:
    - qfq: 前复权
    - hfq: 后复权
    - none: 不复权
    """
    a = (adjust or "hfq").strip().lower()
    if a in {"", "nfq", "raw"}:
        a = "none"
    if a not in {"qfq", "hfq", "none"}:
        raise ValueError(f"invalid adjust={adjust}")
    return a


def upsert_etf_pool(
    db: Session,
    *,
    code: str,
    name: str,
    start_date: str | None,
    end_date: str | None,
    validation_policy_id: int | None = None,
    max_abs_return_override: float | None = None,
) -> EtfPool:
    existing = db.execute(select(EtfPool).where(EtfPool.code == code)).scalar_one_or_none()
    if existing is None:
        obj = EtfPool(
            code=code,
            name=name,
            start_date=start_date,
            end_date=end_date,
            validation_policy_id=validation_policy_id,
            max_abs_return_override=max_abs_return_override,
        )
        db.add(obj)
        db.flush()
        return obj
    existing.name = name
    existing.start_date = start_date
    existing.end_date = end_date
    if validation_policy_id is not None:
        existing.validation_policy_id = validation_policy_id
    existing.max_abs_return_override = max_abs_return_override
    db.flush()
    return existing


def list_etf_pool(db: Session) -> list[EtfPool]:
    return list(db.execute(select(EtfPool).order_by(EtfPool.code.asc())).scalars().all())


def get_etf_pool_by_code(db: Session, code: str) -> EtfPool | None:
    return db.execute(select(EtfPool).where(EtfPool.code == code)).scalar_one_or_none()


def update_etf_pool_data_range(db: Session, *, code: str) -> tuple[str | None, str | None]:
    """
    Update etf_pool.last_data_start_date/last_data_end_date from etf_prices.
    Returns (start_yyyymmdd, end_yyyymmdd).
    """
    obj = get_etf_pool_by_code(db, code)
    if obj is None:
        return (None, None)

    start_d, end_d = db.execute(
        select(func.min(EtfPrice.trade_date), func.max(EtfPrice.trade_date)).where(EtfPrice.code == code)
    ).one()
    if start_d is None or end_d is None:
        obj.last_data_start_date = None
        obj.last_data_end_date = None
        db.flush()
        return (None, None)

    obj.last_data_start_date = start_d.strftime("%Y%m%d")
    obj.last_data_end_date = end_d.strftime("%Y%m%d")
    db.flush()
    return (obj.last_data_start_date, obj.last_data_end_date)


def get_price_date_range(db: Session, *, code: str, adjust: str) -> tuple[str | None, str | None]:
    adj = normalize_adjust(adjust)
    start_d, end_d = db.execute(
        select(func.min(EtfPrice.trade_date), func.max(EtfPrice.trade_date)).where(EtfPrice.code == code, EtfPrice.adjust == adj)
    ).one()
    if start_d is None or end_d is None:
        return (None, None)
    return (start_d.strftime("%Y%m%d"), end_d.strftime("%Y%m%d"))


def delete_etf_pool(db: Session, code: str) -> bool:
    obj = get_etf_pool_by_code(db, code)
    if obj is None:
        return False
    db.delete(obj)
    db.flush()
    return True


def purge_etf_data(db: Session, *, code: str) -> dict[str, int]:
    """
    Permanently delete all persisted data for an ETF code:
    - all price rows (hfq/qfq/none)
    - ingestion batches + items + audits
    - snapshot files referenced by ingestion batches (best-effort)

    Returns deletion counts.
    """
    batches = list(db.execute(select(IngestionBatch).where(IngestionBatch.code == code)).scalars().all())
    snapshot_paths = [b.snapshot_path for b in batches if b.snapshot_path]

    # Delete child tables first (safe even if FK constraints are enabled).
    r_items = db.execute(delete(IngestionItem).where(IngestionItem.code == code))
    r_audits = db.execute(delete(EtfPriceAudit).where(EtfPriceAudit.code == code))
    r_batches = db.execute(delete(IngestionBatch).where(IngestionBatch.code == code))

    n_prices = delete_prices(db, code=code)  # all adjusts

    # Best-effort: delete snapshot files that were kept (normally snapshots are ephemeral and removed after ingestion/rollback).
    n_snaps = 0
    for p in snapshot_paths:
        try:
            fp = Path(str(p))
            if fp.exists():
                fp.unlink()
                n_snaps += 1
        except Exception:  # pylint: disable=broad-exception-caught
            continue

    return {
        "prices": int(n_prices),
        "batches": int(getattr(r_batches, "rowcount", 0) or 0),
        "items": int(getattr(r_items, "rowcount", 0) or 0),
        "audits": int(getattr(r_audits, "rowcount", 0) or 0),
        "snapshots": int(n_snaps),
    }


def list_validation_policies(db: Session) -> list[ValidationPolicy]:
    return list(db.execute(select(ValidationPolicy).order_by(ValidationPolicy.name.asc())).scalars().all())


def get_validation_policy_by_name(db: Session, name: str) -> ValidationPolicy | None:
    return db.execute(select(ValidationPolicy).where(ValidationPolicy.name == name)).scalar_one_or_none()


def get_validation_policy_by_id(db: Session, policy_id: int) -> ValidationPolicy | None:
    return db.execute(select(ValidationPolicy).where(ValidationPolicy.id == policy_id)).scalar_one_or_none()


def mark_fetch_status(
    db: Session,
    *,
    code: str,
    status: str,
    message: str | None = None,
    when: dt.datetime | None = None,
) -> None:
    obj = get_etf_pool_by_code(db, code)
    if obj is None:
        return
    obj.last_fetch_at = when or dt.datetime.now(dt.timezone.utc)
    obj.last_fetch_status = status
    obj.last_fetch_message = message
    db.flush()


def upsert_prices(db: Session, rows: list[PriceRow]) -> int:
    if not rows:
        return 0

    values = [
        dict(
            code=r.code,
            trade_date=r.trade_date,
            open=r.open,
            high=r.high,
            low=r.low,
            close=r.close,
            volume=r.volume,
            amount=r.amount,
            source=r.source,
            adjust=r.adjust,
        )
        for r in rows
    ]

    # SQLite has a hard limit on the number of bound variables per statement.
    # Large ETFs (e.g. 510300) can easily exceed it if we insert everything in one VALUES(...),(...) statement.
    #
    # EtfPrice has 10 bound params per row here (code, trade_date, ohlc, volume, amount, source, adjust),
    # plus a small constant overhead from SQLAlchemy, so a conservative chunk size keeps us safe across platforms.
    chunk_size = 1000

    total = 0
    for i in range(0, len(values), chunk_size):
        chunk = values[i : i + chunk_size]
        stmt = insert(EtfPrice).values(chunk)
        # On conflict, update OHLCV/amount/source/adjust to latest ingested values.
        stmt = stmt.on_conflict_do_update(
            index_elements=[EtfPrice.code, EtfPrice.trade_date, EtfPrice.adjust],
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
        result = db.execute(stmt)
        total += int(getattr(result, "rowcount", 0) or 0)

    # sqlite returns rowcount for DML
    return total


def list_prices(
    db: Session,
    *,
    code: str,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    adjust: str | None = None,
    limit: int = 5000,
) -> list[EtfPrice]:
    stmt = select(EtfPrice).where(EtfPrice.code == code)
    if adjust is not None:
        stmt = stmt.where(EtfPrice.adjust == normalize_adjust(adjust))
    if start_date is not None:
        stmt = stmt.where(EtfPrice.trade_date >= start_date)
    if end_date is not None:
        stmt = stmt.where(EtfPrice.trade_date <= end_date)
    stmt = stmt.order_by(EtfPrice.trade_date.asc()).limit(limit)
    return list(db.execute(stmt).scalars().all())


def delete_prices(
    db: Session,
    *,
    code: str,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    adjust: str | None = None,
) -> int:
    stmt = delete(EtfPrice).where(EtfPrice.code == code)
    if adjust is not None:
        stmt = stmt.where(EtfPrice.adjust == normalize_adjust(adjust))
    if start_date is not None:
        stmt = stmt.where(EtfPrice.trade_date >= start_date)
    if end_date is not None:
        stmt = stmt.where(EtfPrice.trade_date <= end_date)
    result = db.execute(stmt)
    return int(getattr(result, "rowcount", 0) or 0)


def create_ingestion_batch(
    db: Session,
    *,
    code: str,
    start_date: str,
    end_date: str,
    source: str = "eastmoney",
    adjust: str = "qfq",
    snapshot_path: str | None = None,
    val_max_abs_return: float | None = None,
    val_max_hl_spread: float | None = None,
    val_max_gap_days: int | None = None,
) -> IngestionBatch:
    b = IngestionBatch(
        code=code,
        start_date=start_date,
        end_date=end_date,
        source=source,
        adjust=adjust,
        status="running",
        snapshot_path=snapshot_path,
        val_max_abs_return=val_max_abs_return,
        val_max_hl_spread=val_max_hl_spread,
        val_max_gap_days=val_max_gap_days,
    )
    db.add(b)
    db.flush()
    return b


def update_ingestion_batch(
    db: Session,
    *,
    batch_id: int,
    status: str,
    message: str | None = None,
    pre_fingerprint: str | None = None,
    post_fingerprint: str | None = None,
) -> None:
    stmt = (
        update(IngestionBatch)
        .where(IngestionBatch.id == batch_id)
        .values(status=status, message=message, pre_fingerprint=pre_fingerprint, post_fingerprint=post_fingerprint)
    )
    db.execute(stmt)


def get_ingestion_batch(db: Session, batch_id: int) -> IngestionBatch | None:
    return db.execute(select(IngestionBatch).where(IngestionBatch.id == batch_id)).scalar_one_or_none()


def list_ingestion_batches(db: Session, *, code: str | None = None, limit: int = 50) -> list[IngestionBatch]:
    stmt = select(IngestionBatch).order_by(IngestionBatch.id.desc()).limit(limit)
    if code:
        stmt = stmt.where(IngestionBatch.code == code)
    return list(db.execute(stmt).scalars().all())


def _price_fingerprint_rows(rows: list[EtfPrice]) -> str:
    """
    Stable fingerprint for a set of price rows.
    """
    payload = [
        dict(
            trade_date=r.trade_date.isoformat(),
            open=r.open,
            high=r.high,
            low=r.low,
            close=r.close,
            volume=r.volume,
            amount=r.amount,
            source=r.source,
            adjust=r.adjust,
        )
        for r in sorted(rows, key=lambda x: x.trade_date)
    ]
    s = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def compute_price_fingerprint(
    db: Session,
    *,
    code: str,
    start_date: dt.date,
    end_date: dt.date,
    adjust: str,
) -> str:
    rows = list_prices(db, code=code, start_date=start_date, end_date=end_date, adjust=adjust, limit=1000000)
    return _price_fingerprint_rows(rows)


def record_ingestion_items(
    db: Session,
    *,
    batch_id: int,
    code: str,
    items: list[tuple[dt.date, str]],
) -> None:
    for trade_date, action in items:
        db.add(IngestionItem(batch_id=batch_id, code=code, trade_date=trade_date, action=action))
    db.flush()


def record_price_audit(
    db: Session,
    *,
    batch_id: int,
    rows: list[EtfPrice],
) -> None:
    for r in rows:
        db.add(
            EtfPriceAudit(
                batch_id=batch_id,
                code=r.code,
                trade_date=r.trade_date,
                open=r.open,
                high=r.high,
                low=r.low,
                close=r.close,
                volume=r.volume,
                amount=r.amount,
                source=r.source,
                adjust=r.adjust,
            )
        )
    db.flush()

