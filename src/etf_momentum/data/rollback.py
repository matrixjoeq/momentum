from __future__ import annotations

import datetime as dt
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..db.models import EtfPrice, EtfPriceAudit, IngestionBatch, IngestionItem
from ..db.repo import compute_price_fingerprint, get_ingestion_batch, list_prices, normalize_adjust, update_ingestion_batch
from ..db.session import make_engine, make_session_factory
from ..settings import get_settings
from ..validation.price_validate import PricePoint, ValidationError, ValidationPolicyParams, validate_price_series


@dataclass(frozen=True)
class RollbackResult:
    batch_id: int
    status: str  # "success" | "failed" | "snapshot_restored"
    message: str | None = None


def _parse_yyyymmdd(x: str) -> dt.date:
    return dt.datetime.strptime(x, "%Y%m%d").date()


def logical_rollback_batch(db: Session, batch: IngestionBatch) -> None:
    """
    Roll back price changes made by this batch using ingestion_item + audit.
    """
    # delete inserted rows
    ins_dates = list(
        db.execute(
            select(IngestionItem.trade_date).where(IngestionItem.batch_id == batch.id, IngestionItem.action == "insert")
        ).all()
    )
    if ins_dates:
        db.execute(
            delete(EtfPrice).where(
                EtfPrice.code == batch.code,
                EtfPrice.adjust == normalize_adjust(batch.adjust),
                EtfPrice.trade_date.in_([d[0] for d in ins_dates]),
            )
        )

    # restore updated rows from audit
    audits = list(db.execute(select(EtfPriceAudit).where(EtfPriceAudit.batch_id == batch.id)).scalars().all())
    for a in audits:
        row = db.execute(
            select(EtfPrice).where(EtfPrice.code == a.code, EtfPrice.trade_date == a.trade_date, EtfPrice.adjust == a.adjust)
        ).scalar_one_or_none()
        if row is None:
            # should not happen: update implies existed; but keep safe by inserting
            row = EtfPrice(code=a.code, trade_date=a.trade_date, adjust=a.adjust, source=a.source)
            db.add(row)
        row.open = a.open
        row.high = a.high
        row.low = a.low
        row.close = a.close
        row.volume = a.volume
        row.amount = a.amount
        row.source = a.source
        row.adjust = a.adjust
    db.flush()


def restore_snapshot(sqlite_path: Path, snapshot_path: Path) -> None:
    """
    Restore sqlite database file from snapshot.
    Caller must ensure no active connections.
    """
    shutil.copy2(snapshot_path, sqlite_path)


def _validate_db_prices(
    db: Session,
    *,
    code: str,
    start_date: dt.date,
    end_date: dt.date,
    adjust: str,
    policy: ValidationPolicyParams,
    lookback_days: int = 30,
    allow_empty: bool = False,
) -> None:
    rows = list_prices(
        db,
        code=code,
        start_date=start_date - dt.timedelta(days=lookback_days),
        end_date=end_date,
        adjust=adjust,
        limit=1000000,
    )
    if not rows and allow_empty:
        return
    points = [
        PricePoint(
            trade_date=r.trade_date,
            open=r.open,
            high=r.high,
            low=r.low,
            close=r.close,
            volume=r.volume,
            amount=r.amount,
        )
        for r in rows
    ]
    validate_price_series(points, policy=policy)


def rollback_batch_with_fallback(
    db: Session,
    *,
    batch_id: int,
    validate_after: bool = True,
) -> RollbackResult:
    """
    Attempt logical rollback; if post-rollback fingerprint mismatch and snapshot exists,
    restore snapshot file (best-effort).
    """
    settings = get_settings()
    batch = get_ingestion_batch(db, batch_id)
    if batch is None:
        return RollbackResult(batch_id=batch_id, status="failed", message="batch not found")

    start_d = _parse_yyyymmdd(batch.start_date)
    end_d = _parse_yyyymmdd(batch.end_date)
    adj = normalize_adjust(batch.adjust)

    pre_fp = batch.pre_fingerprint
    empty_fp = hashlib.sha256(b"[]").hexdigest()
    allow_empty = pre_fp == empty_fp if pre_fp is not None else False
    if batch.val_max_abs_return is not None and batch.val_max_hl_spread is not None and batch.val_max_gap_days is not None:
        policy = ValidationPolicyParams(
            max_abs_return=float(batch.val_max_abs_return),
            max_hl_spread=float(batch.val_max_hl_spread),
            max_gap_days=int(batch.val_max_gap_days),
        )
    else:
        policy = ValidationPolicyParams(max_abs_return=0.35, max_hl_spread=0.6, max_gap_days=15)
    try:
        logical_rollback_batch(db, batch)
        update_ingestion_batch(db, batch_id=batch.id, status="rolled_back", message="logical rollback")
        db.commit()
    except Exception as e:  # pylint: disable=broad-exception-caught
        db.rollback()
        update_ingestion_batch(db, batch_id=batch.id, status="rollback_failed", message=str(e))
        db.commit()
        return RollbackResult(batch_id=batch.id, status="failed", message=str(e))

    # Validate after logical rollback (full validator). If fingerprint is available, also require match.
    needs_snapshot = False
    cur_fp = None
    validation_exc: ValidationError | None = None
    if validate_after:
        try:
            _validate_db_prices(db, code=batch.code, start_date=start_d, end_date=end_d, policy=policy, allow_empty=allow_empty, adjust=adj)
        except ValidationError as e:
            validation_exc = e
            needs_snapshot = True

    if pre_fp is not None:
        cur_fp = compute_price_fingerprint(db, code=batch.code, start_date=start_d, end_date=end_d, adjust=adj)
        if cur_fp != pre_fp:
            needs_snapshot = True

    if not needs_snapshot:
        return RollbackResult(batch_id=batch.id, status="success", message="rolled back (validated)")

    # Fallback to snapshot restore if available
    if batch.snapshot_path:
        snap = Path(batch.snapshot_path)
        sqlite_path = settings.sqlite_path
        # Close current engine connections by disposing if running under app; here we only best-effort
        try:
            bind = db.get_bind()
            if bind is not None:
                bind.dispose()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        try:
            restore_snapshot(sqlite_path, snap)
        except Exception as e:  # pylint: disable=broad-exception-caught
            return RollbackResult(batch_id=batch.id, status="failed", message=f"snapshot restore failed: {e}")

        # Re-validate using a fresh engine/session after restoring the file.
        engine = make_engine(str(sqlite_path))
        sf = make_session_factory(engine)
        with sf() as db2:
            if validate_after:
                _validate_db_prices(db2, code=batch.code, start_date=start_d, end_date=end_d, policy=policy, allow_empty=allow_empty, adjust=adj)
            if pre_fp is not None:
                fp2 = compute_price_fingerprint(db2, code=batch.code, start_date=start_d, end_date=end_d, adjust=adj)
                if fp2 != pre_fp:
                    return RollbackResult(batch_id=batch.id, status="failed", message="snapshot restored but fingerprint mismatch")
            update_ingestion_batch(db2, batch_id=batch.id, status="snapshot_restored", message="snapshot restored after rollback validation/fingerprint mismatch")
            db2.commit()
        return RollbackResult(batch_id=batch.id, status="snapshot_restored", message="snapshot restored")

    if validation_exc is not None:
        return RollbackResult(batch_id=batch.id, status="failed", message=validation_exc.to_json())
    return RollbackResult(batch_id=batch.id, status="failed", message="rollback validation/fingerprint mismatch and no snapshot")

