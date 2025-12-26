from __future__ import annotations

import datetime as dt
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from ..data.akshare_fetcher import FetchRequest, fetch_etf_daily_qfq
from ..db.models import EtfPool, IngestionBatch
from ..db.repo import (
    PriceRow,
    compute_price_fingerprint,
    create_ingestion_batch,
    get_etf_pool_by_code,
    get_validation_policy_by_id,
    list_prices,
    normalize_adjust,
    record_ingestion_items,
    record_price_audit,
    update_ingestion_batch,
    upsert_prices,
)
from ..settings import get_settings
from ..validation.price_validate import PricePoint, ValidationError, ValidationPolicyParams, validate_price_series


@dataclass(frozen=True)
class IngestResult:
    batch_id: int
    code: str
    upserted: int
    status: str
    message: str | None = None


def _parse_yyyymmdd(x: str) -> dt.date:
    return dt.datetime.strptime(x, "%Y%m%d").date()


def _policy_params_for_pool(db: Session, pool: EtfPool) -> ValidationPolicyParams:
    policy = get_validation_policy_by_id(db, pool.validation_policy_id) if pool.validation_policy_id else None
    if policy is None:
        # safe default
        return ValidationPolicyParams(max_abs_return=0.35, max_hl_spread=0.6, max_gap_days=15)
    max_abs_return = pool.max_abs_return_override if pool.max_abs_return_override is not None else policy.max_abs_return
    return ValidationPolicyParams(
        max_abs_return=max_abs_return,
        max_hl_spread=policy.max_hl_spread,
        max_gap_days=policy.max_gap_days,
    )


def make_sqlite_snapshot(sqlite_path: Path, *, batch_id: int) -> Path:
    backups_dir = sqlite_path.parent / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    snap = backups_dir / f"{sqlite_path.stem}_batch{batch_id}_{ts}{sqlite_path.suffix}"
    shutil.copy2(sqlite_path, snap)
    return snap


def _cleanup_snapshot_for_batch(db: Session, *, batch_id: int, snapshot_path: Path | None) -> None:
    """
    Best-effort cleanup: remove snapshot file and clear snapshot_path in batch row.

    Per product requirement, snapshots are ephemeral and should not accumulate.
    """
    if snapshot_path is None:
        return
    try:
        if snapshot_path.exists():
            snapshot_path.unlink()
    except Exception:  # pylint: disable=broad-exception-caught
        # If deletion fails, keep the file (best-effort).
        return
    try:
        with Session(bind=db.get_bind()) as s2:
            b2 = s2.get(IngestionBatch, batch_id)
            if b2 is not None:
                b2.snapshot_path = None
            s2.commit()
    except Exception:  # pylint: disable=broad-exception-caught
        # best-effort
        return


def ingest_one_etf(
    db: Session,
    *,
    ak,
    code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    adjust: str | None = None,
) -> IngestResult:
    """
    Batch ingestion with:
    - snapshot (file)
    - pre/post fingerprint
    - audit (old values)
    - per-date ingestion items (insert/update)
    - validation before commit (so data is unchanged on failure)
    """
    settings = get_settings()
    pool = get_etf_pool_by_code(db, code)
    if pool is None:
        raise ValueError(f"ETF {code} not found in pool")

    start = start_date or pool.start_date or settings.default_start_date
    end = end_date or pool.end_date or settings.default_end_date
    adj = normalize_adjust(adjust)
    policy_params = _policy_params_for_pool(db, pool)

    # Create a batch record first (committed) so failures are traceable.
    sqlite_path = settings.sqlite_path
    # ensure db file exists (sqlite creates on connect; but we snapshot only if exists)
    if not sqlite_path.exists():
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        # touch
        sqlite_path.touch(exist_ok=True)

    # Create batch record
    batch = create_ingestion_batch(
        db,
        code=code,
        start_date=start,
        end_date=end,
        adjust=adj,
        snapshot_path=None,
        val_max_abs_return=policy_params.max_abs_return,
        val_max_hl_spread=policy_params.max_hl_spread,
        val_max_gap_days=policy_params.max_gap_days,
    )
    db.commit()

    snapshot_path: Path | None = None
    if sqlite_path.exists() and os.path.getsize(sqlite_path) > 0:
        snapshot_path = make_sqlite_snapshot(sqlite_path, batch_id=batch.id)
        with Session(bind=db.get_bind()) as s2:  # separate session to update snapshot path
            b2 = s2.get(type(batch), batch.id)
            if b2 is not None:
                b2.snapshot_path = str(snapshot_path)
            s2.commit()

    # Ingest in one transaction: any failure -> rollback -> data unchanged
    try:
        start_d = _parse_yyyymmdd(start)
        end_d = _parse_yyyymmdd(end)

        pre_fp = compute_price_fingerprint(db, code=code, start_date=start_d, end_date=end_d, adjust=adj)

        # akshare uses "" for no-adjust; we store "none" in DB
        ak_adjust = "" if adj == "none" else adj
        rows = fetch_etf_daily_qfq(ak, FetchRequest(code=code, start_date=start, end_date=end, adjust=ak_adjust))
        rows = [
            PriceRow(
                code=r.code,
                trade_date=r.trade_date,
                open=r.open,
                high=r.high,
                low=r.low,
                close=r.close,
                volume=r.volume,
                amount=r.amount,
                source=r.source,
                adjust=adj,
            )
            for r in rows
        ]

        # fetch existing rows for touched dates to audit + determine insert/update
        existing_rows = list_prices(db, code=code, start_date=start_d, end_date=end_d, adjust=adj, limit=1000000)
        existing_by_date = {r.trade_date: r for r in existing_rows}
        touched = [(r.trade_date, "update" if r.trade_date in existing_by_date else "insert") for r in rows]

        # Build series to validate: we validate only fetched rows for now (MVP)
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
        validate_price_series(points, policy=policy_params)

        # Audit previous values for those we will update
        audit_rows = [existing_by_date[r.trade_date] for r in rows if r.trade_date in existing_by_date]
        record_price_audit(db, batch_id=batch.id, rows=audit_rows)

        # Record ingestion items
        record_ingestion_items(db, batch_id=batch.id, code=code, items=touched)

        # Upsert
        upsert_rows = [
            PriceRow(
                code=code,
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
        n = upsert_prices(db, upsert_rows)

        # Validate merged DB state (same transaction): if this fails, rollback keeps DB unchanged.
        merged_rows = list_prices(
            db,
            code=code,
            start_date=start_d - dt.timedelta(days=30),
            end_date=end_d,
            adjust=adj,
            limit=1000000,
        )
        merged_points = [
            PricePoint(
                trade_date=r.trade_date,
                open=r.open,
                high=r.high,
                low=r.low,
                close=r.close,
                volume=r.volume,
                amount=r.amount,
            )
            for r in merged_rows
        ]
        validate_price_series(merged_points, policy=policy_params)

        post_fp = compute_price_fingerprint(db, code=code, start_date=start_d, end_date=end_d, adjust=adj)
        update_ingestion_batch(db, batch_id=batch.id, status="success", message=f"rows={len(rows)} upserted={n}", pre_fingerprint=pre_fp, post_fingerprint=post_fp)
        db.commit()
        _cleanup_snapshot_for_batch(db, batch_id=batch.id, snapshot_path=snapshot_path)
        return IngestResult(batch_id=batch.id, code=code, upserted=n, status="success")
    except Exception as e:  # pylint: disable=broad-exception-caught
        db.rollback()
        # Update batch as failed in a new transaction
        msg = e.to_json() if isinstance(e, ValidationError) else str(e)
        update_ingestion_batch(db, batch_id=batch.id, status="failed", message=msg)
        db.commit()
        _cleanup_snapshot_for_batch(db, batch_id=batch.id, snapshot_path=snapshot_path)
        return IngestResult(batch_id=batch.id, code=code, upserted=0, status="failed", message=msg)

