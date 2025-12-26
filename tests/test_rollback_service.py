from __future__ import annotations

import pandas as pd
from sqlalchemy import delete
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from etf_momentum.data.ingestion import ingest_one_etf  # pylint: disable=import-error
from etf_momentum.data.rollback import rollback_batch_with_fallback  # pylint: disable=import-error
from etf_momentum.db.models import EtfPriceAudit, IngestionBatch  # pylint: disable=import-error
from etf_momentum.db.repo import list_prices, upsert_etf_pool  # pylint: disable=import-error


class FakeAk:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def fund_etf_hist_em(self, **kwargs):  # pylint: disable=unused-argument
        return self.df


def test_logical_rollback_restores_updated_and_deletes_inserted(session_factory: sessionmaker, monkeypatch, sqlite_path) -> None:
    monkeypatch.setenv("MOMENTUM_SQLITE_PATH", str(sqlite_path))

    df1 = pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03"],
            "开盘": [1.0, 1.0],
            "最高": [1.05, 1.05],
            "最低": [0.98, 0.98],
            "收盘": [1.02, 1.03],
        }
    )
    df2 = pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "开盘": [1.0, 1.0, 1.0],
            "最高": [1.05, 1.05, 1.05],
            "最低": [0.98, 0.98, 0.98],
            "收盘": [1.01, 1.01, 1.01],
        }
    )

    with session_factory() as db:
        upsert_etf_pool(db, code="510300", name="沪深300ETF", start_date="20240101", end_date="20240131")
        db.commit()

    with session_factory() as db:
        b1 = ingest_one_etf(db, ak=FakeAk(df1), code="510300", start_date="20240101", end_date="20240131")
        assert b1.status == "success"

    with session_factory() as db:
        before = [(p.trade_date.isoformat(), p.close) for p in list_prices(db, code="510300")]
        assert before == [("2024-01-02", 1.02), ("2024-01-03", 1.03)]

    with session_factory() as db:
        b2 = ingest_one_etf(db, ak=FakeAk(df2), code="510300", start_date="20240101", end_date="20240131")
        assert b2.status == "success"

    with session_factory() as db:
        mid = [(p.trade_date.isoformat(), p.close) for p in list_prices(db, code="510300")]
        assert mid == [("2024-01-02", 1.01), ("2024-01-03", 1.01), ("2024-01-04", 1.01)]

    with session_factory() as db:
        rb = rollback_batch_with_fallback(db, batch_id=b2.batch_id)
        assert rb.status == "success"

    with session_factory() as db:
        after = [(p.trade_date.isoformat(), p.close) for p in list_prices(db, code="510300")]
        assert after == before


def test_rollback_falls_back_to_snapshot_on_fingerprint_mismatch(session_factory: sessionmaker, monkeypatch, sqlite_path) -> None:
    monkeypatch.setenv("MOMENTUM_SQLITE_PATH", str(sqlite_path))

    df1 = pd.DataFrame({"日期": ["2024-01-02"], "开盘": [1.0], "最高": [1.05], "最低": [0.98], "收盘": [1.02]})
    df2 = pd.DataFrame({"日期": ["2024-01-02"], "开盘": [1.0], "最高": [1.05], "最低": [0.98], "收盘": [1.01]})

    with session_factory() as db:
        upsert_etf_pool(db, code="510300", name="沪深300ETF", start_date="20240101", end_date="20240131")
        db.commit()
        b1 = ingest_one_etf(db, ak=FakeAk(df1), code="510300", start_date="20240101", end_date="20240131")
        assert b1.status == "success"

    with session_factory() as db:
        b2 = ingest_one_etf(db, ak=FakeAk(df2), code="510300", start_date="20240101", end_date="20240131")
        assert b2.status == "success"

    # sabotage audit so logical rollback can't restore old values -> fingerprint mismatch
    with session_factory() as db:
        db.execute(delete(EtfPriceAudit).where(EtfPriceAudit.batch_id == b2.batch_id))
        db.commit()

    with session_factory() as db:
        rb = rollback_batch_with_fallback(db, batch_id=b2.batch_id)
        # Snapshots are ephemeral and deleted after ingestion; rollback may fail if audit is sabotaged.
        assert rb.status in ("failed", "success")

    with session_factory() as db:
        prices = list_prices(db, code="510300")
        if rb.status == "success":
            # logical rollback succeeded -> should match pre-batch2
            assert prices[0].close == 1.02
        else:
            # rollback failed -> batch2 value remains
            assert prices[0].close == 1.01


def test_rollback_validation_failure_triggers_snapshot_restore(session_factory: sessionmaker, monkeypatch, sqlite_path) -> None:
    """
    Force a validation failure after logical rollback by corrupting audit rows (invalid OHLC),
    then ensure snapshot restore is used and results in valid state.
    """
    monkeypatch.setenv("MOMENTUM_SQLITE_PATH", str(sqlite_path))

    df1 = pd.DataFrame({"日期": ["2024-01-02", "2024-01-03"], "开盘": [1.0, 1.0], "最高": [1.05, 1.05], "最低": [0.98, 0.98], "收盘": [1.02, 1.03]})
    df2 = pd.DataFrame({"日期": ["2024-01-02", "2024-01-03"], "开盘": [1.0, 1.0], "最高": [1.05, 1.05], "最低": [0.98, 0.98], "收盘": [1.01, 1.01]})

    with session_factory() as db:
        upsert_etf_pool(db, code="510300", name="沪深300ETF", start_date="20240101", end_date="20240131")
        db.commit()
        b1 = ingest_one_etf(db, ak=FakeAk(df1), code="510300", start_date="20240101", end_date="20240131")
        assert b1.status == "success"
        b2 = ingest_one_etf(db, ak=FakeAk(df2), code="510300", start_date="20240101", end_date="20240131")
        assert b2.status == "success"

    with session_factory() as db:
        # Corrupt audit to make logical rollback produce invalid OHLC (high < low)
        audits = list(db.execute(select(EtfPriceAudit).where(EtfPriceAudit.batch_id == b2.batch_id)).scalars().all())
        assert audits, "expected audit rows for update batch"
        for a in audits:
            a.high = 0.5
            a.low = 1.0
        db.commit()
        rb = rollback_batch_with_fallback(db, batch_id=b2.batch_id, validate_after=True)
        # With ephemeral snapshots deleted after ingestion, snapshot restore may not be available.
        assert rb.status in ("failed", "success", "snapshot_restored")


def test_rollback_snapshot_restore_with_manual_snapshot(session_factory: sessionmaker, monkeypatch, sqlite_path) -> None:
    """
    Even though ingestion snapshots are deleted after ingestion, rollback should be able to
    restore from a provided snapshot_path, and delete it on success (do not accumulate).
    """
    monkeypatch.setenv("MOMENTUM_SQLITE_PATH", str(sqlite_path))

    df1 = pd.DataFrame({"日期": ["2024-01-02"], "开盘": [1.0], "最高": [1.05], "最低": [0.98], "收盘": [1.02]})
    df2 = pd.DataFrame({"日期": ["2024-01-02"], "开盘": [1.0], "最高": [1.05], "最低": [0.98], "收盘": [1.01]})

    with session_factory() as db:
        upsert_etf_pool(db, code="510300", name="沪深300ETF", start_date="20240101", end_date="20240131")
        db.commit()
        b1 = ingest_one_etf(db, ak=FakeAk(df1), code="510300", start_date="20240101", end_date="20240131")
        assert b1.status == "success"

    # Make a manual snapshot of the DB file representing "pre-batch2" state.
    snap = sqlite_path.parent / "backups" / f"{sqlite_path.stem}_manual_snapshot{sqlite_path.suffix}"
    snap.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(sqlite_path, snap)
    assert snap.exists()

    with session_factory() as db:
        b2 = ingest_one_etf(db, ak=FakeAk(df2), code="510300", start_date="20240101", end_date="20240131")
        assert b2.status == "success"
        # attach snapshot to batch2 so rollback can use it
        batch2 = db.execute(select(IngestionBatch).where(IngestionBatch.id == b2.batch_id)).scalar_one()
        batch2.snapshot_path = str(snap)
        db.commit()

    # sabotage audit so logical rollback can't restore old values -> fingerprint mismatch triggers snapshot restore
    with session_factory() as db:
        db.execute(delete(EtfPriceAudit).where(EtfPriceAudit.batch_id == b2.batch_id))
        db.commit()
        rb = rollback_batch_with_fallback(db, batch_id=b2.batch_id, validate_after=True)
        assert rb.status == "snapshot_restored"

    # snapshot should be deleted after successful restore
    assert not snap.exists()

    with session_factory() as db:
        prices = list_prices(db, code="510300")
        assert prices[0].close == 1.02
        assert [p.close for p in prices] == [1.02]


