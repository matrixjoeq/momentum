from __future__ import annotations

import pathlib
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from etf_momentum.data.ingestion import ingest_one_etf  # pylint: disable=import-error
from etf_momentum.db.models import EtfPriceAudit, IngestionBatch, IngestionItem  # pylint: disable=import-error
from etf_momentum.db.repo import get_ingestion_batch, list_prices, upsert_etf_pool  # pylint: disable=import-error


class FakeAk:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def fund_etf_hist_em(self, **kwargs):  # pylint: disable=unused-argument
        return self.df


def test_ingest_success_creates_batch_items_and_prices(session_factory: sessionmaker, monkeypatch, sqlite_path) -> None:
    monkeypatch.setenv("MOMENTUM_SQLITE_PATH", str(sqlite_path))

    df = pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03"],
            "开盘": [1.0, 1.02],
            "最高": [1.05, 1.06],
            "最低": [0.98, 1.00],
            "收盘": [1.02, 1.03],
            "成交量": [10.0, 20.0],
            "成交额": [100.0, 200.0],
        }
    )

    with session_factory() as db:
        upsert_etf_pool(db, code="510300", name="沪深300ETF", start_date="20240101", end_date="20240131")
        db.commit()

    with session_factory() as db:
        res = ingest_one_etf(db, ak=FakeAk(df), code="510300", start_date="20240101", end_date="20240131")
        assert res.status == "success"
        assert res.upserted >= 2

    with session_factory() as db:
        prices = list_prices(db, code="510300")
        assert [p.trade_date.isoformat() for p in prices] == ["2024-01-02", "2024-01-03"]

        batch = get_ingestion_batch(db, res.batch_id)
        assert batch is not None
        assert batch.status == "success"
        assert batch.pre_fingerprint is not None
        assert batch.post_fingerprint is not None
        # snapshot is ephemeral: should be deleted after ingestion completes
        assert batch.snapshot_path is None

        items = list(db.execute(select(IngestionItem).where(IngestionItem.batch_id == res.batch_id)).scalars().all())
        assert len(items) == 2
        assert set(i.action for i in items) == {"insert"}

        audits = list(db.execute(select(EtfPriceAudit).where(EtfPriceAudit.batch_id == res.batch_id)).scalars().all())
        assert audits == []

    backups_dir = pathlib.Path(sqlite_path).parent / "backups"
    if backups_dir.exists():
        assert list(backups_dir.glob("*.sqlite3")) == []


def test_ingest_update_records_audit(session_factory: sessionmaker, monkeypatch, sqlite_path) -> None:
    monkeypatch.setenv("MOMENTUM_SQLITE_PATH", str(sqlite_path))

    df1 = pd.DataFrame(
        {"日期": ["2024-01-02"], "开盘": [1.0], "最高": [1.05], "最低": [0.98], "收盘": [1.02]}
    )
    df2 = pd.DataFrame(
        {"日期": ["2024-01-02"], "开盘": [1.01], "最高": [1.06], "最低": [0.99], "收盘": [1.03]}
    )

    with session_factory() as db:
        upsert_etf_pool(db, code="510300", name="沪深300ETF", start_date="20240101", end_date="20240131")
        db.commit()

    with session_factory() as db:
        res1 = ingest_one_etf(db, ak=FakeAk(df1), code="510300", start_date="20240101", end_date="20240131")
        assert res1.status == "success"

    with session_factory() as db:
        res2 = ingest_one_etf(db, ak=FakeAk(df2), code="510300", start_date="20240101", end_date="20240131")
        assert res2.status == "success"

    with session_factory() as db:
        audits = list(db.execute(select(EtfPriceAudit).where(EtfPriceAudit.batch_id == res2.batch_id)).scalars().all())
        assert len(audits) == 1
        assert audits[0].close == 1.02

        items = list(db.execute(select(IngestionItem).where(IngestionItem.batch_id == res2.batch_id)).scalars().all())
        assert len(items) == 1
        assert items[0].action == "update"


def test_ingest_validation_failure_keeps_data_unchanged(session_factory: sessionmaker, monkeypatch, sqlite_path) -> None:
    monkeypatch.setenv("MOMENTUM_SQLITE_PATH", str(sqlite_path))

    good = pd.DataFrame(
        {"日期": ["2024-01-02", "2024-01-03"], "开盘": [1.0, 1.0], "最高": [1.1, 1.1], "最低": [0.9, 0.9], "收盘": [1.0, 1.0]}
    )
    bad = pd.DataFrame(
        {"日期": ["2024-01-02", "2024-01-03"], "开盘": [1.0, 1.0], "最高": [1.1, 2.0], "最低": [0.9, 0.9], "收盘": [1.0, 2.0]}
    )

    with session_factory() as db:
        upsert_etf_pool(db, code="510300", name="沪深300ETF", start_date="20240101", end_date="20240131")
        db.commit()

    with session_factory() as db:
        res1 = ingest_one_etf(db, ak=FakeAk(good), code="510300", start_date="20240101", end_date="20240131")
        assert res1.status == "success"

    with session_factory() as db:
        before = [(p.trade_date, p.close) for p in list_prices(db, code="510300")]

    # bad ingest should fail and not change prices
    with session_factory() as db:
        res2 = ingest_one_etf(db, ak=FakeAk(bad), code="510300", start_date="20240101", end_date="20240131")
        assert res2.status == "failed"
        batch = db.execute(select(IngestionBatch).where(IngestionBatch.id == res2.batch_id)).scalar_one()
        assert batch.status == "failed"
        assert batch.message is not None
        assert batch.message.startswith("{")
        assert "\"error_type\":\"validation\"" in batch.message

    with session_factory() as db:
        after = [(p.trade_date, p.close) for p in list_prices(db, code="510300")]
        assert after == before

