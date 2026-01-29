from __future__ import annotations

import pandas as pd
from sqlalchemy import delete
from sqlalchemy.orm import sessionmaker

from etf_momentum.data.ingestion import ingest_one_etf  # pylint: disable=import-error
from etf_momentum.data.rollback import rollback_batch_with_fallback  # pylint: disable=import-error
from etf_momentum.db.models import EtfPriceAudit  # pylint: disable=import-error
from etf_momentum.db.repo import list_prices, upsert_etf_pool  # pylint: disable=import-error


class FakeAk:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def fund_etf_hist_em(self, **kwargs):  # pylint: disable=unused-argument
        return self.df


def test_logical_rollback_restores_updated_and_deletes_inserted(session_factory: sessionmaker) -> None:
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


def test_rollback_fails_if_audit_missing(session_factory: sessionmaker) -> None:
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
        assert rb.status == "failed"

    with session_factory() as db:
        prices = list_prices(db, code="510300")
        assert prices[0].close == 1.01


