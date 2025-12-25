from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import sessionmaker

from etf_momentum.db.repo import (
    PriceRow,
    delete_etf_pool,
    delete_prices,
    list_etf_pool,
    list_prices,
    upsert_etf_pool,
    upsert_prices,
)


def test_etf_pool_crud(session_factory: sessionmaker) -> None:
    with session_factory() as db:
        obj = upsert_etf_pool(db, code="510300", name="沪深300ETF", start_date="20200101", end_date="20201231")
        db.commit()
        assert obj.id > 0

    with session_factory() as db:
        items = list_etf_pool(db)
        assert [i.code for i in items] == ["510300"]

        # update
        obj2 = upsert_etf_pool(db, code="510300", name="沪深300ETF(更新)", start_date=None, end_date=None)
        db.commit()
        assert obj2.name == "沪深300ETF(更新)"

    with session_factory() as db:
        assert delete_etf_pool(db, "510300") is True
        db.commit()
    with session_factory() as db:
        assert list_etf_pool(db) == []


def test_upsert_prices_idempotent(session_factory: sessionmaker) -> None:
    d = dt.date(2024, 1, 2)
    rows = [
        PriceRow(
            code="510300",
            trade_date=d,
            open=1.0,
            high=2.0,
            low=0.5,
            close=1.5,
            volume=100.0,
            amount=200.0,
        )
    ]
    with session_factory() as db:
        n1 = upsert_prices(db, rows)
        db.commit()
        assert n1 >= 1

    with session_factory() as db:
        n2 = upsert_prices(db, rows)
        db.commit()
        assert n2 >= 1


def test_list_and_delete_prices(session_factory: sessionmaker) -> None:
    rows = [
        PriceRow(
            code="510300",
            trade_date=dt.date(2024, 1, 2),
            open=1.0,
            high=1.2,
            low=0.8,
            close=1.1,
            volume=10.0,
            amount=100.0,
        ),
        PriceRow(
            code="510300",
            trade_date=dt.date(2024, 1, 3),
            open=2.0,
            high=2.2,
            low=1.8,
            close=2.1,
            volume=20.0,
            amount=200.0,
        ),
    ]
    with session_factory() as db:
        upsert_prices(db, rows)
        db.commit()

    with session_factory() as db:
        got = list_prices(db, code="510300", start_date=dt.date(2024, 1, 3))
        assert [x.trade_date.isoformat() for x in got] == ["2024-01-03"]

    with session_factory() as db:
        deleted = delete_prices(db, code="510300", end_date=dt.date(2024, 1, 2))
        db.commit()
        assert deleted == 1

    with session_factory() as db:
        got2 = list_prices(db, code="510300")
        assert [x.trade_date.isoformat() for x in got2] == ["2024-01-03"]

