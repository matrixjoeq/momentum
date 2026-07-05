from __future__ import annotations

import datetime as dt

import pytest

from etf_momentum.db.global_benchmark_repo import (
    get_global_benchmark_date_range,
    list_global_benchmark_pool,
    normalize_adjust,
    upsert_global_benchmark_pool,
)
from etf_momentum.db.models import GlobalBenchmarkPrice


def test_global_benchmark_pool_upsert_and_list(session_factory) -> None:
    with session_factory() as db:
        upsert_global_benchmark_pool(
            db,
            code="^GSPC",
            series_kind="price",
            name="标普500",
            code_format="yahoo",
            provider_hint="auto",
            provider_symbol="^GSPC",
            source_locked=False,
            fallback_sources=None,
            start_date="20000101",
            end_date="20250101",
        )
        upsert_global_benchmark_pool(
            db,
            code="000300",
            series_kind="price",
            name="沪深300",
            code_format="cn_6",
            provider_hint="tencent",
            provider_symbol="000300",
            source_locked=False,
            fallback_sources=None,
            start_date="20050101",
            end_date="20250101",
        )
        db.commit()
        rows = list_global_benchmark_pool(db)
        assert [(x.code, x.series_kind) for x in rows] == [
            ("000300", "price"),
            ("^GSPC", "price"),
        ]

        upsert_global_benchmark_pool(
            db,
            code="^GSPC",
            series_kind="price",
            name="标普500指数",
            code_format="yahoo",
            provider_hint="yahoo",
            provider_symbol="^GSPC",
            source_locked=True,
            fallback_sources=[{"provider": "stooq", "symbol": "^GSPC"}],
            start_date="20000101",
            end_date="20250601",
        )
        db.commit()
        rows2 = {(x.code, x.series_kind): x for x in list_global_benchmark_pool(db)}
        assert rows2[("^GSPC", "price")].name == "标普500指数"
        assert rows2[("^GSPC", "price")].provider_hint == "yahoo"


def test_global_benchmark_date_range_uses_none_only(session_factory) -> None:
    with session_factory() as db:
        d0 = dt.date(2024, 1, 2)
        d1 = dt.date(2024, 1, 5)
        db.add(
            GlobalBenchmarkPrice(
                code="^NDX",
                trade_date=d0,
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=10.0,
                amount=10.0,
                source="unit",
                adjust="none",
            )
        )
        db.add(
            GlobalBenchmarkPrice(
                code="^NDX",
                trade_date=d1,
                open=2.0,
                high=2.0,
                low=2.0,
                close=2.0,
                volume=10.0,
                amount=20.0,
                source="unit",
                adjust="none",
            )
        )
        db.commit()
        assert get_global_benchmark_date_range(db, code="^NDX", adjust="none") == (
            "20240102",
            "20240105",
        )
        with pytest.raises(ValueError):
            _ = get_global_benchmark_date_range(db, code="^NDX", adjust="hfq")


def test_global_benchmark_normalize_adjust_rejects_non_none() -> None:
    assert normalize_adjust("none") == "none"
    assert normalize_adjust("raw") == "none"
    with pytest.raises(ValueError):
        _ = normalize_adjust("qfq")
