from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from etf_momentum.data.akshare_fetcher import FetchRequest, fetch_etf_daily_qfq


class FakeAk:
    def __init__(self, df, supports_range: bool = True):
        self.df = df
        self.supports_range = supports_range

    def fund_etf_hist_em(self, **kwargs):
        if not self.supports_range and ("start_date" in kwargs or "end_date" in kwargs):
            raise TypeError("no start/end support")
        return self.df


def test_fetch_returns_empty_on_empty_df() -> None:
    ak = FakeAk(pd.DataFrame(), supports_range=True)
    rows = fetch_etf_daily_qfq(ak, FetchRequest(code="510300", start_date="20240101", end_date="20240131"))
    assert rows == []


def test_fetch_raises_on_missing_required_columns() -> None:
    df = pd.DataFrame({"日期": ["2024-01-02"], "开盘": [1.0]})
    ak = FakeAk(df)
    with pytest.raises(ValueError):
        fetch_etf_daily_qfq(ak, FetchRequest(code="510300", start_date="20240101", end_date="20240131"))


def test_fetch_parses_datetime_and_date_objects() -> None:
    df = pd.DataFrame(
        {
            "日期": [dt.datetime(2024, 1, 2), dt.date(2024, 1, 3)],
            "开盘": [1.0, 2.0],
            "最高": [1.2, 2.2],
            "最低": [0.8, 1.8],
            "收盘": [1.1, 2.1],
        }
    )
    ak = FakeAk(df)
    rows = fetch_etf_daily_qfq(ak, FetchRequest(code="510300", start_date="20240101", end_date="20240131"))
    assert [r.trade_date.isoformat() for r in rows] == ["2024-01-02", "2024-01-03"]

