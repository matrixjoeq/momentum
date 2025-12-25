from __future__ import annotations

import pandas as pd

from etf_momentum.data.akshare_fetcher import FetchRequest, fetch_etf_daily_qfq


class FakeAk:
    def __init__(self, df: pd.DataFrame, supports_range: bool):
        self._df = df
        self._supports_range = supports_range

    def fund_etf_hist_em(self, **kwargs):
        if not self._supports_range and ("start_date" in kwargs or "end_date" in kwargs):
            raise TypeError("no start/end support")
        return self._df


def test_fetch_filters_and_maps_columns() -> None:
    df = pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "开盘": [1, 2, 3],
            "最高": [1.2, 2.2, 3.2],
            "最低": [0.8, 1.8, 2.8],
            "收盘": [1.1, 2.1, 3.1],
            "成交量": [10, 20, 30],
            "成交额": [100, 200, 300],
        }
    )
    ak = FakeAk(df, supports_range=False)
    req = FetchRequest(code="510300", start_date="20240103", end_date="20240104", adjust="qfq")
    rows = fetch_etf_daily_qfq(ak, req)
    assert [r.trade_date.isoformat() for r in rows] == ["2024-01-03", "2024-01-04"]
    assert rows[0].open == 2.0
    assert rows[0].amount == 200.0


def test_fetch_dedup_by_trade_date_last_wins() -> None:
    df = pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-02"],
            "开盘": [1, 9],
            "最高": [1.2, 9.2],
            "最低": [0.8, 8.8],
            "收盘": [1.1, 9.1],
        }
    )
    ak = FakeAk(df, supports_range=True)
    req = FetchRequest(code="510300", start_date="20240101", end_date="20240131", adjust="qfq")
    rows = fetch_etf_daily_qfq(ak, req)
    assert len(rows) == 1
    assert rows[0].open == 9.0

