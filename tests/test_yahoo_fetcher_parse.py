import datetime as dt

import pandas as pd

from etf_momentum.data.yahoo_fetcher import _extract_chart_series


def test_extract_chart_series_parses_dates_and_close():
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1704153600, 1704240000],  # 2024-01-02, 2024-01-03 UTC
                    "indicators": {"quote": [{"close": [10.0, 11.5]}]},
                }
            ],
            "error": None,
        }
    }
    df = _extract_chart_series(payload)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["date", "close"]
    assert df["date"].tolist() == [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    assert df["close"].tolist() == [10.0, 11.5]

