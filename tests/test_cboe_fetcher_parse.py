import datetime as dt

import pandas as pd

from etf_momentum.data.cboe_fetcher import _parse_cboe_history_csv


def test_parse_cboe_history_csv_reads_date_and_close():
    text = "DATE,OPEN,HIGH,LOW,CLOSE\n2024-01-02,13,14,12,13.5\n2024-01-03,12,13,11,12.5\n"
    df = _parse_cboe_history_csv(text)
    assert isinstance(df, pd.DataFrame)
    assert df["date"].tolist() == [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    assert df["close"].tolist() == [13.5, 12.5]


def test_parse_cboe_history_csv_supports_gvz_two_column_format():
    text = "DATE,GVZ\n09/18/2009,22.62\n09/21/2009,23.22\n"
    df = _parse_cboe_history_csv(text)
    assert df["date"].tolist() == [dt.date(2009, 9, 18), dt.date(2009, 9, 21)]
    assert df["close"].tolist() == [22.62, 23.22]

