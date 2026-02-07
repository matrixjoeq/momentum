import datetime as dt

import pandas as pd

from etf_momentum.strategy.vix_signal import backtest_vix_next_day_signal


def test_backtest_returns_desc_trades_and_nav_lengths():
    cn_dates = [
        dt.date(2024, 1, 2),
        dt.date(2024, 1, 3),
        dt.date(2024, 1, 4),
        dt.date(2024, 1, 5),
        dt.date(2024, 1, 8),
        dt.date(2024, 1, 9),
    ]
    etf_close = pd.Series([100.0, 101.0, 100.0, 102.0, 101.0, 103.0], index=cn_dates)

    us_dates = [
        dt.date(2024, 1, 1),
        dt.date(2024, 1, 2),
        dt.date(2024, 1, 3),
        dt.date(2024, 1, 4),
        dt.date(2024, 1, 5),
        dt.date(2024, 1, 8),
        dt.date(2024, 1, 9),
    ]
    vix_close = pd.Series([10.0, 11.0, 10.0, 12.0, 11.5, 11.0, 12.0], index=us_dates)

    out = backtest_vix_next_day_signal(
        etf_close_cn=etf_close,
        etf_open_cn=etf_close.copy(),
        index_close_us=vix_close,
        start=cn_dates[0],
        end=cn_dates[-1],
        index_align="none",
        lookback_window=20,
        threshold_quantile=0.01,  # allow trades
        trade_cost_bps=0.0,
        initial_position="long",
        exec_model="open_open",
    )
    assert out["ok"] is True
    series = out["series"]
    assert len(series["dates"]) == len(series["nav_strategy"]) == len(series["nav_buy_hold"])
    trades = out["trades"]
    assert trades[0]["date"] >= trades[-1]["date"]  # desc

