import datetime as dt

from etf_momentum.calendar.trading_calendar import is_trading_day, shift_to_trading_day


def test_shift_to_trading_day_prev_weekend_xshg():
    # 2024-01-06 is a Saturday
    d = dt.date(2024, 1, 6)
    assert is_trading_day(d) is False
    prev = shift_to_trading_day(d, shift="prev")
    assert prev == dt.date(2024, 1, 5)

