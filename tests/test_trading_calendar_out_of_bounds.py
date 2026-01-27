import datetime as dt


def test_is_trading_day_date_out_of_bounds_returns_true(monkeypatch):
    from etf_momentum.calendar import trading_calendar as tc  # pylint: disable=import-error

    class _FakeCal:
        def is_session(self, _ts):
            import exchange_calendars as xcals  # pylint: disable=import-error

            # Raise the exact exception type that broke cloud sync.
            raise xcals.errors.DateOutOfBounds(None, None, "date")

    monkeypatch.setattr(tc, "_get_calendar", lambda _name="XSHG": _FakeCal())

    assert tc.is_trading_day(dt.date(2099, 1, 1), cal="XSHG") is True


def test_shift_to_trading_day_date_out_of_bounds_returns_original(monkeypatch):
    from etf_momentum.calendar import trading_calendar as tc  # pylint: disable=import-error

    class _FakeCal:
        def is_session(self, _ts):
            import exchange_calendars as xcals  # pylint: disable=import-error

            raise xcals.errors.DateOutOfBounds(None, None, "date")

    monkeypatch.setattr(tc, "_get_calendar", lambda _name="XSHG": _FakeCal())

    d = dt.date(2099, 1, 1)
    assert tc.shift_to_trading_day(d, shift="prev", cal="XSHG") == d

