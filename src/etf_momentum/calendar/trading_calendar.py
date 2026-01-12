from __future__ import annotations

import datetime as dt
from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=8)
def _get_calendar(name: str = "XSHG"):
    """
    Return an exchange calendar. Default uses Shanghai Stock Exchange sessions.

    Notes:
    - This is intended for *live simulation scheduling* (pre-knowledge of weekends/holidays).
    - For historical backtests, the DB price calendar already reflects trading days.
    """
    import exchange_calendars as xcals

    return xcals.get_calendar(name)


def is_trading_day(d: dt.date, *, cal: str = "XSHG") -> bool:
    c = _get_calendar(cal)
    ts = pd.Timestamp(d)
    return bool(c.is_session(ts))


def shift_to_trading_day(d: dt.date, *, shift: str = "prev", cal: str = "XSHG") -> dt.date:
    """
    If d is a non-trading day, shift to prev/next trading day.
    """
    s = (shift or "prev").strip().lower()
    if s not in {"prev", "next"}:
        raise ValueError("shift must be one of: prev|next")
    c = _get_calendar(cal)
    ts = pd.Timestamp(d)
    if c.is_session(ts):
        return d
    # exchange_calendars expects session labels for previous_session/next_session.
    # Use date_to_session with direction to handle non-session inputs (weekends/holidays).
    direction = "previous" if s == "prev" else "next"
    sess = c.date_to_session(ts, direction=direction)
    return pd.Timestamp(sess).date()


def trading_days(start: dt.date, end: dt.date, *, cal: str = "XSHG") -> list[dt.date]:
    """
    List trading days (sessions) in [start, end].
    """
    c = _get_calendar(cal)
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    sessions = c.sessions_in_range(s, e)
    return [x.date() for x in sessions]

