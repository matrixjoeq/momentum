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
    import exchange_calendars as xcals  # pylint: disable=import-error

    # IMPORTANT:
    # Some exchange_calendars builds ship with limited precomputed date ranges for some calendars.
    # If we don't explicitly request a sufficiently wide range, `is_session()` can raise
    # DateOutOfBounds for "future" dates (e.g., after year-end) and break background jobs.
    today = dt.date.today()
    end = today + dt.timedelta(days=365 * 20)
    # XSHG holidays are only recorded back to 1991 in exchange_calendars.
    # Trying to instantiate earlier will raise an exception and break cloud sync jobs.
    start = "1991-01-01"
    try:
        return xcals.get_calendar(name, start=start, end=end.isoformat())
    except Exception:  # pylint: disable=broad-exception-caught
        # Fallback to library defaults if explicit range is not supported for this calendar build.
        return xcals.get_calendar(name)


def is_trading_day(d: dt.date, *, cal: str = "XSHG") -> bool:
    import exchange_calendars as xcals  # pylint: disable=import-error

    c = _get_calendar(cal)
    ts = pd.Timestamp(d)
    try:
        return bool(c.is_session(ts))
    except xcals.errors.DateOutOfBounds:
        # For operational jobs (market sync), prefer to "run anyway" rather than crash or
        # incorrectly mark as non-trading day forever due to calendar range limitations.
        return True


def shift_to_trading_day(d: dt.date, *, shift: str = "prev", cal: str = "XSHG") -> dt.date:
    """
    If d is a non-trading day, shift to prev/next trading day.
    """
    s = (shift or "prev").strip().lower()
    if s not in {"prev", "next"}:
        raise ValueError("shift must be one of: prev|next")
    import exchange_calendars as xcals  # pylint: disable=import-error

    c = _get_calendar(cal)
    ts = pd.Timestamp(d)
    try:
        if c.is_session(ts):
            return d
    except xcals.errors.DateOutOfBounds:
        # Same rationale as is_trading_day: don't crash operational paths on out-of-bounds.
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
    import exchange_calendars as xcals  # pylint: disable=import-error

    c = _get_calendar(cal)
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    try:
        sessions = c.sessions_in_range(s, e)
        return [x.date() for x in sessions]
    except xcals.errors.DateOutOfBounds:
        # Fallback: approximate with business days if calendar is out-of-bounds.
        return [x.date() for x in pd.bdate_range(s, e)]

