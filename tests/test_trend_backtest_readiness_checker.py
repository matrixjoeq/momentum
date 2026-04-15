from __future__ import annotations

import datetime as dt

from scripts.check_trend_backtest_readiness import (
    _max_abs_return,
    _parse_codes,
    _parse_date_yyyymmdd,
)


def test_parse_codes_dedup_and_normalize() -> None:
    out = _parse_codes(" rb0, IF0,rb0 ,  ")
    assert out == ["RB0", "IF0"]


def test_parse_date_yyyymmdd() -> None:
    d = _parse_date_yyyymmdd("20250131")
    assert d == dt.date(2025, 1, 31)


def test_max_abs_return_handles_empty_and_values() -> None:
    assert _max_abs_return([]) is None
    assert _max_abs_return([100.0]) is None
    out = _max_abs_return([100.0, 110.0, 99.0])
    assert out is not None
    assert round(out, 3) == 0.100
