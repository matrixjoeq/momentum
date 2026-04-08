from __future__ import annotations

import datetime as dt

from etf_momentum.db.models import EtfPrice


def add_price_all_adjustments(
    db,
    *,
    code: str,
    day: dt.date,
    close: float,
    open_price: float | None = None,
    high: float | None = None,
    low: float | None = None,
    source: str = "eastmoney",
    adjust_order: tuple[str, str, str] = ("none", "hfq", "qfq"),
) -> None:
    """Insert one price row for each adjust basis: none/hfq/qfq."""
    c = float(close)
    o = float(open_price if open_price is not None else c)
    h = float(high if high is not None else c)
    low_px = float(low if low is not None else c)
    for adj in adjust_order:
        db.add(
            EtfPrice(
                code=str(code),
                trade_date=day,
                open=o,
                high=h,
                low=low_px,
                close=c,
                source=str(source),
                adjust=adj,
            )
        )


def seed_close_series_all_adjustments(
    db,
    *,
    code: str,
    dates: list[dt.date],
    closes: list[float],
    high_mult: float = 1.0,
    low_mult: float = 1.0,
    adjust_order: tuple[str, str, str] = ("hfq", "qfq", "none"),
) -> None:
    """Insert a close series for all adjust bases with optional high/low bands."""
    assert len(dates) == len(closes)
    for d, px in zip(dates, closes):
        c = float(px)
        add_price_all_adjustments(
            db,
            code=code,
            day=d,
            close=c,
            open_price=c,
            high=float(c * float(high_mult)),
            low=float(c * float(low_mult)),
            adjust_order=adjust_order,
        )
