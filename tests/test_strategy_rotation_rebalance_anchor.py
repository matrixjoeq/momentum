import datetime as dt

import pandas as pd
import pytest

from etf_momentum.db.models import EtfPrice
from etf_momentum.strategy.rotation import RotationInputs, backtest_rotation


def _add_price(db, *, code: str, day: dt.date, close: float) -> None:
    for adj in ("none", "hfq", "qfq"):
        db.add(
            EtfPrice(
                code=code,
                trade_date=day,
                open=float(close),
                high=float(close),
                low=float(close),
                close=float(close),
                source="eastmoney",
                adjust=adj,
            )
        )


def _first_start_date(out: dict) -> str:
    rows = out.get("holdings") or []
    assert rows
    return str(rows[0].get("start_date"))


def test_weekly_anchor_wednesday_decision_day_is_wednesday(session_factory):
    sf = session_factory
    with sf() as db:
        dates = [d.date() for d in pd.date_range("2024-01-01", "2024-01-31", freq="B")]
        for i, d in enumerate(dates):
            _add_price(db, code="A1", day=d, close=100.0 + i * 1.0)
            _add_price(db, code="B1", day=d, close=90.0 + i * 0.8)
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["A1", "B1"],
                start=dates[0],
                end=dates[-1],
                rebalance="weekly",
                rebalance_anchor=3,  # Wed
                top_k=1,
                lookback_days=3,
            ),
        )

    assert out["holdings"]
    # 2024-01-03 is Wednesday
    assert out["holdings"][0]["decision_date"] == "2024-01-03"


def test_monthly_anchor_non_trading_day_shift_modes(session_factory):
    sf = session_factory
    with sf() as db:
        dates = [d.date() for d in pd.date_range("2024-01-01", "2024-03-31", freq="B")]
        for i, d in enumerate(dates):
            _add_price(db, code="A1", day=d, close=100.0 + i * 1.1)
            _add_price(db, code="B1", day=d, close=100.0 + i * 0.7)
        db.commit()

        base = RotationInputs(
            codes=["A1", "B1"],
            start=dates[0],
            end=dates[-1],
            rebalance="monthly",
            rebalance_anchor=6,  # 2024-01-06 is Saturday
            top_k=1,
            lookback_days=3,
        )
        out_prev = backtest_rotation(db, base)
        out_next = backtest_rotation(db, RotationInputs(**{**base.__dict__, "rebalance_shift": "next"}))
        out_skip = backtest_rotation(db, RotationInputs(**{**base.__dict__, "rebalance_shift": "skip"}))

    assert _first_start_date(out_prev) == "2024-01-08"
    assert _first_start_date(out_next) == "2024-01-09"
    assert _first_start_date(out_skip) == "2024-02-07"
    assert (out_prev["holdings"][0].get("rebalance_hit_mode")) == "prev"
    assert (out_next["holdings"][0].get("rebalance_hit_mode")) == "next"
    assert (out_skip["holdings"][0].get("rebalance_hit_mode")) == "exact"


def test_exec_price_default_is_open(session_factory):
    sf = session_factory
    with sf() as db:
        dates = [d.date() for d in pd.date_range("2024-01-01", "2024-01-31", freq="B")]
        for i, d in enumerate(dates):
            # Keep open != close to ensure non-trivial execution basis.
            for adj in ("none", "hfq", "qfq"):
                db.add(
                    EtfPrice(
                        code="A1",
                        trade_date=d,
                        open=100.0 + i * 0.8,
                        high=101.0 + i * 0.8,
                        low=99.0 + i * 0.8,
                        close=100.5 + i * 1.1,
                        source="eastmoney",
                        adjust=adj,
                    )
                )
                db.add(
                    EtfPrice(
                        code="B1",
                        trade_date=d,
                        open=90.0 + i * 0.5,
                        high=91.0 + i * 0.5,
                        low=89.0 + i * 0.5,
                        close=90.2 + i * 0.9,
                        source="eastmoney",
                        adjust=adj,
                    )
                )
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["A1", "B1"],
                start=dates[0],
                end=dates[-1],
                rebalance="weekly",
                top_k=1,
                lookback_days=3,
            ),
        )

    assert out.get("exec_price") == "open"


@pytest.mark.parametrize(
    "rebalance,anchor,msg_regex",
    [
        ("weekly", 6, r"weekly rebalance_anchor must be within \[1\.\.5\]"),
        ("quarterly", 91, r"quarterly rebalance_anchor must be within \[1\.\.90\]"),
        ("yearly", 366, r"yearly rebalance_anchor must be within \[1\.\.365\]"),
    ],
)
def test_anchor_range_validation(session_factory, rebalance, anchor, msg_regex):
    sf = session_factory
    with sf() as db:
        dates = [d.date() for d in pd.date_range("2024-01-01", "2024-02-29", freq="B")]
        for i, d in enumerate(dates):
            _add_price(db, code="A1", day=d, close=100.0 + i)
            _add_price(db, code="B1", day=d, close=90.0 + i * 0.5)
        db.commit()

        with pytest.raises(ValueError, match=msg_regex):
            backtest_rotation(
                db,
                RotationInputs(
                    codes=["A1", "B1"],
                    start=dates[0],
                    end=dates[-1],
                    rebalance=rebalance,
                    rebalance_anchor=anchor,
                    top_k=1,
                    lookback_days=3,
                ),
            )
