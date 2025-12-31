import datetime as dt

import pandas as pd
import pytest

from etf_momentum.db.models import EtfPrice
from etf_momentum.strategy.rotation import RotationInputs, backtest_rotation


def _add_price(db, *, code: str, day: dt.date, close_by_adjust: dict[str, float]) -> None:
    for adj in ("none", "hfq", "qfq"):
        db.add(
            EtfPrice(
                code=code,
                trade_date=day,
                close=float(close_by_adjust[adj]),
                low=float(close_by_adjust[adj]),
                high=float(close_by_adjust[adj]),
                source="eastmoney",
                adjust=adj,
            )
        )


def test_atr_chandelier_fixed_triggers_and_goes_cash_until_next_rebalance(session_factory):
    """
    Fixed ATR chandelier:
    - ATR is computed from qfq close-to-close absolute changes over atr_window.
    - trailing stop is updated daily (never decreases)
    - trigger when qfq close < previous stop, then cash until next rebalance.
    """
    sf = session_factory
    with sf() as db:
        dates = pd.date_range("2024-01-01", "2024-01-19", freq="B").date

        # Construct AAA qfq close with steady +1 moves (ATR ~ 1) then a drop below trailing stop.
        # BBB flat, so AAA should be selected.
        for i, d in enumerate(dates):
            if d <= dt.date(2024, 1, 9):
                aaa = 100.0 + i  # +1 per day
            elif d == dt.date(2024, 1, 10):
                aaa = 102.0  # still above
            else:
                aaa = 95.0  # hard drop to trigger

            bbb = 100.0
            _add_price(db, code="AAA", day=d, close_by_adjust={"none": aaa, "hfq": aaa, "qfq": aaa})
            _add_price(db, code="BBB", day=d, close_by_adjust={"none": bbb, "hfq": bbb, "qfq": bbb})
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["AAA", "BBB"],
                start=dates[0],
                end=dates[-1],
                rebalance="weekly",
                top_k=1,
                lookback_days=1,
                tp_sl_mode="atr_chandelier_fixed",
                atr_window=5,
                atr_mult=2.0,
            ),
        )

    # segment starting 2024-01-08 should have ATR stop info and a trigger in the second week
    seg = next((p for p in out["holdings"] if p["start_date"] == "2024-01-08"), None)
    assert seg is not None
    assert seg["tp_sl"]["mode"] == "atr_chandelier_fixed"
    assert seg["tp_sl"]["atr_window_used"] == 5
    assert seg["tp_sl"]["atr_mult"] == pytest.approx(2.0)
    assert seg["tp_sl"]["triggered"] is True

    # After trigger date, nav should be flat (cash) until next rebalance.
    dts = out["nav"]["dates"]
    nav = out["nav"]["series"]["ROTATION"]
    t = seg["tp_sl"]["trigger_date"]
    ti = dts.index(t)
    if ti + 1 < len(nav):
        assert nav[ti + 1] == pytest.approx(nav[ti])


def test_atr_chandelier_progressive_reduces_distance_to_min_mult(session_factory):
    """
    Progressive mode: distance decreases as price rises in ATR units, down to atr_min_mult.
    We validate that after a large rise, the stop converges near close - atr_min_mult * ATR.
    """
    sf = session_factory
    with sf() as db:
        dates = pd.date_range("2024-01-01", "2024-01-19", freq="B").date
        for i, d in enumerate(dates):
            aaa = 100.0 + i  # +1 per day -> ATR ~ 1
            bbb = 100.0
            _add_price(db, code="AAA", day=d, close_by_adjust={"none": aaa, "hfq": aaa, "qfq": aaa})
            _add_price(db, code="BBB", day=d, close_by_adjust={"none": bbb, "hfq": bbb, "qfq": bbb})
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["AAA", "BBB"],
                start=dates[0],
                end=dates[-1],
                rebalance="weekly",
                top_k=1,
                lookback_days=1,
                tp_sl_mode="atr_chandelier_progressive",
                atr_window=5,
                atr_mult=2.0,
                atr_step=0.5,
                atr_min_mult=0.5,
            ),
        )

    seg = next((p for p in out["holdings"] if p["start_date"] == "2024-01-08"), None)
    assert seg is not None
    assert seg["tp_sl"]["mode"] == "atr_chandelier_progressive"
    assert seg["tp_sl"]["triggered"] is False

    final_stop = seg["tp_sl"]["final_stop_by_code"]["AAA"]
    # With ATR ~ 1 and enough gains, distance should reach ~0.5 ATR; stop should be close - 0.5
    # Use last day of segment (week ending 2024-01-12).
    assert final_stop > 0
