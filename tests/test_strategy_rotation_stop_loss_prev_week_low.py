import datetime as dt

import pandas as pd
import pytest

from etf_momentum.db.models import EtfPrice
from etf_momentum.strategy.rotation import RotationInputs, backtest_rotation


def _add_price(
    db,
    *,
    code: str,
    day: dt.date,
    close_by_adjust: dict[str, float],
    low: float = 1.0,
    high: float = 1.0,
) -> None:
    """
    Insert EtfPrice rows for each adjust. Allows qfq close to differ from hfq/none for testing
    stop-loss (qfq) vs P&L (hfq/none) separation.
    """
    for adj in ("none", "hfq", "qfq"):
        db.add(
            EtfPrice(
                code=code,
                trade_date=day,
                close=float(close_by_adjust[adj]),
                low=float(low),
                high=float(high),
                source="eastmoney",
                adjust=adj,
            )
        )


def test_prev_week_low_stop_loss_triggers_and_goes_cash_until_next_rebalance(session_factory):
    """
    Weekly rebalance:
    - decision on Friday; holdings apply from next trading day (Monday).
    - initial stop for new position uses previous rebalance-period min(qfq close) (first segment falls back to current period).
    - if any day's qfq close < stop, stop triggers and portfolio stays in cash until next rebalance.
    """
    sf = session_factory
    with sf() as db:
        # business days, spanning 3 Fridays: 2024-01-05, 2024-01-12, 2024-01-19
        dates = pd.date_range("2024-01-01", "2024-01-19", freq="B").date

        # Two assets: AAA trends up; BBB flat.
        # We craft AAA's qfq close to breach stop within the 2nd week.
        for i, d in enumerate(dates):
            # closes
            close_aaa = 100.0 + i
            close_bbb = 100.0

            # Trigger stop on 2024-01-10 (Wed): qfq close dips below stop(=100.0)
            if d == dt.date(2024, 1, 10):
                close_aaa = 99.0

            _add_price(db, code="AAA", day=d, close_by_adjust={"none": close_aaa, "hfq": close_aaa, "qfq": close_aaa})
            _add_price(db, code="BBB", day=d, close_by_adjust={"none": close_bbb, "hfq": close_bbb, "qfq": close_bbb})

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
                skip_days=0,
                risk_off=False,
                cost_bps=0.0,
                tp_sl_mode="prev_week_low_stop",
            ),
        )

    # Find the holding segment that starts at 2024-01-08 (Mon after 2024-01-05 Fri decision).
    seg = next((p for p in out["holdings"] if p["start_date"] == "2024-01-08"), None)
    assert seg is not None
    assert seg["tp_sl"]["mode"] == "prev_week_low_stop"
    assert seg["tp_sl"]["triggered"] is True
    assert seg["tp_sl"]["trigger_date"] == "2024-01-10"
    # initial stop should be last-week low (trailing 5 trading days ending 2024-01-05) ~= 100
    sl = seg["tp_sl"]["stop_loss_level_by_code"]["AAA"]
    assert sl == pytest.approx(100.0, abs=1e-9)

    # After stop triggers on 2024-01-10, portfolio should be cash on 2024-01-11 and 2024-01-12.
    dts = out["nav"]["dates"]
    nav = out["nav"]["series"]["ROTATION"]
    i10 = dts.index("2024-01-10")
    i11 = dts.index("2024-01-11")
    i12 = dts.index("2024-01-12")
    assert nav[i11] == pytest.approx(nav[i10])
    assert nav[i12] == pytest.approx(nav[i10])


def test_prev_week_low_stop_loss_supports_monthly_and_updates_when_holding_unchanged(session_factory):
    """
    Monthly rebalance:
    - stop for new entry in the first rebalance month falls back to current-month min(qfq close)
      because there is no previous month in-range.
    - if holding remains unchanged at next rebalance, stop updates to the current month min(qfq close).
    """
    sf = session_factory
    with sf() as db:
        dates = pd.date_range("2024-01-02", "2024-03-08", freq="B").date
        for i, d in enumerate(dates):
            # AAA uptrend; BBB flat -> AAA should stay selected.
            close_aaa = 100.0 + i * 0.5
            close_bbb = 100.0
            _add_price(db, code="AAA", day=d, close_by_adjust={"none": close_aaa, "hfq": close_aaa, "qfq": close_aaa})
            _add_price(db, code="BBB", day=d, close_by_adjust={"none": close_bbb, "hfq": close_bbb, "qfq": close_bbb})
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["AAA", "BBB"],
                start=dates[0],
                end=dates[-1],
                rebalance="monthly",
                top_k=1,
                lookback_days=5,
                skip_days=0,
                risk_off=False,
                cost_bps=0.0,
                tp_sl_mode="prev_week_low_stop",
            ),
        )

    # Segment starting 2024-02-01 (after Jan decision) uses Jan min close as stop (fallback path).
    seg_feb = next((p for p in out["holdings"] if p["start_date"] == "2024-02-01"), None)
    assert seg_feb is not None
    assert seg_feb["tp_sl"]["stop_loss_level_by_code"]["AAA"] == pytest.approx(100.0, abs=1e-9)

    # Segment starting 2024-03-01 should update stop to Feb min close (holding unchanged).
    seg_mar = next((p for p in out["holdings"] if p["start_date"] == "2024-03-01"), None)
    assert seg_mar is not None
    # Feb business days start at 2024-02-01; AAA close there is 100 + index*0.5, min is at 2024-02-01.
    # That value is deterministic given the construction above.
    # Compute expected from the nav date list to avoid hardcoding offsets.
    feb_1_idx = list(dates).index(dt.date(2024, 2, 1))
    expected_feb_min = 100.0 + feb_1_idx * 0.5
    assert seg_mar["tp_sl"]["stop_loss_level_by_code"]["AAA"] == pytest.approx(expected_feb_min, abs=1e-9)


def test_prev_week_low_stop_loss_fallback_to_current_period_min_when_decision_close_below_prev_min(session_factory):
    """
    Special rule:
    If on rebalance day the qfq close is already below previous-period min(qfq close),
    use current-period min(qfq close) as the stop (for the new entry case).
    """
    sf = session_factory
    with sf() as db:
        dates = pd.date_range("2024-01-01", "2024-01-19", freq="B").date

        for i, d in enumerate(dates):
            # Make BBB selected in week1 (rising), then switch to AAA in week2 (rising).
            # Keep hfq smooth for ranking; craft qfq dip on 2024-01-12 to trigger fallback logic for AAA.
            # AAA hfq rises in week2; BBB hfq rises in week1.
            aaa_hfq = 100.0 + (0.0 if d <= dt.date(2024, 1, 5) else (i - 4) * 2.0)
            bbb_hfq = 100.0 + (i * 2.0 if d <= dt.date(2024, 1, 5) else 0.0)
            # Execution/none follows hfq for simplicity.
            aaa_none = aaa_hfq
            bbb_none = bbb_hfq

            # qfq close mostly follows hfq, but on decision day 2024-01-12 drop AAA qfq below its previous-week min.
            aaa_qfq = aaa_hfq
            if d == dt.date(2024, 1, 12):
                aaa_qfq = 90.0
            bbb_qfq = bbb_hfq

            _add_price(db, code="AAA", day=d, close_by_adjust={"none": aaa_none, "hfq": aaa_hfq, "qfq": aaa_qfq})
            _add_price(db, code="BBB", day=d, close_by_adjust={"none": bbb_none, "hfq": bbb_hfq, "qfq": bbb_qfq})
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
                skip_days=0,
                risk_off=False,
                cost_bps=0.0,
                tp_sl_mode="prev_week_low_stop",
            ),
        )

    # Week2 decision at 2024-01-12 should switch into AAA, and because AAA qfq close(90) < prev_week_min,
    # stop should use current-week min (which is 90.0).
    seg = next((p for p in out["holdings"] if p["start_date"] == "2024-01-15"), None)
    assert seg is not None
    assert seg["picks"] == ["AAA"]
    assert seg["tp_sl"]["stop_loss_level_by_code"]["AAA"] == pytest.approx(90.0, abs=1e-9)
    assert seg["tp_sl"]["triggered"] is False

