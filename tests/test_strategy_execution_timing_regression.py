import datetime as dt

import pytest

from etf_momentum.analysis.calendar_timing_strategy import (
    CalendarTimingStrategyInputs,
    compute_calendar_timing_strategy_backtest,
)
from etf_momentum.analysis.bt_trend import compute_trend_backtest_bt
from etf_momentum.analysis.trend import TrendInputs, compute_trend_backtest
from etf_momentum.db.models import EtfPrice
from etf_momentum.strategy.rotation import RotationInputs, backtest_rotation


def _seed_one(
    db,
    *,
    code: str,
    dates: list[dt.date],
    ohlc_none: list[tuple[float, float]],
    close_qfq: list[float],
) -> None:
    for d, (o_n, c_n), c_q in zip(dates, ohlc_none, close_qfq, strict=True):
        o_q = float(c_q)
        h_q = max(o_q, float(c_q)) * 1.01
        l_q = min(o_q, float(c_q)) * 0.99
        db.add(
            EtfPrice(
                code=code,
                trade_date=d,
                open=o_q,
                high=h_q,
                low=l_q,
                close=float(c_q),
                source="eastmoney",
                adjust="qfq",
            )
        )

        h_n = max(float(o_n), float(c_n)) * 1.01
        l_n = min(float(o_n), float(c_n)) * 0.99
        for adj in ("none", "hfq"):
            db.add(
                EtfPrice(
                    code=code,
                    trade_date=d,
                    open=float(o_n),
                    high=h_n,
                    low=l_n,
                    close=float(c_n),
                    source="eastmoney",
                    adjust=adj,
                )
            )


@pytest.mark.parametrize(
    ("exec_price", "expected_nav"),
    [
        # open: 执行日使用当日 open->close 收益，故与仅 forward 时不同
        ("open", 1.148989898989899),
        ("close", 7.0 / 9.0),
        ("oc2", 0.8565505482172148),
    ],
)
def test_rotation_exec_price_uses_execution_day_forward_return(
    session_factory, exec_price: str, expected_nav: float
):
    sf = session_factory
    with sf() as db:
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(5)]

        # AAA always wins momentum on qfq; execution path is controlled by none/hfq OHLC.
        aaa_none = [(10.0, 10.0), (11.0, 12.0), (9.0, 9.0), (9.9, 13.0), (8.0, 7.0)]
        aaa_qfq = [10.0, 11.0, 12.0, 13.0, 14.0]
        _seed_one(db, code="AAA", dates=dates, ohlc_none=aaa_none, close_qfq=aaa_qfq)

        # BBB stays flat and should never be selected.
        bbb_none = [(10.0, 10.0)] * len(dates)
        bbb_qfq = [10.0] * len(dates)
        _seed_one(db, code="BBB", dates=dates, ohlc_none=bbb_none, close_qfq=bbb_qfq)
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["AAA", "BBB"],
                start=start,
                end=dates[-1],
                rebalance="daily",
                top_k=1,
                lookback_days=1,
                skip_days=0,
                exec_price=exec_price,
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )

    final_nav = float(out["nav"]["series"]["ROTATION"][-1])
    assert final_nav == pytest.approx(expected_nav, rel=0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("exec_price", "expected_nav"),
    [
        ("open", 1.148989898989899),
        ("close", 7.0 / 13.0),
    ],
)
def test_trend_exec_price_uses_execution_day_timing_rule(
    session_factory, exec_price: str, expected_nav: float
):
    sf = session_factory
    with sf() as db:
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(5)]
        aaa_none = [(10.0, 10.0), (11.0, 12.0), (9.0, 9.0), (9.9, 13.0), (8.0, 7.0)]
        aaa_qfq = [10.0, 11.0, 12.0, 13.0, 14.0]
        _seed_one(db, code="AAA", dates=dates, ohlc_none=aaa_none, close_qfq=aaa_qfq)
        db.commit()

        out = compute_trend_backtest(
            db,
            TrendInputs(
                code="AAA",
                start=start,
                end=dates[-1],
                strategy="tsmom",
                mom_lookback=2,
                exec_price=exec_price,
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )

    final_nav = float(out["nav"]["series"]["STRAT"][-1])
    assert final_nav == pytest.approx(expected_nav, rel=0.0, abs=1e-12)


@pytest.mark.parametrize(
    "runner",
    [compute_trend_backtest, compute_trend_backtest_bt],
)
def test_close_execution_initializes_atr_state_from_t_plus_one_close(
    session_factory, runner
):
    sf = session_factory
    with sf() as db:
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(7)]
        close = [100.0, 100.0, 100.0, 110.0, 120.0, 130.0, 140.0]
        _seed_one(
            db,
            code="AAA",
            dates=dates,
            ohlc_none=[(value, value) for value in close],
            close_qfq=close,
        )
        db.commit()

        out = runner(
            db,
            TrendInputs(
                code="AAA",
                start=start,
                end=dates[-1],
                strategy="tsmom",
                mom_lookback=2,
                tsmom_entry_threshold=0.02,
                tsmom_exit_threshold=0.0,
                exec_price="close",
                atr_stop_mode="static",
                atr_stop_window=2,
                atr_stop_n=2.0,
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )

    trace = out["risk_controls"]["atr_stop"]["trace_last_rows"]
    entry = next(row for row in trace if row.get("event_type") == "entry")
    market_index = dates.index(dt.date.fromisoformat(str(entry["date"])))
    # The trace date is already mapped to the actual T+1 market session, so
    # every market observation on the row must belong to that same date.
    assert entry["close"] == pytest.approx(close[market_index])
    assert entry["open"] == pytest.approx(close[market_index])
    assert entry["low"] == pytest.approx(close[market_index] * 0.99)
    # Wilder RMA(alpha=1/2): prior ATR 6.55, current TR 11.20 -> 8.875.
    assert entry["atr"] == pytest.approx(8.875)
    assert entry["stop_after"] == pytest.approx(120.0 - 2.0 * 8.875)
    episode = out["trade_statistics"]["trades"][0]
    assert episode["entry_date"] == entry["date"]
    assert episode["entry_price"] == pytest.approx(120.0)
    assert episode["initial_r_pct_nav"] == pytest.approx((2.0 * 8.875) / 120.0)


@pytest.mark.parametrize("exec_price", ["open", "close"])
def test_trend_and_bt_trend_execution_timing_are_aligned(
    session_factory, exec_price: str
):
    sf = session_factory
    with sf() as db:
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(5)]
        # This path magnifies timing differences:
        # - d2 open->close = +9.09%
        # - d3 open->close = -30%
        # If open execution is accidentally delayed by one extra day, final NAV drifts.
        aaa_none = [(10.0, 10.0), (11.0, 12.0), (9.0, 9.0), (9.9, 13.0), (8.0, 7.0)]
        aaa_qfq = [10.0, 11.0, 12.0, 13.0, 14.0]
        _seed_one(db, code="AAA", dates=dates, ohlc_none=aaa_none, close_qfq=aaa_qfq)
        db.commit()

        inp = TrendInputs(
            code="AAA",
            start=start,
            end=dates[-1],
            strategy="tsmom",
            mom_lookback=2,
            exec_price=exec_price,
            cost_bps=0.0,
            slippage_rate=0.0,
        )
        out_trend = compute_trend_backtest(db, inp)
        out_bt = compute_trend_backtest_bt(db, inp)

    nav_trend = float(out_trend["nav"]["series"]["STRAT"][-1])
    nav_bt = float(out_bt["nav"]["series"]["STRAT"][-1])
    assert nav_bt == pytest.approx(nav_trend, rel=0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("exec_price", "expected_reentry_index", "expected_d9_ret"),
    [
        ("open", 9, 0.20),
        ("close", 8, 0.0),
    ],
)
def test_trend_atr_stop_reentry_timing_no_lookahead(
    session_factory,
    exec_price: str,
    expected_reentry_index: int,
    expected_d9_ret: float,
):
    sf = session_factory
    with sf() as db:
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(10)]
        # qfq close for signal/ATR:
        # - steady uptrend to let trailing stop move upward
        # - d7 has an intraday low that keeps the base signal long but breaks the stop
        # - d8 base stays long to allow stop_reentry
        # - d9 distinguishes open from close execution-day return
        close_qfq = [
            100.0,
            102.0,
            104.0,
            106.0,
            108.0,
            110.0,
            106.5,
            112.0,
            120.0,
            120.0,
        ]
        # none/hfq execution OHLC:
        # - mostly open=close
        # - d9 has large intraday open->close move (20%)
        #   * open mode re-enters at d9 open and realizes it
        #   * close mode re-enters at d8 close and has zero d9 close return
        ohlc_none = [(p, p) for p in close_qfq]
        ohlc_none[9] = (100.0, 120.0)
        _seed_one(db, code="AAA", dates=dates, ohlc_none=ohlc_none, close_qfq=close_qfq)
        db.flush()
        qfq_d7 = (
            db.query(EtfPrice)
            .filter(
                EtfPrice.code == "AAA",
                EtfPrice.trade_date == dates[7],
                EtfPrice.adjust == "qfq",
            )
            .one()
        )
        qfq_d7.low = 90.0
        db.commit()

        out = compute_trend_backtest(
            db,
            TrendInputs(
                code="AAA",
                start=start,
                end=dates[-1],
                strategy="ma_filter",
                sma_window=6,
                atr_stop_mode="trailing",
                atr_stop_window=2,
                atr_stop_n=1.0,
                atr_stop_reentry_mode="reenter",
                exec_price=exec_price,
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )

    d = out["nav"]["dates"]
    nav = out["nav"]["series"]["STRAT"]
    eff = out["signals"]["position_effective"]
    atr = out["risk_controls"]["atr_stop"]
    trace = out["next_plan"]["trace"]["atr_stop"]["trace_last_rows"]

    i7 = d.index(dates[7].isoformat())
    i8 = d.index(dates[8].isoformat())
    i9 = d.index(dates[9].isoformat())

    # The d7 low is the first eligible T+1 stop observation.
    assert int(atr["trigger_count"]) >= 1
    assert atr["trigger_dates"] and atr["trigger_dates"][-1] == dates[7].isoformat()
    assert float(eff[i7]) == pytest.approx(0.0, abs=1e-12)

    expected_reentry_i = d.index(dates[expected_reentry_index].isoformat())
    assert float(eff[expected_reentry_i]) == pytest.approx(1.0, abs=1e-12)
    if exec_price == "open":
        assert float(eff[i8]) == pytest.approx(0.0, abs=1e-12)
    has_stop_reentry = any(
        (str(r.get("date")) == dates[8].isoformat())
        and (str(r.get("event_type")) == "entry")
        and (str(r.get("event_reason")) == "stop_reentry")
        for r in trace
    )
    assert has_stop_reentry

    d9_ret = float(nav[i9] / nav[i8] - 1.0)
    assert d9_ret == pytest.approx(expected_d9_ret, rel=0.0, abs=1e-12)


def test_calendar_open_exec_includes_entry_day_and_excludes_exit_day(session_factory):
    sf = session_factory
    with sf() as db:
        start = dt.date(2024, 1, 2)
        dates = [start + dt.timedelta(days=i) for i in range(35)]
        ohlc_none: list[tuple[float, float]] = []
        close_qfq: list[float] = []
        for d in dates:
            if d == dt.date(2024, 1, 3):
                o, c = 100.0, 110.0  # entry execution day: +10%
            elif d == dt.date(2024, 1, 4):
                o, c = (
                    100.0,
                    80.0,
                )  # exit execution day: -20% should NOT be counted in open mode
            else:
                o, c = 100.0, 100.0
            ohlc_none.append((o, c))
            close_qfq.append(c)
        _seed_one(db, code="AAA", dates=dates, ohlc_none=ohlc_none, close_qfq=close_qfq)
        db.commit()

        out = compute_calendar_timing_strategy_backtest(
            db,
            CalendarTimingStrategyInputs(
                mode="single",
                code="AAA",
                codes=None,
                start=dates[0],
                end=dates[-1],
                decision_day=2,  # 2024-01-02 as decision, 2024-01-03 as execution
                hold_days=1,
                exec_price="open",
                cost_bps=0.0,
                slippage_rate=0.0,
                rebalance_shift="prev",
            ),
        )

    nav_dates = out["nav"]["dates"]
    nav = [float(x) for x in out["nav"]["series"]["STRAT"]]
    i_entry = nav_dates.index("2024-01-03")
    i_exit = nav_dates.index("2024-01-04")
    # Open-buy: execution day return is counted.
    assert float(nav[i_entry] / nav[i_entry - 1] - 1.0) == pytest.approx(
        0.10, rel=0.0, abs=1e-12
    )
    # Open-sell: execution day return is not counted.
    assert float(nav[i_exit] / nav[i_entry] - 1.0) == pytest.approx(
        0.0, rel=0.0, abs=1e-12
    )


def test_trend_benchmark_nav_uses_exec_price_basis(session_factory):
    sf = session_factory
    with sf() as db:
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(6)]
        # Close stays flat, open->close has +10% each day.
        # Benchmark in open mode should grow; close mode should stay flat.
        ohlc_none = [(100.0, 110.0)] * len(dates)
        close_qfq = [110.0] * len(dates)
        _seed_one(db, code="AAA", dates=dates, ohlc_none=ohlc_none, close_qfq=close_qfq)
        db.commit()

        out_open = compute_trend_backtest(
            db,
            TrendInputs(
                code="AAA",
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=2,
                exec_price="open",
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )
        out_close = compute_trend_backtest(
            db,
            TrendInputs(
                code="AAA",
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=2,
                exec_price="close",
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )

    bh_open = [float(x) for x in out_open["nav"]["series"]["BUY_HOLD"]]
    bh_close = [float(x) for x in out_close["nav"]["series"]["BUY_HOLD"]]
    assert bh_open[-1] > 1.0
    assert bh_close[-1] == pytest.approx(1.0, rel=0.0, abs=1e-12)
