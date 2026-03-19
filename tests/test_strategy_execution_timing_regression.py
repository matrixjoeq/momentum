import datetime as dt

import pytest

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
            ),
        )

    final_nav = float(out["nav"]["series"]["ROTATION"][-1])
    assert final_nav == pytest.approx(expected_nav, rel=0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("exec_price", "expected_nav"),
    [
        ("open", 1.148989898989899),
        ("close", 7.0 / 13.0),
        ("oc2", 0.8679341491841491),
    ],
)
def test_trend_exec_price_uses_execution_day_forward_return(
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
            ),
        )

    final_nav = float(out["nav"]["series"]["STRAT"][-1])
    assert final_nav == pytest.approx(expected_nav, rel=0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("exec_price", "expected_d8_ret"),
    [
        ("open", 0.20),
        ("close", 0.0),
    ],
)
def test_trend_atr_stop_reentry_timing_no_lookahead(session_factory, exec_price: str, expected_d8_ret: float):
    sf = session_factory
    with sf() as db:
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(10)]
        # qfq close for signal/ATR:
        # - steady uptrend to let trailing stop move upward
        # - d6 dip keeps base signal long but breaks trailing stop
        # - d7 base stays long to allow stop_reentry decision
        # - d8/d9 used to distinguish open vs close execution-day return
        close_qfq = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 106.5, 112.0, 120.0, 120.0]
        # none/hfq execution OHLC:
        # - mostly open=close
        # - d8 has large intraday open->close move (20%)
        #   * open mode should realize it on d8
        #   * close mode should not realize it on d8
        ohlc_none = [(p, p) for p in close_qfq]
        ohlc_none[8] = (100.0, 120.0)
        _seed_one(db, code="AAA", dates=dates, ohlc_none=ohlc_none, close_qfq=close_qfq)
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
            ),
        )

    d = out["nav"]["dates"]
    nav = out["nav"]["series"]["STRAT"]
    eff = out["signals"]["position_effective"]
    atr = out["risk_controls"]["atr_stop"]
    trace = out["next_plan"]["trace"]["atr_stop"]["trace_last_rows"]

    i6 = d.index(dates[6].isoformat())
    i7 = d.index(dates[7].isoformat())
    i8 = d.index(dates[8].isoformat())

    # Stop is decided on d6 and executed on d7 (effective weight at d7 must be 0).
    assert int(atr["trigger_count"]) >= 1
    assert atr["trigger_dates"] and atr["trigger_dates"][-1] == dates[6].isoformat()
    assert float(eff[i7]) == pytest.approx(0.0, abs=1e-12)
    # d7 is stop execution day, so it must have no strategy return.
    assert float(nav[i7]) == pytest.approx(float(nav[i6]), rel=0.0, abs=1e-12)

    # Re-entry decision can happen at d7; position becomes effective at d8.
    assert float(eff[i8]) == pytest.approx(1.0, abs=1e-12)
    has_stop_reentry = any(
        (str(r.get("date")) == dates[7].isoformat())
        and (str(r.get("event_type")) == "entry")
        and (str(r.get("event_reason")) == "stop_reentry")
        for r in trace
    )
    assert has_stop_reentry

    # d8 return behavior depends on execution price:
    # - open: has same-day open->close return
    # - close: no same-day return
    d8_ret = float(nav[i8] / nav[i7] - 1.0)
    assert d8_ret == pytest.approx(expected_d8_ret, rel=0.0, abs=1e-12)

