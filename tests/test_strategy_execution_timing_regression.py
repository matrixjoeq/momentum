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
        ("open", 8.0 / 9.0),
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
        ("open", 8.0 / 9.9),
        ("close", 7.0 / 13.0),
        ("oc2", 1.0 + 0.5 * ((8.0 / 9.9 - 1.0) + (7.0 / 13.0 - 1.0))),
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

