import datetime as dt

import pytest

from etf_momentum.db.models import EtfPrice
from etf_momentum.strategy.rotation import RotationInputs, backtest_rotation


def _seed_prices(db, *, code: str, dates: list[dt.date], closes: list[float]) -> None:
    assert len(dates) == len(closes)
    for d, px in zip(dates, closes):
        # For tests that exercise technical-analysis filters we also provide qfq.
        for adj in ("hfq", "qfq", "none"):
            c = float(px)
            # Provide synthetic OHLC so that indicators like ADX (high/low/close) can be computed.
            hi = c * 1.01
            lo = c * 0.99
            db.add(
                EtfPrice(
                    code=code,
                    trade_date=d,
                    open=c,
                    high=hi,
                    low=lo,
                    close=c,
                    source="eastmoney",
                    adjust=adj,
                )
            )


def test_trend_filter_self_ma_blocks_risk_and_goes_cash(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["AAA", "BBB"]
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(80)]
        # both down -> universe average below SMA => trend block
        a = [100 - i * 0.3 for i in range(len(dates))]
        b = [100 - i * 0.2 for i in range(len(dates))]
        _seed_prices(db, code="AAA", dates=dates, closes=a)
        _seed_prices(db, code="BBB", dates=dates, closes=b)
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=codes,
                start=start,
                end=dates[-1],
                rebalance="monthly",
                top_k=1,
                lookback_days=10,
                trend_filter=True,
                trend_sma_window=10,
                cost_bps=0.0,
            ),
        )

    assert out["holdings"]
    assert any((p.get("mode") == "cash") for p in out["holdings"])
    assert out["nav"]["series"]["ROTATION"][-1] == pytest.approx(1.0)


def test_rsi_filter_blocks_overbought_candidate(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["AAA", "BBB"]
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(90)]
        # AAA strong up (overbought), BBB flat; momentum would pick AAA but RSI filter blocks it.
        a = [100 + i * 1.0 for i in range(len(dates))]
        b = [100.0 for _ in range(len(dates))]
        _seed_prices(db, code="AAA", dates=dates, closes=a)
        _seed_prices(db, code="BBB", dates=dates, closes=b)
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=codes,
                start=start,
                end=dates[-1],
                rebalance="monthly",
                top_k=1,
                lookback_days=10,
                rsi_filter=True,
                rsi_window=14,
                rsi_overbought=55.0,  # low threshold to make the test deterministic
                rsi_block_overbought=True,
                cost_bps=0.0,
            ),
        )

    assert out["holdings"]
    assert any((p.get("mode") == "cash") for p in out["holdings"])


def test_vol_monitor_scales_exposure_down(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["AAA", "BBB"]
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(120)]
        # AAA: alternating large swings -> high vol; BBB flat.
        a = [100.0]
        for i in range(1, len(dates)):
            # Keep it volatile but with a slight upward drift so momentum prefers AAA over BBB.
            a.append(a[-1] * (1.07 if i % 2 == 0 else 0.96))
        b = [100.0 for _ in range(len(dates))]
        _seed_prices(db, code="AAA", dates=dates, closes=a)
        _seed_prices(db, code="BBB", dates=dates, closes=b)
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=codes,
                start=start,
                end=dates[-1],
                rebalance="monthly",
                top_k=1,
                lookback_days=10,
                vol_monitor=True,
                vol_window=10,
                vol_target_ann=0.05,  # very low target -> scale down
                vol_max_ann=10.0,  # don't hard-stop
                cost_bps=0.0,
            ),
        )

    # At least one period should have exposure < 1 when holding risk assets.
    exposures = [float(p.get("exposure")) for p in out["holdings"] if p.get("mode") == "risk_on"]
    assert exposures, "expected some risk_on periods"
    assert any(x < 0.99 for x in exposures)


def test_momentum_signal_uses_qfq_not_hfq(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["AAA", "BBB"]
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(90)]

        # qfq: AAA up / BBB flat -> qfq momentum should prefer AAA.
        qfq_a = [100.0 + i * 0.6 for i in range(len(dates))]
        qfq_b = [100.0 for _ in range(len(dates))]
        # hfq intentionally opposite to verify ranking no longer depends on hfq.
        hfq_a = [100.0 - i * 0.4 for i in range(len(dates))]
        hfq_b = [100.0 + i * 0.3 for i in range(len(dates))]
        # execution basis can stay stable.
        none_a = [100.0 + i * 0.2 for i in range(len(dates))]
        none_b = [100.0 + i * 0.1 for i in range(len(dates))]

        for d, qa, qb, ha, hb, na, nb in zip(dates, qfq_a, qfq_b, hfq_a, hfq_b, none_a, none_b):
            db.add(EtfPrice(code="AAA", trade_date=d, open=qa, high=qa * 1.01, low=qa * 0.99, close=qa, source="eastmoney", adjust="qfq"))
            db.add(EtfPrice(code="BBB", trade_date=d, open=qb, high=qb * 1.01, low=qb * 0.99, close=qb, source="eastmoney", adjust="qfq"))
            db.add(EtfPrice(code="AAA", trade_date=d, open=ha, high=ha * 1.01, low=ha * 0.99, close=ha, source="eastmoney", adjust="hfq"))
            db.add(EtfPrice(code="BBB", trade_date=d, open=hb, high=hb * 1.01, low=hb * 0.99, close=hb, source="eastmoney", adjust="hfq"))
            db.add(EtfPrice(code="AAA", trade_date=d, open=na, high=na * 1.01, low=na * 0.99, close=na, source="eastmoney", adjust="none"))
            db.add(EtfPrice(code="BBB", trade_date=d, open=nb, high=nb * 1.01, low=nb * 0.99, close=nb, source="eastmoney", adjust="none"))
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=codes,
                start=start,
                end=dates[-1],
                rebalance="monthly",
                top_k=1,
                lookback_days=20,
                cost_bps=0.0,
            ),
        )

    picks = [p.get("picks") for p in out["holdings"] if p.get("mode") == "risk_on"]
    assert picks, "expected some risk_on periods"
    assert any(ps and ps[0] == "AAA" for ps in picks)


def test_chop_filter_adx_excludes_low_trend_asset(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["AAA", "BBB", "CCC"]
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(140)]
        # AAA: choppy but slightly up overall (so momentum can rank it well)
        a = [100.0]
        for i in range(1, len(dates)):
            a.append(a[-1] * (1.01 if i % 2 == 0 else 0.99))
        # BBB: steadier uptrend
        b = [100.0 + i * 0.15 for i in range(len(dates))]
        # CCC: another steady uptrend to keep candidate pool sufficient when AAA is filtered.
        c = [100.0 + i * 0.12 for i in range(len(dates))]
        _seed_prices(db, code="AAA", dates=dates, closes=a)
        _seed_prices(db, code="BBB", dates=dates, closes=b)
        _seed_prices(db, code="CCC", dates=dates, closes=c)
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=codes,
                start=start,
                end=dates[-1],
                rebalance="monthly",
                top_k=1,
                lookback_days=20,
                cost_bps=0.0,
                chop_filter=True,
                chop_mode="adx",
                chop_adx_window=14,
                chop_adx_threshold=15.0,
            ),
        )

    # When ADX filter is active, at least one period should exclude the choppy asset.
    picks = [p.get("picks") for p in out["holdings"] if p.get("mode") == "risk_on"]
    assert picks, "expected some risk_on periods"
    assert any((ps is not None) and ("AAA" not in (ps or [])) for ps in picks)


def test_trend_filters_do_not_use_future_data(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["AAA", "BBB"]
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(140)]

        # Baseline worlds: similar upward drifts with mild differences.
        a = [100.0 + i * 0.25 for i in range(len(dates))]
        b = [100.0 + i * 0.20 for i in range(len(dates))]

        for d, pa, pb in zip(dates, a, b):
            for adj in ("hfq", "qfq", "none"):
                db.add(EtfPrice(code="AAA", trade_date=d, open=pa, high=pa * 1.01, low=pa * 0.99, close=pa, source="eastmoney", adjust=adj))
                db.add(EtfPrice(code="BBB", trade_date=d, open=pb, high=pb * 1.01, low=pb * 0.99, close=pb, source="eastmoney", adjust=adj))
        db.commit()

        base = backtest_rotation(
            db,
            RotationInputs(
                codes=codes,
                start=start,
                end=dates[-1],
                rebalance="weekly",
                top_k=1,
                lookback_days=20,
                trend_filter=True,
                trend_exit_filter=True,
                trend_sma_window=10,
                trend_ma_type="ema",
                cost_bps=0.0,
            ),
        )

    with sf() as db2:
        codes2 = ["AAA_FUT", "BBB_FUT"]
        cutoff = 80
        for i, d in enumerate(dates):
            pa = float(a[i])
            pb = float(b[i])
            # Perturb only the distant future region.
            if i >= cutoff + 10:
                pa = pa * (1.0 + 0.02 * float(i - cutoff))
                pb = pb * (1.0 - 0.01 * float(i - cutoff))
            for adj in ("hfq", "qfq", "none"):
                db2.add(
                    EtfPrice(
                        code="AAA_FUT",
                        trade_date=d,
                        open=pa,
                        high=pa * 1.01,
                        low=pa * 0.99,
                        close=pa,
                        source="eastmoney",
                        adjust=adj,
                    )
                )
                db2.add(
                    EtfPrice(
                        code="BBB_FUT",
                        trade_date=d,
                        open=pb,
                        high=pb * 1.01,
                        low=pb * 0.99,
                        close=pb,
                        source="eastmoney",
                        adjust=adj,
                    )
                )
        db2.commit()

        fut = backtest_rotation(
            db2,
            RotationInputs(
                codes=codes2,
                start=start,
                end=dates[-1],
                rebalance="weekly",
                top_k=1,
                lookback_days=20,
                trend_filter=True,
                trend_exit_filter=True,
                trend_sma_window=10,
                trend_ma_type="ema",
                cost_bps=0.0,
            ),
        )

    nav_base = list(base["nav"]["series"]["ROTATION"])
    nav_fut = list(fut["nav"]["series"]["ROTATION"])
    # Historical NAV before the perturbation window must be identical.
    assert nav_base[: cutoff + 8] == pytest.approx(nav_fut[: cutoff + 8], rel=0.0, abs=1e-12)
