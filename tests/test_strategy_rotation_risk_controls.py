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


def test_trend_filter_universe_blocks_risk_and_goes_cash(session_factory):
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
                risk_off=False,  # key: trend filter alone should be able to block entry
                trend_filter=True,
                trend_mode="universe",
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
                risk_off=False,
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
                risk_off=False,
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


def test_score_method_return_over_vol_prefers_stable_winner(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["AAA", "BBB"]
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(120)]
        # AAA: big swings with high average return (but unstable)
        a = [100.0]
        for i in range(1, len(dates)):
            a.append(a[-1] * (1.10 if i % 2 == 0 else 0.92))
        # BBB: gentle steady up (more stable)
        b = [100.0 + i * 0.15 for i in range(len(dates))]
        _seed_prices(db, code="AAA", dates=dates, closes=a)
        _seed_prices(db, code="BBB", dates=dates, closes=b)
        db.commit()

        out_raw = backtest_rotation(
            db,
            RotationInputs(
                codes=codes,
                start=start,
                end=dates[-1],
                rebalance="monthly",
                top_k=1,
                lookback_days=20,
                score_method="raw_mom",
                cost_bps=0.0,
            ),
        )
        out_rav = backtest_rotation(
            db,
            RotationInputs(
                codes=codes,
                start=start,
                end=dates[-1],
                rebalance="monthly",
                top_k=1,
                lookback_days=20,
                score_method="return_over_vol",
                cost_bps=0.0,
            ),
        )

    # Ensure the two methods produce different choices at least once.
    raw_picks = [p.get("picks") for p in out_raw["holdings"]]
    rav_picks = [p.get("picks") for p in out_rav["holdings"]]
    assert raw_picks != rav_picks


def test_chop_filter_adx_excludes_low_trend_asset(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["AAA", "BBB"]
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(140)]
        # AAA: choppy but slightly up overall (so momentum can rank it well)
        a = [100.0]
        for i in range(1, len(dates)):
            a.append(a[-1] * (1.01 if i % 2 == 0 else 0.99))
        # BBB: steadier uptrend
        b = [100.0 + i * 0.15 for i in range(len(dates))]
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
                top_k=2,
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


