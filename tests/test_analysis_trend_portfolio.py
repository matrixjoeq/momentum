import datetime as dt

import pandas as pd

from etf_momentum.analysis.trend import TrendPortfolioInputs, compute_trend_portfolio_backtest
from tests.helpers.price_seed import add_price_all_adjustments


def _add_price(db, *, code: str, day: dt.date, close: float) -> None:
    add_price_all_adjustments(
        db,
        code=code,
        day=day,
        close=float(close),
        open_price=float(close),
        high=float(close),
        low=float(close),
    )


def test_trend_portfolio_all_active_candidates_and_outputs(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-03-31", freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            _add_price(db, code="A1", day=d, close=100 + i * 1.2)
            _add_price(db, code="A2", day=d, close=100 + i * 1.0)
            _add_price(db, code="B1", day=d, close=100 + i * 0.8)
        db.commit()

        out = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1", "A2", "B1"],
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=10,
            ),
        )
    assert out["meta"]["type"] == "trend_portfolio_backtest"
    assert "weights" in out and "holdings" in out
    assert out["meta"]["params"]["selection_mode"] == "all_active_candidates"
    if out["holdings"]:
        one = out["holdings"][0]
        assert "scores" in one


def test_trend_portfolio_invests_when_candidates_are_active(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-03-31", freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            _add_price(db, code="A1", day=d, close=100 + i * 1.0)
            _add_price(db, code="A2", day=d, close=100 + i * 0.9)
        db.commit()

        out = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1", "A2"],
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=10,
            ),
        )

    strat = out["nav"]["series"]["STRAT"]
    assert strat
    assert any(float(x) > 1.0 for x in strat)
    assert out["holdings"]
    assert any((h.get("picks") or []) for h in out["holdings"])


def test_trend_portfolio_ma_cross_supports_ema_type(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-03-31", freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            _add_price(db, code="A1", day=d, close=100 + i * 1.0)
            _add_price(db, code="B1", day=d, close=100 + i * 0.7)
        db.commit()
        out = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1", "B1"],
                start=dates[0],
                end=dates[-1],
                strategy="ma_cross",
                fast_window=8,
                slow_window=20,
                ma_type="ema",
            ),
        )
    assert out["meta"]["strategy"] == "ma_cross"
    assert ((out.get("meta") or {}).get("params") or {}).get("ma_type") == "ema"


def test_trend_portfolio_excludes_decision_day_return_for_all_strategies(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=8, freq="B")]
    # Decision-day jump on d3 for both assets. If wrongly counted on decision day, NAV would jump sharply there.
    px_a = [100.0, 100.0, 100.0, 200.0, 200.0, 220.0, 220.0, 220.0]
    px_b = [80.0, 80.0, 80.0, 160.0, 160.0, 176.0, 176.0, 176.0]
    strategies = [
        ("ma_filter", {"sma_window": 2}),
        ("ma_cross", {"fast_window": 2, "slow_window": 3, "ma_type": "sma"}),
        ("donchian", {"donchian_entry": 2, "donchian_exit": 2}),
        ("tsmom", {"mom_lookback": 2}),
        ("linreg_slope", {"sma_window": 3}),
        ("bias", {"bias_ma_window": 2, "bias_entry": 1.0, "bias_hot": 50.0, "bias_cold": -10.0, "bias_pos_mode": "binary"}),
        ("macd_cross", {"macd_fast": 2, "macd_slow": 3, "macd_signal": 2}),
        ("macd_zero_filter", {"macd_fast": 2, "macd_slow": 3, "macd_signal": 2}),
        ("macd_v", {"macd_fast": 2, "macd_slow": 3, "macd_signal": 2, "macd_v_atr_window": 2, "macd_v_scale": 100.0}),
        ("hybrid_trend", {"fast_window": 2, "slow_window": 3, "donchian_entry": 2, "donchian_exit": 2, "mom_lookback": 2, "macd_fast": 2, "macd_slow": 3, "macd_signal": 2, "sma_window": 3, "hybrid_entry_n": 1, "hybrid_exit_m": 1}),
    ]

    with sf() as db:
        for d, a, b in zip(dates, px_a, px_b):
            _add_price(db, code="A1", day=d, close=a)
            _add_price(db, code="B1", day=d, close=b)
        db.commit()

        for strat, params in strategies:
            out = compute_trend_portfolio_backtest(
                db,
                TrendPortfolioInputs(
                    codes=["A1", "B1"],
                    start=dates[0],
                    end=dates[-1],
                    strategy=strat,
                    cost_bps=0.0,
                    position_sizing="equal",
                    **params,
                ),
            )
            nav = [float(x) for x in out["nav"]["series"]["STRAT"]]
            assert len(nav) >= 4
            assert nav[3] <= 1.0000001, f"{strat} appears to include portfolio decision-day return"

