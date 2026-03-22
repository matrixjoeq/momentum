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
                exec_price="close",
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


def test_trend_portfolio_fixed_ratio_skip_respects_weight_and_holding_caps(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-02-29", freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            _add_price(db, code="A1", day=d, close=100 + i * 1.2)
            _add_price(db, code="A2", day=d, close=100 + i * 1.0)
            _add_price(db, code="A3", day=d, close=100 + i * 0.8)
        db.commit()
        out = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1", "A2", "A3"],
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=6,
                position_sizing="fixed_ratio",
                fixed_pos_ratio=0.4,
                fixed_max_holdings=2,
                fixed_overcap_policy="skip",
                cost_bps=0.0,
            ),
        )
    ext = ((out.get("risk_controls") or {}).get("position_extension") or {})
    assert ext.get("position_sizing") == "fixed_ratio"
    assert int(ext.get("skipped_count") or 0) > 0
    assert int(ext.get("skipped_over_weight_count") or 0) > 0
    assert int(ext.get("skipped_over_count_count") or 0) > 0
    # Skip policy should keep effective holding count within cap.
    w = pd.DataFrame((out.get("weights") or {}).get("series") or {})
    if not w.empty:
        max_cnt = int((w > 1e-12).sum(axis=1).max())
        assert max_cnt <= 2


def test_trend_portfolio_fixed_ratio_extend_allows_weight_and_holding_overcaps(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-02-29", freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            _add_price(db, code="A1", day=d, close=100 + i * 1.2)
            _add_price(db, code="A2", day=d, close=100 + i * 1.0)
            _add_price(db, code="A3", day=d, close=100 + i * 0.8)
        db.commit()
        out = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1", "A2", "A3"],
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=6,
                position_sizing="fixed_ratio",
                fixed_pos_ratio=0.6,
                fixed_max_holdings=2,
                fixed_overcap_policy="extend",
                cost_bps=0.0,
            ),
        )
    ext = ((out.get("risk_controls") or {}).get("position_extension") or {})
    usage = ((out.get("risk_controls") or {}).get("position_usage") or {})
    assert int(ext.get("extension_count") or 0) > 0
    assert int(ext.get("extension_over_weight_count") or 0) > 0
    assert int(ext.get("extension_over_count_count") or 0) > 0
    # Extend policy allows >100% and >holding-cap states.
    assert int(usage.get("over_100pct_days") or 0) > 0
    w = pd.DataFrame((out.get("weights") or {}).get("series") or {})
    if not w.empty:
        max_cnt = int((w > 1e-12).sum(axis=1).max())
        assert max_cnt > 2


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


def test_trend_portfolio_group_enforce_respects_group_cap(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-04-30", freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            _add_price(db, code="A1", day=d, close=100 + i * 1.20)
            _add_price(db, code="A2", day=d, close=100 + i * 0.60)
            _add_price(db, code="B1", day=d, close=100 + i * 0.90)
        db.commit()
        out = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1", "A2", "B1"],
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=5,
                position_sizing="equal",
                group_enforce=True,
                group_pick_policy="highest_sharpe",
                group_max_holdings=1,
                asset_groups={"A1": "G1", "A2": "G1", "B1": "G2"},
            ),
        )

    params = ((out.get("meta") or {}).get("params") or {})
    assert params.get("group_enforce") is True
    assert params.get("group_pick_policy") == "highest_sharpe"
    assert int(params.get("group_max_holdings") or 0) == 1

    hs = out.get("holdings") or []
    assert hs
    for h in hs:
        gf = (h or {}).get("group_filter") or {}
        if not gf.get("enabled"):
            continue
        gp = gf.get("group_picks") or {}
        g1 = list(gp.get("G1") or [])
        assert len(g1) <= 1
        picks = list((h or {}).get("picks") or [])
        assert sum(1 for c in picks if c in {"A1", "A2"}) <= 1


def test_trend_portfolio_trade_statistics_have_samples_user_case_like(session_factory):
    """
    Use a user-case-like long-span setup (12 assets, 20111209~20260320) and verify
    trade statistics are not empty under cost+slippage when positions persist to the end.
    """
    sf = session_factory
    start = dt.date(2011, 12, 9)
    end = dt.date(2026, 3, 20)
    dates = [d.date() for d in pd.date_range(start, end, freq="B")]
    codes = [f"T{i:02d}" for i in range(1, 13)]
    with sf() as db:
        n = max(1, len(dates) - 1)
        for k, code in enumerate(codes):
            drift = 0.00012 + 0.000025 * float(k)
            for i, d in enumerate(dates):
                px = 100.0 * ((1.0 + drift) ** (float(i) / float(n) * float(n)))
                _add_price(db, code=code, day=d, close=float(px))
        db.commit()
        out = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=codes,
                start=start,
                end=end,
                dynamic_universe=True,
                strategy="ma_filter",
                position_sizing="equal",
                sma_window=20,
                exec_price="close",
                cost_bps=2.0,
                slippage_rate=0.001,
            ),
        )
    ts = (out.get("trade_statistics") or {})
    overall = (ts.get("overall") or {})
    by_code = (ts.get("by_code") or {})
    assert int(overall.get("total_trades") or 0) > 0
    assert any(int((v or {}).get("total_trades") or 0) > 0 for v in by_code.values())
