import datetime as dt

import numpy as np
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
    assert "avg_daily_turnover" in out["metrics"]["strategy"]
    assert "avg_annual_turnover" in out["metrics"]["strategy"]
    assert "avg_daily_trade_count" in out["metrics"]["strategy"]
    assert "avg_annual_trade_count" in out["metrics"]["strategy"]
    assert (out.get("market_regime") or {}).get("enabled") is True
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


def test_trend_portfolio_er_entry_filter_blocks_choppy_entries(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=90, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            px1 = 100.0 + ((-1.0) ** i) * 0.8 + i * 0.01
            px2 = 90.0 + ((-1.0) ** (i + 1)) * 0.7 + i * 0.01
            _add_price(db, code="A1", day=d, close=px1)
            _add_price(db, code="A2", day=d, close=px2)
        db.commit()

        out_no_filter = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1", "A2"],
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=2,
                er_filter=False,
                cost_bps=0.0,
            ),
        )
        out_with_filter = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1", "A2"],
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=2,
                er_filter=True,
                er_window=10,
                er_threshold=0.8,
                cost_bps=0.0,
            ),
        )

    w_no = pd.DataFrame((out_no_filter.get("weights") or {}).get("series") or {})
    w_yes = pd.DataFrame((out_with_filter.get("weights") or {}).get("series") or {})
    assert not w_no.empty
    assert any(float(v) > 0.0 for v in w_no.to_numpy().ravel())
    assert all(float(v) == 0.0 for v in w_yes.to_numpy().ravel())
    params = ((out_with_filter.get("meta") or {}).get("params") or {})
    assert params.get("er_filter") is True
    assert int(params.get("er_window") or 0) == 10


def test_trend_portfolio_impulse_entry_filter_blocks_entries(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=100, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            _add_price(db, code="I1", day=d, close=100.0 + i * 0.6)
            _add_price(db, code="I2", day=d, close=120.0 + i * 0.5)
        db.commit()
        out = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["I1", "I2"],
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=2,
                impulse_entry_filter=True,
                impulse_allow_bull=False,
                impulse_allow_bear=False,
                impulse_allow_neutral=False,
                position_sizing="equal",
                cost_bps=0.0,
            ),
        )
    w = pd.DataFrame(((out.get("weights") or {}).get("series") or {}))
    assert not w.empty
    assert all(float(v) == 0.0 for v in w.to_numpy().ravel())
    params = ((out.get("meta") or {}).get("params") or {})
    assert params.get("impulse_entry_filter") is True
    ts = (out.get("trade_statistics") or {})
    overall = (ts.get("overall") or {})
    blocked = int(overall.get("impulse_filter_blocked_entry_count") or 0)
    blocked_split = (
        int(overall.get("impulse_filter_blocked_entry_count_bull") or 0)
        + int(overall.get("impulse_filter_blocked_entry_count_bear") or 0)
        + int(overall.get("impulse_filter_blocked_entry_count_neutral") or 0)
    )
    assert blocked > 0
    assert blocked_split == blocked
    by_code = (ts.get("by_code") or {})
    assert int(((by_code.get("I1") or {}).get("impulse_filter_blocked_entry_count") or 0) >= 0)
    assert int(((by_code.get("I2") or {}).get("impulse_filter_blocked_entry_count") or 0) >= 0)


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


def test_trend_portfolio_random_entry_seed_controls_reproducibility(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=80, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            _add_price(db, code="A1", day=d, close=100 + i * 0.2)
            _add_price(db, code="A2", day=d, close=90 + i * 0.15)
        db.commit()
        out_a = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1", "A2"],
                start=dates[0],
                end=dates[-1],
                strategy="random_entry",
                random_hold_days=20,
                random_seed=42,
                cost_bps=0.0,
            ),
        )
        out_b = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1", "A2"],
                start=dates[0],
                end=dates[-1],
                strategy="random_entry",
                random_hold_days=20,
                random_seed=42,
                cost_bps=0.0,
            ),
        )
        out_c = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1", "A2"],
                start=dates[0],
                end=dates[-1],
                strategy="random_entry",
                random_hold_days=20,
                random_seed=43,
                cost_bps=0.0,
            ),
        )
    wa = (out_a.get("weights") or {}).get("series") or {}
    wb = (out_b.get("weights") or {}).get("series") or {}
    wc = (out_c.get("weights") or {}).get("series") or {}
    assert wa == wb
    assert wa != wc
    params = ((out_a.get("meta") or {}).get("params") or {})
    assert int(params.get("random_hold_days") or 0) == 20
    assert int(params.get("random_seed") or -1) == 42


def test_trend_portfolio_random_entry_allows_system_random_seed(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=40, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            _add_price(db, code="A1", day=d, close=100 + i * 0.2)
            _add_price(db, code="A2", day=d, close=90 + i * 0.15)
        db.commit()
        out = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1", "A2"],
                start=dates[0],
                end=dates[-1],
                strategy="random_entry",
                random_hold_days=20,
                random_seed=None,
                cost_bps=0.0,
            ),
        )
    params = ((out.get("meta") or {}).get("params") or {})
    assert params.get("random_seed") is None


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
    User-case-like trade-stats check (shorter span than prod debug) with cost+slippage;
    staggered drifts keep ranked assets so MA trend portfolio produces closable trades.
    """
    sf = session_factory
    start = dt.date(2022, 1, 4)
    end = dt.date(2023, 10, 31)
    dates = [d.date() for d in pd.date_range(start, end, freq="B")]
    codes = [f"T{i:02d}" for i in range(1, 9)]
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
    rs = (out.get("r_statistics") or {})
    assert int(((rs.get("overall") or {}).get("trade_count") or 0) > 0)
    assert "recent_100" in rs
    assert int(((rs.get("recent_100") or {}).get("effective_count") or 0) > 0)
    assert "sqn" in ((rs.get("overall") or {}))
    first_trade = ((ts.get("trades") or [None])[0] or {})
    assert "initial_r_amount" in first_trade
    assert "r_multiple" in first_trade


def test_trend_portfolio_risk_budget_position_sizing(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=90, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            _add_price(db, code="A1", day=d, close=100 + i * 0.9)
            _add_price(db, code="A2", day=d, close=80 + i * 0.7)
        db.commit()
        out = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1", "A2"],
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=8,
                position_sizing="risk_budget",
                risk_budget_atr_window=20,
                risk_budget_pct=0.01,
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )
    params = ((out.get("meta") or {}).get("params") or {})
    assert str(params.get("position_sizing") or "") == "risk_budget"
    w = pd.DataFrame((out.get("weights") or {}).get("series") or {})
    if not w.empty:
        expo = w.sum(axis=1)
        assert float(expo.max()) <= 1.0000001
        assert float(expo.max()) > 0.0


def test_trend_portfolio_risk_budget_pct_upper_bound(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=90, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            _add_price(db, code="A1", day=d, close=100 + i * 0.9)
            _add_price(db, code="A2", day=d, close=80 + i * 0.7)
        db.commit()
        try:
            compute_trend_portfolio_backtest(
                db,
                TrendPortfolioInputs(
                    codes=["A1", "A2"],
                    start=dates[0],
                    end=dates[-1],
                    strategy="ma_filter",
                    sma_window=8,
                    position_sizing="risk_budget",
                    risk_budget_atr_window=20,
                    risk_budget_pct=0.03,
                    cost_bps=0.0,
                    slippage_rate=0.0,
                ),
            )
            assert False, "expected ValueError for risk_budget_pct > 0.02"
        except ValueError as exc:
            assert "risk_budget_pct must be in [0.001, 0.02]" in str(exc)


def test_trend_portfolio_monthly_risk_budget_blocks_entries(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=100, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            p1 = 100.0 + 2.2 * np.sin(i / 2.0) + 0.08 * i
            p2 = 90.0 + 1.8 * np.sin(i / 2.3 + 0.7) + 0.06 * i
            _add_price(db, code="A1", day=d, close=float(p1))
            _add_price(db, code="A2", day=d, close=float(p2))
        db.commit()
        out = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1", "A2"],
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                ma_type="ema",
                sma_window=5,
                position_sizing="equal",
                monthly_risk_budget_enabled=True,
                monthly_risk_budget_pct=0.01,
                monthly_risk_budget_include_new_trade_risk=True,
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )
    ts = (out.get("trade_statistics") or {})
    overall = (ts.get("overall") or {})
    by_code = (ts.get("by_code") or {})
    assert int(overall.get("monthly_risk_budget_blocked_entry_count") or 0) > 0
    assert int(((by_code.get("A1") or {}).get("monthly_risk_budget_blocked_entry_count") or 0)) >= 0
    assert int(((by_code.get("A2") or {}).get("monthly_risk_budget_blocked_entry_count") or 0)) >= 0
    m = ((out.get("metrics") or {}).get("strategy") or {})
    assert int(m.get("monthly_risk_budget_blocked_entry_count") or 0) > 0


def test_trend_portfolio_bias_has_base_exit_hot_cold(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=120, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            if i < 60:
                px = 100.0 + i * 0.8
            elif i < 90:
                px = 148.0 - (i - 60) * 0.3
            else:
                px = 139.0 + (i - 90) * 0.9
            _add_price(db, code="A1", day=d, close=px)
        db.commit()
        out = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1"],
                start=dates[0],
                end=dates[-1],
                strategy="bias",
                bias_ma_window=8,
                bias_entry=2.0,
                bias_hot=8.0,
                bias_cold=-2.0,
                bias_pos_mode="binary",
                position_sizing="equal",
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )
    w = pd.DataFrame((out.get("weights_decision") or {}).get("series") or {})
    assert not w.empty
    ws = w["A1"].astype(float)
    assert any(v > 0 for v in ws.tolist())
    assert any(v <= 0 for v in ws.tolist())


def test_trend_portfolio_exposes_r_take_profit_controls(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=120, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            if i < 70:
                a1 = 100.0 + i * 0.7
                a2 = 90.0 + i * 0.5
            else:
                a1 = 149.0 - (i - 70) * 0.4
                a2 = 125.0 - (i - 70) * 0.2
            _add_price(db, code="A1", day=d, close=a1)
            _add_price(db, code="A2", day=d, close=a2)
        db.commit()
        out = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1", "A2"],
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=10,
                atr_stop_mode="none",
                r_take_profit_enabled=True,
                r_take_profit_reentry_mode="reenter",
                r_take_profit_tiers=[
                    {"r_multiple": 2.0, "retrace_ratio": 0.5},
                    {"r_multiple": 3.0, "retrace_ratio": 0.3},
                ],
                position_sizing="equal",
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )
    rc = (out.get("risk_controls") or {})
    rtp = (rc.get("r_take_profit") or {})
    assert bool(rtp.get("enabled")) is True
    assert str(rtp.get("initial_r_mode") or "") == "virtual_atr_fallback"
    assert isinstance((rtp.get("tier_trigger_counts") or {}), dict)
    params = (((out.get("meta") or {}).get("params") or {}))
    assert bool(params.get("r_take_profit_enabled")) is True
    metrics = ((out.get("metrics") or {}).get("strategy") or {})
    assert "r_take_profit_trigger_count" in metrics
    assert isinstance((metrics.get("r_take_profit_tier_trigger_counts") or {}), dict)
    ts = (out.get("trade_statistics") or {})
    overall = (ts.get("overall") or {})
    by_code = (ts.get("by_code") or {})
    assert "atr_stop_trigger_count" in overall
    assert "r_take_profit_trigger_count" in overall
    assert isinstance((overall.get("r_take_profit_tier_trigger_counts") or {}), dict)
    for c in ["A1", "A2"]:
        one = (by_code.get(c) or {})
        assert "atr_stop_trigger_count" in one
        assert "r_take_profit_trigger_count" in one


def test_trend_portfolio_kama_std_band_reduces_trades(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=160, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            p1 = 100.0 + 0.10 * i + 4.5 * np.sin(i / 3.0)
            p2 = 90.0 + 0.08 * i + 3.5 * np.sin(i / 3.7 + 0.6)
            _add_price(db, code="A1", day=d, close=float(p1))
            _add_price(db, code="A2", day=d, close=float(p2))
        db.commit()
        out_lo = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1", "A2"],
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                ma_type="kama",
                sma_window=20,
                kama_std_window=20,
                kama_std_coef=0.0,
                position_sizing="equal",
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )
        out_hi = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1", "A2"],
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                ma_type="kama",
                sma_window=20,
                kama_std_window=20,
                kama_std_coef=3.0,
                position_sizing="equal",
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )
    lo_trades = int((((out_lo.get("trade_statistics") or {}).get("overall") or {}).get("total_trades") or 0))
    hi_trades = int((((out_hi.get("trade_statistics") or {}).get("overall") or {}).get("total_trades") or 0))
    assert hi_trades <= lo_trades

