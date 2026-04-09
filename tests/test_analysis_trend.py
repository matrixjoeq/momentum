import datetime as dt

import numpy as np
import pandas as pd

from etf_momentum.analysis.trend import (
    TrendInputs,
    _apply_atr_stop,
    _apply_r_multiple_take_profit,
    _pos_from_random_entry_hold,
    compute_trend_backtest,
)
from etf_momentum.db.models import EtfPrice
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


def test_trailing_stop_latest_atr_moves_up_on_volatility_drop() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    base_pos = pd.Series([1.0] * len(idx), index=idx, dtype=float)
    close = pd.Series([100.0] * len(idx), index=idx, dtype=float)
    high = pd.Series([120.0, 120.0, 110.0, 105.0, 103.0, 102.0], index=idx, dtype=float)
    low = pd.Series([80.0, 80.0, 90.0, 95.0, 97.0, 98.0], index=idx, dtype=float)

    out_pos, stats = _apply_atr_stop(
        base_pos,
        close=close,
        high=high,
        low=low,
        mode="trailing",
        atr_basis="latest",
        reentry_mode="reenter",
        atr_window=2,
        n_mult=2.0,
        m_step=0.5,
    )

    trace = list(stats.get("trace_last_rows") or [])
    entry_rows = [r for r in trace if str(r.get("event_type")) == "entry"]
    assert entry_rows
    entry_stop = float(entry_rows[0]["stop_after"])
    latest_stop = float(stats["latest_stop_price"])

    # Flat close + shrinking ATR should tighten stop upward for latest-ATR trailing mode.
    assert latest_stop > entry_stop
    # Trailing stop must remain monotonic in favorable direction (long: only move up).
    assert latest_stop <= float(close.iloc[-1])
    hold_rows = [r for r in trace if str(r.get("event_type")) == "hold"]
    assert hold_rows
    assert any(r.get("stop_candidate") is not None for r in hold_rows)
    assert float(out_pos.iloc[-1]) == 1.0


def test_trailing_stop_latest_atr_can_rise_when_price_drops() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    base_pos = pd.Series([1.0] * len(idx), index=idx, dtype=float)
    close = pd.Series([100.0, 100.0, 99.0, 99.0, 99.0], index=idx, dtype=float)
    # ATR drops sharply after entry although close is lower, so candidate stop still rises.
    high = pd.Series([120.0, 120.0, 105.0, 103.0, 102.0], index=idx, dtype=float)
    low = pd.Series([80.0, 80.0, 93.0, 95.0, 96.0], index=idx, dtype=float)

    out_pos, stats = _apply_atr_stop(
        base_pos,
        close=close,
        high=high,
        low=low,
        mode="trailing",
        atr_basis="latest",
        reentry_mode="reenter",
        atr_window=2,
        n_mult=2.0,
        m_step=0.5,
    )

    trace = list(stats.get("trace_last_rows") or [])
    entry_rows = [r for r in trace if str(r.get("event_type")) == "entry"]
    assert entry_rows
    entry_stop = float(entry_rows[0]["stop_after"])
    latest_stop = float(stats["latest_stop_price"])

    assert float(close.iloc[2]) < float(close.iloc[1])
    assert latest_stop > entry_stop
    assert float(out_pos.iloc[-1]) == 1.0


def test_trend_single_risk_budget_sizing_applies_params(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=40, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            px = 100.0 + i * 0.6
            _add_price(db, code=code, day=d, close=px)
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=5,
                position_sizing="risk_budget",
                risk_budget_atr_window=2,
                risk_budget_pct=0.005,
                cost_bps=0.0,
            ),
        )

    params = ((out.get("meta") or {}).get("params") or {})
    assert str(params.get("position_sizing") or "") == "risk_budget"
    assert int(params.get("risk_budget_atr_window") or 0) == 2
    assert float(params.get("risk_budget_pct") or 0.0) == 0.005
    eff = [float(x) for x in ((out.get("signals") or {}).get("position_effective") or [])]
    assert eff
    assert any((x > 0.0) and (x < 1.0) for x in eff)


def test_trend_single_er_entry_filter_blocks_choppy_entries(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=80, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            px = 100.0 + ((-1.0) ** i) * 0.8 + i * 0.01
            _add_price(db, code=code, day=d, close=px)
        db.commit()
        out_no_filter = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=2,
                er_filter=False,
                cost_bps=0.0,
            ),
        )
        out_with_filter = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
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

    pos_no_filter = [float(x) for x in out_no_filter["signals"]["position"]]
    pos_with_filter = [float(x) for x in out_with_filter["signals"]["position"]]
    assert any(x > 0.0 for x in pos_no_filter)
    assert all(x == 0.0 for x in pos_with_filter)
    params = ((out_with_filter.get("meta") or {}).get("params") or {})
    assert params.get("er_filter") is True
    assert int(params.get("er_window") or 0) == 10
    ts = (out_with_filter.get("trade_statistics") or {})
    overall = (ts.get("overall") or {})
    by_code = (ts.get("by_code") or {}).get(code, {})
    assert int(overall.get("er_filter_blocked_entry_count") or 0) > 0
    assert int(overall.get("er_filter_attempted_entry_count") or 0) >= int(overall.get("er_filter_blocked_entry_count") or 0)
    assert int(overall.get("er_filter_allowed_entry_count") or 0) >= 0
    assert int(by_code.get("er_filter_blocked_entry_count") or 0) > 0


def test_trend_single_er_exit_filter_triggers_on_high_er(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=120, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            px = 100.0 + i * 0.6
            _add_price(db, code=code, day=d, close=px)
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=2,
                er_exit_filter=True,
                er_exit_window=10,
                er_exit_threshold=0.8,
                cost_bps=0.0,
            ),
        )

    params = ((out.get("meta") or {}).get("params") or {})
    assert params.get("er_exit_filter") is True
    assert int(params.get("er_exit_window") or 0) == 10
    ts = (out.get("trade_statistics") or {})
    overall = (ts.get("overall") or {})
    by_code = (ts.get("by_code") or {}).get(code, {})
    assert int(overall.get("er_exit_filter_trigger_count") or 0) > 0
    assert int(by_code.get("er_exit_filter_trigger_count") or 0) > 0


def test_trend_ma_filter_smoke(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-06-30", freq="B")]
    with sf() as db:
        # up then down to create at least one regime change
        for i, d in enumerate(dates):
            px = 100.0 + (i * 0.5 if i < 60 else (60 * 0.5) - (i - 60) * 0.8)
            _add_price(db, code=code, day=d, close=px)
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(code=code, start=dates[0], end=dates[-1], strategy="ma_filter", sma_window=20, cost_bps=0.0),
        )
    assert out["meta"]["type"] == "trend_backtest"
    assert out["meta"]["code"] == code
    assert "nav" in out and "series" in out["nav"]
    s = out["nav"]["series"]
    assert len(out["nav"]["dates"]) == len(s["STRAT"]) == len(s["BUY_HOLD"]) == len(s["EXCESS"])
    assert "event_study" in out
    assert "avg_daily_turnover" in out["metrics"]["strategy"]
    assert "avg_annual_turnover" in out["metrics"]["strategy"]
    assert "avg_daily_trade_count" in out["metrics"]["strategy"]
    assert "avg_annual_trade_count" in out["metrics"]["strategy"]
    assert (out.get("market_regime") or {}).get("enabled") is True
    assert "strategy_state_contribution" in (out.get("market_regime") or {})
    assert set((out["event_study"] or {}).get("windows", {}).keys()) >= {"1d", "5d", "10d", "20d"}
    ev1 = (((out.get("event_study") or {}).get("windows") or {}).get("1d") or {})
    assert "profit_frequency" in (ev1.get("signal") or {})
    assert "bucket_probabilities" in (ev1.get("signal") or {})
    assert "bucket_profiles" in (ev1.get("signal") or {})
    assert "profit_frequency_mean" in (ev1.get("random_baseline") or {})
    assert "bucket_profiles_mean" in (ev1.get("random_baseline") or {})
    assert "delta_profit_frequency" in (ev1.get("comparison") or {})
    assert "delta_bucket_profiles" in (ev1.get("comparison") or {})
    # should have some non-trivial positions
    pos = out["signals"]["position"]
    assert any(x > 0 for x in pos)
    r_stats = out.get("r_statistics") or {}
    assert "overall" in r_stats
    assert "recent_100" in r_stats
    recent = (r_stats.get("recent_100") or {})
    assert int(recent.get("effective_count") or 0) <= int((r_stats.get("overall") or {}).get("trade_count") or 0)
    assert "sqn" in ((r_stats.get("overall") or {}))
    ts = out.get("trade_statistics") or {}
    trades = list((ts.get("trades") or []))
    if trades:
        t0 = trades[0]
        assert "initial_r_amount" in t0
        assert "initial_r_pct_nav" in t0
        assert "pnl_amount" in t0
        assert "r_multiple" in t0


def test_trend_ma_filter_ema_smoke(session_factory):
    """均线过滤策略使用 ma_type=ema 时行为与合并前的 ema_filter 一致。"""
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-06-30", freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            px = 100.0 + (i * 0.4 if i < 50 else (50 * 0.4) - (i - 50) * 0.6)
            _add_price(db, code=code, day=d, close=px)
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                ma_type="ema",
                sma_window=20,
                cost_bps=0.0,
            ),
        )
    assert out["meta"]["params"]["sma_window"] == 20
    assert out["meta"]["params"]["ma_type"] == "ema"
    assert out["meta"]["strategy"] == "ma_filter"
    assert any(x > 0 for x in out["signals"]["position"])


def test_trend_ma_filter_kama_smoke(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-06-30", freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            px = 100.0 + (i * 0.4 if i < 50 else (50 * 0.4) - (i - 50) * 0.6)
            _add_price(db, code=code, day=d, close=px)
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                ma_type="kama",
                sma_window=20,
                kama_er_window=10,
                kama_fast_window=2,
                kama_slow_window=30,
                kama_std_window=20,
                kama_std_coef=1.0,
                cost_bps=0.0,
            ),
        )
    params = out["meta"]["params"]
    assert params["ma_type"] == "kama"
    assert params["kama_er_window"] == 10
    assert params["kama_fast_window"] == 2
    assert params["kama_slow_window"] == 30
    assert params["kama_std_window"] == 20
    assert float(params["kama_std_coef"]) == 1.0
    assert out["meta"]["strategy"] == "ma_filter"
    assert any(x > 0 for x in out["signals"]["position"])


def test_trend_ma_filter_kama_std_band_reduces_trades(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=160, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            # Oscillating trend to trigger multiple entries/exits under KAMA filter.
            px = 100.0 + 0.12 * i + 4.0 * np.sin(i / 3.0)
            _add_price(db, code=code, day=d, close=float(px))
        db.commit()
        out_lo = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                ma_type="kama",
                sma_window=20,
                kama_std_window=20,
                kama_std_coef=0.0,
                cost_bps=0.0,
            ),
        )
        out_hi = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                ma_type="kama",
                sma_window=20,
                kama_std_window=20,
                kama_std_coef=3.0,
                cost_bps=0.0,
            ),
        )
    lo_trades = int((((out_lo.get("trade_statistics") or {}).get("overall") or {}).get("total_trades") or 0))
    hi_trades = int((((out_hi.get("trade_statistics") or {}).get("overall") or {}).get("total_trades") or 0))
    assert hi_trades <= lo_trades


def test_trend_ma_cross_supports_ema_type(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-06-30", freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            px = 100.0 + (i * 0.35 if i < 55 else (55 * 0.35) - (i - 55) * 0.5)
            _add_price(db, code=code, day=d, close=px)
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="ma_cross",
                fast_window=10,
                slow_window=30,
                ma_type="ema",
                cost_bps=0.0,
            ),
        )
    assert out["meta"]["strategy"] == "ma_cross"
    assert out["meta"]["params"]["ma_type"] == "ema"
    assert len(out["signals"]["position"]) == len(out["nav"]["dates"])


def test_trend_linreg_slope_smoke(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-06-30", freq="B")]
    with sf() as db:
        # Mostly upward drift so regression slope should be positive for many windows.
        for i, d in enumerate(dates):
            px = 100.0 * (1.0 + 0.001) ** i
            _add_price(db, code=code, day=d, close=px)
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(code=code, start=dates[0], end=dates[-1], strategy="linreg_slope", sma_window=30, cost_bps=0.0),
        )
    assert out["meta"]["strategy"] == "linreg_slope"
    assert any(x > 0 for x in out["signals"]["position"])


def test_trend_bias_binary_and_continuous_modes(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-06-30", freq="B")]

    # Build a step-wise series that guarantees:
    # - in_trend stays True (price well above long MA once regime starts)
    # - entry is triggered (negative bias during pullback)
    # - continuous mode produces fractional exposures (small positive bias after entry but below exit threshold)
    px = []
    for i in range(len(dates)):
        if i < 80:
            px.append(100.0)  # flat base
        elif i < 90:
            px.append(120.0)  # regime shift up (trend MA lags below)
        elif i < 100:
            px.append(110.0)  # pullback: negative bias but still above long MA -> enter
        elif i < 110:
            px.append(113.0)  # mild recovery: small positive bias -> fractional sizing
        else:
            px.append(125.0)  # overheat: should eventually exit in binary mode

    with sf() as db:
        for d, p in zip(dates, px):
            _add_price(db, code=code, day=d, close=float(p))
        db.commit()

        out_bin = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="bias",
                bias_ma_window=5,
                bias_entry=2.0,
                bias_hot=10.0,
                bias_cold=-2.0,
                bias_pos_mode="binary",
                cost_bps=0.0,
            ),
        )
        out_cont = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="bias",
                bias_ma_window=5,
                bias_entry=2.0,
                bias_hot=10.0,
                bias_cold=-2.0,
                bias_pos_mode="continuous",
                cost_bps=0.0,
            ),
        )

    pos_bin = out_bin["signals"]["position"]
    assert any(x == 0 for x in pos_bin) and any(x == 1 for x in pos_bin)

    pos_cont = out_cont["signals"]["position"]
    assert any(x == 0 for x in pos_cont) and any(x > 0 for x in pos_cont)
    # continuous mode should generate some fractional exposures
    assert any((x > 0) and (x < 1) for x in pos_cont)


def test_trend_nav_uses_none_with_hfq_fallback_on_corporate_action_cliff(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-01-10", freq="B")]
    # Build a fake split-like cliff in none price; hfq remains smooth; qfq used for signals remains smooth.
    none_px = [100, 101, 50, 51, 52, 53, 54, 55]
    hfq_px = [100, 101, 102, 103, 104, 105, 106, 107]
    qfq_px = [100, 101, 102, 103, 104, 105, 106, 107]
    with sf() as db:
        for d, n, h, q in zip(dates, none_px, hfq_px, qfq_px):
            db.add(EtfPrice(code=code, trade_date=d, open=float(n), high=float(n), low=float(n), close=float(n), volume=1.0, amount=1.0, source="eastmoney", adjust="none"))
            db.add(EtfPrice(code=code, trade_date=d, open=float(h), high=float(h), low=float(h), close=float(h), volume=1.0, amount=1.0, source="eastmoney", adjust="hfq"))
            db.add(EtfPrice(code=code, trade_date=d, open=float(q), high=float(q), low=float(q), close=float(q), volume=1.0, amount=1.0, source="eastmoney", adjust="qfq"))
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(code=code, start=dates[0], end=dates[-1], strategy="ma_filter", sma_window=2, cost_bps=0.0),
        )
    nav = out["nav"]["series"]["STRAT"]
    # If none cliff was applied directly while long, nav would roughly halve; with hfq fallback it should not.
    assert min(nav) > 0.75


def test_trend_macd_family_smoke(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-06-30", freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            # trend with oscillation to trigger crossovers
            px = 100.0 + i * 0.25 + (2.0 if (i % 10 < 5) else -2.0)
            _add_price(db, code=code, day=d, close=px)
        db.commit()
        out_cross = compute_trend_backtest(
            db,
            TrendInputs(code=code, start=dates[0], end=dates[-1], strategy="macd_cross", cost_bps=0.0),
        )
        out_zero = compute_trend_backtest(
            db,
            TrendInputs(code=code, start=dates[0], end=dates[-1], strategy="macd_zero_filter", cost_bps=0.0),
        )
        out_v = compute_trend_backtest(
            db,
            TrendInputs(code=code, start=dates[0], end=dates[-1], strategy="macd_v", cost_bps=0.0),
        )
    assert out_cross["meta"]["strategy"] == "macd_cross"
    assert out_zero["meta"]["strategy"] == "macd_zero_filter"
    assert out_v["meta"]["strategy"] == "macd_v"
    assert len(out_v["signals"]["position"]) == len(out_v["nav"]["dates"])


def test_trend_excludes_decision_day_return_for_all_strategies(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=8, freq="B")]
    # Day index:
    # d0=100, d1=100, d2=100, d3=200 (decision-day big jump), d4=200, d5=220, d6=220, d7=220
    # If decision-day return were included after signal flip at d3, NAV would jump by ~100%.
    pxs = [100.0, 100.0, 100.0, 200.0, 200.0, 220.0, 220.0, 220.0]
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
        for d, p in zip(dates, pxs):
            _add_price(db, code=code, day=d, close=p)
        db.commit()

        for strat, params in strategies:
            out = compute_trend_backtest(
                db,
                TrendInputs(
                    code=code,
                    start=dates[0],
                    end=dates[-1],
                    strategy=strat,
                    cost_bps=0.0,
                    **params,
                ),
            )
            nav = [float(x) for x in out["nav"]["series"]["STRAT"]]
            pos = [float(x) for x in out["signals"]["position"]]
            # Ensure there is at least one signal-on day so this test is meaningful.
            assert any(x > 0 for x in pos), f"{strat} did not produce any long signal"
            # The first post-jump NAV point must remain ~1.0 (decision-day return excluded).
            # We allow tiny epsilon for float operations.
            assert nav[3] <= 1.0000001, f"{strat} appears to include decision-day return"


def test_random_entry_signal_generator_is_deterministic() -> None:
    idx = pd.bdate_range("2024-01-01", periods=20)
    pos = _pos_from_random_entry_hold(idx, hold_days=3, seed=1)
    assert [int(x) for x in pos.tolist()] == [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]


def test_trend_random_entry_seed_controls_reproducibility(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=80, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            _add_price(db, code=code, day=d, close=100.0 + i * 0.1)
        db.commit()
        out_a = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="random_entry",
                random_hold_days=20,
                random_seed=42,
                cost_bps=0.0,
            ),
        )
        out_b = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="random_entry",
                random_hold_days=20,
                random_seed=42,
                cost_bps=0.0,
            ),
        )
        out_c = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="random_entry",
                random_hold_days=20,
                random_seed=43,
                cost_bps=0.0,
            ),
        )
    pos_a = list((out_a.get("signals") or {}).get("position") or [])
    pos_b = list((out_b.get("signals") or {}).get("position") or [])
    pos_c = list((out_c.get("signals") or {}).get("position") or [])
    assert pos_a == pos_b
    assert pos_a != pos_c
    params = ((out_a.get("meta") or {}).get("params") or {})
    assert int(params.get("random_hold_days") or 0) == 20
    assert int(params.get("random_seed") or -1) == 42


def test_trend_random_entry_allows_system_random_seed(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=40, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            _add_price(db, code=code, day=d, close=100.0 + i * 0.1)
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
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


def test_r_take_profit_triggers_on_peak_drawdown_with_virtual_atr_fallback() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    base_pos = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0, 1.0], index=idx, dtype=float)
    close = pd.Series([100.0, 100.0, 110.0, 120.0, 130.0, 120.0], index=idx, dtype=float)
    high = close.copy()
    low = close.copy()

    out_pos, stats = _apply_r_multiple_take_profit(
        base_pos,
        close=close,
        high=high,
        low=low,
        enabled=True,
        reentry_mode="reenter",
        atr_window=2,
        atr_n=1.0,
        tiers=[
            {"r_multiple": 2.0, "retrace_ratio": 0.5},
            {"r_multiple": 3.0, "retrace_ratio": 0.3},
        ],
        atr_stop_enabled=False,
    )

    assert float(out_pos.iloc[-1]) == 0.0
    assert int(stats.get("trigger_count") or 0) >= 1
    assert bool(stats.get("fallback_mode_used")) is True
    assert str(stats.get("initial_r_mode") or "") == "virtual_atr_fallback"


def test_trend_backtest_exposes_r_take_profit_controls(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=90, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            if i < 60:
                px = 100.0 + i * 0.8
            else:
                px = 148.0 - (i - 60) * 0.5
            _add_price(db, code=code, day=d, close=px)
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
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
                cost_bps=0.0,
            ),
        )
    rtp = (((out.get("risk_controls") or {}).get("r_take_profit") or {}))
    assert bool(rtp.get("enabled")) is True
    assert str(rtp.get("initial_r_mode") or "") == "virtual_atr_fallback"
    assert isinstance((rtp.get("tier_trigger_counts") or {}), dict)
    params = (((out.get("meta") or {}).get("params") or {}))
    assert bool(params.get("r_take_profit_enabled")) is True
    metrics = ((out.get("metrics") or {}).get("strategy") or {})
    assert "r_take_profit_trigger_count" not in metrics
    ts = (out.get("trade_statistics") or {})
    overall = (ts.get("overall") or {})
    by_code = (ts.get("by_code") or {}).get(code, {})
    assert "atr_stop_trigger_count" in overall
    assert "r_take_profit_trigger_count" in overall
    assert isinstance((overall.get("r_take_profit_tier_trigger_counts") or {}), dict)
    assert "atr_stop_trigger_count" in by_code
    assert "r_take_profit_trigger_count" in by_code

