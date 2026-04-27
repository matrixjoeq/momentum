import datetime as dt

import numpy as np
import pandas as pd
import pytest

from etf_momentum.analysis.trend import (
    TrendInputs,
    _risk_budget_dynamic_weights,
    _next_vol_regime_state,
    _trade_stats_from_returns,
    _semi_variance_run_stats_from_returns,
    _apply_impulse_entry_filter,
    _apply_atr_stop,
    _apply_intraday_stop_execution_single,
    _apply_r_multiple_take_profit,
    _apply_bias_v_take_profit,
    _position_risk_from_stop_params,
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


def _add_price_hl(
    db, *, code: str, day: dt.date, close: float, high: float, low: float
) -> None:
    add_price_all_adjustments(
        db,
        code=code,
        day=day,
        close=float(close),
        open_price=float(close),
        high=float(high),
        low=float(low),
    )


def test_risk_budget_dynamic_weights_entry_extreme_state_not_counted_as_dynamic_adjust() -> (
    None
):
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    active = pd.Series([1, 1, 1, 1, 1, 1], index=idx, dtype=float)
    close = pd.Series([100, 100, 100, 100, 100, 100], index=idx, dtype=float)
    atr_b = pd.Series([10, 10, 10, 10, 10, 10], index=idx, dtype=float)
    atr_fast = pd.Series([20, 20, 20, 20, 20, 20], index=idx, dtype=float)
    atr_slow = pd.Series([10, 10, 10, 10, 10, 10], index=idx, dtype=float)
    w, stats = _risk_budget_dynamic_weights(
        active,
        close=close,
        atr_for_budget=atr_b,
        atr_fast=atr_fast,
        atr_slow=atr_slow,
        risk_budget_pct=0.01,
        dynamic_enabled=True,
        expand_threshold=1.45,
        contract_threshold=0.65,
        normal_threshold=1.05,
        extreme_threshold=2.0,
    )
    assert all(float(x) > 0.0 for x in w.tolist())
    assert int(stats.get("vol_risk_entry_state_reduce_on_expand_count") or 0) == 1
    assert int(stats.get("vol_risk_entry_state_increase_on_contract_count") or 0) == 0
    assert int(stats.get("vol_risk_adjust_total_count") or 0) == 0


def test_risk_budget_dynamic_weights_supports_extreme_tier_transition() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    active = pd.Series([1, 1, 1, 1], index=idx, dtype=float)
    close = pd.Series([100, 100, 100, 100], index=idx, dtype=float)
    atr_b = pd.Series([20, 12, 12, 12], index=idx, dtype=float)
    atr_fast = pd.Series([2.2, 1.8, 1.8, 1.8], index=idx, dtype=float)
    atr_slow = pd.Series([1, 1, 1, 1], index=idx, dtype=float)

    w, stats = _risk_budget_dynamic_weights(
        active,
        close=close,
        atr_for_budget=atr_b,
        atr_fast=atr_fast,
        atr_slow=atr_slow,
        risk_budget_pct=0.1,
        dynamic_enabled=True,
        expand_threshold=1.45,
        contract_threshold=0.65,
        normal_threshold=1.05,
        extreme_threshold=2.0,
    )

    # EXTREME -> REDUCED transition should rebalance once.
    assert float(w.iloc[1]) > float(w.iloc[0])
    assert int(stats.get("vol_risk_adjust_total_count") or 0) == 1


def test_vol_regime_state_machine_allows_direct_cross_tier_jumps() -> None:
    assert (
        _next_vol_regime_state(
            "REDUCED",
            ratio=0.50,
            expand_threshold=1.45,
            contract_threshold=0.65,
            normal_threshold=1.05,
            extreme_threshold=2.0,
        )
        == "INCREASED"
    )
    assert (
        _next_vol_regime_state(
            "INCREASED",
            ratio=2.20,
            expand_threshold=1.45,
            contract_threshold=0.65,
            normal_threshold=1.05,
            extreme_threshold=2.0,
        )
        == "EXTREME"
    )
    assert (
        _next_vol_regime_state(
            "EXTREME",
            ratio=0.55,
            expand_threshold=1.45,
            contract_threshold=0.65,
            normal_threshold=1.05,
            extreme_threshold=2.0,
        )
        == "INCREASED"
    )


def test_semi_variance_run_stats_split_continuous_profit_loss_segments() -> None:
    returns = [0.01, 0.02, -0.01, -0.02, -0.03, 0.05, 0.01, 0.0, -0.01]
    stats = _semi_variance_run_stats_from_returns(returns)

    assert int(stats.get("total_segments") or 0) == 4
    assert int(stats.get("profit_segments") or 0) == 2
    assert int(stats.get("loss_segments") or 0) == 2
    assert float(stats.get("win_rate_ex_zero") or 0.0) == pytest.approx(0.5)

    profit_ret = (stats.get("profit_return_stats") or {}).get("max")
    loss_ret = (stats.get("loss_return_stats") or {}).get("min")
    assert float(profit_ret or 0.0) > 0.05
    assert float(loss_ret or 0.0) < -0.05

    profit_len_stats = stats.get("profit_count_stats") or {}
    loss_len_stats = stats.get("loss_count_stats") or {}
    assert int(round(float(profit_len_stats.get("max") or 0.0))) == 2
    assert int(round(float(loss_len_stats.get("max") or 0.0))) == 3
    assert int(round(float(loss_len_stats.get("min") or 0.0))) == 1


def test_trade_stats_risk_of_ruin_matches_expected_formula_value() -> None:
    stats = _trade_stats_from_returns(
        [0.03, -0.01, 0.03, -0.01, 0.03],
        risk_of_ruin_maxrisk=0.30,
    )
    ror = stats.get("risk_of_ruin") or {}
    # P = 3/5 = 0.6, A = 0.01, configured maxrisk = 0.30 -> exponent = 30
    assert float(ror.get("P") or 0.0) == pytest.approx(0.6, rel=0.0, abs=1e-12)
    assert float(ror.get("A") or 0.0) == pytest.approx(0.01, rel=0.0, abs=1e-12)
    assert float(ror.get("maxrisk") or 0.0) == pytest.approx(0.30, rel=0.0, abs=1e-12)
    assert str(ror.get("maxrisk_basis") or "") == "configured_risk_of_ruin_maxrisk"
    assert float(ror.get("probability") or 0.0) == pytest.approx(
        ((1.0 - 0.6) / 0.6) ** (0.30 / 0.01), rel=0.0, abs=1e-12
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


def test_same_day_stop_triggers_on_entry_session_for_futures_t0() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    base_pos = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0], index=idx, dtype=float)
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=idx, dtype=float)
    high = pd.Series([106.0, 106.0, 106.0, 106.0, 106.0], index=idx, dtype=float)
    low = pd.Series([100.0, 100.0, 97.6, 96.0, 96.0], index=idx, dtype=float)
    open_ = pd.Series([100.0, 100.0, 100.0, 98.0, 100.0], index=idx, dtype=float)
    out_nd, _ = _apply_atr_stop(
        base_pos,
        open_=open_,
        close=close,
        high=high,
        low=low,
        mode="static",
        atr_basis="entry",
        reentry_mode="reenter",
        atr_window=2,
        n_mult=0.2,
        m_step=0.5,
        same_day_stop=False,
    )
    out_sd, stats_sd = _apply_atr_stop(
        base_pos,
        open_=open_,
        close=close,
        high=high,
        low=low,
        mode="static",
        atr_basis="entry",
        reentry_mode="reenter",
        atr_window=2,
        n_mult=0.2,
        m_step=0.5,
        same_day_stop=True,
    )
    assert float(out_nd.iloc[2]) == 1.0
    assert float(out_sd.iloc[2]) == 0.0
    assert int(stats_sd.get("trigger_count") or 0) >= 1


def test_atr_stop_intraday_trigger_on_low_and_gap_open_fill() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    base_pos = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx, dtype=float)
    close = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx, dtype=float)
    high = pd.Series([106.0, 106.0, 106.0, 106.0], index=idx, dtype=float)
    low_touch = pd.Series([94.0, 96.0, 97.6, 96.0], index=idx, dtype=float)
    open_touch = pd.Series([100.0, 98.0, 99.0, 100.0], index=idx, dtype=float)
    out_touch, stats_touch = _apply_atr_stop(
        base_pos,
        open_=open_touch,
        close=close,
        high=high,
        low=low_touch,
        mode="static",
        atr_basis="entry",
        reentry_mode="reenter",
        atr_window=2,
        n_mult=0.2,
        m_step=0.5,
    )
    assert float(out_touch.iloc[3]) == 0.0
    ev_touch = list(stats_touch.get("trigger_events") or [])
    assert ev_touch
    assert str(ev_touch[0].get("trigger_source")) == "low_touch_stop"
    assert float(ev_touch[0].get("fill_price")) == pytest.approx(97.8)

    low_gap = pd.Series([94.0, 96.0, 97.9, 95.0], index=idx, dtype=float)
    open_gap = pd.Series([100.0, 98.0, 100.0, 96.8], index=idx, dtype=float)
    out_gap, stats_gap = _apply_atr_stop(
        base_pos,
        open_=open_gap,
        close=close,
        high=high,
        low=low_gap,
        mode="static",
        atr_basis="entry",
        reentry_mode="reenter",
        atr_window=2,
        n_mult=0.2,
        m_step=0.5,
    )
    assert float(out_gap.iloc[3]) == 0.0
    ev_gap = list(stats_gap.get("trigger_events") or [])
    assert ev_gap
    assert str(ev_gap[0].get("trigger_source")) == "gap_open_below_stop"
    assert bool(ev_gap[0].get("gap_open_triggered")) is True
    assert float(ev_gap[0].get("fill_price")) == pytest.approx(96.8)


def test_atr_stop_short_intraday_trigger_on_high_and_gap_open_fill() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    base_pos = pd.Series([-1.0, -1.0, -1.0, -1.0], index=idx, dtype=float)
    close = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx, dtype=float)
    high = pd.Series([106.0, 106.0, 102.5, 106.0], index=idx, dtype=float)
    low = pd.Series([94.0, 96.0, 96.0, 96.0], index=idx, dtype=float)
    open_touch = pd.Series([100.0, 98.0, 99.0, 100.0], index=idx, dtype=float)
    out_touch, stats_touch = _apply_atr_stop(
        base_pos,
        open_=open_touch,
        close=close,
        high=high,
        low=low,
        mode="static",
        atr_basis="entry",
        reentry_mode="reenter",
        atr_window=2,
        n_mult=0.2,
        m_step=0.5,
    )
    assert float(out_touch.iloc[3]) == 0.0
    ev_touch = list(stats_touch.get("trigger_events") or [])
    assert ev_touch
    assert str(ev_touch[0].get("trigger_source")) == "high_touch_stop"
    assert float(ev_touch[0].get("fill_price")) == pytest.approx(102.2)

    open_gap = pd.Series([100.0, 98.0, 99.0, 103.0], index=idx, dtype=float)
    low_gap = pd.Series([94.0, 96.0, 96.0, 95.0], index=idx, dtype=float)
    out_gap, stats_gap = _apply_atr_stop(
        base_pos,
        open_=open_gap,
        close=close,
        high=high,
        low=low_gap,
        mode="static",
        atr_basis="entry",
        reentry_mode="reenter",
        atr_window=2,
        n_mult=0.2,
        m_step=0.5,
    )
    assert float(out_gap.iloc[3]) == 0.0
    ev_gap = list(stats_gap.get("trigger_events") or [])
    assert ev_gap
    assert str(ev_gap[0].get("trigger_source")) == "gap_open_above_stop"
    assert bool(ev_gap[0].get("gap_open_triggered")) is True
    assert float(ev_gap[0].get("fill_price")) == pytest.approx(103.0)


def test_static_atr_stop_is_fixed_and_ignores_atr_basis() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    base_pos = pd.Series([1.0] * len(idx), index=idx, dtype=float)
    close = pd.Series(
        [100.0, 101.0, 103.0, 102.0, 104.0, 105.0], index=idx, dtype=float
    )
    high = pd.Series([102.0, 108.0, 112.0, 111.0, 115.0, 118.0], index=idx, dtype=float)
    low = pd.Series([98.0, 96.0, 95.0, 97.0, 94.0, 93.0], index=idx, dtype=float)
    open_ = close.copy()
    out_entry, stats_entry = _apply_atr_stop(
        base_pos,
        open_=open_,
        close=close,
        high=high,
        low=low,
        mode="static",
        atr_basis="entry",
        reentry_mode="reenter",
        atr_window=2,
        n_mult=2.0,
        m_step=0.5,
    )
    out_latest, stats_latest = _apply_atr_stop(
        base_pos,
        open_=open_,
        close=close,
        high=high,
        low=low,
        mode="static",
        atr_basis="latest",
        reentry_mode="reenter",
        atr_window=2,
        n_mult=2.0,
        m_step=0.5,
    )
    row_e = [
        r
        for r in list(stats_entry.get("trace_last_rows") or [])
        if float(r.get("decision_pos") or 0.0) > 0.0
    ]
    row_l = [
        r
        for r in list(stats_latest.get("trace_last_rows") or [])
        if float(r.get("decision_pos") or 0.0) > 0.0
    ]
    assert row_e and row_l
    stop_vals_e = [
        float(r.get("stop_after")) for r in row_e if r.get("stop_after") is not None
    ]
    stop_vals_l = [
        float(r.get("stop_after")) for r in row_l if r.get("stop_after") is not None
    ]
    assert stop_vals_e and stop_vals_l
    first_e = float(stop_vals_e[0])
    first_l = float(stop_vals_l[0])
    for v in stop_vals_e:
        assert v == pytest.approx(first_e)
    for v in stop_vals_l:
        assert v == pytest.approx(first_l)
    assert first_e == pytest.approx(first_l)
    assert len(out_entry) == len(base_pos)
    assert len(out_latest) == len(base_pos)


def test_r_take_profit_intraday_retrace_trigger_and_gap_fill() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    base_pos = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0], index=idx, dtype=float)
    close = pd.Series([100.0, 100.0, 108.0, 108.0, 108.0], index=idx, dtype=float)
    high = pd.Series([100.0, 103.0, 110.0, 112.0, 112.0], index=idx, dtype=float)
    low = pd.Series([100.0, 97.0, 107.0, 104.0, 104.0], index=idx, dtype=float)
    open_touch = pd.Series([100.0, 100.0, 109.0, 112.0, 108.0], index=idx, dtype=float)
    tiers = [{"r_multiple": 1.5, "retrace_ratio": 0.3}]

    out_touch, stats_touch = _apply_r_multiple_take_profit(
        base_pos,
        open_=open_touch,
        close=close,
        high=high,
        low=low,
        enabled=True,
        reentry_mode="reenter",
        atr_window=2,
        atr_n=1.0,
        tiers=tiers,
        atr_stop_enabled=True,
    )
    assert float(out_touch.iloc[3]) == 0.0
    ev_touch = list(stats_touch.get("trigger_events") or [])
    assert ev_touch
    assert str(ev_touch[0].get("trigger_source")) == "low_touch_tp_retrace"
    assert float(ev_touch[0].get("fill_price")) == pytest.approx(108.4)

    open_gap = pd.Series([100.0, 100.0, 106.0, 106.0, 108.0], index=idx, dtype=float)
    out_gap, stats_gap = _apply_r_multiple_take_profit(
        base_pos,
        open_=open_gap,
        close=close,
        high=high,
        low=low,
        enabled=True,
        reentry_mode="reenter",
        atr_window=2,
        atr_n=1.0,
        tiers=tiers,
        atr_stop_enabled=True,
    )
    assert float(out_gap.iloc[3]) == 0.0
    ev_gap = list(stats_gap.get("trigger_events") or [])
    assert ev_gap
    assert str(ev_gap[0].get("trigger_source")) == "gap_open_below_tp"
    assert bool(ev_gap[0].get("gap_open_triggered")) is True
    assert float(ev_gap[0].get("fill_price")) == pytest.approx(106.0)


def test_bias_v_take_profit_intraday_high_trigger_and_gap_fill() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    base_pos = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0], index=idx, dtype=float)
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=idx, dtype=float)
    high = pd.Series([100.0, 100.0, 100.0, 108.0, 108.0], index=idx, dtype=float)
    low = pd.Series([99.0, 99.0, 99.0, 99.0, 99.0], index=idx, dtype=float)
    open_touch = pd.Series([100.0, 100.0, 100.0, 102.0, 102.0], index=idx, dtype=float)
    out_touch, stats_touch = _apply_bias_v_take_profit(
        base_pos,
        open_=open_touch,
        close=close,
        high=high,
        low=low,
        enabled=True,
        reentry_mode="reenter",
        ma_window=2,
        atr_window=2,
        threshold=0.5,
    )
    assert float(out_touch.iloc[3]) == 0.0
    ev_touch = list(stats_touch.get("trigger_events") or [])
    assert ev_touch
    assert str(ev_touch[0].get("trigger_source")) == "high_touch_bias_v_tp"
    assert 100.0 < float(ev_touch[0].get("fill_price")) < 108.0

    open_gap = pd.Series([100.0, 100.0, 100.0, 106.0, 104.0], index=idx, dtype=float)
    out_gap, stats_gap = _apply_bias_v_take_profit(
        base_pos,
        open_=open_gap,
        close=close,
        high=high,
        low=low,
        enabled=True,
        reentry_mode="reenter",
        ma_window=2,
        atr_window=2,
        threshold=0.5,
    )
    assert float(out_gap.iloc[3]) == 0.0
    ev_gap = list(stats_gap.get("trigger_events") or [])
    assert ev_gap
    assert str(ev_gap[0].get("trigger_source")) == "gap_open_above_bias_v_tp"
    assert bool(ev_gap[0].get("gap_open_triggered")) is True
    assert float(ev_gap[0].get("fill_price")) == pytest.approx(106.0)


def test_atr_stop_next_day_execution_triggers_on_close_and_schedules_next_day() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    base_pos = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0], index=idx, dtype=float)
    open_ = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=idx, dtype=float)
    close = pd.Series([100.0, 100.0, 100.0, 99.5, 99.0], index=idx, dtype=float)
    high = pd.Series([101.0, 101.0, 101.0, 101.0, 101.0], index=idx, dtype=float)
    low = pd.Series([99.8, 99.8, 99.8, 99.8, 99.8], index=idx, dtype=float)
    out, stats = _apply_atr_stop(
        base_pos,
        open_=open_,
        close=close,
        high=high,
        low=low,
        mode="static",
        atr_basis="entry",
        reentry_mode="wait_next_entry",
        execution_mode="next_day",
        atr_window=2,
        n_mult=0.2,
        m_step=0.5,
    )
    # Close trigger day keeps position; next day executes stop and exits.
    assert float(out.iloc[3]) == 1.0
    assert float(out.iloc[4]) == 0.0
    ev = list(stats.get("trigger_events") or [])
    assert ev
    assert str(ev[0].get("execution_mode")) == "next_day"
    assert str(ev[0].get("trigger_date")) == idx[3].date().isoformat()
    assert str(ev[0].get("execution_date")) == idx[4].date().isoformat()
    assert str(ev[0].get("date")) == idx[4].date().isoformat()
    assert ev[0].get("fill_price") is None


def test_intraday_stop_override_open_exec_uses_prev_close_base() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    weights = pd.Series([0.0, 1.0, 1.0], index=idx, dtype=float)
    open_sig = pd.Series([100.0, 100.0, 100.0], index=idx, dtype=float)
    close_sig = pd.Series([100.0, 110.0, 105.0], index=idx, dtype=float)
    stats = {
        "trigger_events": [
            {
                "date": idx[2].date().isoformat(),
                "execution_mode": "intraday",
                "fill_price": 95.0,
            }
        ]
    }
    w_adj, override = _apply_intraday_stop_execution_single(
        weights=weights,
        atr_stop_stats=stats,
        exec_price="open",
        stop_execution_mode="intraday",
        open_sig=open_sig,
        close_sig=close_sig,
    )
    assert float(w_adj.iloc[2]) == 0.0
    assert float(override.iloc[2]) == pytest.approx(95.0 / 110.0 - 1.0)


@pytest.mark.parametrize(
    ("exec_price", "expected"),
    [
        ("open", 98.0 / 110.0 - 1.0),
        ("close", 96.0 / 110.0 - 1.0),
    ],
)
def test_next_day_stop_override_uses_strategy_execution_price(
    exec_price: str, expected: float
) -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    weights = pd.Series([0.0, 1.0, 1.0], index=idx, dtype=float)
    open_sig = pd.Series([100.0, 100.0, 98.0], index=idx, dtype=float)
    close_sig = pd.Series([100.0, 110.0, 96.0], index=idx, dtype=float)
    stats = {
        "trigger_events": [
            {
                "date": idx[2].date().isoformat(),
                "trigger_date": idx[1].date().isoformat(),
                "execution_date": idx[2].date().isoformat(),
                "execution_mode": "next_day",
                "fill_price": None,
            }
        ]
    }
    w_adj, override = _apply_intraday_stop_execution_single(
        weights=weights,
        atr_stop_stats=stats,
        exec_price=exec_price,
        stop_execution_mode="next_day",
        open_sig=open_sig,
        close_sig=close_sig,
    )
    assert float(w_adj.iloc[2]) == 0.0
    assert float(override.iloc[2]) == pytest.approx(expected)


def test_atr_stop_requires_one_effective_holding_day_before_trigger() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    base_pos = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0], index=idx, dtype=float)
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=idx, dtype=float)
    high = pd.Series([106.0, 106.0, 106.0, 106.0, 106.0], index=idx, dtype=float)
    low = pd.Series([100.0, 100.0, 97.0, 96.0, 96.0], index=idx, dtype=float)
    open_ = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=idx, dtype=float)
    out, stats = _apply_atr_stop(
        base_pos,
        open_=open_,
        close=close,
        high=high,
        low=low,
        mode="static",
        atr_basis="entry",
        reentry_mode="reenter",
        atr_window=2,
        n_mult=0.2,
        m_step=0.5,
        same_day_stop=False,
    )
    # Entry decision at t1 means effective entry at t2; stop can only trigger from t3.
    assert float(out.iloc[2]) == 1.0
    assert float(out.iloc[3]) == 0.0
    ev = list(stats.get("trigger_events") or [])
    assert ev
    assert str(ev[0].get("date")) == idx[3].date().isoformat()


def test_r_take_profit_requires_one_effective_holding_day_before_trigger() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    base_pos = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0], index=idx, dtype=float)
    close = pd.Series([100.0, 100.0, 108.0, 108.0, 108.0], index=idx, dtype=float)
    high = pd.Series([101.0, 103.0, 110.0, 110.0, 110.0], index=idx, dtype=float)
    low = pd.Series([99.0, 97.0, 107.0, 107.0, 107.0], index=idx, dtype=float)
    open_ = pd.Series([100.0, 100.0, 109.0, 109.0, 109.0], index=idx, dtype=float)
    out, stats = _apply_r_multiple_take_profit(
        base_pos,
        open_=open_,
        close=close,
        high=high,
        low=low,
        enabled=True,
        reentry_mode="reenter",
        atr_window=2,
        atr_n=1.0,
        tiers=[{"r_multiple": 1.5, "retrace_ratio": 0.3}],
        atr_stop_enabled=True,
    )
    # Retrace condition is already met on t2 but trigger starts from t3.
    assert float(out.iloc[2]) == 1.0
    assert float(out.iloc[3]) == 0.0
    ev = list(stats.get("trigger_events") or [])
    assert ev
    assert str(ev[0].get("date")) == idx[3].date().isoformat()


def test_bias_v_take_profit_requires_one_effective_holding_day_before_trigger() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    base_pos = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0], index=idx, dtype=float)
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=idx, dtype=float)
    high = pd.Series([100.0, 100.0, 108.0, 108.0, 108.0], index=idx, dtype=float)
    low = pd.Series([99.0, 99.0, 99.0, 99.0, 99.0], index=idx, dtype=float)
    open_ = pd.Series([100.0, 100.0, 102.0, 102.0, 102.0], index=idx, dtype=float)
    out, stats = _apply_bias_v_take_profit(
        base_pos,
        open_=open_,
        close=close,
        high=high,
        low=low,
        enabled=True,
        reentry_mode="reenter",
        ma_window=2,
        atr_window=2,
        threshold=0.5,
    )
    # Bias trigger condition is met on t2 but can only execute from t3.
    assert float(out.iloc[2]) == 1.0
    assert float(out.iloc[3]) == 0.0
    ev = list(stats.get("trigger_events") or [])
    assert ev
    assert str(ev[0].get("date")) == idx[3].date().isoformat()


def test_atr_stop_trade_records_capture_required_fields() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    base_pos = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx, dtype=float)
    close = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx, dtype=float)
    high = pd.Series([106.0, 106.0, 106.0, 106.0], index=idx, dtype=float)
    low = pd.Series([94.0, 96.0, 95.0, 96.0], index=idx, dtype=float)
    open_ = pd.Series([100.0, 98.0, 96.8, 100.0], index=idx, dtype=float)
    _, stats = _apply_atr_stop(
        base_pos,
        open_=open_,
        close=close,
        high=high,
        low=low,
        mode="static",
        atr_basis="entry",
        reentry_mode="reenter",
        atr_window=2,
        n_mult=0.2,
        m_step=0.5,
    )
    recs = list(stats.get("trade_records") or [])
    assert recs
    one = recs[0]
    assert one.get("entry_decision_date")
    assert "initial_stop_price" in one
    assert "trigger_stop_price" in one
    assert "execution_stop_price" in one


def test_bias_v_trade_records_capture_required_fields() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    base_pos = pd.Series([0.0, 1.0, 1.0, 1.0], index=idx, dtype=float)
    close = pd.Series([100.0, 100.0, 104.0, 105.0], index=idx, dtype=float)
    high = pd.Series([100.0, 101.0, 106.0, 106.0], index=idx, dtype=float)
    low = pd.Series([100.0, 99.0, 103.0, 104.0], index=idx, dtype=float)
    open_ = close.copy()
    _, stats = _apply_bias_v_take_profit(
        base_pos,
        open_=open_,
        close=close,
        high=high,
        low=low,
        enabled=True,
        reentry_mode="reenter",
        ma_window=2,
        atr_window=2,
        threshold=0.5,
    )
    recs = list(stats.get("trade_records") or [])
    assert recs
    one = recs[0]
    assert one.get("entry_decision_date")
    assert "initial_take_profit_price" in one
    assert "trigger_take_profit_price" in one
    assert "execution_take_profit_price" in one


def test_bias_v_take_profit_trigger_price_is_non_decreasing_in_position() -> None:
    idx = pd.date_range("2024-01-01", periods=7, freq="B")
    base_pos = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], index=idx, dtype=float)
    close = pd.Series(
        [100.0, 104.0, 102.0, 101.0, 100.5, 100.0, 100.0], index=idx, dtype=float
    )
    high = pd.Series(
        [100.0, 104.0, 103.0, 102.0, 101.0, 100.0, 100.0], index=idx, dtype=float
    )
    low = pd.Series(
        [100.0, 103.0, 101.0, 100.0, 99.5, 99.0, 100.0], index=idx, dtype=float
    )
    open_ = close.copy()
    out, stats = _apply_bias_v_take_profit(
        base_pos,
        open_=open_,
        close=close,
        high=high,
        low=low,
        enabled=True,
        reentry_mode="wait_next_entry",
        ma_window=2,
        atr_window=2,
        threshold=0.5,
    )
    rows = [
        r
        for r in list(stats.get("trace_last_rows") or [])
        if float(r.get("decision_pos") or 0.0) > 0.0
    ]
    eff_vals = [
        float(r.get("tp_trigger_price_eff"))
        for r in rows
        if r.get("tp_trigger_price_eff") is not None
    ]
    assert eff_vals
    for i in range(1, len(eff_vals)):
        assert eff_vals[i] >= eff_vals[i - 1] - 1e-9
    assert any(float(x) >= 0.0 for x in out.tolist())


def test_trend_next_plan_exposes_strict_entry_exec_price_with_slippage(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=14, freq="B")]
    opens = [100.0 + i * 0.7 for i in range(len(dates))]
    closes = [100.3 + i * 0.7 for i in range(len(dates))]
    with sf() as db:
        for i, d in enumerate(dates):
            add_price_all_adjustments(
                db,
                code=code,
                day=d,
                close=float(closes[i]),
                open_price=float(opens[i]),
                high=float(max(opens[i], closes[i]) + 0.2),
                low=float(min(opens[i], closes[i]) - 0.2),
            )
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=2,
                exec_price="open",
                slippage_rate=0.2,
                cost_bps=0.0,
            ),
        )
    eff = [
        float(x) for x in ((out.get("signals") or {}).get("position_effective") or [])
    ]
    assert eff and float(eff[-1]) > 0.0
    entry_idx = None
    prev = 0.0
    for i, w in enumerate(eff):
        if w > 1e-12 and prev <= 1e-12:
            entry_idx = i
        prev = w
    assert entry_idx is not None
    m = (
        (out.get("next_plan") or {}).get("entry_exec_price_with_slippage_by_asset")
    ) or {}
    got = float(m.get(code))
    expected = float(opens[int(entry_idx)] + 0.1)
    assert got == pytest.approx(expected)


def test_trend_atr_plan_stop_prices_are_exposed_from_engine(session_factory):
    sf = session_factory
    code = "ASTP"
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=20, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            px = 100.0 + i * 0.6
            add_price_all_adjustments(
                db,
                code=code,
                day=d,
                close=float(px),
                open_price=float(px),
                high=float(px + 1.0),
                low=float(px - 1.0),
            )
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=2,
                atr_stop_mode="static",
                atr_stop_atr_basis="latest",
                atr_stop_n=2.0,
                atr_stop_window=2,
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )
    atr = ((out.get("risk_controls") or {}).get("atr_stop")) or {}
    assert "plan_stop_current" in atr
    assert "plan_stop_next" in atr
    if (
        float(((out.get("signals") or {}).get("position_effective") or [0.0])[-1])
        > 1e-12
    ):
        assert (atr.get("plan_stop_next") is None) or np.isfinite(
            float(atr.get("plan_stop_next"))
        )


def test_trend_non_quick_atr_trade_records_include_entry_execution_fields(
    session_factory,
):
    sf = session_factory
    code = "ATRREC"
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=25, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            px = 100.0 + min(i, 12) * 0.8
            hi = px + 1.0
            lo = px - 1.0
            if i == 15:
                lo = px - 10.0
            add_price_all_adjustments(
                db,
                code=code,
                day=d,
                close=float(px),
                open_price=float(px),
                high=float(hi),
                low=float(lo),
            )
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=2,
                atr_stop_mode="static",
                atr_stop_atr_basis="entry",
                atr_stop_window=2,
                atr_stop_n=0.8,
                exec_price="open",
                slippage_rate=0.2,
                cost_bps=0.0,
                quick_mode=False,
            ),
        )
    atr = ((out.get("risk_controls") or {}).get("atr_stop")) or {}
    recs = list(atr.get("trade_records") or [])
    if recs:
        one = recs[0]
        assert one.get("entry_execution_date")
        assert one.get("entry_execution_price") is not None


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

    params = (out.get("meta") or {}).get("params") or {}
    assert str(params.get("position_sizing") or "") == "risk_budget"
    assert int(params.get("risk_budget_atr_window") or 0) == 2
    assert float(params.get("risk_budget_pct") or 0.0) == 0.005
    eff = [
        float(x) for x in ((out.get("signals") or {}).get("position_effective") or [])
    ]
    assert eff
    assert any(x > 0.0 for x in eff)
    positive_eff = [x for x in eff if x > 1e-12]
    assert len(positive_eff) >= 3
    # Risk-budget sizing should be frozen at entry while the position stays open.
    assert max(positive_eff) - min(positive_eff) <= 1e-12


def test_trend_single_risk_budget_pct_upper_bound(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=40, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            px = 100.0 + i * 0.6
            _add_price(db, code=code, day=d, close=px)
        db.commit()
        try:
            compute_trend_backtest(
                db,
                TrendInputs(
                    code=code,
                    start=dates[0],
                    end=dates[-1],
                    strategy="ma_filter",
                    sma_window=5,
                    position_sizing="risk_budget",
                    risk_budget_atr_window=2,
                    risk_budget_pct=0.03,
                    cost_bps=0.0,
                ),
            )
            assert False, "expected ValueError for risk_budget_pct > 0.02"
        except ValueError as exc:
            assert "risk_budget_pct must be in [0.001, 0.02]" in str(exc)


def test_trend_single_monthly_risk_budget_blocks_entries(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=80, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            px = 100.0 + 2.0 * np.sin(i / 2.0) + 0.05 * i
            _add_price(db, code=code, day=d, close=float(px))
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                ma_type="ema",
                sma_window=5,
                monthly_risk_budget_enabled=True,
                monthly_risk_budget_pct=0.01,
                monthly_risk_budget_include_new_trade_risk=True,
                cost_bps=0.0,
            ),
        )
    ts = out.get("trade_statistics") or {}
    overall = ts.get("overall") or {}
    by_code = (ts.get("by_code") or {}).get(code) or {}
    assert int(overall.get("monthly_risk_budget_blocked_entry_count") or 0) > 0
    attempted = int(overall.get("monthly_risk_budget_attempted_entry_count") or 0)
    blocked = int(overall.get("monthly_risk_budget_blocked_entry_count") or 0)
    rate = float(overall.get("monthly_risk_budget_blocked_entry_rate") or 0.0)
    assert attempted >= blocked >= 0
    assert 0.0 <= rate <= 1.0
    if attempted > 0:
        assert rate == pytest.approx(blocked / attempted, rel=1e-6, abs=1e-9)
    assert int(by_code.get("monthly_risk_budget_blocked_entry_count") or 0) > 0
    by_attempted = int(by_code.get("monthly_risk_budget_attempted_entry_count") or 0)
    by_blocked = int(by_code.get("monthly_risk_budget_blocked_entry_count") or 0)
    by_rate = float(by_code.get("monthly_risk_budget_blocked_entry_rate") or 0.0)
    assert by_attempted >= by_blocked >= 0
    assert 0.0 <= by_rate <= 1.0
    if by_attempted > 0:
        assert by_rate == pytest.approx(by_blocked / by_attempted, rel=1e-6, abs=1e-9)
    m = (out.get("metrics") or {}).get("strategy") or {}
    assert int(m.get("monthly_risk_budget_blocked_entry_count") or 0) > 0


def test_trend_single_monthly_risk_budget_include_new_trade_switch(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=40, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            _add_price(db, code=code, day=d, close=float(100.0 + i * 0.8))
        db.commit()
        out_no_new = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=2,
                ma_type="ema",
                atr_stop_mode="none",
                monthly_risk_budget_enabled=True,
                monthly_risk_budget_pct=0.01,
                monthly_risk_budget_include_new_trade_risk=False,
                cost_bps=0.0,
            ),
        )
        out_with_new = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=2,
                ma_type="ema",
                atr_stop_mode="none",
                monthly_risk_budget_enabled=True,
                monthly_risk_budget_pct=0.01,
                monthly_risk_budget_include_new_trade_risk=True,
                cost_bps=0.0,
            ),
        )
    ts_no = out_no_new.get("trade_statistics") or {}
    ts_yes = out_with_new.get("trade_statistics") or {}
    no_trades = int(((ts_no.get("overall") or {}).get("total_trades") or 0))
    yes_trades = int(((ts_yes.get("overall") or {}).get("total_trades") or 0))
    no_block = int(
        (
            (ts_no.get("overall") or {}).get("monthly_risk_budget_blocked_entry_count")
            or 0
        )
    )
    yes_block = int(
        (
            (ts_yes.get("overall") or {}).get("monthly_risk_budget_blocked_entry_count")
            or 0
        )
    )
    assert no_trades > 0
    assert yes_trades == 0
    assert yes_block > 0
    assert no_block <= yes_block


def test_monthly_risk_position_formula_with_and_without_valid_stop() -> None:
    # Valid stop case (static ATR stop): stop = entry - n*atr = 100 - 2*5 = 90
    # position risk = ((100-90)/100) * 0.30 = 0.03
    r_with_stop = _position_risk_from_stop_params(
        atr_stop_enabled=True,
        atr_mode="static",
        atr_basis="entry",
        atr_n=2.0,
        atr_m=0.5,
        entry_px=100.0,
        entry_atr=5.0,
        curr_close=100.0,
        curr_atr=5.0,
        position_weight=0.30,
        fallback_position_risk=0.01,
    )
    assert r_with_stop == pytest.approx(0.03, rel=0.0, abs=1e-12)

    # No valid stop case (e.g. ATR stop disabled): fixed 1% per position, independent of weight.
    r_no_stop_w10 = _position_risk_from_stop_params(
        atr_stop_enabled=False,
        atr_mode="none",
        atr_basis="latest",
        atr_n=2.0,
        atr_m=0.5,
        entry_px=100.0,
        entry_atr=5.0,
        curr_close=100.0,
        curr_atr=5.0,
        position_weight=0.10,
        fallback_position_risk=0.01,
    )
    r_no_stop_w60 = _position_risk_from_stop_params(
        atr_stop_enabled=False,
        atr_mode="none",
        atr_basis="latest",
        atr_n=2.0,
        atr_m=0.5,
        entry_px=100.0,
        entry_atr=5.0,
        curr_close=100.0,
        curr_atr=5.0,
        position_weight=0.60,
        fallback_position_risk=0.01,
    )
    assert r_no_stop_w10 == pytest.approx(0.01, rel=0.0, abs=1e-12)
    assert r_no_stop_w60 == pytest.approx(0.01, rel=0.0, abs=1e-12)


def test_trend_single_risk_budget_vol_regime_dynamic_adjust_counts(session_factory):
    sf = session_factory
    code = "RBVR1"
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=180, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            px = 100.0 + i * 0.4
            if i < 70:
                hi = px * 1.005
                lo = px * 0.995
            elif i < 100:
                hi = px * 1.08
                lo = px * 0.92
            else:
                hi = px * 1.01
                lo = px * 0.99
            _add_price_hl(db, code=code, day=d, close=px, high=hi, low=lo)
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=2,
                position_sizing="risk_budget",
                risk_budget_atr_window=20,
                risk_budget_pct=0.01,
                vol_regime_risk_mgmt_enabled=True,
                vol_ratio_fast_atr_window=5,
                vol_ratio_slow_atr_window=50,
                vol_ratio_expand_threshold=1.45,
                vol_ratio_contract_threshold=0.65,
                vol_ratio_normal_threshold=1.05,
                cost_bps=0.0,
            ),
        )
    ts = out.get("trade_statistics") or {}
    overall = ts.get("overall") or {}
    by_code = (ts.get("by_code") or {}).get(code, {})
    assert int(overall.get("vol_risk_adjust_total_count") or 0) > 0
    assert int(overall.get("vol_risk_adjust_reduce_on_expand_count") or 0) > 0
    assert int(overall.get("vol_risk_adjust_recover_from_expand_count") or 0) > 0
    assert int(by_code.get("vol_risk_adjust_total_count") or 0) == int(
        overall.get("vol_risk_adjust_total_count") or 0
    )


def test_monthly_risk_position_formula_entry_le_stop_returns_zero_not_fallback() -> (
    None
):
    # Trailing mode with entry basis can produce stop >= entry when price rises enough.
    # Example: entry=100, entry_atr=5, n=2 => stop = curr_close - 10.
    # Use curr_close=120 => stop=110 >= entry, so risk must be 0 (not fallback 1%).
    r = _position_risk_from_stop_params(
        atr_stop_enabled=True,
        atr_mode="trailing",
        atr_basis="entry",
        atr_n=2.0,
        atr_m=0.5,
        entry_px=100.0,
        entry_atr=5.0,
        curr_close=120.0,
        curr_atr=5.0,
        position_weight=0.40,
        fallback_position_risk=0.01,
    )
    assert r == pytest.approx(0.0, rel=0.0, abs=1e-12)


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
    params = (out_with_filter.get("meta") or {}).get("params") or {}
    assert params.get("er_filter") is True
    assert int(params.get("er_window") or 0) == 10
    ts = out_with_filter.get("trade_statistics") or {}
    overall = ts.get("overall") or {}
    by_code = (ts.get("by_code") or {}).get(code, {})
    assert int(overall.get("er_filter_blocked_entry_count") or 0) > 0
    assert int(overall.get("er_filter_attempted_entry_count") or 0) >= int(
        overall.get("er_filter_blocked_entry_count") or 0
    )
    assert int(overall.get("er_filter_allowed_entry_count") or 0) >= 0
    assert int(by_code.get("er_filter_blocked_entry_count") or 0) > 0


def test_impulse_entry_filter_blocks_by_state_counts_are_split() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    raw_pos = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], index=idx, dtype=float)
    impulse_state = pd.Series(
        ["BULL", "BEAR", "NEUTRAL", "BULL", "BEAR", "NEUTRAL"], index=idx, dtype=object
    )
    out, stats = _apply_impulse_entry_filter(
        raw_pos,
        impulse_state=impulse_state,
        allow_bull=False,
        allow_bear=False,
        allow_neutral=False,
    )
    assert all(float(x) == 0.0 for x in out.tolist())
    assert int(stats.get("attempted_entry_count") or 0) == 3
    assert int(stats.get("allowed_entry_count") or 0) == 0
    assert int(stats.get("blocked_entry_count") or 0) == 3
    assert int(stats.get("blocked_entry_count_bull") or 0) == 1
    assert int(stats.get("blocked_entry_count_bear") or 0) == 1
    assert int(stats.get("blocked_entry_count_neutral") or 0) == 1


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

    params = (out.get("meta") or {}).get("params") or {}
    assert params.get("er_exit_filter") is True
    assert int(params.get("er_exit_window") or 0) == 10
    ts = out.get("trade_statistics") or {}
    overall = ts.get("overall") or {}
    by_code = (ts.get("by_code") or {}).get(code, {})
    assert int(overall.get("er_exit_filter_trigger_count") or 0) > 0
    assert int(by_code.get("er_exit_filter_trigger_count") or 0) > 0
    er_exit_rc = (out.get("risk_controls") or {}).get("er_exit_filter") or {}
    assert int(er_exit_rc.get("trigger_count") or 0) > 0
    assert isinstance(er_exit_rc.get("trace_last_rows") or [], list)


def test_trend_single_impulse_entry_filter_blocks_entries(session_factory):
    sf = session_factory
    code = "IMP1"
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=100, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            _add_price(db, code=code, day=d, close=100.0 + i * 0.6)
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=2,
                impulse_entry_filter=True,
                impulse_allow_bull=False,
                impulse_allow_bear=False,
                impulse_allow_neutral=False,
                cost_bps=0.0,
            ),
        )
    pos = [float(x) for x in ((out.get("signals") or {}).get("position") or [])]
    assert all(x == 0.0 for x in pos)
    params = (out.get("meta") or {}).get("params") or {}
    assert params.get("impulse_entry_filter") is True
    ts = out.get("trade_statistics") or {}
    overall = ts.get("overall") or {}
    by_code = (ts.get("by_code") or {}).get(code, {})
    blocked = int(overall.get("impulse_filter_blocked_entry_count") or 0)
    attempted = int(overall.get("impulse_filter_attempted_entry_count") or 0)
    rate = float(overall.get("impulse_filter_blocked_entry_rate") or 0.0)
    blocked_split = (
        int(overall.get("impulse_filter_blocked_entry_count_bull") or 0)
        + int(overall.get("impulse_filter_blocked_entry_count_bear") or 0)
        + int(overall.get("impulse_filter_blocked_entry_count_neutral") or 0)
    )
    assert blocked > 0
    assert attempted >= blocked >= 0
    assert 0.0 <= rate <= 1.0
    if attempted > 0:
        assert rate == pytest.approx(blocked / attempted, rel=1e-6, abs=1e-9)
    assert blocked_split == blocked
    by_blocked = int(by_code.get("impulse_filter_blocked_entry_count") or 0)
    by_attempted = int(by_code.get("impulse_filter_attempted_entry_count") or 0)
    by_rate = float(by_code.get("impulse_filter_blocked_entry_rate") or 0.0)
    assert by_blocked == blocked
    assert by_attempted >= by_blocked >= 0
    assert 0.0 <= by_rate <= 1.0
    if by_attempted > 0:
        assert by_rate == pytest.approx(by_blocked / by_attempted, rel=1e-6, abs=1e-9)


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
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=20,
                cost_bps=0.0,
            ),
        )
    assert out["meta"]["type"] == "trend_backtest"
    assert out["meta"]["code"] == code
    assert "nav" in out and "series" in out["nav"]
    s = out["nav"]["series"]
    assert (
        len(out["nav"]["dates"])
        == len(s["STRAT"])
        == len(s["BUY_HOLD"])
        == len(s["EXCESS"])
    )
    assert "event_study" in out
    assert "avg_daily_turnover" in out["metrics"]["strategy"]
    assert "avg_annual_turnover" in out["metrics"]["strategy"]
    assert "avg_daily_trade_count" in out["metrics"]["strategy"]
    assert "avg_annual_trade_count" in out["metrics"]["strategy"]
    assert (out.get("market_regime") or {}).get("enabled") is True
    assert "strategy_state_contribution" in (out.get("market_regime") or {})
    assert set((out["event_study"] or {}).get("windows", {}).keys()) >= {
        "1d",
        "5d",
        "10d",
        "20d",
    }
    ev1 = ((out.get("event_study") or {}).get("windows") or {}).get("1d") or {}
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
    recent = r_stats.get("recent_100") or {}
    assert int(recent.get("effective_count") or 0) <= int(
        (r_stats.get("overall") or {}).get("trade_count") or 0
    )
    assert "sqn" in (r_stats.get("overall") or {})
    assert "trade_system_score" not in r_stats
    ts = out.get("trade_statistics") or {}
    ecs = ts.get("entry_condition_stats") or {}
    assert "overall" in ecs
    assert "by_code" in ecs
    assert "momentum" in (ecs.get("overall") or {})
    trades = list((ts.get("trades") or []))
    if trades:
        t0 = trades[0]
        assert "initial_r_amount" in t0
        assert "initial_r_pct_nav" in t0
        assert "pnl_amount" in t0
        assert "r_multiple" in t0
        assert "entry_signal_date" in t0
        bins = t0.get("entry_condition_bins") or {}
        assert "momentum" in bins
        assert "er" in bins
        assert "vol_ratio" in bins
        assert "impulse" in bins


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


def test_trend_quick_mode_contains_mfe_r_distribution(session_factory):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=120, freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            close = 100.0 + (i * 0.35 if i < 70 else (70 * 0.35) - (i - 70) * 0.50)
            _add_price_hl(
                db, code=code, day=d, close=close, high=close * 1.02, low=close * 0.99
            )
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=10,
                atr_stop_window=5,
                quick_mode=True,
                exec_price="close",
                cost_bps=0.0,
                slippage_rate=0.0,
            ),
        )
    ts = out.get("trade_statistics") or {}
    mfe = ts.get("mfe_r_distribution") or {}
    overall = mfe.get("overall") or {}
    by_code = mfe.get("by_code") or {}
    recent = mfe.get("recent_100") or {}
    assert int(overall.get("trade_count") or 0) > 0
    assert int(overall.get("valid_mfe_count") or 0) > 0
    assert len((((overall.get("samples") or {}).get("mfe_r_multiple")) or [])) == int(
        overall.get("valid_mfe_count") or 0
    )
    assert str(code) in by_code
    assert int(recent.get("effective_count") or 0) <= int(
        overall.get("trade_count") or 0
    )
    assert ts.get("trades") == []


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
    lo_trades = int(
        (
            ((out_lo.get("trade_statistics") or {}).get("overall") or {}).get(
                "total_trades"
            )
            or 0
        )
    )
    hi_trades = int(
        (
            ((out_hi.get("trade_statistics") or {}).get("overall") or {}).get(
                "total_trades"
            )
            or 0
        )
    )
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
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="linreg_slope",
                sma_window=30,
                cost_bps=0.0,
            ),
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


def test_trend_nav_uses_none_with_hfq_fallback_on_corporate_action_cliff(
    session_factory,
):
    sf = session_factory
    code = "AAA"
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-01-10", freq="B")]
    # Build a fake split-like cliff in none price; hfq remains smooth; qfq used for signals remains smooth.
    none_px = [100, 101, 50, 51, 52, 53, 54, 55]
    hfq_px = [100, 101, 102, 103, 104, 105, 106, 107]
    qfq_px = [100, 101, 102, 103, 104, 105, 106, 107]
    with sf() as db:
        for d, n, h, q in zip(dates, none_px, hfq_px, qfq_px):
            db.add(
                EtfPrice(
                    code=code,
                    trade_date=d,
                    open=float(n),
                    high=float(n),
                    low=float(n),
                    close=float(n),
                    volume=1.0,
                    amount=1.0,
                    source="eastmoney",
                    adjust="none",
                )
            )
            db.add(
                EtfPrice(
                    code=code,
                    trade_date=d,
                    open=float(h),
                    high=float(h),
                    low=float(h),
                    close=float(h),
                    volume=1.0,
                    amount=1.0,
                    source="eastmoney",
                    adjust="hfq",
                )
            )
            db.add(
                EtfPrice(
                    code=code,
                    trade_date=d,
                    open=float(q),
                    high=float(q),
                    low=float(q),
                    close=float(q),
                    volume=1.0,
                    amount=1.0,
                    source="eastmoney",
                    adjust="qfq",
                )
            )
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=2,
                cost_bps=0.0,
            ),
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
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="macd_cross",
                cost_bps=0.0,
            ),
        )
        out_zero = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="macd_zero_filter",
                cost_bps=0.0,
            ),
        )
        out_v = compute_trend_backtest(
            db,
            TrendInputs(
                code=code,
                start=dates[0],
                end=dates[-1],
                strategy="macd_v",
                cost_bps=0.0,
            ),
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
        (
            "bias",
            {
                "bias_ma_window": 2,
                "bias_entry": 1.0,
                "bias_hot": 50.0,
                "bias_cold": -10.0,
                "bias_pos_mode": "binary",
            },
        ),
        ("macd_cross", {"macd_fast": 2, "macd_slow": 3, "macd_signal": 2}),
        ("macd_zero_filter", {"macd_fast": 2, "macd_slow": 3, "macd_signal": 2}),
        (
            "macd_v",
            {
                "macd_fast": 2,
                "macd_slow": 3,
                "macd_signal": 2,
                "macd_v_atr_window": 2,
                "macd_v_scale": 100.0,
            },
        ),
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
            assert nav[3] <= 1.0000001, (
                f"{strat} appears to include decision-day return"
            )


def test_random_entry_signal_generator_is_deterministic() -> None:
    idx = pd.bdate_range("2024-01-01", periods=20)
    pos = _pos_from_random_entry_hold(idx, hold_days=3, seed=1)
    assert [int(x) for x in pos.tolist()] == [
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
    ]


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
    params = (out_a.get("meta") or {}).get("params") or {}
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
    params = (out.get("meta") or {}).get("params") or {}
    assert params.get("random_seed") is None


def test_r_take_profit_triggers_on_peak_drawdown_with_virtual_atr_fallback() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    base_pos = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0, 1.0], index=idx, dtype=float)
    close = pd.Series(
        [100.0, 100.0, 110.0, 120.0, 130.0, 120.0], index=idx, dtype=float
    )
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
                bias_v_take_profit_enabled=True,
                bias_v_take_profit_reentry_mode="reenter",
                bias_v_ma_window=20,
                bias_v_atr_window=20,
                bias_v_take_profit_threshold=3.0,
                cost_bps=0.0,
            ),
        )
    rtp = (out.get("risk_controls") or {}).get("r_take_profit") or {}
    assert bool(rtp.get("enabled")) is True
    assert str(rtp.get("initial_r_mode") or "") == "virtual_atr_fallback"
    assert isinstance((rtp.get("tier_trigger_counts") or {}), dict)
    params = (out.get("meta") or {}).get("params") or {}
    assert bool(params.get("r_take_profit_enabled")) is True
    assert bool(params.get("bias_v_take_profit_enabled")) is True
    bv_rc = (out.get("risk_controls") or {}).get("bias_v_take_profit") or {}
    assert bool(bv_rc.get("enabled")) is True
    metrics = (out.get("metrics") or {}).get("strategy") or {}
    assert "r_take_profit_trigger_count" not in metrics
    ts = out.get("trade_statistics") or {}
    overall = ts.get("overall") or {}
    by_code = (ts.get("by_code") or {}).get(code, {})
    assert "atr_stop_trigger_count" in overall
    assert "r_take_profit_trigger_count" in overall
    assert "bias_v_take_profit_trigger_count" in overall
    assert isinstance((overall.get("r_take_profit_tier_trigger_counts") or {}), dict)
    assert "atr_stop_trigger_count" in by_code
    assert "r_take_profit_trigger_count" in by_code
    assert "bias_v_take_profit_trigger_count" in by_code
