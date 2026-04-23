from __future__ import annotations

import pandas as pd
import pytest

from etf_momentum.analysis.futures_trend import (
    _build_cost_profile,
    _resolve_order_size,
    _run_symbol_backtest,
    _run_vectorized_fallback,
)


def _build_monotonic_ohlc() -> pd.DataFrame:
    idx = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ]
    )
    close = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    return pd.DataFrame(
        {
            "Open": close,
            "High": [x + 1 for x in close],
            "Low": [x - 1 for x in close],
            "Close": close,
            "Volume": [1000.0] * len(close),
        },
        index=idx,
    )


def test_vectorized_fallback_uses_next_day_execution_lag_for_close() -> None:
    df = _build_monotonic_ohlc()
    ret = _run_vectorized_fallback(df, fast_ma=2, slow_ma=3, exec_price="close")
    non_zero_dates = [d for d, v in ret.items() if abs(float(v)) > 0]
    assert non_zero_dates, "expected non-zero returns after signal activation"
    # Fast/slow crossover valid after slow MA warms up; with t+1 execution and close legs,
    # first realized non-zero return appears on bar index 4 for this synthetic path.
    assert non_zero_dates[0] == df.index[4]


def test_vectorized_fallback_uses_next_day_execution_lag_for_open() -> None:
    df = _build_monotonic_ohlc()
    ret = _run_vectorized_fallback(df, fast_ma=2, slow_ma=3, exec_price="open")
    non_zero_dates = [d for d, v in ret.items() if abs(float(v)) > 0]
    assert non_zero_dates, "expected non-zero returns after signal activation"
    assert non_zero_dates[0] == df.index[4]


def test_resolve_order_size_maps_full_position_to_fractional_equity_size() -> None:
    assert _resolve_order_size(1.0) < 1.0
    assert _resolve_order_size(1.0) > 0.9
    assert _resolve_order_size(0.25) == 0.25


def test_build_cost_profile_tick_multiple_uses_min_price_tick_per_fill() -> None:
    cp = _build_cost_profile(
        cost_bps=4.0,
        fee_side="one_way",
        slippage_type="tick_multiple",
        slippage_value=2.0,
        slippage_side="one_way",
        price_reference=4000.0,
        contract_multiplier=10.0,
        min_price_tick=0.5,
    )
    assert cp.slippage_tick_multiple == 2
    assert cp.tick_value_per_lot == pytest.approx(5.0)
    # per-fill adverse return ≈ 2 * 0.5 / 4000
    assert cp.spread_per_fill == pytest.approx(0.00025)
    assert cp.commission_per_fill == pytest.approx(0.0004)


def test_build_cost_profile_tick_multiple_rejects_non_integer_multiple() -> None:
    with pytest.raises(ValueError, match="integer"):
        _build_cost_profile(
            cost_bps=0.0,
            fee_side="one_way",
            slippage_type="tick_multiple",
            slippage_value=1.5,
            slippage_side="one_way",
            price_reference=100.0,
            min_price_tick=1.0,
        )


def test_vectorized_fallback_accepts_cost_profile() -> None:
    df = _build_monotonic_ohlc()
    cost = _build_cost_profile(
        cost_bps=10.0,
        fee_side="one_way",
        slippage_type="percent",
        slippage_value=0.0,
        slippage_side="one_way",
        price_reference=float(df["Close"].median()),
    )
    ret = _run_vectorized_fallback(
        df,
        fast_ma=2,
        slow_ma=3,
        exec_price="close",
        cost=cost,
    )
    assert len(ret) == len(df.index)


def test_symbol_backtest_finalizes_open_trade_for_stats() -> None:
    df = _build_monotonic_ohlc()
    cost = _build_cost_profile(
        cost_bps=5.0,
        fee_side="two_way",
        slippage_type="percent",
        slippage_value=0.0005,
        slippage_side="two_way",
        price_reference=float(df["Close"].median()),
    )
    nav, st = _run_symbol_backtest(
        df,
        fast_ma=2,
        slow_ma=3,
        exec_price="close",
        position_size_pct=1.0,
        cost=cost,
    )
    assert nav.iloc[-1] > 1.0
    if st.get("engine") == "backtesting":
        assert int(st.get("trades", 0)) >= 1
