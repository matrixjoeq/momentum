from __future__ import annotations

import pandas as pd

from etf_momentum.analysis.futures_trend import (
    _build_cost_profile,
    _resolve_order_size,
    _run_symbol_backtest,
    _run_vectorized_fallback,
)


def _build_monotonic_ohlc() -> pd.DataFrame:
    idx = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]
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
    ret = _run_vectorized_fallback(df, fast_ma=1, slow_ma=2, exec_price="close")
    non_zero_dates = [d for d, v in ret.items() if abs(float(v)) > 0]
    assert non_zero_dates, "expected non-zero returns after signal activation"
    # Signal turns valid from 2nd bar onward; with t+1 execution and close-close legs,
    # first realized return appears on the 4th bar.
    assert non_zero_dates[0] == df.index[3]


def test_vectorized_fallback_uses_next_day_execution_lag_for_open() -> None:
    df = _build_monotonic_ohlc()
    ret = _run_vectorized_fallback(df, fast_ma=1, slow_ma=2, exec_price="open")
    non_zero_dates = [d for d, v in ret.items() if abs(float(v)) > 0]
    assert non_zero_dates, "expected non-zero returns after signal activation"
    assert non_zero_dates[0] == df.index[3]


def test_resolve_order_size_maps_full_position_to_fractional_equity_size() -> None:
    assert _resolve_order_size(1.0) < 1.0
    assert _resolve_order_size(1.0) > 0.9
    assert _resolve_order_size(0.25) == 0.25


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
