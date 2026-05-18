from __future__ import annotations

import pandas as pd

from etf_momentum.analysis.futures_trend_portfolio_weights import (
    build_ma_panels,
    equal_weights_from_signals,
    risk_budget_weights,
)


def test_build_ma_panels_short_only_negative_signal() -> None:
    idx = pd.date_range("2024-01-02", periods=10, freq="D")
    # Declining closes → fast MA below slow MA → short regime → -1
    closes = [float(100 - i) for i in range(10)]
    exec_by = {
        "X": pd.DataFrame(
            {
                "SignalClose": closes,
                "Close": closes,
                "Open": closes,
                "High": [c + 1.0 for c in closes],
                "Low": [c - 1.0 for c in closes],
            },
            index=idx,
        )
    }
    score, sig = build_ma_panels(
        exec_by,
        common_idx=idx,
        fast_ma=2,
        slow_ma=4,
        ma_type="sma",
        trade_direction="short_only",
        short_entry_filter_ma=2,
    )
    assert score.shape == sig.shape
    last = sig["X"].iloc[-1]
    assert float(last) < -0.5


def test_build_ma_panels_flattens_signal_on_missing_bar() -> None:
    idx = pd.date_range("2024-01-02", periods=8, freq="D")
    closes = [100.0, 101.0, 102.0, 103.0, float("nan"), float("nan"), 106.0, 107.0]
    exec_by = {
        "X": pd.DataFrame(
            {
                "SignalClose": closes,
                "Close": closes,
                "Open": closes,
                "High": [c + 1.0 if pd.notna(c) else float("nan") for c in closes],
                "Low": [c - 1.0 if pd.notna(c) else float("nan") for c in closes],
            },
            index=idx,
        )
    }
    _, sig = build_ma_panels(
        exec_by,
        common_idx=idx,
        fast_ma=2,
        slow_ma=3,
        ma_type="sma",
        trade_direction="long_only",
        long_entry_filter_ma=2,
    )
    # Pre-missing bars can enter long regime, but missing bars must flatten to 0.
    assert float(sig.loc[idx[3], "X"]) > 0.5
    assert abs(float(sig.loc[idx[4], "X"])) < 1e-12
    assert abs(float(sig.loc[idx[5], "X"])) < 1e-12


def test_build_ma_panels_entry_filter_can_be_disabled() -> None:
    idx = pd.date_range("2024-01-02", periods=8, freq="D")
    closes = [float(100 + i) for i in range(8)]
    exec_by = {
        "X": pd.DataFrame(
            {
                "SignalClose": closes,
                "Close": closes,
                "Open": closes,
                "High": [c + 1.0 for c in closes],
                "Low": [c - 1.0 for c in closes],
            },
            index=idx,
        )
    }
    # Large filter windows keep filter MA as NaN in this short sample.
    # Enabled filter rejects entries; disabled filter should still allow trend entry.
    _, sig_on = build_ma_panels(
        exec_by,
        common_idx=idx,
        fast_ma=2,
        slow_ma=3,
        ma_type="sma",
        trade_direction="long_only",
        entry_filter_enabled=True,
        long_entry_filter_ma=200,
        short_entry_filter_ma=200,
    )
    _, sig_off = build_ma_panels(
        exec_by,
        common_idx=idx,
        fast_ma=2,
        slow_ma=3,
        ma_type="sma",
        trade_direction="long_only",
        entry_filter_enabled=False,
        long_entry_filter_ma=200,
        short_entry_filter_ma=200,
    )
    assert abs(float(sig_on["X"].iloc[-1])) < 1e-12
    assert float(sig_off["X"].iloc[-1]) > 0.5


def test_build_ma_panels_kama_filter_supports_both_reversal() -> None:
    idx = pd.date_range("2024-01-02", periods=12, freq="D")
    closes = [
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
        103.0,
        100.0,
        97.0,
        95.0,
        96.0,
        98.0,
    ]
    exec_by = {
        "X": pd.DataFrame(
            {
                "SignalClose": closes,
                "Close": closes,
                "Open": closes,
                "High": [c + 1.0 for c in closes],
                "Low": [c - 1.0 for c in closes],
            },
            index=idx,
        )
    }
    _, sig = build_ma_panels(
        exec_by,
        common_idx=idx,
        trend_strategy="ma_filter",
        fast_ma=2,
        slow_ma=3,
        ma_type="kama",
        trade_direction="both",
        kama_er_window=3,
        kama_fast_window=2,
        kama_slow_window=10,
        kama_std_window=3,
        kama_std_coef=0.0,
    )
    arr = sig["X"].astype(float).tolist()
    assert any(v > 0.5 for v in arr)
    assert any(v < -0.5 for v in arr)
    first_long = next(i for i, v in enumerate(arr) if v > 0.5)
    first_short = next(i for i, v in enumerate(arr) if v < -0.5)
    assert first_short > first_long


def test_build_ma_panels_ma_cross_supports_both_reversal() -> None:
    idx = pd.date_range("2024-01-02", periods=12, freq="D")
    closes = [
        100.0,
        99.0,
        98.0,
        97.0,
        96.0,
        95.0,
        97.0,
        99.0,
        101.0,
        102.0,
        101.0,
        100.0,
    ]
    exec_by = {
        "X": pd.DataFrame(
            {
                "SignalClose": closes,
                "Close": closes,
                "Open": closes,
                "High": [c + 1.0 for c in closes],
                "Low": [c - 1.0 for c in closes],
            },
            index=idx,
        )
    }
    _, sig = build_ma_panels(
        exec_by,
        common_idx=idx,
        trend_strategy="ma_cross",
        fast_ma=2,
        slow_ma=4,
        ma_type="sma",
        trade_direction="both",
    )
    arr = sig["X"].astype(float).tolist()
    assert any(v > 0.5 for v in arr)
    assert any(v < -0.5 for v in arr)
    first_short = next(i for i, v in enumerate(arr) if v < -0.5)
    first_long = next(i for i, v in enumerate(arr) if v > 0.5)
    assert first_long > first_short


def test_equal_weights_signed_both_mode() -> None:
    idx = pd.to_datetime(["2024-01-02"])
    sig = pd.DataFrame({"A": [1.0], "B": [-1.0]}, index=idx)
    w = equal_weights_from_signals(sig)
    assert abs(float(w.loc[idx[0], "A"]) - 0.5) < 1e-9
    assert abs(float(w.loc[idx[0], "B"]) + 0.5) < 1e-9


def test_risk_budget_flattens_inactive_short_position() -> None:
    idx = pd.date_range("2024-01-02", periods=5, freq="D")
    sig = pd.DataFrame({"X": [0.0, -1.0, 0.0, 0.0, 0.0]}, index=idx, dtype=float)
    score = pd.DataFrame({"X": [0.0, 1.0, 0.0, 0.0, 0.0]}, index=idx, dtype=float)
    close = pd.Series([100.0, 99.0, 98.0, 97.0, 96.0], index=idx, dtype=float)
    exec_by = {
        "X": pd.DataFrame(
            {
                "Open": close.values,
                "High": (close + 1.0).values,
                "Low": (close - 1.0).values,
                "Close": close.values,
            },
            index=idx,
        )
    }
    w_out, _ = risk_budget_weights(
        sig_direction_df=sig,
        score_df=score,
        exec_by_code=exec_by,
        common_idx=idx,
        risk_budget_atr_window=2,
        risk_budget_pct=0.01,
        policy="scale",
        max_leverage_multiple=2.0,
    )
    assert float(w_out.loc[idx[1], "X"]) < 0.0
    assert abs(float(w_out.loc[idx[2], "X"])) < 1e-12
