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
