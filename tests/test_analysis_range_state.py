from __future__ import annotations

import pandas as pd

from etf_momentum.analysis.range_state import (
    RANGE_STATE_RANGE,
    RANGE_STATE_TREND,
    RangeStateConfig,
    compute_range_state_monitor,
)


def test_range_state_monitor_er_hysteresis_contract() -> None:
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    # Early segment oscillates around a tight center (low ER), late segment trends up (high ER).
    close = pd.Series(
        [100.0 + ((-1.0) ** i) * 0.9 for i in range(40)]
        + [100.0 + i * 1.2 for i in range(40)],
        index=dates,
    )
    high = close + 0.6
    low = close - 0.6
    out = compute_range_state_monitor(
        high=high,
        low=low,
        close=close,
        config=RangeStateConfig(
            mode="er",
            window=10,
            enter_threshold=0.25,
            exit_threshold=0.45,
        ),
    )
    series = out["series"]
    assert len(series["dates"]) == len(dates)
    assert len(series["state"]) == len(dates)
    assert len(series["range_score"]) == len(dates)
    assert any(s == RANGE_STATE_RANGE for s in series["state"])
    assert any(s == RANGE_STATE_TREND for s in series["state"])
    latest = out["latest"] or {}
    assert latest.get("state") in {RANGE_STATE_RANGE, RANGE_STATE_TREND}
    summary = out["summary"] or {}
    assert int(summary.get("n") or 0) == len(dates)
