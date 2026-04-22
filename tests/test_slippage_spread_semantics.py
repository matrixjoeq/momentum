from __future__ import annotations

import pandas as pd

from etf_momentum.analysis.execution_timing import slippage_return_from_turnover
from etf_momentum.analysis.trend import (
    _trade_returns_from_weight_series as trend_trade_returns,
)
from etf_momentum.strategy.rotation import (
    _trade_returns_from_weight_series as rotation_trade_returns,
)


def test_slippage_spread_scales_with_execution_price():
    idx = pd.date_range("2026-01-05", periods=2, freq="B")
    turnover = pd.Series([0.5, 0.5], index=idx, dtype=float)
    px_low = pd.Series([1.0, 1.0], index=idx, dtype=float)
    px_high = pd.Series([100.0, 100.0], index=idx, dtype=float)

    slip_low = slippage_return_from_turnover(
        turnover, slippage_spread=0.001, exec_price=px_low
    )
    slip_high = slippage_return_from_turnover(
        turnover, slippage_spread=0.001, exec_price=px_high
    )

    assert float(slip_low.iloc[0]) == 0.0005
    assert float(slip_high.iloc[0]) == 0.000005
    assert float(slip_low.sum()) > float(slip_high.sum())


def test_trend_trade_returns_use_price_spread_slippage():
    idx = pd.date_range("2026-02-02", periods=3, freq="B")
    w = pd.Series([0.0, 1.0, 0.0], index=idx, dtype=float)
    ret_exec = pd.Series([0.0, 0.0, 0.0], index=idx, dtype=float)

    low = trend_trade_returns(
        w,
        ret_exec,
        cost_bps=0.0,
        slippage_rate=0.001,
        exec_price=pd.Series([1.0, 1.0, 1.0], index=idx, dtype=float),
        dates=idx,
    )
    high = trend_trade_returns(
        w,
        ret_exec,
        cost_bps=0.0,
        slippage_rate=0.001,
        exec_price=pd.Series([100.0, 100.0, 100.0], index=idx, dtype=float),
        dates=idx,
    )

    assert len(low["returns"]) == 1
    assert len(high["returns"]) == 1
    assert float(low["returns"][0]) < float(high["returns"][0])


def test_rotation_trade_returns_use_price_spread_slippage():
    idx = pd.date_range("2026-02-02", periods=3, freq="B")
    w = pd.Series([0.0, 1.0, 0.0], index=idx, dtype=float)
    ret_exec = pd.Series([0.0, 0.0, 0.0], index=idx, dtype=float)

    low = rotation_trade_returns(
        w,
        ret_exec,
        cost_bps=0.0,
        slippage_rate=0.001,
        exec_price=pd.Series([1.0, 1.0, 1.0], index=idx, dtype=float),
        dates=idx,
    )
    high = rotation_trade_returns(
        w,
        ret_exec,
        cost_bps=0.0,
        slippage_rate=0.001,
        exec_price=pd.Series([100.0, 100.0, 100.0], index=idx, dtype=float),
        dates=idx,
    )

    assert len(low["returns"]) == 1
    assert len(high["returns"]) == 1
    assert float(low["returns"][0]) < float(high["returns"][0])
