"""Unit tests for discrete-lot futures account simulator."""

from __future__ import annotations

import pandas as pd

from etf_momentum.analysis.futures_lot_account import simulate_discrete_lot_portfolio
from etf_momentum.analysis.futures_trend import CostProfile


def _noop_cost() -> CostProfile:
    return CostProfile(
        commission_per_fill=0.0,
        spread_per_fill=0.0,
        commission_input_bps=0.0,
        fee_side="one_way",
        slippage_type="percent",
        slippage_input=0.0,
        slippage_side="one_way",
        contract_multiplier=10.0,
        min_price_tick=1.0,
    )


def test_simulate_produces_equity_series_same_index() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    code = "X0"
    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0],
            "High": [100.0, 100.0, 100.0],
            "Low": [100.0, 100.0, 100.0],
            "Close": [100.0, 100.0, 100.0],
            "Volume": [1.0, 1.0, 1.0],
            "Settle": [100.0, 100.0, 100.0],
        },
        index=idx,
    )
    w_eff = pd.DataFrame({code: [0.0, 1.0, 1.0]}, index=idx)
    eq, meta = simulate_discrete_lot_portfolio(
        common_idx=idx,
        exec_by_code={code: df},
        w_eff=w_eff,
        cost_by_symbol={code: _noop_cost()},
        mults={code: 10.0},
        margin_rate_frac=0.15,
        reserve_ratio=0.5,
        initial_equity_cny=1_000_000.0,
        exec_price="close",
        position_sizing="equal",
        codes_sorted=[code],
    )
    assert len(eq) == len(idx)
    assert meta.get("engine") == "lot_account"
    assert float(eq.iloc[-1]) == float(eq.iloc[0])
