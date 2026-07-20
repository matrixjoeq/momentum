"""Unit tests for discrete-lot futures account simulator."""

from __future__ import annotations

import pandas as pd
import pytest

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


def test_reserve_entry_attempt_counts_only_new_entries_and_close_on_last_day() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    code = "X0"
    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0],
            "High": [101.0, 101.0, 101.0],
            "Low": [99.0, 99.0, 99.0],
            "Close": [100.0, 100.0, 100.0],
            "Volume": [1.0, 1.0, 1.0],
            "Settle": [100.0, 100.0, 100.0],
        },
        index=idx,
    )
    # Keep signal active after first entry; attempted new entries should be counted once.
    w_eff = pd.DataFrame({code: [1.0, 1.0, 1.0]}, index=idx)
    _, meta = simulate_discrete_lot_portfolio(
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
    assert int(meta.get("reserve_margin_attempted_entry_count", 0)) == 1
    assert int(meta.get("reserve_margin_blocked_entry_count", 0)) == 0
    closed = list(meta.get("closed_trades") or [])
    assert len(closed) == 1
    assert closed[0].get("entry_date") == "2024-01-02"
    assert closed[0].get("exit_date") == "2024-01-04"


def test_roll_fees_use_old_and_new_contract_execution_prices() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    code = "X0"
    df = pd.DataFrame(
        {
            "Open": [100.0, 110.0],
            "High": [101.0, 111.0],
            "Low": [99.0, 109.0],
            "Close": [100.0, 110.0],
            "Volume": [1.0, 1.0],
            "Settle": [100.0, 110.0],
            "dominant_contract_suffix": [None, "2405"],
            "roll_from_open": [None, 100.0],
            "roll_to_open": [None, 130.0],
            "roll_from_close": [None, 101.0],
            "roll_to_close": [None, 131.0],
        },
        index=idx,
    )
    w_eff = pd.DataFrame({code: [1.0, 1.0]}, index=idx)
    cost = CostProfile(
        commission_per_fill=0.001,
        spread_per_fill=0.0,
        commission_input_bps=10.0,
        fee_side="one_way",
        slippage_type="percent",
        slippage_input=0.0,
        slippage_side="one_way",
        contract_multiplier=10.0,
        min_price_tick=1.0,
    )
    _, meta = simulate_discrete_lot_portfolio(
        common_idx=idx,
        exec_by_code={code: df},
        w_eff=w_eff,
        cost_by_symbol={code: cost},
        mults={code: 10.0},
        margin_rate_frac=0.15,
        reserve_ratio=0.5,
        initial_equity_cny=1_000_000.0,
        exec_price="open",
        position_sizing="equal",
        codes_sorted=[code],
    )
    # At first day entry lots = floor(500000 / (100*10*0.15)) = 3333
    lots = 3333
    expected = lots * 10.0 * (100.0 + 130.0) * 0.001
    assert meta.get("roll_events") == 1
    assert float(meta.get("roll_fees_cny", 0.0)) == pytest.approx(expected)


def test_open_execution_marks_intraday_pnl_on_post_trade_lots() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    code = "X0"
    df = pd.DataFrame(
        {
            "Open": [100.0, 110.0],
            "High": [100.0, 130.0],
            "Low": [100.0, 110.0],
            "Close": [100.0, 130.0],
            "Volume": [1.0, 1.0],
            "Settle": [100.0, 130.0],
        },
        index=idx,
    )
    w_eff = pd.DataFrame({code: [0.0, 1.0]}, index=idx)
    eq, _ = simulate_discrete_lot_portfolio(
        common_idx=idx,
        exec_by_code={code: df},
        w_eff=w_eff,
        cost_by_symbol={code: _noop_cost()},
        mults={code: 10.0},
        margin_rate_frac=0.1,
        reserve_ratio=0.0,
        initial_equity_cny=100_000.0,
        exec_price="open",
        position_sizing="equal",
        codes_sorted=[code],
    )
    # Day2 entry at open should realize same-day intraday MTM on post-trade lots:
    # open-exec sizing uses execution open (not same-day settle), so
    # lots=floor(100000 / (110*10*0.1))=909, pnl=909*10*(130-110)=181800.
    assert float(eq.iloc[1]) == pytest.approx(281800.0)


def test_open_execution_margin_sizing_does_not_lookahead_same_day_settle() -> None:
    idx = pd.to_datetime(["2024-01-02"])
    code = "X0"
    df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [100.0],
            "Low": [100.0],
            "Close": [100.0],
            "Volume": [1.0],
            "Settle": [1_000_000_000.0],
        },
        index=idx,
    )
    w_eff = pd.DataFrame({code: [1.0]}, index=idx)
    _, meta = simulate_discrete_lot_portfolio(
        common_idx=idx,
        exec_by_code={code: df},
        w_eff=w_eff,
        cost_by_symbol={code: _noop_cost()},
        mults={code: 10.0},
        margin_rate_frac=0.1,
        reserve_ratio=0.0,
        initial_equity_cny=100_000.0,
        exec_price="open",
        position_sizing="equal",
        codes_sorted=[code],
    )
    assert int(meta.get("reserve_margin_attempted_entry_count", 0)) == 1
    assert int(meta.get("reserve_margin_blocked_entry_count", 0)) == 0
    assert len(list(meta.get("closed_trades") or [])) == 1


def test_simulate_forces_liquidation_and_clamps_non_positive_equity() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    code = "X0"
    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0],
            "High": [100.0, 100.0, 100.0],
            "Low": [100.0, 100.0, 100.0],
            "Close": [100.0, 1.0, 1.0],
            "Volume": [1.0, 1.0, 1.0],
            "Settle": [100.0, 1.0, 1.0],
        },
        index=idx,
    )
    w_eff = pd.DataFrame({code: [1.0, 1.0, 1.0]}, index=idx)
    eq, meta = simulate_discrete_lot_portfolio(
        common_idx=idx,
        exec_by_code={code: df},
        w_eff=w_eff,
        cost_by_symbol={code: _noop_cost()},
        mults={code: 100.0},
        margin_rate_frac=0.1,
        reserve_ratio=0.0,
        initial_equity_cny=100_000.0,
        exec_price="close",
        position_sizing="equal",
        codes_sorted=[code],
    )
    assert float(eq.min()) >= 0.0
    assert float(eq.iloc[1]) == 0.0
    assert float(eq.iloc[2]) == 0.0
    assert bool(meta.get("insolvency_forced_liquidation")) is True
    assert str(meta.get("insolvency_first_date")) == "2024-01-03"
