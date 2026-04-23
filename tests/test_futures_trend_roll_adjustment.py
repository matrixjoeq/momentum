"""Tests for main-contract roll adjustment on futures trend portfolio returns."""

from __future__ import annotations

import pandas as pd

from etf_momentum.analysis.futures_trend import (
    CostProfile,
    _apply_main_contract_roll_adjustments,
)


def test_roll_day_replaces_return_and_deducts_round_turn_fees() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    c = "M888"
    ret_mat = pd.DataFrame({c: [0.01, 0.02, 0.03]}, index=idx)
    w_eff = pd.DataFrame({c: [0.5, 0.5, 0.5]}, index=idx)
    exec_aligned = {
        c: pd.DataFrame(
            {
                "Close": [3000.0, 3020.0, 2880.0],
                "dominant_contract_suffix": [None, None, "2405"],
            },
            index=idx,
        )
    }
    cost = CostProfile(
        commission_per_fill=0.001,
        spread_per_fill=0.0005,
        commission_input_bps=10.0,
        fee_side="two_way",
        slippage_type="percent",
        slippage_input=0.0005,
        slippage_side="two_way",
    )
    out, meta = _apply_main_contract_roll_adjustments(
        ret_mat,
        w_eff=w_eff,
        exec_aligned=exec_aligned,
        cost_by_symbol={c: cost},
    )
    rt_fee = 2.0 * (0.001 + 0.0005)
    r_cc = 2880.0 / 3020.0 - 1.0
    assert meta["total_roll_events"] == 1
    assert abs(float(out.loc[idx[2], c]) - (r_cc - rt_fee)) < 1e-9
    assert float(out.loc[idx[0], c]) == 0.01


def test_no_suffix_column_skips_adjustment() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    c = "X888"
    ret_mat = pd.DataFrame({c: [0.01, 0.02]}, index=idx)
    w_eff = pd.DataFrame({c: [1.0, 1.0]}, index=idx)
    exec_aligned = {c: pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)}
    cost = CostProfile(
        commission_per_fill=0.0,
        spread_per_fill=0.0,
        commission_input_bps=0.0,
        fee_side="two_way",
        slippage_type="percent",
        slippage_input=0.0,
        slippage_side="two_way",
    )
    out, meta = _apply_main_contract_roll_adjustments(
        ret_mat,
        w_eff=w_eff,
        exec_aligned=exec_aligned,
        cost_by_symbol={c: cost},
    )
    assert meta["total_roll_events"] == 0
    pd.testing.assert_frame_equal(out, ret_mat.astype(float))


def test_roll_day_short_weight_uses_negative_bar_return() -> None:
    """Short leg: stitched close-up day → strategy return ~ −r_cc minus round-trip fees."""
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    c = "M888"
    ret_mat = pd.DataFrame({c: [-0.01, -0.02, -0.03]}, index=idx)
    w_eff = pd.DataFrame({c: [-0.5, -0.5, -0.5]}, index=idx)
    exec_aligned = {
        c: pd.DataFrame(
            {
                "Close": [3000.0, 3020.0, 3080.0],
                "dominant_contract_suffix": [None, None, "2405"],
            },
            index=idx,
        )
    }
    cost = CostProfile(
        commission_per_fill=0.001,
        spread_per_fill=0.0005,
        commission_input_bps=10.0,
        fee_side="two_way",
        slippage_type="percent",
        slippage_input=0.0005,
        slippage_side="two_way",
    )
    out, meta = _apply_main_contract_roll_adjustments(
        ret_mat,
        w_eff=w_eff,
        exec_aligned=exec_aligned,
        cost_by_symbol={c: cost},
    )
    rt_fee = 2.0 * (0.001 + 0.0005)
    r_cc = 3080.0 / 3020.0 - 1.0
    expected = -r_cc - rt_fee
    assert meta["total_roll_events"] == 1
    assert abs(float(out.loc[idx[2], c]) - expected) < 1e-9


def test_zero_weight_skips_roll_fee() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    c = "M888"
    ret_mat = pd.DataFrame({c: [0.01, 0.02]}, index=idx)
    w_eff = pd.DataFrame({c: [0.5, 0.0]}, index=idx)
    exec_aligned = {
        c: pd.DataFrame(
            {
                "Close": [3000.0, 2880.0],
                "dominant_contract_suffix": [None, "2405"],
            },
            index=idx,
        )
    }
    cost = CostProfile(
        commission_per_fill=0.001,
        spread_per_fill=0.001,
        commission_input_bps=10.0,
        fee_side="two_way",
        slippage_type="percent",
        slippage_input=0.001,
        slippage_side="two_way",
    )
    out, meta = _apply_main_contract_roll_adjustments(
        ret_mat,
        w_eff=w_eff,
        exec_aligned=exec_aligned,
        cost_by_symbol={c: cost},
    )
    assert meta["total_roll_events"] == 0
    assert float(out.loc[idx[1], c]) == 0.02
