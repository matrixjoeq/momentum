from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from ..strategy.rotation import RotationInputs, backtest_rotation


@dataclass(frozen=True)
class RotationAnalysisInputs:
    codes: list[str]
    start: dt.date
    end: dt.date
    rebalance: str = "weekly"
    rebalance_anchor: int | None = None  # weekly:1..5; monthly:1..28; quarterly:1..90; yearly:1..365
    rebalance_shift: str = "prev"  # prev|next|skip when anchor falls on non-trading day
    top_k: int = 1  # non-zero: >0 top-K, <0 bottom-K; effective=min(|k|, pool)
    position_mode: str = "adaptive"  # adaptive|fixed|risk_budget
    risk_budget_atr_window: int = 20
    risk_budget_pct: float = 0.01
    entry_backfill: bool = False
    entry_match_n: int = 0
    exit_match_n: int = 0
    lookback_days: int = 20
    skip_days: int = 0
    risk_free_rate: float = 0.025
    cost_bps: float = 0.0
    slippage_rate: float = 0.001
    atr_stop_mode: str = "none"
    atr_stop_atr_basis: str = "latest"
    atr_stop_reentry_mode: str = "reenter"
    atr_stop_window: int = 14
    atr_stop_n: float = 2.0
    atr_stop_m: float = 0.5
    score_method: str = "raw_mom"
    # Risk controls (defaults off)
    trend_filter: bool = False
    trend_exit_filter: bool = False
    trend_sma_window: int = 20
    trend_ma_type: str = "sma"
    bias_filter: bool = False
    bias_exit_filter: bool = False
    bias_type: str = "bias"
    bias_ma_window: int = 20
    bias_level_window: str = "all"
    bias_threshold_type: str = "quantile"
    bias_quantile: float = 95.0
    bias_fixed_value: float = 10.0
    bias_min_periods: int = 20
    group_enforce: bool = False
    group_pick_policy: str = "strongest_score"
    asset_groups: dict[str, str] | None = None
    dynamic_universe: bool = False
    exec_price: str = "open"  # close|open|oc2
    # Phase-1 per-asset parameter rules (optional)
    asset_momentum_floor_rules: list[dict[str, Any]] | None = None
    asset_trend_rules: list[dict[str, Any]] | None = None
    asset_bias_rules: list[dict[str, Any]] | None = None
    asset_rc_rules: list[dict[str, Any]] | None = None
    asset_vol_index_rules: list[dict[str, Any]] | None = None
    vol_index_close: dict[str, Any] | None = None


def compute_rotation_backtest(db: Session, inp: RotationAnalysisInputs) -> dict[str, Any]:
    # Pylint may resolve imported dataclasses from an installed package instead of workspace source,
    # which can lag during local dev. Keep behavior correct; suppress false-positive for new fields.
    # pylint: disable=unexpected-keyword-arg
    return backtest_rotation(
        db,
        RotationInputs(
            codes=inp.codes,
            start=inp.start,
            end=inp.end,
            rebalance=inp.rebalance,
            rebalance_anchor=inp.rebalance_anchor,
            rebalance_shift=inp.rebalance_shift,
            top_k=inp.top_k,
            position_mode=inp.position_mode,
            risk_budget_atr_window=inp.risk_budget_atr_window,
            risk_budget_pct=inp.risk_budget_pct,
            entry_backfill=inp.entry_backfill,
            entry_match_n=inp.entry_match_n,
            exit_match_n=inp.exit_match_n,
            lookback_days=inp.lookback_days,
            skip_days=inp.skip_days,
            risk_free_rate=inp.risk_free_rate,
            cost_bps=inp.cost_bps,
            slippage_rate=inp.slippage_rate,
            atr_stop_mode=inp.atr_stop_mode,
            atr_stop_atr_basis=inp.atr_stop_atr_basis,
            atr_stop_reentry_mode=inp.atr_stop_reentry_mode,
            atr_stop_window=inp.atr_stop_window,
            atr_stop_n=inp.atr_stop_n,
            atr_stop_m=inp.atr_stop_m,
            score_method=inp.score_method,
            trend_filter=inp.trend_filter,
            trend_exit_filter=inp.trend_exit_filter,
            trend_sma_window=inp.trend_sma_window,
            trend_ma_type=inp.trend_ma_type,
            bias_filter=inp.bias_filter,
            bias_exit_filter=inp.bias_exit_filter,
            bias_type=inp.bias_type,
            bias_ma_window=inp.bias_ma_window,
            bias_level_window=inp.bias_level_window,
            bias_threshold_type=inp.bias_threshold_type,
            bias_quantile=inp.bias_quantile,
            bias_fixed_value=inp.bias_fixed_value,
            bias_min_periods=inp.bias_min_periods,
            group_enforce=inp.group_enforce,
            group_pick_policy=inp.group_pick_policy,
            asset_groups=inp.asset_groups,
            dynamic_universe=inp.dynamic_universe,
            exec_price=inp.exec_price,
            asset_momentum_floor_rules=inp.asset_momentum_floor_rules,
            asset_trend_rules=inp.asset_trend_rules,
            asset_bias_rules=inp.asset_bias_rules,
            asset_rc_rules=inp.asset_rc_rules,
            asset_vol_index_rules=inp.asset_vol_index_rules,
            vol_index_close=inp.vol_index_close,
        ),
    )

