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
    top_k: int = 1
    lookback_days: int = 20
    skip_days: int = 0
    risk_free_rate: float = 0.025
    cost_bps: float = 0.0
    risk_off: bool = False
    defensive_code: str | None = None
    momentum_floor: float = 0.0
    score_method: str = "raw_mom"
    score_lambda: float = 0.0
    score_vol_power: float = 1.0
    # Risk controls (defaults off)
    trend_filter: bool = False
    trend_mode: str = "each"
    trend_sma_window: int = 20
    rsi_filter: bool = False
    rsi_window: int = 20
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_block_overbought: bool = True
    rsi_block_oversold: bool = False
    vol_monitor: bool = False
    vol_window: int = 20
    vol_target_ann: float = 0.20
    vol_max_ann: float = 0.60
    chop_filter: bool = False
    chop_mode: str = "er"
    chop_window: int = 20
    chop_er_threshold: float = 0.25
    chop_adx_window: int = 20
    chop_adx_threshold: float = 20.0
    # Take-profit / stop-loss (qfq)
    tp_sl_mode: str = "none"
    atr_window: int | None = None
    atr_mult: float = 2.0
    atr_step: float = 0.5
    atr_min_mult: float = 0.5
    # Correlation filter (hfq)
    corr_filter: bool = False
    corr_window: int | None = None
    corr_threshold: float = 0.5
    rr_sizing: bool = False
    rr_years: float = 3.0
    rr_thresholds: list[float] | None = None
    rr_weights: list[float] | None = None
    # Drawdown control (strategy NAV)
    dd_control: bool = False
    dd_threshold: float = 0.10
    dd_reduce: float = 1.0
    dd_sleep_days: int = 20


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
            top_k=inp.top_k,
            lookback_days=inp.lookback_days,
            skip_days=inp.skip_days,
            risk_free_rate=inp.risk_free_rate,
            cost_bps=inp.cost_bps,
            risk_off=inp.risk_off,
            defensive_code=inp.defensive_code,
            momentum_floor=inp.momentum_floor,
            score_method=inp.score_method,
            score_lambda=inp.score_lambda,
            score_vol_power=inp.score_vol_power,
            trend_filter=inp.trend_filter,
            trend_mode=inp.trend_mode,
            trend_sma_window=inp.trend_sma_window,
            rsi_filter=inp.rsi_filter,
            rsi_window=inp.rsi_window,
            rsi_overbought=inp.rsi_overbought,
            rsi_oversold=inp.rsi_oversold,
            rsi_block_overbought=inp.rsi_block_overbought,
            rsi_block_oversold=inp.rsi_block_oversold,
            vol_monitor=inp.vol_monitor,
            vol_window=inp.vol_window,
            vol_target_ann=inp.vol_target_ann,
            vol_max_ann=inp.vol_max_ann,
            chop_filter=inp.chop_filter,
            chop_mode=inp.chop_mode,
            chop_window=inp.chop_window,
            chop_er_threshold=inp.chop_er_threshold,
            chop_adx_window=inp.chop_adx_window,
            chop_adx_threshold=inp.chop_adx_threshold,
            tp_sl_mode=inp.tp_sl_mode,
            atr_window=inp.atr_window,
            atr_mult=inp.atr_mult,
            atr_step=inp.atr_step,
            atr_min_mult=inp.atr_min_mult,
            corr_filter=inp.corr_filter,
            corr_window=inp.corr_window,
            corr_threshold=inp.corr_threshold,
            rr_sizing=inp.rr_sizing,
            rr_years=inp.rr_years,
            rr_thresholds=inp.rr_thresholds,
            rr_weights=inp.rr_weights,
            dd_control=inp.dd_control,
            dd_threshold=inp.dd_threshold,
            dd_reduce=inp.dd_reduce,
            dd_sleep_days=inp.dd_sleep_days,
        ),
    )

