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
    trend_sma_window: int = 200
    rsi_filter: bool = False
    rsi_window: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_block_overbought: bool = True
    rsi_block_oversold: bool = False
    vol_monitor: bool = False
    vol_window: int = 20
    vol_target_ann: float = 0.20
    vol_max_ann: float = 0.60


def compute_rotation_backtest(db: Session, inp: RotationAnalysisInputs) -> dict[str, Any]:
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
        ),
    )

