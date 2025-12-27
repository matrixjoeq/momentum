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
        ),
    )

