from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .models import OffFundResearchState


def get_off_fund_research_state(db: Session) -> OffFundResearchState:
    obj = db.execute(
        select(OffFundResearchState).where(OffFundResearchState.id == 1)
    ).scalar_one_or_none()
    if obj is None:
        try:
            with db.begin_nested():
                db.add(OffFundResearchState(id=1))
                db.flush()
        except IntegrityError:
            pass
        obj = db.execute(
            select(OffFundResearchState).where(OffFundResearchState.id == 1)
        ).scalar_one_or_none()
        if obj is None:
            raise RuntimeError("failed to initialize off_fund_research_state")
    return obj


def upsert_off_fund_research_state(
    db: Session,
    *,
    start_date: str | None,
    end_date: str | None,
    adjust: str,
    risk_free_rate: float,
    inner_mode: str,
    rp_window: int,
    rebalance_cycle: str,
    drift_rebalance_enabled: bool,
    drift_abs_threshold: float,
    drift_rel_threshold: float,
    pair_chart_prefs_json: str | None,
) -> OffFundResearchState:
    obj = get_off_fund_research_state(db)
    obj.start_date = start_date
    obj.end_date = end_date
    obj.adjust = str(adjust)
    obj.risk_free_rate = float(risk_free_rate)
    obj.inner_mode = str(inner_mode)
    obj.rp_window = int(rp_window)
    obj.rebalance_cycle = str(rebalance_cycle)
    obj.drift_rebalance_enabled = bool(drift_rebalance_enabled)
    obj.drift_abs_threshold = float(drift_abs_threshold)
    obj.drift_rel_threshold = float(drift_rel_threshold)
    obj.pair_chart_prefs_json = (
        str(pair_chart_prefs_json) if pair_chart_prefs_json is not None else None
    )
    db.flush()
    return obj
