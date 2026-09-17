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
    invest_mode: str,
    dca_base_amount: float,
    dca_periodic_amount: float,
    dca_frequency: str,
    dca_weekly_weekday: int,
    dca_monthly_day: int,
    dca_non_trading_shift: str,
    show_non_group_codes: bool,
    pair_chart_prefs_json: str | None,
    replication_rolling_window: int,
    replication_min_samples: int,
    replication_include_portfolio: bool,
    replication_drop_short_history_factors: bool,
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
    obj.invest_mode = str(invest_mode)
    obj.dca_base_amount = float(dca_base_amount)
    obj.dca_periodic_amount = float(dca_periodic_amount)
    obj.dca_frequency = str(dca_frequency)
    obj.dca_weekly_weekday = int(dca_weekly_weekday)
    obj.dca_monthly_day = int(dca_monthly_day)
    obj.dca_non_trading_shift = str(dca_non_trading_shift)
    obj.show_non_group_codes = bool(show_non_group_codes)
    obj.pair_chart_prefs_json = (
        str(pair_chart_prefs_json) if pair_chart_prefs_json is not None else None
    )
    obj.replication_rolling_window = int(replication_rolling_window)
    obj.replication_min_samples = int(replication_min_samples)
    obj.replication_include_portfolio = bool(replication_include_portfolio)
    obj.replication_drop_short_history_factors = bool(
        replication_drop_short_history_factors
    )
    db.flush()
    return obj
