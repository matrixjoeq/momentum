from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import datetime as dt

import numpy as np
import pandas as pd

from .deps import get_akshare, get_session
from .schemas import (
    BaselineAnalysisRequest,
    BaselineCalendarEffectRequest,
    BaselineMonteCarloRequest,
    BaselineWeekly5EWDashboardRequest,
    RotationCalendarEffectRequest,
    RotationMonteCarloRequest,
    RotationBacktestRequest,
    RotationWeekly5OpenSimRequest,
    RotationNextPlanRequest,
    SimDecisionGenerateRequest,
    SimInitFixedStrategyResponse,
    SimPortfolioCreateRequest,
    SimPortfolioOut,
    SimTradeConfirmRequest,
    SimTradePreviewRequest,
    TrendBacktestRequest,
    EtfPoolOut,
    EtfPoolUpsert,
    FetchResult,
    FetchSelectedRequest,
    IngestionBatchOut,
    PriceOut,
    ValidationPolicyOut,
)
from ..analysis.baseline import (
    TRADING_DAYS_PER_YEAR,
    BaselineInputs,
    _annualized_return,
    _annualized_vol,
    _compute_return_risk_contributions,
    _max_drawdown,
    _max_drawdown_duration_days,
    _rsi_wilder,
    _rolling_max_drawdown,
    _sharpe,
    _sortino,
    _ulcer_index,
    compute_baseline,
    load_close_prices,
)
from ..analysis.calendar_effect import BaselineCalendarEffectInputs, compute_baseline_calendar_effect, compute_rotation_calendar_effect
from ..analysis.calendar_effect import _decision_dates_for_rebalance as _cal_decision_dates_for_rebalance
from ..analysis.calendar_effect import _ew_nav_and_weights_by_decision_dates as _cal_ew_nav_and_weights_by_decision_dates
from ..analysis.montecarlo import MonteCarloConfig, bootstrap_metrics_from_daily_returns
from ..analysis.rotation import RotationAnalysisInputs, compute_rotation_backtest
from ..analysis.trend import TrendInputs, compute_trend_backtest
from ..data.ingestion import ingest_one_etf
from ..data.rollback import logical_rollback_batch, rollback_batch_with_fallback
from ..db.repo import (
    delete_etf_pool,
    delete_prices,
    get_etf_pool_by_code,
    get_price_date_range,
    get_ingestion_batch,
    get_validation_policy_by_id,
    get_validation_policy_by_name,
    list_ingestion_batches,
    list_etf_pool,
    list_validation_policies,
    list_prices,
    mark_fetch_status,
    purge_etf_data,
    update_ingestion_batch,
    upsert_etf_pool,
)
from ..settings import get_settings
from ..validation.policy_infer import infer_policy_name
from ..calendar.trading_calendar import shift_to_trading_day, trading_days
from ..db.models import EtfPrice, SimDecision, SimPortfolio, SimPositionDaily, SimStrategyConfig, SimTrade, SimVariant

logger = logging.getLogger(__name__)

router = APIRouter()

_ALL_ADJUSTS = ("qfq", "hfq", "none")


def _adjust_ranges(db: Session, code: str) -> dict[str, tuple[str | None, str | None]]:
    return {adj: get_price_date_range(db, code=code, adjust=adj) for adj in _ALL_ADJUSTS}


def _ensure_adjust_ranges_consistent(db: Session, code: str) -> tuple[str, str]:
    ranges = _adjust_ranges(db, code)
    vals = list(ranges.values())
    if any(v[0] is None or v[1] is None for v in vals):
        raise ValueError(f"adjust coverage missing for {code}: {ranges}")  # pragma: no cover
    if len(set(vals)) != 1:
        raise ValueError(f"adjust coverage mismatch for {code}: {ranges}")
    return vals[0][0], vals[0][1]


def _rollback_batches_best_effort(db: Session, batch_ids: list[int], *, reason: str) -> None:
    # logical rollback only (no snapshot restore) to avoid cross-adjust interference
    for bid in reversed([x for x in batch_ids if x and x > 0]):
        b = get_ingestion_batch(db, bid)
        if b is None:  # pragma: no cover
            continue
        logical_rollback_batch(db, b)
        update_ingestion_batch(db, batch_id=b.id, status="rolled_back", message=reason)
        db.commit()


def _parse_yyyymmdd(x: str) -> dt.date:
    return dt.datetime.strptime(x, "%Y%m%d").date()


def _iso(d: dt.date | None) -> str | None:
    return None if d is None else d.isoformat()


_FIXED_CODES = ["159915", "511010", "513100", "518880"]
_FIXED_NAMES = {"159915": "创业板ETF", "511010": "国债ETF", "513100": "纳指ETF", "518880": "黄金ETF"}
_WD_LABEL = {0: "MON", 1: "TUE", 2: "WED", 3: "THU", 4: "FRI"}


@router.post("/analysis/baseline")
def baseline_analysis(payload: BaselineAnalysisRequest, db: Session = Depends(get_session)) -> dict:
    inp = BaselineInputs(
        codes=payload.codes,
        start=_parse_yyyymmdd(payload.start),
        end=_parse_yyyymmdd(payload.end),
        benchmark_code=payload.benchmark_code,
        adjust=payload.adjust,
        rebalance=payload.rebalance,
        risk_free_rate=payload.risk_free_rate,
        rolling_weeks=payload.rolling_weeks,
        rolling_months=payload.rolling_months,
        rolling_years=payload.rolling_years,
        fft_windows=payload.fft_windows,
        fft_roll=payload.fft_roll,
        fft_roll_step=payload.fft_roll_step,
    )
    try:
        return compute_baseline(db, inp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/analysis/baseline/calendar-effect")
def baseline_calendar_effect(payload: BaselineCalendarEffectRequest, db: Session = Depends(get_session)) -> dict:
    anchors = payload.anchors
    # backward-compat for weekly-only payloads
    if (payload.weekdays is not None) and (payload.anchors == [0, 1, 2, 3, 4]) and ((payload.rebalance or "weekly").lower() == "weekly"):
        anchors = payload.weekdays
    inp = BaselineCalendarEffectInputs(
        codes=payload.codes,
        start=_parse_yyyymmdd(payload.start),
        end=_parse_yyyymmdd(payload.end),
        adjust=payload.adjust,
        risk_free_rate=payload.risk_free_rate,
        rebalance=payload.rebalance,
        rebalance_shift=payload.rebalance_shift,
        anchors=anchors,
        exec_prices=payload.exec_prices,
    )
    try:
        return compute_baseline_calendar_effect(db, inp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/analysis/rotation")
def rotation_backtest(payload: RotationBacktestRequest, db: Session = Depends(get_session)) -> dict:
    # Pylint may resolve imported dataclasses from an installed package instead of workspace source,
    # which can lag during local dev. Keep behavior correct; suppress false-positive for new fields.
    # pylint: disable=unexpected-keyword-arg
    inp = RotationAnalysisInputs(
        codes=payload.codes,
        start=_parse_yyyymmdd(payload.start),
        end=_parse_yyyymmdd(payload.end),
        rebalance=payload.rebalance,
        rebalance_shift=payload.rebalance_shift,
        top_k=payload.top_k,
        lookback_days=payload.lookback_days,
        skip_days=payload.skip_days,
        risk_off=payload.risk_off,
        defensive_code=payload.defensive_code,
        momentum_floor=payload.momentum_floor,
        score_method=payload.score_method,
        score_lambda=payload.score_lambda,
        score_vol_power=payload.score_vol_power,
        risk_free_rate=payload.risk_free_rate,
        cost_bps=payload.cost_bps,
        tp_sl_mode=payload.tp_sl_mode,
        atr_window=payload.atr_window,
        atr_mult=payload.atr_mult,
        atr_step=payload.atr_step,
        atr_min_mult=payload.atr_min_mult,
        corr_filter=payload.corr_filter,
        corr_window=payload.corr_window,
        corr_threshold=payload.corr_threshold,
        inertia=payload.inertia,
        inertia_min_hold_periods=payload.inertia_min_hold_periods,
        inertia_score_gap=payload.inertia_score_gap,
        inertia_min_turnover=payload.inertia_min_turnover,
        rr_sizing=payload.rr_sizing,
        rr_years=payload.rr_years,
        rr_thresholds=payload.rr_thresholds,
        rr_weights=payload.rr_weights,
        dd_control=payload.dd_control,
        dd_threshold=payload.dd_threshold,
        dd_reduce=payload.dd_reduce,
        dd_sleep_days=payload.dd_sleep_days,
        timing_rsi_gate=payload.timing_rsi_gate,
        timing_rsi_window=payload.timing_rsi_window,
        trend_filter=payload.trend_filter,
        trend_mode=payload.trend_mode,
        trend_sma_window=payload.trend_sma_window,
        rsi_filter=payload.rsi_filter,
        rsi_window=payload.rsi_window,
        rsi_overbought=payload.rsi_overbought,
        rsi_oversold=payload.rsi_oversold,
        rsi_block_overbought=payload.rsi_block_overbought,
        rsi_block_oversold=payload.rsi_block_oversold,
        vol_monitor=payload.vol_monitor,
        vol_window=payload.vol_window,
        vol_target_ann=payload.vol_target_ann,
        vol_max_ann=payload.vol_max_ann,
        chop_filter=payload.chop_filter,
        chop_mode=payload.chop_mode,
        chop_window=payload.chop_window,
        chop_er_threshold=payload.chop_er_threshold,
        chop_adx_window=payload.chop_adx_window,
        chop_adx_threshold=payload.chop_adx_threshold,
    )
    try:
        return compute_rotation_backtest(db, inp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/analysis/trend")
def trend_backtest(payload: TrendBacktestRequest, db: Session = Depends(get_session)) -> dict:
    inp = TrendInputs(
        code=payload.code,
        start=_parse_yyyymmdd(payload.start),
        end=_parse_yyyymmdd(payload.end),
        risk_free_rate=payload.risk_free_rate,
        cost_bps=payload.cost_bps,
        strategy=payload.strategy,
        sma_window=payload.sma_window,
        fast_window=payload.fast_window,
        slow_window=payload.slow_window,
        donchian_entry=payload.donchian_entry,
        donchian_exit=payload.donchian_exit,
        mom_lookback=payload.mom_lookback,
        bias_ma_window=payload.bias_ma_window,
        bias_entry=payload.bias_entry,
        bias_hot=payload.bias_hot,
        bias_cold=payload.bias_cold,
        bias_pos_mode=payload.bias_pos_mode,
    )
    try:
        return compute_trend_backtest(db, inp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/analysis/rotation/calendar-effect")
def rotation_calendar_effect(payload: RotationCalendarEffectRequest, db: Session = Depends(get_session)) -> dict:
    # Reuse all rotation params as the "base" strategy config for the grid; then vary weekday + exec_price.
    # pylint: disable=unexpected-keyword-arg
    base = RotationAnalysisInputs(
        codes=payload.codes,
        start=_parse_yyyymmdd(payload.start),
        end=_parse_yyyymmdd(payload.end),
        rebalance=payload.rebalance,
        rebalance_shift=payload.rebalance_shift,
        top_k=payload.top_k,
        lookback_days=payload.lookback_days,
        skip_days=payload.skip_days,
        risk_off=payload.risk_off,
        defensive_code=payload.defensive_code,
        momentum_floor=payload.momentum_floor,
        score_method=payload.score_method,
        score_lambda=payload.score_lambda,
        score_vol_power=payload.score_vol_power,
        risk_free_rate=payload.risk_free_rate,
        cost_bps=payload.cost_bps,
        tp_sl_mode=payload.tp_sl_mode,
        atr_window=payload.atr_window,
        atr_mult=payload.atr_mult,
        atr_step=payload.atr_step,
        atr_min_mult=payload.atr_min_mult,
        corr_filter=payload.corr_filter,
        corr_window=payload.corr_window,
        corr_threshold=payload.corr_threshold,
        rr_sizing=payload.rr_sizing,
        rr_years=payload.rr_years,
        rr_thresholds=payload.rr_thresholds,
        rr_weights=payload.rr_weights,
        dd_control=payload.dd_control,
        dd_threshold=payload.dd_threshold,
        dd_reduce=payload.dd_reduce,
        dd_sleep_days=payload.dd_sleep_days,
        trend_filter=payload.trend_filter,
        trend_mode=payload.trend_mode,
        trend_sma_window=payload.trend_sma_window,
        rsi_filter=payload.rsi_filter,
        rsi_window=payload.rsi_window,
        rsi_overbought=payload.rsi_overbought,
        rsi_oversold=payload.rsi_oversold,
        rsi_block_overbought=payload.rsi_block_overbought,
        rsi_block_oversold=payload.rsi_block_oversold,
        vol_monitor=payload.vol_monitor,
        vol_window=payload.vol_window,
        vol_target_ann=payload.vol_target_ann,
        vol_max_ann=payload.vol_max_ann,
        chop_filter=payload.chop_filter,
        chop_mode=payload.chop_mode,
        chop_window=payload.chop_window,
        chop_er_threshold=payload.chop_er_threshold,
        chop_adx_window=payload.chop_adx_window,
        chop_adx_threshold=payload.chop_adx_threshold,
        rebalance_anchor=None,
    )
    try:
        anchors = payload.anchors
        if (payload.weekdays is not None) and (payload.anchors == [0, 1, 2, 3, 4]) and ((payload.rebalance or "weekly").lower() == "weekly"):
            anchors = payload.weekdays
        return compute_rotation_calendar_effect(db, base=base, anchors=anchors, exec_prices=payload.exec_prices)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/analysis/rotation/weekly5-open")
def rotation_weekly5_open_sim(payload: RotationWeekly5OpenSimRequest, db: Session = Depends(get_session)) -> dict:
    """
    Mini-program friendly simplified simulation:
    - candidate pool fixed to the 4 ETFs in product spec
    - weekly rebalance, TopK=1, lookback=20, skip=0
    - exec_price=open, rebalance_shift=prev
    - cost=0, all risk controls off
    - run 5 variants for weekly anchor weekday Mon..Fri (0..4)
    """
    codes = ["159915", "511010", "513100", "518880"]
    start = _parse_yyyymmdd(payload.start)
    end = _parse_yyyymmdd(payload.end)
    base = RotationAnalysisInputs(
        codes=codes,
        start=start,
        end=end,
        rebalance="weekly",
        rebalance_shift="prev",
        rebalance_anchor=None,
        top_k=1,
        lookback_days=20,
        skip_days=0,
        cost_bps=0.0,
        risk_off=False,
        defensive_code=None,
        momentum_floor=0.0,
        score_method="raw_mom",
        score_lambda=0.0,
        score_vol_power=1.0,
        # risk controls all off
        trend_filter=False,
        rsi_filter=False,
        vol_monitor=False,
        chop_filter=False,
        corr_filter=False,
        inertia=False,
        rr_sizing=False,
        dd_control=False,
        timing_rsi_gate=False,
        # execution on open
        exec_price="open",
    )
    one_anchor = payload.anchor_weekday
    anchors = [int(one_anchor)] if one_anchor is not None else [0, 1, 2, 3, 4]
    def _slim_for_miniprogram(x: dict) -> dict:
        # Keep only what the mini-program renders (avoid shipping large unused blobs like rolling series).
        keep = [
            "date_range",
            "score_method",
            "tp_sl_mode",
            "score_params",
            "codes",
            "benchmark_codes",
            "price_basis",
            "timing",
            "nav",
            "nav_rsi",
            "attribution",
            "metrics",
            "win_payoff",
            "period_details",
            # used by sim_decision/generate to extract per-period decisions
            "holdings",
        ]
        return {k: x.get(k) for k in keep if k in x}

    by_anchor: dict[str, dict] = {}
    for a in anchors:
        # pylint: disable=unexpected-keyword-arg
        inp = RotationAnalysisInputs(**{**base.__dict__, "rebalance_anchor": int(a)})
        by_anchor[str(a)] = _slim_for_miniprogram(compute_rotation_backtest(db, inp))
    return {
        "meta": {
            "type": "rotation_weekly5_open",
            "codes": codes,
            "start": payload.start,
            "end": payload.end,
            "rebalance": "weekly",
            "rebalance_shift": "prev",
            "exec_price": "open",
            "anchors": anchors,
            "fixed_params": {"top_k": 1, "lookback_days": 20, "skip_days": 0, "cost_bps": 0},
            "risk_controls": "all_off",
        },
        "by_anchor": by_anchor,
        "weekday_map": {"0": "MON", "1": "TUE", "2": "WED", "3": "THU", "4": "FRI"},
    }


@router.post("/analysis/rotation/weekly5-open-lite")
def rotation_weekly5_open_sim_lite(payload: RotationWeekly5OpenSimRequest, db: Session = Depends(get_session)) -> dict:
    """
    Lite version for mini-program first paint:
    - returns only NAV series (and minimal meta) for one anchor (or 5 anchors if anchor_weekday is omitted)
    - omits heavy fields to reduce payload/JSON serialization time
    """
    codes = ["159915", "511010", "513100", "518880"]
    start = _parse_yyyymmdd(payload.start)
    end = _parse_yyyymmdd(payload.end)
    base = RotationAnalysisInputs(
        codes=codes,
        start=start,
        end=end,
        rebalance="weekly",
        rebalance_shift="prev",
        rebalance_anchor=None,
        top_k=1,
        lookback_days=20,
        skip_days=0,
        cost_bps=0.0,
        risk_off=False,
        defensive_code=None,
        momentum_floor=0.0,
        score_method="raw_mom",
        score_lambda=0.0,
        score_vol_power=1.0,
        # risk controls all off
        trend_filter=False,
        rsi_filter=False,
        vol_monitor=False,
        chop_filter=False,
        corr_filter=False,
        inertia=False,
        rr_sizing=False,
        dd_control=False,
        timing_rsi_gate=False,
        # execution on open
        exec_price="open",
    )

    one_anchor = payload.anchor_weekday
    anchors = [int(one_anchor)] if one_anchor is not None else [0, 1, 2, 3, 4]

    def _lite(x: dict) -> dict:
        nav = x.get("nav") if isinstance(x, dict) else None
        return {
            "date_range": (x.get("date_range") if isinstance(x, dict) else None),
            "nav": nav,
        }

    by_anchor: dict[str, dict] = {}
    for a in anchors:
        # pylint: disable=unexpected-keyword-arg
        inp = RotationAnalysisInputs(**{**base.__dict__, "rebalance_anchor": int(a)})
        by_anchor[str(a)] = _lite(compute_rotation_backtest(db, inp))

    return {
        "meta": {
            "type": "rotation_weekly5_open_lite",
            "codes": codes,
            "start": payload.start,
            "end": payload.end,
            "rebalance": "weekly",
            "rebalance_shift": "prev",
            "exec_price": "open",
            "anchors": anchors,
        },
        "by_anchor": by_anchor,
        "weekday_map": {"0": "MON", "1": "TUE", "2": "WED", "3": "THU", "4": "FRI"},
    }


@router.post("/analysis/rotation/next-plan")
def rotation_next_plan(payload: RotationNextPlanRequest, db: Session = Depends(get_session)) -> dict:
    """
    "Tomorrow plan" for the fixed mini-program rotation strategy.
    If the next trading day is a rebalance effective day (open execution), return the top pick based on asof close.
    """
    asof = _parse_yyyymmdd(payload.asof)
    anchor = int(payload.anchor_weekday)
    if anchor not in {0, 1, 2, 3, 4}:
        raise HTTPException(status_code=400, detail="anchor_weekday must be 0..4")

    # next trading day (XSHG) after asof
    try:
        tds = trading_days(asof, asof + dt.timedelta(days=20), cal="XSHG")
        next_td = next((d for d in tds if d > asof), asof)
    except Exception:  # pragma: no cover
        next_td = asof

    # Determine if asof is the decision day for the coming weekly anchor period.
    # For weekly anchor weekday, the period end is the next occurrence of that weekday (>= asof),
    # and decision_date is shifted to previous trading day if end is not a session ("prev").
    delta = (anchor - asof.weekday()) % 7
    anchor_date = asof + dt.timedelta(days=int(delta))
    try:
        decision_date = shift_to_trading_day(anchor_date, shift="prev", cal="XSHG")
    except Exception:  # pragma: no cover
        decision_date = anchor_date
    rebalance_effective_next_day = bool(decision_date == asof)

    codes = _FIXED_CODES[:]
    start = asof - dt.timedelta(days=90)
    px = load_close_prices(db, codes=codes, start=start, end=asof, adjust="hfq")
    if px.empty:
        raise HTTPException(status_code=400, detail="no price data")
    px = px.sort_index().ffill()
    first_valid = {c: px[c].first_valid_index() for c in codes if c in px.columns}
    common_start = max([d for d in first_valid.values() if d is not None])
    px = px.loc[common_start:, codes]
    if len(px) < 21:
        raise HTTPException(status_code=400, detail="insufficient history for lookback_days=20")

    last = px.iloc[-1]
    prev = px.iloc[-21]
    scores: dict[str, float] = {}
    for c in codes:
        a = float(last[c])
        b = float(prev[c])
        s = (a / b - 1.0) if (np.isfinite(a) and np.isfinite(b) and b > 0) else float("nan")
        scores[c] = float(s)
    pick_code = max(scores.keys(), key=lambda k: (scores.get(k) if np.isfinite(scores.get(k, float("nan"))) else -1e18))
    pick_name = _FIXED_NAMES.get(pick_code, pick_code)

    return {
        "asof": asof.strftime("%Y%m%d"),
        "next_trading_day": next_td.isoformat(),
        "rebalance_effective_next_day": rebalance_effective_next_day,
        "pick_code": pick_code,
        "pick_name": pick_name,
        "scores": {c: float(scores[c]) for c in codes},
        "meta": {"anchor_weekday": anchor, "rebalance_shift": "prev", "lookback_days": 20, "top_k": 1, "exec_price": "open"},
    }


@router.post("/analysis/baseline/weekly5-ew-dashboard")
def baseline_weekly5_ew_dashboard(payload: BaselineWeekly5EWDashboardRequest, db: Session = Depends(get_session)) -> dict:
    """
    Mini-program dashboard data:
    - fixed 4 ETFs
    - equal-weight portfolio rebalanced weekly with anchor weekday 0..4 (MON..FRI)
    - price basis: hfq close
    - execution assumption: rebalance at close on decision_date, effective next trading day
    """
    start = _parse_yyyymmdd(payload.start)
    end = _parse_yyyymmdd(payload.end)
    rf = float(payload.risk_free_rate)
    shift = (payload.rebalance_shift or "prev").strip().lower()
    if shift not in {"prev", "next"}:
        raise HTTPException(status_code=400, detail="rebalance_shift must be prev|next")

    codes = _FIXED_CODES[:]
    close = load_close_prices(db, codes=codes, start=start, end=end, adjust="hfq")
    if close.empty:
        raise HTTPException(status_code=400, detail="no price data for given range")
    close = close.sort_index()
    missing = [c for c in codes if c not in close.columns or close[c].dropna().empty]
    if missing:
        raise HTTPException(status_code=400, detail=f"missing hfq data: {missing}")
    close_ff = close.ffill()

    # common start where all codes have data
    first_valid = {c: close[c].first_valid_index() for c in codes if c in close.columns}
    common_start = max([d for d in first_valid.values() if d is not None])
    px = close_ff.loc[common_start:, codes]
    daily_ret = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    idx = px.index

    def _ema(series: pd.Series, span: int) -> pd.Series:
        s = pd.Series(series).astype(float)
        return s.ewm(span=int(span), adjust=False, min_periods=int(span)).mean()

    def _rolling_std(series: pd.Series, window: int) -> pd.Series:
        return pd.Series(series).astype(float).rolling(window=int(window), min_periods=int(window)).std(ddof=1)

    def _drawdown(nav: pd.Series) -> pd.Series:
        peak = nav.cummax()
        return (nav / peak - 1.0).astype(float)

    one_anchor = payload.anchor_weekday
    anchors = [int(one_anchor)] if one_anchor is not None else [0, 1, 2, 3, 4]
    by_anchor: dict[str, dict] = {}
    for a in anchors:
        decision_dates = _cal_decision_dates_for_rebalance(idx, rebalance="weekly", anchor=int(a), shift=shift)
        ew_nav, ew_w = _cal_ew_nav_and_weights_by_decision_dates(daily_ret[codes], decision_dates=decision_dates)
        ew_ret = ew_nav.pct_change().fillna(0.0).astype(float)

        # overlays on EW NAV
        ema252 = _ema(ew_nav, 252)
        sd252 = _rolling_std(ew_nav, 252)
        bb_u = ema252 + 2.0 * sd252
        bb_l = ema252 - 2.0 * sd252

        dd = _drawdown(ew_nav)
        rsi24 = _rsi_wilder(ew_nav, window=24)

        win_3y = 3 * TRADING_DAYS_PER_YEAR
        rr3y = (ew_nav / ew_nav.shift(win_3y) - 1.0).astype(float)
        rdd3y = _rolling_max_drawdown(ew_nav, win_3y).astype(float)

        # metrics
        cum_ret = float(ew_nav.iloc[-1] / ew_nav.iloc[0] - 1.0) if len(ew_nav) else float("nan")
        ann_ret = _annualized_return(ew_nav, ann_factor=TRADING_DAYS_PER_YEAR)
        ann_vol = _annualized_vol(ew_ret, ann_factor=TRADING_DAYS_PER_YEAR)
        mdd = _max_drawdown(ew_nav)
        mdd_dur = _max_drawdown_duration_days(ew_nav)
        sharpe = _sharpe(ew_ret, rf=rf, ann_factor=TRADING_DAYS_PER_YEAR)
        calmar = float(ann_ret / abs(mdd)) if np.isfinite(mdd) and mdd < 0 else float("nan")
        sortino = _sortino(ew_ret, rf=rf, ann_factor=TRADING_DAYS_PER_YEAR)
        ui = _ulcer_index(ew_nav, in_percent=True)
        ui_den = ui / 100.0
        upi = float((ann_ret - rf) / ui_den) if ui_den > 0 else float("nan")

        metrics = {
            "cumulative_return": float(cum_ret),
            "annualized_return": float(ann_ret),
            "annualized_volatility": float(ann_vol),
            "max_drawdown": float(mdd),
            "max_drawdown_recovery_days": int(mdd_dur),
            "sharpe_ratio": float(sharpe),
            "calmar_ratio": float(calmar),
            "sortino_ratio": float(sortino),
            "ulcer_index": float(ui),
            "ulcer_performance_index": float(upi),
        }

        # attribution + correlation (on daily returns)
        corr = daily_ret[codes].corr(method="pearson")
        corr_out = {
            "method": "pearson",
            "n_obs": int(len(daily_ret)),
            "codes": codes,
            "matrix": corr.to_numpy(dtype=float).tolist(),
        }
        attribution = _compute_return_risk_contributions(asset_ret=daily_ret[codes], weights=ew_w[codes], total_return=float(cum_ret))

        # return calendar
        daily = ew_ret.copy()
        monthly = ew_nav.resample("ME").last().pct_change().dropna()
        yearly = ew_nav.resample("YE").last().pct_change().dropna()
        cal = {
            "daily": {"dates": daily.index.date.astype(str).tolist(), "values": daily.astype(float).tolist()},
            "monthly": {"dates": monthly.index.date.astype(str).tolist(), "values": monthly.astype(float).tolist()},
            "yearly": {"dates": yearly.index.date.astype(str).tolist(), "values": yearly.astype(float).tolist()},
        }

        def _tolist(s: pd.Series) -> list[float | None]:
            return [None if (pd.isna(x) or not np.isfinite(float(x))) else float(x) for x in s.to_numpy(dtype=float)]

        by_anchor[str(a)] = {
            "meta": {"anchor_weekday": int(a), "label": _WD_LABEL[int(a)], "rebalance_shift": shift, "price": "hfq_close"},
            "dates": idx.date.astype(str).tolist(),
            "nav": _tolist(ew_nav),
            "ema252": _tolist(ema252),
            "bb_upper": _tolist(bb_u),
            "bb_lower": _tolist(bb_l),
            "drawdown": _tolist(dd),
            "rsi24": _tolist(rsi24),
            "roll3y_return": _tolist(rr3y),
            "roll3y_mdd": _tolist(rdd3y),
            "metrics": metrics,
            "attribution": attribution,
            "correlation": corr_out,
            "calendar": cal,
        }

    return {
        "meta": {
            "type": "baseline_weekly5_ew_dashboard",
            "codes": codes,
            "start": payload.start,
            "end": payload.end,
            "common_start": common_start.date().strftime("%Y%m%d"),
            "rebalance": "weekly",
            "rebalance_shift": shift,
            "price": "hfq_close",
            "anchors": anchors,
        },
        "by_anchor": by_anchor,
        "weekday_map": {"0": "MON", "1": "TUE", "2": "WED", "3": "THU", "4": "FRI"},
    }


@router.post("/analysis/baseline/weekly5-ew-dashboard-lite")
def baseline_weekly5_ew_dashboard_lite(payload: BaselineWeekly5EWDashboardRequest, db: Session = Depends(get_session)) -> dict:
    """
    Lite version for mini-program first paint:
    - returns only chart series needed for (1)~(5) quickly
    - omits metrics/attribution/correlation/calendar
    """
    start = _parse_yyyymmdd(payload.start)
    end = _parse_yyyymmdd(payload.end)
    shift = (payload.rebalance_shift or "prev").strip().lower()
    if shift not in {"prev", "next"}:
        raise HTTPException(status_code=400, detail="rebalance_shift must be prev|next")

    codes = _FIXED_CODES[:]
    close = load_close_prices(db, codes=codes, start=start, end=end, adjust="hfq")
    if close.empty:
        raise HTTPException(status_code=400, detail="no price data for given range")
    close = close.sort_index()
    missing = [c for c in codes if c not in close.columns or close[c].dropna().empty]
    if missing:
        raise HTTPException(status_code=400, detail=f"missing hfq data: {missing}")
    close_ff = close.ffill()

    first_valid = {c: close[c].first_valid_index() for c in codes if c in close.columns}
    common_start = max([d for d in first_valid.values() if d is not None])
    px = close_ff.loc[common_start:, codes]
    daily_ret = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    idx = px.index

    def _ema(series: pd.Series, span: int) -> pd.Series:
        s = pd.Series(series).astype(float)
        return s.ewm(span=int(span), adjust=False, min_periods=int(span)).mean()

    def _rolling_std(series: pd.Series, window: int) -> pd.Series:
        return pd.Series(series).astype(float).rolling(window=int(window), min_periods=int(window)).std(ddof=1)

    def _drawdown(nav: pd.Series) -> pd.Series:
        peak = nav.cummax()
        return (nav / peak - 1.0).astype(float)

    def _tolist(s: pd.Series) -> list[float | None]:
        return [None if (pd.isna(x) or not np.isfinite(float(x))) else float(x) for x in s.to_numpy(dtype=float)]

    one_anchor = payload.anchor_weekday
    anchors = [int(one_anchor)] if one_anchor is not None else [0, 1, 2, 3, 4]
    by_anchor: dict[str, dict] = {}
    for a in anchors:
        decision_dates = _cal_decision_dates_for_rebalance(idx, rebalance="weekly", anchor=int(a), shift=shift)
        ew_nav, _ew_w = _cal_ew_nav_and_weights_by_decision_dates(daily_ret[codes], decision_dates=decision_dates)

        ema252 = _ema(ew_nav, 252)
        sd252 = _rolling_std(ew_nav, 252)
        bb_u = ema252 + 2.0 * sd252
        bb_l = ema252 - 2.0 * sd252

        dd = _drawdown(ew_nav)
        rsi24 = _rsi_wilder(ew_nav, window=24)

        win_3y = 3 * TRADING_DAYS_PER_YEAR
        rr3y = (ew_nav / ew_nav.shift(win_3y) - 1.0).astype(float)
        rdd3y = _rolling_max_drawdown(ew_nav, win_3y).astype(float)

        by_anchor[str(a)] = {
            "meta": {"anchor_weekday": int(a), "label": _WD_LABEL[int(a)], "rebalance_shift": shift, "price": "hfq_close"},
            "dates": idx.date.astype(str).tolist(),
            "nav": _tolist(ew_nav),
            "ema252": _tolist(ema252),
            "bb_upper": _tolist(bb_u),
            "bb_lower": _tolist(bb_l),
            "drawdown": _tolist(dd),
            "rsi24": _tolist(rsi24),
            "roll3y_return": _tolist(rr3y),
            "roll3y_mdd": _tolist(rdd3y),
        }

    return {
        "meta": {
            "type": "baseline_weekly5_ew_dashboard_lite",
            "codes": codes,
            "start": payload.start,
            "end": payload.end,
            "common_start": common_start.date().strftime("%Y%m%d"),
            "rebalance": "weekly",
            "rebalance_shift": shift,
            "price": "hfq_close",
            "anchors": anchors,
        },
        "by_anchor": by_anchor,
        "weekday_map": {"0": "MON", "1": "TUE", "2": "WED", "3": "THU", "4": "FRI"},
    }


@router.post("/sim/portfolio", response_model=SimPortfolioOut)
def sim_create_portfolio(payload: SimPortfolioCreateRequest, db: Session = Depends(get_session)) -> SimPortfolioOut:
    obj = SimPortfolio(name=payload.name, base_ccy="CNY", initial_cash=float(payload.initial_cash))
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return SimPortfolioOut(id=int(obj.id), name=obj.name, base_ccy=obj.base_ccy, initial_cash=float(obj.initial_cash), created_at=obj.created_at.isoformat())


@router.get("/sim/portfolio", response_model=list[SimPortfolioOut])
def sim_list_portfolios(db: Session = Depends(get_session)) -> list[SimPortfolioOut]:
    rows = list(db.query(SimPortfolio).order_by(SimPortfolio.id.asc()).all())
    return [SimPortfolioOut(id=int(x.id), name=x.name, base_ccy=x.base_ccy, initial_cash=float(x.initial_cash), created_at=x.created_at.isoformat()) for x in rows]


@router.post("/sim/portfolio/{portfolio_id}/init-fixed-strategy", response_model=SimInitFixedStrategyResponse)
def sim_init_fixed_strategy(portfolio_id: int, db: Session = Depends(get_session)) -> SimInitFixedStrategyResponse:
    p = db.query(SimPortfolio).filter(SimPortfolio.id == int(portfolio_id)).one_or_none()
    if p is None:
        raise HTTPException(status_code=404, detail="portfolio not found")

    # Ensure pool entries exist (for sync-market / transparency).
    for code in _FIXED_CODES:
        if get_etf_pool_by_code(db, code) is None:
            upsert_etf_pool(db, code=code, name=_FIXED_NAMES.get(code, code), start_date=None, end_date=None)
    db.flush()

    # Create config snapshot (fixed params).
    import json

    cfg = SimStrategyConfig(
        portfolio_id=int(p.id),
        codes_json=json.dumps(_FIXED_CODES, ensure_ascii=False),
        rebalance="weekly",
        lookback_days=20,
        top_k=1,
        exec_price="open",
        rebalance_shift="prev",
        risk_controls_json=json.dumps({"all_off": True}, ensure_ascii=False),
    )
    db.add(cfg)
    db.flush()

    # Create 5 variants (MON..FRI). Default active = MON.
    vids: list[int] = []
    for wd in [0, 1, 2, 3, 4]:
        v = SimVariant(
            portfolio_id=int(p.id),
            config_id=int(cfg.id),
            anchor_weekday=int(wd),
            label=_WD_LABEL[int(wd)],
            is_active=1 if int(wd) == 0 else 0,
        )
        db.add(v)
        db.flush()
        vids.append(int(v.id))

        # Seed initial position snapshot at portfolio creation date? Keep empty; use trade_confirm to create first snapshot.
    db.commit()
    return SimInitFixedStrategyResponse(portfolio_id=int(p.id), config_id=int(cfg.id), variant_ids=vids)


@router.get("/sim/portfolio/{portfolio_id}/variants")
def sim_list_variants(portfolio_id: int, db: Session = Depends(get_session)) -> dict:
    p = db.query(SimPortfolio).filter(SimPortfolio.id == int(portfolio_id)).one_or_none()
    if p is None:
        raise HTTPException(status_code=404, detail="portfolio not found")
    rows = list(db.query(SimVariant).filter(SimVariant.portfolio_id == int(p.id)).order_by(SimVariant.anchor_weekday.asc()).all())
    return {
        "portfolio_id": int(p.id),
        "variants": [
            {"id": int(v.id), "anchor_weekday": int(v.anchor_weekday), "label": v.label, "is_active": bool(int(v.is_active))}
            for v in rows
        ],
    }


@router.post("/sim/variant/{variant_id}/set-active")
def sim_set_active_variant(variant_id: int, db: Session = Depends(get_session)) -> dict:
    v = db.query(SimVariant).filter(SimVariant.id == int(variant_id)).one_or_none()
    if v is None:
        raise HTTPException(status_code=404, detail="variant not found")
    # reset others in portfolio
    db.query(SimVariant).filter(SimVariant.portfolio_id == int(v.portfolio_id)).update({"is_active": 0})
    v.is_active = 1
    db.commit()
    return {"ok": True, "active_variant_id": int(v.id)}


@router.post("/sim/sync-market")
def sim_sync_market(db: Session = Depends(get_session), ak=Depends(get_akshare)) -> dict:
    """
    Sync market data for fixed 4-code pool.
    Reuses ingestion pipeline (fetch all adjusts).
    """
    settings = get_settings()
    items_by_code = {x.code: x for x in list_etf_pool(db)}
    out: list[FetchResult] = []
    for code in _FIXED_CODES:
        item = items_by_code.get(code)
        if item is None:
            upsert_etf_pool(db, code=code, name=_FIXED_NAMES.get(code, code), start_date=None, end_date=None)
            item = get_etf_pool_by_code(db, code)
        start = (item.start_date if item and item.start_date else settings.default_start_date)
        end = (item.end_date if item and item.end_date else settings.default_end_date)
        total = 0
        ok = True
        parts: list[str] = []
        for adj in _ALL_ADJUSTS:
            res = ingest_one_etf(db, ak=ak, code=code, start_date=start, end_date=end, adjust=adj)
            total += int(res.upserted or 0)
            if res.status != "success":
                ok = False
            extra = f",msg={res.message}" if res.status != "success" and res.message else ""
            parts.append(f"{adj}:{res.status}(batch={res.batch_id},upserted={res.upserted}{extra})")
        status = "success" if ok else "failed"
        msg = "; ".join(parts)
        mark_fetch_status(db, code=code, status=status, message=msg)
        out.append(FetchResult(code=code, inserted_or_updated=(total if ok else 0), status=status, message=msg))
    db.commit()
    return {"ok": True, "results": [x.model_dump() for x in out]}


@router.post("/sim/decision/generate")
def sim_generate_decisions(payload: SimDecisionGenerateRequest, db: Session = Depends(get_session)) -> dict:
    """
    Generate sim_decision rows by running the fixed 5-anchor backtest and extracting holding periods.
    """
    p = db.query(SimPortfolio).filter(SimPortfolio.id == int(payload.portfolio_id)).one_or_none()
    if p is None:
        raise HTTPException(status_code=404, detail="portfolio not found")

    start = _parse_yyyymmdd(payload.start)
    end = _parse_yyyymmdd(payload.end)

    # ensure variants exist
    variants = list(db.query(SimVariant).filter(SimVariant.portfolio_id == int(p.id)).order_by(SimVariant.anchor_weekday.asc()).all())
    if not variants:
        raise HTTPException(status_code=400, detail="no variants; call init-fixed-strategy first")
    v_by_wd = {int(v.anchor_weekday): v for v in variants}

    # compute results for all 5 weekdays
    sim_res = rotation_weekly5_open_sim(RotationWeekly5OpenSimRequest(start=payload.start, end=payload.end), db=db)
    by_anchor = sim_res.get("by_anchor") or {}

    import json

    inserted = 0
    for wd_s, res in by_anchor.items():
        wd = int(wd_s)
        v = v_by_wd.get(wd)
        if v is None:
            continue
        # strategy payload exports holdings as a list of per-period dicts
        periods = (res or {}).get("holdings") or []
        prev_code: str | None = None
        for per in periods:
            try:
                d_date = dt.date.fromisoformat(str(per.get("decision_date")))
                eff = dt.date.fromisoformat(str(per.get("start_date")))
            except Exception as e:  # pragma: no cover
                raise HTTPException(status_code=500, detail=f"invalid period dates: {e}") from e
            if d_date < start or d_date > end:
                continue
            picks = per.get("picks") or []
            picked = str(picks[0]) if picks else None
            scores = per.get("scores") or {}
            reason = {"mode": per.get("mode"), "risk_off_triggered": per.get("risk_off_triggered")}

            # upsert (unique: variant_id + decision_date)
            existing = db.query(SimDecision).filter(SimDecision.variant_id == int(v.id), SimDecision.decision_date == d_date).one_or_none()
            if existing is None:
                obj = SimDecision(
                    variant_id=int(v.id),
                    decision_date=d_date,
                    effective_date=eff,
                    picked_code=picked,
                    scores_json=json.dumps(scores, ensure_ascii=False),
                    prev_code=prev_code,
                    reason_json=json.dumps(reason, ensure_ascii=False),
                )
                db.add(obj)
                db.flush()
                inserted += 1
            else:
                existing.effective_date = eff
                existing.picked_code = picked
                existing.scores_json = json.dumps(scores, ensure_ascii=False)
                existing.prev_code = prev_code
                existing.reason_json = json.dumps(reason, ensure_ascii=False)
            prev_code = picked
    db.commit()
    return {"ok": True, "inserted": int(inserted)}


@router.get("/sim/variant/{variant_id}/decisions")
def sim_list_decisions(variant_id: int, start: str | None = None, end: str | None = None, db: Session = Depends(get_session)) -> dict:
    v = db.query(SimVariant).filter(SimVariant.id == int(variant_id)).one_or_none()
    if v is None:
        raise HTTPException(status_code=404, detail="variant not found")
    q = db.query(SimDecision).filter(SimDecision.variant_id == int(v.id))
    if start:
        q = q.filter(SimDecision.decision_date >= _parse_yyyymmdd(start))
    if end:
        q = q.filter(SimDecision.decision_date <= _parse_yyyymmdd(end))
    rows = list(q.order_by(SimDecision.decision_date.asc()).all())
    return {
        "variant_id": int(v.id),
        "decisions": [
            {
                "id": int(x.id),
                "decision_date": x.decision_date.isoformat(),
                "effective_date": x.effective_date.isoformat(),
                "picked_code": x.picked_code,
                "prev_code": x.prev_code,
                "scores": __import__("json").loads(x.scores_json or "{}"),
                "reason": __import__("json").loads(x.reason_json or "{}"),
            }
            for x in rows
        ],
    }


def _get_open_price_hfq(db: Session, *, code: str, day: dt.date) -> float:
    row = (
        db.query(EtfPrice)
        .filter(EtfPrice.code == code, EtfPrice.adjust == "hfq", EtfPrice.trade_date == day)
        .one_or_none()
    )
    if row is None or row.open is None:
        raise HTTPException(status_code=400, detail=f"missing hfq open price for {code} at {day.isoformat()}")
    return float(row.open)


def _latest_position(db: Session, *, variant_id: int, before_or_on: dt.date | None = None) -> SimPositionDaily | None:
    q = db.query(SimPositionDaily).filter(SimPositionDaily.variant_id == int(variant_id))
    if before_or_on is not None:
        q = q.filter(SimPositionDaily.trade_date <= before_or_on)
    return q.order_by(SimPositionDaily.trade_date.desc()).first()


@router.post("/sim/trade/preview")
def sim_trade_preview(payload: SimTradePreviewRequest, db: Session = Depends(get_session)) -> dict:
    d = db.query(SimDecision).filter(SimDecision.id == int(payload.decision_id)).one_or_none()
    if d is None or int(d.variant_id) != int(payload.variant_id):
        raise HTTPException(status_code=404, detail="decision not found")
    pos = _latest_position(db, variant_id=int(payload.variant_id), before_or_on=d.effective_date)
    import json

    cur_positions = {}
    cur_cash = None
    if pos is not None:
        cur_positions = json.loads(pos.positions_json or "{}")
        cur_cash = float(pos.cash)
    cur_code = next((k for k, v in (cur_positions or {}).items() if float(v) > 0), None)

    target = d.picked_code
    sells = []
    buys = []
    if cur_code and cur_code != target:
        sells.append({"code": cur_code, "side": "SELL"})
    if target and target != cur_code:
        buys.append({"code": target, "side": "BUY"})
    return {
        "variant_id": int(payload.variant_id),
        "decision_id": int(d.id),
        "trade_date": d.effective_date.isoformat(),
        "current_code": cur_code,
        "target_code": target,
        "sells": sells,
        "buys": buys,
        "cash": cur_cash,
    }


@router.post("/sim/trade/confirm")
def sim_trade_confirm(payload: SimTradeConfirmRequest, db: Session = Depends(get_session)) -> dict:
    v = db.query(SimVariant).filter(SimVariant.id == int(payload.variant_id)).one_or_none()
    if v is None:
        raise HTTPException(status_code=404, detail="variant not found")
    d = db.query(SimDecision).filter(SimDecision.id == int(payload.decision_id)).one_or_none()
    if d is None or int(d.variant_id) != int(v.id):
        raise HTTPException(status_code=404, detail="decision not found")

    # idempotency: if already confirmed for this decision, return existing.
    existing = db.query(SimTrade).filter(SimTrade.decision_id == int(d.id)).first()
    if existing is not None:
        return {"ok": True, "already_confirmed": True, "decision_id": int(d.id)}

    trade_date = d.effective_date
    pos0 = _latest_position(db, variant_id=int(v.id), before_or_on=trade_date)
    import json

    if pos0 is None:
        # Initialize from portfolio cash.
        p = db.query(SimPortfolio).filter(SimPortfolio.id == int(v.portfolio_id)).one()
        cur_positions = {}
        cash = float(p.initial_cash)
        nav = float(p.initial_cash)
    else:
        cur_positions = json.loads(pos0.positions_json or "{}")
        cash = float(pos0.cash)
        nav = float(pos0.nav)
    cur_code = next((k for k, qty in (cur_positions or {}).items() if float(qty) > 1e-12), None)
    cur_qty = float(cur_positions.get(cur_code, 0.0)) if cur_code else 0.0

    # Sell current at open
    if cur_code and cur_qty > 0:
        px = _get_open_price_hfq(db, code=str(cur_code), day=trade_date)
        amt = float(cur_qty * px)
        t = SimTrade(
            variant_id=int(v.id),
            trade_date=trade_date,
            code=str(cur_code),
            side="SELL",
            price=float(px),
            qty=float(cur_qty),
            amount=float(amt),
            decision_id=int(d.id),
        )
        db.add(t)
        cash += amt
        cur_positions = {}

    # Buy target with all cash
    if d.picked_code:
        px = _get_open_price_hfq(db, code=str(d.picked_code), day=trade_date)
        qty = float(cash / px) if px > 0 else 0.0
        amt = float(qty * px)
        t = SimTrade(
            variant_id=int(v.id),
            trade_date=trade_date,
            code=str(d.picked_code),
            side="BUY",
            price=float(px),
            qty=float(qty),
            amount=float(amt),
            decision_id=int(d.id),
        )
        db.add(t)
        cash -= amt
        cur_positions = {str(d.picked_code): qty}

    nav = float(cash + sum(float(q) * _get_open_price_hfq(db, code=str(c), day=trade_date) for c, q in cur_positions.items()))
    snap = SimPositionDaily(
        variant_id=int(v.id),
        trade_date=trade_date,
        positions_json=json.dumps(cur_positions, ensure_ascii=False),
        cash=float(cash),
        nav=float(nav),
        mdd=None,
    )
    db.add(snap)
    db.commit()
    return {"ok": True, "decision_id": int(d.id), "trade_date": trade_date.isoformat(), "nav": float(nav)}


@router.post("/sim/mark-to-market")
def sim_mark_to_market(variant_id: int, start: str | None = None, end: str | None = None, db: Session = Depends(get_session)) -> dict:
    v = db.query(SimVariant).filter(SimVariant.id == int(variant_id)).one_or_none()
    if v is None:
        raise HTTPException(status_code=404, detail="variant not found")
    # Determine range
    if start is None or end is None:
        raise HTTPException(status_code=400, detail="start and end are required (YYYYMMDD)")
    s = _parse_yyyymmdd(start)
    e = _parse_yyyymmdd(end)
    days = trading_days(s, e)
    if not days:
        return {"ok": True, "updated": 0}

    import json

    # Start from latest snapshot before the first day; if none, use portfolio initial cash.
    pos0 = _latest_position(db, variant_id=int(v.id), before_or_on=days[0])
    if pos0 is None:
        p = db.query(SimPortfolio).filter(SimPortfolio.id == int(v.portfolio_id)).one()
        cash = float(p.initial_cash)
        positions: dict[str, float] = {}
        peak = cash
        mdd = 0.0
    else:
        cash = float(pos0.cash)
        positions = json.loads(pos0.positions_json or "{}")
        peak = float(pos0.nav)
        mdd = float(pos0.mdd or 0.0)

    updated = 0
    for d in days:
        # skip if already exists
        exists = db.query(SimPositionDaily).filter(SimPositionDaily.variant_id == int(v.id), SimPositionDaily.trade_date == d).one_or_none()
        if exists is not None:
            cash = float(exists.cash)
            positions = json.loads(exists.positions_json or "{}")
            peak = max(peak, float(exists.nav))
            mdd = float(exists.mdd or 0.0)
            continue
        nav = float(cash)
        for c, q in positions.items():
            if float(q) <= 0:
                continue
            px = _get_open_price_hfq(db, code=str(c), day=d)
            nav += float(q) * float(px)
        peak = max(peak, nav)
        dd = (nav / peak - 1.0) if peak > 0 else 0.0
        mdd = float(min(mdd, dd))
        snap = SimPositionDaily(
            variant_id=int(v.id),
            trade_date=d,
            positions_json=json.dumps(positions, ensure_ascii=False),
            cash=float(cash),
            nav=float(nav),
            mdd=float(mdd),
        )
        db.add(snap)
        updated += 1
    db.commit()
    return {"ok": True, "updated": int(updated)}


@router.get("/sim/variant/{variant_id}/status")
def sim_variant_status(variant_id: int, db: Session = Depends(get_session)) -> dict:
    v = db.query(SimVariant).filter(SimVariant.id == int(variant_id)).one_or_none()
    if v is None:
        raise HTTPException(status_code=404, detail="variant not found")
    pos = _latest_position(db, variant_id=int(v.id), before_or_on=None)
    import json

    if pos is None:
        p = db.query(SimPortfolio).filter(SimPortfolio.id == int(v.portfolio_id)).one()
        return {"variant_id": int(v.id), "anchor_weekday": int(v.anchor_weekday), "label": v.label, "nav": float(p.initial_cash), "cash": float(p.initial_cash), "positions": {}, "asof": None}
    return {
        "variant_id": int(v.id),
        "anchor_weekday": int(v.anchor_weekday),
        "label": v.label,
        "nav": float(pos.nav),
        "cash": float(pos.cash),
        "positions": json.loads(pos.positions_json or "{}"),
        "asof": pos.trade_date.isoformat(),
        "mdd": pos.mdd,
        "is_active": bool(int(v.is_active)),
    }


@router.get("/sim/variant/{variant_id}/nav")
def sim_variant_nav(variant_id: int, start: str | None = None, end: str | None = None, db: Session = Depends(get_session)) -> dict:
    v = db.query(SimVariant).filter(SimVariant.id == int(variant_id)).one_or_none()
    if v is None:
        raise HTTPException(status_code=404, detail="variant not found")
    q = db.query(SimPositionDaily).filter(SimPositionDaily.variant_id == int(v.id))
    if start:
        q = q.filter(SimPositionDaily.trade_date >= _parse_yyyymmdd(start))
    if end:
        q = q.filter(SimPositionDaily.trade_date <= _parse_yyyymmdd(end))
    rows = list(q.order_by(SimPositionDaily.trade_date.asc()).all())
    return {"variant_id": int(v.id), "dates": [r.trade_date.isoformat() for r in rows], "nav": [float(r.nav) for r in rows], "mdd": [None if r.mdd is None else float(r.mdd) for r in rows]}


@router.get("/sim/variant/{variant_id}/trades")
def sim_variant_trades(variant_id: int, start: str | None = None, end: str | None = None, db: Session = Depends(get_session)) -> dict:
    v = db.query(SimVariant).filter(SimVariant.id == int(variant_id)).one_or_none()
    if v is None:
        raise HTTPException(status_code=404, detail="variant not found")
    q = db.query(SimTrade).filter(SimTrade.variant_id == int(v.id))
    if start:
        q = q.filter(SimTrade.trade_date >= _parse_yyyymmdd(start))
    if end:
        q = q.filter(SimTrade.trade_date <= _parse_yyyymmdd(end))
    rows = list(q.order_by(SimTrade.trade_date.asc(), SimTrade.id.asc()).all())
    return {
        "variant_id": int(v.id),
        "trades": [
            {
                "id": int(t.id),
                "trade_date": t.trade_date.isoformat(),
                "code": t.code,
                "side": t.side,
                "price": float(t.price),
                "qty": float(t.qty),
                "amount": float(t.amount),
                "decision_id": t.decision_id,
            }
            for t in rows
        ],
    }


@router.post("/analysis/baseline/montecarlo")
def baseline_montecarlo(payload: BaselineMonteCarloRequest, db: Session = Depends(get_session)) -> dict:
    # reuse baseline computation to ensure exact same portfolio construction
    base = baseline_analysis(payload, db=db)
    try:
        import pandas as pd

        nav = pd.Series(base["nav"]["series"]["EW"], index=pd.to_datetime(base["nav"]["dates"]), dtype=float)
    except Exception as e:  # pylint: disable=broad-exception-caught
        raise HTTPException(status_code=500, detail=f"invalid baseline nav payload: {e}") from e
    daily_ret = nav.pct_change().fillna(0.0)
    if payload.sample_window_days is not None:
        daily_ret = daily_ret.tail(int(payload.sample_window_days))
    cfg = MonteCarloConfig(n_sims=payload.n_sims, block_size=payload.block_size, seed=payload.seed)
    try:
        mc = bootstrap_metrics_from_daily_returns(daily_ret, rf=float(payload.risk_free_rate), cfg=cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "meta": {
            "type": "baseline",
            "codes": payload.codes,
            "start": payload.start,
            "end": payload.end,
            "sample_window_days": payload.sample_window_days,
        },
        "mc": mc,
    }


@router.post("/analysis/rotation/montecarlo")
def rotation_montecarlo(payload: RotationMonteCarloRequest, db: Session = Depends(get_session)) -> dict:
    rot = rotation_backtest(payload, db=db)
    try:
        import pandas as pd

        nav = pd.Series(rot["nav"]["series"]["ROTATION"], index=pd.to_datetime(rot["nav"]["dates"]), dtype=float)
        excess = pd.Series(rot["nav"]["series"]["EXCESS"], index=pd.to_datetime(rot["nav"]["dates"]), dtype=float)
    except Exception as e:  # pylint: disable=broad-exception-caught
        raise HTTPException(status_code=500, detail=f"invalid rotation nav payload: {e}") from e
    daily_ret = nav.pct_change().fillna(0.0)
    daily_excess = excess.pct_change().fillna(0.0)
    if payload.sample_window_days is not None:
        daily_ret = daily_ret.tail(int(payload.sample_window_days))
        daily_excess = daily_excess.tail(int(payload.sample_window_days))
    cfg = MonteCarloConfig(n_sims=payload.n_sims, block_size=payload.block_size, seed=payload.seed)
    try:
        mc_strategy = bootstrap_metrics_from_daily_returns(daily_ret, rf=float(payload.risk_free_rate), cfg=cfg)
        mc_excess = bootstrap_metrics_from_daily_returns(daily_excess, rf=0.0, cfg=cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "meta": {
            "type": "rotation",
            "codes": payload.codes,
            "start": payload.start,
            "end": payload.end,
            "rebalance": payload.rebalance,
            "sample_window_days": payload.sample_window_days,
        },
        "mc": {"strategy": mc_strategy, "excess": mc_excess},
    }


@router.get("/validation-policies", response_model=list[ValidationPolicyOut])
def get_policies(db: Session = Depends(get_session)) -> list[ValidationPolicyOut]:
    items = list_validation_policies(db)
    return [
        ValidationPolicyOut(
            id=p.id,
            name=p.name,
            description=p.description,
            max_abs_return=p.max_abs_return,
            max_hl_spread=p.max_hl_spread,
            max_gap_days=p.max_gap_days,
        )
        for p in items
    ]


@router.get("/etf", response_model=list[EtfPoolOut])
def get_etfs(adjust: str = "hfq", db: Session = Depends(get_session)) -> list[EtfPoolOut]:
    items = list_etf_pool(db)
    out: list[EtfPoolOut] = []
    for i in items:
        p = get_validation_policy_by_id(db, i.validation_policy_id) if i.validation_policy_id else None
        try:
            rng = get_price_date_range(db, code=i.code, adjust=adjust)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        out.append(
            EtfPoolOut(
                code=i.code,
                name=i.name,
                start_date=i.start_date,
                end_date=i.end_date,
                validation_policy=(
                    ValidationPolicyOut(
                        id=p.id,
                        name=p.name,
                        description=p.description,
                        max_abs_return=p.max_abs_return,
                        max_hl_spread=p.max_hl_spread,
                        max_gap_days=p.max_gap_days,
                    )
                    if p
                    else None
                ),
                max_abs_return_override=i.max_abs_return_override,
                max_abs_return_effective=(
                    i.max_abs_return_override if i.max_abs_return_override is not None else (p.max_abs_return if p else None)
                ),
                last_fetch_status=i.last_fetch_status,
                last_fetch_message=i.last_fetch_message,
                last_data_start_date=rng[0],
                last_data_end_date=rng[1],
            )
        )
    return out


@router.post("/etf", response_model=EtfPoolOut)
def upsert_etf(payload: EtfPoolUpsert, db: Session = Depends(get_session)) -> EtfPoolOut:
    if payload.start_date and len(payload.start_date) != 8:
        raise HTTPException(status_code=400, detail="start_date must be YYYYMMDD")
    if payload.end_date and len(payload.end_date) != 8:
        raise HTTPException(status_code=400, detail="end_date must be YYYYMMDD")
    if payload.start_date and payload.end_date and payload.start_date > payload.end_date:
        raise HTTPException(status_code=400, detail="start_date must be <= end_date")

    policy_id = payload.validation_policy_id
    if policy_id is None:
        # auto-infer using name (MVP); map inferred policy_name -> id
        inferred = infer_policy_name(code=payload.code, name=payload.name)
        p = get_validation_policy_by_name(db, inferred.policy_name)
        policy_id = p.id if p else None

    obj = upsert_etf_pool(
        db,
        code=payload.code,
        name=payload.name,
        start_date=payload.start_date,
        end_date=payload.end_date,
        validation_policy_id=policy_id,
        max_abs_return_override=payload.max_abs_return_override,
    )
    p = get_validation_policy_by_id(db, obj.validation_policy_id) if obj.validation_policy_id else None
    return EtfPoolOut(
        code=obj.code,
        name=obj.name,
        start_date=obj.start_date,
        end_date=obj.end_date,
        validation_policy=(
            ValidationPolicyOut(
                id=p.id,
                name=p.name,
                description=p.description,
                max_abs_return=p.max_abs_return,
                max_hl_spread=p.max_hl_spread,
                max_gap_days=p.max_gap_days,
            )
            if p
            else None
        ),
        max_abs_return_override=obj.max_abs_return_override,
        max_abs_return_effective=(
            obj.max_abs_return_override if obj.max_abs_return_override is not None else (p.max_abs_return if p else None)
        ),
        last_fetch_status=obj.last_fetch_status,
        last_fetch_message=obj.last_fetch_message,
        last_data_start_date=obj.last_data_start_date,
        last_data_end_date=obj.last_data_end_date,
    )


@router.delete("/etf/{code}")
def delete_etf(code: str, purge: bool = False, db: Session = Depends(get_session)) -> dict:
    ok = delete_etf_pool(db, code)
    if not ok:
        raise HTTPException(status_code=404, detail="ETF not found")
    purged = None
    if purge:
        purged = purge_etf_data(db, code=code)
    db.commit()
    return {"deleted": True, "purged": purged}


@router.post("/etf/{code}/fetch", response_model=FetchResult)
def fetch_one(
    code: str,
    adjust: str = "hfq",  # pylint: disable=unused-argument  # backward-compat; ignored (we always fetch all adjusts)
    db: Session = Depends(get_session),
    ak=Depends(get_akshare),
) -> FetchResult:
    settings = get_settings()
    item = next((x for x in list_etf_pool(db) if x.code == code), None)
    if item is None:
        raise HTTPException(status_code=404, detail="ETF not found")

    start = item.start_date or settings.default_start_date
    end = item.end_date or settings.default_end_date

    total = 0
    ok = True
    parts: list[str] = []
    batch_ids: list[int] = []
    for adj in _ALL_ADJUSTS:
        res = ingest_one_etf(db, ak=ak, code=code, start_date=start, end_date=end, adjust=adj)
        batch_ids.append(int(res.batch_id))
        total += int(res.upserted or 0)
        if res.status != "success":
            ok = False
        extra = f",msg={res.message}" if res.status != "success" and res.message else ""
        parts.append(f"{adj}:{res.status}(batch={res.batch_id},upserted={res.upserted}{extra})")

    if ok:
        try:
            _ensure_adjust_ranges_consistent(db, code)
        except ValueError as e:
            ok = False
            parts.append(f"range_check:failed({e})")
            try:
                _rollback_batches_best_effort(db, batch_ids, reason="auto rollback: adjust range mismatch")
            except Exception as rb_e:  # pylint: disable=broad-exception-caught
                parts.append(f"rollback:failed({rb_e})")

    status = "success" if ok else "failed"
    msg = "; ".join(parts)
    mark_fetch_status(db, code=code, status=status, message=msg)
    db.commit()
    if not ok:
        raise HTTPException(status_code=500, detail=msg or "ingestion failed")
    return FetchResult(code=code, inserted_or_updated=total, status="success", message=msg)


@router.post("/fetch-all", response_model=list[FetchResult])
def fetch_all(
    adjust: str = "hfq",  # pylint: disable=unused-argument  # backward-compat; ignored
    db: Session = Depends(get_session),
    ak=Depends(get_akshare),
) -> list[FetchResult]:
    out: list[FetchResult] = []
    for item in list_etf_pool(db):
        start = item.start_date or get_settings().default_start_date
        end = item.end_date or get_settings().default_end_date
        total = 0
        ok = True
        parts: list[str] = []
        batch_ids: list[int] = []
        for adj in _ALL_ADJUSTS:
            res = ingest_one_etf(db, ak=ak, code=item.code, start_date=start, end_date=end, adjust=adj)
            batch_ids.append(int(res.batch_id))
            total += int(res.upserted or 0)
            if res.status != "success":
                ok = False
            extra = f",msg={res.message}" if res.status != "success" and res.message else ""
            parts.append(f"{adj}:{res.status}(batch={res.batch_id},upserted={res.upserted}{extra})")
        if ok:
            try:
                _ensure_adjust_ranges_consistent(db, item.code)
            except ValueError as e:
                ok = False
                parts.append(f"range_check:failed({e})")
                try:
                    _rollback_batches_best_effort(db, batch_ids, reason="auto rollback: adjust range mismatch")
                except Exception as rb_e:  # pylint: disable=broad-exception-caught
                    parts.append(f"rollback:failed({rb_e})")
        status = "success" if ok else "failed"
        msg = "; ".join(parts)
        mark_fetch_status(db, code=item.code, status=status, message=msg)
        out.append(FetchResult(code=item.code, inserted_or_updated=(total if ok else 0), status=status, message=msg))
    return out


@router.post("/fetch-selected", response_model=list[FetchResult])
def fetch_selected(
    payload: FetchSelectedRequest,
    db: Session = Depends(get_session),
    ak=Depends(get_akshare),
) -> list[FetchResult]:
    items_by_code = {x.code: x for x in list_etf_pool(db)}
    out: list[FetchResult] = []
    for code in payload.codes:
        item = items_by_code.get(code)
        if item is None:
            out.append(FetchResult(code=code, inserted_or_updated=0, status="failed", message="ETF not found"))
            continue
        start = item.start_date or get_settings().default_start_date
        end = item.end_date or get_settings().default_end_date
        total = 0
        ok = True
        parts: list[str] = []
        batch_ids: list[int] = []
        for adj in _ALL_ADJUSTS:
            res = ingest_one_etf(db, ak=ak, code=item.code, start_date=start, end_date=end, adjust=adj)
            batch_ids.append(int(res.batch_id))
            total += int(res.upserted or 0)
            if res.status != "success":
                ok = False
            extra = f",msg={res.message}" if res.status != "success" and res.message else ""
            parts.append(f"{adj}:{res.status}(batch={res.batch_id},upserted={res.upserted}{extra})")
        if ok:
            try:
                _ensure_adjust_ranges_consistent(db, item.code)
            except ValueError as e:
                ok = False
                parts.append(f"range_check:failed({e})")
                try:
                    _rollback_batches_best_effort(db, batch_ids, reason="auto rollback: adjust range mismatch")
                except Exception as rb_e:  # pylint: disable=broad-exception-caught
                    parts.append(f"rollback:failed({rb_e})")
        status = "success" if ok else "failed"
        msg = "; ".join(parts)
        mark_fetch_status(db, code=item.code, status=status, message=msg)
        out.append(FetchResult(code=item.code, inserted_or_updated=(total if ok else 0), status=status, message=msg))
    return out


@router.get("/batches", response_model=list[IngestionBatchOut])
def list_batches(code: str | None = None, limit: int = 50, db: Session = Depends(get_session)) -> list[IngestionBatchOut]:
    items = list_ingestion_batches(db, code=code, limit=limit)
    return [
        IngestionBatchOut(
            id=b.id,
            code=b.code,
            start_date=b.start_date,
            end_date=b.end_date,
            source=b.source,
            adjust=b.adjust,
            status=b.status,
            message=b.message,
            snapshot_path=b.snapshot_path,
            pre_fingerprint=b.pre_fingerprint,
            post_fingerprint=b.post_fingerprint,
            val_max_abs_return=b.val_max_abs_return,
            val_max_hl_spread=b.val_max_hl_spread,
            val_max_gap_days=b.val_max_gap_days,
        )
        for b in items
    ]


@router.get("/batches/{batch_id}", response_model=IngestionBatchOut)
def get_batch(batch_id: int, db: Session = Depends(get_session)) -> IngestionBatchOut:
    b = get_ingestion_batch(db, batch_id)
    if b is None:
        raise HTTPException(status_code=404, detail="batch not found")
    return IngestionBatchOut(
        id=b.id,
        code=b.code,
        start_date=b.start_date,
        end_date=b.end_date,
        source=b.source,
        adjust=b.adjust,
        status=b.status,
        message=b.message,
        snapshot_path=b.snapshot_path,
        pre_fingerprint=b.pre_fingerprint,
        post_fingerprint=b.post_fingerprint,
        val_max_abs_return=b.val_max_abs_return,
        val_max_hl_spread=b.val_max_hl_spread,
        val_max_gap_days=b.val_max_gap_days,
    )


@router.post("/batches/{batch_id}/rollback")
def rollback_batch(batch_id: int, db: Session = Depends(get_session)) -> dict[str, str]:
    res = rollback_batch_with_fallback(db, batch_id=batch_id)
    if res.status == "failed":
        raise HTTPException(status_code=500, detail=res.message or "rollback failed")
    return {"status": res.status, "message": res.message or ""}


@router.get("/etf/{code}/prices", response_model=list[PriceOut])
def get_prices(
    code: str,
    start: str | None = None,
    end: str | None = None,
    adjust: str = "hfq",
    limit: int = 5000,
    db: Session = Depends(get_session),
) -> list[PriceOut]:
    start_d = _parse_yyyymmdd(start) if start else None
    end_d = _parse_yyyymmdd(end) if end else None
    rows = list_prices(db, code=code, start_date=start_d, end_date=end_d, adjust=adjust, limit=limit)
    return [
        PriceOut(
            code=r.code,
            trade_date=r.trade_date.isoformat(),
            open=r.open,
            high=r.high,
            low=r.low,
            close=r.close,
            volume=r.volume,
            amount=r.amount,
            source=r.source,
            adjust=r.adjust,
        )
        for r in rows
    ]


@router.delete("/etf/{code}/prices")
def delete_prices_api(
    code: str,
    start: str | None = None,
    end: str | None = None,
    adjust: str = "hfq",
    db: Session = Depends(get_session),
) -> dict[str, int]:
    start_d = _parse_yyyymmdd(start) if start else None
    end_d = _parse_yyyymmdd(end) if end else None
    n = delete_prices(db, code=code, start_date=start_d, end_date=end_d, adjust=adjust)
    db.commit()
    return {"deleted": n}

