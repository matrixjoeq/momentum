from __future__ import annotations

import logging
import json

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Header, Request
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

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
    LeadLagAnalysisRequest,
    LeadLagAnalysisResponse,
    MacroPairLeadLagRequest,
    MacroPairLeadLagResponse,
    MacroStep1Request,
    MacroStep1Response,
    MacroStep2Request,
    MacroStep2Response,
    MacroStep3Request,
    MacroStep3Response,
    MacroStep4Request,
    MacroStep4Response,
    VixNextActionRequest,
    VixNextActionResponse,
    VixSignalBacktestRequest,
    VixSignalBacktestResponse,
    IndexDistributionRequest,
    IndexDistributionResponse,
    VolProxyTimingRequest,
    VolProxyTimingResponse,
    EtfPoolOut,
    EtfPoolUpsert,
    FetchResult,
    FetchSelectedRequest,
    IngestionBatchOut,
    SyncFixedPoolRequest,
    SyncFixedPoolResponse,
    SyncJobOut,
    SyncJobTriggerResponse,
    PriceOut,
    ValidationPolicyOut,
)
from ..api.index_distributions import IndexDistributionInputs, compute_cboe_index_distribution
from ..analysis.baseline import (
    TRADING_DAYS_PER_YEAR,
    BaselineInputs,
    _annualized_return,
    _annualized_vol,
    _compute_return_risk_contributions,
    _max_drawdown,
    _max_drawdown_duration_days,
    _rsi_wilder,
    _rolling_drawdown,
    _sharpe,
    _sortino,
    _ulcer_index,
    compute_baseline,
    load_close_prices,
    load_ohlc_prices,
)
from ..analysis.calendar_effect import BaselineCalendarEffectInputs, compute_baseline_calendar_effect, compute_rotation_calendar_effect
from ..analysis.calendar_effect import _decision_dates_for_rebalance as _cal_decision_dates_for_rebalance
from ..analysis.calendar_effect import _ew_nav_and_weights_by_decision_dates as _cal_ew_nav_and_weights_by_decision_dates
from ..analysis.montecarlo import MonteCarloConfig, bootstrap_metrics_from_daily_returns
from ..analysis.rotation import RotationAnalysisInputs, compute_rotation_backtest
from ..analysis.trend import TrendInputs, compute_trend_backtest
from ..analysis.leadlag import LeadLagInputs, compute_lead_lag, align_us_close_to_cn_next_trading_day
from ..analysis.macro import analyze_pair_leadlag, load_macro_close_series
from ..analysis.vol_proxy import VolProxySpec, compute_vol_proxy_levels
from ..analysis.vol_timing import (
    backtest_tiered_exposure_by_level,
    backtest_tiered_exposure_by_level_rolling_quantiles,
)
from ..calendar.trading_calendar import shift_to_trading_day
from ..data.cboe_fetcher import FetchRequest as CboeFetchRequest
from ..data.cboe_fetcher import fetch_cboe_daily_close
from ..data.fred_fetcher import FetchRequest as FredFetchRequest
from ..data.fred_fetcher import fetch_fred_daily_close
from ..strategy.vix_signal import VixSignalInputs, backtest_vix_next_day_signal, generate_next_action
from ..data.ingestion import ingest_one_etf
from ..data.rollback import logical_rollback_batch, rollback_batch_with_fallback
from ..data.stooq_fetcher import FetchRequest as StooqFetchRequest
from ..data.stooq_fetcher import fetch_stooq_daily_close
from ..data.yahoo_fetcher import FetchRequest as YahooFetchRequest
from ..data.yahoo_fetcher import fetch_yahoo_daily_close_with_alias
from ..db.repo import (
    create_sync_job,
    delete_etf_pool,
    delete_prices,
    get_sync_job_by_dedupe_key,
    get_sync_job,
    get_etf_pool_by_code,
    get_price_date_range,
    get_ingestion_batch,
    update_sync_job,
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
    update_etf_pool_data_range,
)
from ..settings import get_settings
from ..validation.policy_infer import infer_policy_name
from ..calendar.trading_calendar import trading_days
from ..db.models import EtfPrice, SimDecision, SimPortfolio, SimPositionDaily, SimStrategyConfig, SimTrade, SimVariant
from ..calendar.trading_calendar import is_trading_day
from ..services.market_sync import sync_fixed_pool_prices

logger = logging.getLogger(__name__)

router = APIRouter()

_ALL_ADJUSTS = ("qfq", "hfq", "none")

_YAHOO_ALIASES = {
    # User wording sometimes uses "GVX"; Yahoo commonly uses "^GVZ" for CBOE Gold Volatility.
    "GVX": ["^GVX", "^GVZ"],
    "^GVX": ["^GVX", "^GVZ"],
    "GVZ": ["^GVZ"],
    "^GVZ": ["^GVZ"],
    "VXN": ["^VXN"],
    "^VXN": ["^VXN"],
}


def _normalize_vol_index(sym: str) -> str | None:
    s = str(sym or "").strip().upper()
    if s.startswith("^"):
        s = s[1:]
    if s in {"VIX"}:
        return "VIX"
    if s in {"VXN"}:
        return "VXN"
    if s in {"GVZ", "GVX"}:
        return "GVZ"
    return None


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


def _now_shanghai_date() -> dt.date:
    # Keep timezone logic lightweight (no external deps). WeChat cloud runs in CN region by default.
    return (dt.datetime.utcnow() + dt.timedelta(hours=8)).date()


def _dedupe_key_fixed_pool(run_date: dt.date, *, full_refresh: bool, adjusts: list[str]) -> str:
    ad = ",".join([str(x).strip().lower() for x in (adjusts or []) if str(x).strip()])
    return f"fixed_pool:{run_date.strftime('%Y%m%d')}:full={int(bool(full_refresh))}:adj={ad}"


def _job_to_out(job) -> SyncJobOut:
    def _loads(s: str | None) -> dict | None:
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None

    adjusts = [x for x in str(getattr(job, "adjusts", "") or "").split(",") if x]
    return SyncJobOut(
        id=int(job.id),
        status=str(job.status),
        job_type=str(job.job_type),
        dedupe_key=str(job.dedupe_key),
        run_date=(job.run_date.isoformat() if getattr(job, "run_date", None) is not None else None),
        full_refresh=bool(job.full_refresh),
        adjusts=adjusts,
        progress=_loads(getattr(job, "progress_json", None)),
        result=_loads(getattr(job, "result_json", None)),
        error_message=(str(job.error_message) if getattr(job, "error_message", None) else None),
        created_at=job.created_at.isoformat(),
        started_at=(job.started_at.isoformat() if getattr(job, "started_at", None) else None),
        finished_at=(job.finished_at.isoformat() if getattr(job, "finished_at", None) else None),
    )


def _run_sync_job_fixed_pool(
    session_factory: sessionmaker[Session],
    *,
    job_id: int,
    run_date: dt.date,
    adjusts: list[str],
    full_refresh: bool | None,
) -> None:
    """
    Background job runner (runs inside the cloud-run instance).
    """
    import akshare as ak  # local import to keep startup light

    db = session_factory()
    try:
        update_sync_job(db, job_id=job_id, status="running", started_at=dt.datetime.now(dt.timezone.utc), progress={"phase": "starting"})
        db.commit()

        cal = getattr(get_settings(), "auto_sync_calendar", "XSHG")
        if not is_trading_day(run_date, cal=cal):
            update_sync_job(
                db,
                job_id=job_id,
                status="success",
                finished_at=dt.datetime.now(dt.timezone.utc),
                result={"ok": True, "skipped": True, "reason": f"not_trading_day({cal})", "date": run_date.strftime("%Y%m%d")},
            )
            db.commit()
            return

        def _progress(p: dict) -> None:
            # Write lightweight progress so pollers can show activity.
            update_sync_job(db, job_id=job_id, progress=p)
            db.commit()

        res = sync_fixed_pool_prices(
            db=db,
            ak=ak,
            run_date=run_date,
            adjusts=adjusts,
            full_refresh=full_refresh,
            progress_cb=_progress,
        )
        # commit any data ingestion performed during sync
        db.commit()

        ok = bool(res.get("ok"))
        # Summarize failures for quick visibility in job list.
        err_summary: str | None = None
        if not ok:
            failed_pairs: list[str] = []
            codes_obj = (res.get("codes") if isinstance(res, dict) else None) or {}
            if isinstance(codes_obj, dict):
                for code, co in codes_obj.items():
                    ad = (co.get("adjusts") if isinstance(co, dict) else None) or {}
                    if not isinstance(ad, dict):
                        continue
                    for adj, ao in ad.items():
                        if isinstance(ao, dict) and str(ao.get("status")) != "success" and not bool(ao.get("skipped")):
                            failed_pairs.append(f"{code}:{adj}")
            if failed_pairs:
                err_summary = "sync failed: " + ",".join(failed_pairs[:10]) + ("..." if len(failed_pairs) > 10 else "")
        update_sync_job(
            db,
            job_id=job_id,
            status=("success" if ok else "failed"),
            finished_at=dt.datetime.now(dt.timezone.utc),
            result=dict(res),
            error_message=(None if ok else (err_summary or "sync failed; see result.codes for details")),
            progress={"phase": "done", "ok": ok},
        )
        db.commit()
    except Exception as e:  # pylint: disable=broad-exception-caught
        try:
            update_sync_job(
                db,
                job_id=job_id,
                status="failed",
                finished_at=dt.datetime.now(dt.timezone.utc),
                error_message=str(e),
                progress={"phase": "failed"},
            )
            db.commit()
        except Exception:
            pass
        raise
    finally:
        db.close()


@router.post("/admin/sync/fixed-pool", response_model=SyncFixedPoolResponse)
def admin_sync_fixed_pool(
    payload: SyncFixedPoolRequest,
    db: Session = Depends(get_session),
    x_momentum_token: str | None = Header(default=None, alias="X-Momentum-Token"),
) -> SyncFixedPoolResponse:
    """
    Sync fixed 4-ETF pool prices (qfq/hfq/none).

    Designed for WeChat Cloud scheduled trigger (HTTP):
    - Set env MOMENTUM_SYNC_TOKEN in cloud hosting, and pass it via header `X-Momentum-Token` or JSON body `token`.
    - If triggered on non-trading day, it will skip safely.
    """
    settings = get_settings()
    expected = getattr(settings, "sync_token", None)
    if expected:
        provided = (payload.token or x_momentum_token or "").strip()
        if (not provided) or (provided != expected):
            raise HTTPException(status_code=403, detail="invalid sync token")

    run_date = _now_shanghai_date() if not payload.date else _parse_yyyymmdd(payload.date)
    cal = getattr(settings, "auto_sync_calendar", "XSHG")
    if not is_trading_day(run_date, cal=cal):
        return SyncFixedPoolResponse(
            ok=True,
            skipped=True,
            reason=f"not_trading_day({cal})",
            date=run_date.strftime("%Y%m%d"),
            full_refresh=bool(settings.auto_sync_full_refresh if payload.full_refresh is None else payload.full_refresh),
            adjusts=[str(x).strip().lower() for x in (payload.adjusts or [])],
            codes={},
        )

    ak = get_akshare()
    res = sync_fixed_pool_prices(db=db, ak=ak, run_date=run_date, adjusts=payload.adjusts, full_refresh=payload.full_refresh)
    # ensure pool coverage updated (best effort) - already done per code inside service
    for code in _FIXED_CODES:
        try:
            update_etf_pool_data_range(db, code=code)
        except Exception:  # pragma: no cover
            pass
    db.commit()
    return SyncFixedPoolResponse(
        ok=bool(res.get("ok")),
        skipped=False,
        reason=None,
        date=str(res.get("date")),
        full_refresh=bool(res.get("full_refresh")),
        adjusts=list(res.get("adjusts") or []),
        codes=dict(res.get("codes") or {}),
    )


@router.post("/admin/sync/fixed-pool/async", response_model=SyncJobTriggerResponse)
def admin_sync_fixed_pool_async(
    payload: SyncFixedPoolRequest,
    request: Request,
    background: BackgroundTasks,
    db: Session = Depends(get_session),
    x_momentum_token: str | None = Header(default=None, alias="X-Momentum-Token"),
) -> SyncJobTriggerResponse:
    """
    Trigger a long-running sync job and return immediately with job_id.
    Intended for cloud function triggers with short HTTP timeouts.
    """
    settings = get_settings()
    expected = getattr(settings, "sync_token", None)
    if expected:
        provided = (payload.token or x_momentum_token or "").strip()
        if (not provided) or (provided != expected):
            raise HTTPException(status_code=403, detail="invalid sync token")

    run_date = _now_shanghai_date() if not payload.date else _parse_yyyymmdd(payload.date)
    adjusts = [str(x).strip().lower() for x in (payload.adjusts or []) if str(x).strip()]
    if not adjusts:
        adjusts = ["qfq", "hfq", "none"]
    do_full = bool(settings.auto_sync_full_refresh if payload.full_refresh is None else payload.full_refresh)
    dedupe = _dedupe_key_fixed_pool(run_date, full_refresh=do_full, adjusts=adjusts)

    # If an identical job is already queued/running, return it (dedupe).
    existing = get_sync_job_by_dedupe_key(db, dedupe)
    if (not bool(payload.force_new)) and existing is not None and str(existing.status) in {"queued", "running"}:
        # Reconcile "stuck" jobs:
        # - Cloud-run instances can be restarted mid-job, leaving status=running forever.
        # - Some older versions may have written finished_at/result but failed to flip status.
        now = dt.datetime.now(dt.timezone.utc)
        try:
            if getattr(existing, "finished_at", None) is not None:
                ok2: bool | None = None
                try:
                    if getattr(existing, "result_json", None):
                        ok2 = bool((json.loads(str(existing.result_json)) or {}).get("ok"))
                except Exception:  # pragma: no cover
                    ok2 = None
                update_sync_job(
                    db,
                    job_id=int(existing.id),
                    status=("success" if ok2 else "failed"),
                    error_message=(None if ok2 else (existing.error_message or "sync failed; see result.codes for details")),
                )
                db.commit()
                existing = get_sync_job_by_dedupe_key(db, dedupe)
            else:
                # Stale timeout: if running for too long, mark as failed so a retry can be created.
                # (Full refresh can be slow; keep this conservative.)
                started_at = getattr(existing, "started_at", None)
                created_at = getattr(existing, "created_at", None)
                stale = False
                if started_at is not None and (now - started_at) > dt.timedelta(minutes=60):
                    stale = True
                if started_at is None and created_at is not None and (now - created_at) > dt.timedelta(minutes=10):
                    stale = True
                if stale:
                    update_sync_job(
                        db,
                        job_id=int(existing.id),
                        status="failed",
                        finished_at=now,
                        error_message="stale job: previous instance likely restarted; please retry",
                        progress={"phase": "stale"},
                    )
                    db.commit()
                    existing = get_sync_job_by_dedupe_key(db, dedupe)
        except Exception:  # pragma: no cover
            # best-effort; fall through to normal dedupe behavior
            pass

        if existing is not None and str(existing.status) in {"queued", "running"}:
            return SyncJobTriggerResponse(job_id=int(existing.id), status=str(existing.status), dedupe_key=str(existing.dedupe_key))

    job = create_sync_job(
        db,
        dedupe_key=dedupe,
        run_date=run_date,
        full_refresh=do_full,
        adjusts=adjusts,
        force_new=bool(payload.force_new),
    )
    db.commit()

    # Ensure app state exists (engine/session_factory) and schedule background runner.
    sf: sessionmaker[Session] = request.app.state.session_factory
    background.add_task(_run_sync_job_fixed_pool, sf, job_id=int(job.id), run_date=run_date, adjusts=adjusts, full_refresh=do_full)
    return SyncJobTriggerResponse(job_id=int(job.id), status=str(job.status), dedupe_key=str(job.dedupe_key))


@router.get("/admin/sync/jobs/{job_id}", response_model=SyncJobOut)
def admin_sync_job_status(job_id: int, db: Session = Depends(get_session)) -> SyncJobOut:
    job = get_sync_job(db, int(job_id))
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _job_to_out(job)


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


def _load_vol_index_close_for_rotation_rules(
    rules: list[dict] | None, *, end_yyyymmdd: str
) -> dict[str, pd.Series] | None:
    """
    Preload Cboe vol index close series needed by rotation vol-index timing rules.

    Returns dict[index_code -> aligned_series], where aligned_series maps US close date to CN next trading day.
    """
    if rules is None or len(rules) == 0:
        return None
    need = sorted({str(r.get("index") or "").strip().upper() for r in rules if str(r.get("index") or "").strip()})
    if not need:
        return None

    out: dict[str, pd.Series] = {}
    for idx_code in need:
        if idx_code not in {"VIX", "GVZ"}:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported vol index for rotation timing: {idx_code} (expected VIX/GVZ)",
            )
        start_fetch = "19900101" if idx_code == "VIX" else "20080101"
        dfc = fetch_cboe_daily_close(CboeFetchRequest(index=idx_code, start_date=start_fetch, end_date=end_yyyymmdd))
        if dfc is None or dfc.empty:
            raise HTTPException(status_code=400, detail=f"empty Cboe series for {idx_code}")
        s_us = pd.Series(data=dfc["close"].to_numpy(dtype=float), index=dfc["date"].to_list(), dtype=float).dropna()
        out[idx_code] = align_us_close_to_cn_next_trading_day(s_us, cal="XSHG")
    return out


@router.post("/analysis/rotation")
def rotation_backtest(payload: RotationBacktestRequest, db: Session = Depends(get_session)) -> dict:
    # Pylint may resolve imported dataclasses from an installed package instead of workspace source,
    # which can lag during local dev. Keep behavior correct; suppress false-positive for new fields.
    # pylint: disable=unexpected-keyword-arg
    asset_vol_rules = [r.model_dump() for r in payload.asset_vol_index_rules] if payload.asset_vol_index_rules else None
    vol_index_close = _load_vol_index_close_for_rotation_rules(asset_vol_rules, end_yyyymmdd=payload.end)
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
        asset_rc_rules=[r.model_dump() for r in payload.asset_rc_rules] if payload.asset_rc_rules else None,
        asset_vol_index_rules=asset_vol_rules,
        vol_index_close=vol_index_close,
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


@router.post("/analysis/leadlag", response_model=LeadLagAnalysisResponse)
def analysis_leadlag(
    payload: LeadLagAnalysisRequest,
    db: Session = Depends(get_session),
) -> LeadLagAnalysisResponse:
    """
    Analyze lead/lag and Granger causality between:
    - ETF daily close (from DB)
    - Volatility index daily close (Cboe CSV preferred; Yahoo optional fallback)
    """
    start_d = _parse_yyyymmdd(payload.start)
    end_d = _parse_yyyymmdd(payload.end)
    adj = str(payload.adjust or "hfq").strip().lower()
    code = str(payload.etf_code).strip()

    asset_provider = str(getattr(payload, "asset_provider", "db") or "db").strip().lower()
    asset_symbol = getattr(payload, "asset_symbol", None)
    asset_symbol_s = str(asset_symbol).strip() if asset_symbol is not None else ""

    # Asset (left-hand series) selection:
    # - db: load ETF closes from DB (needs etf_code + adjust)
    # - stooq/yahoo: fetch closes externally (uses asset_symbol)
    if asset_provider in {"", "db"}:
        rows = list_prices(db, code=code, start_date=start_d, end_date=end_d, adjust=adj, limit=1000000)
        if not rows:
            return LeadLagAnalysisResponse(ok=False, error="empty_etf_series")
        etf = pd.Series(
            data=[float(r.close) if r.close is not None else np.nan for r in rows],
            index=[r.trade_date for r in rows],
            dtype=float,
        ).dropna()
        if etf.empty:
            return LeadLagAnalysisResponse(ok=False, error="empty_etf_close")
        asset_meta = {"asset_provider": "db", "etf_code": code, "adjust": adj}
    else:
        if not asset_symbol_s:
            return LeadLagAnalysisResponse(ok=False, error="missing_asset_symbol", meta={"asset_provider": asset_provider})
        if asset_provider == "auto":
            # heuristic: prefer stooq for known stooq-style symbols (like qqq.us or ^ndx), else yahoo
            ap = "stooq" if (".us" in asset_symbol_s.lower() or asset_symbol_s.startswith("^")) else "yahoo"
        else:
            ap = asset_provider

        if ap == "stooq":
            df_a, ameta = fetch_stooq_daily_close(StooqFetchRequest(symbol=asset_symbol_s, start_date=payload.start, end_date=payload.end))
            if df_a is None or df_a.empty:
                err = str((ameta or {}).get("error") or "empty_stooq_series")
                return LeadLagAnalysisResponse(ok=False, error=err, meta={"asset_provider": "stooq", "stooq": ameta})
            etf = pd.Series(data=df_a["close"].to_numpy(dtype=float), index=df_a["date"].to_list(), dtype=float).dropna()
            asset_meta = {"asset_provider": "stooq", "asset_symbol": asset_symbol_s, "stooq": ameta}
        else:
            df_a, ameta = fetch_yahoo_daily_close_with_alias(
                YahooFetchRequest(symbol=asset_symbol_s, start_date=payload.start, end_date=payload.end),
                aliases=_YAHOO_ALIASES,
            )
            if df_a is None or df_a.empty:
                return LeadLagAnalysisResponse(ok=False, error="empty_yahoo_series", meta={"asset_provider": ap, "yahoo": ameta})
            etf = pd.Series(data=df_a["close"].to_numpy(dtype=float), index=df_a["date"].to_list(), dtype=float).dropna()
            asset_meta = {"asset_provider": "yahoo", "asset_symbol": asset_symbol_s, "yahoo": ameta}

        if etf.empty:
            return LeadLagAnalysisResponse(ok=False, error="empty_asset_close", meta=asset_meta)

    idx_provider = str(getattr(payload, "index_provider", "cboe") or "cboe").strip().lower()
    idx_align = str(getattr(payload, "index_align", "cn_next_trading_day") or "cn_next_trading_day").strip().lower()
    idx_sym = str(payload.index_symbol).strip()
    idx_code = _normalize_vol_index(idx_sym)
    idx_sym_u = idx_sym.strip().upper()

    idx: pd.Series
    idx_meta: dict

    fred_known = {"DGS2", "DGS5", "DGS10", "DGS30"}
    sina_known = {"DINIW"}
    # Note: some Stooq symbols (notably certain futures like GC.F / DX.F) may require captcha on
    # historical download pages and can return empty content from the CSV endpoint.
    stooq_known = {"XAUUSD", "GC.F"}

    # Provider selection:
    # - cboe: VIX/GVZ only (preferred when available)
    # - fred: FRED daily series (rates)
    # - stooq: Stooq daily CSV (DXY / XAUUSD / GC)
    # - yahoo: fallback for arbitrary symbols
    # - auto: try in the above order based on symbol
    if idx_provider == "auto":
        if idx_code in {"VIX", "GVZ", "VXN"}:
            idx_provider_eff = "cboe"
        elif idx_sym_u in fred_known:
            idx_provider_eff = "fred"
        elif idx_sym_u in sina_known:
            idx_provider_eff = "sina"
        elif idx_sym_u in stooq_known:
            idx_provider_eff = "stooq"
        else:
            idx_provider_eff = "yahoo"
    else:
        idx_provider_eff = idx_provider

    if idx_provider_eff == "cboe":
        if idx_code not in {"VIX", "GVZ", "VXN"}:
            return LeadLagAnalysisResponse(ok=False, error="unsupported_cboe_index", meta={"provider": "cboe", "symbol": idx_sym})
        dfc = fetch_cboe_daily_close(CboeFetchRequest(index=idx_code, start_date=payload.start, end_date=payload.end))
        if dfc is None or dfc.empty:
            return LeadLagAnalysisResponse(ok=False, error="empty_cboe_series", meta={"cboe": {"index": idx_code}})
        idx = pd.Series(data=dfc["close"].to_numpy(dtype=float), index=dfc["date"].to_list(), dtype=float).dropna()
        idx_meta = {"provider": "cboe", "index": idx_code, "align": idx_align}
    elif idx_provider_eff == "fred":
        settings = get_settings()
        df_fred, fmeta = fetch_fred_daily_close(
            FredFetchRequest(series_id=idx_sym_u, start_date=payload.start, end_date=payload.end),
            api_key=settings.fred_api_key,
        )
        if df_fred is None or df_fred.empty:
            err = str((fmeta or {}).get("error") or "empty_fred_series")
            return LeadLagAnalysisResponse(ok=False, error=err, meta={"fred": fmeta})
        idx = pd.Series(data=df_fred["close"].to_numpy(dtype=float), index=df_fred["date"].to_list(), dtype=float).dropna()
        idx_meta = {"provider": "fred", "series_id": idx_sym_u, "align": idx_align, "fred": fmeta}
    elif idx_provider_eff == "stooq":
        df_stooq, smeta = fetch_stooq_daily_close(
            StooqFetchRequest(symbol=idx_sym, start_date=payload.start, end_date=payload.end)
        )
        if df_stooq is None or df_stooq.empty:
            err = str((smeta or {}).get("error") or "empty_stooq_series")
            return LeadLagAnalysisResponse(ok=False, error=err, meta={"stooq": smeta})
        idx = pd.Series(data=df_stooq["close"].to_numpy(dtype=float), index=df_stooq["date"].to_list(), dtype=float).dropna()
        idx_meta = {"provider": "stooq", "symbol": idx_sym, "align": idx_align, "stooq": smeta}
    elif idx_provider_eff == "sina":
        from ..data.sina_fetcher import (  # local import to keep optional dependency surface small
            FetchRequest as SinaFetchRequest,
            fetch_sina_forex_day_kline_daily_close,
        )

        df_sina, smeta = fetch_sina_forex_day_kline_daily_close(
            SinaFetchRequest(symbol=idx_sym_u, start_date=payload.start, end_date=payload.end)
        )
        if df_sina is None or df_sina.empty:
            err = str((smeta or {}).get("error") or "empty_sina_series")
            return LeadLagAnalysisResponse(ok=False, error=err, meta={"sina": smeta})
        idx = pd.Series(data=df_sina["close"].to_numpy(dtype=float), index=df_sina["date"].to_list(), dtype=float).dropna()
        idx_meta = {"provider": "sina", "symbol": idx_sym_u, "align": idx_align, "sina": smeta}
    else:
        # Yahoo fallback (may be blocked by network policy).
        idx_df, ymeta = fetch_yahoo_daily_close_with_alias(
            YahooFetchRequest(symbol=idx_sym, start_date=payload.start, end_date=payload.end),
            aliases=_YAHOO_ALIASES,
        )
        if idx_df is None or idx_df.empty:
            return LeadLagAnalysisResponse(ok=False, error="empty_yahoo_series", meta={"yahoo": ymeta, "provider_requested": idx_provider_eff})
        idx = pd.Series(data=idx_df["close"].to_numpy(dtype=float), index=idx_df["date"].to_list(), dtype=float).dropna()
        idx_meta = {"provider": "yahoo", "symbol": idx_sym, "align": idx_align, "yahoo": ymeta}

    if idx.empty:
        return LeadLagAnalysisResponse(ok=False, error="empty_index_close", meta=idx_meta)

    res = compute_lead_lag(
        LeadLagInputs(
            etf_close=etf,
            idx_close=idx,
            max_lag=int(payload.max_lag),
            granger_max_lag=int(payload.granger_max_lag),
            alpha=float(payload.alpha),
            index_align=idx_align,
            trade_cost_bps=float(getattr(payload, "trade_cost_bps", 0.0) or 0.0),
            rolling_window=int(getattr(payload, "rolling_window", 252) or 252),
            enable_threshold=bool(getattr(payload, "enable_threshold", True)),
            threshold_quantile=float(getattr(payload, "threshold_quantile", 0.80) or 0.80),
            walk_forward=bool(getattr(payload, "walk_forward", True)),
            train_ratio=float(getattr(payload, "train_ratio", 0.60) or 0.60),
            walk_objective=str(getattr(payload, "walk_objective", "sharpe") or "sharpe"),
            vol_timing=bool(getattr(payload, "vol_timing", False)),
            vol_level_quantiles=list(getattr(payload, "vol_level_quantiles", [0.8]) or [0.8]),
            vol_level_exposures=list(getattr(payload, "vol_level_exposures", [1.0, 0.5]) or [1.0, 0.5]),
            vol_level_window=str(getattr(payload, "vol_level_window", "all") or "all"),
        )
    )
    if not bool(res.get("ok")):
        return LeadLagAnalysisResponse(ok=False, error=str(res.get("reason") or "analysis_failed"), meta=idx_meta)

    meta = dict(res.get("meta") or {})
    meta.update(asset_meta)
    meta.update(idx_meta)
    return LeadLagAnalysisResponse(
        ok=True,
        meta=meta,
        series=res.get("series"),
        corr=res.get("corr"),
        granger=res.get("granger"),
        trade=res.get("trade"),
    )


@router.post("/analysis/vol-proxy-timing", response_model=VolProxyTimingResponse)
def analysis_vol_proxy_timing(payload: VolProxyTimingRequest, db: Session = Depends(get_session)) -> VolProxyTimingResponse:
    """
    Volatility-timing backtests using volatility proxies computed from ETF OHLC only.

    This is designed for comparing "no-options-data" alternatives (RV, range-based, EWMA, HAR forecast)
    with implied-vol indices timing (GVZ/VXN) which is available via /analysis/leadlag.
    """
    start_d = _parse_yyyymmdd(payload.start)
    end_d = _parse_yyyymmdd(payload.end)
    adj = str(payload.adjust or "hfq").strip().lower()
    code = str(payload.etf_code).strip()

    ohlc_mat = load_ohlc_prices(db, codes=[code], start=start_d, end=end_d, adjust=adj)
    close_df = ohlc_mat.get("close", pd.DataFrame())
    if close_df is None or close_df.empty or code not in close_df.columns:
        return VolProxyTimingResponse(ok=False, error="empty_etf_ohlc")

    # per-code OHLC Series (best-effort; can be partially missing for some adjusts)
    open_s = (ohlc_mat.get("open", pd.DataFrame())).get(code)
    high_s = (ohlc_mat.get("high", pd.DataFrame())).get(code)
    low_s = (ohlc_mat.get("low", pd.DataFrame())).get(code)
    close_s = (ohlc_mat.get("close", pd.DataFrame())).get(code)
    if close_s is None:
        return VolProxyTimingResponse(ok=False, error="empty_etf_close")
    ohlc = {"open": open_s, "high": high_s, "low": low_s, "close": close_s}

    cnum = pd.to_numeric(close_s, errors="coerce")
    etf_ret = np.log(cnum.where(cnum > 0)).diff()
    df_ret = pd.DataFrame({"etf_ret": etf_ret}).dropna()
    if df_ret.empty or len(df_ret) < 5:
        return VolProxyTimingResponse(ok=False, error="insufficient_etf_returns")

    qs_in = [float(q) for q in (payload.level_quantiles or []) if q is not None and np.isfinite(float(q))]
    qs = sorted([float(min(max(q, 0.01), 0.99)) for q in qs_in])
    exposures = [float(x) for x in (payload.level_exposures or []) if x is not None and np.isfinite(float(x))]
    level_window = str(getattr(payload, "level_window", "all") or "all").strip().lower()

    def _window_days(key: str) -> int | None:
        k = str(key or "all").strip().lower()
        if k in {"", "all"}:
            return None
        if k == "1y":
            return 252
        if k == "3y":
            return 3 * 252
        if k == "5y":
            return 5 * 252
        if k == "10y":
            return 10 * 252
        return None

    wdays = _window_days(level_window)

    results: dict[str, dict] = {}
    for m in payload.methods:
        name = str(m.name).strip()
        if not name:
            continue
        kind = str(m.kind).strip()
        try:
            spec = VolProxySpec(
                kind=kind,  # type: ignore[arg-type]
                window=int(m.window),
                ann=int(m.ann),
                ewma_lambda=float(m.ewma_lambda),
                har_train_window=int(m.har_train_window),
                har_horizons=tuple(int(x) for x in (m.har_horizons or [1, 5, 22])),
            )
        except Exception as e:  # pragma: no cover
            results[name] = {"ok": False, "error": f"bad_method_spec:{e}"}
            continue

        try:
            lvl = compute_vol_proxy_levels(ohlc, spec=spec)
        except Exception as e:
            results[name] = {"ok": False, "error": f"compute_failed:{e}"}
            continue

        df2 = df_ret.join(pd.DataFrame({"idx_close": lvl}), how="inner").dropna()
        lv = pd.to_numeric(df2["idx_close"], errors="coerce").to_numpy(dtype=float)
        lv = lv[np.isfinite(lv)]
        if len(qs) == 0:
            results[name] = {"ok": False, "error": "empty_level_quantiles"}
            continue
        if lv.size < 20:
            results[name] = {"ok": False, "error": "insufficient_level_samples", "n": int(lv.size)}
            continue
        if len(exposures) != len(qs) + 1:
            results[name] = {
                "ok": False,
                "error": "bad_level_exposures_len",
                "need": int(len(qs) + 1),
                "got": int(len(exposures)),
            }
            continue

        levels = pd.to_numeric(df2["idx_close"], errors="coerce")
        if level_window == "static_all":
            # Research mode: full-sample fixed thresholds (has lookahead bias).
            thr_abs = [float(np.quantile(lv, q)) for q in qs]
            out = backtest_tiered_exposure_by_level(
                df2,
                levels,
                thresholds_abs=thr_abs,
                exposures=exposures,
                cost_bps=float(payload.trade_cost_bps or 0.0),
                ann=int(m.ann),
            )
            out["quantiles"] = qs
            out["level_window"] = "static_all"
            out["thresholds_abs_train"] = thr_abs
        elif wdays is None:
            # Recommended "all": expanding quantiles + shift(1) (no lookahead).
            out = backtest_tiered_exposure_by_level_expanding_quantiles(
                df2,
                levels,
                quantiles=qs,
                exposures=exposures,
                cost_bps=float(payload.trade_cost_bps or 0.0),
                ann=int(m.ann),
            )
            out["level_window"] = "all"
            if out.get("thresholds_abs_last") is not None:
                out["thresholds_abs_train"] = out.get("thresholds_abs_last")
        else:
            out = backtest_tiered_exposure_by_level_rolling_quantiles(
                df2,
                levels,
                quantiles=qs,
                window_days=int(wdays),
                exposures=exposures,
                cost_bps=float(payload.trade_cost_bps or 0.0),
                ann=int(m.ann),
            )
            out["level_window"] = level_window
            if out.get("thresholds_abs_last") is not None:
                out["thresholds_abs_train"] = out.get("thresholds_abs_last")

        # Walk-forward: thresholds learned on train only, applied to train/test.
        vol_walk: dict | None = None
        if bool(payload.walk_forward) and len(df2) >= 80 and level_window == "static_all":
            tr = float(min(max(payload.train_ratio, 0.2), 0.85))
            cut = int(max(20, min(len(df2) - 20, int(len(df2) * tr))))
            train_df = df2.iloc[:cut].copy()
            test_df = df2.iloc[cut:].copy()
            lv_tr = pd.to_numeric(train_df["idx_close"], errors="coerce").to_numpy(dtype=float)
            lv_tr = lv_tr[np.isfinite(lv_tr)]
            if lv_tr.size >= 20:
                thr_tr = [float(np.quantile(lv_tr, q)) for q in qs]
                vol_walk = {
                    "ok": True,
                    "train_ratio": float(tr),
                    "thresholds_abs_train": thr_tr,
                    "train": backtest_tiered_exposure_by_level(
                        train_df,
                        pd.to_numeric(train_df["idx_close"], errors="coerce"),
                        thresholds_abs=thr_tr,
                        exposures=exposures,
                        cost_bps=float(payload.trade_cost_bps or 0.0),
                        ann=int(m.ann),
                    ),
                    "test": backtest_tiered_exposure_by_level(
                        test_df,
                        pd.to_numeric(test_df["idx_close"], errors="coerce"),
                        thresholds_abs=thr_tr,
                        exposures=exposures,
                        cost_bps=float(payload.trade_cost_bps or 0.0),
                        ann=int(m.ann),
                    ),
                }
            else:
                vol_walk = {"ok": False, "reason": "insufficient_train_level_samples", "n": int(lv_tr.size)}
        elif bool(payload.walk_forward) and level_window != "static_all":
            vol_walk = {"ok": False, "reason": f"{level_window}_window_mode"}
        out["walk_forward"] = vol_walk

        # Include the level series itself (aligned to df2 dates) for plotting/debugging.
        out["level_series"] = {
            "dates": [d.isoformat() for d in df2.index],
            "level": pd.to_numeric(df2["idx_close"], errors="coerce").astype(float).tolist(),
        }
        results[name] = out

    meta = {"etf_code": code, "adjust": adj, "start": payload.start, "end": payload.end}
    if not results:
        return VolProxyTimingResponse(ok=False, error="no_methods_computed", meta=meta)
    return VolProxyTimingResponse(ok=True, meta=meta, methods=results)


@router.post("/analysis/macro/pair-leadlag", response_model=MacroPairLeadLagResponse)
def analysis_macro_pair_leadlag(payload: MacroPairLeadLagRequest, db: Session = Depends(get_session)) -> MacroPairLeadLagResponse:
    res = analyze_pair_leadlag(
        db,
        a_series_id=str(payload.a_series_id).strip(),
        b_series_id=str(payload.b_series_id).strip(),
        start=str(payload.start),
        end=str(payload.end),
        index_align=str(payload.index_align or "none"),
        max_lag=int(payload.max_lag),
        granger_max_lag=int(payload.granger_max_lag),
        alpha=float(payload.alpha),
        trade_cost_bps=float(payload.trade_cost_bps),
        rolling_window=int(payload.rolling_window),
        enable_threshold=bool(payload.enable_threshold),
        threshold_quantile=float(payload.threshold_quantile),
        walk_forward=bool(payload.walk_forward),
        train_ratio=float(payload.train_ratio),
        walk_objective=str(payload.walk_objective or "sharpe"),
    )
    if not bool(res.get("ok")):
        return MacroPairLeadLagResponse(ok=False, error=str(res.get("reason") or "analysis_failed"), meta=dict(res))
    return MacroPairLeadLagResponse(
        ok=True,
        meta=res.get("meta"),
        series=res.get("series"),
        corr=res.get("corr"),
        granger=res.get("granger"),
        trade=res.get("trade"),
    )


@router.post("/analysis/macro/step1", response_model=MacroStep1Response)
def analysis_macro_step1(payload: MacroStep1Request, db: Session = Depends(get_session)) -> MacroStep1Response:
    start = str(payload.start)
    end = str(payload.end)
    gold_spot_id = str(payload.gold_spot_series_id).strip()
    dxy_id = str(payload.dxy_series_id).strip()
    yld_id = str(payload.yield_series_id).strip()
    gold_fut_id = str(payload.gold_fut_series_id).strip() if payload.gold_fut_series_id else None

    series: dict[str, dict] = {}
    for sid in [gold_spot_id, dxy_id, yld_id] + ([gold_fut_id] if gold_fut_id else []):
        if not sid:
            continue
        s = load_macro_close_series(db, series_id=sid, start=start, end=end)
        if s.empty:
            return MacroStep1Response(ok=False, error=f"empty_series:{sid}")
        series[sid] = {"dates": [d.isoformat() for d in s.index], "close": s.astype(float).tolist()}

    pairs: dict[str, dict] = {}
    common_kwargs = dict(
        start=start,
        end=end,
        index_align=str(payload.index_align or "none"),
        max_lag=int(payload.max_lag),
        granger_max_lag=int(payload.granger_max_lag),
        alpha=float(payload.alpha),
        rolling_window=int(payload.rolling_window),
        trade_cost_bps=float(payload.trade_cost_bps),
        threshold_quantile=float(payload.threshold_quantile),
        walk_forward=bool(payload.walk_forward),
        train_ratio=float(payload.train_ratio),
        walk_objective=str(payload.walk_objective or "sharpe"),
    )

    # spot vs dxy / yield
    pairs[f"{gold_spot_id}__vs__{dxy_id}"] = analyze_pair_leadlag(db, a_series_id=gold_spot_id, b_series_id=dxy_id, **common_kwargs)
    pairs[f"{gold_spot_id}__vs__{yld_id}"] = analyze_pair_leadlag(db, a_series_id=gold_spot_id, b_series_id=yld_id, **common_kwargs)

    # fut (optional)
    if gold_fut_id:
        pairs[f"{gold_fut_id}__vs__{dxy_id}"] = analyze_pair_leadlag(db, a_series_id=gold_fut_id, b_series_id=dxy_id, **common_kwargs)
        pairs[f"{gold_fut_id}__vs__{yld_id}"] = analyze_pair_leadlag(db, a_series_id=gold_fut_id, b_series_id=yld_id, **common_kwargs)

    meta = {
        "start": start,
        "end": end,
        "gold_spot_series_id": gold_spot_id,
        "gold_fut_series_id": gold_fut_id,
        "dxy_series_id": dxy_id,
        "yield_series_id": yld_id,
        "index_align": str(payload.index_align or "none"),
    }
    return MacroStep1Response(ok=True, meta=meta, series=series, pairs=pairs)


@router.post("/analysis/macro/step2", response_model=MacroStep2Response)
def analysis_macro_step2(payload: MacroStep2Request, db: Session = Depends(get_session)) -> MacroStep2Response:
    start = str(payload.start)
    end = str(payload.end)
    cn_spot_id = str(payload.cn_spot_series_id).strip()
    cnh_id = str(payload.cnh_series_id).strip()
    yld_id = str(payload.yield_series_id).strip()
    cn_fut_id = str(payload.cn_fut_series_id).strip() if payload.cn_fut_series_id else None

    series: dict[str, dict] = {}
    for sid in [cn_spot_id, cnh_id, yld_id] + ([cn_fut_id] if cn_fut_id else []):
        if not sid:
            continue
        s = load_macro_close_series(db, series_id=sid, start=start, end=end)
        if s.empty:
            return MacroStep2Response(ok=False, error=f"empty_series:{sid}")
        series[sid] = {"dates": [d.isoformat() for d in s.index], "close": s.astype(float).tolist()}

    common_kwargs = dict(
        start=start,
        end=end,
        index_align=str(payload.index_align or "none"),
        max_lag=int(payload.max_lag),
        granger_max_lag=int(payload.granger_max_lag),
        alpha=float(payload.alpha),
        rolling_window=int(payload.rolling_window),
        trade_cost_bps=float(payload.trade_cost_bps),
        threshold_quantile=float(payload.threshold_quantile),
        walk_forward=bool(payload.walk_forward),
        train_ratio=float(payload.train_ratio),
        walk_objective=str(payload.walk_objective or "sharpe"),
    )
    pairs: dict[str, dict] = {}
    pairs[f"{cn_spot_id}__vs__{cnh_id}"] = analyze_pair_leadlag(db, a_series_id=cn_spot_id, b_series_id=cnh_id, **common_kwargs)
    pairs[f"{cn_spot_id}__vs__{yld_id}"] = analyze_pair_leadlag(db, a_series_id=cn_spot_id, b_series_id=yld_id, **common_kwargs)
    pairs[f"{cnh_id}__vs__{yld_id}"] = analyze_pair_leadlag(db, a_series_id=cnh_id, b_series_id=yld_id, **common_kwargs)
    if cn_fut_id:
        pairs[f"{cn_fut_id}__vs__{cnh_id}"] = analyze_pair_leadlag(db, a_series_id=cn_fut_id, b_series_id=cnh_id, **common_kwargs)
        pairs[f"{cn_fut_id}__vs__{yld_id}"] = analyze_pair_leadlag(db, a_series_id=cn_fut_id, b_series_id=yld_id, **common_kwargs)

    meta = {
        "start": start,
        "end": end,
        "cn_spot_series_id": cn_spot_id,
        "cn_fut_series_id": cn_fut_id,
        "cnh_series_id": cnh_id,
        "yield_series_id": yld_id,
        "index_align": str(payload.index_align or "none"),
    }
    return MacroStep2Response(ok=True, meta=meta, series=series, pairs=pairs)


@router.post("/analysis/macro/step3", response_model=MacroStep3Response)
def analysis_macro_step3(payload: MacroStep3Request, db: Session = Depends(get_session)) -> MacroStep3Response:
    start = str(payload.start)
    end = str(payload.end)
    cn_id = str(payload.cn_gold_series_id).strip()
    glb_id = str(payload.global_gold_series_id).strip()
    fx_id = str(payload.fx_series_id).strip()

    s_cn = load_macro_close_series(db, series_id=cn_id, start=start, end=end)
    s_glb = load_macro_close_series(db, series_id=glb_id, start=start, end=end)
    s_fx = load_macro_close_series(db, series_id=fx_id, start=start, end=end)
    if s_cn.empty:
        return MacroStep3Response(ok=False, error=f"empty_series:{cn_id}")
    if s_glb.empty:
        return MacroStep3Response(ok=False, error=f"empty_series:{glb_id}")
    if s_fx.empty:
        return MacroStep3Response(ok=False, error=f"empty_series:{fx_id}")

    # Derived: global gold converted by FX (best-effort). Use overlap dates only.
    df = pd.DataFrame({"glb": s_glb, "fx": s_fx}).dropna()
    s_glb_fx = (df["glb"] * df["fx"]).dropna()

    series: dict[str, dict] = {
        cn_id: {"dates": [d.isoformat() for d in s_cn.index], "close": s_cn.astype(float).tolist()},
        glb_id: {"dates": [d.isoformat() for d in s_glb.index], "close": s_glb.astype(float).tolist()},
        fx_id: {"dates": [d.isoformat() for d in s_fx.index], "close": s_fx.astype(float).tolist()},
        f"{glb_id}*{fx_id}": {"dates": [d.isoformat() for d in s_glb_fx.index], "close": s_glb_fx.astype(float).tolist()},
    }

    common_kwargs = dict(
        start=start,
        end=end,
        index_align=str(payload.index_align or "none"),
        max_lag=int(payload.max_lag),
        granger_max_lag=int(payload.granger_max_lag),
        alpha=float(payload.alpha),
        rolling_window=int(payload.rolling_window),
        trade_cost_bps=float(payload.trade_cost_bps),
        threshold_quantile=float(payload.threshold_quantile),
        walk_forward=bool(payload.walk_forward),
        train_ratio=float(payload.train_ratio),
        walk_objective=str(payload.walk_objective or "sharpe"),
    )

    pairs: dict[str, dict] = {}
    pairs[f"{cn_id}__vs__{glb_id}"] = analyze_pair_leadlag(db, a_series_id=cn_id, b_series_id=glb_id, **common_kwargs)
    pairs[f"{glb_id}__vs__{fx_id}"] = analyze_pair_leadlag(db, a_series_id=glb_id, b_series_id=fx_id, **common_kwargs)
    # For derived series, compute lead/lag using in-memory series and compute_lead_lag directly.
    res_glb_fx = compute_lead_lag(
        LeadLagInputs(
            etf_close=s_cn,
            idx_close=s_glb_fx,
            max_lag=int(payload.max_lag),
            granger_max_lag=int(payload.granger_max_lag),
            alpha=float(payload.alpha),
            index_align=str(payload.index_align or "none"),
            trade_cost_bps=float(payload.trade_cost_bps),
            rolling_window=int(payload.rolling_window),
            enable_threshold=True,
            threshold_quantile=float(payload.threshold_quantile),
            walk_forward=bool(payload.walk_forward),
            train_ratio=float(payload.train_ratio),
            walk_objective=str(payload.walk_objective or "sharpe"),
        )
    )
    pairs[f"{cn_id}__vs__{glb_id}*{fx_id}"] = res_glb_fx if isinstance(res_glb_fx, dict) else {"ok": False, "reason": "analysis_failed"}

    meta = {
        "start": start,
        "end": end,
        "cn_gold_series_id": cn_id,
        "global_gold_series_id": glb_id,
        "fx_series_id": fx_id,
        "index_align": str(payload.index_align or "none"),
    }
    return MacroStep3Response(ok=True, meta=meta, series=series, pairs=pairs)


@router.post("/analysis/macro/step4", response_model=MacroStep4Response)
def analysis_macro_step4(payload: MacroStep4Request, db: Session = Depends(get_session)) -> MacroStep4Response:
    start_d = _parse_yyyymmdd(str(payload.start))
    end_d = _parse_yyyymmdd(str(payload.end))
    if end_d < start_d:
        return MacroStep4Response(ok=False, error="end_before_start")

    etf_code = str(payload.etf_code).strip()
    adj = str(payload.adjust or "hfq").strip().lower()
    spot_id = str(payload.cn_spot_series_id).strip()

    rows = list_prices(db, code=etf_code, start_date=start_d, end_date=end_d, adjust=adj, limit=1000000)
    if not rows:
        return MacroStep4Response(ok=False, error="empty_etf_series")
    etf = pd.Series(
        data=[float(r.close) if r.close is not None else np.nan for r in rows],
        index=[r.trade_date for r in rows],
        dtype=float,
    ).dropna()
    if etf.empty:
        return MacroStep4Response(ok=False, error="empty_etf_close")

    spot = load_macro_close_series(db, series_id=spot_id, start=str(payload.start), end=str(payload.end))
    if spot.empty:
        return MacroStep4Response(ok=False, error=f"empty_series:{spot_id}")

    res = compute_lead_lag(
        LeadLagInputs(
            etf_close=etf,
            idx_close=spot,
            max_lag=int(payload.max_lag),
            granger_max_lag=int(payload.granger_max_lag),
            alpha=float(payload.alpha),
            index_align=str(payload.index_align or "none"),
            trade_cost_bps=float(payload.trade_cost_bps),
            rolling_window=int(payload.rolling_window),
            enable_threshold=True,
            threshold_quantile=float(payload.threshold_quantile),
            walk_forward=bool(payload.walk_forward),
            train_ratio=float(payload.train_ratio),
            walk_objective=str(payload.walk_objective or "sharpe"),
        )
    )
    if not bool(res.get("ok")):
        return MacroStep4Response(ok=False, error=str(res.get("reason") or "analysis_failed"))

    series = {
        "etf": {"code": etf_code, "adjust": adj, "dates": [d.isoformat() for d in etf.index], "close": etf.astype(float).tolist()},
        "spot": {"series_id": spot_id, "dates": [d.isoformat() for d in spot.index], "close": spot.astype(float).tolist()},
    }
    meta = {
        "start": str(payload.start),
        "end": str(payload.end),
        "etf_code": etf_code,
        "adjust": adj,
        "cn_spot_series_id": spot_id,
        "index_align": str(payload.index_align or "none"),
    }
    return MacroStep4Response(ok=True, meta=meta, series=series, pair=res)


@router.post("/signal/vix-next-action", response_model=VixNextActionResponse)
def signal_vix_next_action(payload: VixNextActionRequest) -> VixNextActionResponse:
    """
    Produce a next-CN-trading-day instruction using VIX/GVZ (Cboe).
    Intended for real trading workflow: BUY / SELL / HOLD.
    """
    idx = str(payload.index or "VIX").strip().upper()
    # Fetch a reasonably long history for threshold estimation (Cboe CSV is small).
    start = "19900101" if idx == "VIX" else "20090101"
    end = _now_shanghai_date().strftime("%Y%m%d")
    df = fetch_cboe_daily_close(CboeFetchRequest(index=idx, start_date=start, end_date=end))
    if df is None or df.empty:
        return VixNextActionResponse(ok=False, error="empty_cboe_series")
    s = pd.Series(df["close"].to_numpy(dtype=float), index=df["date"].to_list(), dtype=float).dropna()
    if s.empty:
        return VixNextActionResponse(ok=False, error="empty_index_close")

    tgt = None
    if payload.target_cn_trade_date:
        try:
            tgt = _parse_yyyymmdd(str(payload.target_cn_trade_date))
        except Exception:  # pylint: disable=broad-exception-caught
            return VixNextActionResponse(ok=False, error="invalid_target_cn_trade_date")
    else:
        mode = str(getattr(payload, "mode", "next_cn_day") or "next_cn_day").strip().lower()
        if mode == "next_cn_day":
            # This matches real trading workflow: you want a decision for the next CN trading session.
            cal = str(payload.calendar or "XSHG")
            today = _now_shanghai_date()
            tgt = shift_to_trading_day(today + dt.timedelta(days=1), shift="next", cal=cal)

    res = generate_next_action(
        VixSignalInputs(
            index_close_us=s,
            index=idx,
            index_align=str(payload.index_align or "cn_next_trading_day"),
            current_position=str(payload.current_position or "unknown").strip().lower(),  # type: ignore[arg-type]
            lookback_window=int(payload.lookback_window),
            threshold_quantile=float(payload.threshold_quantile),
            min_abs_ret=float(payload.min_abs_ret),
            target_cn_trade_date=tgt,
            calendar=str(payload.calendar or "XSHG"),
        )
    )
    if not bool(res.get("ok")):
        return VixNextActionResponse(
            ok=False,
            error=str(res.get("error") or "failed"),
            action_date=(str(res.get("action_date")) if res.get("action_date") else None),
            signal=dict(res.get("signal") or {}) if isinstance(res.get("signal"), dict) else None,
        )
    return VixNextActionResponse(
        ok=True,
        action_date=str(res.get("action_date")),
        action=str(res.get("action")),
        target_position=str(res.get("target_position")),
        current_position=str(res.get("current_position")),
        reason=str(res.get("reason")),
        index=str(res.get("index")),
        index_align=str(res.get("index_align")),
        calendar=str(res.get("calendar")),
        signal=dict(res.get("signal") or {}),
    )


@router.post("/analysis/vix-signal-backtest", response_model=VixSignalBacktestResponse)
def analysis_vix_signal_backtest(payload: VixSignalBacktestRequest, db: Session = Depends(get_session)) -> VixSignalBacktestResponse:
    """
    Backtest the live-tradable VIX-next-day BUY/SELL/HOLD signal on historical data.
    Returns:
    - strategy vs buy&hold NAV curves
    - performance metrics
    - per-day signal/position log (sorted by date desc)
    """
    start_d = _parse_yyyymmdd(payload.start)
    end_d = _parse_yyyymmdd(payload.end)
    if end_d < start_d:
        return VixSignalBacktestResponse(ok=False, error="end_before_start")

    code = str(payload.etf_code).strip()
    adj = str(payload.adjust or "hfq").strip().lower()
    rows = list_prices(db, code=code, start_date=start_d, end_date=end_d, adjust=adj, limit=1000000)
    if not rows:
        return VixSignalBacktestResponse(ok=False, error="empty_etf_series")
    etf_close = pd.Series(
        data=[float(r.close) if r.close is not None else np.nan for r in rows],
        index=[r.trade_date for r in rows],
        dtype=float,
    ).dropna()
    if etf_close.empty:
        return VixSignalBacktestResponse(ok=False, error="empty_etf_close")

    idx = str(payload.index or "VIX").strip().upper()
    # Expand index fetch window to cover threshold lookback (best-effort).
    back_days = int(max(60, int(payload.lookback_window) * 3))
    start_fetch = (start_d - dt.timedelta(days=back_days)).strftime("%Y%m%d")
    end_fetch = end_d.strftime("%Y%m%d")
    dfc = fetch_cboe_daily_close(CboeFetchRequest(index=idx, start_date=start_fetch, end_date=end_fetch))
    if dfc is None or dfc.empty:
        return VixSignalBacktestResponse(ok=False, error="empty_cboe_series")
    idx_close_us = pd.Series(dfc["close"].to_numpy(dtype=float), index=dfc["date"].to_list(), dtype=float).dropna()
    if idx_close_us.empty:
        return VixSignalBacktestResponse(ok=False, error="empty_index_close")

    res = backtest_vix_next_day_signal(
        etf_close_cn=etf_close,
        index_close_us=idx_close_us,
        start=start_d,
        end=end_d,
        index=idx,
        index_align=str(payload.index_align or "cn_next_trading_day"),
        calendar=str(payload.calendar or "XSHG"),
        lookback_window=int(payload.lookback_window),
        threshold_quantile=float(payload.threshold_quantile),
        min_abs_ret=float(payload.min_abs_ret),
        trade_cost_bps=float(payload.trade_cost_bps),
        initial_nav=float(payload.initial_nav),
        initial_position=str(payload.initial_position or "long").strip().lower(),  # type: ignore[arg-type]
    )
    if not bool(res.get("ok")):
        return VixSignalBacktestResponse(ok=False, error=str(res.get("error") or "failed"))

    meta = dict(res.get("meta") or {})
    meta.update({"etf_code": code, "adjust": adj})
    return VixSignalBacktestResponse(
        ok=True,
        meta=meta,
        series=dict(res.get("series") or {}),
        metrics=dict(res.get("metrics") or {}),
        trades=list(res.get("trades") or []),
    )


@router.post("/analysis/index-distribution", response_model=IndexDistributionResponse)
def analysis_index_distribution(payload: IndexDistributionRequest) -> IndexDistributionResponse:
    w = str(payload.window or "all").strip().lower()
    if w not in {"1y", "3y", "5y", "10y", "all"}:
        return IndexDistributionResponse(ok=False, error="invalid_window", meta={"window": w})
    res = compute_cboe_index_distribution(
        IndexDistributionInputs(symbol=str(payload.symbol), window=w, bins=int(payload.bins))
    )
    if not bool(res.get("ok")):
        return IndexDistributionResponse(ok=False, error=str(res.get("error") or "failed"), meta=dict(res.get("meta") or {}))
    return IndexDistributionResponse(
        ok=True,
        meta=dict(res.get("meta") or {}),
        close=dict(res.get("close") or {}),
        ret_log=dict(res.get("ret_log") or {}),
    )


@router.post("/analysis/rotation/calendar-effect")
def rotation_calendar_effect(payload: RotationCalendarEffectRequest, db: Session = Depends(get_session)) -> dict:
    # Reuse all rotation params as the "base" strategy config for the grid; then vary weekday + exec_price.
    # pylint: disable=unexpected-keyword-arg
    asset_vol_rules = [r.model_dump() for r in payload.asset_vol_index_rules] if payload.asset_vol_index_rules else None
    vol_index_close = _load_vol_index_close_for_rotation_rules(asset_vol_rules, end_yyyymmdd=payload.end)
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
        asset_rc_rules=[r.model_dump() for r in payload.asset_rc_rules] if payload.asset_rc_rules else None,
        asset_vol_index_rules=asset_vol_rules,
        vol_index_close=vol_index_close,
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
    asset_vol_rules = [r.model_dump() for r in payload.asset_vol_index_rules] if payload.asset_vol_index_rules else None
    vol_index_close = _load_vol_index_close_for_rotation_rules(asset_vol_rules, end_yyyymmdd=payload.end)
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
        asset_rc_rules=[r.model_dump() for r in payload.asset_rc_rules] if payload.asset_rc_rules else None,
        asset_vol_index_rules=asset_vol_rules,
        vol_index_close=vol_index_close,
    )
    # Mini-program semantics: `anchor_weekday` refers to the *execution day* weekday (Mon..Fri),
    # while the strategy engine's weekly rebalance anchor is the *decision day* weekday.
    # With "open execution", decision is made on previous trading day close, executed on next trading day open.
    # Therefore: decision_weekday = (exec_weekday - 1) mod 5 (Mon exec -> Fri decision).
    one_exec = payload.anchor_weekday
    anchors = [int(one_exec)] if one_exec is not None else [0, 1, 2, 3, 4]
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
    for exec_wd in anchors:
        decision_wd = (int(exec_wd) - 1) % 5
        # pylint: disable=unexpected-keyword-arg
        inp = RotationAnalysisInputs(**{**base.__dict__, "rebalance_anchor": int(decision_wd)})
        out = _slim_for_miniprogram(compute_rotation_backtest(db, inp))
        # Filter period_details to show only rows whose *execution date* (start_date) matches the tab weekday.
        # This prevents cross-tab leakage (e.g. Fri rows showing in Mon tab) and aligns UI date semantics.
        try:
            pds = out.get("period_details") if isinstance(out, dict) else None
            if isinstance(pds, list):
                def _wd_of(s: str | None) -> int | None:
                    if not s:
                        return None
                    return int(dt.date.fromisoformat(str(s)).weekday())
                out["period_details"] = [r for r in pds if _wd_of(r.get("start_date")) == int(exec_wd)]
        except Exception:  # pragma: no cover
            pass
        by_anchor[str(exec_wd)] = out
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
    asset_vol_rules = [r.model_dump() for r in payload.asset_vol_index_rules] if payload.asset_vol_index_rules else None
    vol_index_close = _load_vol_index_close_for_rotation_rules(asset_vol_rules, end_yyyymmdd=payload.end)
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
        asset_rc_rules=[r.model_dump() for r in payload.asset_rc_rules] if payload.asset_rc_rules else None,
        asset_vol_index_rules=asset_vol_rules,
        vol_index_close=vol_index_close,
    )

    one_exec = payload.anchor_weekday
    anchors = [int(one_exec)] if one_exec is not None else [0, 1, 2, 3, 4]

    def _lite(x: dict) -> dict:
        nav = x.get("nav") if isinstance(x, dict) else None
        return {
            "date_range": (x.get("date_range") if isinstance(x, dict) else None),
            "nav": nav,
        }

    by_anchor: dict[str, dict] = {}
    for exec_wd in anchors:
        decision_wd = (int(exec_wd) - 1) % 5
        # pylint: disable=unexpected-keyword-arg
        inp = RotationAnalysisInputs(**{**base.__dict__, "rebalance_anchor": int(decision_wd)})
        by_anchor[str(exec_wd)] = _lite(compute_rotation_backtest(db, inp))

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


@router.post("/analysis/rotation/weekly5-open-combo-lite")
def rotation_weekly5_open_combo_lite(payload: RotationWeekly5OpenSimRequest, db: Session = Depends(get_session)) -> dict:
    """
    Composite (MON~FRI equally weighted) rotation weekly5-open NAV only.
    Intended for the mini-program "mix" page first paint.
    """
    codes = ["159915", "511010", "513100", "518880"]
    start = _parse_yyyymmdd(payload.start)
    end = _parse_yyyymmdd(payload.end)
    asset_vol_rules = [r.model_dump() for r in payload.asset_vol_index_rules] if payload.asset_vol_index_rules else None
    vol_index_close = _load_vol_index_close_for_rotation_rules(asset_vol_rules, end_yyyymmdd=payload.end)
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
        exec_price="open",
        asset_rc_rules=[r.model_dump() for r in payload.asset_rc_rules] if payload.asset_rc_rules else None,
        asset_vol_index_rules=asset_vol_rules,
        vol_index_close=vol_index_close,
    )

    def _nav_arr(x: dict, key: str) -> np.ndarray:
        s = ((x or {}).get("nav") or {}).get("series") or {}
        arr = s.get(key) or []
        return np.asarray([float(v) for v in arr], dtype=float)

    outs: list[dict] = []
    for a in [0, 1, 2, 3, 4]:
        inp = RotationAnalysisInputs(**{**base.__dict__, "rebalance_anchor": int(a)})
        outs.append(compute_rotation_backtest(db, inp))
    if not outs:
        raise HTTPException(status_code=400, detail="no backtest data")  # pragma: no cover

    nav0 = outs[0].get("nav") or {}
    dates = nav0.get("dates") or []
    rot = np.vstack([_nav_arr(o, "ROTATION") for o in outs]).mean(axis=0)
    ew = np.vstack([_nav_arr(o, "EW_REBAL") for o in outs]).mean(axis=0)
    ex = np.where(ew != 0, rot / ew, np.nan)

    by_anchor = {
        "mix": {
            "date_range": outs[0].get("date_range"),
            "nav": {"dates": dates, "series": {"ROTATION": rot.tolist(), "EW_REBAL": ew.tolist(), "EXCESS": ex.tolist()}},
        }
    }
    return {
        "meta": {
            "type": "rotation_weekly5_open_combo_lite",
            "codes": codes,
            "start": payload.start,
            "end": payload.end,
            "rebalance": "weekly",
            "rebalance_shift": "prev",
            "exec_price": "open",
            "anchors": ["mix"],
        },
        "by_anchor": by_anchor,
        "weekday_map": {"mix": "MIX"},
    }


@router.post("/analysis/rotation/weekly5-open-combo")
def rotation_weekly5_open_combo(payload: RotationWeekly5OpenSimRequest, db: Session = Depends(get_session)) -> dict:
    """
    Composite (MON~FRI equally weighted) rotation weekly5-open full payload (slimmed but UI-complete).
    """
    codes = ["159915", "511010", "513100", "518880"]
    start = _parse_yyyymmdd(payload.start)
    end = _parse_yyyymmdd(payload.end)
    asset_vol_rules = [r.model_dump() for r in payload.asset_vol_index_rules] if payload.asset_vol_index_rules else None
    vol_index_close = _load_vol_index_close_for_rotation_rules(asset_vol_rules, end_yyyymmdd=payload.end)
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
        exec_price="open",
        asset_rc_rules=[r.model_dump() for r in payload.asset_rc_rules] if payload.asset_rc_rules else None,
        asset_vol_index_rules=asset_vol_rules,
        vol_index_close=vol_index_close,
    )

    def _nav_arr(x: dict, key: str) -> np.ndarray:
        s = ((x or {}).get("nav") or {}).get("series") or {}
        arr = s.get(key) or []
        return np.asarray([float(v) for v in arr], dtype=float)

    def _avg_share(items: list[dict], field: str) -> float | None:
        xs = []
        for it in items:
            v = it.get(field)
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:  # pragma: no cover
                continue
            if np.isfinite(fv):
                xs.append(fv)
        return float(np.mean(xs)) if xs else None

    def _avg_metric(path: list[str]) -> float | None:
        """
        Average a numeric metric across 5 variants: out["metrics"][path[0]]...[path[n-1]].
        Returns None if no finite values.
        """
        xs: list[float] = []
        for o in outs:
            cur: object = (o or {}).get("metrics") or {}
            ok = True
            for k in path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False
                    break
            if not ok:
                continue
            try:
                fv = float(cur)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                xs.append(fv)
        return float(np.mean(xs)) if xs else None

    outs: list[dict] = []
    for a in [0, 1, 2, 3, 4]:
        inp = RotationAnalysisInputs(**{**base.__dict__, "rebalance_anchor": int(a)})
        outs.append(compute_rotation_backtest(db, inp))
    if not outs:
        raise HTTPException(status_code=400, detail="no backtest data")  # pragma: no cover

    nav0 = outs[0].get("nav") or {}
    dates = nav0.get("dates") or []
    if not dates:
        raise HTTPException(status_code=400, detail="no nav dates")  # pragma: no cover

    rot = np.vstack([_nav_arr(o, "ROTATION") for o in outs]).mean(axis=0)
    ew = np.vstack([_nav_arr(o, "EW_REBAL") for o in outs]).mean(axis=0)
    ex = np.where(ew != 0, rot / ew, np.nan)

    # metrics from composite nav
    dt_idx = pd.to_datetime(dates)
    s_rot = pd.Series(rot, index=dt_idx).astype(float)
    s_ew = pd.Series(ew, index=dt_idx).astype(float)
    r_rot = s_rot.pct_change().fillna(0.0).astype(float)
    r_ew = s_ew.pct_change().fillna(0.0).astype(float)
    excess_ret = (r_rot - r_ew).astype(float)
    ann_ret = _annualized_return(s_rot, ann_factor=TRADING_DAYS_PER_YEAR)
    ann_vol = _annualized_vol(r_rot, ann_factor=TRADING_DAYS_PER_YEAR)
    mdd = _max_drawdown(s_rot)
    mdd_dur = _max_drawdown_duration_days(s_rot)
    sharpe = _sharpe(r_rot, rf=0.0, ann_factor=TRADING_DAYS_PER_YEAR)
    sortino = _sortino(r_rot, rf=0.0, ann_factor=TRADING_DAYS_PER_YEAR)
    calmar = float(ann_ret / abs(mdd)) if mdd < 0 else float("nan")
    ui = _ulcer_index(s_rot, in_percent=True)
    ui_den = ui / 100.0
    upi = float(ann_ret / ui_den) if ui_den > 0 else float("nan")
    ann_excess_arith = float(excess_ret.mean() * TRADING_DAYS_PER_YEAR) if len(excess_ret) else float("nan")
    ann_excess_vol = _annualized_vol(excess_ret, ann_factor=TRADING_DAYS_PER_YEAR)
    ir = float(excess_ret.mean() / excess_ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if float(excess_ret.std(ddof=1) or 0) > 0 else float("nan")
    ex_nav = (1.0 + excess_ret.fillna(0.0)).cumprod()
    if len(ex_nav):
        ex_nav.iloc[0] = 1.0
    ann_excess_geo = _annualized_return(ex_nav, ann_factor=TRADING_DAYS_PER_YEAR) if len(ex_nav) else float("nan")
    ex_mdd = _max_drawdown(ex_nav) if len(ex_nav) else float("nan")
    ex_mdd_dur = _max_drawdown_duration_days(ex_nav) if len(ex_nav) else float("nan")
    metrics = {
        "strategy": {
            "cumulative_return": float(s_rot.iloc[-1] / s_rot.iloc[0] - 1.0) if len(s_rot) else float("nan"),
            "annualized_return": float(ann_ret),
            "annualized_volatility": float(ann_vol),
            "max_drawdown": float(mdd),
            "max_drawdown_recovery_days": int(mdd_dur) if np.isfinite(float(mdd_dur)) else None,
            "sharpe_ratio": float(sharpe),
            "calmar_ratio": float(calmar),
            "sortino_ratio": float(sortino),
            "ulcer_index": float(ui),
            "ulcer_performance_index": float(upi),
            # composite turnover is not directly derivable from averaged NAV; approximate by averaging 5 variants
            "avg_daily_turnover": _avg_metric(["strategy", "avg_daily_turnover"]),
        },
        "equal_weight": {"cumulative_return": float(s_ew.iloc[-1] / s_ew.iloc[0] - 1.0) if len(s_ew) else float("nan")},
        "excess_vs_equal_weight": {
            "cumulative_return": float((s_rot.iloc[-1] / s_rot.iloc[0]) / (s_ew.iloc[-1] / s_ew.iloc[0]) - 1.0) if len(s_rot) and len(s_ew) else float("nan"),
            # annualized excess return (two complementary definitions)
            # - geo: CAGR on EXCESS nav (compound-consistent, recommended)
            # - arith: mean(active_ret)*252 (expected active return per year, not compound)
            "annualized_return": float(ann_excess_geo),  # backward compatible meaning: geo
            "annualized_return_geo": float(ann_excess_geo),
            "annualized_return_arith": float(ann_excess_arith),
            "annualized_volatility": float(ann_excess_vol),
            "information_ratio": float(ir),
            "max_drawdown": float(ex_mdd),
            "max_drawdown_recovery_days": int(ex_mdd_dur) if np.isfinite(float(ex_mdd_dur)) else None,
        },
    }

    # win/payoff and period_details on weekly periods (W-FRI) for display
    rr = s_rot.resample("W-FRI").last().pct_change().dropna()
    bb = s_ew.resample("W-FRI").last().pct_change().dropna()
    idx_w = rr.index.intersection(bb.index)
    pos = []
    neg = []
    period_details: list[dict] = []
    prev_end = None
    for t in idx_w:
        rs = float(rr.loc[t])
        rb = float(bb.loc[t])
        exr = float(rs - rb)
        if exr > 0:
            pos.append(exr)
        elif exr < 0:
            neg.append(exr)
        end_d = t.date().isoformat()
        start_d = (prev_end or end_d)
        period_details.append(
            {
                "start_date": start_d,
                "end_date": end_d,
                "strategy_return": rs,
                "equal_weight_return": rb,
                "excess_return": exr,
                "win": exr > 0,
                "timing_sleep": False,
                "timing_active_ratio": None,
                "buys": [],
                "sells": [],
                "turnover": None,
            }
        )
        prev_end = end_d
    win_rate = float((np.sum([1 for x in pos if x > 0]) / len(idx_w))) if len(idx_w) else float("nan")
    avg_win = float(np.mean(pos)) if pos else float("nan")
    avg_loss = float(np.mean(neg)) if neg else float("nan")
    payoff = float(avg_win / abs(avg_loss)) if (pos and neg and avg_loss != 0) else float("nan")
    def _geo_mean_return(rs: list[float]) -> float:
        if not rs:
            return float("nan")
        a = np.asarray(rs, dtype=float)
        m = np.isfinite(a) & (a > -1.0 + 1e-12)
        if not np.any(m):
            return float("nan")
        return float(np.exp(np.mean(np.log1p(a[m]))) - 1.0)

    avg_win_geo = _geo_mean_return(pos)
    avg_loss_geo = _geo_mean_return(neg)
    payoff_geo = float(avg_win_geo / abs(avg_loss_geo)) if (np.isfinite(avg_win_geo) and np.isfinite(avg_loss_geo) and avg_loss_geo != 0) else float("nan")
    win_payoff = {
        "rebalance": "weekly",
        "periods": int(len(idx_w)),
        "win_rate": float(win_rate),
        # arithmetic means (backward compatible)
        "avg_win_excess": float(avg_win),
        "avg_loss_excess": float(avg_loss),
        "payoff_ratio": float(payoff),
        # geometric means (new)
        "avg_win_excess_geo": float(avg_win_geo),
        "avg_loss_excess_geo": float(avg_loss_geo),
        "payoff_ratio_geo": float(payoff_geo),
        "kelly_fraction": float("nan"),
    }

    # attribution: average variant shares
    ret_by_code: dict[str, list[float]] = {c: [] for c in codes}
    risk_by_code: dict[str, list[float]] = {c: [] for c in codes}
    for o in outs:
        attr = (o or {}).get("attribution") or {}
        for it in ((attr.get("return") or {}).get("by_code") or []):
            c = str(it.get("code") or "")
            if c in ret_by_code and it.get("return_share") is not None:
                ret_by_code[c].append(float(it.get("return_share")))
        for it in ((attr.get("risk") or {}).get("by_code") or []):
            c = str(it.get("code") or "")
            if c in risk_by_code and it.get("risk_share") is not None:
                risk_by_code[c].append(float(it.get("risk_share")))
    attribution = {
        "return": {"by_code": [{"code": c, "return_share": _avg_share([{"return_share": v} for v in ret_by_code[c]], "return_share")} for c in codes]},
        "risk": {"by_code": [{"code": c, "risk_share": _avg_share([{"risk_share": v} for v in risk_by_code[c]], "risk_share")} for c in codes]},
    }

    out = {
        "date_range": outs[0].get("date_range"),
        "codes": codes,
        "nav": {"dates": dates, "series": {"ROTATION": rot.tolist(), "EW_REBAL": ew.tolist(), "EXCESS": ex.tolist()}},
        "metrics": metrics,
        "win_payoff": win_payoff,
        "period_details": period_details,
        "attribution": attribution,
    }
    return {
        "meta": {
            "type": "rotation_weekly5_open_combo",
            "codes": codes,
            "start": payload.start,
            "end": payload.end,
            "rebalance": "weekly",
            "rebalance_shift": "prev",
            "exec_price": "open",
            "anchors": ["mix"],
        },
        "by_anchor": {"mix": out},
        "weekday_map": {"mix": "MIX"},
    }


@router.post("/analysis/rotation/next-plan")
def rotation_next_plan(payload: RotationNextPlanRequest, db: Session = Depends(get_session)) -> dict:
    """
    "Tomorrow plan" for the fixed mini-program rotation strategy.
    If the next trading day is a rebalance effective day (open execution), return the top pick based on asof close.
    """
    requested_asof = _parse_yyyymmdd(payload.asof)
    anchor = int(payload.anchor_weekday)
    if anchor not in {0, 1, 2, 3, 4}:
        raise HTTPException(status_code=400, detail="anchor_weekday must be 0..4")

    codes = _FIXED_CODES[:]
    # Use "last available close <= requested_asof" as the effective decision date.
    # This makes the endpoint naturally do the right thing intraday (today close not ingested yet).
    start = requested_asof - dt.timedelta(days=90)
    px = load_close_prices(db, codes=codes, start=start, end=requested_asof, adjust="hfq")
    if px.empty:
        raise HTTPException(status_code=400, detail="no price data")
    px = px.sort_index().ffill()
    asof = px.index[-1].date()

    # next trading day (XSHG) after effective asof
    try:
        tds = trading_days(asof, asof + dt.timedelta(days=20), cal="XSHG")
        next_td = next((d for d in tds if d > asof), asof)
    except Exception:  # pragma: no cover
        next_td = asof

    # Mini-program semantics: each tab represents the *execution day* weekday (open execution).
    # We only show a plan on the tab whose weekday matches the next trading day.
    # (i.e., decision is made on asof close, executed on next trading day open).
    rebalance_effective_next_day = bool(next_td > asof and int(next_td.weekday()) == int(anchor))

    # If next trading day is not this tab's execution day, skip computing the pick to avoid
    # misleading UI + unnecessary heavy computations.
    if not rebalance_effective_next_day:
        return {
            "asof": asof.strftime("%Y%m%d"),
            "asof_requested": requested_asof.strftime("%Y%m%d"),
            "next_trading_day": next_td.isoformat(),
            "rebalance_effective_next_day": False,
            "pick_code": None,
            "pick_name": None,
            "scores": {},
            "meta": {"anchor_weekday": anchor, "rebalance_shift": "prev", "lookback_days": 20, "top_k": 1, "exec_price": "open"},
        }
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
        "asof_requested": requested_asof.strftime("%Y%m%d"),
        "next_trading_day": next_td.isoformat(),
        "rebalance_effective_next_day": True,
        "pick_code": pick_code,
        "pick_name": pick_name,
        "scores": {c: float(scores[c]) for c in codes},
        "meta": {"anchor_weekday": anchor, "rebalance_shift": "prev", "lookback_days": 20, "top_k": 1, "exec_price": "open"},
    }


@router.post("/analysis/rotation/next-plan-auto")
def rotation_next_plan_auto(payload: dict, db: Session = Depends(get_session)) -> dict:
    """
    Convenience endpoint for the mini-program "mix" page:
    return the plan for the weekday of the next trading day.
    """
    asof = _parse_yyyymmdd(str((payload or {}).get("asof")))
    # Same as rotation_next_plan: use last available close <= asof.
    codes = _FIXED_CODES[:]
    start = asof - dt.timedelta(days=90)
    px = load_close_prices(db, codes=codes, start=start, end=asof, adjust="hfq")
    if not px.empty:
        px = px.sort_index().ffill()
        asof = px.index[-1].date()

    try:
        tds = trading_days(asof, asof + dt.timedelta(days=20), cal="XSHG")
        next_td = next((d for d in tds if d > asof), asof)
    except Exception:  # pragma: no cover
        next_td = asof
    wd = int(next_td.weekday())
    if wd not in {0, 1, 2, 3, 4}:
        return {
            "asof": asof.strftime("%Y%m%d"),
            "next_trading_day": next_td.isoformat(),
            "rebalance_effective_next_day": False,
            "pick_code": None,
            "pick_name": None,
            "scores": {},
            "meta": {"anchor_weekday": wd, "rebalance_shift": "prev", "lookback_days": 20, "top_k": 1, "exec_price": "open"},
        }
    return rotation_next_plan(RotationNextPlanRequest(anchor_weekday=wd, asof=asof.strftime("%Y%m%d")), db=db)


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
        rdd3y = _rolling_drawdown(ew_nav, win_3y).astype(float)

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
            "roll3y_dd": _tolist(rdd3y),
            # backward-compat (deprecated)
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
        rdd3y = _rolling_drawdown(ew_nav, win_3y).astype(float)

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
            "roll3y_dd": _tolist(rdd3y),
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


@router.post("/analysis/baseline/weekly5-ew-dashboard-combo-lite")
def baseline_weekly5_ew_dashboard_combo_lite(payload: BaselineWeekly5EWDashboardRequest, db: Session = Depends(get_session)) -> dict:
    """
    Composite (MON~FRI equally weighted) EW dashboard lite:
    return only series needed for charts (1)~(5) for the mini-program "mix" page.
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

    navs = []
    for a in [0, 1, 2, 3, 4]:
        decision_dates = _cal_decision_dates_for_rebalance(idx, rebalance="weekly", anchor=int(a), shift=shift)
        ew_nav, _ew_w = _cal_ew_nav_and_weights_by_decision_dates(daily_ret[codes], decision_dates=decision_dates)
        navs.append(ew_nav.astype(float))
    nav_df = pd.concat(navs, axis=1)
    nav_mix = nav_df.mean(axis=1).astype(float)

    ema252 = _ema(nav_mix, 252)
    sd252 = _rolling_std(nav_mix, 252)
    bb_u = ema252 + 2.0 * sd252
    bb_l = ema252 - 2.0 * sd252
    dd = _drawdown(nav_mix)
    rsi24 = _rsi_wilder(nav_mix, window=24)
    win_3y = 3 * TRADING_DAYS_PER_YEAR
    rr3y = (nav_mix / nav_mix.shift(win_3y) - 1.0).astype(float)
    rdd3y = _rolling_drawdown(nav_mix, win_3y).astype(float)

    by_anchor = {
        "mix": {
            "meta": {"anchor_weekday": None, "label": "MIX", "rebalance_shift": shift, "price": "hfq_close"},
            "dates": idx.date.astype(str).tolist(),
            "nav": _tolist(nav_mix),
            "ema252": _tolist(ema252),
            "bb_upper": _tolist(bb_u),
            "bb_lower": _tolist(bb_l),
            "drawdown": _tolist(dd),
            "rsi24": _tolist(rsi24),
            "roll3y_return": _tolist(rr3y),
            "roll3y_dd": _tolist(rdd3y),
            "roll3y_mdd": _tolist(rdd3y),
        }
    }
    return {
        "meta": {
            "type": "baseline_weekly5_ew_dashboard_combo_lite",
            "codes": codes,
            "start": payload.start,
            "end": payload.end,
            "common_start": common_start.date().strftime("%Y%m%d"),
            "rebalance": "weekly",
            "rebalance_shift": shift,
            "price": "hfq_close",
            "anchors": ["mix"],
        },
        "by_anchor": by_anchor,
        "weekday_map": {"mix": "MIX"},
    }


@router.post("/analysis/baseline/weekly5-ew-dashboard-combo")
def baseline_weekly5_ew_dashboard_combo(payload: BaselineWeekly5EWDashboardRequest, db: Session = Depends(get_session)) -> dict:
    """
    Composite (MON~FRI equally weighted) EW dashboard full:
    include metrics/attribution/correlation/calendar for the mini-program "mix" page.
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

    navs = []
    ws = []
    for a in [0, 1, 2, 3, 4]:
        decision_dates = _cal_decision_dates_for_rebalance(idx, rebalance="weekly", anchor=int(a), shift=shift)
        ew_nav, ew_w = _cal_ew_nav_and_weights_by_decision_dates(daily_ret[codes], decision_dates=decision_dates)
        navs.append(ew_nav.astype(float))
        ws.append(ew_w[codes].astype(float))
    nav_mix = pd.concat(navs, axis=1).mean(axis=1).astype(float)
    w_mix = sum(ws) / float(len(ws)) if ws else pd.DataFrame(index=idx, columns=codes).fillna(0.0)

    ew_ret = nav_mix.pct_change().fillna(0.0).astype(float)
    ema252 = _ema(nav_mix, 252)
    sd252 = _rolling_std(nav_mix, 252)
    bb_u = ema252 + 2.0 * sd252
    bb_l = ema252 - 2.0 * sd252
    dd = _drawdown(nav_mix)
    rsi24 = _rsi_wilder(nav_mix, window=24)
    win_3y = 3 * TRADING_DAYS_PER_YEAR
    rr3y = (nav_mix / nav_mix.shift(win_3y) - 1.0).astype(float)
    rdd3y = _rolling_drawdown(nav_mix, win_3y).astype(float)

    cum_ret = float(nav_mix.iloc[-1] / nav_mix.iloc[0] - 1.0) if len(nav_mix) else float("nan")
    ann_ret = _annualized_return(nav_mix, ann_factor=TRADING_DAYS_PER_YEAR)
    ann_vol = _annualized_vol(ew_ret, ann_factor=TRADING_DAYS_PER_YEAR)
    mdd = _max_drawdown(nav_mix)
    mdd_dur = _max_drawdown_duration_days(nav_mix)
    sharpe = _sharpe(ew_ret, rf=rf, ann_factor=TRADING_DAYS_PER_YEAR)
    calmar = float(ann_ret / abs(mdd)) if np.isfinite(mdd) and mdd < 0 else float("nan")
    sortino = _sortino(ew_ret, rf=rf, ann_factor=TRADING_DAYS_PER_YEAR)
    ui = _ulcer_index(nav_mix, in_percent=True)
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

    corr = daily_ret[codes].corr(method="pearson")
    corr_out = {
        "method": "pearson",
        "n_obs": int(len(daily_ret)),
        "codes": codes,
        "matrix": corr.to_numpy(dtype=float).tolist(),
    }
    attribution = _compute_return_risk_contributions(asset_ret=daily_ret[codes], weights=w_mix[codes], total_return=float(cum_ret))

    daily = ew_ret.copy()
    monthly = nav_mix.resample("ME").last().pct_change().dropna()
    yearly = nav_mix.resample("YE").last().pct_change().dropna()
    cal = {
        "daily": {"dates": daily.index.date.astype(str).tolist(), "values": daily.astype(float).tolist()},
        "monthly": {"dates": monthly.index.date.astype(str).tolist(), "values": monthly.astype(float).tolist()},
        "yearly": {"dates": yearly.index.date.astype(str).tolist(), "values": yearly.astype(float).tolist()},
    }

    by_anchor = {
        "mix": {
            "meta": {"anchor_weekday": None, "label": "MIX", "rebalance_shift": shift, "price": "hfq_close"},
            "dates": idx.date.astype(str).tolist(),
            "nav": _tolist(nav_mix),
            "ema252": _tolist(ema252),
            "bb_upper": _tolist(bb_u),
            "bb_lower": _tolist(bb_l),
            "drawdown": _tolist(dd),
            "rsi24": _tolist(rsi24),
            "roll3y_return": _tolist(rr3y),
            "roll3y_dd": _tolist(rdd3y),
            "roll3y_mdd": _tolist(rdd3y),
            "metrics": metrics,
            "attribution": attribution,
            "correlation": corr_out,
            "calendar": cal,
        }
    }
    return {
        "meta": {
            "type": "baseline_weekly5_ew_dashboard_combo",
            "codes": codes,
            "start": payload.start,
            "end": payload.end,
            "common_start": common_start.date().strftime("%Y%m%d"),
            "rebalance": "weekly",
            "rebalance_shift": shift,
            "price": "hfq_close",
            "anchors": ["mix"],
        },
        "by_anchor": by_anchor,
        "weekday_map": {"mix": "MIX"},
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
    # For "period return" distribution, align with the same rebalance frequency selection (best-effort).
    reb = (payload.rebalance or "weekly").strip().lower()
    period_freq = {"weekly": "W-FRI", "monthly": "ME", "quarterly": "QE", "yearly": "YE", "daily": "B"}.get(reb, "W-FRI")
    try:
        mc = bootstrap_metrics_from_daily_returns(daily_ret, rf=float(payload.risk_free_rate), cfg=cfg, period_freq=period_freq)
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
    reb = (payload.rebalance or "weekly").strip().lower()
    period_freq = {"weekly": "W-FRI", "monthly": "ME", "quarterly": "QE", "yearly": "YE", "daily": "B"}.get(reb, "W-FRI")
    try:
        mc_strategy = bootstrap_metrics_from_daily_returns(daily_ret, rf=float(payload.risk_free_rate), cfg=cfg, period_freq=period_freq)
        mc_excess = bootstrap_metrics_from_daily_returns(daily_excess, rf=0.0, cfg=cfg, period_freq=period_freq)
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
        # For research UI: show observed (non-simulated) holding-period length distributions
        "observed_holding_len": {
            "nav_dates": (rot.get("nav") or {}).get("dates") or [],
            # Prefer continuous holding streaks (merge unchanged holdings across rebalance periods).
            "holding_streaks": rot.get("holding_streaks") or [],
            # Backward-compat: decision-period holdings (bounded by rebalance schedule).
            "holdings": rot.get("holdings") or [],
        },
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

