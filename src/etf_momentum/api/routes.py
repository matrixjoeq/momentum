from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import datetime as dt

from .deps import get_akshare, get_session
from .schemas import (
    BaselineAnalysisRequest,
    BaselineMonteCarloRequest,
    RotationMonteCarloRequest,
    RotationBacktestRequest,
    EtfPoolOut,
    EtfPoolUpsert,
    FetchResult,
    FetchSelectedRequest,
    IngestionBatchOut,
    PriceOut,
    ValidationPolicyOut,
)
from ..analysis.baseline import BaselineInputs, compute_baseline
from ..analysis.montecarlo import MonteCarloConfig, bootstrap_metrics_from_daily_returns
from ..analysis.rotation import RotationAnalysisInputs, compute_rotation_backtest
from ..data.ingestion import ingest_one_etf
from ..data.rollback import logical_rollback_batch, rollback_batch_with_fallback
from ..db.repo import (
    delete_etf_pool,
    delete_prices,
    get_price_date_range,
    get_ingestion_batch,
    get_validation_policy_by_id,
    get_validation_policy_by_name,
    list_ingestion_batches,
    list_etf_pool,
    list_validation_policies,
    list_prices,
    mark_fetch_status,
    update_ingestion_batch,
    upsert_etf_pool,
)
from ..settings import get_settings
from ..validation.policy_infer import infer_policy_name

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
    )
    try:
        return compute_baseline(db, inp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/analysis/rotation")
def rotation_backtest(payload: RotationBacktestRequest, db: Session = Depends(get_session)) -> dict:
    inp = RotationAnalysisInputs(
        codes=payload.codes,
        start=_parse_yyyymmdd(payload.start),
        end=_parse_yyyymmdd(payload.end),
        rebalance=payload.rebalance,
        top_k=payload.top_k,
        lookback_days=payload.lookback_days,
        skip_days=payload.skip_days,
        risk_off=payload.risk_off,
        defensive_code=payload.defensive_code,
        momentum_floor=payload.momentum_floor,
        risk_free_rate=payload.risk_free_rate,
        cost_bps=payload.cost_bps,
    )
    try:
        return compute_rotation_backtest(db, inp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


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
def delete_etf(code: str, db: Session = Depends(get_session)) -> dict[str, bool]:
    ok = delete_etf_pool(db, code)
    if not ok:
        raise HTTPException(status_code=404, detail="ETF not found")
    return {"deleted": True}


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

