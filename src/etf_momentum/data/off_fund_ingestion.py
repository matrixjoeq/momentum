from __future__ import annotations

# pylint: disable=broad-exception-caught

import datetime as dt
import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from ..db.off_fund_repo import (
    OffFundEventRow,
    OffFundNavRow,
    get_off_fund_pool_by_code,
    mark_off_fund_fetch_status,
    replace_off_fund_events,
    update_off_fund_pool_data_range,
    upsert_off_fund_navs,
)
from ..settings import get_settings


@dataclass(frozen=True)
class IngestOffFundResult:
    code: str
    upserted: int
    status: str
    message: str | None = None


@dataclass(frozen=True)
class FundEvent:
    effective_date: dt.date
    event_type: str  # dividend|split
    event_key: str
    cash_dividend: float | None = None
    split_ratio: float | None = None
    raw_payload: str | None = None


def _parse_yyyymmdd(x: str) -> dt.date:
    return dt.datetime.strptime(x, "%Y%m%d").date()


def _coerce_date(x: Any) -> dt.date | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    if s in {"", "--", "nan", "None", "null"}:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    return v if pd.notna(v) else None


def _pick_col(columns: list[str], keywords: tuple[str, ...]) -> str | None:
    for c in columns:
        cc = str(c)
        if any(k in cc for k in keywords):
            return c
    return None


def _parse_series_df(
    df: pd.DataFrame, *, value_keywords: tuple[str, ...]
) -> dict[dt.date, float]:
    if df is None or df.empty:
        return {}
    cols = [str(c) for c in list(df.columns)]
    date_col = _pick_col(cols, ("净值日期", "日期", "x"))
    val_col = _pick_col(cols, value_keywords + ("y",))
    if date_col is None:
        date_col = cols[0] if cols else None
    if val_col is None:
        # choose first numeric-like column different from date
        for c in cols:
            if c == date_col:
                continue
            if pd.to_numeric(df[c], errors="coerce").notna().any():
                val_col = c
                break
    if date_col is None or val_col is None:
        return {}
    out: dict[dt.date, float] = {}
    for _, r in df.iterrows():
        d = _coerce_date(r.get(date_col))
        v = _to_float(r.get(val_col))
        if d is None or v is None:
            continue
        out[d] = float(v)
    return out


def _parse_split_ratio(x: Any) -> float | None:
    s = str(x or "").strip()
    if not s:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*[:：/]\s*(\d+(?:\.\d+)?)", s)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        if a > 0:
            return b / a
    m2 = re.search(r"(\d+(?:\.\d+)?)\s*(?:份|倍|x|X)", s)
    if m2:
        v = float(m2.group(1))
        if v > 0:
            return v
    f = _to_float(s)
    if f is not None and f > 0:
        return f
    return None


def _call_fund_open_info(ak: Any, *, code: str, indicator: str) -> pd.DataFrame | None:
    """
    AkShare compatibility wrapper:
    - newer versions: fund_open_fund_info_em(symbol=..., indicator=..., period=...)
    - some variants:  fund_open_fund_info_em(fund=..., indicator=...)
    """
    fn = getattr(ak, "fund_open_fund_info_em", None)
    if fn is None:
        return None
    # Preferred: documented signature in current akshare
    try:
        return fn(symbol=code, indicator=indicator, period="成立来")
    except TypeError:
        pass
    # Backward/variant compatibility
    try:
        return fn(fund=code, indicator=indicator)
    except TypeError:
        pass
    # Last fallback: positional
    try:
        return fn(code, indicator)
    except Exception:
        return None


def _parse_fund_events(ak: Any, *, code: str) -> list[FundEvent]:
    events: list[FundEvent] = []

    # Dividend details
    df_div = _call_fund_open_info(ak, code=code, indicator="分红送配详情")
    if isinstance(df_div, pd.DataFrame) and not df_div.empty:
        cols = [str(c) for c in df_div.columns]
        date_col = _pick_col(cols, ("除息", "权益登记", "日期"))
        cash_col = _pick_col(cols, ("分红", "派现", "每份", "现金"))
        for _, r in df_div.iterrows():
            d = _coerce_date(r.get(date_col) if date_col else None)
            if d is None:
                continue
            cash = _to_float(r.get(cash_col) if cash_col else None)
            payload = {
                k: (None if pd.isna(v) else str(v)) for k, v in r.to_dict().items()
            }
            events.append(
                FundEvent(
                    effective_date=d,
                    event_type="dividend",
                    event_key=f"div:{d.isoformat()}:{cash if cash is not None else '-'}",
                    cash_dividend=cash,
                    raw_payload=json.dumps(payload, ensure_ascii=False),
                )
            )

    # Split / conversion details
    df_split = _call_fund_open_info(ak, code=code, indicator="拆分详情")
    if isinstance(df_split, pd.DataFrame) and not df_split.empty:
        cols = [str(c) for c in df_split.columns]
        date_col = _pick_col(cols, ("拆分", "折算", "日期"))
        ratio_col = _pick_col(cols, ("比例", "折算", "拆分"))
        for _, r in df_split.iterrows():
            d = _coerce_date(r.get(date_col) if date_col else None)
            if d is None:
                continue
            ratio = _parse_split_ratio(r.get(ratio_col) if ratio_col else None)
            payload = {
                k: (None if pd.isna(v) else str(v)) for k, v in r.to_dict().items()
            }
            events.append(
                FundEvent(
                    effective_date=d,
                    event_type="split",
                    event_key=f"split:{d.isoformat()}:{ratio if ratio is not None else '-'}",
                    split_ratio=ratio,
                    raw_payload=json.dumps(payload, ensure_ascii=False),
                )
            )
    # stable order
    events.sort(key=lambda x: (x.effective_date, x.event_type, x.event_key))
    return events


def _build_adjusted_by_accum(
    unit_by_date: dict[dt.date, float], accum_by_date: dict[dt.date, float]
) -> tuple[dict[dt.date, float], dict[dt.date, float]]:
    common_dates = sorted(set(unit_by_date.keys()) & set(accum_by_date.keys()))
    if not common_dates:
        return ({}, {})
    ratio = {}
    for d in common_dates:
        u = unit_by_date[d]
        a = accum_by_date[d]
        if u > 0 and a > 0:
            ratio[d] = a / u
    if not ratio:
        return ({}, {})
    last_d = max(ratio.keys())
    base = ratio[last_d]
    qfq: dict[dt.date, float] = {}
    hfq: dict[dt.date, float] = {}
    for d in sorted(unit_by_date.keys()):
        u = unit_by_date[d]
        r = ratio.get(d)
        if u <= 0:
            continue
        if r and base > 0:
            hfq[d] = u * r
            qfq[d] = hfq[d] / base
        else:
            # ratio unavailable on that day: keep fallback continuity to raw
            hfq[d] = u
            qfq[d] = u
    return (qfq, hfq)


def _build_adjusted_by_events(
    unit_by_date: dict[dt.date, float], events: list[FundEvent]
) -> tuple[dict[dt.date, float], dict[dt.date, float], str]:
    """
    Fallback adjustment reconstruction when cumulative NAV is unavailable.
    Uses dividend/split events to build multiplicative adjustment steps.
    """
    if not unit_by_date:
        return ({}, {}, "no_unit_nav")
    dates = sorted(unit_by_date.keys())
    raw = {d: float(unit_by_date[d]) for d in dates}
    qfq = dict(raw)
    applied = 0

    # per event ratio r:
    # qfq(before event) *= r ; latest day remains unchanged
    for ev in sorted(events, key=lambda x: x.effective_date):
        d = ev.effective_date
        if d not in raw:
            # fallback to next available trading day >= event date
            cand = next((x for x in dates if x >= d), None)
            if cand is None:
                continue
            d = cand
        idx = dates.index(d)
        prev_idx = idx - 1
        if prev_idx < 0:
            continue
        prev_p = raw[dates[prev_idx]]
        if prev_p <= 0:
            continue

        r = 1.0
        if ev.event_type == "dividend" and ev.cash_dividend is not None:
            c = float(ev.cash_dividend)
            if c >= 0 and c < prev_p:
                r *= (prev_p - c) / prev_p
        if (
            ev.event_type == "split"
            and ev.split_ratio is not None
            and ev.split_ratio > 0
        ):
            r *= 1.0 / float(ev.split_ratio)
        if r <= 0 or not pd.notna(r) or abs(r - 1.0) < 1e-12:
            continue
        for k in range(0, idx):
            qfq[dates[k]] = qfq[dates[k]] * r
        applied += 1

    total_ratio = (qfq[dates[0]] / raw[dates[0]]) if raw[dates[0]] > 0 else 1.0
    if not pd.notna(total_ratio) or total_ratio <= 0:
        total_ratio = 1.0
    hfq = {d: (qfq[d] / total_ratio) for d in dates}
    note = f"events_applied={applied}"
    return (qfq, hfq, note)


def ingest_one_off_fund(
    db: Session,
    *,
    ak: Any,
    code: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> IngestOffFundResult:
    pool = get_off_fund_pool_by_code(db, code)
    if pool is None:
        raise ValueError(f"off-fund {code} not found in pool")
    settings = get_settings()
    start = start_date or pool.start_date or settings.default_start_date
    end = end_date or pool.end_date or settings.default_end_date
    start_d = _parse_yyyymmdd(start)
    end_d = _parse_yyyymmdd(end)

    try:
        unit_df = _call_fund_open_info(ak, code=code, indicator="单位净值走势")
        if unit_df is None:
            raise ValueError("fund_open_fund_info_em unavailable")
    except Exception as e:
        msg = f"fetch unit nav failed: {e}"
        mark_off_fund_fetch_status(db, code=code, status="failed", message=msg)
        db.commit()
        return IngestOffFundResult(code=code, upserted=0, status="failed", message=msg)
    unit_by_date = _parse_series_df(unit_df, value_keywords=("单位净值",))
    if not unit_by_date:
        msg = "unit nav is empty"
        mark_off_fund_fetch_status(db, code=code, status="failed", message=msg)
        db.commit()
        return IngestOffFundResult(code=code, upserted=0, status="failed", message=msg)

    # optional cumulative NAV (preferred for adjustment)
    accum_df = _call_fund_open_info(ak, code=code, indicator="累计净值走势")
    accum_by_date = (
        _parse_series_df(accum_df, value_keywords=("累计净值",))
        if accum_df is not None
        else {}
    )

    # restrict by requested range
    unit_by_date = {d: v for d, v in unit_by_date.items() if start_d <= d <= end_d}
    accum_by_date = {d: v for d, v in accum_by_date.items() if start_d <= d <= end_d}
    if not unit_by_date:
        msg = "no unit nav in requested range"
        mark_off_fund_fetch_status(db, code=code, status="failed", message=msg)
        db.commit()
        return IngestOffFundResult(code=code, upserted=0, status="failed", message=msg)

    # get dividend/split details for fallback and persistence
    events = _parse_fund_events(ak, code=code)

    qfq_by_date: dict[dt.date, float]
    hfq_by_date: dict[dt.date, float]
    adj_method: str
    if accum_by_date:
        qfq_by_date, hfq_by_date = _build_adjusted_by_accum(unit_by_date, accum_by_date)
        adj_method = "accum_nav"
    else:
        qfq_by_date, hfq_by_date, note = _build_adjusted_by_events(unit_by_date, events)
        adj_method = f"event_rebuild:{note}"

    # fallback safety
    if not qfq_by_date or not hfq_by_date:
        qfq_by_date = dict(unit_by_date)
        hfq_by_date = dict(unit_by_date)
        adj_method = f"{adj_method}|fallback_raw"

    rows_none = [
        OffFundNavRow(
            code=code,
            trade_date=d,
            nav=float(v),
            accum_nav=accum_by_date.get(d),
            source="eastmoney",
            adjust="none",
        )
        for d, v in sorted(unit_by_date.items())
    ]
    rows_qfq = [
        OffFundNavRow(
            code=code,
            trade_date=d,
            nav=float(v),
            accum_nav=accum_by_date.get(d),
            source="eastmoney",
            adjust="qfq",
        )
        for d, v in sorted(qfq_by_date.items())
    ]
    rows_hfq = [
        OffFundNavRow(
            code=code,
            trade_date=d,
            nav=float(v),
            accum_nav=accum_by_date.get(d),
            source="eastmoney",
            adjust="hfq",
        )
        for d, v in sorted(hfq_by_date.items())
    ]

    ev_rows = [
        OffFundEventRow(
            code=code,
            effective_date=e.effective_date,
            event_type=e.event_type,
            event_key=e.event_key,
            cash_dividend=e.cash_dividend,
            split_ratio=e.split_ratio,
            raw_payload=e.raw_payload,
            source="eastmoney",
        )
        for e in events
        if start_d <= e.effective_date <= end_d
    ]

    n0 = upsert_off_fund_navs(db, rows_none)
    n1 = upsert_off_fund_navs(db, rows_qfq)
    n2 = upsert_off_fund_navs(db, rows_hfq)
    replace_off_fund_events(db, code=code, events=ev_rows)
    update_off_fund_pool_data_range(db, code=code, adjust="hfq")
    msg = f"none={len(rows_none)},qfq={len(rows_qfq)},hfq={len(rows_hfq)},adj_method={adj_method},events={len(ev_rows)}"
    mark_off_fund_fetch_status(db, code=code, status="success", message=msg)
    db.commit()
    return IngestOffFundResult(
        code=code, upserted=int(n0 + n1 + n2), status="success", message=msg
    )
