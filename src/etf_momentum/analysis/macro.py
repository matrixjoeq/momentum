from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..analysis.leadlag import LeadLagInputs, compute_lead_lag
from ..db.repo import list_macro_prices


def _parse_yyyymmdd(s: str) -> dt.date:
    return dt.datetime.strptime(str(s), "%Y%m%d").date()


def load_macro_close_series(
    db: Session,
    *,
    series_id: str,
    start: str,
    end: str,
) -> pd.Series:
    start_d = _parse_yyyymmdd(start)
    end_d = _parse_yyyymmdd(end)
    rows = list_macro_prices(db, series_id=series_id, start_date=start_d, end_date=end_d, limit=500000)
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(
        data=[float(r.close) if r.close is not None else np.nan for r in rows],
        index=[r.trade_date for r in rows],
        dtype=float,
    ).dropna()
    return s.sort_index()


def analyze_pair_leadlag(
    db: Session,
    *,
    a_series_id: str,
    b_series_id: str,
    start: str,
    end: str,
    index_align: str = "none",
    max_lag: int = 20,
    granger_max_lag: int = 10,
    alpha: float = 0.05,
    trade_cost_bps: float = 0.0,
    rolling_window: int = 252,
    enable_threshold: bool = True,
    threshold_quantile: float = 0.80,
    walk_forward: bool = True,
    train_ratio: float = 0.60,
    walk_objective: str = "sharpe",
) -> dict[str, Any]:
    """
    Pairwise lead/lag analysis between two macro series (daily close).

    Returns the same structure as compute_lead_lag(), plus meta about series ids.
    """
    a = load_macro_close_series(db, series_id=a_series_id, start=start, end=end)
    b = load_macro_close_series(db, series_id=b_series_id, start=start, end=end)
    if a.empty:
        return {"ok": False, "reason": "empty_series_a", "a_series_id": a_series_id, "b_series_id": b_series_id}
    if b.empty:
        return {"ok": False, "reason": "empty_series_b", "a_series_id": a_series_id, "b_series_id": b_series_id}

    res = compute_lead_lag(
        LeadLagInputs(
            etf_close=a,
            idx_close=b,
            max_lag=int(max_lag),
            granger_max_lag=int(granger_max_lag),
            alpha=float(alpha),
            index_align=str(index_align or "none"),
            trade_cost_bps=float(trade_cost_bps),
            rolling_window=int(rolling_window),
            enable_threshold=bool(enable_threshold),
            threshold_quantile=float(threshold_quantile),
            walk_forward=bool(walk_forward),
            train_ratio=float(train_ratio),
            walk_objective=str(walk_objective or "sharpe"),
        )
    )
    if not bool(res.get("ok")):
        return {"ok": False, "reason": str(res.get("reason") or "analysis_failed"), "a_series_id": a_series_id, "b_series_id": b_series_id}
    meta = dict(res.get("meta") or {})
    meta.update({"a_series_id": a_series_id, "b_series_id": b_series_id, "index_align": index_align})
    res["meta"] = meta
    return res

