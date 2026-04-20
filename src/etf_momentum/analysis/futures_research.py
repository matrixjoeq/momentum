from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sqlalchemy.orm import Session

from ..db.futures_repo import list_futures_pool, list_futures_prices
from ..db.futures_research_repo import FuturesGroupData

RangeKey = Literal["1m", "3m", "6m", "1y", "3y", "5y", "10y", "all"]
RANGE_KEYS: tuple[RangeKey, ...] = ("1m", "3m", "6m", "1y", "3y", "5y", "10y", "all")


@dataclass(frozen=True)
class ResolvedRange:
    start: str
    end: str
    key: RangeKey


def parse_yyyymmdd(x: str) -> dt.date:
    return dt.datetime.strptime(str(x), "%Y%m%d").date()


def yyyymmdd(x: dt.date) -> str:
    return x.strftime("%Y%m%d")


def resolve_quick_range(
    *,
    key: str,
    base_start: str,
    base_end: str,
) -> ResolvedRange:
    k = str(key or "all").strip().lower()
    if k not in RANGE_KEYS:
        k = "all"
    s0 = parse_yyyymmdd(base_start)
    e0 = parse_yyyymmdd(base_end)
    if s0 > e0:
        s0, e0 = e0, s0
    if k == "all":
        return ResolvedRange(start=yyyymmdd(s0), end=yyyymmdd(e0), key="all")
    months_map = {"1m": 1, "3m": 3, "6m": 6}
    years_map = {"1y": 1, "3y": 3, "5y": 5, "10y": 10}
    if k in months_map:
        st = (pd.Timestamp(e0) - pd.DateOffset(months=months_map[k])).date()
    else:
        st = (pd.Timestamp(e0) - pd.DateOffset(years=years_map[k])).date()
    if st < s0:
        st = s0
    return ResolvedRange(start=yyyymmdd(st), end=yyyymmdd(e0), key=k)  # type: ignore[arg-type]


def _load_close_series(
    db: Session,
    *,
    codes: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    s_d = parse_yyyymmdd(start)
    e_d = parse_yyyymmdd(end)
    out: dict[str, pd.Series] = {}
    for code in codes:
        rows = list_futures_prices(db, code=code, adjust="none", start_date=s_d, end_date=e_d, limit=200000)
        idx = [r.trade_date for r in rows]
        vals = [float(r.close) if r.close is not None else float("nan") for r in rows]
        s = pd.Series(vals, index=idx, dtype=float).sort_index()
        out[str(code)] = s
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_index()


def compute_futures_group_correlation(
    db: Session,
    *,
    group: FuturesGroupData,
    start: str,
    end: str,
    dynamic_universe: bool,
    min_obs: int = 20,
) -> dict:
    codes = [str(c) for c in group.codes]
    if len(codes) == 0:
        return {"ok": False, "error": "empty_group", "meta": {"group_name": group.name}}
    px = _load_close_series(db, codes=codes, start=start, end=end)
    if px.empty:
        return {
            "ok": False,
            "error": "empty_prices",
            "meta": {"group_name": group.name, "start": start, "end": end},
        }
    rets = px.pct_change()
    if dynamic_universe:
        corr_df = rets.corr(min_periods=max(2, int(min_obs)))
        ret_used = rets.dropna(how="all")
        mode = "dynamic_pairwise"
    else:
        ret_used = rets.dropna(how="any")
        corr_df = ret_used.corr(min_periods=max(2, int(min_obs)))
        mode = "static_intersection"

    pool_rows = list_futures_pool(db)
    name_by_code = {str(x.code): str(x.name or x.code) for x in pool_rows}
    aliases: list[dict] = []
    for i, code in enumerate(codes, start=1):
        aliases.append({"id": i, "code": code, "name": name_by_code.get(code, code)})
    matrix: list[list[float | None]] = []
    for c_i in codes:
        row: list[float | None] = []
        for c_j in codes:
            v = None
            if c_i in corr_df.index and c_j in corr_df.columns:
                x = corr_df.loc[c_i, c_j]
                if pd.notna(x):
                    v = float(x)
            row.append(v)
        matrix.append(row)

    used_dates = [d.isoformat() for d in ret_used.index.to_list()] if len(ret_used.index) else []
    return {
        "ok": True,
        "meta": {
            "group_name": group.name,
            "start": start,
            "end": end,
            "dynamic_universe": bool(dynamic_universe),
            "mode": mode,
            "min_obs": int(min_obs),
            "samples": int(len(ret_used.index)),
            "used_start": (used_dates[0] if used_dates else None),
            "used_end": (used_dates[-1] if used_dates else None),
        },
        "aliases": aliases,
        "matrix": matrix,
    }


def compute_futures_group_coverage_summary(
    db: Session,
    *,
    group: FuturesGroupData,
    start: str,
    end: str,
    dynamic_universe: bool,
) -> dict:
    codes = [str(c) for c in group.codes]
    if len(codes) == 0:
        return {"ok": False, "error": "empty_group", "meta": {"group_name": group.name}}
    px = _load_close_series(db, codes=codes, start=start, end=end)
    if px.empty:
        return {
            "ok": False,
            "error": "empty_prices",
            "meta": {"group_name": group.name, "start": start, "end": end},
        }

    symbols: list[dict] = []
    union_dates: set[dt.date] = set()
    inter_dates: set[dt.date] | None = None
    for i, code in enumerate(codes, start=1):
        if code not in px.columns:
            symbols.append(
                {
                    "id": i,
                    "code": code,
                    "name": code,
                    "valid_start": None,
                    "valid_end": None,
                    "valid_points": 0,
                }
            )
            inter_dates = set() if inter_dates is None else inter_dates & set()
            continue
        s = px[code].dropna()
        ds = list(s.index)
        if not ds:
            symbols.append(
                {
                    "id": i,
                    "code": code,
                    "name": code,
                    "valid_start": None,
                    "valid_end": None,
                    "valid_points": 0,
                }
            )
            inter_dates = set() if inter_dates is None else inter_dates & set()
            continue
        dset = set(ds)
        union_dates |= dset
        inter_dates = dset if inter_dates is None else inter_dates & dset
        symbols.append(
            {
                "id": i,
                "code": code,
                "name": code,
                "valid_start": min(ds).isoformat(),
                "valid_end": max(ds).isoformat(),
                "valid_points": int(len(ds)),
            }
        )

    inter_dates = inter_dates or set()
    union_n = int(len(union_dates))
    inter_n = int(len(inter_dates))
    mode = "dynamic_union" if dynamic_universe else "static_intersection"
    effective_points = union_n if dynamic_universe else inter_n
    used_dates = sorted(union_dates if dynamic_universe else inter_dates)
    return {
        "ok": True,
        "meta": {
            "group_name": group.name,
            "start": start,
            "end": end,
            "dynamic_universe": bool(dynamic_universe),
            "mode": mode,
            "union_points": union_n,
            "intersection_points": inter_n,
            "effective_points": int(effective_points),
            "used_start": (used_dates[0].isoformat() if used_dates else None),
            "used_end": (used_dates[-1].isoformat() if used_dates else None),
        },
        "symbols": symbols,
    }


def select_symbols_by_correlation(
    *,
    correlation_output: dict,
    mode: Literal["lowest", "highest"],
    score_basis: Literal["mean", "mean_abs"],
    n: int,
) -> dict:
    """
    Select N symbols by their mean pairwise correlation in current matrix.

    Score definition:
    - mean: arithmetic mean of pairwise correlations (avg_corr).
    - mean_abs: arithmetic mean of absolute pairwise correlations (avg_abs_corr).
    """
    if not correlation_output or correlation_output.get("ok") is not True:
        return {"ok": False, "error": "invalid_correlation_output"}
    aliases = correlation_output.get("aliases") or []
    matrix = correlation_output.get("matrix") or []
    if not isinstance(aliases, list) or not isinstance(matrix, list):
        return {"ok": False, "error": "invalid_matrix"}
    size = min(len(aliases), len(matrix))
    if size <= 1:
        return {"ok": False, "error": "insufficient_symbols"}

    rows: list[dict] = []
    for i in range(size):
        vals: list[float] = []
        abs_vals: list[float] = []
        row_i = matrix[i] if i < len(matrix) and isinstance(matrix[i], list) else []
        for j in range(size):
            if i == j:
                continue
            v = row_i[j] if j < len(row_i) else None
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if pd.isna(fv):
                continue
            vals.append(fv)
            abs_vals.append(abs(fv))
        avg_corr = (sum(vals) / len(vals)) if vals else None
        avg_abs_corr = (sum(abs_vals) / len(abs_vals)) if abs_vals else None
        score = avg_abs_corr if score_basis == "mean_abs" else avg_corr
        item = aliases[i] if i < len(aliases) and isinstance(aliases[i], dict) else {}
        rows.append(
            {
                "id": int(item.get("id", i + 1)),
                "code": str(item.get("code", "")),
                "name": str(item.get("name", item.get("code", ""))),
                "avg_corr": avg_corr,
                "avg_abs_corr": avg_abs_corr,
                "score": score,
                "obs_pairs": int(len(vals)),
            }
        )

    valid = [r for r in rows if r.get("score") is not None]
    if not valid:
        return {"ok": False, "error": "no_valid_scores"}
    reverse = mode == "highest"
    valid.sort(key=lambda x: float(x["score"]), reverse=reverse)
    k = max(1, min(int(n), len(valid)))
    picked = valid[:k]
    return {
        "ok": True,
        "mode": mode,
        "score_basis": score_basis,
        "requested_n": int(n),
        "effective_n": int(k),
        "items": picked,
    }
