from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _annualized_from_daily(simple_ret: np.ndarray, ann: int = 252) -> dict[str, float]:
    r = simple_ret[np.isfinite(simple_ret)]
    if r.size < 2:
        return {"cagr": float("nan"), "vol": float("nan"), "sharpe": float("nan")}
    nav = np.cumprod(1.0 + r)
    years = (len(nav) - 1) / float(ann)
    cagr = float(nav[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(np.std(r, ddof=1) * math.sqrt(ann)) if r.size >= 2 else float("nan")
    sharpe = float((np.mean(r) / np.std(r, ddof=1)) * math.sqrt(ann)) if np.std(r, ddof=1) > 0 else float("nan")
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe}


def _max_drawdown(nav: np.ndarray) -> float:
    if nav.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    return float(np.min(dd))


def backtest_tiered_exposure_by_level(
    df: pd.DataFrame,
    levels: pd.Series,
    *,
    thresholds_abs: list[float],
    exposures: list[float],
    cost_bps: float = 0.0,
    ann: int = 252,
) -> dict[str, Any]:
    """
    Tiered exposure backtest driven by a "level" series (e.g. implied vol index level,
    realized vol proxy, range-based vol proxy).

    - thresholds_abs: ascending absolute level cut points (length m)
    - exposures: tier exposures in [0,1], length m+1
    - Strategy: exposure_t = exposures[bucket(level_t)], hold each day; turnover cost on exposure changes.

    df must contain:
      - etf_ret: log return series (index aligned to levels)
    """
    th = [float(x) for x in (thresholds_abs or []) if x is not None and np.isfinite(float(x))]
    th = sorted(th)
    exps = [float(x) for x in (exposures or []) if x is not None and np.isfinite(float(x))]
    if len(exps) != len(th) + 1:
        return {"ok": False, "error": "bad_tier_exposures_len", "thresholds_abs": th, "exposures": exps}
    if any((x < 0.0 or x > 1.0) for x in exps):
        return {"ok": False, "error": "exposures_out_of_range", "thresholds_abs": th, "exposures": exps}
    if any(th[i] > th[i + 1] for i in range(len(th) - 1)):
        return {"ok": False, "error": "thresholds_not_sorted", "thresholds_abs": th, "exposures": exps}

    lvl = pd.to_numeric(levels, errors="coerce").reindex(df.index)
    lv = lvl.to_numpy(dtype=float)
    th_arr = np.array(th, dtype=float)

    exp = np.full(len(df), np.nan, dtype=float)
    bucket = np.full(len(df), -1, dtype=int)
    for i in range(len(df)):
        if not np.isfinite(lv[i]):
            exp[i] = np.nan
            bucket[i] = -1
            continue
        j = int(np.searchsorted(th_arr, lv[i], side="left")) if th_arr.size else 0
        j = int(max(0, min(len(exps) - 1, j)))
        exp[i] = float(exps[j])
        bucket[i] = j

    # bucket usage rates (excluding nan)
    okb = bucket >= 0
    counts = [int(np.sum(bucket[okb] == j)) for j in range(len(exps))]
    total = int(np.sum(okb))
    bucket_rates = [float(c) / float(total) if total > 0 else float("nan") for c in counts]

    r_etf = np.expm1(pd.to_numeric(df["etf_ret"], errors="coerce").to_numpy(dtype=float))
    cost = float(cost_bps) / 10000.0
    turn = np.zeros_like(exp)
    turn[1:] = np.abs(exp[1:] - exp[:-1])
    r_strat = exp * r_etf - turn * cost

    ok2 = np.isfinite(r_strat) & np.isfinite(r_etf) & np.isfinite(exp)
    r_strat_ok = r_strat[ok2]
    r_etf_ok = r_etf[ok2]
    nav_strat = np.cumprod(1.0 + r_strat_ok) if r_strat_ok.size else np.array([], dtype=float)
    nav_etf = np.cumprod(1.0 + r_etf_ok) if r_etf_ok.size else np.array([], dtype=float)

    strat_stats = _annualized_from_daily(r_strat_ok, ann=int(ann))
    bh_stats = _annualized_from_daily(r_etf_ok, ann=int(ann))
    strat_stats["max_drawdown"] = _max_drawdown(nav_strat)
    bh_stats["max_drawdown"] = _max_drawdown(nav_etf)

    return {
        "ok": True,
        "thresholds_abs": th,
        "exposures": exps,
        "bucket_rates": bucket_rates,
        "dates": [d.isoformat() for d in df.index[ok2]],
        "level": lvl.to_numpy(dtype=float)[ok2].astype(float).tolist(),
        "exp": exp[ok2].astype(float).tolist(),
        "ret_strategy": r_strat_ok.astype(float).tolist(),
        "ret_buy_hold": r_etf_ok.astype(float).tolist(),
        "nav_strategy": nav_strat.astype(float).tolist(),
        "nav_buy_hold": nav_etf.astype(float).tolist(),
        "metrics": {"strategy": strat_stats, "buy_hold": bh_stats},
    }


def backtest_tiered_exposure_by_level_rolling_quantiles(
    df: pd.DataFrame,
    levels: pd.Series,
    *,
    quantiles: list[float],
    window_days: int,
    exposures: list[float],
    cost_bps: float = 0.0,
    ann: int = 252,
    min_periods: int = 20,
) -> dict[str, Any]:
    """
    Tiered exposure backtest where thresholds are recomputed each day using trailing-window quantiles.

    This supports "近1年/3年/5年/10年/全区间" style quantile windows without requiring options data.
    """
    qs = [float(q) for q in (quantiles or []) if q is not None and np.isfinite(float(q))]
    qs = sorted([float(min(max(q, 0.01), 0.99)) for q in qs])
    exps = [float(x) for x in (exposures or []) if x is not None and np.isfinite(float(x))]
    if len(qs) == 0:
        return {"ok": False, "error": "empty_level_quantiles"}
    if len(exps) != len(qs) + 1:
        return {"ok": False, "error": "bad_tier_exposures_len", "need": int(len(qs) + 1), "got": int(len(exps))}
    if any((x < 0.0 or x > 1.0) for x in exps):
        return {"ok": False, "error": "exposures_out_of_range", "exposures": exps}

    w = int(max(2, window_days))
    mp = int(max(1, min_periods))
    lvl = pd.to_numeric(levels, errors="coerce").reindex(df.index)
    lv = lvl.to_numpy(dtype=float)

    # rolling thresholds for each quantile
    # IMPORTANT: shift(1) to avoid using today's level to set today's exposure (lookahead).
    thr_df = pd.DataFrame({f"q{int(q*100)}": lvl.rolling(w, min_periods=mp).quantile(q).shift(1) for q in qs})
    thr_mat = thr_df.to_numpy(dtype=float)  # shape (n, m)

    exp = np.full(len(df), np.nan, dtype=float)
    bucket = np.full(len(df), -1, dtype=int)
    for i in range(len(df)):
        if not np.isfinite(lv[i]):
            continue
        th_row = thr_mat[i, :]
        if th_row.size and not np.isfinite(th_row).all():
            continue
        # quantile thresholds are non-decreasing by construction; still be safe
        th_row_sorted = np.sort(th_row) if th_row.size else np.array([], dtype=float)
        j = int(np.searchsorted(th_row_sorted, lv[i], side="left")) if th_row_sorted.size else 0
        j = int(max(0, min(len(exps) - 1, j)))
        exp[i] = float(exps[j])
        bucket[i] = j

    okb = bucket >= 0
    counts = [int(np.sum(bucket[okb] == j)) for j in range(len(exps))]
    total = int(np.sum(okb))
    bucket_rates = [float(c) / float(total) if total > 0 else float("nan") for c in counts]

    r_etf = np.expm1(pd.to_numeric(df["etf_ret"], errors="coerce").to_numpy(dtype=float))
    cost = float(cost_bps) / 10000.0
    turn = np.zeros_like(exp)
    turn[1:] = np.abs(exp[1:] - exp[:-1])
    r_strat = exp * r_etf - turn * cost

    ok2 = np.isfinite(r_strat) & np.isfinite(r_etf) & np.isfinite(exp)
    r_strat_ok = r_strat[ok2]
    r_etf_ok = r_etf[ok2]
    nav_strat = np.cumprod(1.0 + r_strat_ok) if r_strat_ok.size else np.array([], dtype=float)
    nav_etf = np.cumprod(1.0 + r_etf_ok) if r_etf_ok.size else np.array([], dtype=float)

    strat_stats = _annualized_from_daily(r_strat_ok, ann=int(ann))
    bh_stats = _annualized_from_daily(r_etf_ok, ann=int(ann))
    strat_stats["max_drawdown"] = _max_drawdown(nav_strat)
    bh_stats["max_drawdown"] = _max_drawdown(nav_etf)

    # last available thresholds (for display/debug)
    last_thr = None
    for i in range(len(df) - 1, -1, -1):
        row = thr_mat[i, :]
        if row.size and np.isfinite(row).all():
            last_thr = row.astype(float).tolist()
            break

    return {
        "ok": True,
        "quantiles": qs,
        "window_days": int(w),
        "thresholds_abs_last": last_thr,
        "window_mode": "rolling",
        "exposures": exps,
        "bucket_rates": bucket_rates,
        "dates": [d.isoformat() for d in df.index[ok2]],
        "level": lvl.to_numpy(dtype=float)[ok2].astype(float).tolist(),
        "exp": exp[ok2].astype(float).tolist(),
        "ret_strategy": r_strat_ok.astype(float).tolist(),
        "ret_buy_hold": r_etf_ok.astype(float).tolist(),
        "nav_strategy": nav_strat.astype(float).tolist(),
        "nav_buy_hold": nav_etf.astype(float).tolist(),
        "metrics": {"strategy": strat_stats, "buy_hold": bh_stats},
    }


def backtest_tiered_exposure_by_level_expanding_quantiles(
    df: pd.DataFrame,
    levels: pd.Series,
    *,
    quantiles: list[float],
    exposures: list[float],
    cost_bps: float = 0.0,
    ann: int = 252,
    min_periods: int = 20,
) -> dict[str, Any]:
    """
    Tiered exposure backtest where thresholds are recomputed each day using expanding-window quantiles.

    This is the recommended "all" mode to avoid lookahead bias:
    - thresholds at date t are computed from levels up to t-1 (shifted by 1)
    """
    qs = [float(q) for q in (quantiles or []) if q is not None and np.isfinite(float(q))]
    qs = sorted([float(min(max(q, 0.01), 0.99)) for q in qs])
    exps = [float(x) for x in (exposures or []) if x is not None and np.isfinite(float(x))]
    if len(qs) == 0:
        return {"ok": False, "error": "empty_level_quantiles"}
    if len(exps) != len(qs) + 1:
        return {"ok": False, "error": "bad_tier_exposures_len", "need": int(len(qs) + 1), "got": int(len(exps))}
    if any((x < 0.0 or x > 1.0) for x in exps):
        return {"ok": False, "error": "exposures_out_of_range", "exposures": exps}

    mp = int(max(1, min_periods))
    lvl = pd.to_numeric(levels, errors="coerce").reindex(df.index)
    lv = lvl.to_numpy(dtype=float)

    # expanding thresholds for each quantile
    thr_df = pd.DataFrame({f"q{int(q*100)}": lvl.expanding(min_periods=mp).quantile(q).shift(1) for q in qs})
    thr_mat = thr_df.to_numpy(dtype=float)

    exp = np.full(len(df), np.nan, dtype=float)
    bucket = np.full(len(df), -1, dtype=int)
    for i in range(len(df)):
        if not np.isfinite(lv[i]):
            continue
        th_row = thr_mat[i, :]
        if th_row.size and not np.isfinite(th_row).all():
            continue
        th_row_sorted = np.sort(th_row) if th_row.size else np.array([], dtype=float)
        j = int(np.searchsorted(th_row_sorted, lv[i], side="left")) if th_row_sorted.size else 0
        j = int(max(0, min(len(exps) - 1, j)))
        exp[i] = float(exps[j])
        bucket[i] = j

    okb = bucket >= 0
    counts = [int(np.sum(bucket[okb] == j)) for j in range(len(exps))]
    total = int(np.sum(okb))
    bucket_rates = [float(c) / float(total) if total > 0 else float("nan") for c in counts]

    r_etf = np.expm1(pd.to_numeric(df["etf_ret"], errors="coerce").to_numpy(dtype=float))
    cost = float(cost_bps) / 10000.0
    turn = np.zeros_like(exp)
    turn[1:] = np.abs(exp[1:] - exp[:-1])
    r_strat = exp * r_etf - turn * cost

    ok2 = np.isfinite(r_strat) & np.isfinite(r_etf) & np.isfinite(exp)
    r_strat_ok = r_strat[ok2]
    r_etf_ok = r_etf[ok2]
    nav_strat = np.cumprod(1.0 + r_strat_ok) if r_strat_ok.size else np.array([], dtype=float)
    nav_etf = np.cumprod(1.0 + r_etf_ok) if r_etf_ok.size else np.array([], dtype=float)

    strat_stats = _annualized_from_daily(r_strat_ok, ann=int(ann))
    bh_stats = _annualized_from_daily(r_etf_ok, ann=int(ann))
    strat_stats["max_drawdown"] = _max_drawdown(nav_strat)
    bh_stats["max_drawdown"] = _max_drawdown(nav_etf)

    last_thr = None
    for i in range(len(df) - 1, -1, -1):
        row = thr_mat[i, :]
        if row.size and np.isfinite(row).all():
            last_thr = row.astype(float).tolist()
            break

    return {
        "ok": True,
        "quantiles": qs,
        "window_days": None,
        "thresholds_abs_last": last_thr,
        "window_mode": "expanding",
        "exposures": exps,
        "bucket_rates": bucket_rates,
        "dates": [d.isoformat() for d in df.index[ok2]],
        "level": lvl.to_numpy(dtype=float)[ok2].astype(float).tolist(),
        "exp": exp[ok2].astype(float).tolist(),
        "ret_strategy": r_strat_ok.astype(float).tolist(),
        "ret_buy_hold": r_etf_ok.astype(float).tolist(),
        "nav_strategy": nav_strat.astype(float).tolist(),
        "nav_buy_hold": nav_etf.astype(float).tolist(),
        "metrics": {"strategy": strat_stats, "buy_hold": bh_stats},
    }

