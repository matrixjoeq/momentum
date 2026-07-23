#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import trend_momentum_base_param_search as base

DEFAULT_OUTPUT_JSON = (
    "src/etf_momentum/web/data/trend_momentum_walkforward_5y_results.json"
)
DEFAULT_OUTPUT_YEARLY_CSV = (
    "src/etf_momentum/web/data/trend_momentum_walkforward_5y_yearly.csv"
)
DEFAULT_OUTPUT_SEARCH_CSV = (
    "src/etf_momentum/web/data/trend_momentum_walkforward_5y_search_summary.csv"
)
DEFAULT_LOOKBACK_STABILITY_GAP_QUANTILE = 0.5

TRADING_DAYS_PER_YEAR = 252


def _to_float(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _to_yyyymmdd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def _to_iso_date(d: dt.date) -> str:
    return d.isoformat()


def _parse_yyyymmdd(s: str) -> dt.date:
    return dt.datetime.strptime(str(s), "%Y%m%d").date()


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    data = sorted(float(x) for x in values)
    if len(data) == 1:
        return float(data[0])
    qq = float(max(0.0, min(1.0, q)))
    pos = qq * float(len(data) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(data[lo])
    w = float(pos - lo)
    return float(data[lo] * (1.0 - w) + data[hi] * w)


def _safe_div(a: float | None, b: float | None) -> float | None:
    aa = _to_float(a)
    bb = _to_float(b)
    if aa is None or bb is None or abs(bb) <= 1e-12:
        return None
    return float(aa / bb)


def _max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return float("nan")
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def _max_drawdown_duration_days(nav: pd.Series) -> int:
    if nav.empty:
        return 0
    peak = nav.cummax()
    last_peak_idx = nav.index[0]
    max_dur = 0
    for t in nav.index:
        if nav.loc[t] >= peak.loc[t]:
            last_peak_idx = t
        else:
            dur = int((t - last_peak_idx).days)
            if dur > max_dur:
                max_dur = dur
    return int(max_dur)


def _annualized_return(nav: pd.Series) -> float:
    if nav.empty:
        return float("nan")
    n = len(nav) - 1
    if n <= 0:
        return 0.0
    total = float(nav.iloc[-1] / nav.iloc[0])
    return float(total ** (TRADING_DAYS_PER_YEAR / float(n)) - 1.0)


def _annualized_vol(daily_ret: pd.Series) -> float:
    if daily_ret.empty:
        return float("nan")
    return float(daily_ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _sharpe(daily_ret: pd.Series) -> float:
    if daily_ret.empty:
        return float("nan")
    ex = daily_ret.astype(float)
    std = float(ex.std(ddof=1))
    if not math.isfinite(std) or abs(std) <= 1e-12:
        return float("nan")
    return float(ex.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _sortino(daily_ret: pd.Series) -> float:
    if daily_ret.empty:
        return float("nan")
    ex = daily_ret.astype(float)
    downside = ex.where(ex < 0.0, 0.0)
    dd_std = float(downside.std(ddof=1))
    if not math.isfinite(dd_std) or abs(dd_std) <= 1e-12:
        return float("nan")
    return float(ex.mean() / dd_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _ulcer_index(nav: pd.Series) -> float:
    if nav.empty:
        return float("nan")
    peak = nav.cummax()
    dd = nav / peak - 1.0
    underwater = (-dd).clip(lower=0.0)
    x = underwater * 100.0
    return float(np.sqrt(np.mean(np.square(x.to_numpy(dtype=float)))))


def _sqn_recent_100(
    trades: list[dict[str, Any]], *, min_trades: int = 30
) -> dict[str, Any]:
    r_vals: list[float] = []
    ordered = _sort_trades_by_time(trades)
    for tr in ordered[-100:]:
        rv = _to_float(tr.get("r_multiple"))
        if rv is not None:
            r_vals.append(float(rv))
    n = len(r_vals)
    out = {
        "applicable": False,
        "reason": None,
        "trade_count_total": int(len(ordered)),
        "trade_count_used": int(n),
        "min_trades": int(min_trades),
        "max_trades": 100,
        "expectancy_r": None,
        "std_r": None,
        "sqn": None,
    }
    if n < int(min_trades):
        out["reason"] = f"trades_lt_{int(min_trades)}"
        return out
    s = pd.Series(r_vals, dtype=float)
    exp_r = _to_float(s.mean())
    std_r = _to_float(s.std(ddof=0))
    out["expectancy_r"] = exp_r
    out["std_r"] = std_r
    if exp_r is None or std_r is None or std_r <= 0.0:
        out["reason"] = "std_non_positive"
        return out
    out["applicable"] = True
    out["sqn"] = float((exp_r / std_r) * np.sqrt(float(n)))
    return out


def _sort_trades_by_time(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _k(idx_tr: tuple[int, dict[str, Any]]) -> tuple[pd.Timestamp, int]:
        idx, tr = idx_tr
        d_exit = pd.to_datetime(tr.get("exit_date"), errors="coerce")
        d_entry = pd.to_datetime(tr.get("entry_date"), errors="coerce")
        d = d_exit if not pd.isna(d_exit) else d_entry
        if pd.isna(d):
            d = pd.Timestamp.min
        return d, int(idx)

    ordered = sorted(enumerate(trades), key=_k)
    return [tr for _, tr in ordered]


def _rolling_sqn_recent_100_series(
    trades: list[dict[str, Any]], *, min_trades: int = 30
) -> tuple[list[str], list[float | None]]:
    ordered = _sort_trades_by_time(trades)
    out_dates: list[str] = []
    out_vals: list[float | None] = []
    for i in range(len(ordered)):
        window = ordered[max(0, i - 99) : i + 1]
        sqn_obj = _sqn_recent_100(window, min_trades=min_trades)
        out_dates.append(
            str(ordered[i].get("exit_date") or ordered[i].get("entry_date") or "")
        )
        out_vals.append(_to_float(sqn_obj.get("sqn")))
    return out_dates, out_vals


def _rolling_max_drawdown(nav: pd.Series, window: int) -> pd.Series:
    w = int(window)
    return nav.rolling(window=w, min_periods=w).apply(
        lambda x: float((x / np.maximum.accumulate(x) - 1.0).min()),
        raw=True,
    )


def _rolling_ulcer_index(nav: pd.Series, window: int) -> pd.Series:
    w = int(window)
    return nav.rolling(window=w, min_periods=w).apply(
        lambda x: float(
            np.sqrt(
                np.mean(
                    np.square(
                        np.clip(
                            -(x / np.maximum.accumulate(x) - 1.0),
                            0.0,
                            None,
                        )
                        * 100.0
                    )
                )
            )
        ),
        raw=True,
    )


def _rolling_sharpe(daily_ret: pd.Series, window: int) -> pd.Series:
    w = int(window)
    return daily_ret.rolling(window=w, min_periods=w).apply(
        lambda x: (
            float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
            if np.std(x, ddof=1) > 1e-12
            else np.nan
        ),
        raw=True,
    )


def _rolling_sortino(daily_ret: pd.Series, window: int) -> pd.Series:
    w = int(window)

    def _f(x: np.ndarray) -> float:
        downside = np.where(x < 0.0, x, 0.0)
        dd_std = float(np.std(downside, ddof=1))
        if not math.isfinite(dd_std) or dd_std <= 1e-12:
            return float("nan")
        return float(np.mean(x) / dd_std * np.sqrt(TRADING_DAYS_PER_YEAR))

    return daily_ret.rolling(window=w, min_periods=w).apply(_f, raw=True)


def _window_summary(s: pd.Series) -> dict[str, Any]:
    z = s.replace([np.inf, -np.inf], np.nan).dropna()
    if z.empty:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "max": None,
            "mean": None,
            "latest": None,
        }
    return {
        "count": int(len(z)),
        "min": float(z.min()),
        "p10": float(z.quantile(0.10)),
        "p50": float(z.quantile(0.50)),
        "p90": float(z.quantile(0.90)),
        "max": float(z.max()),
        "mean": float(z.mean()),
        "latest": float(z.iloc[-1]),
    }


def _build_year_slices(
    start: dt.date, end: dt.date, train_years: int
) -> list[dict[str, Any]]:
    first_trade_year = int(start.year + int(train_years))
    out: list[dict[str, Any]] = []
    for year in range(first_trade_year, end.year + 1):
        trade_start = dt.date(year, 1, 1)
        trade_end = min(dt.date(year, 12, 31), end)
        if trade_start > end:
            break
        out.append(
            {
                "trade_year": int(year),
                "train_start": dt.date(year - train_years, 1, 1),
                "train_end": dt.date(year - 1, 12, 31),
                "trade_start": trade_start,
                "trade_end": trade_end,
            }
        )
    return out


def _call_trend_api(
    *,
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    url = base._join_url(base_url, endpoint)
    resp = base._http_post_json(url=url, payload=payload, timeout=timeout)
    if not isinstance(resp, dict):
        raise RuntimeError("unexpected response schema")
    return resp


def _row_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    comp = _to_float(row.get("composite_score"))
    sharpe = base._metric_value(row, "sharpe_ratio")
    ann = base._metric_value(row, "annualized_return")
    mdd = base._metric_value(row, "max_drawdown")
    return (
        float(comp if comp is not None else -1e18),
        float(sharpe if sharpe is not None else -1e18),
        float(ann if ann is not None else -1e18),
        float(-abs(mdd) if mdd is not None else -1e18),
    )


def _build_lookback_representatives(
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    rep: dict[int, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("status") or "") != "ok":
            continue
        if not bool(row.get("objective_eligible")):
            continue
        lb = row.get("mom_lookback")
        try:
            lb_i = int(lb)
        except (TypeError, ValueError):
            continue
        cur = rep.get(lb_i)
        if cur is None or _row_rank_key(row) > _row_rank_key(cur):
            rep[lb_i] = row
    return rep


def _pick_best_stable_by_lookback(
    rows: list[dict[str, Any]],
    *,
    lookback_stability_gap_quantile: float,
    lookback_stability_max_gap_override: float | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    rep = _build_lookback_representatives(rows)
    if not rep:
        return None, {
            "selection_mode": "no_eligible_candidate",
            "stability_threshold": None,
            "stability_quantile": float(lookback_stability_gap_quantile),
            "stability_threshold_mode": "no_samples",
        }

    candidates = sorted(rep.items(), key=lambda kv: _row_rank_key(kv[1]), reverse=True)
    gap_samples: list[float] = []
    for lb, row in candidates:
        left = rep.get(lb - 1)
        right = rep.get(lb + 1)
        if left is None or right is None:
            continue
        comp = _to_float(row.get("composite_score"))
        left_comp = _to_float((left or {}).get("composite_score"))
        right_comp = _to_float((right or {}).get("composite_score"))
        if comp is None or left_comp is None or right_comp is None:
            continue
        gap_left = abs(float(comp) - float(left_comp))
        gap_right = abs(float(comp) - float(right_comp))
        gap_samples.append(float(max(gap_left, gap_right)))

    dynamic_threshold = _quantile(gap_samples, float(lookback_stability_gap_quantile))
    threshold_mode = "quantile"
    if lookback_stability_max_gap_override is not None:
        threshold_mode = "fixed_override"
        dynamic_threshold = float(lookback_stability_max_gap_override)

    relaxed_best: tuple[float, float, dict[str, Any], dict[str, Any]] | None = None
    rejected: list[dict[str, Any]] = []
    for rank_idx, (lb, row) in enumerate(candidates, start=1):
        left = rep.get(lb - 1)
        right = rep.get(lb + 1)
        comp = _to_float(row.get("composite_score"))
        left_comp = _to_float((left or {}).get("composite_score"))
        right_comp = _to_float((right or {}).get("composite_score"))
        diag = {
            "rank": int(rank_idx),
            "lookback": int(lb),
            "composite": comp,
            "left_lookback": int(lb - 1),
            "left_composite": left_comp,
            "right_lookback": int(lb + 1),
            "right_composite": right_comp,
        }
        if left is None or right is None:
            rejected.append({**diag, "reason": "missing_neighbor"})
            continue
        if comp is None or left_comp is None or right_comp is None:
            rejected.append({**diag, "reason": "missing_composite"})
            continue
        gap_left = abs(float(comp) - float(left_comp))
        gap_right = abs(float(comp) - float(right_comp))
        max_gap = float(max(gap_left, gap_right))
        mean_gap = float((gap_left + gap_right) / 2.0)
        if dynamic_threshold is not None and max_gap <= float(dynamic_threshold):
            return row, {
                "selection_mode": "stable_primary",
                "stability_threshold": float(dynamic_threshold),
                "stability_quantile": float(lookback_stability_gap_quantile),
                "stability_threshold_mode": threshold_mode,
                "selected_rank": int(rank_idx),
                "selected_lookback": int(lb),
                "selected_composite": float(comp),
                "stability_gap_left": float(gap_left),
                "stability_gap_right": float(gap_right),
                "stability_max_gap": float(max_gap),
                "stability_mean_gap": float(mean_gap),
                "stability_gap_sample_count": int(len(gap_samples)),
                "stability_gap_sample_p10": _quantile(gap_samples, 0.10),
                "stability_gap_sample_p50": _quantile(gap_samples, 0.50),
                "stability_gap_sample_p90": _quantile(gap_samples, 0.90),
                "eligible_lookback_count": int(len(rep)),
                "rejected_samples": rejected[:8],
            }
        rejected.append(
            {
                **diag,
                "reason": "peak_gap_too_large",
                "gap_left": float(gap_left),
                "gap_right": float(gap_right),
                "max_gap": float(max_gap),
            }
        )
        relaxed_key = (float(max_gap), float(-float(comp)))
        relaxed_diag = {
            "selection_mode": "stable_relaxed_min_gap",
            "stability_threshold": (
                float(dynamic_threshold) if dynamic_threshold is not None else None
            ),
            "stability_quantile": float(lookback_stability_gap_quantile),
            "stability_threshold_mode": threshold_mode,
            "selected_rank": int(rank_idx),
            "selected_lookback": int(lb),
            "selected_composite": float(comp),
            "stability_gap_left": float(gap_left),
            "stability_gap_right": float(gap_right),
            "stability_max_gap": float(max_gap),
            "stability_mean_gap": float(mean_gap),
            "stability_gap_sample_count": int(len(gap_samples)),
            "stability_gap_sample_p10": _quantile(gap_samples, 0.10),
            "stability_gap_sample_p50": _quantile(gap_samples, 0.50),
            "stability_gap_sample_p90": _quantile(gap_samples, 0.90),
            "eligible_lookback_count": int(len(rep)),
            "rejected_samples": rejected[:8],
        }
        if relaxed_best is None or relaxed_key < (
            relaxed_best[0],
            relaxed_best[1],
        ):
            relaxed_best = (float(max_gap), float(-float(comp)), row, relaxed_diag)

    if relaxed_best is not None:
        return relaxed_best[2], relaxed_best[3]

    lb0, row0 = candidates[0]
    comp0 = _to_float(row0.get("composite_score"))
    return row0, {
        "selection_mode": "fallback_top_representative",
        "stability_threshold": (
            float(dynamic_threshold) if dynamic_threshold is not None else None
        ),
        "stability_quantile": float(lookback_stability_gap_quantile),
        "stability_threshold_mode": threshold_mode,
        "selected_rank": 1,
        "selected_lookback": int(lb0),
        "selected_composite": float(comp0) if comp0 is not None else None,
        "stability_gap_left": None,
        "stability_gap_right": None,
        "stability_max_gap": None,
        "stability_mean_gap": None,
        "stability_gap_sample_count": int(len(gap_samples)),
        "stability_gap_sample_p10": _quantile(gap_samples, 0.10),
        "stability_gap_sample_p50": _quantile(gap_samples, 0.50),
        "stability_gap_sample_p90": _quantile(gap_samples, 0.90),
        "eligible_lookback_count": int(len(rep)),
        "rejected_samples": rejected[:8],
    }


def _search_window_best(
    *,
    base_url: str,
    endpoint: str,
    base_payload_template: dict[str, Any],
    train_start: dt.date,
    train_end: dt.date,
    param_space: list[dict[str, Any]],
    workers: int,
    max_in_flight: int,
    timeout: float,
    retry_times: int,
    objective_min_avg_annual_trade_count: float,
    objective_min_annualized_return: float,
    lookback_stability_gap_quantile: float,
    lookback_stability_max_gap_override: float | None = None,
) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload_template)
    payload["start"] = _to_yyyymmdd(train_start)
    payload["end"] = _to_yyyymmdd(train_end)
    t0 = time.perf_counter()
    rows = base._run_param_space_bounded(
        base_url=base_url,
        endpoint=endpoint,
        base_payload=payload,
        param_space=param_space,
        workers=int(workers),
        max_in_flight=int(max_in_flight),
        timeout=float(timeout),
        retry_times=int(retry_times),
    )
    metric_bounds = base._assign_composite_scores(
        rows,
        min_avg_annual_trade_count=float(objective_min_avg_annual_trade_count),
        min_annualized_return=float(objective_min_annualized_return),
    )
    base._sort_rows_for_display(rows)
    best, stable_diag = _pick_best_stable_by_lookback(
        rows,
        lookback_stability_gap_quantile=float(lookback_stability_gap_quantile),
        lookback_stability_max_gap_override=_to_float(
            lookback_stability_max_gap_override
        ),
    )
    selection_mode = (stable_diag or {}).get("selection_mode") or "normal"
    if best is None:
        # Relaxed fallback keeps pipeline alive while retaining sqn/min-metric checks.
        _ = base._assign_composite_scores(
            rows,
            min_avg_annual_trade_count=0.0,
            min_annualized_return=0.0,
        )
        best, stable_diag = _pick_best_stable_by_lookback(
            rows,
            lookback_stability_gap_quantile=float(lookback_stability_gap_quantile),
            lookback_stability_max_gap_override=_to_float(
                lookback_stability_max_gap_override
            ),
        )
        sel = (stable_diag or {}).get("selection_mode") or "unknown"
        selection_mode = f"relaxed_no_trade_return_gate/{sel}"
    if best is None:
        raise RuntimeError(
            "no valid parameter found in training window "
            f"{_to_yyyymmdd(train_start)}~{_to_yyyymmdd(train_end)}"
        )
    errors = [x for x in rows if str(x.get("status") or "") != "ok"]
    ineligible = [
        x
        for x in rows
        if str(x.get("status") or "") == "ok" and not bool(x.get("objective_eligible"))
    ]
    top_rows = sorted(
        [x for x in rows if str(x.get("status") or "") == "ok"],
        key=lambda z: (
            float(base._to_float(z.get("composite_score")) or -1e18),
            float(base._metric_value(z, "sharpe_ratio") or -1e18),
            float(base._metric_value(z, "annualized_return") or -1e18),
        ),
        reverse=True,
    )[:10]
    return {
        "best": best,
        "rows": rows,
        "metric_bounds": metric_bounds,
        "selection_mode": selection_mode,
        "selection_diagnostics": stable_diag or {},
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "total_cases": len(rows),
        "success_cases": len(rows) - len(errors),
        "error_cases": len(errors),
        "ineligible_cases": len(ineligible),
        "top10": top_rows,
    }


def _execute_oos_year(
    *,
    base_url: str,
    endpoint: str,
    base_payload_template: dict[str, Any],
    best_row: dict[str, Any],
    trade_start: dt.date,
    trade_end: dt.date,
    timeout: float,
) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload_template)
    payload["start"] = _to_yyyymmdd(trade_start)
    payload["end"] = _to_yyyymmdd(trade_end)
    payload["mom_lookback"] = int(best_row.get("mom_lookback"))
    payload["tsmom_entry_threshold"] = float(best_row.get("tsmom_entry_threshold"))
    payload["tsmom_exit_threshold"] = float(best_row.get("tsmom_exit_threshold"))
    t0 = time.perf_counter()
    resp = _call_trend_api(
        base_url=base_url,
        endpoint=endpoint,
        payload=payload,
        timeout=float(timeout),
    )
    metrics = base._extract_metrics(resp)
    nav_block = resp.get("nav") if isinstance(resp, dict) else {}
    nav_dates = nav_block.get("dates", []) if isinstance(nav_block, dict) else []
    nav_series = nav_block.get("series", {}) if isinstance(nav_block, dict) else {}
    strat_nav = nav_series.get("STRAT", []) if isinstance(nav_series, dict) else []
    if not isinstance(nav_dates, list) or not isinstance(strat_nav, list):
        raise RuntimeError("invalid nav schema")
    if len(nav_dates) != len(strat_nav):
        raise RuntimeError("nav dates/values length mismatch")
    trades = (
        (((resp.get("trade_statistics") or {}).get("trades")) or [])
        if isinstance(resp, dict)
        else []
    )
    if not isinstance(trades, list):
        trades = []
    return {
        "response": resp,
        "metrics": metrics,
        "nav_dates": nav_dates,
        "nav_values": strat_nav,
        "trades": trades,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000.0),
    }


def _stitch_oos_nav(
    year_runs: list[dict[str, Any]],
) -> tuple[pd.Series, pd.Series]:
    out_dates: list[pd.Timestamp] = []
    out_nav: list[float] = []
    running_nav = 1.0
    for yr in year_runs:
        dates = yr.get("nav_dates") or []
        vals = yr.get("nav_values") or []
        if len(dates) != len(vals):
            continue
        local_vals = [float(x) for x in vals if _to_float(x) is not None]
        if not local_vals:
            continue
        # Align by original positions to keep dates synced.
        local_series = []
        for ds, nv in zip(dates, vals):
            nvf = _to_float(nv)
            if nvf is None:
                continue
            local_series.append((pd.to_datetime(ds), float(nvf)))
        if not local_series:
            continue
        local_nav = pd.Series(
            [x[1] for x in local_series],
            index=pd.DatetimeIndex([x[0] for x in local_series]),
            dtype=float,
        ).sort_index()
        local_ret = (
            local_nav.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
        for d, r in local_ret.items():
            rr = float(r) if np.isfinite(float(r)) else 0.0
            running_nav = float(running_nav * (1.0 + rr))
            out_dates.append(pd.Timestamp(d))
            out_nav.append(float(running_nav))
    nav = pd.Series(out_nav, index=pd.DatetimeIndex(out_dates), dtype=float)
    nav = nav[~nav.index.duplicated(keep="last")].sort_index()
    daily_ret = nav.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return nav, daily_ret


def _overall_metrics(
    nav: pd.Series,
    daily_ret: pd.Series,
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    if nav.empty:
        raise RuntimeError("empty stitched oos nav")
    cum_ret = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    ann_ret = _annualized_return(nav)
    ann_vol = _annualized_vol(daily_ret)
    mdd = _max_drawdown(nav)
    mdd_recovery = _max_drawdown_duration_days(nav)
    sharpe = _sharpe(daily_ret)
    sortino = _sortino(daily_ret)
    calmar = _safe_div(ann_ret, abs(mdd) if _to_float(mdd) is not None else None)
    ui = _ulcer_index(nav)
    upi = _safe_div(ann_ret, (ui / 100.0) if _to_float(ui) is not None else None)
    sqn_obj = _sqn_recent_100(trades, min_trades=30)
    sqn_recent_100 = _to_float(sqn_obj.get("sqn"))

    w = 252
    roll_1y_ret = nav / nav.shift(w) - 1.0
    roll_1y_mdd = _rolling_max_drawdown(nav, w)
    roll_1y_sharpe = _rolling_sharpe(daily_ret, w)
    roll_1y_sortino = _rolling_sortino(daily_ret, w)
    roll_1y_ann_ret = (nav / nav.shift(w)) ** (TRADING_DAYS_PER_YEAR / float(w)) - 1.0
    roll_1y_calmar = roll_1y_ann_ret / roll_1y_mdd.abs().replace(0.0, np.nan)
    roll_1y_ui = _rolling_ulcer_index(nav, w)
    roll_1y_upi = roll_1y_ann_ret / (roll_1y_ui / 100.0).replace(0.0, np.nan)

    sqn_dates, sqn_vals = _rolling_sqn_recent_100_series(trades, min_trades=30)
    sqn_roll = pd.Series(
        [np.nan if v is None else float(v) for v in sqn_vals],
        index=pd.to_datetime(sqn_dates, errors="coerce"),
        dtype=float,
    )
    sqn_roll = sqn_roll[~sqn_roll.index.isna()]

    return {
        "cumulative_return": float(cum_ret),
        "annualized_return": _to_float(ann_ret),
        "annualized_volatility": _to_float(ann_vol),
        "max_drawdown": _to_float(mdd),
        "max_drawdown_recovery_days": int(mdd_recovery),
        "sharpe_ratio": _to_float(sharpe),
        "calmar_ratio": _to_float(calmar),
        "sortino_ratio": _to_float(sortino),
        "ulcer_index": _to_float(ui),
        "ulcer_performance_index": _to_float(upi),
        "sqn_recent_100": _to_float(sqn_recent_100),
        "sqn_recent_100_detail": sqn_obj,
        "rolling_1y_return": _window_summary(roll_1y_ret),
        "rolling_1y_drawdown": _window_summary(roll_1y_mdd),
        "rolling_1y_sharpe": _window_summary(roll_1y_sharpe),
        "rolling_1y_calmar": _window_summary(roll_1y_calmar),
        "rolling_1y_sortino": _window_summary(roll_1y_sortino),
        "rolling_1y_ulcer_index": _window_summary(roll_1y_ui),
        "rolling_1y_ulcer_performance_index": _window_summary(roll_1y_upi),
        "rolling_sqn_recent_100": _window_summary(sqn_roll),
        "rolling_series": {
            "dates": [d.strftime("%Y-%m-%d") for d in nav.index],
            "nav": [float(x) for x in nav.to_list()],
            "daily_return": [float(x) for x in daily_ret.to_list()],
            "rolling_1y_return": [
                None if not np.isfinite(float(x)) else float(x)
                for x in roll_1y_ret.to_list()
            ],
            "rolling_1y_drawdown": [
                None if not np.isfinite(float(x)) else float(x)
                for x in roll_1y_mdd.to_list()
            ],
            "rolling_1y_sharpe": [
                None if not np.isfinite(float(x)) else float(x)
                for x in roll_1y_sharpe.to_list()
            ],
            "rolling_1y_calmar": [
                None if not np.isfinite(float(x)) else float(x)
                for x in roll_1y_calmar.to_list()
            ],
            "rolling_1y_sortino": [
                None if not np.isfinite(float(x)) else float(x)
                for x in roll_1y_sortino.to_list()
            ],
            "rolling_1y_ulcer_index": [
                None if not np.isfinite(float(x)) else float(x)
                for x in roll_1y_ui.to_list()
            ],
            "rolling_1y_ulcer_performance_index": [
                None if not np.isfinite(float(x)) else float(x)
                for x in roll_1y_upi.to_list()
            ],
            "rolling_sqn_recent_100_dates": [
                d.strftime("%Y-%m-%d") for d in sqn_roll.index
            ],
            "rolling_sqn_recent_100_values": [
                None if not np.isfinite(float(x)) else float(x)
                for x in sqn_roll.to_list()
            ],
        },
    }


def _write_yearly_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "trade_year",
        "train_start",
        "train_end",
        "trade_start",
        "trade_end",
        "best_mom_lookback",
        "best_entry_pct",
        "best_exit_pct",
        "search_selection_mode",
        "search_stability_threshold",
        "search_stability_max_gap",
        "search_elapsed_seconds",
        "search_total_cases",
        "search_success_cases",
        "search_error_cases",
        "search_ineligible_cases",
        "oos_annualized_return",
        "oos_sharpe_ratio",
        "oos_calmar_ratio",
        "oos_sortino_ratio",
        "oos_ulcer_index",
        "oos_ulcer_performance_index",
        "oos_max_drawdown",
        "oos_trade_count_total",
        "oos_sqn_recent_100",
        "oos_elapsed_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "trade_year": r.get("trade_year"),
                    "train_start": r.get("train_start"),
                    "train_end": r.get("train_end"),
                    "trade_start": r.get("trade_start"),
                    "trade_end": r.get("trade_end"),
                    "best_mom_lookback": r.get("best_params", {}).get("mom_lookback"),
                    "best_entry_pct": r.get("best_params", {}).get(
                        "tsmom_entry_threshold_pct"
                    ),
                    "best_exit_pct": r.get("best_params", {}).get(
                        "tsmom_exit_threshold_pct"
                    ),
                    "search_selection_mode": r.get("search", {}).get("selection_mode"),
                    "search_stability_threshold": r.get("search", {})
                    .get("selection_diagnostics", {})
                    .get("stability_threshold"),
                    "search_stability_max_gap": r.get("search", {})
                    .get("selection_diagnostics", {})
                    .get("stability_max_gap"),
                    "search_elapsed_seconds": r.get("search", {}).get(
                        "elapsed_seconds"
                    ),
                    "search_total_cases": r.get("search", {}).get("total_cases"),
                    "search_success_cases": r.get("search", {}).get("success_cases"),
                    "search_error_cases": r.get("search", {}).get("error_cases"),
                    "search_ineligible_cases": r.get("search", {}).get(
                        "ineligible_cases"
                    ),
                    "oos_annualized_return": r.get("oos_metrics", {}).get(
                        "annualized_return"
                    ),
                    "oos_sharpe_ratio": r.get("oos_metrics", {}).get("sharpe_ratio"),
                    "oos_calmar_ratio": r.get("oos_metrics", {}).get("calmar_ratio"),
                    "oos_sortino_ratio": r.get("oos_metrics", {}).get("sortino_ratio"),
                    "oos_ulcer_index": r.get("oos_metrics", {}).get("ulcer_index"),
                    "oos_ulcer_performance_index": r.get("oos_metrics", {}).get(
                        "ulcer_performance_index"
                    ),
                    "oos_max_drawdown": r.get("oos_metrics", {}).get("max_drawdown"),
                    "oos_trade_count_total": r.get("oos_metrics", {}).get(
                        "sqn_trade_count_total"
                    ),
                    "oos_sqn_recent_100": r.get("oos_metrics", {}).get(
                        "sqn_recent_100"
                    ),
                    "oos_elapsed_ms": r.get("oos_elapsed_ms"),
                }
            )


def _write_search_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "trade_year",
        "rank",
        "mom_lookback",
        "tsmom_entry_threshold_pct",
        "tsmom_exit_threshold_pct",
        "composite_score",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "avg_annual_trade_count",
        "sqn_recent_100",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            year = r.get("trade_year")
            top = r.get("search", {}).get("top10", [])
            for i, x in enumerate(top, start=1):
                m = x.get("metrics") if isinstance(x, dict) else {}
                w.writerow(
                    {
                        "trade_year": year,
                        "rank": i,
                        "mom_lookback": x.get("mom_lookback"),
                        "tsmom_entry_threshold_pct": x.get("tsmom_entry_threshold_pct"),
                        "tsmom_exit_threshold_pct": x.get("tsmom_exit_threshold_pct"),
                        "composite_score": x.get("composite_score"),
                        "annualized_return": (m or {}).get("annualized_return"),
                        "sharpe_ratio": (m or {}).get("sharpe_ratio"),
                        "max_drawdown": (m or {}).get("max_drawdown"),
                        "avg_annual_trade_count": (m or {}).get(
                            "avg_annual_trade_count"
                        ),
                        "sqn_recent_100": (m or {}).get("sqn_recent_100"),
                    }
                )


def run(args: argparse.Namespace) -> int:
    start_date = _parse_yyyymmdd(str(args.start))
    end_date = _parse_yyyymmdd(str(args.end))
    if start_date >= end_date:
        raise ValueError("start must be earlier than end")
    if int(args.train_years) < 1:
        raise ValueError("train_years must be >= 1")

    base_payload_template = base._build_base_payload(base.RAW_BASE_PAYLOAD)
    slices = _build_year_slices(start_date, end_date, int(args.train_years))
    if not slices:
        raise ValueError("no trade-year slices generated; check date range/train_years")

    lookbacks = base._grid_int_values(
        start=int(args.lookback_start),
        end=int(args.lookback_end),
        step=int(args.lookback_step),
    )
    entry_pct_values = base._grid_decimal_values(
        start=float(args.entry_threshold_pct_start),
        end=float(args.entry_threshold_pct_end),
        step=float(args.entry_threshold_pct_step),
    )
    exit_pct_values = base._grid_decimal_values(
        start=float(args.exit_threshold_pct_start),
        end=float(args.exit_threshold_pct_end),
        step=float(args.exit_threshold_pct_step),
    )
    param_space = base._build_param_space(
        lookback_values=lookbacks,
        entry_threshold_pct_values=entry_pct_values,
        exit_threshold_pct_values=exit_pct_values,
        max_cases=None,
    )
    print(
        "[INFO] walk-forward setup: "
        f"years={len(slices)}, train_years={int(args.train_years)}, "
        f"grid={len(lookbacks)}x{len(entry_pct_values)}x{len(exit_pct_values)}={len(param_space)}, "
        f"workers={int(args.workers)}, max_in_flight={int(args.max_in_flight)}, timeout={float(args.timeout)}s"
    )
    all_year_rows: list[dict[str, Any]] = []
    all_oos_trades: list[dict[str, Any]] = []
    all_oos_nav_runs: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, sl in enumerate(slices, start=1):
        print(
            "[INFO] year step "
            f"{i}/{len(slices)}: train={_to_yyyymmdd(sl['train_start'])}~{_to_yyyymmdd(sl['train_end'])}, "
            f"trade={_to_yyyymmdd(sl['trade_start'])}~{_to_yyyymmdd(sl['trade_end'])}"
        )
        search_pack = _search_window_best(
            base_url=str(args.base_url),
            endpoint=str(args.endpoint),
            base_payload_template=base_payload_template,
            train_start=sl["train_start"],
            train_end=sl["train_end"],
            param_space=param_space,
            workers=int(args.workers),
            max_in_flight=int(args.max_in_flight),
            timeout=float(args.timeout),
            retry_times=int(args.retry_times),
            objective_min_avg_annual_trade_count=float(
                args.objective_min_avg_annual_trade_count
            ),
            objective_min_annualized_return=float(
                float(args.objective_min_annualized_return_pct) / 100.0
            ),
            lookback_stability_gap_quantile=float(args.lookback_stability_gap_quantile),
            lookback_stability_max_gap_override=_to_float(
                args.lookback_stability_max_gap_override
            ),
        )
        best_row = search_pack["best"]
        oos = _execute_oos_year(
            base_url=str(args.base_url),
            endpoint=str(args.endpoint),
            base_payload_template=base_payload_template,
            best_row=best_row,
            trade_start=sl["trade_start"],
            trade_end=sl["trade_end"],
            timeout=float(args.timeout),
        )
        trades = oos.get("trades") or []
        oos_sqn = _sqn_recent_100(trades, min_trades=30)
        year_row = {
            "trade_year": int(sl["trade_year"]),
            "train_start": _to_iso_date(sl["train_start"]),
            "train_end": _to_iso_date(sl["train_end"]),
            "trade_start": _to_iso_date(sl["trade_start"]),
            "trade_end": _to_iso_date(sl["trade_end"]),
            "best_params": {
                "mom_lookback": int(best_row.get("mom_lookback")),
                "tsmom_entry_threshold_pct": float(
                    best_row.get("tsmom_entry_threshold_pct")
                ),
                "tsmom_entry_threshold": float(best_row.get("tsmom_entry_threshold")),
                "tsmom_exit_threshold_pct": float(
                    best_row.get("tsmom_exit_threshold_pct")
                ),
                "tsmom_exit_threshold": float(best_row.get("tsmom_exit_threshold")),
                "composite_score": _to_float(best_row.get("composite_score")),
            },
            "search": {
                "selection_mode": search_pack.get("selection_mode"),
                "selection_diagnostics": search_pack.get("selection_diagnostics") or {},
                "elapsed_seconds": search_pack.get("elapsed_seconds"),
                "total_cases": search_pack.get("total_cases"),
                "success_cases": search_pack.get("success_cases"),
                "error_cases": search_pack.get("error_cases"),
                "ineligible_cases": search_pack.get("ineligible_cases"),
                "best_metrics": best_row.get("metrics")
                if isinstance(best_row, dict)
                else {},
                "top10": search_pack.get("top10", []),
            },
            "oos_elapsed_ms": oos.get("elapsed_ms"),
            "oos_metrics": {
                **(oos.get("metrics") or {}),
                "sqn_recent_100": _to_float(oos_sqn.get("sqn")),
                "sqn_recent_100_detail": oos_sqn,
            },
        }
        all_year_rows.append(year_row)
        for tr in trades:
            t2 = dict(tr) if isinstance(tr, dict) else {}
            t2["oos_trade_year"] = int(sl["trade_year"])
            all_oos_trades.append(t2)
        all_oos_nav_runs.append(
            {
                "trade_year": int(sl["trade_year"]),
                "nav_dates": oos.get("nav_dates") or [],
                "nav_values": oos.get("nav_values") or [],
            }
        )
        print(
            "[INFO] year result: "
            f"trade_year={sl['trade_year']}, best=({best_row.get('mom_lookback')}, "
            f"{best_row.get('tsmom_entry_threshold_pct')}%, {best_row.get('tsmom_exit_threshold_pct')}%), "
            f"pick_mode={search_pack.get('selection_mode')}, "
            f"stability_threshold={(search_pack.get('selection_diagnostics') or {}).get('stability_threshold')}, "
            f"stability_max_gap={(search_pack.get('selection_diagnostics') or {}).get('stability_max_gap')}, "
            f"oos_ann={_to_float((oos.get('metrics') or {}).get('annualized_return'))}, "
            f"oos_sharpe={_to_float((oos.get('metrics') or {}).get('sharpe_ratio'))}"
        )

    stitched_nav, stitched_ret = _stitch_oos_nav(all_oos_nav_runs)
    overall = _overall_metrics(stitched_nav, stitched_ret, all_oos_trades)

    payload = {
        "meta": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "description": (
                "Walk-forward momentum test: rolling 5-year optimisation then "
                "next-year execution, iterated yearly."
            ),
            "api_base_url": str(args.base_url),
            "api_endpoint": str(args.endpoint),
            "base_payload": base_payload_template,
            "global_range": {
                "start": _to_yyyymmdd(start_date),
                "end": _to_yyyymmdd(end_date),
            },
            "walkforward": {
                "train_years": int(args.train_years),
                "first_trade_year": int(slices[0]["trade_year"]),
                "last_trade_year": int(slices[-1]["trade_year"]),
                "steps": int(len(slices)),
            },
            "search_space": {
                "mom_lookback_start": int(args.lookback_start),
                "mom_lookback_end": int(args.lookback_end),
                "mom_lookback_step": int(args.lookback_step),
                "entry_threshold_pct_start": float(args.entry_threshold_pct_start),
                "entry_threshold_pct_end": float(args.entry_threshold_pct_end),
                "entry_threshold_pct_step": float(args.entry_threshold_pct_step),
                "exit_threshold_pct_start": float(args.exit_threshold_pct_start),
                "exit_threshold_pct_end": float(args.exit_threshold_pct_end),
                "exit_threshold_pct_step": float(args.exit_threshold_pct_step),
                "lookback_values": lookbacks,
                "entry_threshold_pct_values": entry_pct_values,
                "exit_threshold_pct_values": exit_pct_values,
                "cases_per_window": len(param_space),
            },
            "objective_constraints": {
                "min_avg_annual_trade_count": float(
                    args.objective_min_avg_annual_trade_count
                ),
                "min_annualized_return_pct": float(
                    args.objective_min_annualized_return_pct
                ),
                "lookback_stability_gap_quantile": float(
                    args.lookback_stability_gap_quantile
                ),
                "lookback_stability_max_gap_override": _to_float(
                    args.lookback_stability_max_gap_override
                ),
            },
            "concurrency": {
                "workers": int(args.workers),
                "max_in_flight": int(args.max_in_flight),
                "retry_times": int(args.retry_times),
                "timeout_per_case_seconds": float(args.timeout),
            },
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        },
        "yearly_iterations": all_year_rows,
        "overall": overall,
    }

    output_json = Path(str(args.output_json))
    output_yearly_csv = Path(str(args.output_yearly_csv))
    output_search_csv = Path(str(args.output_search_csv))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_yearly_csv(output_yearly_csv, all_year_rows)
    _write_search_summary_csv(output_search_csv, all_year_rows)

    print(f"[INFO] done in {time.perf_counter() - t0:.1f}s")
    print(f"[INFO] json: {output_json}")
    print(f"[INFO] yearly csv: {output_yearly_csv}")
    print(f"[INFO] search csv: {output_search_csv}")
    print(
        "[INFO] overall: "
        f"cum={overall.get('cumulative_return')}, ann={overall.get('annualized_return')}, "
        f"sharpe={overall.get('sharpe_ratio')}, mdd={overall.get('max_drawdown')}, "
        f"sqn_recent_100={overall.get('sqn_recent_100')}"
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Walk-forward test: rolling 5-year parameter search, then next-year execution."
        )
    )
    p.add_argument("--base-url", default=base.DEFAULT_BASE_URL)
    p.add_argument("--endpoint", default=base.DEFAULT_ENDPOINT)
    p.add_argument("--start", default="20120101")
    p.add_argument("--end", default="20260720")
    p.add_argument("--train-years", type=int, default=5)
    p.add_argument("--lookback-start", type=int, default=15)
    p.add_argument("--lookback-end", type=int, default=30)
    p.add_argument("--lookback-step", type=int, default=1)
    p.add_argument("--entry-threshold-pct-start", type=float, default=0.0)
    p.add_argument("--entry-threshold-pct-end", type=float, default=5.0)
    p.add_argument("--entry-threshold-pct-step", type=float, default=1.0)
    p.add_argument("--exit-threshold-pct-start", type=float, default=-5.0)
    p.add_argument("--exit-threshold-pct-end", type=float, default=0.0)
    p.add_argument("--exit-threshold-pct-step", type=float, default=1.0)
    p.add_argument(
        "--objective-min-avg-annual-trade-count",
        type=float,
        default=50.0,
        help="Eligibility gate for selecting best params in each train window.",
    )
    p.add_argument(
        "--objective-min-annualized-return-pct",
        type=float,
        default=5.0,
        help="Eligibility gate for selecting best params in each train window.",
    )
    p.add_argument(
        "--lookback-stability-gap-quantile",
        type=float,
        default=DEFAULT_LOOKBACK_STABILITY_GAP_QUANTILE,
        help=(
            "Adaptive threshold quantile in [0,1] computed from each train "
            "window's lookback-neighbor max-gap distribution."
        ),
    )
    p.add_argument(
        "--lookback-stability-max-gap-override",
        type=float,
        default=None,
        help=(
            "Optional hard threshold override for stability gap. "
            "If omitted, threshold is fully adaptive per window."
        ),
    )
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--max-in-flight", type=int, default=96)
    p.add_argument("--retry-times", type=int, default=1)
    p.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per parameter-combination total timeout in seconds.",
    )
    p.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    p.add_argument("--output-yearly-csv", default=DEFAULT_OUTPUT_YEARLY_CSV)
    p.add_argument("--output-search-csv", default=DEFAULT_OUTPUT_SEARCH_CSV)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if int(args.workers) < 1:
        raise ValueError("workers must be >= 1")
    if int(args.max_in_flight) < 1:
        raise ValueError("max-in-flight must be >= 1")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
