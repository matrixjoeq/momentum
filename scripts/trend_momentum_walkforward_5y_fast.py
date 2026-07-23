#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
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
import trend_momentum_walkforward_5y_test as wf

DEFAULT_OUTPUT_JSON = (
    "src/etf_momentum/web/data/trend_momentum_walkforward_5y_fast_results.json"
)
DEFAULT_OUTPUT_YEARLY_CSV = (
    "src/etf_momentum/web/data/trend_momentum_walkforward_5y_fast_yearly.csv"
)
DEFAULT_LOOKBACK_STABILITY_GAP_QUANTILE = 0.5


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


def _run_single_case_full(
    *,
    base_url: str,
    endpoint: str,
    base_payload: dict[str, Any],
    case: dict[str, Any],
    timeout: float,
    retry_times: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    payload = copy.deepcopy(base_payload)
    payload["mom_lookback"] = int(case["mom_lookback"])
    payload["tsmom_entry_threshold"] = float(case["tsmom_entry_threshold"])
    payload["tsmom_exit_threshold"] = float(case["tsmom_exit_threshold"])
    url = base._join_url(base_url, endpoint)
    attempts = max(1, int(retry_times) + 1)
    deadline = t0 + float(timeout)
    last_error = ""
    for attempt_idx in range(attempts):
        remain = deadline - time.perf_counter()
        if remain <= 0:
            last_error = (
                f"timed out (case budget {timeout:.1f}s exceeded before attempt "
                f"{attempt_idx + 1})"
            )
            break
        try:
            resp = base._http_post_json(
                url=url, payload=payload, timeout=max(1.0, float(remain))
            )
            if not isinstance(resp, dict):
                raise RuntimeError("unexpected response schema")
            metrics = base._extract_metrics(resp)
            nav_block = resp.get("nav") if isinstance(resp, dict) else {}
            nav_dates = (
                nav_block.get("dates", []) if isinstance(nav_block, dict) else []
            )
            nav_series = (
                nav_block.get("series", {}) if isinstance(nav_block, dict) else {}
            )
            nav_values = (
                nav_series.get("STRAT", []) if isinstance(nav_series, dict) else []
            )
            trades = (
                (((resp.get("trade_statistics") or {}).get("trades")) or [])
                if isinstance(resp, dict)
                else []
            )
            if not isinstance(nav_dates, list) or not isinstance(nav_values, list):
                raise RuntimeError("invalid nav schema")
            if len(nav_dates) != len(nav_values):
                raise RuntimeError("nav dates/values length mismatch")
            if not isinstance(trades, list):
                trades = []
            return {
                "mom_lookback": int(case["mom_lookback"]),
                "tsmom_entry_threshold_pct": float(case["tsmom_entry_threshold_pct"]),
                "tsmom_entry_threshold": float(case["tsmom_entry_threshold"]),
                "tsmom_exit_threshold_pct": float(case["tsmom_exit_threshold_pct"]),
                "tsmom_exit_threshold": float(case["tsmom_exit_threshold"]),
                "status": "ok",
                "elapsed_ms": int((time.perf_counter() - t0) * 1000.0),
                "attempts": int(attempt_idx + 1),
                "metrics_full_range": metrics,
                "nav_dates": nav_dates,
                "nav_values": nav_values,
                "trades": trades,
                "error": None,
            }
        except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            last_error = str(e)
            if attempt_idx + 1 >= attempts:
                break
            remain_after_error = deadline - time.perf_counter()
            if remain_after_error <= 0:
                last_error = f"timed out (case budget {timeout:.1f}s exceeded)"
                break
            time.sleep(min(1.0, 0.2 * (attempt_idx + 1), remain_after_error))
    return {
        "mom_lookback": int(case["mom_lookback"]),
        "tsmom_entry_threshold_pct": float(case["tsmom_entry_threshold_pct"]),
        "tsmom_entry_threshold": float(case["tsmom_entry_threshold"]),
        "tsmom_exit_threshold_pct": float(case["tsmom_exit_threshold_pct"]),
        "tsmom_exit_threshold": float(case["tsmom_exit_threshold"]),
        "status": "error",
        "elapsed_ms": int((time.perf_counter() - t0) * 1000.0),
        "attempts": int(attempts),
        "metrics_full_range": {},
        "nav_dates": [],
        "nav_values": [],
        "trades": [],
        "error": last_error or "unknown_error",
    }


def _run_cases_bounded(
    *,
    base_url: str,
    endpoint: str,
    base_payload: dict[str, Any],
    param_space: list[dict[str, Any]],
    workers: int,
    max_in_flight: int,
    timeout: float,
    retry_times: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers)) as ex:
        total = len(param_space)
        pending: dict[concurrent.futures.Future, dict[str, Any]] = {}
        idx = 0

        def _submit_one() -> bool:
            nonlocal idx
            if idx >= total:
                return False
            case = param_space[idx]
            idx += 1
            fut = ex.submit(
                _run_single_case_full,
                base_url=base_url,
                endpoint=endpoint,
                base_payload=base_payload,
                case=case,
                timeout=timeout,
                retry_times=retry_times,
            )
            pending[fut] = case
            return True

        while len(pending) < max_in_flight and _submit_one():
            pass
        done = 0
        while pending:
            done_set, _ = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done_set:
                pending.pop(fut, None)
                rows.append(fut.result())
                done += 1
            while len(pending) < max_in_flight and _submit_one():
                pass
            if done % 25 == 0 or done == total:
                ok = sum(1 for x in rows if str(x.get("status") or "") == "ok")
                print(
                    f"[INFO] precompute progress={done}/{total}, success={ok}, "
                    f"in_flight={len(pending)}"
                )
    return rows


def _series_from_row(row: dict[str, Any]) -> pd.Series:
    dates = row.get("nav_dates") or []
    vals = row.get("nav_values") or []
    out_pairs: list[tuple[pd.Timestamp, float]] = []
    for d, v in zip(dates, vals):
        ts = pd.to_datetime(d, errors="coerce")
        fv = _to_float(v)
        if pd.isna(ts) or fv is None:
            continue
        out_pairs.append((pd.Timestamp(ts), float(fv)))
    if not out_pairs:
        return pd.Series(dtype=float)
    s = pd.Series(
        [x[1] for x in out_pairs],
        index=pd.DatetimeIndex([x[0] for x in out_pairs]),
        dtype=float,
    )
    return s[~s.index.duplicated(keep="last")].sort_index()


def _slice_nav(nav: pd.Series, start_d: dt.date, end_d: dt.date) -> pd.Series:
    if nav.empty:
        return nav
    return nav.loc[(nav.index.date >= start_d) & (nav.index.date <= end_d)].copy()


def _slice_trades(
    trades: list[dict[str, Any]], start_d: dt.date, end_d: dt.date
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tr in trades:
        exit_ts = pd.to_datetime(tr.get("exit_date"), errors="coerce")
        entry_ts = pd.to_datetime(tr.get("entry_date"), errors="coerce")
        ts = exit_ts if not pd.isna(exit_ts) else entry_ts
        if pd.isna(ts):
            continue
        d = pd.Timestamp(ts).date()
        if start_d <= d <= end_d:
            out.append(dict(tr))
    return out


def _window_metrics(
    nav: pd.Series,
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    if nav.empty or len(nav) < 2:
        return {
            "cumulative_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "max_drawdown": None,
            "max_drawdown_recovery_days": None,
            "sharpe_ratio": None,
            "calmar_ratio": None,
            "sortino_ratio": None,
            "ulcer_index": None,
            "ulcer_performance_index": None,
            "avg_annual_trade_count": None,
            "sqn_recent_100": None,
            "sqn_trade_count_total": float(len(trades)),
            "sqn_min_trades": 30.0,
            "sqn_reason": "empty_nav",
        }
    daily_ret = nav.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ann_ret = wf._annualized_return(nav)
    ann_vol = wf._annualized_vol(daily_ret)
    mdd = wf._max_drawdown(nav)
    sharpe = wf._sharpe(daily_ret)
    sortino = wf._sortino(daily_ret)
    ui = wf._ulcer_index(nav)
    calmar = None
    if _to_float(mdd) is not None and abs(float(mdd)) > 1e-12:
        calmar = float(ann_ret / abs(float(mdd)))
    upi = None
    if _to_float(ui) is not None and abs(float(ui)) > 1e-12:
        upi = float(ann_ret / (float(ui) / 100.0))
    years = max(1e-12, len(nav) / float(wf.TRADING_DAYS_PER_YEAR))
    avg_trade = float(len(trades) / years)
    sqn_obj = wf._sqn_recent_100(trades, min_trades=30)
    return {
        "cumulative_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
        "annualized_return": _to_float(ann_ret),
        "annualized_volatility": _to_float(ann_vol),
        "max_drawdown": _to_float(mdd),
        "max_drawdown_recovery_days": int(wf._max_drawdown_duration_days(nav)),
        "sharpe_ratio": _to_float(sharpe),
        "calmar_ratio": _to_float(calmar),
        "sortino_ratio": _to_float(sortino),
        "ulcer_index": _to_float(ui),
        "ulcer_performance_index": _to_float(upi),
        "avg_annual_trade_count": float(avg_trade),
        "sqn_recent_100": _to_float(sqn_obj.get("sqn")),
        "sqn_trade_count_total": float(sqn_obj.get("trade_count_total") or len(trades)),
        "sqn_min_trades": float(sqn_obj.get("min_trades") or 30.0),
        "sqn_reason": sqn_obj.get("reason"),
    }


def _assign_window_composite(
    rows: list[dict[str, Any]],
    *,
    min_avg_annual_trade_count: float,
    min_annualized_return: float,
) -> None:
    metrics_def = base.OBJECTIVE_METRICS
    groups = base.OBJECTIVE_GROUPS
    eligible: list[dict[str, Any]] = []
    for row in rows:
        row["window_composite_score"] = None
        row["window_objective_eligible"] = False
        row["window_objective_reason"] = None
        m = row.get("window_metrics") if isinstance(row, dict) else {}
        if not isinstance(m, dict):
            row["window_objective_reason"] = "window_metrics_missing"
            continue
        avg_trade = _to_float(m.get("avg_annual_trade_count"))
        if avg_trade is None or avg_trade < min_avg_annual_trade_count:
            row["window_objective_reason"] = (
                "avg_annual_trade_count_lt_min:"
                f"{avg_trade}<"
                f"{min_avg_annual_trade_count}"
            )
            continue
        ann_ret = _to_float(m.get("annualized_return"))
        if ann_ret is None or ann_ret < min_annualized_return:
            row["window_objective_reason"] = (
                f"annualized_return_lt_min:{ann_ret}<{min_annualized_return}"
            )
            continue
        sqn_total = _to_float(m.get("sqn_trade_count_total"))
        sqn_min = _to_float(m.get("sqn_min_trades")) or 30.0
        if sqn_total is None or sqn_total < sqn_min:
            row["window_objective_reason"] = (
                f"sqn_trade_count_lt_min:{sqn_total}<{sqn_min}"
            )
            continue
        missing = [mk for mk, _hb in metrics_def if _to_float(m.get(mk)) is None]
        if missing:
            row["window_objective_reason"] = f"missing_metrics:{','.join(missing)}"
            continue
        row["window_objective_eligible"] = True
        eligible.append(row)

    bounds: dict[str, tuple[float, float]] = {}
    for mk, _hb in metrics_def:
        vals = [
            float(_to_float(r["window_metrics"].get(mk)))
            for r in eligible
            if _to_float(r["window_metrics"].get(mk)) is not None
        ]
        if vals:
            bounds[mk] = (float(min(vals)), float(max(vals)))
    for row in eligible:
        m = row["window_metrics"]
        parts: dict[str, float] = {}
        for mk, hb in metrics_def:
            v = _to_float(m.get(mk))
            if v is None or mk not in bounds:
                continue
            lo, hi = bounds[mk]
            if abs(hi - lo) <= 1e-12:
                s = 0.5
            else:
                base_s = float((v - lo) / (hi - lo))
                s = base_s if hb else float(1.0 - base_s)
            parts[mk] = float(max(0.0, min(1.0, s)))
        ws = 0.0
        wsum = 0.0
        for _gname, g_metrics, gw in groups:
            vals = [parts[k] for k in g_metrics if k in parts]
            has_bounds = any(k in bounds for k in g_metrics)
            if vals:
                g = float(sum(vals) / len(vals))
            elif has_bounds:
                g = 0.0
            else:
                continue
            ws += g * float(gw)
            wsum += float(gw)
        row["window_composite_score"] = float(ws / wsum) if wsum > 0.0 else None
        if _to_float(row["window_composite_score"]) is None:
            row["window_objective_eligible"] = False
            row["window_objective_reason"] = "window_composite_unavailable"


def _pick_window_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: tuple[tuple[float, float, float, float], dict[str, Any]] | None = None
    for row in rows:
        if not bool(row.get("window_objective_eligible")):
            continue
        comp = _to_float(row.get("window_composite_score"))
        m = row.get("window_metrics") if isinstance(row, dict) else {}
        if comp is None or not isinstance(m, dict):
            continue
        sharpe = _to_float(m.get("sharpe_ratio"))
        ann = _to_float(m.get("annualized_return"))
        mdd = _to_float(m.get("max_drawdown"))
        if sharpe is None:
            continue
        key = (
            float(comp),
            float(sharpe),
            float(ann if ann is not None else -1e18),
            float(-abs(mdd) if mdd is not None else -1e18),
        )
        if best is None or key > best[0]:
            best = (key, row)
    return best[1] if best else None


def _window_row_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    comp = _to_float(row.get("window_composite_score"))
    m = row.get("window_metrics") if isinstance(row, dict) else {}
    sharpe = _to_float((m or {}).get("sharpe_ratio")) if isinstance(m, dict) else None
    ann = _to_float((m or {}).get("annualized_return")) if isinstance(m, dict) else None
    mdd = _to_float((m or {}).get("max_drawdown")) if isinstance(m, dict) else None
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
        if not bool(row.get("window_objective_eligible")):
            continue
        lb = row.get("mom_lookback")
        try:
            lb_i = int(lb)
        except (TypeError, ValueError):
            continue
        cur = rep.get(lb_i)
        if cur is None or _window_row_rank_key(row) > _window_row_rank_key(cur):
            rep[lb_i] = row
    return rep


def _pick_window_best_stable(
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

    candidates = sorted(
        rep.items(),
        key=lambda kv: _window_row_rank_key(kv[1]),
        reverse=True,
    )

    gap_samples: list[float] = []
    for lb, row in candidates:
        left = rep.get(lb - 1)
        right = rep.get(lb + 1)
        if left is None or right is None:
            continue
        comp = _to_float(row.get("window_composite_score"))
        left_comp = _to_float((left or {}).get("window_composite_score"))
        right_comp = _to_float((right or {}).get("window_composite_score"))
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

    relaxed_best: tuple[float, float, int, dict[str, Any], dict[str, Any]] | None = None
    rejected: list[dict[str, Any]] = []
    for rank_idx, (lb, row) in enumerate(candidates, start=1):
        left = rep.get(lb - 1)
        right = rep.get(lb + 1)
        comp = _to_float(row.get("window_composite_score"))
        left_comp = _to_float((left or {}).get("window_composite_score"))
        right_comp = _to_float((right or {}).get("window_composite_score"))
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
        # relaxed fallback: choose the "least peak" candidate first, then better score.
        relaxed_key = (
            float(max_gap),
            float(-float(comp)),
        )
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
            relaxed_best = (
                float(max_gap),
                float(-float(comp)),
                int(rank_idx),
                row,
                relaxed_diag,
            )

    if relaxed_best is not None:
        return relaxed_best[3], relaxed_best[4]

    # Last resort: if every top candidate sits on boundary, keep the strongest representative.
    lb0, row0 = candidates[0]
    comp0 = _to_float(row0.get("window_composite_score"))
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
        "best_train_composite",
        "best_train_selection_mode",
        "best_train_stability_threshold",
        "best_train_stability_max_gap",
        "best_train_annualized_return",
        "best_train_sharpe_ratio",
        "oos_annualized_return",
        "oos_sharpe_ratio",
        "oos_calmar_ratio",
        "oos_sortino_ratio",
        "oos_ulcer_index",
        "oos_ulcer_performance_index",
        "oos_max_drawdown",
        "oos_trade_count_total",
        "oos_sqn_recent_100",
        "search_eligible_cases",
        "search_total_cases",
        "search_error_cases",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            bt = r.get("best_train", {})
            tm = bt.get("window_metrics", {}) if isinstance(bt, dict) else {}
            om = r.get("oos_metrics", {})
            w.writerow(
                {
                    "trade_year": r.get("trade_year"),
                    "train_start": r.get("train_start"),
                    "train_end": r.get("train_end"),
                    "trade_start": r.get("trade_start"),
                    "trade_end": r.get("trade_end"),
                    "best_mom_lookback": bt.get("mom_lookback"),
                    "best_entry_pct": bt.get("tsmom_entry_threshold_pct"),
                    "best_exit_pct": bt.get("tsmom_exit_threshold_pct"),
                    "best_train_composite": bt.get("window_composite_score"),
                    "best_train_selection_mode": bt.get("selection_mode"),
                    "best_train_stability_threshold": bt.get("stability_threshold"),
                    "best_train_stability_max_gap": bt.get("stability_max_gap"),
                    "best_train_annualized_return": (tm or {}).get("annualized_return"),
                    "best_train_sharpe_ratio": (tm or {}).get("sharpe_ratio"),
                    "oos_annualized_return": (om or {}).get("annualized_return"),
                    "oos_sharpe_ratio": (om or {}).get("sharpe_ratio"),
                    "oos_calmar_ratio": (om or {}).get("calmar_ratio"),
                    "oos_sortino_ratio": (om or {}).get("sortino_ratio"),
                    "oos_ulcer_index": (om or {}).get("ulcer_index"),
                    "oos_ulcer_performance_index": (om or {}).get(
                        "ulcer_performance_index"
                    ),
                    "oos_max_drawdown": (om or {}).get("max_drawdown"),
                    "oos_trade_count_total": (om or {}).get("sqn_trade_count_total"),
                    "oos_sqn_recent_100": (om or {}).get("sqn_recent_100"),
                    "search_eligible_cases": r.get("search_eligible_cases"),
                    "search_total_cases": r.get("search_total_cases"),
                    "search_error_cases": r.get("search_error_cases"),
                }
            )


def run(args: argparse.Namespace) -> int:
    start_date = _parse_yyyymmdd(str(args.start))
    end_date = _parse_yyyymmdd(str(args.end))
    if start_date >= end_date:
        raise ValueError("start must be earlier than end")
    train_years = int(args.train_years)
    slices = wf._build_year_slices(start_date, end_date, train_years)
    if not slices:
        raise ValueError("no trade-year slices generated")

    lookbacks = base._grid_int_values(
        start=int(args.lookback_start),
        end=int(args.lookback_end),
        step=int(args.lookback_step),
    )
    entry_vals = base._grid_decimal_values(
        start=float(args.entry_threshold_pct_start),
        end=float(args.entry_threshold_pct_end),
        step=float(args.entry_threshold_pct_step),
    )
    exit_vals = base._grid_decimal_values(
        start=float(args.exit_threshold_pct_start),
        end=float(args.exit_threshold_pct_end),
        step=float(args.exit_threshold_pct_step),
    )
    param_space = base._build_param_space(
        lookback_values=lookbacks,
        entry_threshold_pct_values=entry_vals,
        exit_threshold_pct_values=exit_vals,
        max_cases=None,
    )
    full_payload = base._build_base_payload(base.RAW_BASE_PAYLOAD)
    full_payload["start"] = _to_yyyymmdd(start_date)
    full_payload["end"] = _to_yyyymmdd(end_date)

    print(
        "[INFO] fast walk-forward setup: "
        f"years={len(slices)}, grid={len(param_space)}, "
        f"global_range={_to_yyyymmdd(start_date)}~{_to_yyyymmdd(end_date)}, "
        f"workers={int(args.workers)}"
    )
    t0 = time.perf_counter()
    pre_rows = _run_cases_bounded(
        base_url=str(args.base_url),
        endpoint=str(args.endpoint),
        base_payload=full_payload,
        param_space=param_space,
        workers=int(args.workers),
        max_in_flight=int(args.max_in_flight),
        timeout=float(args.timeout),
        retry_times=int(args.retry_times),
    )
    pre_ok = [x for x in pre_rows if str(x.get("status") or "") == "ok"]
    pre_err = [x for x in pre_rows if str(x.get("status") or "") != "ok"]
    print(
        f"[INFO] precompute done: total={len(pre_rows)}, ok={len(pre_ok)}, err={len(pre_err)}"
    )

    oos_runs: list[dict[str, Any]] = []
    oos_trades: list[dict[str, Any]] = []
    yearly: list[dict[str, Any]] = []
    for i, sl in enumerate(slices, start=1):
        print(
            "[INFO] select year "
            f"{i}/{len(slices)}: train={_to_yyyymmdd(sl['train_start'])}~{_to_yyyymmdd(sl['train_end'])}, "
            f"trade={_to_yyyymmdd(sl['trade_start'])}~{_to_yyyymmdd(sl['trade_end'])}"
        )
        window_rows: list[dict[str, Any]] = []
        for row in pre_ok:
            nav = _series_from_row(row)
            nav_win = _slice_nav(nav, sl["train_start"], sl["train_end"])
            trades_win = _slice_trades(
                row.get("trades") or [], sl["train_start"], sl["train_end"]
            )
            row2 = dict(row)
            row2["window_metrics"] = _window_metrics(nav_win, trades_win)
            window_rows.append(row2)
        _assign_window_composite(
            window_rows,
            min_avg_annual_trade_count=float(args.objective_min_avg_annual_trade_count),
            min_annualized_return=float(args.objective_min_annualized_return_pct)
            / 100.0,
        )
        best, selection_diag = _pick_window_best_stable(
            window_rows,
            lookback_stability_gap_quantile=float(args.lookback_stability_gap_quantile),
            lookback_stability_max_gap_override=_to_float(
                args.lookback_stability_max_gap_override
            ),
        )
        if best is None:
            raise RuntimeError(
                "no eligible best in window "
                f"{_to_yyyymmdd(sl['train_start'])}~{_to_yyyymmdd(sl['train_end'])}"
            )
        # OOS execution uses exact year-only backtest with selected params.
        oos = wf._execute_oos_year(
            base_url=str(args.base_url),
            endpoint=str(args.endpoint),
            base_payload_template=full_payload,
            best_row=best,
            trade_start=sl["trade_start"],
            trade_end=sl["trade_end"],
            timeout=float(args.timeout),
        )
        sqn_obj = wf._sqn_recent_100(oos.get("trades") or [], min_trades=30)
        oos_metrics = {
            **(oos.get("metrics") or {}),
            "sqn_recent_100": _to_float(sqn_obj.get("sqn")),
            "sqn_recent_100_detail": sqn_obj,
        }
        yearly_row = {
            "trade_year": int(sl["trade_year"]),
            "train_start": sl["train_start"].isoformat(),
            "train_end": sl["train_end"].isoformat(),
            "trade_start": sl["trade_start"].isoformat(),
            "trade_end": sl["trade_end"].isoformat(),
            "best_train": {
                "mom_lookback": int(best.get("mom_lookback")),
                "tsmom_entry_threshold_pct": float(
                    best.get("tsmom_entry_threshold_pct")
                ),
                "tsmom_entry_threshold": float(best.get("tsmom_entry_threshold")),
                "tsmom_exit_threshold_pct": float(best.get("tsmom_exit_threshold_pct")),
                "tsmom_exit_threshold": float(best.get("tsmom_exit_threshold")),
                "window_composite_score": _to_float(best.get("window_composite_score")),
                "selection_mode": (selection_diag or {}).get("selection_mode"),
                "stability_threshold": (selection_diag or {}).get(
                    "stability_threshold"
                ),
                "stability_max_gap": (selection_diag or {}).get("stability_max_gap"),
                "window_metrics": best.get("window_metrics") or {},
            },
            "search_total_cases": len(window_rows),
            "search_eligible_cases": int(
                sum(1 for x in window_rows if bool(x.get("window_objective_eligible")))
            ),
            "search_error_cases": int(len(pre_err)),
            "selection_diagnostics": selection_diag or {},
            "oos_elapsed_ms": oos.get("elapsed_ms"),
            "oos_metrics": oos_metrics,
        }
        yearly.append(yearly_row)
        for tr in oos.get("trades") or []:
            t2 = dict(tr) if isinstance(tr, dict) else {}
            t2["oos_trade_year"] = int(sl["trade_year"])
            oos_trades.append(t2)
        oos_runs.append(
            {
                "trade_year": int(sl["trade_year"]),
                "nav_dates": oos.get("nav_dates") or [],
                "nav_values": oos.get("nav_values") or [],
            }
        )
        print(
            "[INFO] year result: "
            f"trade_year={sl['trade_year']}, "
            f"best=({best.get('mom_lookback')}, {best.get('tsmom_entry_threshold_pct')}%, {best.get('tsmom_exit_threshold_pct')}%), "
            f"pick_mode={(selection_diag or {}).get('selection_mode')}, "
            f"stability_threshold={(selection_diag or {}).get('stability_threshold')}, "
            f"stability_max_gap={(selection_diag or {}).get('stability_max_gap')}, "
            f"oos_ann={_to_float(oos_metrics.get('annualized_return'))}, "
            f"oos_sharpe={_to_float(oos_metrics.get('sharpe_ratio'))}"
        )

    nav, daily_ret = wf._stitch_oos_nav(oos_runs)
    overall = wf._overall_metrics(nav, daily_ret, oos_trades)

    out_json = Path(str(args.output_json))
    out_year_csv = Path(str(args.output_yearly_csv))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "mode": "fast_precompute_slice",
            "assumption": (
                "Optimisation windows are evaluated by slicing each parameter's "
                "precomputed full-range trajectory; yearly OOS execution is still "
                "run as exact year-only backtests with selected params."
            ),
            "api_base_url": str(args.base_url),
            "api_endpoint": str(args.endpoint),
            "global_range": {
                "start": _to_yyyymmdd(start_date),
                "end": _to_yyyymmdd(end_date),
            },
            "train_years": int(train_years),
            "search_space": {
                "lookback_values": lookbacks,
                "entry_threshold_pct_values": entry_vals,
                "exit_threshold_pct_values": exit_vals,
                "cases": len(param_space),
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
            "precompute": {
                "ok_cases": len(pre_ok),
                "error_cases": len(pre_err),
            },
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        },
        "yearly_iterations": yearly,
        "overall": overall,
        "precompute_errors": pre_err,
    }
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_yearly_csv(out_year_csv, yearly)
    print(f"[INFO] done in {time.perf_counter() - t0:.1f}s")
    print(f"[INFO] json: {out_json}")
    print(f"[INFO] yearly csv: {out_year_csv}")
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
            "Fast walk-forward: precompute param trajectories once, then roll 5y optimise by slicing."
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
    p.add_argument("--objective-min-avg-annual-trade-count", type=float, default=50.0)
    p.add_argument("--objective-min-annualized-return-pct", type=float, default=5.0)
    p.add_argument(
        "--lookback-stability-gap-quantile",
        type=float,
        default=DEFAULT_LOOKBACK_STABILITY_GAP_QUANTILE,
        help=(
            "Adaptive threshold quantile in [0,1] computed from each training "
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
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--max-in-flight", type=int, default=80)
    p.add_argument("--retry-times", type=int, default=1)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    p.add_argument("--output-yearly-csv", default=DEFAULT_OUTPUT_YEARLY_CSV)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if int(args.workers) < 1:
        raise ValueError("workers must be >=1")
    if int(args.max_in_flight) < 1:
        raise ValueError("max-in-flight must be >=1")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
