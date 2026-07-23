#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import datetime as dt
import json
import math
import sys
import time
import urllib.error
import urllib.request
from decimal import Decimal
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_ENDPOINT = "/api/analysis/trend/portfolio"
DEFAULT_OUTPUT_JSON = (
    "src/etf_momentum/web/data/trend_er_entry_param_search_results.json"
)
OBJECTIVE_METRICS: tuple[tuple[str, bool], ...] = (
    ("sharpe_ratio", True),
    ("calmar_ratio", True),
    ("sortino_ratio", True),
    ("ulcer_index", False),
    ("ulcer_performance_index", True),
    ("sqn_recent_100", True),
)
OBJECTIVE_GROUPS: tuple[tuple[str, tuple[str, ...], float], ...] = (
    (
        "risk_adjusted_return",
        ("sharpe_ratio", "calmar_ratio", "sortino_ratio", "ulcer_performance_index"),
        0.45,
    ),
    ("pain_control", ("ulcer_index",), 0.25),
    ("system_quality", ("sqn_recent_100",), 0.30),
)

RAW_BASE_PAYLOAD: dict[str, Any] = {
    "codes": [
        "159570",
        "159792",
        "513070",
        "513090",
        "159928",
        "515220",
        "560280",
        "159870",
        "162411",
        "515880",
        "517520",
        "512480",
        "512010",
    ],
    "position_sizing": "risk_budget",
    "dynamic_universe": True,
    "start": "20120608",
    "end": "20260717",
    "initial_account_amount": None,
    "cost_bps": 2,
    "slippage_rate": 0.001,
    "capacity_window_years": 1,
    "quick_mode": True,
    "search_minimal_mode": True,
    "engine": "legacy",
    "exec_price": "close",
    "strategy": "tsmom",
    "sma_window": 20,
    "fast_window": 5,
    "slow_window": 20,
    "ma_type": "kama",
    "kama_er_window": 10,
    "kama_fast_window": 2,
    "kama_slow_window": 30,
    "kama_std_window": 20,
    "kama_std_coef": 1,
    "donchian_entry": 20,
    "donchian_exit": 10,
    "mom_lookback": 20,
    "tsmom_entry_threshold": 0.02,
    "tsmom_exit_threshold": 0,
    "impulse_entry_filter": False,
    "impulse_allow_bull": True,
    "impulse_allow_bear": False,
    "impulse_allow_neutral": False,
    "er_filter": True,
    "er_window": 10,
    "er_threshold": 0.1,
    "ma_entry_filter_enabled": False,
    "ma_entry_filter_type": "sma",
    "ma_entry_filter_fast": 100,
    "ma_entry_filter_slow": 200,
    "er_exit_filter": False,
    "er_exit_window": 10,
    "er_exit_threshold": 0.88,
    "bias_ma_window": 20,
    "bias_entry": 2,
    "bias_hot": 10,
    "bias_cold": -2,
    "bias_pos_mode": "binary",
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "macd_v_atr_window": 26,
    "macd_v_scale": 100,
    "macd_hist_min": 0,
    "macd_v_hist_min": 0,
    "random_hold_days": 20,
    "random_seed": 42,
    "atr_stop_mode": "static",
    "atr_stop_atr_basis": "latest",
    "atr_stop_reentry_mode": "wait_next_entry",
    "atr_stop_execution_mode": "intraday",
    "atr_stop_execution_time": "close",
    "atr_stop_window": 20,
    "atr_stop_n": 2,
    "atr_stop_m": 0.5,
    "r_take_profit_enabled": True,
    "r_take_profit_reentry_mode": "wait_next_entry",
    "r_take_profit_execution_mode": "intraday",
    "r_take_profit_execution_time": "close",
    "r_take_profit_tiers": [
        {"r_multiple": 1, "retrace_ratio": 1},
        {"r_multiple": 2, "retrace_ratio": 0.5},
        {"r_multiple": 3, "retrace_ratio": 0.33},
        {"r_multiple": 4, "retrace_ratio": 0.25},
        {"r_multiple": 5, "retrace_ratio": 0.2},
        {"r_multiple": 6, "retrace_ratio": 0.17},
        {"r_multiple": 7, "retrace_ratio": 0.14},
        {"r_multiple": 8, "retrace_ratio": 0.13},
        {"r_multiple": 9, "retrace_ratio": 0.11},
        {"r_multiple": 10, "retrace_ratio": 0.1},
    ],
    "r_profit_scaleout_enabled": False,
    "r_profit_scaleout_execution_mode": "intraday",
    "r_profit_scaleout_execution_time": "close",
    "r_profit_scaleout_breakeven_stop_enabled": True,
    "r_profit_scaleout_tiers": [
        {"r_multiple": 2, "reduce_fraction": 0.33},
    ],
    "bias_v_take_profit_enabled": False,
    "bias_v_take_profit_reentry_mode": "wait_next_entry",
    "bias_v_take_profit_execution_mode": "intraday",
    "bias_v_take_profit_execution_time": "close",
    "bias_v_take_profit_breakeven_stop_enabled": True,
    "bias_v_ma_window": 20,
    "bias_v_atr_window": 20,
    "bias_v_take_profit_tiers": [
        {"threshold": 3, "reduce_fraction": 0.33},
        {"threshold": 5, "reduce_fraction": 0.33},
    ],
    "ma_trailing_stop_enabled": False,
    "ma_trailing_stop_ma_type": "sma",
    "ma_trailing_stop_execution_mode": "intraday",
    "ma_trailing_stop_execution_time": "close",
    "ma_trailing_stop_effective_delay_days": 10,
    "ma_trailing_stop_reduce_window": 10,
    "ma_trailing_stop_exit_window": 20,
    "ma_trailing_stop_reduce_fraction": 0.5,
    "monthly_risk_budget_enabled": False,
    "monthly_risk_budget_pct": 6,
    "monthly_risk_budget_include_new_trade_risk": False,
    "fixed_pos_ratio": 0.05,
    "fixed_overcap_policy": "skip",
    "fixed_max_holdings": 20,
    "risk_budget_atr_window": 20,
    "risk_budget_pct": 0.35,
    "risk_budget_overcap_policy": "scale",
    "risk_budget_rebalance_mode": "standard",
    "risk_budget_max_leverage_multiple": 10,
    "vol_regime_risk_mgmt_enabled": False,
    "vol_periodic_risk_mgmt_enabled": False,
    "vol_periodic_rebalance_threshold_pct": 5,
    "vol_ratio_fast_atr_window": 5,
    "vol_ratio_slow_atr_window": 50,
    "vol_ratio_expand_threshold": 1.4,
    "vol_ratio_contract_threshold": 0.7,
    "vol_ratio_normal_threshold": 1,
    "vol_ratio_extreme_threshold": 2.1,
    "risk_of_ruin_maxrisk": 20,
    "group_enforce": False,
    "group_pick_policy": "highest_sharpe",
    "group_max_holdings": 4,
    "asset_groups": {},
}

SEARCH_CONTEXT: dict[str, Any] = {
    "group_name": "行业",
    "backtest_range": {"start": "20120608", "end": "20260717"},
    "dynamic_universe": True,
    "mode": "portfolio",
    "selected_codes": RAW_BASE_PAYLOAD["codes"],
    "single_code": "159570",
    "position_sizing": "risk_budget",
    "asset_groups_text": "",
    "asset_groups_parse_error": None,
    "r_take_profit_tiers_text": "1:1,2:0.5,3:0.33,4:0.25,5:0.2,6:0.17,7:0.14,8:0.13,9:0.11,10:0.1",
    "r_take_profit_tiers_parse_error": None,
    "r_profit_scaleout_tiers_text": "2:0.33",
    "r_profit_scaleout_tiers_parse_error": None,
    "bias_v_take_profit_tiers_text": "3:0.33,5:0.33",
    "bias_v_take_profit_tiers_parse_error": None,
    "exported_at": "2026-07-19T11:49:14.182Z",
}


def _to_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _join_url(base_url: str, endpoint: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("base_url is empty")
    ep = str(endpoint or "").strip()
    if not ep.startswith("/"):
        ep = "/" + ep
    return f"{base}{ep}"


def _http_post_json(*, url: str, payload: dict[str, Any], timeout: float) -> Any:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, method="POST")
    req.add_header("Accept", "application/json")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(str(e)) from e


def _normalize_ratio(v: Any, *, default: float) -> float:
    x = _to_float(v)
    if x is None:
        return float(default)
    return float(x / 100.0) if x > 1.0 else float(x)


def _normalize_percent_or_ratio(
    v: Any, *, default: float, ratio_ceiling: float
) -> float:
    x = _to_float(v)
    if x is None:
        return float(default)
    if x <= float(ratio_ceiling):
        return float(x)
    return float(x / 100.0)


def _percentile_sorted(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    p = float(max(0.0, min(1.0, q)))
    pos = p * float(len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    w = float(pos - lo)
    return float(vals[lo] + (vals[hi] - vals[lo]) * w)


def _sample_std(values: list[float]) -> float | None:
    n = len(values)
    if n < 2:
        return None
    mean = float(sum(values) / n)
    var = float(sum((v - mean) ** 2 for v in values) / (n - 1))
    if not math.isfinite(var) or var < 0.0:
        return None
    std = math.sqrt(var)
    return float(std) if math.isfinite(std) else None


def _trade_sort_ts(tr: dict[str, Any], idx: int) -> tuple[int, int]:
    def _parse_date_to_ord(x: Any) -> int | None:
        s = str(x or "").strip()
        if not s:
            return None
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return dt.datetime.strptime(s, fmt).date().toordinal()
            except ValueError:
                continue
        return None

    t_exit = _parse_date_to_ord(tr.get("exit_date")) if isinstance(tr, dict) else None
    t_entry = _parse_date_to_ord(tr.get("entry_date")) if isinstance(tr, dict) else None
    if t_exit is not None:
        return (t_exit, idx)
    if t_entry is not None:
        return (t_entry, idx)
    return (idx, idx)


def _rolling_100_sqn_median_from_trades(
    trades: Any, *, fallback_trade_count_total: float | None
) -> dict[str, Any]:
    rows = trades if isinstance(trades, list) else []
    rs_with_idx: list[tuple[int, float]] = []
    for i, tr in enumerate(rows):
        if not isinstance(tr, dict):
            continue
        r = _to_float(tr.get("r_multiple"))
        if r is None:
            continue
        rs_with_idx.append((i, float(r)))
    if rs_with_idx:
        ordered = sorted(
            rs_with_idx,
            key=lambda p: _trade_sort_ts(rows[p[0]], p[0]),
        )
        r_values = [float(x[1]) for x in ordered]
    else:
        r_values = []

    n = len(r_values)
    trade_count_total = (
        float(n)
        if n > 0
        else (float(fallback_trade_count_total) if fallback_trade_count_total else None)
    )
    if n < 2:
        return {
            "sqn_recent_100": None,
            "sqn_recent_100_window": None,
            "sqn_recent_100_roll_points": 0.0,
            "sqn_recent_100_median_note": "insufficient_r_samples",
            "sqn_recent_100_trade_count_total": trade_count_total,
        }

    win = 100 if n >= 100 else n
    sqn_points: list[float] = []
    for end_idx in range(win - 1, n):
        seg = r_values[end_idx - win + 1 : end_idx + 1]
        std = _sample_std(seg)
        if std is None or std <= 1e-12:
            continue
        mean = float(sum(seg) / len(seg))
        sqn = float((mean / std) * math.sqrt(len(seg)))
        if math.isfinite(sqn):
            sqn_points.append(sqn)
    sqn_points_sorted = sorted(sqn_points)
    median = _percentile_sorted(sqn_points_sorted, 0.5)
    return {
        "sqn_recent_100": median,
        "sqn_recent_100_window": float(win),
        "sqn_recent_100_roll_points": float(len(sqn_points_sorted)),
        "sqn_recent_100_median_note": (
            "rolling_100_median" if n >= 100 else f"rolling_{win}_median_due_to_lt_100"
        ),
        "sqn_recent_100_trade_count_total": trade_count_total,
    }


def _extract_kelly_by_code_stats(
    resp: dict[str, Any], *, expected_codes: list[str]
) -> dict[str, Any]:
    trade_stats = resp.get("trade_statistics") if isinstance(resp, dict) else None
    by_code = trade_stats.get("by_code") if isinstance(trade_stats, dict) else None
    by_code_dict = by_code if isinstance(by_code, dict) else {}
    required_codes = [str(c) for c in expected_codes]
    kelly_values: list[float] = []
    missing_codes: list[str] = []
    for code in required_codes:
        node = by_code_dict.get(code)
        if not isinstance(node, dict):
            missing_codes.append(code)
            continue
        kelly = _to_float(node.get("kelly_ex_zero"))
        if kelly is None:
            missing_codes.append(code)
            continue
        kelly_values.append(float(kelly))
    valid_count = len(kelly_values)
    required_count = len(required_codes)
    kelly_mean = float(sum(kelly_values) / valid_count) if valid_count > 0 else None
    kelly_std = _sample_std(kelly_values)
    return {
        "kelly_by_code_std": kelly_std,
        "kelly_by_code_mean": kelly_mean,
        "kelly_by_code_min": float(min(kelly_values)) if kelly_values else None,
        "kelly_by_code_max": float(max(kelly_values)) if kelly_values else None,
        "kelly_by_code_valid_count": float(valid_count),
        "kelly_by_code_required_count": float(required_count),
        "kelly_by_code_complete": bool(
            valid_count == required_count and required_count > 0
        ),
        "kelly_by_code_missing_codes": missing_codes,
        "kelly_by_code_missing_count": float(len(missing_codes)),
    }


def _build_base_payload(raw_payload: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(raw_payload)
    out["monthly_risk_budget_pct"] = _normalize_ratio(
        out.get("monthly_risk_budget_pct"), default=0.06
    )
    out["risk_of_ruin_maxrisk"] = _normalize_ratio(
        out.get("risk_of_ruin_maxrisk"), default=0.30
    )
    out["risk_budget_pct"] = _normalize_percent_or_ratio(
        out.get("risk_budget_pct"), default=0.01, ratio_ceiling=0.03
    )
    out["vol_periodic_rebalance_threshold_pct"] = _normalize_percent_or_ratio(
        out.get("vol_periodic_rebalance_threshold_pct"), default=0.05, ratio_ceiling=1.0
    )
    out["er_filter"] = bool(out.get("er_filter", True))
    return out


def _extract_metrics(
    resp: dict[str, Any], *, expected_codes: list[str]
) -> dict[str, Any]:
    metrics_block = resp.get("metrics") if isinstance(resp, dict) else None
    strategy = metrics_block.get("strategy") if isinstance(metrics_block, dict) else {}
    avg_annual_trade_count = _to_float(strategy.get("avg_annual_trade_count"))
    r_stats = resp.get("r_statistics") if isinstance(resp, dict) else None
    r_overall = r_stats.get("overall") if isinstance(r_stats, dict) else None
    sqn_block = r_overall.get("sqn") if isinstance(r_overall, dict) else None

    sqn_reason = (
        str(sqn_block.get("reason"))
        if isinstance(sqn_block, dict) and sqn_block.get("reason") is not None
        else None
    )
    sqn_applicable = (
        bool(sqn_block.get("applicable")) if isinstance(sqn_block, dict) else None
    )
    sqn_trade_count_total = (
        _to_float(sqn_block.get("trade_count_total"))
        if isinstance(sqn_block, dict)
        else None
    )
    sqn_min_trades = (
        _to_float(sqn_block.get("min_trades")) if isinstance(sqn_block, dict) else None
    )
    trade_stats = resp.get("trade_statistics") if isinstance(resp, dict) else None
    overall_trade = (
        trade_stats.get("overall") if isinstance(trade_stats, dict) else None
    )
    trades_list = trade_stats.get("trades") if isinstance(trade_stats, dict) else None
    sqn_roll_pack = _rolling_100_sqn_median_from_trades(
        trades_list,
        fallback_trade_count_total=sqn_trade_count_total,
    )
    sqn_recent_100 = _to_float(sqn_roll_pack.get("sqn_recent_100"))
    sqn_trade_count_total_final = _to_float(
        sqn_roll_pack.get("sqn_recent_100_trade_count_total")
    )
    if sqn_trade_count_total_final is None:
        sqn_trade_count_total_final = sqn_trade_count_total
    sqn_recent_100_insufficient_100 = (
        bool(sqn_trade_count_total_final < 100.0)
        if sqn_trade_count_total_final is not None
        else None
    )
    sqn_recent_100_note = (
        str(sqn_roll_pack.get("sqn_recent_100_median_note") or "").strip() or None
    )
    sqn_trade_count_used_final = _to_float(sqn_roll_pack.get("sqn_recent_100_window"))
    sqn_roll_points = _to_float(sqn_roll_pack.get("sqn_recent_100_roll_points"))
    if sqn_recent_100 is None and sqn_reason:
        if sqn_recent_100_note:
            sqn_recent_100_note = f"{sqn_recent_100_note}|backend:{sqn_reason}"
        else:
            sqn_recent_100_note = f"backend:{sqn_reason}"

    kelly_overall = (
        _to_float(overall_trade.get("kelly_ex_zero"))
        if isinstance(overall_trade, dict)
        else None
    )
    by_code_kelly = _extract_kelly_by_code_stats(resp, expected_codes=expected_codes)

    return {
        "cumulative_return": _to_float(strategy.get("cumulative_return")),
        "annualized_return": _to_float(strategy.get("annualized_return")),
        "annualized_volatility": _to_float(strategy.get("annualized_volatility")),
        "avg_annual_trade_count": avg_annual_trade_count,
        "max_drawdown": _to_float(strategy.get("max_drawdown")),
        "max_drawdown_recovery_days": _to_float(
            strategy.get("max_drawdown_recovery_days")
        ),
        "sharpe_ratio": _to_float(strategy.get("sharpe_ratio")),
        "sortino_ratio": _to_float(strategy.get("sortino_ratio")),
        "calmar_ratio": _to_float(strategy.get("calmar_ratio")),
        "ulcer_index": _to_float(strategy.get("ulcer_index")),
        "ulcer_performance_index": _to_float(strategy.get("ulcer_performance_index")),
        "sqn_recent_100": sqn_recent_100,
        "sqn_applicable": sqn_applicable,
        "sqn_reason": sqn_reason,
        "sqn_trade_count_used": sqn_trade_count_used_final,
        "sqn_trade_count_total": sqn_trade_count_total_final,
        "sqn_min_trades": sqn_min_trades,
        "sqn_recent_100_insufficient_100": sqn_recent_100_insufficient_100,
        "sqn_recent_100_note": sqn_recent_100_note,
        "sqn_recent_100_roll_points": sqn_roll_points,
        "kelly_overall": kelly_overall,
        **by_code_kelly,
    }


def _metric_value(row: dict[str, Any], key: str) -> float | None:
    metrics = row.get("metrics") if isinstance(row, dict) else None
    if not isinstance(metrics, dict):
        return None
    return _to_float(metrics.get(key))


def _assign_composite_scores(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    eligible_rows: list[dict[str, Any]] = []
    for row in rows:
        row["composite_score"] = None
        row["composite_components"] = {}
        row["composite_group_scores"] = {}
        row["objective_eligible"] = False
        row["objective_missing_metrics"] = []
        row["objective_ineligible_reason"] = None
        if str(row.get("status") or "") != "ok":
            continue
        metrics = row.get("metrics") if isinstance(row, dict) else {}
        sqn_trade_count_total = (
            _to_float(metrics.get("sqn_trade_count_total"))
            if isinstance(metrics, dict)
            else None
        )
        sqn_min_trades = (
            _to_float(metrics.get("sqn_min_trades"))
            if isinstance(metrics, dict)
            else None
        )
        sqn_min = sqn_min_trades if sqn_min_trades is not None else 30.0
        if sqn_trade_count_total is not None and sqn_trade_count_total < sqn_min:
            reason = (
                f"sqn_trade_count_lt_min:{int(sqn_trade_count_total)}<{int(sqn_min)}"
            )
            sqn_reason = (
                str(metrics.get("sqn_reason"))
                if isinstance(metrics, dict) and metrics.get("sqn_reason")
                else None
            )
            if sqn_reason:
                reason = f"{reason} (sqn_reason={sqn_reason})"
            row["objective_ineligible_reason"] = reason
            continue
        missing = [
            metric
            for metric, _hb in OBJECTIVE_METRICS
            if _metric_value(row, metric) is None
        ]
        if missing:
            row["objective_missing_metrics"] = missing
            reason = f"missing_metrics:{','.join(missing)}"
            if "sqn_recent_100" in missing and isinstance(metrics, dict):
                sqn_reason = metrics.get("sqn_reason")
                sqn_roll_note = metrics.get("sqn_recent_100_note")
                if sqn_reason:
                    reason = f"{reason} (sqn_reason={sqn_reason})"
                if sqn_roll_note:
                    reason = f"{reason} (sqn_roll_note={sqn_roll_note})"
            row["objective_ineligible_reason"] = reason
            continue
        row["objective_eligible"] = True
        eligible_rows.append(row)

    metric_bounds: dict[str, dict[str, float]] = {}
    for metric, _higher_better in OBJECTIVE_METRICS:
        vals: list[float] = []
        for row in eligible_rows:
            v = _metric_value(row, metric)
            if v is not None:
                vals.append(float(v))
        if vals:
            metric_bounds[metric] = {"min": float(min(vals)), "max": float(max(vals))}
    for row in eligible_rows:
        parts: dict[str, float] = {}
        for metric, higher_better in OBJECTIVE_METRICS:
            v = _metric_value(row, metric)
            bounds = metric_bounds.get(metric)
            if v is None or not isinstance(bounds, dict):
                continue
            lo = _to_float(bounds.get("min"))
            hi = _to_float(bounds.get("max"))
            if lo is None or hi is None:
                continue
            if abs(hi - lo) <= 1e-12:
                score = 0.5
            else:
                base = float((float(v) - lo) / (hi - lo))
                score = base if higher_better else float(1.0 - base)
            parts[metric] = float(max(0.0, min(1.0, score)))
        row["composite_components"] = parts
        group_scores: dict[str, float] = {}
        weighted_sum = 0.0
        weight_sum = 0.0
        for group_name, metrics, group_weight in OBJECTIVE_GROUPS:
            vals = [parts[m] for m in metrics if m in parts]
            has_bounds = any(m in metric_bounds for m in metrics)
            if vals:
                g_score = float(sum(vals) / len(vals))
            elif has_bounds:
                g_score = 0.0
            else:
                continue
            group_scores[group_name] = g_score
            weighted_sum += g_score * float(group_weight)
            weight_sum += float(group_weight)
        row["composite_group_scores"] = group_scores
        row["composite_score"] = (
            float(weighted_sum / weight_sum) if weight_sum > 0.0 else None
        )
        if row.get("composite_score") is None:
            row["objective_eligible"] = False
            row["objective_ineligible_reason"] = "composite_score_unavailable"
            continue
    return metric_bounds


def _grid_decimal_values(*, start: float, end: float, step: float) -> list[float]:
    try:
        s = Decimal(str(start))
        e = Decimal(str(end))
        k = Decimal(str(step))
    except Exception as ex:  # noqa: BLE001
        raise ValueError("invalid grid range") from ex
    if k <= 0:
        raise ValueError("invalid grid step")
    lo = min(s, e)
    hi = max(s, e)
    out: list[float] = []
    cur = lo
    eps = Decimal("1e-12")
    while cur <= hi + eps:
        out.append(float(cur))
        cur += k
    if not out:
        raise ValueError("empty grid range")
    return out


def _run_single_case(
    *,
    base_url: str,
    endpoint: str,
    base_payload: dict[str, Any],
    er_threshold: float,
    expected_codes: list[str],
    timeout: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    payload = copy.deepcopy(base_payload)
    payload["er_filter"] = True
    payload["er_threshold"] = float(er_threshold)
    url = _join_url(base_url, endpoint)
    try:
        resp = _http_post_json(url=url, payload=payload, timeout=timeout)
        if not isinstance(resp, dict):
            raise RuntimeError("unexpected response schema")
        metrics = _extract_metrics(resp, expected_codes=expected_codes)
        return {
            "er_threshold": float(er_threshold),
            "status": "ok",
            "elapsed_ms": int((time.perf_counter() - t0) * 1000.0),
            "metrics": metrics,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        return {
            "er_threshold": float(er_threshold),
            "status": "error",
            "elapsed_ms": int((time.perf_counter() - t0) * 1000.0),
            "metrics": {},
            "error": str(e),
        }


def _verify_er_boundary_support(
    *,
    base_url: str,
    endpoint: str,
    base_payload: dict[str, Any],
    expected_codes: list[str],
    timeout: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for threshold in (0.0, 1.0):
        row = _run_single_case(
            base_url=base_url,
            endpoint=endpoint,
            base_payload=base_payload,
            er_threshold=threshold,
            expected_codes=expected_codes,
            timeout=timeout,
        )
        out.append(row)
    return out


def _pick_best_by_composite(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: tuple[tuple[float, float, float, float, float], dict[str, Any]] | None = None
    for row in rows:
        if str(row.get("status") or "") != "ok":
            continue
        if not bool(row.get("objective_eligible")):
            continue
        composite = _to_float(row.get("composite_score"))
        if composite is None:
            continue
        sharpe = _metric_value(row, "sharpe_ratio")
        if sharpe is None:
            continue
        ann = _metric_value(row, "annualized_return") or -1e18
        cum_ret = _metric_value(row, "cumulative_return") or -1e18
        mdd = _metric_value(row, "max_drawdown")
        mdd_score = -abs(mdd) if mdd is not None else -1e18
        key = (
            float(composite),
            float(sharpe),
            float(ann),
            float(cum_ret),
            float(mdd_score),
        )
        if best is None:
            best = (key, row)
            continue
        best_key, _ = best
        if key > best_key:
            best = (key, row)
    if best is None:
        return None
    return best[1]


def _assign_kelly_std_objective_flags(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["kelly_std_objective_eligible"] = False
        row["kelly_std_objective_ineligible_reason"] = None
        if str(row.get("status") or "") != "ok":
            row["kelly_std_objective_ineligible_reason"] = "status_not_ok"
            continue
        valid_count = _metric_value(row, "kelly_by_code_valid_count")
        required_count = _metric_value(row, "kelly_by_code_required_count")
        std = _metric_value(row, "kelly_by_code_std")
        if required_count is None or required_count <= 0:
            row["kelly_std_objective_ineligible_reason"] = "required_code_count_invalid"
            continue
        if valid_count is None or valid_count < required_count:
            row["kelly_std_objective_ineligible_reason"] = (
                f"missing_kelly_codes:{int(valid_count or 0)}/{int(required_count)}"
            )
            continue
        if std is None:
            row["kelly_std_objective_ineligible_reason"] = "kelly_std_unavailable"
            continue
        row["kelly_std_objective_eligible"] = True


def _pick_best_by_kelly_std_min(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: (
        tuple[tuple[float, float, float, float, float, float], dict[str, Any]] | None
    ) = None
    for row in rows:
        if not bool(row.get("kelly_std_objective_eligible")):
            continue
        std = _metric_value(row, "kelly_by_code_std")
        if std is None:
            continue
        mean_kelly = _metric_value(row, "kelly_by_code_mean") or -1e18
        sharpe = _metric_value(row, "sharpe_ratio") or -1e18
        ann = _metric_value(row, "annualized_return") or -1e18
        mdd = _metric_value(row, "max_drawdown")
        mdd_score = abs(mdd) if mdd is not None else 1e18
        composite = _to_float(row.get("composite_score")) or -1e18
        key = (
            float(std),
            float(-mean_kelly),
            float(-sharpe),
            float(-ann),
            float(mdd_score),
            float(-composite),
        )
        if best is None:
            best = (key, row)
            continue
        best_key, _ = best
        if key < best_key:
            best = (key, row)
    if best is None:
        return None
    return best[1]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "er_threshold",
        "objective_eligible",
        "objective_ineligible_reason",
        "composite_score",
        "kelly_std_objective_eligible",
        "kelly_std_objective_ineligible_reason",
        "status",
        "elapsed_ms",
        "sharpe_ratio",
        "annualized_return",
        "cumulative_return",
        "annualized_volatility",
        "avg_annual_trade_count",
        "sqn_recent_100",
        "sqn_applicable",
        "sqn_reason",
        "sqn_trade_count_used",
        "sqn_trade_count_total",
        "sqn_min_trades",
        "sqn_recent_100_insufficient_100",
        "sqn_recent_100_note",
        "sqn_recent_100_roll_points",
        "kelly_overall",
        "kelly_by_code_std",
        "kelly_by_code_mean",
        "kelly_by_code_min",
        "kelly_by_code_max",
        "kelly_by_code_valid_count",
        "kelly_by_code_required_count",
        "kelly_by_code_complete",
        "kelly_by_code_missing_count",
        "kelly_by_code_missing_codes",
        "max_drawdown",
        "max_drawdown_recovery_days",
        "sortino_ratio",
        "calmar_ratio",
        "ulcer_index",
        "ulcer_performance_index",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            metrics = row.get("metrics") if isinstance(row, dict) else {}
            out = {
                "er_threshold": row.get("er_threshold"),
                "objective_eligible": row.get("objective_eligible"),
                "objective_ineligible_reason": row.get("objective_ineligible_reason"),
                "composite_score": row.get("composite_score"),
                "kelly_std_objective_eligible": row.get("kelly_std_objective_eligible"),
                "kelly_std_objective_ineligible_reason": row.get(
                    "kelly_std_objective_ineligible_reason"
                ),
                "status": row.get("status"),
                "elapsed_ms": row.get("elapsed_ms"),
                "error": row.get("error"),
            }
            for c in cols[6:-1]:
                out[c] = metrics.get(c) if isinstance(metrics, dict) else None
            w.writerow(out)


def run(args: argparse.Namespace) -> int:
    base_payload = _build_base_payload(RAW_BASE_PAYLOAD)
    expected_codes = [str(c) for c in list(base_payload.get("codes") or [])]

    boundary_check = _verify_er_boundary_support(
        base_url=str(args.base_url),
        endpoint=str(args.endpoint),
        base_payload=base_payload,
        expected_codes=expected_codes,
        timeout=float(args.timeout),
    )
    boundary_errors = [x for x in boundary_check if str(x.get("status") or "") != "ok"]
    if boundary_errors and not bool(args.skip_boundary_guard):
        first = boundary_errors[0]
        raise RuntimeError(
            "ER threshold boundary check failed "
            f"(threshold={first.get('er_threshold')}, error={first.get('error')})"
        )

    grid = _grid_decimal_values(
        start=float(args.start_threshold),
        end=float(args.end_threshold),
        step=float(args.step_threshold),
    )
    print(
        f"[INFO] endpoint={args.endpoint}, grid={len(grid)} points, workers={args.workers}"
    )
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        fut_map = {
            ex.submit(
                _run_single_case,
                base_url=str(args.base_url),
                endpoint=str(args.endpoint),
                base_payload=base_payload,
                er_threshold=er,
                expected_codes=expected_codes,
                timeout=float(args.timeout),
            ): er
            for er in grid
        }
        done = 0
        for fut in concurrent.futures.as_completed(fut_map):
            done += 1
            row = fut.result()
            rows.append(row)
            if done % 5 == 0 or done == len(grid):
                ok = sum(1 for x in rows if str(x.get("status") or "") == "ok")
                print(f"[INFO] progress={done}/{len(grid)}, success={ok}")

    metric_bounds = _assign_composite_scores(rows)
    _assign_kelly_std_objective_flags(rows)
    rows.sort(key=lambda x: float(x.get("er_threshold") or 0.0))
    best_composite = _pick_best_by_composite(rows)
    best_kelly_std = _pick_best_by_kelly_std_min(rows)
    objective_method = str(
        getattr(args, "objective_method", "composite_score") or ""
    ).strip()
    if objective_method == "kelly_std_min":
        best = best_kelly_std
    else:
        objective_method = "composite_score"
        best = best_composite
    errors = [x for x in rows if str(x.get("status") or "") != "ok"]
    objective_ineligible = [
        x
        for x in rows
        if str(x.get("status") or "") == "ok" and not bool(x.get("objective_eligible"))
    ]
    kelly_std_ineligible = [
        x
        for x in rows
        if str(x.get("status") or "") == "ok"
        and not bool(x.get("kelly_std_objective_eligible"))
    ]

    output_json = Path(str(args.output_json))
    output_csv = (
        Path(str(args.output_csv))
        if str(args.output_csv).strip()
        else output_json.with_suffix(".csv")
    )
    output_best_csv = (
        Path(str(args.output_best_csv))
        if str(args.output_best_csv).strip()
        else output_json.with_name(output_json.stem + "_best.csv")
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "api_base_url": str(args.base_url),
            "api_endpoint": str(args.endpoint),
            "search_context": SEARCH_CONTEXT,
            "search_space": {
                "parameter": "er_threshold",
                "start": float(args.start_threshold),
                "end": float(args.end_threshold),
                "step": float(args.step_threshold),
                "values": grid,
            },
            "boundary_support_check": boundary_check,
            "fixed_payload": {
                k: v for k, v in base_payload.items() if k != "er_threshold"
            },
            "objective": {
                "selected_method": objective_method,
                "metric": (
                    "kelly_by_code_std"
                    if objective_method == "kelly_std_min"
                    else "composite_score"
                ),
                "mode": "min" if objective_method == "kelly_std_min" else "max",
                "method": (
                    "cross_sectional_kelly_std_min"
                    if objective_method == "kelly_std_min"
                    else "grouped_minmax_weighted"
                ),
                "strict_comparability": not bool(args.allow_incomplete_objective),
                "sqn_policy": {
                    "metric": "sqn_recent_100",
                    "calculation": (
                        "median_of_rolling_sqn_windows;"
                        "window=100_when_trade_count>=100_else_window=trade_count"
                    ),
                    "ineligible_when_trade_count_lt": 30,
                    "mark_insufficient_recent_100_when_trade_count_lt": 100,
                },
                "kelly_std_policy": {
                    "metric": "kelly_by_code_std",
                    "require_all_codes": True,
                    "kelly_field": "trade_statistics.by_code.<code>.kelly_ex_zero",
                    "std_type": "sample_std_n_minus_1",
                },
                "components": [
                    {
                        "name": metric,
                        "higher_is_better": higher_better,
                        "bounds": metric_bounds.get(metric),
                    }
                    for metric, higher_better in OBJECTIVE_METRICS
                ],
                "groups": [
                    {"name": name, "metrics": list(metrics), "weight": float(weight)}
                    for name, metrics, weight in OBJECTIVE_GROUPS
                ],
            },
            "total_cases": len(rows),
            "success_cases": len(rows) - len(errors),
            "error_cases": len(errors),
            "objective_eligible_cases": sum(
                1 for x in rows if bool(x.get("objective_eligible"))
            ),
            "objective_ineligible_cases": len(objective_ineligible),
            "kelly_std_objective_eligible_cases": sum(
                1 for x in rows if bool(x.get("kelly_std_objective_eligible"))
            ),
            "kelly_std_objective_ineligible_cases": len(kelly_std_ineligible),
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        },
        "results": rows,
        "best_single_setting": best,
        "best_by_method": {
            "composite_score": best_composite,
            "kelly_std_min": best_kelly_std,
        },
        "errors": errors,
        "objective_ineligible": objective_ineligible,
        "kelly_std_objective_ineligible": kelly_std_ineligible,
    }
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_csv, rows)
    _write_csv(output_best_csv, [best] if isinstance(best, dict) else [])

    print(f"[INFO] done in {time.perf_counter() - t0:.1f}s")
    print(f"[INFO] json: {output_json}")
    print(f"[INFO] csv: {output_csv}")
    print(f"[INFO] best: {output_best_csv}")
    if best_composite:
        m = best_composite.get("metrics") if isinstance(best_composite, dict) else {}
        print(
            "[INFO] best composite "
            f"er_threshold={best_composite.get('er_threshold')}, "
            f"composite={_to_float(best_composite.get('composite_score'))}, "
            f"sharpe={_to_float((m or {}).get('sharpe_ratio'))}, "
            "sqn_roll100_median="
            f"{_to_float((m or {}).get('sqn_recent_100'))}"
        )
    if best_kelly_std:
        m = best_kelly_std.get("metrics") if isinstance(best_kelly_std, dict) else {}
        print(
            "[INFO] best kelly-std-min "
            f"er_threshold={best_kelly_std.get('er_threshold')}, "
            f"kelly_by_code_std={_to_float((m or {}).get('kelly_by_code_std'))}, "
            f"kelly_mean={_to_float((m or {}).get('kelly_by_code_mean'))}, "
            f"sharpe={_to_float((m or {}).get('sharpe_ratio'))}"
        )
    if best:
        print(
            "[INFO] selected objective best "
            f"method={objective_method}, er_threshold={best.get('er_threshold')}"
        )
    if errors:
        print(f"[WARN] error cases: {len(errors)}")
    if objective_ineligible:
        print(
            f"[WARN] objective ineligible cases: {len(objective_ineligible)} "
            "(missing required objective metrics)"
        )
        sample = objective_ineligible[0]
        print(
            "[WARN] sample ineligible: "
            f"er_threshold={sample.get('er_threshold')}, "
            f"reason={sample.get('objective_ineligible_reason')}"
        )
    if kelly_std_ineligible:
        sample = kelly_std_ineligible[0]
        print(
            "[WARN] kelly-std objective ineligible cases: "
            f"{len(kelly_std_ineligible)} (sample er_threshold={sample.get('er_threshold')}, "
            f"reason={sample.get('kelly_std_objective_ineligible_reason')})"
        )
    if (
        objective_method == "composite_score"
        and objective_ineligible
        and not bool(args.allow_incomplete_objective)
    ):
        print(
            "[ERROR] strict objective comparability check failed; "
            "re-run with --allow-incomplete-objective to bypass.",
            file=sys.stderr,
        )
        return 3
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Grid-search er_threshold for trend portfolio by calling backend API."
    )
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--start-threshold", type=float, default=0.0)
    p.add_argument("--end-threshold", type=float, default=1.0)
    p.add_argument("--step-threshold", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    p.add_argument("--output-csv", default="")
    p.add_argument("--output-best-csv", default="")
    p.add_argument(
        "--objective-method",
        choices=("composite_score", "kelly_std_min"),
        default="composite_score",
        help="Objective selector: composite_score (default) or kelly_std_min.",
    )
    p.add_argument("--skip-boundary-guard", action="store_true")
    p.add_argument("--allow-incomplete-objective", action="store_true")
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if int(args.workers) < 1:
        print("[ERROR] --workers must be >= 1", file=sys.stderr)
        return 2
    try:
        return run(args)
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
