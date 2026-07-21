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
    "src/etf_momentum/web/data/trend_momentum_base_param_search_results.json"
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

# User-provided payload baseline. Script sweeps mom_lookback + entry/exit thresholds.
RAW_BASE_PAYLOAD: dict[str, Any] = {
    "codes": [
        "510300",
        "159915",
        "588000",
        "560280",
        "159928",
        "513070",
        "159570",
        "513090",
        "513500",
        "513100",
        "513030",
        "513520",
        "513310",
        "164824",
        "518800",
        "161226",
        "159980",
        "501018",
        "159981",
        "159985",
        "515220",
        "515180",
        "513690",
        "159792",
        "159870",
        "162411",
        "512010",
        "515880",
        "517520",
        "513080",
        "159920",
        "512480",
        "563300",
        "512660",
        "562500",
        "159869",
    ],
    "position_sizing": "risk_budget",
    "dynamic_universe": True,
    "start": "20111209",
    "end": "20211231",
    "initial_account_amount": None,
    "cost_bps": 2,
    "slippage_rate": 0.001,
    "capacity_window_years": 1,
    "quick_mode": True,
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
    "er_threshold": 0.25,
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
    "r_profit_scaleout_tiers": [{"r_multiple": 2, "reduce_fraction": 0.33}],
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
    "risk_budget_pct": 0.25,
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
    "group_name": "趋势大全",
    "backtest_range": {"start": "20111209", "end": "20211231"},
    "dynamic_universe": True,
    "mode": "portfolio",
    "selected_codes": RAW_BASE_PAYLOAD["codes"],
    "single_code": "510300",
    "position_sizing": "risk_budget",
    "asset_groups_text": "",
    "asset_groups_parse_error": None,
    "r_take_profit_tiers_text": "1:1,2:0.5,3:0.33,4:0.25,5:0.2,6:0.17,7:0.14,8:0.13,9:0.11,10:0.1",
    "r_take_profit_tiers_parse_error": None,
    "r_profit_scaleout_tiers_text": "2:0.33",
    "r_profit_scaleout_tiers_parse_error": None,
    "bias_v_take_profit_tiers_text": "3:0.33,5:0.33",
    "bias_v_take_profit_tiers_parse_error": None,
    "exported_at": "2026-07-21T03:48:38.195Z",
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


def _http_post_json(
    *,
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> Any:
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


def _percent_to_ratio_signed(v: float) -> float:
    x = _to_float(v)
    if x is None:
        raise ValueError("threshold value is invalid")
    if abs(x) <= 1.0:
        return float(x)
    return float(x / 100.0)


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
    return out


def _extract_metrics(resp: dict[str, Any]) -> dict[str, Any]:
    metrics_block = resp.get("metrics") if isinstance(resp, dict) else None
    strategy = metrics_block.get("strategy") if isinstance(metrics_block, dict) else {}
    avg_annual_trade_count = _to_float(strategy.get("avg_annual_trade_count"))
    r_stats = resp.get("r_statistics") if isinstance(resp, dict) else None
    r_overall = r_stats.get("overall") if isinstance(r_stats, dict) else None
    sqn_block = r_overall.get("sqn") if isinstance(r_overall, dict) else None

    sqn_raw = _to_float(sqn_block.get("sqn")) if isinstance(sqn_block, dict) else None
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
    sqn_trade_count_used = (
        _to_float(sqn_block.get("trade_count_used"))
        if isinstance(sqn_block, dict)
        else None
    )
    sqn_min_trades = (
        _to_float(sqn_block.get("min_trades")) if isinstance(sqn_block, dict) else None
    )
    sqn_recent_100 = sqn_raw
    sqn_recent_100_insufficient_100 = (
        bool(sqn_trade_count_total < 100.0)
        if sqn_trade_count_total is not None
        else None
    )
    sqn_recent_100_note = (
        "trades_lt_100" if sqn_recent_100_insufficient_100 is True else None
    )
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
        "sqn_trade_count_used": sqn_trade_count_used,
        "sqn_trade_count_total": sqn_trade_count_total,
        "sqn_min_trades": sqn_min_trades,
        "sqn_recent_100_insufficient_100": sqn_recent_100_insufficient_100,
        "sqn_recent_100_note": sqn_recent_100_note,
    }


def _metric_value(row: dict[str, Any], key: str) -> float | None:
    metrics = row.get("metrics") if isinstance(row, dict) else None
    if not isinstance(metrics, dict):
        return None
    return _to_float(metrics.get(key))


def _assign_composite_scores(
    rows: list[dict[str, Any]],
    *,
    min_avg_annual_trade_count: float,
    min_annualized_return: float,
) -> dict[str, dict[str, float]]:
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
        avg_annual_trade_count = (
            _to_float(metrics.get("avg_annual_trade_count"))
            if isinstance(metrics, dict)
            else None
        )
        if min_avg_annual_trade_count > 0.0:
            if avg_annual_trade_count is None:
                row["objective_ineligible_reason"] = "avg_annual_trade_count_missing"
                continue
            if avg_annual_trade_count < min_avg_annual_trade_count:
                row["objective_ineligible_reason"] = (
                    "avg_annual_trade_count_lt_min:"
                    f"{avg_annual_trade_count:.6g}"
                    f"<{min_avg_annual_trade_count:.6g}"
                )
                continue
        annualized_return = (
            _to_float(metrics.get("annualized_return"))
            if isinstance(metrics, dict)
            else None
        )
        if min_annualized_return > 0.0:
            if annualized_return is None:
                row["objective_ineligible_reason"] = "annualized_return_missing"
                continue
            if annualized_return < min_annualized_return:
                row["objective_ineligible_reason"] = (
                    "annualized_return_lt_min:"
                    f"{annualized_return:.6g}"
                    f"<{min_annualized_return:.6g}"
                )
                continue
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
            for metric, _higher_better in OBJECTIVE_METRICS
            if _metric_value(row, metric) is None
        ]
        if missing:
            row["objective_missing_metrics"] = missing
            reason = f"missing_metrics:{','.join(missing)}"
            if "sqn_recent_100" in missing and isinstance(metrics, dict):
                sqn_reason = metrics.get("sqn_reason")
                if sqn_reason:
                    reason = f"{reason} (sqn_reason={sqn_reason})"
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


def _grid_int_values(*, start: int, end: int, step: int) -> list[int]:
    st = int(step)
    if st <= 0:
        raise ValueError("invalid int grid step")
    lo = int(min(start, end))
    hi = int(max(start, end))
    out = list(range(lo, hi + 1, st))
    if not out:
        raise ValueError("empty int grid range")
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
        if key > best[0]:
            best = (key, row)
    return best[1] if best else None


def _build_param_space(
    *,
    lookback_values: list[int],
    entry_threshold_pct_values: list[float],
    exit_threshold_pct_values: list[float],
    max_cases: int | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for lb in lookback_values:
        for en_pct in entry_threshold_pct_values:
            en = _percent_to_ratio_signed(en_pct)
            for ex_pct in exit_threshold_pct_values:
                ex = _percent_to_ratio_signed(ex_pct)
                out.append(
                    {
                        "mom_lookback": int(lb),
                        "tsmom_entry_threshold_pct": float(en_pct),
                        "tsmom_entry_threshold": float(en),
                        "tsmom_exit_threshold_pct": float(ex_pct),
                        "tsmom_exit_threshold": float(ex),
                    }
                )
                if max_cases is not None and len(out) >= max_cases:
                    return out
    return out


def _run_single_case(
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
    url = _join_url(base_url, endpoint)
    last_error = ""
    attempts = max(1, int(retry_times) + 1)
    deadline = t0 + float(timeout)
    for attempt_idx in range(attempts):
        remain = deadline - time.perf_counter()
        if remain <= 0:
            last_error = (
                f"timed out (case budget {timeout:.1f}s exceeded before attempt "
                f"{attempt_idx + 1})"
            )
            break
        try:
            resp = _http_post_json(
                url=url,
                payload=payload,
                timeout=max(1.0, float(remain)),
            )
            if not isinstance(resp, dict):
                raise RuntimeError("unexpected response schema")
            metrics = _extract_metrics(resp)
            return {
                "mom_lookback": int(case["mom_lookback"]),
                "tsmom_entry_threshold_pct": float(case["tsmom_entry_threshold_pct"]),
                "tsmom_entry_threshold": float(case["tsmom_entry_threshold"]),
                "tsmom_exit_threshold_pct": float(case["tsmom_exit_threshold_pct"]),
                "tsmom_exit_threshold": float(case["tsmom_exit_threshold"]),
                "status": "ok",
                "elapsed_ms": int((time.perf_counter() - t0) * 1000.0),
                "attempts": int(attempt_idx + 1),
                "metrics": metrics,
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
            time.sleep(min(1.5, 0.25 * (attempt_idx + 1), remain_after_error))
    return {
        "mom_lookback": int(case["mom_lookback"]),
        "tsmom_entry_threshold_pct": float(case["tsmom_entry_threshold_pct"]),
        "tsmom_entry_threshold": float(case["tsmom_entry_threshold"]),
        "tsmom_exit_threshold_pct": float(case["tsmom_exit_threshold_pct"]),
        "tsmom_exit_threshold": float(case["tsmom_exit_threshold"]),
        "status": "error",
        "elapsed_ms": int((time.perf_counter() - t0) * 1000.0),
        "attempts": int(attempts),
        "metrics": {},
        "error": last_error or "unknown_error",
    }


def _run_param_space_bounded(
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
    if not param_space:
        return []
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
                _run_single_case,
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
                done += 1
                rows.append(fut.result())
            while len(pending) < max_in_flight and _submit_one():
                pass
            if done % 25 == 0 or done == total:
                ok = sum(1 for x in rows if str(x.get("status") or "") == "ok")
                print(
                    f"[INFO] progress={done}/{total}, success={ok}, in_flight={len(pending)}"
                )
    return rows


def _sort_rows_for_display(rows: list[dict[str, Any]]) -> None:
    rows.sort(
        key=lambda x: (
            int(_to_float(x.get("mom_lookback")) or 0),
            float(_to_float(x.get("tsmom_entry_threshold_pct")) or 0.0),
            float(_to_float(x.get("tsmom_exit_threshold_pct")) or 0.0),
        )
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "mom_lookback",
        "tsmom_entry_threshold_pct",
        "tsmom_entry_threshold",
        "tsmom_exit_threshold_pct",
        "tsmom_exit_threshold",
        "objective_eligible",
        "objective_ineligible_reason",
        "composite_score",
        "status",
        "elapsed_ms",
        "attempts",
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
                "mom_lookback": row.get("mom_lookback"),
                "tsmom_entry_threshold_pct": row.get("tsmom_entry_threshold_pct"),
                "tsmom_entry_threshold": row.get("tsmom_entry_threshold"),
                "tsmom_exit_threshold_pct": row.get("tsmom_exit_threshold_pct"),
                "tsmom_exit_threshold": row.get("tsmom_exit_threshold"),
                "objective_eligible": row.get("objective_eligible"),
                "objective_ineligible_reason": row.get("objective_ineligible_reason"),
                "composite_score": row.get("composite_score"),
                "status": row.get("status"),
                "elapsed_ms": row.get("elapsed_ms"),
                "attempts": row.get("attempts"),
                "error": row.get("error"),
            }
            for c in cols[11:-1]:
                out[c] = metrics.get(c) if isinstance(metrics, dict) else None
            w.writerow(out)


def run(args: argparse.Namespace) -> int:
    base_payload = _build_base_payload(RAW_BASE_PAYLOAD)
    lookback_values = _grid_int_values(
        start=int(args.lookback_start),
        end=int(args.lookback_end),
        step=int(args.lookback_step),
    )
    entry_pct_values = _grid_decimal_values(
        start=float(args.entry_threshold_pct_start),
        end=float(args.entry_threshold_pct_end),
        step=float(args.entry_threshold_pct_step),
    )
    exit_pct_values = _grid_decimal_values(
        start=float(args.exit_threshold_pct_start),
        end=float(args.exit_threshold_pct_end),
        step=float(args.exit_threshold_pct_step),
    )
    total_full = len(lookback_values) * len(entry_pct_values) * len(exit_pct_values)
    max_cases = int(args.max_cases) if int(args.max_cases) > 0 else None
    param_space = _build_param_space(
        lookback_values=lookback_values,
        entry_threshold_pct_values=entry_pct_values,
        exit_threshold_pct_values=exit_pct_values,
        max_cases=max_cases,
    )
    workers = max(1, int(args.workers))
    max_in_flight = int(args.max_in_flight)
    if max_in_flight <= 0:
        max_in_flight = max(8, workers * 8)
    print(
        "[INFO] "
        f"endpoint={args.endpoint}, "
        f"lookback={len(lookback_values)}, entry={len(entry_pct_values)}, exit={len(exit_pct_values)}, "
        f"full_grid={total_full}, run_cases={len(param_space)}, "
        f"workers={workers}, max_in_flight={max_in_flight}"
    )
    t0 = time.perf_counter()
    rows = _run_param_space_bounded(
        base_url=str(args.base_url),
        endpoint=str(args.endpoint),
        base_payload=base_payload,
        param_space=param_space,
        workers=workers,
        max_in_flight=max_in_flight,
        timeout=float(args.timeout),
        retry_times=int(args.retry_times),
    )
    min_avg_annual_trade_count = max(
        0.0, float(args.objective_min_avg_annual_trade_count)
    )
    min_annualized_return = max(
        0.0, float(args.objective_min_annualized_return_pct) / 100.0
    )
    metric_bounds = _assign_composite_scores(
        rows,
        min_avg_annual_trade_count=min_avg_annual_trade_count,
        min_annualized_return=min_annualized_return,
    )
    _sort_rows_for_display(rows)
    best = _pick_best_by_composite(rows)
    errors = [x for x in rows if str(x.get("status") or "") != "ok"]
    objective_ineligible = [
        x
        for x in rows
        if str(x.get("status") or "") == "ok" and not bool(x.get("objective_eligible"))
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
                "parameters": [
                    "mom_lookback",
                    "tsmom_entry_threshold_pct",
                    "tsmom_exit_threshold_pct",
                ],
                "mom_lookback_start": int(args.lookback_start),
                "mom_lookback_end": int(args.lookback_end),
                "mom_lookback_step": int(args.lookback_step),
                "tsmom_entry_threshold_pct_start": float(
                    args.entry_threshold_pct_start
                ),
                "tsmom_entry_threshold_pct_end": float(args.entry_threshold_pct_end),
                "tsmom_entry_threshold_pct_step": float(args.entry_threshold_pct_step),
                "tsmom_exit_threshold_pct_start": float(args.exit_threshold_pct_start),
                "tsmom_exit_threshold_pct_end": float(args.exit_threshold_pct_end),
                "tsmom_exit_threshold_pct_step": float(args.exit_threshold_pct_step),
                "lookback_values": lookback_values,
                "entry_threshold_pct_values": entry_pct_values,
                "exit_threshold_pct_values": exit_pct_values,
                "full_grid_cases": total_full,
                "executed_cases": len(param_space),
                "max_cases_applied": max_cases,
            },
            "concurrency": {
                "workers": workers,
                "max_in_flight": max_in_flight,
                "retry_times": int(args.retry_times),
                "merge_strategy": "bounded_inflight_as_completed",
            },
            "fixed_payload": {
                k: v
                for k, v in base_payload.items()
                if k
                not in {"mom_lookback", "tsmom_entry_threshold", "tsmom_exit_threshold"}
            },
            "objective": {
                "metric": "composite_score",
                "mode": "max",
                "method": "grouped_minmax_weighted",
                "strict_comparability": not bool(args.allow_incomplete_objective),
                "sqn_policy": {
                    "metric": "sqn_recent_100",
                    "ineligible_when_trade_count_lt": 30,
                    "mark_insufficient_recent_100_when_trade_count_lt": 100,
                },
                "eligibility_constraints": {
                    "min_avg_annual_trade_count": min_avg_annual_trade_count,
                    "min_annualized_return": min_annualized_return,
                    "min_annualized_return_pct": min_annualized_return * 100.0,
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
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        },
        "results": rows,
        "best_single_setting": best,
        "errors": errors,
        "objective_ineligible": objective_ineligible,
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
    if best:
        m = best.get("metrics") if isinstance(best, dict) else {}
        print(
            "[INFO] best composite "
            f"lookback={best.get('mom_lookback')}, "
            f"entry={best.get('tsmom_entry_threshold_pct')}%, "
            f"exit={best.get('tsmom_exit_threshold_pct')}%, "
            f"composite={_to_float(best.get('composite_score'))}, "
            f"sharpe={_to_float((m or {}).get('sharpe_ratio'))}"
        )
    if errors:
        print(f"[WARN] error cases: {len(errors)}")
    if objective_ineligible:
        print(
            f"[WARN] objective ineligible cases: {len(objective_ineligible)} "
            "(failed objective eligibility constraints)"
        )
        sample = objective_ineligible[0]
        print(
            "[WARN] sample ineligible: "
            f"lookback={sample.get('mom_lookback')}, "
            f"entry={sample.get('tsmom_entry_threshold_pct')}%, "
            f"exit={sample.get('tsmom_exit_threshold_pct')}%, "
            f"reason={sample.get('objective_ineligible_reason')}"
        )
    if objective_ineligible and not bool(args.allow_incomplete_objective):
        print(
            "[ERROR] strict objective comparability check failed; "
            "re-run with --allow-incomplete-objective to bypass.",
            file=sys.stderr,
        )
        return 3
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "3D grid-search momentum base parameters for trend portfolio via API."
        )
    )
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--lookback-start", type=int, default=5)
    p.add_argument("--lookback-end", type=int, default=60)
    p.add_argument("--lookback-step", type=int, default=10)
    p.add_argument("--entry-threshold-pct-start", type=float, default=0.0)
    p.add_argument("--entry-threshold-pct-end", type=float, default=20.0)
    p.add_argument("--entry-threshold-pct-step", type=float, default=5.0)
    p.add_argument("--exit-threshold-pct-start", type=float, default=-20.0)
    p.add_argument("--exit-threshold-pct-end", type=float, default=0.0)
    p.add_argument("--exit-threshold-pct-step", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument(
        "--max-in-flight",
        type=int,
        default=0,
        help="Max submitted-but-unfinished tasks. <=0 means workers*8.",
    )
    p.add_argument("--retry-times", type=int, default=1)
    p.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per parameter-combination total timeout in seconds (default: 300s = 5m).",
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional cap for smoke testing; <=0 means full grid.",
    )
    p.add_argument(
        "--objective-min-avg-annual-trade-count",
        type=float,
        default=0.0,
        help="Objective eligibility gate: avg annual trade count >= this value.",
    )
    p.add_argument(
        "--objective-min-annualized-return-pct",
        type=float,
        default=0.0,
        help="Objective eligibility gate: annualized return >= this percent value.",
    )
    p.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    p.add_argument("--output-csv", default="")
    p.add_argument("--output-best-csv", default="")
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
