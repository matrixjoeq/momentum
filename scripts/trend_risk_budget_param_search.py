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
    "src/etf_momentum/web/data/trend_risk_budget_param_search_results.json"
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

# User-provided payload baseline. Script only sweeps risk_budget_pct.
RAW_BASE_PAYLOAD: dict[str, Any] = {
    "codes": [
        "518800",
        "161226",
        "159980",
        "501018",
        "159981",
        "159985",
        "513500",
        "513100",
        "513030",
        "513080",
        "164824",
        "513520",
        "513310",
        "510300",
        "159907",
        "159915",
        "588000",
        "515180",
        "159920",
        "513690",
    ],
    "position_sizing": "risk_budget",
    "dynamic_universe": True,
    "start": "20110810",
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
    "group_name": "宽基",
    "backtest_range": {"start": "20110810", "end": "20260717"},
    "dynamic_universe": True,
    "mode": "portfolio",
    "selected_codes": RAW_BASE_PAYLOAD["codes"],
    "single_code": "518800",
    "position_sizing": "risk_budget",
    "asset_groups_text": "",
    "asset_groups_parse_error": None,
    "r_take_profit_tiers_text": "1:1,2:0.5,3:0.33,4:0.25,5:0.2,6:0.17,7:0.14,8:0.13,9:0.11,10:0.1",
    "r_take_profit_tiers_parse_error": None,
    "r_profit_scaleout_tiers_text": "2:0.33",
    "r_profit_scaleout_tiers_parse_error": None,
    "bias_v_take_profit_tiers_text": "3:0.33,5:0.33",
    "bias_v_take_profit_tiers_parse_error": None,
    "exported_at": "2026-07-19T09:42:08.381Z",
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


def _percent_to_ratio(v: Any) -> float:
    x = _to_float(v)
    if x is None:
        raise ValueError("percent value is invalid")
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
        out.get("risk_budget_pct"),
        default=0.01,
        ratio_ceiling=0.03,
    )
    out["vol_periodic_rebalance_threshold_pct"] = _normalize_percent_or_ratio(
        out.get("vol_periodic_rebalance_threshold_pct"),
        default=0.05,
        ratio_ceiling=1.0,
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


def _risk_budget_pct_percent(value: float) -> float:
    v = _to_float(value)
    if v is None:
        raise ValueError("risk_budget_pct percent is invalid")
    return float(v)


def _risk_budget_pct_decimal(value: float) -> float:
    return float(_percent_to_ratio(value))


def _search_space_percent_values(values: list[float]) -> list[float]:
    return [float(v) for v in values]


def _run_single_case(
    *,
    base_url: str,
    endpoint: str,
    base_payload: dict[str, Any],
    risk_budget_pct_input: float,
    timeout: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    risk_budget_pct = _risk_budget_pct_decimal(risk_budget_pct_input)
    payload = copy.deepcopy(base_payload)
    payload["risk_budget_pct"] = float(risk_budget_pct)
    risk_budget_pct_percent = _risk_budget_pct_percent(risk_budget_pct_input)
    url = _join_url(base_url, endpoint)
    try:
        resp = _http_post_json(url=url, payload=payload, timeout=timeout)
        if not isinstance(resp, dict):
            raise RuntimeError("unexpected response schema")
        metrics = _extract_metrics(resp)
        return {
            "risk_budget_pct_percent": float(risk_budget_pct_percent),
            "risk_budget_pct": float(risk_budget_pct),
            "status": "ok",
            "elapsed_ms": int((time.perf_counter() - t0) * 1000.0),
            "metrics": metrics,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        return {
            "risk_budget_pct_percent": float(risk_budget_pct_percent),
            "risk_budget_pct": float(risk_budget_pct),
            "status": "error",
            "elapsed_ms": int((time.perf_counter() - t0) * 1000.0),
            "metrics": {},
            "error": str(e),
        }


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "risk_budget_pct_percent",
        "risk_budget_pct",
        "objective_eligible",
        "objective_ineligible_reason",
        "composite_score",
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
                "risk_budget_pct_percent": row.get("risk_budget_pct_percent"),
                "risk_budget_pct": row.get("risk_budget_pct"),
                "objective_eligible": row.get("objective_eligible"),
                "objective_ineligible_reason": row.get("objective_ineligible_reason"),
                "composite_score": row.get("composite_score"),
                "status": row.get("status"),
                "elapsed_ms": row.get("elapsed_ms"),
                "error": row.get("error"),
            }
            for c in cols[7:-1]:
                out[c] = metrics.get(c) if isinstance(metrics, dict) else None
            w.writerow(out)


def run(args: argparse.Namespace) -> int:
    base_payload = _build_base_payload(RAW_BASE_PAYLOAD)
    grid = _grid_decimal_values(
        start=float(args.start_pct),
        end=float(args.end_pct),
        step=float(args.step_pct),
    )
    grid_percent = _search_space_percent_values(grid)
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
                risk_budget_pct_input=rb,
                timeout=float(args.timeout),
            ): rb
            for rb in grid
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
    rows.sort(key=lambda x: float(x.get("risk_budget_pct_percent") or 0.0))
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
                "risk_budget_pct_start": float(
                    _risk_budget_pct_decimal(args.start_pct)
                ),
                "risk_budget_pct_end": float(_risk_budget_pct_decimal(args.end_pct)),
                "risk_budget_pct_step": float(_risk_budget_pct_decimal(args.step_pct)),
                "risk_budget_pct_percent_start": float(
                    _risk_budget_pct_percent(args.start_pct)
                ),
                "risk_budget_pct_percent_end": float(
                    _risk_budget_pct_percent(args.end_pct)
                ),
                "risk_budget_pct_percent_step": float(
                    _risk_budget_pct_percent(args.step_pct)
                ),
                "values": grid,
                "values_percent": grid_percent,
            },
            "fixed_payload": {
                k: v for k, v in base_payload.items() if k != "risk_budget_pct"
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
            f"risk_budget_pct={best.get('risk_budget_pct_percent')}% "
            f"(decimal={best.get('risk_budget_pct')}), "
            f"composite={_to_float(best.get('composite_score'))}, "
            f"sharpe={_to_float((m or {}).get('sharpe_ratio'))}, "
            f"sqn_recent_100={_to_float((m or {}).get('sqn_recent_100'))}"
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
            f"risk_budget_pct={sample.get('risk_budget_pct_percent')}%, "
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
            "Grid-search risk_budget_pct for trend portfolio by calling backend API."
        )
    )
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument(
        "--start-pct",
        type=float,
        default=0.1,
        help="risk_budget_pct start (decimal or percent).",
    )
    p.add_argument(
        "--end-pct",
        type=float,
        default=1.0,
        help="risk_budget_pct end (decimal or percent).",
    )
    p.add_argument(
        "--step-pct",
        type=float,
        default=0.05,
        help="risk_budget_pct step (decimal or percent).",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=120.0)
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
