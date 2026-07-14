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
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_ENDPOINT = "/api/analysis/rotation"
DEFAULT_OUTPUT_JSON = (
    "src/etf_momentum/web/data/rotation_long_horizon_param_search_results.json"
)

# Fixed baseline payload from user snapshot.
# Variable parameters in this script:
#   - top_k in [1, 20]
#   - rebalance anchor in [1, 28] (monthly day-of-month anchor)
RAW_BASE_PAYLOAD: dict[str, Any] = {
    "codes": [
        "510300",
        "510500",
        "512100",
        "513100",
        "513500",
        "513520",
        "513030",
        "513980",
        "518800",
        "159980",
        "159981",
        "159985",
        "511090",
        "511260",
        "513800",
        "513080",
        "159907",
        "161226",
        "501018",
        "511010",
        "513400",
    ],
    "start": "20110810",
    "end": "20260710",
    "rebalance": "monthly",
    "rebalance_anchor": 14,
    "rebalance_shift": "next",
    "exec_price": "close",
    "top_k_mode": "fixed",
    "top_k": 6,
    "floating_benchmark_code": None,
    "position_mode": "inverse_vol",
    "risk_budget_atr_window": 20,
    "risk_budget_pct": 0.003,
    "entry_backfill": False,
    "daily_rebalance": False,
    "entry_match_n": 0,
    "exit_match_n": 0,
    "lookback_days": 242,
    "skip_days": 20,
    "cost_bps": 2.0,
    "slippage_rate": 0.001,
    "capacity_window_years": 1,
    "score_method": "raw_mom",
    "stop_scheme": "none",
    "equity_stop_risk_pct": 0.02,
    "atr_stop_mode": "static",
    "atr_stop_atr_basis": "latest",
    "atr_stop_execution_mode": "next_day",
    "atr_stop_execution_time": "close",
    "atr_stop_window": 20,
    "atr_stop_n": 2.0,
    "atr_stop_m": 0.5,
    "group_enforce": False,
    "group_pick_policy": "strongest_score",
    "asset_groups": {},
    "dynamic_universe": True,
    "inertia": False,
    "inertia_min_hold_periods": 0,
    "inertia_score_gap": 0.0,
    "inertia_min_turnover": 0.0,
    "rr_sizing": False,
    "rr_years": 3,
    "rr_thresholds": None,
    "rr_weights": None,
    "mirror_control": False,
    "mirror_quantiles": None,
    "mirror_exposures": None,
    "dd_control": False,
    "dd_threshold": 0.1,
    "dd_reduce": 1.0,
    "dd_sleep_days": 20,
    "trend_filter": False,
    "trend_exit_filter": False,
    "trend_sma_window": 20,
    "trend_ma_type": "sma",
    "bias_filter": False,
    "bias_exit_filter": False,
    "bias_type": "bias",
    "bias_ma_window": 20,
    "bias_level_window": "all",
    "bias_threshold_type": "quantile",
    "bias_quantile": 95.0,
    "bias_fixed_value": 10.0,
    "bias_min_periods": 20,
    "rsi_filter": False,
    "rsi_window": 14,
    "rsi_overbought": 70.0,
    "rsi_oversold": 30.0,
    "rsi_block_overbought": True,
    "rsi_block_oversold": False,
    "vol_monitor": False,
    "vol_window": 20,
    "vol_target_ann": 0.2,
    "vol_max_ann": 0.6,
    "chop_filter": False,
    "chop_mode": "er",
    "chop_window": 20,
    "chop_er_threshold": 0.25,
    "chop_adx_window": 20,
    "chop_adx_threshold": 20.0,
    "asset_trend_rules": [],
    "asset_bias_rules": [],
    "asset_momentum_floor_rules": None,
    "asset_rc_rules": None,
    "asset_vol_index_rules": None,
    "benchmark_mode": "EW_REBAL",
}

SEARCH_CONTEXT: dict[str, Any] = {
    "group_name": "动量配置",
    "backtest_range": {"start": "20110810", "end": "20260710"},
    "dynamic_universe": True,
    "selected_codes": RAW_BASE_PAYLOAD["codes"],
    "fixed_params_from_ui": {
        "rebalance": "monthly",
        "rebalance_shift": "next",
        "exec_price": "close",
        "top_k_mode": "fixed",
        "position_mode": "inverse_vol",
        "score_method": "raw_mom",
        "lookback_days": 242,
        "skip_days": 20,
        "cost_bps": 2,
        "slippage_rate": 0.001,
        "capacity_window_years": 1,
        "group_enforce": False,
        "group_pick_policy": "strongest_score",
        "dynamic_universe": True,
        "benchmark_mode": "EW_REBAL",
    },
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


def _grid_int_values(*, start: int, end: int, step: int) -> list[int]:
    s = int(start)
    e = int(end)
    k = int(step)
    if k <= 0:
        raise ValueError("integer grid step must be >= 1")
    lo = min(s, e)
    hi = max(s, e)
    out = [int(i) for i in range(lo, hi + 1, k)]
    if not out:
        raise ValueError("empty integer grid")
    return out


def _extract_metrics_from_rotation_response(resp: dict[str, Any]) -> dict[str, Any]:
    metrics_block = resp.get("metrics") if isinstance(resp, dict) else None
    strategy = metrics_block.get("strategy") if isinstance(metrics_block, dict) else {}
    if not isinstance(strategy, dict):
        strategy = {}
    out = dict(strategy)
    trade_statistics = resp.get("trade_statistics") if isinstance(resp, dict) else None
    overall = (
        trade_statistics.get("overall") if isinstance(trade_statistics, dict) else {}
    )
    if isinstance(overall, dict):
        out["win_rate"] = _to_float(overall.get("win_rate_ex_zero"))
        out["payoff_ratio"] = _to_float(overall.get("payoff_ex_zero"))
        out["kelly_fraction"] = _to_float(overall.get("kelly_ex_zero"))
    return out


def _run_single_case(
    *,
    base_url: str,
    endpoint: str,
    base_payload: dict[str, Any],
    top_k: int,
    anchor: int,
    timeout: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    payload = copy.deepcopy(base_payload)
    payload["top_k"] = int(top_k)
    payload["rebalance_anchor"] = int(anchor)
    payload["exec_price"] = "close"
    ep = str(endpoint or "").strip().lower()
    use_calendar_effect = ep.endswith("/analysis/rotation/calendar-effect")
    if use_calendar_effect:
        payload["anchors"] = [int(anchor)]
        payload["exec_prices"] = ["close"]
    url = _join_url(base_url, endpoint)
    try:
        resp = _http_post_json(url=url, payload=payload, timeout=timeout)
        if not isinstance(resp, dict):
            raise RuntimeError("unexpected response schema")
        elapsed_ms = int((time.perf_counter() - t0) * 1000.0)
        if use_calendar_effect:
            one: dict[str, Any] | None = None
            grid = resp.get("grid")
            if isinstance(grid, list):
                for item in grid:
                    if not isinstance(item, dict):
                        continue
                    if str(item.get("exec_price") or "").lower() != "close":
                        continue
                    try:
                        a = int(item.get("anchor"))
                    except (TypeError, ValueError):
                        continue
                    if int(a) == int(anchor):
                        one = item
                        break
            if one is None:
                return {
                    "top_k": int(top_k),
                    "rebalance_anchor": int(anchor),
                    "status": "error",
                    "elapsed_ms": elapsed_ms,
                    "metrics": {},
                    "error": "missing anchor result from calendar-effect response",
                }
            ok = bool(one.get("ok"))
            return {
                "top_k": int(top_k),
                "rebalance_anchor": int(anchor),
                "status": "ok" if ok else "error",
                "elapsed_ms": elapsed_ms,
                "metrics": dict(one.get("metrics") or {}) if ok else {},
                "error": None if ok else str(one.get("reason") or "unknown"),
            }

        metrics = _extract_metrics_from_rotation_response(resp)
        if not metrics:
            return {
                "top_k": int(top_k),
                "rebalance_anchor": int(anchor),
                "status": "error",
                "elapsed_ms": elapsed_ms,
                "metrics": {},
                "error": "missing metrics.strategy from rotation response",
            }
        return {
            "top_k": int(top_k),
            "rebalance_anchor": int(anchor),
            "status": "ok",
            "elapsed_ms": elapsed_ms,
            "metrics": metrics,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        elapsed_ms = int((time.perf_counter() - t0) * 1000.0)
        return {
            "top_k": int(top_k),
            "rebalance_anchor": int(anchor),
            "status": "error",
            "elapsed_ms": elapsed_ms,
            "metrics": {},
            "error": str(e),
        }


def _pick_best_by_sharpe(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: tuple[tuple[float, float, float, float], dict[str, Any]] | None = None
    for row in rows:
        if str(row.get("status") or "") != "ok":
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        sharpe = _to_float(metrics.get("sharpe_ratio"))
        if sharpe is None:
            continue
        ann = _to_float(metrics.get("annualized_return")) or -1e18
        cum_ret = _to_float(metrics.get("cumulative_return")) or -1e18
        mdd = _to_float(metrics.get("max_drawdown"))
        mdd_score = -abs(mdd) if mdd is not None else -1e18
        key = (float(sharpe), float(ann), float(cum_ret), float(mdd_score))
        if best is None:
            best = (key, row)
            continue
        if key > best[0]:
            best = (key, row)
    if best is None:
        return None
    return best[1]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "top_k",
        "rebalance_anchor",
        "status",
        "elapsed_ms",
        "sharpe_ratio",
        "annualized_return",
        "cumulative_return",
        "annualized_volatility",
        "max_drawdown",
        "max_drawdown_recovery_days",
        "sortino_ratio",
        "calmar_ratio",
        "ulcer_index",
        "ulcer_performance_index",
        "win_rate",
        "payoff_ratio",
        "kelly_fraction",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics") if isinstance(row, dict) else {}
            out = {
                "top_k": row.get("top_k"),
                "rebalance_anchor": row.get("rebalance_anchor"),
                "status": row.get("status"),
                "elapsed_ms": row.get("elapsed_ms"),
                "error": row.get("error"),
            }
            for col in cols[4:-1]:
                out[col] = metrics.get(col) if isinstance(metrics, dict) else None
            writer.writerow(out)


def run(args: argparse.Namespace) -> int:
    top_k_values = _grid_int_values(
        start=int(args.topk_start),
        end=int(args.topk_end),
        step=int(args.topk_step),
    )
    anchor_values = _grid_int_values(
        start=int(args.anchor_start),
        end=int(args.anchor_end),
        step=int(args.anchor_step),
    )
    if min(anchor_values) < 1 or max(anchor_values) > 28:
        raise ValueError("monthly rebalance_anchor must be in [1, 28]")

    base_payload = copy.deepcopy(RAW_BASE_PAYLOAD)
    total_cases = len(top_k_values) * len(anchor_values)
    print(
        "[INFO] "
        f"endpoint={args.endpoint}, top_k={top_k_values}, anchors={anchor_values}, "
        f"total_cases={total_cases}, workers={args.workers}"
    )
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    combos = [
        (int(top_k), int(anchor)) for top_k in top_k_values for anchor in anchor_values
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        fut_map = {
            ex.submit(
                _run_single_case,
                base_url=str(args.base_url),
                endpoint=str(args.endpoint),
                base_payload=base_payload,
                top_k=int(top_k),
                anchor=int(anchor),
                timeout=float(args.timeout),
            ): (int(top_k), int(anchor))
            for top_k, anchor in combos
        }
        done = 0
        for fut in concurrent.futures.as_completed(fut_map):
            done += 1
            one_row = fut.result()
            if isinstance(one_row, dict):
                rows.append(one_row)
            if done % 20 == 0 or done == len(combos):
                ok = sum(1 for x in rows if str(x.get("status") or "") == "ok")
                print(
                    f"[INFO] progress={done}/{len(combos)} cases, "
                    f"success={ok}/{total_cases}"
                )

    rows.sort(
        key=lambda x: (
            int(x.get("top_k") or 0),
            int(x.get("rebalance_anchor") or 0),
        )
    )
    errors = [x for x in rows if str(x.get("status") or "") != "ok"]
    best_overall = _pick_best_by_sharpe(rows)

    by_topk: dict[int, list[dict[str, Any]]] = {}
    by_anchor: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        tk = int(row.get("top_k") or 0)
        anchor = int(row.get("rebalance_anchor") or 0)
        by_topk.setdefault(tk, []).append(row)
        by_anchor.setdefault(anchor, []).append(row)
    best_by_topk = [
        {"top_k": tk, "best": _pick_best_by_sharpe(rs)}
        for tk, rs in sorted(by_topk.items())
    ]
    best_by_anchor = [
        {"rebalance_anchor": a, "best": _pick_best_by_sharpe(rs)}
        for a, rs in sorted(by_anchor.items())
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
                "top_k_start": int(args.topk_start),
                "top_k_end": int(args.topk_end),
                "top_k_step": int(args.topk_step),
                "top_k_values": top_k_values,
                "anchor_start": int(args.anchor_start),
                "anchor_end": int(args.anchor_end),
                "anchor_step": int(args.anchor_step),
                "anchor_values": anchor_values,
                "total_cases": int(total_cases),
            },
            "fixed_payload": {
                k: v
                for k, v in base_payload.items()
                if k not in {"top_k", "rebalance_anchor"}
            },
            "objective": {"metric": "sharpe_ratio", "mode": "max"},
            "total_cases": len(rows),
            "success_cases": len(rows) - len(errors),
            "error_cases": len(errors),
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        },
        "results": rows,
        "best_single_setting": best_overall,
        "best_by_topk": best_by_topk,
        "best_by_anchor": best_by_anchor,
        "errors": errors,
    }
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_csv, rows)
    _write_csv(
        output_best_csv, [best_overall] if isinstance(best_overall, dict) else []
    )

    print(f"[INFO] done in {time.perf_counter() - t0:.1f}s")
    print(f"[INFO] json: {output_json}")
    print(f"[INFO] csv: {output_csv}")
    print(f"[INFO] best: {output_best_csv}")
    print("[INFO] html: /static/rotation_long_horizon_param_search.html")
    if best_overall:
        metrics = best_overall.get("metrics") if isinstance(best_overall, dict) else {}
        print(
            "[INFO] best sharpe "
            f"top_k={best_overall.get('top_k')}, "
            f"rebalance_anchor={best_overall.get('rebalance_anchor')}, "
            f"sharpe={_to_float((metrics or {}).get('sharpe_ratio'))}"
        )
    if errors:
        print(f"[WARN] error cases: {len(errors)}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Long-horizon monthly rotation parameter search. Sweep top_k and "
            "rebalance_anchor, then save full 2D grid results for heatmap/frontier view."
        )
    )
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--topk-start", type=int, default=1)
    p.add_argument("--topk-end", type=int, default=20)
    p.add_argument("--topk-step", type=int, default=1)
    p.add_argument("--anchor-start", type=int, default=1)
    p.add_argument("--anchor-end", type=int, default=28)
    p.add_argument("--anchor-step", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    p.add_argument("--output-csv", default="")
    p.add_argument("--output-best-csv", default="")
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if int(args.workers) < 1:
        print("[ERROR] --workers must be >= 1", file=sys.stderr)
        return 2
    try:
        _grid_int_values(
            start=int(args.topk_start),
            end=int(args.topk_end),
            step=int(args.topk_step),
        )
        _grid_int_values(
            start=int(args.anchor_start),
            end=int(args.anchor_end),
            step=int(args.anchor_step),
        )
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        print(f"[ERROR] invalid search range: {e}", file=sys.stderr)
        return 2

    try:
        return run(args)
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
