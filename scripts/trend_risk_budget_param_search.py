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
DEFAULT_ENDPOINT = "/api/analysis/trend/portfolio"
DEFAULT_OUTPUT_JSON = (
    "src/etf_momentum/web/data/trend_risk_budget_param_search_results.json"
)

# User-provided payload baseline. Script only sweeps risk_budget_pct.
RAW_BASE_PAYLOAD: dict[str, Any] = {
    "codes": [
        "159915",
        "159980",
        "159985",
        "161226",
        "162411",
        "501018",
        "510300",
        "513100",
        "513500",
        "513520",
        "518880",
        "588000",
        "159981",
        "563300",
        "510500",
        "513310",
    ],
    "position_sizing": "risk_budget",
    "dynamic_universe": True,
    "start": "20111209",
    "end": "20260618",
    "initial_account_amount": None,
    "cost_bps": 2,
    "slippage_rate": 0.001,
    "quick_mode": True,
    "engine": "legacy",
    "exec_price": "close",
    "strategy": "tsmom",
    "sma_window": 20,
    "fast_window": 5,
    "slow_window": 20,
    "ma_type": "ema",
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
    "er_filter": False,
    "er_window": 10,
    "er_threshold": 0.3,
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
    "atr_stop_mode": "none",
    "atr_stop_atr_basis": "latest",
    "atr_stop_reentry_mode": "reenter",
    "atr_stop_execution_mode": "intraday",
    "atr_stop_window": 14,
    "atr_stop_n": 2,
    "atr_stop_m": 0.5,
    "r_take_profit_enabled": False,
    "r_take_profit_reentry_mode": "reenter",
    "r_take_profit_execution_mode": "intraday",
    "r_take_profit_tiers": [
        {"r_multiple": 2, "retrace_ratio": 0.5},
        {"r_multiple": 3, "retrace_ratio": 0.3},
        {"r_multiple": 4, "retrace_ratio": 0.2},
        {"r_multiple": 5, "retrace_ratio": 0.1},
        {"r_multiple": 6, "retrace_ratio": 0.05},
    ],
    "r_profit_scaleout_enabled": False,
    "r_profit_scaleout_execution_mode": "intraday",
    "r_profit_scaleout_tiers": [
        {"r_multiple": 2, "reduce_fraction": 0.4},
        {"r_multiple": 3, "reduce_fraction": 0.3},
        {"r_multiple": 4, "reduce_fraction": 0.2},
    ],
    "bias_v_take_profit_enabled": False,
    "bias_v_take_profit_reentry_mode": "wait_next_entry",
    "bias_v_take_profit_execution_mode": "intraday",
    "bias_v_ma_window": 20,
    "bias_v_atr_window": 20,
    "bias_v_take_profit_tiers": [{"threshold": 5, "reduce_fraction": 0.5}],
    "monthly_risk_budget_enabled": False,
    "monthly_risk_budget_pct": 6,
    "monthly_risk_budget_include_new_trade_risk": False,
    "fixed_pos_ratio": 0.04,
    "fixed_overcap_policy": "skip",
    "fixed_max_holdings": 10,
    "risk_budget_atr_window": 20,
    "risk_budget_pct": 0.15,
    "risk_budget_overcap_policy": "scale",
    "risk_budget_max_leverage_multiple": 1,
    "vol_regime_risk_mgmt_enabled": False,
    "vol_ratio_fast_atr_window": 5,
    "vol_ratio_slow_atr_window": 50,
    "vol_ratio_expand_threshold": 1.4,
    "vol_ratio_contract_threshold": 0.7,
    "vol_ratio_normal_threshold": 1,
    "vol_ratio_extreme_threshold": 2.1,
    "risk_of_ruin_maxrisk": 30,
    "group_enforce": False,
    "group_pick_policy": "highest_sharpe",
    "group_max_holdings": 4,
    "asset_groups": {},
}

SEARCH_CONTEXT: dict[str, Any] = {
    "group_name": "趋势大全",
    "backtest_range": {"start": "20111209", "end": "20260618"},
    "dynamic_universe": True,
    "mode": "portfolio",
    "selected_codes": RAW_BASE_PAYLOAD["codes"],
    "single_code": "159915",
    "position_sizing": "risk_budget",
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


def _build_base_payload(raw_payload: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(raw_payload)
    out["monthly_risk_budget_pct"] = _normalize_ratio(
        out.get("monthly_risk_budget_pct"), default=0.06
    )
    out["risk_of_ruin_maxrisk"] = _normalize_ratio(
        out.get("risk_of_ruin_maxrisk"), default=0.30
    )
    out["risk_budget_pct"] = _normalize_ratio(out.get("risk_budget_pct"), default=0.01)
    return out


def _extract_metrics(resp: dict[str, Any]) -> dict[str, float | None]:
    metrics_block = resp.get("metrics") if isinstance(resp, dict) else None
    strategy = metrics_block.get("strategy") if isinstance(metrics_block, dict) else {}
    return {
        "cumulative_return": _to_float(strategy.get("cumulative_return")),
        "annualized_return": _to_float(strategy.get("annualized_return")),
        "annualized_volatility": _to_float(strategy.get("annualized_volatility")),
        "max_drawdown": _to_float(strategy.get("max_drawdown")),
        "max_drawdown_recovery_days": _to_float(
            strategy.get("max_drawdown_recovery_days")
        ),
        "sharpe_ratio": _to_float(strategy.get("sharpe_ratio")),
        "sortino_ratio": _to_float(strategy.get("sortino_ratio")),
        "calmar_ratio": _to_float(strategy.get("calmar_ratio")),
        "ulcer_index": _to_float(strategy.get("ulcer_index")),
        "ulcer_performance_index": _to_float(
            strategy.get("ulcer_performance_index")
        ),
    }


def _grid_percent_values(*, start: float, end: float, step: float) -> list[float]:
    s = _to_float(start)
    e = _to_float(end)
    k = _to_float(step)
    if s is None or e is None or k is None or k <= 0.0:
        raise ValueError("invalid grid range")
    lo = min(s, e)
    hi = max(s, e)
    scale = 10
    lo_i = int(round(lo * scale))
    hi_i = int(round(hi * scale))
    st_i = int(round(k * scale))
    if st_i <= 0:
        raise ValueError("invalid grid step")
    out = [float(i / scale) for i in range(lo_i, hi_i + 1, st_i)]
    if not out:
        raise ValueError("empty grid range")
    return out


def _run_single_case(
    *,
    base_url: str,
    endpoint: str,
    base_payload: dict[str, Any],
    risk_budget_pct_percent: float,
    timeout: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    risk_budget_pct = float(risk_budget_pct_percent / 100.0)
    payload = copy.deepcopy(base_payload)
    payload["risk_budget_pct"] = float(risk_budget_pct)
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


def _pick_best_by_sharpe(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: tuple[tuple[float, float, float, float], dict[str, Any]] | None = None
    for row in rows:
        if str(row.get("status") or "") != "ok":
            continue
        metrics = row.get("metrics") if isinstance(row, dict) else None
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
                "status": row.get("status"),
                "elapsed_ms": row.get("elapsed_ms"),
                "error": row.get("error"),
            }
            for c in cols[4:-1]:
                out[c] = metrics.get(c) if isinstance(metrics, dict) else None
            w.writerow(out)


def run(args: argparse.Namespace) -> int:
    base_payload = _build_base_payload(RAW_BASE_PAYLOAD)
    grid = _grid_percent_values(
        start=float(args.start_pct),
        end=float(args.end_pct),
        step=float(args.step_pct),
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
                risk_budget_pct_percent=rb,
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

    rows.sort(key=lambda x: float(x.get("risk_budget_pct_percent") or 0.0))
    best = _pick_best_by_sharpe(rows)
    errors = [x for x in rows if str(x.get("status") or "") != "ok"]

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
                "risk_budget_pct_percent_start": float(args.start_pct),
                "risk_budget_pct_percent_end": float(args.end_pct),
                "risk_budget_pct_percent_step": float(args.step_pct),
                "values": grid,
            },
            "fixed_payload": {
                k: v for k, v in base_payload.items() if k != "risk_budget_pct"
            },
            "objective": {"metric": "sharpe_ratio", "mode": "max"},
            "total_cases": len(rows),
            "success_cases": len(rows) - len(errors),
            "error_cases": len(errors),
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        },
        "results": rows,
        "best_single_setting": best,
        "errors": errors,
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
            "[INFO] best sharpe "
            f"risk_budget_pct={best.get('risk_budget_pct_percent')}% "
            f"(decimal={best.get('risk_budget_pct')}), "
            f"sharpe={_to_float((m or {}).get('sharpe_ratio'))}"
        )
    if errors:
        print(f"[WARN] error cases: {len(errors)}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Grid-search risk_budget_pct for trend portfolio by calling backend API."
        )
    )
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--start-pct", type=float, default=0.1, help="Percent, inclusive.")
    p.add_argument("--end-pct", type=float, default=3.0, help="Percent, inclusive.")
    p.add_argument("--step-pct", type=float, default=0.1, help="Percent step.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=120.0)
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
        return run(args)
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
