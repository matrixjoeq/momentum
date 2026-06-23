#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_SYMBOLS = ["510300", "510500", "512100", "515180"]
DEFAULT_OUTPUT_JSON = (
    "src/etf_momentum/web/data/calendar_timing_param_search_results.json"
)
OBJECTIVE_CHOICES = {
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "max_drawdown_recovery_days",
    "sharpe_ratio",
    "ulcer_index",
    "ulcer_performance_index",
    "win_rate",
    "payoff_ratio",
    "kelly_fraction",
}


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _join_url(base_url: str, path: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("base_url is empty")
    return f"{base}{path if path.startswith('/') else '/' + path}"


def _http_json(
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url=url, data=data, method=method.upper())
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {e.code} {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"{method} {url} failed: {e}") from e


def _fetch_symbol_ranges(
    *,
    base_url: str,
    symbols: list[str],
    timeout: float,
) -> dict[str, tuple[str, str]]:
    url = _join_url(base_url, "/api/etf?adjust=none")
    rows = _http_json(method="GET", url=url, timeout=timeout)
    if not isinstance(rows, list):
        raise RuntimeError("unexpected response from /api/etf?adjust=none")
    need = set(symbols)
    out: dict[str, tuple[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = str(row.get("code") or "").strip()
        if code not in need:
            continue
        start = str(row.get("last_data_start_date") or "").strip()
        end = str(row.get("last_data_end_date") or "").strip()
        if len(start) == 8 and len(end) == 8:
            out[code] = (start, end)
    missing = [x for x in symbols if x not in out]
    if missing:
        raise RuntimeError(
            "missing date ranges for symbols in /api/etf response: "
            + ",".join(missing)
        )
    return out


def _extract_metrics(resp: dict[str, Any]) -> dict[str, float | None]:
    metrics_block = resp.get("metrics") if isinstance(resp, dict) else None
    strategy = metrics_block.get("strategy") if isinstance(metrics_block, dict) else {}
    trade_stats = resp.get("trade_statistics") if isinstance(resp, dict) else None
    overall = trade_stats.get("overall") if isinstance(trade_stats, dict) else {}
    return {
        "cumulative_return": _to_float(strategy.get("cumulative_return")),
        "annualized_return": _to_float(strategy.get("annualized_return")),
        "annualized_volatility": _to_float(strategy.get("annualized_volatility")),
        "max_drawdown": _to_float(strategy.get("max_drawdown")),
        "max_drawdown_recovery_days": _to_float(
            strategy.get("max_drawdown_recovery_days")
        ),
        "sharpe_ratio": _to_float(strategy.get("sharpe_ratio")),
        "ulcer_index": _to_float(strategy.get("ulcer_index")),
        "ulcer_performance_index": _to_float(
            strategy.get("ulcer_performance_index")
        ),
        "win_rate": _to_float(overall.get("win_rate_ex_zero")),
        "payoff_ratio": _to_float(overall.get("payoff_ex_zero")),
        "kelly_fraction": _to_float(overall.get("kelly_ex_zero")),
    }


def _run_single_case(
    *,
    base_url: str,
    symbol: str,
    start: str,
    end: str,
    decision_day: int,
    hold_days: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "mode": "single",
        "code": symbol,
        "codes": None,
        "start": start,
        "end": end,
        "adjust": "none",
        "decision_day": int(decision_day),
        "hold_days": int(hold_days),
        "position_mode": "equal",
        "fixed_pos_ratio": 1.0,
        "exec_price": "close",
        "cost_bps": 2.0,
        "slippage_rate": 0.001,
        "rebalance_shift": "next",
        "calendar": "XSHG",
        "dynamic_universe": False,
    }
    url = _join_url(base_url, "/api/analysis/calendar-timing")
    resp = _http_json(method="POST", url=url, payload=payload, timeout=timeout)
    if not isinstance(resp, dict):
        raise RuntimeError("unexpected response schema from /api/analysis/calendar-timing")
    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "decision_day": int(decision_day),
        "hold_days": int(hold_days),
        "metrics": _extract_metrics(resp),
    }


def _select_best(
    *,
    rows: list[dict[str, Any]],
    objective: str,
    objective_mode: str,
) -> dict[str, Any] | None:
    if objective not in OBJECTIVE_CHOICES:
        raise ValueError(f"unsupported objective: {objective}")
    maximize = str(objective_mode).strip().lower() == "max"

    valid = []
    for row in rows:
        metrics = row.get("metrics") if isinstance(row, dict) else None
        if not isinstance(metrics, dict):
            continue
        obj = _to_float(metrics.get(objective))
        if obj is None:
            continue
        sharpe = _to_float(metrics.get("sharpe_ratio")) or -1e18
        ann_ret = _to_float(metrics.get("annualized_return")) or -1e18
        mdd = _to_float(metrics.get("max_drawdown")) or -1e18
        cum_ret = _to_float(metrics.get("cumulative_return")) or -1e18
        key = (
            obj if maximize else -obj,
            sharpe,
            ann_ret,
            -(abs(mdd)),
            cum_ret,
        )
        valid.append((key, row))
    if not valid:
        return None
    valid.sort(key=lambda x: x[0], reverse=True)
    return valid[0][1]


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "symbol",
        "start",
        "end",
        "decision_day",
        "hold_days",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "max_drawdown_recovery_days",
        "sharpe_ratio",
        "ulcer_index",
        "ulcer_performance_index",
        "win_rate",
        "payoff_ratio",
        "kelly_fraction",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            metrics = row.get("metrics", {}) if isinstance(row, dict) else {}
            out = {
                "symbol": row.get("symbol"),
                "start": row.get("start"),
                "end": row.get("end"),
                "decision_day": row.get("decision_day"),
                "hold_days": row.get("hold_days"),
            }
            for k in fieldnames[5:]:
                out[k] = metrics.get(k) if isinstance(metrics, dict) else None
            w.writerow(out)


def _build_decision_days(min_day: int, max_day: int) -> list[int]:
    out: list[int] = []
    lo, hi = int(min_day), int(max_day)
    if lo > hi:
        lo, hi = hi, lo
    for d in range(lo, hi + 1):
        if d == 0:
            continue
        if d < -28 or d > 28:
            continue
        out.append(d)
    if not out:
        raise ValueError("decision day range has no valid values in [-28, 28] excluding 0")
    return out


def _build_hold_days(min_day: int, max_day: int) -> list[int]:
    lo, hi = int(min_day), int(max_day)
    if lo > hi:
        lo, hi = hi, lo
    out = [x for x in range(lo, hi + 1) if 1 <= x <= 252]
    if not out:
        raise ValueError("hold day range has no valid values in [1, 252]")
    return out


def run(args: argparse.Namespace) -> int:
    symbols = [
        x.strip()
        for x in str(args.symbols).split(",")
        if str(x).strip()
    ]
    if not symbols:
        symbols = list(DEFAULT_SYMBOLS)
    decision_days = _build_decision_days(args.decision_min, args.decision_max)
    hold_days = _build_hold_days(args.hold_min, args.hold_max)

    print(
        f"[INFO] symbols={symbols}, decision_days={len(decision_days)}, hold_days={len(hold_days)}"
    )
    ranges = _fetch_symbol_ranges(base_url=args.base_url, symbols=symbols, timeout=args.timeout)

    all_rows: list[dict[str, Any]] = []
    all_errors: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for symbol in symbols:
        start, end = ranges[symbol]
        grid = [(d, h) for d in decision_days for h in hold_days]
        total = len(grid)
        print(
            f"[INFO] {symbol}: date_range={start}..{end}, total_cases={total}, workers={args.workers}"
        )
        done = 0
        err_cnt = 0
        t_symbol = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut_map = {
                ex.submit(
                    _run_single_case,
                    base_url=args.base_url,
                    symbol=symbol,
                    start=start,
                    end=end,
                    decision_day=d,
                    hold_days=h,
                    timeout=args.timeout,
                ): (d, h)
                for d, h in grid
            }
            for fut in concurrent.futures.as_completed(fut_map):
                d, h = fut_map[fut]
                done += 1
                try:
                    all_rows.append(fut.result())
                except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                    err_cnt += 1
                    all_errors.append(
                        {
                            "symbol": symbol,
                            "start": start,
                            "end": end,
                            "decision_day": int(d),
                            "hold_days": int(h),
                            "error": str(e),
                        }
                    )
                if done % 100 == 0 or done == total:
                    elapsed = time.perf_counter() - t_symbol
                    print(
                        f"[INFO] {symbol}: progress={done}/{total}, errors={err_cnt}, elapsed={elapsed:.1f}s"
                    )

    best_by_symbol: dict[str, dict[str, Any] | None] = {}
    for symbol in symbols:
        rows = [x for x in all_rows if x.get("symbol") == symbol]
        best_by_symbol[symbol] = _select_best(
            rows=rows,
            objective=str(args.objective),
            objective_mode=str(args.objective_mode),
        )

    output_json = Path(str(args.output_json))
    output_csv = (
        Path(str(args.output_csv))
        if args.output_csv
        else output_json.with_suffix(".csv")
    )
    output_best_csv = (
        Path(str(args.output_best_csv))
        if args.output_best_csv
        else output_json.with_name(output_json.stem + "_best.csv")
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "api_base_url": str(args.base_url),
            "symbols": symbols,
            "symbol_date_ranges": ranges,
            "search_space": {
                "decision_day": decision_days,
                "hold_days": hold_days,
            },
            "fixed_params": {
                "mode": "single",
                "position_mode": "equal",
                "exec_price": "close",
                "rebalance_shift": "next",
                "cost_bps": 2.0,
                "slippage_rate": 0.001,
                "adjust": "none",
                "calendar": "XSHG",
                "dynamic_universe": False,
            },
            "objective": {
                "metric": str(args.objective),
                "mode": str(args.objective_mode),
            },
            "total_success_cases": len(all_rows),
            "total_error_cases": len(all_errors),
        },
        "results": all_rows,
        "best_by_symbol": best_by_symbol,
        "errors": all_errors,
    }

    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv_rows(output_csv, all_rows)
    _write_csv_rows(
        output_best_csv,
        [v for v in best_by_symbol.values() if isinstance(v, dict)],
    )

    elapsed = time.perf_counter() - t0
    print(f"[INFO] done in {elapsed:.1f}s")
    print(f"[INFO] json: {output_json}")
    print(f"[INFO] csv: {output_csv}")
    print(f"[INFO] best: {output_best_csv}")
    if all_errors:
        print(f"[WARN] error cases: {len(all_errors)}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Call /api/analysis/calendar-timing for calendar effect strategy "
            "parameter sweep and save results."
        )
    )
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Backend base URL, for example: http://127.0.0.1:8000",
    )
    p.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated symbols.",
    )
    p.add_argument("--decision-min", type=int, default=-28)
    p.add_argument("--decision-max", type=int, default=28)
    p.add_argument("--hold-min", type=int, default=1)
    p.add_argument("--hold-max", type=int, default=20)
    p.add_argument(
        "--objective",
        choices=sorted(OBJECTIVE_CHOICES),
        default="sharpe_ratio",
        help="Metric used to identify best params by symbol.",
    )
    p.add_argument(
        "--objective-mode",
        choices=["max", "min"],
        default="max",
        help="Whether larger objective is better.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent request workers for each symbol.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="HTTP timeout seconds for each request.",
    )
    p.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path.",
    )
    p.add_argument(
        "--output-csv",
        default="",
        help="Output CSV path; default is output-json with .csv suffix.",
    )
    p.add_argument(
        "--output-best-csv",
        default="",
        help="Output best-only CSV path; default is *_best.csv next to JSON.",
    )
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.workers < 1:
        print("[ERROR] --workers must be >= 1", file=sys.stderr)
        return 2
    try:
        return run(args)
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
