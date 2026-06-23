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
DEFAULT_ENDPOINT = "/api/analysis/calendar-timing"
DEFAULT_CODES = ["510300", "510500", "512100", "515180"]
DEFAULT_OUTPUT_JSON = (
    "src/etf_momentum/web/data/calendar_timing_risk_budget_param_search_results.json"
)


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
        "sortino_ratio": _to_float(strategy.get("sortino_ratio")),
        "calmar_ratio": _to_float(strategy.get("calmar_ratio")),
        "ulcer_index": _to_float(strategy.get("ulcer_index")),
        "ulcer_performance_index": _to_float(
            strategy.get("ulcer_performance_index")
        ),
        "win_rate": _to_float(overall.get("win_rate_ex_zero")),
        "payoff_ratio": _to_float(overall.get("payoff_ex_zero")),
        "kelly_fraction": _to_float(overall.get("kelly_ex_zero")),
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


def _parse_codes(text: str) -> list[str]:
    out = [x.strip() for x in str(text or "").split(",") if x.strip()]
    if not out:
        out = list(DEFAULT_CODES)
    return out


def _build_base_payload(args: argparse.Namespace) -> dict[str, Any]:
    mode = str(args.mode or "portfolio").strip().lower()
    codes = _parse_codes(str(args.codes or ""))
    code = str(args.code or "").strip()
    if mode == "single":
        if not code:
            code = codes[0]
        use_codes: list[str] | None = None
    elif mode == "portfolio":
        use_codes = codes
        code = ""
    else:
        raise ValueError("--mode only supports portfolio|single")
    payload: dict[str, Any] = {
        "mode": mode,
        "code": code if mode == "single" else None,
        "codes": use_codes if mode == "portfolio" else None,
        "start": str(args.start),
        "end": str(args.end),
        "adjust": "none",
        "decision_day": int(args.decision_day),
        "hold_days": int(args.hold_days),
        "position_mode": "risk_budget",
        "fixed_pos_ratio": float(args.fixed_pos_ratio),
        "risk_budget_atr_window": int(args.risk_budget_atr_window),
        "risk_budget_pct": 0.01,
        "dynamic_universe": bool(args.dynamic_universe),
        "exec_price": str(args.exec_price),
        "cost_bps": float(args.cost_bps),
        "slippage_rate": float(args.slippage_rate),
        "rebalance_shift": str(args.rebalance_shift),
        "calendar": str(args.calendar),
    }
    return payload


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
        return {
            "risk_budget_pct_percent": float(risk_budget_pct_percent),
            "risk_budget_pct": float(risk_budget_pct),
            "status": "ok",
            "elapsed_ms": int((time.perf_counter() - t0) * 1000.0),
            "metrics": _extract_metrics(resp),
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
        "win_rate",
        "payoff_ratio",
        "kelly_fraction",
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
    base_payload = _build_base_payload(args)
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
            rows.append(fut.result())
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

    search_context = {
        "mode": str(base_payload.get("mode") or ""),
        "single_code": base_payload.get("code"),
        "selected_codes": base_payload.get("codes"),
        "decision_day": int(base_payload["decision_day"]),
        "hold_days": int(base_payload["hold_days"]),
    }
    payload = {
        "meta": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "api_base_url": str(args.base_url),
            "api_endpoint": str(args.endpoint),
            "search_context": search_context,
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
            "Step-2 search for calendar timing: fix decision_day/hold_days and grid "
            "search risk_budget_pct by calling backend API."
        )
    )
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--mode", choices=["portfolio", "single"], default="portfolio")
    p.add_argument("--code", default="", help="Used only in --mode=single")
    p.add_argument("--codes", default=",".join(DEFAULT_CODES))
    p.add_argument("--start", default="20111209")
    p.add_argument("--end", default="20260618")
    p.add_argument("--decision-day", type=int, default=-5)
    p.add_argument("--hold-days", type=int, default=11)
    p.add_argument("--risk-budget-atr-window", type=int, default=20)
    p.add_argument(
        "--start-pct", type=float, default=0.1, help="Percent, inclusive."
    )
    p.add_argument("--end-pct", type=float, default=3.0, help="Percent, inclusive.")
    p.add_argument("--step-pct", type=float, default=0.1, help="Percent step.")
    p.add_argument("--exec-price", choices=["open", "close"], default="close")
    p.add_argument("--rebalance-shift", choices=["prev", "next", "skip"], default="next")
    p.add_argument("--calendar", default="XSHG")
    p.add_argument("--dynamic-universe", action="store_true")
    p.add_argument("--fixed-pos-ratio", type=float, default=1.0)
    p.add_argument("--cost-bps", type=float, default=2.0)
    p.add_argument("--slippage-rate", type=float, default=0.001)
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
    if int(args.hold_days) < 1:
        print("[ERROR] --hold-days must be >= 1", file=sys.stderr)
        return 2
    if int(args.decision_day) == 0 or int(args.decision_day) < -28 or int(args.decision_day) > 28:
        print("[ERROR] --decision-day must be in [-28,28] and not 0", file=sys.stderr)
        return 2
    try:
        return run(args)
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
