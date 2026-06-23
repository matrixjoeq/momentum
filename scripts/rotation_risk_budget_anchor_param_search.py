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
DEFAULT_ENDPOINT = "/api/analysis/rotation/calendar-effect"
DEFAULT_OUTPUT_JSON = (
    "src/etf_momentum/web/data/rotation_risk_budget_anchor_search_results.json"
)
DEFAULT_ANCHORS = [1, 2, 3, 4, 5]
DEFAULT_TOPK_VALUES = [1, 2, 3]

# Keep exactly the same fixed API parameters from the user payload.
RAW_BASE_PAYLOAD: dict[str, Any] = {
    "codes": [
        "159915",
        "159980",
        "159981",
        "159985",
        "161226",
        "501018",
        "511090",
        "513100",
        "513310",
        "513520",
        "518880",
        "563300",
        "588000",
        "162411",
        "511010",
        "511260",
        "513030",
        "513980",
    ],
    "start": "20111209",
    "end": "20260622",
    "rebalance": "weekly",
    "rebalance_anchor": 4,
    "rebalance_shift": "next",
    "exec_price": "close",
    "top_k_mode": "fixed",
    "top_k": 3,
    "floating_benchmark_code": None,
    "position_mode": "risk_budget",
    "risk_budget_atr_window": 20,
    "risk_budget_pct": 0.002,
    "lookback_days": 20,
    "skip_days": 0,
    "cost_bps": 2,
    "slippage_rate": 0.001,
    "benchmark_mode": "EW_REBAL",
    "dynamic_universe": True,
    "group_enforce": False,
    "group_pick_policy": "strongest_score",
    "asset_groups": {},
    "asset_momentum_floor_rules": None,
    "asset_trend_rules": None,
    "asset_bias_rules": None,
    "asset_rsi_rules": None,
    "asset_vol_monitor_rules": None,
    "asset_chop_rules": None,
    "asset_vol_index_rules": None,
}

SEARCH_CONTEXT: dict[str, Any] = {
    "group_name": "相对动量精选",
    "backtest_range": {"start": "20111209", "end": "20260622"},
    "dynamic_universe": True,
    "selected_codes": RAW_BASE_PAYLOAD["codes"],
    "ui_rebalance": "weekly",
    "position_mode": "risk_budget",
    "fixed_params_from_ui": {
        "top_k_mode": "fixed",
        "risk_budget_atr_window": 20,
        "lookback_days": 20,
        "skip_days": 0,
        "cost_bps": 2,
        "slippage_rate": 0.001,
        "exec_price": "close",
        "rebalance_shift": "next",
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


def _grid_int_values(*, start: int, end: int, step: int) -> list[int]:
    s = int(start)
    e = int(end)
    k = int(step)
    if k <= 0:
        raise ValueError("invalid integer grid step")
    lo = min(s, e)
    hi = max(s, e)
    out = [int(i) for i in range(lo, hi + 1, k)]
    if not out:
        raise ValueError("empty integer grid range")
    return out


def _parse_anchors(text: str) -> list[int]:
    out: list[int] = []
    for x in str(text or "").split(","):
        s = x.strip()
        if not s:
            continue
        v = int(s)
        if v < 1 or v > 5:
            raise ValueError("rebalance_anchor must be in [1,5]")
        if v not in out:
            out.append(v)
    if not out:
        out = list(DEFAULT_ANCHORS)
    return out


def _build_base_payload(raw_payload: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(raw_payload)
    rb = _to_float(out.get("risk_budget_pct"))
    if rb is None:
        out["risk_budget_pct"] = 0.002
    else:
        out["risk_budget_pct"] = float(rb / 100.0) if rb > 1.0 else float(rb)
    return out


def _run_single_case(
    *,
    base_url: str,
    endpoint: str,
    base_payload: dict[str, Any],
    rebalance_anchors: list[int],
    top_k_value: int,
    risk_budget_pct_percent: float,
    timeout: float,
) -> list[dict[str, Any]]:
    t0 = time.perf_counter()
    risk_budget_pct = float(risk_budget_pct_percent / 100.0)
    payload = copy.deepcopy(base_payload)
    payload["anchors"] = [int(x) for x in rebalance_anchors]
    payload["exec_prices"] = ["close"]
    payload["top_k"] = int(top_k_value)
    payload["risk_budget_pct"] = float(risk_budget_pct)
    url = _join_url(base_url, endpoint)
    try:
        resp = _http_post_json(url=url, payload=payload, timeout=timeout)
        if not isinstance(resp, dict):
            raise RuntimeError("unexpected response schema")
        elapsed_ms = int((time.perf_counter() - t0) * 1000.0)
        grid = resp.get("grid") if isinstance(resp, dict) else None
        by_anchor: dict[int, dict[str, Any]] = {}
        if isinstance(grid, list):
            for item in grid:
                if not isinstance(item, dict):
                    continue
                if str(item.get("exec_price") or "").lower() != "close":
                    continue
                anchor_v = item.get("anchor")
                try:
                    anchor = int(anchor_v)
                except (TypeError, ValueError):
                    continue
                by_anchor[int(anchor)] = item
        rows: list[dict[str, Any]] = []
        for anchor in rebalance_anchors:
            one = by_anchor.get(int(anchor))
            if one is None:
                rows.append(
                    {
                        "top_k": int(top_k_value),
                        "rebalance_anchor": int(anchor),
                        "risk_budget_pct_percent": float(risk_budget_pct_percent),
                        "risk_budget_pct": float(risk_budget_pct),
                        "status": "error",
                        "elapsed_ms": elapsed_ms,
                        "metrics": {},
                        "error": "missing anchor result in calendar-effect response",
                    }
                )
                continue
            ok = bool(one.get("ok"))
            rows.append(
                {
                    "top_k": int(top_k_value),
                    "rebalance_anchor": int(anchor),
                    "risk_budget_pct_percent": float(risk_budget_pct_percent),
                    "risk_budget_pct": float(risk_budget_pct),
                    "status": "ok" if ok else "error",
                    "elapsed_ms": elapsed_ms,
                    "metrics": (
                        one.get("metrics")
                        if ok and isinstance(one.get("metrics"), dict)
                        else {}
                    ),
                    "error": None if ok else str(one.get("reason") or "unknown"),
                }
            )
        return rows
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        elapsed_ms = int((time.perf_counter() - t0) * 1000.0)
        return [
            {
                "top_k": int(top_k_value),
                "rebalance_anchor": int(anchor),
                "risk_budget_pct_percent": float(risk_budget_pct_percent),
                "risk_budget_pct": float(risk_budget_pct),
                "status": "error",
                "elapsed_ms": elapsed_ms,
                "metrics": {},
                "error": str(e),
            }
            for anchor in rebalance_anchors
        ]


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


def _pick_robust_by_sharpe_mdd(
    rows: list[dict[str, Any]], *, top_k: int, mdd_limit_pct: float
) -> dict[str, Any]:
    ok_rows = [r for r in rows if str(r.get("status") or "") == "ok"]
    sorted_by_sharpe = sorted(
        ok_rows,
        key=lambda r: (
            _to_float(((r.get("metrics") or {}).get("sharpe_ratio"))) or -1e18,
            _to_float(((r.get("metrics") or {}).get("annualized_return"))) or -1e18,
        ),
        reverse=True,
    )
    top_rows = sorted_by_sharpe[: max(1, int(top_k))]
    mdd_limit = abs(float(mdd_limit_pct)) / 100.0
    robust_rows = []
    for row in top_rows:
        mdd = _to_float(((row.get("metrics") or {}).get("max_drawdown")))
        if mdd is None:
            continue
        if abs(mdd) <= mdd_limit:
            robust_rows.append(row)
    if not robust_rows:
        return {
            "candidate_count": 0,
            "interval_percent": None,
            "candidate_settings": [],
        }
    xs = [float(r.get("risk_budget_pct_percent") or 0.0) for r in robust_rows]
    interval = {"min": float(min(xs)), "max": float(max(xs))}
    candidates = [
        {
            "risk_budget_pct_percent": float(r.get("risk_budget_pct_percent") or 0.0),
            "risk_budget_pct": float(r.get("risk_budget_pct") or 0.0),
            "sharpe_ratio": _to_float(((r.get("metrics") or {}).get("sharpe_ratio"))),
            "annualized_return": _to_float(
                ((r.get("metrics") or {}).get("annualized_return"))
            ),
            "max_drawdown": _to_float(((r.get("metrics") or {}).get("max_drawdown"))),
        }
        for r in robust_rows
    ]
    return {
        "candidate_count": len(candidates),
        "interval_percent": interval,
        "candidate_settings": candidates,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "top_k",
        "rebalance_anchor",
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
                "top_k": row.get("top_k"),
                "rebalance_anchor": row.get("rebalance_anchor"),
                "risk_budget_pct_percent": row.get("risk_budget_pct_percent"),
                "risk_budget_pct": row.get("risk_budget_pct"),
                "status": row.get("status"),
                "elapsed_ms": row.get("elapsed_ms"),
                "error": row.get("error"),
            }
            for c in cols[5:-1]:
                out[c] = metrics.get(c) if isinstance(metrics, dict) else None
            w.writerow(out)


def run(args: argparse.Namespace) -> int:
    base_payload = _build_base_payload(RAW_BASE_PAYLOAD)
    anchors = _parse_anchors(str(args.anchors))
    top_k_values = _grid_int_values(
        start=int(args.topk_start),
        end=int(args.topk_end),
        step=int(args.topk_step),
    )
    grid = _grid_percent_values(
        start=float(args.start_pct),
        end=float(args.end_pct),
        step=float(args.step_pct),
    )

    total_cases = len(top_k_values) * len(anchors) * len(grid)
    print(
        "[INFO] "
        f"endpoint={args.endpoint}, top_k_values={top_k_values}, anchors={anchors}, "
        f"grid={len(grid)} points, total_cases={total_cases}, workers={args.workers}"
    )
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        fut_map = {}
        for tk in top_k_values:
            for rb in grid:
                fut = ex.submit(
                    _run_single_case,
                    base_url=str(args.base_url),
                    endpoint=str(args.endpoint),
                    base_payload=base_payload,
                    rebalance_anchors=anchors,
                    top_k_value=int(tk),
                    risk_budget_pct_percent=float(rb),
                    timeout=float(args.timeout),
                )
                fut_map[fut] = (tk, rb)
        done = 0
        total_calls = len(top_k_values) * len(grid)
        for fut in concurrent.futures.as_completed(fut_map):
            done += 1
            one_rows = fut.result()
            if isinstance(one_rows, list):
                rows.extend(one_rows)
            if done % 5 == 0 or done == total_calls:
                ok = sum(1 for x in rows if str(x.get("status") or "") == "ok")
                print(
                    f"[INFO] progress={done}/{total_calls} risk-budgets, "
                    f"success={ok}/{total_cases}"
                )

    rows.sort(
        key=lambda x: (
            int(x.get("top_k") or 0),
            int(x.get("rebalance_anchor") or 0),
            float(x.get("risk_budget_pct_percent") or 0.0),
        )
    )
    errors = [x for x in rows if str(x.get("status") or "") != "ok"]

    by_topk_anchor: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        tk = int(row.get("top_k") or 0)
        a = int(row.get("rebalance_anchor") or 0)
        key = (tk, a)
        by_topk_anchor.setdefault(key, []).append(row)
    best_overall = _pick_best_by_sharpe(rows)
    best_by_topk_anchor = []
    robust_by_topk_anchor = []
    for tk, anchor in sorted(by_topk_anchor.keys()):
        one_rows = by_topk_anchor[(tk, anchor)]
        best = _pick_best_by_sharpe(one_rows)
        robust = _pick_robust_by_sharpe_mdd(
            one_rows,
            top_k=int(args.robust_top_k),
            mdd_limit_pct=float(args.robust_mdd_limit_pct),
        )
        best_by_topk_anchor.append(
            {"top_k": int(tk), "rebalance_anchor": int(anchor), "best": best}
        )
        robust_by_topk_anchor.append(
            {
                "top_k": int(tk),
                "rebalance_anchor": int(anchor),
                "rule": {
                    "top_k_by_sharpe": int(args.robust_top_k),
                    "max_drawdown_limit_pct": float(args.robust_mdd_limit_pct),
                },
                **robust,
            }
        )

    best_rows_for_csv = [
        {
            **(x.get("best") or {}),
            "top_k": int(x["top_k"]),
            "rebalance_anchor": int(x["rebalance_anchor"]),
        }
        for x in best_by_topk_anchor
        if isinstance((x.get("best") or {}), dict)
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
        else output_json.with_name(output_json.stem + "_best_by_topk_anchor.csv")
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "api_base_url": str(args.base_url),
            "api_endpoint": str(args.endpoint),
            "search_context": SEARCH_CONTEXT,
            "search_space": {
                "top_k_values": top_k_values,
                "top_k_start": int(args.topk_start),
                "top_k_end": int(args.topk_end),
                "top_k_step": int(args.topk_step),
                "rebalance_anchors": anchors,
                "risk_budget_pct_percent_start": float(args.start_pct),
                "risk_budget_pct_percent_end": float(args.end_pct),
                "risk_budget_pct_percent_step": float(args.step_pct),
                "risk_budget_values_percent": grid,
            },
            "fixed_payload": {
                k: v
                for k, v in base_payload.items()
                if k not in {"risk_budget_pct", "rebalance_anchor", "top_k"}
            },
            "objective": {"metric": "sharpe_ratio", "mode": "max"},
            "robust_rule_default": {
                "top_k_by_sharpe": int(args.robust_top_k),
                "max_drawdown_limit_pct": float(args.robust_mdd_limit_pct),
            },
            "total_cases": len(rows),
            "success_cases": len(rows) - len(errors),
            "error_cases": len(errors),
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        },
        "results": rows,
        "best_single_setting": best_overall,
        "best_by_topk_anchor": best_by_topk_anchor,
        "robust_by_topk_anchor": robust_by_topk_anchor,
        "errors": errors,
    }
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output_csv, rows)
    _write_csv(output_best_csv, best_rows_for_csv)

    print(f"[INFO] done in {time.perf_counter() - t0:.1f}s")
    print(f"[INFO] json: {output_json}")
    print(f"[INFO] csv: {output_csv}")
    print(f"[INFO] best-by-topk-anchor: {output_best_csv}")
    if best_overall:
        m = best_overall.get("metrics") if isinstance(best_overall, dict) else {}
        print(
            "[INFO] best overall "
            f"top_k={best_overall.get('top_k')}, "
            f"anchor={best_overall.get('rebalance_anchor')}, "
            f"risk_budget_pct={best_overall.get('risk_budget_pct_percent')}% "
            f"(decimal={best_overall.get('risk_budget_pct')}), "
            f"sharpe={_to_float((m or {}).get('sharpe_ratio'))}"
        )
    if errors:
        print(f"[WARN] error cases: {len(errors)}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Grid-search rotation risk_budget_pct under risk_budget position mode "
            "for rebalance_anchor in [1..5], by calling backend API."
        )
    )
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--anchors", default="1,2,3,4,5")
    p.add_argument("--topk-start", type=int, default=DEFAULT_TOPK_VALUES[0])
    p.add_argument("--topk-end", type=int, default=DEFAULT_TOPK_VALUES[-1])
    p.add_argument("--topk-step", type=int, default=1)
    p.add_argument("--start-pct", type=float, default=0.1, help="Percent, inclusive.")
    p.add_argument("--end-pct", type=float, default=3.0, help="Percent, inclusive.")
    p.add_argument("--step-pct", type=float, default=0.1, help="Percent step.")
    p.add_argument("--robust-top-k", type=int, default=3)
    p.add_argument("--robust-mdd-limit-pct", type=float, default=20.0)
    p.add_argument("--workers", type=int, default=2)
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
    if int(args.robust_top_k) < 1:
        print("[ERROR] --robust-top-k must be >= 1", file=sys.stderr)
        return 2
    try:
        _grid_int_values(
            start=int(args.topk_start), end=int(args.topk_end), step=int(args.topk_step)
        )
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] invalid top-k range: {e}", file=sys.stderr)
        return 2
    try:
        _parse_anchors(str(args.anchors))
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] invalid --anchors: {e}", file=sys.stderr)
        return 2
    try:
        return run(args)
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
