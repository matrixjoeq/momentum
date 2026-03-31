from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _to_num_or_none(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _dist_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "sample_size": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    s = pd.Series(values, dtype=float)
    return {
        "sample_size": int(len(s)),
        "mean": _to_num_or_none(s.mean()),
        "std": _to_num_or_none(s.std(ddof=0)),
        "min": _to_num_or_none(s.min()),
        "p05": _to_num_or_none(s.quantile(0.05)),
        "p25": _to_num_or_none(s.quantile(0.25)),
        "p50": _to_num_or_none(s.quantile(0.50)),
        "p75": _to_num_or_none(s.quantile(0.75)),
        "p95": _to_num_or_none(s.quantile(0.95)),
        "max": _to_num_or_none(s.max()),
    }


def _lookup_value(data: pd.Series | pd.DataFrame | None, when: pd.Timestamp, code: str | None = None) -> float:
    if data is None:
        return float("nan")
    if isinstance(data, pd.Series):
        if when in data.index:
            return float(pd.to_numeric(data.loc[when], errors="coerce"))
        return float("nan")
    if data.empty:
        return float("nan")
    if code is None or code not in data.columns or when not in data.index:
        return float("nan")
    return float(pd.to_numeric(data.loc[when, code], errors="coerce"))


def _group_code(trade: dict[str, Any], default_code: str | None) -> str:
    c = str(trade.get("code") or default_code or "").strip()
    return c if c else "_ALL_"


def _summarize_group(trades: list[dict[str, Any]]) -> dict[str, Any]:
    init_r = [float(t["initial_r_amount"]) for t in trades if t.get("initial_r_amount") is not None]
    init_r_pct = [float(t["initial_r_pct_nav"]) for t in trades if t.get("initial_r_pct_nav") is not None]
    r_mult = [float(t["r_multiple"]) for t in trades if t.get("r_multiple") is not None]
    out = {
        "trade_count": int(len(trades)),
        "valid_initial_r_count": int(len(init_r)),
        "valid_r_multiple_count": int(len(r_mult)),
        "initial_r_amount_stats": _dist_stats(init_r),
        "initial_r_pct_nav_stats": _dist_stats(init_r_pct),
        "r_multiple_stats": _dist_stats(r_mult),
        "tail": {
            "p_r_lt_m1": None,
            "p_r_gt_2": None,
            "expectancy_r": None,
        },
    }
    if r_mult:
        rs = pd.Series(r_mult, dtype=float)
        out["tail"]["p_r_lt_m1"] = _to_num_or_none((rs <= -1.0).mean())
        out["tail"]["p_r_gt_2"] = _to_num_or_none((rs >= 2.0).mean())
        out["tail"]["expectancy_r"] = _to_num_or_none(rs.mean())
    return out


def enrich_trades_with_r_metrics(
    trades: list[dict[str, Any]],
    *,
    nav: pd.Series,
    weights: pd.Series | pd.DataFrame,
    exec_price: pd.Series | pd.DataFrame,
    atr: pd.Series | pd.DataFrame | None,
    atr_mult: float,
    risk_budget_pct: float | None,
    cost_bps: float,
    slippage_rate: float,
    gap_buffer_rate: float = 0.0,
    default_code: str | None = None,
) -> dict[str, Any]:
    atr_n = float(atr_mult) if np.isfinite(float(atr_mult)) else float("nan")
    rb = float(risk_budget_pct) if (risk_budget_pct is not None and np.isfinite(float(risk_budget_pct))) else float("nan")
    c_bps = max(0.0, float(cost_bps)) if np.isfinite(float(cost_bps)) else 0.0
    slip = max(0.0, float(slippage_rate)) if np.isfinite(float(slippage_rate)) else 0.0
    gap = max(0.0, float(gap_buffer_rate)) if np.isfinite(float(gap_buffer_rate)) else 0.0

    nav_s = pd.to_numeric(nav, errors="coerce").astype(float)
    trades_out: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    eps = 1e-12

    for t in trades or []:
        tr = dict(t)
        code = _group_code(tr, default_code=default_code)
        entry_raw = tr.get("entry_date")
        entry_ts = pd.to_datetime(entry_raw) if entry_raw else pd.NaT
        if pd.isna(entry_ts):
            tr.update(
                {
                    "code": code if code != "_ALL_" else None,
                    "initial_r_amount": None,
                    "initial_r_pct_nav": None,
                    "pnl_amount": None,
                    "r_multiple": None,
                    "r_method": "unavailable",
                    "r_components": {
                        "price_risk_amount": None,
                        "cost_buffer_amount": None,
                        "slippage_buffer_amount": None,
                        "gap_buffer_amount": None,
                    },
                }
            )
            trades_out.append(tr)
            grouped.setdefault(code, []).append(tr)
            continue

        nav_entry = _lookup_value(nav_s, entry_ts)
        if (not np.isfinite(nav_entry)) or nav_entry <= 0.0:
            nav_entry = 1.0
        w_entry = abs(_lookup_value(weights, entry_ts, code if code != "_ALL_" else None))
        w_entry = w_entry if np.isfinite(w_entry) else 0.0
        px_entry = _lookup_value(exec_price, entry_ts, code if code != "_ALL_" else None)
        atr_entry = _lookup_value(atr, entry_ts, code if code != "_ALL_" else None)

        d_price = float("nan")
        method = "unavailable"
        if np.isfinite(atr_entry) and atr_entry > 0.0 and np.isfinite(px_entry) and px_entry > 0.0 and np.isfinite(atr_n) and atr_n > 0.0:
            d_price = float(atr_n) * float(atr_entry) / float(px_entry)
            method = "atr_proxy"
        elif np.isfinite(rb) and rb > 0.0 and w_entry > eps:
            d_price = float(rb) / float(max(w_entry, eps))
            method = "risk_budget_proxy"

        price_risk_amount = float("nan")
        if np.isfinite(d_price) and d_price >= 0.0:
            price_risk_amount = float(nav_entry) * float(w_entry) * float(d_price)
        cost_buffer = float(nav_entry) * float(w_entry) * (float(c_bps) / 10000.0)
        slippage_buffer = 0.0
        if np.isfinite(px_entry) and px_entry > 0.0:
            slippage_buffer = float(nav_entry) * float(w_entry) * (float(slip) / float(px_entry))
        gap_buffer = float(nav_entry) * float(w_entry) * float(gap)

        initial_r = float("nan")
        if np.isfinite(price_risk_amount):
            initial_r = float(price_risk_amount + cost_buffer + slippage_buffer + gap_buffer)
        initial_r_pct = float(initial_r / nav_entry) if np.isfinite(initial_r) and nav_entry > eps else float("nan")

        tr_ret = _to_num_or_none(tr.get("return"))
        pnl_amount = float("nan")
        if tr_ret is not None:
            pnl_amount = float(nav_entry) * float(tr_ret)
        r_mult = float("nan")
        if np.isfinite(pnl_amount) and np.isfinite(initial_r) and initial_r > eps:
            r_mult = float(pnl_amount / initial_r)

        tr.update(
            {
                "code": code if code != "_ALL_" else None,
                "initial_r_amount": _to_num_or_none(initial_r),
                "initial_r_pct_nav": _to_num_or_none(initial_r_pct),
                "pnl_amount": _to_num_or_none(pnl_amount),
                "r_multiple": _to_num_or_none(r_mult),
                "r_method": method,
                "r_components": {
                    "price_risk_amount": _to_num_or_none(price_risk_amount),
                    "cost_buffer_amount": _to_num_or_none(cost_buffer),
                    "slippage_buffer_amount": _to_num_or_none(slippage_buffer),
                    "gap_buffer_amount": _to_num_or_none(gap_buffer),
                },
            }
        )
        trades_out.append(tr)
        grouped.setdefault(code, []).append(tr)

    by_code = {k: _summarize_group(v) for k, v in grouped.items()}
    overall = _summarize_group(trades_out)
    return {
        "trades": trades_out,
        "statistics": {
            "overall": overall,
            "by_code": by_code,
        },
    }
