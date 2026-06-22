"""
Futures rotation backtest for research page.

Semantics:
- Universe prices use synthetic hfq continuous series ({root}889), same as futures trend.
- Decision is made on day t, target weights execute on t+1 at chosen exec_price.
- trade_direction:
  - long_only: momentum high -> low, positive weights.
  - short_only: momentum low -> high, negative weights.
- position_mode:
  - equal
  - inverse_vol (fixed rolling volatility window, with equal-weight fallback).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..db.futures_repo import get_futures_pool_by_code
from ..db.futures_research_repo import FuturesGroupData
from .futures_lot_account import simulate_discrete_lot_portfolio
from .futures_trend import (
    CostProfile,
    _align_futures_trend_inputs,
    _build_cost_profile,
    _coerce_trading_index,
    _combine_group_returns,
    _safe_rate,
    _trade_stats_from_returns,
)

ExecPrice = Literal["open", "close"]
FeeSide = Literal["one_way", "two_way"]
SlippageType = Literal["percent", "price_spread", "tick_multiple"]
Rebalance = Literal["daily", "weekly", "monthly"]
TradeDirection = Literal["long_only", "short_only"]
PositionMode = Literal["equal", "inverse_vol"]

_SQRT_252 = float(np.sqrt(252.0))


def _series_rows(series: pd.Series) -> list[dict[str, Any]]:
    return [
        {"date": pd.Timestamp(d).date().isoformat(), "value": float(v)}
        for d, v in series.items()
        if pd.notna(v)
    ]


def _rebalance_decision_indices(
    dates: pd.DatetimeIndex, *, rebalance: Rebalance
) -> list[int]:
    if len(dates) == 0:
        return []
    rb = str(rebalance or "weekly").strip().lower()
    if rb == "daily":
        return list(range(len(dates)))
    freq = "W-FRI" if rb == "weekly" else "M"
    labels = dates.to_period(freq)
    idxer = pd.Series(np.arange(len(dates)), index=dates)
    return [int(i) for i in idxer.groupby(labels).max().tolist()]


def _momentum_scores(close_df: pd.DataFrame, *, lookback_days: int) -> pd.DataFrame:
    lb = max(1, int(lookback_days))
    return close_df.astype(float) / close_df.astype(float).shift(lb) - 1.0


def _annualized_vol(close_df: pd.DataFrame, *, window: int) -> pd.DataFrame:
    w = max(2, int(window))
    minp = max(3, w // 2)
    ret = close_df.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    return ret.rolling(window=w, min_periods=minp).std(ddof=1) * _SQRT_252


def _normalize_inverse_vol(
    vol_row: pd.Series, picks: list[str]
) -> tuple[dict[str, float], bool]:
    iv_map: dict[str, float] = {}
    for c in picks:
        v = float(vol_row.get(c)) if c in vol_row.index else float("nan")
        iv_map[str(c)] = (1.0 / v) if (np.isfinite(v) and v > 0.0) else 0.0
    iv_sum = float(sum(iv_map.values()))
    if iv_sum > 0.0:
        return ({c: float(iv_map[c] / iv_sum) for c in picks}, False)
    per = 1.0 / float(len(picks))
    return ({c: float(per) for c in picks}, True)


def _build_target_weights(
    *,
    dates: pd.DatetimeIndex,
    scores_df: pd.DataFrame,
    vol_df: pd.DataFrame | None,
    columns: list[str],
    top_k: int,
    trade_direction: TradeDirection,
    position_mode: PositionMode,
    rebalance: Rebalance,
    lookback_days: int,
    inverse_vol_window: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    w_tgt = pd.DataFrame(0.0, index=dates, columns=columns, dtype=float)
    if len(dates) == 0:
        return w_tgt, {"decision_count": 0, "effective_top_k_max": 0}
    dec_idx = _rebalance_decision_indices(dates, rebalance=rebalance)
    dec_set = {dates[i] for i in dec_idx}
    warmup_days = max(
        int(lookback_days),
        int(inverse_vol_window) if str(position_mode) == "inverse_vol" else 0,
    )
    topk_eff_max = 0
    fallback_inverse_vol_count = 0
    current = {c: 0.0 for c in columns}
    ascending = str(trade_direction) == "short_only"
    sign = -1.0 if ascending else 1.0
    for i, d in enumerate(dates):
        if d in dec_set:
            scores = (
                scores_df.loc[d]
                .astype(float)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if i + 1 < warmup_days:
                scores = pd.Series(dtype=float)
            ordered = scores.sort_values(ascending=ascending)
            picks = [str(x) for x in ordered.index[: int(min(int(top_k), len(ordered)))]]
            topk_eff_max = max(topk_eff_max, len(picks))
            next_map = {c: 0.0 for c in columns}
            if picks:
                if str(position_mode) == "inverse_vol" and vol_df is not None:
                    alloc, used_fallback = _normalize_inverse_vol(vol_df.loc[d], picks)
                    if used_fallback:
                        fallback_inverse_vol_count += 1
                else:
                    per = 1.0 / float(len(picks))
                    alloc = {c: float(per) for c in picks}
                for c, w in alloc.items():
                    next_map[str(c)] = float(sign * w)
            current = next_map
        w_tgt.loc[d] = [float(current.get(c, 0.0)) for c in columns]
    return w_tgt, {
        "decision_count": int(len(dec_idx)),
        "decision_dates": [dates[i].date().isoformat() for i in dec_idx],
        "effective_top_k_max": int(topk_eff_max),
        "inverse_vol_equal_fallback_count": int(fallback_inverse_vol_count),
    }


def compute_futures_group_rotation_backtest(
    db: Session,
    *,
    group: FuturesGroupData,
    start: str,
    end: str,
    dynamic_universe: bool,
    rebalance: Rebalance = "weekly",
    lookback_days: int = 20,
    top_k: int = 1,
    trade_direction: TradeDirection = "long_only",
    position_mode: PositionMode = "equal",
    inverse_vol_window: int = 20,
    exec_price: ExecPrice = "close",
    position_size_pct: float = 1.0,
    min_points: int = 120,
    cost_bps: float = 4.0,
    fee_side: FeeSide = "one_way",
    slippage_type: SlippageType = "tick_multiple",
    slippage_value: float = 1.0,
    slippage_side: FeeSide = "one_way",
    account_capital_wan: float = 500.0,
    backtest_margin_rate_pct: float = 15.0,
    reserve_margin_ratio: float = 0.5,
) -> dict[str, Any]:
    codes = [str(c).strip().upper() for c in group.codes if str(c).strip()]
    if len(codes) == 0:
        return {"ok": False, "error": "empty_group", "meta": {"group_name": group.name}}
    rb = str(rebalance or "weekly").strip().lower()
    if rb not in {"daily", "weekly", "monthly"}:
        return {"ok": False, "error": "invalid_rebalance"}
    td = str(trade_direction or "long_only").strip().lower()
    if td not in {"long_only", "short_only"}:
        return {"ok": False, "error": "invalid_trade_direction"}
    pm = str(position_mode or "equal").strip().lower()
    if pm not in {"equal", "inverse_vol"}:
        return {"ok": False, "error": "invalid_position_mode"}
    ep = str(exec_price or "close").strip().lower()
    if ep not in {"open", "close"}:
        return {"ok": False, "error": "invalid_exec_price"}
    if fee_side not in {"one_way", "two_way"}:
        return {"ok": False, "error": "invalid_fee_side"}
    if slippage_side not in {"one_way", "two_way"}:
        return {"ok": False, "error": "invalid_slippage_side"}
    st = str(slippage_type or "tick_multiple").strip().lower()
    if st not in {"percent", "price_spread", "tick_multiple"}:
        return {"ok": False, "error": "invalid_slippage_type"}
    if float(position_size_pct) <= 0.0 or float(position_size_pct) > 1.0:
        return {"ok": False, "error": "invalid_position_size_pct"}
    bmr = float(backtest_margin_rate_pct)
    if (not np.isfinite(bmr)) or bmr <= 0.0 or bmr > 100.0:
        return {"ok": False, "error": "invalid_backtest_margin_rate_pct"}
    rmr = float(reserve_margin_ratio)
    if (not np.isfinite(rmr)) or rmr < 0.0 or rmr >= 1.0:
        return {"ok": False, "error": "invalid_reserve_margin_ratio"}
    if int(top_k) <= 0:
        return {"ok": False, "error": "invalid_top_k"}
    if int(lookback_days) < 2:
        return {"ok": False, "error": "invalid_lookback_days"}
    if int(inverse_vol_window) < 2:
        return {"ok": False, "error": "invalid_inverse_vol_window"}
    if float(account_capital_wan) <= 0.0:
        return {"ok": False, "error": "invalid_account_capital_wan"}

    bench_col = "Open" if ep == "open" else "Close"
    bench_price_by_symbol: dict[str, pd.Series] = {}
    exec_by_code: dict[str, pd.DataFrame] = {}
    cost_by_symbol: dict[str, CostProfile] = {}
    mults_by_symbol: dict[str, float] = {}
    symbol_stats: list[dict[str, Any]] = []
    errors: list[str] = []

    for code in codes:
        try:
            df_exec, df_bench, detail = _align_futures_trend_inputs(
                db, pool_code=code, start=start, end=end
            )
        except ValueError as ex:
            errors.append(f"{code}:{ex}")
            continue
        if len(df_exec.index) < int(min_points):
            errors.append(f"{code}:points<{int(min_points)}")
            continue
        pool_row = get_futures_pool_by_code(db, code)
        cm = (
            getattr(pool_row, "contract_multiplier", None) if pool_row is not None else None
        )
        mpt = getattr(pool_row, "min_price_tick", None) if pool_row is not None else None
        cm_f = float(cm) if cm is not None and np.isfinite(float(cm)) else None
        mpt_f = float(mpt) if mpt is not None and np.isfinite(float(mpt)) else None
        if cm_f is None or float(cm_f) <= 0.0:
            errors.append(f"{code}:missing_contract_multiplier")
            continue
        if st == "tick_multiple" and (
            mpt_f is None or (not np.isfinite(float(mpt_f))) or float(mpt_f) <= 0.0
        ):
            errors.append(f"{code}:missing_min_price_tick_in_pool")
            continue
        px_ref = (
            float(df_exec[bench_col].median())
            if bench_col in df_exec.columns and len(df_exec.index) > 0
            else 0.0
        )
        try:
            cost = _build_cost_profile(
                cost_bps=float(cost_bps),
                fee_side=fee_side,
                slippage_type=st,  # type: ignore[arg-type]
                slippage_value=float(slippage_value),
                slippage_side=slippage_side,
                price_reference=px_ref,
                contract_multiplier=cm_f,
                min_price_tick=mpt_f,
            )
        except ValueError as ex:
            errors.append(f"{code}:cost_profile:{ex}")
            continue
        exec_by_code[code] = df_exec.copy()
        bench_price_by_symbol[code] = df_bench[bench_col].astype(float)
        cost_by_symbol[code] = cost
        mults_by_symbol[code] = float(cm_f)
        symbol_stats.append(
            {
                "code": code,
                "points": int(len(df_exec.index)),
                "start": str(df_exec.index.min().date()),
                "end": str(df_exec.index.max().date()),
                "engine": "lot_account",
                "commission_per_fill": float(cost.commission_per_fill),
                "spread_per_fill": float(cost.spread_per_fill),
                "contract_multiplier": cost.contract_multiplier,
                "min_price_tick": cost.min_price_tick,
                "tick_value_per_lot": cost.tick_value_per_lot,
                "slippage_tick_multiple": cost.slippage_tick_multiple,
                "rotation_resolution": detail.get("trend_resolution"),
                "rotation_execution_symbol": detail.get("execution_symbol"),
                "rotation_execution_adjust": detail.get("execution_adjust"),
            }
        )

    if not exec_by_code:
        hfq_mark = "missing synthetic hfq continuous series"
        err_kind = (
            "missing_synthetic_hfq_continuous"
            if errors and all(hfq_mark in str(e) for e in errors)
            else "insufficient_data"
        )
        return {
            "ok": False,
            "error": err_kind,
            "meta": {
                "group_name": group.name,
                "start": start,
                "end": end,
                "errors": errors,
            },
        }

    idx_list = [_coerce_trading_index(exec_by_code[c].index) for c in exec_by_code]
    common_idx = idx_list[0]
    for ix in idx_list[1:]:
        common_idx = common_idx.union(ix) if bool(dynamic_universe) else common_idx.intersection(ix)
    if len(common_idx) < int(min_points):
        return {
            "ok": False,
            "error": "insufficient_overlap",
            "meta": {
                "group_name": group.name,
                "start": start,
                "end": end,
                "errors": errors,
            },
        }
    common_idx = common_idx.sort_values()
    cols = sorted(exec_by_code.keys())
    exec_aligned = {
        c: exec_by_code[c].reindex(common_idx).replace([np.inf, -np.inf], np.nan)
        for c in cols
    }
    close_df = pd.DataFrame(
        {
            c: exec_aligned[c]["SignalClose"].astype(float)
            if "SignalClose" in exec_aligned[c].columns
            else exec_aligned[c]["Close"].astype(float)
            for c in cols
        },
        index=common_idx,
    )
    mom_df = _momentum_scores(close_df, lookback_days=int(lookback_days))
    vol_df = (
        _annualized_vol(close_df, window=int(inverse_vol_window))
        if pm == "inverse_vol"
        else None
    )
    w_tgt, w_meta = _build_target_weights(
        dates=common_idx,
        scores_df=mom_df,
        vol_df=vol_df,
        columns=cols,
        top_k=int(top_k),
        trade_direction=td,  # type: ignore[arg-type]
        position_mode=pm,  # type: ignore[arg-type]
        rebalance=rb,  # type: ignore[arg-type]
        lookback_days=int(lookback_days),
        inverse_vol_window=int(inverse_vol_window),
    )
    w_eff = (
        w_tgt.shift(1)
        .fillna(0.0)
        .astype(float)
        .mul(float(position_size_pct))
        .reindex(index=common_idx, columns=cols)
        .fillna(0.0)
    )

    equity_ser, lot_meta = simulate_discrete_lot_portfolio(
        common_idx=common_idx,
        exec_by_code=exec_aligned,
        w_eff=w_eff,
        cost_by_symbol=cost_by_symbol,
        mults=mults_by_symbol,
        margin_rate_frac=float(backtest_margin_rate_pct) / 100.0,
        reserve_ratio=float(reserve_margin_ratio),
        initial_equity_cny=float(account_capital_wan) * 10000.0,
        exec_price=ep,
        position_sizing=str(pm),
        codes_sorted=cols,
    )
    if equity_ser.empty:
        return {
            "ok": False,
            "error": "insufficient_data",
            "meta": {
                "group_name": group.name,
                "start": start,
                "end": end,
                "errors": errors,
            },
        }
    strategy_nav = (equity_ser / float(equity_ser.iloc[0])).ffill().fillna(1.0)

    bench_price_df = pd.DataFrame(bench_price_by_symbol).sort_index().reindex(common_idx)
    if not bool(dynamic_universe):
        bench_price_df = bench_price_df.ffill().bfill()
    bench_ret = _combine_group_returns(
        bench_price_df.pct_change(fill_method=None),
        dynamic_universe=bool(dynamic_universe),
    )
    if td == "short_only":
        bench_ret = -bench_ret
    benchmark_nav = (1.0 + bench_ret.fillna(0.0)).cumprod()

    align_idx = strategy_nav.index.union(benchmark_nav.index).sort_values()
    strategy_nav = strategy_nav.reindex(align_idx).ffill().fillna(1.0)
    benchmark_nav = benchmark_nav.reindex(align_idx).ffill().fillna(1.0)

    lot_closed = list(lot_meta.get("closed_trades") or [])
    reserve_attempted_dir = (
        lot_meta.get("reserve_margin_attempted_entry_count_by_direction") or {}
    )
    reserve_blocked_dir = (
        lot_meta.get("reserve_margin_blocked_entry_count_by_direction") or {}
    )
    reserve_attempted_code_dir = (
        lot_meta.get("reserve_margin_attempted_entry_count_by_code_direction") or {}
    )
    reserve_blocked_code_dir = (
        lot_meta.get("reserve_margin_blocked_entry_count_by_code_direction") or {}
    )

    def _reserve_counts(direction: str, code: str | None = None) -> tuple[int, int]:
        if code is not None:
            ca = reserve_attempted_code_dir.get(str(code), {})
            cb = reserve_blocked_code_dir.get(str(code), {})
            return int(ca.get(direction, 0)), int(cb.get(direction, 0))
        if direction == "both":
            return int(sum(int(v) for v in reserve_attempted_dir.values())), int(
                sum(int(v) for v in reserve_blocked_dir.values())
            )
        return int(reserve_attempted_dir.get(direction, 0)), int(
            reserve_blocked_dir.get(direction, 0)
        )

    def _trade_stats_pack(direction: str, code: str | None = None) -> dict[str, Any]:
        d = str(direction)
        rows = [
            r
            for r in lot_closed
            if (code is None or str(r.get("code")) == str(code))
            and (d == "both" or str(r.get("direction") or "").strip().lower() == d)
        ]
        rs = [
            float(r.get("return"))
            for r in rows
            if r.get("return") is not None and np.isfinite(float(r.get("return")))
        ]
        st_pack = _trade_stats_from_returns(rs)
        ra, rb2 = _reserve_counts(d, code)
        st_pack["reserve_margin_blocked_entry_count"] = int(rb2)
        st_pack["reserve_margin_blocked_entry_rate"] = float(
            _safe_rate(int(rb2), int(ra))
        )
        st_pack["monthly_risk_budget_blocked_entry_count"] = 0
        st_pack["monthly_risk_budget_blocked_entry_rate"] = 0.0
        st_pack["atr_stop_trigger_count"] = 0
        st_pack["atr_stop_trigger_rate"] = 0.0
        st_pack["atr_stop_trigger_multiple_distribution"] = {}
        st_pack["atr_stop_trigger_multiple_values"] = []
        st_pack["trade_marks"] = [
            {
                "code": str(r.get("code") or ""),
                "direction": str(r.get("direction") or ""),
                "entry_date": str(r.get("entry_date") or ""),
                "exit_date": str(r.get("exit_date") or ""),
                "return": (
                    None
                    if r.get("return") is None
                    else (
                        float(r.get("return"))
                        if np.isfinite(float(r.get("return")))
                        else None
                    )
                ),
            }
            for r in rows
            if str(r.get("entry_date") or "").strip()
            and str(r.get("exit_date") or "").strip()
        ]
        return st_pack

    trade_statistics = {
        "overall": {
            "both": _trade_stats_pack("both"),
            "long": _trade_stats_pack("long"),
            "short": _trade_stats_pack("short"),
        },
        "by_code": {
            str(c): {
                "both": _trade_stats_pack("both", str(c)),
                "long": _trade_stats_pack("long", str(c)),
                "short": _trade_stats_pack("short", str(c)),
            }
            for c in cols
        },
        "defaults": {"view_mode": "overall", "direction": "both"},
    }

    symbol_nav_by_code: dict[str, list[dict[str, Any]]] = {}
    for c in cols:
        ex = exec_aligned.get(c)
        if ex is None or bench_col not in ex.columns:
            continue
        px = ex[bench_col].reindex(common_idx).astype(float)
        ret = px.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        nav = (1.0 + ret).cumprod().fillna(1.0)
        symbol_nav_by_code[str(c)] = _series_rows(nav)

    benchmark_last = float(benchmark_nav.iloc[-1]) if len(benchmark_nav.index) else 1.0
    strategy_last = float(strategy_nav.iloc[-1]) if len(strategy_nav.index) else 1.0
    excess = (strategy_last / benchmark_last - 1.0) if benchmark_last != 0 else None
    rb_rule = {
        "daily": "every_trading_day",
        "weekly": "period_end_weekly_friday",
        "monthly": "period_end_monthly",
    }.get(rb, "unknown")
    return {
        "ok": True,
        "meta": {
            "group_name": group.name,
            "start": start,
            "end": end,
            "dynamic_universe": bool(dynamic_universe),
            "mode": "dynamic_union" if dynamic_universe else "static_intersection",
            "rebalance": rb,
            "rebalance_rule": rb_rule,
            "lookback_days": int(lookback_days),
            "top_k": int(top_k),
            "effective_top_k_max": int(w_meta.get("effective_top_k_max", 0)),
            "trade_direction": td,
            "position_mode": pm,
            "inverse_vol_window": (
                int(inverse_vol_window) if pm == "inverse_vol" else None
            ),
            "position_size_pct": float(position_size_pct),
            "exec_price": ep,
            "signal_execution_rule": f"signal_t_execute_t_plus_1_{ep}",
            "signal_lag_trading_days": 1,
            "benchmark_price_basis": ep,
            "cost_bps": float(cost_bps),
            "fee_side": fee_side,
            "slippage_type": st,
            "slippage_value": float(slippage_value),
            "slippage_side": slippage_side,
            "account_capital_wan": float(account_capital_wan),
            "backtest_margin_rate_pct": float(backtest_margin_rate_pct),
            "reserve_margin_ratio": float(reserve_margin_ratio),
            "portfolio": {
                "lot_account": lot_meta,
                "decision_count": int(w_meta.get("decision_count", 0)),
                "decision_dates": list(w_meta.get("decision_dates") or []),
                "inverse_vol_equal_fallback_count": int(
                    w_meta.get("inverse_vol_equal_fallback_count", 0)
                ),
            },
            "effective_symbols": int(len(cols)),
            "skipped": errors,
        },
        "series": {
            "strategy_nav": _series_rows(strategy_nav),
            "benchmark_nav": _series_rows(benchmark_nav),
            "symbol_nav_by_code": symbol_nav_by_code,
        },
        "summary": {
            "strategy_total_return": float(strategy_last - 1.0),
            "benchmark_total_return": float(benchmark_last - 1.0),
            "excess_total_return": (
                float(excess)
                if excess is not None and np.isfinite(float(excess))
                else None
            ),
        },
        "symbols": symbol_stats,
        "trade_statistics": trade_statistics,
    }
