from __future__ import annotations

import calendar
import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..calendar.trading_calendar import is_trading_day, shift_to_trading_day, trading_days
from .baseline import (
    TRADING_DAYS_PER_YEAR,
    _annualized_return,
    _annualized_vol,
    _information_ratio,
    _max_drawdown,
    _max_drawdown_duration_days,
    _sharpe,
    _sortino,
    _ulcer_index,
    load_ohlc_prices,
)
from .execution_timing import corporate_action_mask


@dataclass(frozen=True)
class CalendarTimingStrategyInputs:
    mode: str
    code: str | None
    codes: list[str] | None
    start: dt.date
    end: dt.date
    adjust: str = "none"
    decision_day: int = 1  # [-28, -1] U [1, 28], monthly natural day semantics
    hold_days: int = 1
    position_mode: str = "equal"  # equal | fixed_ratio
    fixed_pos_ratio: float = 1.0
    dynamic_universe: bool = False
    exec_price: str = "open"  # open | close
    cost_bps: float = 2.0
    slippage_rate: float = 0.001
    rebalance_shift: str = "prev"  # prev | next | skip
    risk_free_rate: float = 0.025
    cal: str = "XSHG"


def _monthly_target_date(year: int, month: int, anchor: int) -> dt.date:
    if anchor == 0 or anchor < -28 or anchor > 28:
        raise ValueError("decision_day must be within [-28, 28] and cannot be 0")
    last_day = calendar.monthrange(year, month)[1]
    if anchor > 0:
        day = anchor
    else:
        day = last_day + 1 + anchor
    if day < 1:
        day = 1
    if day > last_day:
        day = last_day
    return dt.date(year, month, day)


def _shift_or_skip(target: dt.date, *, shift: str, cal: str) -> dt.date | None:
    if shift == "skip":
        return target if is_trading_day(target, cal=cal) else None
    if shift in {"prev", "next"}:
        return shift_to_trading_day(target, shift=shift, cal=cal)
    raise ValueError("rebalance_shift must be one of: prev|next|skip")


def _periodic_returns(nav: pd.Series, freq: str) -> list[dict[str, Any]]:
    if nav.empty:
        return []
    s = nav.resample(freq).last().dropna()
    r = s.pct_change().dropna()
    return [{"period_end": d.strftime("%Y-%m-%d"), "return": float(v)} for d, v in r.items()]


def _dist_stats(values: list[float]) -> dict[str, Any]:
    xs = np.asarray([float(x) for x in values if np.isfinite(float(x))], dtype=float)
    if xs.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "quantiles": {k: None for k in ["p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]},
        }
    qv = np.percentile(xs, [1, 5, 10, 25, 50, 75, 90, 95, 99]).tolist()
    return {
        "count": int(xs.size),
        "mean": float(xs.mean()),
        "std": float(xs.std(ddof=1)) if xs.size > 1 else 0.0,
        "min": float(xs.min()),
        "max": float(xs.max()),
        "quantiles": {
            "p01": float(qv[0]),
            "p05": float(qv[1]),
            "p10": float(qv[2]),
            "p25": float(qv[3]),
            "p50": float(qv[4]),
            "p75": float(qv[5]),
            "p90": float(qv[6]),
            "p95": float(qv[7]),
            "p99": float(qv[8]),
        },
    }


def _trade_stats(values: list[float], *, flat_eps: float = 1e-12) -> dict[str, Any]:
    rs = [float(x) for x in (values or []) if np.isfinite(float(x))]
    wins = [x for x in rs if x > float(flat_eps)]
    losses = [x for x in rs if x < -float(flat_eps)]
    flats = [x for x in rs if abs(float(x)) <= float(flat_eps)]
    win_rate_ex_zero: float | None = None
    payoff_ex_zero: float | None = None
    kelly_ex_zero: float | None = None
    if wins and losses:
        avg_win = float(np.mean(np.asarray(wins, dtype=float)))
        avg_loss_abs = float(abs(np.mean(np.asarray(losses, dtype=float))))
        if np.isfinite(avg_win) and np.isfinite(avg_loss_abs) and avg_win > 0.0 and avg_loss_abs > 0.0:
            b = float(avg_win / avg_loss_abs)
            p = float(len(wins) / (len(wins) + len(losses)))
            win_rate_ex_zero = p
            payoff_ex_zero = b
            kelly_ex_zero = float(p - (1.0 - p) / b)
    return {
        "total_trades": int(len(rs)),
        "win_trades": int(len(wins)),
        "loss_trades": int(len(losses)),
        "flat_trades": int(len(flats)),
        "win_rate_ex_zero": win_rate_ex_zero,
        "payoff_ex_zero": payoff_ex_zero,
        "kelly_ex_zero": kelly_ex_zero,
        "returns": [float(x) for x in rs],
        "all_stats": _dist_stats(rs),
        "profit_stats": _dist_stats(wins),
        "loss_stats": _dist_stats(losses),
    }


def compute_calendar_timing_strategy_backtest(db: Session, inp: CalendarTimingStrategyInputs) -> dict[str, Any]:
    mode = str(inp.mode or "portfolio").strip().lower()
    if mode not in {"portfolio", "single"}:
        raise ValueError("mode must be one of: portfolio|single")
    codes = [str(inp.code).strip()] if mode == "single" else list(dict.fromkeys([str(x).strip() for x in (inp.codes or []) if str(x).strip()]))
    if not codes:
        raise ValueError("codes is empty")
    if inp.decision_day == 0 or inp.decision_day < -28 or inp.decision_day > 28:
        raise ValueError("decision_day must be within [-28, 28] and cannot be 0")
    if int(inp.hold_days) < 1:
        raise ValueError("hold_days must be >= 1")
    ep = str(inp.exec_price or "open").strip().lower()
    if ep not in {"open", "close"}:
        raise ValueError("exec_price must be one of: open|close")
    shift = str(inp.rebalance_shift or "prev").strip().lower()
    if shift not in {"prev", "next", "skip"}:
        raise ValueError("rebalance_shift must be one of: prev|next|skip")
    position_mode = str(inp.position_mode or "equal").strip().lower()
    if position_mode not in {"equal", "fixed_ratio"}:
        raise ValueError("position_mode must be one of: equal|fixed_ratio")
    exposure = 1.0 if position_mode == "equal" else float(inp.fixed_pos_ratio)
    dynamic_universe = bool(getattr(inp, "dynamic_universe", False))
    if not np.isfinite(exposure) or exposure < 0.0 or exposure > 1.0:
        raise ValueError("fixed_pos_ratio must be finite and within [0,1]")
    cost_rate = float(inp.cost_bps) / 10000.0
    slip_rate = float(inp.slippage_rate)
    if not np.isfinite(cost_rate) or cost_rate < 0.0:
        raise ValueError("cost_bps must be finite and >= 0")
    if not np.isfinite(slip_rate) or slip_rate < 0.0:
        raise ValueError("slippage_rate must be finite and >= 0")

    # Price-adjustment policy (mandatory):
    # - Strategy NAV: none prices preferred, with hfq return fallback on corporate-action cliff days.
    # - Benchmark NAV: always hfq close (buy&hold total-return proxy).
    ohlc_none = load_ohlc_prices(db, codes=codes, start=inp.start, end=inp.end, adjust="none")
    ohlc_hfq = load_ohlc_prices(db, codes=codes, start=inp.start, end=inp.end, adjust="hfq")
    close_none = (
        ohlc_none.get("close", pd.DataFrame())
        .sort_index()
        .reindex(columns=codes)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )
    close_hfq_base = (
        ohlc_hfq.get("close", pd.DataFrame())
        .sort_index()
        .reindex(columns=codes)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )
    if close_none.empty:
        raise ValueError("no execution price data for given range (none)")
    miss_exec = [c for c in codes if c not in close_none.columns or close_none[c].dropna().empty]
    miss_hfq = [c for c in codes if c not in close_hfq_base.columns or close_hfq_base[c].dropna().empty]
    if (miss_exec or miss_hfq) and (not dynamic_universe):
        if miss_exec:
            raise ValueError(f"missing execution data (none) for: {miss_exec}")
        raise ValueError(f"missing benchmark data (hfq) for: {miss_hfq}")

    valid_codes = [c for c in codes if (c not in miss_exec) and (c not in miss_hfq)]
    if not valid_codes:
        raise ValueError("no valid candidate codes with both none/hfq coverage")
    dropped_codes = [c for c in codes if c not in valid_codes]
    codes = valid_codes

    close_none = close_none.reindex(columns=codes)
    close_hfq_base = close_hfq_base.reindex(columns=codes)
    first_valid_none_by_code = {c: close_none[c].first_valid_index() for c in codes}
    first_valid_hfq_by_code = {c: close_hfq_base[c].first_valid_index() for c in codes}
    first_valid_none = [d for d in first_valid_none_by_code.values() if d is not None]
    first_valid_hfq = [d for d in first_valid_hfq_by_code.values() if d is not None]
    if (not first_valid_none) or (not first_valid_hfq):
        raise ValueError("no valid price data after alignment")
    if dynamic_universe:
        common_start = min(first_valid_none + first_valid_hfq)
    else:
        common_start = max(first_valid_none + first_valid_hfq)
    close_none = close_none.loc[common_start:]
    if len(close_none.index) < 3:
        raise ValueError("insufficient price history")

    idx = pd.DatetimeIndex(close_none.index)
    dates = [d.date() for d in idx]
    date_to_i = {d: i for i, d in enumerate(dates)}

    open_none = (
        ohlc_none.get("open", pd.DataFrame())
        .sort_index()
        .reindex(idx)
        .reindex(columns=codes)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )
    close_none_exec = (
        ohlc_none.get("close", pd.DataFrame())
        .sort_index()
        .reindex(idx)
        .reindex(columns=codes)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )
    open_hfq = (
        ohlc_hfq.get("open", pd.DataFrame())
        .sort_index()
        .reindex(idx)
        .reindex(columns=codes)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )
    close_hfq_exec = (
        ohlc_hfq.get("close", pd.DataFrame())
        .sort_index()
        .reindex(idx)
        .reindex(columns=codes)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )
    close_hfq = close_hfq_exec.astype(float).reindex(idx).reindex(columns=codes).ffill()

    close_none_f = close_none.astype(float).reindex(idx).reindex(columns=codes).ffill()
    close_hfq_f = close_hfq.astype(float).reindex(idx).reindex(columns=codes).ffill()
    if ep == "open":
        exec_o_none = open_none.astype(float).combine_first(close_none_f)
        exec_c_none = close_none_exec.astype(float).combine_first(close_none_f)
        exec_o_hfq = open_hfq.astype(float).combine_first(close_hfq_f)
        exec_c_hfq = close_hfq_exec.astype(float).combine_first(close_hfq_f)
        ret_none = (exec_c_none / exec_o_none - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_hfq_exec = (exec_c_hfq / exec_o_hfq - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    else:
        px_none_exec = close_none_exec.astype(float).combine_first(close_none_f)
        px_hfq_exec = close_hfq_exec.astype(float).combine_first(close_hfq_f)
        ret_none = (px_none_exec.shift(-1).div(px_none_exec) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_hfq_exec = (px_hfq_exec.shift(-1).div(px_hfq_exec) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    ret_hfq = close_hfq_f.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    gross_none = (1.0 + ret_none).astype(float)
    gross_hfq = (1.0 + ret_hfq).astype(float)
    _corp_factor, ca_mask = corporate_action_mask(gross_none, gross_hfq)
    ret_exec = ret_none.where(~ca_mask.fillna(False), other=ret_hfq_exec).astype(float)

    # Build monthly decision dates on natural calendar, then map to trading day.
    m0 = (inp.start.year, inp.start.month)
    m1 = (inp.end.year, inp.end.month)
    ym: list[tuple[int, int]] = []
    y, m = m0
    while (y, m) <= m1:
        ym.append((y, m))
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    all_trade_days = trading_days(inp.start - dt.timedelta(days=40), inp.end + dt.timedelta(days=240), cal=inp.cal)
    all_trade_days = sorted(set(all_trade_days))
    trade_pos = {d: i for i, d in enumerate(all_trade_days)}

    entry_events: list[dict[str, Any]] = []
    for y, m in ym:
        natural = _monthly_target_date(y, m, int(inp.decision_day))
        dec = _shift_or_skip(natural, shift=shift, cal=inp.cal)
        if dec is None:
            continue
        p = trade_pos.get(dec)
        if p is None or p + 1 >= len(all_trade_days):
            continue
        exec_day = all_trade_days[p + 1]
        if exec_day < dates[0]:
            continue
        entry_events.append({"decision_date": dec, "entry_exec_date": exec_day})
    entry_events.sort(key=lambda x: x["entry_exec_date"])

    # Build non-overlapping trades.
    trades_sched: list[dict[str, Any]] = []
    last_exit: dt.date | None = None
    for ev in entry_events:
        e_day = ev["entry_exec_date"]
        if last_exit is not None and e_day <= last_exit:
            continue
        p = trade_pos.get(e_day)
        if p is None:
            continue
        x_pos = min(p + int(inp.hold_days), len(all_trade_days) - 1)
        x_day = all_trade_days[x_pos]
        trades_sched.append(
            {
                "decision_date": ev["decision_date"],
                "entry_exec_date": e_day,
                "exit_exec_date": x_day,
            }
        )
        last_exit = x_day

    # Portfolio target weights when invested.
    n_codes = len(codes)
    active_mask = np.zeros((len(idx), n_codes), dtype=bool)
    for j, c in enumerate(codes):
        t0n = first_valid_none_by_code.get(c)
        t0h = first_valid_hfq_by_code.get(c)
        t0 = max(t0n, t0h) if (t0n is not None and t0h is not None) else None
        if t0 is None:
            continue
        active_mask[:, j] = np.asarray(idx >= pd.Timestamp(t0), dtype=bool)

    def _weights_for_day(i: int) -> np.ndarray:
        out = np.zeros(n_codes, dtype=float)
        if exposure <= 0.0 or n_codes <= 0:
            return out
        if dynamic_universe:
            mask = active_mask[int(i)] if 0 <= int(i) < len(active_mask) else np.zeros(n_codes, dtype=bool)
            n_act = int(mask.sum())
            if n_act > 0:
                out[mask] = float(exposure) / float(n_act)
            return out
        out[:] = float(exposure) / float(n_codes)
        return out

    w_cur = np.zeros(n_codes, dtype=float)

    # Event maps on in-sample trading days.
    entry_by_i: dict[int, dict[str, Any]] = {}
    exit_by_i: dict[int, dict[str, Any]] = {}
    for tr in trades_sched:
        ei = date_to_i.get(tr["entry_exec_date"])
        xi = date_to_i.get(tr["exit_exec_date"])
        if ei is None:
            continue
        tr["entry_idx"] = int(ei)
        tr["exit_idx"] = (int(xi) if xi is not None else None)
        entry_by_i[int(ei)] = tr
        if xi is not None:
            exit_by_i[int(xi)] = tr

    strat_ret = np.zeros(len(idx), dtype=float)
    bench_ret = np.zeros(len(idx), dtype=float)
    turnover = np.zeros(len(idx), dtype=float)

    active_trade: dict[str, Any] | None = None
    trade_returns: list[float] = []
    trade_returns_by_code: dict[str, list[float]] = {c: [] for c in codes}

    for i in range(1, len(idx)):
        r_exec_i = ret_exec.iloc[i].to_numpy(dtype=float)
        r_hfq_i = ret_hfq.iloc[i].to_numpy(dtype=float)
        gross = float(np.dot(w_cur, r_exec_i))
        b_w_i = _weights_for_day(i)
        bench_gross = float(np.dot(b_w_i, r_hfq_i))
        tgt = w_cur.copy()
        if i in entry_by_i:
            tgt = _weights_for_day(i)
            if float(np.sum(tgt)) > 1e-12:
                active_trade = {
                    "entry_idx": int(i),
                    "exit_idx": entry_by_i[i]["exit_idx"],
                    "factor": 1.0,
                    "by_code_factor": {c: 1.0 for c in codes},
                }
            else:
                active_trade = None
        if i in exit_by_i:
            tgt = np.zeros(n_codes, dtype=float)
        t = float(np.abs(tgt - w_cur).sum() / 2.0)
        turnover[i] = t
        cost_today = t * (cost_rate + slip_rate)
        net = gross - cost_today
        strat_ret[i] = net
        bench_ret[i] = bench_gross

        if active_trade is not None:
            # Trade-level aggregate return includes sequential daily compounding and execution-day costs.
            active_trade["factor"] *= float(1.0 + net)
            # Per-code trade returns (allocated on each code's own capital share).
            if n_codes > 0:
                abs_delta = np.abs(tgt - w_cur)
                abs_sum = float(abs_delta.sum())
                for j, c in enumerate(codes):
                    rc = float(r_exec_i[j]) if np.isfinite(r_exec_i[j]) else 0.0
                    # On allocated capital, one side cost = (abs(delta_j) / exposure) * (cost+slippage)
                    alloc = 0.0
                    if exposure > 0.0 and abs_sum > 0.0:
                        # Use a stable invested-capital denominator. On exit days tgt[j] can be 0,
                        # so dividing by tgt would explode per-code net return attribution.
                        invested_base = max(float(w_cur[j]), float(tgt[j]), 1e-12)
                        alloc = (float(abs_delta[j]) / float(abs_sum)) * (t * (cost_rate + slip_rate)) / invested_base
                    cnet = rc - alloc
                    active_trade["by_code_factor"][c] *= float(1.0 + cnet)

        w_cur = tgt

        if active_trade is not None and active_trade["exit_idx"] is not None and i >= int(active_trade["exit_idx"]):
            trade_returns.append(float(active_trade["factor"] - 1.0))
            for c in codes:
                trade_returns_by_code[c].append(float(active_trade["by_code_factor"][c] - 1.0))
            active_trade = None

    # Include the last open trade as mark-to-market sample.
    if active_trade is not None:
        trade_returns.append(float(active_trade["factor"] - 1.0))
        for c in codes:
            trade_returns_by_code[c].append(float(active_trade["by_code_factor"][c] - 1.0))

    strat_nav = pd.Series((1.0 + pd.Series(strat_ret, index=idx, dtype=float)).cumprod(), index=idx, dtype=float)
    bench_nav = pd.Series((1.0 + pd.Series(bench_ret, index=idx, dtype=float)).cumprod(), index=idx, dtype=float)
    asset_nav_exec = (1.0 + ret_exec).cumprod().replace([np.inf, -np.inf], np.nan).ffill()
    strat_nav.iloc[0] = 1.0
    bench_nav.iloc[0] = 1.0
    excess_nav = (strat_nav / bench_nav.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)

    sret = strat_nav.pct_change().fillna(0.0).astype(float)
    bret = bench_nav.pct_change().fillna(0.0).astype(float)
    exret = excess_nav.pct_change().fillna(0.0).astype(float)

    def _metrics(nav: pd.Series, daily: pd.Series, *, rf: float, active: pd.Series | None = None) -> dict[str, Any]:
        ann_ret = _annualized_return(nav, ann_factor=TRADING_DAYS_PER_YEAR)
        ann_vol = _annualized_vol(daily, ann_factor=TRADING_DAYS_PER_YEAR)
        mdd = _max_drawdown(nav)
        mdd_days = _max_drawdown_duration_days(nav)
        sharpe = _sharpe(daily, rf=rf, ann_factor=TRADING_DAYS_PER_YEAR)
        sortino = _sortino(daily, rf=rf, ann_factor=TRADING_DAYS_PER_YEAR)
        calmar = float(ann_ret / abs(mdd)) if np.isfinite(mdd) and mdd < 0 else float("nan")
        ui = _ulcer_index(nav, in_percent=True)
        ui_den = ui / 100.0
        upi = float((ann_ret - rf) / ui_den) if ui_den > 0 else float("nan")
        ir = float(_information_ratio(active, ann_factor=TRADING_DAYS_PER_YEAR)) if active is not None else float("nan")
        return {
            "sample_days": int(len(nav)),
            "cumulative_return": float(nav.iloc[-1] - 1.0) if len(nav) else float("nan"),
            "annualized_return": float(ann_ret),
            "annualized_volatility": float(ann_vol),
            "max_drawdown": float(mdd),
            "max_drawdown_recovery_days": int(mdd_days),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "ulcer_index": float(ui),
            "ulcer_performance_index": float(upi),
            "information_ratio": float(ir),
            "avg_daily_turnover": float(np.mean(turnover[1:])) if len(turnover) > 1 else 0.0,
            "avg_annual_turnover": float(np.mean(turnover[1:]) * TRADING_DAYS_PER_YEAR) if len(turnover) > 1 else 0.0,
        }

    m_strat = _metrics(strat_nav, sret, rf=float(inp.risk_free_rate), active=(sret - bret))
    m_bench = _metrics(bench_nav, bret, rf=float(inp.risk_free_rate), active=None)
    m_ex = _metrics(excess_nav, exret, rf=0.0, active=exret)

    weekly = _periodic_returns(strat_nav, "W-FRI")
    monthly = _periodic_returns(strat_nav, "ME")
    quarterly = _periodic_returns(strat_nav, "QE")
    yearly = _periodic_returns(strat_nav, "YE")

    current_holdings: list[dict[str, Any]] = []
    in_pos = bool(np.sum(w_cur) > 1e-12)
    if in_pos:
        cur_entry_idx = None
        for tr in reversed(trades_sched):
            ei = tr.get("entry_idx")
            xi = tr.get("exit_idx")
            if ei is None:
                continue
            if int(ei) <= len(idx) - 1 and (xi is None or int(xi) >= len(idx) - 1):
                cur_entry_idx = int(ei)
                break
        if cur_entry_idx is not None:
            hold_days_now = max(0, (len(idx) - 1) - cur_entry_idx)
            for j, c in enumerate(codes):
                wj = float(w_cur[j])
                if wj <= 1e-12:
                    continue
                nav0 = float(asset_nav_exec[c].iloc[cur_entry_idx]) if np.isfinite(float(asset_nav_exec[c].iloc[cur_entry_idx])) else np.nan
                nav1 = float(asset_nav_exec[c].iloc[-1]) if np.isfinite(float(asset_nav_exec[c].iloc[-1])) else np.nan
                hr = float(nav1 / nav0 - 1.0) if (np.isfinite(nav0) and np.isfinite(nav1) and nav0 > 0) else float("nan")
                current_holdings.append(
                    {
                        "code": c,
                        "entry_date": str(idx[cur_entry_idx].date()),
                        "holding_days": int(hold_days_now),
                        "weight": wj,
                        "holding_return": hr,
                    }
                )

    # Next execution plan (entry or exit) after end date.
    asof = dates[-1]
    next_plan: dict[str, Any] = {"has_execution_plan": False, "asof_date": str(asof), "next_trading_day": None, "plan": None}
    future_trade_days = [d for d in all_trade_days if d > asof]
    if future_trade_days:
        # compute future entries.
        # Include current month first; if asof is before this month's decision/entry,
        # skipping current month would produce a delayed next-plan date.
        fut_months: list[tuple[int, int]] = []
        y0, m0 = asof.year, asof.month
        y, m = y0, m0
        fut_months.append((y, m))
        for _ in range(4):
            if m == 12:
                y += 1
                m = 1
            else:
                m += 1
            fut_months.append((y, m))
        fut_entries: list[tuple[dt.date, dt.date]] = []  # (decision, entry_exec)
        for y, m in fut_months:
            natural = _monthly_target_date(y, m, int(inp.decision_day))
            dec = _shift_or_skip(natural, shift=shift, cal=inp.cal)
            if dec is None:
                continue
            p = trade_pos.get(dec)
            if p is None or p + 1 >= len(all_trade_days):
                continue
            fut_entries.append((dec, all_trade_days[p + 1]))
        fut_entries.sort(key=lambda x: x[1])

        next_entry = fut_entries[0] if fut_entries else None
        next_exit: dt.date | None = None
        if in_pos:
            # infer current open trade exit from schedule/trade days
            for tr in reversed(trades_sched):
                xi = tr.get("exit_exec_date")
                if xi is not None and tr.get("entry_exec_date") is not None and tr["entry_exec_date"] <= asof <= xi:
                    next_exit = xi if xi > asof else None
                    break

        candidate: tuple[str, dt.date, dt.date | None] | None = None  # (type, exec_date, decision_date)
        if next_entry is not None:
            candidate = ("entry", next_entry[1], next_entry[0])
        if next_exit is not None and (candidate is None or next_exit < candidate[1]):
            candidate = ("exit", next_exit, None)

        if candidate is not None:
            ev_type, exec_d, dec_d = candidate
            next_plan["next_trading_day"] = str(exec_d)
            if ev_type == "entry":
                w_plan = _weights_for_day(len(idx) - 1)
                buys = [{"code": c, "target_weight": float(w_plan[j]), "buy_weight": float(w_plan[j])} for j, c in enumerate(codes) if float(w_plan[j]) > 1e-12]
                planned_exit_date: str | None = None
                p_exec = trade_pos.get(exec_d)
                if p_exec is not None:
                    x_pos = min(int(p_exec) + int(inp.hold_days), len(all_trade_days) - 1)
                    planned_exit_date = str(all_trade_days[x_pos])
                next_plan["has_execution_plan"] = len(buys) > 0
                next_plan["plan"] = {
                    "type": "entry",
                    "decision_date": str(dec_d) if dec_d is not None else None,
                    "execution_date": str(exec_d),
                    "planned_exit_date": planned_exit_date,
                    "buys": buys,
                    "sells": [],
                }
            else:
                sells = [{"code": c, "current_weight": float(w_cur[j]), "sell_weight": float(w_cur[j])} for j, c in enumerate(codes) if float(w_cur[j]) > 1e-12]
                next_plan["has_execution_plan"] = len(sells) > 0
                next_plan["plan"] = {
                    "type": "exit",
                    "decision_date": None,
                    "execution_date": str(exec_d),
                    "buys": [],
                    "sells": sells,
                }

    return {
        "meta": {
            "type": "calendar_timing_strategy",
            "mode": mode,
            "codes": codes,
            "start": inp.start.strftime("%Y%m%d"),
            "end": inp.end.strftime("%Y%m%d"),
            "decision_day": int(inp.decision_day),
            "hold_days": int(inp.hold_days),
            "position_mode": position_mode,
            "fixed_pos_ratio": float(exposure),
            "dynamic_universe": bool(dynamic_universe),
            "dropped_codes": dropped_codes,
            "exec_price": ep,
            "cost_bps": float(inp.cost_bps),
            "slippage_rate": float(inp.slippage_rate),
            "rebalance_shift": shift,
            "adjust": str(inp.adjust),
            "strategy_price_basis": "none preferred + hfq fallback on corporate-action days",
            "benchmark_price_basis": "hfq close",
        },
        "nav": {
            "dates": [str(d.date()) for d in idx],
            "series": {
                "STRAT": [float(x) for x in strat_nav.to_numpy(dtype=float)],
                "BUY_HOLD": [float(x) for x in bench_nav.to_numpy(dtype=float)],
                "EXCESS": [float(x) for x in excess_nav.to_numpy(dtype=float)],
            },
        },
        "asset_nav_exec": {
            "dates": [str(d.date()) for d in idx],
            "series": {
                str(c): [float(x) for x in asset_nav_exec[str(c)].to_numpy(dtype=float)]
                for c in codes
                if str(c) in asset_nav_exec.columns
            },
        },
        "metrics": {"strategy": m_strat, "benchmark": m_bench, "excess": m_ex},
        "period_returns": {
            "weekly": weekly,
            "monthly": monthly,
            "quarterly": quarterly,
            "yearly": yearly,
        },
        "current_holdings": current_holdings,
        "next_execution_plan": next_plan,
        "trade_statistics": {
            "overall": _trade_stats(trade_returns),
            "by_code": {c: _trade_stats(vs) for c, vs in trade_returns_by_code.items()},
        },
    }

