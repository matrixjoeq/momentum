from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..analysis.baseline import (
    _annualized_return,
    _annualized_vol,
    _max_drawdown,
    _max_drawdown_duration_days,
    _rolling_max_drawdown,
    _sharpe,
    _sortino,
    _ulcer_index,
)
from ..analysis.baseline import load_close_prices as _load_close_prices
from ..analysis.baseline import _compute_return_risk_contributions as _compute_return_risk_contributions


@dataclass(frozen=True)
class RotationInputs:
    codes: list[str]
    start: dt.date
    end: dt.date
    rebalance: str = "weekly"  # daily/weekly/monthly/quarterly/yearly
    top_k: int = 1
    lookback_days: int = 20
    skip_days: int = 0  # skip recent trading days (0 means no skip)
    risk_free_rate: float = 0.025
    cost_bps: float = 0.0  # round-trip cost in bps per turnover, simple approximation
    risk_off: bool = False
    defensive_code: str | None = None
    momentum_floor: float = 0.0  # if best score <= floor -> risk-off


def _rebalance_labels(index: pd.DatetimeIndex, rebalance: str) -> pd.PeriodIndex:
    r = (rebalance or "monthly").lower()
    freq_map = {"daily": "D", "weekly": "W-FRI", "monthly": "M", "quarterly": "Q", "yearly": "Y"}
    if r not in freq_map:
        raise ValueError(f"invalid rebalance={rebalance}")
    return index.to_period(freq_map[r])


def _momentum_scores(close_hfq: pd.DataFrame, *, lookback_days: int, skip_days: int) -> pd.DataFrame:
    # score[t] = close[t-skip]/close[t-skip-lookback] - 1
    lag = skip_days
    lb = lookback_days
    return close_hfq.shift(lag) / close_hfq.shift(lag + lb) - 1.0


def _pick_assets(
    scores_row: pd.Series, *, top_k: int, risk_off: bool, defensive_code: str | None, floor: float
) -> tuple[list[str], dict[str, Any]]:
    """
    Pick assets for the next holding period.

    Returns (picks, meta):
    - picks: list of codes to hold (equal-weight). Empty list means "cash" if risk_off triggered
      but no defensive_code is provided.
    - meta: debug info (best_score, risk_off_triggered, mode).
    """
    s = scores_row.dropna()
    if s.empty:
        picks = [defensive_code] if (risk_off and defensive_code) else []
        return picks, {"best_score": None, "risk_off_triggered": bool(risk_off and picks), "mode": "no_signal"}

    s = s.sort_values(ascending=False)
    best = float(s.iloc[0])
    if risk_off and best <= floor:
        if defensive_code:
            return [defensive_code], {"best_score": best, "risk_off_triggered": True, "mode": "defensive"}
        return [], {"best_score": best, "risk_off_triggered": True, "mode": "cash"}

    picks = [str(x) for x in s.index[: max(1, int(top_k))].tolist()]
    return picks, {"best_score": best, "risk_off_triggered": False, "mode": "risk_on"}


def backtest_rotation(db: Session, inp: RotationInputs) -> dict[str, Any]:
    universe = list(dict.fromkeys(inp.codes))
    if not universe:
        raise ValueError("codes is empty")
    if inp.top_k <= 0:
        raise ValueError("top_k must be >= 1")
    if inp.lookback_days <= 0:
        raise ValueError("lookback_days must be > 0")
    if inp.skip_days < 0:
        raise ValueError("skip_days must be >= 0")

    codes = universe[:]  # may include defensive later for strategy holdings
    defensive = (inp.defensive_code or "").strip() or None
    if inp.risk_off and defensive:
        if defensive not in codes:
            codes = codes + [defensive]

    # Load hfq for signal and benchmark(total return), none for execution (trading simulation).
    # Need enough history for momentum.
    ext_start = inp.start - dt.timedelta(days=inp.lookback_days + inp.skip_days + 60)
    close_hfq = _load_close_prices(db, codes=codes, start=ext_start, end=inp.end, adjust="hfq")
    close_none = _load_close_prices(db, codes=codes, start=inp.start, end=inp.end, adjust="none")
    if close_none.empty:
        raise ValueError("no execution price data for given range (none)")

    # Align calendars using execution dates; forward-fill hfq onto those dates.
    close_none = close_none.sort_index().ffill()
    dates = close_none.index
    close_hfq = close_hfq.sort_index().reindex(dates).ffill()

    # Require each selected code has data
    miss_exec = [c for c in codes if c not in close_none.columns or close_none[c].dropna().empty]
    if miss_exec:
        raise ValueError(f"missing execution data (none) for: {miss_exec}")
    miss_sig = [c for c in codes if c not in close_hfq.columns or close_hfq[c].dropna().empty]
    if miss_sig:
        raise ValueError(f"missing signal data (hfq) for: {miss_sig}")

    scores = _momentum_scores(close_hfq[codes], lookback_days=inp.lookback_days, skip_days=inp.skip_days)

    # Determine rebalance decision dates: last trading day within each period.
    # If we rebalance at close on decision_date, then returns on the NEXT trading day onward
    # should reflect the new holdings. Therefore the holdings from one decision apply through
    # the NEXT decision date (inclusive), to avoid "gaps" on decision dates.
    labels = _rebalance_labels(dates, inp.rebalance)
    last_idx = pd.Series(np.arange(len(dates)), index=dates).groupby(labels).max().to_list()
    decision_dates = dates[last_idx]

    # Build weights per date (apply from next trading day after decision date).
    w = pd.DataFrame(0.0, index=dates, columns=codes)
    holdings: dict[str, list[dict[str, Any]]] = {"periods": []}
    for i, d in enumerate(decision_dates):
        # apply from next trading day after decision date
        di = dates.get_loc(d)
        if di + 1 >= len(dates):
            break
        start_i = di + 1
        next_di = (dates.get_loc(decision_dates[i + 1]) if i + 1 < len(decision_dates) else (len(dates) - 1))
        end_i = min(len(dates) - 1, next_di)
        picks, meta = _pick_assets(
            scores.loc[d],
            top_k=inp.top_k,
            risk_off=inp.risk_off,
            defensive_code=defensive,
            floor=inp.momentum_floor,
        )
        picks = [p for p in picks if p in codes]
        if picks:
            weight = 1.0 / len(picks)
            w.loc[dates[start_i] : dates[end_i], picks] = weight
        holdings["periods"].append(
            {
                "decision_date": d.date().isoformat(),
                "start_date": dates[start_i].date().isoformat(),
                "end_date": dates[end_i].date().isoformat(),
                "picks": picks,
                "scores": {k: (None if pd.isna(scores.loc[d, k]) else float(scores.loc[d, k])) for k in picks},
                "best_score": meta.get("best_score"),
                "risk_off_triggered": bool(meta.get("risk_off_triggered")),
                "mode": meta.get("mode"),
            }
        )

    # Daily holding return:
    # - trades are assumed executed at none prices (close-to-close approximation),
    # - BUT to correctly model investor economics across dividends/splits, we apply a corporate-action factor
    #   implied by hfq vs none. In a weight-based backtest, this is equivalent to using hfq daily returns
    #   for holding P&L (total return), while keeping the "execution price basis" as none.
    #
    # This prevents artificial NAV cliffs from splits and also captures dividend cashflows implicitly.
    ret_none = close_none[codes].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_hfq_all = close_hfq[codes].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Corporate action factor (gross): (1+hfq_ret)/(1+none_ret). Close to 1 on normal days,
    # deviates on dividend/split days or bad ticks. We don't need it for P&L (hfq already embeds it),
    # but we surface large deviations for debugging.
    gross_none = (1.0 + ret_none).astype(float)
    gross_hfq = (1.0 + ret_hfq_all).astype(float)
    corp_factor = (gross_hfq / gross_none).replace([np.inf, -np.inf], np.nan)

    # Use hfq return for holding P&L (total return proxy).
    ret_exec = ret_hfq_all
    port_ret = (w * ret_exec).sum(axis=1)
    port_nav = (1.0 + port_ret).cumprod()
    port_nav.iloc[0] = 1.0

    # Equal-weight benchmark (hfq total return) WITH SAME rebalance frequency:
    # equal-weight across the selected universe only (not including defensive unless user selected it).
    bench_codes = universe[:]  # fixed benchmark universe
    ret_hfq = close_hfq[bench_codes].pct_change().fillna(0.0)
    w_ew = pd.DataFrame(0.0, index=dates, columns=bench_codes)
    n_b = len(bench_codes)
    if n_b <= 0:
        raise ValueError("benchmark universe empty")
    w_eq = 1.0 / n_b
    for i, d in enumerate(decision_dates):
        di = dates.get_loc(d)
        if di + 1 >= len(dates):
            break
        start_i = di + 1
        next_di = (dates.get_loc(decision_dates[i + 1]) if i + 1 < len(decision_dates) else (len(dates) - 1))
        end_i = min(len(dates) - 1, next_di)
        w_ew.loc[dates[start_i] : dates[end_i], bench_codes] = w_eq
    ew_ret = (w_ew * ret_hfq).sum(axis=1)
    ew_nav = (1.0 + ew_ret).cumprod()
    ew_nav.iloc[0] = 1.0

    # Simple turnover and cost: turnover = sum |w_t - w_{t-1}| / 2 ; cost applied to return.
    w_prev = w.shift(1).fillna(0.0)
    turnover = (w - w_prev).abs().sum(axis=1) / 2.0
    cost = turnover * (inp.cost_bps / 10000.0)
    port_ret_net = port_ret - cost
    port_nav_net = (1.0 + port_ret_net).cumprod()
    port_nav_net.iloc[0] = 1.0

    active_ret = port_ret_net - ew_ret
    excess_nav = (1.0 + active_ret).cumprod()
    excess_nav.iloc[0] = 1.0

    attribution = _compute_return_risk_contributions(
        asset_ret=ret_exec[codes],
        weights=w[codes],
        total_return=float(port_nav.iloc[-1] - 1.0),  # gross return (before costs)
    )

    # Metrics
    ann_ret = _annualized_return(port_nav_net)
    ann_vol = _annualized_vol(port_ret_net)
    mdd = _max_drawdown(port_nav_net)
    mdd_dur = _max_drawdown_duration_days(port_nav_net)
    sharpe = _sharpe(port_ret_net, rf=float(inp.risk_free_rate))
    sortino = _sortino(port_ret_net, rf=float(inp.risk_free_rate))
    ui = _ulcer_index(port_nav_net, in_percent=True)
    ui_den = ui / 100.0
    upi = float((ann_ret - float(inp.risk_free_rate)) / ui_den) if ui_den > 0 else float("nan")

    ann_excess = _annualized_return(excess_nav)
    ir = _sharpe(active_ret, rf=0.0)  # same formula but zero rf; for consistency name it IR-style

    metrics = {
        "strategy": {
            "cumulative_return": float(port_nav_net.iloc[-1] - 1.0),
            "annualized_return": float(ann_ret),
            "annualized_volatility": float(ann_vol),
            "max_drawdown": float(mdd),
            "max_drawdown_recovery_days": int(mdd_dur),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "ulcer_index": float(ui),
            "ulcer_performance_index": float(upi),
            "avg_daily_turnover": float(turnover.mean()),
        },
        "equal_weight": {
            "cumulative_return": float(ew_nav.iloc[-1] - 1.0),
        },
        "excess_vs_equal_weight": {
            "cumulative_return": float(excess_nav.iloc[-1] - 1.0),
            "annualized_return": float(ann_excess),
            "information_ratio": float(ir),
        },
    }

    # Period details and win rate / payoff ratio by rebalance periods: compare period returns strategy vs ew.
    period_stats = []
    wins = 0
    pos: list[float] = []
    neg: list[float] = []
    abs_wins = 0
    abs_pos: list[float] = []
    abs_neg: list[float] = []
    prev_weights = {c: 0.0 for c in codes}
    for p in holdings["periods"]:
        s = pd.to_datetime(p["start_date"])
        e = pd.to_datetime(p["end_date"])
        nav_s = float(port_nav_net.loc[s])
        nav_e = float(port_nav_net.loc[e])
        ew_s = float(ew_nav.loc[s])
        ew_e = float(ew_nav.loc[e])
        r_s = nav_e / nav_s - 1.0
        r_ew = ew_e / ew_s - 1.0
        ex = float(r_s - r_ew)
        if ex > 0:
            wins += 1
            pos.append(ex)
        elif ex < 0:
            neg.append(ex)
        if r_s > 0:
            abs_wins += 1
            abs_pos.append(float(r_s))
        elif r_s < 0:
            abs_neg.append(float(r_s))
        # trade details at start_date
        cur_w = {c: float(w.loc[s, c]) if c in w.columns else 0.0 for c in codes}
        buys = []
        sells = []
        for c in codes:
            pw = float(prev_weights.get(c, 0.0))
            nw = float(cur_w.get(c, 0.0))
            if nw > pw + 1e-12:
                buys.append({"code": c, "from_weight": pw, "to_weight": nw, "delta_weight": nw - pw})
            elif pw > nw + 1e-12:
                sells.append({"code": c, "from_weight": pw, "to_weight": nw, "delta_weight": nw - pw})
        prev_weights = cur_w
        period_turnover = float(turnover.loc[s]) if s in turnover.index else None
        period_stats.append(
            {
                "start_date": p["start_date"],
                "end_date": p["end_date"],
                "strategy_return": float(r_s),
                "equal_weight_return": float(r_ew),
                "excess_return": ex,
                "win": ex > 0,
                "buys": buys,
                "sells": sells,
                "turnover": period_turnover,
            }
        )
    total_p = len(period_stats)
    win_rate = float(wins / total_p) if total_p else float("nan")
    avg_win = float(np.mean(pos)) if pos else float("nan")
    avg_loss = float(np.mean(neg)) if neg else float("nan")
    payoff = float(avg_win / abs(avg_loss)) if (pos and neg and avg_loss != 0) else float("nan")
    # Kelly fraction (binary approximation): f* = p - (1-p)/b, where b is payoff ratio
    if total_p and np.isfinite(win_rate) and np.isfinite(payoff) and payoff > 0:
        kelly = float(win_rate - (1.0 - win_rate) / payoff)
    else:
        kelly = float("nan")

    abs_win_rate = float(abs_wins / total_p) if total_p else float("nan")
    abs_avg_win = float(np.mean(abs_pos)) if abs_pos else float("nan")
    abs_avg_loss = float(np.mean(abs_neg)) if abs_neg else float("nan")
    abs_payoff = float(abs_avg_win / abs(abs_avg_loss)) if (abs_pos and abs_neg and abs_avg_loss != 0) else float("nan")
    if total_p and np.isfinite(abs_win_rate) and np.isfinite(abs_payoff) and abs_payoff > 0:
        abs_kelly = float(abs_win_rate - (1.0 - abs_win_rate) / abs_payoff)
    else:
        abs_kelly = float("nan")

    stats = {
        "rebalance": inp.rebalance,
        "periods": total_p,
        # relative vs equal-weight (excess)
        "win_rate": win_rate,
        "avg_win_excess": avg_win,
        "avg_loss_excess": avg_loss,
        "payoff_ratio": payoff,
        "kelly_fraction": kelly,
        # absolute (strategy itself)
        "abs_win_rate": abs_win_rate,
        "abs_avg_win": abs_avg_win,
        "abs_avg_loss": abs_avg_loss,
        "abs_payoff_ratio": abs_payoff,
        "abs_kelly_fraction": abs_kelly,
    }

    # Periodic returns for strategy (none) and benchmark (hfq), full lists
    def _period_returns(nav_s: pd.Series, nav_b: pd.Series, freq: str) -> list[dict[str, Any]]:
        s = nav_s.copy()
        s.index = pd.to_datetime(s.index)
        b = nav_b.copy()
        b.index = pd.to_datetime(b.index)
        s_r = s.resample(freq).last().pct_change().dropna()
        b_r = b.resample(freq).last().pct_change().dropna()
        idx = s_r.index.intersection(b_r.index)
        out_rows = []
        for t in idx:
            rs = float(s_r.loc[t])
            rb = float(b_r.loc[t])
            out_rows.append(
                {
                    "period_end": t.date().isoformat(),
                    "strategy_return": rs,
                    "benchmark_return": rb,
                    "excess_return": rs - rb,
                }
            )
        return out_rows

    periodic = {
        "weekly": _period_returns(port_nav_net, ew_nav, "W-FRI"),
        "monthly": _period_returns(port_nav_net, ew_nav, "ME"),
        "quarterly": _period_returns(port_nav_net, ew_nav, "QE"),
        "yearly": _period_returns(port_nav_net, ew_nav, "YE"),
    }

    # Rolling stats for strategy vs benchmark (defaults aligned with baseline UI)
    rolling = {"returns": {}, "max_drawdown": {}}
    for weeks in (4, 12, 52):
        window = weeks * 5
        rolling["returns"][f"{weeks}w"] = (port_nav_net / port_nav_net.shift(window) - 1.0).dropna()
        rolling["max_drawdown"][f"{weeks}w"] = _rolling_max_drawdown(port_nav_net, window).dropna()
    for months in (3, 6, 12):
        window = months * 21
        rolling["returns"][f"{months}m"] = (port_nav_net / port_nav_net.shift(window) - 1.0).dropna()
        rolling["max_drawdown"][f"{months}m"] = _rolling_max_drawdown(port_nav_net, window).dropna()
    for years in (1, 3):
        window = years * 252
        rolling["returns"][f"{years}y"] = (port_nav_net / port_nav_net.shift(window) - 1.0).dropna()
        rolling["max_drawdown"][f"{years}y"] = _rolling_max_drawdown(port_nav_net, window).dropna()
    rolling_out = {
        "returns": {k: {"dates": v.index.date.astype(str).tolist(), "values": v.astype(float).tolist()} for k, v in rolling["returns"].items()},
        "max_drawdown": {k: {"dates": v.index.date.astype(str).tolist(), "values": v.astype(float).tolist()} for k, v in rolling["max_drawdown"].items()},
    }

    # Collect large corporate action factor events for transparency/debugging (cap size to avoid huge payloads).
    corporate_actions: list[dict[str, Any]] = []
    if corp_factor.to_numpy().size:
        # flag: factor deviates > 2% (covers typical cash distributions) or is extreme (splits/merges)
        dev = (corp_factor - 1.0).abs()
        mask = (dev > 0.02) | (corp_factor > 1.2) | (corp_factor < 1.0 / 1.2)
        # cap at 200 events, prioritize largest deviation
        try:
            events = []
            for c in codes:
                if c not in corp_factor.columns:
                    continue
                idx = corp_factor.index[mask[c].fillna(False)]
                for d in idx:
                    f = corp_factor.loc[d, c]
                    if pd.isna(f):
                        continue
                    events.append((float(dev.loc[d, c]), c, d, float(f)))
            events.sort(reverse=True, key=lambda x: x[0])
            for _, c, d, f in events[:200]:
                corporate_actions.append(
                    {
                        "code": c,
                        "date": d.date().isoformat(),
                        "none_return": float(ret_none.loc[d, c]),
                        "hfq_return": float(ret_hfq_all.loc[d, c]),
                        "corp_factor": float(f),
                    }
                )
        except (ValueError, TypeError, KeyError):  # pragma: no cover - defensive, should not break backtest
            corporate_actions = []

    out = {
        "date_range": {"start": inp.start.strftime("%Y%m%d"), "end": inp.end.strftime("%Y%m%d")},
        "codes": codes,
        "benchmark_codes": bench_codes,
        "price_basis": {
            "signal": "hfq",
            "strategy_nav": "none execution + hfq-implied corporate action factor (total return proxy)",
            "benchmark_nav": "hfq",
        },
        "nav": {
            "dates": dates.date.astype(str).tolist(),
            "series": {
                "ROTATION": port_nav_net.astype(float).tolist(),
                "EW_REBAL": ew_nav.astype(float).tolist(),
                "EXCESS": excess_nav.astype(float).tolist(),
            },
        },
        "attribution": attribution,
        "metrics": metrics,
        "win_payoff": stats,
        "period_returns": periodic,
        "rolling": rolling_out,
        "period_details": period_stats,
        "holdings": holdings["periods"],
        "corporate_actions": corporate_actions,
    }
    return out

