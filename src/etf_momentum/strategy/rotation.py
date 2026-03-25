from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

# Shared helper for rebalance date shifting (to avoid redefining inside blocks)
def _shift_idx_by_rebalance(target: pd.Timestamp, dates: pd.DatetimeIndex, reb_shift: str) -> int:
    t = pd.to_datetime(target).normalize()
    if t in dates:
        return int(dates.get_loc(t))
    pos = int(dates.searchsorted(t))
    if reb_shift == "next":
        return int(min(pos, len(dates) - 1))
    return int(max(pos - 1, 0))

from ..analysis.baseline import (
    TRADING_DAYS_PER_YEAR,
    _annualized_return,
    _annualized_vol,
    _max_drawdown,
    _max_drawdown_duration_days,
    _rolling_max_drawdown,
    _rsi_wilder,
    _sharpe,
    _sortino,
    _ulcer_index,
    hfq_close_daily_equal_weight_returns,
    load_volume_amount as _load_volume_amount,
)
from ..analysis.baseline import load_close_prices as _load_close_prices
from ..analysis.baseline import load_high_low_prices as _load_high_low_prices
from ..analysis.baseline import load_ohlc_prices as _load_ohlc_prices
from ..analysis.baseline import _compute_return_risk_contributions as _compute_return_risk_contributions
from ..analysis.execution_timing import corporate_action_mask, forward_returns


@dataclass(frozen=True)
class RotationInputs:
    codes: list[str]
    start: dt.date
    end: dt.date
    rebalance: str = "weekly"  # daily/weekly/monthly/quarterly/yearly
    rebalance_anchor: int | None = None  # weekly:1..5; monthly:1..28; quarterly:1..90; yearly:1..365
    rebalance_shift: str = "prev"  # prev|next|skip when anchor falls on non-trading day
    top_k: int = 1  # |K|>0: positive=top-K by score; negative=bottom-K (inverse); effective count=min(|K|, pool)
    position_mode: str = "adaptive"  # adaptive | fixed | risk_budget (base sizing before other exposure controls)
    risk_budget_atr_window: int = 20  # n-day ATR window for risk-budget sizing
    risk_budget_pct: float = 0.01  # per-asset risk budget on total NAV (1% => 0.01)
    entry_backfill: bool = False  # if true, refill from lower-ranked candidates when top_k entries are filtered out
    entry_match_n: int = 0  # 0 => default AND(all enabled); otherwise require at least n entry filters
    exit_match_n: int = 0  # 0 => default AND(all enabled); otherwise require at least n exit filters
    lookback_days: int = 20
    skip_days: int = 0  # skip recent trading days (0 means no skip)
    risk_free_rate: float = 0.025
    cost_bps: float = 0.0  # round-trip cost in bps per turnover, simple approximation
    slippage_rate: float = 0.001  # execution slippage per one-way turnover, always adverse
    # --- Ranking method ---
    score_method: str = "raw_mom"  # raw_mom | sharpe_mom | sortino_mom
    # --- Pre-trade risk controls (drawdown prevention heuristics) ---
    # Trend filter: decide whether to buy at all, and/or exclude candidates that are not in trend.
    trend_filter: bool = False
    trend_exit_filter: bool = False
    trend_sma_window: int = 20  # trading days (weekly use-case default)
    trend_ma_type: str = "sma"  # sma | ema | vma(variable/adaptive)
    bias_filter: bool = False
    bias_exit_filter: bool = False
    bias_type: str = "bias"  # bias | bias_v
    bias_ma_window: int = 20
    bias_level_window: str = "all"
    bias_threshold_type: str = "quantile"  # quantile | fixed
    bias_quantile: float = 95.0  # percent, e.g. 95
    bias_fixed_value: float = 10.0  # percent, e.g. 10 means 10%
    bias_min_periods: int = 20
    # RSI filter: avoid buying overbought / oversold assets.
    rsi_filter: bool = False
    rsi_window: int = 20
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_block_overbought: bool = True
    rsi_block_oversold: bool = False
    # Volatility monitor: scale position size (cash remainder) to reduce exposure during high vol.
    vol_monitor: bool = False
    vol_window: int = 20
    vol_target_ann: float = 0.20  # annualized target vol; scales down if realized vol is higher
    vol_max_ann: float = 0.60  # annualized hard stop; above this -> no risk position (cash/defensive)
    # Choppiness (range-bound) filter via Efficiency Ratio (close-only)
    chop_filter: bool = False
    chop_mode: str = "er"  # "er" | "adx"
    chop_window: int = 20
    chop_er_threshold: float = 0.25  # ER < threshold => choppy => exclude
    chop_adx_window: int = 20
    chop_adx_threshold: float = 20.0  # ADX < threshold => choppy => exclude
    # --- Universal ATR stop-loss (qfq price basis; aligned with trend research) ---
    # none | static | trailing | tightening
    atr_stop_mode: str = "none"
    atr_stop_atr_basis: str = "latest"  # entry | latest (for trailing/tightening)
    atr_stop_reentry_mode: str = "reenter"  # reenter | wait_next_entry (reserved for parity with trend)
    atr_stop_window: int = 14
    atr_stop_n: float = 2.0
    atr_stop_m: float = 0.5
    # --- Correlation filter (qfq price basis) ---
    corr_filter: bool = False
    corr_window: int | None = None  # None -> defaults to lookback_days
    corr_threshold: float = 0.5
    # --- Group constraint (cross-strategy diversification) ---
    group_enforce: bool = False
    group_pick_policy: str = "strongest_score"  # strongest_score | earliest_entry | lowest_vol
    asset_groups: dict[str, str] | None = None  # code -> group_id
    # --- Inertia / dampening (avoid frequent rebalances) ---
    inertia: bool = False
    inertia_min_hold_periods: int = 0  # minimum decision periods between holding changes (0 disables)
    inertia_score_gap: float = 0.0  # only for |top_k|=1: require new_score - cur_score >= gap to switch
    inertia_min_turnover: float = 0.0  # if expected turnover < threshold, skip rebalance (0 disables)
    # --- Rolling-return based position sizing (strategy trailing return) ---
    rr_sizing: bool = False
    rr_years: float = 3.0
    rr_thresholds: list[float] | None = None  # max 5
    rr_weights: list[float] | None = None  # len = len(thresholds)+1
    # --- "Rearview mirror" composite deviation-based exposure cap (universe-level, expanding percentiles) ---
    mirror_control: bool = False
    mirror_quantiles: list[float] | None = None  # in (0,1); default [0.9,0.95,0.99]
    mirror_exposures: list[float] | None = None  # in [0,1], len = len(quantiles); default [0.8,0.5,0.2]
    # --- Drawdown control (strategy NAV) ---
    dd_control: bool = False
    dd_threshold: float = 0.10  # decimal, e.g. 0.10 = 10%
    dd_reduce: float = 1.0  # fraction to reduce, e.g. 1.0 => reduce 100% -> cash
    dd_sleep_days: int = 20  # trading days
    # --- Phase-1 per-asset parameter rules (optional) ---
    # These override the corresponding global params when provided.
    asset_momentum_floor_rules: list[dict[str, Any]] | None = None
    asset_trend_rules: list[dict[str, Any]] | None = None
    asset_bias_rules: list[dict[str, Any]] | None = None
    asset_rsi_rules: list[dict[str, Any]] | None = None
    asset_chop_rules: list[dict[str, Any]] | None = None
    asset_vol_monitor_rules: list[dict[str, Any]] | None = None
    # --- Execution price proxy for benchmark/fallback (hfq, aligned to execution calendar) ---
    # Used to study open/close/OC(=avg(open,close)) calendar effects.
    exec_price: str = "open"  # close | open | oc2
    # --- Per-asset risk control (qfq close-based signals; daily weight scaling) ---
    asset_rc_rules: list[dict[str, Any]] | None = None
    # --- Per-asset volatility-index timing (daily weight scaling; cash remainder) ---
    # Rules define which vol index (e.g. VIX/GVZ) to use per ETF.
    # Vol index levels are expected to be preloaded by caller (API layer) to avoid network dependency here.
    asset_vol_index_rules: list[dict[str, Any]] | None = None
    vol_index_close: dict[str, pd.Series] | None = None
    # Dynamic universe over union interval.
    dynamic_universe: bool = False


def _rebalance_labels(index: pd.DatetimeIndex, rebalance: str, *, weekly_anchor: str = "FRI") -> pd.PeriodIndex:
    r = (rebalance or "monthly").lower()
    anchor = str(weekly_anchor).strip().upper()
    if anchor not in {"MON", "TUE", "WED", "THU", "FRI"}:
        raise ValueError(f"invalid weekly_anchor={weekly_anchor} (expected MON..FRI)")
    freq_map = {"daily": "D", "weekly": f"W-{anchor}", "monthly": "M", "quarterly": "Q", "yearly": "Y"}
    if r not in freq_map:
        raise ValueError(f"invalid rebalance={rebalance}")
    return index.to_period(freq_map[r])


def _dist_stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([float(x) for x in (values or []) if np.isfinite(float(x))], dtype=float)
    if arr.size == 0:
        return {
            "count": 0,
            "max": None,
            "min": None,
            "mean": None,
            "std": None,
            "quantiles": {k: None for k in ["p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]},
        }
    q = lambda p: float(np.percentile(arr, p))  # noqa: E731
    return {
        "count": int(arr.size),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0,
        "quantiles": {
            "p01": q(1),
            "p05": q(5),
            "p10": q(10),
            "p25": q(25),
            "p50": q(50),
            "p75": q(75),
            "p90": q(90),
            "p95": q(95),
            "p99": q(99),
        },
    }


def _trade_stats_from_returns(values: list[float], *, flat_eps: float = 1e-12) -> dict[str, Any]:
    rs = [float(x) for x in (values or []) if np.isfinite(float(x))]
    wins = [x for x in rs if x > float(flat_eps)]
    losses = [x for x in rs if x < -float(flat_eps)]
    flats = [x for x in rs if abs(float(x)) <= float(flat_eps)]
    # Kelly (exclude flat/zero-return trades): f* = p - (1-p)/b, b=avg_win/|avg_loss|
    # Return None when the denominator is not well-defined (no wins/losses).
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


def _trade_returns_from_weight_series(
    w: pd.Series,
    ret_exec: pd.Series,
    *,
    cost_bps: float,
    slippage_rate: float,
    dates: pd.Index,
    eps: float = 1e-12,
) -> dict[str, Any]:
    ww = pd.to_numeric(w, errors="coerce").astype(float).reindex(dates).fillna(0.0)
    rr = pd.to_numeric(ret_exec, errors="coerce").astype(float).reindex(dates).fillna(0.0)
    n = int(len(dates))
    if n <= 0:
        return {"returns": [], "trades": []}
    cost_rate = float(cost_bps) / 10000.0
    slip_rate = float(slippage_rate)
    returns: list[float] = []
    trades: list[dict[str, Any]] = []
    active = False
    start_i = -1
    start_nav = 1.0
    nav_prev = 1.0
    for i in range(n):
        cur = float(ww.iloc[i])
        prev = float(ww.iloc[i - 1]) if i > 0 else 0.0
        r = float(rr.iloc[i]) if np.isfinite(float(rr.iloc[i])) else 0.0
        turnover = abs(float(cur) - float(prev)) / 2.0
        day_ret = float(cur) * float(r) - float(turnover) * (float(cost_rate) + float(slip_rate))
        if (not active) and (prev <= eps) and (cur > eps):
            active = True
            start_i = int(i)
            start_nav = float(nav_prev)
        nav_cur = float(nav_prev) * (1.0 + float(day_ret))
        # Trade ends on the execution day when position becomes flat (exit cost is booked on this day).
        if active and (prev > eps) and (cur <= eps):
            tr = (float(nav_cur) / float(start_nav) - 1.0) if float(start_nav) != 0 else float("nan")
            returns.append(float(tr))
            trades.append(
                {
                    "entry_date": str(pd.to_datetime(dates[start_i]).date()) if start_i >= 0 else None,
                    "exit_date": str(pd.to_datetime(dates[i]).date()),
                    "return": float(tr),
                    "closed": True,
                }
            )
            active = False
            start_i = -1
            start_nav = float(nav_cur)
        nav_prev = float(nav_cur)
    # If trade is still open at the end, include mark-to-market return up to last available date.
    if active and start_i >= 0:
        tr = (float(nav_prev) / float(start_nav) - 1.0) if float(start_nav) != 0 else float("nan")
        returns.append(float(tr))
        trades.append(
            {
                "entry_date": str(pd.to_datetime(dates[start_i]).date()),
                "exit_date": str(pd.to_datetime(dates[n - 1]).date()),
                "return": float(tr),
                "closed": False,
            }
        )
    return {"returns": returns, "trades": trades}


def _trade_returns_from_weight_df(
    w: pd.DataFrame,
    ret_exec: pd.DataFrame,
    *,
    cost_bps: float,
    slippage_rate: float,
    dates: pd.Index,
    eps: float = 1e-12,
) -> dict[str, Any]:
    by_code_returns: dict[str, list[float]] = {}
    by_code_trades: dict[str, list[dict[str, Any]]] = {}
    for c in [str(x) for x in w.columns]:
        one = _trade_returns_from_weight_series(
            w[c] if c in w.columns else pd.Series(dtype=float),
            ret_exec[c] if c in ret_exec.columns else pd.Series(dtype=float),
            cost_bps=float(cost_bps),
            slippage_rate=float(slippage_rate),
            dates=dates,
            eps=float(eps),
        )
        by_code_returns[str(c)] = [float(x) for x in (one.get("returns") or [])]
        by_code_trades[str(c)] = list(one.get("trades") or [])
    all_returns: list[float] = []
    all_trades: list[dict[str, Any]] = []
    for c in by_code_returns:
        all_returns.extend(by_code_returns[c])
        all_trades.extend([{**x, "code": str(c)} for x in by_code_trades.get(c, [])])
    return {
        "returns": [float(x) for x in all_returns],
        "trades": all_trades,
        "returns_by_code": by_code_returns,
        "trades_by_code": by_code_trades,
    }


def _momentum_scores(close_qfq: pd.DataFrame, *, lookback_days: int, skip_days: int) -> pd.DataFrame:
    # score[t] = close[t-skip]/close[t-skip-lookback] - 1
    lag = skip_days
    lb = lookback_days
    return close_qfq.shift(lag) / close_qfq.shift(lag + lb) - 1.0


def _rolling_prod_minus_1(gross: pd.DataFrame, *, window: int) -> pd.DataFrame:
    w = max(1, int(window))
    # rolling product is not built-in for DataFrame; use apply on ndarray for speed-enough.
    return gross.rolling(window=w, min_periods=max(2, w // 2)).apply(lambda x: float(np.prod(x)) - 1.0, raw=True)


def _risk_adjusted_scores(
    close_qfq: pd.DataFrame,
    *,
    lookback_days: int,
    skip_days: int,
    method: str,
    rf_annual: float,
) -> pd.DataFrame:
    """
    Compute alternative momentum scores for ranking:
    - sharpe_mom: Sharpe ratio over lookback window
    - sortino_mom: Sortino ratio over lookback window

    All are computed on qfq daily close-to-close returns, with window ending at (t - skip_days).
    """
    m = (method or "raw_mom").strip().lower()
    if m not in {"sharpe_mom", "sortino_mom"}:
        raise ValueError(f"invalid score_method={method}")

    lb = max(1, int(lookback_days))
    lag = max(0, int(skip_days))
    rf_daily = float(rf_annual) / 252.0

    ret = close_qfq.pct_change().replace([np.inf, -np.inf], np.nan)
    # Align the window to end at (t - lag): shift returns forward by lag.
    ret = ret.shift(lag)

    # window stats
    mean = ret.rolling(window=lb, min_periods=max(3, lb // 2)).mean()
    std = ret.rolling(window=lb, min_periods=max(3, lb // 2)).std(ddof=1)

    # robust division: if denom is ~0, map to large +/- depending on numerator sign (for deterministic ranking)
    def safe_div(numer: pd.DataFrame, denom: pd.DataFrame) -> pd.DataFrame:
        eps = 1e-12
        out = numer / denom.replace(0.0, np.nan)
        small = denom.abs() < eps
        if small.to_numpy().any():
            sign = np.sign(numer.where(small, other=0.0))
            out = out.mask(small & (sign > 0), other=1e9)
            out = out.mask(small & (sign < 0), other=-1e9)
            out = out.mask(small & (sign == 0), other=0.0)
        return out

    # sharpe/sortino use excess mean over rf
    excess_mean = (mean - rf_daily).astype(float)
    if m == "sharpe_mom":
        return safe_div(excess_mean, std.astype(float))

    # sortino: downside deviation on (ret - rf_daily)
    downside = (ret - rf_daily).clip(upper=0.0)
    dd = downside.rolling(window=lb, min_periods=max(3, lb // 2)).std(ddof=1)
    return safe_div(excess_mean, dd.astype(float))


def _trend_ok_each(close: pd.DataFrame, *, ma_window: int, ma_type: str = "sma", op: str = ">") -> pd.DataFrame:
    """
    Simple trend filter: close > MA(close, window).
    Returns boolean DataFrame aligned to `close`.
    """
    w = max(1, int(ma_window))
    mt = str(ma_type or "sma").strip().lower()
    if mt == "ema":
        ma = close.ewm(span=w, adjust=False, min_periods=max(2, w // 2)).mean()
    elif mt == "vma":
        # VMA (Variable/Adaptive MA): EMA with adaptive alpha driven by variability.
        # alpha_t = base_alpha * |CMO_t|, CMO computed over `w`.
        base_alpha = 2.0 / (float(w) + 1.0)
        diff = close.diff()
        up = diff.clip(lower=0.0).rolling(window=w, min_periods=max(2, w // 2)).sum()
        down = (-diff.clip(upper=0.0)).rolling(window=w, min_periods=max(2, w // 2)).sum()
        den = (up + down).replace(0.0, np.nan)
        cmo_abs = (up - down).abs() / den
        alpha_df = (base_alpha * cmo_abs).clip(lower=0.0, upper=1.0).fillna(0.0)
        ma = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        for col in close.columns:
            px = pd.to_numeric(close[col], errors="coerce").astype(float)
            aa = pd.to_numeric(alpha_df[col], errors="coerce").astype(float).fillna(0.0)
            vals = px.to_numpy(dtype=float, copy=False)
            alphas = aa.to_numpy(dtype=float, copy=False)
            out = np.full(len(vals), np.nan, dtype=float)
            prev = np.nan
            for i, v in enumerate(vals):
                if not np.isfinite(v):
                    out[i] = prev
                    continue
                if not np.isfinite(prev):
                    prev = float(v)
                else:
                    prev = float(prev + alphas[i] * (float(v) - prev))
                out[i] = prev
            ma[col] = out
    else:
        ma = close.rolling(window=w, min_periods=max(2, w // 2)).mean()
    o = _normalize_cmp_op(op)
    if o == "gt":
        out = close > ma
    elif o == "lt":
        out = close < ma
    elif o == "ge":
        out = close >= ma
    elif o == "le":
        out = close <= ma
    elif o == "eq":
        out = close == ma
    elif o == "ne":
        out = close != ma
    else:
        out = close > ma
    return out.fillna(False)


def _rsi(close: pd.DataFrame, *, window: int) -> pd.DataFrame:
    """
    RSI (Wilder-style smoothing via EWM).
    Returns DataFrame in [0, 100], aligned to `close`.
    """
    w = max(1, int(window))
    diff = close.diff()
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)
    # Wilder uses alpha=1/w
    avg_gain = gain.ewm(alpha=1.0 / w, adjust=False, min_periods=w).mean()
    avg_loss = loss.ewm(alpha=1.0 / w, adjust=False, min_periods=w).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # If avg_loss==0 and avg_gain>0, RSI should be 100. If both 0, RSI undefined -> NaN.
    rsi = rsi.fillna(100.0).clip(lower=0.0, upper=100.0)
    return rsi


def _ann_realized_vol_from_close(close: pd.DataFrame, *, window: int) -> pd.DataFrame:
    """
    Annualized realized vol from daily close-to-close returns.
    """
    w = max(2, int(window))
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    vol_daily = ret.rolling(window=w, min_periods=max(3, w // 2)).std(ddof=1)
    return (vol_daily * np.sqrt(252.0)).astype(float)


def _atr_from_hlc(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, *, window: int) -> pd.DataFrame:
    """
    Classic ATR (Wilder-style RMA) from high/low/close.
    """
    n = max(2, int(window))
    hi = high.astype(float)
    lo = low.astype(float)
    cl = close.astype(float)
    prev_cl = cl.shift(1)
    tr1 = (hi - lo).abs()
    tr2 = (hi - prev_cl).abs()
    tr3 = (lo - prev_cl).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=0).groupby(level=0).max()
    atr = tr.ewm(alpha=1.0 / float(n), adjust=False, min_periods=n).mean()
    return atr.replace([np.inf, -np.inf], np.nan).astype(float)


def _efficiency_ratio(close: pd.DataFrame, *, window: int) -> pd.DataFrame:
    """
    Kaufman Efficiency Ratio (ER) using close only.
    ER = abs(close_t - close_{t-n}) / sum_{i=1..n} abs(close_{t-i+1} - close_{t-i})
    ER near 0 => choppy; ER near 1 => trending.
    """
    w = max(2, int(window))
    net = (close - close.shift(w)).abs()
    noise = close.diff().abs().rolling(window=w, min_periods=max(3, w // 2)).sum()
    er = net / noise.replace(0.0, np.nan)
    return er.astype(float)


def _adx(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, *, window: int) -> pd.DataFrame:
    """
    Average Directional Index (ADX) with Wilder-style smoothing (RMA via EWM alpha=1/window).

    Output is in the standard 0..100 scale; lower ADX indicates weaker trend / more range-bound.
    """
    n = max(2, int(window))
    hi = high.astype(float)
    lo = low.astype(float)
    cl = close.astype(float)

    prev_cl = cl.shift(1)
    prev_hi = hi.shift(1)
    prev_lo = lo.shift(1)

    tr1 = (hi - lo).abs()
    tr2 = (hi - prev_cl).abs()
    tr3 = (lo - prev_cl).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=0).groupby(level=0).max()

    up_move = hi - prev_hi
    down_move = prev_lo - lo
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)

    alpha = 1.0 / float(n)
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=n).mean()
    plus_di = 100.0 * (plus_dm.ewm(alpha=alpha, adjust=False, min_periods=n).mean() / atr)
    minus_di = 100.0 * (minus_dm.ewm(alpha=alpha, adjust=False, min_periods=n).mean() / atr)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=n).mean()
    return adx.replace([np.inf, -np.inf], np.nan).astype(float)


def _merge_rule(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any]:
    """
    Field-level merge for per-asset rules.
    - override keys with non-None values take precedence
    - `code` is kept from override (or base)
    """
    out: dict[str, Any] = dict(base or {})
    o = dict(override or {})
    for k, v in o.items():
        if k == "code":
            continue
        if v is not None:
            out[k] = v
    out["code"] = str(o.get("code") or out.get("code") or "*")
    return out


def _effective_rules_for_code(code: str, rules: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """
    Resolve effective rule list for a given asset code.

    Semantics:
    - `code="*"` is the default rule (if present)
    - any number of per-code rules may exist; each is merged with the default (field-level)
    - if no per-code rule exists, fall back to default-only (if present)
    """
    if rules is None or len(rules) == 0:
        return []
    c = str(code or "").strip()
    if not c:
        return []
    default_rule: dict[str, Any] | None = None
    specifics: list[dict[str, Any]] = []
    for r in rules:
        rc = str((r or {}).get("code") or "").strip()
        if rc == "*":
            default_rule = dict(r or {})
        elif rc == c:
            specifics.append(dict(r or {}))
    if specifics:
        return [_merge_rule(default_rule, r) for r in specifics]
    return [dict(default_rule)] if default_rule else []


def _normalize_cmp_op(op: Any) -> str:
    s = str(op or "").strip().lower()
    mapping = {
        ">": "gt",
        "gt": "gt",
        "greater": "gt",
        "<": "lt",
        "lt": "lt",
        "less": "lt",
        ">=": "ge",
        "ge": "ge",
        "=>": "ge",
        "<=": "le",
        "le": "le",
        "=<": "le",
        "==": "eq",
        "=": "eq",
        "eq": "eq",
        "!=": "ne",
        "<>": "ne",
        "ne": "ne",
    }
    return mapping.get(s, "gt")


def _compare_with_op(x: float, y: float, op: str) -> bool:
    o = _normalize_cmp_op(op)
    if o == "gt":
        return bool(float(x) > float(y))
    if o == "lt":
        return bool(float(x) < float(y))
    if o == "ge":
        return bool(float(x) >= float(y))
    if o == "le":
        return bool(float(x) <= float(y))
    if o == "eq":
        return bool(float(x) == float(y))
    if o == "ne":
        return bool(float(x) != float(y))
    return bool(float(x) > float(y))


def _momentum_rule_threshold_raw(rule: dict[str, Any], fallback: float) -> float:
    v = (rule or {}).get("threshold")
    if v is None:
        v = (rule or {}).get("momentum_floor")
    if v is None:
        v = fallback
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = float(fallback)
    if not np.isfinite(x):
        x = float(fallback)
    unit = str((rule or {}).get("threshold_unit") or "raw").strip().lower()
    if unit in {"pct", "percent", "%"}:
        return float(x) / 100.0
    return float(x)


def _momentum_rules_for_stage(code: str, *, rules: list[dict[str, Any]] | None, stage: str) -> list[dict[str, Any]]:
    eff = _effective_rules_for_code(code, rules)
    out: list[dict[str, Any]] = []
    st = str(stage or "entry").strip().lower()
    for r in eff:
        rs = str((r or {}).get("stage") or "entry").strip().lower()
        if rs in {"both", st}:
            out.append(dict(r or {}))
    return out


def _momentum_rules_pass(score: float, *, rules: list[dict[str, Any]], fallback_floor: float) -> bool:
    if not np.isfinite(float(score)):
        return False
    if not rules:
        return bool(float(score) > float(fallback_floor))
    for r in rules:
        op = _normalize_cmp_op((r or {}).get("op") or ">")
        thr = _momentum_rule_threshold_raw(dict(r or {}), fallback=float(fallback_floor))
        if not _compare_with_op(float(score), float(thr), op):
            return False
    return True


def _apply_asset_rc_rules(
    w: pd.DataFrame,
    *,
    close_qfq: pd.DataFrame,
    rules: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """
    Apply per-asset risk-control rules by scaling weights daily (cash remainder).

    Notes:
    - Signals are computed on qfq close series.
    - Thresholds use full-sample percentiles (research convenience; may introduce look-ahead bias).
    - Decision uses yesterday's signal to affect today's exposure (avoid look-ahead on the day itself).
    """
    if rules is None or len(rules) == 0 or w.empty:
        return {"enabled": False, "rules": []}

    idx = w.index
    out_rules: list[dict[str, Any]] = []

    def _pct(v: Any) -> float:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return float("nan")
        return x

    def _sig_series(px: pd.Series, *, sig_type: str, k: int) -> pd.Series:
        s = px.astype(float).replace([np.inf, -np.inf], np.nan).ffill()
        r = s.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        if sig_type == "return":
            return (s / s.shift(int(k)) - 1.0).astype(float)
        if sig_type == "volatility":
            return r.rolling(window=int(k), min_periods=max(2, int(k) // 2)).std(ddof=1).astype(float)
        if sig_type == "downside_vol":
            # rolling std of negative returns within window
            def f(x: np.ndarray) -> float:
                xx = x[np.isfinite(x)]
                xx = xx[xx < 0.0]
                if xx.size < 2:
                    return float("nan")
                return float(np.std(xx, ddof=1))

            return r.rolling(window=int(k), min_periods=max(2, int(k) // 2)).apply(lambda x: f(x.to_numpy(dtype=float)), raw=False)
        if sig_type == "drawdown":
            peak = s.rolling(window=int(k), min_periods=max(2, int(k) // 2)).max()
            dd = s / peak - 1.0
            return (-dd).astype(float)
        return pd.Series(np.nan, index=s.index, dtype=float)

    def _quantile(x: pd.Series, q: float) -> float:
        v = x.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        if v.empty:
            return float("nan")
        return float(np.quantile(v.to_numpy(dtype=float), q))

    for r in rules:
        code = str(r.get("code") or "")
        if not code:
            continue
        if code not in w.columns or code not in close_qfq.columns:
            continue
        sig_type = str(r.get("sig_type") or "").strip().lower()
        if sig_type not in {"return", "volatility", "downside_vol", "drawdown"}:
            continue
        k = int(max(2, int(float(r.get("k") or 20))))
        p_in = _pct(r.get("p_in"))
        if not np.isfinite(p_in) or p_in <= 0 or p_in >= 100:
            continue
        reduce_pct = _pct(r.get("reduce_pct"))
        reduce_pct = float(np.clip(reduce_pct if np.isfinite(reduce_pct) else 0.0, 0.0, 100.0))
        rec = str(r.get("recovery_mode") or "immediate").strip().lower()
        if rec not in {"immediate", "hysteresis", "cooldown"}:
            rec = "immediate"
        p_out = _pct(r.get("p_out"))
        if not np.isfinite(p_out):
            p_out = float("nan")
        cd = int(max(0, int(float(r.get("cooldown_days") or 0))))

        px = close_qfq[code].reindex(idx).astype(float).replace([np.inf, -np.inf], np.nan).ffill()
        sig = _sig_series(px, sig_type=sig_type, k=k).reindex(idx)

        is_low_tail = sig_type == "return"
        q_in = (1.0 - p_in / 100.0) if is_low_tail else (p_in / 100.0)
        thr_in = _quantile(sig, q_in)
        thr_out = float("nan")
        if rec == "hysteresis" and np.isfinite(p_out) and 0 < p_out < 100:
            q_out = (1.0 - p_out / 100.0) if is_low_tail else (p_out / 100.0)
            thr_out = _quantile(sig, q_out)

        expo = np.ones(len(idx), dtype=float)
        in_reduced = False
        cd_left = 0
        reduce = 1.0 - float(reduce_pct) / 100.0
        for t in range(1, len(idx)):
            s_prev = float(sig.iloc[t - 1]) if (t - 1) < len(sig) else float("nan")
            trig_in = False
            if np.isfinite(s_prev) and np.isfinite(thr_in):
                trig_in = (s_prev <= thr_in) if is_low_tail else (s_prev >= thr_in)
            if not in_reduced:
                if trig_in:
                    in_reduced = True
                    cd_left = cd
            else:
                if cd_left > 0:
                    cd_left -= 1
                elif rec == "hysteresis" and np.isfinite(thr_out):
                    trig_out = False
                    if np.isfinite(s_prev):
                        trig_out = (s_prev >= thr_out) if is_low_tail else (s_prev <= thr_out)
                    if trig_out:
                        in_reduced = False
                else:
                    if not trig_in:
                        in_reduced = False
            expo[t] = reduce if in_reduced else 1.0

        w[code] = w[code].astype(float).to_numpy(dtype=float) * expo
        out_rules.append(
            {
                "code": code,
                "sig_type": sig_type,
                "k": int(k),
                "p_in": float(p_in),
                "reduce_pct": float(reduce_pct),
                "recovery_mode": rec,
                "p_out": (None if not np.isfinite(p_out) else float(p_out)),
                "cooldown_days": int(cd),
            }
        )

    return {"enabled": bool(out_rules), "rules": out_rules}


def _vol_level_window_days(window: str) -> int | None:
    """
    Map window key to trading days. Supported:
    - rolling: 30d/90d/180d/1y/3y/5y/10y
    - expanding: all
    """
    w = str(window or "all").strip().lower()
    if w in {"all", "expanding"}:
        return None
    if w.endswith("d"):
        try:
            return int(max(2, int(w[:-1])))
        except (TypeError, ValueError):
            return None
    if w.endswith("y"):
        try:
            y = float(w[:-1])
        except (TypeError, ValueError):
            return None
        return int(max(2, round(TRADING_DAYS_PER_YEAR * y)))
    return None


def _tiered_exposure_from_level_quantiles(
    levels: pd.Series,
    *,
    level_window: str,
    quantiles: list[float],
    exposures: list[float],
    min_periods: int = 20,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Compute daily tiered exposure series from a "level" series using quantile thresholds.

    No-lookahead guarantee:
    - rolling/expanding quantile thresholds are computed from history and shifted by 1 day.

    Exposure on day t uses:
    - level_t (assumed observable by decision time; e.g. aligned US vol index close mapped to CN t)
    - thresholds_{t-1} (computed without using level_t)
    """
    lvl = pd.to_numeric(levels, errors="coerce").astype(float)
    qs = [float(q) for q in (quantiles or [])]
    exps = [float(x) for x in (exposures or [])]
    if len(exps) != len(qs) + 1:
        raise ValueError("bad_tier_exposures_len (expected len(exposures)=len(quantiles)+1)")
    if any((q <= 0.0 or q >= 1.0) for q in qs):
        raise ValueError("quantiles_out_of_range (expected 0<q<1)")
    if any((x < 0.0 or x > 1.0) for x in exps):
        raise ValueError("exposures_out_of_range (expected 0<=x<=1)")

    win_days = _vol_level_window_days(level_window)
    mp = int(max(2, int(min_periods)))

    if win_days is None:
        thr_df = pd.DataFrame(
            {f"q{int(q*10000)}": lvl.expanding(min_periods=mp).quantile(q) for q in qs},
            index=lvl.index,
        ).shift(1)
        win_mode = "expanding"
    else:
        thr_df = pd.DataFrame(
            {f"q{int(q*10000)}": lvl.rolling(window=int(win_days), min_periods=mp).quantile(q) for q in qs},
            index=lvl.index,
        ).shift(1)
        win_mode = f"rolling_{int(win_days)}"

    lv = lvl.to_numpy(dtype=float)
    exp = np.ones(len(lvl), dtype=float)
    bucket = np.full(len(lvl), -1, dtype=int)
    thr_vals = thr_df.to_numpy(dtype=float)

    for i in range(len(lvl)):
        if not np.isfinite(lv[i]):
            exp[i] = 1.0
            bucket[i] = -1
            continue
        row = thr_vals[i, :]
        row = row[np.isfinite(row)]
        if row.size == 0:
            # warm-up: keep full exposure until thresholds available
            exp[i] = 1.0
            bucket[i] = 0
            continue
        row_sorted = np.sort(row)
        j = int(np.searchsorted(row_sorted, lv[i], side="left"))
        j = int(max(0, min(len(exps) - 1, j)))
        exp[i] = float(exps[j])
        bucket[i] = j

    okb = bucket >= 0
    counts = [int(np.sum(bucket[okb] == j)) for j in range(len(exps))]
    total = int(np.sum(okb))
    bucket_rates = [float(c) / float(total) if total > 0 else float("nan") for c in counts]

    meta = {
        "window": str(level_window or "all"),
        "window_mode": win_mode,
        "quantiles": qs,
        "exposures": exps,
        "min_periods": int(mp),
        "bucket_rates": bucket_rates,
        "avg_exposure": float(np.nanmean(exp)) if exp.size else float("nan"),
    }
    return pd.Series(exp, index=lvl.index, dtype=float), meta


def _bias_series_from_close(
    close: pd.DataFrame,
    *,
    ma_window: int,
    bias_type: str = "bias",
) -> pd.DataFrame:
    w = max(2, int(ma_window))
    bt = str(bias_type or "bias").strip().lower()
    ma = close.rolling(window=w, min_periods=max(2, w // 2)).mean()
    if bt == "bias_v":
        # Close-only ATR proxy consistent with the rest of strategy internals.
        atr = close.diff().abs().rolling(window=w, min_periods=max(2, w // 2)).mean()
        return ((close - ma) / atr.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).astype(float)
    return (close / ma - 1.0).replace([np.inf, -np.inf], np.nan).astype(float)


def _bias_threshold_series(
    bias_df: pd.DataFrame,
    *,
    level_window: str,
    threshold_type: str,
    quantile: float,
    fixed_value: float,
    min_periods: int,
) -> pd.DataFrame:
    tt = str(threshold_type or "quantile").strip().lower()
    if tt == "fixed":
        bt = str(getattr(bias_df, "attrs", {}).get("bias_type", "bias") or "bias").strip().lower()
        # fixed threshold semantics:
        # - bias: percentage points => convert 10 to 0.10
        # - bias_v: ATR multiples => use raw value (e.g., 1.5 means 1.5x ATR)
        v = float(fixed_value) if bt == "bias_v" else (float(fixed_value) / 100.0)
        return pd.DataFrame(v, index=bias_df.index, columns=bias_df.columns, dtype=float)
    q = float(quantile)
    if q > 1.0:
        q = q / 100.0
    q = float(min(max(q, 1e-6), 1.0 - 1e-6))
    mp = int(max(2, int(min_periods)))
    win_days = _vol_level_window_days(level_window)
    if win_days is None:
        return bias_df.expanding(min_periods=mp).quantile(q).shift(1)
    return bias_df.rolling(window=int(win_days), min_periods=mp).quantile(q).shift(1)


def _apply_asset_vol_index_rules(
    w: pd.DataFrame,
    *,
    rules: list[dict[str, Any]] | None,
    vol_index_close: dict[str, pd.Series] | None,
) -> dict[str, Any]:
    """
    Apply per-asset volatility-index timing by scaling weights daily (cash remainder).

    Expected rule fields:
      - code: ETF code
      - index: vol index code, e.g. VIX / GVZ
      - level_window: 30d/90d/180d/1y/3y/5y/10y/all (all means expanding)
      - level_quantiles: list[float] (0..1)
      - level_exposures: list[float] (len = len(quantiles)+1)
      - min_periods: int (optional)

    Caller should preload vol_index_close[index] as a pandas Series indexed by date (dt.date or Timestamp),
    already aligned to the decision calendar (typically CN next trading day).
    """
    if rules is None or len(rules) == 0 or w.empty:
        return {"enabled": False, "rules": []}
    if vol_index_close is None or len(vol_index_close) == 0:
        raise ValueError("asset_vol_index_rules set but vol_index_close missing")

    idx = w.index
    out_rules: list[dict[str, Any]] = []

    for r in rules:
        code = str(r.get("code") or "").strip()
        if not code or code not in w.columns:
            continue
        index_code = str(r.get("index") or "").strip().upper()
        if not index_code:
            continue
        level_window = str(r.get("level_window") or "all").strip().lower()
        quantiles = r.get("level_quantiles") or [0.8]
        exposures = r.get("level_exposures") or [1.0, 0.5]
        min_periods = int(r.get("min_periods") or 20)

        s_raw = None
        if index_code == "WAVOL":
            # Per-asset weekly rolling annualized volatility (computed on the asset's own close series)
            # is stored as a per-code entry.
            s_raw = vol_index_close.get(f"WAVOL:{code}")
        if s_raw is None:
            s_raw = vol_index_close.get(index_code)
        if s_raw is None or getattr(s_raw, "empty", True):
            raise ValueError(f"missing vol index close for {index_code}")
        s_raw2 = pd.to_numeric(s_raw, errors="coerce").dropna().astype(float)
        # normalize index to Timestamp for alignment with weight index
        s_ts = s_raw2.copy()
        if not isinstance(s_ts.index, pd.DatetimeIndex):
            s_ts.index = pd.to_datetime(list(s_ts.index))
        s = s_ts.reindex(idx).ffill()
        # WAVOL is derived from the asset's own prices; shift by 1 day to avoid
        # using same-day close-derived volatility for same-day exposure scaling.
        if index_code == "WAVOL":
            s = s.shift(1)

        expo, emeta = _tiered_exposure_from_level_quantiles(
            s,
            level_window=level_window,
            quantiles=list(quantiles),
            exposures=list(exposures),
            min_periods=min_periods,
        )
        w[code] = w[code].astype(float).to_numpy(dtype=float) * expo.to_numpy(dtype=float)

        out_rules.append(
            {
                "code": code,
                "index": index_code,
                **emeta,
                "dates": idx.date.astype(str).tolist(),
                "level": s.to_numpy(dtype=float).astype(float).tolist(),
                "exp": expo.to_numpy(dtype=float).astype(float).tolist(),
            }
        )

    return {"enabled": bool(out_rules), "rules": out_rules}


def _pick_assets(
    scores_row: pd.Series, *, top_k: int
) -> tuple[list[str], dict[str, Any]]:
    """
    Pick assets for the next holding period.

    Returns (picks, meta):
    - picks: list of codes to hold (equal-weight). Empty list means "cash".
    - meta: debug info (best_score, mode).

    ``top_k`` may be negative: hold the bottom |K| by score (inverse). Effective count is
    ``min(abs(top_k), len(pool))`` for dynamic pools.
    """
    s = scores_row.dropna()
    if s.empty:
        return [], {"best_score": None, "mode": "no_signal"}

    tk = int(top_k)
    if tk == 0:
        raise ValueError("top_k must be non-zero")
    k_abs = int(abs(tk))
    ascending = tk < 0
    s_ord = s.sort_values(ascending=ascending)
    eff = int(min(k_abs, int(len(s_ord))))
    picks = [str(x) for x in s_ord.index[:eff].tolist()]
    universe_best = float(s.sort_values(ascending=False).iloc[0])
    return picks, {"best_score": universe_best, "mode": "risk_on"}


def _reduce_scores_by_group(
    scores_row: pd.Series,
    *,
    group_enforce: bool,
    asset_groups: dict[str, str] | None,
    policy: str,
    current_holdings: set[str] | None = None,
    vol_row: pd.Series | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Apply cross-asset group hard constraint before Top-K:
    each group can keep at most one candidate.

    Supported policies:
    - strongest_score: highest score in each group (default)
    - earliest_entry: prefer currently held code in each group; fallback to strongest
    - lowest_vol: pick lowest realized vol in each group; fallback to strongest
    """
    meta: dict[str, Any] = {
        "enabled": bool(group_enforce),
        "policy": str(policy or "strongest_score"),
        "before": [],
        "after": [],
        "group_winners": {},
        "group_eliminated": {},
    }
    s = scores_row.dropna()
    if (not group_enforce) or s.empty:
        meta["before"] = [str(c) for c in s.sort_values(ascending=False).index.tolist()]
        meta["after"] = list(meta["before"])
        return s, meta

    policy2 = str(policy or "strongest_score").strip().lower()
    if policy2 not in {"strongest_score", "earliest_entry", "lowest_vol"}:
        raise ValueError(f"invalid group_pick_policy={policy}")

    s = s.sort_values(ascending=False)
    meta["before"] = [str(c) for c in s.index.tolist()]
    groups = {str(k): str(v) for k, v in (asset_groups or {}).items()}
    cur = set(str(x) for x in (current_holdings or set()))

    bucket: dict[str, list[str]] = {}
    for code in s.index.tolist():
        c = str(code)
        gid = str(groups.get(c) or c)
        bucket.setdefault(gid, []).append(c)

    winners: list[str] = []
    eliminated: dict[str, list[str]] = {}
    winner_map: dict[str, str] = {}

    for gid, codes in bucket.items():
        winner = codes[0]
        if policy2 == "earliest_entry":
            held = [c for c in codes if c in cur]
            if held:
                winner = held[0]
        elif policy2 == "lowest_vol" and vol_row is not None:
            vol_pairs: list[tuple[str, float]] = []
            for c in codes:
                v = vol_row.get(c) if c in vol_row.index else np.nan
                try:
                    vv = float(v)
                except (TypeError, ValueError):
                    vv = np.nan
                if np.isfinite(vv):
                    vol_pairs.append((c, vv))
            if vol_pairs:
                # Tie-break: lower vol -> higher score -> lexicographic code
                vol_pairs = sorted(vol_pairs, key=lambda x: (float(x[1]), -float(s.get(x[0], np.nan)), str(x[0])))
                winner = str(vol_pairs[0][0])

        winners.append(winner)
        winner_map[gid] = winner
        eliminated[gid] = [c for c in codes if c != winner]

    # Keep global ranking semantics after group reduction.
    reduced = s.reindex(winners).dropna().sort_values(ascending=False)
    meta["after"] = [str(c) for c in reduced.index.tolist()]
    meta["group_winners"] = winner_map
    meta["group_eliminated"] = eliminated
    return reduced, meta


def _holding_streaks_from_weights(
    w: pd.DataFrame,
    *,
    codes: list[str],
    eps: float = 1e-12,
) -> list[dict[str, Any]]:
    """
    Build continuous holding "streak" segments from daily weights.

    This is different from decision-period holdings:
    - decision periods are bounded by the rebalance schedule, even if holdings don't change
    - streaks merge consecutive days as long as the held-code set stays the same

    Returns list of {start_date, end_date, picks}.
    """
    if w.empty:
        return []
    cols = [c for c in codes if c in w.columns]
    if not cols:
        return []
    idx = pd.to_datetime(w.index)
    arr = w[cols].astype(float).to_numpy(dtype=float)
    out: list[dict[str, Any]] = []
    cur_key: tuple[str, ...] | None = None
    start_i = 0
    for i in range(len(idx)):
        picks = [str(cols[j]) for j in range(len(cols)) if float(arr[i, j]) > float(eps)]
        key = tuple(sorted(picks))
        if cur_key is None:
            cur_key = key
            start_i = i
            continue
        if key != cur_key:
            out.append(
                {
                    "start_date": idx[start_i].date().isoformat(),
                    "end_date": idx[i - 1].date().isoformat(),
                    "picks": list(cur_key),
                }
            )
            cur_key = key
            start_i = i
    if cur_key is not None:
        out.append(
            {
                "start_date": idx[start_i].date().isoformat(),
                "end_date": idx[-1].date().isoformat(),
                "picks": list(cur_key),
            }
        )
    return out


def backtest_rotation(
    db: Session,
    inp: RotationInputs,
    *,
    return_weights_end: bool = False,
    allow_virtual_end: bool = False,
    lightweight: bool = False,
) -> dict[str, Any]:
    universe = list(dict.fromkeys(inp.codes))
    if not universe:
        raise ValueError("codes is empty")
    top_k_signed = int(inp.top_k)
    if top_k_signed == 0:
        raise ValueError("top_k must be non-zero")
    top_k_abs = int(abs(top_k_signed))
    inverse_top_k = bool(top_k_signed < 0)
    pos_mode = str(inp.position_mode or "adaptive").strip().lower()
    if pos_mode not in {"adaptive", "fixed", "risk_budget"}:
        raise ValueError("position_mode must be one of: adaptive|fixed|risk_budget")
    risk_budget_atr_window = int(getattr(inp, "risk_budget_atr_window", 20) or 20)
    if risk_budget_atr_window < 2:
        raise ValueError("risk_budget_atr_window must be >= 2")
    risk_budget_pct = float(getattr(inp, "risk_budget_pct", 0.01) or 0.01)
    if (not np.isfinite(risk_budget_pct)) or risk_budget_pct < 0.001 or risk_budget_pct > 0.03:
        raise ValueError("risk_budget_pct must be in [0.001, 0.03]")
    if inp.lookback_days <= 0:
        raise ValueError("lookback_days must be > 0")
    if int(inp.entry_match_n) < 0:
        raise ValueError("entry_match_n must be >= 0")
    if int(inp.exit_match_n) < 0:
        raise ValueError("exit_match_n must be >= 0")
    if inp.skip_days < 0:
        raise ValueError("skip_days must be >= 0")
    if not np.isfinite(float(inp.cost_bps)) or float(inp.cost_bps) < 0.0:
        raise ValueError("cost_bps must be finite and >= 0")
    if not np.isfinite(float(inp.slippage_rate)) or float(inp.slippage_rate) < 0.0:
        raise ValueError("slippage_rate must be finite and >= 0")
    sm = (inp.score_method or "raw_mom").strip().lower()
    if sm not in {
        "raw_mom",
        "sharpe_mom",
        "sortino_mom",
    }:
        raise ValueError(f"invalid score_method={inp.score_method}")
    if inp.trend_sma_window <= 0:
        raise ValueError("trend_sma_window must be > 0")
    trend_ma_type = str(inp.trend_ma_type or "sma").strip().lower()
    if trend_ma_type not in {"sma", "ema", "vma"}:
        raise ValueError("trend_ma_type must be one of: sma|ema|vma")
    if int(inp.bias_ma_window) <= 1:
        raise ValueError("bias_ma_window must be > 1")
    b_type = str(inp.bias_type or "bias").strip().lower()
    if b_type not in {"bias", "bias_v"}:
        raise ValueError("bias_type must be one of: bias|bias_v")
    b_thr_type = str(inp.bias_threshold_type or "quantile").strip().lower()
    if b_thr_type not in {"quantile", "fixed"}:
        raise ValueError("bias_threshold_type must be one of: quantile|fixed")
    if not np.isfinite(float(inp.bias_quantile)) or float(inp.bias_quantile) <= 0.0 or float(inp.bias_quantile) >= 100.0:
        raise ValueError("bias_quantile must be within (0,100)")
    if not np.isfinite(float(inp.bias_fixed_value)) or float(inp.bias_fixed_value) < 0.0:
        raise ValueError("bias_fixed_value must be >= 0")
    if int(inp.bias_min_periods) < 2:
        raise ValueError("bias_min_periods must be >= 2")
    if inp.rsi_window <= 0:
        raise ValueError("rsi_window must be > 0")
    if inp.vol_window <= 0:
        raise ValueError("vol_window must be > 0")
    if inp.chop_window <= 1:
        raise ValueError("chop_window must be > 1")
    if not np.isfinite(float(inp.chop_er_threshold)):
        raise ValueError("chop_er_threshold must be finite")
    if float(inp.chop_er_threshold) <= 0:
        raise ValueError("chop_er_threshold must be > 0")
    cm = (inp.chop_mode or "er").strip().lower()
    if cm not in {"er", "adx"}:
        raise ValueError(f"invalid chop_mode={inp.chop_mode}")
    if cm == "adx":
        if inp.chop_adx_window <= 1:
            raise ValueError("chop_adx_window must be > 1")
        if not np.isfinite(float(inp.chop_adx_threshold)):
            raise ValueError("chop_adx_threshold must be finite")
        if float(inp.chop_adx_threshold) <= 0:
            raise ValueError("chop_adx_threshold must be > 0")
    atr_stop_mode = (inp.atr_stop_mode or "none").strip().lower()
    if atr_stop_mode not in {"none", "static", "trailing", "tightening"}:
        raise ValueError("atr_stop_mode must be one of: none|static|trailing|tightening")
    atr_stop_atr_basis = (inp.atr_stop_atr_basis or "latest").strip().lower()
    if atr_stop_atr_basis not in {"entry", "latest"}:
        raise ValueError("atr_stop_atr_basis must be one of: entry|latest")
    atr_stop_reentry_mode = (getattr(inp, "atr_stop_reentry_mode", "reenter") or "reenter").strip().lower()
    if atr_stop_reentry_mode not in {"reenter", "wait_next_entry"}:
        raise ValueError("atr_stop_reentry_mode must be one of: reenter|wait_next_entry")
    if int(inp.atr_stop_window) < 2:
        raise ValueError("atr_stop_window must be >= 2")
    if not np.isfinite(float(inp.atr_stop_n)) or float(inp.atr_stop_n) <= 0:
        raise ValueError("atr_stop_n must be finite and > 0")
    if not np.isfinite(float(inp.atr_stop_m)) or float(inp.atr_stop_m) <= 0:
        raise ValueError("atr_stop_m must be finite and > 0")
    if atr_stop_mode == "tightening" and float(inp.atr_stop_n) <= float(inp.atr_stop_m):
        raise ValueError("atr_stop_n must be > atr_stop_m when atr_stop_mode=tightening")
    atr_stop_exec_mode = {
        "none": "none",
        "static": "atr_stop_static",
        "trailing": "atr_stop_trailing",
        "tightening": "atr_stop_tightening",
    }[atr_stop_mode]
    if inp.corr_window is not None and int(inp.corr_window) < 2:
        raise ValueError("corr_window must be >= 2")
    if not np.isfinite(float(inp.corr_threshold)) or float(inp.corr_threshold) < -1.0 or float(inp.corr_threshold) > 1.0:
        raise ValueError("corr_threshold must be within [-1,1]")
    if int(inp.inertia_min_hold_periods) < 0:
        raise ValueError("inertia_min_hold_periods must be >= 0")
    if not np.isfinite(float(inp.inertia_score_gap)) or float(inp.inertia_score_gap) < 0.0:
        raise ValueError("inertia_score_gap must be finite and >= 0")
    if not np.isfinite(float(inp.inertia_min_turnover)) or float(inp.inertia_min_turnover) < 0.0 or float(inp.inertia_min_turnover) > 1.0:
        raise ValueError("inertia_min_turnover must be within [0,1]")
    if not np.isfinite(float(inp.rr_years)) or float(inp.rr_years) <= 0:
        raise ValueError("rr_years must be finite and > 0")
    rr_thresholds = inp.rr_thresholds if inp.rr_thresholds is not None else [0.5, 1.0, 1.5, 2.0, 2.5]
    rr_weights = inp.rr_weights if inp.rr_weights is not None else [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
    if bool(inp.rr_sizing):
        if len(rr_thresholds) > 5:
            raise ValueError("rr_thresholds must have at most 5 thresholds")
        if any((not np.isfinite(float(x))) for x in rr_thresholds):
            raise ValueError("rr_thresholds must be finite")
        if any(float(rr_thresholds[i]) >= float(rr_thresholds[i + 1]) for i in range(len(rr_thresholds) - 1)):
            raise ValueError("rr_thresholds must be strictly increasing")
        if len(rr_weights) != len(rr_thresholds) + 1:
            raise ValueError("rr_weights length must be len(rr_thresholds)+1")
        if any((not np.isfinite(float(x))) for x in rr_weights):
            raise ValueError("rr_weights must be finite")
        if any((float(x) < 0.0 or float(x) > 1.0) for x in rr_weights):
            raise ValueError("rr_weights must be within [0,1]")

    # Mirror (composite deviation) exposure cap (optional)
    mirror_enabled = bool(inp.mirror_control)
    mirror_qs = inp.mirror_quantiles if inp.mirror_quantiles is not None else [0.90, 0.95, 0.99]
    mirror_exps = inp.mirror_exposures if inp.mirror_exposures is not None else [0.80, 0.50, 0.20]
    if mirror_enabled:
        if not mirror_qs:
            raise ValueError("mirror_quantiles cannot be empty when mirror_control=true")
        if len(mirror_exps) != len(mirror_qs):
            raise ValueError("mirror_exposures length must equal mirror_quantiles length")
        if any((not np.isfinite(float(x))) for x in mirror_qs):
            raise ValueError("mirror_quantiles must be finite")
        if any((float(x) <= 0.0 or float(x) >= 1.0) for x in mirror_qs):
            raise ValueError("mirror_quantiles must be within (0,1)")
        if any(float(mirror_qs[i]) >= float(mirror_qs[i + 1]) for i in range(len(mirror_qs) - 1)):
            raise ValueError("mirror_quantiles must be strictly increasing")
        if any((not np.isfinite(float(x))) for x in mirror_exps):
            raise ValueError("mirror_exposures must be finite")
        if any((float(x) < 0.0 or float(x) > 1.0) for x in mirror_exps):
            raise ValueError("mirror_exposures must be within [0,1]")
    if not (0.0 <= float(inp.rsi_oversold) <= 100.0) or not (0.0 <= float(inp.rsi_overbought) <= 100.0):
        raise ValueError("rsi thresholds must be within [0,100]")
    if float(inp.vol_target_ann) <= 0:
        raise ValueError("vol_target_ann must be > 0")
    if float(inp.vol_max_ann) <= 0:
        raise ValueError("vol_max_ann must be > 0")
    if not np.isfinite(float(inp.dd_threshold)) or float(inp.dd_threshold) <= 0.0 or float(inp.dd_threshold) >= 1.0:
        raise ValueError("dd_threshold must be within (0,1)")
    if not np.isfinite(float(inp.dd_reduce)) or float(inp.dd_reduce) < 0.0 or float(inp.dd_reduce) > 1.0:
        raise ValueError("dd_reduce must be within [0,1]")
    if int(inp.dd_sleep_days) < 1:
        raise ValueError("dd_sleep_days must be >= 1")
    reb_shift = (inp.rebalance_shift or "prev").strip().lower()
    if reb_shift not in {"prev", "next", "skip"}:
        raise ValueError("rebalance_shift must be one of: prev|next|skip")
    ep = (inp.exec_price or "open").strip().lower()
    if ep not in {"close", "open", "oc2"}:
        raise ValueError("exec_price must be one of: close|open|oc2")
    group_policy = str(inp.group_pick_policy or "strongest_score").strip().lower()
    if group_policy not in {"strongest_score", "earliest_entry", "lowest_vol"}:
        raise ValueError(f"invalid group_pick_policy={inp.group_pick_policy}")

    codes = universe[:]
    rank_codes = universe[:]  # ranking / filters apply to the original universe only
    group_map: dict[str, str] = {}
    for c in rank_codes:
        cc = str(c)
        gid = str((inp.asset_groups or {}).get(cc) or cc).strip() or cc
        group_map[cc] = gid
    # Load:
    # - qfq: momentum score + technical analysis (trend/RSI/vol/chop filters)
    # - hfq: benchmark/corporate-action fallback calculations
    # - none: execution/trading price basis
    # Per-asset rules: if provided, override the corresponding global params.
    use_floor_rules = bool(inp.asset_momentum_floor_rules)
    use_trend_rules = bool((inp.trend_filter or inp.trend_exit_filter) and inp.asset_trend_rules)
    use_bias_rules = bool((inp.bias_filter or inp.bias_exit_filter) and inp.asset_bias_rules)
    use_rsi_rules = bool(inp.rsi_filter and inp.asset_rsi_rules)
    use_chop_rules = bool(inp.chop_filter and inp.asset_chop_rules)
    use_vol_rules = bool(inp.vol_monitor and inp.asset_vol_monitor_rules)

    trend_windows = [int(inp.trend_sma_window)]
    trend_ma_types = {str(trend_ma_type)}
    # Stage-aware defaults:
    # - entry trend: price above MA
    # - exit trend: price below MA (trend-break exit)
    trend_ops = {_normalize_cmp_op(">")}
    if bool(inp.trend_exit_filter):
        trend_ops.add(_normalize_cmp_op("<"))
    if use_trend_rules:
        for r in inp.asset_trend_rules or []:
            try:
                w = int((r or {}).get("trend_sma_window") or inp.trend_sma_window)
            except (TypeError, ValueError):
                w = int(inp.trend_sma_window)
            if w > 0:
                trend_windows.append(w)
            mt = str((r or {}).get("trend_ma_type") or "").strip().lower()
            if mt in {"sma", "ema", "vma"}:
                trend_ma_types.add(mt)
            rs = str((r or {}).get("stage") or "entry").strip().lower()
            default_op = "<" if rs == "exit" else ">"
            trend_ops.add(_normalize_cmp_op((r or {}).get("op") or default_op))

    bias_windows = [int(inp.bias_ma_window)]
    bias_rule_cfgs: set[tuple[str, int, str, str, float, float, int]] = {
        (
            str(b_type),
            int(inp.bias_ma_window),
            str(inp.bias_level_window or "all").strip().lower(),
            str(inp.bias_threshold_type or "quantile").strip().lower(),
            float(inp.bias_quantile),
            float(inp.bias_fixed_value),
            int(inp.bias_min_periods),
        )
    }
    if use_bias_rules:
        for r in inp.asset_bias_rules or []:
            try:
                w = int((r or {}).get("bias_ma_window") or inp.bias_ma_window)
            except (TypeError, ValueError):
                w = int(inp.bias_ma_window)
            if w > 1:
                bias_windows.append(w)
            bt_rule = str((r or {}).get("bias_type") or b_type).strip().lower()
            if bt_rule not in {"bias", "bias_v"}:
                bt_rule = str(b_type)
            cfg = (
                bt_rule,
                int(max(2, w)),
                str((r or {}).get("level_window") or inp.bias_level_window or "all").strip().lower(),
                str((r or {}).get("threshold_type") or inp.bias_threshold_type or "quantile").strip().lower(),
                float((r or {}).get("quantile") if (r or {}).get("quantile") is not None else inp.bias_quantile),
                float((r or {}).get("fixed_value") if (r or {}).get("fixed_value") is not None else inp.bias_fixed_value),
                int((r or {}).get("min_periods") if (r or {}).get("min_periods") is not None else inp.bias_min_periods),
            )
            bias_rule_cfgs.add(cfg)

    rsi_windows = [int(inp.rsi_window)]
    if use_rsi_rules:
        for r in inp.asset_rsi_rules or []:
            try:
                w = int((r or {}).get("rsi_window") or inp.rsi_window)
            except (TypeError, ValueError):
                w = int(inp.rsi_window)
            if w > 0:
                rsi_windows.append(w)

    vol_windows = [int(inp.vol_window)]
    if use_vol_rules:
        for r in inp.asset_vol_monitor_rules or []:
            try:
                w = int((r or {}).get("vol_window") or inp.vol_window)
            except (TypeError, ValueError):
                w = int(inp.vol_window)
            if w > 0:
                vol_windows.append(w)

    chop_modes = {str(cm)}
    chop_er_windows = [int(inp.chop_window)]
    chop_adx_windows = [int(inp.chop_adx_window)]
    if use_chop_rules:
        for r in inp.asset_chop_rules or []:
            m = str((r or {}).get("chop_mode") or cm).strip().lower()
            if m:
                chop_modes.add(m)
            if m == "adx":
                try:
                    w = int((r or {}).get("chop_adx_window") or inp.chop_adx_window)
                except (TypeError, ValueError):
                    w = int(inp.chop_adx_window)
                if w > 1:
                    chop_adx_windows.append(w)
            else:
                try:
                    w = int((r or {}).get("chop_window") or inp.chop_window)
                except (TypeError, ValueError):
                    w = int(inp.chop_window)
                if w > 1:
                    chop_er_windows.append(w)

    # Need enough history for momentum + optional risk controls (using max window).
    need_hist = inp.lookback_days + inp.skip_days + 60
    if inp.trend_filter:
        need_hist = max(need_hist, int(max(trend_windows)) + 60)
    if inp.bias_filter or inp.bias_exit_filter:
        need_hist = max(need_hist, int(max(bias_windows)) + 60)
    if inp.rsi_filter:
        need_hist = max(need_hist, int(max(rsi_windows)) + 60)
    if inp.vol_monitor:
        need_hist = max(need_hist, int(max(vol_windows)) + 60)
    if inp.chop_filter:
        if "adx" in chop_modes:
            need_hist = max(need_hist, int(max(chop_adx_windows)) + 60)
        else:
            need_hist = max(need_hist, int(max(chop_er_windows)) + 60)
    ext_start = inp.start - dt.timedelta(days=int(need_hist))
    close_hfq = _load_close_prices(db, codes=codes, start=ext_start, end=inp.end, adjust="hfq")
    # qfq is the unified decision basis (signal/filters/risk controls), so it is always required.
    need_qfq = True
    close_qfq = _load_close_prices(db, codes=codes, start=ext_start, end=inp.end, adjust="qfq") if need_qfq else pd.DataFrame()
    need_qfq_hl = bool(
        (inp.chop_filter and (("adx" in chop_modes) if use_chop_rules else (cm == "adx")))
        or (pos_mode == "risk_budget")
    )
    high_qfq, low_qfq = (
        _load_high_low_prices(db, codes=codes, start=ext_start, end=inp.end, adjust="qfq") if need_qfq_hl else (pd.DataFrame(), pd.DataFrame())
    )
    close_none = _load_close_prices(db, codes=codes, start=inp.start, end=inp.end, adjust="none")
    if close_none.empty:
        raise ValueError("no execution price data for given range (none)")

    # Mirror control signal (optional, universe-level):
    # - build 3 deviation series (log return dev, log vol dev, log volume dev) per asset
    # - cross-sectional aggregate per day (median across universe)
    # - for each deviation, compute expanding percentile rank
    # - composite score = mean of the 3 percentiles
    # - composite percentile = expanding percentile rank of composite score
    mirror_pct: pd.Series | None = None
    if mirror_enabled:
        try:
            vol_df, amt_df = _load_volume_amount(db, codes=rank_codes, start=ext_start, end=inp.end, adjust="none")
        except (TypeError, ValueError, RuntimeError):  # pragma: no cover (defensive)
            vol_df, amt_df = pd.DataFrame(), pd.DataFrame()

        act = pd.DataFrame(index=close_qfq.index, columns=rank_codes, dtype=float)
        if vol_df is not None and (not vol_df.empty):
            act = act.combine_first(vol_df.reindex(index=act.index, columns=rank_codes))
        if amt_df is not None and (not amt_df.empty):
            # fill remaining gaps with amount as fallback proxy
            act = act.combine_first(amt_df.reindex(index=act.index, columns=rank_codes))

        # Per-asset log-return dev (MA20 of log returns)
        ret = close_qfq[rank_codes].pct_change().replace([np.inf, -np.inf], np.nan)
        lr = np.log1p(ret).replace([np.inf, -np.inf], np.nan)
        lr_ma = lr.rolling(window=20, min_periods=5).mean()
        lr_dev = (lr - lr_ma).replace([np.inf, -np.inf], np.nan)

        # Per-asset log-vol dev (MA20 of log realized vol)
        std = ret.rolling(window=20, min_periods=10).std(ddof=1) * np.sqrt(252.0)
        lv = np.log(std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        lv_ma = lv.rolling(window=20, min_periods=5).mean()
        lv_dev = (lv - lv_ma).replace([np.inf, -np.inf], np.nan)

        # Per-asset log-volume dev (MA20 of log activity)
        la = np.log(act.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        la_ma = la.rolling(window=20, min_periods=5).mean()
        la_dev = (la - la_ma).replace([np.inf, -np.inf], np.nan)

        # Aggregate to a single scalar per day (median across universe)
        s_lr = lr_dev.median(axis=1, skipna=True)
        s_lv = lv_dev.median(axis=1, skipna=True)
        s_la = la_dev.median(axis=1, skipna=True)

        def _expanding_pct_rank(s: pd.Series) -> pd.Series:
            """
            Expanding percentile rank without lookahead.
            For ties, use mid-rank: (less + 0.5*equal) / n.
            """
            from bisect import bisect_left, bisect_right, insort

            out = np.full(len(s), np.nan, dtype=float)
            hist: list[float] = []
            for i, v in enumerate(s.to_numpy(dtype=float, copy=False)):
                if not np.isfinite(v):
                    continue
                # insert then compute rank among history inclusive
                insort(hist, float(v))
                n = len(hist)
                lo = bisect_left(hist, float(v))
                hi = bisect_right(hist, float(v))
                out[i] = (float(lo) + 0.5 * float(hi - lo)) / float(n)
            return pd.Series(out, index=s.index, dtype=float)

        p_lr = _expanding_pct_rank(s_lr)
        p_lv = _expanding_pct_rank(s_lv)
        p_la = _expanding_pct_rank(s_la)
        comp = (p_lr + p_lv + p_la) / 3.0
        comp = comp.replace([np.inf, -np.inf], np.nan).dropna()
        mirror_pct = _expanding_pct_rank(comp).reindex(close_qfq.index)

    # Execution return basis:
    # - Strategy NAV uses NONE prices (tradeable) by default, with HFQ fallback on corporate-action cliff days.
    # - For plotting/benchmark comparisons we still compute HFQ series.
    # Return decomposition needs open/close legs for all execution modes.
    need_hfq_ohlc = True
    need_none_ohlc = True
    ohlc_hfq = (
        _load_ohlc_prices(db, codes=codes, start=ext_start, end=inp.end, adjust="hfq") if need_hfq_ohlc else {"open": pd.DataFrame(), "high": pd.DataFrame(), "low": pd.DataFrame(), "close": pd.DataFrame()}
    )
    ohlc_none = (
        _load_ohlc_prices(db, codes=codes, start=inp.start, end=inp.end, adjust="none") if need_none_ohlc else {"open": pd.DataFrame(), "high": pd.DataFrame(), "low": pd.DataFrame(), "close": pd.DataFrame()}
    )

    # Align calendars using execution dates; forward-fill hfq onto those dates.
    close_none = close_none.sort_index().ffill()
    if bool(allow_virtual_end) and (not close_none.empty):
        try:
            # For "next-plan" style use-cases: we may want to compute the next execution day's weights
            # even if the DB does not yet have prices for that future trading day. In that case,
            # we extend the execution calendar to `inp.end` using forward-filled last known prices.
            last_exec = close_none.index[-1].date()
            if last_exec < inp.end:
                from ..calendar.trading_calendar import trading_days as _trading_days

                extra = _trading_days(last_exec, inp.end, cal="XSHG")
                extra = [d for d in extra if d > last_exec]
                if extra:
                    extra_idx = pd.to_datetime(extra)
                    new_idx = close_none.index.union(extra_idx)
                    close_none = close_none.reindex(new_idx).ffill()
        except (TypeError, ValueError, IndexError):  # pragma: no cover (defensive)
            pass

    dates = close_none.index
    dynamic_u = bool(getattr(inp, "dynamic_universe", False))
    close_hfq_raw_align = (
        close_hfq.sort_index().reindex(dates).reindex(columns=list(dict.fromkeys(codes))).astype(float)
    )
    close_hfq = close_hfq_raw_align.ffill()
    if need_hfq_ohlc:
        for k in ["open", "high", "low", "close"]:
            ohlc_hfq[k] = ohlc_hfq[k].sort_index().reindex(dates).ffill()
    if need_none_ohlc:
        for k in ["open", "high", "low", "close"]:
            ohlc_none[k] = ohlc_none[k].sort_index().reindex(dates).ffill()
    if need_qfq and not close_qfq.empty:
        close_qfq = close_qfq.sort_index().reindex(dates).ffill()
    if need_qfq_hl:
        high_qfq = high_qfq.sort_index().reindex(dates).ffill()
        low_qfq = low_qfq.sort_index().reindex(dates).ffill()

    # Require each selected code has data (legacy mode only).
    miss_exec = [c for c in codes if c not in close_none.columns or close_none[c].dropna().empty]
    if miss_exec and (not bool(getattr(inp, "dynamic_universe", False))):
        raise ValueError(f"missing execution data (none) for: {miss_exec}")
    miss_sig = [c for c in codes if c not in close_qfq.columns or close_qfq[c].dropna().empty]
    if miss_sig and (not bool(getattr(inp, "dynamic_universe", False))):
        raise ValueError(f"missing signal data (qfq) for: {miss_sig}")
    # qfq is required for both signal and technical-analysis features
    if need_qfq:
        miss_ta = [c for c in codes if c not in close_qfq.columns or close_qfq[c].dropna().empty]
        if miss_ta and (not bool(getattr(inp, "dynamic_universe", False))):
            raise ValueError(f"missing technical-analysis data (qfq) for: {miss_ta}")
    if need_qfq_hl:
        miss_hi = [c for c in codes if c not in high_qfq.columns or high_qfq[c].dropna().empty]
        miss_lo = [c for c in codes if c not in low_qfq.columns or low_qfq[c].dropna().empty]
        miss_hl = sorted(set(miss_hi + miss_lo))
        if miss_hl and (not bool(getattr(inp, "dynamic_universe", False))):
            raise ValueError(f"missing technical-analysis high/low data (qfq) for: {miss_hl}")

    if sm == "raw_mom":
        scores = _momentum_scores(close_qfq[rank_codes], lookback_days=inp.lookback_days, skip_days=inp.skip_days)
    else:
        base_scores = _risk_adjusted_scores(
            close_qfq[rank_codes],
            lookback_days=inp.lookback_days,
            skip_days=inp.skip_days,
            method=inp.score_method,
            rf_annual=float(inp.risk_free_rate),
        )
        scores = base_scores

    # Pre-compute risk-control signals on qfq close (aligned to execution calendar).
    ta_close = close_qfq[rank_codes] if need_qfq else None
    trend_ok_each_by_key: dict[tuple[int, str, str], pd.DataFrame] = {}
    bias_by_key: dict[tuple[int, str], pd.DataFrame] = {}
    bias_thr_by_cfg: dict[tuple[str, int, str, str, float, float, int], pd.DataFrame] = {}
    rsi_by_window: dict[int, pd.DataFrame] = {}
    ann_vol_by_window: dict[int, pd.DataFrame] = {}
    er_by_window: dict[int, pd.DataFrame] = {}
    adx_by_window: dict[int, pd.DataFrame] = {}

    if ta_close is not None:
        if inp.trend_filter or inp.trend_exit_filter:
            for w in sorted(set(int(x) for x in trend_windows if int(x) > 0)):
                for mt in sorted(set(str(x) for x in trend_ma_types if str(x) in {"sma", "ema", "vma"})):
                    for op in sorted(set(str(x) for x in trend_ops)):
                        trend_ok_each_by_key[(int(w), str(mt), str(op))] = _trend_ok_each(
                            ta_close,
                            ma_window=int(w),
                            ma_type=str(mt),
                            op=str(op),
                        )
        if inp.bias_filter or inp.bias_exit_filter:
            all_bias_types = {str(b_type)}
            if use_bias_rules:
                for r in inp.asset_bias_rules or []:
                    all_bias_types.add(str((r or {}).get("bias_type") or b_type).strip().lower())
            for w in sorted(set(int(x) for x in bias_windows if int(x) > 1)):
                for bt in sorted(x for x in all_bias_types if x in {"bias", "bias_v"}):
                    bdf = _bias_series_from_close(ta_close, ma_window=int(w), bias_type=str(bt))
                    bdf.attrs["bias_type"] = str(bt)
                    bias_by_key[(int(w), str(bt))] = bdf
            for cfg in sorted(list(bias_rule_cfgs)):
                bt, w, lvw, tt, qv, fvv, mp = cfg
                bdf = bias_by_key.get((int(w), str(bt)))
                if bdf is None:
                    continue
                bias_thr_by_cfg[cfg] = _bias_threshold_series(
                    bdf,
                    level_window=str(lvw),
                    threshold_type=str(tt),
                    quantile=float(qv),
                    fixed_value=float(fvv),
                    min_periods=int(mp),
                )
        if inp.rsi_filter:
            for w in sorted(set(int(x) for x in rsi_windows if int(x) > 0)):
                rsi_by_window[w] = _rsi(ta_close, window=int(w))
        if inp.vol_monitor:
            for w in sorted(set(int(x) for x in vol_windows if int(x) > 0)):
                ann_vol_by_window[w] = _ann_realized_vol_from_close(ta_close, window=int(w))
        if inp.chop_filter:
            if "er" in chop_modes:
                for w in sorted(set(int(x) for x in chop_er_windows if int(x) > 1)):
                    er_by_window[w] = _efficiency_ratio(ta_close, window=int(w))
            if "adx" in chop_modes:
                for w in sorted(set(int(x) for x in chop_adx_windows if int(x) > 1)):
                    adx_by_window[w] = _adx(high_qfq[rank_codes], low_qfq[rank_codes], ta_close, window=int(w))

    atr_budget = pd.DataFrame()
    if pos_mode == "risk_budget":
        atr_budget = _atr_from_hlc(
            high_qfq[rank_codes].astype(float),
            low_qfq[rank_codes].astype(float),
            close_qfq[rank_codes].astype(float),
            window=int(risk_budget_atr_window),
        )

    decision_hit_mode: dict[int, str] = {}
    decision_target_date: dict[int, str] = {}

    def _decision_indices_for_rebalance(*, rebalance: str, anchor: int | None) -> list[int]:
        """
        Decision dates are where we compute picks; holdings apply from the next trading day.

        anchor semantics:
        - weekly: 1=Mon..5=Fri
        - monthly: day-of-month 1..28
        - quarterly: day-of-quarter 1..90
        - yearly: day-of-year 1..365
        - if anchor is None: keep legacy behavior (period-end)
        """
        r = (rebalance or "monthly").lower()
        if r not in {"daily", "weekly", "monthly", "quarterly", "yearly"}:
            raise ValueError(f"invalid rebalance={rebalance}")
        if r == "daily":
            return list(range(len(dates)))

        if r == "weekly":
            if anchor is None:
                labels_local = _rebalance_labels(dates, r, weekly_anchor="FRI")
                out = pd.Series(np.arange(len(dates)), index=dates).groupby(labels_local).max().to_list()
                for i_local in out:
                    decision_hit_mode[int(i_local)] = "period_end"
                    decision_target_date[int(i_local)] = dates[int(i_local)].date().isoformat()
                return out
            else:
                # 调仓日=决策日；1=Mon..5=Fri（周度仅接受 1-5；日历效应前端传 0-4 时由调用方转为 1-5）
                wd_map_local = {1: "MON", 2: "TUE", 3: "WED", 4: "THU", 5: "FRI"}
                wd = int(anchor)
                if wd not in wd_map_local:
                    raise ValueError("weekly rebalance_anchor must be within [1..5] (Mon..Fri)")
                labels_local = _rebalance_labels(dates, r, weekly_anchor=wd_map_local[wd])
            out: list[int] = []
            seen: set[int] = set()
            for p in pd.unique(labels_local):
                target = pd.Timestamp(p.end_time).normalize()
                if reb_shift == "skip" and (target not in dates):
                    continue
                i = _shift_idx_by_rebalance(target, dates, ("prev" if reb_shift == "skip" else reb_shift))
                if i not in seen:
                    out.append(i)
                    seen.add(i)
                    if target in dates:
                        hm = "exact"
                    else:
                        hm = "prev" if reb_shift == "prev" else "next"
                    decision_hit_mode[int(i)] = hm
                    decision_target_date[int(i)] = pd.to_datetime(target).date().isoformat()
            # Ensure chronological order; pd.unique(periods) order is not guaranteed.
            return sorted(out)

        if r == "monthly":
            if anchor is None:
                labels_local = _rebalance_labels(dates, r, weekly_anchor="FRI")
                out = pd.Series(np.arange(len(dates)), index=dates).groupby(labels_local).max().to_list()
                for i_local in out:
                    decision_hit_mode[int(i_local)] = "period_end"
                    decision_target_date[int(i_local)] = dates[int(i_local)].date().isoformat()
                return out
            dom = int(anchor)
            if dom < 1 or dom > 28:
                raise ValueError("monthly rebalance_anchor must be within [1..28] (day-of-month)")
            labels_local = dates.to_period("M")

            # _shift_idx moved to module-level helper _shift_idx_by_rebalance

            out: list[int] = []
            seen: set[int] = set()
            for p in pd.unique(labels_local):
                target = pd.Timestamp(dt.date(int(p.year), int(p.month), dom))
                if reb_shift == "skip" and (target not in dates):
                    continue
                i = _shift_idx_by_rebalance(target, dates, ("prev" if reb_shift == "skip" else reb_shift))
                if i not in seen:
                    out.append(i)
                    seen.add(i)
                    if target in dates:
                        hm = "exact"
                    else:
                        hm = "prev" if reb_shift == "prev" else "next"
                    decision_hit_mode[int(i)] = hm
                    decision_target_date[int(i)] = pd.to_datetime(target).date().isoformat()
            return sorted(out)

        # quarterly/yearly use calendar day-of-period anchors.
        if anchor is None:
            labels_local = _rebalance_labels(dates, r, weekly_anchor="FRI")
            out = pd.Series(np.arange(len(dates)), index=dates).groupby(labels_local).max().to_list()
            for i_local in out:
                decision_hit_mode[int(i_local)] = "period_end"
                decision_target_date[int(i_local)] = dates[int(i_local)].date().isoformat()
            return out
        n = int(anchor)
        if r == "quarterly":
            if n < 1 or n > 90:
                raise ValueError("quarterly rebalance_anchor must be within [1..90] (day-of-quarter)")
            labels_local = dates.to_period("Q")
            out: list[int] = []
            seen: set[int] = set()
            for p in pd.unique(labels_local):
                q_start = pd.Timestamp(p.start_time).normalize()
                target = q_start + pd.Timedelta(days=int(n) - 1)
                if reb_shift == "skip" and (target not in dates):
                    continue
                i = _shift_idx_by_rebalance(target, dates, ("prev" if reb_shift == "skip" else reb_shift))
                if i not in seen:
                    out.append(i)
                    seen.add(i)
                    if target in dates:
                        hm = "exact"
                    else:
                        hm = "prev" if reb_shift == "prev" else "next"
                    decision_hit_mode[int(i)] = hm
                    decision_target_date[int(i)] = pd.to_datetime(target).date().isoformat()
            return sorted(out)
        if n < 1 or n > 365:
            raise ValueError("yearly rebalance_anchor must be within [1..365] (day-of-year)")
        labels_local = dates.to_period("Y")
        out: list[int] = []
        seen: set[int] = set()
        for p in pd.unique(labels_local):
            y_start = pd.Timestamp(dt.date(int(p.year), 1, 1))
            target = y_start + pd.Timedelta(days=int(n) - 1)
            if reb_shift == "skip" and (target not in dates):
                continue
            i = _shift_idx_by_rebalance(target, dates, ("prev" if reb_shift == "skip" else reb_shift))
            if i not in seen:
                out.append(i)
                seen.add(i)
                if target in dates:
                    hm = "exact"
                else:
                    hm = "prev" if reb_shift == "prev" else "next"
                decision_hit_mode[int(i)] = hm
                decision_target_date[int(i)] = pd.to_datetime(target).date().isoformat()
        return sorted(out)

    # Determine rebalance decision dates.
    # If we rebalance at close on decision_date, then returns on the NEXT trading day onward
    # should reflect the new holdings. Therefore the holdings from one decision apply through
    # the NEXT decision date (inclusive), to avoid "gaps" on decision dates.
    anchor_val = inp.rebalance_anchor
    last_idx = _decision_indices_for_rebalance(rebalance=inp.rebalance, anchor=anchor_val)
    decision_dates = dates[last_idx]

    # Period labels are used by rebalance-period aggregation features.
    # Keep behavior backward-compatible:
    # - weekly: use the same weekly anchor as the decision schedule (default FRI)
    # - monthly/quarterly/yearly: natural calendar periods
    if (inp.rebalance or "weekly").lower() == "weekly":
        wd_map = {1: "MON", 2: "TUE", 3: "WED", 4: "THU", 5: "FRI"}
        w_anchor = wd_map.get(int(anchor_val), "FRI") if anchor_val is not None else "FRI"
        labels = _rebalance_labels(dates, inp.rebalance, weekly_anchor=w_anchor)
    else:
        labels = _rebalance_labels(dates, inp.rebalance, weekly_anchor="FRI")

    # Precompute execution returns for strategy NAV:
    # - prefer NONE (tradeable) prices
    # - if corporate-action cliff detected, use HFQ return on that day to avoid artificial NAV jump
    def _has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
        return (df is not None) and (not df.empty) and all((c in df.columns) for c in cols)

    # Build forward execution returns (t -> t+1), so execution-day weights never
    # consume pre-trade returns from (t-1 -> t).
    o_hfq = ohlc_hfq.get("open", pd.DataFrame())
    c_hfq = ohlc_hfq.get("close", pd.DataFrame())
    o_none = ohlc_none.get("open", pd.DataFrame())
    c_none = ohlc_none.get("close", pd.DataFrame())
    if _has_cols(o_hfq, codes):
        o_hfq = o_hfq[codes].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    else:
        o_hfq = close_hfq[codes].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    if _has_cols(c_hfq, codes):
        c_hfq = c_hfq[codes].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    else:
        c_hfq = close_hfq[codes].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    if _has_cols(o_none, codes):
        o_none = o_none[codes].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    else:
        o_none = close_none[codes].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    if _has_cols(c_none, codes):
        c_none = c_none[codes].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    else:
        c_none = close_none[codes].astype(float).replace([np.inf, -np.inf], np.nan).ffill()

    if ep == "open":
        ret_exec_none = forward_returns(o_none)
        ret_exec_hfq = forward_returns(o_hfq)
    elif ep == "close":
        # 成交价=收盘价时，使用执行日（调仓日后一交易日）的收盘价，非调仓日收盘价：
        # ret[t]=close[t+1]/close[t]-1，权重从执行日 t 起生效，故入场价为 close[t]。
        ret_exec_none = forward_returns(c_none)
        ret_exec_hfq = forward_returns(c_hfq)
    else:
        # OC2 means half execution at open and half at close.
        ret_exec_none = 0.5 * (forward_returns(o_none) + forward_returns(c_none))
        ret_exec_hfq = 0.5 * (forward_returns(o_hfq) + forward_returns(c_hfq))

    # Corporate-action cliff detection must align with execution-return horizon.
    # forward_returns is indexed by t with return over [t -> t+1], so corp_factor
    # needs the same forward horizon; otherwise fallback is applied one day late.
    ret_none_close = close_none[codes].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    ret_hfq_close = close_hfq[codes].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    gross_none_fwd_close = close_none[codes].shift(-1).div(close_none[codes]).replace([np.inf, -np.inf], np.nan)
    gross_hfq_fwd_close = close_hfq[codes].shift(-1).div(close_hfq[codes]).replace([np.inf, -np.inf], np.nan)
    corp_factor, corp_mask = corporate_action_mask(gross_none_fwd_close, gross_hfq_fwd_close)

    # Final execution returns for NAV: none preferred, hfq fallback on cliff days.
    ret_exec_all = ret_exec_none.copy()
    for c in codes:
        if c in ret_exec_all.columns and c in ret_exec_hfq.columns and c in corp_mask.columns:
            m = corp_mask[c].fillna(False)
            if bool(m.any()):
                ret_exec_all.loc[m, c] = ret_exec_hfq.loc[m, c]
    ret_exec_all = ret_exec_all.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    # 调仓日=决策日，执行日=下一交易日；收益从执行日开始计算。
    # 开盘价可享受执行当日收益；收盘价不享受执行当日收益（保持 forward return）。
    exec_day_indices = [i + 1 for i in last_idx if i + 1 < len(dates)]
    if ep == "open" and exec_day_indices:
        # Same-day return (open->close) on execution days; align to dates
        _cn = c_none.reindex(dates).ffill()
        _on = o_none.reindex(dates).ffill()
        same_day_none = (_cn[codes] / _on[codes] - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        _ch = c_hfq.reindex(dates).ffill()
        _oh = o_hfq.reindex(dates).ffill()
        same_day_hfq = (_ch[codes] / _oh[codes] - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        for j in exec_day_indices:
            for c in codes:
                if c not in ret_exec_all.columns:
                    continue
                if c in corp_mask.columns and corp_mask[c].iloc[j]:
                    ret_exec_all.loc[ret_exec_all.index[j], c] = float(same_day_hfq.loc[ret_exec_all.index[j], c])
                else:
                    ret_exec_all.loc[ret_exec_all.index[j], c] = float(same_day_none.loc[ret_exec_all.index[j], c])

    # Build weights per date (apply from next trading day after decision date).
    w = pd.DataFrame(0.0, index=dates, columns=codes)
    holdings: dict[str, list[dict[str, Any]]] = {"periods": []}
    daily_exit_events: list[dict[str, Any]] = []
    # Stop-loss carry: track prior picks and whether a stop-out occurred in the prior holding segment.
    prev_picks_key: tuple[str, ...] | None = None
    prev_segment_stopped_out: bool = False
    # ATR from qfq close only (close-to-close absolute range); aligned to execution calendar.
    # Note: classic ATR uses high/low/prev close; this is a close-only approximation per spec.
    atr_style_mode = atr_stop_exec_mode in {"atr_stop_static", "atr_stop_trailing", "atr_stop_tightening"}
    if atr_style_mode:
        w_atr = int(inp.atr_stop_window)
        w_atr = max(2, w_atr)
        close_for_atr = close_qfq[rank_codes].astype(float)
        atr = close_for_atr.diff().abs().rolling(window=w_atr, min_periods=max(2, w_atr // 2)).mean()
    else:
        w_atr = None
        atr = pd.DataFrame()

    # Correlation filter params (qfq).
    corr_enabled = bool(inp.corr_filter)
    corr_window = int(inp.corr_window) if inp.corr_window is not None else int(inp.lookback_days)
    corr_window = max(2, corr_window)
    corr_threshold = float(inp.corr_threshold)

    inertia_enabled = bool(inp.inertia)
    inertia_min_hold = int(max(0, int(inp.inertia_min_hold_periods)))
    inertia_score_gap = float(inp.inertia_score_gap)
    inertia_min_turnover = float(inp.inertia_min_turnover)
    last_change_decision_i = -10**9  # decision index when holdings last changed (for min-hold)

    def _pair_corr_qfq(*, code_a: str, code_b: str, end_pos: int) -> float | None:
        """
        Pearson corr of daily returns (pct_change) for qfq close over a lookback window ending at end_pos.
        end_pos is an integer index into `dates` (aligned calendar).
        """
        if code_a == code_b:
            return 1.0
        if code_a not in close_qfq.columns or code_b not in close_qfq.columns:
            return None
        start_pos = max(0, int(end_pos) - int(corr_window))
        # Need at least 3 return observations => at least 4 prices.
        if end_pos - start_pos < 3:
            return None
        pa = close_qfq[code_a].iloc[start_pos : end_pos + 1].astype(float)
        pb = close_qfq[code_b].iloc[start_pos : end_pos + 1].astype(float)
        ra = pa.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        rb = pb.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        idx = ra.index.intersection(rb.index)
        if len(idx) < 3:
            return None
        xa = ra.loc[idx].to_numpy(dtype=float)
        xb = rb.loc[idx].to_numpy(dtype=float)
        if not (np.isfinite(xa).all() and np.isfinite(xb).all()):
            return None
        if float(np.std(xa, ddof=1)) == 0.0 or float(np.std(xb, ddof=1)) == 0.0:
            return None
        return float(np.corrcoef(xa, xb)[0, 1])

    # Rolling-return sizing (strategy trailing net return).
    rr_enabled = bool(inp.rr_sizing)
    rr_window_days = max(2, int(round(float(inp.rr_years) * 252.0)))
    nav_running = pd.Series(np.ones(len(dates), dtype=float), index=dates, dtype=float)
    processed_idx = 0

    def _advance_nav_to(idx: int) -> None:
        nonlocal processed_idx
        idx = int(idx)
        if idx <= processed_idx:
            return
        rng = np.arange(processed_idx + 1, idx + 1, dtype=int)
        w_slice = w.iloc[rng].astype(float)
        r_slice = ret_exec_all.iloc[rng].astype(float)
        port_ret = (w_slice * r_slice).sum(axis=1).astype(float)
        # Turnover must be computed by position, not by index alignment.
        w_np = w_slice.to_numpy(dtype=float)
        w_prev_np = w.iloc[rng - 1].astype(float).to_numpy(dtype=float)
        turnover_np = np.abs(w_np - w_prev_np).sum(axis=1) / 2.0
        cost_np = turnover_np * (float(inp.cost_bps) / 10000.0)
        slippage_np = turnover_np * float(inp.slippage_rate)
        nav = float(nav_running.iloc[processed_idx])
        # Use raw arrays to avoid any index-alignment surprises.
        xnet = port_ret.to_numpy(dtype=float) - cost_np.astype(float) - slippage_np.astype(float)
        out = np.empty(len(xnet), dtype=float)
        for j, x in enumerate(xnet):
            nav *= (1.0 + float(x))
            out[j] = nav
        nav_running.iloc[rng] = out
        processed_idx = idx

    def _rr_bucket_exposure(trailing_return: float) -> tuple[int, float]:
        for i, thr in enumerate(rr_thresholds):
            if trailing_return < float(thr):
                return i, float(rr_weights[i])
        return len(rr_thresholds), float(rr_weights[-1])

    # Drawdown control (strategy NAV drawdown, net of turnover cost and slippage).
    dd_enabled = bool(inp.dd_control)
    dd_threshold = float(inp.dd_threshold)
    dd_reduce = float(inp.dd_reduce)
    dd_scale = float(max(0.0, 1.0 - dd_reduce))
    dd_sleep_days = int(max(1, int(inp.dd_sleep_days)))
    dd_sleep_until_idx = -1  # last trading-day index that remains in "sleep"
    dd_nav = 1.0
    dd_peak = 1.0
    dd_prev_w = np.zeros(len(codes), dtype=float)
    dd_processed_idx = 0
    dd_prev_drawdown = 0.0

    def _apply_dd_control_segment(*, seg_start_i: int, seg_end_i: int) -> dict[str, Any]:
        """
        Advance running NAV across a segment and apply drawdown control if triggered.
        Trigger rule:
        - compute drawdown from running peak NAV: dd = 1 - nav/peak
        - if dd >= threshold and not currently sleeping -> scale positions by (1-reduce) from next trading day
          and enter sleep for dd_sleep_days trading days (starting next day).
        """
        nonlocal dd_sleep_until_idx, dd_nav, dd_peak, dd_prev_w, dd_processed_idx, dd_prev_drawdown
        seg_start_i = int(seg_start_i)
        seg_end_i = int(seg_end_i)
        meta: dict[str, Any] = {
            "enabled": bool(dd_enabled),
            "threshold": float(dd_threshold),
            "reduce": float(dd_reduce),
            "sleep_days": int(dd_sleep_days),
            "sleep_until": (dates[dd_sleep_until_idx].date().isoformat() if dd_sleep_until_idx >= 0 else None),
            "triggered": False,
            "trigger_date": None,
            "trigger_drawdown": None,
            "trigger_nav": None,
            "trigger_peak": None,
        }
        if not dd_enabled:
            # Disabled: keep payload shape stable but avoid extra simulation cost.
            meta.update({"in_sleep": False, "nav_asof": None, "peak_asof": None, "drawdown_asof": None})
            return meta

        start = max(int(dd_processed_idx) + 1, seg_start_i)
        if start > seg_end_i:
            # Nothing to do; still provide as-of stats for decision day visibility.
            dd_now = 0.0 if dd_peak <= 0 else float(1.0 - dd_nav / dd_peak)
            meta.update(
                {
                    "nav_asof": float(dd_nav),
                    "peak_asof": float(dd_peak),
                    "drawdown_asof": float(dd_now),
                    "in_sleep": bool(dd_enabled and (seg_start_i <= dd_sleep_until_idx)),
                }
            )
            return meta

        # Simulate day-by-day; when triggered, scale future weights inside this segment.
        for t in range(start, seg_end_i + 1):
            w_row = w.iloc[t].to_numpy(dtype=float)
            r_row = ret_exec_all.iloc[t].to_numpy(dtype=float)
            port_ret = float(np.dot(w_row, r_row))
            turnover = float(np.abs(w_row - dd_prev_w).sum() / 2.0)
            cost = float(turnover * (float(inp.cost_bps) / 10000.0))
            slip = float(turnover * float(inp.slippage_rate))
            dd_nav *= (1.0 + float(port_ret) - float(cost) - float(slip))
            dd_peak = float(max(float(dd_peak), float(dd_nav)))
            dd_prev_w = w_row
            dd_processed_idx = int(t)

            # Trigger check on end-of-day NAV; apply from next trading day.
            if not dd_enabled:
                continue
            if int(t) <= int(dd_sleep_until_idx):
                continue  # already sleeping
            if float(dd_peak) <= 0:
                continue
            dd_now = float(1.0 - float(dd_nav) / float(dd_peak))
            # IMPORTANT: to avoid re-triggering every time sleep ends while dd stays above threshold,
            # only trigger on a crossing from below -> above.
            crossed = (float(dd_prev_drawdown) < float(dd_threshold)) and (float(dd_now) >= float(dd_threshold))
            if crossed:
                meta["triggered"] = True
                meta["trigger_date"] = dates[int(t)].date().isoformat()
                meta["trigger_drawdown"] = float(dd_now)
                meta["trigger_nav"] = float(dd_nav)
                meta["trigger_peak"] = float(dd_peak)
                # enter sleep starting next day
                dd_sleep_until_idx = min(len(dates) - 1, int(t) + int(dd_sleep_days))
                meta["sleep_until"] = dates[int(dd_sleep_until_idx)].date().isoformat()
                # reduce from next trading day to segment end
                if int(t) + 1 <= seg_end_i:
                    if dd_scale <= 0.0:
                        w.iloc[int(t) + 1 : seg_end_i + 1, :] = 0.0
                    else:
                        w.iloc[int(t) + 1 : seg_end_i + 1, :] = w.iloc[int(t) + 1 : seg_end_i + 1, :].astype(float) * float(
                            dd_scale
                        )
            # keep tracking drawdown for crossing detection
            dd_prev_drawdown = float(dd_now)

        dd_now = 0.0 if dd_peak <= 0 else float(1.0 - dd_nav / dd_peak)
        meta.update(
            {
                "nav_asof": float(dd_nav),
                "peak_asof": float(dd_peak),
                "drawdown_asof": float(dd_now),
                "in_sleep": bool(dd_enabled and (seg_start_i <= dd_sleep_until_idx)),
            }
        )
        return meta

    for i, d in enumerate(decision_dates):
        # apply from next trading day after decision date
        di = dates.get_loc(d)
        if di + 1 >= len(dates):
            break
        start_i = di + 1
        next_di = (dates.get_loc(decision_dates[i + 1]) if i + 1 < len(decision_dates) else (len(dates) - 1))
        end_i = min(len(dates) - 1, next_di)
        dd_in_sleep = bool(dd_enabled and (start_i <= dd_sleep_until_idx))
        dd_meta: dict[str, Any] = {
            "enabled": bool(dd_enabled),
            "threshold": float(dd_threshold),
            "reduce": float(dd_reduce),
            "sleep_days": int(dd_sleep_days),
            "sleep_until": (dates[dd_sleep_until_idx].date().isoformat() if dd_sleep_until_idx >= 0 else None),
            "in_sleep": bool(dd_in_sleep),
            "triggered": False,
            "trigger_date": None,
            "trigger_drawdown": None,
        }

        candidate_scores: pd.Series | None = None
        # Sleep branch: keep previous day's weights; skip new decisions.
        if dd_in_sleep:
            prev_w_row = w.iloc[start_i - 1].astype(float)
            w.iloc[start_i : end_i + 1, :] = prev_w_row.to_numpy(dtype=float)
            held = [c for c in codes if float(prev_w_row.get(c, 0.0)) > 0.0]
            picks = [c for c in held if c in codes]
            meta = {"best_score": None, "mode": "dd_sleep"}
            group_meta = {
                "enabled": False,
                "policy": group_policy,
                "before": [],
                "after": [],
                "group_winners": {},
                "group_eliminated": {},
            }
        else:
            prev_w_row = w.iloc[start_i - 1].astype(float)
            # Per-asset momentum entry rules on the scoring series.
            scores_row = scores.loc[d]
            if use_floor_rules:
                s2 = scores_row.copy()
                for c in list(s2.index):
                    try:
                        sc = float(s2.get(c))
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(sc):
                        continue
                    eff_rules_all = _effective_rules_for_code(str(c), inp.asset_momentum_floor_rules)
                    entry_rules = _momentum_rules_for_stage(
                        str(c),
                        rules=inp.asset_momentum_floor_rules,
                        stage="entry",
                    )
                    if eff_rules_all and (not entry_rules):
                        # This asset only has exit-stage momentum rules; do not apply entry filtering.
                        continue
                    if not _momentum_rules_pass(
                        float(sc),
                        rules=entry_rules,
                        fallback_floor=0.0,
                    ):
                        s2.loc[c] = np.nan
                scores_row = s2

            candidate_scores = scores_row.dropna()
            candidate_count = int(candidate_scores.shape[0])
            if candidate_count <= 0:
                picks = []
                best = float(candidate_scores.max()) if candidate_count > 0 else None
                meta = {
                    "best_score": best,
                    "mode": "no_signal",
                    "candidate_count": candidate_count,
                    "target_top_k": int(inp.top_k),
                    "effective_top_k": 0,
                }
                group_meta = {
                    "enabled": bool(inp.group_enforce),
                    "policy": group_policy,
                    "before": [str(x) for x in candidate_scores.index.tolist()],
                    "after": [],
                    "group_winners": {},
                    "group_eliminated": {},
                }
            else:
                cur_holdings = {str(c) for c in rank_codes if float(prev_w_row.get(c, 0.0)) > 1e-12}
                group_vol_row = None
                if group_policy == "lowest_vol":
                    gv = ann_vol_by_window.get(int(inp.lookback_days))
                    if gv is not None and d in gv.index:
                        group_vol_row = gv.loc[d]
                reduced_scores, group_meta = _reduce_scores_by_group(
                    scores_row,
                    group_enforce=bool(inp.group_enforce),
                    asset_groups=group_map,
                    policy=group_policy,
                    current_holdings=cur_holdings,
                    vol_row=group_vol_row,
                )
                picks, meta = _pick_assets(
                    reduced_scores,
                    top_k=inp.top_k,
                )
                meta["candidate_count"] = int(candidate_count)
                meta["target_top_k"] = int(inp.top_k)
                meta["effective_top_k"] = int(min(top_k_abs, int(len(reduced_scores))))
        # picks == [] => cash
        reasons: list[str] = []
        details: dict[str, Any] = {}
        if dd_in_sleep:
            reasons.append("dd_sleep")

        picks = [p for p in picks if p in codes]
        risk_picks = [p for p in picks if p in rank_codes]  # only rank codes are considered "risk assets"

        # Inertia (dampening) is applied after core pick + risk-control decisions.
        inertia_meta: dict[str, Any] = {
            "enabled": bool(inertia_enabled),
            "min_hold_periods": int(inertia_min_hold),
            "score_gap": float(inertia_score_gap),
            "min_turnover": float(inertia_min_turnover),
            "blocked": False,
            "reason": None,
            "current_holdings": [],
            "new_picks": [],
            "expected_turnover": None,
        }

        # Universal ATR stop-loss metadata.
        atr_stop: dict[str, Any] = {"mode": atr_stop_mode, "atr_stop_atr_basis": atr_stop_atr_basis}
        stop_trigger_date: str | None = None
        stop_triggered = False

        # Apply pre-trade risk controls only when we are in risk-on mode (holding risk assets).
        candidate_ranked: list[str] = []
        entry_rejected_codes: set[str] = set()
        backfill_used = False
        backfill_added: list[str] = []
        backfill_initial: list[str] = list(risk_picks)
        if group_meta.get("after"):
            raw_ranked = [str(x) for x in list(group_meta.get("after") or [])]
            candidate_ranked = list(reversed(raw_ranked)) if inverse_top_k else raw_ranked
        elif candidate_scores is not None and hasattr(candidate_scores, "dropna"):
            cand = candidate_scores.dropna()
            asc = bool(inverse_top_k)
            candidate_ranked = [str(x) for x in cand.sort_values(ascending=asc).index.tolist()]
        details["score_by_code"] = {
            str(k): float(v)
            for k, v in (
                (scores_row.dropna().sort_values(ascending=False).to_dict().items())
                if hasattr(scores_row, "dropna")
                else []
            )
        }
        details["candidate_ranked"] = [str(x) for x in candidate_ranked]

        entry_enabled_count = int(bool(inp.chop_filter)) + int(bool(inp.trend_filter)) + int(bool(inp.bias_filter)) + int(bool(inp.rsi_filter))
        raw_entry_n = int(inp.entry_match_n or 0)
        if entry_enabled_count <= 0:
            entry_required = 0
        elif raw_entry_n <= 0:
            entry_required = int(entry_enabled_count)  # default: all enabled filters must pass (AND)
        else:
            entry_required = int(min(entry_enabled_count, max(1, raw_entry_n)))
        use_entry_nofm = bool(entry_enabled_count > 0 and entry_required < entry_enabled_count)
        if entry_enabled_count > 0:
            details["entry_gate"] = {
                "enabled_count": int(entry_enabled_count),
                "required": int(entry_required),
                "mode": ("and" if int(entry_required) >= int(entry_enabled_count) else "n_of_m"),
            }

        def _entry_ok_for_code(code: str, asof_d: pd.Timestamp) -> tuple[bool, int, int, dict[str, Any]]:
            p = str(code)
            passed = 0
            enabled = 0
            by_filter: dict[str, Any] = {}
            # Momentum entry filter (informational in trace; core pre-filter is applied on score series upstream).
            mom_enabled = bool(use_floor_rules)
            mom_ok: bool | None = None
            if mom_enabled:
                try:
                    sc0 = float(scores.loc[asof_d, p]) if (p in scores.columns and asof_d in scores.index) else float("nan")
                except (KeyError, TypeError, ValueError):
                    sc0 = float("nan")
                if np.isfinite(sc0):
                    entry_rules = _momentum_rules_for_stage(
                        str(p),
                        rules=inp.asset_momentum_floor_rules,
                        stage="entry",
                    )
                    eff_rules_all = _effective_rules_for_code(str(p), inp.asset_momentum_floor_rules)
                    if eff_rules_all and (not entry_rules):
                        mom_ok = True
                    else:
                        mom_ok = bool(
                            _momentum_rules_pass(
                                float(sc0),
                                rules=entry_rules,
                                fallback_floor=0.0,
                            )
                        )
            by_filter["momentum_rule"] = {"enabled": bool(mom_enabled), "ok": mom_ok, "checks": (1 if mom_enabled else 0)}
            # Choppiness filter
            if inp.chop_filter:
                enabled += 1
                eff = _effective_rules_for_code(p, inp.asset_chop_rules) if use_chop_rules else []
                if not eff:
                    eff = [
                        {
                            "chop_mode": cm,
                            "chop_window": int(inp.chop_window),
                            "chop_er_threshold": float(inp.chop_er_threshold),
                            "chop_adx_window": int(inp.chop_adx_window),
                            "chop_adx_threshold": float(inp.chop_adx_threshold),
                        }
                    ]
                oks: list[bool] = []
                for r in eff:
                    m = str((r or {}).get("chop_mode") or cm).strip().lower()
                    if m == "adx":
                        win = int((r or {}).get("chop_adx_window") or inp.chop_adx_window)
                        thr = float((r or {}).get("chop_adx_threshold") or inp.chop_adx_threshold)
                        a = adx_by_window.get(int(win))
                        if a is None or asof_d not in a.index:
                            continue
                        v = None if pd.isna(a.loc[asof_d, p]) else float(a.loc[asof_d, p])
                        oks.append(bool(v is not None and np.isfinite(v) and float(v) >= float(thr)))
                    else:
                        win = int((r or {}).get("chop_window") or inp.chop_window)
                        thr = float((r or {}).get("chop_er_threshold") or inp.chop_er_threshold)
                        e = er_by_window.get(int(win))
                        if e is None or asof_d not in e.index:
                            continue
                        v = None if pd.isna(e.loc[asof_d, p]) else float(e.loc[asof_d, p])
                        oks.append(bool(v is not None and np.isfinite(v) and float(v) >= float(thr)))
                ok_chop = (bool(all(oks)) if oks else True)
                if ok_chop:
                    passed += 1
                by_filter["chop"] = {
                    "enabled": True,
                    "ok": bool(ok_chop),
                    "checks": int(len(oks)),
                }
            else:
                by_filter["chop"] = {"enabled": False, "ok": None, "checks": 0}

            # Trend filter
            if inp.trend_filter:
                enabled += 1
                eff = _momentum_rules_for_stage(
                    str(p),
                    rules=inp.asset_trend_rules,
                    stage="entry",
                ) if use_trend_rules else []
                if not eff:
                    eff = [{"trend_sma_window": int(inp.trend_sma_window), "trend_ma_type": str(trend_ma_type), "op": ">"}]
                oks: list[bool] = []
                for r in eff:
                    win = int((r or {}).get("trend_sma_window") or inp.trend_sma_window)
                    mt = str((r or {}).get("trend_ma_type") or trend_ma_type).strip().lower()
                    if mt not in {"sma", "ema", "vma"}:
                        mt = str(trend_ma_type)
                    op = _normalize_cmp_op((r or {}).get("op") or ">")
                    df = trend_ok_each_by_key.get((int(win), str(mt), str(op)))
                    if df is None or asof_d not in df.index:
                        oks.append(False)
                    else:
                        oks.append(bool(df.loc[asof_d, p]))
                ok_trend = (bool(all(oks)) if oks else True)
                if ok_trend:
                    passed += 1
                by_filter["trend"] = {
                    "enabled": True,
                    "ok": bool(ok_trend),
                    "checks": int(len(oks)),
                }
            else:
                by_filter["trend"] = {"enabled": False, "ok": None, "checks": 0}

            # BIAS filter
            if inp.bias_filter:
                enabled += 1
                eff = _momentum_rules_for_stage(
                    str(p),
                    rules=inp.asset_bias_rules,
                    stage="entry",
                ) if use_bias_rules else []
                if not eff:
                    eff = [{
                        "bias_type": str(b_type),
                        "bias_ma_window": int(inp.bias_ma_window),
                        "level_window": str(inp.bias_level_window or "all"),
                        "threshold_type": str(inp.bias_threshold_type or "quantile"),
                        "quantile": float(inp.bias_quantile),
                        "fixed_value": float(inp.bias_fixed_value),
                        "min_periods": int(inp.bias_min_periods),
                        "op": ">",
                    }]
                oks: list[bool] = []
                for r in eff:
                    bty = str((r or {}).get("bias_type") or b_type).strip().lower()
                    if bty not in {"bias", "bias_v"}:
                        bty = str(b_type)
                    win = int((r or {}).get("bias_ma_window") or inp.bias_ma_window)
                    lvw = str((r or {}).get("level_window") or inp.bias_level_window or "all").strip().lower()
                    tt = str((r or {}).get("threshold_type") or inp.bias_threshold_type or "quantile").strip().lower()
                    qv = float((r or {}).get("quantile") if (r or {}).get("quantile") is not None else inp.bias_quantile)
                    fvv = float((r or {}).get("fixed_value") if (r or {}).get("fixed_value") is not None else inp.bias_fixed_value)
                    mp = int((r or {}).get("min_periods") if (r or {}).get("min_periods") is not None else inp.bias_min_periods)
                    op = _normalize_cmp_op((r or {}).get("op") or ">")
                    bdf = bias_by_key.get((int(win), str(bty)))
                    tdf = bias_thr_by_cfg.get((str(bty), int(win), str(lvw), str(tt), float(qv), float(fvv), int(mp)))
                    if bdf is None or tdf is None or asof_d not in bdf.index or asof_d not in tdf.index:
                        oks.append(False)
                        continue
                    sig = None if pd.isna(bdf.loc[asof_d, p]) else float(bdf.loc[asof_d, p])
                    thr = None if pd.isna(tdf.loc[asof_d, p]) else float(tdf.loc[asof_d, p])
                    if sig is None or thr is None or (not np.isfinite(sig)) or (not np.isfinite(thr)):
                        oks.append(False)
                    else:
                        oks.append(_compare_with_op(float(sig), float(thr), op))
                ok_bias = (bool(all(oks)) if oks else True)
                if ok_bias:
                    passed += 1
                by_filter["bias"] = {
                    "enabled": True,
                    "ok": bool(ok_bias),
                    "checks": int(len(oks)),
                }
            else:
                by_filter["bias"] = {"enabled": False, "ok": None, "checks": 0}

            # RSI filter
            if inp.rsi_filter:
                enabled += 1
                eff = _effective_rules_for_code(p, inp.asset_rsi_rules) if use_rsi_rules else []
                if not eff:
                    eff = [
                        {
                            "rsi_window": int(inp.rsi_window),
                            "rsi_overbought": float(inp.rsi_overbought),
                            "rsi_oversold": float(inp.rsi_oversold),
                            "rsi_block_overbought": bool(inp.rsi_block_overbought),
                            "rsi_block_oversold": bool(inp.rsi_block_oversold),
                        }
                    ]
                oks: list[bool] = []
                for r in eff:
                    win = int((r or {}).get("rsi_window") or inp.rsi_window)
                    df = rsi_by_window.get(int(win))
                    if df is None or asof_d not in df.index:
                        continue
                    v = None if pd.isna(df.loc[asof_d, p]) else float(df.loc[asof_d, p])
                    if v is None or (not np.isfinite(v)):
                        oks.append(True)
                        continue
                    ob = float((r or {}).get("rsi_overbought") if (r or {}).get("rsi_overbought") is not None else inp.rsi_overbought)
                    os = float((r or {}).get("rsi_oversold") if (r or {}).get("rsi_oversold") is not None else inp.rsi_oversold)
                    blk_ob = bool((r or {}).get("rsi_block_overbought") if (r or {}).get("rsi_block_overbought") is not None else inp.rsi_block_overbought)
                    blk_os = bool((r or {}).get("rsi_block_oversold") if (r or {}).get("rsi_block_oversold") is not None else inp.rsi_block_oversold)
                    ok = True
                    if blk_ob and float(v) > float(ob):
                        ok = False
                    if blk_os and float(v) < float(os):
                        ok = False
                    oks.append(bool(ok))
                ok_rsi = (bool(all(oks)) if oks else True)
                if ok_rsi:
                    passed += 1
                by_filter["rsi"] = {
                    "enabled": True,
                    "ok": bool(ok_rsi),
                    "checks": int(len(oks)),
                }
            else:
                by_filter["rsi"] = {"enabled": False, "ok": None, "checks": 0}
            ok_all = (passed >= enabled) if enabled > 0 else True
            return ok_all, int(passed), int(enabled), by_filter

        entry_checks_by_code: dict[str, dict[str, Any]] = {}
        if candidate_ranked:
            for cc in candidate_ranked:
                ok_all, pass_cnt, enabled_cnt, by_filter = _entry_ok_for_code(str(cc), d)
                need_n = int(enabled_cnt) if int(enabled_cnt) > 0 else 0
                if use_entry_nofm:
                    need_n = int(entry_required)
                ok_entry = bool(ok_all) if (not use_entry_nofm) else (int(pass_cnt) >= int(need_n))
                entry_checks_by_code[str(cc)] = {
                    "ok_all": bool(ok_all),
                    "ok_gate": bool(ok_entry),
                    "pass_count": int(pass_cnt),
                    "enabled_count": int(enabled_cnt),
                    "required": int(need_n),
                    "by_filter": by_filter,
                }
        details["entry_checks_by_code"] = entry_checks_by_code

        if risk_picks and (not dd_in_sleep):
            # 0) Choppiness filter (ER / ADX)
            if inp.chop_filter:
                before = risk_picks[:]
                er_map: dict[str, float | None] = {}
                adx_map: dict[str, float | None] = {}
                ok_map: dict[str, bool] = {}

                for p in before:
                    eff = _effective_rules_for_code(p, inp.asset_chop_rules) if use_chop_rules else []
                    if not eff:
                        eff = [
                            {
                                "chop_mode": cm,
                                "chop_window": int(inp.chop_window),
                                "chop_er_threshold": float(inp.chop_er_threshold),
                                "chop_adx_window": int(inp.chop_adx_window),
                                "chop_adx_threshold": float(inp.chop_adx_threshold),
                            }
                        ]

                    oks: list[bool] = []
                    for r in eff:
                        m = str((r or {}).get("chop_mode") or cm).strip().lower()
                        if m == "adx":
                            win = int((r or {}).get("chop_adx_window") or inp.chop_adx_window)
                            thr = float((r or {}).get("chop_adx_threshold") or inp.chop_adx_threshold)
                            a = adx_by_window.get(int(win))
                            if a is None or d not in a.index:
                                continue
                            v = None if pd.isna(a.loc[d, p]) else float(a.loc[d, p])
                            adx_map[p] = v
                            oks.append(bool(v is not None and np.isfinite(v) and float(v) >= float(thr)))
                        else:
                            win = int((r or {}).get("chop_window") or inp.chop_window)
                            thr = float((r or {}).get("chop_er_threshold") or inp.chop_er_threshold)
                            e = er_by_window.get(int(win))
                            if e is None or d not in e.index:
                                continue
                            v = None if pd.isna(e.loc[d, p]) else float(e.loc[d, p])
                            er_map[p] = v
                            oks.append(bool(v is not None and np.isfinite(v) and float(v) >= float(thr)))

                    # If indicators are not available on this day (warm-up), do not exclude.
                    ok_map[p] = bool(all(oks)) if oks else True

                if er_map:
                    details["er"] = er_map
                if adx_map:
                    details["adx"] = adx_map

                risk_picks = [p for p in before if ok_map.get(p, False)]
                removed = [p for p in before if p not in risk_picks]
                if removed:
                    entry_rejected_codes.update([str(x) for x in removed])
                    reasons.append(f"chop_exclude:{','.join(removed)}")
                    picks = [p for p in picks if (p not in rank_codes) or (p in risk_picks)]

            # 1) Trend filter
            if inp.trend_filter:
                before = risk_picks[:]
                ok_map: dict[str, bool] = {}
                for p in before:
                    eff = _momentum_rules_for_stage(
                        str(p),
                        rules=inp.asset_trend_rules,
                        stage="entry",
                    ) if use_trend_rules else []
                    if not eff:
                        eff = [{"trend_sma_window": int(inp.trend_sma_window), "trend_ma_type": str(trend_ma_type), "op": ">"}]
                    oks: list[bool] = []
                    for r in eff:
                        win = int((r or {}).get("trend_sma_window") or inp.trend_sma_window)
                        mt = str((r or {}).get("trend_ma_type") or trend_ma_type).strip().lower()
                        if mt not in {"sma", "ema", "vma"}:
                            mt = str(trend_ma_type)
                        op = _normalize_cmp_op((r or {}).get("op") or ">")
                        df = trend_ok_each_by_key.get((int(win), str(mt), str(op)))
                        if df is None or d not in df.index:
                            oks.append(False)
                        else:
                            oks.append(bool(df.loc[d, p]))
                    ok_map[p] = bool(all(oks)) if oks else True
                details["trend_each_ok"] = ok_map
                risk_picks = [p for p in before if ok_map.get(p, False)]
                removed = [p for p in before if p not in risk_picks]
                if removed:
                    entry_rejected_codes.update([str(x) for x in removed])
                    reasons.append(f"trend_exclude:{','.join(removed)}")
                    picks = [p for p in picks if (p not in rank_codes) or (p in risk_picks)]

            # 2) BIAS filter
            if risk_picks and inp.bias_filter:
                before = risk_picks[:]
                ok_map: dict[str, bool] = {}
                for p in before:
                    eff = _momentum_rules_for_stage(
                        str(p),
                        rules=inp.asset_bias_rules,
                        stage="entry",
                    ) if use_bias_rules else []
                    if not eff:
                        eff = [{
                            "bias_type": str(b_type),
                            "bias_ma_window": int(inp.bias_ma_window),
                            "level_window": str(inp.bias_level_window or "all"),
                            "threshold_type": str(inp.bias_threshold_type or "quantile"),
                            "quantile": float(inp.bias_quantile),
                            "fixed_value": float(inp.bias_fixed_value),
                            "min_periods": int(inp.bias_min_periods),
                            "op": ">",
                        }]
                    oks: list[bool] = []
                    for r in eff:
                        bty = str((r or {}).get("bias_type") or b_type).strip().lower()
                        if bty not in {"bias", "bias_v"}:
                            bty = str(b_type)
                        win = int((r or {}).get("bias_ma_window") or inp.bias_ma_window)
                        lvw = str((r or {}).get("level_window") or inp.bias_level_window or "all").strip().lower()
                        tt = str((r or {}).get("threshold_type") or inp.bias_threshold_type or "quantile").strip().lower()
                        qv = float((r or {}).get("quantile") if (r or {}).get("quantile") is not None else inp.bias_quantile)
                        fvv = float((r or {}).get("fixed_value") if (r or {}).get("fixed_value") is not None else inp.bias_fixed_value)
                        mp = int((r or {}).get("min_periods") if (r or {}).get("min_periods") is not None else inp.bias_min_periods)
                        op = _normalize_cmp_op((r or {}).get("op") or ">")
                        bdf = bias_by_key.get((int(win), str(bty)))
                        tdf = bias_thr_by_cfg.get((str(bty), int(win), str(lvw), str(tt), float(qv), float(fvv), int(mp)))
                        if bdf is None or tdf is None or d not in bdf.index or d not in tdf.index:
                            oks.append(False)
                            continue
                        sig = None if pd.isna(bdf.loc[d, p]) else float(bdf.loc[d, p])
                        thr = None if pd.isna(tdf.loc[d, p]) else float(tdf.loc[d, p])
                        if sig is None or thr is None or (not np.isfinite(sig)) or (not np.isfinite(thr)):
                            oks.append(False)
                        else:
                            oks.append(_compare_with_op(float(sig), float(thr), op))
                    ok_map[p] = bool(all(oks)) if oks else True
                details["bias_each_ok"] = ok_map
                risk_picks = [p for p in before if ok_map.get(p, False)]
                removed = [p for p in before if p not in risk_picks]
                if removed:
                    entry_rejected_codes.update([str(x) for x in removed])
                    reasons.append(f"bias_exclude:{','.join(removed)}")
                    picks = [p for p in picks if (p not in rank_codes) or (p in risk_picks)]

            # 3) RSI filter
            if risk_picks and inp.rsi_filter:
                before = risk_picks[:]
                rsi_map: dict[str, float | None] = {}
                ok_map: dict[str, bool] = {}
                for p in before:
                    eff = _effective_rules_for_code(p, inp.asset_rsi_rules) if use_rsi_rules else []
                    if not eff:
                        eff = [
                            {
                                "rsi_window": int(inp.rsi_window),
                                "rsi_overbought": float(inp.rsi_overbought),
                                "rsi_oversold": float(inp.rsi_oversold),
                                "rsi_block_overbought": bool(inp.rsi_block_overbought),
                                "rsi_block_oversold": bool(inp.rsi_block_oversold),
                            }
                        ]
                    oks: list[bool] = []
                    for r in eff:
                        win = int((r or {}).get("rsi_window") or inp.rsi_window)
                        df = rsi_by_window.get(int(win))
                        if df is None or d not in df.index:
                            continue
                        v = None if pd.isna(df.loc[d, p]) else float(df.loc[d, p])
                        rsi_map[p] = v
                        if v is None or (not np.isfinite(v)):
                            # missing RSI: do not block
                            oks.append(True)
                            continue
                        ob = float((r or {}).get("rsi_overbought") if (r or {}).get("rsi_overbought") is not None else inp.rsi_overbought)
                        os = float((r or {}).get("rsi_oversold") if (r or {}).get("rsi_oversold") is not None else inp.rsi_oversold)
                        blk_ob = bool((r or {}).get("rsi_block_overbought") if (r or {}).get("rsi_block_overbought") is not None else inp.rsi_block_overbought)
                        blk_os = bool((r or {}).get("rsi_block_oversold") if (r or {}).get("rsi_block_oversold") is not None else inp.rsi_block_oversold)
                        ok = True
                        if blk_ob and float(v) > float(ob):
                            ok = False
                        if blk_os and float(v) < float(os):
                            ok = False
                        oks.append(bool(ok))
                    ok_map[p] = bool(all(oks)) if oks else True
                if rsi_map:
                    details["rsi"] = rsi_map
                risk_picks = [p for p in before if ok_map.get(p, False)]
                removed = [p for p in before if p not in risk_picks]
                if removed:
                    entry_rejected_codes.update([str(x) for x in removed])
                    reasons.append(f"rsi_exclude:{','.join(removed)}")
                    picks = [p for p in picks if (p not in rank_codes) or (p in risk_picks)]

            # If filters removed all risk assets, fall back to cash (planned no-buy).
            if (not risk_picks) and (meta.get("mode") == "risk_on"):
                reasons.append("risk_controls_block_all")
                picks = []
                meta = {"best_score": meta.get("best_score"), "mode": "cash"}

            # Optional n-of-m aggregation for entry filters (trend/rsi/chop):
            # default n=0 keeps original AND behavior.
            if use_entry_nofm and backfill_initial:
                allowed_top: list[str] = []
                pass_count_by_code: dict[str, int] = {}
                for p in backfill_initial:
                    _, pass_cnt, _, _ = _entry_ok_for_code(str(p), d)
                    pass_count_by_code[str(p)] = int(pass_cnt)
                    if int(pass_cnt) >= int(entry_required):
                        allowed_top.append(str(p))
                removed_top = [str(p) for p in backfill_initial if str(p) not in set(allowed_top)]
                if removed_top:
                    entry_rejected_codes.update(removed_top)
                    reasons.append(f"entry_nofm_exclude:{','.join(removed_top)}")
                details["entry_gate"] = {
                    **(details.get("entry_gate") or {}),
                    "pass_count_by_code": pass_count_by_code,
                }
                risk_picks = [p for p in allowed_top if p in rank_codes]
                picks = [p for p in picks if p not in rank_codes] + risk_picks
                if (not risk_picks) and (meta.get("mode") == "risk_on"):
                    picks = []
                    meta = {"best_score": meta.get("best_score"), "mode": "cash"}

            # Optional refill from lower-ranked candidates after entry filters.
            # Only applies to risk-on branch (no defensive/cash replacement).
            if bool(inp.entry_backfill) and (meta.get("mode") == "risk_on") and candidate_ranked and (len(risk_picks) < top_k_abs):
                current = [str(x) for x in risk_picks]
                cur_set = set(current)
                for c in candidate_ranked:
                    cc = str(c)
                    if cc in cur_set:
                        continue
                    ok_all, pass_cnt, enabled_cnt, _ = _entry_ok_for_code(cc, d)
                    need_n = int(enabled_cnt) if int(enabled_cnt) > 0 else 0
                    if use_entry_nofm:
                        need_n = int(entry_required)
                    ok_entry = bool(ok_all) if (not use_entry_nofm) else (int(pass_cnt) >= int(need_n))
                    if not ok_entry:
                        continue
                    current.append(cc)
                    cur_set.add(cc)
                    backfill_added.append(cc)
                    if len(current) >= top_k_abs:
                        break
                if backfill_added:
                    risk_picks = current[:top_k_abs]
                    picks = [p for p in picks if p not in rank_codes] + risk_picks
                    backfill_used = True
                    reasons.append(f"entry_backfill:{','.join(backfill_added)}")

        # Correlation gate (qfq): if new picks are too correlated with current holdings, skip rebalance this period.
        corr_meta: dict[str, Any] = {"enabled": bool(corr_enabled), "window": int(corr_window), "threshold": float(corr_threshold)}
        if corr_enabled:
            cur_hold = [] if prev_segment_stopped_out else (list(prev_picks_key) if prev_picks_key else [])
            cur_key = tuple(sorted([c for c in cur_hold if c in rank_codes]))
            new_key = tuple(sorted([c for c in (picks or []) if c in rank_codes]))
            corr_meta["current_holdings"] = list(cur_key)
            corr_meta["new_picks"] = list(new_key)
            corr_meta["blocked"] = False
            if cur_key and new_key and (cur_key != new_key):
                end_pos = int(dates.get_loc(d))
                max_corr = None
                max_pair = None
                for a in new_key:
                    for b in cur_key:
                        cval = _pair_corr_qfq(code_a=a, code_b=b, end_pos=end_pos)
                        if cval is None or (not np.isfinite(cval)):
                            continue
                        if (max_corr is None) or (float(cval) > float(max_corr)):
                            max_corr = float(cval)
                            max_pair = (a, b)
                corr_meta["max_corr"] = max_corr
                corr_meta["max_pair"] = list(max_pair) if max_pair else None
                if max_corr is not None and float(max_corr) > float(corr_threshold):
                    corr_meta["blocked"] = True
                    # keep current holdings (no rebalance)
                    picks = list(cur_key)
                    # also reset risk_picks to match picks for downstream sizing logic
                    risk_picks = list(cur_key)

        # Inertia: block frequent holding changes.
        if inertia_enabled and (not dd_in_sleep):
            cur_key = tuple(sorted([c for c in rank_codes if float(prev_w_row.get(c, 0.0)) > 1e-12]))
            cur_def = tuple(sorted([c for c in codes if (c not in rank_codes) and float(prev_w_row.get(c, 0.0)) > 1e-12]))
            new_key = tuple(sorted([c for c in (risk_picks or []) if c in rank_codes]))
            inertia_meta["current_holdings"] = list(cur_key)
            inertia_meta["new_picks"] = list(new_key)

            # Only dampen when both sides are "risk-on" and holdings would change.
            if cur_key and new_key and (cur_key != new_key):
                # 1) Minimum holding periods (decision-count based)
                if inertia_min_hold > 0 and (int(i) - int(last_change_decision_i)) < int(inertia_min_hold):
                    inertia_meta["blocked"] = True
                    inertia_meta["reason"] = "min_hold"
                    reasons.append(f"inertia_min_hold<{inertia_min_hold}")
                    picks = list(cur_key) + list(cur_def)
                    risk_picks = list(cur_key)
                # 2) Score gap (only meaningful for top_k=1)
                elif (float(inertia_score_gap) > 0.0) and (top_k_abs == 1) and (len(cur_key) == 1) and (len(new_key) == 1):
                    cur_c = str(cur_key[0])
                    new_c = str(new_key[0])
                    try:
                        cur_s = float(scores.loc[d, cur_c])
                        new_s = float(scores.loc[d, new_c])
                    except (KeyError, TypeError, ValueError):  # pragma: no cover (defensive)
                        cur_s = float("nan")
                        new_s = float("nan")
                    if np.isfinite(cur_s) and np.isfinite(new_s):
                        gap = float(new_s - cur_s)
                        inertia_meta["score_gap_now"] = gap
                        if gap < float(inertia_score_gap):
                            inertia_meta["blocked"] = True
                            inertia_meta["reason"] = "score_gap"
                            reasons.append(f"inertia_score_gap<{inertia_score_gap}")
                            picks = list(cur_key) + list(cur_def)
                            risk_picks = list(cur_key)

        # Rolling-return based exposure scaling (cash remainder).
        rr_meta: dict[str, Any] = {"enabled": bool(rr_enabled), "years": float(inp.rr_years), "window_days": int(rr_window_days)}
        rr_exposure = 1.0
        if rr_enabled and (not dd_in_sleep):
            _advance_nav_to(int(di))
            start_rr = max(0, int(di) - int(rr_window_days))
            base_nav = float(nav_running.iloc[start_rr])
            cur_nav = float(nav_running.iloc[int(di)])
            trailing = (cur_nav / base_nav - 1.0) if base_nav > 0 else float("nan")
            bucket, rr_exposure = _rr_bucket_exposure(float(trailing) if np.isfinite(trailing) else -1e9)
            rr_meta.update(
                {
                    "asof": d.date().isoformat(),
                    "trailing_return": float(trailing),
                    "bucket": int(bucket),
                    "exposure": float(rr_exposure),
                    "thresholds": [float(x) for x in rr_thresholds],
                    "weights": [float(x) for x in rr_weights],
                }
            )
        else:
            rr_meta.update({"asof": d.date().isoformat(), "trailing_return": None, "bucket": None, "exposure": None})

        # 3) Base sizing + volatility scaling (cash remainder).
        exposure = 1.0
        weight_map: dict[str, float] = {}
        if picks and (not dd_in_sleep):
            risk_picks = [p for p in picks if p in rank_codes]
            if not risk_picks:
                exposure = 0.0
            else:
                base_weight_map: dict[str, float] = {}
                if pos_mode == "risk_budget":
                    rb_meta: dict[str, Any] = {
                        "enabled": True,
                        "atr_window": int(risk_budget_atr_window),
                        "risk_budget_pct": float(risk_budget_pct),
                        "by_code": {},
                        "scaled_to_cap": False,
                    }
                    for p in risk_picks:
                        px = float(close_qfq.loc[d, p]) if (p in close_qfq.columns and d in close_qfq.index) else float("nan")
                        a = float(atr_budget.loc[d, p]) if (not atr_budget.empty and p in atr_budget.columns and d in atr_budget.index) else float("nan")
                        if np.isfinite(px) and px > 0.0 and np.isfinite(a) and a > 0.0:
                            w_raw = float(risk_budget_pct) * float(px) / float(a)
                        else:
                            w_raw = 0.0
                        base_weight_map[p] = max(0.0, float(w_raw))
                        rb_meta["by_code"][str(p)] = {
                            "close": (None if (not np.isfinite(px)) else float(px)),
                            "atr": (None if (not np.isfinite(a)) else float(a)),
                            "weight_raw": float(w_raw),
                        }
                    s_raw = float(sum(base_weight_map.values()))
                    if s_raw > 1.0 + 1e-12:
                        k = 1.0 / s_raw
                        for p in list(base_weight_map.keys()):
                            base_weight_map[p] = float(base_weight_map[p]) * float(k)
                        rb_meta["scaled_to_cap"] = True
                        rb_meta["raw_sum_before_cap"] = float(s_raw)
                    details["risk_budget"] = rb_meta
                else:
                    if not risk_picks:
                        per = 0.0
                    elif pos_mode == "fixed":
                        per = 1.0 / max(1, top_k_abs)
                    else:
                        per = 1.0 / len(risk_picks)
                    for p in risk_picks:
                        base_weight_map[p] = float(per)

                scales = {p: 1.0 for p in risk_picks}
                if inp.vol_monitor and ta_close is not None:
                    vol_map: dict[str, float | None] = {}
                    for p in risk_picks:
                        eff = _effective_rules_for_code(p, inp.asset_vol_monitor_rules) if use_vol_rules else []
                        if not eff:
                            eff = [
                                {
                                    "vol_window": int(inp.vol_window),
                                    "vol_target_ann": float(inp.vol_target_ann),
                                    "vol_max_ann": float(inp.vol_max_ann),
                                }
                            ]
                        p_scales: list[float] = []
                        p_vols: list[float] = []
                        for r in eff:
                            win = int((r or {}).get("vol_window") or inp.vol_window)
                            df = ann_vol_by_window.get(int(win))
                            if df is None or d not in df.index:
                                continue
                            v = None if pd.isna(df.loc[d, p]) else float(df.loc[d, p])
                            if v is None or (not np.isfinite(v)):
                                # missing vol: conservative block for this rule
                                p_scales.append(0.0)
                                continue
                            p_vols.append(float(v))
                            if float(v) <= 0:
                                p_scales.append(1.0)
                                continue
                            vmax = float((r or {}).get("vol_max_ann") or inp.vol_max_ann)
                            vtar = float((r or {}).get("vol_target_ann") or inp.vol_target_ann)
                            if float(v) >= float(vmax):
                                p_scales.append(0.0)
                            else:
                                p_scales.append(float(min(1.0, float(vtar) / float(v))))

                        if p_scales:
                            scales[p] = float(min(p_scales))
                            vol_map[p] = float(p_vols[-1]) if p_vols else None
                        else:
                            # warm-up: keep scale=1 when indicators unavailable
                            scales[p] = float(scales.get(p, 1.0))
                            vol_map[p] = None

                    if vol_map:
                        details["ann_vol"] = vol_map
                for p in risk_picks:
                    weight_map[p] = float(scales.get(p, 0.0) * base_weight_map.get(p, 0.0))
                exposure = float(sum(weight_map.values()))

                # If vol sizing zeroed all positions, fall back to cash.
                if exposure <= 0.0:
                    reasons.append("vol_block_all")
                    weight_map = {}
                    picks = []
                    meta = {"best_score": meta.get("best_score"), "mode": "cash"}
                    exposure = 0.0

            # Apply rolling-return exposure scaling as the final layer (cash remainder).
            if rr_enabled and weight_map and float(rr_exposure) < 1.0:
                for c in list(weight_map.keys()):
                    weight_map[c] = float(weight_map[c]) * float(rr_exposure)
                exposure = float(exposure) * float(rr_exposure)

            # Mirror composite-deviation exposure cap (cash remainder).
            mirror_meta: dict[str, Any] = {
                "enabled": bool(mirror_enabled),
                "quantiles": [float(x) for x in mirror_qs],
                "exposures": [float(x) for x in mirror_exps],
            }
            if mirror_enabled and weight_map and mirror_pct is not None:
                try:
                    p = float(mirror_pct.loc[d])
                except Exception:  # pragma: no cover (defensive)
                    p = float("nan")
                mirror_meta["asof"] = d.date().isoformat()
                mirror_meta["percentile"] = (None if (not np.isfinite(p)) else float(p))
                target = 1.0
                if np.isfinite(p):
                    for thr, exp in zip(mirror_qs, mirror_exps):
                        if float(p) > float(thr):
                            target = float(exp)
                mirror_meta["target_exposure"] = float(target)
                if np.isfinite(p) and float(exposure) > float(target) + 1e-12:
                    # Scale down all risk assets proportionally; keep defensive (if any) untouched.
                    risk_keys = [c for c in weight_map.keys() if c in rank_codes]
                    if risk_keys:
                        factor = float(target) / float(exposure) if float(exposure) > 0 else 0.0
                        for c in risk_keys:
                            weight_map[c] = float(weight_map[c]) * float(factor)
                        exposure = float(target)
                        reasons.append(f"mirror_cap<{target:.3f}")
                details["mirror_control"] = mirror_meta
            else:
                details["mirror_control"] = mirror_meta

            # Mirror composite-deviation exposure cap (cash remainder).
            mirror_meta: dict[str, Any] = {"enabled": bool(mirror_enabled), "quantiles": [float(x) for x in mirror_qs], "exposures": [float(x) for x in mirror_exps]}
            if mirror_enabled and weight_map and mirror_pct is not None:
                try:
                    p = float(mirror_pct.loc[d])
                except Exception:  # pragma: no cover (defensive)
                    p = float("nan")
                mirror_meta["asof"] = d.date().isoformat()
                mirror_meta["percentile"] = (None if (not np.isfinite(p)) else float(p))
                target = 1.0
                if np.isfinite(p):
                    for thr, exp in zip(mirror_qs, mirror_exps):
                        if float(p) > float(thr):
                            target = float(exp)
                mirror_meta["target_exposure"] = float(target)
                if np.isfinite(p) and float(exposure) > float(target) + 1e-12:
                    # scale down all risk assets proportionally; keep defensive (if any) untouched
                    risk_keys = [c for c in weight_map.keys() if c in rank_codes]
                    if risk_keys:
                        factor = float(target) / float(exposure) if float(exposure) > 0 else 0.0
                        for c in risk_keys:
                            weight_map[c] = float(weight_map[c]) * float(factor)
                        exposure = float(target)
                        reasons.append(f"mirror_cap<{target:.3f}")
                details["mirror_control"] = mirror_meta
            else:
                details["mirror_control"] = {"enabled": bool(mirror_enabled), "quantiles": [float(x) for x in mirror_qs], "exposures": [float(x) for x in mirror_exps]}

            # Inertia: turnover threshold gate (applied after sizing determines final target weights).
            if inertia_enabled and float(inertia_min_turnover) > 0.0:
                prev_np = prev_w_row.to_numpy(dtype=float)
                new_np = np.zeros(len(codes), dtype=float)
                if weight_map:
                    for j, c in enumerate(codes):
                        if c in weight_map:
                            new_np[j] = float(weight_map[c])
                exp_turn = float(np.abs(new_np - prev_np).sum() / 2.0)
                inertia_meta["expected_turnover"] = exp_turn
                if exp_turn < float(inertia_min_turnover):
                    inertia_meta["blocked"] = True
                    inertia_meta["reason"] = "min_turnover"
                    reasons.append(f"inertia_turnover<{inertia_min_turnover}")
                    # keep previous holdings for the whole segment
                    w.iloc[start_i : end_i + 1, :] = prev_np
                    picks = [c for c in codes if float(prev_w_row.get(c, 0.0)) > 1e-12]
                    risk_picks = [c for c in rank_codes if float(prev_w_row.get(c, 0.0)) > 1e-12]
                    exposure = float(np.sum(prev_np))
                    weight_map = {}

            # write weights for the whole holding segment (unless overridden by inertia turnover gate above)
            if weight_map:
                for c, wt in weight_map.items():
                    if wt and wt > 0:
                        w.loc[dates[start_i] : dates[end_i], c] = float(wt)
        elif dd_in_sleep:
            # weights already copied from previous day; compute exposure for reporting
            exposure = float(w.iloc[start_i].sum()) if start_i < len(dates) else 0.0

        # Compute ATR stop-loss metadata AFTER final picks are fixed.
        # IMPORTANT: stop-loss uses qfq close for both stop level and trigger.
        if atr_stop_exec_mode in {"atr_stop_static", "atr_stop_trailing", "atr_stop_tightening"} and (not dd_in_sleep) and risk_picks:
            atr_stop["atr_window_used"] = int(w_atr) if w_atr is not None else None
            atr_stop["atr_stop_n"] = float(inp.atr_stop_n)
            atr_stop["atr_stop_m"] = float(inp.atr_stop_m)
            atr_stop["atr_stop_min_distance_mult"] = float(inp.atr_stop_m)
            atr_stop["atr_stop_mode"] = str(atr_stop_mode)
            atr_stop["atr_stop_atr_basis"] = str(atr_stop_atr_basis)

        # In-segment ATR stop-loss check (after weights are written for the segment).
        # We approximate execution as: hold through close on trigger day, then go cash from next trading day.
        if atr_stop_exec_mode in {"atr_stop_static", "atr_stop_trailing", "atr_stop_tightening"} and risk_picks:
            seg_dates = dates[start_i : end_i + 1]
            # Build/initialize per-asset state at segment start.
            # We maintain a trailing stop that never decreases.
            entry_px: dict[str, float] = {}
            entry_atr: dict[str, float] = {}
            prev_close: dict[str, float] = {}
            stop: dict[str, float] = {}
            atr_n = float(inp.atr_stop_n)
            atr_m = float(inp.atr_stop_m)
            atr_min = float(inp.atr_stop_m)
            for c in risk_picks:
                try:
                    p0 = float(close_qfq.loc[seg_dates[0], c])
                    a0 = float(atr.loc[seg_dates[0], c])
                except (KeyError, TypeError, ValueError):  # pragma: no cover
                    continue
                if not (np.isfinite(p0) and np.isfinite(a0) and a0 > 0):
                    continue
                entry_px[c] = p0
                entry_atr[c] = a0
                prev_close[c] = p0
                stop[c] = p0 - float(atr_n) * a0
            atr_stop["entry_price_by_code"] = {k: float(v) for k, v in entry_px.items()}
            atr_stop["initial_stop_by_code"] = {k: float(v) for k, v in stop.items()}

            trig_idx: int | None = None
            trig_code: str | None = None
            # Iterate days: first check trigger vs previous stop, then update stop for next day.
            for j, day in enumerate(seg_dates):
                # 1) check trigger
                for c in risk_picks:
                    if c not in stop:
                        continue
                    try:
                        px = float(close_qfq.loc[day, c])
                    except (KeyError, TypeError, ValueError):  # pragma: no cover
                        continue
                    if np.isfinite(px) and px < float(stop[c]):
                        trig_idx = j
                        trig_code = c
                        break
                if trig_idx is not None:
                    break

                # 2) update stop using today's close and today's ATR (for next day)
                for c in risk_picks:
                    if c not in stop:
                        continue
                    try:
                        px = float(close_qfq.loc[day, c])
                        a = float(atr.loc[day, c])
                    except (KeyError, TypeError, ValueError):  # pragma: no cover
                        continue
                    if not (np.isfinite(px) and np.isfinite(a) and a > 0):
                        continue

                    should_update = True
                    if atr_stop_exec_mode == "atr_stop_static":
                        should_update = False
                        dist_mult = float(atr_n)
                    elif atr_stop_exec_mode == "atr_stop_trailing":
                        dist_mult = float(atr_n)
                        ep = float(entry_px.get(c, px))
                        pc = float(prev_close.get(c, px))
                        should_update = bool((px > ep) or (px > pc))
                    else:
                        # Tightening ATR basis: entry or latest, based on atr_stop_atr_basis.
                        ep = float(entry_px.get(c, px))
                        a_ref = float(entry_atr.get(c, a)) if atr_stop_atr_basis == "entry" else float(a)
                        gain_units = (px - ep) / max(a_ref, 1e-12)
                        steps = int(np.floor(gain_units / float(atr_m))) if np.isfinite(gain_units) else 0
                        dist_mult = float(atr_n) - float(steps) * float(atr_m)
                        dist_mult = float(max(float(atr_min), dist_mult))
                        pc = float(prev_close.get(c, px))
                        should_update = bool((px > ep) or (px > pc))

                    if should_update:
                        a_ref = float(entry_atr.get(c, a)) if atr_stop_atr_basis == "entry" else float(a)
                        cand = px - dist_mult * a_ref
                        # stop never decreases
                        stop[c] = float(max(float(stop[c]), float(cand)))
                    prev_close[c] = px

            if trig_idx is not None:
                stop_triggered = True
                stop_trigger_date = seg_dates[trig_idx].date().isoformat()
                atr_stop["triggered"] = True
                atr_stop["trigger_date"] = stop_trigger_date
                atr_stop["trigger_code"] = trig_code
                if trig_idx + 1 < len(seg_dates):
                    w.loc[seg_dates[trig_idx + 1] : seg_dates[-1], :] = 0.0
            else:
                atr_stop["triggered"] = False
            atr_stop["final_stop_by_code"] = {k: float(v) for k, v in stop.items()}

        # Daily close decision for exit controls; execute next trading day.
        use_trend_exit = bool(inp.trend_exit_filter)
        use_bias_exit = bool(inp.bias_exit_filter)
        exit_enabled_count = int(bool(use_floor_rules)) + int(bool(use_trend_exit)) + int(bool(use_bias_exit))
        raw_exit_n = int(inp.exit_match_n or 0)
        if exit_enabled_count <= 0:
            exit_required = 0
        elif raw_exit_n <= 0:
            exit_required = int(exit_enabled_count)  # default: all enabled exit filters must hit (AND)
        else:
            exit_required = int(min(exit_enabled_count, max(1, raw_exit_n)))
        daily_exit_meta: dict[str, Any] = {
            "enabled": bool(use_floor_rules or use_trend_exit or use_bias_exit),
            "momentum_enabled": bool(use_floor_rules),
            "trend_enabled": bool(use_trend_exit),
            "bias_enabled": bool(use_bias_exit),
            "gate": {
                "enabled_count": int(exit_enabled_count),
                "required": int(exit_required),
                "mode": ("and" if int(exit_required) >= int(exit_enabled_count) else "n_of_m"),
            },
            "triggered": False,
            "events": [],
            "checks_by_day": [],
        }
        if (use_floor_rules or use_trend_exit or use_bias_exit) and (not dd_in_sleep) and risk_picks:
            cur_risk_set = {str(c) for c in risk_picks if c in rank_codes}
            for t in range(int(start_i), int(end_i)):
                sig_d = dates[int(t)]
                exec_d = dates[int(t) + 1]
                if not cur_risk_set:
                    break
                day_checks: list[dict[str, Any]] = []
                for c in list(cur_risk_set):
                    prev_wt = float(w.loc[sig_d, c]) if c in w.columns else 0.0
                    if prev_wt <= 1e-12:
                        continue
                    hit_map: dict[str, bool] = {}
                    momentum_score: float | None = None

                    # 1) momentum-based exit: condition hit => trigger candidate exit
                    if use_floor_rules and c in scores.columns:
                        try:
                            sc = float(scores.loc[sig_d, c])
                        except (KeyError, TypeError, ValueError):
                            sc = float("nan")
                        if np.isfinite(sc):
                            momentum_score = float(sc)
                            exit_rules = _momentum_rules_for_stage(
                                str(c),
                                rules=inp.asset_momentum_floor_rules,
                                stage="exit",
                            )
                            if exit_rules:
                                hit_map["momentum_rule"] = bool(
                                    _momentum_rules_pass(
                                    float(sc),
                                    rules=exit_rules,
                                fallback_floor=0.0,
                                )
                                )

                    # 2) trend-based exit: condition hit => trigger candidate exit
                    if use_trend_exit:
                        trend_rules = _momentum_rules_for_stage(
                            str(c),
                            rules=inp.asset_trend_rules,
                            stage="exit",
                        ) if use_trend_rules else []
                        if not trend_rules:
                            trend_rules = [{"trend_sma_window": int(inp.trend_sma_window), "trend_ma_type": str(trend_ma_type), "op": "<"}]
                        trend_hit = True
                        for r in trend_rules:
                            win = int((r or {}).get("trend_sma_window") or inp.trend_sma_window)
                            mt = str((r or {}).get("trend_ma_type") or trend_ma_type).strip().lower()
                            if mt not in {"sma", "ema", "vma"}:
                                mt = str(trend_ma_type)
                            op = _normalize_cmp_op((r or {}).get("op") or "<")
                            df = trend_ok_each_by_key.get((int(win), str(mt), str(op)))
                            if df is None or sig_d not in df.index:
                                trend_hit = False
                                break
                            if c not in df.columns or (not bool(df.loc[sig_d, c])):
                                trend_hit = False
                                break
                        hit_map["trend_rule"] = bool(trend_hit)

                    # 3) bias-based exit: condition hit => trigger candidate exit
                    if use_bias_exit:
                        bias_rules = _momentum_rules_for_stage(
                            str(c),
                            rules=inp.asset_bias_rules,
                            stage="exit",
                        ) if use_bias_rules else []
                        if not bias_rules:
                            bias_rules = [{
                                "bias_type": str(b_type),
                                "bias_ma_window": int(inp.bias_ma_window),
                                "level_window": str(inp.bias_level_window or "all"),
                                "threshold_type": str(inp.bias_threshold_type or "quantile"),
                                "quantile": float(inp.bias_quantile),
                                "fixed_value": float(inp.bias_fixed_value),
                                "min_periods": int(inp.bias_min_periods),
                                "op": ">",
                            }]
                        bias_hit = True
                        for r in bias_rules:
                            bty = str((r or {}).get("bias_type") or b_type).strip().lower()
                            if bty not in {"bias", "bias_v"}:
                                bty = str(b_type)
                            win = int((r or {}).get("bias_ma_window") or inp.bias_ma_window)
                            lvw = str((r or {}).get("level_window") or inp.bias_level_window or "all").strip().lower()
                            tt = str((r or {}).get("threshold_type") or inp.bias_threshold_type or "quantile").strip().lower()
                            qv = float((r or {}).get("quantile") if (r or {}).get("quantile") is not None else inp.bias_quantile)
                            fvv = float((r or {}).get("fixed_value") if (r or {}).get("fixed_value") is not None else inp.bias_fixed_value)
                            mp = int((r or {}).get("min_periods") if (r or {}).get("min_periods") is not None else inp.bias_min_periods)
                            op = _normalize_cmp_op((r or {}).get("op") or ">")
                            bdf = bias_by_key.get((int(win), str(bty)))
                            tdf = bias_thr_by_cfg.get((str(bty), int(win), str(lvw), str(tt), float(qv), float(fvv), int(mp)))
                            if bdf is None or tdf is None or sig_d not in bdf.index or sig_d not in tdf.index:
                                bias_hit = False
                                break
                            sig = None if pd.isna(bdf.loc[sig_d, c]) else float(bdf.loc[sig_d, c])
                            thr = None if pd.isna(tdf.loc[sig_d, c]) else float(tdf.loc[sig_d, c])
                            if sig is None or thr is None or (not np.isfinite(sig)) or (not np.isfinite(thr)):
                                bias_hit = False
                                break
                            if not _compare_with_op(float(sig), float(thr), op):
                                bias_hit = False
                                break
                        hit_map["bias_rule"] = bool(bias_hit)

                    hit_count = int(sum(1 for v in hit_map.values() if bool(v)))
                    hit_conditions = [k for k, v in hit_map.items() if bool(v)]
                    trigger_now = bool(int(exit_required) > 0 and hit_count >= int(exit_required) and bool(hit_conditions))
                    day_checks.append(
                        {
                            "code": str(c),
                            "decision_date": sig_d.date().isoformat(),
                            "execution_date": exec_d.date().isoformat(),
                            "from_weight": float(prev_wt),
                            "hit_count": int(hit_count),
                            "required": int(exit_required),
                            "hit_conditions": [str(x) for x in hit_conditions],
                            "triggered": bool(trigger_now),
                            "by_filter": {
                                "momentum_rule": (bool(hit_map.get("momentum_rule")) if bool(use_floor_rules) and ("momentum_rule" in hit_map) else None),
                                "trend_rule": (bool(hit_map.get("trend_rule")) if bool(use_trend_exit) and ("trend_rule" in hit_map) else None),
                                "bias_rule": (bool(hit_map.get("bias_rule")) if bool(use_bias_exit) and ("bias_rule" in hit_map) else None),
                            },
                        }
                    )
                    if int(exit_required) <= 0 or hit_count < int(exit_required):
                        continue
                    if not hit_conditions:
                        continue
                    primary_type = hit_conditions[0]
                    ev: dict[str, Any] = {
                        "type": str(primary_type),
                        "code": str(c),
                        "decision_date": sig_d.date().isoformat(),
                        "execution_date": exec_d.date().isoformat(),
                        "from_weight": float(prev_wt),
                        "to_weight": 0.0,
                        "delta_weight": float(-prev_wt),
                        "hit_count": int(hit_count),
                        "required": int(exit_required),
                        "hit_conditions": [str(x) for x in hit_conditions],
                    }
                    if momentum_score is not None and np.isfinite(float(momentum_score)):
                        ev["score"] = float(momentum_score)
                    w.loc[exec_d : dates[end_i], c] = 0.0
                    daily_exit_meta["events"].append(ev)
                    daily_exit_events.append(ev)
                    cur_risk_set.remove(c)
                if day_checks:
                    daily_exit_meta["checks_by_day"].append(
                        {
                            "decision_date": sig_d.date().isoformat(),
                            "execution_date": exec_d.date().isoformat(),
                            "checks": day_checks,
                        }
                    )
            if daily_exit_meta["events"]:
                daily_exit_meta["triggered"] = True

        # Carry forward for next rebalance decision (risk holdings only; stop-out means "cash" next decision).
        prev_picks_key = tuple(sorted([p for p in (risk_picks or []) if p in rank_codes])) if risk_picks else None
        prev_segment_stopped_out = bool(stop_triggered)

        # Update last-change marker for inertia min-hold.
        if inertia_enabled and (not dd_in_sleep):
            cur_key = tuple(sorted([c for c in rank_codes if float(prev_w_row.get(c, 0.0)) > 1e-12]))
            new_key = tuple(sorted([c for c in (risk_picks or []) if c in rank_codes]))
            if cur_key != new_key:
                last_change_decision_i = int(i)

        # Apply drawdown control (may further scale weights inside this segment and update sleep state).
        dd_meta = _apply_dd_control_segment(seg_start_i=start_i, seg_end_i=end_i)
        # propagate trigger info for convenience
        if dd_meta.get("triggered"):
            dd_in_sleep = True
            dd_meta["in_sleep"] = True

        holdings["periods"].append(
            {
                "decision_date": d.date().isoformat(),
                "rebalance_target_date": decision_target_date.get(int(di), d.date().isoformat()),
                "rebalance_hit_mode": decision_hit_mode.get(int(di), "period_end"),
                "start_date": dates[start_i].date().isoformat(),
                "end_date": dates[end_i].date().isoformat(),
                "picks": picks,
                "scores": {k: (None if pd.isna(scores.loc[d, k]) else float(scores.loc[d, k])) for k in picks},
                "best_score": meta.get("best_score"),
                "mode": meta.get("mode"),
                "exposure": float(exposure),
                "atr_stop": atr_stop,
                "corr_filter": corr_meta,
                "inertia": inertia_meta,
                "rr_sizing": rr_meta,
                "dd_control": dd_meta,
                "group_filter": group_meta,
                "backfill": {
                    "enabled": bool(inp.entry_backfill),
                    "used": bool(backfill_used),
                    "initial_risk_picks": [str(x) for x in backfill_initial],
                    "added": [str(x) for x in backfill_added],
                    "rejected": sorted([str(x) for x in entry_rejected_codes]),
                },
                "daily_exit": daily_exit_meta,
                "risk_controls": {"reasons": reasons, **details},
            }
        )

    # Corporate-action diagnostics (close-based):
    # We compute these earlier and also use the mask to apply HFQ fallback for NAV stability.
    ret_none = ret_none_close
    ret_hfq_all = ret_hfq_close

    # Daily holding return (hfq fallback-adjusted execution proxy).
    ret_exec = ret_exec_all

    # Apply per-asset risk-control rules (daily exposure scaling) on final weights.
    asset_rc_meta = _apply_asset_rc_rules(w, close_qfq=close_qfq, rules=inp.asset_rc_rules)
    # Apply per-asset vol-index timing (daily exposure scaling) on final weights.
    asset_vol_index_meta = _apply_asset_vol_index_rules(
        w,
        rules=inp.asset_vol_index_rules,
        vol_index_close=inp.vol_index_close,
    )
    # Continuous holding streaks (by actual daily weights, not rebalance schedule).
    holding_streaks = _holding_streaks_from_weights(w, codes=codes, eps=1e-12)

    # Benchmark: daily HFQ close equal-weight rebalance (no costs); RP uses same daily return matrix.
    bench_codes = universe[:]  # fixed benchmark universe
    n_b = len(bench_codes)
    if n_b <= 0:
        raise ValueError("benchmark universe empty")
    ch_bench = close_hfq_raw_align.reindex(columns=bench_codes, fill_value=np.nan)
    if not dynamic_u:
        ch_bench = ch_bench.ffill()
    w_eq = 1.0 / n_b
    ew_ret = hfq_close_daily_equal_weight_returns(ch_bench, dynamic_universe=dynamic_u).reindex(dates).fillna(0.0)
    ew_nav = (1.0 + ew_ret).cumprod()
    ew_nav.iloc[0] = 1.0

    # Risk parity (inverse-vol) benchmark with the same rebalance schedule as decision_dates.
    # We estimate vol using past HFQ close-to-close returns up to the decision date (inclusive).
    r_for_rp = ch_bench.astype(float).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    rp_window = int(max(20, int(inp.lookback_days or 20)))
    w_rp = pd.DataFrame(0.0, index=dates, columns=bench_codes, dtype=float)
    for i, d in enumerate(decision_dates):
        di = dates.get_loc(d)
        if di + 1 >= len(dates):
            break
        start_i = di + 1
        next_di = (dates.get_loc(decision_dates[i + 1]) if i + 1 < len(decision_dates) else (len(dates) - 1))
        end_i = min(len(dates) - 1, next_di)
        hist = r_for_rp.iloc[max(0, di - rp_window + 1) : di + 1].astype(float)
        vol = hist.std(ddof=1).replace(0.0, np.nan)
        inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
        s = float(np.nansum(inv.to_numpy(dtype=float)))
        if np.isfinite(s) and s > 0:
            wv = (inv / s).fillna(0.0).astype(float)
        else:
            wv = pd.Series(w_eq, index=bench_codes, dtype=float)
        w_rp.loc[dates[start_i] : dates[end_i], bench_codes] = wv.to_numpy(dtype=float)
    rp_ret = (w_rp * r_for_rp).sum(axis=1).astype(float)
    rp_nav = (1.0 + rp_ret).cumprod()
    rp_nav.iloc[0] = 1.0

    # Timed strategy returns/costs.
    port_ret = (w * ret_exec).sum(axis=1).astype(float)
    port_nav = (1.0 + port_ret).cumprod()
    port_nav.iloc[0] = 1.0
    w_prev = w.shift(1).fillna(0.0)
    turnover = (w - w_prev).abs().sum(axis=1) / 2.0
    cost = turnover * (inp.cost_bps / 10000.0)
    slippage = turnover * float(inp.slippage_rate)
    port_ret_net = (port_ret - cost - slippage).astype(float)
    port_nav_net = (1.0 + port_ret_net).cumprod()
    port_nav_net.iloc[0] = 1.0
    # Return decomposition (for explainability panel):
    # - close/oc2: split gross via close->next-open + next-open->next-close + interaction
    # - open: use the same split for open->open horizon, and on execution days
    #   (where we intentionally use same-day open->close), force overnight/interaction=0
    #   and intraday=gross to preserve exact day-level accounting.
    ret_overnight_none = (
        o_none.shift(-1).div(c_none) - 1.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    ret_intraday_none = (
        c_none.shift(-1).div(o_none.shift(-1)) - 1.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    ret_overnight_hfq = (
        o_hfq.shift(-1).div(c_hfq) - 1.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    ret_intraday_hfq = (
        c_hfq.shift(-1).div(o_hfq.shift(-1)) - 1.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    ret_overnight = ret_overnight_none.where(~corp_mask.fillna(False), other=ret_overnight_hfq).astype(float)
    ret_intraday = ret_intraday_none.where(~corp_mask.fillna(False), other=ret_intraday_hfq).astype(float)
    comp_overnight_close = (w * ret_overnight).sum(axis=1).astype(float)
    comp_intraday_close = (w * ret_intraday).sum(axis=1).astype(float)
    comp_interaction_close = (w * (ret_overnight * ret_intraday)).sum(axis=1).astype(float)
    if ep == "open":
        comp_overnight = (w * ret_overnight).sum(axis=1).astype(float)
        comp_intraday = (w * ret_intraday).sum(axis=1).astype(float)
        comp_interaction = (w * (ret_overnight * ret_intraday)).sum(axis=1).astype(float)
        if exec_day_indices:
            exec_dates = [dates[j] for j in exec_day_indices if 0 <= int(j) < len(dates)]
            if exec_dates:
                comp_overnight.loc[exec_dates] = 0.0
                comp_interaction.loc[exec_dates] = 0.0
                comp_intraday.loc[exec_dates] = port_ret.loc[exec_dates].astype(float)
    elif ep == "close":
        comp_overnight = comp_overnight_close
        comp_intraday = comp_intraday_close
        comp_interaction = comp_interaction_close
    else:
        comp_overnight = (0.5 * comp_overnight_close).astype(float)
        comp_intraday = (0.5 * port_ret + 0.5 * comp_intraday_close).astype(float)
        comp_interaction = (0.5 * comp_interaction_close).astype(float)
    decomp_cost = (cost + slippage).astype(float)
    decomp_gross = (comp_overnight + comp_intraday + comp_interaction).astype(float)
    decomp_net = (decomp_gross - decomp_cost).astype(float)
    asset_nav_exec = (1.0 + ret_exec[codes].astype(float).fillna(0.0)).cumprod()
    if not asset_nav_exec.empty:
        asset_nav_exec.iloc[0] = 1.0
    trade_pack = _trade_returns_from_weight_df(
        w.astype(float).fillna(0.0),
        ret_exec[codes].astype(float).fillna(0.0),
        cost_bps=float(inp.cost_bps),
        slippage_rate=float(inp.slippage_rate),
        dates=dates,
    )
    trade_stats = {
        "overall": _trade_stats_from_returns(trade_pack.get("returns", [])),
        "by_code": {
            str(c): _trade_stats_from_returns((trade_pack.get("returns_by_code") or {}).get(str(c), []))
            for c in codes
        },
        "trades": trade_pack.get("trades", []),
        "trades_by_code": trade_pack.get("trades_by_code", {}),
    }

    active_ret = port_ret_net - ew_ret
    excess_nav = (1.0 + active_ret).cumprod()
    excess_nav.iloc[0] = 1.0

    active_ret_rp = port_ret_net - rp_ret
    excess_nav_rp = (1.0 + active_ret_rp).cumprod()
    excess_nav_rp.iloc[0] = 1.0

    if bool(lightweight):
        ann_ret = _annualized_return(port_nav_net)
        mdd = _max_drawdown(port_nav_net)
        expo_mean = float(np.mean(w.sum(axis=1).to_numpy(dtype=float))) if (not w.empty) else 0.0
        out_lite: dict[str, Any] = {
            "metrics": {
                "strategy": {
                    "annualized_return": float(ann_ret),
                    "max_drawdown": float(mdd),
                    "cumulative_return": float(port_nav_net.iloc[-1] - 1.0) if len(port_nav_net) else float("nan"),
                }
            },
            "avg_exposure": expo_mean,
        }
        if return_weights_end:
            if not w.empty:
                last_w = w.iloc[-1].astype(float)
                out_lite["weights_end"] = {str(k): float(v) for k, v in last_w.items() if np.isfinite(float(v))}
            else:
                out_lite["weights_end"] = {}
        return out_lite

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
    calmar = float(ann_ret / abs(mdd)) if mdd < 0 else float("nan")
    sortino = _sortino(port_ret_net, rf=float(inp.risk_free_rate))
    ui = _ulcer_index(port_nav_net, in_percent=True)
    ui_den = ui / 100.0
    upi = float((ann_ret - float(inp.risk_free_rate)) / ui_den) if ui_den > 0 else float("nan")

    ann_excess = _annualized_return(excess_nav)
    ann_excess_vol = _annualized_vol(active_ret)
    ir = _sharpe(active_ret, rf=0.0)  # same formula but zero rf; for consistency name it IR-style
    ann_excess_arith = float(active_ret.mean() * TRADING_DAYS_PER_YEAR) if len(active_ret) else float("nan")
    ex_mdd = _max_drawdown(excess_nav)
    ex_mdd_dur = _max_drawdown_duration_days(excess_nav)

    ann_excess_rp = _annualized_return(excess_nav_rp)
    ann_excess_rp_vol = _annualized_vol(active_ret_rp)
    ir_rp = _sharpe(active_ret_rp, rf=0.0)
    ann_excess_rp_arith = float(active_ret_rp.mean() * TRADING_DAYS_PER_YEAR) if len(active_ret_rp) else float("nan")
    ex_rp_mdd = _max_drawdown(excess_nav_rp)
    ex_rp_mdd_dur = _max_drawdown_duration_days(excess_nav_rp)

    metrics = {
        "strategy": {
            "cumulative_return": float(port_nav_net.iloc[-1] - 1.0),
            "annualized_return": float(ann_ret),
            "annualized_volatility": float(ann_vol),
            "max_drawdown": float(mdd),
            "max_drawdown_recovery_days": int(mdd_dur),
            "sharpe_ratio": float(sharpe),
            "calmar_ratio": float(calmar),
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
            # annualized excess return (two complementary definitions)
            # - geo: CAGR on excess NAV (compound-consistent, recommended)
            # - arith: mean(active_ret)*252 (expected active return per year, not compound)
            "annualized_return": float(ann_excess),  # backward compatible (geo)
            "annualized_return_geo": float(ann_excess),
            "annualized_return_arith": float(ann_excess_arith),
            "annualized_volatility": float(ann_excess_vol),
            "information_ratio": float(ir),
            "max_drawdown": float(ex_mdd),
            "max_drawdown_recovery_days": int(ex_mdd_dur),
        },
        "risk_parity": {
            "cumulative_return": float(rp_nav.iloc[-1] - 1.0),
            "rp_window": int(rp_window),
        },
        "excess_vs_risk_parity": {
            "cumulative_return": float(excess_nav_rp.iloc[-1] - 1.0),
            "annualized_return": float(ann_excess_rp),
            "annualized_return_geo": float(ann_excess_rp),
            "annualized_return_arith": float(ann_excess_rp_arith),
            "annualized_volatility": float(ann_excess_rp_vol),
            "information_ratio": float(ir_rp),
            "max_drawdown": float(ex_rp_mdd),
            "max_drawdown_recovery_days": int(ex_rp_mdd_dur),
            "rp_window": int(rp_window),
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
        rp_s = float(rp_nav.loc[s])
        rp_e = float(rp_nav.loc[e])
        r_s = nav_e / nav_s - 1.0
        r_ew = ew_e / ew_s - 1.0
        r_rp = rp_e / rp_s - 1.0
        ex = float(r_s - r_ew)
        ex_rp = float(r_s - r_rp)
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
                "decision_date": p.get("decision_date"),
                "rebalance_target_date": p.get("rebalance_target_date"),
                "rebalance_hit_mode": p.get("rebalance_hit_mode"),
                "start_date": p["start_date"],
                "end_date": p["end_date"],
                "strategy_return": float(r_s),
                "equal_weight_return": float(r_ew),
                "risk_parity_return": float(r_rp),
                "excess_return": ex,
                "excess_return_rp": ex_rp,
                "win": ex > 0,
                "win_rp": ex_rp > 0,
                "buys": buys,
                "sells": sells,
                "turnover": period_turnover,
                "backfill_used": bool(((p.get("backfill") or {}).get("used"))),
            }
        )
    total_p = len(period_stats)
    win_rate = float(wins / total_p) if total_p else float("nan")
    avg_win = float(np.mean(pos)) if pos else float("nan")
    avg_loss = float(np.mean(neg)) if neg else float("nan")
    payoff = float(avg_win / abs(avg_loss)) if (pos and neg and avg_loss != 0) else float("nan")
    # geometric means for period returns (compound-friendly): exp(mean(log1p(r))) - 1
    def _geo_mean_return(rs: list[float]) -> float:
        if not rs:
            return float("nan")
        a = np.asarray(rs, dtype=float)
        m = np.isfinite(a) & (a > -1.0 + 1e-12)
        if not np.any(m):
            return float("nan")
        return float(np.exp(np.mean(np.log1p(a[m]))) - 1.0)

    avg_win_geo = _geo_mean_return(pos)
    avg_loss_geo = _geo_mean_return(neg)
    payoff_geo = float(avg_win_geo / abs(avg_loss_geo)) if (np.isfinite(avg_win_geo) and np.isfinite(avg_loss_geo) and avg_loss_geo != 0) else float("nan")
    # Kelly fraction (binary approximation): f* = p - (1-p)/b, where b is payoff ratio
    if total_p and np.isfinite(win_rate) and np.isfinite(payoff) and payoff > 0:
        kelly = float(win_rate - (1.0 - win_rate) / payoff)
    else:
        kelly = float("nan")

    abs_win_rate = float(abs_wins / total_p) if total_p else float("nan")
    abs_avg_win = float(np.mean(abs_pos)) if abs_pos else float("nan")
    abs_avg_loss = float(np.mean(abs_neg)) if abs_neg else float("nan")
    abs_payoff = float(abs_avg_win / abs(abs_avg_loss)) if (abs_pos and abs_neg and abs_avg_loss != 0) else float("nan")
    abs_avg_win_geo = _geo_mean_return(abs_pos)
    abs_avg_loss_geo = _geo_mean_return(abs_neg)
    abs_payoff_geo = float(abs_avg_win_geo / abs(abs_avg_loss_geo)) if (np.isfinite(abs_avg_win_geo) and np.isfinite(abs_avg_loss_geo) and abs_avg_loss_geo != 0) else float("nan")
    if total_p and np.isfinite(abs_win_rate) and np.isfinite(abs_payoff) and abs_payoff > 0:
        abs_kelly = float(abs_win_rate - (1.0 - abs_win_rate) / abs_payoff)
    else:
        abs_kelly = float("nan")
    # Robust variant-1: ignore near-zero period returns to reduce inactivity distortion.
    nz_eps = 1e-10
    abs_pos_nz = [x for x in abs_pos if np.isfinite(x) and abs(float(x)) > nz_eps]
    abs_neg_nz = [x for x in abs_neg if np.isfinite(x) and abs(float(x)) > nz_eps]
    total_p_nz = int(len(abs_pos_nz) + len(abs_neg_nz))
    abs_win_rate_nz = float(len(abs_pos_nz) / total_p_nz) if total_p_nz > 0 else float("nan")
    abs_avg_win_nz = float(np.mean(abs_pos_nz)) if abs_pos_nz else float("nan")
    abs_avg_loss_nz = float(np.mean(abs_neg_nz)) if abs_neg_nz else float("nan")
    abs_payoff_nz = float(abs_avg_win_nz / abs(abs_avg_loss_nz)) if (abs_pos_nz and abs_neg_nz and abs_avg_loss_nz != 0) else float("nan")
    if total_p_nz > 0 and np.isfinite(abs_win_rate_nz) and np.isfinite(abs_payoff_nz) and abs_payoff_nz > 0:
        abs_kelly_nonzero = float(abs_win_rate_nz - (1.0 - abs_win_rate_nz) / abs_payoff_nz)
    else:
        abs_kelly_nonzero = float("nan")
    # Robust variant-2: daily mean-variance Kelly proxy (single risky asset approximation).
    dret = pd.to_numeric(port_ret_net, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    d_mu = float(dret.mean()) if len(dret) else float("nan")
    d_var = float(dret.var(ddof=1)) if len(dret) >= 2 else float("nan")
    if np.isfinite(d_mu) and np.isfinite(d_var) and d_var > 0:
        daily_mv_kelly_proxy = float(d_mu / d_var)
    else:
        daily_mv_kelly_proxy = float("nan")

    stats = {
        "rebalance": inp.rebalance,
        "periods": total_p,
        # relative vs equal-weight (excess)
        "win_rate": win_rate,
        # arithmetic means (existing fields; kept for backward compatibility)
        "avg_win_excess": avg_win,
        "avg_loss_excess": avg_loss,
        "payoff_ratio": payoff,
        # geometric means (new)
        "avg_win_excess_geo": avg_win_geo,
        "avg_loss_excess_geo": avg_loss_geo,
        "payoff_ratio_geo": payoff_geo,
        "kelly_fraction": kelly,
        # absolute (strategy itself)
        "abs_win_rate": abs_win_rate,
        # arithmetic means (existing fields; kept for backward compatibility)
        "abs_avg_win": abs_avg_win,
        "abs_avg_loss": abs_avg_loss,
        "abs_payoff_ratio": abs_payoff,
        # geometric means (new)
        "abs_avg_win_geo": abs_avg_win_geo,
        "abs_avg_loss_geo": abs_avg_loss_geo,
        "abs_payoff_ratio_geo": abs_payoff_geo,
        "abs_kelly_fraction": abs_kelly,
        "abs_kelly_nonzero_fraction": abs_kelly_nonzero,
        "daily_mv_kelly_proxy": daily_mv_kelly_proxy,
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
    periodic_rp = {
        "weekly": _period_returns(port_nav_net, rp_nav, "W-FRI"),
        "monthly": _period_returns(port_nav_net, rp_nav, "ME"),
        "quarterly": _period_returns(port_nav_net, rp_nav, "QE"),
        "yearly": _period_returns(port_nav_net, rp_nav, "YE"),
    }

    # Rolling stats for strategy vs benchmark (defaults aligned with baseline UI)
    # NOTE: "drawdown" is rolling drawdown; "max_drawdown" is kept for backward-compat (deprecated).
    rolling = {"returns": {}, "drawdown": {}, "max_drawdown": {}}
    for weeks in (4, 12, 52):
        window = weeks * 5
        rolling["returns"][f"{weeks}w"] = (port_nav_net / port_nav_net.shift(window) - 1.0).dropna()
        rolling["drawdown"][f"{weeks}w"] = (port_nav_net / port_nav_net.rolling(window=window, min_periods=window).max() - 1.0).dropna()
        rolling["max_drawdown"][f"{weeks}w"] = _rolling_max_drawdown(port_nav_net, window).dropna()
    for months in (3, 6, 12):
        window = months * 21
        rolling["returns"][f"{months}m"] = (port_nav_net / port_nav_net.shift(window) - 1.0).dropna()
        rolling["drawdown"][f"{months}m"] = (port_nav_net / port_nav_net.rolling(window=window, min_periods=window).max() - 1.0).dropna()
        rolling["max_drawdown"][f"{months}m"] = _rolling_max_drawdown(port_nav_net, window).dropna()
    for years in (1, 3):
        window = years * 252
        rolling["returns"][f"{years}y"] = (port_nav_net / port_nav_net.shift(window) - 1.0).dropna()
        rolling["drawdown"][f"{years}y"] = (port_nav_net / port_nav_net.rolling(window=window, min_periods=window).max() - 1.0).dropna()
        rolling["max_drawdown"][f"{years}y"] = _rolling_max_drawdown(port_nav_net, window).dropna()
    rolling_out = {
        "returns": {k: {"dates": v.index.date.astype(str).tolist(), "values": v.astype(float).tolist()} for k, v in rolling["returns"].items()},
        "drawdown": {k: {"dates": v.index.date.astype(str).tolist(), "values": v.astype(float).tolist()} for k, v in rolling["drawdown"].items()},
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
        "score_method": (inp.score_method or "raw_mom"),
        "atr_stop_mode": atr_stop_mode,
        "atr_stop_atr_basis": atr_stop_atr_basis,
        "atr_stop_reentry_mode": atr_stop_reentry_mode,
        "codes": codes,
        "benchmark_codes": bench_codes,
        "price_basis": {
            "signal": "qfq",
            "strategy_nav": f"none execution({ep}) + hfq-implied corporate action factor (total return proxy)",
            "benchmark_nav": "hfq",
        },
        "exec_price": ep,
        "position_mode": pos_mode,
        "risk_budget_atr_window": int(risk_budget_atr_window),
        "risk_budget_pct": float(risk_budget_pct),
        "entry_backfill": bool(inp.entry_backfill),
        "entry_match_n": int(inp.entry_match_n or 0),
        "exit_match_n": int(inp.exit_match_n or 0),
        "asset_rc": asset_rc_meta,
        "asset_vol_index_timing": asset_vol_index_meta,
        "nav": {
            "dates": dates.date.astype(str).tolist(),
            "series": {
                "ROTATION": port_nav_net.astype(float).tolist(),
                "EW_REBAL": ew_nav.astype(float).tolist(),
                "RP_REBAL": rp_nav.astype(float).tolist(),
                "EXCESS": excess_nav.astype(float).tolist(),
                "EXCESS_RP": excess_nav_rp.astype(float).tolist(),
            },
        },
        "asset_nav_exec": {
            "dates": dates.date.astype(str).tolist(),
            "series": {str(c): asset_nav_exec[str(c)].astype(float).tolist() for c in codes if str(c) in asset_nav_exec.columns},
        },
        "nav_rsi": {
            "windows": [6, 12, 24],
            "dates": dates.date.astype(str).tolist(),
            "series": {
                "ROTATION": {},
                "EW_REBAL": {},
                "RP_REBAL": {},
            },
        },
        "attribution": attribution,
        "trade_statistics": trade_stats,
        "return_decomposition": {
            "dates": dates.date.astype(str).tolist(),
            "series": {
                "overnight": comp_overnight.astype(float).tolist(),
                "intraday": comp_intraday.astype(float).tolist(),
                "interaction": comp_interaction.astype(float).tolist(),
                "cost": decomp_cost.astype(float).tolist(),
                "gross": decomp_gross.astype(float).tolist(),
                "net": decomp_net.astype(float).tolist(),
            },
            "summary": {
                "ann_overnight": float(comp_overnight.iloc[1:].mean() * TRADING_DAYS_PER_YEAR) if len(comp_overnight) > 1 else 0.0,
                "ann_intraday": float(comp_intraday.iloc[1:].mean() * TRADING_DAYS_PER_YEAR) if len(comp_intraday) > 1 else 0.0,
                "ann_interaction": float(comp_interaction.iloc[1:].mean() * TRADING_DAYS_PER_YEAR) if len(comp_interaction) > 1 else 0.0,
                "ann_cost": float(decomp_cost.iloc[1:].mean() * TRADING_DAYS_PER_YEAR) if len(decomp_cost) > 1 else 0.0,
                "ann_gross": float(decomp_gross.iloc[1:].mean() * TRADING_DAYS_PER_YEAR) if len(decomp_gross) > 1 else 0.0,
                "ann_net": float(decomp_net.iloc[1:].mean() * TRADING_DAYS_PER_YEAR) if len(decomp_net) > 1 else 0.0,
            },
        },
        "metrics": metrics,
        "win_payoff": stats,
        "period_returns": periodic,
        "period_returns_rp": periodic_rp,
        "rolling": rolling_out,
        "period_details": period_stats,
        "holdings": holdings["periods"],
        "holding_streaks": holding_streaks,
        "daily_exit_events": daily_exit_events,
        "corporate_actions": corporate_actions,
    }
    if return_weights_end:
        try:
            d_last = dates[-1]
            out["weights_end"] = {
                "date": d_last.date().isoformat(),
                "weights": {str(c): float(w.loc[d_last, c]) for c in codes if c in w.columns},
            }
        except (KeyError, IndexError, TypeError, ValueError):  # pragma: no cover (defensive)
            out["weights_end"] = {"date": None, "weights": {}}
    # fill nav RSI series (avoid recomputing windows extraction twice)
    for w in out["nav_rsi"]["windows"]:
        out["nav_rsi"]["series"]["ROTATION"][str(w)] = _rsi_wilder(port_nav_net, window=int(w)).astype(float).tolist()
        out["nav_rsi"]["series"]["EW_REBAL"][str(w)] = _rsi_wilder(ew_nav, window=int(w)).astype(float).tolist()
        out["nav_rsi"]["series"]["RP_REBAL"][str(w)] = _rsi_wilder(rp_nav, window=int(w)).astype(float).tolist()
    return out

