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
    _rsi_wilder,
)
from ..analysis.baseline import load_close_prices as _load_close_prices
from ..analysis.baseline import load_high_low_prices as _load_high_low_prices
from ..analysis.baseline import load_ohlc_prices as _load_ohlc_prices
from ..analysis.baseline import _compute_return_risk_contributions as _compute_return_risk_contributions


@dataclass(frozen=True)
class RotationInputs:
    codes: list[str]
    start: dt.date
    end: dt.date
    rebalance: str = "weekly"  # daily/weekly/monthly/quarterly/yearly
    rebalance_weekday: int | None = None  # only used when rebalance=weekly; 0=Mon..4=Fri; default Fri when None
    rebalance_anchor: int | None = None  # weekly:0..4; monthly:1..28; quarterly/yearly:nth trading day (1..)
    rebalance_shift: str = "prev"  # prev|next when anchor falls on non-trading day (used with rebalance_anchor)
    top_k: int = 1
    lookback_days: int = 20
    skip_days: int = 0  # skip recent trading days (0 means no skip)
    risk_free_rate: float = 0.025
    cost_bps: float = 0.0  # round-trip cost in bps per turnover, simple approximation
    risk_off: bool = False
    defensive_code: str | None = None
    momentum_floor: float = 0.0  # if best score <= floor -> risk-off
    # --- Ranking method ---
    score_method: str = "raw_mom"  # raw_mom | sharpe_mom | sortino_mom | return_over_vol
    score_lambda: float = 0.0  # used by mom_minus_lambda_vol
    score_vol_power: float = 1.0  # used by mom_over_vol_power
    # --- Pre-trade risk controls (drawdown prevention heuristics) ---
    # Trend filter: decide whether to buy at all, and/or exclude candidates that are not in trend.
    trend_filter: bool = False
    trend_mode: str = "each"  # "each" | "universe"
    trend_sma_window: int = 20  # trading days (weekly use-case default)
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
    # --- Take-profit / stop-loss (qfq price basis) ---
    # Mode names are intentionally stringly-typed to keep API/UI flexible.
    # - none: disabled
    # - prev_week_low_stop: stop-loss based on last-week low (initial) and rolling update on rebalance
    tp_sl_mode: str = "none"
    atr_window: int | None = None  # None -> defaults to lookback_days
    atr_mult: float = 2.0
    atr_step: float = 0.5
    atr_min_mult: float = 0.5
    # --- Correlation filter (hfq price basis) ---
    corr_filter: bool = False
    corr_window: int | None = None  # None -> defaults to lookback_days
    corr_threshold: float = 0.5
    # --- Inertia / dampening (avoid frequent rebalances) ---
    inertia: bool = False
    inertia_min_hold_periods: int = 0  # minimum decision periods between holding changes (0 disables)
    inertia_score_gap: float = 0.0  # only for top_k=1: require new_score - cur_score >= gap to switch
    inertia_min_turnover: float = 0.0  # if expected turnover < threshold, skip rebalance (0 disables)
    # --- Rolling-return based position sizing (strategy trailing return) ---
    rr_sizing: bool = False
    rr_years: float = 3.0
    rr_thresholds: list[float] | None = None  # max 5
    rr_weights: list[float] | None = None  # len = len(thresholds)+1
    # --- Drawdown control (strategy NAV) ---
    dd_control: bool = False
    dd_threshold: float = 0.10  # decimal, e.g. 0.10 = 10%
    dd_reduce: float = 1.0  # fraction to reduce, e.g. 1.0 => reduce 100% -> cash
    dd_sleep_days: int = 20  # trading days
    # --- Timing control (strategy NAV RSI gate; uses shadow NAV that ignores this timing gate) ---
    timing_rsi_gate: bool = False
    timing_rsi_window: int = 24  # typical choices: 6/12/24; default=24
    # --- Execution price proxy (hfq, aligned to execution calendar) ---
    # Used to study open/close/OC(=avg(open,close)) calendar effects. Default "close" matches existing behavior.
    exec_price: str = "close"  # close | open | oc2


def _rebalance_labels(index: pd.DatetimeIndex, rebalance: str, *, weekly_anchor: str = "FRI") -> pd.PeriodIndex:
    r = (rebalance or "monthly").lower()
    anchor = str(weekly_anchor).strip().upper()
    if anchor not in {"MON", "TUE", "WED", "THU", "FRI"}:
        raise ValueError(f"invalid weekly_anchor={weekly_anchor} (expected MON..FRI)")
    freq_map = {"daily": "D", "weekly": f"W-{anchor}", "monthly": "M", "quarterly": "Q", "yearly": "Y"}
    if r not in freq_map:
        raise ValueError(f"invalid rebalance={rebalance}")
    return index.to_period(freq_map[r])


def _momentum_scores(close_hfq: pd.DataFrame, *, lookback_days: int, skip_days: int) -> pd.DataFrame:
    # score[t] = close[t-skip]/close[t-skip-lookback] - 1
    lag = skip_days
    lb = lookback_days
    return close_hfq.shift(lag) / close_hfq.shift(lag + lb) - 1.0


def _rolling_prod_minus_1(gross: pd.DataFrame, *, window: int) -> pd.DataFrame:
    w = max(1, int(window))
    # rolling product is not built-in for DataFrame; use apply on ndarray for speed-enough.
    return gross.rolling(window=w, min_periods=max(2, w // 2)).apply(lambda x: float(np.prod(x)) - 1.0, raw=True)


def _risk_adjusted_scores(
    close_hfq: pd.DataFrame,
    *,
    lookback_days: int,
    skip_days: int,
    method: str,
    rf_annual: float,
) -> pd.DataFrame:
    """
    Compute alternative momentum scores for ranking:
    - return_over_vol: cumulative return / realized vol over same window
    - sharpe_mom: Sharpe ratio over lookback window
    - sortino_mom: Sortino ratio over lookback window

    All are computed on hfq daily close-to-close returns, with window ending at (t - skip_days).
    """
    m = (method or "raw_mom").strip().lower()
    if m not in {"return_over_vol", "sharpe_mom", "sortino_mom", "mom_minus_lambda_vol", "mom_over_vol_power"}:
        raise ValueError(f"invalid score_method={method}")

    lb = max(1, int(lookback_days))
    lag = max(0, int(skip_days))
    rf_daily = float(rf_annual) / 252.0

    ret = close_hfq.pct_change().replace([np.inf, -np.inf], np.nan)
    # Align the window to end at (t - lag): shift returns forward by lag.
    ret = ret.shift(lag)

    # window stats
    mean = ret.rolling(window=lb, min_periods=max(3, lb // 2)).mean()
    std = ret.rolling(window=lb, min_periods=max(3, lb // 2)).std(ddof=1)
    ann_vol = std * np.sqrt(252.0)

    # cumulative return over the window
    gross = (1.0 + ret).fillna(1.0)
    cum_ret = _rolling_prod_minus_1(gross, window=lb)

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

    if m == "return_over_vol":
        return safe_div(cum_ret.astype(float), ann_vol.astype(float))

    if m == "mom_minus_lambda_vol":
        # Placeholder; actual lambda is applied in backtest_rotation for consistency with input validation.
        return cum_ret.astype(float)  # will be adjusted by caller

    if m == "mom_over_vol_power":
        return cum_ret.astype(float)  # will be adjusted by caller

    # sharpe/sortino use excess mean over rf
    excess_mean = (mean - rf_daily).astype(float)
    if m == "sharpe_mom":
        return safe_div(excess_mean, std.astype(float))

    # sortino: downside deviation on (ret - rf_daily)
    downside = (ret - rf_daily).clip(upper=0.0)
    dd = downside.rolling(window=lb, min_periods=max(3, lb // 2)).std(ddof=1)
    return safe_div(excess_mean, dd.astype(float))


def _trend_ok_each(close: pd.DataFrame, *, sma_window: int) -> pd.DataFrame:
    """
    Simple trend filter: close > SMA(close, window).
    Returns boolean DataFrame aligned to `close`.
    """
    w = max(1, int(sma_window))
    sma = close.rolling(window=w, min_periods=max(2, w // 2)).mean()
    return (close > sma).fillna(False)


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
    sm = (inp.score_method or "raw_mom").strip().lower()
    if sm not in {
        "raw_mom",
        "sharpe_mom",
        "sortino_mom",
        "return_over_vol",
        "mom_minus_lambda_vol",
        "mom_over_vol_power",
    }:
        raise ValueError(f"invalid score_method={inp.score_method}")
    if not np.isfinite(float(inp.score_lambda)):
        raise ValueError("score_lambda must be finite")
    if not np.isfinite(float(inp.score_vol_power)):
        raise ValueError("score_vol_power must be finite")
    if inp.trend_sma_window <= 0:
        raise ValueError("trend_sma_window must be > 0")
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
    if inp.trend_mode not in {"each", "universe"}:
        raise ValueError(f"invalid trend_mode={inp.trend_mode}")
    tp_sl_mode = (inp.tp_sl_mode or "none").strip().lower()
    if tp_sl_mode not in {"none", "prev_week_low_stop", "atr_chandelier_fixed", "atr_chandelier_progressive"}:
        raise ValueError(f"invalid tp_sl_mode={inp.tp_sl_mode}")
    if inp.atr_window is not None and int(inp.atr_window) < 2:
        raise ValueError("atr_window must be >= 2")
    if not np.isfinite(float(inp.atr_mult)) or float(inp.atr_mult) <= 0:
        raise ValueError("atr_mult must be finite and > 0")
    if not np.isfinite(float(inp.atr_step)) or float(inp.atr_step) <= 0:
        raise ValueError("atr_step must be finite and > 0")
    if not np.isfinite(float(inp.atr_min_mult)) or float(inp.atr_min_mult) <= 0:
        raise ValueError("atr_min_mult must be finite and > 0")
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
    if int(inp.timing_rsi_window) < 2:
        raise ValueError("timing_rsi_window must be >= 2")
    reb_shift = (inp.rebalance_shift or "next").strip().lower()
    if reb_shift not in {"prev", "next"}:
        raise ValueError("rebalance_shift must be one of: prev|next")
    ep = (inp.exec_price or "close").strip().lower()
    if ep not in {"close", "open", "oc2"}:
        raise ValueError("exec_price must be one of: close|open|oc2")
    if inp.rebalance_weekday is not None:
        if int(inp.rebalance_weekday) < 0 or int(inp.rebalance_weekday) > 4:
            raise ValueError("rebalance_weekday must be within [0..4] (Mon..Fri)")
    if inp.rebalance_anchor is not None:
        if int(inp.rebalance_anchor) < 0:
            raise ValueError("rebalance_anchor must be >= 0")

    codes = universe[:]  # may include defensive later for strategy holdings
    rank_codes = universe[:]  # ranking / filters apply to the original universe only
    defensive = (inp.defensive_code or "").strip() or None
    if inp.risk_off and defensive:
        if defensive not in codes:
            codes = codes + [defensive]

    # Load:
    # - hfq: momentum score + momentum_floor (as requested)
    # - qfq: technical analysis (trend/RSI/vol/chop filters)
    # - none: execution/trading price basis
    # Need enough history for momentum + optional risk controls.
    need_hist = inp.lookback_days + inp.skip_days + 60
    if inp.trend_filter:
        need_hist = max(need_hist, int(inp.trend_sma_window) + 60)
    if inp.rsi_filter:
        need_hist = max(need_hist, int(inp.rsi_window) + 60)
    if inp.vol_monitor:
        need_hist = max(need_hist, int(inp.vol_window) + 60)
    if inp.chop_filter:
        need_hist = max(need_hist, (int(inp.chop_adx_window) if cm == "adx" else int(inp.chop_window)) + 60)
    ext_start = inp.start - dt.timedelta(days=int(need_hist))
    close_hfq = _load_close_prices(db, codes=codes, start=ext_start, end=inp.end, adjust="hfq")
    # Only load qfq when we actually need technical-analysis indicators.
    need_qfq = bool(inp.trend_filter or inp.rsi_filter or inp.vol_monitor or inp.chop_filter or tp_sl_mode != "none")
    close_qfq = _load_close_prices(db, codes=codes, start=ext_start, end=inp.end, adjust="qfq") if need_qfq else pd.DataFrame()
    need_qfq_hl = bool(inp.chop_filter and cm == "adx")
    high_qfq, low_qfq = (
        _load_high_low_prices(db, codes=codes, start=ext_start, end=inp.end, adjust="qfq") if need_qfq_hl else (pd.DataFrame(), pd.DataFrame())
    )
    close_none = _load_close_prices(db, codes=codes, start=inp.start, end=inp.end, adjust="none")
    if close_none.empty:
        raise ValueError("no execution price data for given range (none)")

    # Execution return basis:
    # - Strategy NAV uses NONE prices (tradeable) by default, with HFQ fallback on corporate-action cliff days.
    # - For plotting/benchmark comparisons we still compute HFQ series.
    # We only load OHLC when needed (open/oc2).
    need_hfq_ohlc = ep in {"open", "oc2"}
    need_none_ohlc = ep in {"open", "oc2"}
    ohlc_hfq = (
        _load_ohlc_prices(db, codes=codes, start=ext_start, end=inp.end, adjust="hfq") if need_hfq_ohlc else {"open": pd.DataFrame(), "high": pd.DataFrame(), "low": pd.DataFrame(), "close": pd.DataFrame()}
    )
    ohlc_none = (
        _load_ohlc_prices(db, codes=codes, start=inp.start, end=inp.end, adjust="none") if need_none_ohlc else {"open": pd.DataFrame(), "high": pd.DataFrame(), "low": pd.DataFrame(), "close": pd.DataFrame()}
    )

    # Align calendars using execution dates; forward-fill hfq onto those dates.
    close_none = close_none.sort_index().ffill()
    dates = close_none.index
    close_hfq = close_hfq.sort_index().reindex(dates).ffill()
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

    # Require each selected code has data
    miss_exec = [c for c in codes if c not in close_none.columns or close_none[c].dropna().empty]
    if miss_exec:
        raise ValueError(f"missing execution data (none) for: {miss_exec}")
    miss_sig = [c for c in codes if c not in close_hfq.columns or close_hfq[c].dropna().empty]
    if miss_sig:
        raise ValueError(f"missing signal data (hfq) for: {miss_sig}")
    # qfq is required for technical analysis features
    if need_qfq:
        miss_ta = [c for c in codes if c not in close_qfq.columns or close_qfq[c].dropna().empty]
        if miss_ta:
            raise ValueError(f"missing technical-analysis data (qfq) for: {miss_ta}")
    if need_qfq_hl:
        miss_hi = [c for c in codes if c not in high_qfq.columns or high_qfq[c].dropna().empty]
        miss_lo = [c for c in codes if c not in low_qfq.columns or low_qfq[c].dropna().empty]
        miss_hl = sorted(set(miss_hi + miss_lo))
        if miss_hl:
            raise ValueError(f"missing technical-analysis high/low data (qfq) for: {miss_hl}")

    if sm == "raw_mom":
        scores = _momentum_scores(close_hfq[rank_codes], lookback_days=inp.lookback_days, skip_days=inp.skip_days)
    else:
        base_scores = _risk_adjusted_scores(
            close_hfq[rank_codes],
            lookback_days=inp.lookback_days,
            skip_days=inp.skip_days,
            method=inp.score_method,
            rf_annual=float(inp.risk_free_rate),
        )
        if sm == "mom_minus_lambda_vol":
            # mom is base cumulative return; subtract lambda * annualized vol over the same window
            lb = max(2, int(inp.lookback_days))
            lag = max(0, int(inp.skip_days))
            ret = close_hfq[rank_codes].pct_change().replace([np.inf, -np.inf], np.nan).shift(lag)
            vol = ret.rolling(window=lb, min_periods=max(3, lb // 2)).std(ddof=1) * np.sqrt(252.0)
            scores = base_scores - float(inp.score_lambda) * vol.astype(float)
        elif sm == "mom_over_vol_power":
            lb = max(2, int(inp.lookback_days))
            lag = max(0, int(inp.skip_days))
            ret = close_hfq[rank_codes].pct_change().replace([np.inf, -np.inf], np.nan).shift(lag)
            vol = ret.rolling(window=lb, min_periods=max(3, lb // 2)).std(ddof=1) * np.sqrt(252.0)
            denom = vol.astype(float).pow(float(inp.score_vol_power))
            scores = base_scores / denom.replace(0.0, np.nan)
            # keep deterministic ranking for zero-vol series
            scores = scores.replace([np.inf, -np.inf], np.nan)
        else:
            scores = base_scores

    # Pre-compute risk-control signals on hfq close (aligned to execution calendar).
    # IMPORTANT: per your rule, all TA uses qfq; only momentum scoring & momentum floor uses hfq.
    ta_close = close_qfq[rank_codes] if need_qfq else None
    trend_ok_each = _trend_ok_each(ta_close, sma_window=int(inp.trend_sma_window)) if (inp.trend_filter and ta_close is not None) else None
    rsi = _rsi(ta_close, window=int(inp.rsi_window)) if (inp.rsi_filter and ta_close is not None) else None
    ann_vol = _ann_realized_vol_from_close(ta_close, window=int(inp.vol_window)) if (inp.vol_monitor and ta_close is not None) else None
    er = _efficiency_ratio(ta_close, window=int(inp.chop_window)) if (inp.chop_filter and cm == "er" and ta_close is not None) else None
    adx = _adx(high_qfq[rank_codes], low_qfq[rank_codes], ta_close, window=int(inp.chop_adx_window)) if (inp.chop_filter and cm == "adx" and ta_close is not None) else None
    # Universe-level trend uses average price series across the selected universe.
    if inp.trend_filter and inp.trend_mode == "universe":
        if ta_close is None:  # pragma: no cover
            raise ValueError("trend_filter requires qfq data")
        uni_close = ta_close.mean(axis=1).to_frame("UNIVERSE")
        uni_ok = _trend_ok_each(uni_close, sma_window=int(inp.trend_sma_window))["UNIVERSE"]
    else:
        uni_ok = None

    def _decision_indices_for_rebalance(*, rebalance: str, anchor: int | None) -> list[int]:
        """
        Decision dates are where we compute picks; holdings apply from the next trading day.

        anchor semantics:
        - weekly: 0=Mon..4=Fri (week ending on that weekday; choose last trading day in that weekly period)
        - monthly: day-of-month 1..28 (choose first trading day with day>=anchor in month; fallback month-end)
        - quarterly/yearly: Nth trading day in period (1-indexed; fallback period-end)
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
                return pd.Series(np.arange(len(dates)), index=dates).groupby(labels_local).max().to_list()
            else:
                wd_map_local = {0: "MON", 1: "TUE", 2: "WED", 3: "THU", 4: "FRI"}
                if int(anchor) not in wd_map_local:
                    raise ValueError("weekly rebalance_anchor must be within [0..4] (Mon..Fri)")
                labels_local = _rebalance_labels(dates, r, weekly_anchor=wd_map_local[int(anchor)])
            # If anchor calendar day is non-trading, choose prev/next trading day per rebalance_shift.
            def _shift_idx(target: pd.Timestamp) -> int:
                t = pd.to_datetime(target).normalize()
                if t in dates:
                    return int(dates.get_loc(t))
                pos = int(dates.searchsorted(t))
                if reb_shift == "next":
                    return int(min(pos, len(dates) - 1))
                return int(max(pos - 1, 0))

            out: list[int] = []
            seen: set[int] = set()
            for p in pd.unique(labels_local):
                target = pd.Timestamp(p.end_time).normalize()
                i = _shift_idx(target)
                if i not in seen:
                    out.append(i)
                    seen.add(i)
            # Ensure chronological order; pd.unique(periods) order is not guaranteed.
            return sorted(out)

        if r == "monthly":
            if anchor is None:
                labels_local = _rebalance_labels(dates, r, weekly_anchor="FRI")
                return pd.Series(np.arange(len(dates)), index=dates).groupby(labels_local).max().to_list()
            dom = int(anchor)
            if dom < 1 or dom > 28:
                raise ValueError("monthly rebalance_anchor must be within [1..28] (day-of-month)")
            labels_local = dates.to_period("M")

            def _shift_idx(target: pd.Timestamp) -> int:
                t = pd.to_datetime(target).normalize()
                if t in dates:
                    return int(dates.get_loc(t))
                pos = int(dates.searchsorted(t))
                if reb_shift == "next":
                    return int(min(pos, len(dates) - 1))
                return int(max(pos - 1, 0))

            out: list[int] = []
            seen: set[int] = set()
            for p in pd.unique(labels_local):
                target = pd.Timestamp(dt.date(int(p.year), int(p.month), dom))
                i = _shift_idx(target)
                if i not in seen:
                    out.append(i)
                    seen.add(i)
            return sorted(out)

        # quarterly/yearly
        if anchor is None:
            labels_local = _rebalance_labels(dates, r, weekly_anchor="FRI")
            return pd.Series(np.arange(len(dates)), index=dates).groupby(labels_local).max().to_list()
        n = int(anchor)
        if n < 1:
            raise ValueError("quarterly/yearly rebalance_anchor must be >= 1 (Nth trading day)")
        labels_local = dates.to_period("Q" if r == "quarterly" else "Y")
        out = []
        for _, pos in pd.Series(np.arange(len(dates)), index=dates).groupby(labels_local):
            arr = pos.to_numpy(dtype=int)
            k = min(n - 1, len(arr) - 1)
            out.append(int(arr[int(k)]))
        return out

    # Determine rebalance decision dates.
    # If we rebalance at close on decision_date, then returns on the NEXT trading day onward
    # should reflect the new holdings. Therefore the holdings from one decision apply through
    # the NEXT decision date (inclusive), to avoid "gaps" on decision dates.
    anchor_val = inp.rebalance_anchor
    if anchor_val is None and inp.rebalance_weekday is not None and (inp.rebalance or "weekly").lower() == "weekly":
        anchor_val = int(inp.rebalance_weekday)
    last_idx = _decision_indices_for_rebalance(rebalance=inp.rebalance, anchor=anchor_val)
    decision_dates = dates[last_idx]

    # Period labels are used by some features (e.g. prev_week_low_stop) to aggregate within a rebalance period.
    # Keep behavior backward-compatible:
    # - weekly: use the same weekly anchor as the decision schedule (default FRI)
    # - monthly/quarterly/yearly: natural calendar periods
    if (inp.rebalance or "weekly").lower() == "weekly":
        wd_map = {0: "MON", 1: "TUE", 2: "WED", 3: "THU", 4: "FRI"}
        w_anchor = wd_map.get(int(anchor_val), "FRI") if anchor_val is not None else "FRI"
        labels = _rebalance_labels(dates, inp.rebalance, weekly_anchor=w_anchor)
    else:
        labels = _rebalance_labels(dates, inp.rebalance, weekly_anchor="FRI")

    # Precompute execution returns for strategy NAV:
    # - prefer NONE (tradeable) prices
    # - if corporate-action cliff detected, use HFQ return on that day to avoid artificial NAV jump
    def _has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
        return (df is not None) and (not df.empty) and all((c in df.columns) for c in cols)

    # HFQ exec proxy (used for benchmark + corp-action fallback returns).
    if ep == "close":
        px_exec_hfq = close_hfq[codes]
    elif ep == "open":
        px_exec_hfq = ohlc_hfq.get("open", pd.DataFrame())
        if not _has_cols(px_exec_hfq, codes):
            px_exec_hfq = close_hfq[codes]
        else:
            px_exec_hfq = px_exec_hfq[codes]
    else:
        o = ohlc_hfq.get("open", pd.DataFrame())
        c = ohlc_hfq.get("close", pd.DataFrame())
        if not (_has_cols(o, codes) and _has_cols(c, codes)):
            px_exec_hfq = close_hfq[codes]
        else:
            px_exec_hfq = (o[codes].astype(float) + c[codes].astype(float)) / 2.0
    px_exec_hfq = px_exec_hfq.astype(float).replace([np.inf, -np.inf], np.nan).ffill()

    # NONE exec proxy (strategy tradeable prices).
    if ep == "close":
        px_exec_none = close_none[codes]
    elif ep == "open":
        px_exec_none = ohlc_none.get("open", pd.DataFrame())
        if not _has_cols(px_exec_none, codes):
            px_exec_none = close_none[codes]
        else:
            px_exec_none = px_exec_none[codes]
    else:
        o = ohlc_none.get("open", pd.DataFrame())
        c = ohlc_none.get("close", pd.DataFrame())
        if not (_has_cols(o, codes) and _has_cols(c, codes)):
            px_exec_none = close_none[codes]
        else:
            px_exec_none = (o[codes].astype(float) + c[codes].astype(float)) / 2.0
    px_exec_none = px_exec_none.astype(float).replace([np.inf, -np.inf], np.nan).ffill()

    ret_exec_none = px_exec_none.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    ret_exec_hfq = px_exec_hfq.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    # Corporate action factor (gross): (1+hfq_ret)/(1+none_ret) on CLOSE series.
    # Use this to identify cliff days, then swap that day's execution return to hfq for NAV stability.
    ret_none_close = close_none[codes].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    ret_hfq_close = close_hfq[codes].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    gross_none_close = (1.0 + ret_none_close).astype(float)
    gross_hfq_close = (1.0 + ret_hfq_close).astype(float)
    corp_factor = (gross_hfq_close / gross_none_close).replace([np.inf, -np.inf], np.nan)
    dev = (corp_factor - 1.0).abs()
    corp_mask = (dev > 0.02) | (corp_factor > 1.2) | (corp_factor < 1.0 / 1.2)

    # Final execution returns for NAV: none preferred, hfq fallback on cliff days.
    ret_exec_all = ret_exec_none.copy()
    for c in codes:
        if c in ret_exec_all.columns and c in ret_exec_hfq.columns and c in corp_mask.columns:
            m = corp_mask[c].fillna(False)
            if bool(m.any()):
                ret_exec_all.loc[m, c] = ret_exec_hfq.loc[m, c]
    ret_exec_all = ret_exec_all.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    # Build weights per date (apply from next trading day after decision date).
    w = pd.DataFrame(0.0, index=dates, columns=codes)
    holdings: dict[str, list[dict[str, Any]]] = {"periods": []}
    # Stop-loss carry: track prior picks and whether a stop-out occurred in the prior holding segment.
    prev_picks_key: tuple[str, ...] | None = None
    prev_segment_stopped_out: bool = False
    period_min_close_qfq = (
        close_qfq[rank_codes].astype(float).groupby(labels).min() if (tp_sl_mode == "prev_week_low_stop" and need_qfq) else pd.DataFrame()
    )
    # ATR from qfq close only (close-to-close absolute range); aligned to execution calendar.
    # Note: classic ATR uses high/low/prev close; this is a close-only approximation per spec.
    if tp_sl_mode in {"atr_chandelier_fixed", "atr_chandelier_progressive"}:
        w_atr = int(inp.atr_window) if inp.atr_window is not None else int(inp.lookback_days)
        w_atr = max(2, w_atr)
        close_for_atr = close_qfq[rank_codes].astype(float)
        atr = close_for_atr.diff().abs().rolling(window=w_atr, min_periods=max(2, w_atr // 2)).mean()
    else:
        w_atr = None
        atr = pd.DataFrame()

    # Correlation filter params (hfq).
    corr_enabled = bool(inp.corr_filter)
    corr_window = int(inp.corr_window) if inp.corr_window is not None else int(inp.lookback_days)
    corr_window = max(2, corr_window)
    corr_threshold = float(inp.corr_threshold)

    inertia_enabled = bool(inp.inertia)
    inertia_min_hold = int(max(0, int(inp.inertia_min_hold_periods)))
    inertia_score_gap = float(inp.inertia_score_gap)
    inertia_min_turnover = float(inp.inertia_min_turnover)
    last_change_decision_i = -10**9  # decision index when holdings last changed (for min-hold)

    def _pair_corr_hfq(*, code_a: str, code_b: str, end_pos: int) -> float | None:
        """
        Pearson corr of daily returns (pct_change) for hfq close over a lookback window ending at end_pos.
        end_pos is an integer index into `dates` (aligned calendar).
        """
        if code_a == code_b:
            return 1.0
        if code_a not in close_hfq.columns or code_b not in close_hfq.columns:
            return None
        start_pos = max(0, int(end_pos) - int(corr_window))
        # Need at least 3 return observations => at least 4 prices.
        if end_pos - start_pos < 3:
            return None
        pa = close_hfq[code_a].iloc[start_pos : end_pos + 1].astype(float)
        pb = close_hfq[code_b].iloc[start_pos : end_pos + 1].astype(float)
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
        nav = float(nav_running.iloc[processed_idx])
        # Use raw arrays to avoid any index-alignment surprises.
        xnet = port_ret.to_numpy(dtype=float) - cost_np.astype(float)
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

    # Drawdown control (strategy NAV drawdown, net of turnover cost).
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
            dd_nav *= (1.0 + float(port_ret) - float(cost))
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

        # Sleep branch: keep previous day's weights; skip new decisions.
        if dd_in_sleep:
            prev_w_row = w.iloc[start_i - 1].astype(float)
            w.iloc[start_i : end_i + 1, :] = prev_w_row.to_numpy(dtype=float)
            held = [c for c in codes if float(prev_w_row.get(c, 0.0)) > 0.0]
            picks = [c for c in held if c in codes]
            meta = {"best_score": None, "risk_off_triggered": True, "mode": "dd_sleep"}
        else:
            prev_w_row = w.iloc[start_i - 1].astype(float)
            picks, meta = _pick_assets(
                scores.loc[d],
                top_k=inp.top_k,
                risk_off=inp.risk_off,
                defensive_code=defensive,
                floor=inp.momentum_floor,
            )
        # Defensive/cash branch from momentum floor:
        # - picks == [defensive] => invest 100% in defensive (if provided)
        # - picks == [] => cash
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

        # Take-profit / stop-loss: initialized here, but stop levels are computed after the final picks
        # are stabilized (risk controls / corr / inertia). This avoids mismatches when picks are modified.
        tp_sl: dict[str, Any] = {"mode": tp_sl_mode}
        stop_levels: dict[str, float] = {}
        stop_trigger_date: str | None = None
        stop_triggered = False

        # Apply pre-trade risk controls only when we are in risk-on mode (holding risk assets).
        if risk_picks and (not dd_in_sleep):
            # 0) Choppiness filter (ER / ADX)
            if inp.chop_filter:
                if cm == "er" and er is not None and d in er.index:
                    er_map = {p: (None if pd.isna(er.loc[d, p]) else float(er.loc[d, p])) for p in risk_picks}
                    details["er"] = er_map
                    before = risk_picks[:]
                    thr = float(inp.chop_er_threshold)
                    risk_picks = [p for p in risk_picks if (er_map.get(p) is not None) and (er_map[p] >= thr)]
                    removed = [p for p in before if p not in risk_picks]
                    if removed:
                        reasons.append(f"chop_er_exclude<{thr}:{','.join(removed)}")
                        picks = [p for p in picks if (p not in rank_codes) or (p in risk_picks)]
                if cm == "adx" and adx is not None and d in adx.index:
                    adx_map = {p: (None if pd.isna(adx.loc[d, p]) else float(adx.loc[d, p])) for p in risk_picks}
                    details["adx"] = adx_map
                    before = risk_picks[:]
                    thr = float(inp.chop_adx_threshold)
                    risk_picks = [p for p in risk_picks if (adx_map.get(p) is not None) and (adx_map[p] >= thr)]
                    removed = [p for p in before if p not in risk_picks]
                    if removed:
                        reasons.append(f"chop_adx_exclude<{thr}:{','.join(removed)}")
                        picks = [p for p in picks if (p not in rank_codes) or (p in risk_picks)]

            # 1) Trend filter
            if inp.trend_filter:
                if inp.trend_mode == "universe":
                    ok = bool(uni_ok.loc[d]) if (uni_ok is not None and d in uni_ok.index) else False
                    details["trend_universe_ok"] = ok
                    if not ok:
                        reasons.append("trend_universe_block")
                        if inp.risk_off and defensive:
                            picks = [defensive]
                            risk_picks = []
                            meta = {"best_score": meta.get("best_score"), "risk_off_triggered": True, "mode": "defensive"}
                        else:
                            picks = []
                            risk_picks = []
                            meta = {"best_score": meta.get("best_score"), "risk_off_triggered": True, "mode": "cash"}
                else:
                    ok_map = {}
                    if trend_ok_each is not None and d in trend_ok_each.index:
                        for p in risk_picks:
                            ok_map[p] = bool(trend_ok_each.loc[d, p])
                    details["trend_each_ok"] = ok_map
                    before = risk_picks[:]
                    risk_picks = [p for p in risk_picks if ok_map.get(p, False)]
                    removed = [p for p in before if p not in risk_picks]
                    if removed:
                        reasons.append(f"trend_each_exclude:{','.join(removed)}")
                        picks = [p for p in picks if (p not in rank_codes) or (p in risk_picks)]

            # 2) RSI filter
            if risk_picks and inp.rsi_filter and rsi is not None and d in rsi.index:
                rsi_map = {p: (None if pd.isna(rsi.loc[d, p]) else float(rsi.loc[d, p])) for p in risk_picks}
                details["rsi"] = rsi_map
                before = risk_picks[:]
                if inp.rsi_block_overbought:
                    risk_picks = [p for p in risk_picks if (rsi_map.get(p) is None) or (rsi_map[p] <= float(inp.rsi_overbought))]
                if inp.rsi_block_oversold:
                    risk_picks = [p for p in risk_picks if (rsi_map.get(p) is None) or (rsi_map[p] >= float(inp.rsi_oversold))]
                removed = [p for p in before if p not in risk_picks]
                if removed:
                    reasons.append(f"rsi_exclude:{','.join(removed)}")
                    picks = [p for p in picks if (p not in rank_codes) or (p in risk_picks)]

            # If filters removed all risk assets, fall back to defensive/cash (planned no-buy).
            if (not risk_picks) and (meta.get("mode") == "risk_on"):
                reasons.append("risk_controls_block_all")
                if inp.risk_off and defensive:
                    picks = [defensive]
                    meta = {"best_score": meta.get("best_score"), "risk_off_triggered": True, "mode": "defensive"}
                else:
                    picks = []
                    meta = {"best_score": meta.get("best_score"), "risk_off_triggered": True, "mode": "cash"}

        # Correlation gate (hfq): if new picks are too correlated with current holdings, skip rebalance this period.
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
                        cval = _pair_corr_hfq(code_a=a, code_b=b, end_pos=end_pos)
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
                elif (float(inertia_score_gap) > 0.0) and (int(inp.top_k) == 1) and (len(cur_key) == 1) and (len(new_key) == 1):
                    cur_c = str(cur_key[0])
                    new_c = str(new_key[0])
                    try:
                        cur_s = float(scores.loc[d, cur_c])
                        new_s = float(scores.loc[d, new_c])
                    except Exception:  # pragma: no cover (defensive)
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

        # 3) Volatility sizing: scale down weights of risk assets (cash remainder).
        exposure = 1.0
        weight_map: dict[str, float] = {}
        if picks and (not dd_in_sleep):
            if picks == [defensive] and defensive:
                weight_map[defensive] = 1.0
                exposure = 1.0
            else:
                risk_picks = [p for p in picks if p in rank_codes]
                if not risk_picks:
                    exposure = 0.0
                else:
                    # default equal-weight, full exposure
                    scales = {p: 1.0 for p in risk_picks}
                    if inp.vol_monitor and ann_vol is not None and d in ann_vol.index:
                        vol_map = {p: (None if pd.isna(ann_vol.loc[d, p]) else float(ann_vol.loc[d, p])) for p in risk_picks}
                        details["ann_vol"] = vol_map
                        for p in risk_picks:
                            v = vol_map.get(p)
                            if v is None or (not np.isfinite(v)) or v <= 0:
                                # If vol is missing/invalid, we conservatively skip this asset.
                                # If vol is (near) zero (e.g. flat price in synthetic tests), it should not force
                                # a "no-buy" decision; treat it as low-vol and cap scale to 1.
                                if v is not None and np.isfinite(v) and v <= 0:
                                    scales[p] = 1.0
                                else:
                                    scales[p] = 0.0
                                continue
                            if v >= float(inp.vol_max_ann):
                                scales[p] = 0.0
                                continue
                            scales[p] = float(min(1.0, float(inp.vol_target_ann) / v))
                    exposure = float(np.mean(list(scales.values()))) if scales else 0.0
                    per = 0.0 if not risk_picks else 1.0 / len(risk_picks)
                    for p in risk_picks:
                        weight_map[p] = float(scales.get(p, 0.0) * per)

                    # If vol sizing zeroed all positions, fall back to defensive/cash.
                    if exposure <= 0.0:
                        reasons.append("vol_block_all")
                        if inp.risk_off and defensive:
                            weight_map = {defensive: 1.0}
                            picks = [defensive]
                            meta = {"best_score": meta.get("best_score"), "risk_off_triggered": True, "mode": "defensive"}
                            exposure = 1.0
                        else:
                            weight_map = {}
                            picks = []
                            meta = {"best_score": meta.get("best_score"), "risk_off_triggered": True, "mode": "cash"}
                            exposure = 0.0

            # Apply rolling-return exposure scaling as the final layer (cash remainder).
            if rr_enabled and weight_map and float(rr_exposure) < 1.0:
                for c in list(weight_map.keys()):
                    weight_map[c] = float(weight_map[c]) * float(rr_exposure)
                exposure = float(exposure) * float(rr_exposure)

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

        # Compute stop-loss / take-profit metadata AFTER final picks are fixed.
        # IMPORTANT: per requirement, stop-loss uses qfq close for both stop level and trigger.
        if tp_sl_mode == "prev_week_low_stop" and (not dd_in_sleep) and risk_picks:
            picks_key = tuple(sorted([c for c in risk_picks if c in rank_codes]))
            di2 = dates.get_loc(d)
            cur_label = labels[di2]
            prev_label = labels[dates.get_loc(decision_dates[i - 1])] if i - 1 >= 0 else None
            for c in picks_key:
                cur_min = None
                prev_min = None
                if not period_min_close_qfq.empty:
                    if c in period_min_close_qfq.columns and cur_label in period_min_close_qfq.index:
                        v = period_min_close_qfq.loc[cur_label, c]
                        cur_min = (None if pd.isna(v) else float(v))
                    if prev_label is not None and c in period_min_close_qfq.columns and prev_label in period_min_close_qfq.index:
                        v = period_min_close_qfq.loc[prev_label, c]
                        prev_min = (None if pd.isna(v) else float(v))

                # base stop: new entry uses prev_min; hold-unchanged uses cur_min
                if (prev_picks_key == picks_key) and (not prev_segment_stopped_out):
                    base = cur_min
                else:
                    base = prev_min if prev_min is not None else cur_min

                # special rule: if decision-day close already below prev_min -> use cur_min
                if prev_min is not None:
                    try:
                        d_close = float(close_qfq.loc[d, c])
                    except (KeyError, TypeError, ValueError):  # pragma: no cover
                        d_close = float("nan")
                    if np.isfinite(d_close) and d_close < float(prev_min) and (cur_min is not None):
                        base = cur_min

                if base is not None and np.isfinite(float(base)):
                    stop_levels[c] = float(base)
            tp_sl["stop_loss_level_by_code"] = {k: float(v) for k, v in stop_levels.items()}
        elif tp_sl_mode in {"atr_chandelier_fixed", "atr_chandelier_progressive"} and (not dd_in_sleep) and risk_picks:
            tp_sl["atr_window_used"] = int(w_atr) if w_atr is not None else None
            tp_sl["atr_mult"] = float(inp.atr_mult)
            tp_sl["atr_step"] = float(inp.atr_step)
            tp_sl["atr_min_mult"] = float(inp.atr_min_mult)

        # In-segment stop-loss check (after weights are written for the segment).
        # We approximate execution as: hold through close on trigger day, then go cash from next trading day.
        if tp_sl_mode == "prev_week_low_stop" and risk_picks and stop_levels:
            seg_dates = dates[start_i : end_i + 1]
            # Find earliest trigger among held assets.
            trig_idx: int | None = None
            trig_code: str | None = None
            for j, day in enumerate(seg_dates):
                for c in risk_picks:
                    sl = stop_levels.get(c)
                    if sl is None:
                        continue
                    try:
                        px = float(close_qfq.loc[day, c])
                    except (KeyError, TypeError, ValueError):  # pragma: no cover
                        continue
                    if np.isfinite(px) and px < float(sl):
                        trig_idx = j
                        trig_code = c
                        break
                if trig_idx is not None:
                    break

            if trig_idx is not None:
                stop_triggered = True
                stop_trigger_date = seg_dates[trig_idx].date().isoformat()
                tp_sl["triggered"] = True
                tp_sl["trigger_date"] = stop_trigger_date
                tp_sl["trigger_code"] = trig_code
                # go cash from next trading day after trigger date, until segment end
                if trig_idx + 1 < len(seg_dates):
                    w.loc[seg_dates[trig_idx + 1] : seg_dates[-1], :] = 0.0
            else:
                tp_sl["triggered"] = False
        elif tp_sl_mode in {"atr_chandelier_fixed", "atr_chandelier_progressive"} and risk_picks:
            seg_dates = dates[start_i : end_i + 1]
            # Build/initialize per-asset state at segment start.
            # We maintain a trailing stop that never decreases.
            entry_px: dict[str, float] = {}
            stop: dict[str, float] = {}
            for c in risk_picks:
                try:
                    p0 = float(close_qfq.loc[seg_dates[0], c])
                    a0 = float(atr.loc[seg_dates[0], c])
                except (KeyError, TypeError, ValueError):  # pragma: no cover
                    continue
                if not (np.isfinite(p0) and np.isfinite(a0) and a0 > 0):
                    continue
                entry_px[c] = p0
                stop[c] = p0 - float(inp.atr_mult) * a0
            tp_sl["entry_price_by_code"] = {k: float(v) for k, v in entry_px.items()}
            tp_sl["initial_stop_by_code"] = {k: float(v) for k, v in stop.items()}

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

                # 2) update trailing stop using today's close and today's ATR (for next day)
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

                    if tp_sl_mode == "atr_chandelier_fixed":
                        dist_mult = float(inp.atr_mult)
                    else:
                        # progressive distance reduction:
                        # distance_mult = max(min_mult, atr_mult - floor(gain/step)*step)
                        # gain is measured in ATR units using current ATR.
                        ep = float(entry_px.get(c, px))
                        gain_units = (px - ep) / a
                        steps = int(np.floor(gain_units / float(inp.atr_step))) if np.isfinite(gain_units) else 0
                        dist_mult = float(inp.atr_mult) - float(steps) * float(inp.atr_step)
                        dist_mult = float(max(float(inp.atr_min_mult), dist_mult))

                    cand = px - dist_mult * a
                    # chandelier stop never decreases
                    stop[c] = float(max(float(stop[c]), float(cand)))

            if trig_idx is not None:
                stop_triggered = True
                stop_trigger_date = seg_dates[trig_idx].date().isoformat()
                tp_sl["triggered"] = True
                tp_sl["trigger_date"] = stop_trigger_date
                tp_sl["trigger_code"] = trig_code
                if trig_idx + 1 < len(seg_dates):
                    w.loc[seg_dates[trig_idx + 1] : seg_dates[-1], :] = 0.0
            else:
                tp_sl["triggered"] = False
            tp_sl["final_stop_by_code"] = {k: float(v) for k, v in stop.items()}

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
                "start_date": dates[start_i].date().isoformat(),
                "end_date": dates[end_i].date().isoformat(),
                "picks": picks,
                "scores": {k: (None if pd.isna(scores.loc[d, k]) else float(scores.loc[d, k])) for k in picks},
                "best_score": meta.get("best_score"),
                "risk_off_triggered": bool(meta.get("risk_off_triggered")),
                "mode": meta.get("mode"),
                "exposure": float(exposure),
                "tp_sl": tp_sl,
                "corr_filter": corr_meta,
                "inertia": inertia_meta,
                "rr_sizing": rr_meta,
                "dd_control": dd_meta,
                "risk_controls": {"reasons": reasons, **details},
            }
        )

    # Corporate-action diagnostics (close-based):
    # We compute these earlier and also use the mask to apply HFQ fallback for NAV stability.
    ret_none = ret_none_close
    ret_hfq_all = ret_hfq_close

    # Daily holding return (hfq, configurable exec_price proxy).
    # Note: close-based hfq is still the default and remains the recommended "total return proxy".
    ret_exec = ret_exec_all
    port_ret = (w * ret_exec).sum(axis=1)
    port_nav = (1.0 + port_ret).cumprod()
    port_nav.iloc[0] = 1.0

    # Equal-weight benchmark WITH SAME rebalance frequency and SAME exec_price proxy:
    # equal-weight across the selected universe only (not including defensive unless user selected it).
    bench_codes = universe[:]  # fixed benchmark universe
    if ep == "close":
        bench_px = close_hfq[bench_codes].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    elif ep == "open":
        b = ohlc_hfq.get("open", pd.DataFrame())
        if not _has_cols(b, bench_codes):
            bench_px = close_hfq[bench_codes].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
        else:
            bench_px = b[bench_codes].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    else:
        bc = ohlc_hfq.get("close", pd.DataFrame())
        bo = ohlc_hfq.get("open", pd.DataFrame())
        if not (_has_cols(bo, bench_codes) and _has_cols(bc, bench_codes)):
            bench_px = close_hfq[bench_codes].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
        else:
            bench_px = (bo[bench_codes].astype(float) + bc[bench_codes].astype(float)) / 2.0
            bench_px = bench_px.replace([np.inf, -np.inf], np.nan).ffill()
    ret_hfq = bench_px.pct_change().fillna(0.0)
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
    # NOTE: we first compute a "shadow" (ungated) NAV for timing signals, then optionally apply timing gating
    # to weights and recompute strategy NAV/metrics.
    w_prev = w.shift(1).fillna(0.0)
    turnover = (w - w_prev).abs().sum(axis=1) / 2.0
    cost = turnover * (inp.cost_bps / 10000.0)
    port_ret_net_shadow = (port_ret - cost).astype(float)
    port_nav_net_shadow = (1.0 + port_ret_net_shadow).cumprod()
    port_nav_net_shadow.iloc[0] = 1.0

    timing_meta: dict[str, Any] = {
        "enabled": bool(inp.timing_rsi_gate),
        "window": int(inp.timing_rsi_window),
        "threshold": 50.0,
        "signal_source": "shadow_nav_net",
        "active_days": None,
        "inactive_days": None,
        "active_ratio": None,
        "rsi": None,
        "active": None,
    }

    timing_expo = pd.Series(np.ones(len(dates), dtype=float), index=dates, dtype=float)
    if bool(inp.timing_rsi_gate):
        win = int(inp.timing_rsi_window)
        thr = 50.0
        # RSI computed on shadow NAV (ignores this timing gate). NaN RSI => do not block (remain active).
        rsi_sig = _rsi_wilder(port_nav_net_shadow, window=win)
        # Rule:
        # - RSI <= 50 => switch to equal-weight holding (EW_REBAL)
        # - RSI > 50 => activate rotation strategy
        active = ((rsi_sig > thr) | rsi_sig.isna()).astype(bool)
        # Use yesterday's RSI signal to decide today's exposure (avoid look-ahead).
        expo = active.shift(1).fillna(True).astype(float)
        timing_expo = expo.astype(float)
        # When inactive (RSI <= 50), switch to equal-weight holdings (same as EW_REBAL),
        # rather than going to cash. During active periods, use rotation weights.
        w_ew_full = pd.DataFrame(0.0, index=dates, columns=w.columns)
        for c in bench_codes:
            if c in w_ew.columns and c in w_ew_full.columns:
                w_ew_full[c] = w_ew[c].astype(float)
        w = (w.mul(expo, axis=0) + w_ew_full.mul(1.0 - expo, axis=0)).astype(float)
        # Recompute strategy return/cost/NAV on timed weights.
        port_ret = (w * ret_exec).sum(axis=1).astype(float)
        port_nav = (1.0 + port_ret).cumprod()
        port_nav.iloc[0] = 1.0
        w_prev = w.shift(1).fillna(0.0)
        turnover = (w - w_prev).abs().sum(axis=1) / 2.0
        cost = turnover * (inp.cost_bps / 10000.0)
        port_ret_net = (port_ret - cost).astype(float)
        port_nav_net = (1.0 + port_ret_net).cumprod()
        port_nav_net.iloc[0] = 1.0

        # timing summary payload (cap arrays by using full; UI needs dates anyway)
        timing_meta["active_days"] = int(active.sum())
        timing_meta["inactive_days"] = int((~active).sum())
        timing_meta["active_ratio"] = float(active.mean()) if len(active) else None
        timing_meta["rsi"] = {"dates": dates.date.astype(str).tolist(), "values": rsi_sig.astype(float).tolist()}
        timing_meta["active"] = {"dates": dates.date.astype(str).tolist(), "values": expo.astype(float).tolist()}
    else:
        # keep original (ungated) values
        port_ret_net = port_ret_net_shadow
        port_nav_net = port_nav_net_shadow

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
    calmar = float(ann_ret / abs(mdd)) if mdd < 0 else float("nan")
    sortino = _sortino(port_ret_net, rf=float(inp.risk_free_rate))
    ui = _ulcer_index(port_nav_net, in_percent=True)
    ui_den = ui / 100.0
    upi = float((ann_ret - float(inp.risk_free_rate)) / ui_den) if ui_den > 0 else float("nan")

    ann_excess = _annualized_return(excess_nav)
    ann_excess_vol = _annualized_vol(active_ret)
    ir = _sharpe(active_ret, rf=0.0)  # same formula but zero rf; for consistency name it IR-style
    ex_mdd = _max_drawdown(excess_nav)
    ex_mdd_dur = _max_drawdown_duration_days(excess_nav)

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
            "annualized_return": float(ann_excess),
            "annualized_volatility": float(ann_excess_vol),
            "information_ratio": float(ir),
            "max_drawdown": float(ex_mdd),
            "max_drawdown_recovery_days": int(ex_mdd_dur),
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
        # Timing sleep indicator for this holding period (based on exposure series).
        timing_ratio = None
        timing_sleep = False
        if bool(inp.timing_rsi_gate):
            try:
                seg = timing_expo.loc[s:e].astype(float)
                timing_ratio = float(seg.mean()) if len(seg) else None
                timing_sleep = bool((timing_ratio is not None) and (timing_ratio <= 1e-12))
            except (KeyError, TypeError, ValueError):  # pragma: no cover (defensive)
                timing_ratio = None
                timing_sleep = False
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
                "timing_sleep": bool(timing_sleep),
                "timing_active_ratio": timing_ratio,
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
        "tp_sl_mode": tp_sl_mode,
        "score_params": {"lambda": float(inp.score_lambda), "vol_power": float(inp.score_vol_power)},
        "codes": codes,
        "benchmark_codes": bench_codes,
        "price_basis": {
            "signal": "hfq",
            "strategy_nav": "none execution + hfq-implied corporate action factor (total return proxy)",
            "benchmark_nav": "hfq",
        },
        "timing": timing_meta,
        "nav": {
            "dates": dates.date.astype(str).tolist(),
            "series": {
                "ROTATION": port_nav_net.astype(float).tolist(),
                "EW_REBAL": ew_nav.astype(float).tolist(),
                "EXCESS": excess_nav.astype(float).tolist(),
            },
        },
        "nav_rsi": {
            "windows": [6, 12, 24],
            "dates": dates.date.astype(str).tolist(),
            "series": {
                "ROTATION": {},
                "EW_REBAL": {},
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
    # fill nav RSI series (avoid recomputing windows extraction twice)
    for w in out["nav_rsi"]["windows"]:
        out["nav_rsi"]["series"]["ROTATION"][str(w)] = _rsi_wilder(port_nav_net, window=int(w)).astype(float).tolist()
        out["nav_rsi"]["series"]["EW_REBAL"][str(w)] = _rsi_wilder(ew_nav, window=int(w)).astype(float).tolist()
    return out

