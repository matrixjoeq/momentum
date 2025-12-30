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
from ..analysis.baseline import load_high_low_prices as _load_high_low_prices
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
    if not (0.0 <= float(inp.rsi_oversold) <= 100.0) or not (0.0 <= float(inp.rsi_overbought) <= 100.0):
        raise ValueError("rsi thresholds must be within [0,100]")
    if float(inp.vol_target_ann) <= 0:
        raise ValueError("vol_target_ann must be > 0")
    if float(inp.vol_max_ann) <= 0:
        raise ValueError("vol_max_ann must be > 0")

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
    need_qfq = bool(inp.trend_filter or inp.rsi_filter or inp.vol_monitor or inp.chop_filter)
    close_qfq = _load_close_prices(db, codes=codes, start=ext_start, end=inp.end, adjust="qfq") if need_qfq else pd.DataFrame()
    need_qfq_hl = bool(inp.chop_filter and cm == "adx")
    high_qfq, low_qfq = _load_high_low_prices(db, codes=codes, start=ext_start, end=inp.end, adjust="qfq") if need_qfq_hl else (pd.DataFrame(), pd.DataFrame())
    close_none = _load_close_prices(db, codes=codes, start=inp.start, end=inp.end, adjust="none")
    if close_none.empty:
        raise ValueError("no execution price data for given range (none)")

    # Align calendars using execution dates; forward-fill hfq onto those dates.
    close_none = close_none.sort_index().ffill()
    dates = close_none.index
    close_hfq = close_hfq.sort_index().reindex(dates).ffill()
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
        # Defensive/cash branch from momentum floor:
        # - picks == [defensive] => invest 100% in defensive (if provided)
        # - picks == [] => cash
        reasons: list[str] = []
        details: dict[str, Any] = {}

        picks = [p for p in picks if p in codes]
        risk_picks = [p for p in picks if p in rank_codes]  # only rank codes are considered "risk assets"

        # Apply pre-trade risk controls only when we are in risk-on mode (holding risk assets).
        if risk_picks:
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

        # 3) Volatility sizing: scale down weights of risk assets (cash remainder).
        exposure = 1.0
        weight_map: dict[str, float] = {}
        if picks:
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

            # write weights for the whole holding segment
            if weight_map:
                for c, wt in weight_map.items():
                    if wt and wt > 0:
                        w.loc[dates[start_i] : dates[end_i], c] = float(wt)
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
                "risk_controls": {"reasons": reasons, **details},
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
        "score_method": (inp.score_method or "raw_mom"),
        "score_params": {"lambda": float(inp.score_lambda), "vol_power": float(inp.score_vol_power)},
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

