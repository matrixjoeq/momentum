from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
Session = Any  # runtime: keep dependency-free typing

from .baseline import (
    TRADING_DAYS_PER_YEAR,
    _annualized_return,
    _annualized_vol,
    _max_drawdown,
    _max_drawdown_duration_days,
    _sharpe,
    _sortino,
    _ulcer_index,
    load_close_prices,
)


@dataclass(frozen=True)
class TrendInputs:
    code: str
    start: dt.date
    end: dt.date
    risk_free_rate: float = 0.025
    cost_bps: float = 0.0
    # strategy selection
    strategy: str = "ma_filter"  # ma_filter | ema_filter | ma_cross | donchian | tsmom | linreg_slope | bias
    # parameters
    sma_window: int = 200  # ma_filter
    fast_window: int = 50  # ma_cross
    slow_window: int = 200  # ma_cross
    donchian_entry: int = 20  # donchian
    donchian_exit: int = 10  # donchian
    mom_lookback: int = 252  # tsmom
    # bias (deviation from EMA) trend-following
    # BIAS = (LN(C) - LN(EMA(C,N))) * 100  (percent)
    bias_ma_window: int = 20  # EMA(C,N) window in trading days
    bias_entry: float = 2.0  # enter when BIAS > entry (percent)
    bias_hot: float = 10.0  # take-profit exit when BIAS >= hot (percent)
    bias_cold: float = -2.0  # stop-loss exit when BIAS <= cold (percent)
    bias_pos_mode: str = "binary"  # binary | continuous (default binary)


def _rolling_linreg_slope(y: np.ndarray) -> float:
    """
    Rolling OLS slope on y over an implicit x = 0..n-1 (centered).
    Returns NaN if any non-finite values exist in the window.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return float("nan")
    if not np.all(np.isfinite(y)):
        return float("nan")
    x = np.arange(y.size, dtype=float)
    x = x - x.mean()
    y0 = y - y.mean()
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y0) / denom)


def _turnover_cost_from_weights(w: pd.Series, *, cost_bps: float) -> pd.Series:
    w_prev = w.shift(1).fillna(0.0).astype(float)
    turnover = (w.astype(float) - w_prev).abs() / 2.0
    cost = turnover * (float(cost_bps) / 10000.0)
    return cost.astype(float)


def _pos_from_donchian(close: pd.Series, *, entry: int, exit_: int) -> pd.Series:
    """
    Long/cash Donchian channel:
    - enter long when close > max(close, entry window) excluding today
    - exit to cash when close < min(close, exit window) excluding today
    """
    e = max(2, int(entry))
    x = max(2, int(exit_))
    hi = close.shift(1).rolling(window=e, min_periods=e).max()
    lo = close.shift(1).rolling(window=x, min_periods=x).min()
    pos = np.zeros(len(close), dtype=float)
    in_pos = False
    for i in range(len(close)):
        c = float(close.iloc[i]) if pd.notna(close.iloc[i]) else float("nan")
        if not np.isfinite(c):
            pos[i] = 1.0 if in_pos else 0.0
            continue
        h = hi.iloc[i]
        l = lo.iloc[i]
        if (not in_pos) and pd.notna(h) and c > float(h):
            in_pos = True
        elif in_pos and pd.notna(l) and c < float(l):
            in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=close.index, dtype=float)


def compute_trend_backtest(db: Session, inp: TrendInputs) -> dict[str, Any]:
    code = (inp.code or "").strip()
    if not code:
        raise ValueError("code is empty")
    if float(inp.cost_bps) < 0:
        raise ValueError("cost_bps must be >= 0")
    if not np.isfinite(float(inp.risk_free_rate)):
        raise ValueError("risk_free_rate must be finite")

    strat = (inp.strategy or "ma_filter").strip().lower()
    if strat not in {"ma_filter", "ema_filter", "ma_cross", "donchian", "tsmom", "linreg_slope", "bias"}:
        raise ValueError(f"invalid strategy={inp.strategy}")

    # validate params
    if int(inp.sma_window) < 2:
        raise ValueError("sma_window must be >= 2")
    if int(inp.fast_window) < 2 or int(inp.slow_window) < 2:
        raise ValueError("fast_window/slow_window must be >= 2")
    if int(inp.fast_window) >= int(inp.slow_window):
        raise ValueError("fast_window must be < slow_window")
    if int(inp.donchian_entry) < 2 or int(inp.donchian_exit) < 2:
        raise ValueError("donchian_entry/donchian_exit must be >= 2")
    if int(inp.mom_lookback) < 2:
        raise ValueError("mom_lookback must be >= 2")
    if int(inp.bias_ma_window) < 2:
        raise ValueError("bias_ma_window must be >= 2")
    if not np.isfinite(float(inp.bias_entry)):
        raise ValueError("bias_entry must be finite")
    if not np.isfinite(float(inp.bias_hot)):
        raise ValueError("bias_hot must be finite")
    if not np.isfinite(float(inp.bias_cold)):
        raise ValueError("bias_cold must be finite")
    if not (float(inp.bias_cold) < float(inp.bias_entry) < float(inp.bias_hot)):
        raise ValueError("bias thresholds must satisfy: cold < entry < hot")
    bias_mode = (inp.bias_pos_mode or "binary").strip().lower()
    if bias_mode not in {"binary", "continuous"}:
        raise ValueError("bias_pos_mode must be binary|continuous")

    # Price basis consistent with rotation research:
    # - Signal/TA: qfq close
    # - Execution/NAV: none close, with hfq return fallback on corporate-action days to avoid false cliffs
    # - Benchmark (buy&hold): hfq close (total return proxy; non-tradable)
    need_hist = max(int(inp.sma_window), int(inp.slow_window), int(inp.donchian_entry), int(inp.mom_lookback), 20) + 60
    ext_start = inp.start - dt.timedelta(days=int(need_hist) * 2)

    close_none = load_close_prices(db, codes=[code], start=inp.start, end=inp.end, adjust="none")
    if close_none.empty or (code not in close_none.columns) or close_none[code].dropna().empty:
        raise ValueError("no execution price data for given range (none)")
    dates = close_none.sort_index().ffill().index

    close_qfq = load_close_prices(db, codes=[code], start=ext_start, end=inp.end, adjust="qfq").sort_index().reindex(dates).ffill()
    close_hfq = load_close_prices(db, codes=[code], start=ext_start, end=inp.end, adjust="hfq").sort_index().reindex(dates).ffill()

    for name, df in [("qfq", close_qfq), ("hfq", close_hfq)]:
        if df.empty or (code not in df.columns) or df[code].dropna().empty:
            raise ValueError(f"missing {name} close data for: {code}")

    px_sig = close_qfq[code].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    px_exec_none = close_none[code].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    px_bh = close_hfq[code].astype(float).replace([np.inf, -np.inf], np.nan).ffill()

    px_sig = px_sig.dropna()
    # align everything to execution calendar
    px_sig = px_sig.reindex(dates).ffill()
    px_exec_none = px_exec_none.reindex(dates).ffill()
    px_bh = px_bh.reindex(dates).ffill()

    if px_exec_none.dropna().empty or px_sig.dropna().empty or px_bh.dropna().empty:
        raise ValueError("no valid price series after alignment")

    # Returns:
    # - strategy execution: primarily none returns, but on corporate-action days use hfq return
    ret_none = px_exec_none.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    ret_hfq = px_bh.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    gross_none = (1.0 + ret_none).astype(float)
    gross_hfq = (1.0 + ret_hfq).astype(float)
    corp_factor = (gross_hfq / gross_none).replace([np.inf, -np.inf], np.nan)
    # same heuristic as rotation debug events: >2% or extreme ratios
    dev = (corp_factor - 1.0).abs()
    ca_mask = (dev > 0.02) | (corp_factor > 1.2) | (corp_factor < 1.0 / 1.2)
    ret_exec = ret_none.where(~ca_mask.fillna(False), other=ret_hfq).astype(float)

    if strat == "ma_filter":
        sma = px_sig.rolling(window=int(inp.sma_window), min_periods=max(2, int(inp.sma_window) // 2)).mean()
        raw_pos = (px_sig > sma).astype(float).fillna(0.0)
    elif strat == "ema_filter":
        # Use EMA on signal price basis (qfq close), applied with next-day execution.
        span = int(inp.sma_window)
        ema = px_sig.ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()
        raw_pos = (px_sig > ema).astype(float).fillna(0.0)
    elif strat == "ma_cross":
        fast = px_sig.rolling(window=int(inp.fast_window), min_periods=max(2, int(inp.fast_window) // 2)).mean()
        slow = px_sig.rolling(window=int(inp.slow_window), min_periods=max(2, int(inp.slow_window) // 2)).mean()
        raw_pos = (fast > slow).astype(float).fillna(0.0)
    elif strat == "donchian":
        raw_pos = _pos_from_donchian(px_sig, entry=int(inp.donchian_entry), exit_=int(inp.donchian_exit)).astype(float)
    elif strat == "linreg_slope":
        # Linear regression slope of log price over window; long if slope > 0.
        n = int(inp.sma_window)
        y = np.log(px_sig.clip(lower=1e-12).astype(float))
        slope = y.rolling(window=n, min_periods=max(2, n // 2)).apply(_rolling_linreg_slope, raw=True)
        raw_pos = (slope > 0.0).astype(float).fillna(0.0)
    elif strat == "bias":
        # BIAS rising-follow strategy:
        # - Trend filter removed (per research need)
        # - BIAS computed from log-diff to EMA: (ln(C) - ln(EMA(C,N))) * 100 (percent)
        # - Enter when bias > entry; exit on take-profit (bias >= hot) or stop-loss (bias <= cold)
        b_win = int(inp.bias_ma_window)
        ema = px_sig.ewm(span=b_win, adjust=False, min_periods=max(2, b_win // 2)).mean()
        ln_c = np.log(px_sig.clip(lower=1e-12).astype(float))
        ln_ema = np.log(ema.clip(lower=1e-12).astype(float))
        bias = (ln_c - ln_ema) * 100.0

        entry = float(inp.bias_entry)
        hot = float(inp.bias_hot)
        cold = float(inp.bias_cold)
        pos_mode = bias_mode

        pos = np.zeros(len(px_sig), dtype=float)
        in_pos = False
        for i, d in enumerate(px_sig.index):
            if not np.isfinite(float(bias.loc[d])):
                in_pos = False
                pos[i] = 0.0
                continue

            b = float(bias.loc[d])
            if not in_pos:
                if b > entry:
                    in_pos = True
            else:
                if (b >= hot) or (b <= cold):
                    in_pos = False

            if not in_pos:
                pos[i] = 0.0
            elif pos_mode == "binary":
                pos[i] = 1.0
            else:
                # Continuous sizing: scale exposure with bias inside [cold, hot]
                # cold -> 0, hot -> 1, clipped.
                w = (b - cold) / (hot - cold)
                pos[i] = float(np.clip(w, 0.0, 1.0))

        raw_pos = pd.Series(pos, index=px_sig.index, dtype=float)
    else:
        mom = px_sig / px_sig.shift(int(inp.mom_lookback)) - 1.0
        raw_pos = (mom > 0.0).astype(float).fillna(0.0)

    # apply signal from next trading day (avoid look-ahead)
    w = raw_pos.shift(1).fillna(0.0).astype(float).clip(lower=0.0, upper=1.0)
    cost = _turnover_cost_from_weights(w, cost_bps=float(inp.cost_bps))
    strat_ret = (w * ret_exec - cost).astype(float)

    nav = (1.0 + strat_ret).cumprod()
    if len(nav) > 0:
        nav.iloc[0] = 1.0

    bh_nav = (1.0 + ret_hfq).cumprod()
    if len(bh_nav) > 0:
        bh_nav.iloc[0] = 1.0

    active = strat_ret - ret_hfq
    excess_nav = (1.0 + active).cumprod()
    if len(excess_nav) > 0:
        excess_nav.iloc[0] = 1.0

    # metrics
    m_strat = {
        "cumulative_return": float(nav.iloc[-1] - 1.0),
        "annualized_return": float(_annualized_return(nav, ann_factor=TRADING_DAYS_PER_YEAR)),
        "annualized_volatility": float(_annualized_vol(strat_ret, ann_factor=TRADING_DAYS_PER_YEAR)),
        "max_drawdown": float(_max_drawdown(nav)),
        "max_drawdown_recovery_days": int(_max_drawdown_duration_days(nav)),
        "sharpe_ratio": float(_sharpe(strat_ret, rf=float(inp.risk_free_rate), ann_factor=TRADING_DAYS_PER_YEAR)),
        "sortino_ratio": float(_sortino(strat_ret, rf=float(inp.risk_free_rate), ann_factor=TRADING_DAYS_PER_YEAR)),
        "ulcer_index": float(_ulcer_index(nav, in_percent=True)),
        "avg_daily_turnover": float(((w - w.shift(1).fillna(0.0)).abs() / 2.0).mean()),
    }
    m_bh = {
        "cumulative_return": float(bh_nav.iloc[-1] - 1.0),
        "annualized_return": float(_annualized_return(bh_nav, ann_factor=TRADING_DAYS_PER_YEAR)),
        "annualized_volatility": float(_annualized_vol(ret_hfq, ann_factor=TRADING_DAYS_PER_YEAR)),
        "max_drawdown": float(_max_drawdown(bh_nav)),
        "max_drawdown_recovery_days": int(_max_drawdown_duration_days(bh_nav)),
        "sharpe_ratio": float(_sharpe(ret_hfq, rf=float(inp.risk_free_rate), ann_factor=TRADING_DAYS_PER_YEAR)),
        "sortino_ratio": float(_sortino(ret_hfq, rf=float(inp.risk_free_rate), ann_factor=TRADING_DAYS_PER_YEAR)),
        "ulcer_index": float(_ulcer_index(bh_nav, in_percent=True)),
    }
    m_ex = {
        "cumulative_return": float(excess_nav.iloc[-1] - 1.0),
        "annualized_return": float(_annualized_return(excess_nav, ann_factor=TRADING_DAYS_PER_YEAR)),
        "information_ratio": float(_sharpe(active, rf=0.0, ann_factor=TRADING_DAYS_PER_YEAR)),
    }

    out = {
        "meta": {
            "type": "trend_backtest",
            "code": code,
            "start": inp.start.strftime("%Y%m%d"),
            "end": inp.end.strftime("%Y%m%d"),
            "strategy": strat,
            "price_basis": {
                "signal": "qfq close",
                "strategy_nav": "none close preferred; hfq return fallback on corporate-action days",
                "benchmark_nav": "hfq close",
            },
            "params": {
                "sma_window": int(inp.sma_window),
                "fast_window": int(inp.fast_window),
                "slow_window": int(inp.slow_window),
                "donchian_entry": int(inp.donchian_entry),
                "donchian_exit": int(inp.donchian_exit),
                "mom_lookback": int(inp.mom_lookback),
                "bias_ma_window": int(inp.bias_ma_window),
                "bias_entry": float(inp.bias_entry),
                "bias_hot": float(inp.bias_hot),
                "bias_cold": float(inp.bias_cold),
                "bias_pos_mode": bias_mode,
                "cost_bps": float(inp.cost_bps),
                "risk_free_rate": float(inp.risk_free_rate),
            },
        },
        "nav": {
            "dates": nav.index.date.astype(str).tolist(),
            "series": {
                "STRAT": nav.astype(float).tolist(),
                "BUY_HOLD": bh_nav.reindex(nav.index).astype(float).tolist(),
                "EXCESS": excess_nav.reindex(nav.index).astype(float).tolist(),
            },
        },
        "signals": {
            "position": raw_pos.reindex(nav.index).astype(float).tolist(),
        },
        "metrics": {"strategy": m_strat, "benchmark": m_bh, "excess": m_ex},
        "corporate_actions": (
            [
                {
                    "date": d.date().isoformat(),
                    "none_return": float(ret_none.loc[d]),
                    "hfq_return": float(ret_hfq.loc[d]),
                    "corp_factor": float(corp_factor.loc[d]),
                }
                for d in corp_factor.index[ca_mask.fillna(False)]
            ][:200]
        ),
    }
    return out

