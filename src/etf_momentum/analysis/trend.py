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
    _information_ratio,
    _max_drawdown,
    _max_drawdown_duration_days,
    _sharpe,
    _sortino,
    _ulcer_index,
    load_close_prices,
    load_high_low_prices,
    load_ohlc_prices,
)
from .execution_timing import forward_align_returns

# 各趋势策略的执行说明（信号日与收益归属）：统一为 T 日收盘后确定信号，T+1 日执行，收益不包含决策日当日。
TREND_STRATEGY_EXECUTION_DESCRIPTIONS: dict[str, str] = {
    "ma_filter": "信号在 T 日收盘后根据价格与均线(SMA/EMA)关系确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "ma_cross": "信号在 T 日收盘后根据快慢线金叉/死叉确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "donchian": "信号在 T 日收盘后根据是否突破通道上轨/下轨确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "tsmom": "信号在 T 日收盘后根据回看期收益与动量入/出场阈值确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "linreg_slope": "信号在 T 日收盘后根据回看窗口线性回归斜率正负确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "bias": "信号在 T 日收盘后根据 BIAS 与进出场阈值确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "macd_cross": "信号在 T 日收盘后根据 MACD 与信号线金叉/死叉确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "macd_zero_filter": "信号在 T 日收盘后根据 MACD 是否大于零确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "macd_v": "信号在 T 日收盘后根据 ATR 归一化 MACD 与信号线关系确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
}


@dataclass(frozen=True)
class TrendInputs:
    code: str
    start: dt.date
    end: dt.date
    risk_free_rate: float = 0.025
    cost_bps: float = 0.0
    exec_price: str = "open"  # open|close|oc2
    # strategy selection
    strategy: str = "ma_filter"  # ma_filter | ma_cross | donchian | tsmom | linreg_slope | bias | macd_cross | macd_zero_filter | macd_v
    # parameters
    sma_window: int = 200  # ma_filter
    fast_window: int = 50  # ma_cross
    slow_window: int = 200  # ma_cross
    ma_type: str = "sma"  # ma_cross: sma | ema
    donchian_entry: int = 20  # donchian
    donchian_exit: int = 10  # donchian
    mom_lookback: int = 252  # tsmom
    tsmom_entry_threshold: float = 0.0  # tsmom: enter when momentum > this value
    tsmom_exit_threshold: float = 0.0  # tsmom: exit when momentum <= this value
    atr_stop_mode: str = "none"  # none | static | trailing | tightening
    atr_stop_window: int = 14  # ATR lookback window
    atr_stop_n: float = 2.0  # stop distance multiplier by ATR
    atr_stop_m: float = 0.5  # tightening step in ATR multiples
    # bias (deviation from EMA) trend-following
    # BIAS = (LN(C) - LN(EMA(C,N))) * 100  (percent)
    bias_ma_window: int = 20  # EMA(C,N) window in trading days
    bias_entry: float = 2.0  # enter when BIAS > entry (percent)
    bias_hot: float = 10.0  # take-profit exit when BIAS >= hot (percent)
    bias_cold: float = -2.0  # stop-loss exit when BIAS <= cold (percent)
    bias_pos_mode: str = "binary"  # binary | continuous (default binary)
    # MACD/MACD-V params
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_v_atr_window: int = 26
    macd_v_scale: float = 100.0


@dataclass(frozen=True)
class TrendPortfolioInputs:
    codes: list[str]
    start: dt.date
    end: dt.date
    risk_free_rate: float = 0.025
    cost_bps: float = 0.0
    exec_price: str = "open"  # open|close|oc2
    strategy: str = "ma_filter"
    position_sizing: str = "equal"  # equal | vol_target
    vol_window: int = 20
    vol_target_ann: float = 0.20
    dynamic_universe: bool = False
    # single-strategy params
    sma_window: int = 200
    fast_window: int = 50
    slow_window: int = 200
    ma_type: str = "sma"
    donchian_entry: int = 20
    donchian_exit: int = 10
    mom_lookback: int = 252
    tsmom_entry_threshold: float = 0.0
    tsmom_exit_threshold: float = 0.0
    atr_stop_mode: str = "none"
    atr_stop_window: int = 14
    atr_stop_n: float = 2.0
    atr_stop_m: float = 0.5
    bias_ma_window: int = 20
    bias_entry: float = 2.0
    bias_hot: float = 10.0
    bias_cold: float = -2.0
    bias_pos_mode: str = "binary"
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_v_atr_window: int = 26
    macd_v_scale: float = 100.0


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


def _ema(s: pd.Series, window: int) -> pd.Series:
    w = max(2, int(window))
    return s.ewm(span=w, adjust=False, min_periods=max(2, w // 2)).mean()


def _moving_average(s: pd.Series, *, window: int, ma_type: str) -> pd.Series:
    t = str(ma_type or "sma").strip().lower()
    if t == "ema":
        return _ema(s, int(window))
    w = max(2, int(window))
    return s.rolling(window=w, min_periods=max(2, w // 2)).mean()


def _pos_from_tsmom(
    mom: pd.Series,
    *,
    entry_threshold: float,
    exit_threshold: float,
) -> pd.Series:
    """
    Long/cash TSMOM with configurable hysteresis thresholds:
    - enter long when momentum > entry_threshold
    - exit to cash when momentum <= exit_threshold
    """
    ent = float(entry_threshold)
    ex = float(exit_threshold)
    pos = np.zeros(len(mom), dtype=float)
    in_pos = False
    for i, v in enumerate(mom.to_numpy(dtype=float)):
        if not np.isfinite(v):
            in_pos = False
            pos[i] = 0.0
            continue
        if not in_pos:
            if v > ent:
                in_pos = True
        elif v <= ex:
            in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=mom.index, dtype=float)


def _apply_atr_stop(
    base_pos: pd.Series,
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    mode: str,
    atr_window: int,
    n_mult: float,
    m_step: float,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Apply universal ATR stop-loss overlay on top of base strategy position:
    - static: stop = entry_price - n * ATR(entry), fixed after entry
    - trailing: if close rises (vs entry or prior close), move stop upward to close - n * ATR(today)
    - tightening: trailing + tighten stop distance by m ATR for each +m ATR rise from entry,
      with minimum distance floored at m ATR.
    Stop trigger rule: when today's close < stop, exit is decided today and executes next day.
    """
    mode_v = str(mode or "none").strip().lower()
    if mode_v == "none":
        out_none = base_pos.fillna(0.0).astype(float)
        return out_none, {
            "enabled": False,
            "mode": "none",
            "trigger_count": 0,
            "trigger_dates": [],
            "first_trigger_date": None,
            "last_trigger_date": None,
            "entries": int(((out_none > 0.0) & (out_none.shift(1).fillna(0.0) <= 0.0)).sum()),
            "exits": int(((out_none <= 0.0) & (out_none.shift(1).fillna(0.0) > 0.0)).sum()),
            "trigger_exit_share": 0.0,
        }

    atr = _atr_from_hlc(high.astype(float), low.astype(float), close.astype(float), window=int(atr_window)).astype(float)
    bp = base_pos.astype(float).fillna(0.0)
    cl = close.astype(float)

    out = np.zeros(len(bp), dtype=float)
    in_pos = False
    stop_px = float("nan")
    entry_px = float("nan")
    entry_atr = float("nan")
    prev_close = float("nan")
    trigger_dates: list[str] = []

    for i in range(len(bp)):
        b = float(bp.iloc[i]) if np.isfinite(float(bp.iloc[i])) else 0.0
        c = float(cl.iloc[i]) if np.isfinite(float(cl.iloc[i])) else float("nan")
        a = float(atr.iloc[i]) if np.isfinite(float(atr.iloc[i])) else float("nan")

        if not in_pos:
            if b <= 0.0 or (not np.isfinite(c)) or (not np.isfinite(a)) or a <= 0.0:
                out[i] = 0.0
                continue
            in_pos = True
            entry_px = c
            entry_atr = a
            prev_close = c
            stop_px = entry_px - float(n_mult) * entry_atr
            out[i] = b
            continue

        # If base strategy already exits, clear stop state directly.
        if b <= 0.0:
            in_pos = False
            out[i] = 0.0
            stop_px = float("nan")
            entry_px = float("nan")
            entry_atr = float("nan")
            prev_close = float("nan")
            continue

        # Stop trigger check uses the current effective stop; avoid applying updates
        # first to prevent masking a real stop hit on this close.
        if np.isfinite(c) and np.isfinite(stop_px) and (c < stop_px):
            in_pos = False
            out[i] = 0.0
            d = bp.index[i]
            trigger_dates.append((d.date().isoformat() if hasattr(d, "date") else str(d)))
            stop_px = float("nan")
            entry_px = float("nan")
            entry_atr = float("nan")
            prev_close = float("nan")
            continue

        if mode_v in {"trailing", "tightening"} and np.isfinite(c) and np.isfinite(a) and a > 0.0:
            should_follow = (c > entry_px) or (np.isfinite(prev_close) and c > prev_close)
            if should_follow:
                dist_mult = float(n_mult)
                if mode_v == "tightening":
                    ent_atr = float(entry_atr) if (np.isfinite(entry_atr) and entry_atr > 0.0) else a
                    rise_in_atr = max(0.0, (c - entry_px) / max(ent_atr, 1e-12))
                    steps = int(np.floor(rise_in_atr / float(m_step))) if float(m_step) > 0 else 0
                    dist_mult = max(float(m_step), float(n_mult) - steps * float(m_step))
                stop_candidate = c - dist_mult * a
                if (not np.isfinite(stop_px)) or (stop_candidate > stop_px):
                    stop_px = stop_candidate

        out[i] = b
        prev_close = c

    out_s = pd.Series(out, index=base_pos.index, dtype=float)
    entries = int(((out_s > 0.0) & (out_s.shift(1).fillna(0.0) <= 0.0)).sum())
    exits = int(((out_s <= 0.0) & (out_s.shift(1).fillna(0.0) > 0.0)).sum())
    trigger_count = int(len(trigger_dates))
    stats = {
        "enabled": True,
        "mode": mode_v,
        "trigger_count": trigger_count,
        "trigger_dates": trigger_dates[:200],
        "first_trigger_date": (trigger_dates[0] if trigger_dates else None),
        "last_trigger_date": (trigger_dates[-1] if trigger_dates else None),
        "entries": entries,
        "exits": exits,
        "trigger_exit_share": (float(trigger_count) / float(exits) if exits > 0 else 0.0),
    }
    return out_s, stats


def _macd_core(close: pd.Series, *, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, int(fast))
    ema_slow = _ema(close, int(slow))
    macd = (ema_fast - ema_slow).astype(float)
    sig = _ema(macd, int(signal)).astype(float)
    hist = (macd - sig).astype(float)
    return macd, sig, hist


def _atr_from_hlc(high: pd.Series, low: pd.Series, close: pd.Series, *, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    w = max(2, int(window))
    return tr.rolling(window=w, min_periods=max(2, w // 2)).mean().astype(float)


def compute_trend_backtest(db: Session, inp: TrendInputs) -> dict[str, Any]:
    code = (inp.code or "").strip()
    if not code:
        raise ValueError("code is empty")
    if float(inp.cost_bps) < 0:
        raise ValueError("cost_bps must be >= 0")
    ep = str(getattr(inp, "exec_price", "open") or "open").strip().lower()
    if ep not in {"open", "close", "oc2"}:
        raise ValueError("exec_price must be one of: open|close|oc2")
    if not np.isfinite(float(inp.risk_free_rate)):
        raise ValueError("risk_free_rate must be finite")

    strat = (inp.strategy or "ma_filter").strip().lower()
    if strat not in {
        "ma_filter",
        "ma_cross",
        "donchian",
        "tsmom",
        "linreg_slope",
        "bias",
        "macd_cross",
        "macd_zero_filter",
        "macd_v",
    }:
        raise ValueError(f"invalid strategy={inp.strategy}")

    # validate params
    if int(inp.sma_window) < 2:
        raise ValueError("sma_window must be >= 2")
    if int(inp.fast_window) < 2 or int(inp.slow_window) < 2:
        raise ValueError("fast_window/slow_window must be >= 2")
    if int(inp.fast_window) >= int(inp.slow_window):
        raise ValueError("fast_window must be < slow_window")
    ma_type = str(getattr(inp, "ma_type", "sma") or "sma").strip().lower()
    if ma_type not in {"sma", "ema"}:
        raise ValueError("ma_type must be one of: sma|ema")
    if not np.isfinite(float(inp.tsmom_entry_threshold)):
        raise ValueError("tsmom_entry_threshold must be finite")
    if not np.isfinite(float(inp.tsmom_exit_threshold)):
        raise ValueError("tsmom_exit_threshold must be finite")
    if float(inp.tsmom_entry_threshold) < float(inp.tsmom_exit_threshold):
        raise ValueError("tsmom thresholds must satisfy: entry >= exit")
    atr_mode = str(getattr(inp, "atr_stop_mode", "none") or "none").strip().lower()
    if atr_mode not in {"none", "static", "trailing", "tightening"}:
        raise ValueError("atr_stop_mode must be one of: none|static|trailing|tightening")
    if int(inp.atr_stop_window) < 2:
        raise ValueError("atr_stop_window must be >= 2")
    if (not np.isfinite(float(inp.atr_stop_n))) or float(inp.atr_stop_n) <= 0:
        raise ValueError("atr_stop_n must be finite and > 0")
    if (not np.isfinite(float(inp.atr_stop_m))) or float(inp.atr_stop_m) <= 0:
        raise ValueError("atr_stop_m must be finite and > 0")
    if atr_mode == "tightening" and float(inp.atr_stop_n) <= float(inp.atr_stop_m):
        raise ValueError("atr_stop_n must be > atr_stop_m when atr_stop_mode=tightening")
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
    if int(inp.macd_fast) < 2 or int(inp.macd_slow) < 2 or int(inp.macd_signal) < 2:
        raise ValueError("macd_fast/macd_slow/macd_signal must be >= 2")
    if int(inp.macd_fast) >= int(inp.macd_slow):
        raise ValueError("macd_fast must be < macd_slow")
    if int(inp.macd_v_atr_window) < 2:
        raise ValueError("macd_v_atr_window must be >= 2")
    if (not np.isfinite(float(inp.macd_v_scale))) or float(inp.macd_v_scale) <= 0:
        raise ValueError("macd_v_scale must be finite and > 0")

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

    high_qfq_df, low_qfq_df = load_high_low_prices(db, codes=[code], start=ext_start, end=inp.end, adjust="qfq")
    high_qfq = (
        high_qfq_df[code].astype(float).replace([np.inf, -np.inf], np.nan).reindex(dates).ffill()
        if (not high_qfq_df.empty and code in high_qfq_df.columns)
        else px_sig
    )
    low_qfq = (
        low_qfq_df[code].astype(float).replace([np.inf, -np.inf], np.nan).reindex(dates).ffill()
        if (not low_qfq_df.empty and code in low_qfq_df.columns)
        else px_sig
    )

    # Returns:
    # - strategy execution: choose open/close/oc2 by exec_price, then apply none->hfq fallback on CA days
    # - benchmark: hfq close
    ohlc_none = load_ohlc_prices(db, codes=[code], start=inp.start, end=inp.end, adjust="none")
    ohlc_hfq = load_ohlc_prices(db, codes=[code], start=inp.start, end=inp.end, adjust="hfq")

    def _pick_series(ohlc: dict[str, pd.DataFrame], field: str, fallback: pd.Series) -> pd.Series:
        df = ohlc.get(field, pd.DataFrame())
        if df is None or df.empty or code not in df.columns:
            return fallback.astype(float)
        return pd.to_numeric(df[code], errors="coerce").astype(float).reindex(dates).ffill().combine_first(fallback.astype(float))

    o_none = _pick_series(ohlc_none, "open", px_exec_none)
    c_none = _pick_series(ohlc_none, "close", px_exec_none)
    o_hfq = _pick_series(ohlc_hfq, "open", px_bh)
    c_hfq = _pick_series(ohlc_hfq, "close", px_bh)
    if ep == "open":
        exec_none = o_none.combine_first(px_exec_none)
        exec_hfq = o_hfq.combine_first(px_bh)
        ret_none = exec_none.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_exec_hfq = exec_hfq.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    elif ep == "close":
        # 成交价=收盘价时使用执行日（T+1）收盘价：ret_exec 经 forward_align 后为 close[t+1]/close[t]-1，执行日 t 入场价为 close[t]。
        exec_none = c_none.combine_first(px_exec_none)
        exec_hfq = c_hfq.combine_first(px_bh)
        ret_none = exec_none.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_exec_hfq = exec_hfq.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    else:
        ret_open_none = o_none.combine_first(px_exec_none).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_close_none = c_none.combine_first(px_exec_none).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_none = (0.5 * (ret_open_none + ret_close_none)).astype(float)
        ret_open_hfq = o_hfq.combine_first(px_bh).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_close_hfq = c_hfq.combine_first(px_bh).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_exec_hfq = (0.5 * (ret_open_hfq + ret_close_hfq)).astype(float)
    ret_hfq = px_bh.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    gross_none = (1.0 + ret_none).astype(float)
    gross_hfq = (1.0 + ret_hfq).astype(float)
    corp_factor = (gross_hfq / gross_none).replace([np.inf, -np.inf], np.nan)
    # same heuristic as rotation debug events: >2% or extreme ratios
    dev = (corp_factor - 1.0).abs()
    ca_mask = (dev > 0.02) | (corp_factor > 1.2) | (corp_factor < 1.0 / 1.2)
    ret_exec = ret_none.where(~ca_mask.fillna(False), other=ret_exec_hfq).astype(float)

    if strat == "ma_filter":
        ma = _moving_average(px_sig, window=int(inp.sma_window), ma_type=ma_type)
        raw_pos = (px_sig > ma).astype(float).fillna(0.0)
    elif strat == "ma_cross":
        fast = _moving_average(px_sig, window=int(inp.fast_window), ma_type=ma_type)
        slow = _moving_average(px_sig, window=int(inp.slow_window), ma_type=ma_type)
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
    elif strat == "macd_cross":
        macd, sig, _ = _macd_core(
            px_sig,
            fast=int(inp.macd_fast),
            slow=int(inp.macd_slow),
            signal=int(inp.macd_signal),
        )
        raw_pos = (macd > sig).astype(float).fillna(0.0)
    elif strat == "macd_zero_filter":
        macd, _, _ = _macd_core(
            px_sig,
            fast=int(inp.macd_fast),
            slow=int(inp.macd_slow),
            signal=int(inp.macd_signal),
        )
        raw_pos = (macd > 0.0).astype(float).fillna(0.0)
    elif strat == "macd_v":
        macd, _, _ = _macd_core(
            px_sig,
            fast=int(inp.macd_fast),
            slow=int(inp.macd_slow),
            signal=int(inp.macd_signal),
        )
        atr = _atr_from_hlc(high_qfq, low_qfq, px_sig, window=int(inp.macd_v_atr_window))
        macd_v = (macd / atr.replace(0.0, np.nan)) * float(inp.macd_v_scale)
        macd_v_sig = _ema(macd_v, int(inp.macd_signal))
        raw_pos = (macd_v > macd_v_sig).astype(float).fillna(0.0)
    else:
        mom = px_sig / px_sig.shift(int(inp.mom_lookback)) - 1.0
        raw_pos = _pos_from_tsmom(
            mom,
            entry_threshold=float(inp.tsmom_entry_threshold),
            exit_threshold=float(inp.tsmom_exit_threshold),
        ).astype(float)

    raw_pos, atr_stop_stats = _apply_atr_stop(
        raw_pos,
        close=px_sig,
        high=high_qfq,
        low=low_qfq,
        mode=atr_mode,
        atr_window=int(inp.atr_stop_window),
        n_mult=float(inp.atr_stop_n),
        m_step=float(inp.atr_stop_m),
    )
    raw_pos = raw_pos.astype(float)

    # Weights become effective on execution day; return uses forward interval (t -> t+1).
    w = raw_pos.shift(1).fillna(0.0).astype(float).clip(lower=0.0, upper=1.0)
    ret_exec_fwd = forward_align_returns(ret_exec.to_frame("r"))["r"].astype(float)
    cost = _turnover_cost_from_weights(w, cost_bps=float(inp.cost_bps))
    strat_ret = (w * ret_exec_fwd - cost).astype(float)

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
            "strategy_execution_description": TREND_STRATEGY_EXECUTION_DESCRIPTIONS.get(strat, ""),
            "price_basis": {
                "signal": "qfq close",
                "strategy_nav": "none close preferred; hfq return fallback on corporate-action days",
                "benchmark_nav": "hfq close",
            },
            "params": {
                "sma_window": int(inp.sma_window),
                "fast_window": int(inp.fast_window),
                "slow_window": int(inp.slow_window),
                "ma_type": ma_type,
                "donchian_entry": int(inp.donchian_entry),
                "donchian_exit": int(inp.donchian_exit),
                "mom_lookback": int(inp.mom_lookback),
                "tsmom_entry_threshold": float(inp.tsmom_entry_threshold),
                "tsmom_exit_threshold": float(inp.tsmom_exit_threshold),
                "atr_stop_mode": str(atr_mode),
                "atr_stop_window": int(inp.atr_stop_window),
                "atr_stop_n": float(inp.atr_stop_n),
                "atr_stop_m": float(inp.atr_stop_m),
                "bias_ma_window": int(inp.bias_ma_window),
                "bias_entry": float(inp.bias_entry),
                "bias_hot": float(inp.bias_hot),
                "bias_cold": float(inp.bias_cold),
                "bias_pos_mode": bias_mode,
                "macd_fast": int(inp.macd_fast),
                "macd_slow": int(inp.macd_slow),
                "macd_signal": int(inp.macd_signal),
                "macd_v_atr_window": int(inp.macd_v_atr_window),
                "macd_v_scale": float(inp.macd_v_scale),
                "exec_price": str(ep),
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
        "risk_controls": {"atr_stop": atr_stop_stats},
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


def compute_trend_portfolio_backtest(
    db: Session,
    inp: TrendPortfolioInputs,
    data_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run trend portfolio backtest. When data_override is provided, db is not used
    (can be None); used for OOS bootstrap with synthetic in-sample data.
    """
    codes = list(dict.fromkeys([str(c).strip() for c in (inp.codes or []) if str(c).strip()]))
    if not codes:
        raise ValueError("codes is empty")
    if float(inp.cost_bps) < 0:
        raise ValueError("cost_bps must be >= 0")
    ep = str(getattr(inp, "exec_price", "open") or "open").strip().lower()
    if ep not in {"open", "close", "oc2"}:
        raise ValueError("exec_price must be one of: open|close|oc2")
    if not np.isfinite(float(inp.risk_free_rate)):
        raise ValueError("risk_free_rate must be finite")
    ps = str(inp.position_sizing or "equal").strip().lower()
    if ps not in {"equal", "vol_target"}:
        raise ValueError("position_sizing must be equal|vol_target")
    if int(inp.vol_window) < 2:
        raise ValueError("vol_window must be >= 2")
    if (not np.isfinite(float(inp.vol_target_ann))) or float(inp.vol_target_ann) <= 0:
        raise ValueError("vol_target_ann must be finite and > 0")
    ma_type = str(getattr(inp, "ma_type", "sma") or "sma").strip().lower()
    if ma_type not in {"sma", "ema"}:
        raise ValueError("ma_type must be one of: sma|ema")
    atr_mode = str(getattr(inp, "atr_stop_mode", "none") or "none").strip().lower()
    if atr_mode not in {"none", "static", "trailing", "tightening"}:
        raise ValueError("atr_stop_mode must be one of: none|static|trailing|tightening")
    if int(inp.atr_stop_window) < 2:
        raise ValueError("atr_stop_window must be >= 2")
    if (not np.isfinite(float(inp.atr_stop_n))) or float(inp.atr_stop_n) <= 0:
        raise ValueError("atr_stop_n must be finite and > 0")
    if (not np.isfinite(float(inp.atr_stop_m))) or float(inp.atr_stop_m) <= 0:
        raise ValueError("atr_stop_m must be finite and > 0")
    if atr_mode == "tightening" and float(inp.atr_stop_n) <= float(inp.atr_stop_m):
        raise ValueError("atr_stop_n must be > atr_stop_m when atr_stop_mode=tightening")
    if not np.isfinite(float(inp.tsmom_entry_threshold)):
        raise ValueError("tsmom_entry_threshold must be finite")
    if not np.isfinite(float(inp.tsmom_exit_threshold)):
        raise ValueError("tsmom_exit_threshold must be finite")
    if float(inp.tsmom_entry_threshold) < float(inp.tsmom_exit_threshold):
        raise ValueError("tsmom thresholds must satisfy: entry >= exit")

    strat = str(inp.strategy or "ma_filter").strip().lower()

    if data_override is not None:
        dates = data_override["dates"]
        close_qfq = data_override["close_qfq"].reindex(columns=codes).reindex(dates).ffill()
        close_hfq = data_override["close_hfq"].reindex(columns=codes).reindex(dates).ffill()
        high_qfq_df = data_override.get("high_qfq_df")
        if high_qfq_df is None or high_qfq_df.empty:
            high_qfq_df = pd.DataFrame(index=dates, columns=codes)
        else:
            high_qfq_df = high_qfq_df.reindex(columns=codes).reindex(dates).ffill()
        low_qfq_df = data_override.get("low_qfq_df")
        if low_qfq_df is None or low_qfq_df.empty:
            low_qfq_df = pd.DataFrame(index=dates, columns=codes)
        else:
            low_qfq_df = low_qfq_df.reindex(columns=codes).reindex(dates).ffill()
        ret_exec = data_override["ret_exec"].reindex(columns=codes).reindex(dates).fillna(0.0).astype(float)
        ret_hfq = data_override["ret_hfq"].reindex(columns=codes).reindex(dates).fillna(0.0).astype(float)
    else:
        if db is None:
            raise ValueError("db is required when data_override is not set")
        need_hist = max(int(inp.sma_window), int(inp.slow_window), int(inp.donchian_entry), int(inp.mom_lookback), int(inp.macd_slow), int(inp.macd_v_atr_window), 20) + 60
        ext_start = inp.start - dt.timedelta(days=int(need_hist) * 2)
        close_none = load_close_prices(db, codes=codes, start=inp.start, end=inp.end, adjust="none").sort_index().ffill()
        close_qfq = load_close_prices(db, codes=codes, start=ext_start, end=inp.end, adjust="qfq").sort_index()
        close_hfq = load_close_prices(db, codes=codes, start=ext_start, end=inp.end, adjust="hfq").sort_index()
        if close_none.empty:
            raise ValueError("no execution price data for given range (none)")
        if not bool(getattr(inp, "dynamic_universe", False)):
            miss = [c for c in codes if c not in close_none.columns or close_none[c].dropna().empty]
            if miss:
                raise ValueError(f"missing execution data (none) for: {miss}")
            first_valid = [close_none[c].first_valid_index() for c in codes if close_none[c].first_valid_index() is not None]
            if not first_valid:
                raise ValueError("no valid first trading date for selected codes")
            common_start = max(first_valid)
            close_none = close_none.loc[common_start:]
            close_qfq = close_qfq.loc[common_start:]
            close_hfq = close_hfq.loc[common_start:]
        dates = close_none.index
        close_qfq = close_qfq.reindex(dates).ffill()
        close_hfq = close_hfq.reindex(dates).ffill()
        high_qfq_df, low_qfq_df = load_high_low_prices(db, codes=codes, start=ext_start, end=inp.end, adjust="qfq")
        high_qfq_df = high_qfq_df.sort_index().reindex(dates).ffill() if not high_qfq_df.empty else pd.DataFrame(index=dates, columns=codes)
        low_qfq_df = low_qfq_df.sort_index().reindex(dates).ffill() if not low_qfq_df.empty else pd.DataFrame(index=dates, columns=codes)

        ohlc_none = load_ohlc_prices(db, codes=codes, start=inp.start, end=inp.end, adjust="none")
        ohlc_hfq = load_ohlc_prices(db, codes=codes, start=inp.start, end=inp.end, adjust="hfq")
        open_none = ohlc_none.get("open", pd.DataFrame()).sort_index().reindex(dates).reindex(columns=codes).ffill()
        close_none_exec = ohlc_none.get("close", pd.DataFrame()).sort_index().reindex(dates).reindex(columns=codes).ffill()
        open_hfq = ohlc_hfq.get("open", pd.DataFrame()).sort_index().reindex(dates).reindex(columns=codes).ffill()
        close_hfq_exec = ohlc_hfq.get("close", pd.DataFrame()).sort_index().reindex(dates).reindex(columns=codes).ffill()
        close_none_f = close_none.reindex(columns=codes).astype(float)
        close_hfq_f = close_hfq.reindex(columns=codes).astype(float)
        if ep == "open":
            px_none_exec = open_none.astype(float).combine_first(close_none_f)
            px_hfq_exec = open_hfq.astype(float).combine_first(close_hfq_f)
            ret_none = px_none_exec.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            ret_hfq_exec = px_hfq_exec.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        elif ep == "close":
            px_none_exec = close_none_exec.astype(float).combine_first(close_none_f)
            px_hfq_exec = close_hfq_exec.astype(float).combine_first(close_hfq_f)
            ret_none = px_none_exec.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            ret_hfq_exec = px_hfq_exec.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        else:
            ret_open_none = open_none.astype(float).combine_first(close_none_f).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            ret_close_none = close_none_exec.astype(float).combine_first(close_none_f).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            ret_none = (0.5 * (ret_open_none + ret_close_none)).astype(float)
            ret_open_hfq = open_hfq.astype(float).combine_first(close_hfq_f).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            ret_close_hfq = close_hfq_exec.astype(float).combine_first(close_hfq_f).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            ret_hfq_exec = (0.5 * (ret_open_hfq + ret_close_hfq)).astype(float)
        ret_hfq = close_hfq[codes].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        gross_none = 1.0 + ret_none
        gross_hfq = 1.0 + ret_hfq
        corp_factor = (gross_hfq / gross_none).replace([np.inf, -np.inf], np.nan)
        ca_mask = ((corp_factor - 1.0).abs() > 0.02) | (corp_factor > 1.2) | (corp_factor < 1.0 / 1.2)
        ret_exec = ret_none.where(~ca_mask.fillna(False), other=ret_hfq_exec).astype(float)

    sig_pos = pd.DataFrame(index=dates, columns=codes, dtype=float)
    sig_score = pd.DataFrame(index=dates, columns=codes, dtype=float)
    atr_stop_by_asset: dict[str, dict[str, Any]] = {}
    for c in codes:
        px = close_qfq[c].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
        if px.dropna().empty:
            sig_pos[c] = 0.0
            sig_score[c] = np.nan
            continue
        if strat == "ma_filter":
            ma = _moving_average(px, window=int(inp.sma_window), ma_type=ma_type)
            pos = (px > ma).astype(float)
            score = (px / ma - 1.0).astype(float)
        elif strat == "ma_cross":
            fast = _moving_average(px, window=int(inp.fast_window), ma_type=ma_type)
            slow = _moving_average(px, window=int(inp.slow_window), ma_type=ma_type)
            pos = (fast > slow).astype(float)
            score = (fast / slow - 1.0).astype(float)
        elif strat == "donchian":
            pos = _pos_from_donchian(px, entry=int(inp.donchian_entry), exit_=int(inp.donchian_exit))
            hi = px.shift(1).rolling(window=max(2, int(inp.donchian_entry)), min_periods=max(2, int(inp.donchian_entry))).max()
            score = (px / hi - 1.0).astype(float)
        elif strat == "linreg_slope":
            n = int(inp.sma_window)
            y = np.log(px.clip(lower=1e-12).astype(float))
            slope = y.rolling(window=n, min_periods=max(2, n // 2)).apply(_rolling_linreg_slope, raw=True)
            pos = (slope > 0.0).astype(float)
            score = slope.astype(float)
        elif strat == "bias":
            b_win = int(inp.bias_ma_window)
            ema = _ema(px, b_win)
            bias = (np.log(px.clip(lower=1e-12)) - np.log(ema.clip(lower=1e-12))) * 100.0
            pos = (bias > float(inp.bias_entry)).astype(float)
            score = bias.astype(float)
        elif strat == "macd_cross":
            macd, sig, _ = _macd_core(px, fast=int(inp.macd_fast), slow=int(inp.macd_slow), signal=int(inp.macd_signal))
            pos = (macd > sig).astype(float)
            score = (macd - sig).astype(float)
        elif strat == "macd_zero_filter":
            macd, _, _ = _macd_core(px, fast=int(inp.macd_fast), slow=int(inp.macd_slow), signal=int(inp.macd_signal))
            pos = (macd > 0.0).astype(float)
            score = macd.astype(float)
        elif strat == "macd_v":
            macd, _, _ = _macd_core(px, fast=int(inp.macd_fast), slow=int(inp.macd_slow), signal=int(inp.macd_signal))
            h = high_qfq_df[c] if (c in high_qfq_df.columns) else px
            l = low_qfq_df[c] if (c in low_qfq_df.columns) else px
            atr = _atr_from_hlc(h.astype(float).fillna(px), l.astype(float).fillna(px), px, window=int(inp.macd_v_atr_window))
            macd_v = (macd / atr.replace(0.0, np.nan)) * float(inp.macd_v_scale)
            macd_v_sig = _ema(macd_v, int(inp.macd_signal))
            pos = (macd_v > macd_v_sig).astype(float)
            score = (macd_v - macd_v_sig).astype(float)
        else:
            mom = px / px.shift(int(inp.mom_lookback)) - 1.0
            pos = _pos_from_tsmom(
                mom,
                entry_threshold=float(inp.tsmom_entry_threshold),
                exit_threshold=float(inp.tsmom_exit_threshold),
            )
            score = mom.astype(float)
        h = high_qfq_df[c] if (c in high_qfq_df.columns) else px
        l = low_qfq_df[c] if (c in low_qfq_df.columns) else px
        pos, one_stop_stats = _apply_atr_stop(
            pos.fillna(0.0).astype(float),
            close=px,
            high=h.astype(float).fillna(px),
            low=l.astype(float).fillna(px),
            mode=atr_mode,
            atr_window=int(inp.atr_stop_window),
            n_mult=float(inp.atr_stop_n),
            m_step=float(inp.atr_stop_m),
        )
        sig_pos[c] = pos.fillna(0.0)
        sig_score[c] = score.replace([np.inf, -np.inf], np.nan)
        atr_stop_by_asset[str(c)] = one_stop_stats

    vol_ann = ret_hfq.rolling(window=int(inp.vol_window), min_periods=max(3, int(inp.vol_window) // 2)).std(ddof=1) * np.sqrt(252.0)
    w_decision = pd.DataFrame(0.0, index=dates, columns=codes, dtype=float)
    holdings: list[dict[str, Any]] = []
    prev_key: tuple[str, ...] | None = None
    for d in dates:
        active = sig_pos.loc[d]
        scores = sig_score.loc[d].where(active > 0.0, other=np.nan).replace([np.inf, -np.inf], np.nan)
        active_codes = [str(c) for c in scores.dropna().sort_values(ascending=False).index.tolist()]
        picks = active_codes
        if not picks:
            w_row = pd.Series(0.0, index=codes, dtype=float)
        elif ps == "equal":
            per = 1.0 / float(len(picks))
            w_row = pd.Series(0.0, index=codes, dtype=float)
            for c in picks:
                w_row.loc[c] = per
        else:
            inv: dict[str, float] = {}
            for c in picks:
                try:
                    av = float(vol_ann.loc[d, c])
                except (TypeError, ValueError, KeyError):
                    av = float("nan")
                if (not np.isfinite(av)) or av <= 0:
                    inv[c] = 0.0
                else:
                    inv[c] = 1.0 / av
            den = float(sum(inv.values()))
            w_row = pd.Series(0.0, index=codes, dtype=float)
            if den > 0:
                raw = {c: float(v) / den for c, v in inv.items()}
                # portfolio target-vol scalar
                port_vol = float(np.sqrt(np.sum([(raw[c] ** 2) * ((float(vol_ann.loc[d, c]) if np.isfinite(float(vol_ann.loc[d, c])) else 0.0) ** 2) for c in picks])))
                scale = 1.0 if port_vol <= 1e-12 else min(1.0, float(inp.vol_target_ann) / port_vol)
                for c in picks:
                    w_row.loc[c] = raw[c] * scale
            else:
                per = 1.0 / float(len(picks))
                for c in picks:
                    w_row.loc[c] = per
        w_decision.loc[d] = w_row.to_numpy(dtype=float)
        key = tuple(sorted([c for c in picks]))
        if key != prev_key:
            holdings.append(
                {
                    "decision_date": d.date().isoformat(),
                    "picks": list(key),
                    "scores": {c: (None if pd.isna(scores.get(c)) else float(scores.get(c))) for c in key},
                }
            )
            prev_key = key

    # Weights become effective on execution day; return uses forward interval (t -> t+1).
    w = w_decision.shift(1).fillna(0.0).astype(float)
    ret_exec_fwd = forward_align_returns(ret_exec).astype(float)
    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1) / 2.0
    cost = turnover * (float(inp.cost_bps) / 10000.0)
    port_ret = (w * ret_exec_fwd).sum(axis=1) - cost
    nav = (1.0 + port_ret).cumprod()
    if len(nav) > 0:
        nav.iloc[0] = 1.0

    bench_ret = ret_hfq.mean(axis=1).fillna(0.0)
    bench_nav = (1.0 + bench_ret).cumprod()
    if len(bench_nav) > 0:
        bench_nav.iloc[0] = 1.0
    active = (port_ret - bench_ret).astype(float)
    ex_nav = (1.0 + active).cumprod()
    if len(ex_nav) > 0:
        ex_nav.iloc[0] = 1.0

    m_strat = {
        "cumulative_return": float(nav.iloc[-1] - 1.0),
        "annualized_return": float(_annualized_return(nav, ann_factor=TRADING_DAYS_PER_YEAR)),
        "annualized_volatility": float(_annualized_vol(port_ret, ann_factor=TRADING_DAYS_PER_YEAR)),
        "max_drawdown": float(_max_drawdown(nav)),
        "max_drawdown_recovery_days": int(_max_drawdown_duration_days(nav)),
        "sharpe_ratio": float(_sharpe(port_ret, rf=float(inp.risk_free_rate), ann_factor=TRADING_DAYS_PER_YEAR)),
        "sortino_ratio": float(_sortino(port_ret, rf=float(inp.risk_free_rate), ann_factor=TRADING_DAYS_PER_YEAR)),
        "ulcer_index": float(_ulcer_index(nav, in_percent=True)),
        "avg_daily_turnover": float(turnover.mean()),
    }
    m_bench = {
        "cumulative_return": float(bench_nav.iloc[-1] - 1.0),
        "annualized_return": float(_annualized_return(bench_nav, ann_factor=TRADING_DAYS_PER_YEAR)),
        "annualized_volatility": float(_annualized_vol(bench_ret, ann_factor=TRADING_DAYS_PER_YEAR)),
        "max_drawdown": float(_max_drawdown(bench_nav)),
        "max_drawdown_recovery_days": int(_max_drawdown_duration_days(bench_nav)),
        "sharpe_ratio": float(_sharpe(bench_ret, rf=float(inp.risk_free_rate), ann_factor=TRADING_DAYS_PER_YEAR)),
        "sortino_ratio": float(_sortino(bench_ret, rf=float(inp.risk_free_rate), ann_factor=TRADING_DAYS_PER_YEAR)),
        "ulcer_index": float(_ulcer_index(bench_nav, in_percent=True)),
    }
    m_ex = {
        "cumulative_return": float(ex_nav.iloc[-1] - 1.0),
        "annualized_return": float(_annualized_return(ex_nav, ann_factor=TRADING_DAYS_PER_YEAR)),
        "information_ratio": float(_information_ratio(active, ann_factor=TRADING_DAYS_PER_YEAR)),
    }

    total_triggers = int(sum(int((v or {}).get("trigger_count", 0)) for v in atr_stop_by_asset.values()))
    uniq_trigger_dates = sorted(
        {
            str(d)
            for v in atr_stop_by_asset.values()
            for d in (v or {}).get("trigger_dates", [])
            if str(d).strip()
        }
    )

    return {
        "meta": {
            "type": "trend_portfolio_backtest",
            "codes": codes,
            "start": inp.start.strftime("%Y%m%d"),
            "end": inp.end.strftime("%Y%m%d"),
            "strategy": strat,
            "strategy_execution_description": TREND_STRATEGY_EXECUTION_DESCRIPTIONS.get(strat, ""),
            "params": {
                "position_sizing": ps,
                "vol_window": int(inp.vol_window),
                "vol_target_ann": float(inp.vol_target_ann),
                "selection_mode": "all_active_candidates",
                "ma_type": ma_type,
                "mom_lookback": int(inp.mom_lookback),
                "tsmom_entry_threshold": float(inp.tsmom_entry_threshold),
                "tsmom_exit_threshold": float(inp.tsmom_exit_threshold),
                "atr_stop_mode": str(atr_mode),
                "atr_stop_window": int(inp.atr_stop_window),
                "atr_stop_n": float(inp.atr_stop_n),
                "atr_stop_m": float(inp.atr_stop_m),
                "exec_price": str(ep),
            },
        },
        "nav": {
            "dates": nav.index.date.astype(str).tolist(),
            "series": {
                "STRAT": nav.astype(float).tolist(),
                "BUY_HOLD_EW": bench_nav.astype(float).tolist(),
                "EXCESS": ex_nav.astype(float).tolist(),
            },
        },
        "weights": {
            "dates": w.index.date.astype(str).tolist(),
            "series": {c: w[c].astype(float).tolist() for c in codes},
        },
        "holdings": holdings,
        "metrics": {"strategy": m_strat, "benchmark": m_bench, "excess": m_ex},
        "risk_controls": {
            "atr_stop": {
                "enabled": bool(atr_mode != "none"),
                "mode": str(atr_mode),
                "trigger_count": total_triggers,
                "trigger_days": int(len(uniq_trigger_dates)),
                "first_trigger_date": (uniq_trigger_dates[0] if uniq_trigger_dates else None),
                "last_trigger_date": (uniq_trigger_dates[-1] if uniq_trigger_dates else None),
                "trigger_dates": uniq_trigger_dates[:200],
                "by_asset": atr_stop_by_asset,
            }
        },
    }