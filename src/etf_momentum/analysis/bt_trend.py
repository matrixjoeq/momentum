from __future__ import annotations

# pylint: disable=broad-exception-caught,cell-var-from-loop,unused-variable,possibly-unused-variable

import datetime as dt
import math
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from .baseline import (
    _compute_return_risk_contributions,
    _annualized_return,
    _annualized_vol,
    _information_ratio,
    _max_drawdown,
    _max_drawdown_duration_days,
    _rolling_drawdown,
    _sharpe,
    _sortino,
    _ulcer_index,
    hfq_close_daily_equal_weight_returns,
    load_close_prices,
    load_high_low_prices,
    load_ohlc_prices,
)
from .event_study import compute_event_study, entry_dates_from_exposure
from .market_regime import build_market_regime_report
from .r_multiple import build_trade_mfe_r_distribution, enrich_trades_with_r_metrics
from .execution_timing import corporate_action_mask, slippage_return_from_turnover

Session = Any

try:
    import talib
except Exception:  # noqa: BLE001
    talib = None

_SUPPORTED_STRATEGIES = {
    "ma_filter",
    "ma_cross",
    "donchian",
    "tsmom",
    "linreg_slope",
    "bias",
    "macd_cross",
    "macd_zero_filter",
    "macd_v",
    "random_entry",
}

TREND_STRATEGY_EXECUTION_DESCRIPTIONS: dict[str, str] = {
    "ma_filter": "信号在 T 日收盘后根据价格与均线(SMA/EMA/KAMA)关系确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "ma_cross": "信号在 T 日收盘后根据快慢线金叉/死叉确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "donchian": "信号在 T 日收盘后根据是否突破通道上轨/下轨确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "tsmom": "信号在 T 日收盘后根据回看期收益与动量入/出场阈值确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "linreg_slope": "信号在 T 日收盘后根据回看窗口线性回归斜率正负确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "bias": "信号在 T 日收盘后根据 BIAS 与进出场阈值确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "macd_cross": "信号在 T 日收盘后根据 MACD 与信号线金叉/死叉确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "macd_zero_filter": "信号在 T 日收盘后根据 MACD 是否大于零确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "macd_v": "信号在 T 日收盘后根据 ATR 归一化 MACD 与信号线关系确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "random_entry": "信号在 T 日收盘后仅当当前无持仓时抛硬币随机决定是否入场（1=入场，0=空仓）；入场后按持有交易日数到期离场，均在 T+1 日执行；策略收益不包含决策日(T日)当日收益。",
}

DEFAULT_R_TAKE_PROFIT_TIERS: list[dict[str, float]] = [
    {"r_multiple": 2.0, "retrace_ratio": 0.50},
    {"r_multiple": 3.0, "retrace_ratio": 0.30},
    {"r_multiple": 4.0, "retrace_ratio": 0.20},
    {"r_multiple": 5.0, "retrace_ratio": 0.10},
    {"r_multiple": 6.0, "retrace_ratio": 0.05},
]


def _talib_enabled() -> bool:
    return talib is not None


def _as_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _as_float_like(x: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.apply(lambda col: pd.to_numeric(col, errors="coerce").astype(float))
    return _as_float_series(x)


def _forward_simple_return(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    x = _as_float_like(s)
    return (
        (x.shift(-1).div(x) - 1.0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )


def _ratio_simple_return(
    numer: pd.Series | pd.DataFrame,
    denom: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    n = _as_float_like(numer)
    d = _as_float_like(denom).reindex(n.index)
    if isinstance(n, pd.DataFrame) and isinstance(d, pd.DataFrame):
        d = d.reindex(columns=n.columns)
    return (n.div(d) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def _clone_like_input(inp: Any, **overrides: Any) -> Any:
    if hasattr(inp, "__dataclass_fields__"):
        base = asdict(inp)
    else:
        base = dict(vars(inp))
    base.update(overrides)
    return SimpleNamespace(**base)


def _stable_code_seed(code: str) -> int:
    v = 0
    for i, ch in enumerate(str(code)):
        v = (v + (i + 1) * ord(ch)) % 2_147_483_647
    return int(v)


def _pos_from_random_entry_hold(
    index: pd.Index, *, hold_days: int, seed: int | None
) -> pd.Series:
    hold_n = max(1, int(hold_days))
    rng = np.random.default_rng() if seed is None else np.random.default_rng(int(seed))
    out = np.zeros(len(index), dtype=float)
    in_pos = False
    days_left = 0
    for i in range(len(index)):
        if in_pos:
            out[i] = 1.0
            days_left -= 1
            if days_left <= 0:
                in_pos = False
            continue
        toss = int(rng.integers(0, 2))
        if toss == 1:
            in_pos = True
            days_left = hold_n
            out[i] = 1.0
            days_left -= 1
            if days_left <= 0:
                in_pos = False
    return pd.Series(out, index=index, dtype=float)


def _ema_fallback(s: pd.Series, window: int) -> pd.Series:
    x = _as_float_series(s)
    w = max(2, int(window))
    return x.ewm(span=w, adjust=False, min_periods=max(2, w // 2)).mean()


def _kama_fallback(
    s: pd.Series,
    *,
    er_window: int = 10,
    fast_window: int = 2,
    slow_window: int = 30,
) -> pd.Series:
    p = _as_float_series(s)
    er_w = max(2, int(er_window))
    fast_w = max(1, int(fast_window))
    slow_w = max(2, int(slow_window))
    if fast_w >= slow_w:
        fast_w = max(1, slow_w - 1)
    change = _momentum_delta(p, er_w).abs().astype(float)
    abs_diff = _momentum_delta(p, 1).abs().astype(float)
    vol_legacy = abs_diff.rolling(window=er_w, min_periods=er_w).sum()
    volatility = _talib_unary_series(
        abs_diff,
        "SUM",
        fallback=vol_legacy,
        kwargs={"timeperiod": er_w},
    )
    er = (change / volatility.replace(0.0, np.nan)).clip(lower=0.0, upper=1.0)
    fast_sc = 2.0 / (float(fast_w) + 1.0)
    slow_sc = 2.0 / (float(slow_w) + 1.0)
    sc = ((er * (fast_sc - slow_sc) + slow_sc) ** 2).astype(float)
    out = np.full(len(p), np.nan, dtype=float)
    vals = p.to_numpy(dtype=float)
    sc_vals = sc.to_numpy(dtype=float)
    first_idx = None
    for i, v in enumerate(vals):
        if np.isfinite(v):
            first_idx = i
            break
    if first_idx is None:
        return pd.Series(out, index=p.index, dtype=float)
    out[first_idx] = float(vals[first_idx])
    for i in range(first_idx + 1, len(vals)):
        v = vals[i]
        prev = out[i - 1]
        if not np.isfinite(v):
            out[i] = prev
            continue
        if not np.isfinite(prev):
            out[i] = v
            continue
        alpha = sc_vals[i] if np.isfinite(sc_vals[i]) else (slow_sc * slow_sc)
        out[i] = float(prev + alpha * (v - prev))
    return pd.Series(out, index=p.index, dtype=float)


def _moving_average_fallback(
    s: pd.Series,
    *,
    window: int,
    ma_type: str,
    kama_er_window: int = 10,
    kama_fast_window: int = 2,
    kama_slow_window: int = 30,
) -> pd.Series:
    t = str(ma_type or "sma").strip().lower()
    if t == "ema":
        return _ema_fallback(s, int(window))
    if t == "kama":
        return _kama_fallback(
            s,
            er_window=int(kama_er_window),
            fast_window=int(kama_fast_window),
            slow_window=int(kama_slow_window),
        )
    w = max(2, int(window))
    return _rolling_sma(
        _as_float_series(s),
        window=w,
        min_periods=max(2, w // 2),
    )


def _macd_core_fallback(
    close: pd.Series, *, fast: int, slow: int, signal: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    x = _as_float_series(close)
    ema_fast = _ema_fallback(x, int(fast))
    ema_slow = _ema_fallback(x, int(slow))
    macd = (ema_fast - ema_slow).astype(float)
    sig = _ema_fallback(macd, int(signal)).astype(float)
    hist = (macd - sig).astype(float)
    return macd, sig, hist


def _atr_from_hlc_fallback(
    high: pd.Series, low: pd.Series, close: pd.Series, *, window: int
) -> pd.Series:
    h = _as_float_series(high)
    l = _as_float_series(low).reindex(h.index).combine_first(h)  # noqa: E741
    c = _as_float_series(close).reindex(h.index).ffill()
    prev_close = c.shift(1)
    tr = pd.concat(
        [
            (h - l).abs(),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    w = max(2, int(window))
    return (
        tr.ewm(alpha=1.0 / float(w), adjust=False, min_periods=w).mean().astype(float)
    )


def _rolling_linreg_slope_raw(y: np.ndarray) -> float:
    yy = np.asarray(y, dtype=float)
    if yy.size < 2:
        return float("nan")
    if not np.all(np.isfinite(yy)):
        return float("nan")
    x = np.arange(yy.size, dtype=float)
    x = x - x.mean()
    y0 = yy - yy.mean()
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y0) / denom)


def _prefer_legacy_on_diff(
    out: pd.Series,
    legacy: pd.Series,
    *,
    tol: float = 1e-12,
) -> pd.Series:
    o = _as_float_series(out).reindex(legacy.index)
    l = _as_float_series(legacy)  # noqa: E741
    merged = o.combine_first(l)
    mask = (
        merged.notna()
        & l.notna()
        & np.isfinite(merged.to_numpy(dtype=float))
        & np.isfinite(l.to_numpy(dtype=float))
        & ((merged - l).abs() > float(tol))
    )
    if bool(mask.any()):
        merged.loc[mask] = l.loc[mask]
    return merged.astype(float)


def _talib_unary_series(
    s: pd.Series,
    func_name: str,
    *,
    fallback: pd.Series | None = None,
    kwargs: dict[str, Any] | None = None,
) -> pd.Series:
    x = _as_float_series(s)
    if not _talib_enabled():
        return (_as_float_series(fallback) if fallback is not None else x).astype(float)
    fn = getattr(talib, str(func_name), None)
    if not callable(fn):
        return (_as_float_series(fallback) if fallback is not None else x).astype(float)
    try:
        out = fn(x.to_numpy(dtype=float), **(kwargs or {}))  # pylint: disable=not-callable
        out_s = pd.Series(out, index=x.index, dtype=float)
        if fallback is not None:
            out_s = _prefer_legacy_on_diff(out_s, _as_float_series(fallback))
        return out_s.astype(float)
    except Exception:  # noqa: BLE001
        return (_as_float_series(fallback) if fallback is not None else x).astype(float)


def _ema(s: pd.Series, window: int) -> pd.Series:
    w = max(2, int(window))
    legacy = _ema_fallback(_as_float_series(s), w)
    return _talib_unary_series(s, "EMA", fallback=legacy, kwargs={"timeperiod": w})


def _moving_average(
    s: pd.Series,
    *,
    window: int,
    ma_type: str,
    kama_er_window: int = 10,
    kama_fast_window: int = 2,
    kama_slow_window: int = 30,
) -> pd.Series:
    t = str(ma_type or "sma").strip().lower()
    if t == "ema":
        return _ema(s, int(window))
    if t == "kama":
        # TA-Lib KAMA only exposes timeperiod; when custom fast/slow is used,
        # keep legacy implementation to preserve exact semantics.
        if (
            _talib_enabled()
            and int(kama_fast_window) == 2
            and int(kama_slow_window) == 30
        ):
            legacy = _moving_average_fallback(
                s,
                window=int(window),
                ma_type=t,
                kama_er_window=int(kama_er_window),
                kama_fast_window=int(kama_fast_window),
                kama_slow_window=int(kama_slow_window),
            )
            return _talib_unary_series(
                s,
                "KAMA",
                fallback=legacy,
                kwargs={"timeperiod": max(2, int(kama_er_window))},
            )
        return _moving_average_fallback(
            s,
            window=int(window),
            ma_type=t,
            kama_er_window=int(kama_er_window),
            kama_fast_window=int(kama_fast_window),
            kama_slow_window=int(kama_slow_window),
        )
    w = max(2, int(window))
    legacy = _moving_average_fallback(
        s,
        window=w,
        ma_type="sma",
        kama_er_window=int(kama_er_window),
        kama_fast_window=int(kama_fast_window),
        kama_slow_window=int(kama_slow_window),
    )
    return _talib_unary_series(s, "SMA", fallback=legacy, kwargs={"timeperiod": w})


def _macd_core(
    close: pd.Series, *, fast: int, slow: int, signal: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    x = _as_float_series(close)
    legacy = _macd_core_fallback(x, fast=int(fast), slow=int(slow), signal=int(signal))
    if not _talib_enabled():
        return legacy
    macd_fn = getattr(talib, "MACD", None)
    if not callable(macd_fn):
        return legacy
    try:
        m, s, h = macd_fn(  # pylint: disable=not-callable
            x.to_numpy(dtype=float),
            fastperiod=max(2, int(fast)),
            slowperiod=max(2, int(slow)),
            signalperiod=max(1, int(signal)),
        )
        macd = _prefer_legacy_on_diff(
            pd.Series(m, index=x.index, dtype=float), legacy[0].astype(float)
        )
        sig = _prefer_legacy_on_diff(
            pd.Series(s, index=x.index, dtype=float), legacy[1].astype(float)
        )
        hist = _prefer_legacy_on_diff(
            pd.Series(h, index=x.index, dtype=float), legacy[2].astype(float)
        )
        return macd.astype(float), sig.astype(float), hist.astype(float)
    except Exception:  # noqa: BLE001
        return legacy


def _atr_from_hlc(
    high: pd.Series, low: pd.Series, close: pd.Series, *, window: int
) -> pd.Series:
    h = _as_float_series(high)
    l = _as_float_series(low).reindex(h.index).combine_first(h)  # noqa: E741
    c = _as_float_series(close).reindex(h.index).ffill()
    w = max(2, int(window))
    legacy = _atr_from_hlc_fallback(h, l, c, window=w)
    if not _talib_enabled():
        return legacy
    atr_fn = getattr(talib, "ATR", None)
    if not callable(atr_fn):
        return legacy
    try:
        out = atr_fn(  # pylint: disable=not-callable
            h.to_numpy(dtype=float),
            l.to_numpy(dtype=float),
            c.to_numpy(dtype=float),
            timeperiod=w,
        )
        out_s = _prefer_legacy_on_diff(
            pd.Series(out, index=h.index, dtype=float), legacy.astype(float)
        )
        return out_s.astype(float)
    except Exception:  # noqa: BLE001
        return legacy


def _rolling_linreg_slope(s: pd.Series, window: int) -> pd.Series:
    x = _as_float_series(s)
    n = max(2, int(window))
    legacy = x.rolling(window=n, min_periods=max(2, n // 2)).apply(
        _rolling_linreg_slope_raw, raw=True
    )
    if not _talib_enabled():
        return legacy.astype(float)
    lr_fn = getattr(talib, "LINEARREG_SLOPE", None)
    if not callable(lr_fn):
        return legacy.astype(float)
    try:
        out = lr_fn(x.to_numpy(dtype=float), timeperiod=n)  # pylint: disable=not-callable
        out_s = _prefer_legacy_on_diff(
            pd.Series(out, index=x.index, dtype=float), legacy.astype(float)
        )
        return out_s.astype(float)
    except Exception:  # noqa: BLE001
        return legacy.astype(float)


def _donchian_prev_high(s: pd.Series, entry: int) -> pd.Series:
    x = _as_float_series(s).shift(1)
    n = max(2, int(entry))
    legacy = x.rolling(window=n, min_periods=n).max()
    if not _talib_enabled():
        return legacy.astype(float)
    return _talib_unary_series(x, "MAX", fallback=legacy, kwargs={"timeperiod": n})


def _donchian_prev_low(s: pd.Series, exit_: int) -> pd.Series:
    x = _as_float_series(s).shift(1)
    n = max(2, int(exit_))
    legacy = x.rolling(window=n, min_periods=n).min()
    if not _talib_enabled():
        return legacy.astype(float)
    return _talib_unary_series(x, "MIN", fallback=legacy, kwargs={"timeperiod": n})


def _tsmom_rocp(s: pd.Series, lookback: int) -> pd.Series:
    x = _as_float_series(s)
    n = max(1, int(lookback))
    legacy = (x / x.shift(n) - 1.0).replace([np.inf, -np.inf], np.nan)
    if not _talib_enabled():
        return legacy.astype(float)
    return _talib_unary_series(x, "ROCP", fallback=legacy, kwargs={"timeperiod": n})


def _momentum_delta(s: pd.Series, periods: int = 1) -> pd.Series:
    x = _as_float_series(s)
    n = max(1, int(periods))
    legacy = x.diff(n)
    if not _talib_enabled():
        return legacy.astype(float)
    mom_fn = getattr(talib, "MOM", None)
    if not callable(mom_fn):
        return legacy.astype(float)
    try:
        out = mom_fn(x.to_numpy(dtype=float), timeperiod=n)  # pylint: disable=not-callable
        out_s = pd.Series(out, index=x.index, dtype=float)
        return _prefer_legacy_on_diff(out_s, legacy.astype(float))
    except Exception:  # noqa: BLE001
        return legacy.astype(float)


def _rolling_std(
    s: pd.Series,
    *,
    window: int,
    min_periods: int,
    ddof: int = 1,
) -> pd.Series:
    x = _as_float_series(s)
    w = max(2, int(window))
    mp = max(1, int(min_periods))
    legacy = x.rolling(window=w, min_periods=mp).std(ddof=int(ddof))
    if not _talib_enabled():
        return legacy.astype(float)
    stddev_fn = getattr(talib, "STDDEV", None)
    if not callable(stddev_fn):
        return legacy.astype(float)
    try:
        out = stddev_fn(x.to_numpy(dtype=float), timeperiod=w, nbdev=1)  # pylint: disable=not-callable
        out_s = pd.Series(out, index=x.index, dtype=float)
        if int(ddof) == 1 and w > 1:
            out_s = out_s * float(np.sqrt(float(w) / float(w - 1)))
        # Preserve legacy warmup/min_periods behavior and exact parity.
        return _prefer_legacy_on_diff(out_s, legacy.astype(float))
    except Exception:  # noqa: BLE001
        return legacy.astype(float)


def _rolling_sma(s: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    x = _as_float_series(s)
    w = max(2, int(window))
    mp = max(1, int(min_periods))
    legacy = x.rolling(window=w, min_periods=mp).mean()
    if not _talib_enabled():
        return legacy.astype(float)
    sma_fn = getattr(talib, "SMA", None)
    if not callable(sma_fn):
        return legacy.astype(float)
    try:
        out = sma_fn(x.to_numpy(dtype=float), timeperiod=w)  # pylint: disable=not-callable
        out_s = pd.Series(out, index=x.index, dtype=float)
        return _prefer_legacy_on_diff(out_s, legacy.astype(float))
    except Exception:  # noqa: BLE001
        return legacy.astype(float)


def _efficiency_ratio(price: pd.Series, *, window: int) -> pd.Series:
    """
    Kaufman ER with TA-Lib-assisted volatility sum when available.
    ER_t = |P_t - P_{t-window}| / SUM(|ΔP|, window)
    """
    w = max(2, int(window))
    p = _as_float_series(price)
    change = _momentum_delta(p, w).abs().astype(float)
    abs_diff = _momentum_delta(p, 1).abs().astype(float)
    vol_legacy = abs_diff.rolling(window=w, min_periods=w).sum()
    if _talib_enabled():
        sum_fn = getattr(talib, "SUM", None)
        if not callable(sum_fn):
            volatility = vol_legacy.astype(float)
        else:
            try:
                vol_ta = pd.Series(
                    sum_fn(abs_diff.to_numpy(dtype=float), timeperiod=w),  # pylint: disable=not-callable
                    index=p.index,
                    dtype=float,
                )
                volatility = _prefer_legacy_on_diff(vol_ta, vol_legacy.astype(float))
            except Exception:  # noqa: BLE001
                volatility = vol_legacy.astype(float)
    else:
        volatility = vol_legacy.astype(float)
    out = (
        (change / volatility.replace(0.0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .clip(lower=0.0, upper=1.0)
    )
    return out.astype(float)


def _pos_from_donchian(close: pd.Series, *, entry: int, exit_: int) -> pd.Series:
    px = _as_float_series(close)
    hi = _donchian_prev_high(px, int(entry))
    lo = _donchian_prev_low(px, int(exit_))
    pos = np.zeros(len(px), dtype=float)
    in_pos = False
    for i in range(len(px)):
        c = float(px.iloc[i]) if np.isfinite(float(px.iloc[i])) else float("nan")
        if not np.isfinite(c):
            pos[i] = 1.0 if in_pos else 0.0
            continue
        h = hi.iloc[i]
        low_v = lo.iloc[i]
        if (not in_pos) and pd.notna(h) and c > float(h):
            in_pos = True
        elif in_pos and pd.notna(low_v) and c < float(low_v):
            in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=px.index, dtype=float)


def _pos_from_tsmom(
    mom: pd.Series,
    *,
    entry_threshold: float,
    exit_threshold: float,
) -> pd.Series:
    m = _as_float_series(mom)
    ent = float(entry_threshold)
    ex = float(exit_threshold)
    pos = np.zeros(len(m), dtype=float)
    in_pos = False
    for i, v in enumerate(m.to_numpy(dtype=float)):
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
    return pd.Series(pos, index=m.index, dtype=float)


def _pos_from_band(
    price: pd.Series,
    center: pd.Series,
    *,
    band: pd.Series,
) -> pd.Series:
    px = _as_float_series(price)
    cc = _as_float_series(center).reindex(px.index)
    bb = _as_float_series(band).reindex(px.index).fillna(0.0)
    out = np.zeros(len(px), dtype=float)
    in_pos = False
    for i in range(len(px)):
        p = float(px.iloc[i]) if np.isfinite(float(px.iloc[i])) else float("nan")
        c = float(cc.iloc[i]) if np.isfinite(float(cc.iloc[i])) else float("nan")
        b = max(0.0, float(bb.iloc[i])) if np.isfinite(float(bb.iloc[i])) else 0.0
        if (not np.isfinite(p)) or (not np.isfinite(c)):
            out[i] = 1.0 if in_pos else 0.0
            continue
        upper = c + b
        lower = c - b
        if (not in_pos) and p > upper:
            in_pos = True
        elif in_pos and p < lower:
            in_pos = False
        out[i] = 1.0 if in_pos else 0.0
    return pd.Series(out, index=px.index, dtype=float)


def _compute_impulse_state(
    close: pd.Series,
    *,
    ema_window: int = 13,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.Series:
    px = _as_float_series(close)
    ema = _ema(px, int(ema_window))
    _, _, hist = _macd_core(
        px,
        fast=int(macd_fast),
        slow=int(macd_slow),
        signal=int(macd_signal),
    )
    ema_diff = _momentum_delta(ema, 1)
    hist_diff = _momentum_delta(hist, 1)
    state = pd.Series("NEUTRAL", index=px.index, dtype=object)
    bull = (ema_diff > 0.0) & (hist_diff > 0.0)
    bear = (ema_diff < 0.0) & (hist_diff < 0.0)
    state.loc[bull.fillna(False)] = "BULL"
    state.loc[bear.fillna(False)] = "BEAR"
    return state


def _apply_er_entry_filter(
    raw_pos: pd.Series, *, er: pd.Series, threshold: float
) -> tuple[pd.Series, dict[str, int]]:
    """Entry-only ER filter."""
    thr = float(threshold)
    out = np.zeros(len(raw_pos), dtype=float)
    blocked_count = 0
    attempted_entry_count = 0
    allowed_entry_count = 0
    in_pos = False
    idx = raw_pos.index
    for i, d in enumerate(idx):
        desired = float(raw_pos.iloc[i]) if np.isfinite(float(raw_pos.iloc[i])) else 0.0
        desired = max(0.0, desired)
        if not in_pos:
            if desired > 0.0:
                attempted_entry_count += 1
                er_ok = bool(np.isfinite(float(er.loc[d]))) and float(er.loc[d]) >= thr
                if er_ok:
                    in_pos = True
                    allowed_entry_count += 1
                    out[i] = desired
                else:
                    blocked_count += 1
                    out[i] = 0.0
            else:
                out[i] = 0.0
        else:
            if desired <= 0.0:
                in_pos = False
                out[i] = 0.0
            else:
                out[i] = desired
    return pd.Series(out, index=idx, dtype=float), {
        "blocked_entry_count": int(blocked_count),
        "attempted_entry_count": int(attempted_entry_count),
        "allowed_entry_count": int(allowed_entry_count),
    }


def _apply_impulse_entry_filter(
    raw_pos: pd.Series,
    *,
    impulse_state: pd.Series,
    allow_bull: bool,
    allow_bear: bool,
    allow_neutral: bool,
) -> tuple[pd.Series, dict[str, int]]:
    """Entry-only filter using impulse states."""
    out = np.zeros(len(raw_pos), dtype=float)
    blocked_count = 0
    attempted_entry_count = 0
    allowed_entry_count = 0
    blocked_by_state = {"BULL": 0, "BEAR": 0, "NEUTRAL": 0}
    in_pos = False
    idx = raw_pos.index
    for i, d in enumerate(idx):
        desired = float(raw_pos.iloc[i]) if np.isfinite(float(raw_pos.iloc[i])) else 0.0
        desired = max(0.0, desired)
        st = str(impulse_state.loc[d]) if d in impulse_state.index else "NEUTRAL"
        st = st if st in {"BULL", "BEAR", "NEUTRAL"} else "NEUTRAL"
        is_allowed = (
            (st == "BULL" and bool(allow_bull))
            or (st == "BEAR" and bool(allow_bear))
            or (st == "NEUTRAL" and bool(allow_neutral))
        )
        if not in_pos:
            if desired > 0.0:
                attempted_entry_count += 1
                if is_allowed:
                    in_pos = True
                    allowed_entry_count += 1
                    out[i] = desired
                else:
                    blocked_count += 1
                    blocked_by_state[st] = int(blocked_by_state.get(st, 0) + 1)
                    out[i] = 0.0
            else:
                out[i] = 0.0
        else:
            if desired <= 0.0:
                in_pos = False
                out[i] = 0.0
            else:
                out[i] = desired
    return pd.Series(out, index=idx, dtype=float), {
        "blocked_entry_count": int(blocked_count),
        "attempted_entry_count": int(attempted_entry_count),
        "allowed_entry_count": int(allowed_entry_count),
        "blocked_entry_count_bull": int(blocked_by_state.get("BULL", 0)),
        "blocked_entry_count_bear": int(blocked_by_state.get("BEAR", 0)),
        "blocked_entry_count_neutral": int(blocked_by_state.get("NEUTRAL", 0)),
    }


def _apply_er_exit_filter(
    raw_pos: pd.Series, *, er: pd.Series, threshold: float
) -> tuple[pd.Series, dict[str, Any]]:
    """Exit-only ER filter."""
    thr = float(threshold)
    out = np.zeros(len(raw_pos), dtype=float)
    trigger_count = 0
    trigger_dates: list[str] = []
    trace_last_rows: list[dict[str, Any]] = []
    in_pos = False
    idx = raw_pos.index
    for i, d in enumerate(idx):
        desired = float(raw_pos.iloc[i]) if np.isfinite(float(raw_pos.iloc[i])) else 0.0
        desired = max(0.0, desired)
        if not in_pos:
            if desired > 0.0:
                in_pos = True
                out[i] = desired
            else:
                out[i] = 0.0
            continue
        er_hit = bool(np.isfinite(float(er.loc[d]))) and float(er.loc[d]) >= thr
        if er_hit:
            in_pos = False
            out[i] = 0.0
            trigger_count += 1
            try:
                trigger_dates.append(pd.Timestamp(d).strftime("%Y%m%d"))
            except (TypeError, ValueError, OverflowError):
                pass
            trace_last_rows.append(
                {
                    "date": (
                        pd.Timestamp(d).date().isoformat()
                        if hasattr(d, "date")
                        else str(d)
                    ),
                    "event_type": "exit",
                    "event_reason": "er_exit_filter",
                    "base_pos": float(desired),
                    "decision_pos": 0.0,
                    "er_value": (
                        float(er.loc[d]) if np.isfinite(float(er.loc[d])) else None
                    ),
                    "er_threshold": float(thr),
                    "er_triggered": True,
                }
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]
        elif desired <= 0.0:
            in_pos = False
            out[i] = 0.0
        else:
            out[i] = desired
    return pd.Series(out, index=idx, dtype=float), {
        "trigger_count": int(trigger_count),
        "trigger_dates": sorted(set(trigger_dates)),
        "trace_last_rows": trace_last_rows[-80:],
    }


def _round_half_up(value: float, ndigits: int = 0) -> float:
    if not np.isfinite(float(value)):
        return float("nan")
    fac = float(10 ** int(max(0, ndigits)))
    x = float(value) * fac
    if x >= 0.0:
        return float(np.floor(x + 0.5) / fac)
    return float(np.ceil(x - 0.5) / fac)


def _bucketize_momentum_series(momentum: pd.Series) -> pd.Series:
    out = pd.Series("NA", index=momentum.index, dtype=object)
    m = pd.to_numeric(momentum, errors="coerce").astype(float) * 100.0
    for d in out.index:
        v = float(m.loc[d]) if d in m.index else float("nan")
        if not np.isfinite(v):
            continue
        iv = int(_round_half_up(v, ndigits=0))
        out.loc[d] = f"{iv:d}%"
    return out


def _bucketize_er_series(er: pd.Series) -> pd.Series:
    out = pd.Series("NA", index=er.index, dtype=object)
    e = pd.to_numeric(er, errors="coerce").astype(float)
    for d in out.index:
        v = float(e.loc[d]) if d in e.index else float("nan")
        if not np.isfinite(v):
            continue
        vv = float(np.clip(v, 0.0, 1.0))
        b = _round_half_up(vv, ndigits=1)
        out.loc[d] = f"{b:.1f}"
    return out


def _bucketize_vol_ratio_series(vol_ratio: pd.Series) -> pd.Series:
    out = pd.Series("NA", index=vol_ratio.index, dtype=object)
    r = pd.to_numeric(vol_ratio, errors="coerce").astype(float)
    for d in out.index:
        v = float(r.loc[d]) if d in r.index else float("nan")
        if not np.isfinite(v):
            continue
        out.loc[d] = f"{_round_half_up(v, ndigits=1):.1f}"
    return out


def _bucketize_impulse_series(impulse_state: pd.Series) -> pd.Series:
    out = pd.Series("NA", index=impulse_state.index, dtype=object)
    for d in out.index:
        st = (
            str(impulse_state.loc[d]).strip().upper()
            if d in impulse_state.index
            else "NA"
        )
        if st in {"BULL", "BEAR", "NEUTRAL"}:
            out.loc[d] = st
    return out


def _rolling_pack(nav_s: pd.Series) -> dict[str, dict[str, Any]]:
    rolling: dict[str, dict[str, Any]] = {
        "returns": {},
        "drawdown": {},
        "max_drawdown": {},
    }
    for weeks in [4, 12, 52]:
        window = int(weeks) * 5
        r = (nav_s / nav_s.shift(window) - 1.0).dropna()
        d = _rolling_drawdown(nav_s, window).dropna()
        rolling["returns"][f"{weeks}w"] = {
            "dates": r.index.date.astype(str).tolist(),
            "values": r.astype(float).tolist(),
        }
        rolling["drawdown"][f"{weeks}w"] = {
            "dates": d.index.date.astype(str).tolist(),
            "values": d.astype(float).tolist(),
        }
        rolling["max_drawdown"][f"{weeks}w"] = {
            "dates": d.index.date.astype(str).tolist(),
            "values": d.astype(float).tolist(),
        }
    for months in [3, 6, 12]:
        window = int(months) * 21
        r = (nav_s / nav_s.shift(window) - 1.0).dropna()
        d = _rolling_drawdown(nav_s, window).dropna()
        rolling["returns"][f"{months}m"] = {
            "dates": r.index.date.astype(str).tolist(),
            "values": r.astype(float).tolist(),
        }
        rolling["drawdown"][f"{months}m"] = {
            "dates": d.index.date.astype(str).tolist(),
            "values": d.astype(float).tolist(),
        }
        rolling["max_drawdown"][f"{months}m"] = {
            "dates": d.index.date.astype(str).tolist(),
            "values": d.astype(float).tolist(),
        }
    for years in [1, 3]:
        window = int(years) * 252
        r = (nav_s / nav_s.shift(window) - 1.0).dropna()
        d = _rolling_drawdown(nav_s, window).dropna()
        rolling["returns"][f"{years}y"] = {
            "dates": r.index.date.astype(str).tolist(),
            "values": r.astype(float).tolist(),
        }
        rolling["drawdown"][f"{years}y"] = {
            "dates": d.index.date.astype(str).tolist(),
            "values": d.astype(float).tolist(),
        }
        rolling["max_drawdown"][f"{years}y"] = {
            "dates": d.index.date.astype(str).tolist(),
            "values": d.astype(float).tolist(),
        }
    return rolling


def _dist_stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(
        [float(x) for x in (values or []) if np.isfinite(float(x))], dtype=float
    )
    if arr.size == 0:
        return {
            "count": 0,
            "max": None,
            "min": None,
            "mean": None,
            "std": None,
            "quantiles": {
                k: None
                for k in ["p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
            },
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


def _trade_stats_from_returns(
    values: list[float], *, flat_eps: float = 1e-12
) -> dict[str, Any]:
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
        if (
            np.isfinite(avg_win)
            and np.isfinite(avg_loss_abs)
            and avg_win > 0.0
            and avg_loss_abs > 0.0
        ):
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


def _series_index_to_date_str(series: pd.Series) -> pd.Series:
    s = series.copy()
    s.index = pd.to_datetime(s.index).date.astype(str)
    return s


def _build_entry_signal_date_map(dates: pd.Index) -> dict[str, str | None]:
    ds = [str(pd.to_datetime(d).date()) for d in dates]
    out: dict[str, str | None] = {}
    for i, d in enumerate(ds):
        out[d] = ds[i - 1] if i > 0 else None
    return out


def _attach_entry_condition_bins_to_trades(
    trades: list[dict[str, Any]],
    *,
    condition_bins_by_code: dict[str, dict[str, pd.Series]],
    dates: pd.Index,
    default_code: str | None = None,
) -> list[dict[str, Any]]:
    prev_map = _build_entry_signal_date_map(dates)
    normalized: dict[str, dict[str, pd.Series]] = {}
    for code, mp in (condition_bins_by_code or {}).items():
        one: dict[str, pd.Series] = {}
        for k, s in (mp or {}).items():
            one[str(k)] = _series_index_to_date_str(
                s if isinstance(s, pd.Series) else pd.Series(dtype=object)
            )
        normalized[str(code)] = one
    out: list[dict[str, Any]] = []
    for tr in list(trades or []):
        row = dict(tr or {})
        code = str(row.get("code") or default_code or "").strip()
        entry_date = str(row.get("entry_date") or "").strip()
        signal_date = prev_map.get(entry_date)
        bins: dict[str, str] = {}
        one_code = normalized.get(code, {})
        for cond in ("momentum", "er", "vol_ratio", "impulse"):
            ss = one_code.get(cond)
            v = "NA"
            if ss is not None and signal_date and (signal_date in ss.index):
                vv = str(ss.loc[signal_date]).strip()
                v = vv if vv else "NA"
            bins[cond] = v
        row["code"] = code
        row["entry_signal_date"] = signal_date
        row["entry_condition_bins"] = bins
        out.append(row)
    return out


def _normal_two_sided_p_from_z(z: float) -> float | None:
    if not np.isfinite(float(z)):
        return None
    return float(max(0.0, min(1.0, math.erfc(abs(float(z)) / math.sqrt(2.0)))))


def _two_proportion_z_test(
    *, wins_a: int, losses_a: int, wins_b: int, losses_b: int
) -> float | None:
    na = int(wins_a) + int(losses_a)
    nb = int(wins_b) + int(losses_b)
    if na <= 0 or nb <= 0:
        return None
    pa = float(wins_a) / float(na)
    pb = float(wins_b) / float(nb)
    p_pool = float(wins_a + wins_b) / float(na + nb)
    denom = p_pool * (1.0 - p_pool) * (1.0 / float(na) + 1.0 / float(nb))
    if denom <= 0.0:
        return None
    z = (pa - pb) / float(np.sqrt(denom))
    return _normal_two_sided_p_from_z(z)


def _welch_t_test_normal_approx(a: list[float], b: list[float]) -> float | None:
    xa = np.asarray([float(x) for x in (a or []) if np.isfinite(float(x))], dtype=float)
    xb = np.asarray([float(x) for x in (b or []) if np.isfinite(float(x))], dtype=float)
    if xa.size < 2 or xb.size < 2:
        return None
    va = float(np.var(xa, ddof=1))
    vb = float(np.var(xb, ddof=1))
    denom = (va / float(xa.size)) + (vb / float(xb.size))
    if (not np.isfinite(denom)) or denom <= 0.0:
        return None
    z = (float(np.mean(xa)) - float(np.mean(xb))) / float(np.sqrt(denom))
    return _normal_two_sided_p_from_z(z)


def _bh_qvalues(ps: list[float | None]) -> list[float | None]:
    pairs = [
        (i, float(p))
        for i, p in enumerate(ps)
        if (p is not None and np.isfinite(float(p)))
    ]
    m = int(len(pairs))
    out: list[float | None] = [None for _ in ps]
    if m <= 0:
        return out
    pairs_sorted = sorted(pairs, key=lambda x: x[1])
    q_raw: list[float] = [0.0] * m
    for rank, (_, p) in enumerate(pairs_sorted, start=1):
        q_raw[rank - 1] = float(min(1.0, p * m / float(rank)))
    q_adj: list[float] = [0.0] * m
    running = 1.0
    for i in range(m - 1, -1, -1):
        running = min(running, q_raw[i])
        q_adj[i] = float(running)
    for i, (orig_idx, _) in enumerate(pairs_sorted):
        out[orig_idx] = q_adj[i]
    return out


def _stratified_permutation_pvalue(
    rows: list[dict[str, Any]],
    *,
    in_bin: str,
    value_key: str,
    strata_key: str,
    n_perm: int,
    seed: int,
) -> float | None:
    vals: list[tuple[str, bool, float]] = []
    for r in rows:
        b = str(r.get("bucket") or "")
        v = r.get(value_key)
        if v is None or (not np.isfinite(float(v))):
            continue
        vals.append((str(r.get(strata_key) or "NA"), b == in_bin, float(v)))
    if not vals:
        return None
    sum_in = float(sum(v for _, f, v in vals if f))
    n_in = int(sum(1 for _, f, _ in vals if f))
    sum_out = float(sum(v for _, f, v in vals if (not f)))
    n_out = int(sum(1 for _, f, _ in vals if (not f)))
    if n_in <= 0 or n_out <= 0:
        return None
    obs = (sum_in / float(n_in)) - (sum_out / float(n_out))
    by_strata: dict[str, list[tuple[bool, float]]] = {}
    for st, f, v in vals:
        by_strata.setdefault(st, []).append((f, v))
    rng = np.random.default_rng(int(seed))
    exceed = 0
    perm_n = int(max(50, n_perm))
    for _ in range(perm_n):
        p_sum_in = 0.0
        p_n_in = 0
        p_sum_out = 0.0
        p_n_out = 0
        for rs in by_strata.values():
            n = int(len(rs))
            if n <= 0:
                continue
            vals_s = np.asarray([float(x[1]) for x in rs], dtype=float)
            k = int(sum(1 for x in rs if x[0]))
            if k <= 0:
                p_sum_out += float(vals_s.sum())
                p_n_out += n
                continue
            if k >= n:
                p_sum_in += float(vals_s.sum())
                p_n_in += n
                continue
            idx_in = rng.choice(n, size=k, replace=False)
            mask = np.zeros(n, dtype=bool)
            mask[idx_in] = True
            p_sum_in += float(vals_s[mask].sum())
            p_n_in += int(mask.sum())
            p_sum_out += float(vals_s[~mask].sum())
            p_n_out += int((~mask).sum())
        if p_n_in <= 0 or p_n_out <= 0:
            continue
        diff = (p_sum_in / float(p_n_in)) - (p_sum_out / float(p_n_out))
        if abs(float(diff)) >= abs(float(obs)) - 1e-18:
            exceed += 1
    return float((exceed + 1) / float(perm_n + 1))


def _sorted_condition_buckets(condition: str, buckets: list[str]) -> list[str]:
    uniq = [str(b) for b in buckets if str(b).strip()]
    if str(condition) == "impulse":
        order = {"BULL": 0, "NEUTRAL": 1, "BEAR": 2, "NA": 99}
        return sorted(set(uniq), key=lambda x: (order.get(str(x).upper(), 50), str(x)))

    def _num_key(x: str) -> tuple[int, float]:
        s = str(x).strip().replace("%", "")
        try:
            return (0, float(s))
        except ValueError:
            return (1, float("inf"))

    return sorted(set(uniq), key=_num_key)


def _stable_seed_from_text(text: str) -> int:
    acc = 0
    for i, ch in enumerate(str(text)):
        acc = (acc + (i + 1) * ord(ch)) % 1_000_003
    return int(acc)


def _build_entry_condition_stats(
    trades: list[dict[str, Any]],
    *,
    by_code: bool,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    closed = []
    for tr in list(trades or []):
        if not bool(tr.get("closed", False)):
            continue
        rr = tr.get("return")
        if rr is None or (not np.isfinite(float(rr))):
            continue
        bins = tr.get("entry_condition_bins") or {}
        if not isinstance(bins, dict):
            bins = {}
        entry_signal_date = str(tr.get("entry_signal_date") or "")
        yyyy = ""
        if entry_signal_date:
            try:
                yyyy = str(pd.to_datetime(entry_signal_date).year)
            except (TypeError, ValueError):
                yyyy = ""
        closed.append(
            {
                "code": str(tr.get("code") or ""),
                "return": float(rr),
                "bucket_map": {k: str(v) for k, v in bins.items()},
                "year": yyyy,
            }
        )
    out: dict[str, Any] = {}
    conditions = ("momentum", "er", "vol_ratio", "impulse")
    for cond in conditions:
        rows: list[dict[str, Any]] = []
        for r in closed:
            b = str((r.get("bucket_map") or {}).get(cond, "NA") or "NA")
            rr = float(r["return"])
            rows.append(
                {
                    "bucket": b,
                    "return": rr,
                    "is_win": 1 if rr > 1e-12 else 0,
                    "is_loss": 1 if rr < -1e-12 else 0,
                    "strata": (
                        f"{r['year']}" if by_code else f"{r['code']}|{r['year']}"
                    ),
                }
            )
        uniq = _sorted_condition_buckets(
            cond, [str(x.get("bucket") or "NA") for x in rows]
        )
        bins_out: list[dict[str, Any]] = []
        p_quasi_win: list[float | None] = []
        p_quasi_ret: list[float | None] = []
        p_strong_win: list[float | None] = []
        p_strong_ret: list[float | None] = []
        for b in uniq:
            in_rows = [x for x in rows if str(x.get("bucket")) == b]
            out_rows = [x for x in rows if str(x.get("bucket")) != b]
            in_rets = [float(x["return"]) for x in in_rows]
            out_rets = [float(x["return"]) for x in out_rows]
            in_nonflat = [
                x for x in in_rows if int(x["is_win"]) == 1 or int(x["is_loss"]) == 1
            ]
            out_nonflat = [
                x for x in out_rows if int(x["is_win"]) == 1 or int(x["is_loss"]) == 1
            ]
            in_wins = int(sum(int(x["is_win"]) for x in in_nonflat))
            in_losses = int(sum(int(x["is_loss"]) for x in in_nonflat))
            out_wins = int(sum(int(x["is_win"]) for x in out_nonflat))
            out_losses = int(sum(int(x["is_loss"]) for x in out_nonflat))
            n_in_eff = int(in_wins + in_losses)
            n_out_eff = int(out_wins + out_losses)
            wr_in = (float(in_wins) / float(n_in_eff)) if n_in_eff > 0 else None
            wr_out = (float(out_wins) / float(n_out_eff)) if n_out_eff > 0 else None
            mean_in = (
                float(np.mean(np.asarray(in_rets, dtype=float))) if in_rets else None
            )
            mean_out = (
                float(np.mean(np.asarray(out_rets, dtype=float))) if out_rets else None
            )
            uplift_win = (
                (wr_in - wr_out) if (wr_in is not None and wr_out is not None) else None
            )
            uplift_ret = (
                (mean_in - mean_out)
                if (mean_in is not None and mean_out is not None)
                else None
            )
            quasi_p_win = _two_proportion_z_test(
                wins_a=in_wins,
                losses_a=in_losses,
                wins_b=out_wins,
                losses_b=out_losses,
            )
            quasi_p_ret = _welch_t_test_normal_approx(in_rets, out_rets)
            p_quasi_win.append(quasi_p_win)
            p_quasi_ret.append(quasi_p_ret)
            win_rows_perm = []
            for x in rows:
                if int(x["is_win"]) == 1 or int(x["is_loss"]) == 1:
                    win_rows_perm.append(
                        {
                            "bucket": str(x["bucket"]),
                            "value": float(x["is_win"]),
                            "strata": str(x["strata"]),
                        }
                    )
            ret_rows_perm = [
                {
                    "bucket": str(x["bucket"]),
                    "value": float(x["return"]),
                    "strata": str(x["strata"]),
                }
                for x in rows
            ]
            strong_p_win = _stratified_permutation_pvalue(
                win_rows_perm,
                in_bin=str(b),
                value_key="value",
                strata_key="strata",
                n_perm=int(n_perm),
                seed=int(seed + _stable_seed_from_text(f"{cond}|{b}|win")),
            )
            strong_p_ret = _stratified_permutation_pvalue(
                ret_rows_perm,
                in_bin=str(b),
                value_key="value",
                strata_key="strata",
                n_perm=int(n_perm),
                seed=int(seed + _stable_seed_from_text(f"{cond}|{b}|ret")),
            )
            p_strong_win.append(strong_p_win)
            p_strong_ret.append(strong_p_ret)
            bins_out.append(
                {
                    "bucket": str(b),
                    "closed_trade_count": int(len(in_rows)),
                    "win_rate_ex_zero": wr_in,
                    "mean_return": mean_in,
                    "quasi_causal": {
                        "uplift_win_rate_ex_zero": uplift_win,
                        "uplift_mean_return": uplift_ret,
                        "p_value_win_rate": quasi_p_win,
                        "p_value_mean_return": quasi_p_ret,
                    },
                    "strong_causal": {
                        "uplift_win_rate_ex_zero": uplift_win,
                        "uplift_mean_return": uplift_ret,
                        "p_value_win_rate": strong_p_win,
                        "p_value_mean_return": strong_p_ret,
                        "method": "stratified_permutation",
                    },
                }
            )
        q_quasi_win = _bh_qvalues(p_quasi_win)
        q_quasi_ret = _bh_qvalues(p_quasi_ret)
        q_strong_win = _bh_qvalues(p_strong_win)
        q_strong_ret = _bh_qvalues(p_strong_ret)
        for i in range(len(bins_out)):
            bins_out[i]["quasi_causal"]["q_value_win_rate"] = q_quasi_win[i]
            bins_out[i]["quasi_causal"]["q_value_mean_return"] = q_quasi_ret[i]
            bins_out[i]["strong_causal"]["q_value_win_rate"] = q_strong_win[i]
            bins_out[i]["strong_causal"]["q_value_mean_return"] = q_strong_ret[i]
        out[cond] = {
            "closed_trade_count": int(len(rows)),
            "bucket_count": int(len(bins_out)),
            "bins": bins_out,
        }
    return out


def _trade_returns_from_weight_series(
    w: pd.Series,
    ret_exec: pd.Series,
    *,
    cost_bps: float,
    slippage_rate: float,
    exec_price: pd.Series,
    dates: pd.Index,
    eps: float = 1e-12,
) -> dict[str, Any]:
    ww = pd.to_numeric(w, errors="coerce").astype(float).reindex(dates).fillna(0.0)
    rr = (
        pd.to_numeric(ret_exec, errors="coerce")
        .astype(float)
        .reindex(dates)
        .fillna(0.0)
    )
    n = int(len(dates))
    if n <= 0:
        return {"returns": [], "trades": []}
    cost_rate = float(cost_bps) / 10000.0
    px = (
        pd.to_numeric(exec_price, errors="coerce")
        .astype(float)
        .reindex(dates)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )
    slip_spread = float(slippage_rate)
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
        px_i = (
            float(px.iloc[i])
            if np.isfinite(float(px.iloc[i])) and float(px.iloc[i]) > 0.0
            else float("nan")
        )
        slip_ret = (
            float(turnover) * (float(slip_spread) / float(px_i))
            if np.isfinite(px_i) and float(slip_spread) > 0.0
            else 0.0
        )
        day_ret = (
            float(cur) * float(r) - float(turnover) * float(cost_rate) - float(slip_ret)
        )
        if (not active) and (prev <= eps) and (cur > eps):
            active = True
            start_i = int(i)
            start_nav = float(nav_prev)
        nav_cur = float(nav_prev) * (1.0 + float(day_ret))
        if active and (prev > eps) and (cur <= eps):
            tr = (
                (float(nav_cur) / float(start_nav) - 1.0)
                if float(start_nav) != 0
                else float("nan")
            )
            returns.append(float(tr))
            trades.append(
                {
                    "entry_date": str(pd.to_datetime(dates[start_i]).date())
                    if start_i >= 0
                    else None,
                    "exit_date": str(pd.to_datetime(dates[i]).date()),
                    "return": float(tr),
                    "closed": True,
                }
            )
            active = False
            start_i = -1
            start_nav = float(nav_cur)
        nav_prev = float(nav_cur)
    if active and start_i >= 0:
        tr = (
            (float(nav_prev) / float(start_nav) - 1.0)
            if float(start_nav) != 0
            else float("nan")
        )
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
    exec_price: pd.DataFrame,
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
            exec_price=exec_price[c]
            if c in exec_price.columns
            else pd.Series(dtype=float),
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


def _reduce_active_codes_by_group(
    *,
    active_codes: list[str],
    score_row: pd.Series,
    sharpe_row: pd.Series,
    group_enforce: bool,
    asset_groups: dict[str, str] | None,
    group_pick_policy: str,
    group_max_holdings: int,
    current_holdings: set[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    meta: dict[str, Any] = {
        "enabled": bool(group_enforce),
        "policy": str(group_pick_policy or "highest_sharpe"),
        "max_holdings_per_group": int(group_max_holdings),
        "before": list(active_codes),
        "after": list(active_codes),
        "group_picks": {},
        "group_eliminated": {},
        "group_winners": {},
    }
    if (not group_enforce) or (not active_codes):
        return list(active_codes), meta

    policy = str(group_pick_policy or "highest_sharpe").strip().lower()
    if policy not in {"earliest_entry", "highest_sharpe"}:
        raise ValueError(
            "group_pick_policy must be one of: earliest_entry|highest_sharpe"
        )
    kmax = int(group_max_holdings)
    if kmax < 1 or kmax > 10:
        raise ValueError("group_max_holdings must be in [1,10]")

    groups = {str(k): str(v) for k, v in (asset_groups or {}).items()}
    cur = set(str(x) for x in (current_holdings or set()))
    active_set = set(active_codes)
    order_by_score = [
        str(c)
        for c in pd.Series(score_row)
        .reindex(active_codes)
        .sort_values(ascending=False)
        .index.tolist()
        if str(c) in active_set
    ]
    if len(order_by_score) != len(active_codes):
        order_by_score = list(active_codes)

    bucket: dict[str, list[str]] = {}
    for c in order_by_score:
        gid = str(groups.get(str(c)) or str(c)).strip() or str(c)
        bucket.setdefault(gid, []).append(str(c))

    kept_set: set[str] = set()
    for gid, codes_in_gid in bucket.items():
        chosen: list[str] = []
        if policy == "earliest_entry":
            held = [c for c in codes_in_gid if c in cur]
            chosen.extend(held[:kmax])
            for c in codes_in_gid:
                if len(chosen) >= kmax:
                    break
                if c not in chosen:
                    chosen.append(c)
        else:
            ranked = sorted(
                codes_in_gid,
                key=lambda c: (
                    -(
                        float(sharpe_row.get(c))
                        if np.isfinite(float(sharpe_row.get(c)))
                        else -1e18
                    ),
                    -(
                        float(score_row.get(c))
                        if np.isfinite(float(score_row.get(c)))
                        else -1e18
                    ),
                    str(c),
                ),
            )
            chosen = ranked[:kmax]

        eliminated = [c for c in codes_in_gid if c not in set(chosen)]
        for c in chosen:
            kept_set.add(c)
        meta["group_picks"][gid] = list(chosen)
        meta["group_eliminated"][gid] = list(eliminated)
        meta["group_winners"][gid] = chosen[0] if chosen else ""

    reduced = [c for c in order_by_score if c in kept_set]
    meta["after"] = list(reduced)
    return reduced, meta


def _risk_budget_dynamic_weights(
    active_signal: pd.Series,
    *,
    close: pd.Series,
    atr_for_budget: pd.Series,
    atr_fast: pd.Series,
    atr_slow: pd.Series,
    risk_budget_pct: float,
    dynamic_enabled: bool,
    expand_threshold: float,
    contract_threshold: float,
    normal_threshold: float,
) -> tuple[pd.Series, dict[str, int]]:
    idx = active_signal.index
    sig = active_signal.reindex(idx).astype(float).fillna(0.0).clip(lower=0.0)
    px = close.reindex(idx).astype(float)
    atr_b = atr_for_budget.reindex(idx).astype(float)
    atr_f = atr_fast.reindex(idx).astype(float)
    atr_s = atr_slow.reindex(idx).astype(float)
    base_target = (
        (float(risk_budget_pct) * px / atr_b.replace(0.0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
        .astype(float)
    )
    ratio = (
        (atr_f / atr_s.replace(0.0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )

    out = pd.Series(0.0, index=idx, dtype=float)
    state = "FLAT"
    held_weight = 0.0
    stats = {
        "vol_risk_adjust_total_count": 0,
        "vol_risk_adjust_reduce_on_expand_count": 0,
        "vol_risk_adjust_increase_on_contract_count": 0,
        "vol_risk_adjust_recover_from_expand_count": 0,
        "vol_risk_adjust_recover_from_contract_count": 0,
        "vol_risk_entry_state_reduce_on_expand_count": 0,
        "vol_risk_entry_state_increase_on_contract_count": 0,
    }
    for d in idx:
        desired = float(sig.loc[d]) if np.isfinite(float(sig.loc[d])) else 0.0
        desired = max(0.0, desired)
        if desired <= 1e-12:
            state = "FLAT"
            held_weight = 0.0
            out.loc[d] = 0.0
            continue
        if state == "FLAT":
            held_weight = (
                float(base_target.loc[d])
                if np.isfinite(float(base_target.loc[d]))
                else 0.0
            )
            held_weight = float(min(1.0, max(0.0, held_weight)))
            if dynamic_enabled:
                r = (
                    float(ratio.loc[d])
                    if np.isfinite(float(ratio.loc[d]))
                    else float("nan")
                )
                if np.isfinite(r) and r > float(expand_threshold):
                    state = "REDUCED"
                    stats["vol_risk_entry_state_reduce_on_expand_count"] += 1
                elif np.isfinite(r) and r < float(contract_threshold):
                    state = "INCREASED"
                    stats["vol_risk_entry_state_increase_on_contract_count"] += 1
                else:
                    state = "NORMAL"
            else:
                state = "NORMAL"
            out.loc[d] = held_weight
            continue
        if dynamic_enabled:
            r = (
                float(ratio.loc[d])
                if np.isfinite(float(ratio.loc[d]))
                else float("nan")
            )
            can_recalc = np.isfinite(float(base_target.loc[d]))
            if state == "NORMAL":
                if np.isfinite(r) and r > float(expand_threshold) and can_recalc:
                    held_weight = float(base_target.loc[d])
                    state = "REDUCED"
                    stats["vol_risk_adjust_total_count"] += 1
                    stats["vol_risk_adjust_reduce_on_expand_count"] += 1
                elif np.isfinite(r) and r < float(contract_threshold) and can_recalc:
                    held_weight = float(base_target.loc[d])
                    state = "INCREASED"
                    stats["vol_risk_adjust_total_count"] += 1
                    stats["vol_risk_adjust_increase_on_contract_count"] += 1
            elif state == "REDUCED":
                if np.isfinite(r) and r < float(normal_threshold) and can_recalc:
                    held_weight = float(base_target.loc[d])
                    state = "NORMAL"
                    stats["vol_risk_adjust_total_count"] += 1
                    stats["vol_risk_adjust_recover_from_expand_count"] += 1
            elif state == "INCREASED":
                if np.isfinite(r) and r > float(normal_threshold) and can_recalc:
                    held_weight = float(base_target.loc[d])
                    state = "NORMAL"
                    stats["vol_risk_adjust_total_count"] += 1
                    stats["vol_risk_adjust_recover_from_contract_count"] += 1
        out.loc[d] = float(min(1.0, max(0.0, held_weight)))
    return out.astype(float), {k: int(v) for k, v in stats.items()}


def _month_key(d: Any) -> str:
    try:
        return pd.Timestamp(d).strftime("%Y-%m")
    except (TypeError, ValueError, OverflowError):
        return ""


def _position_risk_from_stop_params(
    *,
    atr_stop_enabled: bool,
    atr_mode: str,
    atr_basis: str,
    atr_n: float,
    atr_m: float,
    entry_px: float,
    entry_atr: float,
    curr_close: float,
    curr_atr: float,
    position_weight: float,
    fallback_position_risk: float = 0.02,
) -> float:
    fb = float(fallback_position_risk)
    if (not np.isfinite(fb)) or fb < 0.0:
        fb = 0.02
    w = float(position_weight) if np.isfinite(float(position_weight)) else 0.0
    w = max(0.0, w)
    if not atr_stop_enabled:
        return fb
    if (not np.isfinite(entry_px)) or entry_px <= 0.0:
        return fb
    if (not np.isfinite(entry_atr)) or entry_atr <= 0.0:
        return fb
    n_mult = float(atr_n) if np.isfinite(float(atr_n)) else 2.0
    m_step = float(atr_m) if np.isfinite(float(atr_m)) else 0.5
    mode = str(atr_mode or "none").strip().lower()
    basis = str(atr_basis or "latest").strip().lower()
    stop_px = float("nan")
    if mode == "static":
        stop_px = float(entry_px - n_mult * entry_atr)
    elif mode in {"trailing", "tightening"}:
        ref_atr = entry_atr if basis == "entry" else curr_atr
        if (
            (not np.isfinite(ref_atr))
            or ref_atr <= 0.0
            or (not np.isfinite(curr_close))
        ):
            return fb
        if mode == "trailing":
            stop_px = float(curr_close - n_mult * ref_atr)
        else:
            if (not np.isfinite(m_step)) or m_step <= 0.0:
                m_step = 0.5
            rise = max(0.0, float(curr_close - entry_px))
            den = float(m_step * ref_atr)
            steps = int(np.floor(rise / den)) if den > 0.0 else 0
            dist_mult = max(float(n_mult - steps * m_step), float(m_step))
            stop_px = float(curr_close - dist_mult * ref_atr)
    else:
        return fb
    if (not np.isfinite(stop_px)) or (entry_px <= stop_px):
        return 0.0 if np.isfinite(stop_px) else fb
    unit_risk = (float(entry_px) - float(stop_px)) / float(entry_px)
    if not np.isfinite(unit_risk):
        return fb
    return float(max(0.0, unit_risk) * w)


def _apply_monthly_risk_budget_gate(
    decision_weights: pd.DataFrame,
    *,
    close: pd.DataFrame,
    atr: pd.DataFrame,
    enabled: bool,
    budget_pct: float,
    include_new_trade_risk: bool,
    atr_stop_enabled: bool,
    atr_mode: str,
    atr_basis: str,
    atr_n: float,
    atr_m: float,
    fallback_position_risk: float = 0.02,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    w = (
        decision_weights.reindex(index=close.index, columns=close.columns)
        .astype(float)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    if not enabled:
        return w, {
            "enabled": False,
            "budget_pct": float(budget_pct),
            "include_new_trade_risk": bool(include_new_trade_risk),
            "attempted_entry_count": 0,
            "attempted_entry_count_by_code": {str(c): 0 for c in w.columns},
            "blocked_entry_count": 0,
            "blocked_entry_count_by_code": {str(c): 0 for c in w.columns},
        }
    bgt = float(budget_pct)
    if (not np.isfinite(bgt)) or bgt < 0.01 or bgt > 0.06:
        bgt = 0.06

    out = pd.DataFrame(0.0, index=w.index, columns=w.columns, dtype=float)
    attempted_by_code: dict[str, int] = {str(c): 0 for c in w.columns}
    blocked_by_code: dict[str, int] = {str(c): 0 for c in w.columns}
    realized_by_month: dict[str, float] = {}
    positions: dict[str, dict[str, float]] = {}
    eps = 1e-12

    for d in w.index:
        month = _month_key(d)
        desired = w.loc[d].astype(float).fillna(0.0).clip(lower=0.0).copy()
        monthly_realized = float(realized_by_month.get(month, 0.0))
        monthly_loss = max(0.0, -monthly_realized)

        current_holding_risk = 0.0
        for c, st in positions.items():
            wt = float(st.get("weight", 0.0))
            if wt <= eps:
                continue
            entry_px = float(st.get("entry_px", float("nan")))
            entry_atr = float(st.get("entry_atr", float("nan")))
            curr_close = (
                float(close.loc[d, c])
                if (
                    c in close.columns
                    and d in close.index
                    and np.isfinite(float(close.loc[d, c]))
                )
                else float("nan")
            )
            curr_atr = (
                float(atr.loc[d, c])
                if (
                    c in atr.columns
                    and d in atr.index
                    and np.isfinite(float(atr.loc[d, c]))
                )
                else float("nan")
            )
            pos_risk = _position_risk_from_stop_params(
                atr_stop_enabled=bool(atr_stop_enabled),
                atr_mode=atr_mode,
                atr_basis=atr_basis,
                atr_n=float(atr_n),
                atr_m=float(atr_m),
                entry_px=entry_px,
                entry_atr=entry_atr,
                curr_close=curr_close,
                curr_atr=curr_atr,
                position_weight=wt,
                fallback_position_risk=float(fallback_position_risk),
            )
            current_holding_risk += float(max(0.0, pos_risk))

        budget_used = float(monthly_loss + current_holding_risk)
        accepted = desired.copy()
        for c in w.columns:
            code = str(c)
            target_w = float(desired.get(c, 0.0))
            is_new_entry = (code not in positions) and (target_w > eps)
            if not is_new_entry:
                continue
            attempted_by_code[code] = int(attempted_by_code.get(code, 0) + 1)
            new_trade_risk = 0.0
            if include_new_trade_risk:
                entry_px = (
                    float(close.loc[d, c])
                    if (
                        c in close.columns
                        and d in close.index
                        and np.isfinite(float(close.loc[d, c]))
                    )
                    else float("nan")
                )
                entry_atr = (
                    float(atr.loc[d, c])
                    if (
                        c in atr.columns
                        and d in atr.index
                        and np.isfinite(float(atr.loc[d, c]))
                    )
                    else float("nan")
                )
                new_trade_risk = _position_risk_from_stop_params(
                    atr_stop_enabled=bool(atr_stop_enabled),
                    atr_mode=atr_mode,
                    atr_basis=atr_basis,
                    atr_n=float(atr_n),
                    atr_m=float(atr_m),
                    entry_px=entry_px,
                    entry_atr=entry_atr,
                    curr_close=entry_px,
                    curr_atr=entry_atr,
                    position_weight=target_w,
                    fallback_position_risk=float(fallback_position_risk),
                )
                new_trade_risk = float(max(0.0, new_trade_risk))
            if float(budget_used + new_trade_risk) >= float(bgt) - 1e-12:
                accepted.loc[c] = 0.0
                blocked_by_code[code] = int(blocked_by_code.get(code, 0) + 1)
                continue
            budget_used += float(new_trade_risk)

        out.loc[d] = accepted.to_numpy(dtype=float)

        for c in list(positions.keys()):
            nw = float(accepted.get(c, 0.0))
            if nw > eps:
                positions[c]["weight"] = nw
                continue
            entry_px = float(positions[c].get("entry_px", float("nan")))
            entry_w = float(
                positions[c].get("entry_weight", positions[c].get("weight", 0.0))
            )
            exit_px = (
                float(close.loc[d, c])
                if (
                    c in close.columns
                    and d in close.index
                    and np.isfinite(float(close.loc[d, c]))
                )
                else float("nan")
            )
            realized = 0.0
            if np.isfinite(entry_px) and entry_px > 0.0 and np.isfinite(exit_px):
                realized = float((exit_px / entry_px - 1.0) * entry_w)
            realized_by_month[month] = float(
                realized_by_month.get(month, 0.0) + realized
            )
            positions.pop(c, None)

        for c in w.columns:
            code = str(c)
            nw = float(accepted.get(c, 0.0))
            if nw <= eps or code in positions:
                continue
            entry_px = (
                float(close.loc[d, c])
                if (
                    c in close.columns
                    and d in close.index
                    and np.isfinite(float(close.loc[d, c]))
                )
                else float("nan")
            )
            entry_atr = (
                float(atr.loc[d, c])
                if (
                    c in atr.columns
                    and d in atr.index
                    and np.isfinite(float(atr.loc[d, c]))
                )
                else float("nan")
            )
            positions[code] = {
                "entry_px": float(entry_px),
                "entry_atr": float(entry_atr),
                "entry_weight": float(nw),
                "weight": float(nw),
            }

    total_attempted = int(sum(int(v) for v in attempted_by_code.values()))
    total_blocked = int(sum(int(v) for v in blocked_by_code.values()))
    return out.astype(float), {
        "enabled": True,
        "budget_pct": float(bgt),
        "include_new_trade_risk": bool(include_new_trade_risk),
        "attempted_entry_count": int(total_attempted),
        "attempted_entry_count_by_code": {
            str(k): int(v) for k, v in attempted_by_code.items()
        },
        "blocked_entry_count": int(total_blocked),
        "blocked_entry_count_by_code": {
            str(k): int(v) for k, v in blocked_by_code.items()
        },
    }


def _latest_entry_exec_price_with_slippage(
    *,
    effective_weight: pd.Series,
    exec_price_series: pd.Series,
    slippage_spread: float,
) -> float | None:
    w = effective_weight.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if w.empty:
        return None
    if float(w.iloc[-1]) <= 1e-12:
        return None
    w_prev = w.shift(1).fillna(0.0).astype(float)
    is_entry = (w > 1e-12) & (w_prev <= 1e-12)
    entry_dates = list(w.index[is_entry])
    if not entry_dates:
        return None
    entry_dt = entry_dates[-1]
    px = (
        pd.to_numeric(exec_price_series, errors="coerce")
        .astype(float)
        .reindex(w.index)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )
    px0 = (
        float(px.loc[entry_dt])
        if entry_dt in px.index and np.isfinite(float(px.loc[entry_dt]))
        else float("nan")
    )
    if (not np.isfinite(px0)) or px0 <= 0.0:
        return None
    spread = (
        float(slippage_spread)
        if np.isfinite(float(slippage_spread)) and float(slippage_spread) > 0.0
        else 0.0
    )
    return float(px0 + 0.5 * spread)


def _extract_atr_plan_stops_from_trace(
    stats: dict[str, Any],
) -> dict[str, float | None]:
    rows = list((stats or {}).get("trace_last_rows") or [])
    if not rows:
        return {"plan_stop_current": None, "plan_stop_next": None}
    episode: list[dict[str, Any]] = []
    for r in rows:
        rr = r or {}
        in_pos = bool(rr.get("in_pos_after")) or (
            float(rr.get("decision_pos") or 0.0) > 1e-12
        )
        if in_pos:
            episode.append(rr)
        else:
            episode = []
    if not episode:
        return {"plan_stop_current": None, "plan_stop_next": None}

    def _to_finite(v: Any) -> float:
        try:
            x = float(v)
        except Exception:
            return float("nan")
        return x if np.isfinite(x) else float("nan")

    last = episode[-1] or {}
    prev = episode[-2] if len(episode) >= 2 else None
    stop_next = _to_finite(last.get("stop_after"))
    stop_cur = _to_finite(last.get("stop_before"))
    if (not np.isfinite(stop_cur)) and isinstance(prev, dict):
        stop_cur = _to_finite(prev.get("stop_after"))
    if (not np.isfinite(stop_cur)) and np.isfinite(stop_next):
        stop_cur = stop_next
    return {
        "plan_stop_current": (float(stop_cur) if np.isfinite(stop_cur) else None),
        "plan_stop_next": (float(stop_next) if np.isfinite(stop_next) else None),
    }


def _normalize_r_take_profit_tiers(
    tiers: list[dict[str, float]] | None,
) -> list[dict[str, float]]:
    raw = tiers if tiers is not None else DEFAULT_R_TAKE_PROFIT_TIERS
    out: list[dict[str, float]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        r_mult = float(item.get("r_multiple"))
        retrace = float(item.get("retrace_ratio"))
        if (not np.isfinite(r_mult)) or r_mult <= 0.0:
            continue
        if (not np.isfinite(retrace)) or retrace <= 0.0 or retrace >= 1.0:
            continue
        out.append({"r_multiple": float(r_mult), "retrace_ratio": float(retrace)})
    if not out:
        out = [dict(x) for x in DEFAULT_R_TAKE_PROFIT_TIERS]
    uniq: dict[float, float] = {}
    for row in out:
        uniq[float(row["r_multiple"])] = float(row["retrace_ratio"])
    rows = [
        {"r_multiple": float(k), "retrace_ratio": float(v)} for k, v in uniq.items()
    ]
    rows.sort(key=lambda x: float(x["r_multiple"]))
    return rows


def _apply_r_multiple_take_profit(
    base_pos: pd.Series,
    *,
    open_: pd.Series | None = None,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    enabled: bool,
    reentry_mode: str,
    atr_window: int,
    atr_n: float,
    tiers: list[dict[str, float]] | None,
    atr_stop_enabled: bool,
) -> tuple[pd.Series, dict[str, Any]]:
    tiers_v = _normalize_r_take_profit_tiers(tiers)
    reentry_v = str(reentry_mode or "reenter").strip().lower()
    if reentry_v not in {"reenter", "wait_next_entry"}:
        reentry_v = "reenter"
    if not bool(enabled):
        out_none = base_pos.fillna(0.0).astype(float)
        return out_none, {
            "enabled": False,
            "reentry_mode": reentry_v,
            "atr_window": int(atr_window),
            "atr_n": float(atr_n),
            "fallback_mode_used": False,
            "initial_r_mode": "disabled",
            "trigger_count": 0,
            "trigger_dates": [],
            "trigger_events": [],
            "first_trigger_date": None,
            "last_trigger_date": None,
            "entries": int(
                ((out_none > 0.0) & (out_none.shift(1).fillna(0.0) <= 0.0)).sum()
            ),
            "exits": int(
                ((out_none <= 0.0) & (out_none.shift(1).fillna(0.0) > 0.0)).sum()
            ),
            "trigger_exit_share": 0.0,
            "wait_next_entry_lock_active": False,
            "tiers": tiers_v,
            "tier_trigger_counts": {},
            "trigger_rule": "low_vs_peak_retrace_same_day_exit",
            "fill_rule": "fill=min(trigger_price,open_price) for long",
            "trace_last_rows": [],
        }
    bp = base_pos.fillna(0.0).astype(float)
    cl = close.astype(float)
    op = (
        (open_.astype(float) if isinstance(open_, pd.Series) else cl.astype(float))
        .reindex(bp.index)
        .astype(float)
    )
    hi = high.astype(float).fillna(cl)
    lo = low.astype(float).fillna(cl)
    atr = _atr_from_hlc(hi, lo, cl, window=int(atr_window)).astype(float)
    out = np.zeros(len(bp), dtype=float)
    trigger_dates: list[str] = []
    trigger_events: list[dict[str, Any]] = []
    tier_trigger_counts: dict[str, int] = {}
    trace_last_rows: list[dict[str, Any]] = []
    in_pos = False
    prev_base = 0.0
    wait_next_entry_lock = False
    entry_px = float("nan")
    initial_r_pct = float("nan")
    peak_profit_pct = float("nan")
    invalid_r_entries = 0
    eps = 1e-12

    for i in range(len(bp)):
        b = float(bp.iloc[i]) if np.isfinite(float(bp.iloc[i])) else 0.0
        o = float(op.iloc[i]) if np.isfinite(float(op.iloc[i])) else float("nan")
        c = float(cl.iloc[i]) if np.isfinite(float(cl.iloc[i])) else float("nan")
        h = float(hi.iloc[i]) if np.isfinite(float(hi.iloc[i])) else float("nan")
        l = float(lo.iloc[i]) if np.isfinite(float(lo.iloc[i])) else float("nan")  # noqa: E741
        a = float(atr.iloc[i]) if np.isfinite(float(atr.iloc[i])) else float("nan")
        d = bp.index[i]
        ds = d.date().isoformat() if hasattr(d, "date") else str(d)
        base_entry_event = bool((b > 0.0) and (prev_base <= 0.0))

        if not in_pos:
            if b <= 0.0 or (not np.isfinite(c)) or c <= 0.0:
                out[i] = 0.0
                prev_base = b
                continue
            if (
                reentry_v == "wait_next_entry"
                and wait_next_entry_lock
                and (not base_entry_event)
            ):
                out[i] = 0.0
                prev_base = b
                continue
            in_pos = True
            wait_next_entry_lock = False
            entry_px = c
            initial_r_pct = (
                float(atr_n) * a / c if (np.isfinite(a) and a > 0.0) else float("nan")
            )
            if (not np.isfinite(initial_r_pct)) or initial_r_pct <= eps:
                invalid_r_entries += 1
                initial_r_pct = float("nan")
            peak_profit_pct = 0.0
            out[i] = b
            trace_last_rows.append(
                {
                    "date": ds,
                    "event_type": "entry",
                    "event_reason": (
                        "base_entry_signal" if base_entry_event else "tp_reentry"
                    ),
                    "base_pos": float(b),
                    "entry_price": float(entry_px),
                    "atr_entry": (float(a) if np.isfinite(a) else None),
                    "initial_r_pct": (
                        float(initial_r_pct) if np.isfinite(initial_r_pct) else None
                    ),
                    "peak_profit_pct": 0.0,
                    "peak_r_multiple": 0.0,
                    "active_tier_r": None,
                    "active_tier_retrace": None,
                    "drawdown_from_peak": 0.0,
                    "tp_triggered": False,
                    "open": (float(o) if np.isfinite(o) else None),
                    "high": (float(h) if np.isfinite(h) else None),
                    "low": (float(l) if np.isfinite(l) else None),
                    "tp_fill_price": None,
                    "tp_trigger_source": None,
                    "gap_open_triggered": None,
                    "stop_fill_price": None,
                    "stop_trigger_source": None,
                    "decision_pos": float(out[i]),
                    "in_pos_after": bool(in_pos),
                    "wait_next_entry_lock": bool(wait_next_entry_lock),
                }
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]
            prev_base = b
            continue

        if b <= 0.0:
            in_pos = False
            out[i] = 0.0
            entry_px = float("nan")
            initial_r_pct = float("nan")
            peak_profit_pct = float("nan")
            trace_last_rows.append(
                {
                    "date": ds,
                    "event_type": "exit",
                    "event_reason": "base_exit_signal",
                    "base_pos": float(b),
                    "entry_price": None,
                    "atr_entry": None,
                    "initial_r_pct": None,
                    "peak_profit_pct": None,
                    "peak_r_multiple": None,
                    "active_tier_r": None,
                    "active_tier_retrace": None,
                    "drawdown_from_peak": None,
                    "tp_triggered": False,
                    "open": (float(o) if np.isfinite(o) else None),
                    "high": (float(h) if np.isfinite(h) else None),
                    "low": (float(l) if np.isfinite(l) else None),
                    "tp_fill_price": None,
                    "tp_trigger_source": None,
                    "gap_open_triggered": None,
                    "stop_fill_price": None,
                    "stop_trigger_source": None,
                    "decision_pos": float(out[i]),
                    "in_pos_after": bool(in_pos),
                    "wait_next_entry_lock": bool(wait_next_entry_lock),
                }
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]
            prev_base = b
            continue

        cur_peak_profit_pct = (
            (h / entry_px - 1.0)
            if (np.isfinite(h) and np.isfinite(entry_px) and entry_px > eps)
            else float("nan")
        )
        cur_low_profit_pct = (
            (l / entry_px - 1.0)
            if (np.isfinite(l) and np.isfinite(entry_px) and entry_px > eps)
            else float("nan")
        )
        if np.isfinite(cur_peak_profit_pct):
            peak_profit_pct = max(float(peak_profit_pct), float(cur_peak_profit_pct))
        peak_r_mult = (
            (float(peak_profit_pct) / float(initial_r_pct))
            if (
                np.isfinite(peak_profit_pct)
                and np.isfinite(initial_r_pct)
                and initial_r_pct > eps
            )
            else float("nan")
        )
        active_tier: dict[str, float] | None = None
        if np.isfinite(peak_r_mult):
            for row in tiers_v:
                if peak_r_mult >= float(row["r_multiple"]):
                    active_tier = row
        dd_from_peak = (
            (float(peak_profit_pct) - float(cur_low_profit_pct))
            / float(peak_profit_pct)
            if (
                np.isfinite(peak_profit_pct)
                and peak_profit_pct > eps
                and np.isfinite(cur_low_profit_pct)
            )
            else float("nan")
        )
        tp_triggered = bool(
            active_tier is not None
            and np.isfinite(dd_from_peak)
            and dd_from_peak >= float(active_tier["retrace_ratio"])
        )
        if tp_triggered:
            in_pos = False
            out[i] = 0.0
            trigger_profit_pct = float(peak_profit_pct) * (
                1.0 - float(active_tier["retrace_ratio"])
            )
            trigger_px = (
                float(entry_px) * (1.0 + float(trigger_profit_pct))
                if np.isfinite(entry_px) and np.isfinite(trigger_profit_pct)
                else float("nan")
            )
            gap_open_triggered = bool(
                np.isfinite(o) and np.isfinite(trigger_px) and (o <= trigger_px)
            )
            fill_price = (
                float(o) if gap_open_triggered and np.isfinite(o) else float(trigger_px)
            )
            trigger_source = (
                "gap_open_below_tp" if gap_open_triggered else "low_touch_tp_retrace"
            )
            if reentry_v == "wait_next_entry":
                wait_next_entry_lock = True
            trigger_dates.append(ds)
            trigger_events.append(
                {
                    "date": ds,
                    "trigger_price": (
                        float(trigger_px) if np.isfinite(trigger_px) else None
                    ),
                    "open_price": (float(o) if np.isfinite(o) else None),
                    "low_price": (float(l) if np.isfinite(l) else None),
                    "fill_price": (
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    "trigger_source": trigger_source,
                    "gap_open_triggered": bool(gap_open_triggered),
                    "peak_r_multiple": (
                        float(peak_r_mult) if np.isfinite(peak_r_mult) else None
                    ),
                    "active_tier_r": float(active_tier["r_multiple"])
                    if active_tier
                    else None,
                    "active_tier_retrace": float(active_tier["retrace_ratio"])
                    if active_tier
                    else None,
                }
            )
            tier_label = (
                f"{float(active_tier['r_multiple']):g}R" if active_tier else "unknown"
            )
            tier_trigger_counts[tier_label] = int(
                tier_trigger_counts.get(tier_label, 0) + 1
            )
            trace_last_rows.append(
                {
                    "date": ds,
                    "event_type": "exit",
                    "event_reason": "r_take_profit",
                    "base_pos": float(b),
                    "entry_price": (float(entry_px) if np.isfinite(entry_px) else None),
                    "atr_entry": None,
                    "initial_r_pct": (
                        float(initial_r_pct) if np.isfinite(initial_r_pct) else None
                    ),
                    "peak_profit_pct": (
                        float(peak_profit_pct) if np.isfinite(peak_profit_pct) else None
                    ),
                    "peak_r_multiple": (
                        float(peak_r_mult) if np.isfinite(peak_r_mult) else None
                    ),
                    "active_tier_r": float(active_tier["r_multiple"])
                    if active_tier
                    else None,
                    "active_tier_retrace": float(active_tier["retrace_ratio"])
                    if active_tier
                    else None,
                    "drawdown_from_peak": (
                        float(dd_from_peak) if np.isfinite(dd_from_peak) else None
                    ),
                    "tp_triggered": True,
                    "open": (float(o) if np.isfinite(o) else None),
                    "high": (float(h) if np.isfinite(h) else None),
                    "low": (float(l) if np.isfinite(l) else None),
                    "tp_fill_price": (
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    "tp_trigger_source": trigger_source,
                    "gap_open_triggered": bool(gap_open_triggered),
                    "stop_fill_price": (
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    "stop_trigger_source": trigger_source,
                    "decision_pos": float(out[i]),
                    "in_pos_after": bool(in_pos),
                    "wait_next_entry_lock": bool(wait_next_entry_lock),
                }
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]
            entry_px = float("nan")
            initial_r_pct = float("nan")
            peak_profit_pct = float("nan")
            prev_base = b
            continue

        out[i] = b
        trace_last_rows.append(
            {
                "date": ds,
                "event_type": "hold",
                "event_reason": "r_take_profit_watch",
                "base_pos": float(b),
                "entry_price": (float(entry_px) if np.isfinite(entry_px) else None),
                "atr_entry": None,
                "initial_r_pct": (
                    float(initial_r_pct) if np.isfinite(initial_r_pct) else None
                ),
                "peak_profit_pct": (
                    float(peak_profit_pct) if np.isfinite(peak_profit_pct) else None
                ),
                "peak_r_multiple": (
                    float(peak_r_mult) if np.isfinite(peak_r_mult) else None
                ),
                "active_tier_r": float(active_tier["r_multiple"])
                if active_tier
                else None,
                "active_tier_retrace": float(active_tier["retrace_ratio"])
                if active_tier
                else None,
                "drawdown_from_peak": (
                    float(dd_from_peak) if np.isfinite(dd_from_peak) else None
                ),
                "tp_triggered": False,
                "open": (float(o) if np.isfinite(o) else None),
                "high": (float(h) if np.isfinite(h) else None),
                "low": (float(l) if np.isfinite(l) else None),
                "tp_fill_price": None,
                "tp_trigger_source": None,
                "gap_open_triggered": None,
                "stop_fill_price": None,
                "stop_trigger_source": None,
                "decision_pos": float(out[i]),
                "in_pos_after": bool(in_pos),
                "wait_next_entry_lock": bool(wait_next_entry_lock),
            }
        )
        if len(trace_last_rows) > 120:
            trace_last_rows = trace_last_rows[-120:]
        prev_base = b

    out_s = pd.Series(out, index=base_pos.index, dtype=float)
    exits = int(((out_s <= 0.0) & (out_s.shift(1).fillna(0.0) > 0.0)).sum())
    trigger_count = int(len(trigger_dates))
    stats = {
        "enabled": True,
        "reentry_mode": reentry_v,
        "atr_window": int(atr_window),
        "atr_n": float(atr_n),
        "fallback_mode_used": bool(not atr_stop_enabled),
        "initial_r_mode": ("atr_stop" if atr_stop_enabled else "virtual_atr_fallback"),
        "trigger_count": trigger_count,
        "trigger_dates": trigger_dates[:200],
        "trigger_events": trigger_events[:200],
        "first_trigger_date": (trigger_dates[0] if trigger_dates else None),
        "last_trigger_date": (trigger_dates[-1] if trigger_dates else None),
        "entries": int(((out_s > 0.0) & (out_s.shift(1).fillna(0.0) <= 0.0)).sum()),
        "exits": exits,
        "trigger_exit_share": (
            float(trigger_count) / float(exits) if exits > 0 else 0.0
        ),
        "wait_next_entry_lock_active": bool(wait_next_entry_lock),
        "tiers": tiers_v,
        "tier_trigger_counts": dict(tier_trigger_counts),
        "invalid_initial_r_entries": int(invalid_r_entries),
        "trigger_rule": "low_vs_peak_retrace_same_day_exit",
        "fill_rule": "fill=min(trigger_price,open_price) for long",
        "trace_last_rows": trace_last_rows[-80:],
    }
    return out_s, stats


def _apply_atr_stop(
    base_pos: pd.Series,
    *,
    open_: pd.Series | None = None,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    mode: str,
    atr_basis: str,
    reentry_mode: str,
    atr_window: int,
    n_mult: float,
    m_step: float,
) -> tuple[pd.Series, dict[str, Any]]:
    mode_v = str(mode or "none").strip().lower()
    atr_basis_v = str(atr_basis or "latest").strip().lower()
    reentry_v = str(reentry_mode or "reenter").strip().lower()

    def _event_row(
        *,
        idx: Any,
        b: float,
        c: float | None,
        o: float | None,
        l: float | None,  # noqa: E741
        a: float | None,
        stop_before: float | None,
        stop_candidate: float | None,
        stop_after: float | None,
        stop_fill_price: float | None,
        stop_trigger_source: str | None,
        gap_open_triggered: bool | None,
        decision_pos: float,
        in_pos_after: bool,
        wait_lock: bool,
        event_type: str,
        event_reason: str,
        base_entry_event: bool,
        stop_triggered: bool,
    ) -> dict[str, Any]:
        return {
            "date": (idx.date().isoformat() if hasattr(idx, "date") else str(idx)),
            "event_type": str(event_type),
            "event_reason": str(event_reason),
            "base_pos": float(b),
            "base_entry_event": bool(base_entry_event),
            "close": (
                None if c is None else (float(c) if np.isfinite(float(c)) else None)
            ),
            "open": (
                None if o is None else (float(o) if np.isfinite(float(o)) else None)
            ),
            "low": (
                None if l is None else (float(l) if np.isfinite(float(l)) else None)
            ),
            "atr": (
                None if a is None else (float(a) if np.isfinite(float(a)) else None)
            ),
            "stop_before": stop_before,
            "stop_candidate": stop_candidate,
            "stop_after": stop_after,
            "stop_triggered": bool(stop_triggered),
            "stop_fill_price": stop_fill_price,
            "stop_trigger_source": stop_trigger_source,
            "gap_open_triggered": (
                None if gap_open_triggered is None else bool(gap_open_triggered)
            ),
            "decision_pos": float(decision_pos),
            "in_pos_after": bool(in_pos_after),
            "wait_next_entry_lock": bool(wait_lock),
        }

    if mode_v == "none":
        out_none = base_pos.fillna(0.0).astype(float)
        bp = out_none.astype(float)
        trace_rows: list[dict[str, Any]] = []
        prev_b = 0.0
        for i in range(len(bp)):
            b = float(bp.iloc[i]) if np.isfinite(float(bp.iloc[i])) else 0.0
            d = bp.index[i]
            base_entry_event = bool((b > 0.0) and (prev_b <= 0.0))
            base_exit_event = bool((b <= 0.0) and (prev_b > 0.0))
            if base_entry_event:
                trace_rows.append(
                    _event_row(
                        idx=d,
                        b=b,
                        c=None,
                        o=None,
                        l=None,
                        a=None,
                        stop_before=None,
                        stop_candidate=None,
                        stop_after=None,
                        stop_fill_price=None,
                        stop_trigger_source=None,
                        gap_open_triggered=None,
                        decision_pos=b,
                        in_pos_after=bool(b > 0.0),
                        wait_lock=False,
                        event_type="entry",
                        event_reason="base_entry_signal",
                        base_entry_event=base_entry_event,
                        stop_triggered=False,
                    )
                )
            elif base_exit_event:
                trace_rows.append(
                    _event_row(
                        idx=d,
                        b=b,
                        c=None,
                        o=None,
                        l=None,
                        a=None,
                        stop_before=None,
                        stop_candidate=None,
                        stop_after=None,
                        stop_fill_price=None,
                        stop_trigger_source=None,
                        gap_open_triggered=None,
                        decision_pos=b,
                        in_pos_after=bool(b > 0.0),
                        wait_lock=False,
                        event_type="exit",
                        event_reason="base_exit_signal",
                        base_entry_event=base_entry_event,
                        stop_triggered=False,
                    )
                )
            prev_b = b
        return out_none, {
            "enabled": False,
            "mode": "none",
            "atr_basis": atr_basis_v,
            "reentry_mode": reentry_v,
            "trigger_count": 0,
            "trigger_dates": [],
            "trigger_events": [],
            "first_trigger_date": None,
            "last_trigger_date": None,
            "entries": int(
                ((out_none > 0.0) & (out_none.shift(1).fillna(0.0) <= 0.0)).sum()
            ),
            "exits": int(
                ((out_none <= 0.0) & (out_none.shift(1).fillna(0.0) > 0.0)).sum()
            ),
            "trigger_exit_share": 0.0,
            "latest_stop_price": None,
            "latest_stop_date": None,
            "wait_next_entry_lock_active": False,
            "trigger_rule": "low_le_stop_same_day_exit",
            "fill_rule": "fill=min(stop_price,open_price) for long",
            "trace_last_rows": trace_rows[-80:],
            "trade_records": [],
        }

    atr = _atr_from_hlc(
        high.astype(float),
        low.astype(float),
        close.astype(float),
        window=int(atr_window),
    ).astype(float)
    bp = base_pos.astype(float).fillna(0.0)
    op = (
        (open_.astype(float) if isinstance(open_, pd.Series) else close.astype(float))
        .reindex(bp.index)
        .astype(float)
    )
    cl = close.astype(float)
    lo = low.astype(float)

    out = np.zeros(len(bp), dtype=float)
    in_pos = False
    stop_px = float("nan")
    entry_px = float("nan")
    entry_atr = float("nan")
    prev_base = 0.0
    wait_next_entry_lock = False
    trigger_dates: list[str] = []
    trigger_events: list[dict[str, Any]] = []
    trade_records: list[dict[str, Any]] = []
    latest_stop_date: str | None = None
    trace_last_rows: list[dict[str, Any]] = []
    entry_decision_date: str | None = None
    initial_stop_price: float = float("nan")

    for i in range(len(bp)):
        b = float(bp.iloc[i]) if np.isfinite(float(bp.iloc[i])) else 0.0
        o = float(op.iloc[i]) if np.isfinite(float(op.iloc[i])) else float("nan")
        c = float(cl.iloc[i]) if np.isfinite(float(cl.iloc[i])) else float("nan")
        l = float(lo.iloc[i]) if np.isfinite(float(lo.iloc[i])) else float("nan")  # noqa: E741
        a = float(atr.iloc[i]) if np.isfinite(float(atr.iloc[i])) else float("nan")
        base_entry_event = bool((b > 0.0) and (prev_base <= 0.0))

        if not in_pos:
            if b <= 0.0 or (not np.isfinite(c)) or (not np.isfinite(a)) or a <= 0.0:
                out[i] = 0.0
                prev_base = b
                continue
            if (
                reentry_v == "wait_next_entry"
                and wait_next_entry_lock
                and (not base_entry_event)
            ):
                out[i] = 0.0
                prev_base = b
                continue
            in_pos = True
            wait_next_entry_lock = False
            entry_px = c
            entry_atr = a
            stop_px = entry_px - float(n_mult) * a
            entry_decision_date = (
                bp.index[i].date().isoformat()
                if hasattr(bp.index[i], "date")
                else str(bp.index[i])
            )
            initial_stop_price = (
                float(stop_px) if np.isfinite(stop_px) else float("nan")
            )
            out[i] = b
            if np.isfinite(stop_px):
                d = bp.index[i]
                latest_stop_date = (
                    d.date().isoformat() if hasattr(d, "date") else str(d)
                )
            trace_last_rows.append(
                _event_row(
                    idx=bp.index[i],
                    b=b,
                    c=c,
                    o=o,
                    l=l,
                    a=a,
                    stop_before=None,
                    stop_candidate=(float(stop_px) if np.isfinite(stop_px) else None),
                    stop_after=(float(stop_px) if np.isfinite(stop_px) else None),
                    stop_fill_price=None,
                    stop_trigger_source=None,
                    gap_open_triggered=None,
                    decision_pos=float(out[i]),
                    in_pos_after=bool(in_pos),
                    wait_lock=bool(wait_next_entry_lock),
                    event_type="entry",
                    event_reason=(
                        "base_entry_signal" if base_entry_event else "stop_reentry"
                    ),
                    base_entry_event=bool(base_entry_event),
                    stop_triggered=False,
                )
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]
            prev_base = b
            continue

        if b <= 0.0:
            stop_before = float(stop_px) if np.isfinite(stop_px) else None
            in_pos = False
            out[i] = 0.0
            stop_px = float("nan")
            entry_px = float("nan")
            entry_atr = float("nan")
            entry_decision_date = None
            initial_stop_price = float("nan")
            trace_last_rows.append(
                _event_row(
                    idx=bp.index[i],
                    b=b,
                    c=c,
                    o=o,
                    l=l,
                    a=a,
                    stop_before=stop_before,
                    stop_candidate=None,
                    stop_after=None,
                    stop_fill_price=None,
                    stop_trigger_source=None,
                    gap_open_triggered=None,
                    decision_pos=float(out[i]),
                    in_pos_after=bool(in_pos),
                    wait_lock=bool(wait_next_entry_lock),
                    event_type="exit",
                    event_reason="base_exit_signal",
                    base_entry_event=bool(base_entry_event),
                    stop_triggered=False,
                )
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]
            prev_base = b
            continue

        stop_before = float(stop_px) if np.isfinite(stop_px) else None
        if np.isfinite(l) and np.isfinite(stop_px) and (l <= stop_px):
            in_pos = False
            out[i] = 0.0
            d = bp.index[i]
            ds = d.date().isoformat() if hasattr(d, "date") else str(d)
            trigger_dates.append(ds)
            gap_open_triggered = bool(np.isfinite(o) and (o <= stop_px))
            fill_price = (
                float(o) if gap_open_triggered and np.isfinite(o) else float(stop_px)
            )
            trigger_source = (
                "gap_open_below_stop" if gap_open_triggered else "low_touch_stop"
            )
            trigger_events.append(
                {
                    "date": ds,
                    "stop_price": (float(stop_px) if np.isfinite(stop_px) else None),
                    "open_price": (float(o) if np.isfinite(o) else None),
                    "low_price": (float(l) if np.isfinite(l) else None),
                    "fill_price": (
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    "trigger_source": trigger_source,
                    "gap_open_triggered": bool(gap_open_triggered),
                }
            )
            trade_records.append(
                {
                    "entry_decision_date": entry_decision_date,
                    "entry_execution_date": None,
                    "entry_execution_price": None,
                    "initial_stop_price": (
                        float(initial_stop_price)
                        if np.isfinite(initial_stop_price)
                        else None
                    ),
                    "trigger_stop_price": (
                        float(stop_px) if np.isfinite(stop_px) else None
                    ),
                    "execution_stop_price": (
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    "trigger_date": ds,
                }
            )
            if reentry_v == "wait_next_entry":
                wait_next_entry_lock = True
            stop_px = float("nan")
            entry_px = float("nan")
            entry_atr = float("nan")
            entry_decision_date = None
            initial_stop_price = float("nan")
            trace_last_rows.append(
                _event_row(
                    idx=bp.index[i],
                    b=b,
                    c=c,
                    o=o,
                    l=l,
                    a=a,
                    stop_before=stop_before,
                    stop_candidate=None,
                    stop_after=None,
                    stop_fill_price=(
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    stop_trigger_source=trigger_source,
                    gap_open_triggered=bool(gap_open_triggered),
                    decision_pos=float(out[i]),
                    in_pos_after=bool(in_pos),
                    wait_lock=bool(wait_next_entry_lock),
                    event_type="exit",
                    event_reason="atr_stop",
                    base_entry_event=bool(base_entry_event),
                    stop_triggered=True,
                )
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]
            prev_base = b
            continue

        stop_candidate: float | None = None
        if (
            mode_v in {"trailing", "tightening"}
            and np.isfinite(c)
            and np.isfinite(a)
            and a > 0.0
        ):
            atr_ref = (
                float(entry_atr)
                if (
                    atr_basis_v == "entry"
                    and np.isfinite(entry_atr)
                    and float(entry_atr) > 0.0
                )
                else float(a)
            )
            dist_mult = float(n_mult)
            if mode_v == "tightening":
                rise_in_atr = max(0.0, (c - entry_px) / max(atr_ref, 1e-12))
                steps = (
                    int(np.floor(rise_in_atr / float(m_step)))
                    if float(m_step) > 0
                    else 0
                )
                dist_mult = max(float(m_step), float(n_mult) - steps * float(m_step))
            stop_candidate_v = c - dist_mult * atr_ref
            stop_candidate = (
                float(stop_candidate_v) if np.isfinite(stop_candidate_v) else None
            )
            if np.isfinite(stop_candidate_v) and (
                (not np.isfinite(stop_px)) or (stop_candidate_v > stop_px)
            ):
                stop_px = stop_candidate_v

        trace_last_rows.append(
            _event_row(
                idx=bp.index[i],
                b=b,
                c=c,
                o=o,
                l=l,
                a=a,
                stop_before=stop_before,
                stop_candidate=stop_candidate,
                stop_after=(float(stop_px) if np.isfinite(stop_px) else None),
                stop_fill_price=None,
                stop_trigger_source=None,
                gap_open_triggered=None,
                decision_pos=float(b),
                in_pos_after=bool(in_pos),
                wait_lock=bool(wait_next_entry_lock),
                event_type="hold",
                event_reason="atr_update",
                base_entry_event=bool(base_entry_event),
                stop_triggered=False,
            )
        )
        if len(trace_last_rows) > 120:
            trace_last_rows = trace_last_rows[-120:]

        out[i] = b
        if np.isfinite(stop_px):
            d = bp.index[i]
            latest_stop_date = d.date().isoformat() if hasattr(d, "date") else str(d)
        prev_base = b

    out_s = pd.Series(out, index=base_pos.index, dtype=float)
    entries = int(((out_s > 0.0) & (out_s.shift(1).fillna(0.0) <= 0.0)).sum())
    exits = int(((out_s <= 0.0) & (out_s.shift(1).fillna(0.0) > 0.0)).sum())
    trigger_count = int(len(trigger_dates))
    stats = {
        "enabled": True,
        "mode": mode_v,
        "atr_basis": atr_basis_v,
        "reentry_mode": reentry_v,
        "trigger_count": trigger_count,
        "trigger_dates": trigger_dates[:200],
        "trigger_events": trigger_events[:200],
        "first_trigger_date": (trigger_dates[0] if trigger_dates else None),
        "last_trigger_date": (trigger_dates[-1] if trigger_dates else None),
        "entries": entries,
        "exits": exits,
        "trigger_exit_share": (
            float(trigger_count) / float(exits) if exits > 0 else 0.0
        ),
        "latest_stop_price": (
            float(stop_px) if (in_pos and np.isfinite(stop_px)) else None
        ),
        "latest_stop_date": (
            latest_stop_date if (in_pos and np.isfinite(stop_px)) else None
        ),
        "wait_next_entry_lock_active": bool(wait_next_entry_lock),
        "trigger_rule": "low_le_stop_same_day_exit",
        "fill_rule": "fill=min(stop_price,open_price) for long",
        "trace_last_rows": trace_last_rows[-80:],
        "trade_records": trade_records,
    }
    return out_s, stats


def _stop_fill_return(
    *,
    exec_price: str,
    open_px: float,
    prev_close_px: float,
    fill_px: float,
) -> float:
    ep = str(exec_price or "close").strip().lower()
    if (not np.isfinite(fill_px)) or fill_px <= 0.0:
        return 0.0
    if ep == "open":
        base = open_px
        if (not np.isfinite(base)) or base <= 0.0:
            return 0.0
        return float(fill_px / base - 1.0)
    if ep == "oc2":
        r1 = 0.0
        r2 = 0.0
        if np.isfinite(open_px) and open_px > 0.0:
            r1 = float(fill_px / open_px - 1.0)
        if np.isfinite(prev_close_px) and prev_close_px > 0.0:
            r2 = float(fill_px / prev_close_px - 1.0)
        return float(0.5 * (r1 + r2))
    base = prev_close_px
    if (not np.isfinite(base)) or base <= 0.0:
        base = open_px
    if (not np.isfinite(base)) or base <= 0.0:
        return 0.0
    return float(fill_px / base - 1.0)


def _apply_intraday_stop_execution_single(
    *,
    weights: pd.Series,
    atr_stop_stats: dict[str, Any],
    exec_price: str,
    open_sig: pd.Series,
    close_sig: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    w0 = weights.astype(float).copy()
    w_adj = w0.copy()
    override = pd.Series(0.0, index=w0.index, dtype=float)
    events = list((atr_stop_stats or {}).get("trigger_events") or [])
    if not events:
        return w_adj, override
    prev_close = close_sig.astype(float).shift(1)
    for e in events:
        d_raw = e.get("date")
        try:
            d = pd.Timestamp(d_raw)
        except Exception:
            continue
        if d not in w_adj.index:
            continue
        wv = float(w_adj.loc[d]) if np.isfinite(float(w_adj.loc[d])) else 0.0
        if wv <= 1e-12:
            continue
        fill_raw = e.get("fill_price")
        try:
            fill_px = float(fill_raw)
        except (TypeError, ValueError):
            fill_px = float("nan")
        if not np.isfinite(fill_px):
            fill_px = float("nan")
        o = (
            float(open_sig.loc[d])
            if (d in open_sig.index and np.isfinite(float(open_sig.loc[d])))
            else float("nan")
        )
        pc = (
            float(prev_close.loc[d])
            if (d in prev_close.index and np.isfinite(float(prev_close.loc[d])))
            else float("nan")
        )
        day_ret = _stop_fill_return(
            exec_price=exec_price, open_px=o, prev_close_px=pc, fill_px=fill_px
        )
        override.loc[d] = float(override.loc[d] + wv * day_ret)
        w_adj.loc[d] = 0.0
    return w_adj, override


def _apply_intraday_stop_execution_portfolio(
    *,
    weights: pd.DataFrame,
    atr_stop_by_asset: dict[str, dict[str, Any]],
    exec_price: str,
    open_sig_df: pd.DataFrame,
    close_sig_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    w0 = weights.astype(float).copy()
    w_adj = w0.copy()
    override = pd.Series(0.0, index=w0.index, dtype=float)
    prev_close_df = close_sig_df.astype(float).shift(1)
    for c in w_adj.columns:
        stats = atr_stop_by_asset.get(str(c)) or {}
        events = list(stats.get("trigger_events") or [])
        if not events:
            continue
        for e in events:
            d_raw = e.get("date")
            try:
                d = pd.Timestamp(d_raw)
            except Exception:
                continue
            if d not in w_adj.index:
                continue
            wv = float(w_adj.loc[d, c]) if np.isfinite(float(w_adj.loc[d, c])) else 0.0
            if wv <= 1e-12:
                continue
            fill_raw = e.get("fill_price")
            try:
                fill_px = float(fill_raw)
            except (TypeError, ValueError):
                fill_px = float("nan")
            if not np.isfinite(fill_px):
                fill_px = float("nan")
            o = (
                float(open_sig_df.loc[d, c])
                if (
                    d in open_sig_df.index
                    and c in open_sig_df.columns
                    and np.isfinite(float(open_sig_df.loc[d, c]))
                )
                else float("nan")
            )
            pc = (
                float(prev_close_df.loc[d, c])
                if (
                    d in prev_close_df.index
                    and c in prev_close_df.columns
                    and np.isfinite(float(prev_close_df.loc[d, c]))
                )
                else float("nan")
            )
            day_ret = _stop_fill_return(
                exec_price=exec_price, open_px=o, prev_close_px=pc, fill_px=fill_px
            )
            override.loc[d] = float(override.loc[d] + wv * day_ret)
            w_adj.loc[d, c] = 0.0
    return w_adj, override


def _apply_bias_v_take_profit(
    base_pos: pd.Series,
    *,
    open_: pd.Series | None = None,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    enabled: bool,
    reentry_mode: str,
    ma_window: int,
    atr_window: int,
    threshold: float,
) -> tuple[pd.Series, dict[str, Any]]:
    reentry_v = str(reentry_mode or "reenter").strip().lower()
    if reentry_v not in {"reenter", "wait_next_entry"}:
        reentry_v = "reenter"
    if not bool(enabled):
        out_none = base_pos.fillna(0.0).astype(float)
        return out_none, {
            "enabled": False,
            "reentry_mode": reentry_v,
            "ma_window": int(ma_window),
            "atr_window": int(atr_window),
            "threshold": float(threshold),
            "trigger_count": 0,
            "trigger_dates": [],
            "trigger_events": [],
            "first_trigger_date": None,
            "last_trigger_date": None,
            "entries": int(
                ((out_none > 0.0) & (out_none.shift(1).fillna(0.0) <= 0.0)).sum()
            ),
            "exits": int(
                ((out_none <= 0.0) & (out_none.shift(1).fillna(0.0) > 0.0)).sum()
            ),
            "trigger_exit_share": 0.0,
            "wait_next_entry_lock_active": False,
            "trigger_rule": "high_ge_threshold_same_day_exit",
            "fill_rule": "fill=max(trigger_price,open_price) for long",
            "trace_last_rows": [],
            "trade_records": [],
        }
    bp = base_pos.fillna(0.0).astype(float)
    cl = close.astype(float)
    op = (
        (open_.astype(float) if isinstance(open_, pd.Series) else cl.astype(float))
        .reindex(bp.index)
        .astype(float)
    )
    hi = high.astype(float).fillna(cl)
    lo = low.astype(float).fillna(cl)
    ma_w = max(2, int(ma_window))
    ma = _rolling_sma(cl, window=ma_w, min_periods=ma_w).astype(float)
    atr = _atr_from_hlc(hi, lo, cl, window=max(2, int(atr_window))).astype(float)
    bias_v = (
        ((cl - ma) / atr.replace(0.0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )

    out = np.zeros(len(bp), dtype=float)
    trigger_dates: list[str] = []
    trigger_events: list[dict[str, Any]] = []
    trade_records: list[dict[str, Any]] = []
    trace_last_rows: list[dict[str, Any]] = []
    in_pos = False
    prev_base = 0.0
    wait_next_entry_lock = False
    entry_decision_date: str | None = None
    initial_tp_price: float = float("nan")

    for i in range(len(bp)):
        b = float(bp.iloc[i]) if np.isfinite(float(bp.iloc[i])) else 0.0
        o = float(op.iloc[i]) if np.isfinite(float(op.iloc[i])) else float("nan")
        c = float(cl.iloc[i]) if np.isfinite(float(cl.iloc[i])) else float("nan")
        h = float(hi.iloc[i]) if np.isfinite(float(hi.iloc[i])) else float("nan")
        l = float(lo.iloc[i]) if np.isfinite(float(lo.iloc[i])) else float("nan")  # noqa: E741
        m = float(ma.iloc[i]) if np.isfinite(float(ma.iloc[i])) else float("nan")
        a = float(atr.iloc[i]) if np.isfinite(float(atr.iloc[i])) else float("nan")
        v = (
            float(bias_v.iloc[i])
            if np.isfinite(float(bias_v.iloc[i]))
            else float("nan")
        )
        d = bp.index[i]
        ds = d.date().isoformat() if hasattr(d, "date") else str(d)
        base_entry_event = bool((b > 0.0) and (prev_base <= 0.0))

        if not in_pos:
            if b <= 0.0 or (not np.isfinite(c)) or c <= 0.0:
                out[i] = 0.0
                prev_base = b
                continue
            if (
                reentry_v == "wait_next_entry"
                and wait_next_entry_lock
                and (not base_entry_event)
            ):
                out[i] = 0.0
                prev_base = b
                continue
            in_pos = True
            wait_next_entry_lock = False
            raw_px = (
                float(m + float(threshold) * a)
                if (np.isfinite(m) and np.isfinite(a))
                else float("nan")
            )
            tp_line_px = float(raw_px) if np.isfinite(raw_px) else float("nan")
            entry_decision_date = ds
            initial_tp_price = (
                float(tp_line_px) if np.isfinite(tp_line_px) else float("nan")
            )
            out[i] = b
            trace_last_rows.append(
                {
                    "date": ds,
                    "event_type": "entry",
                    "event_reason": (
                        "base_entry_signal" if base_entry_event else "tp_reentry"
                    ),
                    "base_pos": float(b),
                    "open": (float(o) if np.isfinite(o) else None),
                    "high": (float(h) if np.isfinite(h) else None),
                    "low": (float(l) if np.isfinite(l) else None),
                    "close": (float(c) if np.isfinite(c) else None),
                    "ma": (float(m) if np.isfinite(m) else None),
                    "atr": (float(a) if np.isfinite(a) else None),
                    "bias_v": (float(v) if np.isfinite(v) else None),
                    "threshold": float(threshold),
                    "tp_trigger_price_raw": (
                        float(raw_px) if np.isfinite(raw_px) else None
                    ),
                    "tp_trigger_price_eff": (
                        float(tp_line_px) if np.isfinite(tp_line_px) else None
                    ),
                    "tp_triggered": False,
                    "tp_fill_price": None,
                    "tp_trigger_source": None,
                    "gap_open_triggered": None,
                    "stop_fill_price": None,
                    "stop_trigger_source": None,
                    "decision_pos": float(out[i]),
                    "in_pos_after": bool(in_pos),
                    "wait_next_entry_lock": bool(wait_next_entry_lock),
                }
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]
            prev_base = b
            continue

        if b <= 0.0:
            in_pos = False
            out[i] = 0.0
            tp_line_px = float("nan")
            entry_decision_date = None
            initial_tp_price = float("nan")
            trace_last_rows.append(
                {
                    "date": ds,
                    "event_type": "exit",
                    "event_reason": "base_exit_signal",
                    "base_pos": float(b),
                    "open": (float(o) if np.isfinite(o) else None),
                    "high": (float(h) if np.isfinite(h) else None),
                    "low": (float(l) if np.isfinite(l) else None),
                    "close": (float(c) if np.isfinite(c) else None),
                    "ma": (float(m) if np.isfinite(m) else None),
                    "atr": (float(a) if np.isfinite(a) else None),
                    "bias_v": (float(v) if np.isfinite(v) else None),
                    "threshold": float(threshold),
                    "tp_trigger_price_raw": None,
                    "tp_trigger_price_eff": None,
                    "tp_triggered": False,
                    "tp_fill_price": None,
                    "tp_trigger_source": None,
                    "gap_open_triggered": None,
                    "stop_fill_price": None,
                    "stop_trigger_source": None,
                    "decision_pos": float(out[i]),
                    "in_pos_after": bool(in_pos),
                    "wait_next_entry_lock": bool(wait_next_entry_lock),
                }
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]
            prev_base = b
            continue

        trigger_px_raw = (
            float(m + float(threshold) * a)
            if (np.isfinite(m) and np.isfinite(a))
            else float("nan")
        )
        if np.isfinite(trigger_px_raw):
            tp_line_px = (
                float(trigger_px_raw)
                if (not np.isfinite(tp_line_px))
                else max(float(tp_line_px), float(trigger_px_raw))
            )
        trigger_px = float(tp_line_px) if np.isfinite(tp_line_px) else float("nan")
        tp_triggered = bool(
            np.isfinite(h) and np.isfinite(trigger_px) and (h >= trigger_px)
        )
        if tp_triggered:
            in_pos = False
            out[i] = 0.0
            gap_open_triggered = bool(
                np.isfinite(o) and np.isfinite(trigger_px) and (o >= trigger_px)
            )
            fill_price = (
                float(o) if gap_open_triggered and np.isfinite(o) else float(trigger_px)
            )
            trigger_source = (
                "gap_open_above_bias_v_tp"
                if gap_open_triggered
                else "high_touch_bias_v_tp"
            )
            if reentry_v == "wait_next_entry":
                wait_next_entry_lock = True
            trigger_dates.append(ds)
            trigger_events.append(
                {
                    "date": ds,
                    "trigger_price": (
                        float(trigger_px) if np.isfinite(trigger_px) else None
                    ),
                    "trigger_price_raw": (
                        float(trigger_px_raw) if np.isfinite(trigger_px_raw) else None
                    ),
                    "trigger_price_eff": (
                        float(trigger_px) if np.isfinite(trigger_px) else None
                    ),
                    "open_price": (float(o) if np.isfinite(o) else None),
                    "high_price": (float(h) if np.isfinite(h) else None),
                    "fill_price": (
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    "trigger_source": trigger_source,
                    "gap_open_triggered": bool(gap_open_triggered),
                    "threshold": float(threshold),
                    "bias_v": (float(v) if np.isfinite(v) else None),
                }
            )
            trade_records.append(
                {
                    "entry_decision_date": entry_decision_date,
                    "entry_execution_date": None,
                    "entry_execution_price": None,
                    "initial_take_profit_price": (
                        float(initial_tp_price)
                        if np.isfinite(initial_tp_price)
                        else None
                    ),
                    "trigger_take_profit_price": (
                        float(trigger_px) if np.isfinite(trigger_px) else None
                    ),
                    "execution_take_profit_price": (
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    "trigger_date": ds,
                }
            )
            trace_last_rows.append(
                {
                    "date": ds,
                    "event_type": "exit",
                    "event_reason": "bias_v_take_profit",
                    "base_pos": float(b),
                    "open": (float(o) if np.isfinite(o) else None),
                    "high": (float(h) if np.isfinite(h) else None),
                    "low": (float(l) if np.isfinite(l) else None),
                    "close": (float(c) if np.isfinite(c) else None),
                    "ma": (float(m) if np.isfinite(m) else None),
                    "atr": (float(a) if np.isfinite(a) else None),
                    "bias_v": (float(v) if np.isfinite(v) else None),
                    "threshold": float(threshold),
                    "tp_trigger_price_raw": (
                        float(trigger_px_raw) if np.isfinite(trigger_px_raw) else None
                    ),
                    "tp_trigger_price_eff": (
                        float(trigger_px) if np.isfinite(trigger_px) else None
                    ),
                    "tp_triggered": True,
                    "tp_fill_price": (
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    "tp_trigger_source": trigger_source,
                    "gap_open_triggered": bool(gap_open_triggered),
                    "stop_fill_price": (
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    "stop_trigger_source": trigger_source,
                    "decision_pos": float(out[i]),
                    "in_pos_after": bool(in_pos),
                    "wait_next_entry_lock": bool(wait_next_entry_lock),
                }
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]
            tp_line_px = float("nan")
            entry_decision_date = None
            initial_tp_price = float("nan")
            prev_base = b
            continue

        out[i] = b
        trace_last_rows.append(
            {
                "date": ds,
                "event_type": "hold",
                "event_reason": "bias_v_take_profit_watch",
                "base_pos": float(b),
                "open": (float(o) if np.isfinite(o) else None),
                "high": (float(h) if np.isfinite(h) else None),
                "low": (float(l) if np.isfinite(l) else None),
                "close": (float(c) if np.isfinite(c) else None),
                "ma": (float(m) if np.isfinite(m) else None),
                "atr": (float(a) if np.isfinite(a) else None),
                "bias_v": (float(v) if np.isfinite(v) else None),
                "threshold": float(threshold),
                "tp_trigger_price_raw": (
                    float(trigger_px_raw) if np.isfinite(trigger_px_raw) else None
                ),
                "tp_trigger_price_eff": (
                    float(trigger_px) if np.isfinite(trigger_px) else None
                ),
                "tp_triggered": False,
                "tp_fill_price": None,
                "tp_trigger_source": None,
                "gap_open_triggered": None,
                "stop_fill_price": None,
                "stop_trigger_source": None,
                "decision_pos": float(out[i]),
                "in_pos_after": bool(in_pos),
                "wait_next_entry_lock": bool(wait_next_entry_lock),
            }
        )
        if len(trace_last_rows) > 120:
            trace_last_rows = trace_last_rows[-120:]
        prev_base = b

    out_s = pd.Series(out, index=base_pos.index, dtype=float)
    exits = int(((out_s <= 0.0) & (out_s.shift(1).fillna(0.0) > 0.0)).sum())
    trigger_count = int(len(trigger_dates))
    stats = {
        "enabled": True,
        "reentry_mode": reentry_v,
        "ma_window": int(ma_window),
        "atr_window": int(atr_window),
        "threshold": float(threshold),
        "trigger_count": trigger_count,
        "trigger_dates": trigger_dates[:200],
        "trigger_events": trigger_events[:200],
        "first_trigger_date": (trigger_dates[0] if trigger_dates else None),
        "last_trigger_date": (trigger_dates[-1] if trigger_dates else None),
        "entries": int(((out_s > 0.0) & (out_s.shift(1).fillna(0.0) <= 0.0)).sum()),
        "exits": exits,
        "trigger_exit_share": (
            float(trigger_count) / float(exits) if exits > 0 else 0.0
        ),
        "wait_next_entry_lock_active": bool(wait_next_entry_lock),
        "trigger_rule": "high_ge_threshold_same_day_exit",
        "fill_rule": "fill=max(trigger_price,open_price) for long",
        "trace_last_rows": trace_last_rows[-80:],
        "trade_records": trade_records,
    }
    return out_s, stats


def _validate_bt_single_inputs(inp: Any) -> None:
    strat = str(inp.strategy or "ma_filter").strip().lower()
    if strat not in _SUPPORTED_STRATEGIES:
        raise ValueError(f"invalid strategy={inp.strategy}")
    ma_type = str(getattr(inp, "ma_type", "sma") or "sma").strip().lower()
    if ma_type not in {"sma", "ema", "kama"}:
        raise ValueError("ma_type must be one of: sma|ema|kama")
    kama_fast_window = int(getattr(inp, "kama_fast_window", 2) or 2)
    kama_slow_window = int(getattr(inp, "kama_slow_window", 30) or 30)
    if kama_fast_window >= kama_slow_window:
        raise ValueError("kama_fast_window must be < kama_slow_window")
    if strat == "ma_cross" and ma_type == "kama":
        raise ValueError("ma_type=kama is only supported for ma_filter")
    ps = str(getattr(inp, "position_sizing", "equal") or "equal").strip().lower()
    if ps not in {"equal", "vol_target", "fixed_ratio", "risk_budget"}:
        raise ValueError(
            "position_sizing must be equal|vol_target|fixed_ratio|risk_budget"
        )
    risk_budget_pct = float(getattr(inp, "risk_budget_pct", 0.01) or 0.01)
    if (
        (not np.isfinite(risk_budget_pct))
        or risk_budget_pct < 0.001
        or risk_budget_pct > 0.02
    ):
        raise ValueError("risk_budget_pct must be in [0.001, 0.02]")
    if bool(getattr(inp, "vol_regime_risk_mgmt_enabled", False)):
        vt_expand = float(getattr(inp, "vol_ratio_expand_threshold", 1.45) or 1.45)
        vt_contract = float(getattr(inp, "vol_ratio_contract_threshold", 0.65) or 0.65)
        vt_normal = float(getattr(inp, "vol_ratio_normal_threshold", 1.05) or 1.05)
        if vt_expand <= vt_normal:
            raise ValueError(
                "vol_ratio_expand_threshold must be > vol_ratio_normal_threshold"
            )
        if vt_contract >= vt_normal:
            raise ValueError(
                "vol_ratio_contract_threshold must be < vol_ratio_normal_threshold"
            )


def _build_meta_params(inp: Any) -> dict[str, Any]:
    out = {
        "exec_price": str(getattr(inp, "exec_price", "open") or "open"),
        "cost_bps": float(getattr(inp, "cost_bps", 0.0)),
        "slippage_rate": float(getattr(inp, "slippage_rate", 0.0)),
        "position_sizing": str(getattr(inp, "position_sizing", "equal") or "equal"),
        "sma_window": int(getattr(inp, "sma_window", 20) or 20),
        "fast_window": int(getattr(inp, "fast_window", 5) or 5),
        "slow_window": int(getattr(inp, "slow_window", 20) or 20),
        "donchian_entry": int(getattr(inp, "donchian_entry", 20) or 20),
        "donchian_exit": int(getattr(inp, "donchian_exit", 10) or 10),
        "mom_lookback": int(getattr(inp, "mom_lookback", 252) or 252),
        "tsmom_entry_threshold": float(
            getattr(inp, "tsmom_entry_threshold", 0.0) or 0.0
        ),
        "tsmom_exit_threshold": float(getattr(inp, "tsmom_exit_threshold", 0.0) or 0.0),
        "bias_ma_window": int(getattr(inp, "bias_ma_window", 20) or 20),
        "bias_entry": float(getattr(inp, "bias_entry", 2.0) or 2.0),
        "bias_hot": float(getattr(inp, "bias_hot", 5.0) or 5.0),
        "bias_cold": float(getattr(inp, "bias_cold", -2.0) or -2.0),
        "bias_pos_mode": str(getattr(inp, "bias_pos_mode", "binary") or "binary"),
        "macd_fast": int(getattr(inp, "macd_fast", 12) or 12),
        "macd_slow": int(getattr(inp, "macd_slow", 26) or 26),
        "macd_signal": int(getattr(inp, "macd_signal", 9) or 9),
        "macd_v_atr_window": int(getattr(inp, "macd_v_atr_window", 14) or 14),
        "macd_v_scale": float(getattr(inp, "macd_v_scale", 1.0) or 1.0),
        "random_hold_days": int(getattr(inp, "random_hold_days", 20) or 20),
        "random_seed": (
            None
            if getattr(inp, "random_seed", 42) is None
            else int(getattr(inp, "random_seed", 42))
        ),
        "vol_window": int(getattr(inp, "vol_window", 20) or 20),
        "vol_target_ann": float(getattr(inp, "vol_target_ann", 0.20) or 0.20),
        "fixed_pos_ratio": float(getattr(inp, "fixed_pos_ratio", 0.04) or 0.04),
        "fixed_max_holdings": int(getattr(inp, "fixed_max_holdings", 10) or 10),
        "fixed_overcap_policy": str(
            getattr(inp, "fixed_overcap_policy", "extend") or "extend"
        ),
        "quick_mode": bool(getattr(inp, "quick_mode", False)),
        "risk_budget_atr_window": int(getattr(inp, "risk_budget_atr_window", 20) or 20),
        "risk_budget_pct": float(getattr(inp, "risk_budget_pct", 0.01) or 0.01),
        "risk_budget_overcap_policy": str(
            getattr(inp, "risk_budget_overcap_policy", "scale") or "scale"
        ),
        "risk_budget_max_leverage_multiple": float(
            getattr(inp, "risk_budget_max_leverage_multiple", 2.0) or 2.0
        ),
        "vol_regime_risk_mgmt_enabled": bool(
            getattr(inp, "vol_regime_risk_mgmt_enabled", False)
        ),
        "vol_ratio_fast_atr_window": int(
            getattr(inp, "vol_ratio_fast_atr_window", 5) or 5
        ),
        "vol_ratio_slow_atr_window": int(
            getattr(inp, "vol_ratio_slow_atr_window", 50) or 50
        ),
        "vol_ratio_expand_threshold": float(
            getattr(inp, "vol_ratio_expand_threshold", 1.45) or 1.45
        ),
        "vol_ratio_contract_threshold": float(
            getattr(inp, "vol_ratio_contract_threshold", 0.65) or 0.65
        ),
        "vol_ratio_normal_threshold": float(
            getattr(inp, "vol_ratio_normal_threshold", 1.05) or 1.05
        ),
        "atr_stop_mode": str(getattr(inp, "atr_stop_mode", "none") or "none"),
        "atr_stop_atr_basis": str(
            getattr(inp, "atr_stop_atr_basis", "latest") or "latest"
        ),
        "atr_stop_reentry_mode": str(
            getattr(inp, "atr_stop_reentry_mode", "reenter") or "reenter"
        ),
        "atr_stop_window": int(getattr(inp, "atr_stop_window", 14) or 14),
        "atr_stop_n": float(getattr(inp, "atr_stop_n", 2.0) or 2.0),
        "atr_stop_m": float(getattr(inp, "atr_stop_m", 0.5) or 0.5),
        "r_take_profit_enabled": bool(getattr(inp, "r_take_profit_enabled", False)),
        "r_take_profit_reentry_mode": str(
            getattr(inp, "r_take_profit_reentry_mode", "reenter") or "reenter"
        ),
        "bias_v_take_profit_enabled": bool(
            getattr(inp, "bias_v_take_profit_enabled", False)
        ),
        "bias_v_take_profit_reentry_mode": str(
            getattr(inp, "bias_v_take_profit_reentry_mode", "reenter") or "reenter"
        ),
        "bias_v_ma_window": int(getattr(inp, "bias_v_ma_window", 20) or 20),
        "bias_v_atr_window": int(getattr(inp, "bias_v_atr_window", 20) or 20),
        "bias_v_take_profit_threshold": float(
            getattr(inp, "bias_v_take_profit_threshold", 5.0) or 5.0
        ),
        "monthly_risk_budget_enabled": bool(
            getattr(inp, "monthly_risk_budget_enabled", False)
        ),
        "monthly_risk_budget_pct": float(
            getattr(inp, "monthly_risk_budget_pct", 0.06) or 0.06
        ),
        "monthly_risk_budget_include_new_trade_risk": bool(
            getattr(inp, "monthly_risk_budget_include_new_trade_risk", False)
        ),
        "er_filter": bool(getattr(inp, "er_filter", False)),
        "er_window": int(getattr(inp, "er_window", 10) or 10),
        "er_threshold": float(getattr(inp, "er_threshold", 0.30) or 0.30),
        "er_exit_filter": bool(getattr(inp, "er_exit_filter", False)),
        "er_exit_window": int(getattr(inp, "er_exit_window", 10) or 10),
        "er_exit_threshold": float(getattr(inp, "er_exit_threshold", 0.88) or 0.88),
        "impulse_entry_filter": bool(getattr(inp, "impulse_entry_filter", False)),
        "impulse_allow_bull": bool(getattr(inp, "impulse_allow_bull", True)),
        "impulse_allow_bear": bool(getattr(inp, "impulse_allow_bear", False)),
        "impulse_allow_neutral": bool(getattr(inp, "impulse_allow_neutral", False)),
        "ma_type": str(getattr(inp, "ma_type", "sma") or "sma"),
        "kama_er_window": int(getattr(inp, "kama_er_window", 10) or 10),
        "kama_fast_window": int(getattr(inp, "kama_fast_window", 2) or 2),
        "kama_slow_window": int(getattr(inp, "kama_slow_window", 30) or 30),
        "kama_std_window": int(getattr(inp, "kama_std_window", 20) or 20),
        "kama_std_coef": float(getattr(inp, "kama_std_coef", 1.0) or 1.0),
        "r_take_profit_tiers": _normalize_r_take_profit_tiers(
            getattr(inp, "r_take_profit_tiers", None)
        ),
        "risk_free_rate": float(getattr(inp, "risk_free_rate", 0.0) or 0.0),
        "dynamic_universe": bool(getattr(inp, "dynamic_universe", False)),
        "selection_mode": str(
            getattr(inp, "selection_mode", "all_active_candidates")
            or "all_active_candidates"
        ),
        "group_enforce": bool(getattr(inp, "group_enforce", False)),
        "group_pick_policy": str(
            getattr(inp, "group_pick_policy", "highest_sharpe") or "highest_sharpe"
        ),
        "group_max_holdings": int(getattr(inp, "group_max_holdings", 4) or 4),
        "asset_groups": dict(getattr(inp, "asset_groups", {}) or {}),
    }
    return out


def _risk_budget_frozen_weight(
    signal: pd.Series,
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr_window: int,
    risk_budget_pct: float,
) -> pd.Series:
    sig = (
        pd.to_numeric(signal, errors="coerce").astype(float).fillna(0.0).clip(lower=0.0)
    )
    c = pd.to_numeric(close, errors="coerce").astype(float).reindex(sig.index).ffill()
    h = (
        pd.to_numeric(high, errors="coerce")
        .astype(float)
        .reindex(sig.index)
        .ffill()
        .combine_first(c)
    )
    low_series = (
        pd.to_numeric(low, errors="coerce")
        .astype(float)
        .reindex(sig.index)
        .ffill()
        .combine_first(c)
    )
    atr = _atr_from_hlc(h, low_series, c, window=max(2, int(atr_window)))
    out = np.zeros(len(sig), dtype=float)
    in_pos = False
    frozen = 0.0
    rb = float(risk_budget_pct)
    for i, d in enumerate(sig.index):
        s = float(sig.loc[d]) if np.isfinite(float(sig.loc[d])) else 0.0
        if s <= 0.0:
            in_pos = False
            frozen = 0.0
            out[i] = 0.0
            continue
        if not in_pos:
            a = float(atr.loc[d]) if np.isfinite(float(atr.loc[d])) else np.nan
            px = float(c.loc[d]) if np.isfinite(float(c.loc[d])) else np.nan
            if np.isfinite(a) and a > 0.0 and np.isfinite(px) and px > 0.0:
                frozen = max(0.0, float(rb * px / a))
            else:
                frozen = 0.0
            in_pos = True
        out[i] = float(frozen)
    return pd.Series(out, index=sig.index, dtype=float)


def _as_nav(ret: pd.Series) -> pd.Series:
    s = (
        pd.to_numeric(ret, errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    nav = (1.0 + s).cumprod().astype(float)
    if len(nav) > 0:
        nav.iloc[0] = 1.0
    return nav


def _period_returns(nav: pd.Series, rule: str) -> list[dict[str, Any]]:
    if nav.empty:
        return []
    p = pd.to_numeric(nav, errors="coerce").astype(float).dropna()
    if p.empty:
        return []
    grp = p.resample(rule).last().dropna()
    if grp.empty:
        return []
    ret = _tsmom_rocp(grp, 1).dropna()
    out: list[dict[str, Any]] = []
    for d, v in ret.items():
        ds = pd.Timestamp(d).strftime("%Y-%m-%d")
        rv = float(v)
        out.append(
            {
                "date": ds,
                "strategy_return": rv,
                # Legacy-compatible aliases.
                "period_end": ds,
                "return": rv,
            }
        )
    return out


def _metrics_from_ret(ret: pd.Series, rf: float) -> dict[str, float]:
    s = (
        pd.to_numeric(ret, errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    nav = (1.0 + s).cumprod().astype(float)
    if len(nav) > 0:
        nav.iloc[0] = 1.0
    ann_ret = float(_annualized_return(nav))
    ulcer = float(_ulcer_index(nav))
    mdd_dur = float(_max_drawdown_duration_days(nav))
    out = {
        "cumulative_return": float((1.0 + s).prod() - 1.0),
        "annualized_return": ann_ret,
        "annualized_volatility": float(_annualized_vol(s)),
        "max_drawdown": float(_max_drawdown(nav)),
        "max_drawdown_duration_days": mdd_dur,
        "max_drawdown_recovery_days": int(mdd_dur),
        "sharpe_ratio": float(_sharpe(s, rf=float(rf))),
        "sortino_ratio": float(_sortino(s, rf=float(rf))),
        "ulcer_index": ulcer,
    }
    out["ulcer_performance_index"] = (
        float((ann_ret - float(rf)) / (ulcer / 100.0)) if ulcer > 0.0 else float("nan")
    )
    return out


def _normalize_trace_rows(rows: Any, keys: list[str]) -> list[dict[str, Any]]:
    src = list(rows or [])
    if not src:
        return [{k: None for k in keys}]
    out: list[dict[str, Any]] = []
    for r in src:
        rr = dict(r or {})
        for k in keys:
            rr.setdefault(k, None)
        out.append(rr)
    return out


def _build_signal_position(
    inp: Any,
    *,
    signal_close: pd.Series,
    signal_high: pd.Series,
    signal_low: pd.Series,
    code: str,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    strat = str(inp.strategy or "ma_filter").strip().lower()
    ma_type = str(getattr(inp, "ma_type", "sma") or "sma").strip().lower()
    if strat not in _SUPPORTED_STRATEGIES:
        raise ValueError(f"invalid strategy={inp.strategy}")
    if ma_type not in {"sma", "ema", "kama"}:
        raise ValueError("ma_type must be one of: sma|ema|kama")

    px = pd.to_numeric(signal_close, errors="coerce").astype(float)
    raw_pos: pd.Series
    score: pd.Series
    debug: dict[str, Any] = {"strategy": strat}

    if strat == "ma_filter":
        ma = _moving_average(
            px,
            window=int(inp.sma_window),
            ma_type=ma_type,
            kama_er_window=int(getattr(inp, "kama_er_window", 10)),
            kama_fast_window=int(getattr(inp, "kama_fast_window", 2)),
            kama_slow_window=int(getattr(inp, "kama_slow_window", 30)),
        )
        if ma_type == "kama":
            kstd_win = int(getattr(inp, "kama_std_window", 20))
            kstd = _rolling_std(
                ma.astype(float),
                window=kstd_win,
                min_periods=max(2, kstd_win // 2),
                ddof=0,
            ).fillna(0.0)
            raw_pos = _pos_from_band(
                px, ma, band=float(getattr(inp, "kama_std_coef", 1.0)) * kstd
            ).astype(float)
        else:
            raw_pos = (px > ma).astype(float).fillna(0.0)
        score = (px / ma - 1.0).replace([np.inf, -np.inf], np.nan).astype(float)
    elif strat == "ma_cross":
        fast = _moving_average(px, window=int(inp.fast_window), ma_type=ma_type)
        slow = _moving_average(px, window=int(inp.slow_window), ma_type=ma_type)
        raw_pos = (fast > slow).astype(float).fillna(0.0)
        score = (fast / slow - 1.0).replace([np.inf, -np.inf], np.nan).astype(float)
    elif strat == "donchian":
        raw_pos = _pos_from_donchian(
            px, entry=int(inp.donchian_entry), exit_=int(inp.donchian_exit)
        ).astype(float)
        hi = _donchian_prev_high(px, int(inp.donchian_entry))
        score = (px / hi - 1.0).replace([np.inf, -np.inf], np.nan).astype(float)
    elif strat == "linreg_slope":
        n = int(inp.sma_window)
        y = np.log(px.clip(lower=1e-12).astype(float))
        slope = _rolling_linreg_slope(y, n)
        raw_pos = (slope > 0.0).astype(float).fillna(0.0)
        score = slope.astype(float)
    elif strat == "bias":
        b_win = int(inp.bias_ma_window)
        ema = _ema(px, b_win)
        ln_c = np.log(px.clip(lower=1e-12))
        ln_ema = np.log(ema.clip(lower=1e-12))
        bias = (ln_c - ln_ema) * 100.0
        entry = float(inp.bias_entry)
        hot = float(inp.bias_hot)
        cold = float(inp.bias_cold)
        mode = str(inp.bias_pos_mode or "binary").strip().lower()
        pos = np.zeros(len(px), dtype=float)
        in_pos = False
        for i, d in enumerate(px.index):
            b = float(bias.loc[d]) if np.isfinite(float(bias.loc[d])) else np.nan
            if not np.isfinite(b):
                pos[i] = 0.0
                in_pos = False
                continue
            if not in_pos:
                if b > entry:
                    in_pos = True
                    pos[i] = (
                        1.0
                        if mode == "binary"
                        else float(np.clip((b - cold) / (hot - cold), 0.0, 1.0))
                    )
                else:
                    pos[i] = 0.0
            else:
                if b >= hot or b <= cold:
                    in_pos = False
                    pos[i] = 0.0
                else:
                    pos[i] = (
                        1.0
                        if mode == "binary"
                        else float(np.clip((b - cold) / (hot - cold), 0.0, 1.0))
                    )
        raw_pos = pd.Series(pos, index=px.index, dtype=float)
        score = bias.replace([np.inf, -np.inf], np.nan).astype(float)
    elif strat == "macd_cross":
        macd, sig, _ = _macd_core(
            px,
            fast=int(inp.macd_fast),
            slow=int(inp.macd_slow),
            signal=int(inp.macd_signal),
        )
        raw_pos = (macd > sig).astype(float).fillna(0.0)
        score = (macd - sig).replace([np.inf, -np.inf], np.nan).astype(float)
    elif strat == "macd_zero_filter":
        macd, _, _ = _macd_core(
            px,
            fast=int(inp.macd_fast),
            slow=int(inp.macd_slow),
            signal=int(inp.macd_signal),
        )
        raw_pos = (macd > 0.0).astype(float).fillna(0.0)
        score = macd.replace([np.inf, -np.inf], np.nan).astype(float)
    elif strat == "macd_v":
        macd, _, _ = _macd_core(
            px,
            fast=int(inp.macd_fast),
            slow=int(inp.macd_slow),
            signal=int(inp.macd_signal),
        )
        atr = _atr_from_hlc(
            signal_high, signal_low, px, window=int(inp.macd_v_atr_window)
        )
        macd_v = (macd / atr.replace(0.0, np.nan)) * float(inp.macd_v_scale)
        macd_v_sig = _ema(macd_v, int(inp.macd_signal))
        raw_pos = (macd_v > macd_v_sig).astype(float).fillna(0.0)
        score = (macd_v - macd_v_sig).replace([np.inf, -np.inf], np.nan).astype(float)
    elif strat == "random_entry":
        raw_pos = _pos_from_random_entry_hold(
            px.index,
            hold_days=int(getattr(inp, "random_hold_days", 20)),
            seed=getattr(inp, "random_seed", 42),
        )
        score = raw_pos.astype(float)
    else:
        mom = _tsmom_rocp(px, int(inp.mom_lookback))
        raw_pos = _pos_from_tsmom(
            mom,
            entry_threshold=float(inp.tsmom_entry_threshold),
            exit_threshold=float(inp.tsmom_exit_threshold),
        )
        score = mom.replace([np.inf, -np.inf], np.nan).astype(float)

    out = raw_pos.astype(float).fillna(0.0).clip(lower=0.0)

    if bool(getattr(inp, "er_filter", False)):
        er = _efficiency_ratio(px, window=int(getattr(inp, "er_window", 10)))
        out, er_stats = _apply_er_entry_filter(
            out, er=er, threshold=float(getattr(inp, "er_threshold", 0.30))
        )
        debug["er_filter"] = er_stats
    if bool(getattr(inp, "impulse_entry_filter", False)):
        st = _compute_impulse_state(
            px,
            ema_window=13,
            macd_fast=int(inp.macd_fast),
            macd_slow=int(inp.macd_slow),
            macd_signal=int(inp.macd_signal),
        )
        out, imp_stats = _apply_impulse_entry_filter(
            out,
            impulse_state=st,
            allow_bull=bool(getattr(inp, "impulse_allow_bull", True)),
            allow_bear=bool(getattr(inp, "impulse_allow_bear", False)),
            allow_neutral=bool(getattr(inp, "impulse_allow_neutral", False)),
        )
        debug["impulse_filter"] = imp_stats
    if bool(getattr(inp, "er_exit_filter", False)):
        er_exit = _efficiency_ratio(px, window=int(getattr(inp, "er_exit_window", 10)))
        out, ex_stats = _apply_er_exit_filter(
            out, er=er_exit, threshold=float(getattr(inp, "er_exit_threshold", 0.88))
        )
        debug["er_exit_filter"] = ex_stats
    debug["signal_mode"] = (
        "fractional_to_binary"
        if (out.max() > 1.0 or out.min() < 0.0 or (out % 1.0).abs().sum() > 0.0)
        else "binary"
    )
    if strat == "random_entry" and getattr(inp, "random_seed", None) is None:
        debug["random_seed_note"] = (
            f"random_seed=None for {code}, run is non-deterministic"
        )
    debug["score_available_count"] = int(
        pd.Series(score).replace([np.inf, -np.inf], np.nan).notna().sum()
    )
    return out, score.replace([np.inf, -np.inf], np.nan).astype(float), debug


def _build_bt_frame(
    db: Session,
    *,
    code: str,
    start: dt.date,
    end: dt.date,
) -> tuple[pd.DataFrame, pd.Series]:
    ohlc_none = load_ohlc_prices(db, codes=[code], start=start, end=end, adjust="none")
    ohlc_qfq = load_ohlc_prices(db, codes=[code], start=start, end=end, adjust="qfq")
    ohlc_hfq = load_ohlc_prices(db, codes=[code], start=start, end=end, adjust="hfq")
    close_none = load_close_prices(
        db, codes=[code], start=start, end=end, adjust="none"
    )
    close_qfq = load_close_prices(db, codes=[code], start=start, end=end, adjust="qfq")
    close_hfq = load_close_prices(db, codes=[code], start=start, end=end, adjust="hfq")
    high_qfq, low_qfq = load_high_low_prices(
        db, codes=[code], start=start, end=end, adjust="qfq"
    )

    if close_none.empty or code not in close_none.columns:
        raise ValueError(f"no execution data for {code}")
    idx = close_none.sort_index().index

    def _pick(
        ohlc: dict[str, pd.DataFrame], field: str, fallback: pd.Series
    ) -> pd.Series:
        df = (
            ohlc.get(field, pd.DataFrame())
            if isinstance(ohlc, dict)
            else pd.DataFrame()
        )
        if df is None or df.empty or code not in df.columns:
            return fallback.astype(float)
        return (
            pd.to_numeric(df[code], errors="coerce")
            .astype(float)
            .reindex(idx)
            .ffill()
            .combine_first(fallback.astype(float))
        )

    px_none = (
        pd.to_numeric(close_none[code], errors="coerce")
        .astype(float)
        .reindex(idx)
        .ffill()
    )
    bt_df = pd.DataFrame(index=idx)
    bt_df["Open"] = _pick(ohlc_none, "open", px_none)
    bt_df["High"] = _pick(ohlc_none, "high", bt_df["Open"])
    bt_df["Low"] = _pick(ohlc_none, "low", bt_df["Open"])
    bt_df["Close"] = _pick(ohlc_none, "close", px_none)
    bt_df["Volume"] = _pick(ohlc_none, "volume", pd.Series(0.0, index=idx))

    sig_close = (
        pd.to_numeric(close_qfq.get(code, px_none), errors="coerce")
        .astype(float)
        .reindex(idx)
        .ffill()
    )
    sig_open = _pick(ohlc_qfq, "open", sig_close)
    sig_high = (
        pd.to_numeric(high_qfq.get(code, sig_close), errors="coerce")
        .astype(float)
        .reindex(idx)
        .ffill()
    )
    sig_low = (
        pd.to_numeric(low_qfq.get(code, sig_close), errors="coerce")
        .astype(float)
        .reindex(idx)
        .ffill()
    )
    hfq_close = (
        pd.to_numeric(close_hfq.get(code, bt_df["Close"]), errors="coerce")
        .astype(float)
        .reindex(idx)
        .ffill()
    )
    hfq_open = _pick(ohlc_hfq, "open", hfq_close)

    bt_df = bt_df.dropna(subset=["Open", "High", "Low", "Close"], how="any")
    sig_close = sig_close.reindex(bt_df.index).ffill()
    sig_open = sig_open.reindex(bt_df.index).ffill()
    sig_high = sig_high.reindex(bt_df.index).ffill()
    sig_low = sig_low.reindex(bt_df.index).ffill()
    hfq_close = hfq_close.reindex(bt_df.index).ffill()
    hfq_open = hfq_open.reindex(bt_df.index).ffill()
    bt_df["HfqOpen"] = hfq_open
    bt_df["HfqClose"] = hfq_close
    bt_df["SigOpen"] = sig_open
    bt_df["SigClose"] = sig_close
    bt_df["SigHigh"] = sig_high
    bt_df["SigLow"] = sig_low
    return bt_df, hfq_close


def _run_single_backtesting(
    db: Session,
    inp: Any,
    *,
    code: str,
    random_seed: int | None,
) -> dict[str, Any]:
    use_backtesting = True
    try:
        from backtesting import Backtest, Strategy
    except Exception:
        use_backtesting = False

    bt_df, hfq_close = _build_bt_frame(db, code=code, start=inp.start, end=inp.end)
    if bt_df.empty:
        raise ValueError(f"no valid OHLC rows for {code}")

    inp_local = _clone_like_input(inp, code=code, random_seed=random_seed)
    raw_pos, score_sig, debug_sig = _build_signal_position(
        inp_local,
        signal_close=bt_df["SigClose"],
        signal_high=bt_df["SigHigh"],
        signal_low=bt_df["SigLow"],
        code=code,
    )
    ep = str(getattr(inp, "exec_price", "open") or "open").strip().lower()
    open_none = bt_df["Open"].astype(float).combine_first(bt_df["Close"].astype(float))
    close_none = bt_df["Close"].astype(float)
    open_hfq = (
        bt_df["HfqOpen"].astype(float).combine_first(bt_df["HfqClose"].astype(float))
    )
    close_hfq = bt_df["HfqClose"].astype(float)
    if ep == "open":
        ret_exec_none = _forward_simple_return(open_none)
        ret_exec_hfq = _forward_simple_return(open_hfq)
        px_exec_none = open_none.astype(float)
        px_exec_hfq = open_hfq.astype(float)
    elif ep == "oc2":
        ret_open_none = _forward_simple_return(open_none)
        ret_close_none = _forward_simple_return(close_none)
        ret_open_hfq = _forward_simple_return(open_hfq)
        ret_close_hfq = _forward_simple_return(close_hfq)
        ret_exec_none = (0.5 * (ret_open_none + ret_close_none)).astype(float)
        ret_exec_hfq = (0.5 * (ret_open_hfq + ret_close_hfq)).astype(float)
        px_exec_none = (0.5 * (open_none + close_none)).astype(float)
        px_exec_hfq = (0.5 * (open_hfq + close_hfq)).astype(float)
    else:
        ret_exec_none = _forward_simple_return(close_none)
        ret_exec_hfq = _forward_simple_return(close_hfq)
        px_exec_none = close_none.astype(float)
        px_exec_hfq = close_hfq.astype(float)
    gross_none = (1.0 + ret_exec_none).astype(float)
    gross_hfq = (1.0 + ret_exec_hfq).astype(float)
    _, ca_mask = corporate_action_mask(
        gross_none.to_frame(code), gross_hfq.to_frame(code)
    )
    ca_m = (
        ca_mask[code].reindex(bt_df.index).fillna(False)
        if isinstance(ca_mask, pd.DataFrame) and code in ca_mask.columns
        else pd.Series(False, index=bt_df.index)
    )
    ret_exec = ret_exec_none.where(~ca_m, other=ret_exec_hfq).astype(float)
    ret_exec_raw = ret_exec.copy().astype(float)
    px_exec_slip = (
        px_exec_none.where(~ca_m, other=px_exec_hfq)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .astype(float)
    )

    atr_mode = str(getattr(inp, "atr_stop_mode", "none") or "none").strip().lower()
    atr_basis = (
        str(getattr(inp, "atr_stop_atr_basis", "latest") or "latest").strip().lower()
    )
    atr_reentry_mode = (
        str(getattr(inp, "atr_stop_reentry_mode", "reenter") or "reenter")
        .strip()
        .lower()
    )
    rtp_enabled = bool(getattr(inp, "r_take_profit_enabled", False))
    rtp_reentry_mode = (
        str(getattr(inp, "r_take_profit_reentry_mode", "reenter") or "reenter")
        .strip()
        .lower()
    )
    bias_v_tp_enabled = bool(getattr(inp, "bias_v_take_profit_enabled", False))
    bias_v_tp_reentry_mode = (
        str(getattr(inp, "bias_v_take_profit_reentry_mode", "reenter") or "reenter")
        .strip()
        .lower()
    )
    monthly_enabled = bool(getattr(inp, "monthly_risk_budget_enabled", False))
    ps = str(getattr(inp, "position_sizing", "equal") or "equal").strip().lower()
    simple_backtesting_mode = bool(
        use_backtesting
        and ps == "equal"
        and atr_mode == "none"
        and (not rtp_enabled)
        and (not bias_v_tp_enabled)
        and (not monthly_enabled)
        and float(raw_pos.max()) <= 1.0
        and float(raw_pos.min()) >= 0.0
        and float((raw_pos % 1.0).abs().sum()) <= 1e-12
    )

    atr_stop_stats: dict[str, Any] = {
        "enabled": False,
        "trigger_count": 0,
        "trigger_events": [],
        "trace_last_rows": [],
    }
    bias_v_tp_stats: dict[str, Any] = {
        "enabled": False,
        "trigger_count": 0,
        "trigger_events": [],
        "trace_last_rows": [],
    }
    r_take_profit_stats: dict[str, Any] = {
        "enabled": False,
        "trigger_count": 0,
        "tier_trigger_counts": {},
        "trigger_events": [],
        "trace_last_rows": [],
    }
    vol_risk_stats = {
        "vol_risk_adjust_total_count": 0,
        "vol_risk_adjust_reduce_on_expand_count": 0,
        "vol_risk_adjust_increase_on_contract_count": 0,
        "vol_risk_adjust_recover_from_expand_count": 0,
        "vol_risk_adjust_recover_from_contract_count": 0,
        "vol_risk_entry_state_reduce_on_expand_count": 0,
        "vol_risk_entry_state_increase_on_contract_count": 0,
    }
    monthly_gate_stats = {
        "enabled": False,
        "budget_pct": float(getattr(inp, "monthly_risk_budget_pct", 0.06) or 0.06),
        "include_new_trade_risk": bool(
            getattr(inp, "monthly_risk_budget_include_new_trade_risk", False)
        ),
        "attempted_entry_count": 0,
        "attempted_entry_count_by_code": {str(code): 0},
        "blocked_entry_count": 0,
        "blocked_entry_count_by_code": {str(code): 0},
    }

    base_pos = raw_pos.astype(float).fillna(0.0)
    raw_pos_for_exec = base_pos.copy()
    if not simple_backtesting_mode:
        raw_pos_for_exec, atr_stop_stats = _apply_atr_stop(
            raw_pos_for_exec,
            open_=bt_df["SigOpen"].astype(float),
            close=bt_df["SigClose"].astype(float),
            high=bt_df["SigHigh"].astype(float),
            low=bt_df["SigLow"].astype(float),
            mode=atr_mode,
            atr_basis=atr_basis,
            reentry_mode=atr_reentry_mode,
            atr_window=int(getattr(inp, "atr_stop_window", 14)),
            n_mult=float(getattr(inp, "atr_stop_n", 2.0)),
            m_step=float(getattr(inp, "atr_stop_m", 0.5)),
        )
        atr_stop_stats = {
            **(atr_stop_stats or {}),
            **_extract_atr_plan_stops_from_trace(atr_stop_stats or {}),
        }
        raw_pos_for_exec, bias_v_tp_stats = _apply_bias_v_take_profit(
            raw_pos_for_exec,
            open_=bt_df["SigOpen"].astype(float),
            close=bt_df["SigClose"].astype(float),
            high=bt_df["SigHigh"].astype(float),
            low=bt_df["SigLow"].astype(float),
            enabled=bias_v_tp_enabled,
            reentry_mode=bias_v_tp_reentry_mode,
            ma_window=int(getattr(inp, "bias_v_ma_window", 20)),
            atr_window=int(getattr(inp, "bias_v_atr_window", 20)),
            threshold=float(getattr(inp, "bias_v_take_profit_threshold", 5.0)),
        )
        raw_pos_for_exec, r_take_profit_stats = _apply_r_multiple_take_profit(
            raw_pos_for_exec,
            open_=bt_df["SigOpen"].astype(float),
            close=bt_df["SigClose"].astype(float),
            high=bt_df["SigHigh"].astype(float),
            low=bt_df["SigLow"].astype(float),
            enabled=rtp_enabled,
            reentry_mode=rtp_reentry_mode,
            atr_window=int(getattr(inp, "atr_stop_window", 14)),
            atr_n=float(getattr(inp, "atr_stop_n", 2.0)),
            tiers=_normalize_r_take_profit_tiers(
                getattr(inp, "r_take_profit_tiers", None)
            ),
            atr_stop_enabled=bool(atr_mode != "none"),
        )

        sizing_scale = pd.Series(1.0, index=raw_pos_for_exec.index, dtype=float)
        if ps == "fixed_ratio":
            sizing_scale = pd.Series(
                float(getattr(inp, "fixed_pos_ratio", 0.04) or 0.04),
                index=raw_pos_for_exec.index,
                dtype=float,
            )
        elif ps == "risk_budget":
            atr_rb = _atr_from_hlc(
                bt_df["SigHigh"],
                bt_df["SigLow"],
                bt_df["SigClose"],
                window=int(getattr(inp, "risk_budget_atr_window", 20)),
            )
            atr_fast = _atr_from_hlc(
                bt_df["SigHigh"],
                bt_df["SigLow"],
                bt_df["SigClose"],
                window=int(getattr(inp, "vol_ratio_fast_atr_window", 5)),
            )
            atr_slow = _atr_from_hlc(
                bt_df["SigHigh"],
                bt_df["SigLow"],
                bt_df["SigClose"],
                window=int(getattr(inp, "vol_ratio_slow_atr_window", 50)),
            )
            sizing_scale, vol_risk_stats = _risk_budget_dynamic_weights(
                raw_pos_for_exec.astype(float).fillna(0.0),
                close=bt_df["SigClose"].astype(float),
                atr_for_budget=atr_rb.astype(float),
                atr_fast=atr_fast.astype(float),
                atr_slow=atr_slow.astype(float),
                risk_budget_pct=float(getattr(inp, "risk_budget_pct", 0.01) or 0.01),
                dynamic_enabled=bool(
                    getattr(inp, "vol_regime_risk_mgmt_enabled", False)
                ),
                expand_threshold=float(
                    getattr(inp, "vol_ratio_expand_threshold", 1.45) or 1.45
                ),
                contract_threshold=float(
                    getattr(inp, "vol_ratio_contract_threshold", 0.65) or 0.65
                ),
                normal_threshold=float(
                    getattr(inp, "vol_ratio_normal_threshold", 1.05) or 1.05
                ),
            )
        elif ps == "vol_target":
            asset_vol = (
                _rolling_std(
                    ret_exec,
                    window=max(2, int(getattr(inp, "vol_window", 20) or 20)),
                    min_periods=max(2, int(getattr(inp, "vol_window", 20) or 20)),
                    ddof=1,
                ).mul(np.sqrt(252))
            ).replace([np.inf, -np.inf], np.nan)
            sizing_scale = (
                (float(getattr(inp, "vol_target_ann", 0.20) or 0.20) / asset_vol)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .clip(lower=0.0, upper=1.0)
                .astype(float)
            )
        raw_pos_for_exec = (
            raw_pos_for_exec.clip(lower=0.0, upper=1.0) * sizing_scale
        ).astype(float)

        if monthly_enabled:
            atr_gate = _atr_from_hlc(
                bt_df["SigHigh"],
                bt_df["SigLow"],
                bt_df["SigClose"],
                window=int(getattr(inp, "atr_stop_window", 14)),
            )
            gated_df, monthly_gate_stats = _apply_monthly_risk_budget_gate(
                raw_pos_for_exec.to_frame(code),
                close=bt_df["SigClose"].to_frame(code),
                atr=atr_gate.to_frame(code),
                enabled=True,
                budget_pct=float(getattr(inp, "monthly_risk_budget_pct", 0.06) or 0.06),
                include_new_trade_risk=bool(
                    getattr(inp, "monthly_risk_budget_include_new_trade_risk", False)
                ),
                atr_stop_enabled=bool(atr_mode != "none"),
                atr_mode=str(atr_mode),
                atr_basis=str(atr_basis),
                atr_n=float(getattr(inp, "atr_stop_n", 2.0) or 2.0),
                atr_m=float(getattr(inp, "atr_stop_m", 0.5) or 0.5),
                fallback_position_risk=0.02,
            )
            raw_pos_for_exec = gated_df[code].astype(float)

    signal_pos = raw_pos_for_exec.astype(float).fillna(0.0)
    bt_df["DesiredPos"] = (
        (signal_pos > 0.0).astype(float)
        if simple_backtesting_mode
        else signal_pos.astype(float)
    )
    if simple_backtesting_mode:
        signal_lookback = 2 if ep in {"close", "oc2"} else 1

        class BtTrendStrategy(Strategy):
            def init(self) -> None:
                return

            def next(self) -> None:
                if len(self.data.Close) < signal_lookback:
                    return
                raw_target = float(self.data.DesiredPos[-signal_lookback])
                target = raw_target if np.isfinite(raw_target) else 0.0
                if target > 0.0 and not self.position:
                    self.buy(size=0.999999)
                elif target <= 0.0 and self.position:
                    self.position.close()

        trade_on_close = ep in {"close", "oc2"}
        bt = Backtest(
            bt_df,
            BtTrendStrategy,
            cash=1_000_000.0,
            spread=float(getattr(inp, "slippage_rate", 0.0) or 0.0),
            commission=float(getattr(inp, "cost_bps", 0.0) or 0.0) / 10000.0,
            trade_on_close=trade_on_close,
            exclusive_orders=True,
            finalize_trades=True,
        )
        stats = bt.run()
        equity_curve = stats.get("_equity_curve")
        if equity_curve is None or "Equity" not in equity_curve:
            raise ValueError(f"failed to build equity curve for {code}")
        eq = pd.Series(
            equity_curve["Equity"],
            index=pd.to_datetime(equity_curve.index),
            dtype=float,
        ).sort_index()
        nav = (eq / float(eq.iloc[0])).ffill().fillna(1.0)
        strat_ret = _tsmom_rocp(nav, 1).fillna(0.0).astype(float)
        pos_eff = bt_df["DesiredPos"].shift(1).fillna(0.0).astype(float)
        atr_override_ret = pd.Series(0.0, index=bt_df.index, dtype=float)
        bias_override_ret = pd.Series(0.0, index=bt_df.index, dtype=float)
        rtp_override_ret = pd.Series(0.0, index=bt_df.index, dtype=float)
        runtime_engine = "backtesting"
    else:
        pos_eff = bt_df["DesiredPos"].shift(1).fillna(0.0).astype(float).clip(lower=0.0)
        atr_override_ret = pd.Series(0.0, index=bt_df.index, dtype=float)
        bias_override_ret = pd.Series(0.0, index=bt_df.index, dtype=float)
        rtp_override_ret = pd.Series(0.0, index=bt_df.index, dtype=float)
        if not simple_backtesting_mode:
            pos_eff, atr_override_ret = _apply_intraday_stop_execution_single(
                weights=pos_eff,
                atr_stop_stats=atr_stop_stats,
                exec_price=str(ep),
                open_sig=bt_df["SigOpen"].astype(float),
                close_sig=bt_df["SigClose"].astype(float),
            )
            pos_eff, bias_override_ret = _apply_intraday_stop_execution_single(
                weights=pos_eff,
                atr_stop_stats=bias_v_tp_stats,
                exec_price=str(ep),
                open_sig=bt_df["SigOpen"].astype(float),
                close_sig=bt_df["SigClose"].astype(float),
            )
            pos_eff, rtp_override_ret = _apply_intraday_stop_execution_single(
                weights=pos_eff,
                atr_stop_stats=r_take_profit_stats,
                exec_price=str(ep),
                open_sig=bt_df["SigOpen"].astype(float),
                close_sig=bt_df["SigClose"].astype(float),
            )
        ret_exec_use = ret_exec.copy().astype(float)
        if ep in {"open", "oc2"}:
            same_day_none = _ratio_simple_return(bt_df["Close"], bt_df["Open"])
            same_day_hfq = _ratio_simple_return(bt_df["HfqClose"], bt_df["HfqOpen"])
            open_fwd_none = _forward_simple_return(bt_df["Open"])
            open_fwd_hfq = _forward_simple_return(bt_df["HfqOpen"])
            close_fwd_none = _forward_simple_return(bt_df["Close"])
            close_fwd_hfq = _forward_simple_return(bt_df["HfqClose"])
            cm = ca_m.reindex(ret_exec_use.index).fillna(False).astype(bool)
            w_ix = pos_eff.reindex(ret_exec_use.index).fillna(0.0).astype(float)
            if ep == "open":
                for d in ret_exec_use.index:
                    if float(w_ix.loc[d]) <= 1e-12:
                        continue
                    ret_exec_use.loc[d] = (
                        float(same_day_hfq.loc[d])
                        if bool(cm.loc[d])
                        else float(same_day_none.loc[d])
                    )
            else:
                ret_blend_none = pd.Series(0.0, index=ret_exec_use.index, dtype=float)
                ret_blend_hfq = pd.Series(0.0, index=ret_exec_use.index, dtype=float)
                for d in ret_exec_use.index:
                    hold = float(w_ix.loc[d]) > 1e-12
                    po_n = (
                        float(same_day_none.loc[d])
                        if hold
                        else float(open_fwd_none.loc[d])
                    )
                    po_h = (
                        float(same_day_hfq.loc[d])
                        if hold
                        else float(open_fwd_hfq.loc[d])
                    )
                    cn = float(close_fwd_none.loc[d])
                    ch = float(close_fwd_hfq.loc[d])
                    ret_blend_none.loc[d] = 0.5 * (po_n + cn)
                    ret_blend_hfq.loc[d] = 0.5 * (po_h + ch)
                ret_exec_use = ret_blend_none.where(~cm, ret_blend_hfq).astype(float)
        base_ret = (
            (pos_eff * ret_exec_use).astype(float)
            + atr_override_ret.astype(float)
            + bias_override_ret.astype(float)
            + rtp_override_ret.astype(float)
        )
        turnover = pos_eff.diff().abs().fillna(pos_eff.abs()) / 2.0
        cost_comm = turnover * (float(getattr(inp, "cost_bps", 0.0) or 0.0) / 10000.0)
        cost_slip = slippage_return_from_turnover(
            turnover,
            slippage_spread=float(getattr(inp, "slippage_rate", 0.0) or 0.0),
            exec_price=px_exec_slip,
        )
        strat_ret = (base_ret - cost_comm - cost_slip).fillna(0.0).astype(float)
        nav = _as_nav(strat_ret)
        ret_exec = ret_exec_use.astype(float)
        stats = {
            "_trades": pd.DataFrame(),
            "# Trades": int(
                ((pos_eff > 0) & (pos_eff.shift(1).fillna(0.0) <= 0)).sum()
            ),
        }
        runtime_engine = "semantic_vectorized"

    ret_hfq_cc = (
        _tsmom_rocp(bt_df["HfqClose"].astype(float), 1).fillna(0.0).astype(float)
    )
    cm_bh = ca_m.reindex(ret_hfq_cc.index).fillna(False).astype(bool)
    bh_same_none = _ratio_simple_return(bt_df["Close"], bt_df["Open"])
    bh_same_hfq = _ratio_simple_return(bt_df["HfqClose"], bt_df["HfqOpen"])
    if ep == "close":
        bh_ret = ret_hfq_cc.astype(float)
    elif ep == "open":
        bh_ret = (
            bh_same_none.where(~cm_bh, bh_same_hfq)
            .astype(float)
            .reindex(ret_hfq_cc.index)
            .fillna(0.0)
        )
    else:
        cf_none = _forward_simple_return(bt_df["Close"])
        cf_hfq = _forward_simple_return(bt_df["HfqClose"])
        blend_bh_none = (0.5 * (bh_same_none + cf_none)).astype(float)
        blend_bh_hfq = (0.5 * (bh_same_hfq + cf_hfq)).astype(float)
        bh_ret = (
            blend_bh_none.where(~cm_bh, blend_bh_hfq)
            .astype(float)
            .reindex(ret_hfq_cc.index)
            .fillna(0.0)
        )
    bh_nav = _as_nav(bh_ret)
    excess_nav = (nav / bh_nav.replace(0.0, np.nan)).fillna(1.0)
    excess_ret = _tsmom_rocp(excess_nav, 1).fillna(0.0).astype(float)

    trades_df = stats.get("_trades")
    trades: list[dict[str, Any]] = []
    if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
        for _, row in trades_df.iterrows():
            et = row.get("EntryTime")
            xt = row.get("ExitTime")
            entry_date = pd.Timestamp(et).strftime("%Y-%m-%d") if pd.notna(et) else None
            exit_date = pd.Timestamp(xt).strftime("%Y-%m-%d") if pd.notna(xt) else None
            trades.append(
                {
                    "code": str(code),
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": float(row.get("EntryPrice"))
                    if pd.notna(row.get("EntryPrice"))
                    else None,
                    "exit_price": float(row.get("ExitPrice"))
                    if pd.notna(row.get("ExitPrice"))
                    else None,
                    "return_pct": float(row.get("ReturnPct"))
                    if pd.notna(row.get("ReturnPct"))
                    else None,
                    "pnl": float(row.get("PnL")) if pd.notna(row.get("PnL")) else None,
                    "size": float(row.get("Size"))
                    if pd.notna(row.get("Size"))
                    else None,
                    "duration_days": int(row.get("Duration").days)
                    if pd.notna(row.get("Duration"))
                    else None,
                }
            )

    if not trades:
        sig = bt_df["DesiredPos"].reindex(nav.index).fillna(0.0).astype(float)
        in_pos = False
        entry_date = None
        entry_price = None
        for d in sig.index:
            s = float(sig.loc[d])
            if (not in_pos) and s > 0.0:
                in_pos = True
                entry_date = pd.Timestamp(d).strftime("%Y-%m-%d")
                entry_price = float(bt_df.loc[d, "Close"]) if d in bt_df.index else None
            elif in_pos and s <= 0.0:
                exit_date = pd.Timestamp(d).strftime("%Y-%m-%d")
                exit_price = float(bt_df.loc[d, "Close"]) if d in bt_df.index else None
                r = None
                if entry_price and exit_price and entry_price > 0:
                    r = float(exit_price / entry_price - 1.0)
                trades.append(
                    {
                        "code": str(code),
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return_pct": r,
                        "pnl": None,
                        "size": 1.0,
                        "duration_days": None,
                    }
                )
                in_pos = False
                entry_date = None
                entry_price = None

    return {
        "code": code,
        "dates": [d.strftime("%Y-%m-%d") for d in nav.index],
        "nav": nav,
        "buy_hold_nav": bh_nav.reindex(nav.index).ffill().fillna(1.0),
        "excess_nav": excess_nav.reindex(nav.index).ffill().fillna(1.0),
        "strat_ret": strat_ret,
        "bench_ret": bh_ret.reindex(nav.index).fillna(0.0),
        "excess_ret": excess_ret,
        "desired_pos": bt_df["DesiredPos"].reindex(nav.index).fillna(0.0),
        "base_pos": base_pos.reindex(nav.index).fillna(0.0).astype(float),
        "signal_pos": signal_pos.reindex(nav.index).fillna(0.0).astype(float),
        "sig_open": bt_df["SigOpen"].reindex(nav.index).ffill(),
        "sig_close": bt_df["SigClose"].reindex(nav.index).ffill(),
        "sig_high": bt_df["SigHigh"].reindex(nav.index).ffill(),
        "sig_low": bt_df["SigLow"].reindex(nav.index).ffill(),
        "ret_exec": ret_exec.reindex(nav.index).fillna(0.0).astype(float),
        "ret_exec_raw": ret_exec_raw.reindex(nav.index).fillna(0.0).astype(float),
        "px_exec_slip": px_exec_slip.reindex(nav.index).ffill().astype(float),
        "ret_exec_none": ret_exec_none.reindex(nav.index).fillna(0.0).astype(float),
        "ret_exec_hfq": ret_exec_hfq.reindex(nav.index).fillna(0.0).astype(float),
        "exec_open_none": bt_df["Open"].reindex(nav.index).astype(float),
        "exec_close_none": bt_df["Close"].reindex(nav.index).astype(float),
        "exec_open_hfq": bt_df["HfqOpen"].reindex(nav.index).astype(float),
        "exec_close_hfq": bt_df["HfqClose"].reindex(nav.index).astype(float),
        "corp_factor": (gross_hfq / gross_none.replace(0.0, np.nan))
        .reindex(nav.index)
        .replace([np.inf, -np.inf], np.nan)
        .astype(float),
        "ca_mask": ca_m.reindex(nav.index).fillna(False).astype(bool),
        "trades": trades,
        "trade_count": int(stats.get("# Trades", 0) or 0),
        "signal_debug": debug_sig,
        "signal_score": score_sig.reindex(nav.index).astype(float),
        "runtime_engine": runtime_engine,
        "semantic_stats": {
            "atr_stop": atr_stop_stats,
            "bias_v_take_profit": bias_v_tp_stats,
            "r_take_profit": r_take_profit_stats,
            "vol_risk_adjust": vol_risk_stats,
            "monthly_risk_budget_gate": monthly_gate_stats,
        },
    }


def compute_trend_backtest_bt(db: Session, inp: Any) -> dict[str, Any]:
    code = str(inp.code or "").strip()
    if not code:
        raise ValueError("code is empty")
    _validate_bt_single_inputs(inp)
    strat = str(inp.strategy or "ma_filter").strip().lower()
    ep = str(getattr(inp, "exec_price", "open") or "open").strip().lower()

    single = _run_single_backtesting(
        db, inp, code=code, random_seed=getattr(inp, "random_seed", 42)
    )
    nav = single["nav"]
    bh_nav = single["buy_hold_nav"]
    excess_nav = single["excess_nav"]
    strat_ret = single["strat_ret"]
    bench_ret = single["bench_ret"]
    excess_ret = single["excess_ret"]
    active_ret = (
        (
            strat_ret.reindex(nav.index).astype(float)
            - bench_ret.reindex(nav.index).astype(float)
        )
        .fillna(0.0)
        .astype(float)
    )

    ps = str(getattr(inp, "position_sizing", "equal") or "equal").strip().lower()
    if ps == "risk_budget":
        weight_s = _risk_budget_frozen_weight(
            single["desired_pos"],
            close=single["sig_close"],
            high=single["sig_high"],
            low=single["sig_low"],
            atr_window=int(getattr(inp, "risk_budget_atr_window", 20) or 20),
            risk_budget_pct=float(getattr(inp, "risk_budget_pct", 0.01) or 0.01),
        )
    else:
        weight_s = single["desired_pos"].astype(float)
    w_eff = single["desired_pos"].shift(1).fillna(0.0).astype(float).clip(lower=0.0)
    sig_open = (
        single.get("sig_open", single["sig_close"])
        .astype(float)
        .reindex(nav.index)
        .ffill()
    )
    sig_close = single["sig_close"].astype(float).reindex(nav.index).ffill()
    sig_high = single["sig_high"].astype(float).reindex(nav.index).ffill()
    sig_low = single["sig_low"].astype(float).reindex(nav.index).ffill()
    ret_exec_s = (
        single.get("ret_exec", strat_ret).astype(float).reindex(nav.index).fillna(0.0)
    )
    px_exec_s = (
        single.get("px_exec_slip", sig_close).astype(float).reindex(nav.index).ffill()
    )
    turnover_one_way = (w_eff - w_eff.shift(1).fillna(0.0)).abs() / 2.0
    cost_s = turnover_one_way * (float(getattr(inp, "cost_bps", 0.0) or 0.0) / 10000.0)
    slip_s = slippage_return_from_turnover(
        turnover_one_way.astype(float),
        slippage_spread=float(getattr(inp, "slippage_rate", 0.0) or 0.0),
        exec_price=px_exec_s.astype(float),
    ).astype(float)
    ret_overnight = (
        (sig_open / sig_close.shift(1) - 1.0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )
    ret_intraday = (
        (sig_close / sig_open - 1.0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )
    sem_dbg = single.get("semantic_stats") or {}
    atr_over = pd.Series(0.0, index=nav.index, dtype=float)
    bv_over = pd.Series(0.0, index=nav.index, dtype=float)
    rtp_over = pd.Series(0.0, index=nav.index, dtype=float)
    if str(single.get("runtime_engine") or "") != "backtesting":
        w_tmp = w_eff.copy()
        w_tmp, atr_over = _apply_intraday_stop_execution_single(
            weights=w_tmp,
            atr_stop_stats=dict((sem_dbg.get("atr_stop") or {})),
            exec_price=str(getattr(inp, "exec_price", "open") or "open"),
            open_sig=sig_open,
            close_sig=sig_close,
        )
        w_tmp, bv_over = _apply_intraday_stop_execution_single(
            weights=w_tmp,
            atr_stop_stats=dict((sem_dbg.get("bias_v_take_profit") or {})),
            exec_price=str(getattr(inp, "exec_price", "open") or "open"),
            open_sig=sig_open,
            close_sig=sig_close,
        )
        w_tmp, rtp_over = _apply_intraday_stop_execution_single(
            weights=w_tmp,
            atr_stop_stats=dict((sem_dbg.get("r_take_profit") or {})),
            exec_price=str(getattr(inp, "exec_price", "open") or "open"),
            open_sig=sig_open,
            close_sig=sig_close,
        )
        w_eff = w_tmp.astype(float)
    return_decomposition = None
    quick_mode = bool(getattr(inp, "quick_mode", False))
    if not quick_mode:
        decomp_overnight = (w_eff * ret_overnight).astype(float)
        decomp_intraday = (w_eff * ret_intraday).astype(float)
        decomp_interaction = (w_eff * ret_overnight * ret_intraday).astype(float)
        decomp_risk = (atr_over + bv_over + rtp_over).astype(float)
        decomp_cost = (cost_s + slip_s).astype(float)
        decomp_gross = (
            decomp_overnight + decomp_intraday + decomp_interaction + decomp_risk
        ).astype(float)
        decomp_net = (decomp_gross - decomp_cost).astype(float)
        return_decomposition = {
            "dates": [d.strftime("%Y-%m-%d") for d in nav.index],
            "series": {
                "overnight": decomp_overnight.tolist(),
                "intraday": decomp_intraday.tolist(),
                "interaction": decomp_interaction.tolist(),
                "atr_stop_override": atr_over.tolist(),
                "bias_v_take_profit_override": bv_over.tolist(),
                "r_take_profit_override": rtp_over.tolist(),
                "risk_exit_override": decomp_risk.tolist(),
                "cost": decomp_cost.tolist(),
                "gross": decomp_gross.tolist(),
                "net": decomp_net.tolist(),
            },
            "summary": {
                "ann_overnight": float(decomp_overnight.iloc[1:].mean() * 252.0)
                if len(decomp_overnight) > 1
                else 0.0,
                "ann_intraday": float(decomp_intraday.iloc[1:].mean() * 252.0)
                if len(decomp_intraday) > 1
                else 0.0,
                "ann_interaction": float(decomp_interaction.iloc[1:].mean() * 252.0)
                if len(decomp_interaction) > 1
                else 0.0,
                "ann_atr_stop_override": float(atr_over.iloc[1:].mean() * 252.0)
                if len(atr_over) > 1
                else 0.0,
                "ann_bias_v_take_profit_override": float(
                    bv_over.iloc[1:].mean() * 252.0
                )
                if len(bv_over) > 1
                else 0.0,
                "ann_r_take_profit_override": float(rtp_over.iloc[1:].mean() * 252.0)
                if len(rtp_over) > 1
                else 0.0,
                "ann_risk_exit_override": float(decomp_risk.iloc[1:].mean() * 252.0)
                if len(decomp_risk) > 1
                else 0.0,
                "ann_cost": float(decomp_cost.iloc[1:].mean() * 252.0)
                if len(decomp_cost) > 1
                else 0.0,
                "ann_gross": float(decomp_gross.iloc[1:].mean() * 252.0)
                if len(decomp_gross) > 1
                else 0.0,
                "ann_net": float(decomp_net.iloc[1:].mean() * 252.0)
                if len(decomp_net) > 1
                else 0.0,
            },
        }
    event_study = (
        None
        if quick_mode
        else compute_event_study(
            dates=nav.index,
            daily_returns=strat_ret.reindex(nav.index).astype(float),
            entry_dates=entry_dates_from_exposure(
                w_eff.reindex(nav.index).astype(float)
            ),
        )
    )
    market_regime = build_market_regime_report(
        close=sig_close.to_frame(code).astype(float),
        high=sig_high.to_frame(code).astype(float),
        low=sig_low.to_frame(code).astype(float),
        weights=w_eff.to_frame(code).astype(float),
        asset_returns=ret_exec_s.to_frame(code).astype(float),
        strategy_returns=strat_ret.reindex(nav.index).astype(float),
        ann_factor=252,
    )
    attribution = _compute_return_risk_contributions(
        asset_ret=ret_exec_s.to_frame(code)
        .reindex(nav.index)
        .astype(float)
        .fillna(0.0),
        weights=w_eff.to_frame(code).reindex(nav.index).astype(float).fillna(0.0),
        total_return=float(nav.iloc[-1] - 1.0) if len(nav) else 0.0,
    )
    latest_entry_exec_px = _latest_entry_exec_price_with_slippage(
        effective_weight=w_eff.reindex(nav.index).astype(float),
        exec_price_series=px_exec_s.reindex(nav.index).ffill().astype(float),
        slippage_spread=float(getattr(inp, "slippage_rate", 0.0) or 0.0),
    )
    trade_one = _trade_returns_from_weight_series(
        w_eff.reindex(nav.index).astype(float),
        ret_exec_s.reindex(nav.index).astype(float),
        cost_bps=float(getattr(inp, "cost_bps", 0.0) or 0.0),
        slippage_rate=float(getattr(inp, "slippage_rate", 0.0) or 0.0),
        exec_price=px_exec_s.reindex(nav.index).ffill().astype(float),
        dates=nav.index,
    )
    atr_risk = _atr_from_hlc(
        sig_high.reindex(nav.index)
        .astype(float)
        .fillna(sig_close.reindex(nav.index).astype(float)),
        sig_low.reindex(nav.index)
        .astype(float)
        .fillna(sig_close.reindex(nav.index).astype(float)),
        sig_close.reindex(nav.index).astype(float),
        window=int(getattr(inp, "atr_stop_window", 14) or 14),
    ).reindex(nav.index)
    trade_r_pack = enrich_trades_with_r_metrics(
        trade_one.get("trades", []),
        nav=nav.astype(float),
        weights=w_eff.reindex(nav.index).astype(float),
        exec_price=px_exec_s.reindex(nav.index).ffill().astype(float),
        atr=atr_risk.astype(float),
        atr_mult=float(getattr(inp, "atr_stop_n", 2.0) or 2.0),
        risk_budget_pct=(
            float(getattr(inp, "risk_budget_pct", 0.01) or 0.01)
            if ps == "risk_budget"
            else None
        ),
        cost_bps=float(getattr(inp, "cost_bps", 0.0) or 0.0),
        slippage_rate=float(getattr(inp, "slippage_rate", 0.0) or 0.0),
        default_code=str(code),
        ulcer_index=float(_ulcer_index(nav, in_percent=True)) if len(nav) else None,
        annual_trade_count=(
            float(len(trade_one.get("returns", []))) / max(1.0, float(len(nav))) * 252.0
        )
        if len(nav)
        else None,
        backtest_years=(float(len(nav)) / 252.0) if len(nav) else None,
        score_sqn_weight=0.60,
        score_ulcer_weight=0.40,
    )
    trades_with_r = list(trade_r_pack.get("trades") or [])
    r_stats_out = dict(trade_r_pack.get("statistics") or {})
    r_stats_out.pop("trade_system_score", None)
    if not quick_mode:
        mom_for_entry = _tsmom_rocp(
            sig_close.reindex(nav.index).astype(float),
            int(getattr(inp, "mom_lookback", 252) or 252),
        ).astype(float)
        er_for_entry = _efficiency_ratio(
            sig_close.reindex(nav.index).astype(float),
            window=int(getattr(inp, "er_window", 10) or 10),
        ).astype(float)
        atr_fast_for_entry = _atr_from_hlc(
            sig_high.reindex(nav.index)
            .astype(float)
            .fillna(sig_close.reindex(nav.index).astype(float)),
            sig_low.reindex(nav.index)
            .astype(float)
            .fillna(sig_close.reindex(nav.index).astype(float)),
            sig_close.reindex(nav.index).astype(float),
            window=int(getattr(inp, "vol_ratio_fast_atr_window", 5) or 5),
        ).astype(float)
        atr_slow_for_entry = _atr_from_hlc(
            sig_high.reindex(nav.index)
            .astype(float)
            .fillna(sig_close.reindex(nav.index).astype(float)),
            sig_low.reindex(nav.index)
            .astype(float)
            .fillna(sig_close.reindex(nav.index).astype(float)),
            sig_close.reindex(nav.index).astype(float),
            window=int(getattr(inp, "vol_ratio_slow_atr_window", 50) or 50),
        ).astype(float)
        vol_ratio_for_entry = (
            atr_fast_for_entry / atr_slow_for_entry.replace(0.0, np.nan)
        ).astype(float)
        impulse_state = _compute_impulse_state(
            sig_close.reindex(nav.index).astype(float),
            ema_window=13,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
        )
        condition_bins_by_code_single = {
            str(code): {
                "momentum": _bucketize_momentum_series(
                    mom_for_entry.reindex(nav.index)
                ),
                "er": _bucketize_er_series(er_for_entry.reindex(nav.index)),
                "vol_ratio": _bucketize_vol_ratio_series(
                    vol_ratio_for_entry.reindex(nav.index)
                ),
                "impulse": _bucketize_impulse_series(
                    (
                        impulse_state
                        if impulse_state is not None
                        else pd.Series(index=nav.index, dtype=object)
                    ).reindex(nav.index)
                ),
            }
        }
        trades_with_r = _attach_entry_condition_bins_to_trades(
            trades_with_r,
            condition_bins_by_code=condition_bins_by_code_single,
            dates=nav.index,
            default_code=str(code),
        )
    mfe_r_distribution = build_trade_mfe_r_distribution(
        trade_one.get("trades", []),
        close=sig_close.reindex(nav.index).astype(float).ffill(),
        high=sig_high.reindex(nav.index).astype(float).ffill(),
        atr=atr_risk.astype(float).reindex(nav.index),
        atr_mult=float(getattr(inp, "atr_stop_n", 2.0) or 2.0),
        default_code=str(code),
    )
    sig_dbg = single.get("signal_debug") or {}
    sem_dbg = single.get("semantic_stats") or {}
    er_stats = sig_dbg.get("er_filter") or {}
    imp_stats = sig_dbg.get("impulse_filter") or {}
    er_exit_stats = sig_dbg.get("er_exit_filter") or {}
    atr_stats = sem_dbg.get("atr_stop") or {}
    rtp_stats = sem_dbg.get("r_take_profit") or {}
    bv_stats = sem_dbg.get("bias_v_take_profit") or {}
    vol_stats = sem_dbg.get("vol_risk_adjust") or {}
    month_stats = sem_dbg.get("monthly_risk_budget_gate") or {}
    impulse_attempted = int(imp_stats.get("attempted_entry_count", 0))
    impulse_blocked = int(imp_stats.get("blocked_entry_count", 0))
    monthly_attempted = int(month_stats.get("attempted_entry_count", 0))
    monthly_blocked = int(month_stats.get("blocked_entry_count", 0))
    overall_stats = {
        **_trade_stats_from_returns(trade_one.get("returns", [])),
        "n": int(single["trade_count"]),
        "atr_stop_trigger_count": int(atr_stats.get("trigger_count", 0)),
        "r_take_profit_trigger_count": int(rtp_stats.get("trigger_count", 0)),
        "bias_v_take_profit_trigger_count": int(bv_stats.get("trigger_count", 0)),
        "r_take_profit_tier_trigger_counts": dict(
            rtp_stats.get("tier_trigger_counts") or {}
        ),
        "er_filter_blocked_entry_count": int(er_stats.get("blocked_entry_count", 0)),
        "er_filter_attempted_entry_count": int(
            er_stats.get("attempted_entry_count", 0)
        ),
        "er_filter_allowed_entry_count": int(er_stats.get("allowed_entry_count", 0)),
        "impulse_filter_blocked_entry_count": impulse_blocked,
        "impulse_filter_attempted_entry_count": impulse_attempted,
        "impulse_filter_allowed_entry_count": int(
            imp_stats.get("allowed_entry_count", 0)
        ),
        "impulse_filter_blocked_entry_rate": (
            float(impulse_blocked / impulse_attempted) if impulse_attempted > 0 else 0.0
        ),
        "impulse_filter_blocked_entry_count_bull": int(
            imp_stats.get("blocked_entry_count_bull", 0)
        ),
        "impulse_filter_blocked_entry_count_bear": int(
            imp_stats.get("blocked_entry_count_bear", 0)
        ),
        "impulse_filter_blocked_entry_count_neutral": int(
            imp_stats.get("blocked_entry_count_neutral", 0)
        ),
        "er_exit_filter_trigger_count": int(er_exit_stats.get("trigger_count", 0)),
        "vol_risk_adjust_total_count": int(
            vol_stats.get("vol_risk_adjust_total_count", 0)
        ),
        "vol_risk_adjust_reduce_on_expand_count": int(
            vol_stats.get("vol_risk_adjust_reduce_on_expand_count", 0)
        ),
        "vol_risk_adjust_increase_on_contract_count": int(
            vol_stats.get("vol_risk_adjust_increase_on_contract_count", 0)
        ),
        "vol_risk_adjust_recover_from_expand_count": int(
            vol_stats.get("vol_risk_adjust_recover_from_expand_count", 0)
        ),
        "vol_risk_adjust_recover_from_contract_count": int(
            vol_stats.get("vol_risk_adjust_recover_from_contract_count", 0)
        ),
        "vol_risk_entry_state_reduce_on_expand_count": int(
            vol_stats.get("vol_risk_entry_state_reduce_on_expand_count", 0)
        ),
        "vol_risk_entry_state_increase_on_contract_count": int(
            vol_stats.get("vol_risk_entry_state_increase_on_contract_count", 0)
        ),
        "monthly_risk_budget_attempted_entry_count": monthly_attempted,
        "monthly_risk_budget_blocked_entry_count": monthly_blocked,
        "monthly_risk_budget_blocked_entry_rate": (
            float(monthly_blocked / monthly_attempted) if monthly_attempted > 0 else 0.0
        ),
    }
    by_code_stats = {
        str(code): {
            **_trade_stats_from_returns(trade_one.get("returns", [])),
            **dict(overall_stats),
        }
    }
    trade_stats_trades = [] if quick_mode else list(trades_with_r)
    trade_stats_trades_by_code = {
        str(code): ([] if quick_mode else list(trades_with_r))
    }
    sample_days = int(len(strat_ret))
    complete_trade_count = int(len(trade_one.get("returns", [])))
    avg_daily_turnover = (
        float(turnover_one_way.mean()) if len(turnover_one_way) else 0.0
    )
    avg_annual_turnover = float(avg_daily_turnover * 252.0)
    avg_daily_trade_count = (
        float(complete_trade_count / sample_days) if sample_days > 0 else 0.0
    )
    avg_annual_trade_count = float(avg_daily_trade_count * 252.0)

    out = {
        "meta": {
            "type": "trend_backtest",
            "engine": "bt",
            "runtime_engine": str(single.get("runtime_engine") or "unknown"),
            "code": code,
            "start": inp.start.strftime("%Y%m%d"),
            "end": inp.end.strftime("%Y%m%d"),
            "strategy": strat,
            "strategy_execution_description": TREND_STRATEGY_EXECUTION_DESCRIPTIONS.get(
                strat, ""
            ),
            "price_basis": {
                "signal": "qfq close",
                "strategy_nav": "none close preferred; hfq return fallback on corporate-action days",
                "benchmark_nav": {
                    "close": "HFQ close-to-close daily returns (BUY_HOLD line; excess vs strategy uses this series)",
                    "open": "same-day open→close (none; hfq on corporate-action days); BUY_HOLD aligned to open execution",
                    "oc2": "50% same-day open→close + 50% HFQ close-to-close next day; BUY_HOLD aligned to OC2 execution",
                }.get(ep, "unknown exec_price"),
            },
            "params": _build_meta_params(inp),
            "limitations": [],
        },
        "nav": {
            "dates": single["dates"],
            "series": {
                "STRAT": [float(x) for x in nav.values],
                "BUY_HOLD": [float(x) for x in bh_nav.values],
                "EXCESS": [float(x) for x in excess_nav.values],
            },
        },
        "signals": {
            "dates": single["dates"],
            "base_position": [float(x) for x in single["base_pos"].values],
            "position": [float(x) for x in single["signal_pos"].values],
            "position_effective": [float(x) for x in w_eff.values],
        },
        "weights": {
            "dates": single["dates"],
            "series": {code: [float(x) for x in weight_s.values]},
        },
        "metrics": {
            "strategy": {
                **_metrics_from_ret(strat_ret, float(inp.risk_free_rate)),
                "avg_daily_turnover": float(avg_daily_turnover),
                "avg_annual_turnover": float(avg_annual_turnover),
                "avg_annual_turnover_rate": float(avg_annual_turnover),
                "avg_daily_trade_count": float(avg_daily_trade_count),
                "avg_annual_trade_count": float(avg_annual_trade_count),
                "r_take_profit_tier_trigger_counts": dict(
                    rtp_stats.get("tier_trigger_counts") or {}
                ),
                "impulse_filter_blocked_entry_count": int(
                    imp_stats.get("blocked_entry_count", 0)
                ),
                "impulse_filter_blocked_entry_count_bull": int(
                    imp_stats.get("blocked_entry_count_bull", 0)
                ),
                "impulse_filter_blocked_entry_count_bear": int(
                    imp_stats.get("blocked_entry_count_bear", 0)
                ),
                "impulse_filter_blocked_entry_count_neutral": int(
                    imp_stats.get("blocked_entry_count_neutral", 0)
                ),
                "monthly_risk_budget_blocked_entry_count": int(
                    month_stats.get("blocked_entry_count", 0)
                ),
            },
            "benchmark": _metrics_from_ret(bench_ret, float(inp.risk_free_rate)),
            "excess": {
                **_metrics_from_ret(excess_ret, float(inp.risk_free_rate)),
                "information_ratio": float(_sharpe(active_ret, rf=0.0)),
            },
        },
        "period_returns": {
            "weekly": _period_returns(nav, "W-FRI"),
            "monthly": _period_returns(nav, "ME"),
            "quarterly": _period_returns(nav, "QE"),
            "yearly": _period_returns(nav, "YE"),
        },
        "rolling": _rolling_pack(nav),
        "attribution": attribution,
        "trade_statistics": {
            "all": {"n": int(single["trade_count"])},
            "overall": overall_stats,
            "by_code": by_code_stats,
            "trades": trade_stats_trades,
            "trades_by_code": trade_stats_trades_by_code,
            "mfe_r_distribution": mfe_r_distribution,
        },
        "r_statistics": r_stats_out,
        "trades": ([] if quick_mode else trades_with_r),
        "risk_controls": {
            "vol_regime_risk_mgmt": {
                "enabled": bool(getattr(inp, "vol_regime_risk_mgmt_enabled", False)),
                "fast_atr_window": int(
                    getattr(inp, "vol_ratio_fast_atr_window", 5) or 5
                ),
                "slow_atr_window": int(
                    getattr(inp, "vol_ratio_slow_atr_window", 50) or 50
                ),
                "expand_threshold": float(
                    getattr(inp, "vol_ratio_expand_threshold", 1.45) or 1.45
                ),
                "contract_threshold": float(
                    getattr(inp, "vol_ratio_contract_threshold", 0.65) or 0.65
                ),
                "normal_threshold": float(
                    getattr(inp, "vol_ratio_normal_threshold", 1.05) or 1.05
                ),
                "adjust_total_count": int(
                    vol_stats.get("vol_risk_adjust_total_count", 0)
                ),
                "adjust_reduce_on_expand_count": int(
                    vol_stats.get("vol_risk_adjust_reduce_on_expand_count", 0)
                ),
                "adjust_increase_on_contract_count": int(
                    vol_stats.get("vol_risk_adjust_increase_on_contract_count", 0)
                ),
                "adjust_recover_from_expand_count": int(
                    vol_stats.get("vol_risk_adjust_recover_from_expand_count", 0)
                ),
                "adjust_recover_from_contract_count": int(
                    vol_stats.get("vol_risk_adjust_recover_from_contract_count", 0)
                ),
                "entry_state_reduce_on_expand_count": int(
                    vol_stats.get("vol_risk_entry_state_reduce_on_expand_count", 0)
                ),
                "entry_state_increase_on_contract_count": int(
                    vol_stats.get("vol_risk_entry_state_increase_on_contract_count", 0)
                ),
            },
            "er_exit_filter": {
                "enabled": bool(getattr(inp, "er_exit_filter", False)),
                "window": int(getattr(inp, "er_exit_window", 10) or 10),
                "threshold": float(getattr(inp, "er_exit_threshold", 0.88) or 0.88),
                "trigger_count": int(er_exit_stats.get("trigger_count", 0)),
                "trigger_dates": list(er_exit_stats.get("trigger_dates") or []),
                "trace_last_rows": list(er_exit_stats.get("trace_last_rows") or []),
            },
            "atr_stop": dict(sem_dbg.get("atr_stop") or {}),
            "bias_v_take_profit": dict(sem_dbg.get("bias_v_take_profit") or {}),
            "r_take_profit": dict(sem_dbg.get("r_take_profit") or {}),
            "monthly_risk_budget_gate": {
                **dict(sem_dbg.get("monthly_risk_budget_gate") or {}),
                "enabled": bool(getattr(inp, "monthly_risk_budget_enabled", False)),
                "budget_pct": float(
                    getattr(inp, "monthly_risk_budget_pct", 0.06) or 0.06
                ),
                "include_new_trade_risk": bool(
                    getattr(inp, "monthly_risk_budget_include_new_trade_risk", False)
                ),
            },
            "monthly_risk_budget": {
                **dict(sem_dbg.get("monthly_risk_budget_gate") or {}),
                "enabled": bool(getattr(inp, "monthly_risk_budget_enabled", False)),
                "budget_pct": float(
                    getattr(inp, "monthly_risk_budget_pct", 0.06) or 0.06
                ),
                "include_new_trade_risk": bool(
                    getattr(inp, "monthly_risk_budget_include_new_trade_risk", False)
                ),
            },
        },
        "return_decomposition": return_decomposition,
        "event_study": event_study,
        "market_regime": market_regime,
        "next_plan": {
            "decision_date": (str(nav.index[-1].date()) if len(nav.index) else None),
            "current_effective_weight": (float(w_eff.iloc[-1]) if len(w_eff) else 0.0),
            "target_weight": (
                float(single["desired_pos"].reindex(nav.index).iloc[-1])
                if len(nav.index)
                else 0.0
            ),
            "entry_exec_price_with_slippage_by_asset": (
                {str(code): float(latest_entry_exec_px)}
                if latest_entry_exec_px is not None
                else {}
            ),
            "trace": {
                "atr_stop_mode": str(getattr(inp, "atr_stop_mode", "none") or "none"),
                "atr_stop_atr_basis": str(
                    getattr(inp, "atr_stop_atr_basis", "latest") or "latest"
                ),
                "atr_stop_reentry_mode": str(
                    getattr(inp, "atr_stop_reentry_mode", "reenter") or "reenter"
                ),
                "base_signal_prev": (
                    float(single["base_pos"].iloc[-2])
                    if len(single["base_pos"]) >= 2
                    else 0.0
                ),
                "base_signal_today": (
                    float(single["base_pos"].iloc[-1])
                    if len(single["base_pos"])
                    else 0.0
                ),
                "base_entry_event_today": (
                    bool(
                        (single["base_pos"].iloc[-1] > 0.0)
                        and (single["base_pos"].iloc[-2] <= 0.0)
                    )
                    if len(single["base_pos"]) >= 2
                    else bool(single["base_pos"].iloc[-1] > 0.0)
                    if len(single["base_pos"])
                    else False
                ),
                "strategy": str(strat),
                "atr_stop": dict(atr_stats),
                "bias_v_take_profit": dict(bv_stats),
                "r_take_profit": dict(rtp_stats),
                "er_exit_filter": {
                    "enabled": bool(getattr(inp, "er_exit_filter", False)),
                    "window": int(getattr(inp, "er_exit_window", 10) or 10),
                    "threshold": float(getattr(inp, "er_exit_threshold", 0.88) or 0.88),
                    "trigger_count": int(er_exit_stats.get("trigger_count", 0)),
                    "trigger_dates": list(er_exit_stats.get("trigger_dates") or []),
                    "trace_last_rows": list(er_exit_stats.get("trace_last_rows") or []),
                },
            },
        },
        "corporate_actions": (
            [
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "none_return": float(single["ret_exec_none"].loc[d]),
                    "hfq_return": float(single["ret_exec_hfq"].loc[d]),
                    "corp_factor": (
                        float(single["corp_factor"].loc[d])
                        if np.isfinite(float(single["corp_factor"].loc[d]))
                        else None
                    ),
                }
                for d in nav.index
                if bool(single["ca_mask"].loc[d])
            ][:200]
        ),
        "signal_debug": single["signal_debug"],
    }
    if not quick_mode:
        out["trade_statistics"]["entry_condition_stats"] = {
            "scope": "closed_trades_only",
            "signal_day_basis": "signal_day_before_entry_execution",
            "quasi_causal_method": "uplift + two_proportion_z / welch_t_normal_approx + BH",
            "strong_causal_method": "uplift + stratified_permutation + BH",
            "overall": _build_entry_condition_stats(
                trades_with_r, by_code=False, n_perm=300, seed=20260410
            ),
            "by_code": {
                str(code): _build_entry_condition_stats(
                    trades_with_r, by_code=True, n_perm=200, seed=20260410
                )
            },
        }
    return out


def compute_trend_portfolio_backtest_bt(db: Session, inp: Any) -> dict[str, Any]:
    codes = list(
        dict.fromkeys([str(c).strip() for c in (inp.codes or []) if str(c).strip()])
    )
    if not codes:
        raise ValueError("codes is empty")
    single_validation = SimpleNamespace(
        code="__BT_VALIDATION__",
        start=inp.start,
        end=inp.end,
        strategy=inp.strategy,
        ma_type=inp.ma_type,
        kama_fast_window=inp.kama_fast_window,
        kama_slow_window=inp.kama_slow_window,
        position_sizing=inp.position_sizing,
        risk_budget_pct=inp.risk_budget_pct,
        vol_regime_risk_mgmt_enabled=inp.vol_regime_risk_mgmt_enabled,
        vol_ratio_expand_threshold=inp.vol_ratio_expand_threshold,
        vol_ratio_contract_threshold=inp.vol_ratio_contract_threshold,
        vol_ratio_normal_threshold=inp.vol_ratio_normal_threshold,
    )
    _validate_bt_single_inputs(single_validation)
    strat = str(inp.strategy or "ma_filter").strip().lower()
    need_hist = (
        max(
            int(getattr(inp, "sma_window", 20) or 20),
            int(getattr(inp, "slow_window", 20) or 20),
            int(getattr(inp, "donchian_entry", 20) or 20),
            int(getattr(inp, "mom_lookback", 252) or 252),
            int(getattr(inp, "macd_slow", 26) or 26),
            int(getattr(inp, "macd_v_atr_window", 14) or 14),
            20,
        )
        + 60
    )
    ext_start = inp.start - dt.timedelta(days=int(need_hist) * 2)

    nav_map: dict[str, pd.Series] = {}
    weight_map: dict[str, pd.Series] = {}
    ret_exec_map: dict[str, pd.Series] = {}
    ret_hfq_map: dict[str, pd.Series] = {}
    px_exec_slip_map: dict[str, pd.Series] = {}
    sig_open_map: dict[str, pd.Series] = {}
    sig_close_map: dict[str, pd.Series] = {}
    score_map: dict[str, pd.Series] = {}
    trades: list[dict[str, Any]] = []
    failures: list[str] = []
    signal_debug_by_code: dict[str, dict[str, Any]] = {}
    semantic_debug_by_code: dict[str, dict[str, Any]] = {}
    price_sig_by_code: dict[str, pd.DataFrame] = {}
    corporate_actions_rows: list[dict[str, Any]] = []

    for c in codes:
        seed_base = getattr(inp, "random_seed", 42)
        code_seed = (
            None
            if seed_base is None
            else (int(seed_base) + _stable_code_seed(c)) % 2_147_483_647
        )
        single_inp = SimpleNamespace(
            code=c,
            start=ext_start,
            end=inp.end,
            risk_free_rate=inp.risk_free_rate,
            cost_bps=inp.cost_bps,
            slippage_rate=inp.slippage_rate,
            exec_price=inp.exec_price,
            strategy=inp.strategy,
            sma_window=inp.sma_window,
            fast_window=inp.fast_window,
            slow_window=inp.slow_window,
            ma_type=inp.ma_type,
            kama_er_window=inp.kama_er_window,
            kama_fast_window=inp.kama_fast_window,
            kama_slow_window=inp.kama_slow_window,
            kama_std_window=inp.kama_std_window,
            kama_std_coef=inp.kama_std_coef,
            donchian_entry=inp.donchian_entry,
            donchian_exit=inp.donchian_exit,
            mom_lookback=inp.mom_lookback,
            tsmom_entry_threshold=inp.tsmom_entry_threshold,
            tsmom_exit_threshold=inp.tsmom_exit_threshold,
            bias_ma_window=inp.bias_ma_window,
            bias_entry=inp.bias_entry,
            bias_hot=inp.bias_hot,
            bias_cold=inp.bias_cold,
            bias_pos_mode=inp.bias_pos_mode,
            macd_fast=inp.macd_fast,
            macd_slow=inp.macd_slow,
            macd_signal=inp.macd_signal,
            macd_v_atr_window=inp.macd_v_atr_window,
            macd_v_scale=inp.macd_v_scale,
            random_hold_days=inp.random_hold_days,
            random_seed=code_seed,
            er_filter=inp.er_filter,
            er_window=inp.er_window,
            er_threshold=inp.er_threshold,
            impulse_entry_filter=inp.impulse_entry_filter,
            impulse_allow_bull=inp.impulse_allow_bull,
            impulse_allow_bear=inp.impulse_allow_bear,
            impulse_allow_neutral=inp.impulse_allow_neutral,
            er_exit_filter=inp.er_exit_filter,
            er_exit_window=inp.er_exit_window,
            er_exit_threshold=inp.er_exit_threshold,
            atr_stop_mode=inp.atr_stop_mode,
            atr_stop_atr_basis=inp.atr_stop_atr_basis,
            atr_stop_reentry_mode=inp.atr_stop_reentry_mode,
            atr_stop_window=inp.atr_stop_window,
            atr_stop_n=inp.atr_stop_n,
            atr_stop_m=inp.atr_stop_m,
            r_take_profit_enabled=inp.r_take_profit_enabled,
            r_take_profit_reentry_mode=inp.r_take_profit_reentry_mode,
            r_take_profit_tiers=inp.r_take_profit_tiers,
            bias_v_take_profit_enabled=inp.bias_v_take_profit_enabled,
            bias_v_take_profit_reentry_mode=inp.bias_v_take_profit_reentry_mode,
            bias_v_ma_window=inp.bias_v_ma_window,
            bias_v_atr_window=inp.bias_v_atr_window,
            bias_v_take_profit_threshold=inp.bias_v_take_profit_threshold,
            monthly_risk_budget_enabled=inp.monthly_risk_budget_enabled,
            monthly_risk_budget_pct=inp.monthly_risk_budget_pct,
            # Portfolio semantics: monthly risk budget gate is applied once at
            # portfolio decision level, not inside each per-asset signal leg.
            monthly_risk_budget_include_new_trade_risk=False,
            # Keep per-asset signal generation in equal mode; portfolio-level sizing
            # remains handled by the portfolio engine path.
            position_sizing="equal",
            fixed_pos_ratio=inp.fixed_pos_ratio,
            fixed_overcap_policy=inp.fixed_overcap_policy,
            fixed_max_holdings=inp.fixed_max_holdings,
            risk_budget_atr_window=inp.risk_budget_atr_window,
            risk_budget_pct=inp.risk_budget_pct,
            vol_regime_risk_mgmt_enabled=inp.vol_regime_risk_mgmt_enabled,
            vol_ratio_fast_atr_window=inp.vol_ratio_fast_atr_window,
            vol_ratio_slow_atr_window=inp.vol_ratio_slow_atr_window,
            vol_ratio_expand_threshold=inp.vol_ratio_expand_threshold,
            vol_ratio_contract_threshold=inp.vol_ratio_contract_threshold,
            vol_ratio_normal_threshold=inp.vol_ratio_normal_threshold,
            group_enforce=inp.group_enforce,
            group_pick_policy=inp.group_pick_policy,
            group_max_holdings=inp.group_max_holdings,
            asset_groups=inp.asset_groups,
            quick_mode=inp.quick_mode,
        )
        # Keep per-asset signal generation free of portfolio-level monthly gate.
        single_inp = _clone_like_input(single_inp, monthly_risk_budget_enabled=False)
        try:
            one = _run_single_backtesting(db, single_inp, code=c, random_seed=code_seed)
        except ValueError as exc:
            failures.append(f"{c}:{exc}")
            continue
        trim_ix = one["nav"].index[
            (one["nav"].index.date >= inp.start) & (one["nav"].index.date <= inp.end)
        ]
        if len(trim_ix) == 0:
            failures.append(f"{c}:no rows in requested date range")
            continue
        one = {
            **one,
            "dates": [d.strftime("%Y-%m-%d") for d in trim_ix],
            "nav": one["nav"].reindex(trim_ix).astype(float),
            "buy_hold_nav": one["buy_hold_nav"].reindex(trim_ix).astype(float),
            "excess_nav": one["excess_nav"].reindex(trim_ix).astype(float),
            "strat_ret": one["strat_ret"].reindex(trim_ix).fillna(0.0).astype(float),
            "bench_ret": one["bench_ret"].reindex(trim_ix).fillna(0.0).astype(float),
            "excess_ret": one["excess_ret"].reindex(trim_ix).fillna(0.0).astype(float),
            "desired_pos": one["desired_pos"]
            .reindex(trim_ix)
            .fillna(0.0)
            .astype(float),
            "base_pos": one["base_pos"].reindex(trim_ix).fillna(0.0).astype(float),
            "signal_pos": one["signal_pos"].reindex(trim_ix).fillna(0.0).astype(float),
            "sig_open": one["sig_open"].reindex(trim_ix).ffill().astype(float),
            "sig_close": one["sig_close"].reindex(trim_ix).ffill().astype(float),
            "sig_high": one["sig_high"].reindex(trim_ix).ffill().astype(float),
            "sig_low": one["sig_low"].reindex(trim_ix).ffill().astype(float),
            "ret_exec": one["ret_exec"].reindex(trim_ix).fillna(0.0).astype(float),
            "ret_exec_raw": one["ret_exec_raw"]
            .reindex(trim_ix)
            .fillna(0.0)
            .astype(float),
            "px_exec_slip": one["px_exec_slip"].reindex(trim_ix).ffill().astype(float),
            "ret_exec_none": one["ret_exec_none"]
            .reindex(trim_ix)
            .fillna(0.0)
            .astype(float),
            "ret_exec_hfq": one["ret_exec_hfq"]
            .reindex(trim_ix)
            .fillna(0.0)
            .astype(float),
            "exec_open_none": one["exec_open_none"].reindex(trim_ix).astype(float),
            "exec_close_none": one["exec_close_none"].reindex(trim_ix).astype(float),
            "exec_open_hfq": one["exec_open_hfq"].reindex(trim_ix).astype(float),
            "exec_close_hfq": one["exec_close_hfq"].reindex(trim_ix).astype(float),
            "corp_factor": one["corp_factor"].reindex(trim_ix).astype(float),
            "ca_mask": one["ca_mask"].reindex(trim_ix).fillna(False).astype(bool),
            "trades": [
                t
                for t in list(one.get("trades") or [])
                if str(t.get("entry_date") or "") >= str(inp.start)
            ],
        }
        nav_map[c] = one["nav"]
        weight_map[c] = one["desired_pos"].astype(float)
        ep_port = str(getattr(inp, "exec_price", "open") or "open").strip().lower()
        ps_port = (
            str(getattr(inp, "position_sizing", "equal") or "equal").strip().lower()
        )
        if ep_port == "open" and ps_port == "risk_budget":
            ret_exec_map[c] = one["ret_exec"].astype(float)
            ret_hfq_map[c] = one["ret_exec_hfq"].astype(float)
            px_exec_slip_map[c] = one["px_exec_slip"].astype(float)
            sig_open_map[c] = one["sig_open"].astype(float)
            sig_close_map[c] = one["sig_close"].astype(float)
            score_map[c] = one.get(
                "signal_score", pd.Series(np.nan, index=one["nav"].index)
            ).astype(float)
            trades.extend(one["trades"])
            runtime_engine = str(one.get("runtime_engine") or "unknown")
            signal_debug_by_code[c] = dict(one.get("signal_debug") or {})
            semantic_debug_by_code[c] = dict(one.get("semantic_stats") or {})
            price_sig_by_code[c] = pd.DataFrame(
                {
                    "close": one["sig_close"].astype(float),
                    "high": one["sig_high"].astype(float),
                    "low": one["sig_low"].astype(float),
                }
            )
            ca_mask_s = one.get("ca_mask", pd.Series(False, index=one["nav"].index))
            for d in one["nav"].index:
                if bool(ca_mask_s.loc[d]):
                    corporate_actions_rows.append(
                        {
                            "date": d.strftime("%Y-%m-%d")
                            if hasattr(d, "strftime")
                            else str(d),
                            "code": str(c),
                            "none_return": float(one["ret_exec_none"].loc[d]),
                            "hfq_return": float(one["ret_exec_hfq"].loc[d]),
                            "corp_factor": (
                                float(one["corp_factor"].loc[d])
                                if np.isfinite(float(one["corp_factor"].loc[d]))
                                else None
                            ),
                        }
                    )
            continue
        open_none = one["exec_open_none"].astype(float)
        close_none = one["exec_close_none"].astype(float)
        open_hfq = one["exec_open_hfq"].astype(float)
        close_hfq = one["exec_close_hfq"].astype(float)
        if ep_port == "open":
            ret_none_one = _forward_simple_return(open_none)
            ret_hfq_one = _forward_simple_return(open_hfq)
            px_none_one = open_none.astype(float)
            px_hfq_one = open_hfq.astype(float)
        elif ep_port == "close":
            ret_none_one = _forward_simple_return(close_none)
            ret_hfq_one = _forward_simple_return(close_hfq)
            px_none_one = close_none.astype(float)
            px_hfq_one = close_hfq.astype(float)
        else:
            ret_open_none_one = _forward_simple_return(open_none)
            ret_close_none_one = _forward_simple_return(close_none)
            ret_none_one = (0.5 * (ret_open_none_one + ret_close_none_one)).astype(
                float
            )
            ret_open_hfq_one = _forward_simple_return(open_hfq)
            ret_close_hfq_one = _forward_simple_return(close_hfq)
            ret_hfq_one = (0.5 * (ret_open_hfq_one + ret_close_hfq_one)).astype(float)
            px_none_one = (0.5 * (open_none + close_none)).astype(float)
            px_hfq_one = (0.5 * (open_hfq + close_hfq)).astype(float)
        gross_none_one = (1.0 + ret_none_one).astype(float).to_frame(c)
        gross_hfq_one = (1.0 + ret_hfq_one).astype(float).to_frame(c)
        _, ca_mask_one_df = corporate_action_mask(gross_none_one, gross_hfq_one)
        if isinstance(ca_mask_one_df, pd.DataFrame) and c in ca_mask_one_df.columns:
            ca_mask_one = (
                ca_mask_one_df[c].reindex(ret_none_one.index).fillna(False).astype(bool)
            )
        else:
            ca_mask_one = pd.Series(False, index=ret_none_one.index, dtype=bool)
        ret_exec_map[c] = ret_none_one.where(~ca_mask_one, other=ret_hfq_one).astype(
            float
        )
        ret_hfq_map[c] = ret_hfq_one.astype(float)
        px_exec_slip_map[c] = (
            px_none_one.where(~ca_mask_one, other=px_hfq_one)
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .astype(float)
        )
        sig_open_map[c] = one["sig_open"].astype(float)
        sig_close_map[c] = one["sig_close"].astype(float)
        score_map[c] = one.get(
            "signal_score", pd.Series(np.nan, index=one["nav"].index)
        ).astype(float)
        trades.extend(one["trades"])
        runtime_engine = str(one.get("runtime_engine") or "unknown")
        signal_debug_by_code[c] = dict(one.get("signal_debug") or {})
        semantic_debug_by_code[c] = dict(one.get("semantic_stats") or {})
        price_sig_by_code[c] = pd.DataFrame(
            {
                "close": one["sig_close"].astype(float),
                "high": one["sig_high"].astype(float),
                "low": one["sig_low"].astype(float),
            }
        )
        ca_mask_s = one.get("ca_mask", pd.Series(False, index=one["nav"].index))
        for d in one["nav"].index:
            if bool(ca_mask_s.loc[d]):
                corporate_actions_rows.append(
                    {
                        "date": d.strftime("%Y-%m-%d")
                        if hasattr(d, "strftime")
                        else str(d),
                        "code": str(c),
                        "none_return": float(one["ret_exec_none"].loc[d]),
                        "hfq_return": float(one["ret_exec_hfq"].loc[d]),
                        "corp_factor": (
                            float(one["corp_factor"].loc[d])
                            if np.isfinite(float(one["corp_factor"].loc[d]))
                            else None
                        ),
                    }
                )

    if not nav_map:
        raise ValueError("no valid symbol data for bt trend portfolio")

    dynamic_universe = bool(getattr(inp, "dynamic_universe", False))
    if not dynamic_universe:
        first_valid: list[pd.Timestamp] = []
        for c in codes:
            s = sig_close_map.get(c, pd.Series(dtype=float))
            fv = s.first_valid_index() if isinstance(s, pd.Series) else None
            if fv is None:
                raise ValueError(f"missing execution data (none) for: ['{c}']")
            first_valid.append(pd.Timestamp(fv))
        if not first_valid:
            raise ValueError("no valid first trading date for selected codes")
        common_start = max(first_valid)

        def _trim_series_map(m: dict[str, pd.Series]) -> dict[str, pd.Series]:
            out: dict[str, pd.Series] = {}
            for k, s in m.items():
                if isinstance(s, pd.Series):
                    out[str(k)] = s.loc[s.index >= common_start]
            return out

        nav_map = _trim_series_map(nav_map)
        weight_map = _trim_series_map(weight_map)
        ret_exec_map = _trim_series_map(ret_exec_map)
        ret_hfq_map = _trim_series_map(ret_hfq_map)
        px_exec_slip_map = _trim_series_map(px_exec_slip_map)
        sig_open_map = _trim_series_map(sig_open_map)
        sig_close_map = _trim_series_map(sig_close_map)
        score_map = _trim_series_map(score_map)
        price_sig_by_code = {
            str(k): v.loc[v.index >= common_start].copy()
            for k, v in price_sig_by_code.items()
            if isinstance(v, pd.DataFrame)
        }

    nav_df = pd.DataFrame(nav_map).sort_index()
    wdf = pd.DataFrame(weight_map).reindex(nav_df.index).fillna(0.0)
    score_df = pd.DataFrame(score_map).reindex(index=wdf.index, columns=wdf.columns)
    ret_hfq_df = (
        pd.DataFrame(ret_hfq_map)
        .reindex(index=wdf.index, columns=wdf.columns)
        .fillna(0.0)
        .astype(float)
    )
    group_enforce = bool(getattr(inp, "group_enforce", False))
    group_pick_policy = (
        str(getattr(inp, "group_pick_policy", "highest_sharpe") or "highest_sharpe")
        .strip()
        .lower()
    )
    group_max_holdings = int(getattr(inp, "group_max_holdings", 4) or 4)
    group_map = {
        str(k).strip(): str(v).strip()
        for k, v in ((getattr(inp, "asset_groups", None) or {}).items())
        if str(k).strip()
    }
    sharpe_like = pd.DataFrame(
        index=ret_hfq_df.index, columns=ret_hfq_df.columns, dtype=float
    )
    sharpe_win = max(20, int(getattr(inp, "vol_window", 20) or 20))
    sharpe_minp = max(10, sharpe_win // 2)
    for c in ret_hfq_df.columns:
        rs = ret_hfq_df[c].astype(float)
        mu = _rolling_sma(rs, window=sharpe_win, min_periods=sharpe_minp)
        sd = _rolling_std(
            rs, window=sharpe_win, min_periods=sharpe_minp, ddof=1
        ).replace(0.0, np.nan)
        sharpe_like[c] = (mu / sd) * np.sqrt(252.0)
    group_filter_meta_by_date: dict[pd.Timestamp, dict[str, Any]] = {}
    if group_enforce:
        prev_group_holdings: set[str] = set()
        for d in wdf.index:
            row = wdf.loc[d].astype(float)
            score_row = score_df.loc[d].astype(float).replace([np.inf, -np.inf], np.nan)
            scores = score_row.where(row > 1e-12, other=np.nan)
            active_raw = [
                str(c)
                for c in scores.dropna().sort_values(ascending=False).index.tolist()
            ]
            reduced, group_meta = _reduce_active_codes_by_group(
                active_codes=active_raw,
                score_row=scores,
                sharpe_row=(
                    sharpe_like.loc[d]
                    if d in sharpe_like.index
                    else pd.Series(dtype=float)
                ),
                group_enforce=group_enforce,
                asset_groups=group_map,
                group_pick_policy=group_pick_policy,
                group_max_holdings=group_max_holdings,
                current_holdings=prev_group_holdings,
            )
            group_filter_meta_by_date[pd.Timestamp(d)] = dict(group_meta or {})
            mask = pd.Series(0.0, index=wdf.columns, dtype=float)
            for c in reduced:
                if c in mask.index:
                    mask.loc[c] = 1.0
            wdf.loc[d] = row.mul(mask, fill_value=0.0).to_numpy(dtype=float)
            prev_group_holdings = set(str(c) for c in reduced)
    ps = str(getattr(inp, "position_sizing", "equal") or "equal").strip().lower()
    if ps in {"equal", "vol_target"}:
        w_decision = pd.DataFrame(
            0.0, index=wdf.index, columns=wdf.columns, dtype=float
        )
        if ps == "equal":
            for d in wdf.index:
                row = wdf.loc[d].astype(float)
                score_row = (
                    score_df.loc[d].astype(float).replace([np.inf, -np.inf], np.nan)
                )
                scores = score_row.where(row > 1e-12, other=np.nan)
                active = [
                    str(c)
                    for c in scores.dropna().sort_values(ascending=False).index.tolist()
                ]
                if not active:
                    continue
                per = 1.0 / float(len(active))
                for c in active:
                    w_decision.loc[d, c] = float(per)
        else:
            vol_window = int(getattr(inp, "vol_window", 20) or 20)
            vol_ann = pd.DataFrame(
                index=ret_hfq_df.index, columns=ret_hfq_df.columns, dtype=float
            )
            for c in ret_hfq_df.columns:
                vol_ann[c] = _rolling_std(
                    ret_hfq_df[c].astype(float),
                    window=vol_window,
                    min_periods=max(3, vol_window // 2),
                    ddof=1,
                ) * np.sqrt(252.0)
            vol_target_ann = float(getattr(inp, "vol_target_ann", 0.20) or 0.20)
            for d in wdf.index:
                row = wdf.loc[d].astype(float)
                score_row = (
                    score_df.loc[d].astype(float).replace([np.inf, -np.inf], np.nan)
                )
                scores = score_row.where(row > 1e-12, other=np.nan)
                active = [
                    str(c)
                    for c in scores.dropna().sort_values(ascending=False).index.tolist()
                ]
                if not active:
                    continue
                inv: dict[str, float] = {}
                for c in active:
                    av = (
                        float(vol_ann.loc[d, c])
                        if (c in vol_ann.columns and d in vol_ann.index)
                        else float("nan")
                    )
                    inv[c] = (1.0 / av) if (np.isfinite(av) and av > 0.0) else 0.0
                den = float(sum(inv.values()))
                if den > 0.0:
                    raw = {c: (float(v) / den) for c, v in inv.items()}
                    port_vol = float(
                        np.sqrt(
                            np.sum(
                                [
                                    (raw[c] ** 2)
                                    * (
                                        (
                                            float(vol_ann.loc[d, c])
                                            if np.isfinite(float(vol_ann.loc[d, c]))
                                            else 0.0
                                        )
                                        ** 2
                                    )
                                    for c in active
                                ]
                            )
                        )
                    )
                    scale = (
                        1.0
                        if port_vol <= 1e-12
                        else min(1.0, float(vol_target_ann) / port_vol)
                    )
                    for c in active:
                        w_decision.loc[d, c] = float(raw[c] * scale)
                else:
                    per = 1.0 / float(len(active))
                    for c in active:
                        w_decision.loc[d, c] = float(per)
        wdf = w_decision.astype(float)
    if ps == "risk_budget":
        # Keep binary active signals here; portfolio risk-budget sizing is
        # applied below with a stateful day-by-day loop.
        wdf = wdf.astype(float).clip(lower=0.0)
    overcap_scale_by_code = {str(c): 0 for c in wdf.columns}
    overcap_scale_total = 0
    overcap_skip_decision_total = 0
    overcap_skip_episode_total = 0
    overcap_skip_decision_by_code = {str(c): 0 for c in wdf.columns}
    overcap_skip_episode_by_code = {str(c): 0 for c in wdf.columns}
    overcap_skip_episode_active = {str(c): False for c in wdf.columns}
    overcap_replace_total = 0
    overcap_replace_out_by_code = {str(c): 0 for c in wdf.columns}
    overcap_replace_in_by_code = {str(c): 0 for c in wdf.columns}
    overcap_leverage_usage_total = 0
    overcap_leverage_usage_by_code = {str(c): 0 for c in wdf.columns}
    overcap_leverage_max_multiple = 0.0
    overcap_leverage_max_multiple_by_code = {str(c): 0.0 for c in wdf.columns}
    risk_budget_overcap_daily_counts: dict[str, dict[str, Any]] = {}
    fixed_ext_events: list[dict[str, Any]] = []
    fixed_skip_events: list[dict[str, Any]] = []
    if ps == "fixed_ratio":
        fixed_ratio = float(getattr(inp, "fixed_pos_ratio", 0.04) or 0.04)
        fixed_max_holding_n = int(getattr(inp, "fixed_max_holdings", 10) or 10)
        fixed_overcap_policy = (
            str(getattr(inp, "fixed_overcap_policy", "skip") or "skip").strip().lower()
        )
        prev_fixed_w = pd.Series(0.0, index=wdf.columns, dtype=float)
        for d in wdf.index:
            row = wdf.loc[d].astype(float)
            score_row = score_df.loc[d].astype(float).replace([np.inf, -np.inf], np.nan)
            scores = score_row.where(row > 1e-12, other=np.nan)
            active_codes = [
                str(c)
                for c in scores.dropna().sort_values(ascending=False).index.tolist()
            ]
            active_set = set(active_codes)
            w_row = prev_fixed_w.copy().astype(float).reindex(wdf.columns).fillna(0.0)
            for c in wdf.columns:
                key = str(c)
                if (float(w_row.loc[c]) > 1e-12) and (key not in active_set):
                    w_row.loc[c] = 0.0
            for key in active_set:
                if key in w_row.index and float(w_row.loc[key]) > 1e-12:
                    w_row.loc[key] = float(fixed_ratio)
            for key in active_codes:
                if key in w_row.index and float(w_row.loc[key]) > 1e-12:
                    continue
                cur_total = float(w_row.sum())
                proposed_total = float(cur_total + fixed_ratio)
                cur_count = int((w_row > 1e-12).sum())
                proposed_count = int(cur_count + 1)
                over_weight = bool(proposed_total > 1.0 + 1e-12)
                over_count = bool(proposed_count > fixed_max_holding_n)
                if over_weight or over_count:
                    evt = {
                        "date": pd.Timestamp(d).date().isoformat(),
                        "code": str(key),
                        "current_total": float(cur_total),
                        "proposed_total": float(proposed_total),
                        "current_count": int(cur_count),
                        "proposed_count": int(proposed_count),
                        "fixed_max_holdings": int(fixed_max_holding_n),
                        "fixed_pos_ratio": float(fixed_ratio),
                        "over_weight": bool(over_weight),
                        "over_count": bool(over_count),
                    }
                    if fixed_overcap_policy == "skip":
                        fixed_skip_events.append(evt)
                        continue
                    fixed_ext_events.append(evt)
                if key in w_row.index:
                    w_row.loc[key] = float(fixed_ratio)
            wdf.loc[d] = w_row.to_numpy(dtype=float)
            prev_fixed_w = w_row.copy()
    if ps == "risk_budget":
        policy = (
            str(getattr(inp, "risk_budget_overcap_policy", "scale") or "scale")
            .strip()
            .lower()
        )
        max_lev = float(getattr(inp, "risk_budget_max_leverage_multiple", 2.0) or 2.0)
        if (not np.isfinite(max_lev)) or max_lev <= 1.0:
            max_lev = 2.0
        eps = 1e-12
        risk_budget_pct = float(getattr(inp, "risk_budget_pct", 0.01) or 0.01)
        risk_budget_atr_window = int(getattr(inp, "risk_budget_atr_window", 20) or 20)
        vol_regime_risk_mgmt_enabled = bool(
            getattr(inp, "vol_regime_risk_mgmt_enabled", False)
        )
        vol_ratio_fast_atr_window = int(
            getattr(inp, "vol_ratio_fast_atr_window", 5) or 5
        )
        vol_ratio_slow_atr_window = int(
            getattr(inp, "vol_ratio_slow_atr_window", 50) or 50
        )
        vol_ratio_expand_threshold = float(
            getattr(inp, "vol_ratio_expand_threshold", 1.45) or 1.45
        )
        vol_ratio_contract_threshold = float(
            getattr(inp, "vol_ratio_contract_threshold", 0.65) or 0.65
        )
        vol_ratio_normal_threshold = float(
            getattr(inp, "vol_ratio_normal_threshold", 1.05) or 1.05
        )

        atr_budget_df = pd.DataFrame(index=wdf.index, columns=wdf.columns, dtype=float)
        atr_ratio_fast_df = pd.DataFrame(
            index=wdf.index, columns=wdf.columns, dtype=float
        )
        atr_ratio_slow_df = pd.DataFrame(
            index=wdf.index, columns=wdf.columns, dtype=float
        )
        for c in wdf.columns:
            pxc = price_sig_by_code.get(str(c), pd.DataFrame(index=wdf.index))
            cl = (
                pxc.get("close", pd.Series(np.nan, index=wdf.index))
                .reindex(wdf.index)
                .astype(float)
            )
            hi = (
                pxc.get("high", pd.Series(np.nan, index=wdf.index))
                .reindex(wdf.index)
                .astype(float)
                .fillna(cl)
            )
            lo = (
                pxc.get("low", pd.Series(np.nan, index=wdf.index))
                .reindex(wdf.index)
                .astype(float)
                .fillna(cl)
            )
            atr_budget_df[c] = _atr_from_hlc(
                hi, lo, cl, window=int(risk_budget_atr_window)
            ).astype(float)
            atr_ratio_fast_df[c] = _atr_from_hlc(
                hi, lo, cl, window=int(vol_ratio_fast_atr_window)
            ).astype(float)
            atr_ratio_slow_df[c] = _atr_from_hlc(
                hi, lo, cl, window=int(vol_ratio_slow_atr_window)
            ).astype(float)

        prev_rb_w = pd.Series(0.0, index=wdf.columns, dtype=float)
        rb_state_by_code: dict[str, str] = {str(c): "FLAT" for c in wdf.columns}
        rb_entry_price_by_code: dict[str, float] = {
            str(c): float("nan") for c in wdf.columns
        }
        rb_entry_seq_by_code: dict[str, int] = {str(c): -1 for c in wdf.columns}
        day_seq = 0
        for d in wdf.index:
            day_seq += 1
            d_key = str(pd.Timestamp(d).date())
            score_row = score_df.loc[d].astype(float).replace([np.inf, -np.inf], np.nan)
            sig_row = wdf.loc[d].astype(float).clip(lower=0.0)
            active_codes = [
                str(c)
                for c in score_row.where(sig_row > eps, other=np.nan)
                .dropna()
                .sort_values(ascending=False)
                .index.tolist()
            ]
            active_set = set(active_codes)
            w_row = prev_rb_w.copy().astype(float).reindex(wdf.columns).fillna(0.0)
            skipped_today: set[str] = set()

            def _inc_overcap_daily(kind: str, n: int = 1) -> None:
                nn = int(n)
                if nn <= 0:
                    return
                row_d = risk_budget_overcap_daily_counts.setdefault(
                    d_key,
                    {
                        "scale": 0,
                        "skip_entry": 0,
                        "replace_entry": 0,
                        "leverage_entry": 0,
                        "leverage_multiple_max": 0.0,
                    },
                )
                row_d[str(kind)] = int(row_d.get(str(kind), 0) + nn)

            def _apply_overcap_scale_once(cap_multiple: float = 1.0) -> None:
                nonlocal w_row, overcap_scale_total
                cap_v = (
                    float(cap_multiple)
                    if np.isfinite(float(cap_multiple)) and float(cap_multiple) > 0.0
                    else 1.0
                )
                s_now = float(w_row.sum())
                if s_now <= cap_v + eps:
                    return
                pre_scale = w_row.copy().astype(float)
                w_row = (w_row * (cap_v / s_now)).astype(float)
                overcap_scale_total += 1
                _inc_overcap_daily("scale", 1)
                for cc in w_row.index:
                    key_cc = str(cc)
                    before = (
                        float(pre_scale.loc[cc])
                        if np.isfinite(float(pre_scale.loc[cc]))
                        else 0.0
                    )
                    after = (
                        float(w_row.loc[cc])
                        if np.isfinite(float(w_row.loc[cc]))
                        else 0.0
                    )
                    if before > after + eps:
                        overcap_scale_by_code[key_cc] = int(
                            overcap_scale_by_code.get(key_cc, 0) + 1
                        )

            def _set_new_risk_budget_entry(key: str, base_target: float) -> None:
                nonlocal w_row
                w_row.loc[key] = float(base_target)
                px_now = float(
                    price_sig_by_code.get(key, pd.DataFrame(index=wdf.index))
                    .get("close", pd.Series(np.nan, index=wdf.index))
                    .reindex(wdf.index)
                    .loc[d]
                )
                rb_entry_price_by_code[key] = (
                    float(px_now)
                    if np.isfinite(px_now) and px_now > 0.0
                    else float("nan")
                )
                rb_entry_seq_by_code[key] = int(day_seq)
                if bool(vol_regime_risk_mgmt_enabled):
                    af = (
                        float(atr_ratio_fast_df.loc[d, key])
                        if (
                            key in atr_ratio_fast_df.columns
                            and d in atr_ratio_fast_df.index
                        )
                        else float("nan")
                    )
                    aslow = (
                        float(atr_ratio_slow_df.loc[d, key])
                        if (
                            key in atr_ratio_slow_df.columns
                            and d in atr_ratio_slow_df.index
                        )
                        else float("nan")
                    )
                    ratio = (
                        (af / aslow)
                        if (np.isfinite(af) and np.isfinite(aslow) and aslow > 0.0)
                        else float("nan")
                    )
                    if np.isfinite(ratio) and ratio > float(vol_ratio_expand_threshold):
                        rb_state_by_code[key] = "REDUCED"
                    elif np.isfinite(ratio) and ratio < float(
                        vol_ratio_contract_threshold
                    ):
                        rb_state_by_code[key] = "INCREASED"
                    else:
                        rb_state_by_code[key] = "NORMAL"
                else:
                    rb_state_by_code[key] = "NORMAL"

            def _select_replace_out_code(new_code: str) -> str | None:
                cand: list[tuple[float, int, str]] = []
                for cc in wdf.columns:
                    key_cc = str(cc)
                    if key_cc == str(new_code):
                        continue
                    if float(w_row.loc[key_cc]) <= eps:
                        continue
                    cur_px = float(
                        price_sig_by_code.get(key_cc, pd.DataFrame(index=wdf.index))
                        .get("close", pd.Series(np.nan, index=wdf.index))
                        .reindex(wdf.index)
                        .loc[d]
                    )
                    ent_px = float(rb_entry_price_by_code.get(key_cc, float("nan")))
                    ret = (
                        ((cur_px / ent_px) - 1.0)
                        if (
                            np.isfinite(cur_px)
                            and cur_px > 0.0
                            and np.isfinite(ent_px)
                            and ent_px > 0.0
                        )
                        else float("inf")
                    )
                    seq = int(rb_entry_seq_by_code.get(key_cc, 10**9))
                    if seq < 0:
                        seq = 10**9
                    cand.append((float(ret), int(seq), key_cc))
                if not cand:
                    return None
                cand.sort(key=lambda x: (x[0], x[1], x[2]))
                return str(cand[0][2])

            for c in wdf.columns:
                key_c = str(c)
                if (float(w_row.loc[c]) > eps) and (key_c not in active_set):
                    w_row.loc[c] = 0.0
                    rb_state_by_code[key_c] = "FLAT"
                    rb_entry_price_by_code[key_c] = float("nan")
                    rb_entry_seq_by_code[key_c] = -1

            for c in active_codes:
                px = float(
                    price_sig_by_code.get(str(c), pd.DataFrame(index=wdf.index))
                    .get("close", pd.Series(np.nan, index=wdf.index))
                    .reindex(wdf.index)
                    .loc[d]
                )
                a = (
                    float(atr_budget_df.loc[d, c])
                    if (c in atr_budget_df.columns and d in atr_budget_df.index)
                    else float("nan")
                )
                base_target = float("nan")
                if np.isfinite(px) and px > 0.0 and np.isfinite(a) and a > 0.0:
                    base_target = float(risk_budget_pct) * float(px) / float(a)
                has_pos = bool(float(w_row.loc[c]) > eps)
                key = str(c)
                if not has_pos:
                    if np.isfinite(base_target) and base_target > 0.0:
                        proposed_total = float(w_row.sum() + float(base_target))
                        overcap_on_new_entry = bool(proposed_total > 1.0 + eps)
                        if overcap_on_new_entry and str(policy) == "skip_entry":
                            overcap_skip_decision_total += 1
                            _inc_overcap_daily("skip_entry", 1)
                            overcap_skip_decision_by_code[key] = int(
                                overcap_skip_decision_by_code.get(key, 0) + 1
                            )
                            skipped_today.add(key)
                            if not bool(overcap_skip_episode_active.get(key, False)):
                                overcap_skip_episode_active[key] = True
                                overcap_skip_episode_total += 1
                                overcap_skip_episode_by_code[key] = int(
                                    overcap_skip_episode_by_code.get(key, 0) + 1
                                )
                            continue
                        if overcap_on_new_entry and str(policy) == "replace_entry":
                            out_code = _select_replace_out_code(key)
                            if out_code:
                                w_row.loc[out_code] = 0.0
                                rb_state_by_code[out_code] = "FLAT"
                                rb_entry_price_by_code[out_code] = float("nan")
                                rb_entry_seq_by_code[out_code] = -1
                                overcap_replace_total += 1
                                _inc_overcap_daily("replace_entry", 1)
                                overcap_replace_out_by_code[out_code] = int(
                                    overcap_replace_out_by_code.get(out_code, 0) + 1
                                )
                                overcap_replace_in_by_code[key] = int(
                                    overcap_replace_in_by_code.get(key, 0) + 1
                                )
                        _set_new_risk_budget_entry(key, float(base_target))
                        if overcap_on_new_entry and str(policy) == "replace_entry":
                            _apply_overcap_scale_once()
                        elif overcap_on_new_entry and str(policy) == "leverage_entry":
                            lev_now = float(w_row.sum())
                            if lev_now > 1.0 + eps:
                                overcap_leverage_usage_total += 1
                                _inc_overcap_daily("leverage_entry", 1)
                                landed_lev = float(min(float(lev_now), float(max_lev)))
                                overcap_leverage_max_multiple = float(
                                    max(
                                        float(overcap_leverage_max_multiple), landed_lev
                                    )
                                )
                                row_d = risk_budget_overcap_daily_counts.setdefault(
                                    d_key,
                                    {
                                        "scale": 0,
                                        "skip_entry": 0,
                                        "replace_entry": 0,
                                        "leverage_entry": 0,
                                        "leverage_multiple_max": 0.0,
                                    },
                                )
                                row_d["leverage_multiple_max"] = float(
                                    max(
                                        float(
                                            row_d.get("leverage_multiple_max", 0.0)
                                            or 0.0
                                        ),
                                        landed_lev,
                                    )
                                )
                                for cc in wdf.columns:
                                    key_cc = str(cc)
                                    if float(w_row.loc[key_cc]) > eps:
                                        overcap_leverage_usage_by_code[key_cc] = int(
                                            overcap_leverage_usage_by_code.get(
                                                key_cc, 0
                                            )
                                            + 1
                                        )
                                        overcap_leverage_max_multiple_by_code[
                                            key_cc
                                        ] = float(
                                            max(
                                                float(
                                                    overcap_leverage_max_multiple_by_code.get(
                                                        key_cc, 0.0
                                                    )
                                                ),
                                                landed_lev,
                                            )
                                        )
                                if lev_now > float(max_lev) + eps:
                                    _apply_overcap_scale_once(float(max_lev))
                    continue

                if not bool(vol_regime_risk_mgmt_enabled):
                    continue
                st = str(rb_state_by_code.get(key, "NORMAL") or "NORMAL").upper()
                af = (
                    float(atr_ratio_fast_df.loc[d, c])
                    if (c in atr_ratio_fast_df.columns and d in atr_ratio_fast_df.index)
                    else float("nan")
                )
                aslow = (
                    float(atr_ratio_slow_df.loc[d, c])
                    if (c in atr_ratio_slow_df.columns and d in atr_ratio_slow_df.index)
                    else float("nan")
                )
                ratio = (
                    (af / aslow)
                    if (np.isfinite(af) and np.isfinite(aslow) and aslow > 0.0)
                    else float("nan")
                )
                if not np.isfinite(base_target):
                    continue
                if st == "NORMAL":
                    if np.isfinite(ratio) and ratio > float(vol_ratio_expand_threshold):
                        w_row.loc[c] = float(base_target)
                        rb_state_by_code[key] = "REDUCED"
                    elif np.isfinite(ratio) and ratio < float(
                        vol_ratio_contract_threshold
                    ):
                        w_row.loc[c] = float(base_target)
                        rb_state_by_code[key] = "INCREASED"
                elif st == "REDUCED":
                    if np.isfinite(ratio) and ratio < float(vol_ratio_normal_threshold):
                        w_row.loc[c] = float(base_target)
                        rb_state_by_code[key] = "NORMAL"
                elif st == "INCREASED":
                    if np.isfinite(ratio) and ratio > float(vol_ratio_normal_threshold):
                        w_row.loc[c] = float(base_target)
                        rb_state_by_code[key] = "NORMAL"

            w_row = w_row.clip(lower=0.0)
            if str(policy) != "leverage_entry":
                _apply_overcap_scale_once()
            for key_cc in wdf.columns:
                kk = str(key_cc)
                if bool(overcap_skip_episode_active.get(kk, False)) and (
                    kk not in skipped_today
                ):
                    overcap_skip_episode_active[kk] = False
            wdf.loc[d] = w_row.to_numpy(dtype=float)
            prev_rb_w = w_row.copy()

    monthly_attempted_total = 0
    monthly_blocked_total = 0
    if bool(getattr(inp, "monthly_risk_budget_enabled", False)):
        close_df = pd.DataFrame(
            {
                c: price_sig_by_code.get(c, pd.DataFrame(index=wdf.index))
                .get("close", pd.Series(np.nan, index=wdf.index))
                .reindex(wdf.index)
                .astype(float)
                for c in wdf.columns
            },
            index=wdf.index,
        )
        atr_df = pd.DataFrame(index=wdf.index)
        for c in wdf.columns:
            pxc = price_sig_by_code.get(c, pd.DataFrame(index=wdf.index))
            atr_df[c] = (
                _atr_from_hlc(
                    pxc.get("high", pd.Series(np.nan, index=wdf.index))
                    .reindex(wdf.index)
                    .astype(float),
                    pxc.get("low", pd.Series(np.nan, index=wdf.index))
                    .reindex(wdf.index)
                    .astype(float),
                    pxc.get("close", pd.Series(np.nan, index=wdf.index))
                    .reindex(wdf.index)
                    .astype(float),
                    window=int(getattr(inp, "atr_stop_window", 14) or 14),
                )
                .reindex(wdf.index)
                .astype(float)
            )
        wdf, gate_stats = _apply_monthly_risk_budget_gate(
            wdf.astype(float),
            close=close_df.astype(float),
            atr=atr_df.astype(float),
            enabled=True,
            budget_pct=float(getattr(inp, "monthly_risk_budget_pct", 0.06) or 0.06),
            include_new_trade_risk=bool(
                getattr(inp, "monthly_risk_budget_include_new_trade_risk", False)
            ),
            atr_stop_enabled=bool(
                str(getattr(inp, "atr_stop_mode", "none") or "none").strip().lower()
                != "none"
            ),
            atr_mode=str(getattr(inp, "atr_stop_mode", "none") or "none"),
            atr_basis=str(getattr(inp, "atr_stop_atr_basis", "latest") or "latest"),
            atr_n=float(getattr(inp, "atr_stop_n", 2.0) or 2.0),
            atr_m=float(getattr(inp, "atr_stop_m", 0.5) or 0.5),
            fallback_position_risk=0.02,
        )
        monthly_attempted_total = int(
            (gate_stats or {}).get("attempted_entry_count", 0)
        )
        monthly_blocked_total = int((gate_stats or {}).get("blocked_entry_count", 0))

    ret_exec_df = (
        pd.DataFrame(ret_exec_map)
        .reindex(index=wdf.index, columns=wdf.columns)
        .fillna(0.0)
        .astype(float)
    )
    px_exec_slip_df = (
        pd.DataFrame(px_exec_slip_map)
        .reindex(index=wdf.index, columns=wdf.columns)
        .ffill()
        .astype(float)
    )
    open_sig_df = (
        pd.DataFrame(sig_open_map)
        .reindex(index=wdf.index, columns=wdf.columns)
        .ffill()
        .astype(float)
    )
    close_sig_df = (
        pd.DataFrame(sig_close_map)
        .reindex(index=wdf.index, columns=wdf.columns)
        .ffill()
        .astype(float)
    )
    ret_overnight_df_comp: pd.DataFrame | None = None
    ret_intraday_df_comp: pd.DataFrame | None = None
    try:
        ccodes = [str(c) for c in wdf.columns]
        idx = wdf.index
        ep_port = str(getattr(inp, "exec_price", "open") or "open").strip().lower()
        ps_port = (
            str(getattr(inp, "position_sizing", "equal") or "equal").strip().lower()
        )
        if (
            ep_port == "open"
            and ps_port == "risk_budget"
            and (not bool(getattr(inp, "quick_mode", False)))
        ):
            raise RuntimeError(
                "keep single-derived mapping for open+risk_budget parity"
            )
        close_none_df = (
            load_close_prices(
                db, codes=ccodes, start=inp.start, end=inp.end, adjust="none"
            )
            .reindex(index=idx, columns=ccodes)
            .ffill()
            .astype(float)
        )
        close_hfq_df = (
            load_close_prices(
                db, codes=ccodes, start=inp.start, end=inp.end, adjust="hfq"
            )
            .reindex(index=idx, columns=ccodes)
            .ffill()
            .astype(float)
        )
        close_qfq_df = (
            load_close_prices(
                db, codes=ccodes, start=inp.start, end=inp.end, adjust="qfq"
            )
            .reindex(index=idx, columns=ccodes)
            .ffill()
            .astype(float)
        )
        ohlc_none = load_ohlc_prices(
            db, codes=ccodes, start=inp.start, end=inp.end, adjust="none"
        )
        ohlc_hfq = load_ohlc_prices(
            db, codes=ccodes, start=inp.start, end=inp.end, adjust="hfq"
        )
        ohlc_qfq = load_ohlc_prices(
            db, codes=ccodes, start=inp.start, end=inp.end, adjust="qfq"
        )

        def _raw_df(ohlc: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
            df = (
                ohlc.get(field, pd.DataFrame())
                if isinstance(ohlc, dict)
                else pd.DataFrame()
            )
            if df is None or df.empty:
                return pd.DataFrame(index=idx, columns=ccodes, dtype=float)
            return (
                df.sort_index().reindex(index=idx, columns=ccodes).astype(float).ffill()
            )

        open_none_raw = _raw_df(ohlc_none, "open")
        close_none_raw = _raw_df(ohlc_none, "close")
        open_hfq_raw = _raw_df(ohlc_hfq, "open")
        close_hfq_raw = _raw_df(ohlc_hfq, "close")
        open_qfq_raw = _raw_df(ohlc_qfq, "open")
        close_qfq_raw = _raw_df(ohlc_qfq, "close")
        open_none_exec = open_none_raw.combine_first(close_none_df)
        close_none_exec = close_none_raw.combine_first(close_none_df)
        open_hfq_exec = open_hfq_raw.combine_first(close_hfq_df)
        close_hfq_exec = close_hfq_raw.combine_first(close_hfq_df)
        if ep_port == "open":
            ret_none_base = _forward_simple_return(open_none_exec)
            ret_hfq_base = _forward_simple_return(open_hfq_exec)
            px_none_base = open_none_exec.astype(float)
            px_hfq_base = open_hfq_exec.astype(float)
        elif ep_port == "close":
            ret_none_base = _forward_simple_return(close_none_exec)
            ret_hfq_base = _forward_simple_return(close_hfq_exec)
            px_none_base = close_none_exec.astype(float)
            px_hfq_base = close_hfq_exec.astype(float)
        else:
            ret_open_none_base = _forward_simple_return(open_none_exec)
            ret_close_none_base = _forward_simple_return(close_none_exec)
            ret_none_base = (0.5 * (ret_open_none_base + ret_close_none_base)).astype(
                float
            )
            ret_open_hfq_base = _forward_simple_return(open_hfq_exec)
            ret_close_hfq_base = _forward_simple_return(close_hfq_exec)
            ret_hfq_base = (0.5 * (ret_open_hfq_base + ret_close_hfq_base)).astype(
                float
            )
            px_none_base = (0.5 * (open_none_exec + close_none_exec)).astype(float)
            px_hfq_base = (0.5 * (open_hfq_exec + close_hfq_exec)).astype(float)
        _, ca_mask_base = corporate_action_mask(
            (1.0 + ret_none_base).astype(float), (1.0 + ret_hfq_base).astype(float)
        )
        ca_mask_base = (
            ca_mask_base.reindex(index=idx, columns=ccodes).fillna(False).astype(bool)
        )
        ret_exec_df = ret_none_base.where(~ca_mask_base, other=ret_hfq_base).astype(
            float
        )
        px_exec_slip_df = (
            px_none_base.where(~ca_mask_base, other=px_hfq_base)
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .astype(float)
        )
        open_sig_df = open_qfq_raw.astype(float)
        close_sig_df = close_qfq_df.astype(float)
        ret_overnight_none_close = _ratio_simple_return(
            open_none_raw.shift(-1), close_none_raw
        )
        ret_intraday_none_close = _ratio_simple_return(
            close_none_raw.shift(-1), open_none_raw.shift(-1)
        )
        ret_overnight_hfq_close = _ratio_simple_return(
            open_hfq_raw.shift(-1), close_hfq_raw
        )
        ret_intraday_hfq_close = _ratio_simple_return(
            close_hfq_raw.shift(-1), open_hfq_raw.shift(-1)
        )
        ret_intraday_none_open = _ratio_simple_return(close_none_raw, open_none_raw)
        ret_overnight_none_open = _ratio_simple_return(
            open_none_raw.shift(-1), close_none_raw
        )
        ret_intraday_hfq_open = _ratio_simple_return(close_hfq_raw, open_hfq_raw)
        ret_overnight_hfq_open = _ratio_simple_return(
            open_hfq_raw.shift(-1), close_hfq_raw
        )
        ret_overnight_close = ret_overnight_none_close.where(
            ~ca_mask_base, other=ret_overnight_hfq_close
        ).astype(float)
        ret_intraday_close = ret_intraday_none_close.where(
            ~ca_mask_base, other=ret_intraday_hfq_close
        ).astype(float)
        ret_overnight_open = ret_overnight_none_open.where(
            ~ca_mask_base, other=ret_overnight_hfq_open
        ).astype(float)
        ret_intraday_open = ret_intraday_none_open.where(
            ~ca_mask_base, other=ret_intraday_hfq_open
        ).astype(float)
        if ep_port == "open":
            ret_overnight_df_comp = ret_overnight_open.astype(float)
            ret_intraday_df_comp = ret_intraday_open.astype(float)
        elif ep_port == "close":
            ret_overnight_df_comp = ret_overnight_close.astype(float)
            ret_intraday_df_comp = ret_intraday_close.astype(float)
        else:
            ret_overnight_df_comp = (0.5 * ret_overnight_close).astype(float)
            ret_intraday_df_comp = (
                0.5 * ret_exec_df + 0.5 * ret_intraday_close
            ).astype(float)
    except Exception:
        pass
    atr_stop_by_asset = {
        str(c): dict(
            (semantic_debug_by_code.get(str(c), {}) or {}).get("atr_stop") or {}
        )
        for c in wdf.columns
    }
    bias_v_tp_by_asset = {
        str(c): dict(
            (semantic_debug_by_code.get(str(c), {}) or {}).get("bias_v_take_profit")
            or {}
        )
        for c in wdf.columns
    }
    rtp_by_asset = {
        str(c): dict(
            (semantic_debug_by_code.get(str(c), {}) or {}).get("r_take_profit") or {}
        )
        for c in wdf.columns
    }

    w_eff = wdf.shift(1).fillna(0.0).astype(float).clip(lower=0.0)
    w_eff, atr_stop_override_ret = _apply_intraday_stop_execution_portfolio(
        weights=w_eff,
        atr_stop_by_asset=atr_stop_by_asset,
        exec_price=str(getattr(inp, "exec_price", "open") or "open"),
        open_sig_df=open_sig_df,
        close_sig_df=close_sig_df,
    )
    w_eff, bias_v_take_profit_override_ret = _apply_intraday_stop_execution_portfolio(
        weights=w_eff,
        atr_stop_by_asset=bias_v_tp_by_asset,
        exec_price=str(getattr(inp, "exec_price", "open") or "open"),
        open_sig_df=open_sig_df,
        close_sig_df=close_sig_df,
    )
    w_eff, r_take_profit_override_ret = _apply_intraday_stop_execution_portfolio(
        weights=w_eff,
        atr_stop_by_asset=rtp_by_asset,
        exec_price=str(getattr(inp, "exec_price", "open") or "open"),
        open_sig_df=open_sig_df,
        close_sig_df=close_sig_df,
    )
    turnover = (w_eff - w_eff.shift(1).fillna(0.0)).abs().sum(axis=1) / 2.0
    turnover_by_asset = (w_eff - w_eff.shift(1).fillna(0.0)).abs() / 2.0
    cost = turnover * (float(getattr(inp, "cost_bps", 0.0) or 0.0) / 10000.0)
    slippage = (
        slippage_return_from_turnover(
            turnover_by_asset.astype(float),
            slippage_spread=float(getattr(inp, "slippage_rate", 0.0) or 0.0),
            exec_price=px_exec_slip_df.astype(float),
        )
        .sum(axis=1)
        .astype(float)
    )
    base_ret = (w_eff * ret_exec_df).sum(axis=1).astype(float)
    decomp_risk = (
        atr_stop_override_ret.reindex(w_eff.index).fillna(0.0).astype(float)
        + bias_v_take_profit_override_ret.reindex(w_eff.index).fillna(0.0).astype(float)
        + r_take_profit_override_ret.reindex(w_eff.index).fillna(0.0).astype(float)
    ).astype(float)
    decomp_cost = (cost + slippage).astype(float)
    port_ret = (base_ret + decomp_risk - decomp_cost).fillna(0.0).astype(float)
    _ep_rt = str(getattr(inp, "exec_price", "open") or "open").strip().lower()
    _ps_rt = str(getattr(inp, "position_sizing", "equal") or "equal").strip().lower()
    _keep_base_port_ret = bool(_ep_rt == "open" and _ps_rt == "risk_budget")
    if ret_overnight_df_comp is None or ret_intraday_df_comp is None:
        ret_overnight_df = (
            (open_sig_df / close_sig_df.shift(1) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_intraday_df = (
            (close_sig_df / open_sig_df - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
    else:
        ret_overnight_df = (
            ret_overnight_df_comp.reindex(index=w_eff.index, columns=w_eff.columns)
            .fillna(0.0)
            .astype(float)
        )
        ret_intraday_df = (
            ret_intraday_df_comp.reindex(index=w_eff.index, columns=w_eff.columns)
            .fillna(0.0)
            .astype(float)
        )
    return_decomposition = None
    if not bool(getattr(inp, "quick_mode", False)):
        comp_overnight = (w_eff * ret_overnight_df).sum(axis=1).astype(float)
        comp_intraday = (w_eff * ret_intraday_df).sum(axis=1).astype(float)
        comp_interaction = (
            (w_eff * ret_overnight_df * ret_intraday_df).sum(axis=1).astype(float)
        )
        decomp_gross = (
            comp_overnight + comp_intraday + comp_interaction + decomp_risk
        ).astype(float)
        decomp_net = (decomp_gross - decomp_cost).astype(float)
        if not _keep_base_port_ret:
            port_ret = decomp_net.fillna(0.0).astype(float)
        return_decomposition = {
            "dates": [d.strftime("%Y-%m-%d") for d in w_eff.index],
            "series": {
                "overnight": comp_overnight.tolist(),
                "intraday": comp_intraday.tolist(),
                "interaction": comp_interaction.tolist(),
                "atr_stop_override": atr_stop_override_ret.reindex(w_eff.index)
                .fillna(0.0)
                .astype(float)
                .tolist(),
                "bias_v_take_profit_override": bias_v_take_profit_override_ret.reindex(
                    w_eff.index
                )
                .fillna(0.0)
                .astype(float)
                .tolist(),
                "r_take_profit_override": r_take_profit_override_ret.reindex(
                    w_eff.index
                )
                .fillna(0.0)
                .astype(float)
                .tolist(),
                "risk_exit_override": decomp_risk.tolist(),
                "cost": decomp_cost.tolist(),
                "gross": decomp_gross.tolist(),
                "net": decomp_net.tolist(),
            },
            "summary": {
                "ann_overnight": float(comp_overnight.iloc[1:].mean() * 252.0)
                if len(comp_overnight) > 1
                else 0.0,
                "ann_intraday": float(comp_intraday.iloc[1:].mean() * 252.0)
                if len(comp_intraday) > 1
                else 0.0,
                "ann_interaction": float(comp_interaction.iloc[1:].mean() * 252.0)
                if len(comp_interaction) > 1
                else 0.0,
                "ann_atr_stop_override": float(
                    atr_stop_override_ret.iloc[1:].mean() * 252.0
                )
                if len(atr_stop_override_ret) > 1
                else 0.0,
                "ann_bias_v_take_profit_override": (
                    float(bias_v_take_profit_override_ret.iloc[1:].mean() * 252.0)
                    if len(bias_v_take_profit_override_ret) > 1
                    else 0.0
                ),
                "ann_r_take_profit_override": float(
                    r_take_profit_override_ret.iloc[1:].mean() * 252.0
                )
                if len(r_take_profit_override_ret) > 1
                else 0.0,
                "ann_risk_exit_override": float(decomp_risk.iloc[1:].mean() * 252.0)
                if len(decomp_risk) > 1
                else 0.0,
                "ann_cost": float(decomp_cost.iloc[1:].mean() * 252.0)
                if len(decomp_cost) > 1
                else 0.0,
                "ann_gross": float(decomp_gross.iloc[1:].mean() * 252.0)
                if len(decomp_gross) > 1
                else 0.0,
                "ann_net": float(decomp_net.iloc[1:].mean() * 252.0)
                if len(decomp_net) > 1
                else 0.0,
            },
        }
    nav = _as_nav(port_ret)
    close_hfq = (
        load_close_prices(
            db, codes=list(nav_map.keys()), start=inp.start, end=inp.end, adjust="hfq"
        )
        .reindex(nav.index)
        .ffill()
    )
    bh_ret = (
        hfq_close_daily_equal_weight_returns(
            close_hfq, dynamic_universe=dynamic_universe
        )
        .reindex(nav.index)
        .fillna(0.0)
    )
    bh_nav = _as_nav(bh_ret)
    excess_nav = (nav / bh_nav.replace(0.0, np.nan)).fillna(1.0)
    excess_ret = _tsmom_rocp(excess_nav, 1).fillna(0.0).astype(float)
    active_ret = (
        (
            port_ret.reindex(nav.index).astype(float)
            - bh_ret.reindex(nav.index).astype(float)
        )
        .fillna(0.0)
        .astype(float)
    )
    event_study = (
        None
        if bool(getattr(inp, "quick_mode", False))
        else compute_event_study(
            dates=nav.index,
            daily_returns=port_ret.reindex(nav.index).astype(float),
            entry_dates=entry_dates_from_exposure(
                w_eff.sum(axis=1).reindex(nav.index).astype(float)
            ),
        )
    )
    high_sig_df = pd.DataFrame(
        {
            c: price_sig_by_code.get(c, pd.DataFrame(index=nav.index))
            .get("high", pd.Series(np.nan, index=nav.index))
            .reindex(nav.index)
            .astype(float)
            for c in w_eff.columns
        },
        index=nav.index,
    )
    low_sig_df = pd.DataFrame(
        {
            c: price_sig_by_code.get(c, pd.DataFrame(index=nav.index))
            .get("low", pd.Series(np.nan, index=nav.index))
            .reindex(nav.index)
            .astype(float)
            for c in w_eff.columns
        },
        index=nav.index,
    )
    market_regime = build_market_regime_report(
        close=close_sig_df.reindex(index=nav.index, columns=w_eff.columns).astype(
            float
        ),
        high=high_sig_df.reindex(index=nav.index, columns=w_eff.columns).astype(float),
        low=low_sig_df.reindex(index=nav.index, columns=w_eff.columns).astype(float),
        weights=w_eff.reindex(index=nav.index, columns=w_eff.columns)
        .astype(float)
        .fillna(0.0),
        asset_returns=ret_exec_df.reindex(index=nav.index, columns=w_eff.columns)
        .astype(float)
        .fillna(0.0),
        strategy_returns=port_ret.reindex(nav.index).astype(float),
        ann_factor=252,
    )
    attribution = _compute_return_risk_contributions(
        asset_ret=ret_exec_df.reindex(index=nav.index, columns=w_eff.columns)
        .astype(float)
        .fillna(0.0),
        weights=w_eff.reindex(index=nav.index, columns=w_eff.columns)
        .astype(float)
        .fillna(0.0),
        total_return=float(nav.iloc[-1] - 1.0) if len(nav) else 0.0,
    )
    holdings: list[dict[str, Any]] = []
    prev_picks: tuple[str, ...] | None = None
    for d in wdf.index:
        picks = tuple(
            sorted([str(c) for c in wdf.columns if float(wdf.loc[d, c]) > 1e-12])
        )
        if picks != prev_picks:
            score_row = (
                score_df.loc[d] if d in score_df.index else pd.Series(dtype=float)
            )
            group_meta_raw = dict(
                group_filter_meta_by_date.get(pd.Timestamp(d), {}) or {}
            )
            group_filter_norm = {
                "enabled": bool(group_meta_raw.get("enabled", False)),
                "policy": str(
                    group_meta_raw.get(
                        "policy",
                        getattr(inp, "group_pick_policy", "highest_sharpe")
                        or "highest_sharpe",
                    )
                ),
                "max_holdings_per_group": int(
                    group_meta_raw.get(
                        "max_holdings_per_group",
                        getattr(inp, "group_max_holdings", 4) or 4,
                    )
                ),
                "before": [str(x) for x in (group_meta_raw.get("before") or [])],
                "after": [str(x) for x in (group_meta_raw.get("after") or [])],
                "group_winners": {
                    str(k): [str(x) for x in (v or [])]
                    for k, v in (
                        (group_meta_raw.get("group_winners") or {}) or {}
                    ).items()
                },
                "group_eliminated": {
                    str(k): [str(x) for x in (v or [])]
                    for k, v in (
                        (group_meta_raw.get("group_eliminated") or {}) or {}
                    ).items()
                },
                "group_picks": {
                    str(k): [str(x) for x in (v or [])]
                    for k, v in (
                        (group_meta_raw.get("group_picks") or {}) or {}
                    ).items()
                },
            }
            holdings.append(
                {
                    "decision_date": d.date().isoformat()
                    if hasattr(d, "date")
                    else str(d),
                    "picks": list(picks),
                    "grouped_picks": {
                        str(k): [str(x) for x in (v or [])]
                        for k, v in (
                            group_filter_norm.get("group_picks", {}) or {}
                        ).items()
                    },
                    "scores": {
                        str(c): (
                            None
                            if pd.isna(score_row.get(c))
                            else float(score_row.get(c))
                        )
                        for c in picks
                    },
                    "group_filter": group_filter_norm,
                }
            )
            prev_picks = picks
    group_filter_enabled_segments = int(
        sum(
            1
            for h in holdings
            if bool(((h or {}).get("group_filter") or {}).get("enabled"))
        )
    )
    group_filter_effective_segments = int(
        sum(
            1
            for h in holdings
            if bool(((h or {}).get("group_filter") or {}).get("enabled"))
            and (
                len((((h or {}).get("group_filter") or {}).get("before") or []))
                > len((((h or {}).get("group_filter") or {}).get("after") or []))
            )
        )
    )
    trade_pack = _trade_returns_from_weight_df(
        w_eff.reindex(index=nav.index, columns=w_eff.columns).astype(float).fillna(0.0),
        ret_exec_df.reindex(index=nav.index, columns=w_eff.columns)
        .astype(float)
        .fillna(0.0),
        cost_bps=float(getattr(inp, "cost_bps", 0.0) or 0.0),
        slippage_rate=float(getattr(inp, "slippage_rate", 0.0) or 0.0),
        exec_price=px_exec_slip_df.reindex(
            index=nav.index, columns=w_eff.columns
        ).ffill(),
        dates=nav.index,
    )
    atr_risk_df = pd.DataFrame(index=nav.index, columns=w_eff.columns, dtype=float)
    for c in w_eff.columns:
        cl = close_sig_df[c].reindex(nav.index).astype(float)
        hi = high_sig_df[c].reindex(nav.index).astype(float).fillna(cl)
        lo = low_sig_df[c].reindex(nav.index).astype(float).fillna(cl)
        atr_risk_df[c] = (
            _atr_from_hlc(
                hi,
                lo,
                cl,
                window=int(getattr(inp, "atr_stop_window", 14) or 14),
            )
            .reindex(nav.index)
            .astype(float)
        )
    trade_r_pack = enrich_trades_with_r_metrics(
        trade_pack.get("trades", []),
        nav=nav.astype(float),
        weights=w_eff.reindex(index=nav.index, columns=w_eff.columns)
        .astype(float)
        .fillna(0.0),
        exec_price=px_exec_slip_df.reindex(index=nav.index, columns=w_eff.columns)
        .ffill()
        .astype(float),
        atr=atr_risk_df.reindex(index=nav.index, columns=w_eff.columns).astype(float),
        atr_mult=float(getattr(inp, "atr_stop_n", 2.0) or 2.0),
        risk_budget_pct=(
            float(getattr(inp, "risk_budget_pct", 0.01) or 0.01)
            if ps == "risk_budget"
            else None
        ),
        cost_bps=float(getattr(inp, "cost_bps", 0.0) or 0.0),
        slippage_rate=float(getattr(inp, "slippage_rate", 0.0) or 0.0),
        ulcer_index=float(_ulcer_index(nav, in_percent=True)) if len(nav) else None,
        annual_trade_count=(
            float(len(trade_pack.get("returns", [])))
            / max(1.0, float(len(nav)))
            * 252.0
        )
        if len(nav)
        else None,
        backtest_years=(float(len(nav)) / 252.0) if len(nav) else None,
        score_sqn_weight=0.60,
        score_ulcer_weight=0.40,
    )
    trades_with_r = list(trade_r_pack.get("trades") or [])
    r_stats_out = dict(trade_r_pack.get("statistics") or {})
    r_stats_out.pop("trade_system_score", None)
    if not bool(getattr(inp, "quick_mode", False)):
        condition_bins_by_code: dict[str, dict[str, pd.Series]] = {}
        for c in w_eff.columns:
            ck = str(c)
            cl = close_sig_df[ck].reindex(nav.index).astype(float)
            mom_for_entry = _tsmom_rocp(
                cl,
                int(getattr(inp, "mom_lookback", 252) or 252),
            ).astype(float)
            er_for_entry = _efficiency_ratio(
                cl, window=int(getattr(inp, "er_window", 10) or 10)
            ).astype(float)
            atr_fast = _atr_from_hlc(
                high_sig_df[ck].reindex(nav.index).astype(float).fillna(cl),
                low_sig_df[ck].reindex(nav.index).astype(float).fillna(cl),
                cl,
                window=int(getattr(inp, "vol_ratio_fast_atr_window", 5) or 5),
            ).astype(float)
            atr_slow = _atr_from_hlc(
                high_sig_df[ck].reindex(nav.index).astype(float).fillna(cl),
                low_sig_df[ck].reindex(nav.index).astype(float).fillna(cl),
                cl,
                window=int(getattr(inp, "vol_ratio_slow_atr_window", 50) or 50),
            ).astype(float)
            vol_ratio = (atr_fast / atr_slow.replace(0.0, np.nan)).astype(float)
            impulse_state = _compute_impulse_state(
                cl,
                ema_window=13,
                macd_fast=12,
                macd_slow=26,
                macd_signal=9,
            )
            condition_bins_by_code[ck] = {
                "momentum": _bucketize_momentum_series(
                    mom_for_entry.reindex(nav.index)
                ),
                "er": _bucketize_er_series(er_for_entry.reindex(nav.index)),
                "vol_ratio": _bucketize_vol_ratio_series(vol_ratio.reindex(nav.index)),
                "impulse": _bucketize_impulse_series(
                    (
                        impulse_state
                        if impulse_state is not None
                        else pd.Series(index=nav.index, dtype=object)
                    ).reindex(nav.index)
                ),
            }
        trades_with_r = _attach_entry_condition_bins_to_trades(
            trades_with_r,
            condition_bins_by_code=condition_bins_by_code,
            dates=nav.index,
            default_code=None,
        )
    mfe_r_distribution = build_trade_mfe_r_distribution(
        trade_pack.get("trades", []),
        close=close_sig_df.reindex(index=nav.index, columns=w_eff.columns)
        .astype(float)
        .ffill(),
        high=high_sig_df.reindex(index=nav.index, columns=w_eff.columns)
        .astype(float)
        .ffill(),
        atr=atr_risk_df.reindex(index=nav.index, columns=w_eff.columns).astype(float),
        atr_mult=float(getattr(inp, "atr_stop_n", 2.0) or 2.0),
        default_code=None,
    )

    er_blocked = sum(
        int(
            (signal_debug_by_code.get(c, {}).get("er_filter") or {}).get(
                "blocked_entry_count", 0
            )
        )
        for c in signal_debug_by_code
    )
    er_attempted = sum(
        int(
            (signal_debug_by_code.get(c, {}).get("er_filter") or {}).get(
                "attempted_entry_count", 0
            )
        )
        for c in signal_debug_by_code
    )
    er_allowed = sum(
        int(
            (signal_debug_by_code.get(c, {}).get("er_filter") or {}).get(
                "allowed_entry_count", 0
            )
        )
        for c in signal_debug_by_code
    )
    imp_blocked = sum(
        int(
            (signal_debug_by_code.get(c, {}).get("impulse_filter") or {}).get(
                "blocked_entry_count", 0
            )
        )
        for c in signal_debug_by_code
    )
    imp_attempted = sum(
        int(
            (signal_debug_by_code.get(c, {}).get("impulse_filter") or {}).get(
                "attempted_entry_count", 0
            )
        )
        for c in signal_debug_by_code
    )
    imp_allowed = sum(
        int(
            (signal_debug_by_code.get(c, {}).get("impulse_filter") or {}).get(
                "allowed_entry_count", 0
            )
        )
        for c in signal_debug_by_code
    )
    imp_bull = sum(
        int(
            (signal_debug_by_code.get(c, {}).get("impulse_filter") or {}).get(
                "blocked_entry_count_bull", 0
            )
        )
        for c in signal_debug_by_code
    )
    imp_bear = sum(
        int(
            (signal_debug_by_code.get(c, {}).get("impulse_filter") or {}).get(
                "blocked_entry_count_bear", 0
            )
        )
        for c in signal_debug_by_code
    )
    imp_neutral = sum(
        int(
            (signal_debug_by_code.get(c, {}).get("impulse_filter") or {}).get(
                "blocked_entry_count_neutral", 0
            )
        )
        for c in signal_debug_by_code
    )
    er_exit_trigger = sum(
        int(
            (signal_debug_by_code.get(c, {}).get("er_exit_filter") or {}).get(
                "trigger_count", 0
            )
        )
        for c in signal_debug_by_code
    )
    vol_adj_total = sum(
        int(
            (semantic_debug_by_code.get(c, {}).get("vol_risk_adjust") or {}).get(
                "vol_risk_adjust_total_count", 0
            )
        )
        for c in semantic_debug_by_code
    )
    atr_trigger_total = sum(
        int(
            (semantic_debug_by_code.get(c, {}).get("atr_stop") or {}).get(
                "trigger_count", 0
            )
        )
        for c in semantic_debug_by_code
    )
    rtp_trigger_total = sum(
        int(
            (semantic_debug_by_code.get(c, {}).get("r_take_profit") or {}).get(
                "trigger_count", 0
            )
        )
        for c in semantic_debug_by_code
    )
    bias_v_tp_trigger_total = sum(
        int(
            (semantic_debug_by_code.get(c, {}).get("bias_v_take_profit") or {}).get(
                "trigger_count", 0
            )
        )
        for c in semantic_debug_by_code
    )
    rtp_tier_counts: dict[str, int] = {}
    for c in semantic_debug_by_code:
        tiers = (semantic_debug_by_code.get(c, {}).get("r_take_profit") or {}).get(
            "tier_trigger_counts"
        ) or {}
        for k, v in dict(tiers).items():
            kk = str(k)
            rtp_tier_counts[kk] = int(rtp_tier_counts.get(kk, 0) + int(v))
    if (
        not bool(getattr(inp, "monthly_risk_budget_enabled", False))
    ) and semantic_debug_by_code:
        monthly_attempted_total = sum(
            int(
                (
                    semantic_debug_by_code.get(c, {}).get("monthly_risk_budget_gate")
                    or {}
                ).get("attempted_entry_count", 0)
            )
            for c in semantic_debug_by_code
        )
        monthly_blocked_total = sum(
            int(
                (
                    semantic_debug_by_code.get(c, {}).get("monthly_risk_budget_gate")
                    or {}
                ).get("blocked_entry_count", 0)
            )
            for c in semantic_debug_by_code
        )
    imp_rate = float(imp_blocked / imp_attempted) if imp_attempted > 0 else 0.0
    monthly_rate = (
        float(monthly_blocked_total / monthly_attempted_total)
        if monthly_attempted_total > 0
        else 0.0
    )
    overall_stats = {
        **_trade_stats_from_returns(trade_pack.get("returns", [])),
        "n": len(trades),
        "atr_stop_trigger_count": int(atr_trigger_total),
        "r_take_profit_trigger_count": int(rtp_trigger_total),
        "bias_v_take_profit_trigger_count": int(bias_v_tp_trigger_total),
        "r_take_profit_tier_trigger_counts": dict(rtp_tier_counts),
        "er_filter_blocked_entry_count": int(er_blocked),
        "er_filter_attempted_entry_count": int(er_attempted),
        "er_filter_allowed_entry_count": int(er_allowed),
        "impulse_filter_blocked_entry_count": int(imp_blocked),
        "impulse_filter_attempted_entry_count": int(imp_attempted),
        "impulse_filter_allowed_entry_count": int(imp_allowed),
        "impulse_filter_blocked_entry_rate": float(imp_rate),
        "impulse_filter_blocked_entry_count_bull": int(imp_bull),
        "impulse_filter_blocked_entry_count_bear": int(imp_bear),
        "impulse_filter_blocked_entry_count_neutral": int(imp_neutral),
        "er_exit_filter_trigger_count": int(er_exit_trigger),
        "vol_risk_adjust_total_count": int(vol_adj_total),
        "vol_risk_adjust_reduce_on_expand_count": int(
            sum(
                int(
                    (
                        (semantic_debug_by_code.get(str(c), {}) or {}).get(
                            "vol_risk_adjust"
                        )
                        or {}
                    ).get("vol_risk_adjust_reduce_on_expand_count", 0)
                )
                for c in wdf.columns
            )
        ),
        "vol_risk_adjust_increase_on_contract_count": int(
            sum(
                int(
                    (
                        (semantic_debug_by_code.get(str(c), {}) or {}).get(
                            "vol_risk_adjust"
                        )
                        or {}
                    ).get("vol_risk_adjust_increase_on_contract_count", 0)
                )
                for c in wdf.columns
            )
        ),
        "vol_risk_adjust_recover_from_expand_count": int(
            sum(
                int(
                    (
                        (semantic_debug_by_code.get(str(c), {}) or {}).get(
                            "vol_risk_adjust"
                        )
                        or {}
                    ).get("vol_risk_adjust_recover_from_expand_count", 0)
                )
                for c in wdf.columns
            )
        ),
        "vol_risk_adjust_recover_from_contract_count": int(
            sum(
                int(
                    (
                        (semantic_debug_by_code.get(str(c), {}) or {}).get(
                            "vol_risk_adjust"
                        )
                        or {}
                    ).get("vol_risk_adjust_recover_from_contract_count", 0)
                )
                for c in wdf.columns
            )
        ),
        "vol_risk_entry_state_reduce_on_expand_count": int(
            sum(
                int(
                    (
                        (semantic_debug_by_code.get(str(c), {}) or {}).get(
                            "vol_risk_adjust"
                        )
                        or {}
                    ).get("vol_risk_entry_state_reduce_on_expand_count", 0)
                )
                for c in wdf.columns
            )
        ),
        "vol_risk_entry_state_increase_on_contract_count": int(
            sum(
                int(
                    (
                        (semantic_debug_by_code.get(str(c), {}) or {}).get(
                            "vol_risk_adjust"
                        )
                        or {}
                    ).get("vol_risk_entry_state_increase_on_contract_count", 0)
                )
                for c in wdf.columns
            )
        ),
        "monthly_risk_budget_attempted_entry_count": int(monthly_attempted_total),
        "monthly_risk_budget_blocked_entry_count": int(monthly_blocked_total),
        "monthly_risk_budget_blocked_entry_rate": float(monthly_rate),
        "vol_risk_overcap_scale_count": int(overcap_scale_total),
        "vol_risk_overcap_skip_entry_decision_count": int(overcap_skip_decision_total),
        "vol_risk_overcap_skip_entry_episode_count": int(overcap_skip_episode_total),
        "vol_risk_overcap_replace_entry_count": int(overcap_replace_total),
        "vol_risk_overcap_replace_out_count": int(overcap_replace_total),
        "vol_risk_overcap_replace_in_count": int(overcap_replace_total),
        "vol_risk_overcap_leverage_usage_count": int(overcap_leverage_usage_total),
        "vol_risk_overcap_leverage_max_multiple": float(overcap_leverage_max_multiple),
    }
    by_code_stats: dict[str, dict[str, Any]] = {}
    for c in wdf.columns:
        d = signal_debug_by_code.get(c, {})
        er_stats = d.get("er_filter") or {}
        imp_stats = d.get("impulse_filter") or {}
        er_exit_stats = d.get("er_exit_filter") or {}
        sem = semantic_debug_by_code.get(c, {})
        one_imp_attempted = int(imp_stats.get("attempted_entry_count", 0))
        one_imp_blocked = int(imp_stats.get("blocked_entry_count", 0))
        one_month_attempted = int(
            (sem.get("monthly_risk_budget_gate") or {}).get("attempted_entry_count", 0)
        )
        one_month_blocked = int(
            (sem.get("monthly_risk_budget_gate") or {}).get("blocked_entry_count", 0)
        )
        by_code_stats[str(c)] = {
            **_trade_stats_from_returns(
                (trade_pack.get("returns_by_code") or {}).get(str(c), [])
            ),
            "n": int(sum(1 for t in trades if str(t.get("code")) == str(c))),
            "atr_stop_trigger_count": int(
                (sem.get("atr_stop") or {}).get("trigger_count", 0)
            ),
            "r_take_profit_trigger_count": int(
                (sem.get("r_take_profit") or {}).get("trigger_count", 0)
            ),
            "bias_v_take_profit_trigger_count": int(
                (sem.get("bias_v_take_profit") or {}).get("trigger_count", 0)
            ),
            "r_take_profit_tier_trigger_counts": dict(
                (sem.get("r_take_profit") or {}).get("tier_trigger_counts") or {}
            ),
            "er_filter_blocked_entry_count": int(
                er_stats.get("blocked_entry_count", 0)
            ),
            "er_filter_attempted_entry_count": int(
                er_stats.get("attempted_entry_count", 0)
            ),
            "er_filter_allowed_entry_count": int(
                er_stats.get("allowed_entry_count", 0)
            ),
            "impulse_filter_blocked_entry_count": one_imp_blocked,
            "impulse_filter_attempted_entry_count": one_imp_attempted,
            "impulse_filter_allowed_entry_count": int(
                imp_stats.get("allowed_entry_count", 0)
            ),
            "impulse_filter_blocked_entry_rate": (
                float(one_imp_blocked / one_imp_attempted)
                if one_imp_attempted > 0
                else 0.0
            ),
            "impulse_filter_blocked_entry_count_bull": int(
                imp_stats.get("blocked_entry_count_bull", 0)
            ),
            "impulse_filter_blocked_entry_count_bear": int(
                imp_stats.get("blocked_entry_count_bear", 0)
            ),
            "impulse_filter_blocked_entry_count_neutral": int(
                imp_stats.get("blocked_entry_count_neutral", 0)
            ),
            "er_exit_filter_trigger_count": int(er_exit_stats.get("trigger_count", 0)),
            "vol_risk_adjust_total_count": int(
                (sem.get("vol_risk_adjust") or {}).get("vol_risk_adjust_total_count", 0)
            ),
            "vol_risk_adjust_reduce_on_expand_count": int(
                (sem.get("vol_risk_adjust") or {}).get(
                    "vol_risk_adjust_reduce_on_expand_count", 0
                )
            ),
            "vol_risk_adjust_increase_on_contract_count": int(
                (sem.get("vol_risk_adjust") or {}).get(
                    "vol_risk_adjust_increase_on_contract_count", 0
                )
            ),
            "vol_risk_adjust_recover_from_expand_count": int(
                (sem.get("vol_risk_adjust") or {}).get(
                    "vol_risk_adjust_recover_from_expand_count", 0
                )
            ),
            "vol_risk_adjust_recover_from_contract_count": int(
                (sem.get("vol_risk_adjust") or {}).get(
                    "vol_risk_adjust_recover_from_contract_count", 0
                )
            ),
            "vol_risk_entry_state_reduce_on_expand_count": int(
                (sem.get("vol_risk_adjust") or {}).get(
                    "vol_risk_entry_state_reduce_on_expand_count", 0
                )
            ),
            "vol_risk_entry_state_increase_on_contract_count": int(
                (sem.get("vol_risk_adjust") or {}).get(
                    "vol_risk_entry_state_increase_on_contract_count", 0
                )
            ),
            "monthly_risk_budget_attempted_entry_count": one_month_attempted,
            "monthly_risk_budget_blocked_entry_count": one_month_blocked,
            "monthly_risk_budget_blocked_entry_rate": (
                float(one_month_blocked / one_month_attempted)
                if one_month_attempted > 0
                else 0.0
            ),
            "vol_risk_overcap_scale_count": int(overcap_scale_by_code.get(str(c), 0)),
            "vol_risk_overcap_skip_entry_decision_count": int(
                overcap_skip_decision_by_code.get(str(c), 0)
            ),
            "vol_risk_overcap_skip_entry_episode_count": int(
                overcap_skip_episode_by_code.get(str(c), 0)
            ),
            "vol_risk_overcap_replace_out_count": int(
                overcap_replace_out_by_code.get(str(c), 0)
            ),
            "vol_risk_overcap_replace_in_count": int(
                overcap_replace_in_by_code.get(str(c), 0)
            ),
            "vol_risk_overcap_leverage_usage_count": int(
                overcap_leverage_usage_by_code.get(str(c), 0)
            ),
            "vol_risk_overcap_leverage_max_multiple": float(
                overcap_leverage_max_multiple_by_code.get(str(c), 0.0)
            ),
        }
    atr_trigger_dates = sorted(
        {
            str(d)
            for v in atr_stop_by_asset.values()
            for d in list((v or {}).get("trigger_dates") or [])
            if str(d).strip()
        }
    )
    rtp_trigger_dates = sorted(
        {
            str(d)
            for v in rtp_by_asset.values()
            for d in list((v or {}).get("trigger_dates") or [])
            if str(d).strip()
        }
    )
    bias_v_tp_trigger_dates = sorted(
        {
            str(d)
            for v in bias_v_tp_by_asset.values()
            for d in list((v or {}).get("trigger_dates") or [])
            if str(d).strip()
        }
    )
    er_exit_by_asset = {}
    for c in wdf.columns:
        one = dict((signal_debug_by_code.get(c, {}).get("er_exit_filter") or {}))
        er_exit_by_asset[str(c)] = {
            **one,
            "trigger_count": int(one.get("trigger_count", 0)),
            "trigger_dates": [
                str(x) for x in (one.get("trigger_dates") or []) if str(x).strip()
            ],
            "trace_last_rows": list(one.get("trace_last_rows") or []),
        }
    er_exit_trigger_dates = sorted(
        {
            str(d)
            for v in er_exit_by_asset.values()
            for d in list((v or {}).get("trigger_dates") or [])
            if str(d).strip()
        }
    )
    monthly_attempted_by_code: dict[str, int] = {}
    monthly_blocked_by_code: dict[str, int] = {}
    for c in wdf.columns:
        gate = semantic_debug_by_code.get(c, {}).get("monthly_risk_budget_gate") or {}
        one_attempted = int(gate.get("attempted_entry_count", 0))
        one_blocked = int(gate.get("blocked_entry_count", 0))
        monthly_attempted_by_code[str(c)] = one_attempted
        monthly_blocked_by_code[str(c)] = one_blocked
    fixed_ext_dates = sorted(
        {str(e.get("date")) for e in fixed_ext_events if str(e.get("date", "")).strip()}
    )
    fixed_skip_dates = sorted(
        {
            str(e.get("date"))
            for e in fixed_skip_events
            if str(e.get("date", "")).strip()
        }
    )
    fixed_ext_over_weight_count = int(
        sum(1 for e in fixed_ext_events if bool(e.get("over_weight")))
    )
    fixed_ext_over_count_count = int(
        sum(1 for e in fixed_ext_events if bool(e.get("over_count")))
    )
    fixed_ext_over_both_count = int(
        sum(
            1
            for e in fixed_ext_events
            if bool(e.get("over_weight")) and bool(e.get("over_count"))
        )
    )
    fixed_skip_over_weight_count = int(
        sum(1 for e in fixed_skip_events if bool(e.get("over_weight")))
    )
    fixed_skip_over_count_count = int(
        sum(1 for e in fixed_skip_events if bool(e.get("over_count")))
    )
    fixed_skip_over_both_count = int(
        sum(
            1
            for e in fixed_skip_events
            if bool(e.get("over_weight")) and bool(e.get("over_count"))
        )
    )
    vol_risk_adjust_by_asset = {
        str(c): dict((semantic_debug_by_code.get(c, {}).get("vol_risk_adjust") or {}))
        for c in wdf.columns
    }
    bias_v_trace_keys = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "ma",
        "atr",
        "threshold",
        "bias_v",
        "base_pos",
        "decision_pos",
        "tp_triggered",
        "tp_trigger_source",
        "tp_trigger_price_raw",
        "tp_trigger_price_eff",
        "tp_fill_price",
        "stop_trigger_source",
        "stop_fill_price",
        "gap_open_triggered",
        "event_type",
        "event_reason",
        "in_pos_after",
        "wait_next_entry_lock",
    ]
    atr_trade_record_keys = [
        "entry_decision_date",
        "entry_execution_date",
        "trigger_date",
        "entry_execution_price",
        "initial_stop_price",
        "trigger_stop_price",
        "execution_stop_price",
    ]
    atr_trigger_event_keys = [
        "date",
        "stop_price",
        "open_price",
        "low_price",
        "fill_price",
        "trigger_source",
        "gap_open_triggered",
    ]
    bias_v_trade_record_keys = [
        "entry_decision_date",
        "entry_execution_date",
        "trigger_date",
        "entry_execution_price",
        "initial_take_profit_price",
        "trigger_take_profit_price",
        "execution_take_profit_price",
    ]
    bias_v_trigger_event_keys = [
        "date",
        "trigger_price",
        "trigger_price_raw",
        "trigger_price_eff",
        "open_price",
        "high_price",
        "fill_price",
        "trigger_source",
        "gap_open_triggered",
        "bias_v",
        "threshold",
    ]
    rtp_trace_keys = [
        "date",
        "open",
        "high",
        "low",
        "base_pos",
        "decision_pos",
        "entry_price",
        "atr_entry",
        "initial_r_pct",
        "peak_profit_pct",
        "peak_r_multiple",
        "drawdown_from_peak",
        "active_tier_r",
        "active_tier_retrace",
        "tp_triggered",
        "tp_trigger_source",
        "tp_fill_price",
        "stop_trigger_source",
        "stop_fill_price",
        "gap_open_triggered",
        "event_type",
        "event_reason",
        "in_pos_after",
        "wait_next_entry_lock",
    ]
    for c in wdf.columns:
        ck = str(c)
        a = dict(atr_stop_by_asset.get(ck) or {})
        a["trade_records"] = _normalize_trace_rows(
            a.get("trade_records"), atr_trade_record_keys
        )
        a["trigger_events"] = _normalize_trace_rows(
            a.get("trigger_events"), atr_trigger_event_keys
        )
        atr_stop_by_asset[ck] = a
        b = dict(bias_v_tp_by_asset.get(ck) or {})
        b["trace_last_rows"] = _normalize_trace_rows(
            b.get("trace_last_rows"), bias_v_trace_keys
        )
        b["trade_records"] = _normalize_trace_rows(
            b.get("trade_records"), bias_v_trade_record_keys
        )
        b["trigger_events"] = _normalize_trace_rows(
            b.get("trigger_events"), bias_v_trigger_event_keys
        )
        bias_v_tp_by_asset[ck] = b
        r = dict(rtp_by_asset.get(ck) or {})
        r.setdefault("invalid_initial_r_entries", 0)
        r["trace_last_rows"] = _normalize_trace_rows(
            r.get("trace_last_rows"), rtp_trace_keys
        )
        rtp_by_asset[ck] = r

    sample_days = int(len(port_ret))
    complete_trade_count = int(len(trade_pack.get("returns", [])))
    avg_daily_turnover = float(turnover.mean()) if len(turnover) else 0.0
    avg_annual_turnover = float(avg_daily_turnover * 252.0)
    avg_daily_trade_count = (
        float(complete_trade_count / sample_days) if sample_days > 0 else 0.0
    )
    avg_annual_trade_count = float(avg_daily_trade_count * 252.0)
    quick_mode = bool(getattr(inp, "quick_mode", False))
    trades_by_code = {
        str(c): [t for t in trades_with_r if str(t.get("code")) == str(c)]
        for c in wdf.columns
    }
    if quick_mode:
        trades = []
        trades_by_code = {str(c): [] for c in wdf.columns}
    entry_exec_price_with_slippage_by_asset: dict[str, float] = {}
    for c in w_eff.columns:
        one = _latest_entry_exec_price_with_slippage(
            effective_weight=w_eff[c].reindex(nav.index).astype(float),
            exec_price_series=px_exec_slip_df[c]
            .reindex(nav.index)
            .ffill()
            .astype(float),
            slippage_spread=float(getattr(inp, "slippage_rate", 0.0) or 0.0),
        )
        if one is not None:
            entry_exec_price_with_slippage_by_asset[str(c)] = float(one)

    out = {
        "meta": {
            "type": "trend_portfolio_backtest",
            "engine": "bt",
            "runtime_engine": runtime_engine
            if "runtime_engine" in locals()
            else "unknown",
            "start": inp.start.strftime("%Y%m%d"),
            "end": inp.end.strftime("%Y%m%d"),
            "strategy": strat,
            "codes": list(nav_map.keys()),
            "failed_codes": failures,
            "strategy_execution_description": TREND_STRATEGY_EXECUTION_DESCRIPTIONS.get(
                strat, ""
            ),
            "params": _build_meta_params(inp),
            "limitations": [],
        },
        "nav": {
            "dates": [d.strftime("%Y-%m-%d") for d in nav.index],
            "series": {
                "STRAT": [float(x) for x in nav.values],
                "BUY_HOLD_EW": [float(x) for x in bh_nav.values],
                "BUY_HOLD": [float(x) for x in bh_nav.values],
                "EXCESS": [float(x) for x in excess_nav.values],
            },
        },
        "weights": {
            "dates": [d.strftime("%Y-%m-%d") for d in nav.index],
            "series": {c: [float(x) for x in w_eff[c].values] for c in w_eff.columns},
        },
        "weights_decision": {
            "dates": [d.strftime("%Y-%m-%d") for d in wdf.index],
            "series": {c: [float(x) for x in wdf[c].values] for c in wdf.columns},
        },
        "asset_nav_exec": {
            "dates": [d.strftime("%Y-%m-%d") for d in ret_exec_df.index],
            "series": {
                c: [
                    float(x)
                    for x in (1.0 + ret_exec_df[c].astype(float)).cumprod().values
                ]
                for c in ret_exec_df.columns
            },
        },
        "signals": {
            "dates": [d.strftime("%Y-%m-%d") for d in nav.index],
            "position_effective": [float(x) for x in (w_eff > 0.0).mean(axis=1).values],
        },
        "metrics": {
            "strategy": {
                **_metrics_from_ret(port_ret, float(inp.risk_free_rate)),
                "avg_daily_turnover": float(avg_daily_turnover),
                "avg_annual_turnover": float(avg_annual_turnover),
                "avg_annual_turnover_rate": float(avg_annual_turnover),
                "avg_daily_trade_count": float(avg_daily_trade_count),
                "avg_annual_trade_count": float(avg_annual_trade_count),
                "r_take_profit_tier_trigger_counts": dict(rtp_tier_counts),
                "r_take_profit_trigger_count": int(rtp_trigger_total),
                "bias_v_take_profit_trigger_count": int(bias_v_tp_trigger_total),
                "atr_stop_trigger_count": int(atr_trigger_total),
                "impulse_filter_blocked_entry_count": int(imp_blocked),
                "impulse_filter_blocked_entry_count_bull": int(imp_bull),
                "impulse_filter_blocked_entry_count_bear": int(imp_bear),
                "impulse_filter_blocked_entry_count_neutral": int(imp_neutral),
                "vol_risk_overcap_scale_count": int(overcap_scale_total),
                "vol_risk_overcap_skip_entry_decision_count": int(
                    overcap_skip_decision_total
                ),
                "vol_risk_overcap_skip_entry_episode_count": int(
                    overcap_skip_episode_total
                ),
                "vol_risk_overcap_replace_entry_count": int(overcap_replace_total),
                "vol_risk_overcap_leverage_usage_count": int(
                    overcap_leverage_usage_total
                ),
                "vol_risk_overcap_leverage_max_multiple": float(
                    overcap_leverage_max_multiple
                ),
                "monthly_risk_budget_blocked_entry_count": int(monthly_blocked_total),
            },
            "benchmark": _metrics_from_ret(bh_ret, float(inp.risk_free_rate)),
            "excess": {
                **_metrics_from_ret(excess_ret, float(inp.risk_free_rate)),
                "information_ratio": float(_information_ratio(active_ret)),
            },
        },
        "period_returns": {
            "weekly": _period_returns(nav, "W-FRI"),
            "monthly": _period_returns(nav, "ME"),
            "quarterly": _period_returns(nav, "QE"),
            "yearly": _period_returns(nav, "YE"),
        },
        "rolling": _rolling_pack(nav),
        "attribution": attribution,
        "trade_statistics": {
            "all": {"n": len(trades)},
            "overall": overall_stats,
            "by_code": by_code_stats,
            "trades": ([] if quick_mode else trades_with_r),
            "trades_by_code": trades_by_code,
            "mfe_r_distribution": mfe_r_distribution,
        },
        "r_statistics": r_stats_out,
        "trades": ([] if quick_mode else trades_with_r),
        "next_plan": {
            "decision_date": (str(nav.index[-1].date()) if len(nav.index) else None),
            "entry_exec_price_with_slippage_by_asset": entry_exec_price_with_slippage_by_asset,
        },
        "risk_controls": {
            "vol_regime_risk_mgmt": {
                "enabled": bool(getattr(inp, "vol_regime_risk_mgmt_enabled", False)),
                "fast_atr_window": int(
                    getattr(inp, "vol_ratio_fast_atr_window", 5) or 5
                ),
                "slow_atr_window": int(
                    getattr(inp, "vol_ratio_slow_atr_window", 50) or 50
                ),
                "expand_threshold": float(
                    getattr(inp, "vol_ratio_expand_threshold", 1.45) or 1.45
                ),
                "contract_threshold": float(
                    getattr(inp, "vol_ratio_contract_threshold", 0.65) or 0.65
                ),
                "normal_threshold": float(
                    getattr(inp, "vol_ratio_normal_threshold", 1.05) or 1.05
                ),
                "adjust_total_count": int(
                    overall_stats.get("vol_risk_adjust_total_count", 0)
                ),
                "adjust_reduce_on_expand_count": int(
                    overall_stats.get("vol_risk_adjust_reduce_on_expand_count", 0)
                ),
                "adjust_increase_on_contract_count": int(
                    overall_stats.get("vol_risk_adjust_increase_on_contract_count", 0)
                ),
                "adjust_recover_from_expand_count": int(
                    overall_stats.get("vol_risk_adjust_recover_from_expand_count", 0)
                ),
                "adjust_recover_from_contract_count": int(
                    overall_stats.get("vol_risk_adjust_recover_from_contract_count", 0)
                ),
                "entry_state_reduce_on_expand_count": int(
                    overall_stats.get("vol_risk_entry_state_reduce_on_expand_count", 0)
                ),
                "entry_state_increase_on_contract_count": int(
                    overall_stats.get(
                        "vol_risk_entry_state_increase_on_contract_count", 0
                    )
                ),
                "overcap_policy": str(
                    getattr(inp, "risk_budget_overcap_policy", "scale") or "scale"
                ),
                "overcap_max_leverage_multiple": float(
                    getattr(inp, "risk_budget_max_leverage_multiple", 2.0) or 2.0
                ),
                "overcap_scale_count": int(overcap_scale_total),
                "overcap_skip_entry_decision_count": int(overcap_skip_decision_total),
                "overcap_skip_entry_episode_count": int(overcap_skip_episode_total),
                "overcap_skip_entry_decision_count_by_code": dict(
                    overcap_skip_decision_by_code
                ),
                "overcap_skip_entry_episode_count_by_code": dict(
                    overcap_skip_episode_by_code
                ),
                "overcap_replace_entry_count": int(overcap_replace_total),
                "overcap_replace_out_count_by_code": dict(overcap_replace_out_by_code),
                "overcap_replace_in_count_by_code": dict(overcap_replace_in_by_code),
                "overcap_leverage_usage_count": int(overcap_leverage_usage_total),
                "overcap_leverage_max_multiple": float(overcap_leverage_max_multiple),
                "overcap_leverage_usage_count_by_code": dict(
                    overcap_leverage_usage_by_code
                ),
                "overcap_leverage_max_multiple_by_code": dict(
                    overcap_leverage_max_multiple_by_code
                ),
                "overcap_daily_counts": [
                    {
                        "date": str(k),
                        "scale": int((v or {}).get("scale", 0)),
                        "skip_entry": int((v or {}).get("skip_entry", 0)),
                        "replace_entry": int((v or {}).get("replace_entry", 0)),
                        "leverage_entry": int((v or {}).get("leverage_entry", 0)),
                        "leverage_multiple_max": float(
                            (v or {}).get("leverage_multiple_max", 0.0) or 0.0
                        ),
                    }
                    for k, v in sorted(
                        (risk_budget_overcap_daily_counts or {}).items(),
                        key=lambda x: str(x[0]),
                    )
                ],
                "by_asset": dict(vol_risk_adjust_by_asset),
            },
            "atr_stop": {
                "enabled": bool(
                    str(getattr(inp, "atr_stop_mode", "none") or "none").strip().lower()
                    != "none"
                ),
                "mode": str(getattr(inp, "atr_stop_mode", "none") or "none"),
                "atr_basis": str(
                    getattr(inp, "atr_stop_atr_basis", "latest") or "latest"
                ),
                "reentry_mode": str(
                    getattr(inp, "atr_stop_reentry_mode", "reenter") or "reenter"
                ),
                "trigger_count": int(atr_trigger_total),
                "trigger_days": int(len(atr_trigger_dates)),
                "first_trigger_date": (
                    atr_trigger_dates[0] if atr_trigger_dates else None
                ),
                "last_trigger_date": (
                    atr_trigger_dates[-1] if atr_trigger_dates else None
                ),
                "trigger_dates": atr_trigger_dates[:200],
                "by_asset": atr_stop_by_asset,
            },
            "r_take_profit": {
                "enabled": bool(getattr(inp, "r_take_profit_enabled", False)),
                "reentry_mode": str(
                    getattr(inp, "r_take_profit_reentry_mode", "reenter") or "reenter"
                ),
                "tiers": _normalize_r_take_profit_tiers(
                    getattr(inp, "r_take_profit_tiers", None)
                ),
                "trigger_count": int(rtp_trigger_total),
                "tier_trigger_counts": dict(rtp_tier_counts),
                "trigger_days": int(len(rtp_trigger_dates)),
                "first_trigger_date": (
                    rtp_trigger_dates[0] if rtp_trigger_dates else None
                ),
                "last_trigger_date": (
                    rtp_trigger_dates[-1] if rtp_trigger_dates else None
                ),
                "trigger_dates": rtp_trigger_dates[:200],
                "fallback_mode_used": bool(
                    str(getattr(inp, "atr_stop_mode", "none") or "none").strip().lower()
                    == "none"
                    and bool(getattr(inp, "r_take_profit_enabled", False))
                ),
                "initial_r_mode": (
                    "atr_stop"
                    if str(getattr(inp, "atr_stop_mode", "none") or "none")
                    .strip()
                    .lower()
                    != "none"
                    else "virtual_atr_fallback"
                ),
                "by_asset": rtp_by_asset,
            },
            "bias_v_take_profit": {
                "enabled": bool(getattr(inp, "bias_v_take_profit_enabled", False)),
                "reentry_mode": str(
                    getattr(inp, "bias_v_take_profit_reentry_mode", "reenter")
                    or "reenter"
                ),
                "ma_window": int(getattr(inp, "bias_v_ma_window", 20) or 20),
                "atr_window": int(getattr(inp, "bias_v_atr_window", 20) or 20),
                "threshold": float(
                    getattr(inp, "bias_v_take_profit_threshold", 5.0) or 5.0
                ),
                "trigger_count": int(bias_v_tp_trigger_total),
                "trigger_days": int(len(bias_v_tp_trigger_dates)),
                "first_trigger_date": (
                    bias_v_tp_trigger_dates[0] if bias_v_tp_trigger_dates else None
                ),
                "last_trigger_date": (
                    bias_v_tp_trigger_dates[-1] if bias_v_tp_trigger_dates else None
                ),
                "trigger_dates": bias_v_tp_trigger_dates[:200],
                "by_asset": bias_v_tp_by_asset,
            },
            "er_exit_filter": {
                "enabled": bool(getattr(inp, "er_exit_filter", False)),
                "window": int(getattr(inp, "er_exit_window", 10) or 10),
                "threshold": float(getattr(inp, "er_exit_threshold", 0.88) or 0.88),
                "trigger_count": int(er_exit_trigger),
                "trigger_days": int(len(er_exit_trigger_dates)),
                "first_trigger_date": (
                    er_exit_trigger_dates[0] if er_exit_trigger_dates else None
                ),
                "last_trigger_date": (
                    er_exit_trigger_dates[-1] if er_exit_trigger_dates else None
                ),
                "trigger_dates": er_exit_trigger_dates[:200],
                "by_asset": er_exit_by_asset,
            },
            "monthly_risk_budget_gate": {
                "enabled": bool(getattr(inp, "monthly_risk_budget_enabled", False)),
                "budget_pct": float(
                    getattr(inp, "monthly_risk_budget_pct", 0.06) or 0.06
                ),
                "include_new_trade_risk": bool(
                    getattr(inp, "monthly_risk_budget_include_new_trade_risk", False)
                ),
                "attempted_entry_count": int(monthly_attempted_total),
                "attempted_entry_count_by_code": dict(monthly_attempted_by_code),
                "blocked_entry_count": int(monthly_blocked_total),
                "blocked_entry_count_by_code": dict(monthly_blocked_by_code),
            },
            "monthly_risk_budget": {
                "enabled": bool(getattr(inp, "monthly_risk_budget_enabled", False)),
                "budget_pct": float(
                    getattr(inp, "monthly_risk_budget_pct", 0.06) or 0.06
                ),
                "include_new_trade_risk": bool(
                    getattr(inp, "monthly_risk_budget_include_new_trade_risk", False)
                ),
                "attempted_entry_count": int(monthly_attempted_total),
                "attempted_entry_count_by_code": dict(monthly_attempted_by_code),
                "blocked_entry_count": int(monthly_blocked_total),
                "blocked_entry_count_by_code": dict(monthly_blocked_by_code),
            },
            "group_filter": {
                "enabled": bool(getattr(inp, "group_enforce", False)),
                "policy": str(
                    getattr(inp, "group_pick_policy", "highest_sharpe")
                    or "highest_sharpe"
                ),
                "max_holdings_per_group": int(
                    getattr(inp, "group_max_holdings", 4) or 4
                ),
                "decision_segments_with_group_filter": int(
                    group_filter_enabled_segments
                ),
                "decision_segments_effective": int(group_filter_effective_segments),
            },
            "position_extension": {
                "enabled": bool(
                    ps == "fixed_ratio"
                    and str(getattr(inp, "fixed_overcap_policy", "extend") or "extend")
                    == "extend"
                ),
                "position_sizing": str(ps),
                "fixed_pos_ratio": float(getattr(inp, "fixed_pos_ratio", 0.04) or 0.04),
                "overcap_policy": str(
                    getattr(inp, "fixed_overcap_policy", "extend") or "extend"
                ),
                "fixed_max_holdings": int(getattr(inp, "fixed_max_holdings", 10) or 10),
                "extension_count": int(len(fixed_ext_events)),
                "extension_over_weight_count": int(fixed_ext_over_weight_count),
                "extension_over_count_count": int(fixed_ext_over_count_count),
                "extension_over_both_count": int(fixed_ext_over_both_count),
                "extension_days": int(len(fixed_ext_dates)),
                "first_extension_date": (
                    fixed_ext_dates[0] if fixed_ext_dates else None
                ),
                "last_extension_date": (
                    fixed_ext_dates[-1] if fixed_ext_dates else None
                ),
                "extension_dates": fixed_ext_dates[:200],
                "extensions": fixed_ext_events[:200],
                "skipped_count": int(len(fixed_skip_events)),
                "skipped_over_weight_count": int(fixed_skip_over_weight_count),
                "skipped_over_count_count": int(fixed_skip_over_count_count),
                "skipped_over_both_count": int(fixed_skip_over_both_count),
                "skipped_days": int(len(fixed_skip_dates)),
                "first_skipped_date": (
                    fixed_skip_dates[0] if fixed_skip_dates else None
                ),
                "last_skipped_date": (
                    fixed_skip_dates[-1] if fixed_skip_dates else None
                ),
                "skipped_dates": fixed_skip_dates[:200],
                "skipped": fixed_skip_events[:200],
            },
            "position_usage": {
                "enabled": bool(ps in {"fixed_ratio", "risk_budget"}),
                "position_sizing": str(ps),
                "cash_as_residual": True,
                "min_exposure": (
                    float(w_eff.sum(axis=1).min()) if len(w_eff) else float("nan")
                ),
                "max_exposure": (
                    float(w_eff.sum(axis=1).max()) if len(w_eff) else float("nan")
                ),
                "mean_exposure": (
                    float(w_eff.sum(axis=1).mean()) if len(w_eff) else float("nan")
                ),
                "quantiles": {
                    "p05": (
                        float(w_eff.sum(axis=1).quantile(0.05))
                        if len(w_eff)
                        else float("nan")
                    ),
                    "p25": (
                        float(w_eff.sum(axis=1).quantile(0.25))
                        if len(w_eff)
                        else float("nan")
                    ),
                    "p50": (
                        float(w_eff.sum(axis=1).quantile(0.50))
                        if len(w_eff)
                        else float("nan")
                    ),
                    "p75": (
                        float(w_eff.sum(axis=1).quantile(0.75))
                        if len(w_eff)
                        else float("nan")
                    ),
                    "p95": (
                        float(w_eff.sum(axis=1).quantile(0.95))
                        if len(w_eff)
                        else float("nan")
                    ),
                },
                "over_100pct_days": int((w_eff.sum(axis=1) > 1.0 + 1e-12).sum())
                if len(w_eff)
                else 0,
                "under_100pct_days": int((w_eff.sum(axis=1) < 1.0 - 1e-12).sum())
                if len(w_eff)
                else 0,
            },
        },
        "return_decomposition": return_decomposition,
        "event_study": event_study,
        "market_regime": market_regime,
        "holdings": holdings,
        "corporate_actions": sorted(
            corporate_actions_rows,
            key=lambda x: (str(x.get("date")), str(x.get("code"))),
        )[:200],
    }
    if not quick_mode:
        out["trade_statistics"]["entry_condition_stats"] = {
            "scope": "closed_trades_only",
            "signal_day_basis": "signal_day_before_entry_execution",
            "quasi_causal_method": "uplift + two_proportion_z / welch_t_normal_approx + BH",
            "strong_causal_method": "uplift + stratified_permutation + BH",
            "overall": _build_entry_condition_stats(
                trades_with_r, by_code=False, n_perm=300, seed=20260410
            ),
            "by_code": {
                str(c): _build_entry_condition_stats(
                    trades_by_code.get(str(c), []),
                    by_code=True,
                    n_perm=200,
                    seed=20260410,
                )
                for c in wdf.columns
            },
        }
    return out
