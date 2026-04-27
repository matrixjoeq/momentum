from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


RANGE_STATE_RANGE = "RANGE"
RANGE_STATE_TREND = "TREND"
RANGE_STATE_TRANSITION = "TRANSITION"
RANGE_STATE_UNKNOWN = "UNKNOWN"
_STATE_ORDER = [
    RANGE_STATE_RANGE,
    RANGE_STATE_TREND,
    RANGE_STATE_TRANSITION,
    RANGE_STATE_UNKNOWN,
]


@dataclass(frozen=True)
class RangeStateConfig:
    mode: str = "adx"
    window: int = 14
    enter_threshold: float = 20.0
    exit_threshold: float = 25.0


def _wilder_mean(x: pd.Series, *, window: int) -> pd.Series:
    return x.ewm(alpha=1.0 / float(window), adjust=False, min_periods=window).mean()


def _compute_er(close: pd.Series, *, window: int) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    direction = (c - c.shift(window)).abs()
    volatility = c.diff().abs().rolling(window=window, min_periods=window).sum()
    er = direction / volatility.replace(0.0, np.nan)
    return er.clip(lower=0.0, upper=1.0)


def _compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, *, window: int
) -> pd.Series:
    h = pd.to_numeric(high, errors="coerce")
    low_s = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")

    up_move = h.diff()
    down_move = -low_s.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=h.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=h.index
    )

    prev_c = c.shift(1)
    tr = pd.concat(
        [(h - low_s).abs(), (h - prev_c).abs(), (low_s - prev_c).abs()], axis=1
    ).max(axis=1)
    atr = _wilder_mean(tr, window=window)
    plus_di = 100.0 * _wilder_mean(plus_dm, window=window) / atr.replace(0.0, np.nan)
    minus_di = 100.0 * _wilder_mean(minus_dm, window=window) / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return _wilder_mean(dx, window=window).clip(lower=0.0, upper=100.0)


def _compute_adxr(
    high: pd.Series, low: pd.Series, close: pd.Series, *, window: int
) -> pd.Series:
    adx = _compute_adx(high, low, close, window=window)
    return ((adx + adx.shift(window)) / 2.0).clip(lower=0.0, upper=100.0)


def _raw_state(v: float, *, enter: float, exit_: float) -> str:
    if not np.isfinite(v):
        return RANGE_STATE_UNKNOWN
    if v <= enter:
        return RANGE_STATE_RANGE
    if v >= exit_:
        return RANGE_STATE_TREND
    return RANGE_STATE_TRANSITION


def _hysteresis_state(
    metric: pd.Series, *, enter: float, exit_: float
) -> tuple[pd.Series, pd.Series]:
    out_state: list[str] = []
    out_raw: list[str] = []
    prev = RANGE_STATE_UNKNOWN
    for x in metric.to_numpy(dtype=float):
        raw = _raw_state(x, enter=enter, exit_=exit_)
        out_raw.append(raw)
        if raw == RANGE_STATE_UNKNOWN:
            out_state.append(RANGE_STATE_UNKNOWN)
            prev = RANGE_STATE_UNKNOWN
            continue
        if prev == RANGE_STATE_UNKNOWN:
            if raw == RANGE_STATE_TRANSITION:
                state = RANGE_STATE_UNKNOWN
            else:
                state = raw
        elif prev == RANGE_STATE_RANGE:
            state = RANGE_STATE_TREND if x >= exit_ else RANGE_STATE_RANGE
        elif prev == RANGE_STATE_TREND:
            state = RANGE_STATE_RANGE if x <= enter else RANGE_STATE_TREND
        else:
            state = raw if raw != RANGE_STATE_TRANSITION else RANGE_STATE_UNKNOWN
        out_state.append(state)
        prev = state
    return (
        pd.Series(out_state, index=metric.index, dtype="object"),
        pd.Series(out_raw, index=metric.index, dtype="object"),
    )


def _range_score(metric: pd.Series, *, enter: float, exit_: float) -> pd.Series:
    x = pd.to_numeric(metric, errors="coerce")
    if exit_ <= enter:
        score = pd.Series(np.where(x <= enter, 1.0, 0.0), index=x.index, dtype=float)
    else:
        score = (exit_ - x) / (exit_ - enter)
    return score.clip(lower=0.0, upper=1.0)


def _build_summary(state: pd.Series) -> dict[str, float | int]:
    n = int(state.notna().sum())
    if n <= 0:
        return {
            "n": 0,
            "range_days": 0,
            "trend_days": 0,
            "transition_days": 0,
            "unknown_days": 0,
            "range_ratio": float("nan"),
        }
    cnt = state.value_counts(dropna=False)
    range_days = int(cnt.get(RANGE_STATE_RANGE, 0))
    trend_days = int(cnt.get(RANGE_STATE_TREND, 0))
    transition_days = int(cnt.get(RANGE_STATE_TRANSITION, 0))
    unknown_days = int(cnt.get(RANGE_STATE_UNKNOWN, 0))
    denom = max(1, range_days + trend_days + transition_days + unknown_days)
    return {
        "n": n,
        "range_days": range_days,
        "trend_days": trend_days,
        "transition_days": transition_days,
        "unknown_days": unknown_days,
        "range_ratio": float(range_days / float(denom)),
    }


def compute_range_state_monitor(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    config: RangeStateConfig,
) -> dict:
    idx = close.index
    mode = str(config.mode or "adx").strip().lower()
    window = int(config.window)
    enter = float(config.enter_threshold)
    exit_ = float(config.exit_threshold)

    if mode == "er":
        metric = _compute_er(close, window=window)
    elif mode == "adxr":
        metric = _compute_adxr(high, low, close, window=window)
    else:
        metric = _compute_adx(high, low, close, window=window)
    metric = metric.reindex(idx)

    state_hys, state_raw = _hysteresis_state(metric, enter=enter, exit_=exit_)
    score = _range_score(metric, enter=enter, exit_=exit_)
    subscore_band = 1.0 - (
        (metric - (enter + exit_) / 2.0).abs() / max(1e-12, exit_ - enter)
    )
    subscore_band = subscore_band.clip(lower=0.0, upper=1.0)

    last_valid_idx = metric.last_valid_index()
    latest = None
    if last_valid_idx is not None:
        i = int(idx.get_loc(last_valid_idx))
        latest = {
            "date": str(pd.Timestamp(last_valid_idx).date()),
            "mode": mode,
            "metric": None
            if not np.isfinite(metric.iloc[i])
            else float(metric.iloc[i]),
            "range_score": None
            if not np.isfinite(score.iloc[i])
            else float(score.iloc[i]),
            "subscore_band": (
                None
                if not np.isfinite(subscore_band.iloc[i])
                else float(subscore_band.iloc[i])
            ),
            "raw_state": str(state_raw.iloc[i]),
            "state": str(state_hys.iloc[i]),
            "threshold_enter": enter,
            "threshold_exit": exit_,
            "window": window,
        }

    date_list = [str(pd.Timestamp(x).date()) for x in idx]
    metric_list = [
        None if not np.isfinite(v) else float(v)
        for v in metric.to_numpy(dtype=float, na_value=np.nan)
    ]
    score_list = [
        None if not np.isfinite(v) else float(v)
        for v in score.to_numpy(dtype=float, na_value=np.nan)
    ]
    subscore_band_list = [
        None if not np.isfinite(v) else float(v)
        for v in subscore_band.to_numpy(dtype=float, na_value=np.nan)
    ]
    state_raw_list = [str(v) for v in state_raw.to_numpy(dtype=object)]
    state_hys_list = [str(v) for v in state_hys.to_numpy(dtype=object)]

    return {
        "meta": {
            "mode": mode,
            "window": window,
            "enter_threshold": enter,
            "exit_threshold": exit_,
            "state_order": list(_STATE_ORDER),
        },
        "series": {
            "dates": date_list,
            "metric": metric_list,
            "range_score": score_list,
            "subscore_band": subscore_band_list,
            "raw_state": state_raw_list,
            "state": state_hys_list,
        },
        "latest": latest,
        "summary": _build_summary(state_hys),
    }
