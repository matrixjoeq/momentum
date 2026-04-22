from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

REGIME_ORDER: list[str] = [
    "UP_NARROW",
    "UP_WIDE",
    "DOWN_NARROW",
    "DOWN_WIDE",
    "SIDE_NARROW",
    "SIDE_WIDE",
]
UNCLASSIFIED = "UNCLASSIFIED"


def _rolling_linreg_slope(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    n = int(y.size)
    if n < 2:
        return float("nan")
    if (not np.isfinite(y).all()) or np.isclose(float(np.nanstd(y)), 0.0):
        return float("nan")
    x = np.arange(n, dtype=float)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    var_x = float(np.sum((x - x_mean) ** 2))
    if var_x <= 0.0:
        return float("nan")
    cov_xy = float(np.sum((x - x_mean) * (y - y_mean)))
    return float(cov_xy / var_x)


def _annualized_return_from_returns(r: pd.Series, ann_factor: int) -> float:
    rr = pd.Series(r).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(rr))
    if n == 0:
        return float("nan")
    nav = (1.0 + rr).cumprod()
    if float(nav.iloc[-1]) <= 0.0:
        return float("nan")
    return float(nav.iloc[-1] ** (float(ann_factor) / float(n)) - 1.0)


def _max_drawdown_from_returns(r: pd.Series) -> float:
    rr = pd.Series(r).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if rr.empty:
        return float("nan")
    nav = (1.0 + rr).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min()) if len(dd) else float("nan")


def _segment_perf_stats(r: pd.Series, ann_factor: int) -> dict[str, float | int | None]:
    rr = pd.Series(r).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(rr))
    if n == 0:
        return {
            "sample_days": 0,
            "cumulative_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe_ratio": None,
            "max_drawdown": None,
            "win_rate": None,
            "avg_daily_return": None,
        }
    cum = float((1.0 + rr).prod() - 1.0)
    ann_ret = _annualized_return_from_returns(rr, ann_factor)
    ann_vol = float(rr.std(ddof=0) * np.sqrt(float(ann_factor)))
    sharpe = (
        float((rr.mean() / rr.std(ddof=0)) * np.sqrt(float(ann_factor)))
        if float(rr.std(ddof=0)) > 0
        else float("nan")
    )
    mdd = _max_drawdown_from_returns(rr)
    win = float((rr > 0).mean())
    avg = float(rr.mean())
    return {
        "sample_days": n,
        "cumulative_return": cum,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "win_rate": win,
        "avg_daily_return": avg,
    }


def _classify_regimes(
    close: pd.DataFrame,
    high: pd.DataFrame | None,
    low: pd.DataFrame | None,
    *,
    slope_window: int,
    vol_window: int,
    direction_threshold_ann: float,
    vol_quantile: float,
    ann_factor: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    c = close.astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    sw = max(5, int(slope_window))
    vw = max(5, int(vol_window))
    minp = max(5, sw // 2)
    logp = np.log(c.clip(lower=1e-12))
    slope = logp.rolling(window=sw, min_periods=minp).apply(
        _rolling_linreg_slope, raw=True
    )
    slope_ann = slope * float(ann_factor)

    direction = pd.DataFrame("SIDE", index=c.index, columns=c.columns, dtype=object)
    direction = direction.mask(slope_ann > float(direction_threshold_ann), "UP")
    direction = direction.mask(slope_ann < -float(direction_threshold_ann), "DOWN")

    h = (
        high.reindex(index=c.index, columns=c.columns)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .combine_first(c)
        if high is not None
        else c.copy()
    )
    low_px = (
        low.reindex(index=c.index, columns=c.columns)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .combine_first(c)
        if low is not None
        else c.copy()
    )
    prev = c.shift(1)
    tr1 = (h - low_px).abs()
    tr2 = (h - prev).abs()
    tr3 = (low_px - prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=0).groupby(level=0).max()
    vol_metric = tr.div(prev.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    vol_metric = vol_metric.rolling(window=vw, min_periods=max(5, vw // 2)).mean()

    vol_thresh = (
        vol_metric.expanding(min_periods=max(vw, 30))
        .quantile(float(vol_quantile))
        .shift(1)
    )
    amplitude = pd.DataFrame("NARROW", index=c.index, columns=c.columns, dtype=object)
    amplitude = amplitude.mask(vol_metric >= vol_thresh, "WIDE")

    regimes = (direction.astype(str) + "_" + amplitude.astype(str)).astype(object)
    known = set(REGIME_ORDER)
    regimes = regimes.where(regimes.isin(known), other=UNCLASSIFIED)
    return regimes, slope_ann.astype(float), vol_metric.astype(float)


def build_market_regime_report(
    *,
    close: pd.DataFrame,
    high: pd.DataFrame | None,
    low: pd.DataFrame | None,
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    strategy_returns: pd.Series,
    ann_factor: int = 252,
    slope_window: int = 20,
    vol_window: int = 20,
    direction_threshold_ann: float = 0.03,
    vol_quantile: float = 0.50,
) -> dict[str, Any]:
    w = weights.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    r = asset_returns.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dates = w.index.intersection(r.index)
    cols = [c for c in w.columns if c in r.columns and c in close.columns]
    if len(dates) == 0 or len(cols) == 0:
        return {"enabled": False, "reason": "insufficient_overlap"}

    w = w.reindex(index=dates, columns=cols).fillna(0.0)
    r = r.reindex(index=dates, columns=cols).fillna(0.0)
    c = close.reindex(index=dates, columns=cols).astype(float)
    h = None if high is None else high.reindex(index=dates, columns=cols).astype(float)
    low_px = (
        None if low is None else low.reindex(index=dates, columns=cols).astype(float)
    )

    regimes, slope_ann, vol_metric = _classify_regimes(
        c,
        h,
        low_px,
        slope_window=int(slope_window),
        vol_window=int(vol_window),
        direction_threshold_ann=float(direction_threshold_ann),
        vol_quantile=float(vol_quantile),
        ann_factor=int(ann_factor),
    )
    regimes = regimes.reindex(index=dates, columns=cols).fillna(UNCLASSIFIED)
    contrib = (w * r).astype(float)
    abs_expo = w.abs().astype(float)

    exposure_total = float(abs_expo.sum().sum())
    by_state: dict[str, Any] = {}
    exposure_series_by_state: dict[str, pd.Series] = {}
    weighted_series_by_state: dict[str, pd.Series] = {}

    states_full = REGIME_ORDER + [UNCLASSIFIED]
    for state in states_full:
        mask = regimes.eq(state)
        expo_s = abs_expo.where(mask, 0.0).sum(axis=1).astype(float)
        pnl_s = contrib.where(mask, 0.0).sum(axis=1).astype(float)
        weighted_ret_s = pnl_s.div(expo_s.replace(0.0, np.nan))
        expo_sum = float(expo_s.sum())
        pnl_sum = float(pnl_s.sum())
        by_state[state] = {
            "sample_days_with_exposure": int((expo_s > 1e-12).sum()),
            "exposure_time_share": (
                float(expo_sum / exposure_total) if exposure_total > 0 else None
            ),
            "total_contribution_return": pnl_sum,
            "annualized_contribution_return": float(pnl_s.mean() * float(ann_factor)),
            "contribution_volatility_ann": float(
                pnl_s.std(ddof=0) * np.sqrt(float(ann_factor))
            ),
            "avg_weighted_asset_return_on_exposed_days": (
                float(weighted_ret_s.dropna().mean())
                if int(weighted_ret_s.notna().sum()) > 0
                else None
            ),
            "win_rate_on_exposed_days": (
                float((pnl_s[expo_s > 1e-12] > 0).mean())
                if int((expo_s > 1e-12).sum()) > 0
                else None
            ),
        }
        exposure_series_by_state[state] = expo_s
        weighted_series_by_state[state] = weighted_ret_s

    dom_expo = pd.DataFrame(
        {k: exposure_series_by_state[k] for k in REGIME_ORDER}, index=dates
    ).astype(float)
    dominant = dom_expo.idxmax(axis=1)
    dominant = dominant.where(dom_expo.max(axis=1) > 1e-12, other=UNCLASSIFIED)
    strat = (
        strategy_returns.reindex(dates)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    dominant_partition = {
        s: _segment_perf_stats(strat[dominant == s], int(ann_factor))
        for s in states_full
    }

    by_asset: dict[str, Any] = {}
    for c0 in cols:
        rc = r[c0].astype(float)
        wc = w[c0].astype(float)
        rg = regimes[c0].astype(str)
        one: dict[str, Any] = {}
        for s in states_full:
            m = rg == s
            rr = rc[m]
            ww = wc[m]
            one[s] = {
                "sample_days": int(m.sum()),
                "avg_asset_return": (float(rr.mean()) if len(rr) else None),
                "annualized_asset_return": (
                    float(rr.mean() * float(ann_factor)) if len(rr) else None
                ),
                "avg_weight_when_state": (float(ww.mean()) if len(ww) else None),
            }
        by_asset[str(c0)] = one

    return {
        "enabled": True,
        "method": {
            "name": "two_stage_rule_engine",
            "direction": "rolling log-price OLS slope with annualized threshold",
            "amplitude": "rolling ATR% (TR-based) split by expanding quantile",
            "portfolio_aggregation": [
                "exposure_contribution_attribution",
                "dominant_state_partition",
            ],
        },
        "params": {
            "slope_window": int(slope_window),
            "vol_window": int(vol_window),
            "direction_threshold_ann": float(direction_threshold_ann),
            "vol_quantile": float(vol_quantile),
        },
        "states_order": REGIME_ORDER,
        "series": {
            "dates": dates.date.astype(str).tolist(),
            "regime_by_asset": {
                str(c0): regimes[c0].astype(str).tolist() for c0 in cols
            },
            "dominant_regime": dominant.astype(str).tolist(),
            "exposure_share_by_state": {
                s: exposure_series_by_state[s].astype(float).tolist()
                for s in REGIME_ORDER
            },
            "weighted_asset_return_by_state": {
                s: weighted_series_by_state[s]
                .astype(float)
                .where(np.isfinite(weighted_series_by_state[s]), np.nan)
                .tolist()
                for s in REGIME_ORDER
            },
            "slope_ann_by_asset": {
                str(c0): slope_ann[c0].astype(float).tolist() for c0 in cols
            },
            "vol_metric_by_asset": {
                str(c0): vol_metric[c0].astype(float).tolist() for c0 in cols
            },
        },
        "strategy_state_contribution": by_state,
        "strategy_by_dominant_state": dominant_partition,
        "asset_state_summary": by_asset,
    }
