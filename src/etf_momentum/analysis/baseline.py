from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import EtfPrice


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BaselineInputs:
    codes: list[str]
    start: dt.date
    end: dt.date
    benchmark_code: str | None = None
    adjust: str = "hfq"  # qfq/hfq/none (global)
    rebalance: str = "yearly"  # daily/weekly/monthly/quarterly/yearly/none
    risk_free_rate: float = 0.025  # annualized rf (decimal), default 2.5% CN 0-1y gov
    rolling_weeks: list[int] | None = None
    rolling_months: list[int] | None = None
    rolling_years: list[int] | None = None
    fft_windows: list[int] | None = None
    fft_roll: bool = True
    fft_roll_step: int = 5
    # Risk parity (inverse-vol) portfolio config (used by research UI)
    rp_window_days: int = 60


def _inv_vol_weights(vol: pd.Series) -> pd.Series:
    v = pd.Series(vol).astype(float).replace([np.inf, -np.inf], np.nan)
    inv = 1.0 / v.replace(0.0, np.nan)
    inv = inv.where(np.isfinite(inv), other=np.nan)
    s = float(np.nansum(inv.to_numpy(dtype=float)))
    if not np.isfinite(s) or s <= 0:
        # fallback: equal weight over finite vols
        m = np.isfinite(v.to_numpy(dtype=float)) & (v.to_numpy(dtype=float) > 0)
        if not np.any(m):
            return pd.Series(0.0, index=v.index, dtype=float)
        w = np.zeros(len(v), dtype=float)
        w[m] = 1.0 / float(np.sum(m))
        return pd.Series(w, index=v.index, dtype=float)
    return (inv / s).fillna(0.0).astype(float)


def _compute_risk_parity_nav_and_weights(
    daily_ret: pd.DataFrame,
    *,
    rebalance: str,
    window: int,
    weekly_anchor: str = "FRI",
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Risk parity (inverse-vol) portfolio NAV and pre-return weights.

    Implementation notes (no lookahead):
    - We rebalance weights at the same schedule as EW (daily/weekly/monthly/quarterly/yearly/none).
    - At each rebalance boundary, we estimate vol using *past* returns only (exclude current-day return).
    - Weights are proportional to 1/vol (diagonal covariance approximation).
    """
    reb = (rebalance or "yearly").lower()
    if reb not in {"none", "daily", "weekly", "monthly", "quarterly", "yearly"}:
        raise ValueError(f"invalid rebalance={rebalance}")
    if daily_ret.empty:
        return pd.Series(dtype=float, name="RP"), pd.DataFrame()

    r = daily_ret.astype(float).fillna(0.0)
    cols = list(r.columns)
    n = len(cols)
    if n <= 0:
        return pd.Series(dtype=float, name="RP"), pd.DataFrame()

    w0 = np.full(n, 1.0 / n, dtype=float)
    w_df = pd.DataFrame(index=r.index, columns=cols, dtype=float)

    # none: buy-and-hold with initial inverse-vol weights estimated on the first window
    win = max(2, int(window))
    if reb == "none":
        # use the earliest available window (excluding day0 return which is 0 by construction)
        hist = r.iloc[1 : 1 + win] if len(r) >= 1 + win else r.iloc[1:]
        vol = hist.std(ddof=1)
        w_init = _inv_vol_weights(vol).reindex(cols).fillna(0.0).to_numpy(dtype=float)
        if float(np.sum(w_init)) <= 0:
            w_init = w0.copy()
        indiv_nav = (1.0 + r).cumprod()
        indiv_nav.iloc[0, :] = 1.0
        nav = (indiv_nav * w_init.reshape(1, n)).sum(axis=1)
        # weights drift with asset NAV
        w = indiv_nav.mul(w_init, axis=1)
        w = w.div(w.sum(axis=1), axis=0).fillna(0.0)
        w_df.loc[:, :] = w.reindex(columns=cols).to_numpy(dtype=float)
        if len(nav) > 0:
            nav.iloc[0] = 1.0
        return nav.rename("RP"), w_df

    if reb == "daily":
        # pre-return weights for day i use returns up to i-1
        for i, t in enumerate(r.index):
            hist = r.iloc[max(0, i - win) : i]
            if len(hist) < win:
                w_df.loc[t, :] = w0
            else:
                w_df.loc[t, :] = _inv_vol_weights(hist.std(ddof=1)).reindex(cols).fillna(0.0)
        port_ret = (w_df * r).sum(axis=1)
        nav = (1.0 + port_ret).cumprod()
        if len(nav) > 0:
            nav.iloc[0] = 1.0
        return nav.rename("RP"), w_df

    freq_map = {"weekly": f"W-{str(weekly_anchor).strip().upper()}", "monthly": "M", "quarterly": "Q", "yearly": "Y"}
    labels = r.index.to_period(freq_map[reb])
    prev_label = None
    nav = 1.0
    w = w0.copy()
    nav_out: list[float] = []
    w_out: list[np.ndarray] = []
    rmat = r.to_numpy(dtype=float)
    for i, lab in enumerate(labels):
        if prev_label is None or lab != prev_label:
            hist = r.iloc[max(0, i - win) : i]
            if len(hist) >= win:
                ww = _inv_vol_weights(hist.std(ddof=1)).reindex(cols).fillna(0.0).to_numpy(dtype=float)
                if float(np.sum(ww)) > 0:
                    w = ww
                else:
                    w = w0.copy()
            else:
                w = w0.copy()
        w_out.append(w.copy())
        rr = rmat[i]
        port_r = float(np.dot(w, rr))
        nav *= (1.0 + port_r)
        if 1.0 + port_r != 0.0:
            w = w * (1.0 + rr) / (1.0 + port_r)
        nav_out.append(nav)
        prev_label = lab
    nav_s = pd.Series(nav_out, index=r.index, name="RP")
    if len(nav_s) > 0:
        nav_s.iloc[0] = 1.0
    w_df2 = pd.DataFrame(np.vstack(w_out), index=r.index, columns=cols, dtype=float)
    return nav_s, w_df2


def _to_date(x: str) -> dt.date:
    return dt.datetime.strptime(x, "%Y%m%d").date()


def _max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def _max_drawdown_duration_days(nav: pd.Series) -> int:
    peak = nav.cummax()
    in_dd = nav < peak
    if not in_dd.any():
        return 0
    # duration since last peak high
    last_peak_idx = nav.index[0]
    max_dur = 0
    for t in nav.index:
        if nav.loc[t] >= peak.loc[t]:
            last_peak_idx = t
        else:
            dur = (t - last_peak_idx).days
            if dur > max_dur:
                max_dur = dur
    return int(max_dur)


def _annualized_return(nav: pd.Series, ann_factor: int = TRADING_DAYS_PER_YEAR) -> float:
    if nav.empty:
        return float("nan")
    n = len(nav) - 1
    if n <= 0:
        return 0.0
    total = nav.iloc[-1] / nav.iloc[0]
    return float(total ** (ann_factor / n) - 1.0)


def _annualized_vol(daily_ret: pd.Series, ann_factor: int = TRADING_DAYS_PER_YEAR) -> float:
    if daily_ret.empty:
        return float("nan")
    return float(daily_ret.std(ddof=1) * np.sqrt(ann_factor))


def _sharpe(daily_ret: pd.Series, rf: float = 0.0, ann_factor: int = TRADING_DAYS_PER_YEAR) -> float:
    if daily_ret.empty:
        return float("nan")
    ex = daily_ret - rf / ann_factor
    std = ex.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(ex.mean() / std * np.sqrt(ann_factor))


def _sortino(daily_ret: pd.Series, rf: float = 0.0, ann_factor: int = TRADING_DAYS_PER_YEAR) -> float:
    if daily_ret.empty:
        return float("nan")
    ex = daily_ret - rf / ann_factor
    downside = ex.where(ex < 0, 0.0)
    dd_std = downside.std(ddof=1)
    if dd_std == 0 or np.isnan(dd_std):
        return float("nan")
    return float(ex.mean() / dd_std * np.sqrt(ann_factor))


def _information_ratio(active_daily: pd.Series, ann_factor: int = TRADING_DAYS_PER_YEAR) -> float:
    if active_daily.empty:
        return float("nan")
    std = active_daily.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(active_daily.mean() / std * np.sqrt(ann_factor))


def _rolling_max_drawdown(nav: pd.Series, window: int) -> pd.Series:
    # rolling apply on values; window in trading days
    def f(x: np.ndarray) -> float:
        peak = np.maximum.accumulate(x)
        dd = x / peak - 1.0
        return float(dd.min())

    return nav.rolling(window=window, min_periods=window).apply(lambda x: f(x.to_numpy()), raw=False)


def _rolling_drawdown(nav: pd.Series, window: int) -> pd.Series:
    """
    Rolling drawdown at each date, computed within the trailing window:
      DD_t(window) = nav_t / max(nav_{t-window+1..t}) - 1
    """
    w = int(window)
    peak = nav.rolling(window=w, min_periods=w).max()
    dd = nav / peak - 1.0
    return dd.astype(float)

def _ulcer_index(nav: pd.Series, *, in_percent: bool = True) -> float:
    """
    Ulcer Index (UI): RMS of percentage drawdowns from prior peaks.

    Common definition uses percent drawdown (0..100). We default to percent units.
    """
    if nav.empty:
        return float("nan")
    peak = nav.cummax()
    dd = nav / peak - 1.0  # <= 0
    underwater = (-dd).clip(lower=0.0)
    x = underwater * (100.0 if in_percent else 1.0)
    return float(np.sqrt(np.mean(np.square(x.to_numpy(dtype=float)))))


def _rsi_wilder(price: pd.Series, *, window: int) -> pd.Series:
    """
    RSI (Wilder-style smoothing via EWM alpha=1/window) on a price-like series.
    Returns a Series in [0,100], aligned to `price`. The first ~window points are NaN.
    """
    w = max(1, int(window))
    s = pd.Series(price).astype(float).replace([np.inf, -np.inf], np.nan)
    diff = s.diff()
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / w, adjust=False, min_periods=w).mean()
    avg_loss = loss.ewm(alpha=1.0 / w, adjust=False, min_periods=w).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # handle division-by-zero edge cases deterministically
    both0 = (avg_gain == 0.0) & (avg_loss == 0.0)
    rsi = rsi.mask(both0, other=50.0)
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), other=100.0)
    rsi = rsi.mask((avg_gain == 0.0) & (avg_loss > 0.0), other=0.0)
    return rsi.clip(lower=0.0, upper=100.0)


def _fft_summary_from_returns(
    daily_ret: pd.Series,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    FFT features from daily returns (approx log-return is recommended, but pct-return is ok for small moves).

    Notes:
    - Uses Hann window + rFFT power spectrum.
    - Returns band energy ratios and dominant periods (in trading days).
    """
    x = pd.Series(daily_ret).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 8:
        return {"n": int(len(x)), "ok": False, "reason": "not enough samples (<8)", "peaks": [], "band_energy": {}, "spectral_entropy": None}

    # de-mean
    arr = x.to_numpy(dtype=float)
    arr = arr - float(np.mean(arr))
    n = len(arr)
    win = np.hanning(n)
    y = arr * win

    spec = np.fft.rfft(y)
    power = (np.abs(spec) ** 2).astype(float)
    freqs = np.fft.rfftfreq(n, d=1.0)  # cycles / day

    # exclude DC component
    power = power[1:]
    freqs = freqs[1:]
    if len(power) == 0 or float(np.sum(power)) <= 0:
        return {"n": int(n), "ok": False, "reason": "empty spectrum", "peaks": [], "band_energy": {}, "spectral_entropy": None}

    total = float(np.sum(power))
    p = power / total

    # band energies by (approx) period thresholds (in trading days)
    # low: >=60d, mid: 20-60d, high: <20d
    # convert period->freq: f = 1/period
    f_low = 1.0 / 60.0
    f_mid = 1.0 / 20.0
    low_mask = freqs <= f_low
    mid_mask = (freqs > f_low) & (freqs <= f_mid)
    high_mask = freqs > f_mid
    band = {
        "low": float(np.sum(p[low_mask])) if np.any(low_mask) else 0.0,
        "mid": float(np.sum(p[mid_mask])) if np.any(mid_mask) else 0.0,
        "high": float(np.sum(p[high_mask])) if np.any(high_mask) else 0.0,
    }

    # spectral entropy (normalized)
    eps = 1e-12
    ent = float(-np.sum(p * np.log(p + eps)) / np.log(len(p) + eps))

    # dominant peaks
    idx = np.argsort(power)[::-1]
    peaks: list[dict[str, float]] = []
    for i in idx[: int(top_k)]:
        f = float(freqs[i])
        if f <= 0:
            continue
        period = 1.0 / f
        peaks.append(
            {
                "period_days": float(period),
                "freq": float(f),
                "power_share": float(p[i]),
            }
        )

    return {
        "n": int(n),
        "ok": True,
        "method": "hann_rfft_power",
        "peaks": peaks,
        "band_energy": band,
        "spectral_entropy": ent,
    }


def _fft_analysis(
    close: pd.DataFrame,
    *,
    ew_nav: pd.Series,
    windows: list[int] | None = None,
) -> dict[str, Any]:
    """
    Fourier analysis for candidate ETF series under the same adjustment basis as `close`.

    We compute on log returns (diff(log(price))) for robustness.
    """
    if close.empty:
        return {"ok": False, "reason": "empty close", "per_code": {}, "ew": {}}
    windows = windows or [252, 126]  # ~1y, ~0.5y
    # sanitize windows
    win_clean: list[int] = []
    for w in windows:
        try:
            wi = int(w)
        except Exception:
            continue
        if wi >= 8:
            win_clean.append(wi)
    windows = sorted(list(dict.fromkeys(win_clean)), reverse=True)  # unique, desc
    px = close.astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    logret = np.log(px).diff()
    out: dict[str, Any] = {"ok": True, "method": "fft_on_log_returns", "windows": windows, "per_code": {}, "ew": {}}

    for c in px.columns:
        s = logret[c].dropna()
        per = {"full": _fft_summary_from_returns(s)}
        for w in windows:
            per[f"last_{w}"] = _fft_summary_from_returns(s.tail(int(w)))
        out["per_code"][str(c)] = per

    # EW portfolio series (based on ew_nav)
    ew = pd.Series(ew_nav).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    ew_lr = np.log(ew).diff().dropna()
    ew_per = {"full": _fft_summary_from_returns(ew_lr)}
    for w in windows:
        ew_per[f"last_{w}"] = _fft_summary_from_returns(ew_lr.tail(int(w)))
    out["ew"] = ew_per
    return out


def _histogram_from_samples(samples: np.ndarray, *, bins: int = 40, clip_q: tuple[float, float] = (0.01, 0.99)) -> dict[str, Any]:
    """
    Compute histogram from samples, similar to montecarlo._histogram.
    """
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"bin_edges": [], "counts": [], "underflow": 0, "overflow": 0}
    lo = float(np.quantile(x, clip_q[0]))
    hi = float(np.quantile(x, clip_q[1]))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = float(np.min(x)), float(np.max(x))
    if lo == hi:
        delta = max(1e-6, abs(lo) * 1e-9, 1e-12)
        lo = lo - delta
        hi = hi + delta
    under = int(np.sum(x < lo))
    over = int(np.sum(x > hi))
    core = x[(x >= lo) & (x <= hi)]
    counts, edges = np.histogram(core, bins=int(bins), range=(lo, hi))
    return {
        "bin_edges": edges.astype(float).tolist(),
        "counts": counts.astype(int).tolist(),
        "underflow": under,
        "overflow": over,
    }


def _quantiles_from_samples(samples: np.ndarray, quantiles: list[float] | None = None) -> dict[str, float]:
    """
    Compute quantiles from samples.
    """
    if quantiles is None:
        quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"q{int(q*100):02d}": float("nan") for q in quantiles}
    result = {}
    for q in quantiles:
        val = float(np.quantile(x, q))
        result[f"q{int(q*100):02d}"] = val
    return result


def _expanding_percentile_rank(s: pd.Series) -> pd.Series:
    """
    Expanding percentile rank without lookahead.

    For each time t (iterating forward), compute the percentile rank of s[t] within
    the history up to t inclusive. For ties, uses mid-rank:
        (count(<v) + 0.5*count(==v)) / n
    """
    from bisect import bisect_left, bisect_right, insort

    xs = pd.to_numeric(s, errors="coerce").astype(float)
    out = np.full(len(xs), np.nan, dtype=float)
    hist: list[float] = []
    for i, v in enumerate(xs.to_numpy(dtype=float, copy=False)):
        if not np.isfinite(v):
            continue
        insort(hist, float(v))
        n = len(hist)
        lo = bisect_left(hist, float(v))
        hi = bisect_right(hist, float(v))
        out[i] = (float(lo) + 0.5 * float(hi - lo)) / float(n)
    return pd.Series(out, index=xs.index, dtype=float)


def _compute_periodic_returns_and_volatility(
    daily_ret: pd.DataFrame,
    *,
    codes: list[str],
    daily_close: pd.DataFrame | None = None,
    daily_volume: pd.DataFrame | None = None,
    daily_amount: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Compute periodic returns and volatility for each code across different frequencies.
    
    Returns distributions and quantiles for:
    - Daily returns
    - Weekly returns (resampled)
    - Monthly returns (resampled)
    - Quarterly returns (resampled)
    - Yearly returns (resampled)
    - Daily volatility (rolling window)
    - Weekly volatility (resampled)
    - Monthly volatility (resampled)
    - Quarterly volatility (resampled)
    - Yearly volatility (resampled)

    Also supports (when `daily_close` is provided):
    - Price deviation (BIAS) distributions (end-of-period), where
        bias_t = close_t / MA252(close)_t - 1
      This measures how far price deviates from a long-term trendline (252 trading days).

    Also supports (when `daily_volume` or `daily_amount` is provided):
    - "Activity" distributions (sum within period): use volume if available, otherwise fallback to amount.
    - Relative activity (RVOL) distributions (mean within period), where
        rvol_t = activity_t / rolling_mean(activity, 20)
    - Log-activity deviation distributions (mean within period), where
        dev_t = log(activity_t) - rolling_mean(log(activity), 20)

    These are intended as "crowding" (拥挤度) proxies and do not require ETF shares data.
    """
    ret_df = daily_ret[codes].copy()
    ret_df.index = pd.to_datetime(ret_df.index)
    close_df = None
    if daily_close is not None and not daily_close.empty:
        close_df = daily_close.copy()
        close_df.index = pd.to_datetime(close_df.index)
    vol_df = None
    if daily_volume is not None and not daily_volume.empty:
        vol_df = daily_volume.copy()
        vol_df.index = pd.to_datetime(vol_df.index)
    amt_df = None
    if daily_amount is not None and not daily_amount.empty:
        amt_df = daily_amount.copy()
        amt_df.index = pd.to_datetime(amt_df.index)
    
    result: dict[str, Any] = {}
    
    for code in codes:
        if code not in ret_df.columns:
            continue
        
        code_ret = ret_df[code].dropna()
        if code_ret.empty:
            continue
        
        code_result: dict[str, Any] = {}
        
        # Daily returns
        daily_ret_vals = code_ret.values
        code_result["daily_return"] = {
            "hist": _histogram_from_samples(daily_ret_vals),
            "quantiles": _quantiles_from_samples(daily_ret_vals),
            "mean": float(np.mean(daily_ret_vals)),
            "std": float(np.std(daily_ret_vals, ddof=1)),
            "count": int(len(daily_ret_vals)),
            "current": float(code_ret.iloc[-1]),
            "current_date": pd.to_datetime(code_ret.index[-1]).date().isoformat(),
        }
        # Log-return deviation: log(1+r) - MA20(log(1+r))
        lr = np.log1p(pd.to_numeric(code_ret, errors="coerce").astype(float))
        lr = lr.replace([np.inf, -np.inf], np.nan).dropna()
        if not lr.empty:
            lr_dev = (lr - lr.rolling(window=20, min_periods=5).mean()).dropna()
            if not lr_dev.empty:
                vals = lr_dev.to_numpy(dtype=float)
                code_result["daily_log_return_dev"] = {
                    "hist": _histogram_from_samples(vals),
                    "quantiles": _quantiles_from_samples(vals),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) >= 2 else float("nan"),
                    "count": int(len(vals)),
                    "current": float(lr_dev.iloc[-1]),
                    "current_date": pd.to_datetime(lr_dev.index[-1]).date().isoformat(),
                }
        
        # Daily volatility (rolling 20-day window)
        daily_vol = code_ret.rolling(window=20, min_periods=5).std(ddof=1).dropna()
        if not daily_vol.empty:
            daily_vol_vals = daily_vol.values
            code_result["daily_volatility"] = {
                "hist": _histogram_from_samples(daily_vol_vals),
                "quantiles": _quantiles_from_samples(daily_vol_vals),
                "mean": float(np.mean(daily_vol_vals)),
                "std": float(np.std(daily_vol_vals, ddof=1)),
                "count": int(len(daily_vol_vals)),
                "current": float(daily_vol.iloc[-1]),
                "current_date": pd.to_datetime(daily_vol.index[-1]).date().isoformat(),
            }
            # Log-volatility (often closer to normal)
            lv = pd.to_numeric(daily_vol, errors="coerce").astype(float)
            lv = lv[lv > 0].dropna()
            if not lv.empty:
                lv_vals = np.log(lv.to_numpy(dtype=float))
                code_result["daily_log_volatility"] = {
                    "hist": _histogram_from_samples(lv_vals),
                    "quantiles": _quantiles_from_samples(lv_vals),
                    "mean": float(np.mean(lv_vals)),
                    "std": float(np.std(lv_vals, ddof=1)) if len(lv_vals) >= 2 else float("nan"),
                    "count": int(len(lv_vals)),
                    "current": float(np.log(float(lv.iloc[-1]))),
                    "current_date": pd.to_datetime(lv.index[-1]).date().isoformat(),
                }
                # Log-vol deviation: log(vol) - MA20(log(vol))
                lv_log = np.log(lv).replace([np.inf, -np.inf], np.nan).dropna()
                lv_dev = (lv_log - lv_log.rolling(window=20, min_periods=5).mean()).dropna()
                if not lv_dev.empty:
                    vals = lv_dev.to_numpy(dtype=float)
                    code_result["daily_log_vol_dev"] = {
                        "hist": _histogram_from_samples(vals),
                        "quantiles": _quantiles_from_samples(vals),
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals, ddof=1)) if len(vals) >= 2 else float("nan"),
                        "count": int(len(vals)),
                        "current": float(lv_dev.iloc[-1]),
                        "current_date": pd.to_datetime(lv_dev.index[-1]).date().isoformat(),
                    }
        
        # Weekly returns and volatility
        weekly_nav = (1.0 + code_ret).resample("W-FRI").prod()
        weekly_ret = weekly_nav.pct_change().dropna()
        if not weekly_ret.empty:
            weekly_ret_vals = weekly_ret.values
            code_result["weekly_return"] = {
                "hist": _histogram_from_samples(weekly_ret_vals),
                "quantiles": _quantiles_from_samples(weekly_ret_vals),
                "mean": float(np.mean(weekly_ret_vals)),
                "std": float(np.std(weekly_ret_vals, ddof=1)),
                "count": int(len(weekly_ret_vals)),
                "current": float(weekly_ret.iloc[-1]),
                "current_date": pd.to_datetime(weekly_ret.index[-1]).date().isoformat(),
            }
            wlr = np.log1p(pd.to_numeric(weekly_ret, errors="coerce").astype(float))
            wlr = wlr.replace([np.inf, -np.inf], np.nan).dropna()
            if not wlr.empty:
                wlr_dev = (wlr - wlr.rolling(window=20, min_periods=5).mean()).dropna()
                if not wlr_dev.empty:
                    vals = wlr_dev.to_numpy(dtype=float)
                    code_result["weekly_log_return_dev"] = {
                        "hist": _histogram_from_samples(vals),
                        "quantiles": _quantiles_from_samples(vals),
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals, ddof=1)) if len(vals) >= 2 else float("nan"),
                        "count": int(len(vals)),
                        "current": float(wlr_dev.iloc[-1]),
                        "current_date": pd.to_datetime(wlr_dev.index[-1]).date().isoformat(),
                    }
            # Weekly volatility (rolling 4-week window)
            weekly_vol = weekly_ret.rolling(window=4, min_periods=2).std(ddof=1).dropna()
            if not weekly_vol.empty:
                weekly_vol_vals = weekly_vol.values
                code_result["weekly_volatility"] = {
                    "hist": _histogram_from_samples(weekly_vol_vals),
                    "quantiles": _quantiles_from_samples(weekly_vol_vals),
                    "mean": float(np.mean(weekly_vol_vals)),
                    "std": float(np.std(weekly_vol_vals, ddof=1)),
                    "count": int(len(weekly_vol_vals)),
                    "current": float(weekly_vol.iloc[-1]),
                    "current_date": pd.to_datetime(weekly_vol.index[-1]).date().isoformat(),
                }
                # Log-vol deviation: log(vol) - MA20(log(vol))
                wlv = pd.to_numeric(weekly_vol, errors="coerce").astype(float)
                wlv = wlv[wlv > 0].dropna()
                if not wlv.empty:
                    wlv_log = np.log(wlv)
                    wlv_dev = (wlv_log - wlv_log.rolling(window=20, min_periods=5).mean()).dropna()
                    if not wlv_dev.empty:
                        vals = wlv_dev.to_numpy(dtype=float)
                        code_result["weekly_log_vol_dev"] = {
                            "hist": _histogram_from_samples(vals),
                            "quantiles": _quantiles_from_samples(vals),
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals, ddof=1)) if len(vals) >= 2 else float("nan"),
                            "count": int(len(vals)),
                            "current": float(wlv_dev.iloc[-1]),
                            "current_date": pd.to_datetime(wlv_dev.index[-1]).date().isoformat(),
                        }
                wv = pd.to_numeric(weekly_vol, errors="coerce").astype(float)
                wv = wv[wv > 0].dropna()
                if not wv.empty:
                    wv_vals = np.log(wv.to_numpy(dtype=float))
                    code_result["weekly_log_volatility"] = {
                        "hist": _histogram_from_samples(wv_vals),
                        "quantiles": _quantiles_from_samples(wv_vals),
                        "mean": float(np.mean(wv_vals)),
                        "std": float(np.std(wv_vals, ddof=1)) if len(wv_vals) >= 2 else float("nan"),
                        "count": int(len(wv_vals)),
                        "current": float(np.log(float(wv.iloc[-1]))),
                        "current_date": pd.to_datetime(wv.index[-1]).date().isoformat(),
                    }
        
        # Monthly returns and volatility
        monthly_nav = (1.0 + code_ret).resample("ME").prod()
        monthly_ret = monthly_nav.pct_change().dropna()
        if not monthly_ret.empty:
            monthly_ret_vals = monthly_ret.values
            code_result["monthly_return"] = {
                "hist": _histogram_from_samples(monthly_ret_vals),
                "quantiles": _quantiles_from_samples(monthly_ret_vals),
                "mean": float(np.mean(monthly_ret_vals)),
                "std": float(np.std(monthly_ret_vals, ddof=1)),
                "count": int(len(monthly_ret_vals)),
                "current": float(monthly_ret.iloc[-1]),
                "current_date": pd.to_datetime(monthly_ret.index[-1]).date().isoformat(),
            }
            mlr = np.log1p(pd.to_numeric(monthly_ret, errors="coerce").astype(float))
            mlr = mlr.replace([np.inf, -np.inf], np.nan).dropna()
            if not mlr.empty:
                mlr_dev = (mlr - mlr.rolling(window=20, min_periods=5).mean()).dropna()
                if not mlr_dev.empty:
                    vals = mlr_dev.to_numpy(dtype=float)
                    code_result["monthly_log_return_dev"] = {
                        "hist": _histogram_from_samples(vals),
                        "quantiles": _quantiles_from_samples(vals),
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals, ddof=1)) if len(vals) >= 2 else float("nan"),
                        "count": int(len(vals)),
                        "current": float(mlr_dev.iloc[-1]),
                        "current_date": pd.to_datetime(mlr_dev.index[-1]).date().isoformat(),
                    }
            # Monthly volatility (rolling 3-month window)
            monthly_vol = monthly_ret.rolling(window=3, min_periods=2).std(ddof=1).dropna()
            if not monthly_vol.empty:
                monthly_vol_vals = monthly_vol.values
                code_result["monthly_volatility"] = {
                    "hist": _histogram_from_samples(monthly_vol_vals),
                    "quantiles": _quantiles_from_samples(monthly_vol_vals),
                    "mean": float(np.mean(monthly_vol_vals)),
                    "std": float(np.std(monthly_vol_vals, ddof=1)),
                    "count": int(len(monthly_vol_vals)),
                    "current": float(monthly_vol.iloc[-1]),
                    "current_date": pd.to_datetime(monthly_vol.index[-1]).date().isoformat(),
                }
                mlv = pd.to_numeric(monthly_vol, errors="coerce").astype(float)
                mlv = mlv[mlv > 0].dropna()
                if not mlv.empty:
                    mlv_log = np.log(mlv)
                    mlv_dev = (mlv_log - mlv_log.rolling(window=20, min_periods=5).mean()).dropna()
                    if not mlv_dev.empty:
                        vals = mlv_dev.to_numpy(dtype=float)
                        code_result["monthly_log_vol_dev"] = {
                            "hist": _histogram_from_samples(vals),
                            "quantiles": _quantiles_from_samples(vals),
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals, ddof=1)) if len(vals) >= 2 else float("nan"),
                            "count": int(len(vals)),
                            "current": float(mlv_dev.iloc[-1]),
                            "current_date": pd.to_datetime(mlv_dev.index[-1]).date().isoformat(),
                        }
                mv = pd.to_numeric(monthly_vol, errors="coerce").astype(float)
                mv = mv[mv > 0].dropna()
                if not mv.empty:
                    mv_vals = np.log(mv.to_numpy(dtype=float))
                    code_result["monthly_log_volatility"] = {
                        "hist": _histogram_from_samples(mv_vals),
                        "quantiles": _quantiles_from_samples(mv_vals),
                        "mean": float(np.mean(mv_vals)),
                        "std": float(np.std(mv_vals, ddof=1)) if len(mv_vals) >= 2 else float("nan"),
                        "count": int(len(mv_vals)),
                        "current": float(np.log(float(mv.iloc[-1]))),
                        "current_date": pd.to_datetime(mv.index[-1]).date().isoformat(),
                    }
        
        # Quarterly returns and volatility
        quarterly_nav = (1.0 + code_ret).resample("QE").prod()
        quarterly_ret = quarterly_nav.pct_change().dropna()
        if not quarterly_ret.empty:
            quarterly_ret_vals = quarterly_ret.values
            code_result["quarterly_return"] = {
                "hist": _histogram_from_samples(quarterly_ret_vals),
                "quantiles": _quantiles_from_samples(quarterly_ret_vals),
                "mean": float(np.mean(quarterly_ret_vals)),
                "std": float(np.std(quarterly_ret_vals, ddof=1)),
                "count": int(len(quarterly_ret_vals)),
                "current": float(quarterly_ret.iloc[-1]),
                "current_date": pd.to_datetime(quarterly_ret.index[-1]).date().isoformat(),
            }
            qlr = np.log1p(pd.to_numeric(quarterly_ret, errors="coerce").astype(float))
            qlr = qlr.replace([np.inf, -np.inf], np.nan).dropna()
            if not qlr.empty:
                qlr_dev = (qlr - qlr.rolling(window=20, min_periods=5).mean()).dropna()
                if not qlr_dev.empty:
                    vals = qlr_dev.to_numpy(dtype=float)
                    code_result["quarterly_log_return_dev"] = {
                        "hist": _histogram_from_samples(vals),
                        "quantiles": _quantiles_from_samples(vals),
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals, ddof=1)) if len(vals) >= 2 else float("nan"),
                        "count": int(len(vals)),
                        "current": float(qlr_dev.iloc[-1]),
                        "current_date": pd.to_datetime(qlr_dev.index[-1]).date().isoformat(),
                    }
            # Quarterly volatility (rolling 2-quarter window)
            quarterly_vol = quarterly_ret.rolling(window=2, min_periods=2).std(ddof=1).dropna()
            if not quarterly_vol.empty:
                quarterly_vol_vals = quarterly_vol.values
                code_result["quarterly_volatility"] = {
                    "hist": _histogram_from_samples(quarterly_vol_vals),
                    "quantiles": _quantiles_from_samples(quarterly_vol_vals),
                    "mean": float(np.mean(quarterly_vol_vals)),
                    "std": float(np.std(quarterly_vol_vals, ddof=1)),
                    "count": int(len(quarterly_vol_vals)),
                    "current": float(quarterly_vol.iloc[-1]),
                    "current_date": pd.to_datetime(quarterly_vol.index[-1]).date().isoformat(),
                }
                qlv = pd.to_numeric(quarterly_vol, errors="coerce").astype(float)
                qlv = qlv[qlv > 0].dropna()
                if not qlv.empty:
                    qlv_log = np.log(qlv)
                    qlv_dev = (qlv_log - qlv_log.rolling(window=20, min_periods=5).mean()).dropna()
                    if not qlv_dev.empty:
                        vals = qlv_dev.to_numpy(dtype=float)
                        code_result["quarterly_log_vol_dev"] = {
                            "hist": _histogram_from_samples(vals),
                            "quantiles": _quantiles_from_samples(vals),
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals, ddof=1)) if len(vals) >= 2 else float("nan"),
                            "count": int(len(vals)),
                            "current": float(qlv_dev.iloc[-1]),
                            "current_date": pd.to_datetime(qlv_dev.index[-1]).date().isoformat(),
                        }
                qv = pd.to_numeric(quarterly_vol, errors="coerce").astype(float)
                qv = qv[qv > 0].dropna()
                if not qv.empty:
                    qv_vals = np.log(qv.to_numpy(dtype=float))
                    code_result["quarterly_log_volatility"] = {
                        "hist": _histogram_from_samples(qv_vals),
                        "quantiles": _quantiles_from_samples(qv_vals),
                        "mean": float(np.mean(qv_vals)),
                        "std": float(np.std(qv_vals, ddof=1)) if len(qv_vals) >= 2 else float("nan"),
                        "count": int(len(qv_vals)),
                        "current": float(np.log(float(qv.iloc[-1]))),
                        "current_date": pd.to_datetime(qv.index[-1]).date().isoformat(),
                    }
        
        # Yearly returns and volatility
        yearly_nav = (1.0 + code_ret).resample("YE").prod()
        yearly_ret = yearly_nav.pct_change().dropna()
        if not yearly_ret.empty:
            yearly_ret_vals = yearly_ret.values
            code_result["yearly_return"] = {
                "hist": _histogram_from_samples(yearly_ret_vals),
                "quantiles": _quantiles_from_samples(yearly_ret_vals),
                "mean": float(np.mean(yearly_ret_vals)),
                "std": float(np.std(yearly_ret_vals, ddof=1)),
                "count": int(len(yearly_ret_vals)),
                "current": float(yearly_ret.iloc[-1]),
                "current_date": pd.to_datetime(yearly_ret.index[-1]).date().isoformat(),
            }
            # Yearly volatility (use all available years)
            if len(yearly_ret) >= 2:
                yearly_vol_vals = yearly_ret.values
                code_result["yearly_volatility"] = {
                    "hist": _histogram_from_samples(yearly_vol_vals),
                    "quantiles": _quantiles_from_samples(yearly_vol_vals),
                    "mean": float(np.mean(yearly_vol_vals)),
                    "std": float(np.std(yearly_vol_vals, ddof=1)),
                    "count": int(len(yearly_vol_vals)),
                    "current": float(yearly_ret.iloc[-1]),
                    "current_date": pd.to_datetime(yearly_ret.index[-1]).date().isoformat(),
                }

        # Price deviation (log-price deviation from long-term trendline), optional.
        #
        # Compute by frequency using end-of-period closes and frequency-specific windows:
        # - daily: N=252
        # - weekly: N=52
        # - monthly: N=12
        # - quarterly: N=4
        # - yearly: N=3
        #
        # dev = log(P) - MA_N(log(P))  (equivalently log(P / GM_N(P)))
        if close_df is not None and code in close_df.columns:
            px = pd.to_numeric(close_df[code], errors="coerce").astype(float).dropna()
            if not px.empty:
                def _add_dev(kind: str, s: pd.Series) -> None:
                    if s is None or s.empty:
                        return
                    xs = s.to_numpy(dtype=float)
                    code_result[f"{kind}_price_dev"] = {
                        "hist": _histogram_from_samples(xs),
                        "quantiles": _quantiles_from_samples(xs),
                        "mean": float(np.mean(xs)),
                        "std": float(np.std(xs, ddof=1)) if len(xs) >= 2 else float("nan"),
                        "count": int(len(xs)),
                        "current": float(s.iloc[-1]),
                        "current_date": pd.to_datetime(s.index[-1]).date().isoformat(),
                    }

                def _log_dev(s: pd.Series, *, n: int) -> pd.Series:
                    s2 = pd.to_numeric(s, errors="coerce").astype(float).replace(0.0, np.nan)
                    s2 = s2.replace([np.inf, -np.inf], np.nan).dropna()
                    if s2.empty:
                        return pd.Series([], dtype=float)
                    lp = np.log(s2)
                    mp = lp.rolling(window=int(n), min_periods=max(3, min(20, int(n)))).mean()
                    out = (lp - mp).replace([np.inf, -np.inf], np.nan).dropna()
                    return out

                px_d = px
                px_w = px.resample("W-FRI").last().dropna()
                px_m = px.resample("ME").last().dropna()
                px_q = px.resample("QE").last().dropna()
                px_y = px.resample("YE").last().dropna()

                _add_dev("daily", _log_dev(px_d, n=252))
                _add_dev("weekly", _log_dev(px_w, n=52))
                _add_dev("monthly", _log_dev(px_m, n=12))
                _add_dev("quarterly", _log_dev(px_q, n=4))
                _add_dev("yearly", _log_dev(px_y, n=3))

        # Activity (volume/amount) and crowding proxies (optional)
        act: pd.Series | None = None
        act_src: str | None = None
        if vol_df is not None and code in vol_df.columns:
            s = pd.to_numeric(vol_df[code], errors="coerce").astype(float).dropna()
            if not s.empty:
                act = s
                act_src = "volume"
        if act is None and amt_df is not None and code in amt_df.columns:
            s = pd.to_numeric(amt_df[code], errors="coerce").astype(float).dropna()
            if not s.empty:
                act = s
                act_src = "amount"

        if act is not None and act_src is not None:
            def _resample_sum(s: pd.Series, freq: str) -> pd.Series:
                out = s.resample(freq).sum(min_count=1)
                return pd.to_numeric(out, errors="coerce").astype(float).dropna()

            def _resample_mean(s: pd.Series, freq: str) -> pd.Series:
                out = s.resample(freq).mean()
                return pd.to_numeric(out, errors="coerce").astype(float).dropna()

            # Activity distributions (sum within period)
            for kind, ser in [
                ("daily", act),
                ("weekly", _resample_sum(act, "W-FRI")),
                ("monthly", _resample_sum(act, "ME")),
                ("quarterly", _resample_sum(act, "QE")),
                ("yearly", _resample_sum(act, "YE")),
            ]:
                if ser is None or ser.empty:
                    continue
                xs = ser.to_numpy(dtype=float)
                code_result[f"{kind}_volume"] = {
                    "hist": _histogram_from_samples(xs),
                    "quantiles": _quantiles_from_samples(xs),
                    "mean": float(np.mean(xs)),
                    "std": float(np.std(xs, ddof=1)) if len(xs) >= 2 else float("nan"),
                    "count": int(len(xs)),
                    "source": act_src,
                    "current": float(ser.iloc[-1]),
                    "current_date": pd.to_datetime(ser.index[-1]).date().isoformat(),
                }

            # Crowding proxy #1: RVOL on activity (mean within period = sustained crowding)
            den = act.rolling(window=20, min_periods=5).mean()
            rvol = (act / den).replace([np.inf, -np.inf], np.nan).dropna()
            for kind, ser in [
                ("daily", rvol),
                ("weekly", _resample_mean(rvol, "W-FRI")),
                ("monthly", _resample_mean(rvol, "ME")),
                ("quarterly", _resample_mean(rvol, "QE")),
                ("yearly", _resample_mean(rvol, "YE")),
            ]:
                if ser is None or ser.empty:
                    continue
                xs = ser.to_numpy(dtype=float)
                code_result[f"{kind}_rvol"] = {
                    "hist": _histogram_from_samples(xs),
                    "quantiles": _quantiles_from_samples(xs),
                    "mean": float(np.mean(xs)),
                    "std": float(np.std(xs, ddof=1)) if len(xs) >= 2 else float("nan"),
                    "count": int(len(xs)),
                    "source": act_src,
                    "current": float(ser.iloc[-1]),
                    "current_date": pd.to_datetime(ser.index[-1]).date().isoformat(),
                }

            # Crowding proxy #2: log-activity deviation (mean within period = sustained crowding)
            lv = np.log(act.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
            if not lv.empty:
                dev = (lv - lv.rolling(window=20, min_periods=5).mean()).replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                for kind, ser in [
                    ("daily", dev),
                    ("weekly", _resample_mean(dev, "W-FRI")),
                    ("monthly", _resample_mean(dev, "ME")),
                    ("quarterly", _resample_mean(dev, "QE")),
                    ("yearly", _resample_mean(dev, "YE")),
                ]:
                    if ser is None or ser.empty:
                        continue
                    xs = ser.to_numpy(dtype=float)
                    code_result[f"{kind}_vol_dev"] = {
                        "hist": _histogram_from_samples(xs),
                        "quantiles": _quantiles_from_samples(xs),
                        "mean": float(np.mean(xs)),
                        "std": float(np.std(xs, ddof=1)) if len(xs) >= 2 else float("nan"),
                        "count": int(len(xs)),
                        "source": act_src,
                        "current": float(ser.iloc[-1]),
                        "current_date": pd.to_datetime(ser.index[-1]).date().isoformat(),
                    }
        
        result[code] = code_result
    
    return result


def _fft_roll_timeseries_from_returns(
    log_returns: pd.Series,
    *,
    windows: list[int],
    step: int = 5,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Rolling FFT feature time series computed on log returns.

    Output per window w:
    - spectral_entropy
    - high_band_energy (share)

    To keep runtime manageable, compute every `step` trading days.
    """
    # sanitize windows similarly to _fft_analysis
    win_clean: list[int] = []
    for w in windows or []:
        try:
            wi = int(w)
        except Exception:
            continue
        if wi >= 8:
            win_clean.append(wi)
    win_clean = sorted(list(dict.fromkeys(win_clean)), reverse=True)

    x = pd.Series(log_returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(x))
    step_i = int(step) if int(step) > 0 else 1
    out: dict[str, Any] = {
        "ok": True,
        "method": "rolling_fft_on_log_returns",
        "windows": win_clean,
        "step": int(step_i),
        "series": {},
    }
    if n < 8:
        out["ok"] = False
        out["reason"] = "not enough samples (<8)"
        return out

    idx = x.index
    arr = x.to_numpy(dtype=float)

    for wi in win_clean:
        if n < wi:
            out["series"][f"last_{wi}"] = {"ok": False, "reason": f"not enough samples (<{wi})", "dates": [], "spectral_entropy": [], "high_band_energy": []}
            continue
        dates: list[str] = []
        ent: list[float] = []
        high: list[float] = []
        for end_i in range(wi - 1, n, step_i):
            seg = arr[end_i - wi + 1 : end_i + 1]
            s = _fft_summary_from_returns(pd.Series(seg), top_k=top_k)
            if not s.get("ok"):
                continue
            dates.append(pd.to_datetime(idx[end_i]).date().isoformat())
            ent.append(float(s.get("spectral_entropy")))
            be = s.get("band_energy") or {}
            high.append(float(be.get("high", 0.0)))
        out["series"][f"last_{wi}"] = {"ok": True, "dates": dates, "spectral_entropy": ent, "high_band_energy": high}
    return out


def load_close_prices(
    db: Session,
    *,
    codes: list[str],
    start: dt.date,
    end: dt.date,
    adjust: str,
) -> pd.DataFrame:
    stmt = (
        select(EtfPrice.trade_date, EtfPrice.code, EtfPrice.close)
        .where(EtfPrice.code.in_(codes))
        .where(EtfPrice.adjust == adjust)
        .where(EtfPrice.trade_date >= start)
        .where(EtfPrice.trade_date <= end)
        .order_by(EtfPrice.trade_date.asc())
    )
    rows = db.execute(stmt).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "code", "close"])
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="code", values="close", aggfunc="last").sort_index()
    return pivot


def load_volume_amount(
    db: Session,
    *,
    codes: list[str],
    start: dt.date,
    end: dt.date,
    adjust: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load volume/amount matrices for each code.

    Note: volume/amount are typically not adjusted; we still filter by `adjust` to
    stay consistent with the price loaders.
    """
    stmt = (
        select(EtfPrice.trade_date, EtfPrice.code, EtfPrice.volume, EtfPrice.amount)
        .where(EtfPrice.code.in_(codes))
        .where(EtfPrice.adjust == adjust)
        .where(EtfPrice.trade_date >= start)
        .where(EtfPrice.trade_date <= end)
        .order_by(EtfPrice.trade_date.asc())
    )
    rows = db.execute(stmt).all()
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "code", "volume", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    volume = df.pivot_table(index="date", columns="code", values="volume", aggfunc="last").sort_index()
    amount = df.pivot_table(index="date", columns="code", values="amount", aggfunc="last").sort_index()
    return volume, amount


def load_high_low_prices(
    db: Session,
    *,
    codes: list[str],
    start: dt.date,
    end: dt.date,
    adjust: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load high/low price matrices for each code.

    Returns:
      (high_df, low_df) indexed by date with columns as codes.
    """
    stmt = (
        select(EtfPrice.trade_date, EtfPrice.code, EtfPrice.high, EtfPrice.low)
        .where(EtfPrice.code.in_(codes))
        .where(EtfPrice.adjust == adjust)
        .where(EtfPrice.trade_date >= start)
        .where(EtfPrice.trade_date <= end)
        .order_by(EtfPrice.trade_date.asc())
    )
    rows = db.execute(stmt).all()
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "code", "high", "low"])
    df["date"] = pd.to_datetime(df["date"])
    high = df.pivot_table(index="date", columns="code", values="high", aggfunc="last").sort_index()
    low = df.pivot_table(index="date", columns="code", values="low", aggfunc="last").sort_index()
    return high, low


def load_ohlc_prices(
    db: Session,
    *,
    codes: list[str],
    start: dt.date,
    end: dt.date,
    adjust: str,
) -> dict[str, pd.DataFrame]:
    """
    Load OHLC price matrices for each code.

    Returns a dict with keys: open/high/low/close, each a DataFrame indexed by date with columns as codes.
    """
    stmt = (
        select(EtfPrice.trade_date, EtfPrice.code, EtfPrice.open, EtfPrice.high, EtfPrice.low, EtfPrice.close)
        .where(EtfPrice.code.in_(codes))
        .where(EtfPrice.adjust == adjust)
        .where(EtfPrice.trade_date >= start)
        .where(EtfPrice.trade_date <= end)
        .order_by(EtfPrice.trade_date.asc())
    )
    rows = db.execute(stmt).all()
    if not rows:
        return {"open": pd.DataFrame(), "high": pd.DataFrame(), "low": pd.DataFrame(), "close": pd.DataFrame()}
    df = pd.DataFrame(rows, columns=["date", "code", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    for k in ["open", "high", "low", "close"]:
        out[k] = df.pivot_table(index="date", columns="code", values=k, aggfunc="last").sort_index()
    return out


def _compute_equal_weight_nav(
    daily_ret: pd.DataFrame,
    *,
    rebalance: str,
    weekly_anchor: str = "FRI",
) -> pd.Series:
    """
    Equal-weight portfolio NAV under different rebalancing schedules.

    rebalance:
    - none: buy-and-hold equal initial weights (no rebalancing)
    - daily/weekly/monthly/quarterly/yearly: reset to equal weights at period boundaries
    """
    reb = (rebalance or "yearly").lower()
    if reb not in {"none", "daily", "weekly", "monthly", "quarterly", "yearly"}:
        raise ValueError(f"invalid rebalance={rebalance}")

    # buy & hold (equal initial weights): mean of individual NAVs
    if reb == "none":
        indiv_nav = (1.0 + daily_ret.fillna(0.0)).cumprod()
        indiv_nav.iloc[0, :] = 1.0
        return indiv_nav.mean(axis=1)

    if reb == "daily":
        ew_ret = daily_ret.fillna(0.0).mean(axis=1)
        ew_nav = (1.0 + ew_ret).cumprod()
        ew_nav.iloc[0] = 1.0
        return ew_nav

    freq_map = {
        "weekly": f"W-{str(weekly_anchor).strip().upper()}",
        "monthly": "M",
        "quarterly": "Q",
        "yearly": "Y",
    }
    labels = daily_ret.index.to_period(freq_map[reb])

    nav = 1.0
    n = daily_ret.shape[1]
    w = np.full(n, 1.0 / n, dtype=float)
    out = []
    prev_label = None
    rmat = daily_ret.fillna(0.0).to_numpy(dtype=float)
    for i, lab in enumerate(labels):
        if prev_label is None or lab != prev_label:
            w[:] = 1.0 / n
        r = rmat[i]
        port_r = float(np.dot(w, r))
        nav *= (1.0 + port_r)
        # update weights post-move
        if 1.0 + port_r != 0.0:
            w = w * (1.0 + r) / (1.0 + port_r)
        out.append(nav)
        prev_label = lab
    s = pd.Series(out, index=daily_ret.index, name="EW")
    if len(s) > 0:
        s.iloc[0] = 1.0
    return s


def _compute_equal_weight_nav_and_weights(
    daily_ret: pd.DataFrame,
    *,
    rebalance: str,
    weekly_anchor: str = "FRI",
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Compute equal-weight NAV and the corresponding pre-return weights per day.

    Weights are aligned so that for date t, weights[t] apply to the return r[t]
    (where r[t] is close[t]/close[t-1]-1).
    """
    reb = (rebalance or "yearly").lower()
    if reb not in {"none", "daily", "weekly", "monthly", "quarterly", "yearly"}:
        raise ValueError(f"invalid rebalance={rebalance}")
    if daily_ret.empty:
        return pd.Series(dtype=float, name="EW"), pd.DataFrame()

    cols = list(daily_ret.columns)
    n = len(cols)
    if n <= 0:
        return pd.Series(dtype=float, name="EW"), pd.DataFrame()

    rmat = daily_ret.fillna(0.0).to_numpy(dtype=float)

    # none: buy-and-hold equal initial weights
    if reb == "none":
        indiv_nav = (1.0 + daily_ret.fillna(0.0)).cumprod()
        indiv_nav.iloc[0, :] = 1.0
        nav = indiv_nav.mean(axis=1)
        w = indiv_nav.div(indiv_nav.sum(axis=1), axis=0).fillna(0.0)
        w = w.reindex(columns=cols)
        if len(nav) > 0:
            nav.iloc[0] = 1.0
        return nav.rename("EW"), w

    # daily: reset to equal weights every day (pre-return weights always equal)
    if reb == "daily":
        w = pd.DataFrame((1.0 / n), index=daily_ret.index, columns=cols, dtype=float)
        ew_ret = daily_ret.fillna(0.0).mean(axis=1)
        nav = (1.0 + ew_ret).cumprod()
        if len(nav) > 0:
            nav.iloc[0] = 1.0
        return nav.rename("EW"), w

    freq_map = {"weekly": f"W-{str(weekly_anchor).strip().upper()}", "monthly": "M", "quarterly": "Q", "yearly": "Y"}
    labels = daily_ret.index.to_period(freq_map[reb])
    prev_label = None
    nav = 1.0
    w = np.full(n, 1.0 / n, dtype=float)
    nav_out: list[float] = []
    w_out: list[np.ndarray] = []
    for i, lab in enumerate(labels):
        if prev_label is None or lab != prev_label:
            w[:] = 1.0 / n
        # record pre-return weights for attribution
        w_out.append(w.copy())
        r = rmat[i]
        port_r = float(np.dot(w, r))
        nav *= (1.0 + port_r)
        # update weights post-move
        if 1.0 + port_r != 0.0:
            w = w * (1.0 + r) / (1.0 + port_r)
        nav_out.append(nav)
        prev_label = lab
    nav_s = pd.Series(nav_out, index=daily_ret.index, name="EW")
    if len(nav_s) > 0:
        nav_s.iloc[0] = 1.0
    w_df = pd.DataFrame(np.vstack(w_out), index=daily_ret.index, columns=cols, dtype=float)
    return nav_s, w_df


def _compute_return_risk_contributions(
    *,
    asset_ret: pd.DataFrame,
    weights: pd.DataFrame,
    total_return: float,
) -> dict[str, Any]:
    """
    Return & risk contribution (by code) for a portfolio with time-varying weights.

    - Return contribution uses a log-return based attribution and is scaled to sum to total_return.
    - Risk contribution uses variance decomposition based on sample covariance of returns and mean weights.
    """
    r = asset_ret.astype(float).fillna(0.0)
    w = weights.reindex(index=r.index, columns=r.columns).astype(float).fillna(0.0)
    # log-return attribution (scaled)
    log_r = np.log1p(r.clip(lower=-0.999999))
    log_contrib = (w * log_r).sum(axis=0).astype(float)
    approx_port_log = float(np.nansum(log_contrib.to_numpy(dtype=float)))
    if approx_port_log == 0.0 or np.isnan(approx_port_log):
        share = log_contrib * np.nan
        contrib = log_contrib * np.nan
    else:
        share = (log_contrib / approx_port_log).astype(float)
        contrib = (share * float(total_return)).astype(float)

    return_rows = []
    for c in r.columns:
        return_rows.append(
            {
                "code": str(c),
                "return_contribution": (None if pd.isna(contrib.get(c)) else float(contrib.get(c))),
                "return_share": (None if pd.isna(share.get(c)) else float(share.get(c))),
            }
        )

    # variance contribution
    cov = r.cov()
    w_bar = w.mean(axis=0).astype(float)
    w_vec = w_bar.reindex(cov.index).fillna(0.0).to_numpy(dtype=float)
    cov_mat = cov.to_numpy(dtype=float)
    port_var = float(w_vec.T @ cov_mat @ w_vec) if len(w_vec) else float("nan")
    if port_var == 0.0 or np.isnan(port_var):
        risk_rows = [{"code": str(c), "risk_share": None} for c in cov.index]
    else:
        m = cov_mat @ w_vec  # marginal contribution to variance
        rc = w_vec * m
        rc_share = rc / port_var
        risk_rows = []
        for i, c in enumerate(cov.index):
            risk_rows.append({"code": str(c), "risk_share": float(rc_share[i])})

    return {
        "return": {
            "method": "log_scaled",
            "total_return": float(total_return),
            "approx_port_log": approx_port_log,
            "by_code": return_rows,
        },
        "risk": {
            "method": "variance_share",
            "portfolio_variance": port_var,
            "by_code": risk_rows,
        },
    }


def compute_baseline(db: Session, inp: BaselineInputs) -> dict[str, Any]:
    codes = list(dict.fromkeys(inp.codes))
    if not codes:
        raise ValueError("codes is empty")

    close = load_close_prices(db, codes=codes, start=inp.start, end=inp.end, adjust=inp.adjust)
    if close.empty:
        raise ValueError("no price data for given range")

    close = close.sort_index()
    missing = [c for c in codes if c not in close.columns or close[c].dropna().empty]
    if missing:
        raise ValueError(f"missing data for adjust={inp.adjust}: {missing}")
    # individual NAVs (forward-fill for plotting continuity)
    close_ff = close.ffill()
    daily_ret = close_ff.pct_change()
    nav = (1.0 + daily_ret).cumprod()
    nav.iloc[0, :] = 1.0

    # common start where all selected have data after ffill (i.e. each has first valid close)
    first_valid = {c: close[c].first_valid_index() for c in codes if c in close.columns}
    common_start = max([d for d in first_valid.values() if d is not None])
    nav_common = nav.loc[common_start:]
    ret_common = daily_ret.loc[common_start:].fillna(0.0)

    # Activity (volume/amount) should reflect trading, not adjusted prices.
    # Therefore always load it from `adjust="none"` (if available in DB).
    volume, amount = load_volume_amount(
        db,
        codes=codes,
        start=inp.start,
        end=inp.end,
        adjust="none",
    )
    volume_common: pd.DataFrame | None = None
    if volume is not None and not volume.empty:
        volume = volume.sort_index()
        volume_common = volume.loc[common_start:]
    amount_common: pd.DataFrame | None = None
    if amount is not None and not amount.empty:
        amount = amount.sort_index()
        amount_common = amount.loc[common_start:]

    # correlation matrix (using full backtest range after common_start)
    corr_ret = ret_common[codes].astype(float)
    corr = corr_ret.corr(method="pearson")
    corr_out = {
        "method": "pearson",
        "n_obs": int(len(corr_ret.index)),
        "codes": [c for c in codes if c in corr.columns],
        "matrix": corr.to_numpy(dtype=float).tolist(),
    }

    # equal-weight with configurable rebalancing (also return weights for attribution)
    ew_nav, ew_w = _compute_equal_weight_nav_and_weights(ret_common[codes], rebalance=inp.rebalance)
    ew_ret = ew_nav.pct_change().fillna(0.0)

    # risk parity (inverse-vol) portfolio with the same rebalancing schedule
    rp_win = max(2, int(getattr(inp, "rp_window_days", 60) or 60))
    rp_nav, rp_w = _compute_risk_parity_nav_and_weights(ret_common[codes], rebalance=inp.rebalance, window=rp_win)
    rp_ret = rp_nav.pct_change().fillna(0.0)

    # risk parity (inverse-vol) portfolio with the same rebalancing schedule
    rp_win = max(2, int(getattr(inp, "rp_window_days", 60) or 60))
    rp_nav, rp_w = _compute_risk_parity_nav_and_weights(ret_common[codes], rebalance=inp.rebalance, window=rp_win)
    rp_ret = rp_nav.pct_change().fillna(0.0)

    # benchmark
    bench_code = inp.benchmark_code
    if bench_code is None:
        bench_code = "510300" if "510300" in codes else codes[0]
    bench_ret = ret_common[bench_code] if bench_code in ret_common.columns else ew_ret * 0.0
    bench_nav = (1.0 + bench_ret.fillna(0.0)).cumprod()
    bench_nav.iloc[0] = 1.0

    # periodic returns
    def period_returns(series: pd.Series, freq: str) -> pd.DataFrame:
        s = series.copy()
        s.index = pd.to_datetime(s.index)
        r = s.resample(freq).last().pct_change().dropna()
        return pd.DataFrame({"period_end": r.index.date.astype(str), "return": r.values})

    weekly = period_returns(ew_nav, "W-FRI")
    monthly = period_returns(ew_nav, "ME")
    quarterly = period_returns(ew_nav, "QE")
    yearly = period_returns(ew_nav, "YE")
    weekly_rp = period_returns(rp_nav, "W-FRI")
    monthly_rp = period_returns(rp_nav, "ME")
    quarterly_rp = period_returns(rp_nav, "QE")
    yearly_rp = period_returns(rp_nav, "YE")

    def _win_payoff_kelly(df: pd.DataFrame) -> dict[str, float]:
        r = pd.Series(df["return"].astype(float).to_numpy()) if (df is not None and not df.empty) else pd.Series([], dtype=float)
        if r.empty:
            return {"win_rate": float("nan"), "payoff_ratio": float("nan"), "kelly_fraction": float("nan")}
        pos = r[r > 0]
        neg = r[r < 0]
        win_rate = float(len(pos) / len(r)) if len(r) else float("nan")
        avg_win = float(pos.mean()) if len(pos) else float("nan")
        avg_loss = float(neg.mean()) if len(neg) else float("nan")  # negative
        payoff = float(avg_win / abs(avg_loss)) if (len(pos) and len(neg) and avg_loss != 0) else float("nan")
        if np.isfinite(win_rate) and np.isfinite(payoff) and payoff > 0:
            kelly = float(win_rate - (1.0 - win_rate) / payoff)
        else:
            kelly = float("nan")
        return {"win_rate": win_rate, "payoff_ratio": payoff, "kelly_fraction": kelly}

    weekly_wp = _win_payoff_kelly(weekly)
    monthly_wp = _win_payoff_kelly(monthly)
    quarterly_wp = _win_payoff_kelly(quarterly)
    yearly_wp = _win_payoff_kelly(yearly)
    weekly_wp_rp = _win_payoff_kelly(weekly_rp)
    monthly_wp_rp = _win_payoff_kelly(monthly_rp)
    quarterly_wp_rp = _win_payoff_kelly(quarterly_rp)
    yearly_wp_rp = _win_payoff_kelly(yearly_rp)

    def _metrics_from_nav(
        nav_s: pd.Series,
        nav_ret: pd.Series,
        *,
        wp_w: dict[str, float],
        wp_m: dict[str, float],
        wp_q: dict[str, float],
        wp_y: dict[str, float],
    ) -> dict[str, Any]:
        cum_ret = float(nav_s.iloc[-1] / nav_s.iloc[0] - 1.0)
        ann_ret = _annualized_return(nav_s)
        ann_vol = _annualized_vol(nav_ret)
        mdd = _max_drawdown(nav_s)
        mdd_dur = _max_drawdown_duration_days(nav_s)
        sharpe = _sharpe(nav_ret, rf=float(inp.risk_free_rate))
        calmar = float(ann_ret / abs(mdd)) if mdd < 0 else float("nan")
        sortino = _sortino(nav_ret, rf=float(inp.risk_free_rate))
        ir = _information_ratio(nav_ret - bench_ret.fillna(0.0))
        ui = _ulcer_index(nav_s, in_percent=True)
        ui_den = ui / 100.0
        upi = float((ann_ret - float(inp.risk_free_rate)) / ui_den) if ui_den > 0 else float("nan")
        return {
            "benchmark_code": bench_code,
            "rebalance": inp.rebalance,
            "risk_free_rate": float(inp.risk_free_rate),
            "cumulative_return": cum_ret,
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "max_drawdown": mdd,
            "max_drawdown_recovery_days": mdd_dur,
            "sharpe_ratio": sharpe,
            "calmar_ratio": calmar,
            "sortino_ratio": sortino,
            "information_ratio": ir,
            "ulcer_index": ui,
            "ulcer_performance_index": upi,
            # holding (absolute) win/payoff/kelly by period
            "holding_weekly_win_rate": wp_w["win_rate"],
            "holding_weekly_payoff_ratio": wp_w["payoff_ratio"],
            "holding_weekly_kelly_fraction": wp_w["kelly_fraction"],
            "holding_monthly_win_rate": wp_m["win_rate"],
            "holding_monthly_payoff_ratio": wp_m["payoff_ratio"],
            "holding_monthly_kelly_fraction": wp_m["kelly_fraction"],
            "holding_quarterly_win_rate": wp_q["win_rate"],
            "holding_quarterly_payoff_ratio": wp_q["payoff_ratio"],
            "holding_quarterly_kelly_fraction": wp_q["kelly_fraction"],
            "holding_yearly_win_rate": wp_y["win_rate"],
            "holding_yearly_payoff_ratio": wp_y["payoff_ratio"],
            "holding_yearly_kelly_fraction": wp_y["kelly_fraction"],
        }

    metrics_ew = _metrics_from_nav(ew_nav, ew_ret, wp_w=weekly_wp, wp_m=monthly_wp, wp_q=quarterly_wp, wp_y=yearly_wp)
    metrics_rp = _metrics_from_nav(
        rp_nav,
        rp_ret,
        wp_w=weekly_wp_rp,
        wp_m=monthly_wp_rp,
        wp_q=quarterly_wp_rp,
        wp_y=yearly_wp_rp,
    )
    # Backward-compat: top-level metrics remains EW.
    metrics = metrics_ew

    def _rolling_pack(nav_s: pd.Series) -> dict[str, dict[str, Any]]:
        rolling = {"returns": {}, "drawdown": {}, "max_drawdown": {}}
        for weeks in inp.rolling_weeks or []:
            window = weeks * 5
            rolling["returns"][f"{weeks}w"] = (nav_s / nav_s.shift(window) - 1.0).dropna()
            rolling["drawdown"][f"{weeks}w"] = _rolling_drawdown(nav_s, window).dropna()
            # backward-compat (deprecated): previously "rolling max drawdown"
            rolling["max_drawdown"][f"{weeks}w"] = _rolling_max_drawdown(nav_s, window).dropna()
        for months in inp.rolling_months or []:
            window = months * 21
            rolling["returns"][f"{months}m"] = (nav_s / nav_s.shift(window) - 1.0).dropna()
            rolling["drawdown"][f"{months}m"] = _rolling_drawdown(nav_s, window).dropna()
            rolling["max_drawdown"][f"{months}m"] = _rolling_max_drawdown(nav_s, window).dropna()
        for years in inp.rolling_years or []:
            window = years * 252
            rolling["returns"][f"{years}y"] = (nav_s / nav_s.shift(window) - 1.0).dropna()
            rolling["drawdown"][f"{years}y"] = _rolling_drawdown(nav_s, window).dropna()
            rolling["max_drawdown"][f"{years}y"] = _rolling_max_drawdown(nav_s, window).dropna()
        return {
            "returns": {k: {"dates": v.index.date.astype(str).tolist(), "values": v.astype(float).tolist()} for k, v in rolling["returns"].items()},
            "drawdown": {k: {"dates": v.index.date.astype(str).tolist(), "values": v.astype(float).tolist()} for k, v in rolling["drawdown"].items()},
            "max_drawdown": {k: {"dates": v.index.date.astype(str).tolist(), "values": v.astype(float).tolist()} for k, v in rolling["max_drawdown"].items()},
        }

    rolling_by_portfolio = {"EW": _rolling_pack(ew_nav), "RP": _rolling_pack(rp_nav)}

    # package series for UI (plotly expects arrays)
    dates = nav_common.index.date.astype(str).tolist()
    series = {c: nav_common[c].astype(float).fillna(np.nan).tolist() for c in codes if c in nav_common.columns}
    series["EW"] = ew_nav.astype(float).tolist()
    series["RP"] = rp_nav.astype(float).tolist()
    series[f"BENCH:{bench_code}"] = bench_nav.astype(float).tolist()

    # NAV RSI (EW + benchmark), windows configurable
    rsi_windows = [6, 12, 24]
    nav_rsi = {
        "windows": rsi_windows,
        "dates": dates,
        "series": {
            "EW": {str(w): _rsi_wilder(ew_nav, window=int(w)).astype(float).tolist() for w in rsi_windows},
            "RP": {str(w): _rsi_wilder(rp_nav, window=int(w)).astype(float).tolist() for w in rsi_windows},
            f"BENCH:{bench_code}": {str(w): _rsi_wilder(bench_nav, window=int(w)).astype(float).tolist() for w in rsi_windows},
        },
    }
    rolling_out = rolling_by_portfolio["EW"]  # backward-compat

    attribution_ew = _compute_return_risk_contributions(
        asset_ret=ret_common[codes],
        weights=ew_w[codes] if not ew_w.empty else pd.DataFrame(index=ret_common.index, columns=codes),
        total_return=float(metrics_ew["cumulative_return"]),
    )
    attribution_rp = _compute_return_risk_contributions(
        asset_ret=ret_common[codes],
        weights=rp_w[codes] if not rp_w.empty else pd.DataFrame(index=ret_common.index, columns=codes),
        total_return=float(metrics_rp["cumulative_return"]),
    )
    attribution = attribution_ew  # backward-compat

    # FFT analysis (on log returns, using the same adjustment basis as baseline close)
    fft = _fft_analysis(close_ff.loc[common_start:, codes], ew_nav=ew_nav, windows=inp.fft_windows)
    fft_roll = {"ok": False, "reason": "disabled", "windows": fft.get("windows", []), "step": int(inp.fft_roll_step), "series": {}}
    if bool(inp.fft_roll):
        ew = pd.Series(ew_nav).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        ew_lr = np.log(ew).diff().dropna()
        fft_roll = _fft_roll_timeseries_from_returns(
            ew_lr,
            windows=list(fft.get("windows", []) or []),
            step=int(inp.fft_roll_step),
            top_k=5,
        )

    # Compute periodic returns and volatility distributions for each code
    period_distributions = _compute_periodic_returns_and_volatility(
        ret_common,
        codes=codes,
        daily_close=close_ff.loc[common_start:],
        daily_volume=volume_common,
        daily_amount=amount_common,
    )

    # "Mirror" composite deviation indicator time-series (per asset, by period).
    #
    # This is used for UI visualization (distribution panel) and is deliberately "rearview":
    # expanding percentiles only, no rolling-window quantiles (to avoid lookahead).
    #
    # For each selected period (daily/weekly/monthly/quarterly/yearly), build:
    # - log_return_dev: log(1+r_period) - MA20(log(1+r_period))
    # - log_vol_dev: log(vol_period) - MA20(log(vol_period)), where vol_period is a short rolling std on period returns
    # - log_volume_dev: log(activity) - MA20(log(activity)), aggregated by mean within period
    #
    # Then:
    # - percentile each deviation via expanding percentile rank
    # - composite = mean of the 3 percentiles (requires all 3)
    # - mirror = expanding percentile rank of composite
    mirror_timeseries: dict[str, Any] = {}
    close_m = close_ff.loc[common_start:, codes].copy()
    close_m.index = pd.to_datetime(close_m.index)
    vol_m = volume_common.copy() if (volume_common is not None and not volume_common.empty) else pd.DataFrame()
    amt_m = amount_common.copy() if (amount_common is not None and not amount_common.empty) else pd.DataFrame()
    if not vol_m.empty:
        vol_m.index = pd.to_datetime(vol_m.index)
    if not amt_m.empty:
        amt_m.index = pd.to_datetime(amt_m.index)

    def _pack_ts(s: pd.Series) -> dict[str, Any]:
        s2 = pd.to_numeric(s, errors="coerce").astype(float)
        return {"dates": s2.index.date.astype(str).tolist(), "values": s2.tolist()}

    freq_by_period = {"daily": None, "weekly": "W-FRI", "monthly": "ME", "quarterly": "QE", "yearly": "YE"}
    vol_win = {"daily": (20, 5), "weekly": (4, 2), "monthly": (3, 2), "quarterly": (2, 2), "yearly": (2, 2)}

    for code in codes:
        if code not in close_m.columns:
            continue
        px = pd.to_numeric(close_m[code], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if px.empty:
            continue
        r = px.pct_change().replace([np.inf, -np.inf], np.nan)

        act = pd.Series(index=px.index, dtype=float)
        if (not vol_m.empty) and (code in vol_m.columns):
            act = act.combine_first(pd.to_numeric(vol_m[code], errors="coerce").astype(float))
        if (not amt_m.empty) and (code in amt_m.columns):
            act = act.combine_first(pd.to_numeric(amt_m[code], errors="coerce").astype(float))
        act = act.replace([np.inf, -np.inf], np.nan)
        la = np.log(act.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        la_dev_daily = (la - la.rolling(window=20, min_periods=5).mean()).replace([np.inf, -np.inf], np.nan)

        out_by_period: dict[str, Any] = {}
        gross = (1.0 + r).replace([np.inf, -np.inf], np.nan).fillna(1.0).astype(float)
        for per, freq in freq_by_period.items():
            # Period returns
            if freq is None:
                ret_p = pd.to_numeric(r, errors="coerce").astype(float)
            else:
                nav_p = gross.resample(freq).prod()
                ret_p = nav_p.pct_change()
            ret_p = ret_p.replace([np.inf, -np.inf], np.nan).dropna()
            if ret_p.empty:
                continue

            # log-return deviation
            lr_p = np.log1p(ret_p).replace([np.inf, -np.inf], np.nan).dropna()
            lr_dev_p = (lr_p - lr_p.rolling(window=20, min_periods=5).mean()).replace([np.inf, -np.inf], np.nan).dropna()
            if lr_dev_p.empty:
                continue

            # realized vol deviation (vol on period returns)
            wv, mp = vol_win[per]
            vol_p = ret_p.rolling(window=int(wv), min_periods=int(mp)).std(ddof=1).replace([np.inf, -np.inf], np.nan).dropna()
            vol_p = pd.to_numeric(vol_p, errors="coerce").astype(float)
            vol_p = vol_p[vol_p > 0].dropna()
            lv_p = np.log(vol_p).replace([np.inf, -np.inf], np.nan).dropna()
            lv_dev_p = (lv_p - lv_p.rolling(window=20, min_periods=5).mean()).replace([np.inf, -np.inf], np.nan).dropna()
            if lv_dev_p.empty:
                continue

            # activity deviation aggregated by mean within period
            if freq is None:
                la_dev_p = la_dev_daily
            else:
                la_dev_p = la_dev_daily.resample(freq).mean()
            la_dev_p = pd.to_numeric(la_dev_p, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            if la_dev_p.empty:
                continue

            p1 = _expanding_percentile_rank(lr_dev_p)
            p2 = _expanding_percentile_rank(lv_dev_p)
            p3 = _expanding_percentile_rank(la_dev_p)
            comp = pd.concat([p1, p2, p3], axis=1).mean(axis=1, skipna=False).replace([np.inf, -np.inf], np.nan).dropna()
            if comp.empty:
                continue
            mirror = _expanding_percentile_rank(comp)
            out_by_period[per] = _pack_ts(mirror)

        if out_by_period:
            mirror_timeseries[code] = out_by_period

    return {
        "date_range": {"start": inp.start.strftime("%Y%m%d"), "end": inp.end.strftime("%Y%m%d"), "common_start": common_start.date().strftime("%Y%m%d")},
        "codes": codes,
        "nav": {"dates": dates, "series": series},
        "nav_rsi": nav_rsi,
        "period_returns": {
            "weekly": weekly.to_dict(orient="records"),
            "monthly": monthly.to_dict(orient="records"),
            "quarterly": quarterly.to_dict(orient="records"),
            "yearly": yearly.to_dict(orient="records"),
        },
        "metrics": metrics,
        "metrics_by_portfolio": {"EW": metrics_ew, "RP": metrics_rp},
        "correlation": corr_out,
        "rolling": rolling_out,
        "rolling_by_portfolio": rolling_by_portfolio,
        "attribution": attribution,
        "attribution_by_portfolio": {"EW": attribution_ew, "RP": attribution_rp},
        "fft": fft,
        "fft_roll": {"ew": fft_roll},
        "period_distributions": period_distributions,
        "mirror_timeseries": mirror_timeseries,
    }

