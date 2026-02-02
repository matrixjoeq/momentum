from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


VolProxyKind = Literal[
    # close-based
    "rv_close",
    "ewma_close",
    # range-based (needs OHLC)
    "parkinson",
    "garman_klass",
    "rogers_satchell",
    "yang_zhang",
    # forecast (based on realized variance of close)
    "har_rv",
]


@dataclass(frozen=True)
class VolProxySpec:
    """
    Volatility proxy specification.

    All outputs are annualized volatility levels (decimal), i.e. 0.2 = 20% annualized.
    """

    kind: VolProxyKind
    window: int = 20  # rolling window (trading days) for level smoothing
    ann: int = 252  # annualization factor

    # EWMA parameters
    ewma_lambda: float = 0.94

    # HAR forecast parameters
    har_train_window: int = 252
    har_horizons: tuple[int, int, int] = (1, 5, 22)  # daily/weekly/monthly RV means


def _log_return(close: pd.Series) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    c = c.where(c > 0)
    return np.log(c).diff()


def realized_vol_close(
    close: pd.Series,
    *,
    window: int,
    ann: int = 252,
) -> pd.Series:
    r = _log_return(close)
    # Annualized volatility level (std of daily log-returns)
    return r.rolling(int(window)).std(ddof=1) * math.sqrt(float(ann))


def ewma_vol_close(
    close: pd.Series,
    *,
    lam: float = 0.94,
    ann: int = 252,
) -> pd.Series:
    """
    EWMA volatility (RiskMetrics-style) from close log-returns.

    sigma_t^2 = lam*sigma_{t-1}^2 + (1-lam)*r_t^2
    """
    r = _log_return(close)
    lam = float(lam)
    lam = min(max(lam, 0.01), 0.9999)
    r2 = (pd.to_numeric(r, errors="coerce") ** 2).astype(float)

    out = np.full(len(r2), np.nan, dtype=float)
    prev = float("nan")
    for i, v in enumerate(r2.to_numpy(dtype=float)):
        if not np.isfinite(v):
            out[i] = float("nan")
            continue
        if not np.isfinite(prev):
            prev = v
        else:
            prev = lam * prev + (1.0 - lam) * v
        out[i] = math.sqrt(prev * float(ann))
    return pd.Series(out, index=r2.index)


def parkinson_vol(
    high: pd.Series,
    low: pd.Series,
    *,
    window: int,
    ann: int = 252,
) -> pd.Series:
    """
    Parkinson range-based volatility estimator.

    Daily variance: (1/(4 ln 2)) * (ln(H/L))^2
    Rolling mean of daily variance -> annualized vol.
    """
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    x = (h / l).where((h > 0) & (l > 0))
    u2 = (np.log(x) ** 2) / (4.0 * math.log(2.0))
    var = u2.rolling(int(window)).mean()
    return np.sqrt(var * float(ann))


def garman_klass_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    window: int,
    ann: int = 252,
) -> pd.Series:
    """
    Garman-Klass volatility estimator (uses OHLC).
    """
    o = pd.to_numeric(open_, errors="coerce")
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")

    ok = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    o = o.where(ok)
    h = h.where(ok)
    l = l.where(ok)
    c = c.where(ok)

    log_hl = np.log(h / l)
    log_co = np.log(c / o)
    var_d = 0.5 * (log_hl**2) - (2.0 * math.log(2.0) - 1.0) * (log_co**2)
    var = var_d.rolling(int(window)).mean()
    return np.sqrt(var * float(ann))


def rogers_satchell_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    window: int,
    ann: int = 252,
) -> pd.Series:
    """
    Rogers-Satchell volatility estimator (uses OHLC).
    """
    o = pd.to_numeric(open_, errors="coerce")
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")

    ok = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    o = o.where(ok)
    h = h.where(ok)
    l = l.where(ok)
    c = c.where(ok)

    log_ho = np.log(h / o)
    log_lo = np.log(l / o)
    log_co = np.log(c / o)
    var_d = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    var = var_d.rolling(int(window)).mean()
    return np.sqrt(var * float(ann))


def yang_zhang_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    window: int,
    ann: int = 252,
) -> pd.Series:
    """
    Yang-Zhang volatility estimator (uses OHLC).

    This implementation uses:
    - overnight returns: log(O_t / C_{t-1})
    - open-to-close returns: log(C_t / O_t)
    - Rogers-Satchell for intraday
    Combined with a k weighting term.
    """
    w = int(window)
    o = pd.to_numeric(open_, errors="coerce")
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")

    ok = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    o = o.where(ok)
    h = h.where(ok)
    l = l.where(ok)
    c = c.where(ok)

    oc = np.log(o / c.shift(1))
    co = np.log(c / o)

    rs_var_d = (np.log(h / o) * (np.log(h / o) - co)) + (np.log(l / o) * (np.log(l / o) - co))

    # Rolling variances
    var_oc = oc.rolling(w).var(ddof=1)
    var_co = co.rolling(w).var(ddof=1)
    var_rs = rs_var_d.rolling(w).mean()

    k = 0.34 / (1.34 + (w + 1.0) / max(1.0, w - 1.0))
    var = var_oc + k * var_co + (1.0 - k) * var_rs
    return np.sqrt(var * float(ann))


def har_forecast_vol(
    close: pd.Series,
    *,
    train_window: int = 252,
    horizons: tuple[int, int, int] = (1, 5, 22),
    ann: int = 252,
) -> pd.Series:
    """
    HAR-RV style one-step-ahead forecast of realized volatility.

    We fit a linear model on realized variance:
      rv_{t} = b0 + b1*rv_{t-1} + b2*avg(rv_{t-5:t-1}) + b3*avg(rv_{t-22:t-1}) + e_t

    using a rolling training window, then forecast rv_{t+1}.
    Output is annualized volatility level (sqrt(rv_forecast*ann)).
    """
    r = _log_return(close)
    rv = (pd.to_numeric(r, errors="coerce") ** 2).astype(float)  # daily realized variance proxy

    h1, h5, h22 = horizons
    h1 = int(max(1, h1))
    h5 = int(max(2, h5))
    h22 = int(max(5, h22))
    tw = int(max(30, train_window))

    # Features available at time t (to predict rv at time t)
    x1 = rv.shift(1)
    x5 = rv.rolling(h5).mean().shift(1)
    x22 = rv.rolling(h22).mean().shift(1)
    y = rv

    out = np.full(len(rv), np.nan, dtype=float)
    idx = rv.index

    X = np.column_stack(
        [
            np.ones(len(rv)),
            x1.to_numpy(dtype=float),
            x5.to_numpy(dtype=float),
            x22.to_numpy(dtype=float),
        ]
    )
    Y = y.to_numpy(dtype=float)

    for i in range(len(rv)):
        # need enough history AND a full train window
        if i < tw:
            continue
        wsl = slice(i - tw, i)
        Xw = X[wsl]
        Yw = Y[wsl]
        m = np.isfinite(Yw) & np.isfinite(Xw).all(axis=1)
        if int(np.sum(m)) < 20:
            continue
        Xw2 = Xw[m]
        Yw2 = Yw[m]
        try:
            beta, *_ = np.linalg.lstsq(Xw2, Yw2, rcond=None)
        except Exception:
            continue
        # forecast for time i using features at i (which are shifted already)
        Xi = X[i]
        if not np.isfinite(Xi).all():
            continue
        yhat = float(np.dot(Xi, beta))
        if not np.isfinite(yhat) or yhat < 0:
            continue
        out[i] = math.sqrt(yhat * float(ann))

    return pd.Series(out, index=idx)


def compute_vol_proxy_levels(
    ohlc: dict[str, pd.Series],
    *,
    spec: VolProxySpec,
) -> pd.Series:
    """
    Compute an annualized volatility *level* series (decimal) for timing.
    """
    kind = spec.kind
    w = int(max(2, spec.window))
    ann = int(max(1, spec.ann))

    close = ohlc.get("close", pd.Series(dtype=float))
    open_ = ohlc.get("open", pd.Series(dtype=float))
    high = ohlc.get("high", pd.Series(dtype=float))
    low = ohlc.get("low", pd.Series(dtype=float))

    if kind == "rv_close":
        return realized_vol_close(close, window=w, ann=ann)
    if kind == "ewma_close":
        return ewma_vol_close(close, lam=spec.ewma_lambda, ann=ann)
    if kind == "parkinson":
        return parkinson_vol(high, low, window=w, ann=ann)
    if kind == "garman_klass":
        return garman_klass_vol(open_, high, low, close, window=w, ann=ann)
    if kind == "rogers_satchell":
        return rogers_satchell_vol(open_, high, low, close, window=w, ann=ann)
    if kind == "yang_zhang":
        return yang_zhang_vol(open_, high, low, close, window=w, ann=ann)
    if kind == "har_rv":
        # HAR returns already annualized; optionally smooth it with rolling window (w)
        har = har_forecast_vol(close, train_window=spec.har_train_window, horizons=spec.har_horizons, ann=ann)
        return har.rolling(w).mean()

    raise ValueError(f"unknown vol proxy kind={kind}")

