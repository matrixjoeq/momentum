from __future__ import annotations

# pylint: disable=broad-exception-caught

import math
import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import chi2, gennorm, jf_skew_t, kstest, t as t_dist


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
    low_px = pd.to_numeric(low, errors="coerce")
    x = (h / low_px).where((h > 0) & (low_px > 0))
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
    low_px = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")

    ok = (o > 0) & (h > 0) & (low_px > 0) & (c > 0)
    o = o.where(ok)
    h = h.where(ok)
    low_px = low_px.where(ok)
    c = c.where(ok)

    log_hl = np.log(h / low_px)
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
    low_px = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")

    ok = (o > 0) & (h > 0) & (low_px > 0) & (c > 0)
    o = o.where(ok)
    h = h.where(ok)
    low_px = low_px.where(ok)
    c = c.where(ok)

    log_ho = np.log(h / o)
    log_lo = np.log(low_px / o)
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
    low_px = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")

    ok = (o > 0) & (h > 0) & (low_px > 0) & (c > 0)
    o = o.where(ok)
    h = h.where(ok)
    low_px = low_px.where(ok)
    c = c.where(ok)

    oc = np.log(o / c.shift(1))
    co = np.log(c / o)

    rs_var_d = (np.log(h / o) * (np.log(h / o) - co)) + (
        np.log(low_px / o) * (np.log(low_px / o) - co)
    )

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
    rv = (pd.to_numeric(r, errors="coerce") ** 2).astype(
        float
    )  # daily realized variance proxy

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
        har = har_forecast_vol(
            close,
            train_window=spec.har_train_window,
            horizons=spec.har_horizons,
            ann=ann,
        )
        return har.rolling(w).mean()

    raise ValueError(f"unknown vol proxy kind={kind}")


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def _stable_u32_seed(*parts: Any) -> int:
    """
    Build a stable uint32 seed from deterministic text payload.
    """
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _arma_grid_filter(
    series: pd.Series,
    *,
    max_p: int = 3,
    max_q: int = 3,
    arch_lags: int = 10,
) -> dict[str, Any] | None:
    """
    Auto-select ARMA(p,q) mean model by residual diagnostics and BIC.
    Returns selected residual/fitted mean with candidate summaries.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.shape[0] < 40:
        return None
    try:
        from statsmodels.tsa.arima.model import ARIMA  # type: ignore
    except Exception:
        return None

    candidates: list[dict[str, Any]] = []
    for p in range(0, int(max_p) + 1):
        for q in range(0, int(max_q) + 1):
            try:
                fit = ARIMA(
                    s,
                    order=(int(p), 0, int(q)),
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
            except Exception:
                continue
            resid = pd.to_numeric(getattr(fit, "resid", None), errors="coerce").astype(
                float
            )
            fitted = pd.to_numeric(
                getattr(fit, "fittedvalues", None), errors="coerce"
            ).astype(float)
            resid = resid.replace([np.inf, -np.inf], np.nan).dropna()
            fitted = fitted.replace([np.inf, -np.inf], np.nan).reindex(resid.index)
            if resid.shape[0] < 30:
                continue
            resid_arr = resid.to_numpy(dtype=float)
            acf_p = _safe_float(
                _acf_ljung_test(resid_arr, lags=arch_lags).get("pvalue")
            )
            arch_p = _safe_float(_arch_lm_test(resid_arr, lags=arch_lags).get("pvalue"))
            bic = _safe_float(getattr(fit, "bic", None))
            aic = _safe_float(getattr(fit, "aic", None))
            candidates.append(
                {
                    "p": int(p),
                    "q": int(q),
                    "model": f"arma({int(p)},{int(q)})",
                    "resid": resid,
                    "fitted": fitted,
                    "aic": float(aic) if np.isfinite(aic) else None,
                    "bic": float(bic) if np.isfinite(bic) else None,
                    "resid_acf_pvalue": float(acf_p) if np.isfinite(acf_p) else None,
                    "resid_arch_pvalue": float(arch_p) if np.isfinite(arch_p) else None,
                    "acf_pass": bool(np.isfinite(acf_p) and acf_p >= 0.05),
                    "arch_pass": bool(np.isfinite(arch_p) and arch_p >= 0.05),
                }
            )
    if not candidates:
        return None

    def _sort_key(c: dict[str, Any]) -> tuple:
        acf_pass = bool(c.get("acf_pass"))
        arch_pass = bool(c.get("arch_pass"))
        bic = _safe_float(c.get("bic"))
        acf_p = _safe_float(c.get("resid_acf_pvalue"))
        arch_p = _safe_float(c.get("resid_arch_pvalue"))
        return (
            0 if acf_pass else 1,
            0 if arch_pass else 1,
            0 if np.isfinite(bic) else 1,
            float(bic) if np.isfinite(bic) else float("inf"),
            -float(acf_p) if np.isfinite(acf_p) else 1.0,
            -float(arch_p) if np.isfinite(arch_p) else 1.0,
        )

    candidates_sorted = sorted(candidates, key=_sort_key)
    best = candidates_sorted[0]
    return {
        "model": str(best.get("model") or "arma(0,0)"),
        "resid": best["resid"],
        "fitted": best["fitted"],
        "resid_acf_pvalue": best.get("resid_acf_pvalue"),
        "resid_arch_pvalue": best.get("resid_arch_pvalue"),
        "aic": best.get("aic"),
        "bic": best.get("bic"),
        "candidates": [
            {
                "model": c.get("model"),
                "aic": c.get("aic"),
                "bic": c.get("bic"),
                "resid_acf_pvalue": c.get("resid_acf_pvalue"),
                "resid_arch_pvalue": c.get("resid_arch_pvalue"),
                "acf_pass": c.get("acf_pass"),
                "arch_pass": c.get("arch_pass"),
            }
            for c in candidates_sorted
        ],
    }


def _arch_lm_test(series: np.ndarray, *, lags: int) -> dict[str, Any]:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    lag_n = int(max(1, lags))
    n = int(x.size)
    out: dict[str, Any] = {
        "ok": False,
        "lags": lag_n,
        "n_obs": n,
        "stat": None,
        "pvalue": None,
        "significant": None,
    }
    if n <= lag_n + 5:
        return out

    x = x - float(np.mean(x))
    e2 = np.square(x)
    y = e2[lag_n:]
    if y.size <= lag_n:
        return out

    x_cols = [np.ones(int(y.size), dtype=float)]
    for k in range(1, lag_n + 1):
        x_cols.append(e2[lag_n - k : -k])
    design = np.column_stack(x_cols)
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return out
    y_hat = design @ beta
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    if not np.isfinite(sst) or sst <= 0:
        r2 = 0.0
    else:
        ssr = float(np.sum((y - y_hat) ** 2))
        r2 = float(max(0.0, min(1.0, 1.0 - ssr / sst)))
    stat = float(y.size * r2)
    pvalue = float(1.0 - chi2.cdf(stat, df=lag_n))
    out.update(
        {
            "ok": True,
            "stat": stat,
            "pvalue": pvalue,
            "significant": bool(pvalue < 0.05),
            "conclusion": (
                "显著（拒绝无 ARCH 效应原假设）"
                if bool(pvalue < 0.05)
                else "不显著（未拒绝无 ARCH 效应原假设）"
            ),
        }
    )
    return out


def _jarque_bera_test(series: np.ndarray) -> dict[str, Any]:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    out: dict[str, Any] = {
        "ok": False,
        "n_obs": n,
        "stat": None,
        "pvalue": None,
        "significant": None,
        "conclusion": "样本不足",
    }
    if n < 8:
        return out
    mean = float(np.mean(x))
    cen = x - mean
    m2 = float(np.mean(cen**2))
    if (not np.isfinite(m2)) or m2 <= 0:
        return out
    m3 = float(np.mean(cen**3))
    m4 = float(np.mean(cen**4))
    skew = float(m3 / (m2**1.5))
    kurt = float(m4 / (m2**2))
    stat = float((n / 6.0) * (skew * skew + ((kurt - 3.0) ** 2) / 4.0))
    pvalue = float(1.0 - chi2.cdf(stat, df=2))
    sig = bool(pvalue < 0.05)
    out.update(
        {
            "ok": True,
            "stat": stat,
            "pvalue": pvalue,
            "significant": sig,
            "conclusion": (
                "显著（拒绝正态分布原假设）"
                if sig
                else "不显著（未拒绝正态分布原假设）"
            ),
        }
    )
    return out


def _adf_stationarity_test(series: np.ndarray) -> dict[str, Any]:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    out: dict[str, Any] = {
        "ok": False,
        "n_obs": n,
        "stat": None,
        "pvalue": None,
        "significant": None,
        "conclusion": "样本不足或依赖不可用",
    }
    if n < 20:
        return out
    try:
        from arch.unitroot import ADF as _ArchADF  # type: ignore
    except Exception:
        return out
    try:
        adf_res = _ArchADF(x, trend="c")
        stat = _safe_float(getattr(adf_res, "stat", None))
        pvalue = _safe_float(getattr(adf_res, "pvalue", None))
        sig = bool(np.isfinite(pvalue) and pvalue < 0.05)
        out.update(
            {
                "ok": bool(np.isfinite(stat) and np.isfinite(pvalue)),
                "stat": float(stat) if np.isfinite(stat) else None,
                "pvalue": float(pvalue) if np.isfinite(pvalue) else None,
                "significant": sig if np.isfinite(pvalue) else None,
                "conclusion": (
                    "显著（拒绝单位根原假设，序列更可能平稳）"
                    if sig
                    else "不显著（未拒绝单位根原假设，序列平稳性较弱）"
                )
                if np.isfinite(pvalue)
                else "统计量不可用",
            }
        )
        return out
    except Exception:
        return out


def _acf_ljung_test(series: np.ndarray, *, lags: int) -> dict[str, Any]:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    lag_n = int(max(1, lags))
    n = int(x.size)
    out: dict[str, Any] = {
        "ok": False,
        "lags": lag_n,
        "n_obs": n,
        "stat": None,
        "pvalue": None,
        "significant": None,
        "max_abs_acf": None,
        "conclusion": "样本不足",
    }
    if n <= lag_n + 5:
        return out
    x = x - float(np.mean(x))
    denom = float(np.dot(x, x))
    if (not np.isfinite(denom)) or denom <= 0:
        return out
    acf_vals: list[float] = []
    for k in range(1, lag_n + 1):
        num = float(np.dot(x[k:], x[:-k]))
        acf_vals.append(float(num / denom))
    q_acc = 0.0
    q_stat = float("nan")
    for k, v in enumerate(acf_vals, start=1):
        denom_k = float(n - k)
        if denom_k <= 0:
            continue
        q_acc += (float(v) * float(v)) / denom_k
        q_stat = float(n * (n + 2) * q_acc)
    pvalue = (
        float(1.0 - chi2.cdf(q_stat, df=lag_n)) if np.isfinite(q_stat) else float("nan")
    )
    sig = bool(np.isfinite(pvalue) and pvalue < 0.05)
    max_abs_acf = (
        float(np.max(np.abs(np.asarray(acf_vals, dtype=float))))
        if acf_vals
        else float("nan")
    )
    out.update(
        {
            "ok": bool(np.isfinite(q_stat) and np.isfinite(pvalue)),
            "stat": float(q_stat) if np.isfinite(q_stat) else None,
            "pvalue": float(pvalue) if np.isfinite(pvalue) else None,
            "significant": sig if np.isfinite(pvalue) else None,
            "max_abs_acf": float(max_abs_acf) if np.isfinite(max_abs_acf) else None,
            "conclusion": (
                "显著（拒绝收益率无自相关原假设）"
                if sig
                else "不显著（未拒绝收益率无自相关原假设）"
            )
            if np.isfinite(pvalue)
            else "统计量不可用",
        }
    )
    return out


def _student_t_gof_test(series: np.ndarray) -> dict[str, Any]:
    """
    Student-t goodness-of-fit test (estimated-parameter KS).
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    out: dict[str, Any] = {
        "ok": False,
        "n_obs": n,
        "df": None,
        "loc": None,
        "scale": None,
        "stat": None,
        "pvalue": None,
        "pvalue_asymptotic": None,
        "significant": None,
        "method": "ks_bootstrap_with_fitted_t",
        "bootstrap_reps": 0,
        "bootstrap_valid_reps": 0,
        "conclusion": "样本不足",
        "note": "",
    }
    if n < 30:
        out["note"] = "样本偏少，结论可靠性较低"
        return out
    try:
        df_hat, loc_hat, scale_hat = t_dist.fit(x)
    except Exception:
        out["conclusion"] = "t 分布拟合失败"
        return out
    df_v = _safe_float(df_hat)
    loc_v = _safe_float(loc_hat)
    scale_v = _safe_float(scale_hat)
    if (
        (not np.isfinite(df_v))
        or (not np.isfinite(loc_v))
        or (not np.isfinite(scale_v))
    ):
        out["conclusion"] = "t 分布参数不可用"
        return out
    if scale_v <= 0.0:
        out["conclusion"] = "t 分布参数异常"
        out.update(
            {
                "df": float(df_v) if np.isfinite(df_v) else None,
                "loc": float(loc_v) if np.isfinite(loc_v) else None,
                "scale": float(scale_v) if np.isfinite(scale_v) else None,
            }
        )
        return out
    try:
        ks_stat, ks_pvalue = kstest(x, "t", args=(df_v, loc_v, scale_v))
    except Exception:
        out["conclusion"] = "t 分布 KS 检验失败"
        out.update(
            {
                "df": float(df_v),
                "loc": float(loc_v),
                "scale": float(scale_v),
            }
        )
        return out
    # Parametric bootstrap calibration for KS with estimated parameters.
    # This avoids optimistic p-values from asymptotic KS under fitted distributions.
    rng_seed = _stable_u32_seed(
        "t_gof",
        f"{float(df_v):.6f}",
        f"{float(loc_v):.6f}",
        f"{float(scale_v):.6f}",
        int(n),
    )
    rng = np.random.default_rng(rng_seed)
    reps = 200
    valid_reps = 0
    exceed = 0
    ks_obs = float(ks_stat) if np.isfinite(ks_stat) else float("nan")
    for _ in range(reps):
        try:
            sim = t_dist.rvs(df_v, loc=loc_v, scale=scale_v, size=n, random_state=rng)
            sim = np.asarray(sim, dtype=float)
            sim = sim[np.isfinite(sim)]
            if sim.size < max(30, n // 2):
                continue
            b_df, b_loc, b_scale = t_dist.fit(sim)
            b_df = _safe_float(b_df)
            b_loc = _safe_float(b_loc)
            b_scale = _safe_float(b_scale)
            if (
                (not np.isfinite(b_df))
                or (not np.isfinite(b_loc))
                or (not np.isfinite(b_scale))
                or b_scale <= 0.0
            ):
                continue
            b_stat, _ = kstest(sim, "t", args=(b_df, b_loc, b_scale))
            if not np.isfinite(b_stat):
                continue
            valid_reps += 1
            if float(b_stat) >= ks_obs:
                exceed += 1
        except Exception:
            continue
    if valid_reps > 0:
        p_boot = float((exceed + 1.0) / (valid_reps + 1.0))
    else:
        p_boot = float("nan")
    p_use = p_boot if np.isfinite(p_boot) else float(ks_pvalue)
    sig = bool(np.isfinite(p_use) and p_use < 0.05)
    note_parts: list[str] = []
    if df_v <= 2.0:
        note_parts.append("拟合 df<=2（尾部极厚），有限样本下结论可靠性较低")
    if valid_reps < max(50, reps // 4):
        note_parts.append("bootstrap 有效重采样次数偏少，结论稳定性有限")
    if not note_parts:
        note_parts.append("bootstrap 已用于校正 KS p 值")
    out.update(
        {
            "ok": bool(np.isfinite(ks_stat) and np.isfinite(p_use)),
            "df": float(df_v),
            "loc": float(loc_v),
            "scale": float(scale_v),
            "stat": float(ks_stat) if np.isfinite(ks_stat) else None,
            "pvalue": float(p_use) if np.isfinite(p_use) else None,
            "pvalue_asymptotic": float(ks_pvalue) if np.isfinite(ks_pvalue) else None,
            "bootstrap_reps": int(reps),
            "bootstrap_valid_reps": int(valid_reps),
            "significant": sig if np.isfinite(p_use) else None,
            "conclusion": (
                "显著（拒绝服从 t 分布原假设）"
                if sig
                else "不显著（未拒绝服从 t 分布原假设）"
            )
            if np.isfinite(p_use)
            else "统计量不可用",
            "note": "；".join(note_parts),
        }
    )
    return out


def _ged_gof_test(series: np.ndarray) -> dict[str, Any]:
    """
    GED (Generalized Error Distribution, exponential power) goodness-of-fit test
    with parametric-bootstrap calibrated KS p-value.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    out: dict[str, Any] = {
        "ok": False,
        "n_obs": n,
        "beta": None,
        "loc": None,
        "scale": None,
        "stat": None,
        "pvalue": None,
        "pvalue_asymptotic": None,
        "significant": None,
        "method": "ks_bootstrap_with_fitted_ged",
        "bootstrap_reps": 0,
        "bootstrap_valid_reps": 0,
        "conclusion": "样本不足",
        "note": "",
    }
    if n < 30:
        out["note"] = "样本偏少，结论可靠性较低"
        return out
    try:
        beta_hat, loc_hat, scale_hat = gennorm.fit(x)
    except Exception:
        out["conclusion"] = "GED 拟合失败"
        return out
    beta_v = _safe_float(beta_hat)
    loc_v = _safe_float(loc_hat)
    scale_v = _safe_float(scale_hat)
    if (
        (not np.isfinite(beta_v))
        or (not np.isfinite(loc_v))
        or (not np.isfinite(scale_v))
    ):
        out["conclusion"] = "GED 参数不可用"
        return out
    if beta_v <= 0.0 or scale_v <= 0.0:
        out["conclusion"] = "GED 参数异常"
        out.update(
            {
                "beta": float(beta_v) if np.isfinite(beta_v) else None,
                "loc": float(loc_v) if np.isfinite(loc_v) else None,
                "scale": float(scale_v) if np.isfinite(scale_v) else None,
            }
        )
        return out
    try:
        ks_stat, ks_pvalue = kstest(x, "gennorm", args=(beta_v, loc_v, scale_v))
    except Exception:
        out["conclusion"] = "GED KS 检验失败"
        out.update(
            {
                "beta": float(beta_v),
                "loc": float(loc_v),
                "scale": float(scale_v),
            }
        )
        return out

    rng_seed = _stable_u32_seed(
        "ged_gof",
        f"{float(beta_v):.6f}",
        f"{float(loc_v):.6f}",
        f"{float(scale_v):.6f}",
        int(n),
    )
    rng = np.random.default_rng(rng_seed)
    reps = 200
    valid_reps = 0
    exceed = 0
    ks_obs = float(ks_stat) if np.isfinite(ks_stat) else float("nan")
    for _ in range(reps):
        try:
            sim = gennorm.rvs(
                beta_v, loc=loc_v, scale=scale_v, size=n, random_state=rng
            )
            sim = np.asarray(sim, dtype=float)
            sim = sim[np.isfinite(sim)]
            if sim.size < max(30, n // 2):
                continue
            b_beta, b_loc, b_scale = gennorm.fit(sim)
            b_beta = _safe_float(b_beta)
            b_loc = _safe_float(b_loc)
            b_scale = _safe_float(b_scale)
            if (
                (not np.isfinite(b_beta))
                or (not np.isfinite(b_loc))
                or (not np.isfinite(b_scale))
                or b_beta <= 0.0
                or b_scale <= 0.0
            ):
                continue
            b_stat, _ = kstest(sim, "gennorm", args=(b_beta, b_loc, b_scale))
            if not np.isfinite(b_stat):
                continue
            valid_reps += 1
            if float(b_stat) >= ks_obs:
                exceed += 1
        except Exception:
            continue
    if valid_reps > 0:
        p_boot = float((exceed + 1.0) / (valid_reps + 1.0))
    else:
        p_boot = float("nan")
    p_use = p_boot if np.isfinite(p_boot) else float(ks_pvalue)
    sig = bool(np.isfinite(p_use) and p_use < 0.05)
    note_parts: list[str] = []
    if valid_reps < max(50, reps // 4):
        note_parts.append("bootstrap 有效重采样次数偏少，结论稳定性有限")
    if not note_parts:
        note_parts.append("bootstrap 已用于校正 KS p 值")
    out.update(
        {
            "ok": bool(np.isfinite(ks_stat) and np.isfinite(p_use)),
            "beta": float(beta_v),
            "loc": float(loc_v),
            "scale": float(scale_v),
            "stat": float(ks_stat) if np.isfinite(ks_stat) else None,
            "pvalue": float(p_use) if np.isfinite(p_use) else None,
            "pvalue_asymptotic": float(ks_pvalue) if np.isfinite(ks_pvalue) else None,
            "bootstrap_reps": int(reps),
            "bootstrap_valid_reps": int(valid_reps),
            "significant": sig if np.isfinite(p_use) else None,
            "conclusion": (
                "显著（拒绝服从 GED 分布原假设）"
                if sig
                else "不显著（未拒绝服从 GED 分布原假设）"
            )
            if np.isfinite(p_use)
            else "统计量不可用",
            "note": "；".join(note_parts),
        }
    )
    return out


def _skew_t_gof_test(series: np.ndarray) -> dict[str, Any]:
    """
    Jones-Faddy skew-t goodness-of-fit test with bootstrap-calibrated KS p-value.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    out: dict[str, Any] = {
        "ok": False,
        "n_obs": n,
        "a": None,
        "b": None,
        "loc": None,
        "scale": None,
        "stat": None,
        "pvalue": None,
        "pvalue_asymptotic": None,
        "significant": None,
        "method": "ks_bootstrap_with_fitted_skew_t",
        "bootstrap_reps": 0,
        "bootstrap_valid_reps": 0,
        "conclusion": "样本不足",
        "note": "",
    }
    if n < 30:
        out["note"] = "样本偏少，结论可靠性较低"
        return out
    try:
        a_hat, b_hat, loc_hat, scale_hat = jf_skew_t.fit(x)
    except Exception:
        out["conclusion"] = "skew-t 拟合失败"
        return out
    a_v = _safe_float(a_hat)
    b_v = _safe_float(b_hat)
    loc_v = _safe_float(loc_hat)
    scale_v = _safe_float(scale_hat)
    if (
        (not np.isfinite(a_v))
        or (not np.isfinite(b_v))
        or (not np.isfinite(loc_v))
        or (not np.isfinite(scale_v))
    ):
        out["conclusion"] = "skew-t 参数不可用"
        return out
    if a_v <= 0.0 or b_v <= 0.0 or scale_v <= 0.0:
        out["conclusion"] = "skew-t 参数异常"
        out.update(
            {
                "a": float(a_v) if np.isfinite(a_v) else None,
                "b": float(b_v) if np.isfinite(b_v) else None,
                "loc": float(loc_v) if np.isfinite(loc_v) else None,
                "scale": float(scale_v) if np.isfinite(scale_v) else None,
            }
        )
        return out
    try:
        ks_stat, ks_pvalue = kstest(x, "jf_skew_t", args=(a_v, b_v, loc_v, scale_v))
    except Exception:
        out["conclusion"] = "skew-t KS 检验失败"
        out.update(
            {
                "a": float(a_v),
                "b": float(b_v),
                "loc": float(loc_v),
                "scale": float(scale_v),
            }
        )
        return out

    rng_seed = _stable_u32_seed(
        "skew_t_gof",
        f"{float(a_v):.6f}",
        f"{float(b_v):.6f}",
        f"{float(loc_v):.6f}",
        f"{float(scale_v):.6f}",
        int(n),
    )
    rng = np.random.default_rng(rng_seed)
    reps = 200
    valid_reps = 0
    exceed = 0
    ks_obs = float(ks_stat) if np.isfinite(ks_stat) else float("nan")
    for _ in range(reps):
        try:
            sim = jf_skew_t.rvs(
                a_v, b_v, loc=loc_v, scale=scale_v, size=n, random_state=rng
            )
            sim = np.asarray(sim, dtype=float)
            sim = sim[np.isfinite(sim)]
            if sim.size < max(30, n // 2):
                continue
            s_a, s_b, s_loc, s_scale = jf_skew_t.fit(sim)
            s_a = _safe_float(s_a)
            s_b = _safe_float(s_b)
            s_loc = _safe_float(s_loc)
            s_scale = _safe_float(s_scale)
            if (
                (not np.isfinite(s_a))
                or (not np.isfinite(s_b))
                or (not np.isfinite(s_loc))
                or (not np.isfinite(s_scale))
                or s_a <= 0.0
                or s_b <= 0.0
                or s_scale <= 0.0
            ):
                continue
            s_stat, _ = kstest(sim, "jf_skew_t", args=(s_a, s_b, s_loc, s_scale))
            if not np.isfinite(s_stat):
                continue
            valid_reps += 1
            if float(s_stat) >= ks_obs:
                exceed += 1
        except Exception:
            continue
    p_boot = (
        float((exceed + 1.0) / (valid_reps + 1.0)) if valid_reps > 0 else float("nan")
    )
    p_use = p_boot if np.isfinite(p_boot) else float(ks_pvalue)
    sig = bool(np.isfinite(p_use) and p_use < 0.05)
    note_parts: list[str] = []
    if valid_reps < max(50, reps // 4):
        note_parts.append("bootstrap 有效重采样次数偏少，结论稳定性有限")
    if not note_parts:
        note_parts.append("bootstrap 已用于校正 KS p 值")
    out.update(
        {
            "ok": bool(np.isfinite(ks_stat) and np.isfinite(p_use)),
            "a": float(a_v),
            "b": float(b_v),
            "loc": float(loc_v),
            "scale": float(scale_v),
            "stat": float(ks_stat) if np.isfinite(ks_stat) else None,
            "pvalue": float(p_use) if np.isfinite(p_use) else None,
            "pvalue_asymptotic": float(ks_pvalue) if np.isfinite(ks_pvalue) else None,
            "bootstrap_reps": int(reps),
            "bootstrap_valid_reps": int(valid_reps),
            "significant": sig if np.isfinite(p_use) else None,
            "conclusion": (
                "显著（拒绝服从 skew-t 分布原假设）"
                if sig
                else "不显著（未拒绝服从 skew-t 分布原假设）"
            )
            if np.isfinite(p_use)
            else "统计量不可用",
            "note": "；".join(note_parts),
        }
    )
    return out


def _var_hit_backtest(
    *,
    returns_scaled: np.ndarray,
    mu_scaled: float | np.ndarray,
    cond_vol_scaled: np.ndarray,
    nu: float,
    alpha: float,
) -> dict[str, Any]:
    r = np.asarray(returns_scaled, dtype=float)
    s = np.asarray(cond_vol_scaled, dtype=float)
    mu_arr = np.asarray(mu_scaled, dtype=float)
    if mu_arr.ndim == 0:
        mu_vec = np.full(int(r.size), float(mu_arr), dtype=float)
    else:
        mu_vec = mu_arr.reshape(-1)
    k = int(min(r.size, s.size, mu_vec.size))
    out: dict[str, Any] = {
        "ok": False,
        "alpha": float(alpha),
        "n_obs": int(k),
        "expected_hit_rate": float(alpha),
        "hit_count": 0,
        "hit_rate": None,
        "var_mean": None,
        "kupiec_lr_uc": None,
        "kupiec_pvalue": None,
        "kupiec_reject": None,
        "christoffersen_lr_ind": None,
        "christoffersen_pvalue": None,
        "christoffersen_reject": None,
        "conclusion": "样本不足",
    }
    if k < 60 or (not np.isfinite(alpha)) or alpha <= 0.0 or alpha >= 0.5:
        return out
    r = r[-k:]
    s = s[-k:]
    mu_vec = mu_vec[-k:]
    mask = np.isfinite(r) & np.isfinite(s) & np.isfinite(mu_vec) & (s > 0.0)
    if int(np.sum(mask)) < 60:
        return out
    r = r[mask]
    s = s[mask]
    mu_vec = mu_vec[mask]
    k = int(r.size)

    if np.isfinite(nu) and nu > 2.0:
        scale_z = float(np.sqrt(nu / (nu - 2.0)))
        q_alpha = float(t_dist.ppf(alpha, df=float(nu)) / scale_z)
    else:
        z = (r - mu_vec) / s
        z = z[np.isfinite(z)]
        if z.size < 60:
            return out
        q_alpha = float(np.quantile(z, alpha))
    var_series = mu_vec + s * q_alpha
    hit = (r < var_series).astype(int)
    hit_n = int(np.sum(hit))
    hit_rate = float(hit_n / k) if k > 0 else float("nan")

    # Kupiec unconditional coverage test
    p0 = float(alpha)
    ph = float(hit_rate) if np.isfinite(hit_rate) else float("nan")
    lr_uc = float("nan")
    p_uc = float("nan")
    if np.isfinite(ph) and 0.0 < ph < 1.0 and k > 0:
        ll0 = (k - hit_n) * np.log(max(1e-12, 1.0 - p0)) + hit_n * np.log(
            max(1e-12, p0)
        )
        ll1 = (k - hit_n) * np.log(max(1e-12, 1.0 - ph)) + hit_n * np.log(
            max(1e-12, ph)
        )
        lr_uc = float(max(0.0, -2.0 * (ll0 - ll1)))
        p_uc = float(1.0 - chi2.cdf(lr_uc, df=1))

    # Christoffersen independence test
    n00 = n01 = n10 = n11 = 0
    for i in range(1, k):
        prev = int(hit[i - 1])
        cur = int(hit[i])
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1
    den0 = n00 + n01
    den1 = n10 + n11
    pi01 = (n01 / den0) if den0 > 0 else float("nan")
    pi11 = (n11 / den1) if den1 > 0 else float("nan")
    den = n00 + n01 + n10 + n11
    pi = ((n01 + n11) / den) if den > 0 else float("nan")
    lr_ind = float("nan")
    p_ind = float("nan")
    if (
        np.isfinite(pi01)
        and np.isfinite(pi11)
        and np.isfinite(pi)
        and 0.0 < pi01 < 1.0
        and 0.0 < pi11 < 1.0
        and 0.0 < pi < 1.0
    ):
        ll_ind0 = (n00 + n10) * np.log(max(1e-12, 1.0 - pi)) + (n01 + n11) * np.log(
            max(1e-12, pi)
        )
        ll_ind1 = n00 * np.log(max(1e-12, 1.0 - pi01)) + n01 * np.log(max(1e-12, pi01))
        ll_ind1 += n10 * np.log(max(1e-12, 1.0 - pi11)) + n11 * np.log(max(1e-12, pi11))
        lr_ind = float(max(0.0, -2.0 * (ll_ind0 - ll_ind1)))
        p_ind = float(1.0 - chi2.cdf(lr_ind, df=1))

    uc_reject = bool(np.isfinite(p_uc) and p_uc < 0.05)
    ind_reject = bool(np.isfinite(p_ind) and p_ind < 0.05)
    out.update(
        {
            "ok": True,
            "n_obs": int(k),
            "hit_count": int(hit_n),
            "hit_rate": float(hit_rate) if np.isfinite(hit_rate) else None,
            "var_mean": float(np.mean(var_series))
            if np.isfinite(np.mean(var_series))
            else None,
            "kupiec_lr_uc": float(lr_uc) if np.isfinite(lr_uc) else None,
            "kupiec_pvalue": float(p_uc) if np.isfinite(p_uc) else None,
            "kupiec_reject": uc_reject if np.isfinite(p_uc) else None,
            "christoffersen_lr_ind": float(lr_ind) if np.isfinite(lr_ind) else None,
            "christoffersen_pvalue": float(p_ind) if np.isfinite(p_ind) else None,
            "christoffersen_reject": ind_reject if np.isfinite(p_ind) else None,
            "conclusion": (
                "命中率与独立性均通过（回测表现较好）"
                if (
                    not uc_reject
                    and not ind_reject
                    and np.isfinite(p_uc)
                    and np.isfinite(p_ind)
                )
                else "命中率或独立性未通过（需谨慎使用）"
            ),
        }
    )
    return out


def _model_value_assessment(
    *,
    n_obs: int,
    converged: bool,
    persistence: float,
    pre_arch_pvalue: float,
    post_arch_pvalue: float,
    nu: float,
) -> dict[str, Any]:
    score = 0.0
    reasons: list[str] = []

    if converged:
        score += 0.30
        reasons.append("优化收敛，参数估计可用")
    else:
        reasons.append("优化未收敛，模型稳定性不足")

    if n_obs >= 252:
        score += 0.25
        reasons.append("样本长度达到 1 年以上（日频）")
    elif n_obs >= 126:
        score += 0.15
        reasons.append("样本长度达到半年以上（日频）")
    else:
        reasons.append("样本长度偏短，统计置信度受限")

    if np.isfinite(persistence):
        if 0.70 <= persistence < 0.99:
            score += 0.25
            reasons.append("波动持续性位于常见区间（0.70~0.99）")
        elif 0.40 <= persistence < 1.02:
            score += 0.15
            reasons.append("波动持续性可解释但边际稳定性一般")
        else:
            reasons.append("波动持续性异常，需谨慎解释")
    else:
        reasons.append("无法计算波动持续性")

    if np.isfinite(pre_arch_pvalue):
        if pre_arch_pvalue < 0.05:
            score += 0.10
            reasons.append("原始收益存在显著 ARCH 效应（适合 GARCH 类模型）")
        else:
            reasons.append("原始收益未检出显著 ARCH 效应")

    if np.isfinite(post_arch_pvalue):
        if post_arch_pvalue >= 0.05:
            score += 0.10
            reasons.append("标准化残差未检出显著 ARCH 效应（拟合解释较充分）")
        else:
            reasons.append("标准化残差仍有 ARCH 效应（拟合仍有剩余结构）")

    if np.isfinite(nu):
        if 4.0 <= nu <= 80.0:
            score += 0.05
            reasons.append("Student-t 自由度在可解释区间")
        else:
            reasons.append("Student-t 自由度异常，尾部拟合需谨慎")

    score = float(max(0.0, min(1.0, score)))
    if score >= 0.75:
        level = "high"
        summary = "模型诊断表现较好，可作为波动率近似的高价值参考。"
    elif score >= 0.45:
        level = "medium"
        summary = "模型可提供方向性参考，但需结合其他指标交叉验证。"
    else:
        level = "low"
        summary = "模型稳定性或统计显著性不足，参考价值有限。"
    return {
        "model_value": level,
        "value_score": score,
        "summary": summary,
        "reasons": reasons,
    }


def compute_gjr_garch_volatility(
    close: pd.Series,
    *,
    ann_factor: int = 252,
    max_points: int = 1200,
    min_samples: int = 120,
    return_scale: float = 100.0,
    arch_lags: int = 10,
    include_model_comparison: bool = False,
    arch_model_factory: Any | None = None,
) -> dict[str, Any]:
    """
    Fit GJR-GARCH(1,1) with Student-t innovations on close log returns.

    Six-model comparison (APARCH/EGARCH/GJR-GARCH × t/skew-t) runs only when
    ``include_model_comparison`` is true.

    Returns a structured dict with:
    - params
    - diagnostics
    - interpretation
    - aligned price/volatility series for plotting
    """

    ann = int(max(1, ann_factor))
    max_points_i = int(max(0, max_points))
    min_samples_i = int(max(20, min_samples))
    arch_lags_i = int(max(1, arch_lags))
    scale = _safe_float(return_scale)
    if not np.isfinite(scale) or scale <= 0:
        scale = 100.0

    close_raw = pd.to_numeric(close, errors="coerce").astype(float)
    n_raw = int(close_raw.shape[0])
    close_pos = close_raw.replace([np.inf, -np.inf], np.nan).where(close_raw > 0.0)
    close_clean = close_pos.dropna()
    n_close = int(close_clean.shape[0])
    dropped = int(max(0, n_raw - n_close))
    ret = np.log(close_clean).diff().replace([np.inf, -np.inf], np.nan).dropna()
    if max_points_i > 0 and len(ret) > max_points_i:
        ret = ret.iloc[-max_points_i:]
    n_ret = int(ret.shape[0])

    base_meta = {
        "n_obs_raw": n_raw,
        "n_obs_price": n_close,
        "n_obs_returns": n_ret,
        "dropped_obs": dropped,
        "min_samples": min_samples_i,
        "max_points": max_points_i,
    }
    if n_ret < min_samples_i:
        return {"ok": False, "error": "insufficient_samples", "meta": base_meta}

    if arch_model_factory is None:
        try:
            # Lazy import: keep app startup independent from optional arch package.
            from arch import arch_model as _arch_model_factory
        except ModuleNotFoundError:
            return {"ok": False, "error": "dependency_unavailable", "meta": base_meta}
        arch_model_factory = _arch_model_factory
    if not callable(arch_model_factory):
        return {"ok": False, "error": "dependency_unavailable", "meta": base_meta}

    ret_arr = ret.to_numpy(dtype=float)
    pre_arch = _arch_lm_test(ret_arr, lags=arch_lags_i)
    adf_ret = _adf_stationarity_test(ret_arr)
    acf_pre = _acf_ljung_test(ret_arr, lags=arch_lags_i)
    jb_pre = _jarque_bera_test(ret_arr)
    t_pre = _student_t_gof_test(ret_arr)
    ged_pre = _ged_gof_test(ret_arr)
    skew_t_pre = _skew_t_gof_test(ret_arr)
    ret_scaled_raw = ret * float(scale)
    arma_sel = _arma_grid_filter(
        ret_scaled_raw, max_p=3, max_q=3, arch_lags=arch_lags_i
    )
    if arma_sel is None:
        return {
            "ok": False,
            "error": "fit_failed",
            "meta": {**base_meta, "reason": "arma_grid_failed"},
        }
    arma_resid_scaled = arma_sel["resid"]
    arma_mean_scaled = arma_sel["fitted"]
    n_model = int(arma_resid_scaled.shape[0])
    base_meta["mean_model"] = str(arma_sel.get("model") or "arma(0,0)")
    base_meta["n_obs_model"] = n_model
    base_meta["mean_model_selection"] = {
        "selected": str(arma_sel.get("model") or "arma(0,0)"),
        "selected_aic": arma_sel.get("aic"),
        "selected_bic": arma_sel.get("bic"),
        "selected_resid_acf_pvalue": arma_sel.get("resid_acf_pvalue"),
        "selected_resid_arch_pvalue": arma_sel.get("resid_arch_pvalue"),
        "candidate_count": len(arma_sel.get("candidates") or []),
    }
    if n_model < min_samples_i:
        meta = dict(base_meta)
        meta["reason"] = "insufficient_samples_after_arma_grid"
        return {"ok": False, "error": "insufficient_samples", "meta": meta}
    ret_scaled = arma_resid_scaled

    try:
        try:
            model = arch_model_factory(
                ret_scaled,
                mean="Zero",
                vol="GARCH",
                p=1,
                o=1,
                q=1,
                dist="t",
                rescale=False,
            )
        except TypeError:
            model = arch_model_factory(
                ret_scaled,
                mean="Zero",
                vol="GARCH",
                p=1,
                o=1,
                q=1,
                dist="t",
            )
        try:
            fit = model.fit(disp="off", show_warning=False)
        except TypeError:
            try:
                fit = model.fit(disp="off")
            except TypeError:
                fit = model.fit()
    except Exception as exc:  # pragma: no cover - depends on 3rd-party runtime
        meta = dict(base_meta)
        meta["reason"] = str(exc)
        return {"ok": False, "error": "fit_failed", "meta": meta}

    conv_flag = int(getattr(fit, "convergence_flag", 0) or 0)
    converged = conv_flag == 0
    if not converged:
        meta = dict(base_meta)
        meta["convergence_flag"] = conv_flag
        return {"ok": False, "error": "fit_failed", "meta": meta}

    params = getattr(fit, "params", {})
    omega = _safe_float(params.get("omega") if hasattr(params, "get") else None)
    alpha1 = _safe_float(params.get("alpha[1]") if hasattr(params, "get") else None)
    gamma1 = _safe_float(params.get("gamma[1]") if hasattr(params, "get") else None)
    beta1 = _safe_float(params.get("beta[1]") if hasattr(params, "get") else None)
    mu = 0.0
    nu = _safe_float(params.get("nu") if hasattr(params, "get") else None)
    persistence = (
        float(alpha1 + beta1 + 0.5 * gamma1)
        if np.isfinite(alpha1) and np.isfinite(beta1) and np.isfinite(gamma1)
        else float("nan")
    )
    uncond_var_daily = float("nan")
    if np.isfinite(omega) and np.isfinite(persistence) and 0 < persistence < 1:
        denom = 1.0 - persistence
        if denom > 1e-10:
            uncond_var_daily = float(omega / denom / (scale**2))
    uncond_vol_daily = (
        float(math.sqrt(uncond_var_daily))
        if np.isfinite(uncond_var_daily) and uncond_var_daily >= 0
        else float("nan")
    )
    uncond_vol_ann = (
        float(uncond_vol_daily * math.sqrt(float(ann)))
        if np.isfinite(uncond_vol_daily)
        else float("nan")
    )

    cond_vol_scaled = np.asarray(
        getattr(fit, "conditional_volatility", []),
        dtype=float,
    )
    if cond_vol_scaled.size <= 0:
        return {"ok": False, "error": "fit_failed", "meta": base_meta}
    cond_vol_scaled = cond_vol_scaled[-n_model:]
    cond_vol_daily = cond_vol_scaled / float(scale)
    cond_vol_ann = cond_vol_daily * math.sqrt(float(ann))

    std_resid = np.asarray(getattr(fit, "std_resid", []), dtype=float)
    if std_resid.size <= 0:
        resid = np.asarray(getattr(fit, "resid", []), dtype=float)
        if resid.size > 0 and cond_vol_scaled.size > 0:
            k = min(resid.size, cond_vol_scaled.size)
            denom = cond_vol_scaled[-k:]
            num = resid[-k:]
            with np.errstate(divide="ignore", invalid="ignore"):
                std_resid = num / denom
    std_resid = std_resid[np.isfinite(std_resid)]
    # Distributional assumptions in GARCH are about innovations (standardized residuals),
    # not raw returns. Keep these tests on std_resid for statistical consistency.
    jb_ret = _jarque_bera_test(std_resid)
    t_ret = _student_t_gof_test(std_resid)
    ged_ret = _ged_gof_test(std_resid)
    skew_t_ret = _skew_t_gof_test(std_resid)
    adf_post = _adf_stationarity_test(std_resid)
    acf_post = _acf_ljung_test(std_resid, lags=arch_lags_i)
    post_arch = _arch_lm_test(std_resid, lags=arch_lags_i)
    ret_bt = ret_scaled_raw.reindex(ret_scaled.index)
    mean_bt = arma_mean_scaled.reindex(ret_scaled.index)
    var_bt_95 = _var_hit_backtest(
        returns_scaled=ret_bt.to_numpy(dtype=float),
        mu_scaled=mean_bt.to_numpy(dtype=float),
        cond_vol_scaled=cond_vol_scaled,
        nu=float(nu) if np.isfinite(nu) else float("nan"),
        alpha=0.05,
    )
    var_bt_99 = _var_hit_backtest(
        returns_scaled=ret_bt.to_numpy(dtype=float),
        mu_scaled=mean_bt.to_numpy(dtype=float),
        cond_vol_scaled=cond_vol_scaled,
        nu=float(nu) if np.isfinite(nu) else float("nan"),
        alpha=0.01,
    )

    # Horizontal comparison: 6 model variants (opt-in; expensive).
    cmp_models_sorted: list[dict[str, Any]] = []
    cmp_best: str | None = None
    cmp_best_bic: str | None = None
    cmp_candidate_count = 0
    include_cmp = bool(include_model_comparison)

    def _fit_compare_variant(
        *,
        model_id: str,
        label: str,
        vol_name: str,
        dist_name: str,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "model_id": model_id,
            "label": label,
            "vol": vol_name,
            "dist": dist_name,
            "ok": False,
            "converged": False,
            "convergence_flag": None,
            "aic": None,
            "bic": None,
            "loglikelihood": None,
            "n_obs": int(n_model),
            "resid_acf_pvalue": None,
            "resid_arch_pvalue": None,
            "var95_hit_rate": None,
            "var95_kupiec_pvalue": None,
            "var95_ind_pvalue": None,
            "var95_pass": None,
            "var99_hit_rate": None,
            "var99_kupiec_pvalue": None,
            "var99_ind_pvalue": None,
            "var99_pass": None,
            "notes": "",
        }
        try:
            try:
                mdl = arch_model_factory(
                    ret_scaled,
                    mean="Zero",
                    vol=vol_name,
                    p=1,
                    o=1,
                    q=1,
                    dist=dist_name,
                    rescale=False,
                )
            except TypeError:
                mdl = arch_model_factory(
                    ret_scaled,
                    mean="Zero",
                    vol=vol_name,
                    p=1,
                    o=1,
                    q=1,
                    dist=dist_name,
                )
            try:
                fit_i = mdl.fit(disp="off", show_warning=False)
            except TypeError:
                try:
                    fit_i = mdl.fit(disp="off")
                except TypeError:
                    fit_i = mdl.fit()
        except Exception as exc:  # pragma: no cover
            row["notes"] = f"fit_failed:{exc}"
            return row

        conv_i = int(getattr(fit_i, "convergence_flag", 0) or 0)
        row["convergence_flag"] = int(conv_i)
        row["converged"] = bool(conv_i == 0)
        if conv_i != 0:
            row["notes"] = "not_converged"
            return row

        params_i = getattr(fit_i, "params", {})
        nu_i = _safe_float(params_i.get("nu") if hasattr(params_i, "get") else None)
        ll_i = _safe_float(getattr(fit_i, "loglikelihood", None))
        aic_i = _safe_float(getattr(fit_i, "aic", None))
        bic_i = _safe_float(getattr(fit_i, "bic", None))

        cond_i = np.asarray(getattr(fit_i, "conditional_volatility", []), dtype=float)
        resid_i = np.asarray(getattr(fit_i, "resid", []), dtype=float)
        std_i = np.asarray(getattr(fit_i, "std_resid", []), dtype=float)
        if std_i.size <= 0 and resid_i.size > 0 and cond_i.size > 0:
            kk = min(resid_i.size, cond_i.size)
            with np.errstate(divide="ignore", invalid="ignore"):
                std_i = resid_i[-kk:] / cond_i[-kk:]
        std_i = std_i[np.isfinite(std_i)]

        acf_i = _acf_ljung_test(std_i, lags=arch_lags_i)
        arch_i = _arch_lm_test(std_i, lags=arch_lags_i)
        var95_i = _var_hit_backtest(
            returns_scaled=ret_bt.to_numpy(dtype=float),
            mu_scaled=mean_bt.to_numpy(dtype=float),
            cond_vol_scaled=cond_i,
            nu=float(nu_i) if np.isfinite(nu_i) else float("nan"),
            alpha=0.05,
        )
        var99_i = _var_hit_backtest(
            returns_scaled=ret_bt.to_numpy(dtype=float),
            mu_scaled=mean_bt.to_numpy(dtype=float),
            cond_vol_scaled=cond_i,
            nu=float(nu_i) if np.isfinite(nu_i) else float("nan"),
            alpha=0.01,
        )
        var95_pass = bool(
            var95_i.get("ok")
            and (var95_i.get("kupiec_reject") is False)
            and (var95_i.get("christoffersen_reject") is False)
        )
        var99_pass = bool(
            var99_i.get("ok")
            and (var99_i.get("kupiec_reject") is False)
            and (var99_i.get("christoffersen_reject") is False)
        )
        row.update(
            {
                "ok": True,
                "aic": float(aic_i) if np.isfinite(aic_i) else None,
                "bic": float(bic_i) if np.isfinite(bic_i) else None,
                "loglikelihood": float(ll_i) if np.isfinite(ll_i) else None,
                "resid_acf_pvalue": _safe_float(acf_i.get("pvalue")),
                "resid_arch_pvalue": _safe_float(arch_i.get("pvalue")),
                "var95_hit_rate": _safe_float(var95_i.get("hit_rate")),
                "var95_kupiec_pvalue": _safe_float(var95_i.get("kupiec_pvalue")),
                "var95_ind_pvalue": _safe_float(var95_i.get("christoffersen_pvalue")),
                "var95_pass": bool(var95_pass),
                "var99_hit_rate": _safe_float(var99_i.get("hit_rate")),
                "var99_kupiec_pvalue": _safe_float(var99_i.get("kupiec_pvalue")),
                "var99_ind_pvalue": _safe_float(var99_i.get("christoffersen_pvalue")),
                "var99_pass": bool(var99_pass),
            }
        )
        return row

    if include_cmp:
        cmp_specs = [
            ("gjr_garch_t", "GJR-GARCH+t", "GARCH", "t"),
            ("gjr_garch_skewt", "GJR-GARCH+skew-t", "GARCH", "skewt"),
            ("egarch_t", "EGARCH+t", "EGARCH", "t"),
            ("egarch_skewt", "EGARCH+skew-t", "EGARCH", "skewt"),
            ("aparch_t", "APARCH+t", "APARCH", "t"),
            ("aparch_skewt", "APARCH+skew-t", "APARCH", "skewt"),
        ]
        cmp_candidate_count = len(cmp_specs)
        cmp_models = [
            _fit_compare_variant(
                model_id=mid,
                label=lbl,
                vol_name=vol_n,
                dist_name=dist_n,
            )
            for (mid, lbl, vol_n, dist_n) in cmp_specs
        ]

        def _cmp_sort_key(m: dict[str, Any]) -> tuple:
            converged = bool(m.get("converged"))
            var95_pass = bool(m.get("var95_pass"))
            var99_pass = bool(m.get("var99_pass"))
            var_pass_score = (1 if var95_pass else 0) + (1 if var99_pass else 0)
            arch_p = _safe_float(m.get("resid_arch_pvalue"))
            acf_p = _safe_float(m.get("resid_acf_pvalue"))
            bic = _safe_float(m.get("bic"))
            return (
                0 if converged else 1,
                0 if var_pass_score == 2 else (1 if var_pass_score == 1 else 2),
                0 if np.isfinite(arch_p) else 1,
                -float(arch_p) if np.isfinite(arch_p) else 1.0,
                0 if np.isfinite(acf_p) else 1,
                -float(acf_p) if np.isfinite(acf_p) else 1.0,
                0 if np.isfinite(bic) else 1,
                float(bic) if np.isfinite(bic) else float("inf"),
            )

        cmp_models_sorted = sorted(cmp_models, key=_cmp_sort_key)
        cmp_best = (
            cmp_models_sorted[0].get("model_id") if len(cmp_models_sorted) > 0 else None
        )
        cmp_ok = [
            m
            for m in cmp_models
            if bool(m.get("ok")) and np.isfinite(_safe_float(m.get("bic")))
        ]
        cmp_best_bic = (
            min(cmp_ok, key=lambda x: float(_safe_float(x.get("bic")))).get("model_id")
            if cmp_ok
            else None
        )

    resid_mean = float(np.mean(std_resid)) if std_resid.size > 0 else float("nan")
    resid_std = (
        float(np.std(std_resid, ddof=1)) if std_resid.size >= 2 else float("nan")
    )
    if std_resid.size >= 3:
        m = float(np.mean(std_resid))
        centered = std_resid - m
        m2 = float(np.mean(centered**2))
        m3 = float(np.mean(centered**3))
        resid_skew = float(m3 / (m2**1.5)) if m2 > 0 else float("nan")
    else:
        resid_skew = float("nan")
    if std_resid.size >= 4:
        m = float(np.mean(std_resid))
        centered = std_resid - m
        m2 = float(np.mean(centered**2))
        m4 = float(np.mean(centered**4))
        resid_kurt_excess = float(m4 / (m2 * m2) - 3.0) if m2 > 0 else float("nan")
    else:
        resid_kurt_excess = float("nan")

    interp = _model_value_assessment(
        n_obs=n_ret,
        converged=converged,
        persistence=persistence,
        pre_arch_pvalue=_safe_float(pre_arch.get("pvalue")),
        post_arch_pvalue=_safe_float(post_arch.get("pvalue")),
        nu=nu,
    )

    vol_dates = [pd.Timestamp(d).date().isoformat() for d in ret_scaled.index]
    price_aligned = close_clean.reindex(ret_scaled.index)

    def _pack_series(vals: np.ndarray | pd.Series) -> list[float | None]:
        arr = np.asarray(vals, dtype=float)
        out: list[float | None] = []
        for v in arr:
            out.append(float(v) if np.isfinite(v) else None)
        return out

    def _opt(v: float) -> float | None:
        return float(v) if np.isfinite(v) else None

    return {
        "ok": True,
        "meta": base_meta,
        "params": {
            "mu": _opt(mu),
            "omega": _opt(omega),
            "alpha1": _opt(alpha1),
            "gamma1": _opt(gamma1),
            "beta1": _opt(beta1),
            "nu": _opt(nu),
            "persistence": _opt(persistence),
            "unconditional_var_daily": _opt(uncond_var_daily),
            "unconditional_vol_daily": _opt(uncond_vol_daily),
            "unconditional_vol_annualized": _opt(uncond_vol_ann),
        },
        "diagnostics": {
            "converged": converged,
            "convergence_flag": int(conv_flag),
            "n_obs_raw": n_raw,
            "n_obs_price": n_close,
            "n_obs_returns": n_ret,
            "n_obs_model": n_model,
            "dropped_obs": dropped,
            "ann_factor": int(ann),
            "return_scale": float(scale),
            "loglikelihood": _opt(_safe_float(getattr(fit, "loglikelihood", None))),
            "aic": _opt(_safe_float(getattr(fit, "aic", None))),
            "bic": _opt(_safe_float(getattr(fit, "bic", None))),
            "std_resid_mean": _opt(resid_mean),
            "std_resid_std": _opt(resid_std),
            "std_resid_skew": _opt(resid_skew),
            "std_resid_kurtosis_excess": _opt(resid_kurt_excess),
            "normality_jb": jb_ret,
            "t_dist_gof": t_ret,
            "ged_dist_gof": ged_ret,
            "skew_t_dist_gof": skew_t_ret,
            "normality_jb_pre": jb_pre,
            "t_dist_gof_pre": t_pre,
            "ged_dist_gof_pre": ged_pre,
            "skew_t_dist_gof_pre": skew_t_pre,
            "stationarity_adf_pre": adf_ret,
            "stationarity_adf_post": adf_post,
            "autocorr_acf_ljung_pre": acf_pre,
            "autocorr_acf_ljung_post": acf_post,
            "arch_lm_pre": pre_arch,
            "arch_lm_post": post_arch,
            "var_backtest_95": var_bt_95,
            "var_backtest_99": var_bt_99,
            "mean_model_selection": {
                "selected": str(arma_sel.get("model") or "arma(0,0)"),
                "selected_aic": arma_sel.get("aic"),
                "selected_bic": arma_sel.get("bic"),
                "selected_resid_acf_pvalue": arma_sel.get("resid_acf_pvalue"),
                "selected_resid_arch_pvalue": arma_sel.get("resid_arch_pvalue"),
                "candidates": arma_sel.get("candidates") or [],
            },
            "model_comparison": {
                "enabled": include_cmp,
                "models": cmp_models_sorted,
                "best_by_priority": cmp_best,
                "best_by_bic": cmp_best_bic,
                "priority_order": [
                    "converged",
                    "var95_99_pass",
                    "resid_arch_pvalue_desc",
                    "resid_acf_pvalue_desc",
                    "bic_asc",
                ],
                "candidate_count": int(cmp_candidate_count),
            },
        },
        "interpretation": interp,
        "series": {
            "price_dates": vol_dates,
            "price_close": _pack_series(price_aligned.to_numpy(dtype=float)),
            "vol_dates": vol_dates,
            "cond_vol_daily": _pack_series(cond_vol_daily),
            "cond_vol_annualized": _pack_series(cond_vol_ann),
            "log_returns": _pack_series(ret_bt.to_numpy(dtype=float)),
        },
    }
