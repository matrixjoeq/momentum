from __future__ import annotations

# pylint: disable=broad-exception-caught

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import chi2


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
    arch_model_factory: Any | None = None,
) -> dict[str, Any]:
    """
    Fit GJR-GARCH(1,1) with Student-t innovations on close log returns.

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

    pre_arch = _arch_lm_test(ret.to_numpy(dtype=float), lags=arch_lags_i)
    ret_scaled = ret * float(scale)

    try:
        try:
            model = arch_model_factory(
                ret_scaled,
                mean="Constant",
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
                mean="Constant",
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
    mu = _safe_float(params.get("mu") if hasattr(params, "get") else None)
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
    cond_vol_scaled = cond_vol_scaled[-n_ret:]
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
    post_arch = _arch_lm_test(std_resid, lags=arch_lags_i)

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

    vol_dates = [pd.Timestamp(d).date().isoformat() for d in ret.index]
    price_aligned = close_clean.reindex(ret.index)

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
            "arch_lm_pre": pre_arch,
            "arch_lm_post": post_arch,
        },
        "interpretation": interp,
        "series": {
            "price_dates": vol_dates,
            "price_close": _pack_series(price_aligned.to_numpy(dtype=float)),
            "vol_dates": vol_dates,
            "cond_vol_daily": _pack_series(cond_vol_daily),
            "cond_vol_annualized": _pack_series(cond_vol_ann),
            "log_returns": _pack_series(ret.to_numpy(dtype=float)),
        },
    }
