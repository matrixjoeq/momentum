from __future__ import annotations

import datetime as dt
import os
import sys
import multiprocessing as mp
import math
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd

from .baseline import (
    TRADING_DAYS_PER_YEAR,
    _annualized_return,
    _annualized_vol,
    _max_drawdown,
    _sharpe,
)
from ..strategy import rotation as _rot


def _parse_yyyymmdd(s: str) -> dt.date:
    return dt.datetime.strptime(str(s), "%Y%m%d").date()


def _yyyymmdd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def business_days_index(*, start: str, end: str) -> pd.DatetimeIndex:
    """
    Synthetic trading calendar: pandas business days.
    Use this to avoid exchange calendar out-of-bounds (and to support start=1990-01-01).
    """
    s = _parse_yyyymmdd(start)
    e = _parse_yyyymmdd(end)
    if e < s:
        return pd.DatetimeIndex([])
    return pd.date_range(start=s, end=e, freq="B")


def _today_last_business_day_yyyymmdd() -> str:
    d = dt.date.today()
    # If today is weekend, go back to Fri.
    while d.weekday() >= 5:
        d = d - dt.timedelta(days=1)
    return _yyyymmdd(d)


def _metrics_from_nav(nav: pd.Series, *, rf_annual: float = 0.0) -> dict[str, float | None]:
    nav = pd.to_numeric(nav, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if nav.empty or len(nav) < 3:
        return {"cagr": None, "vol": None, "sharpe": None, "max_drawdown": None, "calmar": None}
    daily_ret = nav.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    cagr = float(_annualized_return(nav, ann_factor=TRADING_DAYS_PER_YEAR))
    vol = float(_annualized_vol(daily_ret, ann_factor=TRADING_DAYS_PER_YEAR))
    sharpe = float(_sharpe(daily_ret, rf=float(rf_annual), ann_factor=TRADING_DAYS_PER_YEAR))
    mdd = float(_max_drawdown(nav))
    calmar = float(cagr / abs(mdd)) if np.isfinite(mdd) and float(mdd) < 0 else float("nan")
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_drawdown": mdd, "calmar": calmar}


@dataclass(frozen=True)
class SimConfig:
    n_assets: int = 4
    vol_low: float = 0.05
    vol_high: float = 0.30
    corr_low: float | None = None
    corr_high: float | None = None
    mu_low: float | None = None
    mu_high: float | None = None
    seed: int | None = None


def _resolve_mu_range(cfg: SimConfig) -> tuple[float, float]:
    # If unset, keep "random drift" default behavior.
    if cfg.mu_low is None and cfg.mu_high is None:
        return -0.05, 0.15
    if cfg.mu_low is None or cfg.mu_high is None:
        raise ValueError("invalid_mu_range")
    lo = float(cfg.mu_low)
    hi = float(cfg.mu_high)
    if not (-1.0 <= lo <= hi <= 3.0):
        raise ValueError("invalid_mu_range")
    return lo, hi


def _resolve_corr_range(cfg: SimConfig, *, n_assets: int) -> tuple[float, float] | None:
    # Unset means independent assets.
    if cfg.corr_low is None and cfg.corr_high is None:
        return None
    if cfg.corr_low is None or cfg.corr_high is None:
        raise ValueError("invalid_corr_range")
    lo = float(cfg.corr_low)
    hi = float(cfg.corr_high)
    # Equicorrelation PSD lower bound.
    rho_floor = -1.0 / max(1.0, float(n_assets - 1))
    if not (rho_floor <= lo <= hi < 0.99):
        raise ValueError("invalid_corr_range")
    return lo, hi


def _nearest_correlation_matrix(a: np.ndarray, *, max_iter: int = 40, eps: float = 1e-10) -> np.ndarray:
    x = np.array(a, dtype=float, copy=True)
    x = 0.5 * (x + x.T)
    np.fill_diagonal(x, 1.0)
    y = x.copy()
    delta = np.zeros_like(x)
    for _ in range(int(max_iter)):
        r = y - delta
        w, v = np.linalg.eigh(0.5 * (r + r.T))
        w = np.clip(w, eps, None)
        x_psd = (v * w) @ v.T
        delta = x_psd - r
        y = x_psd.copy()
        np.fill_diagonal(y, 1.0)
    y = 0.5 * (y + y.T)
    np.fill_diagonal(y, 1.0)
    w, v = np.linalg.eigh(y)
    if np.min(w) < eps:
        w = np.clip(w, eps, None)
        y = (v * w) @ v.T
        d = np.sqrt(np.clip(np.diag(y), eps, None))
        y = y / (d[:, None] * d[None, :])
        y = 0.5 * (y + y.T)
        np.fill_diagonal(y, 1.0)
    return y


def _sample_pairwise_corr_matrix(
    *,
    n_assets: int,
    corr_low: float,
    corr_high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(n_assets)
    if n <= 1:
        return np.eye(max(1, n), dtype=float)
    lo = float(corr_low)
    hi = float(corr_high)
    if abs(hi - lo) < 1e-12:
        m = np.full((n, n), lo, dtype=float)
        np.fill_diagonal(m, 1.0)
        return _nearest_correlation_matrix(m)
    iu = np.triu_indices(n, 1)
    best = None
    best_pen = float("inf")
    for _ in range(80):
        t = np.eye(n, dtype=float)
        vals = rng.uniform(low=lo, high=hi, size=len(iu[0])).astype(float)
        t[iu] = vals
        t[(iu[1], iu[0])] = vals
        c = _nearest_correlation_matrix(t)
        off = c[iu]
        pen = float(np.maximum(0.0, lo - np.min(off)) + np.maximum(0.0, np.max(off) - hi))
        if pen < best_pen:
            best_pen = pen
            best = c
        if pen <= 0.01:
            return c
    return best if best is not None else np.eye(n, dtype=float)


def simulate_gbm_prices(
    *,
    start: str = "19900101",
    end: str | None = None,
    cfg: SimConfig = SimConfig(),
) -> dict[str, Any]:
    """
    Simulate GBM-like price series (starting at 1.0) where:
    - daily log returns are Normal(mu_d, sigma_d)
    - annual vol is sampled per asset from [vol_low, vol_high]
    - annual drift is sampled per asset from [mu_low, mu_high]
      (or random default range when mu range is unset)
    - assets are independent by default; optional common-correlation range supported
    """
    end = str(end or _today_last_business_day_yyyymmdd())
    idx = business_days_index(start=start, end=end)
    if len(idx) < 10:
        return {"ok": False, "error": "insufficient_dates", "meta": {"start": start, "end": end}}

    n = int(cfg.n_assets)
    if n <= 1 or n > 20:
        return {"ok": False, "error": "invalid_n_assets", "meta": {"n_assets": n}}

    v0 = float(cfg.vol_low)
    v1 = float(cfg.vol_high)
    if not (0.0 < v0 <= v1 <= 2.0):
        return {"ok": False, "error": "invalid_vol_range", "meta": {"vol_low": v0, "vol_high": v1}}
    try:
        mu0, mu1 = _resolve_mu_range(cfg)
    except ValueError:
        return {"ok": False, "error": "invalid_mu_range", "meta": {"mu_low": cfg.mu_low, "mu_high": cfg.mu_high}}
    try:
        corr_range = _resolve_corr_range(cfg, n_assets=n)
    except ValueError:
        return {"ok": False, "error": "invalid_corr_range", "meta": {"corr_low": cfg.corr_low, "corr_high": cfg.corr_high}}

    rng = np.random.default_rng(None if cfg.seed is None else int(cfg.seed))
    ann_vols = rng.uniform(low=v0, high=v1, size=n).astype(float)
    ann_mus = rng.uniform(low=mu0, high=mu1, size=n).astype(float)
    sig_d = ann_vols / np.sqrt(float(TRADING_DAYS_PER_YEAR))
    mu_d = ann_mus / float(TRADING_DAYS_PER_YEAR)
    if corr_range is None:
        z = rng.normal(loc=0.0, scale=1.0, size=(len(idx), n)).astype(float)
        corr_gen = np.eye(n, dtype=float)
    else:
        corr_gen = _sample_pairwise_corr_matrix(
            n_assets=n,
            corr_low=float(corr_range[0]),
            corr_high=float(corr_range[1]),
            rng=rng,
        )
        l = np.linalg.cholesky(corr_gen)
        z_raw = rng.normal(loc=0.0, scale=1.0, size=(len(idx), n)).astype(float)
        z = (z_raw @ l.T).astype(float)
    # log returns (first day has no return)
    lr = (mu_d.reshape(1, n) + sig_d.reshape(1, n) * z).astype(float)
    lr[0, :] = 0.0
    logp = np.cumsum(lr, axis=0)
    px = np.exp(logp).astype(float)
    codes = [f"SIM{i+1}" for i in range(n)]
    close = pd.DataFrame(px, index=idx, columns=codes, dtype=float)

    # Correlation of daily log returns (exclude first)
    lr_df = pd.DataFrame(lr, index=idx, columns=codes, dtype=float)
    corr = lr_df.iloc[1:].corr().to_numpy(dtype=float).tolist()

    metrics_by_asset: dict[str, dict[str, float | None]] = {}
    for c in codes:
        metrics_by_asset[c] = _metrics_from_nav(close[c].astype(float))

    return {
        "ok": True,
        "meta": {
            "start": start,
            "end": end,
            "n_assets": int(n),
            "vol_low": float(v0),
            "vol_high": float(v1),
            "corr_low": (None if cfg.corr_low is None else float(cfg.corr_low)),
            "corr_high": (None if cfg.corr_high is None else float(cfg.corr_high)),
            "corr_generated_offdiag_mean": (
                float(np.mean(corr_gen[np.triu_indices(n, 1)])) if n > 1 else 0.0
            ),
            "mu_low": (None if cfg.mu_low is None else float(cfg.mu_low)),
            "mu_high": (None if cfg.mu_high is None else float(cfg.mu_high)),
            "seed": (None if cfg.seed is None else int(cfg.seed)),
            "calendar": "pandas_B",
            "return_model": "log1p(r) ~ Normal(mu_d, sigma_d), sigma_d=ann_vol/sqrt(252), mu_d=ann_mu/252",
        },
        "assets": {
            "codes": codes,
            "ann_vols": {c: float(v) for c, v in zip(codes, ann_vols, strict=False)},
            "ann_mus": {c: float(v) for c, v in zip(codes, ann_mus, strict=False)},
        },
        "series": {
            "dates": idx.strftime("%Y-%m-%d").tolist(),
            "close": {c: close[c].astype(float).tolist() for c in codes},
        },
        "corr": {"codes": codes, "matrix": corr},
        "metrics": {"by_asset": metrics_by_asset},
    }


def _weekly_decision_indices(dates: pd.DatetimeIndex) -> list[int]:
    # Decision on Fridays (weekday=4) if present; hold from next business day.
    return [i for i, d in enumerate(dates) if int(d.weekday()) == 4]


def backtest_rotation_basic(
    close: pd.DataFrame,
    *,
    lookback_days: int = 20,
    top_k: int = 1,
) -> dict[str, Any]:
    """
    Minimal rotation backtest for synthetic experiment:
    - weekly decisions on Fridays using raw momentum score = close/close.shift(lookback)-1
    - execute from next business day close-to-close (position applies to next day's return)
    - no filters/timing/risk-off
    """
    if close is None or close.empty:
        return {"ok": False, "error": "empty_prices"}
    close = close.astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    dates = pd.to_datetime(close.index)
    codes = list(close.columns)
    lb = int(max(2, lookback_days))
    _ = int(max(1, min(len(codes), int(top_k))))

    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    score = close / close.shift(lb) - 1.0

    decision_idx = _weekly_decision_indices(dates)
    # start only when lookback is available
    decision_idx = [i for i in decision_idx if i >= lb]
    if not decision_idx:
        return {"ok": False, "error": "no_decision_dates"}

    pick_daily = np.full(len(dates), -1, dtype=int)
    # warm-up: equal-weight until first rebalance-effective day
    first = decision_idx[0]
    if first + 1 < len(dates):
        pick_daily[: first + 1] = 0  # arbitrary (unused in ew warm-up)

    picks_by_decision: list[dict[str, Any]] = []
    for j, di in enumerate(decision_idx):
        start_i = di + 1
        if start_i >= len(dates):
            break
        next_di = decision_idx[j + 1] if j + 1 < len(decision_idx) else (len(dates) - 1)
        end_i = min(len(dates) - 1, next_di)
        row = score.iloc[di].to_numpy(dtype=float)
        # pick max score (ignore NaNs)
        row2 = np.where(np.isfinite(row), row, -1e18)
        pick = int(np.argmax(row2))
        pick_daily[start_i : end_i + 1] = pick
        picks_by_decision.append(
            {
                "decision_date": dates[di].date().isoformat(),
                "start_date": dates[start_i].date().isoformat(),
                "end_date": dates[end_i].date().isoformat(),
                "pick": str(codes[pick]),
                "score": (None if not np.isfinite(float(row[pick])) else float(row[pick])),
            }
        )

    # Compute portfolio return with close-execution timing:
    # decision at d -> trade at next day close -> first full return starts from the following day.
    port_ret = np.zeros(len(dates), dtype=float)
    for t in range(1, len(dates)):
        p = int(pick_daily[t - 1])
        if p < 0:
            port_ret[t] = 0.0
        else:
            port_ret[t] = float(ret.iloc[t, p])
    nav = np.cumprod(1.0 + port_ret).astype(float)
    nav[0] = 1.0
    nav_s = pd.Series(nav, index=dates, dtype=float)
    m = _metrics_from_nav(nav_s)
    return {
        "ok": True,
        "series": {"dates": pd.DatetimeIndex(dates).strftime("%Y-%m-%d").tolist(), "nav": nav_s.astype(float).tolist()},
        "metrics": m,
        "holdings": picks_by_decision,
    }


def backtest_equal_weight_weekly(close: pd.DataFrame) -> dict[str, Any]:
    if close is None or close.empty:
        return {"ok": False, "error": "empty_prices"}
    close = close.astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    dates = pd.to_datetime(close.index)
    codes = list(close.columns)
    n = len(codes)
    if n <= 0:
        return {"ok": False, "error": "empty_universe"}

    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    w = np.zeros((len(dates), n), dtype=float)
    decision_idx = _weekly_decision_indices(dates)
    if not decision_idx:
        w[:, :] = 1.0 / n
    else:
        for j, di in enumerate(decision_idx):
            start_i = di + 1
            if start_i >= len(dates):
                break
            next_di = decision_idx[j + 1] if j + 1 < len(decision_idx) else (len(dates) - 1)
            end_i = min(len(dates) - 1, next_di)
            w[start_i : end_i + 1, :] = 1.0 / n
        # warm-up before first effective day: equal-weight
        w[: decision_idx[0] + 1, :] = 1.0 / n

    w_df = pd.DataFrame(w, index=dates, columns=codes, dtype=float)
    ret_fwd = ret.shift(-1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    port_ret = (w_df.to_numpy(dtype=float) * ret_fwd.to_numpy(dtype=float)).sum(axis=1).astype(float)
    nav = np.cumprod(1.0 + port_ret).astype(float)
    nav[0] = 1.0
    nav_s = pd.Series(nav, index=dates, dtype=float)
    m = _metrics_from_nav(nav_s)
    return {
        "ok": True,
        "series": {"dates": pd.DatetimeIndex(dates).strftime("%Y-%m-%d").tolist(), "nav": nav_s.astype(float).tolist()},
        "metrics": m,
    }


def backtest_risk_parity_inverse_vol(close: pd.DataFrame, *, ann_vols: dict[str, float]) -> dict[str, Any]:
    """
    Risk parity baseline (inverse-vol weights) for the synthetic GBM experiment.

    For the simulated assets, annual vol is constant per asset, so inverse-vol weights are constant.
    """
    if close is None or close.empty:
        return {"ok": False, "error": "empty_prices"}
    close = close.astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    dates = pd.to_datetime(close.index)
    codes = list(close.columns)
    n = len(codes)
    if n <= 0:
        return {"ok": False, "error": "empty_universe"}

    vols = np.asarray([float(ann_vols.get(c, np.nan)) for c in codes], dtype=float)
    inv = 1.0 / np.where((np.isfinite(vols) & (vols > 0)), vols, np.nan)
    s = float(np.nansum(inv))
    if not np.isfinite(s) or s <= 0:
        w = np.full(n, 1.0 / n, dtype=float)
    else:
        w = inv / s
        w = np.where(np.isfinite(w), w, 0.0)
        ss = float(np.sum(w))
        w = (w / ss) if ss > 0 else np.full(n, 1.0 / n, dtype=float)

    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    w_df = pd.DataFrame(np.repeat(w.reshape(1, -1), len(dates), axis=0), index=dates, columns=codes, dtype=float)
    ret_fwd = ret.shift(-1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    port_ret = (ret_fwd.to_numpy(dtype=float) * w_df.to_numpy(dtype=float)).sum(axis=1).astype(float)
    nav = np.cumprod(1.0 + port_ret).astype(float)
    nav[0] = 1.0
    nav_s = pd.Series(nav, index=dates, dtype=float)
    m = _metrics_from_nav(nav_s)
    return {
        "ok": True,
        "series": {"dates": pd.DatetimeIndex(dates).strftime("%Y-%m-%d").tolist(), "nav": nav_s.astype(float).tolist()},
        "metrics": m,
        "weights": {c: float(wi) for c, wi in zip(codes, w, strict=False)},
    }


def apply_position_sizing(
    nav: pd.Series,
    *,
    initial_cash: float,
    position_pct: float,
) -> dict[str, Any]:
    nav = pd.to_numeric(nav, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if nav.empty or len(nav) < 3:
        return {"ok": False, "error": "empty_nav"}
    pos = float(position_pct)
    if not np.isfinite(pos) or pos < 0:
        return {"ok": False, "error": "invalid_position_pct"}
    cash0 = float(initial_cash)
    if not np.isfinite(cash0) or cash0 <= 0:
        return {"ok": False, "error": "invalid_initial_cash"}

    daily_ret = nav.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    equity = np.full(len(nav), float(cash0), dtype=float)
    for i in range(1, len(nav)):
        r = float(daily_ret.iloc[i])
        equity[i] = equity[i - 1] * (1.0 + pos * r)
    eq = pd.Series(equity, index=nav.index, dtype=float)
    min_eq = float(np.min(equity)) if equity.size else float("nan")
    ruin = bool(min_eq <= 0.0)
    return {
        "ok": True,
        "series": {"dates": pd.DatetimeIndex(eq.index).strftime("%Y-%m-%d").tolist(), "equity": eq.astype(float).tolist()},
        "stats": {
            "initial_cash": float(cash0),
            "position_pct": float(pos),
            "min_equity": float(min_eq),
            "min_equity_ratio": float(min_eq / cash0) if cash0 > 0 and np.isfinite(min_eq) else None,
            "ruin": bool(ruin),
        },
    }


def montecarlo_rotation_vs_ew(
    *,
    start: str = "19900101",
    end: str | None = None,
    n_sims: int = 10000,
    chunk_size: int = 200,
    n_assets: int = 4,
    vol_low: float = 0.05,
    vol_high: float = 0.30,
    corr_low: float | None = None,
    corr_high: float | None = None,
    mu_low: float | None = None,
    mu_high: float | None = None,
    seed: int | None = None,
    lookback_days: int = 20,
    strategy_a: dict[str, Any] | None = None,
    n_jobs: int = 0,
    selected_assets: list[int] | None = None,
) -> dict[str, Any]:
    end = str(end or _today_last_business_day_yyyymmdd())
    idx = business_days_index(start=start, end=end)
    if len(idx) < 100:
        return {"ok": False, "error": "insufficient_dates"}
    n_days = len(idx)
    n_assets = int(n_assets)
    # Determine active asset indices (selected by Phase1 highlights) or all assets by default
    all_indices = list(range(n_assets))
    if selected_assets is not None:
        try:
            sel = sorted({int(x) for x in selected_assets})
        except Exception:
            return {"ok": False, "error": "invalid_selected_assets"}
        sel = [i for i in sel if 0 <= i < n_assets]
        asset_indices = sel
    else:
        asset_indices = all_indices
    n_assets_eff = len(asset_indices)
    if n_assets_eff <= 0 or n_assets_eff > 50:
        return {"ok": False, "error": "invalid_n_assets", "meta": {"n_assets": n_assets_eff}}

    n_sims = int(n_sims)
    if n_sims <= 0 or n_sims > 50000:
        return {"ok": False, "error": "invalid_n_sims"}
    chunk_size = int(max(1, min(2000, int(chunk_size))))
    cfg_check = SimConfig(
        n_assets=int(n_assets_eff),
        vol_low=float(vol_low),
        vol_high=float(vol_high),
        corr_low=(None if corr_low is None else float(corr_low)),
        corr_high=(None if corr_high is None else float(corr_high)),
        mu_low=(None if mu_low is None else float(mu_low)),
        mu_high=(None if mu_high is None else float(mu_high)),
        seed=seed,
    )
    try:
        corr_range = _resolve_corr_range(cfg_check, n_assets=int(n_assets_eff))
    except ValueError:
        return {"ok": False, "error": "invalid_corr_range", "meta": {"corr_low": corr_low, "corr_high": corr_high}}
    try:
        mu0, mu1 = _resolve_mu_range(cfg_check)
    except ValueError:
        return {"ok": False, "error": "invalid_mu_range", "meta": {"mu_low": mu_low, "mu_high": mu_high}}

    rng = np.random.default_rng(None if seed is None else int(seed))
    # Precompute decision indices and segments once.
    dates = pd.to_datetime(idx)
    decision_idx = _weekly_decision_indices(dates)
    lb = int(max(2, lookback_days))
    decision_idx = [i for i in decision_idx if i >= lb]
    if not decision_idx:
        return {"ok": False, "error": "no_decision_dates"}

    # For daily pick mapping, we will fill per decision segment.
    # segments are (start_i, end_i) inclusive.
    segs: list[tuple[int, int]] = []
    for j, di in enumerate(decision_idx):
        start_i = di + 1
        if start_i >= n_days:
            break
        next_di = decision_idx[j + 1] if j + 1 < len(decision_idx) else (n_days - 1)
        end_i = min(n_days - 1, next_di)
        segs.append((start_i, end_i))
    if not segs:
        return {"ok": False, "error": "no_effective_segments"}

    # Output arrays
    out_cagr_rot: list[float] = []
    out_cagr_ew: list[float] = []
    out_cagr_rp: list[float] = []
    out_mdd_rot: list[float] = []
    out_mdd_ew: list[float] = []
    out_mdd_rp: list[float] = []

    use_strategy_a = bool(strategy_a)
    if use_strategy_a:
        sim_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n_sims)]
        jobs = int(n_jobs)
        if jobs <= 0:
            jobs = max(1, int(os.cpu_count() or 1))
        jobs = max(1, min(jobs, n_sims))
        jobs_effective = int(jobs)
        if jobs == 1:
            for s in sim_seeds:
                one = _eval_mc_world(
                    start=start,
                    end=end,
                    n_assets=int(n_assets_eff),
                    vol_low=float(vol_low),
                    vol_high=float(vol_high),
                    corr_low=(None if corr_low is None else float(corr_low)),
                    corr_high=(None if corr_high is None else float(corr_high)),
                    mu_low=(None if mu_low is None else float(mu_low)),
                    mu_high=(None if mu_high is None else float(mu_high)),
                    world_seed=int(s),
                    strategy_a=dict(strategy_a or {}),
                )
                if one is None:
                    continue
                cr, ce, cp, dr, de, dp = one
                out_cagr_rot.append(float(cr))
                out_cagr_ew.append(float(ce))
                out_cagr_rp.append(float(cp))
                out_mdd_rot.append(float(dr))
                out_mdd_ew.append(float(de))
                out_mdd_rp.append(float(dp))
        else:
            try:
                mp_ctx = None
                if sys.platform != "win32":
                    try:
                        mp_ctx = mp.get_context("fork")
                    except ValueError:
                        mp_ctx = None
                with ProcessPoolExecutor(max_workers=jobs, mp_context=mp_ctx) as ex:
                    futs = [
                        ex.submit(
                            _eval_mc_world,
                            start=start,
                            end=end,
                            n_assets=int(n_assets_eff),
                            vol_low=float(vol_low),
                            vol_high=float(vol_high),
                            corr_low=(None if corr_low is None else float(corr_low)),
                            corr_high=(None if corr_high is None else float(corr_high)),
                            mu_low=(None if mu_low is None else float(mu_low)),
                            mu_high=(None if mu_high is None else float(mu_high)),
                            world_seed=int(s),
                            strategy_a=dict(strategy_a or {}),
                        )
                        for s in sim_seeds
                    ]
                    for f in futs:
                        one = f.result()
                        if one is None:
                            continue
                        cr, ce, cp, dr, de, dp = one
                        out_cagr_rot.append(float(cr))
                        out_cagr_ew.append(float(ce))
                        out_cagr_rp.append(float(cp))
                        out_mdd_rot.append(float(dr))
                        out_mdd_ew.append(float(de))
                        out_mdd_rp.append(float(dp))
            except (BrokenProcessPool, RuntimeError, OSError, FileNotFoundError):
                jobs_effective = 1
                for s in sim_seeds:
                    one = _eval_mc_world(
                        start=start,
                        end=end,
                        n_assets=int(n_assets_eff),
                        vol_low=float(vol_low),
                        vol_high=float(vol_high),
                        corr_low=(None if corr_low is None else float(corr_low)),
                        corr_high=(None if corr_high is None else float(corr_high)),
                        mu_low=(None if mu_low is None else float(mu_low)),
                        mu_high=(None if mu_high is None else float(mu_high)),
                        world_seed=int(s),
                        strategy_a=dict(strategy_a or {}),
                    )
                    if one is None:
                        continue
                    cr, ce, cp, dr, de, dp = one
                    out_cagr_rot.append(float(cr))
                    out_cagr_ew.append(float(ce))
                    out_cagr_rp.append(float(cp))
                    out_mdd_rot.append(float(dr))
                    out_mdd_ew.append(float(de))
                    out_mdd_rp.append(float(dp))
    else:
        jobs_effective = 1
        done = 0
        while done < n_sims:
            m = min(chunk_size, n_sims - done)
            ann_vols_all = rng.uniform(low=float(vol_low), high=float(vol_high), size=(m, n_assets)).astype(float)
            ann_mus_all = rng.uniform(low=float(mu0), high=float(mu1), size=(m, n_assets)).astype(float)
            ann_vols = ann_vols_all[:, asset_indices]
            ann_mus = ann_mus_all[:, asset_indices]
            sig_d = ann_vols / np.sqrt(float(TRADING_DAYS_PER_YEAR))
            mu_d = ann_mus / float(TRADING_DAYS_PER_YEAR)
            if corr_range is None:
                z = rng.normal(loc=0.0, scale=1.0, size=(m, n_days, n_assets_eff)).astype(float)
            else:
                l_batch = np.empty((m, n_assets_eff, n_assets_eff), dtype=float)
                for i in range(m):
                    c = _sample_pairwise_corr_matrix(
                        n_assets=n_assets_eff,
                        corr_low=float(corr_range[0]),
                        corr_high=float(corr_range[1]),
                        rng=rng,
                    )
                    l_batch[i] = np.linalg.cholesky(c)
                z_raw = rng.normal(loc=0.0, scale=1.0, size=(m, n_days, n_assets_eff)).astype(float)
                z = np.einsum("itk,ijk->itj", z_raw, l_batch).astype(float)
            lr = (mu_d[:, None, :] + sig_d[:, None, :] * z).astype(float)
            lr[:, 0, :] = 0.0
            logp = np.cumsum(lr, axis=1)
            ret = np.exp(lr).astype(float) - 1.0

            dec = np.array(decision_idx, dtype=int)
            mom = np.exp(logp[:, dec, :] - logp[:, dec - lb, :]) - 1.0
            pick = np.argmax(np.where(np.isfinite(mom), mom, -1e18), axis=2).astype(int)
            pick_daily = np.full((m, n_days), -1, dtype=int)
            pick_daily[:, : decision_idx[0] + 1] = 0
            for j, (start_i, end_i) in enumerate(segs):
                if j >= pick.shape[1]:
                    break
                pick_daily[:, start_i : end_i + 1] = pick[:, j].reshape(m, 1)

            rot_ret = np.zeros((m, n_days), dtype=float)
            for t in range(1, n_days):
                p = pick_daily[:, t]
                rot_ret[:, t] = ret[np.arange(m), t, p]
            rot_nav = np.cumprod(1.0 + rot_ret, axis=1).astype(float)
            rot_nav[:, 0] = 1.0
            ew_ret = ret.mean(axis=2).astype(float)
            ew_nav = np.cumprod(1.0 + ew_ret, axis=1).astype(float)
            ew_nav[:, 0] = 1.0
            inv = 1.0 / np.where((np.isfinite(ann_vols) & (ann_vols > 0)), ann_vols, np.nan)
            inv_sum = np.nansum(inv, axis=1, keepdims=True)
            w_rp = np.where(np.isfinite(inv_sum) & (inv_sum > 0), inv / inv_sum, (1.0 / float(n_assets_eff)))
            rp_ret = np.sum(ret * w_rp[:, None, :], axis=2).astype(float)
            rp_nav = np.cumprod(1.0 + rp_ret, axis=1).astype(float)
            rp_nav[:, 0] = 1.0

            years = (n_days - 1) / float(TRADING_DAYS_PER_YEAR)
            cagr_rot = np.power(rot_nav[:, -1], 1.0 / years) - 1.0
            cagr_ew = np.power(ew_nav[:, -1], 1.0 / years) - 1.0
            cagr_rp = np.power(rp_nav[:, -1], 1.0 / years) - 1.0
            peak_rot = np.maximum.accumulate(rot_nav, axis=1)
            mdd_rot = np.min(rot_nav / peak_rot - 1.0, axis=1)
            peak_ew = np.maximum.accumulate(ew_nav, axis=1)
            mdd_ew = np.min(ew_nav / peak_ew - 1.0, axis=1)
            peak_rp = np.maximum.accumulate(rp_nav, axis=1)
            mdd_rp = np.min(rp_nav / peak_rp - 1.0, axis=1)

            out_cagr_rot.extend([float(x) for x in cagr_rot.tolist()])
            out_cagr_ew.extend([float(x) for x in cagr_ew.tolist()])
            out_cagr_rp.extend([float(x) for x in cagr_rp.tolist()])
            out_mdd_rot.extend([float(x) for x in mdd_rot.tolist()])
            out_mdd_ew.extend([float(x) for x in mdd_ew.tolist()])
            out_mdd_rp.extend([float(x) for x in mdd_rp.tolist()])
            done += m

    return {
        "ok": True,
        "meta": {
            "start": start,
            "end": end,
            "n_days": int(n_days),
            "n_assets": int(n_assets),
            "n_sims": int(n_sims),
            "chunk_size": int(chunk_size),
            "vol_low": float(vol_low),
            "vol_high": float(vol_high),
            "corr_low": (None if corr_low is None else float(corr_low)),
            "corr_high": (None if corr_high is None else float(corr_high)),
            "mu_low": (None if mu_low is None else float(mu_low)),
            "mu_high": (None if mu_high is None else float(mu_high)),
            "seed": (None if seed is None else int(seed)),
            "lookback_days": int(lookback_days),
            "strategy_a_applied": bool(use_strategy_a),
            "n_jobs": int(jobs_effective),
        },
        "dist": {
            "rotation": {"cagr": out_cagr_rot, "max_drawdown": out_mdd_rot},
            "equal_weight": {"cagr": out_cagr_ew, "max_drawdown": out_mdd_ew},
            "risk_parity": {"cagr": out_cagr_rp, "max_drawdown": out_mdd_rp},
            "excess": {
                "cagr": [float(a - b) for a, b in zip(out_cagr_rot, out_cagr_ew, strict=False)],
                "max_drawdown": [float(a - b) for a, b in zip(out_mdd_rot, out_mdd_ew, strict=False)],
            },
            "excess_vs_risk_parity": {
                "cagr": [float(a - b) for a, b in zip(out_cagr_rot, out_cagr_rp, strict=False)],
                "max_drawdown": [float(a - b) for a, b in zip(out_mdd_rot, out_mdd_rp, strict=False)],
            },
        },
    }


def _extract_annualized_return(bt: dict[str, Any]) -> float:
    m = (bt or {}).get("metrics") or {}
    if isinstance(m.get("strategy"), dict):
        v = m["strategy"].get("annualized_return")
    else:
        v = m.get("annualized_return")
        if v is None:
            v = m.get("cagr")
    return float(v) if v is not None and np.isfinite(float(v)) else float("nan")


def _extract_max_drawdown_from_bt(bt: dict[str, Any]) -> float:
    m = (bt or {}).get("metrics") or {}
    if isinstance(m.get("strategy"), dict):
        v = m["strategy"].get("max_drawdown")
    else:
        v = m.get("max_drawdown")
    return float(v) if v is not None and np.isfinite(float(v)) else float("nan")


_AB_TARGET_LABELS: dict[str, str] = {
    "cash": "持有现金",
    "equal_weight": "等权再平衡",
    "risk_parity": "风险平价再平衡",
    "rotation_a": "轮动策略A",
    "rotation_b": "轮动策略B",
}


def _normalize_ab_target(target: str | None, *, fallback: str) -> str:
    t = str(target or "").strip().lower()
    return t if t in _AB_TARGET_LABELS else str(fallback).strip().lower()


def _ab_mode_to_targets(mode: str | None) -> tuple[str, str]:
    m = str(mode or "custom_ab").strip().lower()
    if m == "rotation_vs_equal_weight":
        return "rotation_a", "equal_weight"
    if m == "risk_parity_vs_equal_weight":
        return "risk_parity", "equal_weight"
    if m == "rotation_vs_risk_parity":
        return "rotation_a", "risk_parity"
    if m == "equal_weight_vs_cash":
        return "equal_weight", "cash"
    return "rotation_a", "rotation_b"


def _legacy_mode_from_targets(target_a: str, target_b: str) -> str:
    ta = _normalize_ab_target(target_a, fallback="rotation_a")
    tb = _normalize_ab_target(target_b, fallback="rotation_b")
    if ta == "rotation_a" and tb == "equal_weight":
        return "rotation_vs_equal_weight"
    if ta == "risk_parity" and tb == "equal_weight":
        return "risk_parity_vs_equal_weight"
    if ta == "rotation_a" and tb == "risk_parity":
        return "rotation_vs_risk_parity"
    if ta == "equal_weight" and tb == "cash":
        return "equal_weight_vs_cash"
    if ta == "rotation_a" and tb == "rotation_b":
        return "custom_ab"
    return "custom_targets"


def _resolve_ab_targets(
    *,
    target_a: str | None,
    target_b: str | None,
    comparison_mode: str | None,
) -> tuple[str, str, str, str, str]:
    has_explicit_targets = bool(str(target_a or "").strip()) or bool(
        str(target_b or "").strip()
    )
    if has_explicit_targets:
        ta = _normalize_ab_target(target_a, fallback="rotation_a")
        tb = _normalize_ab_target(target_b, fallback="rotation_b")
    else:
        ta, tb = _ab_mode_to_targets(comparison_mode)
    mode = _legacy_mode_from_targets(ta, tb)
    return ta, tb, mode, _AB_TARGET_LABELS[ta], _AB_TARGET_LABELS[tb]


def _sim_ohlc_from_close(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    c = close.astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    o = c.shift(1).ffill()
    if not o.empty:
        o.iloc[0, :] = c.iloc[0, :]
    d = (c / o - 1.0).abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pad = (0.002 + 0.5 * d).clip(lower=0.001, upper=0.08)
    h = np.maximum(c, o) * (1.0 + pad)
    l = np.minimum(c, o) * (1.0 - pad)
    return {
        "open": o.astype(float),
        "high": h.astype(float),
        "low": l.astype(float),
        "close": c.astype(float),
    }


def _sim_vol_index_proxy(close: pd.DataFrame) -> dict[str, pd.Series]:
    # Proxy a smooth "fear index" from cross-sectional realized vol to support
    # vol-index timing rules in synthetic worlds.
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    rv = ret.rolling(20, min_periods=5).std(ddof=1) * np.sqrt(252.0) * 100.0
    lvl = rv.mean(axis=1, skipna=True).astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    if lvl.dropna().empty:
        lvl = pd.Series(20.0, index=close.index, dtype=float)
    return {
        "VIX": lvl,
        "VXN": lvl * 1.05,
        "GVZ": lvl * 0.85,
        "OVX": lvl * 1.10,
        "WAVOL": lvl,
    }


@contextmanager
def _patched_rotation_loaders(
    close_hfq: pd.DataFrame,
    close_qfq: pd.DataFrame,
    close_none: pd.DataFrame,
    ohlc_hfq: dict[str, pd.DataFrame],
    ohlc_none: dict[str, pd.DataFrame],
):
    old_close = _rot._load_close_prices
    old_hl = _rot._load_high_low_prices
    old_ohlc = _rot._load_ohlc_prices
    old_vol_amt = _rot._load_volume_amount

    def _slice_df(df: pd.DataFrame, codes: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        idx = pd.to_datetime(df.index)
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        out = df.copy()
        out.index = idx
        out.columns = [str(c) for c in out.columns]
        out = out.loc[(out.index >= s) & (out.index <= e)]
        out = out.reindex(columns=[str(c) for c in codes])
        return out.sort_index()

    def _slice_ohlc(src: dict[str, pd.DataFrame], codes: list[str], start: dt.date, end: dt.date) -> dict[str, pd.DataFrame]:
        return {k: _slice_df(v, codes, start, end) for k, v in src.items()}

    def _load_close(_db: Any, *, codes: list[str], start: dt.date, end: dt.date, adjust: str) -> pd.DataFrame:
        a = str(adjust or "none").strip().lower()
        if a == "hfq":
            return _slice_df(close_hfq, codes, start, end)
        if a == "qfq":
            return _slice_df(close_qfq, codes, start, end)
        return _slice_df(close_none, codes, start, end)

    def _load_hl(_db: Any, *, codes: list[str], start: dt.date, end: dt.date, adjust: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        src = ohlc_hfq if str(adjust or "").strip().lower() == "qfq" else ohlc_none
        d = _slice_ohlc(src, codes, start, end)
        return d.get("high", pd.DataFrame()), d.get("low", pd.DataFrame())

    def _load_ohlc(_db: Any, *, codes: list[str], start: dt.date, end: dt.date, adjust: str) -> dict[str, pd.DataFrame]:
        src = ohlc_hfq if str(adjust or "").strip().lower() == "hfq" else ohlc_none
        return _slice_ohlc(src, codes, start, end)

    def _load_vol_amt(_db: Any, *, codes: list[str], start: dt.date, end: dt.date, adjust: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        return pd.DataFrame(), pd.DataFrame()

    _rot._load_close_prices = _load_close
    _rot._load_high_low_prices = _load_hl
    _rot._load_ohlc_prices = _load_ohlc
    _rot._load_volume_amount = _load_vol_amt
    try:
        yield
    finally:
        _rot._load_close_prices = old_close
        _rot._load_high_low_prices = old_hl
        _rot._load_ohlc_prices = old_ohlc
        _rot._load_volume_amount = old_vol_amt


def _run_rotation_variant_on_sim(
    close: pd.DataFrame,
    strategy: dict[str, Any],
    *,
    ohlc_hfq: dict[str, pd.DataFrame] | None = None,
    ohlc_none: dict[str, pd.DataFrame] | None = None,
    vol_index_close: dict[str, pd.Series] | None = None,
) -> tuple[float, float]:
    close = close.astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    if close.empty:
        return float("nan"), 0.0
    s = dict(strategy or {})
    start_d = pd.to_datetime(close.index.min()).date()
    end_d = pd.to_datetime(close.index.max()).date()
    codes = [str(c) for c in close.columns]
    bt_type = str(s.get("bias_threshold_type", "quantile")).strip().lower()
    if bt_type == "fixed_value":
        bt_type = "fixed"
    inp = _rot.RotationInputs(
        codes=codes,
        start=start_d,
        end=end_d,
        rebalance=str(s.get("rebalance", "weekly")),
        rebalance_anchor=(None if s.get("rebalance_anchor") is None else int(s.get("rebalance_anchor"))),
        rebalance_shift=str(s.get("rebalance_shift", "prev")),
        exec_price=str(s.get("exec_price", "open")),
        top_k=int(s.get("top_k", 1) or 1),
        position_mode=str(s.get("position_mode", "adaptive")),
        entry_backfill=bool(s.get("entry_backfill", False)),
        entry_match_n=int(s.get("entry_match_n", 0) or 0),
        exit_match_n=int(s.get("exit_match_n", 0) or 0),
        lookback_days=int(s.get("lookback_days", 20) or 20),
        skip_days=int(s.get("skip_days", 0) or 0),
        score_method=str(s.get("score_method", "raw_mom")),
        risk_free_rate=float(s.get("risk_free_rate", 0.025) or 0.025),
        cost_bps=float(s.get("cost_bps", 0.0) or 0.0),
        trend_filter=bool(s.get("trend_filter", False)),
        trend_exit_filter=bool(s.get("trend_exit_filter", False)),
        trend_sma_window=int(s.get("trend_sma_window", 20) or 20),
        trend_ma_type=str(s.get("trend_ma_type", "sma")),
        bias_filter=bool(s.get("bias_filter", False)),
        bias_exit_filter=bool(s.get("bias_exit_filter", False)),
        bias_ma_window=int(s.get("bias_ma_window", 20) or 20),
        bias_level_window=str(s.get("bias_level_window", "all")),
        bias_threshold_type=bt_type,
        bias_quantile=float(s.get("bias_quantile", 95.0) or 95.0),
        bias_fixed_value=float(s.get("bias_fixed_value", 10.0) or 10.0),
        bias_min_periods=int(s.get("bias_min_periods", 20) or 20),
        group_enforce=bool(s.get("group_enforce", False)),
        group_pick_policy=str(s.get("group_pick_policy", "strongest_score")),
        asset_groups=s.get("asset_groups"),
        asset_momentum_floor_rules=s.get("asset_momentum_floor_rules"),
        asset_trend_rules=s.get("asset_trend_rules"),
        asset_bias_rules=s.get("asset_bias_rules"),
        asset_vol_index_rules=s.get("asset_vol_index_rules"),
        vol_index_close=(vol_index_close if vol_index_close is not None else _sim_vol_index_proxy(close)),
        dynamic_universe=True,
    )
    if ohlc_hfq is None:
        ohlc_hfq = _sim_ohlc_from_close(close)
    if ohlc_none is None:
        ohlc_none = ohlc_hfq
    with _patched_rotation_loaders(
        close_hfq=close,
        close_qfq=close,
        close_none=close,
        ohlc_hfq=ohlc_hfq,
        ohlc_none=ohlc_none,
    ):
        out = _rot.backtest_rotation(None, inp, lightweight=True)
    ann = _extract_annualized_return(out)
    exposure = float(out.get("avg_exposure")) if out.get("avg_exposure") is not None else 0.0
    return ann, exposure


def _run_rotation_variant_perf_on_sim(
    close: pd.DataFrame,
    strategy: dict[str, Any],
    *,
    ohlc_hfq: dict[str, pd.DataFrame] | None = None,
    ohlc_none: dict[str, pd.DataFrame] | None = None,
    vol_index_close: dict[str, pd.Series] | None = None,
) -> tuple[float, float]:
    close = close.astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    if close.empty:
        return float("nan"), float("nan")
    s = dict(strategy or {})
    start_d = pd.to_datetime(close.index.min()).date()
    end_d = pd.to_datetime(close.index.max()).date()
    codes = [str(c) for c in close.columns]
    bt_type = str(s.get("bias_threshold_type", "quantile")).strip().lower()
    if bt_type == "fixed_value":
        bt_type = "fixed"
    inp = _rot.RotationInputs(
        codes=codes,
        start=start_d,
        end=end_d,
        rebalance=str(s.get("rebalance", "weekly")),
        rebalance_anchor=(None if s.get("rebalance_anchor") is None else int(s.get("rebalance_anchor"))),
        rebalance_shift=str(s.get("rebalance_shift", "prev")),
        exec_price=str(s.get("exec_price", "open")),
        top_k=int(s.get("top_k", 1) or 1),
        position_mode=str(s.get("position_mode", "adaptive")),
        entry_backfill=bool(s.get("entry_backfill", False)),
        entry_match_n=int(s.get("entry_match_n", 0) or 0),
        exit_match_n=int(s.get("exit_match_n", 0) or 0),
        lookback_days=int(s.get("lookback_days", 20) or 20),
        skip_days=int(s.get("skip_days", 0) or 0),
        score_method=str(s.get("score_method", "raw_mom")),
        risk_free_rate=float(s.get("risk_free_rate", 0.025) or 0.025),
        cost_bps=float(s.get("cost_bps", 0.0) or 0.0),
        trend_filter=bool(s.get("trend_filter", False)),
        trend_exit_filter=bool(s.get("trend_exit_filter", False)),
        trend_sma_window=int(s.get("trend_sma_window", 20) or 20),
        trend_ma_type=str(s.get("trend_ma_type", "sma")),
        bias_filter=bool(s.get("bias_filter", False)),
        bias_exit_filter=bool(s.get("bias_exit_filter", False)),
        bias_ma_window=int(s.get("bias_ma_window", 20) or 20),
        bias_level_window=str(s.get("bias_level_window", "all")),
        bias_threshold_type=bt_type,
        bias_quantile=float(s.get("bias_quantile", 95.0) or 95.0),
        bias_fixed_value=float(s.get("bias_fixed_value", 10.0) or 10.0),
        bias_min_periods=int(s.get("bias_min_periods", 20) or 20),
        group_enforce=bool(s.get("group_enforce", False)),
        group_pick_policy=str(s.get("group_pick_policy", "strongest_score")),
        asset_groups=s.get("asset_groups"),
        asset_momentum_floor_rules=s.get("asset_momentum_floor_rules"),
        asset_trend_rules=s.get("asset_trend_rules"),
        asset_bias_rules=s.get("asset_bias_rules"),
        asset_vol_index_rules=s.get("asset_vol_index_rules"),
        vol_index_close=(vol_index_close if vol_index_close is not None else _sim_vol_index_proxy(close)),
        dynamic_universe=True,
    )
    if ohlc_hfq is None:
        ohlc_hfq = _sim_ohlc_from_close(close)
    if ohlc_none is None:
        ohlc_none = ohlc_hfq
    with _patched_rotation_loaders(
        close_hfq=close,
        close_qfq=close,
        close_none=close,
        ohlc_hfq=ohlc_hfq,
        ohlc_none=ohlc_none,
    ):
        out = _rot.backtest_rotation(None, inp, lightweight=True)
    return _extract_annualized_return(out), _extract_max_drawdown_from_bt(out)


def _run_rotation_variant_with_series_on_sim(
    close: pd.DataFrame,
    strategy: dict[str, Any],
    *,
    ohlc_hfq: dict[str, pd.DataFrame] | None = None,
    ohlc_none: dict[str, pd.DataFrame] | None = None,
    vol_index_close: dict[str, pd.Series] | None = None,
) -> dict[str, Any]:
    close = close.astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    if close.empty:
        return {"ok": False, "error": "empty_prices"}
    s = dict(strategy or {})
    start_d = pd.to_datetime(close.index.min()).date()
    end_d = pd.to_datetime(close.index.max()).date()
    codes = [str(c) for c in close.columns]
    bt_type = str(s.get("bias_threshold_type", "quantile")).strip().lower()
    if bt_type == "fixed_value":
        bt_type = "fixed"
    inp = _rot.RotationInputs(
        codes=codes,
        start=start_d,
        end=end_d,
        rebalance=str(s.get("rebalance", "weekly")),
        rebalance_anchor=(None if s.get("rebalance_anchor") is None else int(s.get("rebalance_anchor"))),
        rebalance_shift=str(s.get("rebalance_shift", "prev")),
        exec_price=str(s.get("exec_price", "open")),
        top_k=int(s.get("top_k", 1) or 1),
        position_mode=str(s.get("position_mode", "adaptive")),
        entry_backfill=bool(s.get("entry_backfill", False)),
        entry_match_n=int(s.get("entry_match_n", 0) or 0),
        exit_match_n=int(s.get("exit_match_n", 0) or 0),
        lookback_days=int(s.get("lookback_days", 20) or 20),
        skip_days=int(s.get("skip_days", 0) or 0),
        score_method=str(s.get("score_method", "raw_mom")),
        risk_free_rate=float(s.get("risk_free_rate", 0.025) or 0.025),
        cost_bps=float(s.get("cost_bps", 0.0) or 0.0),
        trend_filter=bool(s.get("trend_filter", False)),
        trend_exit_filter=bool(s.get("trend_exit_filter", False)),
        trend_sma_window=int(s.get("trend_sma_window", 20) or 20),
        trend_ma_type=str(s.get("trend_ma_type", "sma")),
        bias_filter=bool(s.get("bias_filter", False)),
        bias_exit_filter=bool(s.get("bias_exit_filter", False)),
        bias_ma_window=int(s.get("bias_ma_window", 20) or 20),
        bias_level_window=str(s.get("bias_level_window", "all")),
        bias_threshold_type=bt_type,
        bias_quantile=float(s.get("bias_quantile", 95.0) or 95.0),
        bias_fixed_value=float(s.get("bias_fixed_value", 10.0) or 10.0),
        bias_min_periods=int(s.get("bias_min_periods", 20) or 20),
        group_enforce=bool(s.get("group_enforce", False)),
        group_pick_policy=str(s.get("group_pick_policy", "strongest_score")),
        asset_groups=s.get("asset_groups"),
        asset_momentum_floor_rules=s.get("asset_momentum_floor_rules"),
        asset_trend_rules=s.get("asset_trend_rules"),
        asset_bias_rules=s.get("asset_bias_rules"),
        asset_vol_index_rules=s.get("asset_vol_index_rules"),
        vol_index_close=(vol_index_close if vol_index_close is not None else _sim_vol_index_proxy(close)),
        dynamic_universe=True,
    )
    if ohlc_hfq is None:
        ohlc_hfq = _sim_ohlc_from_close(close)
    if ohlc_none is None:
        ohlc_none = ohlc_hfq
    with _patched_rotation_loaders(
        close_hfq=close,
        close_qfq=close,
        close_none=close,
        ohlc_hfq=ohlc_hfq,
        ohlc_none=ohlc_none,
    ):
        out = _rot.backtest_rotation(None, inp, lightweight=False)
    nav = (out.get("nav") or {})
    dates = list(nav.get("dates") or [])
    vals = list(((nav.get("series") or {}).get("ROTATION") or []))
    m = {}
    if dates and vals and len(dates) == len(vals):
        nav_s = pd.Series(pd.to_numeric(vals, errors="coerce"), index=pd.to_datetime(dates), dtype=float)
        m = _metrics_from_nav(nav_s)
    else:
        ann = _extract_annualized_return(out)
        m = {"cagr": ann if np.isfinite(ann) else None, "vol": None, "sharpe": None, "max_drawdown": None, "calmar": None}
    return {
        "ok": True,
        "series": {"dates": dates, "nav": [float(x) for x in vals]},
        "metrics": m,
        "holdings": list(out.get("holdings") or []),
    }


def _paired_permutation_pvalue(diff: np.ndarray, *, n_perm: int, seed: int | None) -> float:
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    obs = float(np.mean(d))
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(int(max(1000, n_perm))):
        signs = rng.choice([-1.0, 1.0], size=d.size)
        stat = float(np.mean(d * signs))
        if stat >= obs:
            ge += 1
    return float((ge + 1) / (int(max(1000, n_perm)) + 1))


def _bootstrap_ci(diff: np.ndarray, *, n_boot: int, seed: int | None) -> dict[str, list[float]]:
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"mean": [float("nan"), float("nan")], "median": [float("nan"), float("nan")]}
    rng = np.random.default_rng(seed)
    m = np.empty(int(max(1000, n_boot)), dtype=float)
    md = np.empty(int(max(1000, n_boot)), dtype=float)
    n = d.size
    for i in range(m.size):
        samp = d[rng.integers(0, n, size=n)]
        m[i] = float(np.mean(samp))
        md[i] = float(np.median(samp))
    return {
        "mean": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
        "median": [float(np.percentile(md, 2.5)), float(np.percentile(md, 97.5))],
    }


def _normal_sf(z: float) -> float:
    if not np.isfinite(z):
        return float("nan")
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def _sign_test_pvalue_one_sided(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0.0]
    n = int(d.size)
    if n <= 0:
        return float("nan")
    k = int(np.sum(d > 0.0))
    p = 0.0
    for i in range(k, n + 1):
        p += math.comb(n, i) * (0.5 ** n)
    return float(min(1.0, max(0.0, p)))


def _wilcoxon_signed_rank_pvalue_one_sided(diff: np.ndarray) -> float:
    """
    One-sided Wilcoxon signed-rank test (H1: median(diff) > 0), normal approximation.
    """
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0.0]
    n = int(d.size)
    if n <= 0:
        return float("nan")
    abs_d = np.abs(d)
    order = np.argsort(abs_d, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs_d[order[j]] == abs_d[order[i]]:
            j += 1
        r = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = float(r)
        i = j
    w_plus = float(np.sum(ranks[d > 0.0]))
    mean_w = float(n * (n + 1) / 4.0)
    var_w = float(n * (n + 1) * (2 * n + 1) / 24.0)
    if var_w <= 0:
        return float("nan")
    z = (w_plus - mean_w - 0.5) / math.sqrt(var_w)
    return float(min(1.0, max(0.0, _normal_sf(z))))


def _collect_ab_samples(
    *,
    start: str,
    end: str,
    n_worlds: int,
    n_assets: int,
    vol_low: float,
    vol_high: float,
    corr_low: float | None,
    corr_high: float | None,
    mu_low: float | None,
    mu_high: float | None,
    strategy_a: dict[str, Any],
    strategy_b: dict[str, Any],
    target_a: str,
    target_b: str,
    world_seeds: list[int],
    n_jobs: int,
) -> tuple[list[float], list[float], list[float], list[float], int]:
    a_vals: list[float] = []
    b_vals: list[float] = []
    exp_a: list[float] = []
    exp_b: list[float] = []
    jobs = int(n_jobs)
    if jobs <= 0:
        jobs = max(1, int(os.cpu_count() or 1))
    jobs = max(1, min(jobs, int(n_worlds)))
    jobs_effective = int(jobs)

    if jobs == 1:
        for ws in world_seeds:
            one = _eval_ab_world(
                start=start,
                end=end,
                n_assets=int(n_assets),
                vol_low=float(vol_low),
                vol_high=float(vol_high),
                corr_low=(None if corr_low is None else float(corr_low)),
                corr_high=(None if corr_high is None else float(corr_high)),
                mu_low=(None if mu_low is None else float(mu_low)),
                mu_high=(None if mu_high is None else float(mu_high)),
                world_seed=int(ws),
                strategy_a=strategy_a,
                strategy_b=strategy_b,
                target_a=target_a,
                target_b=target_b,
            )
            if one is None:
                continue
            aa, bb, exa, exb = one
            a_vals.append(float(aa))
            b_vals.append(float(bb))
            exp_a.append(float(exa))
            exp_b.append(float(exb))
        return a_vals, b_vals, exp_a, exp_b, jobs_effective

    try:
        mp_ctx = None
        if sys.platform != "win32":
            try:
                mp_ctx = mp.get_context("fork")
            except ValueError:
                mp_ctx = None
        with ProcessPoolExecutor(max_workers=jobs, mp_context=mp_ctx) as ex:
            futs = [
                ex.submit(
                    _eval_ab_world,
                    start=start,
                    end=end,
                    n_assets=int(n_assets),
                    vol_low=float(vol_low),
                    vol_high=float(vol_high),
                    corr_low=(None if corr_low is None else float(corr_low)),
                    corr_high=(None if corr_high is None else float(corr_high)),
                    mu_low=(None if mu_low is None else float(mu_low)),
                    mu_high=(None if mu_high is None else float(mu_high)),
                    world_seed=int(ws),
                    strategy_a=strategy_a,
                    strategy_b=strategy_b,
                    target_a=target_a,
                    target_b=target_b,
                )
                for ws in world_seeds
            ]
            for f in futs:
                one = f.result()
                if one is None:
                    continue
                aa, bb, exa, exb = one
                a_vals.append(float(aa))
                b_vals.append(float(bb))
                exp_a.append(float(exa))
                exp_b.append(float(exb))
    except (BrokenProcessPool, RuntimeError, OSError, FileNotFoundError):
        jobs_effective = 1
        for ws in world_seeds:
            one = _eval_ab_world(
                start=start,
                end=end,
                n_assets=int(n_assets),
                vol_low=float(vol_low),
                vol_high=float(vol_high),
                corr_low=(None if corr_low is None else float(corr_low)),
                corr_high=(None if corr_high is None else float(corr_high)),
                mu_low=(None if mu_low is None else float(mu_low)),
                mu_high=(None if mu_high is None else float(mu_high)),
                world_seed=int(ws),
                strategy_a=strategy_a,
                strategy_b=strategy_b,
                target_a=target_a,
                target_b=target_b,
            )
            if one is None:
                continue
            aa, bb, exa, exb = one
            a_vals.append(float(aa))
            b_vals.append(float(bb))
            exp_a.append(float(exa))
            exp_b.append(float(exb))
    return a_vals, b_vals, exp_a, exp_b, jobs_effective


def _eval_mc_world(
    *,
    start: str,
    end: str,
    n_assets: int,
    vol_low: float,
    vol_high: float,
    corr_low: float | None,
    corr_high: float | None,
    mu_low: float | None,
    mu_high: float | None,
    world_seed: int,
    strategy_a: dict[str, Any],
) -> tuple[float, float, float, float, float, float] | None:
    sim = simulate_gbm_prices(
        start=start,
        end=end,
        cfg=SimConfig(
            n_assets=int(n_assets),
            vol_low=float(vol_low),
            vol_high=float(vol_high),
            corr_low=(None if corr_low is None else float(corr_low)),
            corr_high=(None if corr_high is None else float(corr_high)),
            mu_low=(None if mu_low is None else float(mu_low)),
            mu_high=(None if mu_high is None else float(mu_high)),
            seed=int(world_seed),
        ),
    )
    if not bool(sim.get("ok")):
        return None
    dates = pd.to_datetime(((sim.get("series") or {}).get("dates") or []))
    close_map = ((sim.get("series") or {}).get("close") or {})
    close = pd.DataFrame(close_map, index=dates, dtype=float)
    if close.empty:
        return None
    ohlc_hfq = _sim_ohlc_from_close(close)
    vol_idx = _sim_vol_index_proxy(close)
    cagr_rot, mdd_rot = _run_rotation_variant_perf_on_sim(
        close,
        strategy_a or {},
        ohlc_hfq=ohlc_hfq,
        ohlc_none=ohlc_hfq,
        vol_index_close=vol_idx,
    )
    ew = backtest_equal_weight_weekly(close)
    cagr_ew = _extract_annualized_return(ew)
    mdd_ew = _extract_max_drawdown_from_bt(ew)
    ann_vols = (sim.get("assets") or {}).get("ann_vols") or {}
    rp = backtest_risk_parity_inverse_vol(close, ann_vols=ann_vols)
    cagr_rp = _extract_annualized_return(rp)
    mdd_rp = _extract_max_drawdown_from_bt(rp)
    vals = [cagr_rot, cagr_ew, cagr_rp, mdd_rot, mdd_ew, mdd_rp]
    return tuple(float(x) for x in vals) if all(np.isfinite(float(x)) for x in vals) else None


def _eval_ab_world(
    *,
    start: str,
    end: str,
    n_assets: int,
    vol_low: float,
    vol_high: float,
    corr_low: float | None,
    corr_high: float | None,
    mu_low: float | None,
    mu_high: float | None,
    world_seed: int,
    strategy_a: dict[str, Any],
    strategy_b: dict[str, Any],
    target_a: str = "rotation_a",
    target_b: str = "rotation_b",
) -> tuple[float, float, float, float] | None:
    sim = simulate_gbm_prices(
        start=start,
        end=end,
        cfg=SimConfig(
            n_assets=int(n_assets),
            vol_low=float(vol_low),
            vol_high=float(vol_high),
            corr_low=(None if corr_low is None else float(corr_low)),
            corr_high=(None if corr_high is None else float(corr_high)),
            mu_low=(None if mu_low is None else float(mu_low)),
            mu_high=(None if mu_high is None else float(mu_high)),
            seed=int(world_seed),
        ),
    )
    if not bool(sim.get("ok")):
        return None
    dates = pd.to_datetime(((sim.get("series") or {}).get("dates") or []))
    close_map = ((sim.get("series") or {}).get("close") or {})
    close = pd.DataFrame(close_map, index=dates, dtype=float)
    ohlc_hfq = _sim_ohlc_from_close(close)
    vol_idx = _sim_vol_index_proxy(close)
    ann_vols = (sim.get("assets") or {}).get("ann_vols") or {}

    def _eval_target(target: str) -> tuple[float, float]:
        t = _normalize_ab_target(target, fallback="rotation_a")
        if t == "cash":
            return 0.0, 0.0
        if t == "equal_weight":
            ew = backtest_equal_weight_weekly(close)
            return _extract_annualized_return(ew), 1.0
        if t == "risk_parity":
            rp = backtest_risk_parity_inverse_vol(close, ann_vols=ann_vols)
            return _extract_annualized_return(rp), 1.0
        if t == "rotation_b":
            return _run_rotation_variant_on_sim(
                close,
                strategy_b or {},
                ohlc_hfq=ohlc_hfq,
                ohlc_none=ohlc_hfq,
                vol_index_close=vol_idx,
            )
        return _run_rotation_variant_on_sim(
            close,
            strategy_a or {},
            ohlc_hfq=ohlc_hfq,
            ohlc_none=ohlc_hfq,
            vol_index_close=vol_idx,
        )

    aa, exa = _eval_target(target_a)
    bb, exb = _eval_target(target_b)
    if np.isfinite(aa) and np.isfinite(bb):
        return float(aa), float(bb), float(exa), float(exb)
    return None


def gbm_ab_significance(
    *,
    start: str,
    end: str | None,
    n_worlds: int,
    n_assets: int,
    vol_low: float,
    vol_high: float,
    corr_low: float | None = None,
    corr_high: float | None = None,
    mu_low: float | None = None,
    mu_high: float | None = None,
    seed: int | None,
    strategy_a: dict[str, Any],
    strategy_b: dict[str, Any],
    n_perm: int = 5000,
    n_boot: int = 3000,
    n_jobs: int = 1,
    target_a: str | None = None,
    target_b: str | None = None,
    comparison_mode: str = "custom_ab",
    stability_repeats: int = 0,
    stability_worlds: int = 100,
) -> dict[str, Any]:
    ta, tb, mode, label_a, label_b = _resolve_ab_targets(
        target_a=target_a,
        target_b=target_b,
        comparison_mode=comparison_mode,
    )
    end = str(end or _today_last_business_day_yyyymmdd())
    n_worlds = int(max(2, n_worlds))
    rng = np.random.default_rng(seed)
    world_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n_worlds)]
    a_vals, b_vals, exp_a, exp_b, jobs_effective = _collect_ab_samples(
        start=start,
        end=end,
        n_worlds=int(n_worlds),
        n_assets=int(n_assets),
        vol_low=float(vol_low),
        vol_high=float(vol_high),
        corr_low=(None if corr_low is None else float(corr_low)),
        corr_high=(None if corr_high is None else float(corr_high)),
        mu_low=(None if mu_low is None else float(mu_low)),
        mu_high=(None if mu_high is None else float(mu_high)),
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        target_a=ta,
        target_b=tb,
        world_seeds=world_seeds,
        n_jobs=int(n_jobs),
    )
    a_arr = np.asarray(a_vals, dtype=float)
    b_arr = np.asarray(b_vals, dtype=float)
    diff = a_arr - b_arr
    p = _paired_permutation_pvalue(diff, n_perm=n_perm, seed=seed)
    ci = _bootstrap_ci(diff, n_boot=n_boot, seed=(None if seed is None else int(seed) + 7))
    p_sign = _sign_test_pvalue_one_sided(diff)
    p_wilcoxon = _wilcoxon_signed_rank_pvalue_one_sided(diff)

    rep = int(max(0, stability_repeats))
    ws = int(max(2, stability_worlds))
    stab: dict[str, Any] = {"enabled": bool(rep > 0), "repeats": rep, "worlds_per_repeat": ws}
    if rep > 0:
        rep_seeds: list[int] = []
        rep_mean_diff: list[float] = []
        rep_p_sign: list[float] = []
        rep_p_perm: list[float] = []
        seed_rng = np.random.default_rng(None if seed is None else int(seed) + 7919)
        for _ in range(rep):
            srep = int(seed_rng.integers(0, 2**31 - 1))
            rep_seeds.append(srep)
            world_seeds_rep = [int(seed_rng.integers(0, 2**31 - 1)) for _ in range(ws)]
            a_rep, b_rep, _, _, _ = _collect_ab_samples(
                start=start,
                end=end,
                n_worlds=int(ws),
                n_assets=int(n_assets),
                vol_low=float(vol_low),
                vol_high=float(vol_high),
                corr_low=(None if corr_low is None else float(corr_low)),
                corr_high=(None if corr_high is None else float(corr_high)),
                mu_low=(None if mu_low is None else float(mu_low)),
                mu_high=(None if mu_high is None else float(mu_high)),
                strategy_a=strategy_a,
                strategy_b=strategy_b,
                target_a=ta,
                target_b=tb,
                world_seeds=world_seeds_rep,
                n_jobs=int(n_jobs),
            )
            drep = np.asarray(a_rep, dtype=float) - np.asarray(b_rep, dtype=float)
            rep_mean_diff.append(float(np.mean(drep)) if drep.size else float("nan"))
            rep_p_sign.append(float(_sign_test_pvalue_one_sided(drep)))
            rep_p_perm.append(float(_paired_permutation_pvalue(drep, n_perm=max(1000, min(int(n_perm), 2000)), seed=srep)))
        arr_md = np.asarray(rep_mean_diff, dtype=float)
        arr_ps = np.asarray(rep_p_sign, dtype=float)
        arr_pp = np.asarray(rep_p_perm, dtype=float)
        stab.update(
            {
                "seeds": rep_seeds,
                "mean_diff": [float(x) for x in rep_mean_diff],
                "pvalue_sign_test_one_sided": [float(x) for x in rep_p_sign],
                "pvalue_permutation_one_sided": [float(x) for x in rep_p_perm],
                "positive_mean_fraction": float(np.mean(arr_md > 0.0)) if arr_md.size else float("nan"),
                "sign_sig_fraction": float(np.mean(arr_ps < 0.05)) if arr_ps.size else float("nan"),
                "perm_sig_fraction": float(np.mean(arr_pp < 0.05)) if arr_pp.size else float("nan"),
            }
        )
    return {
        "ok": True,
        "meta": {
            "start": start,
            "end": end,
            "n_worlds": int(n_worlds),
            "n_assets": int(n_assets),
            "vol_low": float(vol_low),
            "vol_high": float(vol_high),
            "corr_low": (None if corr_low is None else float(corr_low)),
            "corr_high": (None if corr_high is None else float(corr_high)),
            "mu_low": (None if mu_low is None else float(mu_low)),
            "mu_high": (None if mu_high is None else float(mu_high)),
            "seed": seed,
            "n_jobs": int(jobs_effective),
            "n_samples": int(diff.size),
            "comparison_mode": mode,
            "comparison_target_a": ta,
            "comparison_target_b": tb,
        },
        "comparison": {
            "mode": mode,
            "target_a": ta,
            "target_b": tb,
            "label_a": label_a,
            "label_b": label_b,
        },
        "stats": {
            "mean_diff": float(np.mean(diff)) if diff.size else float("nan"),
            "median_diff": float(np.median(diff)) if diff.size else float("nan"),
            "win_rate": float(np.mean(diff > 0.0)) if diff.size else float("nan"),
            "pvalue_permutation_one_sided": float(p),
            "pvalue_sign_test_one_sided": float(p_sign),
            "pvalue_wilcoxon_one_sided": float(p_wilcoxon),
            "bootstrap_ci_95": ci,
            "avg_exposure_a": float(np.mean(np.asarray(exp_a, dtype=float))) if exp_a else float("nan"),
            "avg_exposure_b": float(np.mean(np.asarray(exp_b, dtype=float))) if exp_b else float("nan"),
        },
        "robustness": {
            "sign_test_one_sided": float(p_sign),
            "wilcoxon_signed_rank_one_sided": float(p_wilcoxon),
            "seed_stability": stab,
        },
        "dist": {
            "annualized_return_a": [float(x) for x in a_vals],
            "annualized_return_b": [float(x) for x in b_vals],
            "diff_a_minus_b": [float(x) for x in diff.tolist()],
        },
    }

