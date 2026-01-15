from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import secrets
import scipy.stats as st

from .baseline import (
    TRADING_DAYS_PER_YEAR,
    _annualized_return,
    _annualized_vol,
    _max_drawdown,
    _sharpe,
)


@dataclass(frozen=True)
class MonteCarloConfig:
    n_sims: int = 2000
    block_size: int = 5
    seed: int | None = None


def _circular_block_bootstrap_indices(n: int, *, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    b = int(block_size)
    if b <= 0:
        b = 1
    starts = rng.integers(0, n, size=int(np.ceil(n / b)))
    idx = []
    for s in starts:
        idx.extend(((s + np.arange(b)) % n).tolist())
        if len(idx) >= n:
            break
    return np.array(idx[:n], dtype=int)


def _summarize(samples: np.ndarray, *, observed: float) -> dict[str, Any]:
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "observed": float(observed),
            "mean": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p_value_le_0": float("nan"),
        }
    return {
        "observed": float(observed),
        "mean": float(np.mean(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
        "p_value_le_0": float(np.mean(x <= 0.0)),
    }


def _histogram(samples: np.ndarray, *, bins: int = 40, clip_q: tuple[float, float] = (0.01, 0.99)) -> dict[str, Any]:
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"bin_edges": [], "counts": [], "underflow": 0, "overflow": 0}
    lo = float(np.quantile(x, clip_q[0]))
    hi = float(np.quantile(x, clip_q[1]))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = float(np.min(x)), float(np.max(x))
    if lo == hi:
        # Expand range in a scale-aware way (absolute epsilon may be rounded away for huge values).
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


def _stats(x: np.ndarray) -> dict[str, Any]:
    xs = np.asarray(x, dtype=float)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "pos_ratio": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(xs.size),
        "mean": float(np.mean(xs)),
        "std": float(np.std(xs, ddof=1)) if xs.size >= 2 else float("nan"),
        "p05": float(np.quantile(xs, 0.05)),
        "p10": float(np.quantile(xs, 0.10)),
        "p50": float(np.quantile(xs, 0.50)),
        "p90": float(np.quantile(xs, 0.90)),
        "p95": float(np.quantile(xs, 0.95)),
        "pos_ratio": float(np.mean(xs > 0.0)),
        "min": float(np.min(xs)),
        "max": float(np.max(xs)),
    }


def _anderson_darling_stat(x: np.ndarray, cdf: Callable[[np.ndarray], np.ndarray]) -> float:
    # Generic AD statistic (no p-value). Assumes continuous distribution.
    xs = np.asarray(x, dtype=float)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return float("nan")
    xs = np.sort(xs)
    n = xs.size
    p = cdf(xs)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1, dtype=float)
    s = np.sum((2 * i - 1) * (np.log(p) + np.log(1 - p[::-1])))
    return float(-n - s / n)


def _fit_one_distribution(x: np.ndarray, dist: str) -> dict[str, Any]:
    xs = np.asarray(x, dtype=float)
    xs = xs[np.isfinite(xs)]
    n = xs.size
    if n == 0:
        return {"ok": False, "error": "empty"}

    try:
        if dist == "normal":
            mu, sigma = st.norm.fit(xs)
            frozen = st.norm(loc=mu, scale=sigma)
            k = 2
            params = {"mu": float(mu), "sigma": float(sigma)}
        elif dist == "t":
            df, loc, scale = st.t.fit(xs)
            frozen = st.t(df=df, loc=loc, scale=scale)
            k = 3
            params = {"df": float(df), "loc": float(loc), "scale": float(scale)}
        elif dist == "lognorm":
            if np.any(xs <= 0):
                return {"ok": False, "error": "lognorm requires x>0"}
            shape, loc, scale = st.lognorm.fit(xs)
            frozen = st.lognorm(s=shape, loc=loc, scale=scale)
            k = 3
            params = {"s": float(shape), "loc": float(loc), "scale": float(scale)}
        else:
            return {"ok": False, "error": f"unknown dist={dist}"}

        ll = float(np.sum(frozen.logpdf(xs)))
        aic = float(2 * k - 2 * ll)
        bic = float(k * np.log(n) - 2 * ll)
        ks = st.kstest(xs, frozen.cdf)
        ad = _anderson_darling_stat(xs, frozen.cdf)
        return {
            "ok": True,
            "params": params,
            "loglik": ll,
            "aic": aic,
            "bic": bic,
            "ks": {"stat": float(ks.statistic), "p_value": float(ks.pvalue)},
            "ad": {"stat": float(ad)},
        }
    except (ValueError, TypeError, ZeroDivisionError, FloatingPointError) as e:  # pragma: no cover
        return {"ok": False, "error": str(e)}


def _qq_points(x: np.ndarray, frozen, *, n_points: int = 80) -> dict[str, Any]:
    xs = np.asarray(x, dtype=float)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return {"p": [], "emp": [], "theory": []}
    ps = np.linspace(0.01, 0.99, int(n_points))
    emp = np.quantile(xs, ps).astype(float)
    th = frozen.ppf(ps).astype(float)
    emp = emp[np.isfinite(th)]
    th = th[np.isfinite(th)]
    return {"p": ps.astype(float).tolist(), "emp": emp.tolist(), "theory": th.tolist()}


def bootstrap_metrics_from_daily_returns(
    daily_ret: pd.Series,
    *,
    rf: float,
    cfg: MonteCarloConfig,
    ann_factor: int = TRADING_DAYS_PER_YEAR,
    extra_metrics: dict[str, Callable[[pd.Series, pd.Series], float]] | None = None,
    period_freq: str | None = None,
) -> dict[str, Any]:
    """
    Monte Carlo via circular block bootstrap on daily returns.

    Returns summary per metric with (observed, mean, p05/p50/p95, p_value_le_0).
    """
    if cfg.n_sims <= 0:
        raise ValueError("n_sims must be > 0")
    if cfg.block_size <= 0:
        raise ValueError("block_size must be > 0")

    r = daily_ret.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # keep index if provided (for resample-based period returns)
    try:
        r.index = pd.to_datetime(r.index)
    except Exception:  # pragma: no cover
        pass
    # drop the first day if it's a synthetic 0 from pct_change
    if len(r) >= 2 and r.iloc[0] == 0.0:
        r = r.iloc[1:]
    n = int(len(r))
    # Allow very short series (e.g. test fixtures) but results will be low-signal.
    if n <= 0:
        raise ValueError("not enough daily returns for monte carlo")

    nav_obs = (1.0 + r).cumprod()
    nav_obs.iloc[0] = 1.0
    obs = {
        "cumulative_return": float(nav_obs.iloc[-1] - 1.0),
        "annualized_return": float(_annualized_return(nav_obs, ann_factor=ann_factor)),
        "annualized_volatility": float(_annualized_vol(r, ann_factor=ann_factor)),
        "max_drawdown": float(_max_drawdown(nav_obs)),
        "sharpe_ratio": float(_sharpe(r, rf=float(rf), ann_factor=ann_factor)),
    }

    extra_metrics = extra_metrics or {}
    for k, fn in extra_metrics.items():
        try:
            obs[k] = float(fn(r, nav_obs))
        except (ValueError, TypeError):
            obs[k] = float("nan")

    # If seed is not provided, generate one from system entropy so results are non-reproducible by default,
    # but still reproducible if user re-runs with the returned seed_used.
    seed_used = int(cfg.seed) if cfg.seed is not None else secrets.randbits(64)
    rng = np.random.default_rng(seed_used)
    sims: dict[str, list[float]] = {k: [] for k in obs.keys()}
    period_samples: list[float] = []
    for _ in range(int(cfg.n_sims)):
        idx = _circular_block_bootstrap_indices(n, block_size=cfg.block_size, rng=rng)
        rr = pd.Series(r.to_numpy(dtype=float)[idx], index=r.index)
        nav = (1.0 + rr).cumprod()
        nav.iloc[0] = 1.0
        sims["cumulative_return"].append(float(nav.iloc[-1] - 1.0))
        sims["annualized_return"].append(float(_annualized_return(nav, ann_factor=ann_factor)))
        sims["annualized_volatility"].append(float(_annualized_vol(rr, ann_factor=ann_factor)))
        sims["max_drawdown"].append(float(_max_drawdown(nav)))
        sims["sharpe_ratio"].append(float(_sharpe(rr, rf=float(rf), ann_factor=ann_factor)))
        for k, fn in extra_metrics.items():
            try:
                sims[k].append(float(fn(rr, nav)))
            except (ValueError, TypeError):
                sims[k].append(float("nan"))

        if period_freq:
            try:
                pr = nav.resample(str(period_freq)).last().pct_change().dropna().to_numpy(dtype=float)
                if pr.size:
                    period_samples.extend([float(x) for x in pr if np.isfinite(float(x))])
            except Exception:  # pragma: no cover
                pass

    # Fit candidate distributions for each metric.
    candidates = ["normal", "t", "lognorm"]
    out = {}
    for k, v in sims.items():
        arr = np.asarray(v, dtype=float)
        out[k] = _summarize(arr, observed=obs[k])
        out[k]["hist"] = _histogram(arr, bins=40, clip_q=(0.01, 0.99))
        fits = {d: _fit_one_distribution(arr, d) for d in candidates}
        # Determine best by BIC among ok fits
        ok = [(d, fits[d].get("bic")) for d in candidates if fits[d].get("ok") and np.isfinite(fits[d].get("bic"))]
        ok.sort(key=lambda x: x[1])
        best = ok[0][0] if ok else None
        # attach QQ points for each dist (limited points)
        qq = {}
        for d in candidates:
            if not fits[d].get("ok"):
                qq[d] = {"p": [], "emp": [], "theory": []}
                continue
            # rebuild frozen from params for qq
            try:
                if d == "normal":
                    frozen = st.norm(loc=fits[d]["params"]["mu"], scale=fits[d]["params"]["sigma"])
                elif d == "t":
                    frozen = st.t(df=fits[d]["params"]["df"], loc=fits[d]["params"]["loc"], scale=fits[d]["params"]["scale"])
                else:
                    frozen = st.lognorm(s=fits[d]["params"]["s"], loc=fits[d]["params"]["loc"], scale=fits[d]["params"]["scale"])
                qq[d] = _qq_points(arr, frozen, n_points=80)
            except (ValueError, TypeError, ZeroDivisionError, FloatingPointError):  # pragma: no cover
                qq[d] = {"p": [], "emp": [], "theory": []}
        out[k]["fit"] = {"candidates": candidates, "best_by_bic": best, "dists": fits, "qq": qq}
    period_out = None
    if period_freq:
        try:
            obs_pr = nav_obs.resample(str(period_freq)).last().pct_change().dropna().to_numpy(dtype=float)
        except Exception:  # pragma: no cover
            obs_pr = np.asarray([], dtype=float)
        period_out = {
            "freq": str(period_freq),
            "observed": _stats(obs_pr),
            "simulated": _stats(np.asarray(period_samples, dtype=float)),
            "hist": _histogram(np.asarray(period_samples, dtype=float), bins=60, clip_q=(0.01, 0.99)),
        }

    return {
        "method": "circular_block_bootstrap",
        "n_sims": int(cfg.n_sims),
        "block_size": int(cfg.block_size),
        "seed_provided": cfg.seed,
        "seed_used": int(seed_used),
        "metrics": out,
        "period_return": period_out,
    }

