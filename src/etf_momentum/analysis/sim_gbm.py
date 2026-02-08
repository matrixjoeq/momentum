from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .baseline import (
    TRADING_DAYS_PER_YEAR,
    _annualized_return,
    _annualized_vol,
    _max_drawdown,
    _sharpe,
)


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
    seed: int | None = None


def simulate_gbm_prices(
    *,
    start: str = "19900101",
    end: str | None = None,
    cfg: SimConfig = SimConfig(),
) -> dict[str, Any]:
    """
    Simulate independent GBM price series (starting at 1.0) where:
    - daily log returns are Normal(0, sigma_d), sigma_d = ann_vol / sqrt(252)
    - assets are independent (uncorrelated by construction)
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

    rng = np.random.default_rng(None if cfg.seed is None else int(cfg.seed))
    ann_vols = rng.uniform(low=v0, high=v1, size=n).astype(float)
    sig_d = ann_vols / np.sqrt(float(TRADING_DAYS_PER_YEAR))
    # log returns (T-1 values; first day has no return)
    lr = rng.normal(loc=0.0, scale=sig_d.reshape(1, n), size=(len(idx), n)).astype(float)
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
            "seed": (None if cfg.seed is None else int(cfg.seed)),
            "calendar": "pandas_B",
            "return_model": "log1p(r) ~ Normal(0, sigma_d) with sigma_d=ann_vol/sqrt(252)",
        },
        "assets": {
            "codes": codes,
            "ann_vols": {c: float(v) for c, v in zip(codes, ann_vols, strict=False)},
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

    # compute portfolio return (cash for days without a pick: should not happen after first segment)
    port_ret = np.zeros(len(dates), dtype=float)
    for t in range(1, len(dates)):
        p = int(pick_daily[t])
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

    port_ret = (w * ret.to_numpy(dtype=float)).sum(axis=1).astype(float)
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
    port_ret = (ret.to_numpy(dtype=float) * w.reshape(1, -1)).sum(axis=1).astype(float)
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
    seed: int | None = None,
    lookback_days: int = 20,
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

    done = 0
    while done < n_sims:
        m = min(chunk_size, n_sims - done)
        # Randomize vols per sim per asset (respect selected assets)
        ann_vols_all = rng.uniform(low=float(vol_low), high=float(vol_high), size=(m, n_assets)).astype(float)
        ann_vols = ann_vols_all[:, asset_indices]
        sig_d = ann_vols / np.sqrt(float(TRADING_DAYS_PER_YEAR))
        lr = rng.normal(loc=0.0, scale=sig_d[:, None, :], size=(m, n_days, n_assets_eff)).astype(float)
        lr[:, 0, :] = 0.0
        logp = np.cumsum(lr, axis=1)
        # daily simple returns
        ret = np.exp(lr).astype(float) - 1.0

        # momentum scores at decision dates
        # score = exp(logp[t]-logp[t-lb]) - 1
        # gather logp at decision indices and (decision-lb)
        dec = np.array(decision_idx, dtype=int)
        mom = np.exp(logp[:, dec, :] - logp[:, dec - lb, :]) - 1.0  # (m, n_dec, n_assets_eff)
        pick = np.argmax(np.where(np.isfinite(mom), mom, -1e18), axis=2).astype(int)  # (m, n_dec)

        # Build daily pick id for rotation
        pick_daily = np.full((m, n_days), -1, dtype=int)
        # warm-up: use 0 (unused for metrics; but keeps array valid)
        pick_daily[:, : decision_idx[0] + 1] = 0
        for j, (start_i, end_i) in enumerate(segs):
            if j >= pick.shape[1]:
                break
            pick_daily[:, start_i : end_i + 1] = pick[:, j].reshape(m, 1)

        # Rotation portfolio daily returns
        rot_ret = np.zeros((m, n_days), dtype=float)
        for t in range(1, n_days):
            p = pick_daily[:, t]
            rot_ret[:, t] = ret[np.arange(m), t, p]
        rot_nav = np.cumprod(1.0 + rot_ret, axis=1).astype(float)
        rot_nav[:, 0] = 1.0

        # EW weekly-rebalanced (equal weights each segment)
        ew_ret = ret.mean(axis=2).astype(float)  # daily rebalanced equals mean; close enough for synthetic
        ew_nav = np.cumprod(1.0 + ew_ret, axis=1).astype(float)
        ew_nav[:, 0] = 1.0

        # RP baseline: constant inverse-vol weights per simulation
        inv = 1.0 / np.where((np.isfinite(ann_vols) & (ann_vols > 0)), ann_vols, np.nan)  # (m, n_assets_eff)
        inv_sum = np.nansum(inv, axis=1, keepdims=True)  # (m,1)
        w_rp = np.where(np.isfinite(inv_sum) & (inv_sum > 0), inv / inv_sum, (1.0 / float(n_assets_eff)))  # (m,n_assets_eff)
        # portfolio return
        rp_ret = np.sum(ret * w_rp[:, None, :], axis=2).astype(float)  # (m, n_days)
        rp_nav = np.cumprod(1.0 + rp_ret, axis=1).astype(float)
        rp_nav[:, 0] = 1.0

        # Metrics (approx): CAGR and max drawdown
        years = (n_days - 1) / float(TRADING_DAYS_PER_YEAR)
        cagr_rot = np.power(rot_nav[:, -1], 1.0 / years) - 1.0
        cagr_ew = np.power(ew_nav[:, -1], 1.0 / years) - 1.0
        cagr_rp = np.power(rp_nav[:, -1], 1.0 / years) - 1.0
        # max drawdown
        peak_rot = np.maximum.accumulate(rot_nav, axis=1)
        dd_rot = rot_nav / peak_rot - 1.0
        mdd_rot = np.min(dd_rot, axis=1)
        peak_ew = np.maximum.accumulate(ew_nav, axis=1)
        dd_ew = ew_nav / peak_ew - 1.0
        mdd_ew = np.min(dd_ew, axis=1)
        peak_rp = np.maximum.accumulate(rp_nav, axis=1)
        dd_rp = rp_nav / peak_rp - 1.0
        mdd_rp = np.min(dd_rp, axis=1)

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
            "seed": (None if seed is None else int(seed)),
            "lookback_days": int(lookback_days),
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

