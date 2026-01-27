from __future__ import annotations

import math
import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..calendar.trading_calendar import shift_to_trading_day


@dataclass(frozen=True)
class LeadLagInputs:
    etf_close: pd.Series  # index: date, values: close
    idx_close: pd.Series  # index: date, values: close
    max_lag: int = 20  # days, +/- max_lag
    granger_max_lag: int = 10  # 1..granger_max_lag
    alpha: float = 0.05
    # If your index series is US close (Cboe VIX/GVZ), map its DATE to CN next trading day.
    index_align: str = "none"  # none|cn_next_trading_day
    # Trading-evaluation knobs (best-effort; diagnostic only)
    trade_cost_bps: float = 0.0
    rolling_window: int = 252  # trading days
    enable_threshold: bool = True
    threshold_quantile: float = 0.80
    walk_forward: bool = True
    train_ratio: float = 0.60
    walk_objective: str = "sharpe"  # sharpe|cagr


def align_us_close_to_cn_next_trading_day(
    s: pd.Series,
    *,
    cal: str = "XSHG",
) -> pd.Series:
    """
    Align US "close date" to the next China trading day.

    Rationale: US close happens after CN market close on the same calendar date.
    If you compare to CN ETF close, the US close of date D usually impacts CN date D+1 (or next session).
    """
    if s is None or s.empty:
        return pd.Series(dtype=float)
    x = pd.to_numeric(s, errors="coerce").dropna()
    out: dict[dt.date, float] = {}
    for d, v in x.items():
        if not isinstance(d, dt.date):
            continue
        d2 = d + dt.timedelta(days=1)
        d3 = shift_to_trading_day(d2, shift="next", cal=cal)
        try:
            out[d3] = float(v)
        except (TypeError, ValueError):  # pragma: no cover
            continue
    if not out:
        return pd.Series(dtype=float)
    return pd.Series(out).sort_index()


def _quantiles(x: np.ndarray, qs: list[float]) -> dict[str, float]:
    xx = x[np.isfinite(x)]
    if xx.size == 0:
        return {f"q{int(q*100)}": float("nan") for q in qs}
    vals = np.quantile(xx, qs)
    return {f"q{int(q*100)}": float(v) for q, v in zip(qs, vals, strict=False)}


def _annualized_from_daily(simple_ret: np.ndarray, ann: int = 252) -> dict[str, float]:
    r = simple_ret[np.isfinite(simple_ret)]
    if r.size < 2:
        return {"cagr": float("nan"), "vol": float("nan"), "sharpe": float("nan")}
    nav = np.cumprod(1.0 + r)
    years = (len(nav) - 1) / float(ann)
    cagr = float(nav[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(np.std(r, ddof=1) * math.sqrt(ann)) if r.size >= 2 else float("nan")
    sharpe = float((np.mean(r) / np.std(r, ddof=1)) * math.sqrt(ann)) if np.std(r, ddof=1) > 0 else float("nan")
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe}


def _max_drawdown(nav: np.ndarray) -> float:
    if nav.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    return float(np.min(dd))


def _trade_diagnostics(
    df2: pd.DataFrame,
    *,
    best_lag: int,
    best_corr: float,
    cost_bps: float,
    rolling_window: int,
    enable_threshold: bool,
    threshold_quantile: float,
    walk_forward: bool,
    train_ratio: float,
    walk_objective: str,
    max_lag: int,
) -> dict[str, Any]:
    """
    Best-effort "practical" diagnostics:
    - directional hit-rate (using index move to predict ETF move with lag)
    - conditional return distributions
    - simple risk-on/off strategy backtest (no leverage)
    - rolling stability metrics
    """
    lag = int(best_lag)
    idx_sig = df2["idx_ret"].shift(lag)
    etf = df2["etf_ret"]

    # mapping: if corr<0, predict opposite sign; if corr>0, predict same sign
    same = bool(np.isfinite(best_corr) and best_corr > 0)

    # direction labels (ignore zeros)
    y = np.sign(etf.to_numpy(dtype=float))
    x = np.sign(idx_sig.to_numpy(dtype=float))
    m = np.isfinite(y) & np.isfinite(x) & (y != 0) & (x != 0)
    yy = y[m]
    xx = x[m]
    pred = (xx if same else -xx)
    hit = (np.sign(pred) == np.sign(yy))
    hit_rate = float(np.mean(hit)) if hit.size else float("nan")

    # confusion matrix for predicting "up" (1) vs "down" (-1)
    y_up = (yy > 0)
    p_up = (pred > 0)
    tp = int(np.sum(y_up & p_up))
    tn = int(np.sum((~y_up) & (~p_up)))
    fp = int(np.sum((~y_up) & p_up))
    fn = int(np.sum(y_up & (~p_up)))
    total = int(yy.size)

    # conditional distributions: bucket by idx_sig sign
    etf_simple = np.expm1(etf.to_numpy(dtype=float))
    idx_simple = np.expm1(idx_sig.to_numpy(dtype=float))
    mm = np.isfinite(etf_simple) & np.isfinite(idx_simple)
    e = etf_simple[mm]
    v = idx_simple[mm]
    up_mask = v > 0
    dn_mask = v < 0
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    cond = {
        "idx_up": {"n": int(np.sum(up_mask)), "mean": float(np.mean(e[up_mask])) if np.any(up_mask) else float("nan"), **_quantiles(e[up_mask], qs)},
        "idx_down": {"n": int(np.sum(dn_mask)), "mean": float(np.mean(e[dn_mask])) if np.any(dn_mask) else float("nan"), **_quantiles(e[dn_mask], qs)},
    }

    def _strategy_for(
        df_local: pd.DataFrame,
        idx_sig_series: pd.Series,
        *,
        same_sign: bool,
        thr: float | None,
    ) -> dict[str, Any]:
        x_full = idx_sig_series.to_numpy(dtype=float)
        sgn = np.sign(x_full)
        pred_full = (sgn if same_sign else -sgn)

        active = np.isfinite(pred_full)
        thr_used = None
        if thr is not None and np.isfinite(thr):
            thr_used = float(thr)
            active = active & (np.abs(x_full) >= thr_used)

        exp = np.full(len(df_local), np.nan, dtype=float)
        # hold-last exposure; start fully invested (1.0)
        last = 1.0
        for i in range(len(df_local)):
            if not np.isfinite(pred_full[i]):
                exp[i] = np.nan
                continue
            if not bool(active[i]):
                exp[i] = last
                continue
            last = 1.0 if pred_full[i] >= 0 else 0.0
            exp[i] = last

        r_etf = np.expm1(df_local["etf_ret"].to_numpy(dtype=float))
        cost = float(cost_bps) / 10000.0
        turn = np.zeros_like(exp)
        turn[1:] = np.abs(exp[1:] - exp[:-1])
        r_strat = exp * r_etf - turn * cost
        ok2 = np.isfinite(r_strat) & np.isfinite(r_etf)
        r_strat_ok = r_strat[ok2]
        r_etf_ok = r_etf[ok2]
        nav_strat = np.cumprod(1.0 + r_strat_ok) if r_strat_ok.size else np.array([], dtype=float)
        nav_etf = np.cumprod(1.0 + r_etf_ok) if r_etf_ok.size else np.array([], dtype=float)

        strat_stats = _annualized_from_daily(r_strat_ok)
        bh_stats = _annualized_from_daily(r_etf_ok)
        strat_stats["max_drawdown"] = _max_drawdown(nav_strat)
        bh_stats["max_drawdown"] = _max_drawdown(nav_etf)

        act_rate = float(np.mean(active[np.isfinite(active)])) if np.any(np.isfinite(active)) else float("nan")
        return {
            "threshold_abs": thr_used,
            "active_rate": act_rate,
            "dates": [d.isoformat() for d in df_local.index[ok2]],
            "nav_strategy": nav_strat.astype(float).tolist(),
            "nav_buy_hold": nav_etf.astype(float).tolist(),
            "exp": exp[ok2].astype(float).tolist(),
            "ret_strategy": r_strat_ok.astype(float).tolist(),
            "ret_buy_hold": r_etf_ok.astype(float).tolist(),
            "metrics": {"strategy": strat_stats, "buy_hold": bh_stats},
        }

    strat = _strategy_for(df2, idx_sig, same_sign=same, thr=None)

    thr_abs: float | None = None
    strat_thr: dict[str, Any] | None = None
    if bool(enable_threshold):
        abs_sig = np.abs(idx_sig.to_numpy(dtype=float))
        abs_sig = abs_sig[np.isfinite(abs_sig)]
        if abs_sig.size >= 20:
            tq = float(min(max(threshold_quantile, 0.01), 0.99))
            thr_abs = float(np.quantile(abs_sig, tq))
            strat_thr = _strategy_for(df2, idx_sig, same_sign=same, thr=thr_abs)

    walk: dict[str, Any] | None = None
    if bool(walk_forward) and len(df2) >= 80:
        tr = float(min(max(train_ratio, 0.2), 0.85))
        cut = int(max(20, min(len(df2) - 20, int(len(df2) * tr))))
        train_df = df2.iloc[:cut].copy()
        test_df = df2.iloc[cut:].copy()
        obj = str(walk_objective or "sharpe").strip().lower()
        if obj not in {"sharpe", "cagr"}:
            obj = "sharpe"

        best_params: dict[str, Any] | None = None
        best_score = -float("inf")
        tq_grid = [0.6, 0.7, 0.8, 0.9] if bool(enable_threshold) else [None]

        for lag2 in range(-int(max_lag), int(max_lag) + 1):
            idx_sig_tr = train_df["idx_ret"].shift(int(lag2))
            corr_tr, n_tr = _safe_corr(train_df["etf_ret"], idx_sig_tr)
            if not np.isfinite(corr_tr) or n_tr < 50:
                continue
            same2 = bool(corr_tr > 0)
            for tq in tq_grid:
                thr2: float | None = None
                if tq is not None:
                    abs_tr = np.abs(idx_sig_tr.to_numpy(dtype=float))
                    abs_tr = abs_tr[np.isfinite(abs_tr)]
                    if abs_tr.size >= 20:
                        thr2 = float(np.quantile(abs_tr, float(tq)))
                s_tr = _strategy_for(train_df, idx_sig_tr, same_sign=same2, thr=thr2)
                ms = (s_tr.get("metrics") or {}).get("strategy") or {}
                score = float(ms.get(obj) or float("nan"))
                if not np.isfinite(score):
                    continue
                if score > best_score:
                    best_score = score
                    best_params = {"lag": int(lag2), "same_sign": same2, "thr_abs": thr2, "thr_q": tq, "corr_train": float(corr_tr)}

        if best_params is not None:
            lag_star = int(best_params["lag"])
            idx_sig_tr2 = train_df["idx_ret"].shift(lag_star)
            idx_sig_te2 = test_df["idx_ret"].shift(lag_star)
            s_tr2 = _strategy_for(train_df, idx_sig_tr2, same_sign=bool(best_params["same_sign"]), thr=best_params["thr_abs"])
            s_te2 = _strategy_for(test_df, idx_sig_te2, same_sign=bool(best_params["same_sign"]), thr=best_params["thr_abs"])
            walk = {
                "ok": True,
                "objective": obj,
                "best_score_train": float(best_score),
                "chosen": best_params,
                "train": {"start": train_df.index.min().isoformat(), "end": train_df.index.max().isoformat(), "strategy": s_tr2},
                "test": {"start": test_df.index.min().isoformat(), "end": test_df.index.max().isoformat(), "strategy": s_te2},
            }
        else:
            walk = {"ok": False, "reason": "no_valid_params"}

    # rolling stability (on aligned returns, not shifted simple returns)
    rw = int(max(20, rolling_window))
    # rolling corr of aligned log returns (etf_ret vs idx_sig_ret)
    s1 = df2["etf_ret"]
    s2 = idx_sig
    roll_corr = s1.rolling(rw).corr(s2)
    # rolling hit-rate
    def _roll_hit(a: pd.Series, b: pd.Series) -> float:
        aa = np.sign(a.to_numpy(dtype=float))
        bb = np.sign(b.to_numpy(dtype=float))
        mm2 = np.isfinite(aa) & np.isfinite(bb) & (aa != 0) & (bb != 0)
        if not np.any(mm2):
            return float("nan")
        pp = (bb if same else -bb)
        return float(np.mean(np.sign(pp[mm2]) == np.sign(aa[mm2])))

    roll_hit = df2["etf_ret"].rolling(rw).apply(lambda _: np.nan, raw=False)
    # compute via loop to avoid pandas apply overhead with two series
    roll_hit_vals: list[float] = []
    for i in range(len(df2)):
        if i + 1 < rw:
            roll_hit_vals.append(float("nan"))
            continue
        w = slice(i + 1 - rw, i + 1)
        roll_hit_vals.append(_roll_hit(df2["etf_ret"].iloc[w], idx_sig.iloc[w]))
    roll_hit = pd.Series(roll_hit_vals, index=df2.index)

    return {
        "signal": {"lag_used": lag, "same_sign": same, "cost_bps": float(cost_bps), "rolling_window": rw},
        "direction": {"n": total, "hit_rate": hit_rate, "tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "conditional": cond,
        "strategy": strat,
        "threshold": {"enabled": bool(enable_threshold), "quantile": float(threshold_quantile), "strategy": strat_thr} if bool(enable_threshold) else {"enabled": False},
        "walk_forward": walk,
        "rolling": {
            "dates": [d.isoformat() for d in df2.index],
            "corr": roll_corr.astype(float).tolist(),
            "hit_rate": roll_hit.astype(float).tolist(),
        },
    }


def _safe_corr(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 3:
        return float("nan"), n
    return float(np.corrcoef(x[m].to_numpy(dtype=float), y[m].to_numpy(dtype=float))[0, 1]), n


def _corr_pvalue(r: float, n: int) -> float:
    if not (np.isfinite(r) and n >= 3):
        return float("nan")
    if abs(r) >= 1.0:
        return 0.0
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return float(2 * stats.t.sf(abs(t), df=n - 2))


def _granger_f_test(y: np.ndarray, x: np.ndarray, p: int) -> dict[str, Any]:
    """
    Granger causality test (x -> y) with lag order p.
    Model:
      y_t = c + sum a_i y_{t-i} + sum b_i x_{t-i} + e_t
    F-test comparing restricted (no x terms) vs unrestricted.
    """
    n = len(y)
    if p <= 0 or n <= (2 * p + 3):
        return {"ok": False, "p": p, "n": n, "pvalue": None, "f": None, "df1": p, "df2": None}

    # Build design matrices
    # rows correspond to t = p..n-1
    yy = y[p:]
    cols_u = [np.ones(n - p)]
    cols_r = [np.ones(n - p)]
    for i in range(1, p + 1):
        cols_u.append(y[p - i : n - i])
        cols_r.append(y[p - i : n - i])
    for i in range(1, p + 1):
        cols_u.append(x[p - i : n - i])

    Xr = np.column_stack(cols_r)
    Xu = np.column_stack(cols_u)

    # Solve least squares
    br, *_ = np.linalg.lstsq(Xr, yy, rcond=None)
    bu, *_ = np.linalg.lstsq(Xu, yy, rcond=None)
    er = yy - Xr @ br
    eu = yy - Xu @ bu
    rss_r = float(np.sum(er * er))
    rss_u = float(np.sum(eu * eu))

    df1 = p
    df2 = (n - p) - Xu.shape[1]
    if df2 <= 0 or rss_u <= 0:
        return {"ok": False, "p": p, "n": n, "pvalue": None, "f": None, "df1": df1, "df2": df2}

    f = ((rss_r - rss_u) / df1) / (rss_u / df2)
    pval = float(stats.f.sf(f, df1, df2)) if np.isfinite(f) else float("nan")
    return {"ok": True, "p": p, "n": n, "pvalue": pval, "f": float(f), "df1": df1, "df2": int(df2)}


def compute_lead_lag(inputs: LeadLagInputs) -> dict[str, Any]:
    """
    Compute lead/lag relationship between ETF returns and index returns:
    - daily returns correlation at lags [-max_lag..max_lag]
    - best lag classification (lead/sync/lag)
    - Granger causality both directions
    """
    etf = pd.to_numeric(inputs.etf_close, errors="coerce").dropna()
    idx = pd.to_numeric(inputs.idx_close, errors="coerce").dropna()
    if str(getattr(inputs, "index_align", "none") or "none").strip().lower() == "cn_next_trading_day":
        idx = align_us_close_to_cn_next_trading_day(idx)
    if etf.empty or idx.empty:
        return {"ok": False, "reason": "empty_series"}

    df = pd.DataFrame({"etf_close": etf, "idx_close": idx}).dropna()
    if df.empty or len(df) < 5:
        return {"ok": False, "reason": "insufficient_overlap"}

    # Use log returns for stability
    df["etf_ret"] = np.log(df["etf_close"]).diff()
    df["idx_ret"] = np.log(df["idx_close"]).diff()
    df2 = df.dropna(subset=["etf_ret", "idx_ret"]).copy()
    if df2.empty or len(df2) < 10:
        return {"ok": False, "reason": "insufficient_returns"}

    max_lag = int(max(0, inputs.max_lag))
    rows: list[dict[str, Any]] = []
    for lag in range(-max_lag, max_lag + 1):
        r, n = _safe_corr(df2["etf_ret"], df2["idx_ret"].shift(int(lag)))
        pval = _corr_pvalue(r, n)
        rows.append({"lag": int(lag), "corr": r, "n": n, "pvalue": pval})

    # pick best lag by absolute correlation (break ties by smaller |lag|)
    finite = [x for x in rows if np.isfinite(x["corr"])]
    finite_sorted = sorted(finite, key=lambda x: (-abs(float(x["corr"])), abs(int(x["lag"]))))
    best = finite_sorted[0] if finite_sorted else {"lag": 0, "corr": float("nan"), "n": 0, "pvalue": float("nan")}

    lag0_corr, lag0_n = _safe_corr(df2["etf_ret"], df2["idx_ret"])
    lag0_p = _corr_pvalue(lag0_corr, lag0_n)

    # classification
    best_lag = int(best["lag"])
    best_p = float(best["pvalue"]) if best["pvalue"] is not None else float("nan")
    if np.isfinite(best_p) and best_p <= float(inputs.alpha):
        if best_lag > 0:
            relation = "leading"  # index leads ETF
        elif best_lag < 0:
            relation = "lagging"  # index lags ETF
        else:
            relation = "synchronous"
    else:
        relation = "unclear"

    # Granger tests
    y = df2["etf_ret"].to_numpy(dtype=float)
    x = df2["idx_ret"].to_numpy(dtype=float)
    granger_xy: list[dict[str, Any]] = []
    granger_yx: list[dict[str, Any]] = []
    gmax = int(max(1, inputs.granger_max_lag))
    for p in range(1, gmax + 1):
        granger_xy.append(_granger_f_test(y=y, x=x, p=p))
        granger_yx.append(_granger_f_test(y=x, x=y, p=p))

    # summary strings
    def _min_p(tests: list[dict[str, Any]]) -> float | None:
        ps = [float(t["pvalue"]) for t in tests if t.get("ok") and t.get("pvalue") is not None and np.isfinite(float(t["pvalue"]))]
        return None if not ps else float(min(ps))

    min_p_xy = _min_p(granger_xy)
    min_p_yx = _min_p(granger_yx)
    granger_summary = {
        "idx_causes_etf_min_p": min_p_xy,
        "etf_causes_idx_min_p": min_p_yx,
        "alpha": float(inputs.alpha),
    }

    trade = _trade_diagnostics(
        df2,
        best_lag=best_lag,
        best_corr=float(best["corr"]),
        cost_bps=float(getattr(inputs, "trade_cost_bps", 0.0) or 0.0),
        rolling_window=int(getattr(inputs, "rolling_window", 252) or 252),
        enable_threshold=bool(getattr(inputs, "enable_threshold", True)),
        threshold_quantile=float(getattr(inputs, "threshold_quantile", 0.80) or 0.80),
        walk_forward=bool(getattr(inputs, "walk_forward", True)),
        train_ratio=float(getattr(inputs, "train_ratio", 0.60) or 0.60),
        walk_objective=str(getattr(inputs, "walk_objective", "sharpe") or "sharpe"),
        max_lag=int(getattr(inputs, "max_lag", 20) or 20),
    )

    return {
        "ok": True,
        "meta": {
            "n_overlap": int(len(df)),
            "n_returns": int(len(df2)),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
        },
        "series": {
            "dates": [d.isoformat() for d in df2.index],
            "etf_close": df2["etf_close"].astype(float).tolist(),
            "idx_close": df2["idx_close"].astype(float).tolist(),
            "etf_ret": df2["etf_ret"].astype(float).tolist(),
            "idx_ret": df2["idx_ret"].astype(float).tolist(),
        },
        "corr": {
            "lag0": {"corr": lag0_corr, "n": lag0_n, "pvalue": lag0_p},
            "by_lag": rows,
            "best": best,
            "relation": relation,
        },
        "granger": {
            "idx_to_etf": granger_xy,
            "etf_to_idx": granger_yx,
            "summary": granger_summary,
        },
        "trade": trade,
    }

