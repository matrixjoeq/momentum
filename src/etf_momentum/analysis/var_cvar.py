from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

WEIGHT_EPS = 1e-12


def historical_var_cvar_loss(
    returns: pd.Series,
    *,
    confidence: float = 0.95,
) -> tuple[float | None, float | None]:
    """Historical VaR/CVaR as non-negative loss fractions (same formula as trend)."""
    s = (
        pd.to_numeric(returns, errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if s.empty:
        return None, None
    conf = float(confidence)
    if (not np.isfinite(conf)) or conf <= 0.0 or conf >= 1.0:
        conf = 0.95
    alpha = 1.0 - conf
    q = float(s.quantile(alpha))
    if not np.isfinite(q):
        return None, None
    var_loss = float(max(0.0, -q))
    tail = s[s <= q]
    cvar_ret = float(tail.mean()) if not tail.empty else float(q)
    if not np.isfinite(cvar_ret):
        return var_loss, None
    cvar_loss = float(max(0.0, -cvar_ret))
    return var_loss, cvar_loss


def cvar_shrink_scale(
    cvar_loss: float | None,
    budget_pct: float,
    *,
    sample_count: int,
    window: int,
    eps: float = WEIGHT_EPS,
) -> tuple[float, str]:
    win = int(window)
    if win < 20 or int(sample_count) < win:
        return 1.0, "insufficient_samples"
    if cvar_loss is None or (not np.isfinite(float(cvar_loss))):
        return 1.0, "invalid_cvar"
    loss = float(cvar_loss)
    if loss <= float(eps):
        return 1.0, "non_positive_cvar"
    budget = float(budget_pct)
    if (not np.isfinite(budget)) or budget <= 0.0:
        return 1.0, "invalid_cvar"
    return float(min(1.0, budget / max(loss, float(eps)))), "ok"


def collect_hs_portfolio_returns(
    weights_row: pd.Series,
    asset_ret: pd.DataFrame,
    *,
    end_loc: int,
    window: int,
    include_end: bool,
    eps: float = WEIGHT_EPS,
) -> pd.Series:
    """Walk backward from end_loc collecting finite portfolio returns until `window` points."""
    win = int(max(1, window))
    w_row = pd.to_numeric(weights_row, errors="coerce").astype(float).fillna(0.0)
    columns = [str(c) for c in w_row.index]
    r = (
        asset_ret.rename(columns=str)
        .reindex(columns=columns)
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )
    start = int(end_loc) if bool(include_end) else int(end_loc) - 1
    positions, values = _collect_hs_from_arrays(
        w_row.to_numpy(dtype=float),
        r.to_numpy(dtype=float),
        end_loc=start,
        window=win,
        eps=eps,
    )
    if not positions:
        return pd.Series(dtype=float)
    return pd.Series(
        values,
        index=[r.index[i] for i in positions],
        dtype=float,
    )


def _collect_hs_from_arrays(
    weights: np.ndarray,
    asset_ret: np.ndarray,
    *,
    end_loc: int,
    window: int,
    eps: float,
) -> tuple[list[int], list[float]]:
    """Collect at most ``window`` valid observations from prepared float arrays."""
    held = np.flatnonzero(np.isfinite(weights) & (weights > float(eps)))
    start = min(int(end_loc), int(asset_ret.shape[0]) - 1)
    if start < 0:
        return [], []
    if held.size == 0:
        positions = list(range(max(0, start - int(window) + 1), start + 1))
        return positions, [0.0] * len(positions)

    positions_desc: list[int] = []
    values_desc: list[float] = []
    i = start
    while i >= 0 and len(values_desc) < int(window):
        row = asset_ret[i, held]
        if np.isfinite(row).all():
            positions_desc.append(i)
            values_desc.append(float(np.dot(weights[held], row)))
        i -= 1
    positions_desc.reverse()
    values_desc.reverse()
    return positions_desc, values_desc


def _estimate_cvar_prepared(
    weights: np.ndarray,
    asset_ret: np.ndarray,
    *,
    end_loc: int,
    window: int,
    include_end: bool,
    budget_pct: float,
    confidence: float,
    eps: float,
) -> dict[str, Any]:
    start = int(end_loc) if include_end else int(end_loc) - 1
    _, values = _collect_hs_from_arrays(
        weights,
        asset_ret,
        end_loc=start,
        window=window,
        eps=eps,
    )
    sample = pd.Series(values, dtype=float)
    var_loss, cvar_loss = historical_var_cvar_loss(sample, confidence=confidence)
    scale, status = cvar_shrink_scale(
        cvar_loss,
        budget_pct,
        sample_count=len(values),
        window=window,
        eps=eps,
    )
    return {
        "sample_count": int(len(values)),
        "var_95": var_loss,
        "cvar_95": cvar_loss,
        "scale": float(scale),
        "status": str(status),
    }


def estimate_cvar_at(
    weights_row: pd.Series,
    asset_ret: pd.DataFrame,
    *,
    end_loc: int,
    window: int,
    include_end: bool,
    budget_pct: float,
    confidence: float = 0.95,
    eps: float = WEIGHT_EPS,
) -> dict[str, Any]:
    sample = collect_hs_portfolio_returns(
        weights_row,
        asset_ret,
        end_loc=end_loc,
        window=window,
        include_end=include_end,
        eps=eps,
    )
    n = int(len(sample))
    var_loss, cvar_loss = historical_var_cvar_loss(sample, confidence=confidence)
    scale, status = cvar_shrink_scale(
        cvar_loss, budget_pct, sample_count=n, window=window, eps=eps
    )
    return {
        "sample_count": n,
        "var_95": var_loss,
        "cvar_95": cvar_loss,
        "scale": float(scale),
        "status": str(status),
    }


def build_holding_cvar_overlay(
    weights: pd.DataFrame,
    hs_asset_ret: pd.DataFrame,
    orig_port_ret: pd.Series,
    orig_nav: pd.Series,
    *,
    window: int,
    budget_pct: float,
    confidence: float = 0.95,
    eps: float = WEIGHT_EPS,
) -> dict[str, Any]:
    """
    Shrink-only CVaR overlay for a holding-strategy weight path.

    HS uses un-forwarded, unfilled asset returns. Overlay P&L uses orig_port_ret.
    """
    idx = orig_port_ret.index
    w = (
        weights.reindex(index=idx)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    r_hs = (
        hs_asset_ret.rename(columns=str)
        .reindex(index=idx, columns=[str(c) for c in w.columns])
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )
    r_hs_values = r_hs.to_numpy(dtype=float)
    r_orig = (
        pd.to_numeric(orig_port_ret.reindex(idx), errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    win = int(max(1, window))
    budget = float(budget_pct)
    scales = pd.Series(1.0, index=idx, dtype=float)
    statuses: list[str] = []
    sim_sample_counts: list[int] = []
    trigger_dates: list[str] = []
    scale_applied_count = 0
    trigger_episode_count = 0
    in_episode = False
    buckets = {"mild_0.8_1.0": 0, "medium_0.5_0.8": 0, "strong_0.0_0.5": 0}

    n = len(idx)
    for loc in range(n):
        est = _estimate_cvar_prepared(
            w.iloc[loc].to_numpy(dtype=float),
            r_hs_values,
            end_loc=loc,
            window=win,
            include_end=False,
            budget_pct=budget,
            confidence=confidence,
            eps=eps,
        )
        scale = float(est["scale"])
        status = str(est["status"])
        scales.iloc[loc] = scale
        statuses.append(status)
        sim_sample_counts.append(int(est["sample_count"]))
        if scale < 1.0 - float(eps):
            scale_applied_count += 1
            d = idx[loc]
            try:
                d_str = pd.Timestamp(d).date().isoformat()
            except Exception:
                d_str = str(d)
            trigger_dates.append(d_str)
            if scale > 0.8:
                buckets["mild_0.8_1.0"] += 1
            elif scale > 0.5:
                buckets["medium_0.5_0.8"] += 1
            else:
                buckets["strong_0.0_0.5"] += 1
            if not in_episode:
                trigger_episode_count += 1
                in_episode = True
        else:
            in_episode = False

    r_sim = (scales.astype(float) * r_orig).astype(float)
    nav_src = pd.to_numeric(orig_nav.reindex(idx), errors="coerce").astype(float)
    nav_sim = nav_src.copy()
    if len(nav_sim) > 0:
        if (not np.isfinite(float(nav_sim.iloc[0]))) or float(nav_sim.iloc[0]) <= 0:
            nav_sim.iloc[0] = 1.0
        for i in range(1, len(nav_sim)):
            prev = float(nav_sim.iloc[i - 1])
            rr = float(r_sim.iloc[i])
            if not np.isfinite(prev):
                prev = 1.0
            if not np.isfinite(rr):
                rr = 0.0
            nav_sim.iloc[i] = prev * (1.0 + rr)

    last_loc = n - 1
    last_w = w.iloc[last_loc] if last_loc >= 0 else pd.Series(dtype=float)
    prompt_est = (
        _estimate_cvar_prepared(
            last_w.to_numpy(dtype=float),
            r_hs_values,
            end_loc=last_loc,
            window=win,
            include_end=True,
            budget_pct=budget,
            confidence=confidence,
            eps=eps,
        )
        if last_loc >= 0
        else {
            "sample_count": 0,
            "var_95": None,
            "cvar_95": None,
            "scale": 1.0,
            "status": "insufficient_samples",
        }
    )
    prompt_scale = float(prompt_est["scale"])
    risk_sum = float(last_w.clip(lower=0.0).sum()) if last_loc >= 0 else 0.0
    if not np.isfinite(risk_sum):
        risk_sum = 0.0
    suggested_risk = float(prompt_scale) * float(risk_sum)
    suggested_cash = float(max(0.0, 1.0 - suggested_risk))
    suggested_weights: list[dict[str, Any]] = []
    if last_loc >= 0:
        for c in last_w.index:
            wv = float(last_w.loc[c] or 0.0)
            if np.isfinite(wv) and wv > float(eps):
                suggested_weights.append(
                    {"code": str(c), "weight": float(wv * prompt_scale)}
                )
        suggested_weights.sort(key=lambda x: float(x["weight"]), reverse=True)

    asof = ""
    if last_loc >= 0:
        try:
            asof = pd.Timestamp(idx[last_loc]).date().isoformat()
        except Exception:
            asof = str(idx[last_loc])

    w_scaled = w.mul(scales, axis=0).clip(lower=0.0)
    cash = (1.0 - w_scaled.sum(axis=1)).clip(lower=0.0)
    w_with_cash = w_scaled.copy()
    w_with_cash["__cash__"] = cash
    turn = (w_with_cash - w_with_cash.shift(1).fillna(0.0)).abs().sum(axis=1) / 2.0
    avg_daily_turnover = float(turn.mean()) if len(turn) else 0.0

    dates = []
    for d in idx:
        try:
            dates.append(pd.Timestamp(d).date().isoformat())
        except Exception:
            dates.append(str(d))

    return {
        "window": int(win),
        "budget_pct": float(budget),
        "confidence": float(confidence),
        "prompt": {
            "asof": asof,
            "applies_to": "下一交易日",
            "status": str(prompt_est["status"]),
            "sample_count": int(prompt_est["sample_count"]),
            "var_95": prompt_est["var_95"],
            "cvar_95": prompt_est["cvar_95"],
            "scale": float(prompt_scale),
            "suggested_risk_weight": float(suggested_risk),
            "suggested_cash": float(suggested_cash),
            "suggested_weights": suggested_weights,
        },
        "sim": {
            "dates": dates,
            "nav": [float(x) for x in nav_sim.astype(float).tolist()],
            "scale": [float(x) for x in scales.astype(float).tolist()],
            "status": statuses,
            "sample_count": sim_sample_counts,
            "avg_daily_turnover": float(avg_daily_turnover),
            "scale_applied_count": int(scale_applied_count),
            "cvar_scale_trigger_episode_count": int(trigger_episode_count),
            "cvar_scale_trigger_count_by_bucket": {
                k: int(v) for k, v in buckets.items()
            },
            "cvar_scale_trigger_dates": list(dict.fromkeys(trigger_dates))[:200],
            "mean_scale": float(scales.mean()) if len(scales) else 1.0,
            "min_scale": float(scales.min()) if len(scales) else 1.0,
            "max_scale": float(scales.max()) if len(scales) else 1.0,
        },
        "_nav_sim": nav_sim,
        "_r_sim": r_sim,
        "_scales": scales,
    }


def public_cvar_overlay(
    raw: dict[str, Any],
    *,
    cumulative_return: float,
    annualized_return: float,
    annualized_volatility: float,
    max_drawdown: float,
) -> dict[str, Any]:
    sim = dict(raw.get("sim") or {})
    sim["metrics"] = {
        "cumulative_return": float(cumulative_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "max_drawdown": float(max_drawdown),
        "avg_daily_turnover": float(sim.get("avg_daily_turnover") or 0.0),
    }
    return {
        "window": int(raw.get("window") or 0),
        "budget_pct": float(raw.get("budget_pct") or 0.0),
        "confidence": float(raw.get("confidence") or 0.95),
        "prompt": dict(raw.get("prompt") or {}),
        "sim": sim,
    }
