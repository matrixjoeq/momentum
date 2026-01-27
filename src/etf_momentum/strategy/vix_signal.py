from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..calendar.trading_calendar import shift_to_trading_day


Action = Literal["BUY", "SELL", "HOLD"]
Position = Literal["long", "cash", "unknown"]


@dataclass(frozen=True)
class VixSignalInputs:
    """
    Produce a *next CN trading day* trade instruction for an A-share ETF using VIX/GVZ.

    IMPORTANT (no look-ahead):
    - Use Cboe DATE=D (US session close) and map it to CN next trading day.
    - The mapped CN date is the earliest CN session where this US-close info is available pre-open.
    """

    index_close_us: pd.Series  # index: US date (dt.date), values: close
    index: str = "VIX"  # VIX/GVZ
    index_align: str = "cn_next_trading_day"  # none|cn_next_trading_day
    current_position: Position = "unknown"  # long|cash|unknown
    # thresholding: trade only when |idx_ret| >= threshold_abs computed from past window quantile
    lookback_window: int = 252
    threshold_quantile: float = 0.80
    min_abs_ret: float = 0.0  # hard minimum abs log-ret threshold
    # optional explicit target CN trade date (YYYY-MM-DD as dt.date)
    target_cn_trade_date: dt.date | None = None
    calendar: str = "XSHG"


def _map_us_close_to_cn_trade_dates(
    s: pd.Series,
    *,
    align: str,
    cal: str,
) -> tuple[pd.Series, dict[dt.date, dt.date]]:
    """
    Map US close series to CN trade dates.
    Returns (cn_close_series, cn_date -> us_date mapping).
    """
    x = pd.to_numeric(s, errors="coerce").dropna()
    mapping: dict[dt.date, dt.date] = {}

    if x.empty:
        return pd.Series(dtype=float), {}

    if str(align or "none").strip().lower() == "cn_next_trading_day":
        out: dict[dt.date, float] = {}
        for us_d, v in x.items():
            if not isinstance(us_d, dt.date):
                continue
            cn_d = shift_to_trading_day(us_d + dt.timedelta(days=1), shift="next", cal=cal)
            out[cn_d] = float(v)
            mapping[cn_d] = us_d
        return pd.Series(out).sort_index(), mapping

    # no alignment: keep original dates
    out2: dict[dt.date, float] = {}
    for d, v in x.items():
        if isinstance(d, dt.date):
            out2[d] = float(v)
            mapping[d] = d
    return pd.Series(out2).sort_index(), mapping


def generate_next_action(inputs: VixSignalInputs) -> dict[str, Any]:
    """
    Return a dict with:
    - action_date (CN trade date)
    - action: BUY/SELL/HOLD
    - target_position: long/cash
    - reason + signal values
    """
    cn_close, cn_to_us = _map_us_close_to_cn_trade_dates(inputs.index_close_us, align=inputs.index_align, cal=inputs.calendar)
    if cn_close.empty or len(cn_close) < 3:
        return {"ok": False, "error": "insufficient_index_history"}

    idx_ret = np.log(cn_close).diff()
    idx_ret = idx_ret.replace([np.inf, -np.inf], np.nan).dropna()
    if idx_ret.empty:
        return {"ok": False, "error": "empty_index_ret"}

    # Determine action date:
    # - If user passes target date: use it (must be <= latest available mapped date)
    # - else: use latest available mapped date (corresponding to latest US close known)
    latest_avail = idx_ret.index.max()
    if isinstance(latest_avail, pd.Timestamp):  # pragma: no cover
        latest_avail = latest_avail.date()

    if inputs.target_cn_trade_date is not None:
        action_date = inputs.target_cn_trade_date
    else:
        action_date = latest_avail

    if action_date not in idx_ret.index:
        return {
            "ok": False,
            "error": "no_signal_for_date",
            "action_date": action_date.isoformat(),
            "signal": {"latest_available_date": latest_avail.isoformat()},
        }

    # threshold from past window ending at action_date - 1 (avoid using same-day to set its own threshold)
    hist = idx_ret.loc[idx_ret.index < action_date]
    hist = hist.tail(int(max(20, inputs.lookback_window)))
    abs_hist = np.abs(hist.to_numpy(dtype=float))
    abs_hist = abs_hist[np.isfinite(abs_hist)]
    thr_abs = float("nan")
    if abs_hist.size >= 20:
        q = float(min(max(inputs.threshold_quantile, 0.01), 0.99))
        thr_abs = float(np.quantile(abs_hist, q))
    # hard floor
    thr_eff = float(inputs.min_abs_ret) if np.isfinite(float(inputs.min_abs_ret)) else 0.0
    if np.isfinite(thr_abs):
        thr_eff = max(thr_eff, thr_abs)

    sig = float(idx_ret.loc[action_date])
    sig_abs = abs(sig)
    us_date = cn_to_us.get(action_date)
    v_close = float(cn_close.loc[action_date]) if action_date in cn_close.index else float("nan")

    # Risk rule (configurable by sign): for Nasdaq ETF we usually interpret VIX up as risk-off.
    # We'll express as target position; action depends on current_position.
    if not np.isfinite(sig):
        target: Position = "unknown"
        action: Action = "HOLD"
        reason = "signal_nan"
    elif sig_abs < thr_eff:
        target = inputs.current_position if inputs.current_position in {"long", "cash"} else "unknown"
        action = "HOLD"
        reason = "below_threshold"
    else:
        # VIX up => cash; VIX down => long
        target = "cash" if sig > 0 else "long"
        if inputs.current_position == "unknown":
            action = "SELL" if target == "cash" else "BUY"
        elif inputs.current_position == target:
            action = "HOLD"
        else:
            action = "SELL" if target == "cash" else "BUY"
        reason = "threshold_triggered"

    return {
        "ok": True,
        "index": str(inputs.index),
        "index_align": str(inputs.index_align),
        "calendar": str(inputs.calendar),
        "action_date": action_date.isoformat(),
        "action": action,
        "target_position": target,
        "current_position": inputs.current_position,
        "reason": reason,
        "signal": {
            "us_date": (us_date.isoformat() if isinstance(us_date, dt.date) else None),
            "cn_date": action_date.isoformat(),
            "index_close": v_close,
            "index_log_ret": sig,
            "abs_ret": sig_abs,
            "threshold_abs": (float(thr_abs) if np.isfinite(thr_abs) else None),
            "threshold_eff": thr_eff,
            "threshold_quantile": float(inputs.threshold_quantile),
            "lookback_window": int(inputs.lookback_window),
            "lookback_n": int(len(abs_hist)),
        },
    }


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


def backtest_vix_next_day_signal(
    *,
    etf_close_cn: pd.Series,  # index: CN dt.date
    index_close_us: pd.Series,  # index: US dt.date
    start: dt.date,
    end: dt.date,
    index: str = "VIX",
    index_align: str = "cn_next_trading_day",
    calendar: str = "XSHG",
    lookback_window: int = 252,
    threshold_quantile: float = 0.80,
    min_abs_ret: float = 0.0,
    trade_cost_bps: float = 0.0,
    initial_nav: float = 1.0,
    initial_position: Position = "long",
) -> dict[str, Any]:
    """
    Historical backtest for the "VIX next-day" live signal.

    Mechanics (approximation):
    - decision for CN date T uses US-close mapped to T (cn_next_trading_day), known pre-open of T
    - exposure (0/1) is applied to *close-to-close* ETF return on T (as a proxy of day-T PnL)
    - switching cost applies when position changes on T (turnover = |pos_T - pos_{T-1}|)
    """
    etf = pd.to_numeric(etf_close_cn, errors="coerce").dropna()
    if etf.empty:
        return {"ok": False, "error": "empty_etf_close"}
    etf = etf[(etf.index >= start) & (etf.index <= end)].copy()
    if etf.empty or len(etf) < 5:
        return {"ok": False, "error": "insufficient_etf_range"}

    cn_idx_close, cn_to_us = _map_us_close_to_cn_trade_dates(index_close_us, align=index_align, cal=calendar)
    if cn_idx_close.empty or len(cn_idx_close) < 5:
        return {"ok": False, "error": "insufficient_index_history"}

    df = pd.DataFrame({"etf_close": etf}).copy()
    # attach mapped index close on same CN dates (may have NaNs)
    df["idx_close"] = cn_idx_close.reindex(df.index)
    df["idx_us_date"] = [cn_to_us.get(d) for d in df.index]

    # compute returns
    df["etf_ret"] = df["etf_close"].pct_change()
    df["idx_log_ret"] = np.log(df["idx_close"]).diff()
    df = df.replace([np.inf, -np.inf], np.nan)

    # rolling threshold (NO look-ahead): use history up to T-1 to form threshold for T
    rw = int(max(20, lookback_window))
    q = float(min(max(float(threshold_quantile), 0.01), 0.99))
    abs_sig = df["idx_log_ret"].abs()
    thr_abs = abs_sig.rolling(rw, min_periods=20).quantile(q).shift(1)
    thr_eff = np.maximum(thr_abs.to_numpy(dtype=float), float(max(0.0, min_abs_ret)))
    df["thr_abs"] = thr_abs
    df["thr_eff"] = thr_eff

    # signal active?
    df["sig_active"] = (abs_sig.to_numpy(dtype=float) >= df["thr_eff"].to_numpy(dtype=float)) & abs_sig.notna()

    # target position for each date
    # - active: VIX up => cash, VIX down => long
    # - inactive: HOLD previous
    pos = []
    prev = "long" if initial_position not in {"cash"} else "cash"
    for d, r in df.iterrows():
        active = bool(r.get("sig_active")) if pd.notna(r.get("sig_active")) else False
        sig = r.get("idx_log_ret")
        if active and pd.notna(sig):
            prev = "cash" if float(sig) > 0 else "long"
        pos.append(prev)
    df["position"] = pos
    df["pos_num"] = (df["position"] == "long").astype(float)

    # action based on change vs yesterday
    df["pos_prev"] = df["pos_num"].shift(1)
    df["turnover"] = (df["pos_num"] - df["pos_prev"]).abs().fillna(0.0)
    def _action(row) -> str:
        if float(row["turnover"]) <= 0:
            return "HOLD"
        return "BUY" if float(row["pos_num"]) > float(row["pos_prev"] or 0.0) else "SELL"
    df["action"] = df.apply(_action, axis=1)

    # NAV
    cost_rate = float(max(0.0, trade_cost_bps)) / 10000.0
    nav_s = [float(initial_nav)]
    nav_b = [float(initial_nav)]
    for i in range(1, len(df)):
        r = float(df["etf_ret"].iloc[i]) if np.isfinite(float(df["etf_ret"].iloc[i] or 0.0)) else 0.0
        posi = float(df["pos_num"].iloc[i]) if np.isfinite(float(df["pos_num"].iloc[i] or 0.0)) else 0.0
        to = float(df["turnover"].iloc[i]) if np.isfinite(float(df["turnover"].iloc[i] or 0.0)) else 0.0

        ns_prev = nav_s[-1]
        ns = ns_prev * (1.0 - cost_rate * to)
        ns = ns * (1.0 + posi * r)
        nav_s.append(float(ns))

        nb_prev = nav_b[-1]
        nb = nb_prev * (1.0 + r)
        nav_b.append(float(nb))

    df["nav_strategy"] = nav_s
    df["nav_buy_hold"] = nav_b

    # metrics
    strat_ret = pd.Series(df["nav_strategy"]).pct_change().to_numpy(dtype=float)
    bh_ret = pd.Series(df["nav_buy_hold"]).pct_change().to_numpy(dtype=float)
    ms = _annualized_from_daily(strat_ret)
    mb = _annualized_from_daily(bh_ret)
    ms["max_drawdown"] = _max_drawdown(np.asarray(nav_s, dtype=float))
    mb["max_drawdown"] = _max_drawdown(np.asarray(nav_b, dtype=float))

    # trade log (ALL dates, newest first)
    trades = []
    for d, r in df.iloc[::-1].iterrows():
        us_d = r.get("idx_us_date")
        trades.append(
            {
                "date": d.isoformat() if isinstance(d, dt.date) else str(d),
                "action": str(r.get("action") or "HOLD"),
                "position": str(r.get("position") or ""),
                "idx_us_date": us_d.isoformat() if isinstance(us_d, dt.date) else None,
                "idx_close": (float(r["idx_close"]) if pd.notna(r.get("idx_close")) else None),
                "idx_log_ret": (float(r["idx_log_ret"]) if pd.notna(r.get("idx_log_ret")) else None),
                "thr_eff": (float(r["thr_eff"]) if pd.notna(r.get("thr_eff")) else None),
                "sig_active": bool(r.get("sig_active")) if pd.notna(r.get("sig_active")) else False,
                "etf_close": float(r["etf_close"]) if pd.notna(r.get("etf_close")) else None,
                "etf_ret": (float(r["etf_ret"]) if pd.notna(r.get("etf_ret")) else None),
                "turnover": float(r["turnover"]) if pd.notna(r.get("turnover")) else 0.0,
                "nav_strategy": float(r["nav_strategy"]) if pd.notna(r.get("nav_strategy")) else None,
                "nav_buy_hold": float(r["nav_buy_hold"]) if pd.notna(r.get("nav_buy_hold")) else None,
            }
        )

    return {
        "ok": True,
        "meta": {
            "index": str(index),
            "index_align": str(index_align),
            "calendar": str(calendar),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "n": int(len(df)),
            "lookback_window": int(rw),
            "threshold_quantile": float(q),
            "min_abs_ret": float(max(0.0, min_abs_ret)),
            "trade_cost_bps": float(max(0.0, trade_cost_bps)),
            "initial_position": str(initial_position),
        },
        "series": {
            "dates": [d.isoformat() for d in df.index],
            "nav_strategy": df["nav_strategy"].astype(float).tolist(),
            "nav_buy_hold": df["nav_buy_hold"].astype(float).tolist(),
            "position": df["pos_num"].astype(float).tolist(),
        },
        "metrics": {"strategy": ms, "buy_hold": mb},
        "trades": trades,
    }

