from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..analysis.baseline import (
    TRADING_DAYS_PER_YEAR,
    _annualized_return,
    _annualized_vol,
    _information_ratio,
    _max_drawdown,
    _max_drawdown_duration_days,
    _sharpe,
    _sortino,
    _ulcer_index,
)
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


def _period_returns(series: pd.Series, freq: str) -> pd.DataFrame:
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    r = s.resample(freq).last().pct_change().dropna()
    return pd.DataFrame({"period_end": r.index.date.astype(str), "return": r.values})


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


def _dist_stats(x: pd.Series | np.ndarray) -> dict[str, float | int | None]:
    v = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if v.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p01": None,
            "p05": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "pos_ratio": None,
        }
    arr = v.to_numpy(dtype=float)
    out: dict[str, float | int | None] = {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "pos_ratio": float(np.mean(arr > 0.0)),
    }
    for q, k in [
        (0.01, "p01"),
        (0.05, "p05"),
        (0.10, "p10"),
        (0.25, "p25"),
        (0.50, "p50"),
        (0.75, "p75"),
        (0.90, "p90"),
        (0.95, "p95"),
        (0.99, "p99"),
    ]:
        out[k] = float(np.quantile(arr, float(q))) if len(arr) else None
    return out


def _metrics_from_nav(*, nav: pd.Series, rf_annual: float) -> dict[str, float | int | None]:
    nav = pd.to_numeric(nav, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if nav.empty or len(nav) < 3:
        return {
            "cumulative_return": None,
            "annualized_return": None,
            "cagr": None,
            "annualized_volatility": None,
            "vol": None,
            "sharpe": None,
            "sortino": None,
            "max_drawdown": None,
            "max_drawdown_recovery_days": None,
            "calmar": None,
            "ulcer_index": None,
            "ulcer_performance_index": None,
        }
    daily_ret = nav.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    cum = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    ann_ret = float(_annualized_return(nav, ann_factor=TRADING_DAYS_PER_YEAR))
    ann_vol = float(_annualized_vol(daily_ret, ann_factor=TRADING_DAYS_PER_YEAR))
    sharpe = float(_sharpe(daily_ret, rf=float(rf_annual), ann_factor=TRADING_DAYS_PER_YEAR))
    sortino = float(_sortino(daily_ret, rf=float(rf_annual), ann_factor=TRADING_DAYS_PER_YEAR))
    mdd = float(_max_drawdown(nav))
    mdd_dur = int(_max_drawdown_duration_days(nav))
    calmar = float(ann_ret / abs(mdd)) if np.isfinite(mdd) and float(mdd) < 0 else float("nan")
    ui = float(_ulcer_index(nav, in_percent=True))
    ui_den = ui / 100.0
    upi = float((ann_ret - float(rf_annual)) / ui_den) if ui_den > 0 else float("nan")
    return {
        "cumulative_return": cum,
        "annualized_return": ann_ret,
        "cagr": ann_ret,  # backward-compat
        "annualized_volatility": ann_vol,
        "vol": ann_vol,  # backward-compat
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "max_drawdown_recovery_days": mdd_dur,
        "calmar": calmar,
        "ulcer_index": ui,
        "ulcer_performance_index": upi,
    }


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


def backtest_vix_next_day_signal(
    *,
    etf_close_cn: pd.Series,  # index: CN dt.date
    etf_open_cn: pd.Series | None = None,  # index: CN dt.date
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
    exec_model: str = "open_open",  # open_open|close_close (legacy)
) -> dict[str, Any]:
    """
    Historical backtest for the "VIX next-day" live signal.

    Mechanics (no look-ahead, tradable by design):
    - decision for CN date T uses US-close mapped to T (cn_next_trading_day), known pre-open of T
    - execution model:
      - open_open: trade at open(T), PnL uses open(T)->open(T+1) return
      - close_close: (legacy) applies position to close-to-close return (can be optimistic around gaps)
    - switching cost applies when position changes on T (turnover = |pos_T - pos_{T-1}|)
    """
    etf_close = pd.to_numeric(etf_close_cn, errors="coerce").dropna()
    if etf_close.empty:
        return {"ok": False, "error": "empty_etf_close"}
    etf_close = etf_close[(etf_close.index >= start) & (etf_close.index <= end)].copy()
    if etf_close.empty or len(etf_close) < 5:
        return {"ok": False, "error": "insufficient_etf_range"}

    em = str(exec_model or "open_open").strip().lower()
    if em not in {"open_open", "close_close"}:
        return {"ok": False, "error": "invalid_exec_model"}

    etf_open = None
    if em == "open_open":
        if etf_open_cn is None:
            return {"ok": False, "error": "missing_etf_open"}
        etf_open = pd.to_numeric(etf_open_cn, errors="coerce").dropna()
        etf_open = etf_open[(etf_open.index >= start) & (etf_open.index <= end)].copy()
        if etf_open.empty or len(etf_open) < 6:
            return {"ok": False, "error": "insufficient_etf_open_range"}

    cn_idx_close, cn_to_us = _map_us_close_to_cn_trade_dates(index_close_us, align=index_align, cal=calendar)
    if cn_idx_close.empty or len(cn_idx_close) < 5:
        return {"ok": False, "error": "insufficient_index_history"}

    if etf_open is not None:
        df = pd.DataFrame({"etf_open": etf_open}).copy()
        df["etf_close"] = etf_close.reindex(df.index)
    else:
        df = pd.DataFrame({"etf_close": etf_close}).copy()
        df["etf_open"] = np.nan
    # attach mapped index close on same CN dates (may have NaNs)
    df["idx_close"] = cn_idx_close.reindex(df.index)
    df["idx_us_date"] = [cn_to_us.get(d) for d in df.index]

    # compute returns
    if em == "open_open":
        df["etf_ret_exec"] = df["etf_open"].shift(-1) / df["etf_open"] - 1.0
    else:
        df["etf_ret_exec"] = df["etf_close"].pct_change()
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
    # Backtest ends at `end` (inclusive); we don't realize PnL beyond the last available exec return.
    # Avoid a phantom last-day trade affecting logs/metrics when there is no T->T+1 return.
    if len(df) >= 2:
        df.loc[df.index[-1], "turnover"] = 0.0
        df.loc[df.index[-1], "action"] = "HOLD"

    # NAV
    cost_rate = float(max(0.0, trade_cost_bps)) / 10000.0
    nav_s = [float(initial_nav)]
    nav_b = [float(initial_nav)]
    if em == "open_open":
        for i in range(0, len(df) - 1):
            r = float(df["etf_ret_exec"].iloc[i]) if np.isfinite(float(df["etf_ret_exec"].iloc[i] or 0.0)) else 0.0
            posi = float(df["pos_num"].iloc[i]) if np.isfinite(float(df["pos_num"].iloc[i] or 0.0)) else 0.0
            to = float(df["turnover"].iloc[i]) if np.isfinite(float(df["turnover"].iloc[i] or 0.0)) else 0.0

            ns_prev = nav_s[-1]
            ns = ns_prev * (1.0 - cost_rate * to)
            ns = ns * (1.0 + posi * r)
            nav_s.append(float(ns))

            nb_prev = nav_b[-1]
            nb = nb_prev * (1.0 + r)
            nav_b.append(float(nb))
    else:
        # legacy: apply position to same-day close-to-close return
        for i in range(1, len(df)):
            r = float(df["etf_ret_exec"].iloc[i]) if np.isfinite(float(df["etf_ret_exec"].iloc[i] or 0.0)) else 0.0
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
    # Excess NAV (strategy daily active return vs buy&hold)
    strat_daily = pd.Series(df["nav_strategy"], index=pd.to_datetime(df.index)).pct_change().fillna(0.0).astype(float)
    bh_daily = pd.Series(df["nav_buy_hold"], index=pd.to_datetime(df.index)).pct_change().fillna(0.0).astype(float)
    active = (strat_daily - bh_daily).astype(float)
    nav_excess = (1.0 + active).cumprod().astype(float)
    nav_excess.iloc[0] = 1.0
    df["nav_excess"] = nav_excess.to_numpy(dtype=float)

    # metrics
    nav_s_ser = pd.Series(df["nav_strategy"], index=pd.to_datetime(df.index), dtype=float)
    nav_b_ser = pd.Series(df["nav_buy_hold"], index=pd.to_datetime(df.index), dtype=float)
    nav_x_ser = pd.Series(df["nav_excess"], index=pd.to_datetime(df.index), dtype=float)
    ms = _metrics_from_nav(nav=nav_s_ser, rf_annual=0.0)
    mb = _metrics_from_nav(nav=nav_b_ser, rf_annual=0.0)
    mx = _metrics_from_nav(nav=nav_x_ser, rf_annual=0.0)
    # IR on daily active return (strategy - buy&hold)
    ir = float(_information_ratio(active, ann_factor=TRADING_DAYS_PER_YEAR))
    mx["information_ratio"] = ir

    # period returns + win/payoff/kelly
    periods = {
        "daily": pd.DataFrame({"period_end": nav_s_ser.index.date.astype(str), "return": strat_daily.to_numpy(dtype=float)}).iloc[1:].reset_index(drop=True),
        "weekly": _period_returns(nav_s_ser, "W-FRI"),
        "monthly": _period_returns(nav_s_ser, "ME"),
        "quarterly": _period_returns(nav_s_ser, "QE"),
        "yearly": _period_returns(nav_s_ser, "YE"),
    }
    periods_b = {
        "daily": pd.DataFrame({"period_end": nav_b_ser.index.date.astype(str), "return": bh_daily.to_numpy(dtype=float)}).iloc[1:].reset_index(drop=True),
        "weekly": _period_returns(nav_b_ser, "W-FRI"),
        "monthly": _period_returns(nav_b_ser, "ME"),
        "quarterly": _period_returns(nav_b_ser, "QE"),
        "yearly": _period_returns(nav_b_ser, "YE"),
    }
    periods_x = {
        "daily": pd.DataFrame({"period_end": nav_x_ser.index.date.astype(str), "return": active.to_numpy(dtype=float)}).iloc[1:].reset_index(drop=True),
        "weekly": _period_returns(nav_x_ser, "W-FRI"),
        "monthly": _period_returns(nav_x_ser, "ME"),
        "quarterly": _period_returns(nav_x_ser, "QE"),
        "yearly": _period_returns(nav_x_ser, "YE"),
    }

    period_returns_out: dict[str, Any] = {}
    distributions_out: dict[str, Any] = {}
    win_payoff_out: dict[str, Any] = {}
    for k in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
        s_df = periods[k]
        b_df = periods_b[k]
        x_df = periods_x[k]
        period_returns_out[k] = {
            "strategy": s_df.to_dict(orient="records"),
            "buy_hold": b_df.to_dict(orient="records"),
            "excess": x_df.to_dict(orient="records"),
        }
        distributions_out[k] = {
            "strategy": _dist_stats(pd.Series(s_df["return"].astype(float).to_numpy(dtype=float)) if (s_df is not None and not s_df.empty) else pd.Series([], dtype=float)),
            "buy_hold": _dist_stats(pd.Series(b_df["return"].astype(float).to_numpy(dtype=float)) if (b_df is not None and not b_df.empty) else pd.Series([], dtype=float)),
            "excess": _dist_stats(pd.Series(x_df["return"].astype(float).to_numpy(dtype=float)) if (x_df is not None and not x_df.empty) else pd.Series([], dtype=float)),
        }
        # win/payoff/kelly (absolute return, not log)
        win_payoff_out[k] = {
            "strategy": _win_payoff_kelly(s_df),
            "buy_hold": _win_payoff_kelly(b_df),
            "excess": _win_payoff_kelly(x_df),
        }

    # Alignment self-check (guard against future-function risk).
    violations: list[dict[str, Any]] = []
    if str(index_align or "none").strip().lower() == "cn_next_trading_day":
        for cn_d, us_d in zip(df.index, df["idx_us_date"].to_list()):
            if isinstance(cn_d, dt.date) and isinstance(us_d, dt.date):
                if us_d >= cn_d:
                    violations.append({"cn_date": cn_d.isoformat(), "us_date": us_d.isoformat()})

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
                "thr_abs": (float(r["thr_abs"]) if pd.notna(r.get("thr_abs")) else None),
                "thr_eff": (float(r["thr_eff"]) if pd.notna(r.get("thr_eff")) else None),
                "sig_active": bool(r.get("sig_active")) if pd.notna(r.get("sig_active")) else False,
                "etf_open": (float(r["etf_open"]) if pd.notna(r.get("etf_open")) else None),
                "etf_close": float(r["etf_close"]) if pd.notna(r.get("etf_close")) else None,
                "etf_ret_exec": (float(r["etf_ret_exec"]) if pd.notna(r.get("etf_ret_exec")) else None),
                "turnover": float(r["turnover"]) if pd.notna(r.get("turnover")) else 0.0,
                "trade_cost_bps": float(max(0.0, trade_cost_bps)),
                "nav_strategy": float(r["nav_strategy"]) if pd.notna(r.get("nav_strategy")) else None,
                "nav_buy_hold": float(r["nav_buy_hold"]) if pd.notna(r.get("nav_buy_hold")) else None,
                "nav_excess": float(r["nav_excess"]) if pd.notna(r.get("nav_excess")) else None,
            }
        )

    return {
        "ok": True,
        "meta": {
            "index": str(index),
            "index_align": str(index_align),
            "calendar": str(calendar),
            "exec_model": str(em),
            "return_basis": ("etf_open[t]->etf_open[t+1]" if em == "open_open" else "etf_close.pct_change"),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "n": int(len(df)),
            "lookback_window": int(rw),
            "threshold_quantile": float(q),
            "min_abs_ret": float(max(0.0, min_abs_ret)),
            "trade_cost_bps": float(max(0.0, trade_cost_bps)),
            "initial_position": str(initial_position),
            "alignment_check": {
                "ok": bool(len(violations) == 0),
                "violation_count": int(len(violations)),
                "violations": violations[:10],
            },
        },
        "series": {
            "dates": [d.isoformat() for d in df.index],
            "nav_strategy": df["nav_strategy"].astype(float).tolist(),
            "nav_buy_hold": df["nav_buy_hold"].astype(float).tolist(),
            "nav_excess": df["nav_excess"].astype(float).tolist(),
            "position": df["pos_num"].astype(float).tolist(),
            "action": df["action"].astype(str).tolist(),
            "turnover": df["turnover"].astype(float).tolist(),
            "etf_open": df["etf_open"].astype(float).tolist(),
            "etf_close": df["etf_close"].astype(float).tolist(),
            "etf_ret_exec": df["etf_ret_exec"].astype(float).tolist(),
            "idx_close": df["idx_close"].astype(float).tolist(),
            "idx_log_ret": df["idx_log_ret"].astype(float).tolist(),
            "thr_abs": df["thr_abs"].astype(float).tolist(),
            "thr_eff": df["thr_eff"].astype(float).tolist(),
            "sig_active": df["sig_active"].astype(bool).tolist(),
        },
        "metrics": {"strategy": ms, "buy_hold": mb, "excess": mx, "win_payoff_kelly": win_payoff_out},
        "period_returns": period_returns_out,
        "distributions": distributions_out,
        "trades": trades,
    }

