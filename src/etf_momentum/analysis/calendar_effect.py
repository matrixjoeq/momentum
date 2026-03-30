from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from .baseline import (
    TRADING_DAYS_PER_YEAR,
    _annualized_return,
    _annualized_vol,
    _information_ratio,
    _max_drawdown,
    _max_drawdown_duration_days,
    _sharpe,
    _sortino,
    _ulcer_index,
    hfq_close_daily_equal_weight_returns,
    load_close_prices,
    load_ohlc_prices,
)
from .execution_timing import forward_align_returns
from .rotation import RotationAnalysisInputs, compute_rotation_backtest


def _weekly_anchor_from_weekday(weekday: int) -> str:
    """
    调仓日=决策日；1=Mon .. 5=Fri（等权/轮动日历效应统一 1-5）
    """
    m = {1: "MON", 2: "TUE", 3: "WED", 4: "THU", 5: "FRI"}
    if int(weekday) not in m:
        raise ValueError("weekday must be within [1..5] (Mon..Fri)")
    return m[int(weekday)]


def _decision_dates_for_rebalance(
    dates: pd.DatetimeIndex,
    *,
    rebalance: str,
    anchor: int,
    shift: str = "next",
) -> list[pd.Timestamp]:
    """
    Compute rebalance decision dates on a trading-day calendar.

    anchor semantics:
    - weekly: 1=Mon..5=Fri 调仓日=决策日（use last trading day in that weekly period）
    - monthly: day-of-month 1..28 (use first trading day with day >= anchor in that month; fallback to month-end)
    - quarterly/yearly: Nth trading day in the period (1-indexed; fallback to period-end)
    """
    r = (rebalance or "weekly").strip().lower()
    if r not in {"weekly", "monthly", "quarterly", "yearly"}:
        raise ValueError(f"invalid rebalance={rebalance} (expected weekly/monthly/quarterly/yearly)")
    if dates.empty:
        return []

    sm = (shift or "next").strip().lower()
    if sm not in {"prev", "next"}:
        raise ValueError("rebalance_shift must be one of: prev|next")

    def _shift_to_trading_day(target: pd.Timestamp) -> pd.Timestamp:
        t = pd.to_datetime(target).normalize()
        if t in dates:
            return pd.Timestamp(t)
        pos = int(dates.searchsorted(t))
        if sm == "next":
            return dates[min(pos, len(dates) - 1)]
        # prev
        return dates[max(pos - 1, 0)]

    if r == "weekly":
        wd = int(anchor)
        w_anchor = _weekly_anchor_from_weekday(wd)
        labels = dates.to_period(f"W-{w_anchor}")
        periods = pd.unique(labels)
        out: list[pd.Timestamp] = []
        seen: set[pd.Timestamp] = set()
        for p in periods:
            # weekly anchor is the period end calendar day (Mon..Fri).
            target = pd.Timestamp(p.end_time).normalize()
            d = _shift_to_trading_day(target)
            if d not in seen:
                out.append(d)
                seen.add(d)
        return out

    if r == "monthly":
        dom = int(anchor)
        if dom < 1 or dom > 28:
            raise ValueError("monthly anchor must be within [1..28] (day-of-month)")
        labels = dates.to_period("M")
        periods = pd.unique(labels)
        out = []
        seen: set[pd.Timestamp] = set()
        for p in periods:
            target = pd.Timestamp(dt.date(int(p.year), int(p.month), dom))
            d = _shift_to_trading_day(target)
            if d not in seen:
                out.append(d)
                seen.add(d)
        return out

    # quarterly/yearly: Nth trading day in that period
    n = int(anchor)
    if n < 1:
        raise ValueError("anchor must be >= 1 (Nth trading day)")
    labels = dates.to_period("Q" if r == "quarterly" else "Y")
    out = []
    for _, pos in pd.Series(np.arange(len(dates)), index=dates).groupby(labels):
        arr = pos.to_numpy(dtype=int)
        dts = dates[arr]
        k = min(int(n) - 1, len(dts) - 1)
        out.append(dts[int(k)])
    return out


def _ew_nav_and_weights_by_decision_dates(
    daily_ret: pd.DataFrame,
    *,
    decision_dates: list[pd.Timestamp],
    exec_price: str = "open",
    ret_aligned: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Equal-weight portfolio with drifting weights and rebalancing to equal weights at decision dates (applied next trading day).

    调仓日=决策日，执行日=下一交易日；收益从执行日开始计算。
    成交价=收盘价时，使用执行日收盘价：ret_fwd=forward_align(daily_ret)，即 ret_fwd[t]=ret[t+1]，执行日 t 入场价为 close[t]。
    If ret_aligned is provided, it is used as the return series (e.g. with execution-day same-day return for open).
    """
    if daily_ret.empty:
        return pd.Series(dtype=float, name="EW"), pd.DataFrame()
    cols = list(daily_ret.columns)
    n = len(cols)
    if n <= 0:
        return pd.Series(dtype=float, name="EW"), pd.DataFrame()

    r = daily_ret.astype(float).fillna(0.0).to_numpy(dtype=float)
    idx = daily_ret.index
    decision_set = set(pd.to_datetime(decision_dates))

    w = np.full(n, 1.0 / n, dtype=float)
    w_out: list[np.ndarray] = []

    for i, _t in enumerate(idx):
        # apply rebalance from the day after decision date
        if i > 0 and pd.to_datetime(idx[i - 1]) in decision_set:
            w[:] = 1.0 / n
        w_out.append(w.copy())
        ri = r[i]
        port_r = float(np.dot(w, ri))
        if 1.0 + port_r != 0.0:
            w = w * (1.0 + ri) / (1.0 + port_r)

    w_df = pd.DataFrame(np.vstack(w_out), index=idx, columns=cols, dtype=float)
    if ret_aligned is not None and not ret_aligned.empty:
        ret_fwd = ret_aligned.reindex(index=idx, columns=cols).fillna(0.0).astype(float)
    else:
        ret_fwd = forward_align_returns(daily_ret.astype(float).fillna(0.0))
    port_ret = (w_df * ret_fwd).sum(axis=1).astype(float)
    s = (1.0 + port_ret).cumprod().astype(float).rename("EW")
    if len(s) > 0:
        s.iloc[0] = 1.0
    return s, w_df


def _pick_exec_price(ohlc: dict[str, pd.DataFrame], *, exec_price: str) -> pd.DataFrame:
    """Return OHLC column for open or close. OC2 is handled separately (50% open + 50% close daily returns)."""
    ep = (exec_price or "close").strip().lower()
    if ep not in {"open", "close"}:
        raise ValueError(f"invalid exec_price={exec_price} (expected open|close)")
    return ohlc[ep]


def _common_start_from_prices(px: pd.DataFrame, codes: list[str]) -> pd.Timestamp:
    first_valid = {c: (px[c].first_valid_index() if c in px.columns else None) for c in codes}
    valid = [d for d in first_valid.values() if d is not None]
    if not valid:
        raise ValueError("no valid price data")
    return max(valid)


@dataclass(frozen=True)
class BaselineCalendarEffectInputs:
    codes: list[str]
    start: dt.date
    end: dt.date
    adjust: str = "hfq"
    risk_free_rate: float = 0.025
    rebalance: str = "weekly"
    rebalance_shift: str = "prev"
    anchors: list[int] | None = None  # semantics depend on rebalance
    exec_prices: list[str] | None = None  # open/close/ohlc4


def compute_baseline_calendar_effect(db: Session, inp: BaselineCalendarEffectInputs) -> dict[str, Any]:
    codes = list(dict.fromkeys(inp.codes))
    if not codes:
        raise ValueError("codes is empty")

    reb = (inp.rebalance or "weekly").strip().lower()
    anchors = inp.anchors or ([1, 2, 3, 4, 5] if reb == "weekly" else [1])
    exec_prices = inp.exec_prices or ["open", "close", "oc2"]

    ohlc = load_ohlc_prices(db, codes=codes, start=inp.start, end=inp.end, adjust=inp.adjust)
    if ohlc["close"].empty:
        raise ValueError("no price data for given range")

    # Benchmark for information ratio: same-frequency EW rebalancing on hfq close.
    bench_close_hfq = load_close_prices(db, codes=codes, start=inp.start, end=inp.end, adjust="hfq")

    grid: list[dict[str, Any]] = []
    # rolling return stability diagnostics (1/3/5 years)
    years_list = [1, 3, 5]
    win_days = [int(TRADING_DAYS_PER_YEAR * y) for y in years_list]
    roll_step = 5  # downsample to reduce payload
    rolling_dates: list[str] | None = None
    rolling_series: dict[str, dict[str, list[float | None]]] = {}
    rolling_stats: dict[str, dict[str, dict[str, float | int | None]]] = {}

    def _rolling_return(nav: pd.Series, w: int) -> pd.Series:
        return (nav / nav.shift(int(w)) - 1.0).astype(float)

    def _stats(x: pd.Series) -> dict[str, float | int | None]:
        v = x.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        if v.empty:
            return {"count": 0, "mean": None, "std": None, "min": None, "p10": None, "p50": None, "p90": None, "max": None, "pos_ratio": None}
        arr = v.to_numpy(dtype=float)
        return {
            "count": int(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(np.max(arr)),
            "pos_ratio": float(np.mean(arr > 0.0)),
        }
    for a in anchors:
        for ep in exec_prices:
            ep_l = str(ep).strip().lower()
            if ep_l == "oc2":
                oo = ohlc.get("open", pd.DataFrame())
                cc = ohlc.get("close", pd.DataFrame())
                if oo.empty or cc.empty:
                    grid.append({"anchor": int(a), "exec_price": str(ep), "ok": False, "reason": "empty open/close for oc2"})
                    continue
                oo = oo.sort_index().reindex(columns=codes).replace([np.inf, -np.inf], np.nan).ffill()
                cc = cc.sort_index().reindex(columns=codes).replace([np.inf, -np.inf], np.nan).ffill()
                missing = [
                    c
                    for c in codes
                    if c not in oo.columns
                    or c not in cc.columns
                    or oo[c].dropna().empty
                    or cc[c].dropna().empty
                ]
                if missing:
                    grid.append(
                        {
                            "anchor": int(a),
                            "exec_price": str(ep),
                            "ok": False,
                            "reason": f"missing data for adjust={inp.adjust}: {missing}",
                        }
                    )
                    continue
                common_start = _common_start_from_prices(cc, codes)
                px_common = cc.loc[common_start:, codes]
                oo_c = oo.reindex(px_common.index).ffill().reindex(columns=codes)
                cc_c = cc.reindex(px_common.index).ffill().reindex(columns=codes)
                daily_ret_o = oo_c.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
                daily_ret_c = cc_c.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
                daily_ret = (0.5 * (daily_ret_o + daily_ret_c)).astype(float)
                ret_aligned = None
            else:
                px = _pick_exec_price(ohlc, exec_price=str(ep))
                if px.empty:
                    grid.append({"anchor": int(a), "exec_price": str(ep), "ok": False, "reason": "empty price matrix"})
                    continue

                px = px.sort_index()
                missing = [c for c in codes if c not in px.columns or px[c].dropna().empty]
                if missing:
                    grid.append(
                        {
                            "anchor": int(a),
                            "exec_price": str(ep),
                            "ok": False,
                            "reason": f"missing data for adjust={inp.adjust}: {missing}",
                        }
                    )
                    continue

                px_ff = px.ffill()
                common_start = _common_start_from_prices(px, codes)
                px_common = px_ff.loc[common_start:, codes]
                daily_ret = px_common.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
                ret_aligned = None

            decision_dates = _decision_dates_for_rebalance(px_common.index, rebalance=reb, anchor=int(a), shift=str(inp.rebalance_shift))
            if ep_l == "open":
                ret_fwd = forward_align_returns(daily_ret[codes].astype(float).fillna(0.0))
                decision_set = set(pd.to_datetime(decision_dates))
                idx = px_common.index
                exec_day_positions = set()
                for d in decision_dates:
                    dt = pd.to_datetime(d)
                    if dt in idx:
                        pos = idx.get_loc(dt)
                        if pos + 1 < len(idx):
                            exec_day_positions.add(int(pos) + 1)
                if exec_day_positions and "open" in ohlc and "close" in ohlc:
                    co = ohlc["close"][codes].reindex(idx).ffill()
                    oo = ohlc["open"][codes].reindex(idx).ffill()
                    same_day = (co / oo - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
                    for j in exec_day_positions:
                        ret_fwd.iloc[j] = same_day.iloc[j]
                ret_aligned = ret_fwd
            ew_nav, ew_w = _ew_nav_and_weights_by_decision_dates(
                daily_ret[codes], decision_dates=decision_dates, exec_price=str(ep), ret_aligned=ret_aligned
            )
            ew_ret = ew_nav.pct_change().fillna(0.0)

            # Benchmark: daily HFQ close equal-weight (no costs), same calendar as strategy grid.
            bench_ch = bench_close_hfq.sort_index().reindex(px_common.index).reindex(columns=codes).ffill()
            bench_daily = hfq_close_daily_equal_weight_returns(bench_ch, dynamic_universe=False)
            bench_nav = (1.0 + bench_daily).cumprod().astype(float)
            if len(bench_nav) > 0:
                bench_nav.iloc[0] = 1.0
            active_daily = (ew_ret - bench_daily).astype(float)
            info_ratio = _information_ratio(active_daily, ann_factor=TRADING_DAYS_PER_YEAR)

            # implied turnover from weights (same definition as rotation module)
            w_prev = ew_w.shift(1).fillna(0.0)
            turnover = (ew_w - w_prev).abs().sum(axis=1) / 2.0

            ann_ret = _annualized_return(ew_nav, ann_factor=TRADING_DAYS_PER_YEAR)
            ann_vol = _annualized_vol(ew_ret, ann_factor=TRADING_DAYS_PER_YEAR)
            mdd = _max_drawdown(ew_nav)
            mdd_dur = _max_drawdown_duration_days(ew_nav)
            sharpe = _sharpe(ew_ret, rf=float(inp.risk_free_rate), ann_factor=TRADING_DAYS_PER_YEAR)
            sortino = _sortino(ew_ret, rf=float(inp.risk_free_rate), ann_factor=TRADING_DAYS_PER_YEAR)
            calmar = float(ann_ret / abs(mdd)) if np.isfinite(mdd) and float(mdd) < 0 else float("nan")
            ui = _ulcer_index(ew_nav, in_percent=True)
            ui_den = ui / 100.0
            upi = float((ann_ret - float(inp.risk_free_rate)) / ui_den) if ui_den > 0 else float("nan")

            grid.append(
                {
                    "anchor": int(a),
                    "exec_price": str(ep),
                    "ok": True,
                    "metrics": {
                        "cumulative_return": float(ew_nav.iloc[-1] - 1.0) if len(ew_nav) else float("nan"),
                        "annualized_return": float(ann_ret),
                        "annualized_volatility": float(ann_vol),
                        "max_drawdown": float(mdd),
                        "max_drawdown_recovery_days": int(mdd_dur),
                        "sharpe_ratio": float(sharpe),
                        "calmar_ratio": float(calmar),
                        "sortino_ratio": float(sortino),
                        "ulcer_index": float(ui),
                        "ulcer_performance_index": float(upi),
                        "information_ratio": float(info_ratio),
                        "avg_daily_turnover": float(turnover.mean()) if len(turnover) else float("nan"),
                    },
                }
            )

            # rolling return curves + stability stats
            key = f"{int(a)}|{str(ep)}"
            if rolling_dates is None:
                rolling_dates = ew_nav.index[::roll_step].date.astype(str).tolist()
            rolling_series[key] = {}
            rolling_stats[key] = {}
            for y, w in zip(years_list, win_days):
                rr = _rolling_return(ew_nav, w)
                rr_ds = rr.iloc[::roll_step].replace([np.inf, -np.inf], np.nan)
                rolling_series[key][f"{int(y)}y"] = [None if pd.isna(v) else float(v) for v in rr_ds.to_numpy(dtype=float)]
                rolling_stats[key][f"{int(y)}y"] = _stats(rr)

    return {
        "meta": {
            "type": "baseline_calendar_effect",
            "codes": codes,
            "start": inp.start.strftime("%Y%m%d"),
            "end": inp.end.strftime("%Y%m%d"),
            "adjust": inp.adjust,
            "rebalance": reb,
            "rebalance_shift": str(inp.rebalance_shift),
            "anchors": [int(x) for x in anchors],
            "exec_prices": [str(x) for x in exec_prices],
            "rolling_years": years_list,
            "rolling_step": int(roll_step),
        },
        "grid": grid,
        "rolling": {"dates": (rolling_dates or []), "series": rolling_series},
        "rolling_stats": rolling_stats,
        "exec_price_map": {"open": "开盘", "close": "收盘", "oc2": "OC2(50%开+50%收日收益)"},
    }


def compute_rotation_calendar_effect(
    db: Session,
    *,
    base: RotationAnalysisInputs,
    anchors: list[int] | None = None,
    exec_prices: list[str] | None = None,
) -> dict[str, Any]:
    """
    Calendar-effect grid study for rotation strategy:
    - vary weekly rebalance weekday (Mon..Fri)
    - vary execution price proxy (open/close/ohlc4)

    This runs multiple backtests; keep candidate pool size and range reasonable.
    """
    reb = (base.rebalance or "weekly").strip().lower()
    anchors = anchors or ([1, 2, 3, 4, 5] if reb == "weekly" else [1])
    exec_prices = exec_prices or ["open", "close", "oc2"]

    grid: list[dict[str, Any]] = []
    years_list = [1, 3, 5]
    win_days = [int(TRADING_DAYS_PER_YEAR * y) for y in years_list]
    roll_step = 5
    rolling_dates: list[str] | None = None
    rolling_series: dict[str, dict[str, list[float | None]]] = {}
    rolling_stats: dict[str, dict[str, dict[str, float | int | None]]] = {}

    def _rolling_return(nav: pd.Series, w: int) -> pd.Series:
        return (nav / nav.shift(int(w)) - 1.0).astype(float)

    def _stats(x: pd.Series) -> dict[str, float | int | None]:
        v = x.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        if v.empty:
            return {"count": 0, "mean": None, "std": None, "min": None, "p10": None, "p50": None, "p90": None, "max": None, "pos_ratio": None}
        arr = v.to_numpy(dtype=float)
        return {
            "count": int(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(np.max(arr)),
            "pos_ratio": float(np.mean(arr > 0.0)),
        }

    # 调仓日=决策日，1-5=周一~周五（轮动日历效应锚点统一为 1-5）
    for a in anchors:
        for ep in exec_prices:
            try:
                inp = RotationAnalysisInputs(
                    **{
                        **base.__dict__,
                        "rebalance": reb,
                        "rebalance_anchor": int(a),
                        "exec_price": str(ep),
                    }
                )
                res = compute_rotation_backtest(db, inp, include_benchmarks=False)
                strat = (res.get("metrics") or {}).get("strategy") or {}

                nav_dates = (res.get("nav") or {}).get("dates") or []
                nav_series = ((res.get("nav") or {}).get("series") or {}).get("ROTATION") or []
                grid.append(
                    {
                        "anchor": int(a),
                        "exec_price": str(ep),
                        "ok": True,
                        "metrics": strat,
                    }
                )

                # rolling return curves + stability stats (strategy NAV series)
                if nav_dates and nav_series and (rolling_dates is None):
                    rolling_dates = [str(x) for x in nav_dates[::roll_step]]
                if nav_dates and nav_series:
                    idx = pd.to_datetime(nav_dates)
                    nav = pd.Series([float(x) for x in nav_series], index=idx, dtype=float)
                    key = f"{int(a)}|{str(ep)}"
                    rolling_series[key] = {}
                    rolling_stats[key] = {}
                    for y, w in zip(years_list, win_days):
                        rr = _rolling_return(nav, w)
                        rr_ds = rr.iloc[::roll_step].replace([np.inf, -np.inf], np.nan)
                        rolling_series[key][f"{int(y)}y"] = [None if pd.isna(v) else float(v) for v in rr_ds.to_numpy(dtype=float)]
                        rolling_stats[key][f"{int(y)}y"] = _stats(rr)
            except Exception as e:  # pylint: disable=broad-exception-caught
                grid.append({"anchor": int(a), "exec_price": str(ep), "ok": False, "reason": str(e)})

    return {
        "meta": {
            "type": "rotation_calendar_effect",
            "codes": list(dict.fromkeys(base.codes)),
            "start": base.start.strftime("%Y%m%d"),
            "end": base.end.strftime("%Y%m%d"),
            "rebalance": reb,
            "rebalance_shift": str(getattr(base, "rebalance_shift", "next")),
            "anchors": [int(x) for x in anchors],
            "exec_prices": [str(x) for x in exec_prices],
            "rolling_years": years_list,
            "rolling_step": int(roll_step),
        },
        "grid": grid,
        "rolling": {"dates": (rolling_dates or []), "series": rolling_series},
        "rolling_stats": rolling_stats,
        "exec_price_map": {"open": "开盘", "close": "收盘", "oc2": "OC2(50%开+50%收日收益)"},
    }


