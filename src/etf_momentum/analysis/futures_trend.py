"""
Futures trend backtest (research) — price series policy
------------------------------------------------------
When **synthesized continuous** rows exist for a symbol (codes ``{root}88`` /
``{root}888`` / ``{root}889`` under the same ``Date`` calendar), we apply:

- **Benchmark (buy-and-hold comparison):** synthetic **backward-adjusted**
  (**hfq**, e.g. ``{root}889``) OHLC; compare NAV uses the same open/close basis
  as ``exec_price`` on this hfq series.
- **Signals** (here: TA-Lib SMA crossover on “close”): synthetic **forward-adjusted**
  (**qfq**, e.g. ``{root}888``) **close**, aligned on trading dates with execution data.
- **Trade execution & strategy return compounding:** synthetic **no-adjust**
  (**none**, e.g. ``{root}88``) OHLCV passed into ``backtesting.py``.

If ``{root}88`` is missing (no synthesis yet), we fall back to the group’s listed
contract code (typically main ``*0``) **none** rows only; signals and benchmark then
also use that same none series (see per-symbol ``trend_resolution`` in the API).
"""

from __future__ import annotations

# pylint: disable=broad-exception-caught,attribute-defined-outside-init

import datetime as dt
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..data.futures_synthesize import _symbol_root_from_main
from ..db.futures_repo import list_futures_prices
from ..db.futures_research_repo import FuturesGroupData
from .futures_trend_portfolio_weights import (
    atr_ewm_wilder,
    build_ma_panels,
    equal_weights_from_signals,
    risk_budget_weights,
)
from .trend import (
    _apply_atr_stop,
    _apply_intraday_stop_execution_portfolio,
    _apply_monthly_risk_budget_gate,
    _moving_average,
)

ExecPrice = Literal["open", "close"]
FeeSide = Literal["one_way", "two_way"]
SlippageType = Literal["percent", "price_spread"]


@dataclass(frozen=True)
class CostProfile:
    commission_per_fill: float
    spread_per_fill: float
    commission_input_bps: float
    fee_side: FeeSide
    slippage_type: SlippageType
    slippage_input: float
    slippage_side: FeeSide


def _parse_yyyymmdd(value: str) -> dt.date:
    return dt.datetime.strptime(str(value), "%Y%m%d").date()


def _coerce_trading_index(ix: pd.DatetimeIndex | pd.Index) -> pd.DatetimeIndex:
    """
    Normalize futures OHLC indexes to UTC-naive midnight so strategy NAV,
    benchmark prices, and intersection calendars align (fixes flat benchmark
    when mixing tz-aware / naive timestamps).
    """
    dti = pd.DatetimeIndex(ix)
    if dti.tz is not None:
        dti = dti.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(dti.normalize())


def _normalize_per_fill(value: float, side: FeeSide) -> float:
    if side == "one_way":
        return float(value)
    # two_way means the input is round-trip total, split per fill.
    return float(value) / 2.0


def _resolve_order_size(position_size_pct: float) -> float:
    """
    Normalize order size for backtesting.py semantics.

    In backtesting.py, size in (0,1) is interpreted as equity fraction,
    while size >= 1 is interpreted as unit count. We want a portfolio-style
    sizing control where 1.0 means "fully invested", so map it to just below 1.
    """
    pct = float(position_size_pct)
    if pct <= 0:
        return 0.001
    if pct >= 1.0:
        return 0.999999
    return pct


def _load_futures_ohlcv(
    db: Session,
    *,
    code: str,
    start: str,
    end: str,
    adjust: str = "none",
) -> pd.DataFrame:
    """Load OHLCV for ``code`` + ``adjust`` into a backtesting-compatible frame."""
    s_d = _parse_yyyymmdd(start)
    e_d = _parse_yyyymmdd(end)
    rows = list_futures_prices(
        db,
        code=code,
        adjust=adjust,
        start_date=s_d,
        end_date=e_d,
        limit=300000,
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "Open": [float(r.open) if r.open is not None else np.nan for r in rows],
            "High": [float(r.high) if r.high is not None else np.nan for r in rows],
            "Low": [float(r.low) if r.low is not None else np.nan for r in rows],
            "Close": [float(r.close) if r.close is not None else np.nan for r in rows],
            "Volume": [float(r.volume) if r.volume is not None else 0.0 for r in rows],
        },
        index=pd.to_datetime([r.trade_date for r in rows]),
    )
    df = df.sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"], how="any")
    df.index = _coerce_trading_index(df.index)
    return df


def _align_futures_trend_inputs(
    db: Session,
    *,
    pool_code: str,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]] | None:
    """
    Build execution OHLCV (none) + benchmark OHLC (hfq when available) +
    attach ``SignalClose`` (qfq when available) for the same calendar.

    Returns ``None`` when no usable none-style series exists in range.
    """
    root = _symbol_root_from_main(pool_code)
    code_88 = f"{root}88"
    code_888 = f"{root}888"
    code_889 = f"{root}889"

    df88 = _load_futures_ohlcv(db, code=code_88, start=start, end=end, adjust="none")
    df888 = _load_futures_ohlcv(db, code=code_888, start=start, end=end, adjust="qfq")
    df889 = _load_futures_ohlcv(db, code=code_889, start=start, end=end, adjust="hfq")

    if df88.empty:
        df_main = _load_futures_ohlcv(
            db, code=pool_code, start=start, end=end, adjust="none"
        )
        if df_main.empty:
            return None
        out = df_main.copy()
        out["SignalClose"] = out["Close"].astype(float)
        detail: dict[str, Any] = {
            "trend_resolution": "main_contract_none",
            "execution_symbol": pool_code,
            "signal_symbol": pool_code,
            "benchmark_symbol": pool_code,
            "signal_adjust": "none",
            "benchmark_adjust": "none",
        }
        out.index = _coerce_trading_index(out.index)
        ob = out.copy()
        ob.index = _coerce_trading_index(ob.index)
        return out, ob, detail

    idx = df88.index
    detail_a: dict[str, Any] = {
        "execution_symbol": code_88,
        "signal_adjust": "qfq",
        "benchmark_adjust": "hfq",
    }
    if not df888.empty:
        idx = idx.intersection(df888.index)
    else:
        detail_a["signal_adjust"] = "none_fallback"

    if not df889.empty:
        idx = idx.intersection(df889.index)
    else:
        detail_a["benchmark_adjust"] = "none_fallback"

    idx = idx.sort_values()
    if len(idx) == 0:
        return None

    exec_df = df88.loc[idx].copy()
    if not df888.empty:
        exec_df["SignalClose"] = df888.loc[idx, "Close"].astype(float).values
        detail_a["signal_symbol"] = code_888
    else:
        exec_df["SignalClose"] = exec_df["Close"].astype(float)
        detail_a["signal_symbol"] = code_88

    if not df889.empty:
        bench_df = df889.loc[idx].copy()
        detail_a["benchmark_symbol"] = code_889
    else:
        bench_df = exec_df[["Open", "High", "Low", "Close", "Volume"]].copy()
        detail_a["benchmark_symbol"] = code_88

    if (
        detail_a.get("signal_adjust") == "qfq"
        and detail_a.get("benchmark_adjust") == "hfq"
    ):
        detail_a["trend_resolution"] = "synthetic_triple"
    else:
        detail_a["trend_resolution"] = "synthetic_partial"

    return exec_df, bench_df, detail_a


def _build_cost_profile(
    *,
    cost_bps: float,
    fee_side: FeeSide,
    slippage_type: SlippageType,
    slippage_value: float,
    slippage_side: FeeSide,
    price_reference: float,
) -> CostProfile:
    cost_ratio = float(cost_bps) / 10000.0
    commission_per_fill = _normalize_per_fill(cost_ratio, fee_side)
    if slippage_type == "percent":
        spread_ratio = float(slippage_value)
    else:
        # absolute price spread -> relative spread against price reference
        spread_ratio = (
            (float(slippage_value) / float(price_reference))
            if price_reference > 0
            else 0.0
        )
    spread_per_fill = _normalize_per_fill(spread_ratio, slippage_side)
    return CostProfile(
        commission_per_fill=float(max(0.0, commission_per_fill)),
        spread_per_fill=float(max(0.0, spread_per_fill)),
        commission_input_bps=float(cost_bps),
        fee_side=fee_side,
        slippage_type=slippage_type,
        slippage_input=float(slippage_value),
        slippage_side=slippage_side,
    )


def _combine_group_returns(
    ret_df: pd.DataFrame, *, dynamic_universe: bool
) -> pd.Series:
    if ret_df.empty:
        return pd.Series(dtype=float)
    if dynamic_universe:
        out = ret_df.mean(axis=1, skipna=True)
    else:
        out = ret_df.fillna(0.0).mean(axis=1)
    return out.fillna(0.0)


def _run_vectorized_fallback(
    df: pd.DataFrame,
    *,
    fast_ma: int,
    slow_ma: int,
    exec_price: ExecPrice,
    ma_type: str = "sma",
) -> pd.Series:
    if "SignalClose" in df.columns:
        close = df["SignalClose"].astype(float)
    else:
        close = df["Close"].astype(float)
    mt = str(ma_type or "sma").strip().lower()
    fast = _moving_average(close, window=int(fast_ma), ma_type=mt)
    slow = _moving_average(close, window=int(slow_ma), ma_type=mt)
    signal = (fast > slow).astype(float).fillna(0.0)
    # Strict anti-lookahead rule: signal on t executes on t+1.
    # Using close-close/open-open return legs implies first active return is t+2.
    if exec_price == "close":
        px_ret = close.pct_change().fillna(0.0)
        pos = signal.shift(2).fillna(0.0)
        return pos * px_ret
    # open execution approximation in fallback: open-open return with same lag rule.
    open_px = df["Open"].astype(float)
    px_ret = open_px.pct_change().fillna(0.0)
    pos = signal.shift(2).fillna(0.0)
    return pos * px_ret


def _run_symbol_backtest(
    df: pd.DataFrame,
    *,
    fast_ma: int,
    slow_ma: int,
    exec_price: ExecPrice,
    position_size_pct: float,
    cost: CostProfile,
    ma_type: str = "sma",
) -> tuple[pd.Series, dict]:
    order_size = _resolve_order_size(position_size_pct)

    try:
        from backtesting import Backtest, Strategy
    except Exception:  # pragma: no cover - fallback only when lib unavailable
        ret = _run_vectorized_fallback(
            df,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            exec_price=exec_price,
            ma_type=ma_type,
        )
        nav = (1.0 + ret.fillna(0.0)).cumprod()
        return nav, {
            "engine": "fallback_vectorized",
            "trades": 0,
            "ret_total": float(nav.iloc[-1] - 1.0),
        }

    try:
        import talib
    except Exception:  # pragma: no cover - fallback only when lib unavailable
        talib = None

    class TalibMaCrossTrend(Strategy):
        fast = 20
        slow = 60
        ma_kind = "sma"
        size_pct = 0.999999

        def init(self) -> None:
            sig = getattr(self.data, "SignalClose", None)
            close_arr = np.asarray(
                sig if sig is not None else self.data.Close,
                dtype=float,
            )
            mt = str(self.ma_kind or "sma").strip().lower()
            talib_ma = None
            if talib is not None:
                if mt == "sma":
                    talib_ma = getattr(talib, "SMA", None)
                elif mt == "ema":
                    talib_ma = getattr(talib, "EMA", None)
                elif mt == "wma":
                    talib_ma = getattr(talib, "WMA", None)
            if callable(talib_ma):
                self.fast_ma = self.I(talib_ma, close_arr, int(self.fast))
                self.slow_ma = self.I(talib_ma, close_arr, int(self.slow))
            else:
                cs = pd.Series(close_arr, dtype=float)
                fv = _moving_average(cs, window=int(self.fast), ma_type=mt).to_numpy()
                sv = _moving_average(cs, window=int(self.slow), ma_type=mt).to_numpy()
                self.fast_ma = self.I(lambda: fv)
                self.slow_ma = self.I(lambda: sv)

        def next(self) -> None:
            # Strict anti-lookahead rule:
            # - close execution: use yesterday's signal, execute today close
            # - open execution: use today's signal, execute next day open
            sig_idx = -2 if exec_price == "close" else -1
            if sig_idx == -2 and len(self.data.Close) < 2:
                return
            f = (
                float(self.fast_ma[sig_idx])
                if not pd.isna(self.fast_ma[sig_idx])
                else np.nan
            )
            s = (
                float(self.slow_ma[sig_idx])
                if not pd.isna(self.slow_ma[sig_idx])
                else np.nan
            )
            if np.isnan(f) or np.isnan(s):
                return
            if f > s and not self.position:
                self.buy(size=float(self.size_pct))
            elif f < s and self.position:
                self.position.close()

    bt = Backtest(
        df,
        TalibMaCrossTrend,
        cash=1_000_000.0,
        trade_on_close=(exec_price == "close"),
        commission=float(cost.commission_per_fill),
        spread=float(cost.spread_per_fill),
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run(
        fast=int(fast_ma),
        slow=int(slow_ma),
        ma_kind=str(ma_type),
        size_pct=float(order_size),
    )
    eq = stats.get("_equity_curve")
    if eq is None or "Equity" not in eq:
        ret = _run_vectorized_fallback(
            df,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            exec_price=exec_price,
            ma_type=ma_type,
        )
        nav = (1.0 + ret.fillna(0.0)).cumprod()
        return nav, {
            "engine": "fallback_vectorized",
            "trades": 0,
            "ret_total": float(nav.iloc[-1] - 1.0),
        }
    equity = pd.Series(
        eq["Equity"], index=pd.to_datetime(eq.index), dtype=float
    ).sort_index()
    equity.index = _coerce_trading_index(equity.index)
    nav = (equity / float(equity.iloc[0])).ffill().fillna(1.0)
    trades = int(stats.get("# Trades", 0) or 0)
    ret_total = float(nav.iloc[-1] - 1.0)
    win_rate = float(stats.get("Win Rate [%]", 0.0) or 0.0) / 100.0
    return nav, {
        "engine": "backtesting",
        "trades": trades,
        "ret_total": ret_total,
        "win_rate": win_rate,
    }


def compute_futures_group_trend_backtest(
    db: Session,
    *,
    group: FuturesGroupData,
    start: str,
    end: str,
    dynamic_universe: bool,
    exec_price: ExecPrice = "close",
    trend_strategy: str = "ma_cross",
    fast_ma: int = 20,
    slow_ma: int = 60,
    ma_type: str = "sma",
    position_size_pct: float = 1.0,
    min_points: int = 120,
    cost_bps: float = 5.0,
    fee_side: FeeSide = "two_way",
    slippage_type: SlippageType = "percent",
    slippage_value: float = 0.0005,
    slippage_side: FeeSide = "two_way",
    backtest_mode: str = "portfolio",
    single_code: str | None = None,
    position_sizing: str = "equal",
    risk_budget_atr_window: int = 20,
    risk_budget_pct: float = 0.01,
    risk_budget_overcap_policy: str = "scale",
    risk_budget_max_leverage_multiple: float = 2.0,
    monthly_risk_budget_enabled: bool = False,
    monthly_risk_budget_pct: float = 0.06,
    monthly_risk_budget_include_new_trade_risk: bool = False,
    atr_stop_mode: str = "none",
    atr_stop_atr_basis: str = "latest",
    atr_stop_reentry_mode: str = "reenter",
    atr_stop_window: int = 14,
    atr_stop_n: float = 2.0,
    atr_stop_m: float = 0.5,
) -> dict:
    codes = [str(c).strip().upper() for c in group.codes if str(c).strip()]
    if len(codes) == 0:
        return {"ok": False, "error": "empty_group", "meta": {"group_name": group.name}}
    if int(fast_ma) < 2 or int(slow_ma) <= int(fast_ma):
        return {"ok": False, "error": "invalid_ma_windows"}
    ts_raw = str(trend_strategy or "ma_cross").strip().lower()
    if ts_raw != "ma_cross":
        return {"ok": False, "error": "unsupported_trend_strategy"}
    mt_eff = str(ma_type or "sma").strip().lower()
    if mt_eff not in {"sma", "ema", "wma"}:
        return {"ok": False, "error": "invalid_ma_type"}
    if exec_price not in {"open", "close"}:
        return {"ok": False, "error": "invalid_exec_price"}
    if fee_side not in {"one_way", "two_way"}:
        return {"ok": False, "error": "invalid_fee_side"}
    if slippage_side not in {"one_way", "two_way"}:
        return {"ok": False, "error": "invalid_slippage_side"}
    if slippage_type not in {"percent", "price_spread"}:
        return {"ok": False, "error": "invalid_slippage_type"}
    if position_size_pct <= 0 or position_size_pct > 1:
        return {"ok": False, "error": "invalid_position_size_pct"}

    bm = str(backtest_mode or "portfolio").strip().lower()
    if bm not in {"portfolio", "single"}:
        return {"ok": False, "error": "invalid_backtest_mode"}
    ps = str(position_sizing or "equal").strip().lower()
    if bm == "portfolio" and ps not in {"equal", "risk_budget"}:
        return {"ok": False, "error": "invalid_position_sizing"}
    rb_pol = str(risk_budget_overcap_policy or "scale").strip().lower()
    if rb_pol not in {"scale", "skip_entry", "replace_entry", "leverage_entry"}:
        return {"ok": False, "error": "invalid_risk_budget_overcap_policy"}

    atm_raw = str(atr_stop_mode or "none").strip().lower()
    if atm_raw not in {"none", "static", "trailing", "tightening"}:
        return {"ok": False, "error": "invalid_atr_stop_mode"}
    ab_raw = str(atr_stop_atr_basis or "latest").strip().lower()
    if ab_raw not in {"entry", "latest"}:
        return {"ok": False, "error": "invalid_atr_stop_atr_basis"}
    arm_raw = str(atr_stop_reentry_mode or "reenter").strip().lower()
    if arm_raw not in {"reenter", "wait_next_entry"}:
        return {"ok": False, "error": "invalid_atr_stop_reentry_mode"}

    codes_run = codes
    if bm == "single":
        sc = str(single_code or "").strip().upper()
        if not sc:
            return {"ok": False, "error": "missing_single_code"}
        if sc not in set(codes):
            return {"ok": False, "error": "single_code_not_in_group"}
        codes_run = [sc]

    nav_by_symbol: dict[str, pd.Series] = {}
    bench_price_by_symbol: dict[str, pd.Series] = {}
    exec_by_code: dict[str, pd.DataFrame] = {}
    symbol_stats: list[dict] = []
    errors: list[str] = []

    for code in codes_run:
        aligned = _align_futures_trend_inputs(db, pool_code=code, start=start, end=end)
        if aligned is None:
            errors.append(f"{code}:no_price_rows")
            continue
        df_exec, df_bench, trend_detail = aligned
        if len(df_exec.index) < int(min_points):
            errors.append(f"{code}:points<{int(min_points)}")
            continue
        price_ref = float(df_exec["Close"].median()) if len(df_exec.index) else 0.0
        cost = _build_cost_profile(
            cost_bps=cost_bps,
            fee_side=fee_side,
            slippage_type=slippage_type,
            slippage_value=slippage_value,
            slippage_side=slippage_side,
            price_reference=price_ref,
        )
        nav, st = _run_symbol_backtest(
            df_exec,
            fast_ma=int(fast_ma),
            slow_ma=int(slow_ma),
            exec_price=exec_price,
            position_size_pct=float(position_size_pct),
            cost=cost,
            ma_type=mt_eff,
        )
        nav_by_symbol[code] = nav
        exec_by_code[code] = df_exec.copy()
        bench_col = "Open" if exec_price == "open" else "Close"
        bench_price_by_symbol[code] = df_bench[bench_col].astype(float)
        symbol_stats.append(
            {
                "code": code,
                "points": int(len(df_exec.index)),
                "start": str(df_exec.index.min().date()),
                "end": str(df_exec.index.max().date()),
                "ret_total": float(st.get("ret_total", 0.0)),
                "trades": int(st.get("trades", 0)),
                "win_rate": float(st.get("win_rate", 0.0)),
                "engine": str(st.get("engine", "unknown")),
                "commission_per_fill": float(cost.commission_per_fill),
                "spread_per_fill": float(cost.spread_per_fill),
                "trend_resolution": trend_detail.get("trend_resolution"),
                "trend_execution_symbol": trend_detail.get("execution_symbol"),
                "trend_signal_adjust": trend_detail.get("signal_adjust"),
                "trend_benchmark_adjust": trend_detail.get("benchmark_adjust"),
            }
        )

    if not nav_by_symbol:
        return {
            "ok": False,
            "error": "insufficient_data",
            "meta": {
                "group_name": group.name,
                "start": start,
                "end": end,
                "errors": errors,
            },
        }

    common_idx: pd.DatetimeIndex | None = None
    for nav in nav_by_symbol.values():
        ix = _coerce_trading_index(nav.index)
        common_idx = ix if common_idx is None else common_idx.intersection(ix)
    if common_idx is None or len(common_idx) < 2:
        return {
            "ok": False,
            "error": "insufficient_overlap",
            "meta": {
                "group_name": group.name,
                "start": start,
                "end": end,
                "errors": errors,
            },
        }
    common_idx = common_idx.sort_values()

    ret_parts: dict[str, pd.Series] = {}
    for c, nav in nav_by_symbol.items():
        nv = nav.reindex(common_idx).ffill()
        ret_parts[str(c)] = nv.pct_change().fillna(0.0)
    ret_mat = pd.DataFrame(ret_parts).sort_index().astype(float)

    exec_aligned = {c: exec_by_code[c].reindex(common_idx) for c in exec_by_code}

    portfolio_meta: dict[str, Any] = {}
    atr_stop_by_asset: dict[str, dict[str, Any]] = {}
    if bm == "single":
        portfolio_meta["allocation"] = "single_asset_full_notional"
        code = str(codes_run[0])
        _, sig_df = build_ma_panels(
            exec_aligned,
            common_idx=common_idx,
            fast_ma=int(fast_ma),
            slow_ma=int(slow_ma),
            ma_type=mt_eff,
        )
        if atm_raw != "none":
            ex = exec_aligned.get(code)
            if ex is not None:
                out_s, st = _apply_atr_stop(
                    sig_df[code].astype(float).reindex(common_idx).fillna(0.0),
                    open_=ex["Open"].reindex(common_idx).astype(float),
                    close=ex["Close"].reindex(common_idx).astype(float),
                    high=ex["High"].reindex(common_idx).astype(float),
                    low=ex["Low"].reindex(common_idx).astype(float),
                    mode=atm_raw,
                    atr_basis=ab_raw,
                    reentry_mode=arm_raw,
                    atr_window=int(atr_stop_window),
                    n_mult=float(atr_stop_n),
                    m_step=float(atr_stop_m),
                    same_day_stop=True,
                )
                sig_df = sig_df.copy()
                sig_df[code] = out_s.reindex(sig_df.index).fillna(0.0).astype(float)
                atr_stop_by_asset[code] = st
        w_df = equal_weights_from_signals(sig_df)
        portfolio_meta["universal_atr_stop"] = {
            "applied": atm_raw != "none",
            "same_day_stop": True,
            "per_symbol_stats_keys": sorted(atr_stop_by_asset.keys()),
        }
        portfolio_meta["universal_atr_stop_applied_in_engine"] = bool(atm_raw != "none")

        w_eff = (
            w_df.reindex(index=ret_mat.index, columns=ret_mat.columns)
            .fillna(0.0)
            .astype(float)
            .shift(1)
            .fillna(0.0)
        )
        atr_override = pd.Series(0.0, index=ret_mat.index, dtype=float)
        if atm_raw != "none" and atr_stop_by_asset:
            open_df = pd.DataFrame(
                {
                    str(code): exec_aligned[code]["Open"].astype(float),
                },
                index=common_idx,
            )
            close_df_exec = pd.DataFrame(
                {
                    str(code): exec_aligned[code]["Close"].astype(float),
                },
                index=common_idx,
            )
            w_eff, atr_override = _apply_intraday_stop_execution_portfolio(
                weights=w_eff,
                atr_stop_by_asset=atr_stop_by_asset,
                exec_price=str(exec_price),
                open_sig_df=open_df.reindex(
                    index=w_eff.index, columns=w_eff.columns
                ).astype(float),
                close_sig_df=close_df_exec.reindex(
                    index=w_eff.index, columns=w_eff.columns
                ).astype(float),
            )
        group_ret = (w_eff * ret_mat.astype(float)).sum(axis=1).astype(float).fillna(
            0.0
        ) + atr_override.reindex(ret_mat.index).fillna(0.0).astype(float)
    else:
        score_df, sig_df = build_ma_panels(
            exec_aligned,
            common_idx=common_idx,
            fast_ma=int(fast_ma),
            slow_ma=int(slow_ma),
            ma_type=mt_eff,
        )
        if atm_raw != "none":
            sig_adj = sig_df.copy()
            for c in list(sig_df.columns):
                ex = exec_aligned.get(c)
                if ex is None:
                    continue
                o_ = ex["Open"].reindex(common_idx).astype(float)
                cl_ = ex["Close"].reindex(common_idx).astype(float)
                hi_ = ex["High"].reindex(common_idx).astype(float)
                lo_ = ex["Low"].reindex(common_idx).astype(float)
                out_s, st = _apply_atr_stop(
                    sig_df[c].astype(float).reindex(common_idx).fillna(0.0),
                    open_=o_,
                    close=cl_,
                    high=hi_,
                    low=lo_,
                    mode=atm_raw,
                    atr_basis=ab_raw,
                    reentry_mode=arm_raw,
                    atr_window=int(atr_stop_window),
                    n_mult=float(atr_stop_n),
                    m_step=float(atr_stop_m),
                    same_day_stop=True,
                )
                sig_adj[c] = out_s.reindex(sig_adj.index).fillna(0.0).astype(float)
                atr_stop_by_asset[str(c)] = st
            sig_df = sig_adj

        if ps == "equal":
            w_df = equal_weights_from_signals(sig_df)
            portfolio_meta["position_sizing"] = "equal"
            portfolio_meta["note"] = (
                "equal weight among MA-long symbols each day; weights lagged one day "
                "into returns"
            )
        else:
            w_df, rb_stats = risk_budget_weights(
                sig_binary_df=sig_df,
                score_df=score_df,
                exec_by_code=exec_aligned,
                common_idx=common_idx,
                risk_budget_atr_window=int(risk_budget_atr_window),
                risk_budget_pct=float(risk_budget_pct),
                policy=rb_pol,
                max_leverage_multiple=float(risk_budget_max_leverage_multiple),
            )
            portfolio_meta["position_sizing"] = "risk_budget"
            portfolio_meta["risk_budget"] = rb_stats

        portfolio_meta["universal_atr_stop"] = {
            "applied": atm_raw != "none",
            "same_day_stop": True,
            "per_symbol_stats_keys": sorted(atr_stop_by_asset.keys()),
        }
        portfolio_meta["universal_atr_stop_applied_in_engine"] = bool(atm_raw != "none")

        if bm == "portfolio" and bool(monthly_risk_budget_enabled):
            cc = [str(c) for c in ret_mat.columns]
            close_df = pd.DataFrame(
                {
                    c: exec_aligned[c]["Close"].astype(float)
                    for c in cc
                    if c in exec_aligned
                },
                index=common_idx,
            )
            atr_gate = pd.DataFrame(index=common_idx, columns=cc, dtype=float)
            w_atr = max(2, int(atr_stop_window))
            for c in cc:
                if c not in exec_aligned:
                    continue
                ex = exec_aligned[c]
                atr_gate[c] = atr_ewm_wilder(
                    ex["High"].astype(float),
                    ex["Low"].astype(float),
                    ex["Close"].astype(float),
                    window=w_atr,
                )
            atm = atm_raw
            w_df, gate_stats = _apply_monthly_risk_budget_gate(
                w_df.reindex(index=close_df.index, columns=close_df.columns)
                .astype(float)
                .fillna(0.0),
                close=close_df.astype(float),
                atr=atr_gate.astype(float),
                enabled=True,
                budget_pct=float(monthly_risk_budget_pct),
                include_new_trade_risk=bool(monthly_risk_budget_include_new_trade_risk),
                atr_stop_enabled=(atm != "none"),
                atr_mode=atm,
                atr_basis=ab_raw,
                atr_n=float(atr_stop_n),
                atr_m=float(atr_stop_m),
                fallback_position_risk=0.01,
            )
            portfolio_meta["monthly_risk_budget_gate"] = gate_stats
            portfolio_meta["monthly_risk_budget_atr_stop"] = {
                "atr_stop_mode": atm,
                "atr_stop_atr_basis": ab_raw,
                "atr_stop_reentry_mode": arm_raw,
                "atr_stop_window": int(w_atr),
                "atr_stop_n": float(atr_stop_n),
                "atr_stop_m": float(atr_stop_m),
                "fallback_position_risk": 0.01,
            }

        w_eff = (
            w_df.reindex(index=ret_mat.index, columns=ret_mat.columns)
            .fillna(0.0)
            .astype(float)
            .shift(1)
            .fillna(0.0)
        )
        atr_override = pd.Series(0.0, index=ret_mat.index, dtype=float)
        if atm_raw != "none" and atr_stop_by_asset:
            open_df = pd.DataFrame(
                {
                    str(c): exec_aligned[c]["Open"].astype(float)
                    for c in ret_mat.columns
                    if c in exec_aligned
                },
                index=common_idx,
            )
            close_df_exec = pd.DataFrame(
                {
                    str(c): exec_aligned[c]["Close"].astype(float)
                    for c in ret_mat.columns
                    if c in exec_aligned
                },
                index=common_idx,
            )
            w_eff, atr_override = _apply_intraday_stop_execution_portfolio(
                weights=w_eff,
                atr_stop_by_asset=atr_stop_by_asset,
                exec_price=str(exec_price),
                open_sig_df=open_df.reindex(
                    index=w_eff.index, columns=w_eff.columns
                ).astype(float),
                close_sig_df=close_df_exec.reindex(
                    index=w_eff.index, columns=w_eff.columns
                ).astype(float),
            )
        group_ret = (w_eff * ret_mat.astype(float)).sum(axis=1).astype(float).fillna(
            0.0
        ) + atr_override.reindex(ret_mat.index).fillna(0.0).astype(float)

    group_nav = (1.0 + group_ret.fillna(0.0)).cumprod()

    bench_close_df = pd.DataFrame(bench_price_by_symbol).sort_index()
    bench_close_df.index = _coerce_trading_index(bench_close_df.index)
    bench_close_df = bench_close_df.reindex(common_idx).ffill().bfill()
    bench_ret = _combine_group_returns(
        bench_close_df.pct_change(),
        dynamic_universe=bool(dynamic_universe),
    )
    bench_nav = (1.0 + bench_ret.fillna(0.0)).cumprod()

    align_idx = group_nav.index.union(bench_nav.index).sort_values()
    group_nav = group_nav.reindex(align_idx).ffill().fillna(1.0)
    bench_nav = bench_nav.reindex(align_idx).ffill().fillna(1.0)

    def _series_rows(series: pd.Series) -> list[dict]:
        return [
            {"date": pd.Timestamp(d).date().isoformat(), "value": float(v)}
            for d, v in series.items()
            if pd.notna(v)
        ]

    return {
        "ok": True,
        "meta": {
            "group_name": group.name,
            "start": start,
            "end": end,
            "dynamic_universe": bool(dynamic_universe),
            "exec_price": exec_price,
            "trend_strategy": str(ts_raw),
            "ma_type": str(mt_eff),
            "fast_ma": int(fast_ma),
            "slow_ma": int(slow_ma),
            "position_size_pct": float(position_size_pct),
            "cost_bps": float(cost_bps),
            "fee_side": fee_side,
            "slippage_type": slippage_type,
            "slippage_value": float(slippage_value),
            "slippage_side": slippage_side,
            "mode": "dynamic_union" if dynamic_universe else "static_intersection",
            "signal_execution_rule": f"signal_t_execute_t_plus_1_{exec_price}",
            "signal_lag_trading_days": 1,
            "benchmark_price_basis": ("open" if exec_price == "open" else "close"),
            "trend_series_policy": {
                "benchmark": "synthetic_hfq_continuous (889) when present; "
                "else none (fallback)",
                "signals": "synthetic_qfq_continuous (888) close when present; "
                "else none on execution series",
                "execution_and_returns": "synthetic_none_continuous (88) when "
                "present; else main contract none",
            },
            "backtest_mode": bm,
            "single_code": (
                str(single_code or "").strip().upper() if bm == "single" else None
            ),
            "position_sizing": (ps if bm == "portfolio" else None),
            "portfolio": portfolio_meta,
            "risk_budget_atr_window": int(risk_budget_atr_window),
            "risk_budget_pct": float(risk_budget_pct),
            "risk_budget_overcap_policy": rb_pol,
            "risk_budget_max_leverage_multiple": float(
                risk_budget_max_leverage_multiple
            ),
            "monthly_risk_budget_enabled": bool(monthly_risk_budget_enabled),
            "monthly_risk_budget_effective": bool(
                bm == "portfolio" and monthly_risk_budget_enabled
            ),
            "monthly_risk_budget_pct": float(monthly_risk_budget_pct),
            "monthly_risk_budget_include_new_trade_risk": bool(
                monthly_risk_budget_include_new_trade_risk
            ),
            "atr_stop_mode": str(atm_raw),
            "atr_stop_atr_basis": str(ab_raw),
            "atr_stop_reentry_mode": str(arm_raw),
            "atr_stop_window": int(atr_stop_window),
            "atr_stop_n": float(atr_stop_n),
            "atr_stop_m": float(atr_stop_m),
            "effective_symbols": int(len(nav_by_symbol)),
            "skipped": errors,
        },
        "series": {
            "strategy_nav": _series_rows(group_nav),
            "benchmark_nav": _series_rows(bench_nav),
        },
        "summary": {
            "strategy_total_return": float(group_nav.iloc[-1] - 1.0),
            "benchmark_total_return": float(bench_nav.iloc[-1] - 1.0),
            "excess_total_return": float(
                (group_nav.iloc[-1] / bench_nav.iloc[-1]) - 1.0
            ),
        },
        "symbols": symbol_stats,
    }
