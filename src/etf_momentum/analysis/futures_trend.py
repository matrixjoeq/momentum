"""
Futures trend backtest (research) — price series policy
------------------------------------------------------
When **synthesized continuous** rows exist for a symbol (codes ``{root}88`` /
``{root}888`` / ``{root}889`` under the same ``Date`` calendar), we apply:

- **Main-contract roll (88 execution series):** group **strategy** P&L is driven by the
  **discrete-lot account** simulator: MTM on stitched continuous prices plus
  **round-turn** commission+slippage on ``dominant_contract_suffix`` changes (same
  integer lots on exit and re-entry). Legacy per-bar return-matrix roll adjustments are
  not applied once the lot engine is enabled.

- **Benchmark (buy-and-hold comparison):** synthetic **backward-adjusted**
  (**hfq**, e.g. ``{root}889``) OHLC; compare NAV uses the same open/close basis
  as ``exec_price`` on this hfq series.
- **Signals** (here: TA-Lib SMA crossover on “close”): synthetic **forward-adjusted**
  (**qfq**, e.g. ``{root}888``) **close**, aligned on trading dates with execution data.
- **Trade execution & strategy return compounding:** synthetic **no-adjust**
  (**none**, e.g. ``{root}88``) OHLCV passed into ``backtesting.py``.
- **Costs:** default **4 bps per fill** (``fee_side=one_way``, open and close each pay);
  default slippage **tick_multiple** with integer multiple **N** vs pool ``min_price_tick``
  (``tick_value_per_lot = contract_multiplier × min_price_tick`` for reporting).

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
from ..db.futures_repo import get_futures_pool_by_code, list_futures_prices
from ..db.futures_research_repo import FuturesGroupData
from .futures_lot_account import simulate_discrete_lot_portfolio
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
SlippageType = Literal["percent", "price_spread", "tick_multiple"]


@dataclass(frozen=True)
class CostProfile:
    commission_per_fill: float
    spread_per_fill: float
    commission_input_bps: float
    fee_side: FeeSide
    slippage_type: SlippageType
    slippage_input: float
    slippage_side: FeeSide
    contract_multiplier: float | None = None
    min_price_tick: float | None = None
    tick_value_per_lot: float | None = None
    slippage_tick_multiple: int | None = None
    slippage_price_reference: float | None = None


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
    # two_way: input is a round-trip total, split equally per fill (open and close).
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
            "Settle": [
                float(r.settle)
                if getattr(r, "settle", None) is not None
                and np.isfinite(float(r.settle))
                else np.nan
                for r in rows
            ],
            "Volume": [float(r.volume) if r.volume is not None else 0.0 for r in rows],
        },
        index=pd.to_datetime([r.trade_date for r in rows]),
    )
    if any(getattr(r, "dominant_contract_suffix", None) is not None for r in rows):
        suf_cells: list[str | None] = []
        for r in rows:
            raw = getattr(r, "dominant_contract_suffix", None)
            if raw is None:
                suf_cells.append(None)
            else:
                t = str(raw).strip()
                suf_cells.append(t if t else None)
        df["dominant_contract_suffix"] = suf_cells
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
        if "Settle" not in out.columns:
            out["Settle"] = out["Close"].astype(float)
        else:
            sc = pd.to_numeric(out["Settle"], errors="coerce")
            out["Settle"] = sc.fillna(out["Close"].astype(float)).astype(float)
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
    contract_multiplier: float | None = None,
    min_price_tick: float | None = None,
) -> CostProfile:
    cost_ratio = float(cost_bps) / 10000.0
    commission_per_fill = _normalize_per_fill(cost_ratio, fee_side)
    st = str(slippage_type or "percent").strip().lower()
    m_tick: int | None = None
    if st == "percent":
        spread_ratio = float(slippage_value)
    elif st == "price_spread":
        spread_ratio = (
            (float(slippage_value) / float(price_reference))
            if price_reference > 0
            else 0.0
        )
    elif st == "tick_multiple":
        if min_price_tick is None or (not np.isfinite(float(min_price_tick))):
            raise ValueError("min_price_tick required for tick_multiple slippage")
        tick = float(min_price_tick)
        if tick <= 0.0:
            raise ValueError("min_price_tick must be positive for tick_multiple")
        m = int(round(float(slippage_value)))
        if abs(float(slippage_value) - float(m)) > 1e-6:
            raise ValueError(
                "slippage_value must be a non-negative integer for tick_multiple"
            )
        m_tick = int(m)
        if m_tick < 0:
            raise ValueError("slippage tick multiple must be non-negative")
        # Adverse one-way return ≈ m * (price tick) / quote; matches cash / notional since
        # tick notional per lot uses multiplier * tick but (m*tick)/(price) = (m*mult*tick)/(price*mult).
        spread_ratio = (
            float(m_tick) * tick / float(price_reference)
            if price_reference > 0
            else 0.0
        )
    else:
        spread_ratio = 0.0
    spread_per_fill = _normalize_per_fill(spread_ratio, slippage_side)
    tick_val: float | None = None
    if (
        contract_multiplier is not None
        and min_price_tick is not None
        and np.isfinite(float(contract_multiplier))
        and np.isfinite(float(min_price_tick))
        and float(contract_multiplier) > 0.0
        and float(min_price_tick) > 0.0
    ):
        tick_val = float(contract_multiplier) * float(min_price_tick)
    return CostProfile(
        commission_per_fill=float(max(0.0, commission_per_fill)),
        spread_per_fill=float(max(0.0, spread_per_fill)),
        commission_input_bps=float(cost_bps),
        fee_side=fee_side,
        slippage_type=st
        if st in {"percent", "price_spread", "tick_multiple"}
        else "percent",
        slippage_input=float(slippage_value),
        slippage_side=slippage_side,
        contract_multiplier=(
            float(contract_multiplier)
            if contract_multiplier is not None
            and np.isfinite(float(contract_multiplier))
            else None
        ),
        min_price_tick=(
            float(min_price_tick)
            if min_price_tick is not None and np.isfinite(float(min_price_tick))
            else None
        ),
        tick_value_per_lot=tick_val,
        slippage_tick_multiple=m_tick,
        slippage_price_reference=float(price_reference)
        if np.isfinite(float(price_reference))
        else None,
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


def _apply_main_contract_roll_adjustments(
    ret_mat: pd.DataFrame,
    *,
    w_eff: pd.DataFrame,
    exec_aligned: dict[str, pd.DataFrame],
    cost_by_symbol: dict[str, CostProfile],
    min_weight: float = 1e-12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    On days where the stored 88 series marks a **main contract switch**, replace
    that symbol's strategy daily return with the **execution** close-to-close move
    on the stitched series (signed by **lagged weight**: long uses ``+r_cc``, short
    uses ``-r_cc``) and subtract **exit + re-entry** fill costs when the lagged
    portfolio weight is non-zero (``abs(w)``).

    The per-symbol Backtest NAV does not insert an extra trade when the MA signal
    is unchanged across the roll; this adjustment approximates mandatory roll
    turnover while keeping position size (weight) unchanged after the roll.
    """
    out = ret_mat.copy().astype(float)
    meta: dict[str, Any] = {
        "enabled": True,
        "per_symbol_roll_events": {},
        "total_roll_events": 0,
    }
    if out.empty or w_eff.empty:
        meta["enabled"] = False
        return out, meta

    idx = out.index
    w_eff = w_eff.reindex(idx).fillna(0.0).astype(float)

    for c in list(out.columns):
        code = str(c)
        ex = exec_aligned.get(code)
        cost = cost_by_symbol.get(code)
        if ex is None or cost is None:
            continue
        if "dominant_contract_suffix" not in ex.columns:
            continue
        suf = ex["dominant_contract_suffix"].reindex(idx)
        st = suf.astype(str).str.strip()
        roll_day = suf.notna() & st.ne("") & st.ne("nan") & st.ne("None")
        if not bool(roll_day.any()):
            continue

        close = pd.to_numeric(ex["Close"], errors="coerce").astype(float).reindex(idx)
        r_cc = close.pct_change().reindex(idx).astype(float)
        rt_fee = 2.0 * (
            float(max(0.0, cost.commission_per_fill))
            + float(max(0.0, cost.spread_per_fill))
        )

        col = str(c)
        if col not in w_eff.columns:
            continue
        wcol = w_eff[col].reindex(idx).fillna(0.0).astype(float)
        apply_m = roll_day.fillna(False) & (wcol.abs() > float(min_weight))
        if not bool(apply_m.any()):
            continue

        n = int(apply_m.sum())
        meta["per_symbol_roll_events"][str(c)] = n
        meta["total_roll_events"] += n
        # Long: ~ +close return; short: ~ −close return on the same stitched bar.
        adj_series = (np.sign(wcol) * r_cc - rt_fee).astype(float)
        out.loc[apply_m, code] = adj_series.loc[apply_m].astype(float).values

    if meta["total_roll_events"] == 0:
        meta["note"] = (
            "no_roll_events_in_sample_or_missing_dominant_contract_suffix_column"
        )
    return out, meta


def _run_vectorized_fallback(
    df: pd.DataFrame,
    *,
    fast_ma: int,
    slow_ma: int,
    exec_price: ExecPrice,
    ma_type: str = "sma",
    trade_direction: str = "long_only",
    cost: CostProfile | None = None,
) -> pd.Series:
    if "SignalClose" in df.columns:
        close = df["SignalClose"].astype(float)
    else:
        close = df["Close"].astype(float)
    mt = str(ma_type or "sma").strip().lower()
    fast = _moving_average(close, window=int(fast_ma), ma_type=mt)
    slow = _moving_average(close, window=int(slow_ma), ma_type=mt)
    td = str(trade_direction or "long_only").strip().lower()
    if td == "short_only":
        pos_raw = -(fast < slow).astype(float)
    elif td == "both":
        pos_raw = np.sign((fast - slow).astype(float)).fillna(0.0)
    else:
        pos_raw = (fast > slow).astype(float)
    pos_raw = pos_raw.fillna(0.0)
    # Strict anti-lookahead rule: signal on t executes on t+1.
    # Using close-close/open-open return legs implies first active return is t+2.
    if exec_price == "close":
        px_ret = close.pct_change().fillna(0.0)
        pos = pos_raw.shift(2).fillna(0.0)
        gross = pos * px_ret
    else:
        open_px = df["Open"].astype(float)
        px_ret = open_px.pct_change().fillna(0.0)
        pos = pos_raw.shift(2).fillna(0.0)
        gross = pos * px_ret
    if cost is not None:
        cfill = float(max(0.0, cost.commission_per_fill)) + float(
            max(0.0, cost.spread_per_fill)
        )
        if cfill > 0.0:
            flip = pos.diff().abs().fillna(0.0)
            gross = gross - flip * cfill
    return gross


def _run_symbol_backtest(
    df: pd.DataFrame,
    *,
    fast_ma: int,
    slow_ma: int,
    exec_price: ExecPrice,
    position_size_pct: float,
    cost: CostProfile,
    ma_type: str = "sma",
    trade_direction: str = "long_only",
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
            trade_direction=trade_direction,
            cost=cost,
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
        trade_direction = "long_only"

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
            td = str(self.trade_direction or "long_only").strip().lower()
            sz = float(self.size_pct)
            if td == "short_only":
                if f < s and not self.position:
                    self.sell(size=sz)
                elif f > s and self.position.is_short:
                    self.position.close()
                return
            if td == "both":
                if f > s:
                    if self.position.is_short:
                        self.position.close()
                    elif not self.position:
                        self.buy(size=sz)
                elif f < s:
                    if self.position.is_long:
                        self.position.close()
                    elif not self.position:
                        self.sell(size=sz)
                return
            if f > s and not self.position:
                self.buy(size=sz)
            elif f < s and self.position.is_long:
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
        trade_direction=str(trade_direction),
    )
    eq = stats.get("_equity_curve")
    if eq is None or "Equity" not in eq:
        ret = _run_vectorized_fallback(
            df,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            exec_price=exec_price,
            ma_type=ma_type,
            trade_direction=trade_direction,
            cost=cost,
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
    cost_bps: float = 4.0,
    fee_side: FeeSide = "one_way",
    slippage_type: SlippageType = "tick_multiple",
    slippage_value: float = 1.0,
    slippage_side: FeeSide = "one_way",
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
    trade_direction: str = "long_only",
    account_capital_wan: float = 500.0,
    backtest_margin_rate_pct: float = 15.0,
    reserve_margin_ratio: float = 0.5,
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
    slip_raw = str(slippage_type or "tick_multiple").strip().lower()
    if slip_raw not in {"percent", "price_spread", "tick_multiple"}:
        return {"ok": False, "error": "invalid_slippage_type"}
    if slip_raw == "tick_multiple":
        sv_chk = float(slippage_value)
        if (not np.isfinite(sv_chk)) or sv_chk < 0:
            return {"ok": False, "error": "invalid_slippage_tick_multiple"}
        if abs(sv_chk - round(sv_chk)) > 1e-6:
            return {
                "ok": False,
                "error": "slippage_tick_multiple_must_be_integer",
            }
    if position_size_pct <= 0 or position_size_pct > 1:
        return {"ok": False, "error": "invalid_position_size_pct"}
    if float(account_capital_wan) <= 0:
        return {"ok": False, "error": "invalid_account_capital_wan"}
    bmr = float(backtest_margin_rate_pct)
    if (not np.isfinite(bmr)) or bmr <= 0.0 or bmr > 100.0:
        return {"ok": False, "error": "invalid_backtest_margin_rate_pct"}
    rmr = float(reserve_margin_ratio)
    if (not np.isfinite(rmr)) or rmr < 0.0 or rmr >= 1.0:
        return {"ok": False, "error": "invalid_reserve_margin_ratio"}

    bm = str(backtest_mode or "portfolio").strip().lower()
    if bm not in {"portfolio", "single"}:
        return {"ok": False, "error": "invalid_backtest_mode"}
    ps = str(position_sizing or "equal").strip().lower()
    if bm == "portfolio" and ps not in {"equal", "risk_budget"}:
        return {"ok": False, "error": "invalid_position_sizing"}
    td_norm = str(trade_direction or "long_only").strip().lower()
    if td_norm not in {"long_only", "short_only", "both"}:
        return {"ok": False, "error": "invalid_trade_direction"}
    if ps == "risk_budget" and td_norm != "long_only":
        return {
            "ok": False,
            "error": "risk_budget_requires_trade_direction_long_only",
        }
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

    compat_notes: list[str] = []
    atm_eff: str = atm_raw
    monthly_eff: bool = bool(monthly_risk_budget_enabled)
    if td_norm != "long_only":
        if monthly_eff:
            compat_notes.append("monthly_risk_budget_disabled_non_long_trade_direction")
            monthly_eff = False

    codes_run = codes
    if bm == "single":
        sc = str(single_code or "").strip().upper()
        if not sc:
            return {"ok": False, "error": "missing_single_code"}
        if sc not in set(codes):
            return {"ok": False, "error": "single_code_not_in_group"}
        codes_run = [sc]

    bench_price_by_symbol: dict[str, pd.Series] = {}
    exec_by_code: dict[str, pd.DataFrame] = {}
    cost_by_symbol: dict[str, CostProfile] = {}
    mults_by_symbol: dict[str, float] = {}
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
        pool_row = get_futures_pool_by_code(db, code)
        cm = (
            getattr(pool_row, "contract_multiplier", None)
            if pool_row is not None
            else None
        )
        mpt = (
            getattr(pool_row, "min_price_tick", None) if pool_row is not None else None
        )
        cm_f = float(cm) if cm is not None and np.isfinite(float(cm)) else None
        mpt_f = float(mpt) if mpt is not None and np.isfinite(float(mpt)) else None
        if cm_f is None or (not np.isfinite(float(cm_f))) or float(cm_f) <= 0.0:
            errors.append(f"{code}:missing_contract_multiplier")
            continue
        if slip_raw == "tick_multiple" and (
            mpt_f is None or (not np.isfinite(float(mpt_f))) or float(mpt_f) <= 0.0
        ):
            errors.append(f"{code}:missing_min_price_tick_in_pool")
            continue
        try:
            cost = _build_cost_profile(
                cost_bps=cost_bps,
                fee_side=fee_side,
                slippage_type=slip_raw,  # type: ignore[arg-type]
                slippage_value=slippage_value,
                slippage_side=slippage_side,
                price_reference=price_ref,
                contract_multiplier=cm_f,
                min_price_tick=mpt_f,
            )
        except ValueError as ex:
            errors.append(f"{code}:cost_profile:{ex}")
            continue
        cost_by_symbol[code] = cost
        mults_by_symbol[code] = float(cm_f)
        exec_by_code[code] = df_exec.copy()
        bench_col = "Open" if exec_price == "open" else "Close"
        bench_price_by_symbol[code] = df_bench[bench_col].astype(float)
        symbol_stats.append(
            {
                "code": code,
                "points": int(len(df_exec.index)),
                "start": str(df_exec.index.min().date()),
                "end": str(df_exec.index.max().date()),
                "ret_total": None,
                "trades": None,
                "win_rate": None,
                "engine": "lot_account",
                "commission_per_fill": float(cost.commission_per_fill),
                "spread_per_fill": float(cost.spread_per_fill),
                "contract_multiplier": cost.contract_multiplier,
                "min_price_tick": cost.min_price_tick,
                "tick_value_per_lot": cost.tick_value_per_lot,
                "slippage_tick_multiple": cost.slippage_tick_multiple,
                "trend_resolution": trend_detail.get("trend_resolution"),
                "trend_execution_symbol": trend_detail.get("execution_symbol"),
                "trend_signal_adjust": trend_detail.get("signal_adjust"),
                "trend_benchmark_adjust": trend_detail.get("benchmark_adjust"),
            }
        )

    if not exec_by_code:
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
    for code in codes_run:
        if code not in exec_by_code:
            continue
        ix = _coerce_trading_index(exec_by_code[code].index)
        common_idx = ix if common_idx is None else common_idx.intersection(ix)
    if common_idx is None or len(common_idx) < int(min_points):
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
            trade_direction=td_norm,
        )
        if atm_eff != "none":
            ex = exec_aligned.get(code)
            if ex is not None:
                out_s, st = _apply_atr_stop(
                    sig_df[code].astype(float).reindex(common_idx).fillna(0.0),
                    open_=ex["Open"].reindex(common_idx).astype(float),
                    close=ex["Close"].reindex(common_idx).astype(float),
                    high=ex["High"].reindex(common_idx).astype(float),
                    low=ex["Low"].reindex(common_idx).astype(float),
                    mode=atm_eff,
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
            "applied": atm_eff != "none",
            "same_day_stop": True,
            "per_symbol_stats_keys": sorted(atr_stop_by_asset.keys()),
        }
        portfolio_meta["universal_atr_stop_applied_in_engine"] = bool(atm_eff != "none")

        _cols = list(w_df.columns)
        w_eff = (
            w_df.reindex(index=common_idx, columns=_cols)
            .fillna(0.0)
            .astype(float)
            .shift(1)
            .fillna(0.0)
        )
        atr_override = pd.Series(0.0, index=common_idx, dtype=float)
        if atm_eff != "none" and atr_stop_by_asset:
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
    else:
        score_df, sig_df = build_ma_panels(
            exec_aligned,
            common_idx=common_idx,
            fast_ma=int(fast_ma),
            slow_ma=int(slow_ma),
            ma_type=mt_eff,
            trade_direction=td_norm,
        )
        if atm_eff != "none":
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
                    mode=atm_eff,
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
                "equal |weight| among symbols with a non-flat MA regime that day "
                "(sign follows trade_direction); weights lagged one day into returns"
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
            "applied": atm_eff != "none",
            "same_day_stop": True,
            "per_symbol_stats_keys": sorted(atr_stop_by_asset.keys()),
        }
        portfolio_meta["universal_atr_stop_applied_in_engine"] = bool(atm_eff != "none")

        if bm == "portfolio" and monthly_eff:
            cc = [str(c) for c in w_df.columns]
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
            atm = atm_eff
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

        _cols_pf = list(w_df.columns)
        w_eff = (
            w_df.reindex(index=common_idx, columns=_cols_pf)
            .fillna(0.0)
            .astype(float)
            .shift(1)
            .fillna(0.0)
        )
        atr_override = pd.Series(0.0, index=common_idx, dtype=float)
        if atm_eff != "none" and atr_stop_by_asset:
            open_df = pd.DataFrame(
                {
                    str(c): exec_aligned[c]["Open"].astype(float)
                    for c in _cols_pf
                    if c in exec_aligned
                },
                index=common_idx,
            )
            close_df_exec = pd.DataFrame(
                {
                    str(c): exec_aligned[c]["Close"].astype(float)
                    for c in _cols_pf
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

    codes_sorted = sorted(exec_by_code.keys())
    ps_sim = ps if bm == "portfolio" else "equal"
    equity_ser, lot_meta = simulate_discrete_lot_portfolio(
        common_idx=common_idx,
        exec_by_code=exec_aligned,
        w_eff=w_eff,
        cost_by_symbol=cost_by_symbol,
        mults=mults_by_symbol,
        margin_rate_frac=float(backtest_margin_rate_pct) / 100.0,
        reserve_ratio=float(reserve_margin_ratio),
        initial_equity_cny=float(account_capital_wan)
        * 10000.0
        * float(position_size_pct),
        exec_price=str(exec_price),
        position_sizing=str(ps_sim).strip().lower(),
        codes_sorted=codes_sorted,
    )
    group_ret = equity_ser.pct_change().fillna(0.0).astype(float)
    atr_adj = atr_override.reindex(group_ret.index).fillna(0.0).astype(float)
    group_ret = (group_ret + atr_adj).astype(float)
    portfolio_meta["lot_account"] = lot_meta
    portfolio_meta["main_contract_roll"] = {
        "mode": "embedded_in_lot_engine",
        "notes": (
            "Roll fees on dominant_contract_suffix changes and MTM on stitched "
            "continuous prices; legacy ret_mat roll adjustment not applied."
        ),
    }

    portfolio_meta.setdefault("trade_direction", td_norm)
    if compat_notes:
        portfolio_meta["trade_direction_compat"] = compat_notes

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
            "fee_side_semantics": (
                "one_way: cost_bps applies to each fill (open and close each pay full bps); "
                "two_way: cost_bps is round-trip total split equally per fill"
            ),
            "slippage_type": slip_raw,
            "slippage_value": float(slippage_value),
            "slippage_side": slippage_side,
            "slippage_side_semantics": (
                "one_way: full spread ratio per fill; "
                "two_way: round-trip spread total split per fill"
            ),
            "tick_slippage_semantics": (
                "slippage_value is integer N; per-fill adverse return ≈ N * min_price_tick / price "
                "(min_price_tick from futures_pool; tick cash value per lot = contract_multiplier * "
                "min_price_tick for reporting)"
            ),
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
            "monthly_risk_budget_requested": bool(monthly_risk_budget_enabled),
            "monthly_risk_budget_enabled": bool(monthly_eff),
            "monthly_risk_budget_effective": bool(bm == "portfolio" and monthly_eff),
            "monthly_risk_budget_pct": float(monthly_risk_budget_pct),
            "monthly_risk_budget_include_new_trade_risk": bool(
                monthly_risk_budget_include_new_trade_risk
            ),
            "trade_direction": td_norm,
            "trade_direction_compat": compat_notes,
            "atr_stop_mode_requested": str(atm_raw),
            "atr_stop_mode": str(atm_eff),
            "atr_stop_atr_basis": str(ab_raw),
            "atr_stop_reentry_mode": str(arm_raw),
            "atr_stop_window": int(atr_stop_window),
            "atr_stop_n": float(atr_stop_n),
            "atr_stop_m": float(atr_stop_m),
            "effective_symbols": int(len(exec_by_code)),
            "account_capital_wan": float(account_capital_wan),
            "backtest_margin_rate_pct": float(backtest_margin_rate_pct),
            "reserve_margin_ratio": float(reserve_margin_ratio),
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
