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
from .bt_trend import _apply_monthly_risk_budget_gate
from .futures_trend_portfolio_weights import (
    atr_ewm_wilder,
    build_ma_panels,
    combine_weighted_returns,
    equal_weights_from_signals,
    risk_budget_weights,
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
        return out, out.copy(), detail

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
) -> pd.Series:
    if "SignalClose" in df.columns:
        close = df["SignalClose"].astype(float)
    else:
        close = df["Close"].astype(float)
    fast = close.rolling(window=int(fast_ma), min_periods=int(fast_ma)).mean()
    slow = close.rolling(window=int(slow_ma), min_periods=int(slow_ma)).mean()
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

    class TalibSmaTrend(Strategy):
        fast = 20
        slow = 60
        size_pct = 0.999999

        def init(self) -> None:
            sig = getattr(self.data, "SignalClose", None)
            close_arr = np.asarray(
                sig if sig is not None else self.data.Close,
                dtype=float,
            )
            sma_fn = getattr(talib, "SMA", None) if talib is not None else None
            if callable(sma_fn):
                self.fast_ma = self.I(sma_fn, close_arr, int(self.fast))
                self.slow_ma = self.I(sma_fn, close_arr, int(self.slow))
            else:
                # Fallback only when TA-Lib is unavailable.
                s = pd.Series(close_arr, dtype=float)
                self.fast_ma = self.I(
                    lambda x, n: (
                        pd.Series(x, dtype=float)
                        .rolling(n, min_periods=n)
                        .mean()
                        .to_numpy()
                    ),
                    s.to_numpy(),
                    int(self.fast),
                )
                self.slow_ma = self.I(
                    lambda x, n: (
                        pd.Series(x, dtype=float)
                        .rolling(n, min_periods=n)
                        .mean()
                        .to_numpy()
                    ),
                    s.to_numpy(),
                    int(self.slow),
                )

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
        TalibSmaTrend,
        cash=1_000_000.0,
        trade_on_close=(exec_price == "close"),
        commission=float(cost.commission_per_fill),
        spread=float(cost.spread_per_fill),
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = bt.run(fast=int(fast_ma), slow=int(slow_ma), size_pct=float(order_size))
    eq = stats.get("_equity_curve")
    if eq is None or "Equity" not in eq:
        ret = _run_vectorized_fallback(
            df, fast_ma=fast_ma, slow_ma=slow_ma, exec_price=exec_price
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
    fast_ma: int = 20,
    slow_ma: int = 60,
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
    atr_stop_window: int = 14,
    atr_stop_n: float = 2.0,
    atr_stop_m: float = 0.5,
) -> dict:
    codes = [str(c).strip().upper() for c in group.codes if str(c).strip()]
    if len(codes) == 0:
        return {"ok": False, "error": "empty_group", "meta": {"group_name": group.name}}
    if int(fast_ma) < 2 or int(slow_ma) <= int(fast_ma):
        return {"ok": False, "error": "invalid_ma_windows"}
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
        common_idx = (
            nav.index if common_idx is None else common_idx.intersection(nav.index)
        )
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
    if bm == "single":
        group_ret = ret_mat.iloc[:, 0].astype(float)
        portfolio_meta["allocation"] = "single_asset_full_notional"
    else:
        if ps == "equal":
            _, sig_df = build_ma_panels(
                exec_aligned,
                common_idx=common_idx,
                fast_ma=int(fast_ma),
                slow_ma=int(slow_ma),
            )
            w_df = equal_weights_from_signals(sig_df)
            portfolio_meta["position_sizing"] = "equal"
            portfolio_meta["note"] = (
                "equal weight among MA-long symbols each day; weights lagged one day "
                "into returns"
            )
        else:
            score_df, sig_df = build_ma_panels(
                exec_aligned,
                common_idx=common_idx,
                fast_ma=int(fast_ma),
                slow_ma=int(slow_ma),
            )
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
            atm = str(atr_stop_mode or "none").strip().lower()
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
                atr_basis=str(atr_stop_atr_basis or "latest"),
                atr_n=float(atr_stop_n),
                atr_m=float(atr_stop_m),
                fallback_position_risk=0.02,
            )
            portfolio_meta["monthly_risk_budget_gate"] = gate_stats
            portfolio_meta["monthly_risk_budget_atr_stop"] = {
                "atr_stop_mode": atm,
                "atr_stop_atr_basis": str(atr_stop_atr_basis or "latest"),
                "atr_stop_window": int(w_atr),
                "atr_stop_n": float(atr_stop_n),
                "atr_stop_m": float(atr_stop_m),
            }

        group_ret = combine_weighted_returns(ret_mat, w_df)

    group_nav = (1.0 + group_ret.fillna(0.0)).cumprod()

    bench_close_df = (
        pd.DataFrame(bench_price_by_symbol).sort_index().reindex(common_idx).ffill()
    )
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
            "atr_stop_mode": str(atr_stop_mode or "none"),
            "atr_stop_atr_basis": str(atr_stop_atr_basis or "latest"),
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
