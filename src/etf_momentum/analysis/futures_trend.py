from __future__ import annotations

# pylint: disable=broad-exception-caught,attribute-defined-outside-init

import datetime as dt
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..db.futures_repo import list_futures_prices
from ..db.futures_research_repo import FuturesGroupData

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
) -> pd.DataFrame:
    s_d = _parse_yyyymmdd(start)
    e_d = _parse_yyyymmdd(end)
    rows = list_futures_prices(
        db,
        code=code,
        adjust="none",
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
            close_arr = np.asarray(self.data.Close, dtype=float)
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

    nav_by_symbol: dict[str, pd.Series] = {}
    bench_price_by_symbol: dict[str, pd.Series] = {}
    symbol_stats: list[dict] = []
    errors: list[str] = []

    for code in codes:
        df = _load_futures_ohlcv(db, code=code, start=start, end=end)
        if len(df.index) < int(min_points):
            errors.append(f"{code}:points<{int(min_points)}")
            continue
        price_ref = float(df["Close"].median()) if len(df.index) else 0.0
        cost = _build_cost_profile(
            cost_bps=cost_bps,
            fee_side=fee_side,
            slippage_type=slippage_type,
            slippage_value=slippage_value,
            slippage_side=slippage_side,
            price_reference=price_ref,
        )
        nav, st = _run_symbol_backtest(
            df,
            fast_ma=int(fast_ma),
            slow_ma=int(slow_ma),
            exec_price=exec_price,
            position_size_pct=float(position_size_pct),
            cost=cost,
        )
        nav_by_symbol[code] = nav
        bench_col = "Open" if exec_price == "open" else "Close"
        bench_price_by_symbol[code] = df[bench_col].astype(float)
        symbol_stats.append(
            {
                "code": code,
                "points": int(len(df.index)),
                "start": str(df.index.min().date()),
                "end": str(df.index.max().date()),
                "ret_total": float(st.get("ret_total", 0.0)),
                "trades": int(st.get("trades", 0)),
                "win_rate": float(st.get("win_rate", 0.0)),
                "engine": str(st.get("engine", "unknown")),
                "commission_per_fill": float(cost.commission_per_fill),
                "spread_per_fill": float(cost.spread_per_fill),
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

    nav_df = pd.DataFrame(nav_by_symbol).sort_index()
    strat_ret = nav_df.pct_change()
    group_ret = _combine_group_returns(
        strat_ret, dynamic_universe=bool(dynamic_universe)
    )
    group_nav = (1.0 + group_ret).cumprod()

    bench_close_df = pd.DataFrame(bench_price_by_symbol).sort_index()
    bench_ret = _combine_group_returns(
        bench_close_df.pct_change(),
        dynamic_universe=bool(dynamic_universe),
    )
    bench_nav = (1.0 + bench_ret).cumprod()

    common_idx = group_nav.index.union(bench_nav.index).sort_values()
    group_nav = group_nav.reindex(common_idx).ffill().fillna(1.0)
    bench_nav = bench_nav.reindex(common_idx).ffill().fillna(1.0)

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
