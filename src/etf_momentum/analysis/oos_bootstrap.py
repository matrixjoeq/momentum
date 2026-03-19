"""
Out-of-sample bootstrap parameter optimisation (Carver-style).

Splits data into in-sample and out-of-sample by time; runs block-bootstrap
on in-sample returns to get robust parameter estimates, then evaluates once
on OOS. See "Systematic Trading" (Robert Carver), Ch. 3–4.

Used for: rotation strategies, trend strategies (single or portfolio).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .montecarlo import _circular_block_bootstrap_indices

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OosBootstrapConfig:
    """Configuration for out-of-sample bootstrap optimisation."""

    oos_ratio: float = 0.3
    """Fraction of full period reserved for out-of-sample (chronologically at the end)."""

    n_bootstrap: int = 100
    """Number of bootstrap resamples on in-sample data."""

    block_size: int = 21
    """Block length for circular block bootstrap (trading days; preserves time dependence)."""

    seed: Optional[int] = None
    """Random seed for reproducibility."""

    objective: str = "maximize"
    """Optimisation objective: 'maximize' (e.g. Sharpe) or 'minimize' (e.g. drawdown)."""

    objective_metric: str = "sharpe_ratio"
    """Key of the metric dict returned by backtest to use as objective."""


def split_in_sample_oos(
    dates: pd.DatetimeIndex,
    oos_ratio: float,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Split a datetime index into in-sample (earlier) and out-of-sample (later) by time.
    No shuffle; OOS is the last `oos_ratio` of the period.
    """
    if oos_ratio <= 0 or oos_ratio >= 1:
        raise ValueError("oos_ratio must be in (0, 1)")
    n = len(dates)
    if n < 2:
        raise ValueError("Need at least 2 dates to split")
    n_oos = max(1, int(round(n * oos_ratio)))
    n_in = n - n_oos
    in_sample = dates[:n_in]
    oos = dates[n_in:]
    return in_sample, oos


def block_bootstrap_returns(
    returns: pd.DataFrame,
    *,
    block_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Resample rows of `returns` using circular block bootstrap.
    Preserves time-series dependence within blocks. Returns same shape as input.
    """
    n = len(returns)
    if n == 0:
        return returns.copy()
    idx = _circular_block_bootstrap_indices(n, block_size=block_size, rng=rng)
    return returns.iloc[idx].reset_index(drop=True)


def returns_to_close(returns: pd.DataFrame, initial: Optional[pd.Series] = None) -> pd.DataFrame:
    """Build close prices from daily returns; first row is 1.0 (or initial) then cumprod(1+r)."""
    r = returns.fillna(0.0).astype(float)
    if initial is not None:
        r = r.reindex(columns=initial.index).fillna(0.0)
    close = (1 + r).cumprod()
    if initial is not None:
        close = close * (initial.reindex(close.columns).fillna(1.0))
    else:
        close.iloc[0] = 1.0
    return close.astype(float)


def _aggregate_params(
    list_of_params: List[Dict[str, Any]],
    param_grid: Dict[str, Sequence[Any]],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Aggregate bootstrap parameter draws: median for numeric, mode for categorical.
    Values are snapped to the nearest valid grid value when a grid is provided.
    """
    if not list_of_params:
        return {}
    keys = list_of_params[0].keys()
    out: Dict[str, Any] = {}
    for k in keys:
        values = [p.get(k) for p in list_of_params if k in p]
        if not values:
            continue
        grid = param_grid.get(k)
        if grid is not None and len(grid) > 0:
            # Categorical or discrete: use mode, then ensure in grid
            uniq, counts = np.unique(values, return_counts=True)
            mode_val = uniq[np.argmax(counts)].tolist() if hasattr(uniq[0], "tolist") else uniq[np.argmax(counts)]
            if mode_val in grid:
                out[k] = mode_val
            else:
                out[k] = min(grid, key=lambda x: (abs(x - mode_val) if isinstance(mode_val, (int, float)) else 1))
        else:
            numeric = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
            if numeric:
                out[k] = int(round(np.median(numeric))) if all(isinstance(v, int) for v in numeric) else float(np.median(numeric))
            else:
                uniq, counts = np.unique(values, return_counts=True)
                out[k] = uniq[np.argmax(counts)].tolist() if hasattr(uniq[0], "tolist") else uniq[np.argmax(counts)]
    return out


def _run_grid_search(
    close: pd.DataFrame,
    param_grid: Dict[str, Sequence[Any]],
    backtest_fn: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, float]],
    config: OosBootstrapConfig,
) -> Dict[str, Any]:
    """Run full grid over param_grid, return best params by objective metric."""
    from itertools import product

    keys = list(param_grid.keys())
    grids = [param_grid[k] for k in keys]
    best_metric: float = -np.inf if config.objective == "maximize" else np.inf
    best_params: Dict[str, Any] = {}

    for combo in product(*grids):
        params = dict(zip(keys, combo))
        try:
            metrics = backtest_fn(close, params)
        except Exception as e:  # noqa: BLE001
            logger.debug("Backtest failed for %s: %s", params, e)
            continue
        m = metrics.get(config.objective_metric)
        if m is None or not np.isfinite(m):
            continue
        if config.objective == "maximize" and m > best_metric:
            best_metric = m
            best_params = params.copy()
        elif config.objective == "minimize" and m < best_metric:
            best_metric = m
            best_params = params.copy()

    return best_params


def _synthetic_close_with_daily_index(returns_boot: pd.DataFrame) -> pd.DataFrame:
    """Build synthetic close from bootstrapped returns with a contiguous daily DatetimeIndex."""
    close = returns_to_close(returns_boot, initial=None)
    idx = pd.date_range(start="2000-01-01", periods=len(close), freq="B")
    close = close.set_axis(idx)
    return close


def run_rotation_oos_bootstrap(
    close: pd.DataFrame,
    universe: Any,
    param_grid: Dict[str, Sequence[Any]],
    *,
    cost_bps: float = 3.0,
    config: Optional[OosBootstrapConfig] = None,
) -> Dict[str, Any]:
    """
    Out-of-sample bootstrap parameter optimisation for rotation strategy.

    - Splits `close` into in-sample (earlier) and OOS (later) by time.
    - For each of n_bootstrap runs: block-bootstrap in-sample returns -> synthetic
      in-sample close -> grid search -> record best params.
    - Aggregates params (median/mode), then runs one backtest on real OOS data.

    Returns dict with: oos_metrics, chosen_params, bootstrap_params (list), in_sample_end, oos_start.
    """

    from etf_momentum.scripts.rotation_research_runner import (
        RotationStrategyConfig,
        backtest_strategy,
        calculate_metrics,
    )
    from etf_momentum.strategy.rotation_research_config import UniverseConfig

    cfg = config or OosBootstrapConfig()
    if isinstance(universe, UniverseConfig):
        universe_config = universe
    else:
        universe_config = UniverseConfig(name="Custom", codes=close.columns.tolist())

    dates = close.index.sort_values() if hasattr(close.index, "sort_values") else pd.DatetimeIndex(sorted(close.index))
    in_dates, oos_dates = split_in_sample_oos(dates, cfg.oos_ratio)
    close_in = close.loc[in_dates].dropna(how="all").ffill().bfill()
    close_oos = close.loc[oos_dates].reindex(columns=close_in.columns).ffill().bfill()
    if close_in.empty or close_oos.empty or len(close_in) < 30:
        return {
            "error": "Insufficient data after split",
            "in_sample_days": len(close_in),
            "oos_days": len(close_oos),
        }

    returns_in = close_in.pct_change().dropna(how="all").fillna(0.0)
    if returns_in.empty or len(returns_in) < 10:
        return {"error": "Insufficient in-sample returns", "in_sample_days": len(close_in)}

    rng = np.random.default_rng(cfg.seed)
    bootstrap_params: List[Dict[str, Any]] = []

    def backtest_fn(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, float]:
        c = RotationStrategyConfig(
            universe=universe_config,
            score_method=params.get("score_method", "sharpe_mom"),
            lookback_days=int(params.get("lookback_days", 90)),
            top_k=int(params.get("top_k", 2)),
            rebalance=str(params.get("rebalance", "weekly")),
            enable_trend_filter=bool(params.get("enable_trend_filter", True)),
        )
        result = backtest_strategy(df, c, cost_bps=cost_bps)
        return result.metrics

    for b in range(cfg.n_bootstrap):
        boot_idx = _circular_block_bootstrap_indices(
            len(returns_in), block_size=cfg.block_size, rng=rng
        )
        ret_boot = returns_in.iloc[boot_idx].reset_index(drop=True)
        close_boot = _synthetic_close_with_daily_index(ret_boot)
        best = _run_grid_search(close_boot, param_grid, backtest_fn, cfg)
        if best:
            bootstrap_params.append(best)

    if not bootstrap_params:
        return {
            "error": "No valid bootstrap runs",
            "in_sample_end": in_dates[-1].isoformat() if len(in_dates) else None,
            "oos_start": oos_dates[0].isoformat() if len(oos_dates) else None,
        }

    chosen = _aggregate_params(bootstrap_params, param_grid, rng)
    config_final = RotationStrategyConfig(
        universe=universe_config,
        score_method=chosen.get("score_method", "sharpe_mom"),
        lookback_days=int(chosen.get("lookback_days", 90)),
        top_k=int(chosen.get("top_k", 2)),
        rebalance=str(chosen.get("rebalance", "weekly")),
        enable_trend_filter=bool(chosen.get("enable_trend_filter", True)),
    )
    result_oos = backtest_strategy(close_oos, config_final, cost_bps=cost_bps)
    oos_metrics = result_oos.metrics

    return {
        "oos_metrics": oos_metrics,
        "chosen_params": chosen,
        "bootstrap_params": bootstrap_params,
        "in_sample_end": in_dates[-1].isoformat(),
        "oos_start": oos_dates[0].isoformat(),
        "n_bootstrap": cfg.n_bootstrap,
        "oos_ratio": cfg.oos_ratio,
    }


def _load_trend_in_sample_data(
    db: Any,
    codes: List[str],
    start_in: "datetime.date",
    end_in: "datetime.date",
    need_hist: int,
) -> Optional[Dict[str, Any]]:
    """Load close/high/low and returns for in-sample period for trend portfolio bootstrap."""
    import datetime as _dt

    from .baseline import load_close_prices, load_high_low_prices

    ext_start = start_in - _dt.timedelta(days=int(need_hist) * 2)
    close_qfq = load_close_prices(db, codes=codes, start=ext_start, end=end_in, adjust="qfq").sort_index()
    close_hfq = load_close_prices(db, codes=codes, start=ext_start, end=end_in, adjust="hfq").sort_index()
    high_qfq_df, low_qfq_df = load_high_low_prices(db, codes=codes, start=ext_start, end=end_in, adjust="qfq")
    if close_qfq.empty or close_hfq.empty:
        return None
    common = close_qfq.index.intersection(close_hfq.index)
    close_qfq = close_qfq.loc[common].reindex(columns=codes).ffill().bfill()
    close_hfq = close_hfq.loc[common].reindex(columns=codes).ffill().bfill()
    first_valid = close_qfq.first_valid_index()
    if first_valid is None:
        return None
    close_qfq = close_qfq.loc[first_valid:]
    close_hfq = close_hfq.loc[first_valid:]
    in_mask = (close_qfq.index.date >= start_in) & (close_qfq.index.date <= end_in)
    dates = close_qfq.index[in_mask]
    if len(dates) < 30:
        return None
    close_qfq = close_qfq.loc[dates].reindex(columns=codes).ffill().bfill()
    close_hfq = close_hfq.loc[dates].reindex(columns=codes).ffill().bfill()
    returns_in = close_qfq.pct_change().fillna(0.0).astype(float)
    if returns_in.empty or len(returns_in) < 20:
        return None
    high_qfq_df = high_qfq_df.sort_index().reindex(dates).reindex(columns=codes).ffill() if not high_qfq_df.empty else pd.DataFrame(index=dates, columns=codes)
    low_qfq_df = low_qfq_df.sort_index().reindex(dates).reindex(columns=codes).ffill() if not low_qfq_df.empty else pd.DataFrame(index=dates, columns=codes)
    return {
        "dates": dates,
        "close_qfq": close_qfq,
        "close_hfq": close_hfq,
        "high_qfq_df": high_qfq_df,
        "low_qfq_df": low_qfq_df,
        "returns_in": returns_in,
    }


def _trend_params_to_inputs(
    codes: List[str],
    start: "datetime.date",
    end: "datetime.date",
    strategy: str,
    params: Dict[str, Any],
    *,
    risk_free_rate: float = 0.025,
    cost_bps: float = 0.0,
    exec_price: str = "open",
    position_sizing: str = "equal",
    vol_window: int = 20,
    vol_target_ann: float = 0.20,
) -> Any:
    """Build TrendPortfolioInputs from strategy name and param dict (grid values)."""
    from .trend import TrendPortfolioInputs

    return TrendPortfolioInputs(
        codes=codes,
        start=start,
        end=end,
        risk_free_rate=risk_free_rate,
        cost_bps=cost_bps,
        exec_price=exec_price,
        strategy=strategy,
        position_sizing=position_sizing,
        vol_window=vol_window,
        vol_target_ann=vol_target_ann,
        sma_window=int(params.get("sma_window", 200)),
        fast_window=int(params.get("fast_window", 50)),
        slow_window=int(params.get("slow_window", 200)),
        ma_type=str(params.get("ma_type", "sma")),
        donchian_entry=int(params.get("donchian_entry", 20)),
        donchian_exit=int(params.get("donchian_exit", 10)),
        mom_lookback=int(params.get("mom_lookback", 252)),
        tsmom_entry_threshold=float(params.get("tsmom_entry_threshold", 0.0)),
        tsmom_exit_threshold=float(params.get("tsmom_exit_threshold", 0.0)),
        atr_stop_mode=str(params.get("atr_stop_mode", "none")),
        atr_stop_window=int(params.get("atr_stop_window", 14)),
        atr_stop_n=float(params.get("atr_stop_n", 2.0)),
        atr_stop_m=float(params.get("atr_stop_m", 0.5)),
        bias_ma_window=int(params.get("bias_ma_window", 20)),
        bias_entry=float(params.get("bias_entry", 2.0)),
        bias_hot=float(params.get("bias_hot", 10.0)),
        bias_cold=float(params.get("bias_cold", -2.0)),
        bias_pos_mode=str(params.get("bias_pos_mode", "binary")),
        macd_fast=int(params.get("macd_fast", 12)),
        macd_slow=int(params.get("macd_slow", 26)),
        macd_signal=int(params.get("macd_signal", 9)),
        macd_v_atr_window=int(params.get("macd_v_atr_window", 26)),
        macd_v_scale=float(params.get("macd_v_scale", 100.0)),
        hybrid_entry_n=int(params.get("hybrid_entry_n", 1)),
        hybrid_exit_m=int(params.get("hybrid_exit_m", 1)),
    )


def _default_trend_param_grid(strategy: str) -> Dict[str, Sequence[Any]]:
    """Default parameter grid per trend strategy for OOS bootstrap."""
    if strategy == "ma_filter":
        return {"sma_window": [100, 150, 200], "ma_type": ["sma", "ema"]}
    if strategy == "ma_cross":
        return {"fast_window": [50], "slow_window": [150, 200], "ma_type": ["sma", "ema"]}
    if strategy == "donchian":
        return {"donchian_entry": [20, 55], "donchian_exit": [10, 20]}
    if strategy == "tsmom":
        return {
            "mom_lookback": [126, 252],
            "tsmom_entry_threshold": [0.0],
            "tsmom_exit_threshold": [0.0],
        }
    if strategy == "linreg_slope":
        return {"sma_window": [100, 200]}
    if strategy == "bias":
        return {"bias_ma_window": [20], "bias_entry": [2.0], "bias_cold": [-2.0]}
    if strategy in ("macd_cross", "macd_zero_filter"):
        return {"macd_fast": [12], "macd_slow": [26], "macd_signal": [9]}
    if strategy == "macd_v":
        return {"macd_fast": [12], "macd_slow": [26], "macd_signal": [9], "macd_v_atr_window": [26]}
    if strategy == "hybrid_trend":
        return {"hybrid_entry_n": [1, 2, 3], "hybrid_exit_m": [1, 2, 3]}
    return {"sma_window": [100, 200], "ma_type": ["sma"]}


def run_trend_oos_bootstrap(
    db: Any,
    codes: List[str],
    start: "datetime.date",
    end: "datetime.date",
    param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    *,
    strategy: str = "ma_filter",
    config: Optional[OosBootstrapConfig] = None,
    risk_free_rate: float = 0.025,
    cost_bps: float = 0.0,
    exec_price: str = "open",
) -> Dict[str, Any]:
    """
    Out-of-sample bootstrap parameter optimisation for trend (portfolio) strategy.

    Splits [start, end] into in-sample and OOS. For each of n_bootstrap runs:
    block-bootstraps in-sample returns, builds synthetic OHLC, runs grid search
    on synthetic data, records best params. Aggregates params (median/mode), then
    evaluates once on real OOS data.
    """
    from itertools import product

    from .trend import compute_trend_portfolio_backtest

    cfg = config or OosBootstrapConfig()
    codes = list(dict.fromkeys([str(c).strip() for c in codes if str(c).strip()]))
    if not codes:
        return {"error": "codes is empty"}
    dates = pd.date_range(start=start, end=end, freq="B")
    if len(dates) < 60:
        return {"error": "Period too short for split"}
    in_dates, oos_dates = split_in_sample_oos(dates, cfg.oos_ratio)
    start_in = in_dates[0].date() if hasattr(in_dates[0], "date") else in_dates[0]
    end_in = in_dates[-1].date() if hasattr(in_dates[-1], "date") else in_dates[-1]
    start_oos = oos_dates[0].date() if hasattr(oos_dates[0], "date") else oos_dates[0]
    end_oos = oos_dates[-1].date() if hasattr(oos_dates[-1], "date") else oos_dates[-1]

    grid = param_grid if param_grid else _default_trend_param_grid(strategy)
    need_hist = 400
    data_in = _load_trend_in_sample_data(db, codes, start_in, end_in, need_hist)
    if data_in is None:
        return {
            "error": "Insufficient in-sample price data",
            "in_sample_end": end_in.isoformat(),
            "oos_start": start_oos.isoformat(),
        }
    returns_in = data_in["returns_in"]
    if len(returns_in) < 30:
        return {"error": "Insufficient in-sample returns", "in_sample_days": len(returns_in)}

    rng = np.random.default_rng(cfg.seed)
    bootstrap_params: List[Dict[str, Any]] = []
    keys = list(grid.keys())
    grids = [list(grid[k]) for k in keys]

    for b in range(cfg.n_bootstrap):
        boot_idx = _circular_block_bootstrap_indices(len(returns_in), block_size=cfg.block_size, rng=rng)
        ret_boot = returns_in.iloc[boot_idx].reset_index(drop=True)
        close_boot = returns_to_close(ret_boot, initial=None)
        idx = pd.date_range(start="2000-01-01", periods=len(close_boot), freq="B")
        close_boot = close_boot.set_axis(idx)
        high_boot = data_in["high_qfq_df"].iloc[boot_idx].reset_index(drop=True).set_axis(idx)
        low_boot = data_in["low_qfq_df"].iloc[boot_idx].reset_index(drop=True).set_axis(idx)
        ret_exec = ret_boot.set_axis(idx).reindex(idx).fillna(0.0).astype(float)
        ret_hfq = ret_exec.copy()
        syn_start = idx[0].date() if hasattr(idx[0], "date") else idx[0]
        syn_end = idx[-1].date() if hasattr(idx[-1], "date") else idx[-1]
        data_override = {
            "dates": idx,
            "close_qfq": close_boot.reindex(columns=codes).ffill().bfill(),
            "close_hfq": close_boot.reindex(columns=codes).ffill().bfill(),
            "high_qfq_df": high_boot.reindex(columns=codes).ffill().bfill(),
            "low_qfq_df": low_boot.reindex(columns=codes).ffill().bfill(),
            "ret_exec": ret_exec,
            "ret_hfq": ret_hfq,
        }
        best_metric: float = -np.inf if cfg.objective == "maximize" else np.inf
        best_params: Dict[str, Any] = {}
        for combo in product(*grids):
            params = dict(zip(keys, combo))
            inp = _trend_params_to_inputs(codes, syn_start, syn_end, strategy, params, risk_free_rate=risk_free_rate, cost_bps=cost_bps, exec_price=exec_price)
            try:
                out = compute_trend_portfolio_backtest(None, inp, data_override=data_override)
            except Exception as e:  # noqa: BLE001
                logger.debug("Trend bootstrap backtest failed for %s: %s", params, e)
                continue
            metrics = out.get("metrics") or {}
            m_dict = metrics.get("strategy") if isinstance(metrics.get("strategy"), dict) else {}
            m = m_dict.get(cfg.objective_metric)
            if m is None or not np.isfinite(m):
                continue
            if cfg.objective == "maximize" and m > best_metric:
                best_metric = m
                best_params = params.copy()
            elif cfg.objective == "minimize" and m < best_metric:
                best_metric = m
                best_params = params.copy()
        if best_params:
            bootstrap_params.append(best_params)

    if not bootstrap_params:
        return {
            "error": "No valid bootstrap runs",
            "in_sample_end": end_in.isoformat(),
            "oos_start": start_oos.isoformat(),
        }

    chosen = _aggregate_params(bootstrap_params, grid, rng)
    inp_oos = _trend_params_to_inputs(codes, start_oos, end_oos, strategy, chosen, risk_free_rate=risk_free_rate, cost_bps=cost_bps, exec_price=exec_price)
    out_oos = compute_trend_portfolio_backtest(db, inp_oos)
    oos_metrics = (out_oos.get("metrics") or {}).get("strategy")
    if not isinstance(oos_metrics, dict):
        oos_metrics = {}

    return {
        "oos_metrics": oos_metrics,
        "chosen_params": chosen,
        "bootstrap_params": bootstrap_params,
        "in_sample_end": end_in.isoformat(),
        "oos_start": start_oos.isoformat(),
        "n_bootstrap": cfg.n_bootstrap,
        "oos_ratio": cfg.oos_ratio,
        "strategy": strategy,
    }
