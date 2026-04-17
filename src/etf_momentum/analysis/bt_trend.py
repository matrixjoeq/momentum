from __future__ import annotations

import datetime as dt
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .baseline import (
    _compute_return_risk_contributions,
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
    load_high_low_prices,
    load_ohlc_prices,
)
from .event_study import compute_event_study, entry_dates_from_exposure
from .market_regime import build_market_regime_report
from .r_multiple import build_trade_mfe_r_distribution, enrich_trades_with_r_metrics
from .trend import (
    TREND_STRATEGY_EXECUTION_DESCRIPTIONS,
    TrendInputs,
    TrendPortfolioInputs,
    _apply_er_entry_filter,
    _apply_er_exit_filter,
    _apply_impulse_entry_filter,
    _apply_atr_stop,
    _apply_bias_v_take_profit,
    _attach_entry_condition_bins_to_trades,
    _bucketize_er_series,
    _bucketize_impulse_series,
    _bucketize_momentum_series,
    _bucketize_vol_ratio_series,
    _build_entry_condition_stats,
    _apply_intraday_stop_execution_portfolio,
    _apply_intraday_stop_execution_single,
    _apply_monthly_risk_budget_gate,
    _apply_r_multiple_take_profit,
    _atr_from_hlc,
    _compute_impulse_state,
    _efficiency_ratio,
    _ema,
    _extract_atr_plan_stops_from_trace,
    _latest_entry_exec_price_with_slippage,
    _macd_core,
    _moving_average,
    _normalize_r_take_profit_tiers,
    _pos_from_band,
    _pos_from_donchian,
    _pos_from_random_entry_hold,
    _pos_from_tsmom,
    _rolling_pack,
    _risk_budget_dynamic_weights,
    _reduce_active_codes_by_group,
    _trade_returns_from_weight_df,
    _trade_returns_from_weight_series,
    _trade_stats_from_returns,
    _rolling_linreg_slope,
    _stable_code_seed,
)
from .execution_timing import corporate_action_mask, slippage_return_from_turnover

Session = Any

_SUPPORTED_STRATEGIES = {
    "ma_filter",
    "ma_cross",
    "donchian",
    "tsmom",
    "linreg_slope",
    "bias",
    "macd_cross",
    "macd_zero_filter",
    "macd_v",
    "random_entry",
}


def _validate_bt_single_inputs(inp: TrendInputs) -> None:
    strat = str(inp.strategy or "ma_filter").strip().lower()
    if strat not in _SUPPORTED_STRATEGIES:
        raise ValueError(f"invalid strategy={inp.strategy}")
    ma_type = str(getattr(inp, "ma_type", "sma") or "sma").strip().lower()
    if ma_type not in {"sma", "ema", "kama"}:
        raise ValueError("ma_type must be one of: sma|ema|kama")
    kama_fast_window = int(getattr(inp, "kama_fast_window", 2) or 2)
    kama_slow_window = int(getattr(inp, "kama_slow_window", 30) or 30)
    if kama_fast_window >= kama_slow_window:
        raise ValueError("kama_fast_window must be < kama_slow_window")
    if strat == "ma_cross" and ma_type == "kama":
        raise ValueError("ma_type=kama is only supported for ma_filter")
    ps = str(getattr(inp, "position_sizing", "equal") or "equal").strip().lower()
    if ps not in {"equal", "vol_target", "fixed_ratio", "risk_budget"}:
        raise ValueError("position_sizing must be equal|vol_target|fixed_ratio|risk_budget")
    risk_budget_pct = float(getattr(inp, "risk_budget_pct", 0.01) or 0.01)
    if (not np.isfinite(risk_budget_pct)) or risk_budget_pct < 0.001 or risk_budget_pct > 0.02:
        raise ValueError("risk_budget_pct must be in [0.001, 0.02]")
    if bool(getattr(inp, "vol_regime_risk_mgmt_enabled", False)):
        vt_expand = float(getattr(inp, "vol_ratio_expand_threshold", 1.45) or 1.45)
        vt_contract = float(getattr(inp, "vol_ratio_contract_threshold", 0.65) or 0.65)
        vt_normal = float(getattr(inp, "vol_ratio_normal_threshold", 1.05) or 1.05)
        if vt_expand <= vt_normal:
            raise ValueError("vol_ratio_expand_threshold must be > vol_ratio_normal_threshold")
        if vt_contract >= vt_normal:
            raise ValueError("vol_ratio_contract_threshold must be < vol_ratio_normal_threshold")


def _build_meta_params(inp: TrendInputs | TrendPortfolioInputs) -> dict[str, Any]:
    out = {
        "exec_price": str(getattr(inp, "exec_price", "open") or "open"),
        "cost_bps": float(getattr(inp, "cost_bps", 0.0)),
        "slippage_rate": float(getattr(inp, "slippage_rate", 0.0)),
        "position_sizing": str(getattr(inp, "position_sizing", "equal") or "equal"),
        "sma_window": int(getattr(inp, "sma_window", 20) or 20),
        "fast_window": int(getattr(inp, "fast_window", 5) or 5),
        "slow_window": int(getattr(inp, "slow_window", 20) or 20),
        "donchian_entry": int(getattr(inp, "donchian_entry", 20) or 20),
        "donchian_exit": int(getattr(inp, "donchian_exit", 10) or 10),
        "mom_lookback": int(getattr(inp, "mom_lookback", 252) or 252),
        "tsmom_entry_threshold": float(getattr(inp, "tsmom_entry_threshold", 0.0) or 0.0),
        "tsmom_exit_threshold": float(getattr(inp, "tsmom_exit_threshold", 0.0) or 0.0),
        "bias_ma_window": int(getattr(inp, "bias_ma_window", 20) or 20),
        "bias_entry": float(getattr(inp, "bias_entry", 2.0) or 2.0),
        "bias_hot": float(getattr(inp, "bias_hot", 5.0) or 5.0),
        "bias_cold": float(getattr(inp, "bias_cold", -2.0) or -2.0),
        "bias_pos_mode": str(getattr(inp, "bias_pos_mode", "binary") or "binary"),
        "macd_fast": int(getattr(inp, "macd_fast", 12) or 12),
        "macd_slow": int(getattr(inp, "macd_slow", 26) or 26),
        "macd_signal": int(getattr(inp, "macd_signal", 9) or 9),
        "macd_v_atr_window": int(getattr(inp, "macd_v_atr_window", 14) or 14),
        "macd_v_scale": float(getattr(inp, "macd_v_scale", 1.0) or 1.0),
        "random_hold_days": int(getattr(inp, "random_hold_days", 20) or 20),
        "random_seed": (
            None
            if getattr(inp, "random_seed", 42) is None
            else int(getattr(inp, "random_seed", 42))
        ),
        "vol_window": int(getattr(inp, "vol_window", 20) or 20),
        "vol_target_ann": float(getattr(inp, "vol_target_ann", 0.20) or 0.20),
        "fixed_pos_ratio": float(getattr(inp, "fixed_pos_ratio", 0.04) or 0.04),
        "fixed_max_holdings": int(getattr(inp, "fixed_max_holdings", 10) or 10),
        "fixed_overcap_policy": str(getattr(inp, "fixed_overcap_policy", "extend") or "extend"),
        "quick_mode": bool(getattr(inp, "quick_mode", False)),
        "risk_budget_atr_window": int(getattr(inp, "risk_budget_atr_window", 20) or 20),
        "risk_budget_pct": float(getattr(inp, "risk_budget_pct", 0.01) or 0.01),
        "risk_budget_overcap_policy": str(getattr(inp, "risk_budget_overcap_policy", "scale") or "scale"),
        "risk_budget_max_leverage_multiple": float(getattr(inp, "risk_budget_max_leverage_multiple", 2.0) or 2.0),
        "vol_regime_risk_mgmt_enabled": bool(getattr(inp, "vol_regime_risk_mgmt_enabled", False)),
        "vol_ratio_fast_atr_window": int(getattr(inp, "vol_ratio_fast_atr_window", 5) or 5),
        "vol_ratio_slow_atr_window": int(getattr(inp, "vol_ratio_slow_atr_window", 50) or 50),
        "vol_ratio_expand_threshold": float(getattr(inp, "vol_ratio_expand_threshold", 1.45) or 1.45),
        "vol_ratio_contract_threshold": float(getattr(inp, "vol_ratio_contract_threshold", 0.65) or 0.65),
        "vol_ratio_normal_threshold": float(getattr(inp, "vol_ratio_normal_threshold", 1.05) or 1.05),
        "atr_stop_mode": str(getattr(inp, "atr_stop_mode", "none") or "none"),
        "atr_stop_atr_basis": str(getattr(inp, "atr_stop_atr_basis", "latest") or "latest"),
        "atr_stop_reentry_mode": str(getattr(inp, "atr_stop_reentry_mode", "reenter") or "reenter"),
        "atr_stop_window": int(getattr(inp, "atr_stop_window", 14) or 14),
        "atr_stop_n": float(getattr(inp, "atr_stop_n", 2.0) or 2.0),
        "atr_stop_m": float(getattr(inp, "atr_stop_m", 0.5) or 0.5),
        "r_take_profit_enabled": bool(getattr(inp, "r_take_profit_enabled", False)),
        "r_take_profit_reentry_mode": str(getattr(inp, "r_take_profit_reentry_mode", "reenter") or "reenter"),
        "bias_v_take_profit_enabled": bool(getattr(inp, "bias_v_take_profit_enabled", False)),
        "bias_v_take_profit_reentry_mode": str(getattr(inp, "bias_v_take_profit_reentry_mode", "reenter") or "reenter"),
        "bias_v_ma_window": int(getattr(inp, "bias_v_ma_window", 20) or 20),
        "bias_v_atr_window": int(getattr(inp, "bias_v_atr_window", 20) or 20),
        "bias_v_take_profit_threshold": float(getattr(inp, "bias_v_take_profit_threshold", 5.0) or 5.0),
        "monthly_risk_budget_enabled": bool(getattr(inp, "monthly_risk_budget_enabled", False)),
        "monthly_risk_budget_pct": float(getattr(inp, "monthly_risk_budget_pct", 0.06) or 0.06),
        "monthly_risk_budget_include_new_trade_risk": bool(getattr(inp, "monthly_risk_budget_include_new_trade_risk", False)),
        "er_filter": bool(getattr(inp, "er_filter", False)),
        "er_window": int(getattr(inp, "er_window", 10) or 10),
        "er_threshold": float(getattr(inp, "er_threshold", 0.30) or 0.30),
        "er_exit_filter": bool(getattr(inp, "er_exit_filter", False)),
        "er_exit_window": int(getattr(inp, "er_exit_window", 10) or 10),
        "er_exit_threshold": float(getattr(inp, "er_exit_threshold", 0.88) or 0.88),
        "impulse_entry_filter": bool(getattr(inp, "impulse_entry_filter", False)),
        "impulse_allow_bull": bool(getattr(inp, "impulse_allow_bull", True)),
        "impulse_allow_bear": bool(getattr(inp, "impulse_allow_bear", False)),
        "impulse_allow_neutral": bool(getattr(inp, "impulse_allow_neutral", False)),
        "ma_type": str(getattr(inp, "ma_type", "sma") or "sma"),
        "kama_er_window": int(getattr(inp, "kama_er_window", 10) or 10),
        "kama_fast_window": int(getattr(inp, "kama_fast_window", 2) or 2),
        "kama_slow_window": int(getattr(inp, "kama_slow_window", 30) or 30),
        "kama_std_window": int(getattr(inp, "kama_std_window", 20) or 20),
        "kama_std_coef": float(getattr(inp, "kama_std_coef", 1.0) or 1.0),
        "r_take_profit_tiers": _normalize_r_take_profit_tiers(getattr(inp, "r_take_profit_tiers", None)),
        "risk_free_rate": float(getattr(inp, "risk_free_rate", 0.0) or 0.0),
        "dynamic_universe": bool(getattr(inp, "dynamic_universe", False)),
        "selection_mode": str(getattr(inp, "selection_mode", "all_active_candidates") or "all_active_candidates"),
        "group_enforce": bool(getattr(inp, "group_enforce", False)),
        "group_pick_policy": str(getattr(inp, "group_pick_policy", "highest_sharpe") or "highest_sharpe"),
        "group_max_holdings": int(getattr(inp, "group_max_holdings", 4) or 4),
        "asset_groups": dict(getattr(inp, "asset_groups", {}) or {}),
    }
    return out


def _risk_budget_frozen_weight(
    signal: pd.Series,
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr_window: int,
    risk_budget_pct: float,
) -> pd.Series:
    sig = pd.to_numeric(signal, errors="coerce").astype(float).fillna(0.0).clip(lower=0.0)
    c = pd.to_numeric(close, errors="coerce").astype(float).reindex(sig.index).ffill()
    h = pd.to_numeric(high, errors="coerce").astype(float).reindex(sig.index).ffill().combine_first(c)
    l = pd.to_numeric(low, errors="coerce").astype(float).reindex(sig.index).ffill().combine_first(c)
    atr = _atr_from_hlc(h, l, c, window=max(2, int(atr_window)))
    out = np.zeros(len(sig), dtype=float)
    in_pos = False
    frozen = 0.0
    rb = float(risk_budget_pct)
    for i, d in enumerate(sig.index):
        s = float(sig.loc[d]) if np.isfinite(float(sig.loc[d])) else 0.0
        if s <= 0.0:
            in_pos = False
            frozen = 0.0
            out[i] = 0.0
            continue
        if not in_pos:
            a = float(atr.loc[d]) if np.isfinite(float(atr.loc[d])) else np.nan
            px = float(c.loc[d]) if np.isfinite(float(c.loc[d])) else np.nan
            if np.isfinite(a) and a > 0.0 and np.isfinite(px) and px > 0.0:
                frozen = max(0.0, float(rb * px / a))
            else:
                frozen = 0.0
            in_pos = True
        out[i] = float(frozen)
    return pd.Series(out, index=sig.index, dtype=float)


def _as_nav(ret: pd.Series) -> pd.Series:
    s = pd.to_numeric(ret, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    nav = (1.0 + s).cumprod().astype(float)
    if len(nav) > 0:
        nav.iloc[0] = 1.0
    return nav


def _period_returns(nav: pd.Series, rule: str) -> list[dict[str, Any]]:
    if nav.empty:
        return []
    p = pd.to_numeric(nav, errors="coerce").astype(float).dropna()
    if p.empty:
        return []
    grp = p.resample(rule).last().dropna()
    if grp.empty:
        return []
    ret = grp.pct_change().dropna()
    out: list[dict[str, Any]] = []
    for d, v in ret.items():
        ds = pd.Timestamp(d).strftime("%Y-%m-%d")
        rv = float(v)
        out.append(
            {
                "date": ds,
                "strategy_return": rv,
                # Legacy-compatible aliases.
                "period_end": ds,
                "return": rv,
            }
        )
    return out


def _metrics_from_ret(ret: pd.Series, rf: float) -> dict[str, float]:
    s = pd.to_numeric(ret, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    nav = (1.0 + s).cumprod().astype(float)
    if len(nav) > 0:
        nav.iloc[0] = 1.0
    ann_ret = float(_annualized_return(nav))
    ulcer = float(_ulcer_index(nav))
    mdd_dur = float(_max_drawdown_duration_days(nav))
    out = {
        "cumulative_return": float((1.0 + s).prod() - 1.0),
        "annualized_return": ann_ret,
        "annualized_volatility": float(_annualized_vol(s)),
        "max_drawdown": float(_max_drawdown(nav)),
        "max_drawdown_duration_days": mdd_dur,
        "max_drawdown_recovery_days": int(mdd_dur),
        "sharpe_ratio": float(_sharpe(s, rf=float(rf))),
        "sortino_ratio": float(_sortino(s, rf=float(rf))),
        "ulcer_index": ulcer,
    }
    out["ulcer_performance_index"] = float((ann_ret - float(rf)) / (ulcer / 100.0)) if ulcer > 0.0 else float("nan")
    return out


def _normalize_trace_rows(rows: Any, keys: list[str]) -> list[dict[str, Any]]:
    src = list(rows or [])
    if not src:
        return [{k: None for k in keys}]
    out: list[dict[str, Any]] = []
    for r in src:
        rr = dict(r or {})
        for k in keys:
            rr.setdefault(k, None)
        out.append(rr)
    return out


def _build_signal_position(
    inp: TrendInputs,
    *,
    signal_close: pd.Series,
    signal_high: pd.Series,
    signal_low: pd.Series,
    code: str,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    strat = str(inp.strategy or "ma_filter").strip().lower()
    ma_type = str(getattr(inp, "ma_type", "sma") or "sma").strip().lower()
    if strat not in _SUPPORTED_STRATEGIES:
        raise ValueError(f"invalid strategy={inp.strategy}")
    if ma_type not in {"sma", "ema", "kama"}:
        raise ValueError("ma_type must be one of: sma|ema|kama")

    px = pd.to_numeric(signal_close, errors="coerce").astype(float)
    raw_pos: pd.Series
    score: pd.Series
    debug: dict[str, Any] = {"strategy": strat}

    if strat == "ma_filter":
        ma = _moving_average(
            px,
            window=int(inp.sma_window),
            ma_type=ma_type,
            kama_er_window=int(getattr(inp, "kama_er_window", 10)),
            kama_fast_window=int(getattr(inp, "kama_fast_window", 2)),
            kama_slow_window=int(getattr(inp, "kama_slow_window", 30)),
        )
        if ma_type == "kama":
            kstd = (
                ma.astype(float)
                .rolling(window=int(getattr(inp, "kama_std_window", 20)), min_periods=max(2, int(getattr(inp, "kama_std_window", 20)) // 2))
                .std(ddof=0)
                .fillna(0.0)
            )
            raw_pos = _pos_from_band(px, ma, band=float(getattr(inp, "kama_std_coef", 1.0)) * kstd).astype(float)
        else:
            raw_pos = (px > ma).astype(float).fillna(0.0)
        score = (px / ma - 1.0).replace([np.inf, -np.inf], np.nan).astype(float)
    elif strat == "ma_cross":
        fast = _moving_average(px, window=int(inp.fast_window), ma_type=ma_type)
        slow = _moving_average(px, window=int(inp.slow_window), ma_type=ma_type)
        raw_pos = (fast > slow).astype(float).fillna(0.0)
        score = (fast / slow - 1.0).replace([np.inf, -np.inf], np.nan).astype(float)
    elif strat == "donchian":
        raw_pos = _pos_from_donchian(px, entry=int(inp.donchian_entry), exit_=int(inp.donchian_exit)).astype(float)
        hi = px.shift(1).rolling(window=max(2, int(inp.donchian_entry)), min_periods=max(2, int(inp.donchian_entry))).max()
        score = (px / hi - 1.0).replace([np.inf, -np.inf], np.nan).astype(float)
    elif strat == "linreg_slope":
        n = int(inp.sma_window)
        y = np.log(px.clip(lower=1e-12).astype(float))
        slope = y.rolling(window=n, min_periods=max(2, n // 2)).apply(_rolling_linreg_slope, raw=True)
        raw_pos = (slope > 0.0).astype(float).fillna(0.0)
        score = slope.astype(float)
    elif strat == "bias":
        b_win = int(inp.bias_ma_window)
        ema = px.ewm(span=b_win, adjust=False, min_periods=max(2, b_win // 2)).mean()
        ln_c = np.log(px.clip(lower=1e-12))
        ln_ema = np.log(ema.clip(lower=1e-12))
        bias = (ln_c - ln_ema) * 100.0
        entry = float(inp.bias_entry)
        hot = float(inp.bias_hot)
        cold = float(inp.bias_cold)
        mode = str(inp.bias_pos_mode or "binary").strip().lower()
        pos = np.zeros(len(px), dtype=float)
        in_pos = False
        for i, d in enumerate(px.index):
            b = float(bias.loc[d]) if np.isfinite(float(bias.loc[d])) else np.nan
            if not np.isfinite(b):
                pos[i] = 0.0
                in_pos = False
                continue
            if not in_pos:
                if b > entry:
                    in_pos = True
                    pos[i] = 1.0 if mode == "binary" else float(np.clip((b - cold) / (hot - cold), 0.0, 1.0))
                else:
                    pos[i] = 0.0
            else:
                if b >= hot or b <= cold:
                    in_pos = False
                    pos[i] = 0.0
                else:
                    pos[i] = 1.0 if mode == "binary" else float(np.clip((b - cold) / (hot - cold), 0.0, 1.0))
        raw_pos = pd.Series(pos, index=px.index, dtype=float)
        score = bias.replace([np.inf, -np.inf], np.nan).astype(float)
    elif strat == "macd_cross":
        macd, sig, _ = _macd_core(px, fast=int(inp.macd_fast), slow=int(inp.macd_slow), signal=int(inp.macd_signal))
        raw_pos = (macd > sig).astype(float).fillna(0.0)
        score = (macd - sig).replace([np.inf, -np.inf], np.nan).astype(float)
    elif strat == "macd_zero_filter":
        macd, _, _ = _macd_core(px, fast=int(inp.macd_fast), slow=int(inp.macd_slow), signal=int(inp.macd_signal))
        raw_pos = (macd > 0.0).astype(float).fillna(0.0)
        score = macd.replace([np.inf, -np.inf], np.nan).astype(float)
    elif strat == "macd_v":
        macd, _, _ = _macd_core(px, fast=int(inp.macd_fast), slow=int(inp.macd_slow), signal=int(inp.macd_signal))
        atr = _atr_from_hlc(signal_high, signal_low, px, window=int(inp.macd_v_atr_window))
        macd_v = (macd / atr.replace(0.0, np.nan)) * float(inp.macd_v_scale)
        macd_v_sig = _ema(macd_v, int(inp.macd_signal))
        raw_pos = (macd_v > macd_v_sig).astype(float).fillna(0.0)
        score = (macd_v - macd_v_sig).replace([np.inf, -np.inf], np.nan).astype(float)
    elif strat == "random_entry":
        raw_pos = _pos_from_random_entry_hold(
            px.index,
            hold_days=int(getattr(inp, "random_hold_days", 20)),
            seed=getattr(inp, "random_seed", 42),
        )
        score = raw_pos.astype(float)
    else:
        mom = px / px.shift(int(inp.mom_lookback)) - 1.0
        raw_pos = _pos_from_tsmom(
            mom,
            entry_threshold=float(inp.tsmom_entry_threshold),
            exit_threshold=float(inp.tsmom_exit_threshold),
        )
        score = mom.replace([np.inf, -np.inf], np.nan).astype(float)

    out = raw_pos.astype(float).fillna(0.0).clip(lower=0.0)

    if bool(getattr(inp, "er_filter", False)):
        er = _efficiency_ratio(px, window=int(getattr(inp, "er_window", 10)))
        out, er_stats = _apply_er_entry_filter(out, er=er, threshold=float(getattr(inp, "er_threshold", 0.30)))
        debug["er_filter"] = er_stats
    if bool(getattr(inp, "impulse_entry_filter", False)):
        st = _compute_impulse_state(
            px,
            ema_window=13,
            macd_fast=int(inp.macd_fast),
            macd_slow=int(inp.macd_slow),
            macd_signal=int(inp.macd_signal),
        )
        out, imp_stats = _apply_impulse_entry_filter(
            out,
            impulse_state=st,
            allow_bull=bool(getattr(inp, "impulse_allow_bull", True)),
            allow_bear=bool(getattr(inp, "impulse_allow_bear", False)),
            allow_neutral=bool(getattr(inp, "impulse_allow_neutral", False)),
        )
        debug["impulse_filter"] = imp_stats
    if bool(getattr(inp, "er_exit_filter", False)):
        er_exit = _efficiency_ratio(px, window=int(getattr(inp, "er_exit_window", 10)))
        out, ex_stats = _apply_er_exit_filter(out, er=er_exit, threshold=float(getattr(inp, "er_exit_threshold", 0.88)))
        debug["er_exit_filter"] = ex_stats
    debug["signal_mode"] = "fractional_to_binary" if (out.max() > 1.0 or out.min() < 0.0 or (out % 1.0).abs().sum() > 0.0) else "binary"
    if strat == "random_entry" and getattr(inp, "random_seed", None) is None:
        debug["random_seed_note"] = f"random_seed=None for {code}, run is non-deterministic"
    debug["score_available_count"] = int(pd.Series(score).replace([np.inf, -np.inf], np.nan).notna().sum())
    return out, score.replace([np.inf, -np.inf], np.nan).astype(float), debug


def _build_bt_frame(
    db: Session,
    *,
    code: str,
    start: dt.date,
    end: dt.date,
) -> tuple[pd.DataFrame, pd.Series]:
    ohlc_none = load_ohlc_prices(db, codes=[code], start=start, end=end, adjust="none")
    ohlc_qfq = load_ohlc_prices(db, codes=[code], start=start, end=end, adjust="qfq")
    ohlc_hfq = load_ohlc_prices(db, codes=[code], start=start, end=end, adjust="hfq")
    close_none = load_close_prices(db, codes=[code], start=start, end=end, adjust="none")
    close_qfq = load_close_prices(db, codes=[code], start=start, end=end, adjust="qfq")
    close_hfq = load_close_prices(db, codes=[code], start=start, end=end, adjust="hfq")
    high_qfq, low_qfq = load_high_low_prices(db, codes=[code], start=start, end=end, adjust="qfq")

    if close_none.empty or code not in close_none.columns:
        raise ValueError(f"no execution data for {code}")
    idx = close_none.sort_index().index

    def _pick(ohlc: dict[str, pd.DataFrame], field: str, fallback: pd.Series) -> pd.Series:
        df = ohlc.get(field, pd.DataFrame()) if isinstance(ohlc, dict) else pd.DataFrame()
        if df is None or df.empty or code not in df.columns:
            return fallback.astype(float)
        return pd.to_numeric(df[code], errors="coerce").astype(float).reindex(idx).ffill().combine_first(fallback.astype(float))

    px_none = pd.to_numeric(close_none[code], errors="coerce").astype(float).reindex(idx).ffill()
    bt_df = pd.DataFrame(index=idx)
    bt_df["Open"] = _pick(ohlc_none, "open", px_none)
    bt_df["High"] = _pick(ohlc_none, "high", bt_df["Open"])
    bt_df["Low"] = _pick(ohlc_none, "low", bt_df["Open"])
    bt_df["Close"] = _pick(ohlc_none, "close", px_none)
    bt_df["Volume"] = _pick(ohlc_none, "volume", pd.Series(0.0, index=idx))

    sig_close = pd.to_numeric(close_qfq.get(code, px_none), errors="coerce").astype(float).reindex(idx).ffill()
    sig_open = _pick(ohlc_qfq, "open", sig_close)
    sig_high = pd.to_numeric(high_qfq.get(code, sig_close), errors="coerce").astype(float).reindex(idx).ffill()
    sig_low = pd.to_numeric(low_qfq.get(code, sig_close), errors="coerce").astype(float).reindex(idx).ffill()
    hfq_close = pd.to_numeric(close_hfq.get(code, bt_df["Close"]), errors="coerce").astype(float).reindex(idx).ffill()
    hfq_open = _pick(ohlc_hfq, "open", hfq_close)

    bt_df = bt_df.dropna(subset=["Open", "High", "Low", "Close"], how="any")
    sig_close = sig_close.reindex(bt_df.index).ffill()
    sig_open = sig_open.reindex(bt_df.index).ffill()
    sig_high = sig_high.reindex(bt_df.index).ffill()
    sig_low = sig_low.reindex(bt_df.index).ffill()
    hfq_close = hfq_close.reindex(bt_df.index).ffill()
    hfq_open = hfq_open.reindex(bt_df.index).ffill()
    bt_df["HfqOpen"] = hfq_open
    bt_df["HfqClose"] = hfq_close
    bt_df["SigOpen"] = sig_open
    bt_df["SigClose"] = sig_close
    bt_df["SigHigh"] = sig_high
    bt_df["SigLow"] = sig_low
    return bt_df, hfq_close


def _run_single_backtesting(
    db: Session,
    inp: TrendInputs,
    *,
    code: str,
    random_seed: int | None,
) -> dict[str, Any]:
    use_backtesting = True
    try:
        from backtesting import Backtest, Strategy
    except Exception:
        use_backtesting = False

    bt_df, hfq_close = _build_bt_frame(db, code=code, start=inp.start, end=inp.end)
    if bt_df.empty:
        raise ValueError(f"no valid OHLC rows for {code}")

    inp_local = TrendInputs(**{**asdict(inp), "code": code, "random_seed": random_seed})
    raw_pos, score_sig, debug_sig = _build_signal_position(
        inp_local,
        signal_close=bt_df["SigClose"],
        signal_high=bt_df["SigHigh"],
        signal_low=bt_df["SigLow"],
        code=code,
    )
    ep = str(getattr(inp, "exec_price", "open") or "open").strip().lower()
    open_none = bt_df["Open"].astype(float).combine_first(bt_df["Close"].astype(float))
    close_none = bt_df["Close"].astype(float)
    open_hfq = bt_df["HfqOpen"].astype(float).combine_first(bt_df["HfqClose"].astype(float))
    close_hfq = bt_df["HfqClose"].astype(float)
    if ep == "open":
        ret_exec_none = (open_none.shift(-1).div(open_none) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_exec_hfq = (open_hfq.shift(-1).div(open_hfq) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        px_exec_none = open_none.astype(float)
        px_exec_hfq = open_hfq.astype(float)
    elif ep == "oc2":
        ret_open_none = (open_none.shift(-1).div(open_none) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_close_none = (close_none.shift(-1).div(close_none) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_open_hfq = (open_hfq.shift(-1).div(open_hfq) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_close_hfq = (close_hfq.shift(-1).div(close_hfq) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_exec_none = (0.5 * (ret_open_none + ret_close_none)).astype(float)
        ret_exec_hfq = (0.5 * (ret_open_hfq + ret_close_hfq)).astype(float)
        px_exec_none = (0.5 * (open_none + close_none)).astype(float)
        px_exec_hfq = (0.5 * (open_hfq + close_hfq)).astype(float)
    else:
        ret_exec_none = (close_none.shift(-1).div(close_none) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_exec_hfq = (close_hfq.shift(-1).div(close_hfq) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        px_exec_none = close_none.astype(float)
        px_exec_hfq = close_hfq.astype(float)
    gross_none = (1.0 + ret_exec_none).astype(float)
    gross_hfq = (1.0 + ret_exec_hfq).astype(float)
    _, ca_mask = corporate_action_mask(gross_none.to_frame(code), gross_hfq.to_frame(code))
    ca_m = ca_mask[code].reindex(bt_df.index).fillna(False) if isinstance(ca_mask, pd.DataFrame) and code in ca_mask.columns else pd.Series(False, index=bt_df.index)
    ret_exec = ret_exec_none.where(~ca_m, other=ret_exec_hfq).astype(float)
    ret_exec_raw = ret_exec.copy().astype(float)
    px_exec_slip = px_exec_none.where(~ca_m, other=px_exec_hfq).replace([np.inf, -np.inf], np.nan).ffill().astype(float)

    atr_mode = str(getattr(inp, "atr_stop_mode", "none") or "none").strip().lower()
    atr_basis = str(getattr(inp, "atr_stop_atr_basis", "latest") or "latest").strip().lower()
    atr_reentry_mode = str(getattr(inp, "atr_stop_reentry_mode", "reenter") or "reenter").strip().lower()
    rtp_enabled = bool(getattr(inp, "r_take_profit_enabled", False))
    rtp_reentry_mode = str(getattr(inp, "r_take_profit_reentry_mode", "reenter") or "reenter").strip().lower()
    bias_v_tp_enabled = bool(getattr(inp, "bias_v_take_profit_enabled", False))
    bias_v_tp_reentry_mode = str(getattr(inp, "bias_v_take_profit_reentry_mode", "reenter") or "reenter").strip().lower()
    monthly_enabled = bool(getattr(inp, "monthly_risk_budget_enabled", False))
    ps = str(getattr(inp, "position_sizing", "equal") or "equal").strip().lower()
    simple_backtesting_mode = bool(
        use_backtesting
        and ps == "equal"
        and atr_mode == "none"
        and (not rtp_enabled)
        and (not bias_v_tp_enabled)
        and (not monthly_enabled)
        and float(raw_pos.max()) <= 1.0
        and float(raw_pos.min()) >= 0.0
        and float((raw_pos % 1.0).abs().sum()) <= 1e-12
    )

    atr_stop_stats: dict[str, Any] = {
        "enabled": False,
        "trigger_count": 0,
        "trigger_events": [],
        "trace_last_rows": [],
    }
    bias_v_tp_stats: dict[str, Any] = {
        "enabled": False,
        "trigger_count": 0,
        "trigger_events": [],
        "trace_last_rows": [],
    }
    r_take_profit_stats: dict[str, Any] = {
        "enabled": False,
        "trigger_count": 0,
        "tier_trigger_counts": {},
        "trigger_events": [],
        "trace_last_rows": [],
    }
    vol_risk_stats = {
        "vol_risk_adjust_total_count": 0,
        "vol_risk_adjust_reduce_on_expand_count": 0,
        "vol_risk_adjust_increase_on_contract_count": 0,
        "vol_risk_adjust_recover_from_expand_count": 0,
        "vol_risk_adjust_recover_from_contract_count": 0,
        "vol_risk_entry_state_reduce_on_expand_count": 0,
        "vol_risk_entry_state_increase_on_contract_count": 0,
    }
    monthly_gate_stats = {
        "enabled": False,
        "budget_pct": float(getattr(inp, "monthly_risk_budget_pct", 0.06) or 0.06),
        "include_new_trade_risk": bool(getattr(inp, "monthly_risk_budget_include_new_trade_risk", False)),
        "attempted_entry_count": 0,
        "attempted_entry_count_by_code": {str(code): 0},
        "blocked_entry_count": 0,
        "blocked_entry_count_by_code": {str(code): 0},
    }

    base_pos = raw_pos.astype(float).fillna(0.0)
    raw_pos_for_exec = base_pos.copy()
    if not simple_backtesting_mode:
        raw_pos_for_exec, atr_stop_stats = _apply_atr_stop(
            raw_pos_for_exec,
            open_=bt_df["SigOpen"].astype(float),
            close=bt_df["SigClose"].astype(float),
            high=bt_df["SigHigh"].astype(float),
            low=bt_df["SigLow"].astype(float),
            mode=atr_mode,
            atr_basis=atr_basis,
            reentry_mode=atr_reentry_mode,
            atr_window=int(getattr(inp, "atr_stop_window", 14)),
            n_mult=float(getattr(inp, "atr_stop_n", 2.0)),
            m_step=float(getattr(inp, "atr_stop_m", 0.5)),
        )
        atr_stop_stats = {**(atr_stop_stats or {}), **_extract_atr_plan_stops_from_trace(atr_stop_stats or {})}
        raw_pos_for_exec, bias_v_tp_stats = _apply_bias_v_take_profit(
            raw_pos_for_exec,
            open_=bt_df["SigOpen"].astype(float),
            close=bt_df["SigClose"].astype(float),
            high=bt_df["SigHigh"].astype(float),
            low=bt_df["SigLow"].astype(float),
            enabled=bias_v_tp_enabled,
            reentry_mode=bias_v_tp_reentry_mode,
            ma_window=int(getattr(inp, "bias_v_ma_window", 20)),
            atr_window=int(getattr(inp, "bias_v_atr_window", 20)),
            threshold=float(getattr(inp, "bias_v_take_profit_threshold", 5.0)),
        )
        raw_pos_for_exec, r_take_profit_stats = _apply_r_multiple_take_profit(
            raw_pos_for_exec,
            open_=bt_df["SigOpen"].astype(float),
            close=bt_df["SigClose"].astype(float),
            high=bt_df["SigHigh"].astype(float),
            low=bt_df["SigLow"].astype(float),
            enabled=rtp_enabled,
            reentry_mode=rtp_reentry_mode,
            atr_window=int(getattr(inp, "atr_stop_window", 14)),
            atr_n=float(getattr(inp, "atr_stop_n", 2.0)),
            tiers=_normalize_r_take_profit_tiers(getattr(inp, "r_take_profit_tiers", None)),
            atr_stop_enabled=bool(atr_mode != "none"),
        )

        sizing_scale = pd.Series(1.0, index=raw_pos_for_exec.index, dtype=float)
        if ps == "fixed_ratio":
            sizing_scale = pd.Series(float(getattr(inp, "fixed_pos_ratio", 0.04) or 0.04), index=raw_pos_for_exec.index, dtype=float)
        elif ps == "risk_budget":
            atr_rb = _atr_from_hlc(bt_df["SigHigh"], bt_df["SigLow"], bt_df["SigClose"], window=int(getattr(inp, "risk_budget_atr_window", 20)))
            atr_fast = _atr_from_hlc(bt_df["SigHigh"], bt_df["SigLow"], bt_df["SigClose"], window=int(getattr(inp, "vol_ratio_fast_atr_window", 5)))
            atr_slow = _atr_from_hlc(bt_df["SigHigh"], bt_df["SigLow"], bt_df["SigClose"], window=int(getattr(inp, "vol_ratio_slow_atr_window", 50)))
            sizing_scale, vol_risk_stats = _risk_budget_dynamic_weights(
                raw_pos_for_exec.astype(float).fillna(0.0),
                close=bt_df["SigClose"].astype(float),
                atr_for_budget=atr_rb.astype(float),
                atr_fast=atr_fast.astype(float),
                atr_slow=atr_slow.astype(float),
                risk_budget_pct=float(getattr(inp, "risk_budget_pct", 0.01) or 0.01),
                dynamic_enabled=bool(getattr(inp, "vol_regime_risk_mgmt_enabled", False)),
                expand_threshold=float(getattr(inp, "vol_ratio_expand_threshold", 1.45) or 1.45),
                contract_threshold=float(getattr(inp, "vol_ratio_contract_threshold", 0.65) or 0.65),
                normal_threshold=float(getattr(inp, "vol_ratio_normal_threshold", 1.05) or 1.05),
            )
        elif ps == "vol_target":
            asset_vol = (
                ret_exec.rolling(window=max(2, int(getattr(inp, "vol_window", 20) or 20)), min_periods=max(2, int(getattr(inp, "vol_window", 20) or 20)))
                .std()
                .mul(np.sqrt(252))
            ).replace([np.inf, -np.inf], np.nan)
            sizing_scale = (float(getattr(inp, "vol_target_ann", 0.20) or 0.20) / asset_vol).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0).astype(float)
        raw_pos_for_exec = (raw_pos_for_exec.clip(lower=0.0, upper=1.0) * sizing_scale).astype(float)

        if monthly_enabled:
            atr_gate = _atr_from_hlc(bt_df["SigHigh"], bt_df["SigLow"], bt_df["SigClose"], window=int(getattr(inp, "atr_stop_window", 14)))
            gated_df, monthly_gate_stats = _apply_monthly_risk_budget_gate(
                raw_pos_for_exec.to_frame(code),
                close=bt_df["SigClose"].to_frame(code),
                atr=atr_gate.to_frame(code),
                enabled=True,
                budget_pct=float(getattr(inp, "monthly_risk_budget_pct", 0.06) or 0.06),
                include_new_trade_risk=bool(getattr(inp, "monthly_risk_budget_include_new_trade_risk", False)),
                atr_stop_enabled=bool(atr_mode != "none"),
                atr_mode=str(atr_mode),
                atr_basis=str(atr_basis),
                atr_n=float(getattr(inp, "atr_stop_n", 2.0) or 2.0),
                atr_m=float(getattr(inp, "atr_stop_m", 0.5) or 0.5),
                fallback_position_risk=0.02,
            )
            raw_pos_for_exec = gated_df[code].astype(float)

    signal_pos = raw_pos_for_exec.astype(float).fillna(0.0)
    bt_df["DesiredPos"] = (signal_pos > 0.0).astype(float) if simple_backtesting_mode else signal_pos.astype(float)
    if simple_backtesting_mode:
        class BtTrendStrategy(Strategy):
            def init(self) -> None:
                return

            def next(self) -> None:
                if len(self.data.Close) < 2:
                    return
                target = float(self.data.DesiredPos[-2]) if np.isfinite(float(self.data.DesiredPos[-2])) else 0.0
                if target > 0.0 and not self.position:
                    self.buy(size=0.999999)
                elif target <= 0.0 and self.position:
                    self.position.close()

        trade_on_close = ep in {"close", "oc2"}
        bt = Backtest(
            bt_df,
            BtTrendStrategy,
            cash=1_000_000.0,
            spread=float(getattr(inp, "slippage_rate", 0.0) or 0.0),
            commission=float(getattr(inp, "cost_bps", 0.0) or 0.0) / 10000.0,
            trade_on_close=trade_on_close,
            exclusive_orders=True,
            finalize_trades=True,
        )
        stats = bt.run()
        equity_curve = stats.get("_equity_curve")
        if equity_curve is None or "Equity" not in equity_curve:
            raise ValueError(f"failed to build equity curve for {code}")
        eq = pd.Series(equity_curve["Equity"], index=pd.to_datetime(equity_curve.index), dtype=float).sort_index()
        nav = (eq / float(eq.iloc[0])).ffill().fillna(1.0)
        strat_ret = nav.pct_change().fillna(0.0).astype(float)
        pos_eff = bt_df["DesiredPos"].shift(1).fillna(0.0).astype(float)
        atr_override_ret = pd.Series(0.0, index=bt_df.index, dtype=float)
        bias_override_ret = pd.Series(0.0, index=bt_df.index, dtype=float)
        rtp_override_ret = pd.Series(0.0, index=bt_df.index, dtype=float)
        runtime_engine = "backtesting"
    else:
        pos_eff = bt_df["DesiredPos"].shift(1).fillna(0.0).astype(float).clip(lower=0.0)
        atr_override_ret = pd.Series(0.0, index=bt_df.index, dtype=float)
        bias_override_ret = pd.Series(0.0, index=bt_df.index, dtype=float)
        rtp_override_ret = pd.Series(0.0, index=bt_df.index, dtype=float)
        if not simple_backtesting_mode:
            pos_eff, atr_override_ret = _apply_intraday_stop_execution_single(
                weights=pos_eff,
                atr_stop_stats=atr_stop_stats,
                exec_price=str(ep),
                open_sig=bt_df["SigOpen"].astype(float),
                close_sig=bt_df["SigClose"].astype(float),
            )
            pos_eff, bias_override_ret = _apply_intraday_stop_execution_single(
                weights=pos_eff,
                atr_stop_stats=bias_v_tp_stats,
                exec_price=str(ep),
                open_sig=bt_df["SigOpen"].astype(float),
                close_sig=bt_df["SigClose"].astype(float),
            )
            pos_eff, rtp_override_ret = _apply_intraday_stop_execution_single(
                weights=pos_eff,
                atr_stop_stats=r_take_profit_stats,
                exec_price=str(ep),
                open_sig=bt_df["SigOpen"].astype(float),
                close_sig=bt_df["SigClose"].astype(float),
            )
        ret_exec_use = ret_exec.copy().astype(float)
        if ep in {"open", "oc2"}:
            same_day_none = (bt_df["Close"].astype(float) / bt_df["Open"].astype(float) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            same_day_hfq = (bt_df["HfqClose"].astype(float) / bt_df["HfqOpen"].astype(float) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            open_fwd_none = (bt_df["Open"].astype(float).shift(-1).div(bt_df["Open"].astype(float)) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            open_fwd_hfq = (bt_df["HfqOpen"].astype(float).shift(-1).div(bt_df["HfqOpen"].astype(float)) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            close_fwd_none = (bt_df["Close"].astype(float).shift(-1).div(bt_df["Close"].astype(float)) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            close_fwd_hfq = (bt_df["HfqClose"].astype(float).shift(-1).div(bt_df["HfqClose"].astype(float)) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            cm = ca_m.reindex(ret_exec_use.index).fillna(False).astype(bool)
            w_ix = pos_eff.reindex(ret_exec_use.index).fillna(0.0).astype(float)
            if ep == "open":
                for d in ret_exec_use.index:
                    if float(w_ix.loc[d]) <= 1e-12:
                        continue
                    ret_exec_use.loc[d] = float(same_day_hfq.loc[d]) if bool(cm.loc[d]) else float(same_day_none.loc[d])
            else:
                ret_blend_none = pd.Series(0.0, index=ret_exec_use.index, dtype=float)
                ret_blend_hfq = pd.Series(0.0, index=ret_exec_use.index, dtype=float)
                for d in ret_exec_use.index:
                    hold = float(w_ix.loc[d]) > 1e-12
                    po_n = float(same_day_none.loc[d]) if hold else float(open_fwd_none.loc[d])
                    po_h = float(same_day_hfq.loc[d]) if hold else float(open_fwd_hfq.loc[d])
                    cn = float(close_fwd_none.loc[d])
                    ch = float(close_fwd_hfq.loc[d])
                    ret_blend_none.loc[d] = 0.5 * (po_n + cn)
                    ret_blend_hfq.loc[d] = 0.5 * (po_h + ch)
                ret_exec_use = ret_blend_none.where(~cm, ret_blend_hfq).astype(float)
        base_ret = (pos_eff * ret_exec_use).astype(float) + atr_override_ret.astype(float) + bias_override_ret.astype(float) + rtp_override_ret.astype(float)
        turnover = pos_eff.diff().abs().fillna(pos_eff.abs()) / 2.0
        cost_comm = turnover * (float(getattr(inp, "cost_bps", 0.0) or 0.0) / 10000.0)
        cost_slip = slippage_return_from_turnover(turnover, slippage_spread=float(getattr(inp, "slippage_rate", 0.0) or 0.0), exec_price=px_exec_slip)
        strat_ret = (base_ret - cost_comm - cost_slip).fillna(0.0).astype(float)
        nav = _as_nav(strat_ret)
        ret_exec = ret_exec_use.astype(float)
        stats = {"_trades": pd.DataFrame(), "# Trades": int(((pos_eff > 0) & (pos_eff.shift(1).fillna(0.0) <= 0)).sum())}
        runtime_engine = "semantic_vectorized"

    ret_hfq_cc = bt_df["HfqClose"].astype(float).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    cm_bh = ca_m.reindex(ret_hfq_cc.index).fillna(False).astype(bool)
    bh_same_none = (bt_df["Close"].astype(float) / bt_df["Open"].astype(float) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    bh_same_hfq = (bt_df["HfqClose"].astype(float) / bt_df["HfqOpen"].astype(float) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    if ep == "close":
        bh_ret = ret_hfq_cc.astype(float)
    elif ep == "open":
        bh_ret = bh_same_none.where(~cm_bh, bh_same_hfq).astype(float).reindex(ret_hfq_cc.index).fillna(0.0)
    else:
        cf_none = (bt_df["Close"].astype(float).shift(-1).div(bt_df["Close"].astype(float)) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        cf_hfq = (bt_df["HfqClose"].astype(float).shift(-1).div(bt_df["HfqClose"].astype(float)) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        blend_bh_none = (0.5 * (bh_same_none + cf_none)).astype(float)
        blend_bh_hfq = (0.5 * (bh_same_hfq + cf_hfq)).astype(float)
        bh_ret = blend_bh_none.where(~cm_bh, blend_bh_hfq).astype(float).reindex(ret_hfq_cc.index).fillna(0.0)
    bh_nav = _as_nav(bh_ret)
    excess_nav = (nav / bh_nav.replace(0.0, np.nan)).fillna(1.0)
    excess_ret = excess_nav.pct_change().fillna(0.0).astype(float)
    active_ret = (strat_ret.reindex(nav.index).astype(float) - bh_ret.reindex(nav.index).astype(float)).fillna(0.0).astype(float)

    trades_df = stats.get("_trades")
    trades: list[dict[str, Any]] = []
    if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
        for _, row in trades_df.iterrows():
            et = row.get("EntryTime")
            xt = row.get("ExitTime")
            entry_date = pd.Timestamp(et).strftime("%Y-%m-%d") if pd.notna(et) else None
            exit_date = pd.Timestamp(xt).strftime("%Y-%m-%d") if pd.notna(xt) else None
            trades.append(
                {
                    "code": str(code),
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": float(row.get("EntryPrice")) if pd.notna(row.get("EntryPrice")) else None,
                    "exit_price": float(row.get("ExitPrice")) if pd.notna(row.get("ExitPrice")) else None,
                    "return_pct": float(row.get("ReturnPct")) if pd.notna(row.get("ReturnPct")) else None,
                    "pnl": float(row.get("PnL")) if pd.notna(row.get("PnL")) else None,
                    "size": float(row.get("Size")) if pd.notna(row.get("Size")) else None,
                    "duration_days": int(row.get("Duration").days) if pd.notna(row.get("Duration")) else None,
                }
            )

    if not trades:
        sig = bt_df["DesiredPos"].reindex(nav.index).fillna(0.0).astype(float)
        in_pos = False
        entry_date = None
        entry_price = None
        for d in sig.index:
            s = float(sig.loc[d])
            if (not in_pos) and s > 0.0:
                in_pos = True
                entry_date = pd.Timestamp(d).strftime("%Y-%m-%d")
                entry_price = float(bt_df.loc[d, "Close"]) if d in bt_df.index else None
            elif in_pos and s <= 0.0:
                exit_date = pd.Timestamp(d).strftime("%Y-%m-%d")
                exit_price = float(bt_df.loc[d, "Close"]) if d in bt_df.index else None
                r = None
                if entry_price and exit_price and entry_price > 0:
                    r = float(exit_price / entry_price - 1.0)
                trades.append(
                    {
                        "code": str(code),
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return_pct": r,
                        "pnl": None,
                        "size": 1.0,
                        "duration_days": None,
                    }
                )
                in_pos = False
                entry_date = None
                entry_price = None

    return {
        "code": code,
        "dates": [d.strftime("%Y-%m-%d") for d in nav.index],
        "nav": nav,
        "buy_hold_nav": bh_nav.reindex(nav.index).ffill().fillna(1.0),
        "excess_nav": excess_nav.reindex(nav.index).ffill().fillna(1.0),
        "strat_ret": strat_ret,
        "bench_ret": bh_ret.reindex(nav.index).fillna(0.0),
        "excess_ret": excess_ret,
        "desired_pos": bt_df["DesiredPos"].reindex(nav.index).fillna(0.0),
        "base_pos": base_pos.reindex(nav.index).fillna(0.0).astype(float),
        "signal_pos": signal_pos.reindex(nav.index).fillna(0.0).astype(float),
        "sig_open": bt_df["SigOpen"].reindex(nav.index).ffill(),
        "sig_close": bt_df["SigClose"].reindex(nav.index).ffill(),
        "sig_high": bt_df["SigHigh"].reindex(nav.index).ffill(),
        "sig_low": bt_df["SigLow"].reindex(nav.index).ffill(),
        "ret_exec": ret_exec.reindex(nav.index).fillna(0.0).astype(float),
        "ret_exec_raw": ret_exec_raw.reindex(nav.index).fillna(0.0).astype(float),
        "px_exec_slip": px_exec_slip.reindex(nav.index).ffill().astype(float),
        "ret_exec_none": ret_exec_none.reindex(nav.index).fillna(0.0).astype(float),
        "ret_exec_hfq": ret_exec_hfq.reindex(nav.index).fillna(0.0).astype(float),
        "exec_open_none": bt_df["Open"].reindex(nav.index).astype(float),
        "exec_close_none": bt_df["Close"].reindex(nav.index).astype(float),
        "exec_open_hfq": bt_df["HfqOpen"].reindex(nav.index).astype(float),
        "exec_close_hfq": bt_df["HfqClose"].reindex(nav.index).astype(float),
        "corp_factor": (gross_hfq / gross_none.replace(0.0, np.nan)).reindex(nav.index).replace([np.inf, -np.inf], np.nan).astype(float),
        "ca_mask": ca_m.reindex(nav.index).fillna(False).astype(bool),
        "trades": trades,
        "trade_count": int(stats.get("# Trades", 0) or 0),
        "signal_debug": debug_sig,
        "signal_score": score_sig.reindex(nav.index).astype(float),
        "runtime_engine": runtime_engine,
        "semantic_stats": {
            "atr_stop": atr_stop_stats,
            "bias_v_take_profit": bias_v_tp_stats,
            "r_take_profit": r_take_profit_stats,
            "vol_risk_adjust": vol_risk_stats,
            "monthly_risk_budget_gate": monthly_gate_stats,
        },
    }


def compute_trend_backtest_bt(db: Session, inp: TrendInputs) -> dict[str, Any]:
    code = str(inp.code or "").strip()
    if not code:
        raise ValueError("code is empty")
    _validate_bt_single_inputs(inp)
    strat = str(inp.strategy or "ma_filter").strip().lower()
    ep = str(getattr(inp, "exec_price", "open") or "open").strip().lower()

    single = _run_single_backtesting(db, inp, code=code, random_seed=getattr(inp, "random_seed", 42))
    nav = single["nav"]
    bh_nav = single["buy_hold_nav"]
    excess_nav = single["excess_nav"]
    strat_ret = single["strat_ret"]
    bench_ret = single["bench_ret"]
    excess_ret = single["excess_ret"]
    active_ret = (strat_ret.reindex(nav.index).astype(float) - bench_ret.reindex(nav.index).astype(float)).fillna(0.0).astype(float)

    ps = str(getattr(inp, "position_sizing", "equal") or "equal").strip().lower()
    if ps == "risk_budget":
        weight_s = _risk_budget_frozen_weight(
            single["desired_pos"],
            close=single["sig_close"],
            high=single["sig_high"],
            low=single["sig_low"],
            atr_window=int(getattr(inp, "risk_budget_atr_window", 20) or 20),
            risk_budget_pct=float(getattr(inp, "risk_budget_pct", 0.01) or 0.01),
        )
    else:
        weight_s = single["desired_pos"].astype(float)
    w_eff = single["desired_pos"].shift(1).fillna(0.0).astype(float).clip(lower=0.0)
    sig_open = single.get("sig_open", single["sig_close"]).astype(float).reindex(nav.index).ffill()
    sig_close = single["sig_close"].astype(float).reindex(nav.index).ffill()
    sig_high = single["sig_high"].astype(float).reindex(nav.index).ffill()
    sig_low = single["sig_low"].astype(float).reindex(nav.index).ffill()
    ret_exec_s = single.get("ret_exec", strat_ret).astype(float).reindex(nav.index).fillna(0.0)
    px_exec_s = single.get("px_exec_slip", sig_close).astype(float).reindex(nav.index).ffill()
    turnover_one_way = (w_eff - w_eff.shift(1).fillna(0.0)).abs() / 2.0
    cost_s = turnover_one_way * (float(getattr(inp, "cost_bps", 0.0) or 0.0) / 10000.0)
    slip_s = slippage_return_from_turnover(
        turnover_one_way.astype(float),
        slippage_spread=float(getattr(inp, "slippage_rate", 0.0) or 0.0),
        exec_price=px_exec_s.astype(float),
    ).astype(float)
    ret_overnight = (sig_open / sig_close.shift(1) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    ret_intraday = (sig_close / sig_open - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    sem_dbg = single.get("semantic_stats") or {}
    atr_over = pd.Series(0.0, index=nav.index, dtype=float)
    bv_over = pd.Series(0.0, index=nav.index, dtype=float)
    rtp_over = pd.Series(0.0, index=nav.index, dtype=float)
    if str(single.get("runtime_engine") or "") != "backtesting":
        w_tmp = w_eff.copy()
        w_tmp, atr_over = _apply_intraday_stop_execution_single(
            weights=w_tmp,
            atr_stop_stats=dict((sem_dbg.get("atr_stop") or {})),
            exec_price=str(getattr(inp, "exec_price", "open") or "open"),
            open_sig=sig_open,
            close_sig=sig_close,
        )
        w_tmp, bv_over = _apply_intraday_stop_execution_single(
            weights=w_tmp,
            atr_stop_stats=dict((sem_dbg.get("bias_v_take_profit") or {})),
            exec_price=str(getattr(inp, "exec_price", "open") or "open"),
            open_sig=sig_open,
            close_sig=sig_close,
        )
        w_tmp, rtp_over = _apply_intraday_stop_execution_single(
            weights=w_tmp,
            atr_stop_stats=dict((sem_dbg.get("r_take_profit") or {})),
            exec_price=str(getattr(inp, "exec_price", "open") or "open"),
            open_sig=sig_open,
            close_sig=sig_close,
        )
        w_eff = w_tmp.astype(float)
    return_decomposition = None
    quick_mode = bool(getattr(inp, "quick_mode", False))
    if not quick_mode:
        decomp_overnight = (w_eff * ret_overnight).astype(float)
        decomp_intraday = (w_eff * ret_intraday).astype(float)
        decomp_interaction = (w_eff * ret_overnight * ret_intraday).astype(float)
        decomp_risk = (atr_over + bv_over + rtp_over).astype(float)
        decomp_cost = (cost_s + slip_s).astype(float)
        decomp_gross = (decomp_overnight + decomp_intraday + decomp_interaction + decomp_risk).astype(float)
        decomp_net = (decomp_gross - decomp_cost).astype(float)
        return_decomposition = {
            "dates": [d.strftime("%Y-%m-%d") for d in nav.index],
            "series": {
                "overnight": decomp_overnight.tolist(),
                "intraday": decomp_intraday.tolist(),
                "interaction": decomp_interaction.tolist(),
                "atr_stop_override": atr_over.tolist(),
                "bias_v_take_profit_override": bv_over.tolist(),
                "r_take_profit_override": rtp_over.tolist(),
                "risk_exit_override": decomp_risk.tolist(),
                "cost": decomp_cost.tolist(),
                "gross": decomp_gross.tolist(),
                "net": decomp_net.tolist(),
            },
            "summary": {
                "ann_overnight": float(decomp_overnight.iloc[1:].mean() * 252.0) if len(decomp_overnight) > 1 else 0.0,
                "ann_intraday": float(decomp_intraday.iloc[1:].mean() * 252.0) if len(decomp_intraday) > 1 else 0.0,
                "ann_interaction": float(decomp_interaction.iloc[1:].mean() * 252.0) if len(decomp_interaction) > 1 else 0.0,
                "ann_atr_stop_override": float(atr_over.iloc[1:].mean() * 252.0) if len(atr_over) > 1 else 0.0,
                "ann_bias_v_take_profit_override": float(bv_over.iloc[1:].mean() * 252.0) if len(bv_over) > 1 else 0.0,
                "ann_r_take_profit_override": float(rtp_over.iloc[1:].mean() * 252.0) if len(rtp_over) > 1 else 0.0,
                "ann_risk_exit_override": float(decomp_risk.iloc[1:].mean() * 252.0) if len(decomp_risk) > 1 else 0.0,
                "ann_cost": float(decomp_cost.iloc[1:].mean() * 252.0) if len(decomp_cost) > 1 else 0.0,
                "ann_gross": float(decomp_gross.iloc[1:].mean() * 252.0) if len(decomp_gross) > 1 else 0.0,
                "ann_net": float(decomp_net.iloc[1:].mean() * 252.0) if len(decomp_net) > 1 else 0.0,
            },
        }
    event_study = None if quick_mode else compute_event_study(
        dates=nav.index,
        daily_returns=strat_ret.reindex(nav.index).astype(float),
        entry_dates=entry_dates_from_exposure(w_eff.reindex(nav.index).astype(float)),
    )
    market_regime = build_market_regime_report(
        close=sig_close.to_frame(code).astype(float),
        high=sig_high.to_frame(code).astype(float),
        low=sig_low.to_frame(code).astype(float),
        weights=w_eff.to_frame(code).astype(float),
        asset_returns=ret_exec_s.to_frame(code).astype(float),
        strategy_returns=strat_ret.reindex(nav.index).astype(float),
        ann_factor=252,
    )
    attribution = _compute_return_risk_contributions(
        asset_ret=ret_exec_s.to_frame(code).reindex(nav.index).astype(float).fillna(0.0),
        weights=w_eff.to_frame(code).reindex(nav.index).astype(float).fillna(0.0),
        total_return=float(nav.iloc[-1] - 1.0) if len(nav) else 0.0,
    )
    latest_entry_exec_px = _latest_entry_exec_price_with_slippage(
        effective_weight=w_eff.reindex(nav.index).astype(float),
        exec_price_series=px_exec_s.reindex(nav.index).ffill().astype(float),
        slippage_spread=float(getattr(inp, "slippage_rate", 0.0) or 0.0),
    )
    trade_one = _trade_returns_from_weight_series(
        w_eff.reindex(nav.index).astype(float),
        ret_exec_s.reindex(nav.index).astype(float),
        cost_bps=float(getattr(inp, "cost_bps", 0.0) or 0.0),
        slippage_rate=float(getattr(inp, "slippage_rate", 0.0) or 0.0),
        exec_price=px_exec_s.reindex(nav.index).ffill().astype(float),
        dates=nav.index,
    )
    atr_risk = _atr_from_hlc(
        sig_high.reindex(nav.index).astype(float).fillna(sig_close.reindex(nav.index).astype(float)),
        sig_low.reindex(nav.index).astype(float).fillna(sig_close.reindex(nav.index).astype(float)),
        sig_close.reindex(nav.index).astype(float),
        window=int(getattr(inp, "atr_stop_window", 14) or 14),
    ).reindex(nav.index)
    trade_r_pack = enrich_trades_with_r_metrics(
        trade_one.get("trades", []),
        nav=nav.astype(float),
        weights=w_eff.reindex(nav.index).astype(float),
        exec_price=px_exec_s.reindex(nav.index).ffill().astype(float),
        atr=atr_risk.astype(float),
        atr_mult=float(getattr(inp, "atr_stop_n", 2.0) or 2.0),
        risk_budget_pct=(float(getattr(inp, "risk_budget_pct", 0.01) or 0.01) if ps == "risk_budget" else None),
        cost_bps=float(getattr(inp, "cost_bps", 0.0) or 0.0),
        slippage_rate=float(getattr(inp, "slippage_rate", 0.0) or 0.0),
        default_code=str(code),
        ulcer_index=float(_ulcer_index(nav, in_percent=True)) if len(nav) else None,
        annual_trade_count=(float(len(trade_one.get("returns", []))) / max(1.0, float(len(nav))) * 252.0) if len(nav) else None,
        backtest_years=(float(len(nav)) / 252.0) if len(nav) else None,
        score_sqn_weight=0.60,
        score_ulcer_weight=0.40,
    )
    trades_with_r = list(trade_r_pack.get("trades") or [])
    r_stats_out = dict(trade_r_pack.get("statistics") or {})
    r_stats_out.pop("trade_system_score", None)
    if not quick_mode:
        mom_for_entry = (
            sig_close.reindex(nav.index).astype(float)
            / sig_close.reindex(nav.index).astype(float).shift(int(getattr(inp, "mom_lookback", 252) or 252))
            - 1.0
        ).astype(float)
        er_for_entry = _efficiency_ratio(
            sig_close.reindex(nav.index).astype(float),
            window=int(getattr(inp, "er_window", 10) or 10),
        ).astype(float)
        atr_fast_for_entry = _atr_from_hlc(
            sig_high.reindex(nav.index).astype(float).fillna(sig_close.reindex(nav.index).astype(float)),
            sig_low.reindex(nav.index).astype(float).fillna(sig_close.reindex(nav.index).astype(float)),
            sig_close.reindex(nav.index).astype(float),
            window=int(getattr(inp, "vol_ratio_fast_atr_window", 5) or 5),
        ).astype(float)
        atr_slow_for_entry = _atr_from_hlc(
            sig_high.reindex(nav.index).astype(float).fillna(sig_close.reindex(nav.index).astype(float)),
            sig_low.reindex(nav.index).astype(float).fillna(sig_close.reindex(nav.index).astype(float)),
            sig_close.reindex(nav.index).astype(float),
            window=int(getattr(inp, "vol_ratio_slow_atr_window", 50) or 50),
        ).astype(float)
        vol_ratio_for_entry = (atr_fast_for_entry / atr_slow_for_entry.replace(0.0, np.nan)).astype(float)
        impulse_state = _compute_impulse_state(
            sig_close.reindex(nav.index).astype(float),
            ema_window=13,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
        )
        condition_bins_by_code_single = {
            str(code): {
                "momentum": _bucketize_momentum_series(mom_for_entry.reindex(nav.index)),
                "er": _bucketize_er_series(er_for_entry.reindex(nav.index)),
                "vol_ratio": _bucketize_vol_ratio_series(vol_ratio_for_entry.reindex(nav.index)),
                "impulse": _bucketize_impulse_series(
                    (impulse_state if impulse_state is not None else pd.Series(index=nav.index, dtype=object)).reindex(nav.index)
                ),
            }
        }
        trades_with_r = _attach_entry_condition_bins_to_trades(
            trades_with_r,
            condition_bins_by_code=condition_bins_by_code_single,
            dates=nav.index,
            default_code=str(code),
        )
    mfe_r_distribution = build_trade_mfe_r_distribution(
        trade_one.get("trades", []),
        close=sig_close.reindex(nav.index).astype(float).ffill(),
        high=sig_high.reindex(nav.index).astype(float).ffill(),
        atr=atr_risk.astype(float).reindex(nav.index),
        atr_mult=float(getattr(inp, "atr_stop_n", 2.0) or 2.0),
        default_code=str(code),
    )
    sig_dbg = single.get("signal_debug") or {}
    sem_dbg = single.get("semantic_stats") or {}
    er_stats = sig_dbg.get("er_filter") or {}
    imp_stats = sig_dbg.get("impulse_filter") or {}
    er_exit_stats = sig_dbg.get("er_exit_filter") or {}
    atr_stats = sem_dbg.get("atr_stop") or {}
    rtp_stats = sem_dbg.get("r_take_profit") or {}
    bv_stats = sem_dbg.get("bias_v_take_profit") or {}
    vol_stats = sem_dbg.get("vol_risk_adjust") or {}
    month_stats = sem_dbg.get("monthly_risk_budget_gate") or {}
    impulse_attempted = int(imp_stats.get("attempted_entry_count", 0))
    impulse_blocked = int(imp_stats.get("blocked_entry_count", 0))
    monthly_attempted = int(month_stats.get("attempted_entry_count", 0))
    monthly_blocked = int(month_stats.get("blocked_entry_count", 0))
    overall_stats = {
        **_trade_stats_from_returns(trade_one.get("returns", [])),
        "n": int(single["trade_count"]),
        "atr_stop_trigger_count": int(atr_stats.get("trigger_count", 0)),
        "r_take_profit_trigger_count": int(rtp_stats.get("trigger_count", 0)),
        "bias_v_take_profit_trigger_count": int(bv_stats.get("trigger_count", 0)),
        "r_take_profit_tier_trigger_counts": dict(rtp_stats.get("tier_trigger_counts") or {}),
        "er_filter_blocked_entry_count": int(er_stats.get("blocked_entry_count", 0)),
        "er_filter_attempted_entry_count": int(er_stats.get("attempted_entry_count", 0)),
        "er_filter_allowed_entry_count": int(er_stats.get("allowed_entry_count", 0)),
        "impulse_filter_blocked_entry_count": impulse_blocked,
        "impulse_filter_attempted_entry_count": impulse_attempted,
        "impulse_filter_allowed_entry_count": int(imp_stats.get("allowed_entry_count", 0)),
        "impulse_filter_blocked_entry_rate": (float(impulse_blocked / impulse_attempted) if impulse_attempted > 0 else 0.0),
        "impulse_filter_blocked_entry_count_bull": int(imp_stats.get("blocked_entry_count_bull", 0)),
        "impulse_filter_blocked_entry_count_bear": int(imp_stats.get("blocked_entry_count_bear", 0)),
        "impulse_filter_blocked_entry_count_neutral": int(imp_stats.get("blocked_entry_count_neutral", 0)),
        "er_exit_filter_trigger_count": int(er_exit_stats.get("trigger_count", 0)),
        "vol_risk_adjust_total_count": int(vol_stats.get("vol_risk_adjust_total_count", 0)),
        "vol_risk_adjust_reduce_on_expand_count": int(vol_stats.get("vol_risk_adjust_reduce_on_expand_count", 0)),
        "vol_risk_adjust_increase_on_contract_count": int(vol_stats.get("vol_risk_adjust_increase_on_contract_count", 0)),
        "vol_risk_adjust_recover_from_expand_count": int(vol_stats.get("vol_risk_adjust_recover_from_expand_count", 0)),
        "vol_risk_adjust_recover_from_contract_count": int(vol_stats.get("vol_risk_adjust_recover_from_contract_count", 0)),
        "vol_risk_entry_state_reduce_on_expand_count": int(vol_stats.get("vol_risk_entry_state_reduce_on_expand_count", 0)),
        "vol_risk_entry_state_increase_on_contract_count": int(vol_stats.get("vol_risk_entry_state_increase_on_contract_count", 0)),
        "monthly_risk_budget_attempted_entry_count": monthly_attempted,
        "monthly_risk_budget_blocked_entry_count": monthly_blocked,
        "monthly_risk_budget_blocked_entry_rate": (float(monthly_blocked / monthly_attempted) if monthly_attempted > 0 else 0.0),
    }
    by_code_stats = {str(code): {**_trade_stats_from_returns(trade_one.get("returns", [])), **dict(overall_stats)}}
    trade_stats_trades = [] if quick_mode else list(trades_with_r)
    trade_stats_trades_by_code = {str(code): ([] if quick_mode else list(trades_with_r))}
    sample_days = int(len(strat_ret))
    complete_trade_count = int(len(trade_one.get("returns", [])))
    avg_daily_turnover = float(turnover_one_way.mean()) if len(turnover_one_way) else 0.0
    avg_annual_turnover = float(avg_daily_turnover * 252.0)
    avg_daily_trade_count = float(complete_trade_count / sample_days) if sample_days > 0 else 0.0
    avg_annual_trade_count = float(avg_daily_trade_count * 252.0)

    out = {
        "meta": {
            "type": "trend_backtest",
            "engine": "bt",
            "runtime_engine": str(single.get("runtime_engine") or "unknown"),
            "code": code,
            "start": inp.start.strftime("%Y%m%d"),
            "end": inp.end.strftime("%Y%m%d"),
            "strategy": strat,
            "strategy_execution_description": TREND_STRATEGY_EXECUTION_DESCRIPTIONS.get(strat, ""),
            "price_basis": {
                "signal": "qfq close",
                "strategy_nav": "none close preferred; hfq return fallback on corporate-action days",
                "benchmark_nav": {
                    "close": "HFQ close-to-close daily returns (BUY_HOLD line; excess vs strategy uses this series)",
                    "open": "same-day open→close (none; hfq on corporate-action days); BUY_HOLD aligned to open execution",
                    "oc2": "50% same-day open→close + 50% HFQ close-to-close next day; BUY_HOLD aligned to OC2 execution",
                }.get(ep, "unknown exec_price"),
            },
            "params": _build_meta_params(inp),
            "limitations": [],
        },
        "nav": {
            "dates": single["dates"],
            "series": {
                "STRAT": [float(x) for x in nav.values],
                "BUY_HOLD": [float(x) for x in bh_nav.values],
                "EXCESS": [float(x) for x in excess_nav.values],
            },
        },
        "signals": {
            "dates": single["dates"],
            "base_position": [float(x) for x in single["base_pos"].values],
            "position": [float(x) for x in single["signal_pos"].values],
            "position_effective": [float(x) for x in w_eff.values],
        },
        "weights": {
            "dates": single["dates"],
            "series": {code: [float(x) for x in weight_s.values]},
        },
        "metrics": {
            "strategy": {
                **_metrics_from_ret(strat_ret, float(inp.risk_free_rate)),
                "avg_daily_turnover": float(avg_daily_turnover),
                "avg_annual_turnover": float(avg_annual_turnover),
                "avg_annual_turnover_rate": float(avg_annual_turnover),
                "avg_daily_trade_count": float(avg_daily_trade_count),
                "avg_annual_trade_count": float(avg_annual_trade_count),
                "r_take_profit_tier_trigger_counts": dict(rtp_stats.get("tier_trigger_counts") or {}),
                "impulse_filter_blocked_entry_count": int(imp_stats.get("blocked_entry_count", 0)),
                "impulse_filter_blocked_entry_count_bull": int(imp_stats.get("blocked_entry_count_bull", 0)),
                "impulse_filter_blocked_entry_count_bear": int(imp_stats.get("blocked_entry_count_bear", 0)),
                "impulse_filter_blocked_entry_count_neutral": int(imp_stats.get("blocked_entry_count_neutral", 0)),
                "monthly_risk_budget_blocked_entry_count": int(month_stats.get("blocked_entry_count", 0)),
            },
            "benchmark": _metrics_from_ret(bench_ret, float(inp.risk_free_rate)),
            "excess": {
                **_metrics_from_ret(excess_ret, float(inp.risk_free_rate)),
                "information_ratio": float(_sharpe(active_ret, rf=0.0)),
            },
        },
        "period_returns": {
            "weekly": _period_returns(nav, "W-FRI"),
            "monthly": _period_returns(nav, "ME"),
            "quarterly": _period_returns(nav, "QE"),
            "yearly": _period_returns(nav, "YE"),
        },
        "rolling": _rolling_pack(nav),
        "attribution": attribution,
        "trade_statistics": {
            "all": {"n": int(single["trade_count"])},
            "overall": overall_stats,
            "by_code": by_code_stats,
            "trades": trade_stats_trades,
            "trades_by_code": trade_stats_trades_by_code,
            "mfe_r_distribution": mfe_r_distribution,
        },
        "r_statistics": r_stats_out,
        "trades": ([] if quick_mode else trades_with_r),
        "risk_controls": {
            "vol_regime_risk_mgmt": {
                "enabled": bool(getattr(inp, "vol_regime_risk_mgmt_enabled", False)),
                "fast_atr_window": int(getattr(inp, "vol_ratio_fast_atr_window", 5) or 5),
                "slow_atr_window": int(getattr(inp, "vol_ratio_slow_atr_window", 50) or 50),
                "expand_threshold": float(getattr(inp, "vol_ratio_expand_threshold", 1.45) or 1.45),
                "contract_threshold": float(getattr(inp, "vol_ratio_contract_threshold", 0.65) or 0.65),
                "normal_threshold": float(getattr(inp, "vol_ratio_normal_threshold", 1.05) or 1.05),
                "adjust_total_count": int(vol_stats.get("vol_risk_adjust_total_count", 0)),
                "adjust_reduce_on_expand_count": int(vol_stats.get("vol_risk_adjust_reduce_on_expand_count", 0)),
                "adjust_increase_on_contract_count": int(vol_stats.get("vol_risk_adjust_increase_on_contract_count", 0)),
                "adjust_recover_from_expand_count": int(vol_stats.get("vol_risk_adjust_recover_from_expand_count", 0)),
                "adjust_recover_from_contract_count": int(vol_stats.get("vol_risk_adjust_recover_from_contract_count", 0)),
                "entry_state_reduce_on_expand_count": int(vol_stats.get("vol_risk_entry_state_reduce_on_expand_count", 0)),
                "entry_state_increase_on_contract_count": int(vol_stats.get("vol_risk_entry_state_increase_on_contract_count", 0)),
            },
            "er_exit_filter": {
                "enabled": bool(getattr(inp, "er_exit_filter", False)),
                "window": int(getattr(inp, "er_exit_window", 10) or 10),
                "threshold": float(getattr(inp, "er_exit_threshold", 0.88) or 0.88),
                "trigger_count": int(er_exit_stats.get("trigger_count", 0)),
                "trigger_dates": list(er_exit_stats.get("trigger_dates") or []),
                "trace_last_rows": list(er_exit_stats.get("trace_last_rows") or []),
            },
            "atr_stop": dict(sem_dbg.get("atr_stop") or {}),
            "bias_v_take_profit": dict(sem_dbg.get("bias_v_take_profit") or {}),
            "r_take_profit": dict(sem_dbg.get("r_take_profit") or {}),
            "monthly_risk_budget_gate": {
                **dict(sem_dbg.get("monthly_risk_budget_gate") or {}),
                "enabled": bool(getattr(inp, "monthly_risk_budget_enabled", False)),
                "budget_pct": float(getattr(inp, "monthly_risk_budget_pct", 0.06) or 0.06),
                "include_new_trade_risk": bool(getattr(inp, "monthly_risk_budget_include_new_trade_risk", False)),
            },
            "monthly_risk_budget": {
                **dict(sem_dbg.get("monthly_risk_budget_gate") or {}),
                "enabled": bool(getattr(inp, "monthly_risk_budget_enabled", False)),
                "budget_pct": float(getattr(inp, "monthly_risk_budget_pct", 0.06) or 0.06),
                "include_new_trade_risk": bool(getattr(inp, "monthly_risk_budget_include_new_trade_risk", False)),
            },
        },
        "return_decomposition": return_decomposition,
        "event_study": event_study,
        "market_regime": market_regime,
        "next_plan": {
            "decision_date": (str(nav.index[-1].date()) if len(nav.index) else None),
            "current_effective_weight": (float(w_eff.iloc[-1]) if len(w_eff) else 0.0),
            "target_weight": (float(single["desired_pos"].reindex(nav.index).iloc[-1]) if len(nav.index) else 0.0),
            "entry_exec_price_with_slippage_by_asset": (
                {str(code): float(latest_entry_exec_px)} if latest_entry_exec_px is not None else {}
            ),
            "trace": {
                "atr_stop_mode": str(getattr(inp, "atr_stop_mode", "none") or "none"),
                "atr_stop_atr_basis": str(getattr(inp, "atr_stop_atr_basis", "latest") or "latest"),
                "atr_stop_reentry_mode": str(getattr(inp, "atr_stop_reentry_mode", "reenter") or "reenter"),
                "base_signal_prev": (float(single["base_pos"].iloc[-2]) if len(single["base_pos"]) >= 2 else 0.0),
                "base_signal_today": (float(single["base_pos"].iloc[-1]) if len(single["base_pos"]) else 0.0),
                "base_entry_event_today": (
                    bool((single["base_pos"].iloc[-1] > 0.0) and (single["base_pos"].iloc[-2] <= 0.0))
                    if len(single["base_pos"]) >= 2
                    else bool(single["base_pos"].iloc[-1] > 0.0) if len(single["base_pos"]) else False
                ),
                "strategy": str(strat),
                "atr_stop": dict(atr_stats),
                "bias_v_take_profit": dict(bv_stats),
                "r_take_profit": dict(rtp_stats),
                "er_exit_filter": {
                    "enabled": bool(getattr(inp, "er_exit_filter", False)),
                    "window": int(getattr(inp, "er_exit_window", 10) or 10),
                    "threshold": float(getattr(inp, "er_exit_threshold", 0.88) or 0.88),
                    "trigger_count": int(er_exit_stats.get("trigger_count", 0)),
                    "trigger_dates": list(er_exit_stats.get("trigger_dates") or []),
                    "trace_last_rows": list(er_exit_stats.get("trace_last_rows") or []),
                },
            },
        },
        "corporate_actions": (
            [
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "none_return": float(single["ret_exec_none"].loc[d]),
                    "hfq_return": float(single["ret_exec_hfq"].loc[d]),
                    "corp_factor": (
                        float(single["corp_factor"].loc[d]) if np.isfinite(float(single["corp_factor"].loc[d])) else None
                    ),
                }
                for d in nav.index
                if bool(single["ca_mask"].loc[d])
            ][:200]
        ),
        "signal_debug": single["signal_debug"],
    }
    if not quick_mode:
        out["trade_statistics"]["entry_condition_stats"] = {
            "scope": "closed_trades_only",
            "signal_day_basis": "signal_day_before_entry_execution",
            "quasi_causal_method": "uplift + two_proportion_z / welch_t_normal_approx + BH",
            "strong_causal_method": "uplift + stratified_permutation + BH",
            "overall": _build_entry_condition_stats(trades_with_r, by_code=False, n_perm=300, seed=20260410),
            "by_code": {
                str(code): _build_entry_condition_stats(trades_with_r, by_code=True, n_perm=200, seed=20260410)
            },
        }
    return out


def compute_trend_portfolio_backtest_bt(db: Session, inp: TrendPortfolioInputs) -> dict[str, Any]:
    codes = list(dict.fromkeys([str(c).strip() for c in (inp.codes or []) if str(c).strip()]))
    if not codes:
        raise ValueError("codes is empty")
    single_validation = TrendInputs(
        code="__BT_VALIDATION__",
        start=inp.start,
        end=inp.end,
        strategy=inp.strategy,
        ma_type=inp.ma_type,
        kama_fast_window=inp.kama_fast_window,
        kama_slow_window=inp.kama_slow_window,
        position_sizing=inp.position_sizing,
        risk_budget_pct=inp.risk_budget_pct,
        vol_regime_risk_mgmt_enabled=inp.vol_regime_risk_mgmt_enabled,
        vol_ratio_expand_threshold=inp.vol_ratio_expand_threshold,
        vol_ratio_contract_threshold=inp.vol_ratio_contract_threshold,
        vol_ratio_normal_threshold=inp.vol_ratio_normal_threshold,
    )
    _validate_bt_single_inputs(single_validation)
    strat = str(inp.strategy or "ma_filter").strip().lower()
    need_hist = max(
        int(getattr(inp, "sma_window", 20) or 20),
        int(getattr(inp, "slow_window", 20) or 20),
        int(getattr(inp, "donchian_entry", 20) or 20),
        int(getattr(inp, "mom_lookback", 252) or 252),
        int(getattr(inp, "macd_slow", 26) or 26),
        int(getattr(inp, "macd_v_atr_window", 14) or 14),
        20,
    ) + 60
    ext_start = inp.start - dt.timedelta(days=int(need_hist) * 2)

    nav_map: dict[str, pd.Series] = {}
    weight_map: dict[str, pd.Series] = {}
    ret_exec_map: dict[str, pd.Series] = {}
    ret_hfq_map: dict[str, pd.Series] = {}
    px_exec_slip_map: dict[str, pd.Series] = {}
    sig_open_map: dict[str, pd.Series] = {}
    sig_close_map: dict[str, pd.Series] = {}
    score_map: dict[str, pd.Series] = {}
    trades: list[dict[str, Any]] = []
    failures: list[str] = []
    signal_debug_by_code: dict[str, dict[str, Any]] = {}
    semantic_debug_by_code: dict[str, dict[str, Any]] = {}
    price_sig_by_code: dict[str, pd.DataFrame] = {}
    corporate_actions_rows: list[dict[str, Any]] = []

    for c in codes:
        seed_base = getattr(inp, "random_seed", 42)
        code_seed = None if seed_base is None else (int(seed_base) + _stable_code_seed(c)) % 2_147_483_647
        single_inp = TrendInputs(
            code=c,
            start=ext_start,
            end=inp.end,
            risk_free_rate=inp.risk_free_rate,
            cost_bps=inp.cost_bps,
            slippage_rate=inp.slippage_rate,
            exec_price=inp.exec_price,
            strategy=inp.strategy,
            sma_window=inp.sma_window,
            fast_window=inp.fast_window,
            slow_window=inp.slow_window,
            ma_type=inp.ma_type,
            kama_er_window=inp.kama_er_window,
            kama_fast_window=inp.kama_fast_window,
            kama_slow_window=inp.kama_slow_window,
            kama_std_window=inp.kama_std_window,
            kama_std_coef=inp.kama_std_coef,
            donchian_entry=inp.donchian_entry,
            donchian_exit=inp.donchian_exit,
            mom_lookback=inp.mom_lookback,
            tsmom_entry_threshold=inp.tsmom_entry_threshold,
            tsmom_exit_threshold=inp.tsmom_exit_threshold,
            bias_ma_window=inp.bias_ma_window,
            bias_entry=inp.bias_entry,
            bias_hot=inp.bias_hot,
            bias_cold=inp.bias_cold,
            bias_pos_mode=inp.bias_pos_mode,
            macd_fast=inp.macd_fast,
            macd_slow=inp.macd_slow,
            macd_signal=inp.macd_signal,
            macd_v_atr_window=inp.macd_v_atr_window,
            macd_v_scale=inp.macd_v_scale,
            random_hold_days=inp.random_hold_days,
            random_seed=code_seed,
            er_filter=inp.er_filter,
            er_window=inp.er_window,
            er_threshold=inp.er_threshold,
            impulse_entry_filter=inp.impulse_entry_filter,
            impulse_allow_bull=inp.impulse_allow_bull,
            impulse_allow_bear=inp.impulse_allow_bear,
            impulse_allow_neutral=inp.impulse_allow_neutral,
            er_exit_filter=inp.er_exit_filter,
            er_exit_window=inp.er_exit_window,
            er_exit_threshold=inp.er_exit_threshold,
            atr_stop_mode=inp.atr_stop_mode,
            atr_stop_atr_basis=inp.atr_stop_atr_basis,
            atr_stop_reentry_mode=inp.atr_stop_reentry_mode,
            atr_stop_window=inp.atr_stop_window,
            atr_stop_n=inp.atr_stop_n,
            atr_stop_m=inp.atr_stop_m,
            r_take_profit_enabled=inp.r_take_profit_enabled,
            r_take_profit_reentry_mode=inp.r_take_profit_reentry_mode,
            r_take_profit_tiers=inp.r_take_profit_tiers,
            bias_v_take_profit_enabled=inp.bias_v_take_profit_enabled,
            bias_v_take_profit_reentry_mode=inp.bias_v_take_profit_reentry_mode,
            bias_v_ma_window=inp.bias_v_ma_window,
            bias_v_atr_window=inp.bias_v_atr_window,
            bias_v_take_profit_threshold=inp.bias_v_take_profit_threshold,
            monthly_risk_budget_enabled=inp.monthly_risk_budget_enabled,
            monthly_risk_budget_pct=inp.monthly_risk_budget_pct,
            # Portfolio semantics: monthly risk budget gate is applied once at
            # portfolio decision level, not inside each per-asset signal leg.
            monthly_risk_budget_include_new_trade_risk=False,
            # Keep per-asset signal generation in equal mode; portfolio-level sizing
            # remains handled by the portfolio engine path.
            position_sizing="equal",
            fixed_pos_ratio=inp.fixed_pos_ratio,
            fixed_overcap_policy=inp.fixed_overcap_policy,
            fixed_max_holdings=inp.fixed_max_holdings,
            risk_budget_atr_window=inp.risk_budget_atr_window,
            risk_budget_pct=inp.risk_budget_pct,
            vol_regime_risk_mgmt_enabled=inp.vol_regime_risk_mgmt_enabled,
            vol_ratio_fast_atr_window=inp.vol_ratio_fast_atr_window,
            vol_ratio_slow_atr_window=inp.vol_ratio_slow_atr_window,
            vol_ratio_expand_threshold=inp.vol_ratio_expand_threshold,
            vol_ratio_contract_threshold=inp.vol_ratio_contract_threshold,
            vol_ratio_normal_threshold=inp.vol_ratio_normal_threshold,
            group_enforce=inp.group_enforce,
            group_pick_policy=inp.group_pick_policy,
            group_max_holdings=inp.group_max_holdings,
            asset_groups=inp.asset_groups,
            quick_mode=inp.quick_mode,
        )
        # Keep per-asset signal generation free of portfolio-level monthly gate.
        single_inp = TrendInputs(**{**asdict(single_inp), "monthly_risk_budget_enabled": False})
        try:
            one = _run_single_backtesting(db, single_inp, code=c, random_seed=code_seed)
        except ValueError as exc:
            failures.append(f"{c}:{exc}")
            continue
        trim_ix = one["nav"].index[(one["nav"].index.date >= inp.start) & (one["nav"].index.date <= inp.end)]
        if len(trim_ix) == 0:
            failures.append(f"{c}:no rows in requested date range")
            continue
        one = {
            **one,
            "dates": [d.strftime("%Y-%m-%d") for d in trim_ix],
            "nav": one["nav"].reindex(trim_ix).astype(float),
            "buy_hold_nav": one["buy_hold_nav"].reindex(trim_ix).astype(float),
            "excess_nav": one["excess_nav"].reindex(trim_ix).astype(float),
            "strat_ret": one["strat_ret"].reindex(trim_ix).fillna(0.0).astype(float),
            "bench_ret": one["bench_ret"].reindex(trim_ix).fillna(0.0).astype(float),
            "excess_ret": one["excess_ret"].reindex(trim_ix).fillna(0.0).astype(float),
            "desired_pos": one["desired_pos"].reindex(trim_ix).fillna(0.0).astype(float),
            "base_pos": one["base_pos"].reindex(trim_ix).fillna(0.0).astype(float),
            "signal_pos": one["signal_pos"].reindex(trim_ix).fillna(0.0).astype(float),
            "sig_open": one["sig_open"].reindex(trim_ix).ffill().astype(float),
            "sig_close": one["sig_close"].reindex(trim_ix).ffill().astype(float),
            "sig_high": one["sig_high"].reindex(trim_ix).ffill().astype(float),
            "sig_low": one["sig_low"].reindex(trim_ix).ffill().astype(float),
            "ret_exec": one["ret_exec"].reindex(trim_ix).fillna(0.0).astype(float),
            "ret_exec_raw": one["ret_exec_raw"].reindex(trim_ix).fillna(0.0).astype(float),
            "px_exec_slip": one["px_exec_slip"].reindex(trim_ix).ffill().astype(float),
            "ret_exec_none": one["ret_exec_none"].reindex(trim_ix).fillna(0.0).astype(float),
            "ret_exec_hfq": one["ret_exec_hfq"].reindex(trim_ix).fillna(0.0).astype(float),
            "exec_open_none": one["exec_open_none"].reindex(trim_ix).astype(float),
            "exec_close_none": one["exec_close_none"].reindex(trim_ix).astype(float),
            "exec_open_hfq": one["exec_open_hfq"].reindex(trim_ix).astype(float),
            "exec_close_hfq": one["exec_close_hfq"].reindex(trim_ix).astype(float),
            "corp_factor": one["corp_factor"].reindex(trim_ix).astype(float),
            "ca_mask": one["ca_mask"].reindex(trim_ix).fillna(False).astype(bool),
            "trades": [
                t
                for t in list(one.get("trades") or [])
                if str(t.get("entry_date") or "") >= str(inp.start)
            ],
        }
        nav_map[c] = one["nav"]
        weight_map[c] = one["desired_pos"].astype(float)
        ep_port = str(getattr(inp, "exec_price", "open") or "open").strip().lower()
        ps_port = str(getattr(inp, "position_sizing", "equal") or "equal").strip().lower()
        if ep_port == "open" and ps_port == "risk_budget":
            ret_exec_map[c] = one["ret_exec"].astype(float)
            ret_hfq_map[c] = one["ret_exec_hfq"].astype(float)
            px_exec_slip_map[c] = one["px_exec_slip"].astype(float)
            sig_open_map[c] = one["sig_open"].astype(float)
            sig_close_map[c] = one["sig_close"].astype(float)
            score_map[c] = one.get("signal_score", pd.Series(np.nan, index=one["nav"].index)).astype(float)
            trades.extend(one["trades"])
            runtime_engine = str(one.get("runtime_engine") or "unknown")
            signal_debug_by_code[c] = dict(one.get("signal_debug") or {})
            semantic_debug_by_code[c] = dict(one.get("semantic_stats") or {})
            price_sig_by_code[c] = pd.DataFrame(
                {
                    "close": one["sig_close"].astype(float),
                    "high": one["sig_high"].astype(float),
                    "low": one["sig_low"].astype(float),
                }
            )
            ca_mask_s = one.get("ca_mask", pd.Series(False, index=one["nav"].index))
            for d in one["nav"].index:
                if bool(ca_mask_s.loc[d]):
                    corporate_actions_rows.append(
                        {
                            "date": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
                            "code": str(c),
                            "none_return": float(one["ret_exec_none"].loc[d]),
                            "hfq_return": float(one["ret_exec_hfq"].loc[d]),
                            "corp_factor": (
                                float(one["corp_factor"].loc[d]) if np.isfinite(float(one["corp_factor"].loc[d])) else None
                            ),
                        }
                    )
            continue
        open_none = one["exec_open_none"].astype(float)
        close_none = one["exec_close_none"].astype(float)
        open_hfq = one["exec_open_hfq"].astype(float)
        close_hfq = one["exec_close_hfq"].astype(float)
        if ep_port == "open":
            ret_none_one = (open_none.shift(-1).div(open_none) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            ret_hfq_one = (open_hfq.shift(-1).div(open_hfq) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            px_none_one = open_none.astype(float)
            px_hfq_one = open_hfq.astype(float)
        elif ep_port == "close":
            ret_none_one = (close_none.shift(-1).div(close_none) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            ret_hfq_one = (close_hfq.shift(-1).div(close_hfq) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            px_none_one = close_none.astype(float)
            px_hfq_one = close_hfq.astype(float)
        else:
            ret_open_none_one = (open_none.shift(-1).div(open_none) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            ret_close_none_one = (close_none.shift(-1).div(close_none) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            ret_none_one = (0.5 * (ret_open_none_one + ret_close_none_one)).astype(float)
            ret_open_hfq_one = (open_hfq.shift(-1).div(open_hfq) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            ret_close_hfq_one = (close_hfq.shift(-1).div(close_hfq) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            ret_hfq_one = (0.5 * (ret_open_hfq_one + ret_close_hfq_one)).astype(float)
            px_none_one = (0.5 * (open_none + close_none)).astype(float)
            px_hfq_one = (0.5 * (open_hfq + close_hfq)).astype(float)
        gross_none_one = (1.0 + ret_none_one).astype(float).to_frame(c)
        gross_hfq_one = (1.0 + ret_hfq_one).astype(float).to_frame(c)
        _, ca_mask_one_df = corporate_action_mask(gross_none_one, gross_hfq_one)
        if isinstance(ca_mask_one_df, pd.DataFrame) and c in ca_mask_one_df.columns:
            ca_mask_one = ca_mask_one_df[c].reindex(ret_none_one.index).fillna(False).astype(bool)
        else:
            ca_mask_one = pd.Series(False, index=ret_none_one.index, dtype=bool)
        ret_exec_map[c] = ret_none_one.where(~ca_mask_one, other=ret_hfq_one).astype(float)
        ret_hfq_map[c] = ret_hfq_one.astype(float)
        px_exec_slip_map[c] = px_none_one.where(~ca_mask_one, other=px_hfq_one).replace([np.inf, -np.inf], np.nan).ffill().astype(float)
        sig_open_map[c] = one["sig_open"].astype(float)
        sig_close_map[c] = one["sig_close"].astype(float)
        score_map[c] = one.get("signal_score", pd.Series(np.nan, index=one["nav"].index)).astype(float)
        trades.extend(one["trades"])
        runtime_engine = str(one.get("runtime_engine") or "unknown")
        signal_debug_by_code[c] = dict(one.get("signal_debug") or {})
        semantic_debug_by_code[c] = dict(one.get("semantic_stats") or {})
        price_sig_by_code[c] = pd.DataFrame(
            {
                "close": one["sig_close"].astype(float),
                "high": one["sig_high"].astype(float),
                "low": one["sig_low"].astype(float),
            }
        )
        ca_mask_s = one.get("ca_mask", pd.Series(False, index=one["nav"].index))
        for d in one["nav"].index:
            if bool(ca_mask_s.loc[d]):
                corporate_actions_rows.append(
                    {
                        "date": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
                        "code": str(c),
                        "none_return": float(one["ret_exec_none"].loc[d]),
                        "hfq_return": float(one["ret_exec_hfq"].loc[d]),
                        "corp_factor": (
                            float(one["corp_factor"].loc[d]) if np.isfinite(float(one["corp_factor"].loc[d])) else None
                        ),
                    }
                )

    if not nav_map:
        raise ValueError("no valid symbol data for bt trend portfolio")

    dynamic_universe = bool(getattr(inp, "dynamic_universe", False))
    if not dynamic_universe:
        first_valid: list[pd.Timestamp] = []
        for c in codes:
            s = sig_close_map.get(c, pd.Series(dtype=float))
            fv = s.first_valid_index() if isinstance(s, pd.Series) else None
            if fv is None:
                raise ValueError(f"missing execution data (none) for: ['{c}']")
            first_valid.append(pd.Timestamp(fv))
        if not first_valid:
            raise ValueError("no valid first trading date for selected codes")
        common_start = max(first_valid)

        def _trim_series_map(m: dict[str, pd.Series]) -> dict[str, pd.Series]:
            out: dict[str, pd.Series] = {}
            for k, s in m.items():
                if isinstance(s, pd.Series):
                    out[str(k)] = s.loc[s.index >= common_start]
            return out

        nav_map = _trim_series_map(nav_map)
        weight_map = _trim_series_map(weight_map)
        ret_exec_map = _trim_series_map(ret_exec_map)
        ret_hfq_map = _trim_series_map(ret_hfq_map)
        px_exec_slip_map = _trim_series_map(px_exec_slip_map)
        sig_open_map = _trim_series_map(sig_open_map)
        sig_close_map = _trim_series_map(sig_close_map)
        score_map = _trim_series_map(score_map)
        price_sig_by_code = {
            str(k): v.loc[v.index >= common_start].copy()
            for k, v in price_sig_by_code.items()
            if isinstance(v, pd.DataFrame)
        }

    nav_df = pd.DataFrame(nav_map).sort_index()
    wdf = pd.DataFrame(weight_map).reindex(nav_df.index).fillna(0.0)
    score_df = pd.DataFrame(score_map).reindex(index=wdf.index, columns=wdf.columns)
    ret_hfq_df = pd.DataFrame(ret_hfq_map).reindex(index=wdf.index, columns=wdf.columns).fillna(0.0).astype(float)
    group_enforce = bool(getattr(inp, "group_enforce", False))
    group_pick_policy = str(getattr(inp, "group_pick_policy", "highest_sharpe") or "highest_sharpe").strip().lower()
    group_max_holdings = int(getattr(inp, "group_max_holdings", 4) or 4)
    group_map = {
        str(k).strip(): str(v).strip()
        for k, v in ((getattr(inp, "asset_groups", None) or {}).items())
        if str(k).strip()
    }
    sharpe_like = (
        ret_hfq_df.rolling(window=max(20, int(getattr(inp, "vol_window", 20) or 20)), min_periods=max(10, int(getattr(inp, "vol_window", 20) or 20) // 2)).mean()
        / ret_hfq_df.rolling(window=max(20, int(getattr(inp, "vol_window", 20) or 20)), min_periods=max(10, int(getattr(inp, "vol_window", 20) or 20) // 2))
        .std(ddof=1)
        .replace(0.0, np.nan)
    ) * np.sqrt(252.0)
    group_filter_meta_by_date: dict[pd.Timestamp, dict[str, Any]] = {}
    if group_enforce:
        prev_group_holdings: set[str] = set()
        for d in wdf.index:
            row = wdf.loc[d].astype(float)
            score_row = score_df.loc[d].astype(float).replace([np.inf, -np.inf], np.nan)
            scores = score_row.where(row > 1e-12, other=np.nan)
            active_raw = [str(c) for c in scores.dropna().sort_values(ascending=False).index.tolist()]
            reduced, group_meta = _reduce_active_codes_by_group(
                active_codes=active_raw,
                score_row=scores,
                sharpe_row=(sharpe_like.loc[d] if d in sharpe_like.index else pd.Series(dtype=float)),
                group_enforce=group_enforce,
                asset_groups=group_map,
                group_pick_policy=group_pick_policy,
                group_max_holdings=group_max_holdings,
                current_holdings=prev_group_holdings,
            )
            group_filter_meta_by_date[pd.Timestamp(d)] = dict(group_meta or {})
            mask = pd.Series(0.0, index=wdf.columns, dtype=float)
            for c in reduced:
                if c in mask.index:
                    mask.loc[c] = 1.0
            wdf.loc[d] = row.mul(mask, fill_value=0.0).to_numpy(dtype=float)
            prev_group_holdings = set(str(c) for c in reduced)
    ps = str(getattr(inp, "position_sizing", "equal") or "equal").strip().lower()
    if ps in {"equal", "vol_target"}:
        w_decision = pd.DataFrame(0.0, index=wdf.index, columns=wdf.columns, dtype=float)
        if ps == "equal":
            for d in wdf.index:
                row = wdf.loc[d].astype(float)
                score_row = score_df.loc[d].astype(float).replace([np.inf, -np.inf], np.nan)
                scores = score_row.where(row > 1e-12, other=np.nan)
                active = [str(c) for c in scores.dropna().sort_values(ascending=False).index.tolist()]
                if not active:
                    continue
                per = 1.0 / float(len(active))
                for c in active:
                    w_decision.loc[d, c] = float(per)
        else:
            vol_window = int(getattr(inp, "vol_window", 20) or 20)
            vol_ann = ret_hfq_df.rolling(
                window=vol_window,
                min_periods=max(3, vol_window // 2),
            ).std(ddof=1) * np.sqrt(252.0)
            vol_target_ann = float(getattr(inp, "vol_target_ann", 0.20) or 0.20)
            for d in wdf.index:
                row = wdf.loc[d].astype(float)
                score_row = score_df.loc[d].astype(float).replace([np.inf, -np.inf], np.nan)
                scores = score_row.where(row > 1e-12, other=np.nan)
                active = [str(c) for c in scores.dropna().sort_values(ascending=False).index.tolist()]
                if not active:
                    continue
                inv: dict[str, float] = {}
                for c in active:
                    av = float(vol_ann.loc[d, c]) if (c in vol_ann.columns and d in vol_ann.index) else float("nan")
                    inv[c] = (1.0 / av) if (np.isfinite(av) and av > 0.0) else 0.0
                den = float(sum(inv.values()))
                if den > 0.0:
                    raw = {c: (float(v) / den) for c, v in inv.items()}
                    port_vol = float(
                        np.sqrt(
                            np.sum(
                                [
                                    (raw[c] ** 2)
                                    * (
                                        (float(vol_ann.loc[d, c]) if np.isfinite(float(vol_ann.loc[d, c])) else 0.0)
                                        ** 2
                                    )
                                    for c in active
                                ]
                            )
                        )
                    )
                    scale = 1.0 if port_vol <= 1e-12 else min(1.0, float(vol_target_ann) / port_vol)
                    for c in active:
                        w_decision.loc[d, c] = float(raw[c] * scale)
                else:
                    per = 1.0 / float(len(active))
                    for c in active:
                        w_decision.loc[d, c] = float(per)
        wdf = w_decision.astype(float)
    if ps == "risk_budget":
        # Keep binary active signals here; portfolio risk-budget sizing is
        # applied below with a stateful day-by-day loop.
        wdf = wdf.astype(float).clip(lower=0.0)
    overcap_scale_by_code = {str(c): 0 for c in wdf.columns}
    overcap_scale_total = 0
    overcap_skip_decision_total = 0
    overcap_skip_episode_total = 0
    overcap_skip_decision_by_code = {str(c): 0 for c in wdf.columns}
    overcap_skip_episode_by_code = {str(c): 0 for c in wdf.columns}
    overcap_skip_episode_active = {str(c): False for c in wdf.columns}
    overcap_replace_total = 0
    overcap_replace_out_by_code = {str(c): 0 for c in wdf.columns}
    overcap_replace_in_by_code = {str(c): 0 for c in wdf.columns}
    overcap_leverage_usage_total = 0
    overcap_leverage_usage_by_code = {str(c): 0 for c in wdf.columns}
    overcap_leverage_max_multiple = 0.0
    overcap_leverage_max_multiple_by_code = {str(c): 0.0 for c in wdf.columns}
    risk_budget_overcap_daily_counts: dict[str, dict[str, Any]] = {}
    fixed_ext_events: list[dict[str, Any]] = []
    fixed_skip_events: list[dict[str, Any]] = []
    if ps == "fixed_ratio":
        fixed_ratio = float(getattr(inp, "fixed_pos_ratio", 0.04) or 0.04)
        fixed_max_holding_n = int(getattr(inp, "fixed_max_holdings", 10) or 10)
        fixed_overcap_policy = str(getattr(inp, "fixed_overcap_policy", "skip") or "skip").strip().lower()
        prev_fixed_w = pd.Series(0.0, index=wdf.columns, dtype=float)
        for d in wdf.index:
            row = wdf.loc[d].astype(float)
            score_row = score_df.loc[d].astype(float).replace([np.inf, -np.inf], np.nan)
            scores = score_row.where(row > 1e-12, other=np.nan)
            active_codes = [str(c) for c in scores.dropna().sort_values(ascending=False).index.tolist()]
            active_set = set(active_codes)
            w_row = prev_fixed_w.copy().astype(float).reindex(wdf.columns).fillna(0.0)
            for c in wdf.columns:
                key = str(c)
                if (float(w_row.loc[c]) > 1e-12) and (key not in active_set):
                    w_row.loc[c] = 0.0
            for key in active_set:
                if key in w_row.index and float(w_row.loc[key]) > 1e-12:
                    w_row.loc[key] = float(fixed_ratio)
            for key in active_codes:
                if key in w_row.index and float(w_row.loc[key]) > 1e-12:
                    continue
                cur_total = float(w_row.sum())
                proposed_total = float(cur_total + fixed_ratio)
                cur_count = int((w_row > 1e-12).sum())
                proposed_count = int(cur_count + 1)
                over_weight = bool(proposed_total > 1.0 + 1e-12)
                over_count = bool(proposed_count > fixed_max_holding_n)
                if over_weight or over_count:
                    evt = {
                        "date": pd.Timestamp(d).date().isoformat(),
                        "code": str(key),
                        "current_total": float(cur_total),
                        "proposed_total": float(proposed_total),
                        "current_count": int(cur_count),
                        "proposed_count": int(proposed_count),
                        "fixed_max_holdings": int(fixed_max_holding_n),
                        "fixed_pos_ratio": float(fixed_ratio),
                        "over_weight": bool(over_weight),
                        "over_count": bool(over_count),
                    }
                    if fixed_overcap_policy == "skip":
                        fixed_skip_events.append(evt)
                        continue
                    fixed_ext_events.append(evt)
                if key in w_row.index:
                    w_row.loc[key] = float(fixed_ratio)
            wdf.loc[d] = w_row.to_numpy(dtype=float)
            prev_fixed_w = w_row.copy()
    if ps == "risk_budget":
        policy = str(getattr(inp, "risk_budget_overcap_policy", "scale") or "scale").strip().lower()
        max_lev = float(getattr(inp, "risk_budget_max_leverage_multiple", 2.0) or 2.0)
        if (not np.isfinite(max_lev)) or max_lev <= 1.0:
            max_lev = 2.0
        eps = 1e-12
        risk_budget_pct = float(getattr(inp, "risk_budget_pct", 0.01) or 0.01)
        risk_budget_atr_window = int(getattr(inp, "risk_budget_atr_window", 20) or 20)
        vol_regime_risk_mgmt_enabled = bool(getattr(inp, "vol_regime_risk_mgmt_enabled", False))
        vol_ratio_fast_atr_window = int(getattr(inp, "vol_ratio_fast_atr_window", 5) or 5)
        vol_ratio_slow_atr_window = int(getattr(inp, "vol_ratio_slow_atr_window", 50) or 50)
        vol_ratio_expand_threshold = float(getattr(inp, "vol_ratio_expand_threshold", 1.45) or 1.45)
        vol_ratio_contract_threshold = float(getattr(inp, "vol_ratio_contract_threshold", 0.65) or 0.65)
        vol_ratio_normal_threshold = float(getattr(inp, "vol_ratio_normal_threshold", 1.05) or 1.05)

        atr_budget_df = pd.DataFrame(index=wdf.index, columns=wdf.columns, dtype=float)
        atr_ratio_fast_df = pd.DataFrame(index=wdf.index, columns=wdf.columns, dtype=float)
        atr_ratio_slow_df = pd.DataFrame(index=wdf.index, columns=wdf.columns, dtype=float)
        for c in wdf.columns:
            pxc = price_sig_by_code.get(str(c), pd.DataFrame(index=wdf.index))
            cl = pxc.get("close", pd.Series(np.nan, index=wdf.index)).reindex(wdf.index).astype(float)
            hi = pxc.get("high", pd.Series(np.nan, index=wdf.index)).reindex(wdf.index).astype(float).fillna(cl)
            lo = pxc.get("low", pd.Series(np.nan, index=wdf.index)).reindex(wdf.index).astype(float).fillna(cl)
            atr_budget_df[c] = _atr_from_hlc(hi, lo, cl, window=int(risk_budget_atr_window)).astype(float)
            atr_ratio_fast_df[c] = _atr_from_hlc(hi, lo, cl, window=int(vol_ratio_fast_atr_window)).astype(float)
            atr_ratio_slow_df[c] = _atr_from_hlc(hi, lo, cl, window=int(vol_ratio_slow_atr_window)).astype(float)

        prev_rb_w = pd.Series(0.0, index=wdf.columns, dtype=float)
        rb_state_by_code: dict[str, str] = {str(c): "FLAT" for c in wdf.columns}
        rb_entry_price_by_code: dict[str, float] = {str(c): float("nan") for c in wdf.columns}
        rb_entry_seq_by_code: dict[str, int] = {str(c): -1 for c in wdf.columns}
        day_seq = 0
        for d in wdf.index:
            day_seq += 1
            d_key = str(pd.Timestamp(d).date())
            score_row = score_df.loc[d].astype(float).replace([np.inf, -np.inf], np.nan)
            sig_row = wdf.loc[d].astype(float).clip(lower=0.0)
            active_codes = [
                str(c)
                for c in score_row.where(sig_row > eps, other=np.nan).dropna().sort_values(ascending=False).index.tolist()
            ]
            active_set = set(active_codes)
            w_row = prev_rb_w.copy().astype(float).reindex(wdf.columns).fillna(0.0)
            skipped_today: set[str] = set()

            def _inc_overcap_daily(kind: str, n: int = 1) -> None:
                nn = int(n)
                if nn <= 0:
                    return
                row_d = risk_budget_overcap_daily_counts.setdefault(
                    d_key,
                    {
                        "scale": 0,
                        "skip_entry": 0,
                        "replace_entry": 0,
                        "leverage_entry": 0,
                        "leverage_multiple_max": 0.0,
                    },
                )
                row_d[str(kind)] = int(row_d.get(str(kind), 0) + nn)

            def _apply_overcap_scale_once(cap_multiple: float = 1.0) -> None:
                nonlocal w_row, overcap_scale_total
                cap_v = float(cap_multiple) if np.isfinite(float(cap_multiple)) and float(cap_multiple) > 0.0 else 1.0
                s_now = float(w_row.sum())
                if s_now <= cap_v + eps:
                    return
                pre_scale = w_row.copy().astype(float)
                w_row = (w_row * (cap_v / s_now)).astype(float)
                overcap_scale_total += 1
                _inc_overcap_daily("scale", 1)
                for cc in w_row.index:
                    key_cc = str(cc)
                    before = float(pre_scale.loc[cc]) if np.isfinite(float(pre_scale.loc[cc])) else 0.0
                    after = float(w_row.loc[cc]) if np.isfinite(float(w_row.loc[cc])) else 0.0
                    if before > after + eps:
                        overcap_scale_by_code[key_cc] = int(overcap_scale_by_code.get(key_cc, 0) + 1)

            def _set_new_risk_budget_entry(key: str, base_target: float) -> None:
                nonlocal w_row
                w_row.loc[key] = float(base_target)
                px_now = float(price_sig_by_code.get(key, pd.DataFrame(index=wdf.index)).get("close", pd.Series(np.nan, index=wdf.index)).reindex(wdf.index).loc[d])
                rb_entry_price_by_code[key] = (float(px_now) if np.isfinite(px_now) and px_now > 0.0 else float("nan"))
                rb_entry_seq_by_code[key] = int(day_seq)
                if bool(vol_regime_risk_mgmt_enabled):
                    af = float(atr_ratio_fast_df.loc[d, key]) if (key in atr_ratio_fast_df.columns and d in atr_ratio_fast_df.index) else float("nan")
                    aslow = float(atr_ratio_slow_df.loc[d, key]) if (key in atr_ratio_slow_df.columns and d in atr_ratio_slow_df.index) else float("nan")
                    ratio = (af / aslow) if (np.isfinite(af) and np.isfinite(aslow) and aslow > 0.0) else float("nan")
                    if np.isfinite(ratio) and ratio > float(vol_ratio_expand_threshold):
                        rb_state_by_code[key] = "REDUCED"
                    elif np.isfinite(ratio) and ratio < float(vol_ratio_contract_threshold):
                        rb_state_by_code[key] = "INCREASED"
                    else:
                        rb_state_by_code[key] = "NORMAL"
                else:
                    rb_state_by_code[key] = "NORMAL"

            def _select_replace_out_code(new_code: str) -> str | None:
                cand: list[tuple[float, int, str]] = []
                for cc in wdf.columns:
                    key_cc = str(cc)
                    if key_cc == str(new_code):
                        continue
                    if float(w_row.loc[key_cc]) <= eps:
                        continue
                    cur_px = float(price_sig_by_code.get(key_cc, pd.DataFrame(index=wdf.index)).get("close", pd.Series(np.nan, index=wdf.index)).reindex(wdf.index).loc[d])
                    ent_px = float(rb_entry_price_by_code.get(key_cc, float("nan")))
                    ret = ((cur_px / ent_px) - 1.0) if (np.isfinite(cur_px) and cur_px > 0.0 and np.isfinite(ent_px) and ent_px > 0.0) else float("inf")
                    seq = int(rb_entry_seq_by_code.get(key_cc, 10**9))
                    if seq < 0:
                        seq = 10**9
                    cand.append((float(ret), int(seq), key_cc))
                if not cand:
                    return None
                cand.sort(key=lambda x: (x[0], x[1], x[2]))
                return str(cand[0][2])

            for c in wdf.columns:
                key_c = str(c)
                if (float(w_row.loc[c]) > eps) and (key_c not in active_set):
                    w_row.loc[c] = 0.0
                    rb_state_by_code[key_c] = "FLAT"
                    rb_entry_price_by_code[key_c] = float("nan")
                    rb_entry_seq_by_code[key_c] = -1

            for c in active_codes:
                px = float(price_sig_by_code.get(str(c), pd.DataFrame(index=wdf.index)).get("close", pd.Series(np.nan, index=wdf.index)).reindex(wdf.index).loc[d])
                a = float(atr_budget_df.loc[d, c]) if (c in atr_budget_df.columns and d in atr_budget_df.index) else float("nan")
                base_target = float("nan")
                if np.isfinite(px) and px > 0.0 and np.isfinite(a) and a > 0.0:
                    base_target = float(risk_budget_pct) * float(px) / float(a)
                has_pos = bool(float(w_row.loc[c]) > eps)
                key = str(c)
                if not has_pos:
                    if np.isfinite(base_target) and base_target > 0.0:
                        proposed_total = float(w_row.sum() + float(base_target))
                        overcap_on_new_entry = bool(proposed_total > 1.0 + eps)
                        if overcap_on_new_entry and str(policy) == "skip_entry":
                            overcap_skip_decision_total += 1
                            _inc_overcap_daily("skip_entry", 1)
                            overcap_skip_decision_by_code[key] = int(overcap_skip_decision_by_code.get(key, 0) + 1)
                            skipped_today.add(key)
                            if not bool(overcap_skip_episode_active.get(key, False)):
                                overcap_skip_episode_active[key] = True
                                overcap_skip_episode_total += 1
                                overcap_skip_episode_by_code[key] = int(overcap_skip_episode_by_code.get(key, 0) + 1)
                            continue
                        if overcap_on_new_entry and str(policy) == "replace_entry":
                            out_code = _select_replace_out_code(key)
                            if out_code:
                                w_row.loc[out_code] = 0.0
                                rb_state_by_code[out_code] = "FLAT"
                                rb_entry_price_by_code[out_code] = float("nan")
                                rb_entry_seq_by_code[out_code] = -1
                                overcap_replace_total += 1
                                _inc_overcap_daily("replace_entry", 1)
                                overcap_replace_out_by_code[out_code] = int(overcap_replace_out_by_code.get(out_code, 0) + 1)
                                overcap_replace_in_by_code[key] = int(overcap_replace_in_by_code.get(key, 0) + 1)
                        _set_new_risk_budget_entry(key, float(base_target))
                        if overcap_on_new_entry and str(policy) == "replace_entry":
                            _apply_overcap_scale_once()
                        elif overcap_on_new_entry and str(policy) == "leverage_entry":
                            lev_now = float(w_row.sum())
                            if lev_now > 1.0 + eps:
                                overcap_leverage_usage_total += 1
                                _inc_overcap_daily("leverage_entry", 1)
                                landed_lev = float(min(float(lev_now), float(max_lev)))
                                overcap_leverage_max_multiple = float(max(float(overcap_leverage_max_multiple), landed_lev))
                                row_d = risk_budget_overcap_daily_counts.setdefault(
                                    d_key,
                                    {
                                        "scale": 0,
                                        "skip_entry": 0,
                                        "replace_entry": 0,
                                        "leverage_entry": 0,
                                        "leverage_multiple_max": 0.0,
                                    },
                                )
                                row_d["leverage_multiple_max"] = float(max(float(row_d.get("leverage_multiple_max", 0.0) or 0.0), landed_lev))
                                for cc in wdf.columns:
                                    key_cc = str(cc)
                                    if float(w_row.loc[key_cc]) > eps:
                                        overcap_leverage_usage_by_code[key_cc] = int(overcap_leverage_usage_by_code.get(key_cc, 0) + 1)
                                        overcap_leverage_max_multiple_by_code[key_cc] = float(
                                            max(float(overcap_leverage_max_multiple_by_code.get(key_cc, 0.0)), landed_lev)
                                        )
                                if lev_now > float(max_lev) + eps:
                                    _apply_overcap_scale_once(float(max_lev))
                    continue

                if not bool(vol_regime_risk_mgmt_enabled):
                    continue
                st = str(rb_state_by_code.get(key, "NORMAL") or "NORMAL").upper()
                af = float(atr_ratio_fast_df.loc[d, c]) if (c in atr_ratio_fast_df.columns and d in atr_ratio_fast_df.index) else float("nan")
                aslow = float(atr_ratio_slow_df.loc[d, c]) if (c in atr_ratio_slow_df.columns and d in atr_ratio_slow_df.index) else float("nan")
                ratio = (af / aslow) if (np.isfinite(af) and np.isfinite(aslow) and aslow > 0.0) else float("nan")
                if not np.isfinite(base_target):
                    continue
                if st == "NORMAL":
                    if np.isfinite(ratio) and ratio > float(vol_ratio_expand_threshold):
                        w_row.loc[c] = float(base_target)
                        rb_state_by_code[key] = "REDUCED"
                    elif np.isfinite(ratio) and ratio < float(vol_ratio_contract_threshold):
                        w_row.loc[c] = float(base_target)
                        rb_state_by_code[key] = "INCREASED"
                elif st == "REDUCED":
                    if np.isfinite(ratio) and ratio < float(vol_ratio_normal_threshold):
                        w_row.loc[c] = float(base_target)
                        rb_state_by_code[key] = "NORMAL"
                elif st == "INCREASED":
                    if np.isfinite(ratio) and ratio > float(vol_ratio_normal_threshold):
                        w_row.loc[c] = float(base_target)
                        rb_state_by_code[key] = "NORMAL"

            w_row = w_row.clip(lower=0.0)
            if str(policy) != "leverage_entry":
                _apply_overcap_scale_once()
            for key_cc in wdf.columns:
                kk = str(key_cc)
                if bool(overcap_skip_episode_active.get(kk, False)) and (kk not in skipped_today):
                    overcap_skip_episode_active[kk] = False
            wdf.loc[d] = w_row.to_numpy(dtype=float)
            prev_rb_w = w_row.copy()

    monthly_attempted_total = 0
    monthly_blocked_total = 0
    if bool(getattr(inp, "monthly_risk_budget_enabled", False)):
        close_df = pd.DataFrame(
            {
                c: price_sig_by_code.get(c, pd.DataFrame(index=wdf.index)).get("close", pd.Series(np.nan, index=wdf.index)).reindex(wdf.index).astype(float)
                for c in wdf.columns
            },
            index=wdf.index,
        )
        atr_df = pd.DataFrame(index=wdf.index)
        for c in wdf.columns:
            pxc = price_sig_by_code.get(c, pd.DataFrame(index=wdf.index))
            atr_df[c] = _atr_from_hlc(
                pxc.get("high", pd.Series(np.nan, index=wdf.index)).reindex(wdf.index).astype(float),
                pxc.get("low", pd.Series(np.nan, index=wdf.index)).reindex(wdf.index).astype(float),
                pxc.get("close", pd.Series(np.nan, index=wdf.index)).reindex(wdf.index).astype(float),
                window=int(getattr(inp, "atr_stop_window", 14) or 14),
            ).reindex(wdf.index).astype(float)
        wdf, gate_stats = _apply_monthly_risk_budget_gate(
            wdf.astype(float),
            close=close_df.astype(float),
            atr=atr_df.astype(float),
            enabled=True,
            budget_pct=float(getattr(inp, "monthly_risk_budget_pct", 0.06) or 0.06),
            include_new_trade_risk=bool(getattr(inp, "monthly_risk_budget_include_new_trade_risk", False)),
            atr_stop_enabled=bool(str(getattr(inp, "atr_stop_mode", "none") or "none").strip().lower() != "none"),
            atr_mode=str(getattr(inp, "atr_stop_mode", "none") or "none"),
            atr_basis=str(getattr(inp, "atr_stop_atr_basis", "latest") or "latest"),
            atr_n=float(getattr(inp, "atr_stop_n", 2.0) or 2.0),
            atr_m=float(getattr(inp, "atr_stop_m", 0.5) or 0.5),
            fallback_position_risk=0.02,
        )
        monthly_attempted_total = int((gate_stats or {}).get("attempted_entry_count", 0))
        monthly_blocked_total = int((gate_stats or {}).get("blocked_entry_count", 0))

    ret_exec_df = pd.DataFrame(ret_exec_map).reindex(index=wdf.index, columns=wdf.columns).fillna(0.0).astype(float)
    px_exec_slip_df = pd.DataFrame(px_exec_slip_map).reindex(index=wdf.index, columns=wdf.columns).ffill().astype(float)
    open_sig_df = pd.DataFrame(sig_open_map).reindex(index=wdf.index, columns=wdf.columns).ffill().astype(float)
    close_sig_df = pd.DataFrame(sig_close_map).reindex(index=wdf.index, columns=wdf.columns).ffill().astype(float)
    ret_overnight_df_comp: pd.DataFrame | None = None
    ret_intraday_df_comp: pd.DataFrame | None = None
    try:
        ccodes = [str(c) for c in wdf.columns]
        idx = wdf.index
        ep_port = str(getattr(inp, "exec_price", "open") or "open").strip().lower()
        ps_port = str(getattr(inp, "position_sizing", "equal") or "equal").strip().lower()
        if ep_port == "open" and ps_port == "risk_budget" and (not bool(getattr(inp, "quick_mode", False))):
            raise RuntimeError("keep single-derived mapping for open+risk_budget parity")
        close_none_df = load_close_prices(db, codes=ccodes, start=inp.start, end=inp.end, adjust="none").reindex(index=idx, columns=ccodes).ffill().astype(float)
        close_hfq_df = load_close_prices(db, codes=ccodes, start=inp.start, end=inp.end, adjust="hfq").reindex(index=idx, columns=ccodes).ffill().astype(float)
        close_qfq_df = load_close_prices(db, codes=ccodes, start=inp.start, end=inp.end, adjust="qfq").reindex(index=idx, columns=ccodes).ffill().astype(float)
        ohlc_none = load_ohlc_prices(db, codes=ccodes, start=inp.start, end=inp.end, adjust="none")
        ohlc_hfq = load_ohlc_prices(db, codes=ccodes, start=inp.start, end=inp.end, adjust="hfq")
        ohlc_qfq = load_ohlc_prices(db, codes=ccodes, start=inp.start, end=inp.end, adjust="qfq")
        def _raw_df(ohlc: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
            df = ohlc.get(field, pd.DataFrame()) if isinstance(ohlc, dict) else pd.DataFrame()
            if df is None or df.empty:
                return pd.DataFrame(index=idx, columns=ccodes, dtype=float)
            return (
                df.sort_index()
                .reindex(index=idx, columns=ccodes)
                .astype(float)
                .ffill()
            )

        open_none_raw = _raw_df(ohlc_none, "open")
        close_none_raw = _raw_df(ohlc_none, "close")
        open_hfq_raw = _raw_df(ohlc_hfq, "open")
        close_hfq_raw = _raw_df(ohlc_hfq, "close")
        open_qfq_raw = _raw_df(ohlc_qfq, "open")
        close_qfq_raw = _raw_df(ohlc_qfq, "close")
        open_none_exec = open_none_raw.combine_first(close_none_df)
        close_none_exec = close_none_raw.combine_first(close_none_df)
        open_hfq_exec = open_hfq_raw.combine_first(close_hfq_df)
        close_hfq_exec = close_hfq_raw.combine_first(close_hfq_df)
        if ep_port == "open":
            ret_none_base = (open_none_exec.shift(-1).div(open_none_exec) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            ret_hfq_base = (open_hfq_exec.shift(-1).div(open_hfq_exec) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            px_none_base = open_none_exec.astype(float)
            px_hfq_base = open_hfq_exec.astype(float)
        elif ep_port == "close":
            ret_none_base = (close_none_exec.shift(-1).div(close_none_exec) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            ret_hfq_base = (close_hfq_exec.shift(-1).div(close_hfq_exec) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            px_none_base = close_none_exec.astype(float)
            px_hfq_base = close_hfq_exec.astype(float)
        else:
            ret_open_none_base = (open_none_exec.shift(-1).div(open_none_exec) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            ret_close_none_base = (close_none_exec.shift(-1).div(close_none_exec) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            ret_none_base = (0.5 * (ret_open_none_base + ret_close_none_base)).astype(float)
            ret_open_hfq_base = (open_hfq_exec.shift(-1).div(open_hfq_exec) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            ret_close_hfq_base = (close_hfq_exec.shift(-1).div(close_hfq_exec) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            ret_hfq_base = (0.5 * (ret_open_hfq_base + ret_close_hfq_base)).astype(float)
            px_none_base = (0.5 * (open_none_exec + close_none_exec)).astype(float)
            px_hfq_base = (0.5 * (open_hfq_exec + close_hfq_exec)).astype(float)
        _, ca_mask_base = corporate_action_mask((1.0 + ret_none_base).astype(float), (1.0 + ret_hfq_base).astype(float))
        ca_mask_base = ca_mask_base.reindex(index=idx, columns=ccodes).fillna(False).astype(bool)
        ret_exec_df = ret_none_base.where(~ca_mask_base, other=ret_hfq_base).astype(float)
        px_exec_slip_df = px_none_base.where(~ca_mask_base, other=px_hfq_base).replace([np.inf, -np.inf], np.nan).ffill().astype(float)
        open_sig_df = open_qfq_raw.astype(float)
        close_sig_df = close_qfq_df.astype(float)
        ret_overnight_none_close = (open_none_raw.shift(-1).div(close_none_raw) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_intraday_none_close = (close_none_raw.shift(-1).div(open_none_raw.shift(-1)) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_overnight_hfq_close = (open_hfq_raw.shift(-1).div(close_hfq_raw) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_intraday_hfq_close = (close_hfq_raw.shift(-1).div(open_hfq_raw.shift(-1)) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_intraday_none_open = (close_none_raw.div(open_none_raw) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_overnight_none_open = (open_none_raw.shift(-1).div(close_none_raw) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_intraday_hfq_open = (close_hfq_raw.div(open_hfq_raw) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_overnight_hfq_open = (open_hfq_raw.shift(-1).div(close_hfq_raw) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_overnight_close = ret_overnight_none_close.where(~ca_mask_base, other=ret_overnight_hfq_close).astype(float)
        ret_intraday_close = ret_intraday_none_close.where(~ca_mask_base, other=ret_intraday_hfq_close).astype(float)
        ret_overnight_open = ret_overnight_none_open.where(~ca_mask_base, other=ret_overnight_hfq_open).astype(float)
        ret_intraday_open = ret_intraday_none_open.where(~ca_mask_base, other=ret_intraday_hfq_open).astype(float)
        if ep_port == "open":
            ret_overnight_df_comp = ret_overnight_open.astype(float)
            ret_intraday_df_comp = ret_intraday_open.astype(float)
        elif ep_port == "close":
            ret_overnight_df_comp = ret_overnight_close.astype(float)
            ret_intraday_df_comp = ret_intraday_close.astype(float)
        else:
            ret_overnight_df_comp = (0.5 * ret_overnight_close).astype(float)
            ret_intraday_df_comp = (0.5 * ret_exec_df + 0.5 * ret_intraday_close).astype(float)
    except Exception:
        pass
    atr_stop_by_asset = {str(c): dict((semantic_debug_by_code.get(str(c), {}) or {}).get("atr_stop") or {}) for c in wdf.columns}
    bias_v_tp_by_asset = {str(c): dict((semantic_debug_by_code.get(str(c), {}) or {}).get("bias_v_take_profit") or {}) for c in wdf.columns}
    rtp_by_asset = {str(c): dict((semantic_debug_by_code.get(str(c), {}) or {}).get("r_take_profit") or {}) for c in wdf.columns}

    w_eff = wdf.shift(1).fillna(0.0).astype(float).clip(lower=0.0)
    w_eff, atr_stop_override_ret = _apply_intraday_stop_execution_portfolio(
        weights=w_eff,
        atr_stop_by_asset=atr_stop_by_asset,
        exec_price=str(getattr(inp, "exec_price", "open") or "open"),
        open_sig_df=open_sig_df,
        close_sig_df=close_sig_df,
    )
    w_eff, bias_v_take_profit_override_ret = _apply_intraday_stop_execution_portfolio(
        weights=w_eff,
        atr_stop_by_asset=bias_v_tp_by_asset,
        exec_price=str(getattr(inp, "exec_price", "open") or "open"),
        open_sig_df=open_sig_df,
        close_sig_df=close_sig_df,
    )
    w_eff, r_take_profit_override_ret = _apply_intraday_stop_execution_portfolio(
        weights=w_eff,
        atr_stop_by_asset=rtp_by_asset,
        exec_price=str(getattr(inp, "exec_price", "open") or "open"),
        open_sig_df=open_sig_df,
        close_sig_df=close_sig_df,
    )
    turnover = (w_eff - w_eff.shift(1).fillna(0.0)).abs().sum(axis=1) / 2.0
    turnover_by_asset = (w_eff - w_eff.shift(1).fillna(0.0)).abs() / 2.0
    cost = turnover * (float(getattr(inp, "cost_bps", 0.0) or 0.0) / 10000.0)
    slippage = (
        slippage_return_from_turnover(
            turnover_by_asset.astype(float),
            slippage_spread=float(getattr(inp, "slippage_rate", 0.0) or 0.0),
            exec_price=px_exec_slip_df.astype(float),
        )
        .sum(axis=1)
        .astype(float)
    )
    base_ret = (w_eff * ret_exec_df).sum(axis=1).astype(float)
    decomp_risk = (
        atr_stop_override_ret.reindex(w_eff.index).fillna(0.0).astype(float)
        + bias_v_take_profit_override_ret.reindex(w_eff.index).fillna(0.0).astype(float)
        + r_take_profit_override_ret.reindex(w_eff.index).fillna(0.0).astype(float)
    ).astype(float)
    decomp_cost = (cost + slippage).astype(float)
    port_ret = (base_ret + decomp_risk - decomp_cost).fillna(0.0).astype(float)
    _ep_rt = str(getattr(inp, "exec_price", "open") or "open").strip().lower()
    _ps_rt = str(getattr(inp, "position_sizing", "equal") or "equal").strip().lower()
    _keep_base_port_ret = bool(_ep_rt == "open" and _ps_rt == "risk_budget")
    if ret_overnight_df_comp is None or ret_intraday_df_comp is None:
        ret_overnight_df = (open_sig_df / close_sig_df.shift(1) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        ret_intraday_df = (close_sig_df / open_sig_df - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    else:
        ret_overnight_df = ret_overnight_df_comp.reindex(index=w_eff.index, columns=w_eff.columns).fillna(0.0).astype(float)
        ret_intraday_df = ret_intraday_df_comp.reindex(index=w_eff.index, columns=w_eff.columns).fillna(0.0).astype(float)
    return_decomposition = None
    if not bool(getattr(inp, "quick_mode", False)):
        comp_overnight = (w_eff * ret_overnight_df).sum(axis=1).astype(float)
        comp_intraday = (w_eff * ret_intraday_df).sum(axis=1).astype(float)
        comp_interaction = (w_eff * ret_overnight_df * ret_intraday_df).sum(axis=1).astype(float)
        decomp_gross = (comp_overnight + comp_intraday + comp_interaction + decomp_risk).astype(float)
        decomp_net = (decomp_gross - decomp_cost).astype(float)
        if not _keep_base_port_ret:
            port_ret = decomp_net.fillna(0.0).astype(float)
        return_decomposition = {
            "dates": [d.strftime("%Y-%m-%d") for d in w_eff.index],
            "series": {
                "overnight": comp_overnight.tolist(),
                "intraday": comp_intraday.tolist(),
                "interaction": comp_interaction.tolist(),
                "atr_stop_override": atr_stop_override_ret.reindex(w_eff.index).fillna(0.0).astype(float).tolist(),
                "bias_v_take_profit_override": bias_v_take_profit_override_ret.reindex(w_eff.index).fillna(0.0).astype(float).tolist(),
                "r_take_profit_override": r_take_profit_override_ret.reindex(w_eff.index).fillna(0.0).astype(float).tolist(),
                "risk_exit_override": decomp_risk.tolist(),
                "cost": decomp_cost.tolist(),
                "gross": decomp_gross.tolist(),
                "net": decomp_net.tolist(),
            },
            "summary": {
                "ann_overnight": float(comp_overnight.iloc[1:].mean() * 252.0) if len(comp_overnight) > 1 else 0.0,
                "ann_intraday": float(comp_intraday.iloc[1:].mean() * 252.0) if len(comp_intraday) > 1 else 0.0,
                "ann_interaction": float(comp_interaction.iloc[1:].mean() * 252.0) if len(comp_interaction) > 1 else 0.0,
                "ann_atr_stop_override": float(atr_stop_override_ret.iloc[1:].mean() * 252.0) if len(atr_stop_override_ret) > 1 else 0.0,
                "ann_bias_v_take_profit_override": (
                    float(bias_v_take_profit_override_ret.iloc[1:].mean() * 252.0) if len(bias_v_take_profit_override_ret) > 1 else 0.0
                ),
                "ann_r_take_profit_override": float(r_take_profit_override_ret.iloc[1:].mean() * 252.0) if len(r_take_profit_override_ret) > 1 else 0.0,
                "ann_risk_exit_override": float(decomp_risk.iloc[1:].mean() * 252.0) if len(decomp_risk) > 1 else 0.0,
                "ann_cost": float(decomp_cost.iloc[1:].mean() * 252.0) if len(decomp_cost) > 1 else 0.0,
                "ann_gross": float(decomp_gross.iloc[1:].mean() * 252.0) if len(decomp_gross) > 1 else 0.0,
                "ann_net": float(decomp_net.iloc[1:].mean() * 252.0) if len(decomp_net) > 1 else 0.0,
            },
        }
    nav = _as_nav(port_ret)
    close_hfq = load_close_prices(db, codes=list(nav_map.keys()), start=inp.start, end=inp.end, adjust="hfq").reindex(nav.index).ffill()
    bh_ret = hfq_close_daily_equal_weight_returns(close_hfq, dynamic_universe=dynamic_universe).reindex(nav.index).fillna(0.0)
    bh_nav = _as_nav(bh_ret)
    excess_nav = (nav / bh_nav.replace(0.0, np.nan)).fillna(1.0)
    excess_ret = excess_nav.pct_change().fillna(0.0).astype(float)
    active_ret = (port_ret.reindex(nav.index).astype(float) - bh_ret.reindex(nav.index).astype(float)).fillna(0.0).astype(float)
    event_study = None if bool(getattr(inp, "quick_mode", False)) else compute_event_study(
        dates=nav.index,
        daily_returns=port_ret.reindex(nav.index).astype(float),
        entry_dates=entry_dates_from_exposure(w_eff.sum(axis=1).reindex(nav.index).astype(float)),
    )
    high_sig_df = pd.DataFrame(
        {
            c: price_sig_by_code.get(c, pd.DataFrame(index=nav.index)).get("high", pd.Series(np.nan, index=nav.index)).reindex(nav.index).astype(float)
            for c in w_eff.columns
        },
        index=nav.index,
    )
    low_sig_df = pd.DataFrame(
        {
            c: price_sig_by_code.get(c, pd.DataFrame(index=nav.index)).get("low", pd.Series(np.nan, index=nav.index)).reindex(nav.index).astype(float)
            for c in w_eff.columns
        },
        index=nav.index,
    )
    market_regime = build_market_regime_report(
        close=close_sig_df.reindex(index=nav.index, columns=w_eff.columns).astype(float),
        high=high_sig_df.reindex(index=nav.index, columns=w_eff.columns).astype(float),
        low=low_sig_df.reindex(index=nav.index, columns=w_eff.columns).astype(float),
        weights=w_eff.reindex(index=nav.index, columns=w_eff.columns).astype(float).fillna(0.0),
        asset_returns=ret_exec_df.reindex(index=nav.index, columns=w_eff.columns).astype(float).fillna(0.0),
        strategy_returns=port_ret.reindex(nav.index).astype(float),
        ann_factor=252,
    )
    attribution = _compute_return_risk_contributions(
        asset_ret=ret_exec_df.reindex(index=nav.index, columns=w_eff.columns).astype(float).fillna(0.0),
        weights=w_eff.reindex(index=nav.index, columns=w_eff.columns).astype(float).fillna(0.0),
        total_return=float(nav.iloc[-1] - 1.0) if len(nav) else 0.0,
    )
    holdings: list[dict[str, Any]] = []
    prev_picks: tuple[str, ...] | None = None
    for d in wdf.index:
        picks = tuple(sorted([str(c) for c in wdf.columns if float(wdf.loc[d, c]) > 1e-12]))
        if picks != prev_picks:
            score_row = score_df.loc[d] if d in score_df.index else pd.Series(dtype=float)
            group_meta_raw = dict(group_filter_meta_by_date.get(pd.Timestamp(d), {}) or {})
            group_filter_norm = {
                "enabled": bool(group_meta_raw.get("enabled", False)),
                "policy": str(group_meta_raw.get("policy", getattr(inp, "group_pick_policy", "highest_sharpe") or "highest_sharpe")),
                "max_holdings_per_group": int(
                    group_meta_raw.get("max_holdings_per_group", getattr(inp, "group_max_holdings", 4) or 4)
                ),
                "before": [str(x) for x in (group_meta_raw.get("before") or [])],
                "after": [str(x) for x in (group_meta_raw.get("after") or [])],
                "group_winners": {
                    str(k): [str(x) for x in (v or [])]
                    for k, v in ((group_meta_raw.get("group_winners") or {}) or {}).items()
                },
                "group_eliminated": {
                    str(k): [str(x) for x in (v or [])]
                    for k, v in ((group_meta_raw.get("group_eliminated") or {}) or {}).items()
                },
                "group_picks": {
                    str(k): [str(x) for x in (v or [])]
                    for k, v in ((group_meta_raw.get("group_picks") or {}) or {}).items()
                },
            }
            holdings.append(
                {
                    "decision_date": d.date().isoformat() if hasattr(d, "date") else str(d),
                    "picks": list(picks),
                    "grouped_picks": {
                        str(k): [str(x) for x in (v or [])]
                        for k, v in (group_filter_norm.get("group_picks", {}) or {}).items()
                    },
                    "scores": {
                        str(c): (None if pd.isna(score_row.get(c)) else float(score_row.get(c)))
                        for c in picks
                    },
                    "group_filter": group_filter_norm,
                }
            )
            prev_picks = picks
    group_filter_enabled_segments = int(
        sum(1 for h in holdings if bool(((h or {}).get("group_filter") or {}).get("enabled")))
    )
    group_filter_effective_segments = int(
        sum(
            1
            for h in holdings
            if bool(((h or {}).get("group_filter") or {}).get("enabled"))
            and (
                len((((h or {}).get("group_filter") or {}).get("before") or []))
                > len((((h or {}).get("group_filter") or {}).get("after") or []))
            )
        )
    )
    trade_pack = _trade_returns_from_weight_df(
        w_eff.reindex(index=nav.index, columns=w_eff.columns).astype(float).fillna(0.0),
        ret_exec_df.reindex(index=nav.index, columns=w_eff.columns).astype(float).fillna(0.0),
        cost_bps=float(getattr(inp, "cost_bps", 0.0) or 0.0),
        slippage_rate=float(getattr(inp, "slippage_rate", 0.0) or 0.0),
        exec_price=px_exec_slip_df.reindex(index=nav.index, columns=w_eff.columns).ffill(),
        dates=nav.index,
    )
    atr_risk_df = pd.DataFrame(index=nav.index, columns=w_eff.columns, dtype=float)
    for c in w_eff.columns:
        cl = close_sig_df[c].reindex(nav.index).astype(float)
        hi = high_sig_df[c].reindex(nav.index).astype(float).fillna(cl)
        lo = low_sig_df[c].reindex(nav.index).astype(float).fillna(cl)
        atr_risk_df[c] = _atr_from_hlc(
            hi,
            lo,
            cl,
            window=int(getattr(inp, "atr_stop_window", 14) or 14),
        ).reindex(nav.index).astype(float)
    trade_r_pack = enrich_trades_with_r_metrics(
        trade_pack.get("trades", []),
        nav=nav.astype(float),
        weights=w_eff.reindex(index=nav.index, columns=w_eff.columns).astype(float).fillna(0.0),
        exec_price=px_exec_slip_df.reindex(index=nav.index, columns=w_eff.columns).ffill().astype(float),
        atr=atr_risk_df.reindex(index=nav.index, columns=w_eff.columns).astype(float),
        atr_mult=float(getattr(inp, "atr_stop_n", 2.0) or 2.0),
        risk_budget_pct=(float(getattr(inp, "risk_budget_pct", 0.01) or 0.01) if ps == "risk_budget" else None),
        cost_bps=float(getattr(inp, "cost_bps", 0.0) or 0.0),
        slippage_rate=float(getattr(inp, "slippage_rate", 0.0) or 0.0),
        ulcer_index=float(_ulcer_index(nav, in_percent=True)) if len(nav) else None,
        annual_trade_count=(float(len(trade_pack.get("returns", []))) / max(1.0, float(len(nav))) * 252.0) if len(nav) else None,
        backtest_years=(float(len(nav)) / 252.0) if len(nav) else None,
        score_sqn_weight=0.60,
        score_ulcer_weight=0.40,
    )
    trades_with_r = list(trade_r_pack.get("trades") or [])
    r_stats_out = dict(trade_r_pack.get("statistics") or {})
    r_stats_out.pop("trade_system_score", None)
    if not bool(getattr(inp, "quick_mode", False)):
        condition_bins_by_code: dict[str, dict[str, pd.Series]] = {}
        for c in w_eff.columns:
            ck = str(c)
            cl = close_sig_df[ck].reindex(nav.index).astype(float)
            mom_for_entry = (cl / cl.shift(int(getattr(inp, "mom_lookback", 252) or 252)) - 1.0).astype(float)
            er_for_entry = _efficiency_ratio(cl, window=int(getattr(inp, "er_window", 10) or 10)).astype(float)
            atr_fast = _atr_from_hlc(
                high_sig_df[ck].reindex(nav.index).astype(float).fillna(cl),
                low_sig_df[ck].reindex(nav.index).astype(float).fillna(cl),
                cl,
                window=int(getattr(inp, "vol_ratio_fast_atr_window", 5) or 5),
            ).astype(float)
            atr_slow = _atr_from_hlc(
                high_sig_df[ck].reindex(nav.index).astype(float).fillna(cl),
                low_sig_df[ck].reindex(nav.index).astype(float).fillna(cl),
                cl,
                window=int(getattr(inp, "vol_ratio_slow_atr_window", 50) or 50),
            ).astype(float)
            vol_ratio = (atr_fast / atr_slow.replace(0.0, np.nan)).astype(float)
            impulse_state = _compute_impulse_state(
                cl,
                ema_window=13,
                macd_fast=12,
                macd_slow=26,
                macd_signal=9,
            )
            condition_bins_by_code[ck] = {
                "momentum": _bucketize_momentum_series(mom_for_entry.reindex(nav.index)),
                "er": _bucketize_er_series(er_for_entry.reindex(nav.index)),
                "vol_ratio": _bucketize_vol_ratio_series(vol_ratio.reindex(nav.index)),
                "impulse": _bucketize_impulse_series(
                    (impulse_state if impulse_state is not None else pd.Series(index=nav.index, dtype=object)).reindex(nav.index)
                ),
            }
        trades_with_r = _attach_entry_condition_bins_to_trades(
            trades_with_r,
            condition_bins_by_code=condition_bins_by_code,
            dates=nav.index,
            default_code=None,
        )
    mfe_r_distribution = build_trade_mfe_r_distribution(
        trade_pack.get("trades", []),
        close=close_sig_df.reindex(index=nav.index, columns=w_eff.columns).astype(float).ffill(),
        high=high_sig_df.reindex(index=nav.index, columns=w_eff.columns).astype(float).ffill(),
        atr=atr_risk_df.reindex(index=nav.index, columns=w_eff.columns).astype(float),
        atr_mult=float(getattr(inp, "atr_stop_n", 2.0) or 2.0),
        default_code=None,
    )

    er_blocked = sum(int((signal_debug_by_code.get(c, {}).get("er_filter") or {}).get("blocked_entry_count", 0)) for c in signal_debug_by_code)
    er_attempted = sum(int((signal_debug_by_code.get(c, {}).get("er_filter") or {}).get("attempted_entry_count", 0)) for c in signal_debug_by_code)
    er_allowed = sum(int((signal_debug_by_code.get(c, {}).get("er_filter") or {}).get("allowed_entry_count", 0)) for c in signal_debug_by_code)
    imp_blocked = sum(int((signal_debug_by_code.get(c, {}).get("impulse_filter") or {}).get("blocked_entry_count", 0)) for c in signal_debug_by_code)
    imp_attempted = sum(int((signal_debug_by_code.get(c, {}).get("impulse_filter") or {}).get("attempted_entry_count", 0)) for c in signal_debug_by_code)
    imp_allowed = sum(int((signal_debug_by_code.get(c, {}).get("impulse_filter") or {}).get("allowed_entry_count", 0)) for c in signal_debug_by_code)
    imp_bull = sum(int((signal_debug_by_code.get(c, {}).get("impulse_filter") or {}).get("blocked_entry_count_bull", 0)) for c in signal_debug_by_code)
    imp_bear = sum(int((signal_debug_by_code.get(c, {}).get("impulse_filter") or {}).get("blocked_entry_count_bear", 0)) for c in signal_debug_by_code)
    imp_neutral = sum(int((signal_debug_by_code.get(c, {}).get("impulse_filter") or {}).get("blocked_entry_count_neutral", 0)) for c in signal_debug_by_code)
    er_exit_trigger = sum(int((signal_debug_by_code.get(c, {}).get("er_exit_filter") or {}).get("trigger_count", 0)) for c in signal_debug_by_code)
    vol_adj_total = sum(int((semantic_debug_by_code.get(c, {}).get("vol_risk_adjust") or {}).get("vol_risk_adjust_total_count", 0)) for c in semantic_debug_by_code)
    atr_trigger_total = sum(int((semantic_debug_by_code.get(c, {}).get("atr_stop") or {}).get("trigger_count", 0)) for c in semantic_debug_by_code)
    rtp_trigger_total = sum(int((semantic_debug_by_code.get(c, {}).get("r_take_profit") or {}).get("trigger_count", 0)) for c in semantic_debug_by_code)
    bias_v_tp_trigger_total = sum(int((semantic_debug_by_code.get(c, {}).get("bias_v_take_profit") or {}).get("trigger_count", 0)) for c in semantic_debug_by_code)
    rtp_tier_counts: dict[str, int] = {}
    for c in semantic_debug_by_code:
        tiers = ((semantic_debug_by_code.get(c, {}).get("r_take_profit") or {}).get("tier_trigger_counts") or {})
        for k, v in dict(tiers).items():
            kk = str(k)
            rtp_tier_counts[kk] = int(rtp_tier_counts.get(kk, 0) + int(v))
    if (not bool(getattr(inp, "monthly_risk_budget_enabled", False))) and semantic_debug_by_code:
        monthly_attempted_total = sum(int((semantic_debug_by_code.get(c, {}).get("monthly_risk_budget_gate") or {}).get("attempted_entry_count", 0)) for c in semantic_debug_by_code)
        monthly_blocked_total = sum(int((semantic_debug_by_code.get(c, {}).get("monthly_risk_budget_gate") or {}).get("blocked_entry_count", 0)) for c in semantic_debug_by_code)
    imp_rate = float(imp_blocked / imp_attempted) if imp_attempted > 0 else 0.0
    monthly_rate = float(monthly_blocked_total / monthly_attempted_total) if monthly_attempted_total > 0 else 0.0
    overall_stats = {
        **_trade_stats_from_returns(trade_pack.get("returns", [])),
        "n": len(trades),
        "atr_stop_trigger_count": int(atr_trigger_total),
        "r_take_profit_trigger_count": int(rtp_trigger_total),
        "bias_v_take_profit_trigger_count": int(bias_v_tp_trigger_total),
        "r_take_profit_tier_trigger_counts": dict(rtp_tier_counts),
        "er_filter_blocked_entry_count": int(er_blocked),
        "er_filter_attempted_entry_count": int(er_attempted),
        "er_filter_allowed_entry_count": int(er_allowed),
        "impulse_filter_blocked_entry_count": int(imp_blocked),
        "impulse_filter_attempted_entry_count": int(imp_attempted),
        "impulse_filter_allowed_entry_count": int(imp_allowed),
        "impulse_filter_blocked_entry_rate": float(imp_rate),
        "impulse_filter_blocked_entry_count_bull": int(imp_bull),
        "impulse_filter_blocked_entry_count_bear": int(imp_bear),
        "impulse_filter_blocked_entry_count_neutral": int(imp_neutral),
        "er_exit_filter_trigger_count": int(er_exit_trigger),
        "vol_risk_adjust_total_count": int(vol_adj_total),
        "vol_risk_adjust_reduce_on_expand_count": int(
            sum(
                int(((semantic_debug_by_code.get(str(c), {}) or {}).get("vol_risk_adjust") or {}).get("vol_risk_adjust_reduce_on_expand_count", 0))
                for c in wdf.columns
            )
        ),
        "vol_risk_adjust_increase_on_contract_count": int(
            sum(
                int(((semantic_debug_by_code.get(str(c), {}) or {}).get("vol_risk_adjust") or {}).get("vol_risk_adjust_increase_on_contract_count", 0))
                for c in wdf.columns
            )
        ),
        "vol_risk_adjust_recover_from_expand_count": int(
            sum(
                int(((semantic_debug_by_code.get(str(c), {}) or {}).get("vol_risk_adjust") or {}).get("vol_risk_adjust_recover_from_expand_count", 0))
                for c in wdf.columns
            )
        ),
        "vol_risk_adjust_recover_from_contract_count": int(
            sum(
                int(((semantic_debug_by_code.get(str(c), {}) or {}).get("vol_risk_adjust") or {}).get("vol_risk_adjust_recover_from_contract_count", 0))
                for c in wdf.columns
            )
        ),
        "vol_risk_entry_state_reduce_on_expand_count": int(
            sum(
                int(((semantic_debug_by_code.get(str(c), {}) or {}).get("vol_risk_adjust") or {}).get("vol_risk_entry_state_reduce_on_expand_count", 0))
                for c in wdf.columns
            )
        ),
        "vol_risk_entry_state_increase_on_contract_count": int(
            sum(
                int(((semantic_debug_by_code.get(str(c), {}) or {}).get("vol_risk_adjust") or {}).get("vol_risk_entry_state_increase_on_contract_count", 0))
                for c in wdf.columns
            )
        ),
        "monthly_risk_budget_attempted_entry_count": int(monthly_attempted_total),
        "monthly_risk_budget_blocked_entry_count": int(monthly_blocked_total),
        "monthly_risk_budget_blocked_entry_rate": float(monthly_rate),
        "vol_risk_overcap_scale_count": int(overcap_scale_total),
        "vol_risk_overcap_skip_entry_decision_count": int(overcap_skip_decision_total),
        "vol_risk_overcap_skip_entry_episode_count": int(overcap_skip_episode_total),
        "vol_risk_overcap_replace_entry_count": int(overcap_replace_total),
        "vol_risk_overcap_replace_out_count": int(overcap_replace_total),
        "vol_risk_overcap_replace_in_count": int(overcap_replace_total),
        "vol_risk_overcap_leverage_usage_count": int(overcap_leverage_usage_total),
        "vol_risk_overcap_leverage_max_multiple": float(overcap_leverage_max_multiple),
    }
    by_code_stats: dict[str, dict[str, Any]] = {}
    for c in wdf.columns:
        d = signal_debug_by_code.get(c, {})
        er_stats = d.get("er_filter") or {}
        imp_stats = d.get("impulse_filter") or {}
        er_exit_stats = d.get("er_exit_filter") or {}
        sem = semantic_debug_by_code.get(c, {})
        one_imp_attempted = int(imp_stats.get("attempted_entry_count", 0))
        one_imp_blocked = int(imp_stats.get("blocked_entry_count", 0))
        one_month_attempted = int((sem.get("monthly_risk_budget_gate") or {}).get("attempted_entry_count", 0))
        one_month_blocked = int((sem.get("monthly_risk_budget_gate") or {}).get("blocked_entry_count", 0))
        by_code_stats[str(c)] = {
            **_trade_stats_from_returns((trade_pack.get("returns_by_code") or {}).get(str(c), [])),
            "n": int(sum(1 for t in trades if str(t.get("code")) == str(c))),
            "atr_stop_trigger_count": int((sem.get("atr_stop") or {}).get("trigger_count", 0)),
            "r_take_profit_trigger_count": int((sem.get("r_take_profit") or {}).get("trigger_count", 0)),
            "bias_v_take_profit_trigger_count": int((sem.get("bias_v_take_profit") or {}).get("trigger_count", 0)),
            "r_take_profit_tier_trigger_counts": dict((sem.get("r_take_profit") or {}).get("tier_trigger_counts") or {}),
            "er_filter_blocked_entry_count": int(er_stats.get("blocked_entry_count", 0)),
            "er_filter_attempted_entry_count": int(er_stats.get("attempted_entry_count", 0)),
            "er_filter_allowed_entry_count": int(er_stats.get("allowed_entry_count", 0)),
            "impulse_filter_blocked_entry_count": one_imp_blocked,
            "impulse_filter_attempted_entry_count": one_imp_attempted,
            "impulse_filter_allowed_entry_count": int(imp_stats.get("allowed_entry_count", 0)),
            "impulse_filter_blocked_entry_rate": (float(one_imp_blocked / one_imp_attempted) if one_imp_attempted > 0 else 0.0),
            "impulse_filter_blocked_entry_count_bull": int(imp_stats.get("blocked_entry_count_bull", 0)),
            "impulse_filter_blocked_entry_count_bear": int(imp_stats.get("blocked_entry_count_bear", 0)),
            "impulse_filter_blocked_entry_count_neutral": int(imp_stats.get("blocked_entry_count_neutral", 0)),
            "er_exit_filter_trigger_count": int(er_exit_stats.get("trigger_count", 0)),
            "vol_risk_adjust_total_count": int((sem.get("vol_risk_adjust") or {}).get("vol_risk_adjust_total_count", 0)),
            "vol_risk_adjust_reduce_on_expand_count": int((sem.get("vol_risk_adjust") or {}).get("vol_risk_adjust_reduce_on_expand_count", 0)),
            "vol_risk_adjust_increase_on_contract_count": int((sem.get("vol_risk_adjust") or {}).get("vol_risk_adjust_increase_on_contract_count", 0)),
            "vol_risk_adjust_recover_from_expand_count": int((sem.get("vol_risk_adjust") or {}).get("vol_risk_adjust_recover_from_expand_count", 0)),
            "vol_risk_adjust_recover_from_contract_count": int((sem.get("vol_risk_adjust") or {}).get("vol_risk_adjust_recover_from_contract_count", 0)),
            "vol_risk_entry_state_reduce_on_expand_count": int((sem.get("vol_risk_adjust") or {}).get("vol_risk_entry_state_reduce_on_expand_count", 0)),
            "vol_risk_entry_state_increase_on_contract_count": int((sem.get("vol_risk_adjust") or {}).get("vol_risk_entry_state_increase_on_contract_count", 0)),
            "monthly_risk_budget_attempted_entry_count": one_month_attempted,
            "monthly_risk_budget_blocked_entry_count": one_month_blocked,
            "monthly_risk_budget_blocked_entry_rate": (float(one_month_blocked / one_month_attempted) if one_month_attempted > 0 else 0.0),
            "vol_risk_overcap_scale_count": int(overcap_scale_by_code.get(str(c), 0)),
            "vol_risk_overcap_skip_entry_decision_count": int(overcap_skip_decision_by_code.get(str(c), 0)),
            "vol_risk_overcap_skip_entry_episode_count": int(overcap_skip_episode_by_code.get(str(c), 0)),
            "vol_risk_overcap_replace_out_count": int(overcap_replace_out_by_code.get(str(c), 0)),
            "vol_risk_overcap_replace_in_count": int(overcap_replace_in_by_code.get(str(c), 0)),
            "vol_risk_overcap_leverage_usage_count": int(overcap_leverage_usage_by_code.get(str(c), 0)),
            "vol_risk_overcap_leverage_max_multiple": float(overcap_leverage_max_multiple_by_code.get(str(c), 0.0)),
        }
    atr_trigger_dates = sorted(
        {
            str(d)
            for v in atr_stop_by_asset.values()
            for d in list((v or {}).get("trigger_dates") or [])
            if str(d).strip()
        }
    )
    rtp_trigger_dates = sorted(
        {
            str(d)
            for v in rtp_by_asset.values()
            for d in list((v or {}).get("trigger_dates") or [])
            if str(d).strip()
        }
    )
    bias_v_tp_trigger_dates = sorted(
        {
            str(d)
            for v in bias_v_tp_by_asset.values()
            for d in list((v or {}).get("trigger_dates") or [])
            if str(d).strip()
        }
    )
    er_exit_by_asset = {}
    for c in wdf.columns:
        one = dict((signal_debug_by_code.get(c, {}).get("er_exit_filter") or {}))
        er_exit_by_asset[str(c)] = {
            **one,
            "trigger_count": int(one.get("trigger_count", 0)),
            "trigger_dates": [str(x) for x in (one.get("trigger_dates") or []) if str(x).strip()],
            "trace_last_rows": list(one.get("trace_last_rows") or []),
        }
    er_exit_trigger_dates = sorted(
        {
            str(d)
            for v in er_exit_by_asset.values()
            for d in list((v or {}).get("trigger_dates") or [])
            if str(d).strip()
        }
    )
    monthly_attempted_by_code: dict[str, int] = {}
    monthly_blocked_by_code: dict[str, int] = {}
    for c in wdf.columns:
        gate = (semantic_debug_by_code.get(c, {}).get("monthly_risk_budget_gate") or {})
        one_attempted = int(gate.get("attempted_entry_count", 0))
        one_blocked = int(gate.get("blocked_entry_count", 0))
        monthly_attempted_by_code[str(c)] = one_attempted
        monthly_blocked_by_code[str(c)] = one_blocked
    fixed_ext_dates = sorted({str(e.get("date")) for e in fixed_ext_events if str(e.get("date", "")).strip()})
    fixed_skip_dates = sorted({str(e.get("date")) for e in fixed_skip_events if str(e.get("date", "")).strip()})
    fixed_ext_over_weight_count = int(sum(1 for e in fixed_ext_events if bool(e.get("over_weight"))))
    fixed_ext_over_count_count = int(sum(1 for e in fixed_ext_events if bool(e.get("over_count"))))
    fixed_ext_over_both_count = int(sum(1 for e in fixed_ext_events if bool(e.get("over_weight")) and bool(e.get("over_count"))))
    fixed_skip_over_weight_count = int(sum(1 for e in fixed_skip_events if bool(e.get("over_weight"))))
    fixed_skip_over_count_count = int(sum(1 for e in fixed_skip_events if bool(e.get("over_count"))))
    fixed_skip_over_both_count = int(sum(1 for e in fixed_skip_events if bool(e.get("over_weight")) and bool(e.get("over_count"))))
    vol_risk_adjust_by_asset = {
        str(c): dict((semantic_debug_by_code.get(c, {}).get("vol_risk_adjust") or {}))
        for c in wdf.columns
    }
    bias_v_trace_keys = [
        "date", "open", "high", "low", "close", "ma", "atr", "threshold", "bias_v",
        "base_pos", "decision_pos", "tp_triggered", "tp_trigger_source", "tp_trigger_price_raw",
        "tp_trigger_price_eff", "tp_fill_price", "stop_trigger_source", "stop_fill_price",
        "gap_open_triggered", "event_type", "event_reason", "in_pos_after", "wait_next_entry_lock",
    ]
    atr_trade_record_keys = [
        "entry_decision_date",
        "entry_execution_date",
        "trigger_date",
        "entry_execution_price",
        "initial_stop_price",
        "trigger_stop_price",
        "execution_stop_price",
    ]
    atr_trigger_event_keys = [
        "date",
        "stop_price",
        "open_price",
        "low_price",
        "fill_price",
        "trigger_source",
        "gap_open_triggered",
    ]
    bias_v_trade_record_keys = [
        "entry_decision_date",
        "entry_execution_date",
        "trigger_date",
        "entry_execution_price",
        "initial_take_profit_price",
        "trigger_take_profit_price",
        "execution_take_profit_price",
    ]
    bias_v_trigger_event_keys = [
        "date",
        "trigger_price",
        "trigger_price_raw",
        "trigger_price_eff",
        "open_price",
        "high_price",
        "fill_price",
        "trigger_source",
        "gap_open_triggered",
        "bias_v",
        "threshold",
    ]
    rtp_trace_keys = [
        "date", "open", "high", "low", "base_pos", "decision_pos", "entry_price", "atr_entry",
        "initial_r_pct", "peak_profit_pct", "peak_r_multiple", "drawdown_from_peak", "active_tier_r",
        "active_tier_retrace", "tp_triggered", "tp_trigger_source", "tp_fill_price",
        "stop_trigger_source", "stop_fill_price", "gap_open_triggered", "event_type", "event_reason",
        "in_pos_after", "wait_next_entry_lock",
    ]
    for c in wdf.columns:
        ck = str(c)
        a = dict(atr_stop_by_asset.get(ck) or {})
        a["trade_records"] = _normalize_trace_rows(a.get("trade_records"), atr_trade_record_keys)
        a["trigger_events"] = _normalize_trace_rows(a.get("trigger_events"), atr_trigger_event_keys)
        atr_stop_by_asset[ck] = a
        b = dict(bias_v_tp_by_asset.get(ck) or {})
        b["trace_last_rows"] = _normalize_trace_rows(b.get("trace_last_rows"), bias_v_trace_keys)
        b["trade_records"] = _normalize_trace_rows(b.get("trade_records"), bias_v_trade_record_keys)
        b["trigger_events"] = _normalize_trace_rows(b.get("trigger_events"), bias_v_trigger_event_keys)
        bias_v_tp_by_asset[ck] = b
        r = dict(rtp_by_asset.get(ck) or {})
        r.setdefault("invalid_initial_r_entries", 0)
        r["trace_last_rows"] = _normalize_trace_rows(r.get("trace_last_rows"), rtp_trace_keys)
        rtp_by_asset[ck] = r

    sample_days = int(len(port_ret))
    complete_trade_count = int(len(trade_pack.get("returns", [])))
    avg_daily_turnover = float(turnover.mean()) if len(turnover) else 0.0
    avg_annual_turnover = float(avg_daily_turnover * 252.0)
    avg_daily_trade_count = float(complete_trade_count / sample_days) if sample_days > 0 else 0.0
    avg_annual_trade_count = float(avg_daily_trade_count * 252.0)
    quick_mode = bool(getattr(inp, "quick_mode", False))
    trades_by_code = {str(c): [t for t in trades_with_r if str(t.get("code")) == str(c)] for c in wdf.columns}
    if quick_mode:
        trades = []
        trades_by_code = {str(c): [] for c in wdf.columns}
    entry_exec_price_with_slippage_by_asset: dict[str, float] = {}
    for c in w_eff.columns:
        one = _latest_entry_exec_price_with_slippage(
            effective_weight=w_eff[c].reindex(nav.index).astype(float),
            exec_price_series=px_exec_slip_df[c].reindex(nav.index).ffill().astype(float),
            slippage_spread=float(getattr(inp, "slippage_rate", 0.0) or 0.0),
        )
        if one is not None:
            entry_exec_price_with_slippage_by_asset[str(c)] = float(one)

    out = {
        "meta": {
            "type": "trend_portfolio_backtest",
            "engine": "bt",
            "runtime_engine": runtime_engine if "runtime_engine" in locals() else "unknown",
            "start": inp.start.strftime("%Y%m%d"),
            "end": inp.end.strftime("%Y%m%d"),
            "strategy": strat,
            "codes": list(nav_map.keys()),
            "failed_codes": failures,
            "strategy_execution_description": TREND_STRATEGY_EXECUTION_DESCRIPTIONS.get(strat, ""),
            "params": _build_meta_params(inp),
            "limitations": [],
        },
        "nav": {
            "dates": [d.strftime("%Y-%m-%d") for d in nav.index],
            "series": {
                "STRAT": [float(x) for x in nav.values],
                "BUY_HOLD_EW": [float(x) for x in bh_nav.values],
                "BUY_HOLD": [float(x) for x in bh_nav.values],
                "EXCESS": [float(x) for x in excess_nav.values],
            },
        },
        "weights": {
            "dates": [d.strftime("%Y-%m-%d") for d in nav.index],
            "series": {c: [float(x) for x in w_eff[c].values] for c in w_eff.columns},
        },
        "weights_decision": {
            "dates": [d.strftime("%Y-%m-%d") for d in wdf.index],
            "series": {c: [float(x) for x in wdf[c].values] for c in wdf.columns},
        },
        "asset_nav_exec": {
            "dates": [d.strftime("%Y-%m-%d") for d in ret_exec_df.index],
            "series": {
                c: [float(x) for x in (1.0 + ret_exec_df[c].astype(float)).cumprod().values]
                for c in ret_exec_df.columns
            },
        },
        "signals": {
            "dates": [d.strftime("%Y-%m-%d") for d in nav.index],
            "position_effective": [float(x) for x in (w_eff > 0.0).mean(axis=1).values],
        },
        "metrics": {
            "strategy": {
                **_metrics_from_ret(port_ret, float(inp.risk_free_rate)),
                "avg_daily_turnover": float(avg_daily_turnover),
                "avg_annual_turnover": float(avg_annual_turnover),
                "avg_annual_turnover_rate": float(avg_annual_turnover),
                "avg_daily_trade_count": float(avg_daily_trade_count),
                "avg_annual_trade_count": float(avg_annual_trade_count),
                "r_take_profit_tier_trigger_counts": dict(rtp_tier_counts),
                "r_take_profit_trigger_count": int(rtp_trigger_total),
                "bias_v_take_profit_trigger_count": int(bias_v_tp_trigger_total),
                "atr_stop_trigger_count": int(atr_trigger_total),
                "impulse_filter_blocked_entry_count": int(imp_blocked),
                "impulse_filter_blocked_entry_count_bull": int(imp_bull),
                "impulse_filter_blocked_entry_count_bear": int(imp_bear),
                "impulse_filter_blocked_entry_count_neutral": int(imp_neutral),
                "vol_risk_overcap_scale_count": int(overcap_scale_total),
                "vol_risk_overcap_skip_entry_decision_count": int(overcap_skip_decision_total),
                "vol_risk_overcap_skip_entry_episode_count": int(overcap_skip_episode_total),
                "vol_risk_overcap_replace_entry_count": int(overcap_replace_total),
                "vol_risk_overcap_leverage_usage_count": int(overcap_leverage_usage_total),
                "vol_risk_overcap_leverage_max_multiple": float(overcap_leverage_max_multiple),
                "monthly_risk_budget_blocked_entry_count": int(monthly_blocked_total),
            },
            "benchmark": _metrics_from_ret(bh_ret, float(inp.risk_free_rate)),
            "excess": {
                **_metrics_from_ret(excess_ret, float(inp.risk_free_rate)),
                "information_ratio": float(_information_ratio(active_ret)),
            },
        },
        "period_returns": {
            "weekly": _period_returns(nav, "W-FRI"),
            "monthly": _period_returns(nav, "ME"),
            "quarterly": _period_returns(nav, "QE"),
            "yearly": _period_returns(nav, "YE"),
        },
        "rolling": _rolling_pack(nav),
        "attribution": attribution,
        "trade_statistics": {
            "all": {"n": len(trades)},
            "overall": overall_stats,
            "by_code": by_code_stats,
            "trades": ([] if quick_mode else trades_with_r),
            "trades_by_code": trades_by_code,
            "mfe_r_distribution": mfe_r_distribution,
        },
        "r_statistics": r_stats_out,
        "trades": ([] if quick_mode else trades_with_r),
        "next_plan": {
            "decision_date": (str(nav.index[-1].date()) if len(nav.index) else None),
            "entry_exec_price_with_slippage_by_asset": entry_exec_price_with_slippage_by_asset,
        },
        "risk_controls": {
            "vol_regime_risk_mgmt": {
                "enabled": bool(getattr(inp, "vol_regime_risk_mgmt_enabled", False)),
                "fast_atr_window": int(getattr(inp, "vol_ratio_fast_atr_window", 5) or 5),
                "slow_atr_window": int(getattr(inp, "vol_ratio_slow_atr_window", 50) or 50),
                "expand_threshold": float(getattr(inp, "vol_ratio_expand_threshold", 1.45) or 1.45),
                "contract_threshold": float(getattr(inp, "vol_ratio_contract_threshold", 0.65) or 0.65),
                "normal_threshold": float(getattr(inp, "vol_ratio_normal_threshold", 1.05) or 1.05),
                "adjust_total_count": int(overall_stats.get("vol_risk_adjust_total_count", 0)),
                "adjust_reduce_on_expand_count": int(overall_stats.get("vol_risk_adjust_reduce_on_expand_count", 0)),
                "adjust_increase_on_contract_count": int(overall_stats.get("vol_risk_adjust_increase_on_contract_count", 0)),
                "adjust_recover_from_expand_count": int(overall_stats.get("vol_risk_adjust_recover_from_expand_count", 0)),
                "adjust_recover_from_contract_count": int(overall_stats.get("vol_risk_adjust_recover_from_contract_count", 0)),
                "entry_state_reduce_on_expand_count": int(overall_stats.get("vol_risk_entry_state_reduce_on_expand_count", 0)),
                "entry_state_increase_on_contract_count": int(overall_stats.get("vol_risk_entry_state_increase_on_contract_count", 0)),
                "overcap_policy": str(getattr(inp, "risk_budget_overcap_policy", "scale") or "scale"),
                "overcap_max_leverage_multiple": float(getattr(inp, "risk_budget_max_leverage_multiple", 2.0) or 2.0),
                "overcap_scale_count": int(overcap_scale_total),
                "overcap_skip_entry_decision_count": int(overcap_skip_decision_total),
                "overcap_skip_entry_episode_count": int(overcap_skip_episode_total),
                "overcap_skip_entry_decision_count_by_code": dict(overcap_skip_decision_by_code),
                "overcap_skip_entry_episode_count_by_code": dict(overcap_skip_episode_by_code),
                "overcap_replace_entry_count": int(overcap_replace_total),
                "overcap_replace_out_count_by_code": dict(overcap_replace_out_by_code),
                "overcap_replace_in_count_by_code": dict(overcap_replace_in_by_code),
                "overcap_leverage_usage_count": int(overcap_leverage_usage_total),
                "overcap_leverage_max_multiple": float(overcap_leverage_max_multiple),
                "overcap_leverage_usage_count_by_code": dict(overcap_leverage_usage_by_code),
                "overcap_leverage_max_multiple_by_code": dict(overcap_leverage_max_multiple_by_code),
                "overcap_daily_counts": [
                    {
                        "date": str(k),
                        "scale": int((v or {}).get("scale", 0)),
                        "skip_entry": int((v or {}).get("skip_entry", 0)),
                        "replace_entry": int((v or {}).get("replace_entry", 0)),
                        "leverage_entry": int((v or {}).get("leverage_entry", 0)),
                        "leverage_multiple_max": float((v or {}).get("leverage_multiple_max", 0.0) or 0.0),
                    }
                    for k, v in sorted((risk_budget_overcap_daily_counts or {}).items(), key=lambda x: str(x[0]))
                ],
                "by_asset": dict(vol_risk_adjust_by_asset),
            },
            "atr_stop": {
                "enabled": bool(str(getattr(inp, "atr_stop_mode", "none") or "none").strip().lower() != "none"),
                "mode": str(getattr(inp, "atr_stop_mode", "none") or "none"),
                "atr_basis": str(getattr(inp, "atr_stop_atr_basis", "latest") or "latest"),
                "reentry_mode": str(getattr(inp, "atr_stop_reentry_mode", "reenter") or "reenter"),
                "trigger_count": int(atr_trigger_total),
                "trigger_days": int(len(atr_trigger_dates)),
                "first_trigger_date": (atr_trigger_dates[0] if atr_trigger_dates else None),
                "last_trigger_date": (atr_trigger_dates[-1] if atr_trigger_dates else None),
                "trigger_dates": atr_trigger_dates[:200],
                "by_asset": atr_stop_by_asset,
            },
            "r_take_profit": {
                "enabled": bool(getattr(inp, "r_take_profit_enabled", False)),
                "reentry_mode": str(getattr(inp, "r_take_profit_reentry_mode", "reenter") or "reenter"),
                "tiers": _normalize_r_take_profit_tiers(getattr(inp, "r_take_profit_tiers", None)),
                "trigger_count": int(rtp_trigger_total),
                "tier_trigger_counts": dict(rtp_tier_counts),
                "trigger_days": int(len(rtp_trigger_dates)),
                "first_trigger_date": (rtp_trigger_dates[0] if rtp_trigger_dates else None),
                "last_trigger_date": (rtp_trigger_dates[-1] if rtp_trigger_dates else None),
                "trigger_dates": rtp_trigger_dates[:200],
                "fallback_mode_used": bool(
                    str(getattr(inp, "atr_stop_mode", "none") or "none").strip().lower() == "none"
                    and bool(getattr(inp, "r_take_profit_enabled", False))
                ),
                "initial_r_mode": (
                    "atr_stop"
                    if str(getattr(inp, "atr_stop_mode", "none") or "none").strip().lower() != "none"
                    else "virtual_atr_fallback"
                ),
                "by_asset": rtp_by_asset,
            },
            "bias_v_take_profit": {
                "enabled": bool(getattr(inp, "bias_v_take_profit_enabled", False)),
                "reentry_mode": str(getattr(inp, "bias_v_take_profit_reentry_mode", "reenter") or "reenter"),
                "ma_window": int(getattr(inp, "bias_v_ma_window", 20) or 20),
                "atr_window": int(getattr(inp, "bias_v_atr_window", 20) or 20),
                "threshold": float(getattr(inp, "bias_v_take_profit_threshold", 5.0) or 5.0),
                "trigger_count": int(bias_v_tp_trigger_total),
                "trigger_days": int(len(bias_v_tp_trigger_dates)),
                "first_trigger_date": (bias_v_tp_trigger_dates[0] if bias_v_tp_trigger_dates else None),
                "last_trigger_date": (bias_v_tp_trigger_dates[-1] if bias_v_tp_trigger_dates else None),
                "trigger_dates": bias_v_tp_trigger_dates[:200],
                "by_asset": bias_v_tp_by_asset,
            },
            "er_exit_filter": {
                "enabled": bool(getattr(inp, "er_exit_filter", False)),
                "window": int(getattr(inp, "er_exit_window", 10) or 10),
                "threshold": float(getattr(inp, "er_exit_threshold", 0.88) or 0.88),
                "trigger_count": int(er_exit_trigger),
                "trigger_days": int(len(er_exit_trigger_dates)),
                "first_trigger_date": (er_exit_trigger_dates[0] if er_exit_trigger_dates else None),
                "last_trigger_date": (er_exit_trigger_dates[-1] if er_exit_trigger_dates else None),
                "trigger_dates": er_exit_trigger_dates[:200],
                "by_asset": er_exit_by_asset,
            },
            "monthly_risk_budget_gate": {
                "enabled": bool(getattr(inp, "monthly_risk_budget_enabled", False)),
                "budget_pct": float(getattr(inp, "monthly_risk_budget_pct", 0.06) or 0.06),
                "include_new_trade_risk": bool(getattr(inp, "monthly_risk_budget_include_new_trade_risk", False)),
                "attempted_entry_count": int(monthly_attempted_total),
                "attempted_entry_count_by_code": dict(monthly_attempted_by_code),
                "blocked_entry_count": int(monthly_blocked_total),
                "blocked_entry_count_by_code": dict(monthly_blocked_by_code),
            },
            "monthly_risk_budget": {
                "enabled": bool(getattr(inp, "monthly_risk_budget_enabled", False)),
                "budget_pct": float(getattr(inp, "monthly_risk_budget_pct", 0.06) or 0.06),
                "include_new_trade_risk": bool(getattr(inp, "monthly_risk_budget_include_new_trade_risk", False)),
                "attempted_entry_count": int(monthly_attempted_total),
                "attempted_entry_count_by_code": dict(monthly_attempted_by_code),
                "blocked_entry_count": int(monthly_blocked_total),
                "blocked_entry_count_by_code": dict(monthly_blocked_by_code),
            },
            "group_filter": {
                "enabled": bool(getattr(inp, "group_enforce", False)),
                "policy": str(getattr(inp, "group_pick_policy", "highest_sharpe") or "highest_sharpe"),
                "max_holdings_per_group": int(getattr(inp, "group_max_holdings", 4) or 4),
                "decision_segments_with_group_filter": int(group_filter_enabled_segments),
                "decision_segments_effective": int(group_filter_effective_segments),
            },
            "position_extension": {
                "enabled": bool(ps == "fixed_ratio" and str(getattr(inp, "fixed_overcap_policy", "extend") or "extend") == "extend"),
                "position_sizing": str(ps),
                "fixed_pos_ratio": float(getattr(inp, "fixed_pos_ratio", 0.04) or 0.04),
                "overcap_policy": str(getattr(inp, "fixed_overcap_policy", "extend") or "extend"),
                "fixed_max_holdings": int(getattr(inp, "fixed_max_holdings", 10) or 10),
                "extension_count": int(len(fixed_ext_events)),
                "extension_over_weight_count": int(fixed_ext_over_weight_count),
                "extension_over_count_count": int(fixed_ext_over_count_count),
                "extension_over_both_count": int(fixed_ext_over_both_count),
                "extension_days": int(len(fixed_ext_dates)),
                "first_extension_date": (fixed_ext_dates[0] if fixed_ext_dates else None),
                "last_extension_date": (fixed_ext_dates[-1] if fixed_ext_dates else None),
                "extension_dates": fixed_ext_dates[:200],
                "extensions": fixed_ext_events[:200],
                "skipped_count": int(len(fixed_skip_events)),
                "skipped_over_weight_count": int(fixed_skip_over_weight_count),
                "skipped_over_count_count": int(fixed_skip_over_count_count),
                "skipped_over_both_count": int(fixed_skip_over_both_count),
                "skipped_days": int(len(fixed_skip_dates)),
                "first_skipped_date": (fixed_skip_dates[0] if fixed_skip_dates else None),
                "last_skipped_date": (fixed_skip_dates[-1] if fixed_skip_dates else None),
                "skipped_dates": fixed_skip_dates[:200],
                "skipped": fixed_skip_events[:200],
            },
            "position_usage": {
                "enabled": bool(ps in {"fixed_ratio", "risk_budget"}),
                "position_sizing": str(ps),
                "cash_as_residual": True,
                "min_exposure": (float(w_eff.sum(axis=1).min()) if len(w_eff) else float("nan")),
                "max_exposure": (float(w_eff.sum(axis=1).max()) if len(w_eff) else float("nan")),
                "mean_exposure": (float(w_eff.sum(axis=1).mean()) if len(w_eff) else float("nan")),
                "quantiles": {
                    "p05": (float(w_eff.sum(axis=1).quantile(0.05)) if len(w_eff) else float("nan")),
                    "p25": (float(w_eff.sum(axis=1).quantile(0.25)) if len(w_eff) else float("nan")),
                    "p50": (float(w_eff.sum(axis=1).quantile(0.50)) if len(w_eff) else float("nan")),
                    "p75": (float(w_eff.sum(axis=1).quantile(0.75)) if len(w_eff) else float("nan")),
                    "p95": (float(w_eff.sum(axis=1).quantile(0.95)) if len(w_eff) else float("nan")),
                },
                "over_100pct_days": int((w_eff.sum(axis=1) > 1.0 + 1e-12).sum()) if len(w_eff) else 0,
                "under_100pct_days": int((w_eff.sum(axis=1) < 1.0 - 1e-12).sum()) if len(w_eff) else 0,
            },
        },
        "return_decomposition": return_decomposition,
        "event_study": event_study,
        "market_regime": market_regime,
        "holdings": holdings,
        "corporate_actions": sorted(corporate_actions_rows, key=lambda x: (str(x.get("date")), str(x.get("code"))))[:200],
    }
    if not quick_mode:
        out["trade_statistics"]["entry_condition_stats"] = {
            "scope": "closed_trades_only",
            "signal_day_basis": "signal_day_before_entry_execution",
            "quasi_causal_method": "uplift + two_proportion_z / welch_t_normal_approx + BH",
            "strong_causal_method": "uplift + stratified_permutation + BH",
            "overall": _build_entry_condition_stats(trades_with_r, by_code=False, n_perm=300, seed=20260410),
            "by_code": {
                str(c): _build_entry_condition_stats(trades_by_code.get(str(c), []), by_code=True, n_perm=200, seed=20260410)
                for c in wdf.columns
            },
        }
    return out

