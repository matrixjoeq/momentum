from __future__ import annotations

# pylint: disable=broad-exception-caught,cell-var-from-loop

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .baseline import (
    TRADING_DAYS_PER_YEAR,
    _annualized_return,
    _annualized_vol,
    _compute_return_risk_contributions,
    _information_ratio,
    _max_drawdown,
    _max_drawdown_duration_days,
    _rolling_drawdown,
    _sharpe,
    _sortino,
    _ulcer_index,
    hfq_close_buy_hold_returns,
    hfq_close_daily_equal_weight_returns,
    load_close_prices,
    load_high_low_prices,
    load_ohlc_prices,
)
from .event_study import compute_event_study, entry_dates_from_exposure
from .execution_timing import corporate_action_mask, slippage_return_from_turnover
from .market_regime import build_market_regime_report
from .r_multiple import build_trade_mfe_r_distribution, enrich_trades_with_r_metrics

Session = Any  # runtime: keep dependency-free typing
# 各趋势策略的执行说明（信号日与收益归属）：统一为 T 日收盘后确定信号，T+1 日执行，收益不包含决策日当日。
TREND_STRATEGY_EXECUTION_DESCRIPTIONS: dict[str, str] = {
    "ma_filter": "信号在 T 日收盘后根据价格与均线(SMA/EMA/KAMA)关系确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "ma_cross": "信号在 T 日收盘后根据快慢线金叉/死叉确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "donchian": "信号在 T 日收盘后根据是否突破通道上轨/下轨确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "tsmom": "信号在 T 日收盘后根据回看期收益与动量入/出场阈值确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "linreg_slope": "信号在 T 日收盘后根据回看窗口线性回归斜率正负确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "bias": "信号在 T 日收盘后根据 BIAS 与进出场阈值确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "macd_cross": "信号在 T 日收盘后根据 MACD 与信号线金叉/死叉确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "macd_zero_filter": "信号在 T 日收盘后根据 MACD 是否大于零确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "macd_v": "信号在 T 日收盘后根据 ATR 归一化 MACD 与信号线关系确定，T+1 日按仓位执行；策略收益不包含决策日(T日)当日收益。",
    "random_entry": "信号在 T 日收盘后仅当当前无持仓时抛硬币随机决定是否入场（1=入场，0=空仓）；入场后按持有交易日数到期离场，均在 T+1 日执行；策略收益不包含决策日(T日)当日收益。",
}

DEFAULT_R_TAKE_PROFIT_TIERS: list[dict[str, float]] = [
    {"r_multiple": 2.0, "retrace_ratio": 0.50},
    {"r_multiple": 3.0, "retrace_ratio": 0.30},
    {"r_multiple": 4.0, "retrace_ratio": 0.20},
    {"r_multiple": 5.0, "retrace_ratio": 0.10},
    {"r_multiple": 6.0, "retrace_ratio": 0.05},
]


@dataclass(frozen=True)
class TrendInputs:
    code: str
    start: dt.date
    end: dt.date
    risk_free_rate: float = 0.025
    cost_bps: float = 0.0
    slippage_rate: float = (
        0.001  # one-way adverse slippage spread (absolute price diff)
    )
    exec_price: str = "open"  # open|close|oc2
    # strategy selection
    strategy: str = "ma_filter"  # ma_filter | ma_cross | donchian | tsmom | linreg_slope | bias | macd_cross | macd_zero_filter | macd_v | random_entry
    # parameters
    sma_window: int = 200  # ma_filter
    fast_window: int = 50  # ma_cross
    slow_window: int = 200  # ma_cross
    ma_type: str = "sma"  # ma_filter: sma | ema | kama; ma_cross: sma | ema | wma
    kama_er_window: int = 10
    kama_fast_window: int = 2
    kama_slow_window: int = 30
    kama_std_window: int = 20
    kama_std_coef: float = 1.0
    donchian_entry: int = 20  # donchian
    donchian_exit: int = 10  # donchian
    mom_lookback: int = 252  # tsmom
    tsmom_entry_threshold: float = 0.0  # tsmom: enter when momentum > this value
    tsmom_exit_threshold: float = 0.0  # tsmom: exit when momentum <= this value
    atr_stop_mode: str = "none"  # none | static | trailing | tightening
    atr_stop_atr_basis: str = "latest"  # entry | latest (for trailing/tightening)
    atr_stop_reentry_mode: str = "reenter"  # reenter | wait_next_entry
    atr_stop_window: int = 14  # ATR lookback window
    atr_stop_n: float = 2.0  # stop distance multiplier by ATR
    atr_stop_m: float = 0.5  # tightening step in ATR multiples
    r_take_profit_enabled: bool = False
    r_take_profit_reentry_mode: str = "reenter"  # reenter | wait_next_entry
    r_take_profit_tiers: list[dict[str, float]] | None = None
    bias_v_take_profit_enabled: bool = False
    bias_v_take_profit_reentry_mode: str = "reenter"  # reenter | wait_next_entry
    bias_v_ma_window: int = 20
    bias_v_atr_window: int = 20
    bias_v_take_profit_threshold: float = 5.0
    monthly_risk_budget_enabled: bool = False
    monthly_risk_budget_pct: float = 0.06
    monthly_risk_budget_include_new_trade_risk: bool = False
    # bias (deviation from EMA) trend-following
    # BIAS = (LN(C) - LN(EMA(C,N))) * 100  (percent)
    bias_ma_window: int = 20  # EMA(C,N) window in trading days
    bias_entry: float = 2.0  # enter when BIAS > entry (percent)
    bias_hot: float = 10.0  # take-profit exit when BIAS >= hot (percent)
    bias_cold: float = -2.0  # stop-loss exit when BIAS <= cold (percent)
    bias_pos_mode: str = "binary"  # binary | continuous (default binary)
    # MACD/MACD-V params
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_v_atr_window: int = 26
    macd_v_scale: float = 100.0
    random_hold_days: int = 20
    random_seed: int | None = 42
    position_sizing: str = "equal"  # equal | vol_target | fixed_ratio | risk_budget
    vol_window: int = 20
    vol_target_ann: float = 0.20
    fixed_pos_ratio: float = 0.04
    fixed_overcap_policy: str = "skip"
    fixed_max_holdings: int = 10
    risk_budget_atr_window: int = 20
    risk_budget_pct: float = 0.01
    vol_regime_risk_mgmt_enabled: bool = False
    vol_ratio_fast_atr_window: int = 5
    vol_ratio_slow_atr_window: int = 50
    vol_ratio_expand_threshold: float = 1.45
    vol_ratio_contract_threshold: float = 0.65
    vol_ratio_normal_threshold: float = 1.05
    group_enforce: bool = False
    group_pick_policy: str = "highest_sharpe"
    group_max_holdings: int = 4
    asset_groups: dict[str, str] | None = None
    er_filter: bool = False
    er_window: int = 10
    er_threshold: float = 0.30
    impulse_entry_filter: bool = False
    impulse_allow_bull: bool = True
    impulse_allow_bear: bool = False
    impulse_allow_neutral: bool = False
    er_exit_filter: bool = False
    er_exit_window: int = 10
    er_exit_threshold: float = 0.88
    quick_mode: bool = False


@dataclass(frozen=True)
class TrendPortfolioInputs:
    codes: list[str]
    start: dt.date
    end: dt.date
    risk_free_rate: float = 0.025
    cost_bps: float = 0.0
    slippage_rate: float = (
        0.001  # one-way adverse slippage spread (absolute price diff)
    )
    exec_price: str = "open"  # open|close|oc2
    strategy: str = "ma_filter"
    position_sizing: str = "equal"  # equal | vol_target | fixed_ratio | risk_budget
    vol_window: int = 20
    vol_target_ann: float = 0.20
    fixed_pos_ratio: float = 0.04  # used when position_sizing=fixed_ratio
    fixed_overcap_policy: str = (
        "skip"  # skip | extend when position would exceed constraints
    )
    fixed_max_holdings: int = (
        10  # max number of concurrently held assets when position_sizing=fixed_ratio
    )
    risk_budget_atr_window: int = 20  # n-day ATR window for risk-budget sizing
    risk_budget_pct: float = 0.01  # per-asset risk budget on total NAV (1% => 0.01)
    risk_budget_overcap_policy: str = (
        "scale"  # scale | skip_entry | replace_entry | leverage_entry
    )
    risk_budget_max_leverage_multiple: float = (
        2.0  # only used when overcap_policy=leverage_entry
    )
    vol_regime_risk_mgmt_enabled: bool = False
    vol_ratio_fast_atr_window: int = 5
    vol_ratio_slow_atr_window: int = 50
    vol_ratio_expand_threshold: float = 1.45
    vol_ratio_contract_threshold: float = 0.65
    vol_ratio_normal_threshold: float = 1.05
    dynamic_universe: bool = False
    # single-strategy params
    sma_window: int = 200
    fast_window: int = 50
    slow_window: int = 200
    ma_type: str = "sma"  # ma_filter: sma | ema | kama; ma_cross: sma | ema | wma
    kama_er_window: int = 10
    kama_fast_window: int = 2
    kama_slow_window: int = 30
    kama_std_window: int = 20
    kama_std_coef: float = 1.0
    donchian_entry: int = 20
    donchian_exit: int = 10
    mom_lookback: int = 252
    tsmom_entry_threshold: float = 0.0
    tsmom_exit_threshold: float = 0.0
    atr_stop_mode: str = "none"
    atr_stop_atr_basis: str = "latest"
    atr_stop_reentry_mode: str = "reenter"
    atr_stop_window: int = 14
    atr_stop_n: float = 2.0
    atr_stop_m: float = 0.5
    r_take_profit_enabled: bool = False
    r_take_profit_reentry_mode: str = "reenter"
    r_take_profit_tiers: list[dict[str, float]] | None = None
    bias_v_take_profit_enabled: bool = False
    bias_v_take_profit_reentry_mode: str = "reenter"
    bias_v_ma_window: int = 20
    bias_v_atr_window: int = 20
    bias_v_take_profit_threshold: float = 5.0
    monthly_risk_budget_enabled: bool = False
    monthly_risk_budget_pct: float = 0.06
    monthly_risk_budget_include_new_trade_risk: bool = False
    bias_ma_window: int = 20
    bias_entry: float = 2.0
    bias_hot: float = 10.0
    bias_cold: float = -2.0
    bias_pos_mode: str = "binary"
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_v_atr_window: int = 26
    macd_v_scale: float = 100.0
    random_hold_days: int = 20
    random_seed: int | None = 42
    group_enforce: bool = False
    group_pick_policy: str = "highest_sharpe"  # highest_sharpe | earliest_entry
    group_max_holdings: int = 4  # max picks kept in each group (1..10)
    asset_groups: dict[str, str] | None = None  # code -> group_id
    er_filter: bool = False
    er_window: int = 10
    er_threshold: float = 0.30
    impulse_entry_filter: bool = False
    impulse_allow_bull: bool = True
    impulse_allow_bear: bool = False
    impulse_allow_neutral: bool = False
    er_exit_filter: bool = False
    er_exit_window: int = 10
    er_exit_threshold: float = 0.88
    quick_mode: bool = False


def _reduce_active_codes_by_group(
    *,
    active_codes: list[str],
    score_row: pd.Series,
    sharpe_row: pd.Series,
    group_enforce: bool,
    asset_groups: dict[str, str] | None,
    group_pick_policy: str,
    group_max_holdings: int,
    current_holdings: set[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """
    Apply trend portfolio group constraint:
    - each group keeps at most group_max_holdings candidates.
    - policy:
      - earliest_entry: prefer currently-held assets first, then score rank.
      - highest_sharpe: prefer higher rolling sharpe, then score rank.
    """
    meta: dict[str, Any] = {
        "enabled": bool(group_enforce),
        "policy": str(group_pick_policy or "highest_sharpe"),
        "max_holdings_per_group": int(group_max_holdings),
        "before": list(active_codes),
        "after": list(active_codes),
        "group_picks": {},
        "group_eliminated": {},
        # backward-compat (single winner view)
        "group_winners": {},
    }
    if (not group_enforce) or (not active_codes):
        return list(active_codes), meta

    policy = str(group_pick_policy or "highest_sharpe").strip().lower()
    if policy not in {"earliest_entry", "highest_sharpe"}:
        raise ValueError(
            "group_pick_policy must be one of: earliest_entry|highest_sharpe"
        )
    kmax = int(group_max_holdings)
    if kmax < 1 or kmax > 10:
        raise ValueError("group_max_holdings must be in [1,10]")

    groups = {str(k): str(v) for k, v in (asset_groups or {}).items()}
    cur = set(str(x) for x in (current_holdings or set()))
    active_set = set(active_codes)
    # keep global score order semantics
    order_by_score = [
        str(c)
        for c in pd.Series(score_row)
        .reindex(active_codes)
        .sort_values(ascending=False)
        .index.tolist()
        if str(c) in active_set
    ]
    if len(order_by_score) != len(active_codes):
        order_by_score = list(active_codes)

    bucket: dict[str, list[str]] = {}
    for c in order_by_score:
        gid = str(groups.get(str(c)) or str(c)).strip() or str(c)
        bucket.setdefault(gid, []).append(str(c))

    kept_set: set[str] = set()
    for gid, codes_in_gid in bucket.items():
        chosen: list[str] = []
        if policy == "earliest_entry":
            held = [c for c in codes_in_gid if c in cur]
            chosen.extend(held[:kmax])
            for c in codes_in_gid:
                if len(chosen) >= kmax:
                    break
                if c not in chosen:
                    chosen.append(c)
        else:
            ranked = sorted(
                codes_in_gid,
                key=lambda c: (
                    -(
                        float(sharpe_row.get(c))
                        if np.isfinite(float(sharpe_row.get(c)))
                        else -1e18
                    ),
                    -(
                        float(score_row.get(c))
                        if np.isfinite(float(score_row.get(c)))
                        else -1e18
                    ),
                    str(c),
                ),
            )
            chosen = ranked[:kmax]

        eliminated = [c for c in codes_in_gid if c not in set(chosen)]
        for c in chosen:
            kept_set.add(c)
        meta["group_picks"][gid] = list(chosen)
        meta["group_eliminated"][gid] = list(eliminated)
        meta["group_winners"][gid] = chosen[0] if chosen else ""

    reduced = [c for c in order_by_score if c in kept_set]
    meta["after"] = list(reduced)
    return reduced, meta


def _rolling_linreg_slope(y: np.ndarray) -> float:
    """
    Rolling OLS slope on y over an implicit x = 0..n-1 (centered).
    Returns NaN if any non-finite values exist in the window.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return float("nan")
    if not np.all(np.isfinite(y)):
        return float("nan")
    x = np.arange(y.size, dtype=float)
    x = x - x.mean()
    y0 = y - y.mean()
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y0) / denom)


def _efficiency_ratio(price: pd.Series, *, window: int) -> pd.Series:
    """
    Kaufman Efficiency Ratio (ER):
    ER_t = |P_t - P_{t-window}| / sum_{i=t-window+1..t} |ΔP_i|
    """
    w = max(2, int(window))
    p = pd.to_numeric(price, errors="coerce").astype(float)
    change = (p - p.shift(w)).abs()
    volatility = p.diff().abs().rolling(window=w, min_periods=w).sum()
    er = (change / volatility.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return er.clip(lower=0.0, upper=1.0)


def _apply_er_entry_filter(
    raw_pos: pd.Series, *, er: pd.Series, threshold: float
) -> tuple[pd.Series, dict[str, int]]:
    """
    Entry-only filter:
    - new entry (0 -> >0) is allowed only when ER >= threshold
    - exits are unchanged
    """
    thr = float(threshold)
    out = np.zeros(len(raw_pos), dtype=float)
    blocked_count = 0
    attempted_entry_count = 0
    allowed_entry_count = 0
    in_pos = False
    idx = raw_pos.index
    for i, d in enumerate(idx):
        desired = float(raw_pos.iloc[i]) if np.isfinite(float(raw_pos.iloc[i])) else 0.0
        desired = max(0.0, desired)
        if not in_pos:
            if desired > 0.0:
                attempted_entry_count += 1
                er_ok = bool(np.isfinite(float(er.loc[d]))) and float(er.loc[d]) >= thr
                if er_ok:
                    in_pos = True
                    allowed_entry_count += 1
                    out[i] = desired
                else:
                    blocked_count += 1
                    out[i] = 0.0
            else:
                out[i] = 0.0
        else:
            if desired <= 0.0:
                in_pos = False
                out[i] = 0.0
            else:
                out[i] = desired
    return pd.Series(out, index=idx, dtype=float), {
        "blocked_entry_count": int(blocked_count),
        "attempted_entry_count": int(attempted_entry_count),
        "allowed_entry_count": int(allowed_entry_count),
    }


def _compute_impulse_state(
    close: pd.Series,
    *,
    ema_window: int = 13,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.Series:
    """
    Elder Impulse state:
    - BULL: EMA and MACD histogram both rising vs previous day
    - BEAR: EMA and MACD histogram both falling vs previous day
    - otherwise NEUTRAL
    """
    px = pd.to_numeric(close, errors="coerce").astype(float)
    ema = _ema(px, int(ema_window))
    _, _, hist = _macd_core(
        px, fast=int(macd_fast), slow=int(macd_slow), signal=int(macd_signal)
    )
    ema_diff = ema.diff()
    hist_diff = hist.diff()
    state = pd.Series("NEUTRAL", index=px.index, dtype=object)
    bull = (ema_diff > 0.0) & (hist_diff > 0.0)
    bear = (ema_diff < 0.0) & (hist_diff < 0.0)
    state.loc[bull.fillna(False)] = "BULL"
    state.loc[bear.fillna(False)] = "BEAR"
    return state


def _apply_impulse_entry_filter(
    raw_pos: pd.Series,
    *,
    impulse_state: pd.Series,
    allow_bull: bool,
    allow_bear: bool,
    allow_neutral: bool,
) -> tuple[pd.Series, dict[str, int]]:
    """
    Entry-only filter using impulse states.
    - new entry (0 -> >0) is allowed only if current state is configured as allowed
    - exits are unchanged
    """
    out = np.zeros(len(raw_pos), dtype=float)
    blocked_count = 0
    attempted_entry_count = 0
    allowed_entry_count = 0
    blocked_by_state = {"BULL": 0, "BEAR": 0, "NEUTRAL": 0}
    in_pos = False
    idx = raw_pos.index
    for i, d in enumerate(idx):
        desired = float(raw_pos.iloc[i]) if np.isfinite(float(raw_pos.iloc[i])) else 0.0
        desired = max(0.0, desired)
        st = str(impulse_state.loc[d]) if d in impulse_state.index else "NEUTRAL"
        st = st if st in {"BULL", "BEAR", "NEUTRAL"} else "NEUTRAL"
        is_allowed = (
            (st == "BULL" and bool(allow_bull))
            or (st == "BEAR" and bool(allow_bear))
            or (st == "NEUTRAL" and bool(allow_neutral))
        )
        if not in_pos:
            if desired > 0.0:
                attempted_entry_count += 1
                if is_allowed:
                    in_pos = True
                    allowed_entry_count += 1
                    out[i] = desired
                else:
                    blocked_count += 1
                    blocked_by_state[st] = int(blocked_by_state.get(st, 0) + 1)
                    out[i] = 0.0
            else:
                out[i] = 0.0
        else:
            if desired <= 0.0:
                in_pos = False
                out[i] = 0.0
            else:
                out[i] = desired
    return pd.Series(out, index=idx, dtype=float), {
        "blocked_entry_count": int(blocked_count),
        "attempted_entry_count": int(attempted_entry_count),
        "allowed_entry_count": int(allowed_entry_count),
        "blocked_entry_count_bull": int(blocked_by_state.get("BULL", 0)),
        "blocked_entry_count_bear": int(blocked_by_state.get("BEAR", 0)),
        "blocked_entry_count_neutral": int(blocked_by_state.get("NEUTRAL", 0)),
    }


def _apply_er_exit_filter(
    raw_pos: pd.Series, *, er: pd.Series, threshold: float
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Exit-only filter:
    - when currently in position, if ER >= threshold then force exit
    - entry behavior is unchanged
    """
    thr = float(threshold)
    out = np.zeros(len(raw_pos), dtype=float)
    trigger_count = 0
    trigger_dates: list[str] = []
    trace_last_rows: list[dict[str, Any]] = []
    in_pos = False
    idx = raw_pos.index
    for i, d in enumerate(idx):
        desired = float(raw_pos.iloc[i]) if np.isfinite(float(raw_pos.iloc[i])) else 0.0
        desired = max(0.0, desired)
        if not in_pos:
            if desired > 0.0:
                in_pos = True
                out[i] = desired
            else:
                out[i] = 0.0
            continue
        er_hit = bool(np.isfinite(float(er.loc[d]))) and float(er.loc[d]) >= thr
        if er_hit:
            in_pos = False
            out[i] = 0.0
            trigger_count += 1
            try:
                trigger_dates.append(pd.Timestamp(d).strftime("%Y%m%d"))
            except (TypeError, ValueError, OverflowError):
                pass
            trace_last_rows.append(
                {
                    "date": (
                        pd.Timestamp(d).date().isoformat()
                        if hasattr(d, "date")
                        else str(d)
                    ),
                    "event_type": "exit",
                    "event_reason": "er_exit_filter",
                    "base_pos": float(desired),
                    "decision_pos": 0.0,
                    "er_value": (
                        float(er.loc[d]) if np.isfinite(float(er.loc[d])) else None
                    ),
                    "er_threshold": float(thr),
                    "er_triggered": True,
                }
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]
        elif desired <= 0.0:
            in_pos = False
            out[i] = 0.0
        else:
            out[i] = desired
    return pd.Series(out, index=idx, dtype=float), {
        "trigger_count": int(trigger_count),
        "trigger_dates": sorted(set(trigger_dates)),
        "trace_last_rows": trace_last_rows[-80:],
    }


def _month_key(d: Any) -> str:
    try:
        return pd.Timestamp(d).strftime("%Y-%m")
    except (TypeError, ValueError, OverflowError):
        return ""


def _risk_budget_dynamic_weights(
    active_signal: pd.Series,
    *,
    close: pd.Series,
    atr_for_budget: pd.Series,
    atr_fast: pd.Series,
    atr_slow: pd.Series,
    risk_budget_pct: float,
    dynamic_enabled: bool,
    expand_threshold: float,
    contract_threshold: float,
    normal_threshold: float,
) -> tuple[pd.Series, dict[str, int]]:
    """
    Risk-budget weights with post-entry volatility-regime adjustments.

    Base rule: target_weight = risk_budget_pct * close / ATR(budget_window), fixed after entry.
    Optional dynamic regime rule (while in position):
    - NORMAL -> REDUCED when ATR(5)/ATR(50) > expand_threshold (de-risk)
    - NORMAL -> INCREASED when ATR(5)/ATR(50) < contract_threshold (add risk)
    - REDUCED -> NORMAL when ratio < normal_threshold
    - INCREASED -> NORMAL when ratio > normal_threshold
    """
    idx = active_signal.index
    sig = active_signal.reindex(idx).astype(float).fillna(0.0).clip(lower=0.0)
    px = close.reindex(idx).astype(float)
    atr_b = atr_for_budget.reindex(idx).astype(float)
    atr_f = atr_fast.reindex(idx).astype(float)
    atr_s = atr_slow.reindex(idx).astype(float)
    base_target = (
        (float(risk_budget_pct) * px / atr_b.replace(0.0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
        .astype(float)
    )
    ratio = (
        (atr_f / atr_s.replace(0.0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )

    out = pd.Series(0.0, index=idx, dtype=float)
    state = "FLAT"  # FLAT | NORMAL | REDUCED | INCREASED
    held_weight = 0.0
    stats = {
        "vol_risk_adjust_total_count": 0,
        "vol_risk_adjust_reduce_on_expand_count": 0,
        "vol_risk_adjust_increase_on_contract_count": 0,
        "vol_risk_adjust_recover_from_expand_count": 0,
        "vol_risk_adjust_recover_from_contract_count": 0,
        "vol_risk_entry_state_reduce_on_expand_count": 0,
        "vol_risk_entry_state_increase_on_contract_count": 0,
    }
    for d in idx:
        desired = float(sig.loc[d]) if np.isfinite(float(sig.loc[d])) else 0.0
        desired = max(0.0, desired)
        if desired <= 1e-12:
            state = "FLAT"
            held_weight = 0.0
            out.loc[d] = 0.0
            continue
        if state == "FLAT":
            held_weight = (
                float(base_target.loc[d])
                if np.isfinite(float(base_target.loc[d]))
                else 0.0
            )
            held_weight = float(min(1.0, max(0.0, held_weight)))
            if dynamic_enabled:
                r = (
                    float(ratio.loc[d])
                    if np.isfinite(float(ratio.loc[d]))
                    else float("nan")
                )
                if np.isfinite(r) and r > float(expand_threshold):
                    state = "REDUCED"
                    stats["vol_risk_entry_state_reduce_on_expand_count"] += 1
                elif np.isfinite(r) and r < float(contract_threshold):
                    state = "INCREASED"
                    stats["vol_risk_entry_state_increase_on_contract_count"] += 1
                else:
                    state = "NORMAL"
            else:
                state = "NORMAL"
            out.loc[d] = held_weight
            continue
        if dynamic_enabled:
            r = (
                float(ratio.loc[d])
                if np.isfinite(float(ratio.loc[d]))
                else float("nan")
            )
            can_recalc = np.isfinite(float(base_target.loc[d]))
            if state == "NORMAL":
                if np.isfinite(r) and r > float(expand_threshold) and can_recalc:
                    held_weight = float(base_target.loc[d])
                    state = "REDUCED"
                    stats["vol_risk_adjust_total_count"] += 1
                    stats["vol_risk_adjust_reduce_on_expand_count"] += 1
                elif np.isfinite(r) and r < float(contract_threshold) and can_recalc:
                    held_weight = float(base_target.loc[d])
                    state = "INCREASED"
                    stats["vol_risk_adjust_total_count"] += 1
                    stats["vol_risk_adjust_increase_on_contract_count"] += 1
            elif state == "REDUCED":
                if np.isfinite(r) and r < float(normal_threshold) and can_recalc:
                    held_weight = float(base_target.loc[d])
                    state = "NORMAL"
                    stats["vol_risk_adjust_total_count"] += 1
                    stats["vol_risk_adjust_recover_from_expand_count"] += 1
            elif state == "INCREASED":
                if np.isfinite(r) and r > float(normal_threshold) and can_recalc:
                    held_weight = float(base_target.loc[d])
                    state = "NORMAL"
                    stats["vol_risk_adjust_total_count"] += 1
                    stats["vol_risk_adjust_recover_from_contract_count"] += 1
        out.loc[d] = float(min(1.0, max(0.0, held_weight)))
    return out.astype(float), {k: int(v) for k, v in stats.items()}


def _position_risk_from_stop_params(
    *,
    atr_stop_enabled: bool,
    atr_mode: str,
    atr_basis: str,
    atr_n: float,
    atr_m: float,
    entry_px: float,
    entry_atr: float,
    curr_close: float,
    curr_atr: float,
    position_weight: float,
    fallback_position_risk: float = 0.01,
) -> float:
    """
    Return one position's risk contribution as NAV fraction.
    - valid stop: unit_risk * position_weight
    - no valid stop: fixed fallback risk per position (independent of weight)
    """
    fb = float(fallback_position_risk)
    if (not np.isfinite(fb)) or fb < 0.0:
        fb = 0.01
    w = float(position_weight) if np.isfinite(float(position_weight)) else 0.0
    w = max(0.0, w)
    if not atr_stop_enabled:
        return fb
    if (not np.isfinite(entry_px)) or entry_px <= 0.0:
        return fb
    if (not np.isfinite(entry_atr)) or entry_atr <= 0.0:
        return fb
    n_mult = float(atr_n) if np.isfinite(float(atr_n)) else 2.0
    m_step = float(atr_m) if np.isfinite(float(atr_m)) else 0.5
    mode = str(atr_mode or "none").strip().lower()
    basis = str(atr_basis or "latest").strip().lower()
    stop_px = float("nan")
    if mode == "static":
        stop_px = float(entry_px - n_mult * entry_atr)
    elif mode in {"trailing", "tightening"}:
        ref_atr = entry_atr if basis == "entry" else curr_atr
        if (
            (not np.isfinite(ref_atr))
            or ref_atr <= 0.0
            or (not np.isfinite(curr_close))
        ):
            return fb
        if mode == "trailing":
            stop_px = float(curr_close - n_mult * ref_atr)
        else:
            if (not np.isfinite(m_step)) or m_step <= 0.0:
                m_step = 0.5
            rise = max(0.0, float(curr_close - entry_px))
            den = float(m_step * ref_atr)
            steps = int(np.floor(rise / den)) if den > 0.0 else 0
            dist_mult = max(float(n_mult - steps * m_step), float(m_step))
            stop_px = float(curr_close - dist_mult * ref_atr)
    else:
        return fb
    if (not np.isfinite(stop_px)) or (entry_px <= stop_px):
        return 0.0 if np.isfinite(stop_px) else fb
    unit_risk = (float(entry_px) - float(stop_px)) / float(entry_px)
    if not np.isfinite(unit_risk):
        return fb
    return float(max(0.0, unit_risk) * w)


def _apply_monthly_risk_budget_gate(
    decision_weights: pd.DataFrame,
    *,
    close: pd.DataFrame,
    atr: pd.DataFrame,
    enabled: bool,
    budget_pct: float,
    include_new_trade_risk: bool,
    atr_stop_enabled: bool,
    atr_mode: str,
    atr_basis: str,
    atr_n: float,
    atr_m: float,
    fallback_position_risk: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Block new entries when monthly risk budget is exhausted."""
    w = (
        decision_weights.reindex(index=close.index, columns=close.columns)
        .astype(float)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    if not enabled:
        return w, {
            "enabled": False,
            "budget_pct": float(budget_pct),
            "include_new_trade_risk": bool(include_new_trade_risk),
            "attempted_entry_count": 0,
            "attempted_entry_count_by_code": {str(c): 0 for c in w.columns},
            "blocked_entry_count": 0,
            "blocked_entry_count_by_code": {str(c): 0 for c in w.columns},
        }
    bgt = float(budget_pct)
    if (not np.isfinite(bgt)) or bgt < 0.01 or bgt > 0.06:
        bgt = 0.06

    out = pd.DataFrame(0.0, index=w.index, columns=w.columns, dtype=float)
    attempted_by_code: dict[str, int] = {str(c): 0 for c in w.columns}
    blocked_by_code: dict[str, int] = {str(c): 0 for c in w.columns}
    realized_by_month: dict[str, float] = {}
    positions: dict[str, dict[str, float]] = {}
    eps = 1e-12

    for d in w.index:
        month = _month_key(d)
        desired = w.loc[d].astype(float).fillna(0.0).clip(lower=0.0).copy()
        monthly_realized = float(realized_by_month.get(month, 0.0))
        monthly_loss = max(0.0, -monthly_realized)

        current_holding_risk = 0.0
        for c, st in positions.items():
            wt = float(st.get("weight", 0.0))
            if wt <= eps:
                continue
            entry_px = float(st.get("entry_px", float("nan")))
            entry_atr = float(st.get("entry_atr", float("nan")))
            curr_close = (
                float(close.loc[d, c])
                if (
                    c in close.columns
                    and d in close.index
                    and np.isfinite(float(close.loc[d, c]))
                )
                else float("nan")
            )
            curr_atr = (
                float(atr.loc[d, c])
                if (
                    c in atr.columns
                    and d in atr.index
                    and np.isfinite(float(atr.loc[d, c]))
                )
                else float("nan")
            )
            pos_risk = _position_risk_from_stop_params(
                atr_stop_enabled=bool(atr_stop_enabled),
                atr_mode=atr_mode,
                atr_basis=atr_basis,
                atr_n=float(atr_n),
                atr_m=float(atr_m),
                entry_px=entry_px,
                entry_atr=entry_atr,
                curr_close=curr_close,
                curr_atr=curr_atr,
                position_weight=wt,
                fallback_position_risk=float(fallback_position_risk),
            )
            current_holding_risk += float(max(0.0, pos_risk))

        budget_used = float(monthly_loss + current_holding_risk)
        accepted = desired.copy()
        for c in w.columns:
            code = str(c)
            target_w = float(desired.get(c, 0.0))
            is_new_entry = (code not in positions) and (target_w > eps)
            if not is_new_entry:
                continue
            attempted_by_code[code] = int(attempted_by_code.get(code, 0) + 1)
            new_trade_risk = 0.0
            if include_new_trade_risk:
                entry_px = (
                    float(close.loc[d, c])
                    if (
                        c in close.columns
                        and d in close.index
                        and np.isfinite(float(close.loc[d, c]))
                    )
                    else float("nan")
                )
                entry_atr = (
                    float(atr.loc[d, c])
                    if (
                        c in atr.columns
                        and d in atr.index
                        and np.isfinite(float(atr.loc[d, c]))
                    )
                    else float("nan")
                )
                new_trade_risk = _position_risk_from_stop_params(
                    atr_stop_enabled=bool(atr_stop_enabled),
                    atr_mode=atr_mode,
                    atr_basis=atr_basis,
                    atr_n=float(atr_n),
                    atr_m=float(atr_m),
                    entry_px=entry_px,
                    entry_atr=entry_atr,
                    curr_close=entry_px,
                    curr_atr=entry_atr,
                    position_weight=target_w,
                    fallback_position_risk=float(fallback_position_risk),
                )
                new_trade_risk = float(max(0.0, new_trade_risk))
            if float(budget_used + new_trade_risk) >= float(bgt) - 1e-12:
                accepted.loc[c] = 0.0
                blocked_by_code[code] = int(blocked_by_code.get(code, 0) + 1)
                continue
            budget_used += float(new_trade_risk)

        out.loc[d] = accepted.to_numpy(dtype=float)

        # Realize PnL for positions closed by today's accepted decision.
        for c in list(positions.keys()):
            nw = float(accepted.get(c, 0.0))
            if nw > eps:
                positions[c]["weight"] = nw
                continue
            entry_px = float(positions[c].get("entry_px", float("nan")))
            entry_w = float(
                positions[c].get("entry_weight", positions[c].get("weight", 0.0))
            )
            exit_px = (
                float(close.loc[d, c])
                if (
                    c in close.columns
                    and d in close.index
                    and np.isfinite(float(close.loc[d, c]))
                )
                else float("nan")
            )
            realized = 0.0
            if np.isfinite(entry_px) and entry_px > 0.0 and np.isfinite(exit_px):
                realized = float((exit_px / entry_px - 1.0) * entry_w)
            realized_by_month[month] = float(
                realized_by_month.get(month, 0.0) + realized
            )
            positions.pop(c, None)

        # Register new opens accepted today.
        for c in w.columns:
            code = str(c)
            nw = float(accepted.get(c, 0.0))
            if nw <= eps or code in positions:
                continue
            entry_px = (
                float(close.loc[d, c])
                if (
                    c in close.columns
                    and d in close.index
                    and np.isfinite(float(close.loc[d, c]))
                )
                else float("nan")
            )
            entry_atr = (
                float(atr.loc[d, c])
                if (
                    c in atr.columns
                    and d in atr.index
                    and np.isfinite(float(atr.loc[d, c]))
                )
                else float("nan")
            )
            positions[code] = {
                "entry_px": float(entry_px),
                "entry_atr": float(entry_atr),
                "entry_weight": float(nw),
                "weight": float(nw),
            }

    total_attempted = int(sum(int(v) for v in attempted_by_code.values()))
    total_blocked = int(sum(int(v) for v in blocked_by_code.values()))
    return out.astype(float), {
        "enabled": True,
        "budget_pct": float(bgt),
        "include_new_trade_risk": bool(include_new_trade_risk),
        "attempted_entry_count": int(total_attempted),
        "attempted_entry_count_by_code": {
            str(k): int(v) for k, v in attempted_by_code.items()
        },
        "blocked_entry_count": int(total_blocked),
        "blocked_entry_count_by_code": {
            str(k): int(v) for k, v in blocked_by_code.items()
        },
    }


def _turnover_cost_from_weights(w: pd.Series, *, cost_bps: float) -> pd.Series:
    w_prev = w.shift(1).fillna(0.0).astype(float)
    turnover = (w.astype(float) - w_prev).abs() / 2.0
    cost = turnover * (float(cost_bps) / 10000.0)
    return cost.astype(float)


def _latest_entry_exec_price_with_slippage(
    *,
    effective_weight: pd.Series,
    exec_price_series: pd.Series,
    slippage_spread: float,
) -> float | None:
    w = effective_weight.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if w.empty:
        return None
    if float(w.iloc[-1]) <= 1e-12:
        return None
    w_prev = w.shift(1).fillna(0.0).astype(float)
    is_entry = (w > 1e-12) & (w_prev <= 1e-12)
    entry_dates = list(w.index[is_entry])
    if not entry_dates:
        return None
    entry_dt = entry_dates[-1]
    px = (
        pd.to_numeric(exec_price_series, errors="coerce")
        .astype(float)
        .reindex(w.index)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )
    px0 = (
        float(px.loc[entry_dt])
        if entry_dt in px.index and np.isfinite(float(px.loc[entry_dt]))
        else float("nan")
    )
    if (not np.isfinite(px0)) or px0 <= 0.0:
        return None
    spread = (
        float(slippage_spread)
        if np.isfinite(float(slippage_spread)) and float(slippage_spread) > 0.0
        else 0.0
    )
    # Turnover pipeline uses one-way turnover (|dw|/2), so one-side execution price impact is spread/2.
    return float(px0 + 0.5 * spread)


def _extract_atr_plan_stops_from_trace(
    stats: dict[str, Any],
) -> dict[str, float | None]:
    rows = list((stats or {}).get("trace_last_rows") or [])
    if not rows:
        return {"plan_stop_current": None, "plan_stop_next": None}
    episode: list[dict[str, Any]] = []
    for r in rows:
        rr = r or {}
        in_pos = bool(rr.get("in_pos_after")) or (
            float(rr.get("decision_pos") or 0.0) > 1e-12
        )
        if in_pos:
            episode.append(rr)
        else:
            episode = []
    if not episode:
        return {"plan_stop_current": None, "plan_stop_next": None}

    def _to_finite(v: Any) -> float:
        try:
            x = float(v)
        except Exception:
            return float("nan")
        return x if np.isfinite(x) else float("nan")

    last = episode[-1] or {}
    prev = episode[-2] if len(episode) >= 2 else None
    stop_next = _to_finite(last.get("stop_after"))
    stop_cur = _to_finite(last.get("stop_before"))
    if (not np.isfinite(stop_cur)) and isinstance(prev, dict):
        stop_cur = _to_finite(prev.get("stop_after"))
    if (not np.isfinite(stop_cur)) and np.isfinite(stop_next):
        stop_cur = stop_next
    return {
        "plan_stop_current": (float(stop_cur) if np.isfinite(stop_cur) else None),
        "plan_stop_next": (float(stop_next) if np.isfinite(stop_next) else None),
    }


def _enrich_trade_records_with_engine_timeline(
    *,
    records: list[dict[str, Any]] | None,
    effective_weight: pd.Series,
    exec_price_series: pd.Series,
    slippage_spread: float,
) -> list[dict[str, Any]]:
    recs = [dict(r or {}) for r in list(records or [])]
    if not recs:
        return []
    w = effective_weight.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    px = (
        pd.to_numeric(exec_price_series, errors="coerce")
        .astype(float)
        .reindex(w.index)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )
    spread = (
        float(slippage_spread)
        if np.isfinite(float(slippage_spread)) and float(slippage_spread) > 0.0
        else 0.0
    )
    by_trigger_date: dict[str, list[int]] = {}
    for i, rr in enumerate(recs):
        d = str(rr.get("trigger_date") or "").strip()
        if d:
            by_trigger_date.setdefault(d, []).append(i)
    prev = w.shift(1).fillna(0.0).astype(float)
    open_entry_date: str | None = None
    open_entry_price: float | None = None
    for i, dt0 in enumerate(w.index):
        if float(w.iloc[i]) > 1e-12 and float(prev.iloc[i]) <= 1e-12:
            ds = dt0.date().isoformat() if hasattr(dt0, "date") else str(dt0)
            p = float(px.iloc[i]) if np.isfinite(float(px.iloc[i])) else float("nan")
            open_entry_date = str(ds)
            open_entry_price = (
                float(p + 0.5 * spread) if np.isfinite(p) and p > 0.0 else None
            )
        ds = dt0.date().isoformat() if hasattr(dt0, "date") else str(dt0)
        for ridx in by_trigger_date.get(str(ds), []):
            if (
                open_entry_date is not None
                and recs[ridx].get("entry_execution_date") is None
            ):
                recs[ridx]["entry_execution_date"] = str(open_entry_date)
            if (
                open_entry_price is not None
                and recs[ridx].get("entry_execution_price") is None
            ):
                recs[ridx]["entry_execution_price"] = float(open_entry_price)
            open_entry_date = None
            open_entry_price = None
    return recs


def _period_returns(nav: pd.Series, freq: str) -> pd.DataFrame:
    s = pd.Series(nav).astype(float)
    if s.empty:
        return pd.DataFrame(columns=["period_end", "return"])
    p = s.resample(freq).last().dropna()
    if p.empty:
        return pd.DataFrame(columns=["period_end", "return"])
    r = p.pct_change().dropna()
    if r.empty:
        return pd.DataFrame(columns=["period_end", "return"])
    return pd.DataFrame(
        {
            "period_end": r.index.date.astype(str),
            "return": r.astype(float).to_numpy(),
        }
    )


def _rolling_pack(nav_s: pd.Series) -> dict[str, dict[str, Any]]:
    rolling: dict[str, dict[str, Any]] = {
        "returns": {},
        "drawdown": {},
        "max_drawdown": {},
    }
    for weeks in [4, 12, 52]:
        window = int(weeks) * 5
        r = (nav_s / nav_s.shift(window) - 1.0).dropna()
        d = _rolling_drawdown(nav_s, window).dropna()
        rolling["returns"][f"{weeks}w"] = {
            "dates": r.index.date.astype(str).tolist(),
            "values": r.astype(float).tolist(),
        }
        rolling["drawdown"][f"{weeks}w"] = {
            "dates": d.index.date.astype(str).tolist(),
            "values": d.astype(float).tolist(),
        }
        rolling["max_drawdown"][f"{weeks}w"] = {
            "dates": d.index.date.astype(str).tolist(),
            "values": d.astype(float).tolist(),
        }
    for months in [3, 6, 12]:
        window = int(months) * 21
        r = (nav_s / nav_s.shift(window) - 1.0).dropna()
        d = _rolling_drawdown(nav_s, window).dropna()
        rolling["returns"][f"{months}m"] = {
            "dates": r.index.date.astype(str).tolist(),
            "values": r.astype(float).tolist(),
        }
        rolling["drawdown"][f"{months}m"] = {
            "dates": d.index.date.astype(str).tolist(),
            "values": d.astype(float).tolist(),
        }
        rolling["max_drawdown"][f"{months}m"] = {
            "dates": d.index.date.astype(str).tolist(),
            "values": d.astype(float).tolist(),
        }
    for years in [1, 3]:
        window = int(years) * 252
        r = (nav_s / nav_s.shift(window) - 1.0).dropna()
        d = _rolling_drawdown(nav_s, window).dropna()
        rolling["returns"][f"{years}y"] = {
            "dates": r.index.date.astype(str).tolist(),
            "values": r.astype(float).tolist(),
        }
        rolling["drawdown"][f"{years}y"] = {
            "dates": d.index.date.astype(str).tolist(),
            "values": d.astype(float).tolist(),
        }
        rolling["max_drawdown"][f"{years}y"] = {
            "dates": d.index.date.astype(str).tolist(),
            "values": d.astype(float).tolist(),
        }
    return rolling


def _dist_stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(
        [float(x) for x in (values or []) if np.isfinite(float(x))], dtype=float
    )
    if arr.size == 0:
        return {
            "count": 0,
            "max": None,
            "min": None,
            "mean": None,
            "std": None,
            "quantiles": {
                k: None
                for k in ["p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
            },
        }
    q = lambda p: float(np.percentile(arr, p))  # noqa: E731
    return {
        "count": int(arr.size),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0,
        "quantiles": {
            "p01": q(1),
            "p05": q(5),
            "p10": q(10),
            "p25": q(25),
            "p50": q(50),
            "p75": q(75),
            "p90": q(90),
            "p95": q(95),
            "p99": q(99),
        },
    }


def _trade_stats_from_returns(
    values: list[float], *, flat_eps: float = 1e-12
) -> dict[str, Any]:
    rs = [float(x) for x in (values or []) if np.isfinite(float(x))]
    wins = [x for x in rs if x > float(flat_eps)]
    losses = [x for x in rs if x < -float(flat_eps)]
    flats = [x for x in rs if abs(float(x)) <= float(flat_eps)]
    # Kelly (exclude flat/zero-return trades): f* = p - (1-p)/b, b=avg_win/|avg_loss|
    # Return None when the denominator is not well-defined (no wins/losses).
    win_rate_ex_zero: float | None = None
    payoff_ex_zero: float | None = None
    kelly_ex_zero: float | None = None
    if wins and losses:
        avg_win = float(np.mean(np.asarray(wins, dtype=float)))
        avg_loss_abs = float(abs(np.mean(np.asarray(losses, dtype=float))))
        if (
            np.isfinite(avg_win)
            and np.isfinite(avg_loss_abs)
            and avg_win > 0.0
            and avg_loss_abs > 0.0
        ):
            b = float(avg_win / avg_loss_abs)
            p = float(len(wins) / (len(wins) + len(losses)))
            win_rate_ex_zero = p
            payoff_ex_zero = b
            kelly_ex_zero = float(p - (1.0 - p) / b)
    return {
        "total_trades": int(len(rs)),
        "win_trades": int(len(wins)),
        "loss_trades": int(len(losses)),
        "flat_trades": int(len(flats)),
        "win_rate_ex_zero": win_rate_ex_zero,
        "payoff_ex_zero": payoff_ex_zero,
        "kelly_ex_zero": kelly_ex_zero,
        "returns": [float(x) for x in rs],
        "all_stats": _dist_stats(rs),
        "profit_stats": _dist_stats(wins),
        "loss_stats": _dist_stats(losses),
    }


def _round_half_up(value: float, ndigits: int = 0) -> float:
    if not np.isfinite(float(value)):
        return float("nan")
    fac = float(10 ** int(max(0, ndigits)))
    x = float(value) * fac
    if x >= 0.0:
        return float(np.floor(x + 0.5) / fac)
    return float(np.ceil(x - 0.5) / fac)


def _bucketize_momentum_series(momentum: pd.Series) -> pd.Series:
    out = pd.Series("NA", index=momentum.index, dtype=object)
    m = pd.to_numeric(momentum, errors="coerce").astype(float) * 100.0
    for d in out.index:
        v = float(m.loc[d]) if d in m.index else float("nan")
        if not np.isfinite(v):
            continue
        iv = int(_round_half_up(v, ndigits=0))
        out.loc[d] = f"{iv:d}%"
    return out


def _bucketize_er_series(er: pd.Series) -> pd.Series:
    out = pd.Series("NA", index=er.index, dtype=object)
    e = pd.to_numeric(er, errors="coerce").astype(float)
    for d in out.index:
        v = float(e.loc[d]) if d in e.index else float("nan")
        if not np.isfinite(v):
            continue
        vv = float(np.clip(v, 0.0, 1.0))
        b = _round_half_up(vv, ndigits=1)
        out.loc[d] = f"{b:.1f}"
    return out


def _bucketize_vol_ratio_series(vol_ratio: pd.Series) -> pd.Series:
    out = pd.Series("NA", index=vol_ratio.index, dtype=object)
    r = pd.to_numeric(vol_ratio, errors="coerce").astype(float)
    for d in out.index:
        v = float(r.loc[d]) if d in r.index else float("nan")
        if not np.isfinite(v):
            continue
        out.loc[d] = f"{_round_half_up(v, ndigits=1):.1f}"
    return out


def _bucketize_impulse_series(impulse_state: pd.Series) -> pd.Series:
    out = pd.Series("NA", index=impulse_state.index, dtype=object)
    for d in out.index:
        st = (
            str(impulse_state.loc[d]).strip().upper()
            if d in impulse_state.index
            else "NA"
        )
        if st in {"BULL", "BEAR", "NEUTRAL"}:
            out.loc[d] = st
    return out


def _series_index_to_date_str(series: pd.Series) -> pd.Series:
    s = series.copy()
    s.index = pd.to_datetime(s.index).date.astype(str)
    return s


def _build_entry_signal_date_map(dates: pd.Index) -> dict[str, str | None]:
    ds = [str(pd.to_datetime(d).date()) for d in dates]
    out: dict[str, str | None] = {}
    for i, d in enumerate(ds):
        out[d] = ds[i - 1] if i > 0 else None
    return out


def _attach_entry_condition_bins_to_trades(
    trades: list[dict[str, Any]],
    *,
    condition_bins_by_code: dict[str, dict[str, pd.Series]],
    dates: pd.Index,
    default_code: str | None = None,
) -> list[dict[str, Any]]:
    prev_map = _build_entry_signal_date_map(dates)
    normalized: dict[str, dict[str, pd.Series]] = {}
    for code, mp in (condition_bins_by_code or {}).items():
        one: dict[str, pd.Series] = {}
        for k, s in (mp or {}).items():
            one[str(k)] = _series_index_to_date_str(
                s if isinstance(s, pd.Series) else pd.Series(dtype=object)
            )
        normalized[str(code)] = one
    out: list[dict[str, Any]] = []
    for tr in list(trades or []):
        row = dict(tr or {})
        code = str(row.get("code") or default_code or "").strip()
        entry_date = str(row.get("entry_date") or "").strip()
        signal_date = prev_map.get(entry_date)
        bins: dict[str, str] = {}
        one_code = normalized.get(code, {})
        for cond in ("momentum", "er", "vol_ratio", "impulse"):
            ss = one_code.get(cond)
            v = "NA"
            if ss is not None and signal_date and (signal_date in ss.index):
                vv = str(ss.loc[signal_date]).strip()
                v = vv if vv else "NA"
            bins[cond] = v
        row["code"] = code
        row["entry_signal_date"] = signal_date
        row["entry_condition_bins"] = bins
        out.append(row)
    return out


def _normal_two_sided_p_from_z(z: float) -> float | None:
    if not np.isfinite(float(z)):
        return None
    return float(max(0.0, min(1.0, math.erfc(abs(float(z)) / math.sqrt(2.0)))))


def _two_proportion_z_test(
    *, wins_a: int, losses_a: int, wins_b: int, losses_b: int
) -> float | None:
    na = int(wins_a) + int(losses_a)
    nb = int(wins_b) + int(losses_b)
    if na <= 0 or nb <= 0:
        return None
    pa = float(wins_a) / float(na)
    pb = float(wins_b) / float(nb)
    p_pool = float(wins_a + wins_b) / float(na + nb)
    denom = p_pool * (1.0 - p_pool) * (1.0 / float(na) + 1.0 / float(nb))
    if denom <= 0.0:
        return None
    z = (pa - pb) / float(np.sqrt(denom))
    return _normal_two_sided_p_from_z(z)


def _welch_t_test_normal_approx(a: list[float], b: list[float]) -> float | None:
    xa = np.asarray([float(x) for x in (a or []) if np.isfinite(float(x))], dtype=float)
    xb = np.asarray([float(x) for x in (b or []) if np.isfinite(float(x))], dtype=float)
    if xa.size < 2 or xb.size < 2:
        return None
    va = float(np.var(xa, ddof=1))
    vb = float(np.var(xb, ddof=1))
    denom = (va / float(xa.size)) + (vb / float(xb.size))
    if (not np.isfinite(denom)) or denom <= 0.0:
        return None
    z = (float(np.mean(xa)) - float(np.mean(xb))) / float(np.sqrt(denom))
    return _normal_two_sided_p_from_z(z)


def _bh_qvalues(ps: list[float | None]) -> list[float | None]:
    pairs = [
        (i, float(p))
        for i, p in enumerate(ps)
        if (p is not None and np.isfinite(float(p)))
    ]
    m = int(len(pairs))
    out: list[float | None] = [None for _ in ps]
    if m <= 0:
        return out
    pairs_sorted = sorted(pairs, key=lambda x: x[1])
    q_raw: list[float] = [0.0] * m
    for rank, (_, p) in enumerate(pairs_sorted, start=1):
        q_raw[rank - 1] = float(min(1.0, p * m / float(rank)))
    q_adj: list[float] = [0.0] * m
    running = 1.0
    for i in range(m - 1, -1, -1):
        running = min(running, q_raw[i])
        q_adj[i] = float(running)
    for i, (orig_idx, _) in enumerate(pairs_sorted):
        out[orig_idx] = q_adj[i]
    return out


def _stratified_permutation_pvalue(
    rows: list[dict[str, Any]],
    *,
    in_bin: str,
    value_key: str,
    strata_key: str,
    n_perm: int,
    seed: int,
) -> float | None:
    vals: list[tuple[str, bool, float]] = []
    for r in rows:
        b = str(r.get("bucket") or "")
        v = r.get(value_key)
        if v is None or (not np.isfinite(float(v))):
            continue
        vals.append((str(r.get(strata_key) or "NA"), b == in_bin, float(v)))
    if not vals:
        return None
    sum_in = float(sum(v for _, f, v in vals if f))
    n_in = int(sum(1 for _, f, _ in vals if f))
    sum_out = float(sum(v for _, f, v in vals if (not f)))
    n_out = int(sum(1 for _, f, _ in vals if (not f)))
    if n_in <= 0 or n_out <= 0:
        return None
    obs = (sum_in / float(n_in)) - (sum_out / float(n_out))
    by_strata: dict[str, list[tuple[bool, float]]] = {}
    for st, f, v in vals:
        by_strata.setdefault(st, []).append((f, v))
    rng = np.random.default_rng(int(seed))
    exceed = 0
    perm_n = int(max(50, n_perm))
    for _ in range(perm_n):
        p_sum_in = 0.0
        p_n_in = 0
        p_sum_out = 0.0
        p_n_out = 0
        for rs in by_strata.values():
            n = int(len(rs))
            if n <= 0:
                continue
            vals_s = np.asarray([float(x[1]) for x in rs], dtype=float)
            k = int(sum(1 for x in rs if x[0]))
            if k <= 0:
                p_sum_out += float(vals_s.sum())
                p_n_out += n
                continue
            if k >= n:
                p_sum_in += float(vals_s.sum())
                p_n_in += n
                continue
            idx_in = rng.choice(n, size=k, replace=False)
            mask = np.zeros(n, dtype=bool)
            mask[idx_in] = True
            p_sum_in += float(vals_s[mask].sum())
            p_n_in += int(mask.sum())
            p_sum_out += float(vals_s[~mask].sum())
            p_n_out += int((~mask).sum())
        if p_n_in <= 0 or p_n_out <= 0:
            continue
        diff = (p_sum_in / float(p_n_in)) - (p_sum_out / float(p_n_out))
        if abs(float(diff)) >= abs(float(obs)) - 1e-18:
            exceed += 1
    return float((exceed + 1) / float(perm_n + 1))


def _sorted_condition_buckets(condition: str, buckets: list[str]) -> list[str]:
    uniq = [str(b) for b in buckets if str(b).strip()]
    if str(condition) == "impulse":
        order = {"BULL": 0, "NEUTRAL": 1, "BEAR": 2, "NA": 99}
        return sorted(set(uniq), key=lambda x: (order.get(str(x).upper(), 50), str(x)))

    def _num_key(x: str) -> tuple[int, float]:
        s = str(x).strip().replace("%", "")
        try:
            return (0, float(s))
        except ValueError:
            return (1, float("inf"))

    return sorted(set(uniq), key=_num_key)


def _stable_seed_from_text(text: str) -> int:
    acc = 0
    for i, ch in enumerate(str(text)):
        acc = (acc + (i + 1) * ord(ch)) % 1_000_003
    return int(acc)


def _build_entry_condition_stats(
    trades: list[dict[str, Any]],
    *,
    by_code: bool,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    closed = []
    for tr in list(trades or []):
        if not bool(tr.get("closed", False)):
            continue
        rr = tr.get("return")
        if rr is None or (not np.isfinite(float(rr))):
            continue
        bins = tr.get("entry_condition_bins") or {}
        if not isinstance(bins, dict):
            bins = {}
        entry_signal_date = str(tr.get("entry_signal_date") or "")
        yyyy = ""
        if entry_signal_date:
            try:
                yyyy = str(pd.to_datetime(entry_signal_date).year)
            except (TypeError, ValueError):
                yyyy = ""
        closed.append(
            {
                "code": str(tr.get("code") or ""),
                "return": float(rr),
                "bucket_map": {k: str(v) for k, v in bins.items()},
                "year": yyyy,
            }
        )
    out: dict[str, Any] = {}
    conditions = ("momentum", "er", "vol_ratio", "impulse")
    for cond in conditions:
        rows: list[dict[str, Any]] = []
        for r in closed:
            b = str((r.get("bucket_map") or {}).get(cond, "NA") or "NA")
            rr = float(r["return"])
            rows.append(
                {
                    "bucket": b,
                    "return": rr,
                    "is_win": 1 if rr > 1e-12 else 0,
                    "is_loss": 1 if rr < -1e-12 else 0,
                    "strata": (
                        f"{r['year']}" if by_code else f"{r['code']}|{r['year']}"
                    ),
                }
            )
        uniq = _sorted_condition_buckets(
            cond, [str(x.get("bucket") or "NA") for x in rows]
        )
        bins_out: list[dict[str, Any]] = []
        p_quasi_win: list[float | None] = []
        p_quasi_ret: list[float | None] = []
        p_strong_win: list[float | None] = []
        p_strong_ret: list[float | None] = []
        for b in uniq:
            in_rows = [x for x in rows if str(x.get("bucket")) == b]
            out_rows = [x for x in rows if str(x.get("bucket")) != b]
            in_rets = [float(x["return"]) for x in in_rows]
            out_rets = [float(x["return"]) for x in out_rows]
            in_nonflat = [
                x for x in in_rows if int(x["is_win"]) == 1 or int(x["is_loss"]) == 1
            ]
            out_nonflat = [
                x for x in out_rows if int(x["is_win"]) == 1 or int(x["is_loss"]) == 1
            ]
            in_wins = int(sum(int(x["is_win"]) for x in in_nonflat))
            in_losses = int(sum(int(x["is_loss"]) for x in in_nonflat))
            out_wins = int(sum(int(x["is_win"]) for x in out_nonflat))
            out_losses = int(sum(int(x["is_loss"]) for x in out_nonflat))
            n_in_eff = int(in_wins + in_losses)
            n_out_eff = int(out_wins + out_losses)
            wr_in = (float(in_wins) / float(n_in_eff)) if n_in_eff > 0 else None
            wr_out = (float(out_wins) / float(n_out_eff)) if n_out_eff > 0 else None
            mean_in = (
                float(np.mean(np.asarray(in_rets, dtype=float))) if in_rets else None
            )
            mean_out = (
                float(np.mean(np.asarray(out_rets, dtype=float))) if out_rets else None
            )
            uplift_win = (
                (wr_in - wr_out) if (wr_in is not None and wr_out is not None) else None
            )
            uplift_ret = (
                (mean_in - mean_out)
                if (mean_in is not None and mean_out is not None)
                else None
            )
            quasi_p_win = _two_proportion_z_test(
                wins_a=in_wins,
                losses_a=in_losses,
                wins_b=out_wins,
                losses_b=out_losses,
            )
            quasi_p_ret = _welch_t_test_normal_approx(in_rets, out_rets)
            p_quasi_win.append(quasi_p_win)
            p_quasi_ret.append(quasi_p_ret)
            win_rows_perm = []
            for x in rows:
                if int(x["is_win"]) == 1 or int(x["is_loss"]) == 1:
                    win_rows_perm.append(
                        {
                            "bucket": str(x["bucket"]),
                            "value": float(x["is_win"]),
                            "strata": str(x["strata"]),
                        }
                    )
            ret_rows_perm = [
                {
                    "bucket": str(x["bucket"]),
                    "value": float(x["return"]),
                    "strata": str(x["strata"]),
                }
                for x in rows
            ]
            strong_p_win = _stratified_permutation_pvalue(
                win_rows_perm,
                in_bin=str(b),
                value_key="value",
                strata_key="strata",
                n_perm=int(n_perm),
                seed=int(seed + _stable_seed_from_text(f"{cond}|{b}|win")),
            )
            strong_p_ret = _stratified_permutation_pvalue(
                ret_rows_perm,
                in_bin=str(b),
                value_key="value",
                strata_key="strata",
                n_perm=int(n_perm),
                seed=int(seed + _stable_seed_from_text(f"{cond}|{b}|ret")),
            )
            p_strong_win.append(strong_p_win)
            p_strong_ret.append(strong_p_ret)
            bins_out.append(
                {
                    "bucket": str(b),
                    "closed_trade_count": int(len(in_rows)),
                    "win_rate_ex_zero": wr_in,
                    "mean_return": mean_in,
                    "quasi_causal": {
                        "uplift_win_rate_ex_zero": uplift_win,
                        "uplift_mean_return": uplift_ret,
                        "p_value_win_rate": quasi_p_win,
                        "p_value_mean_return": quasi_p_ret,
                    },
                    "strong_causal": {
                        "uplift_win_rate_ex_zero": uplift_win,
                        "uplift_mean_return": uplift_ret,
                        "p_value_win_rate": strong_p_win,
                        "p_value_mean_return": strong_p_ret,
                        "method": "stratified_permutation",
                    },
                }
            )
        q_quasi_win = _bh_qvalues(p_quasi_win)
        q_quasi_ret = _bh_qvalues(p_quasi_ret)
        q_strong_win = _bh_qvalues(p_strong_win)
        q_strong_ret = _bh_qvalues(p_strong_ret)
        for i in range(len(bins_out)):
            bins_out[i]["quasi_causal"]["q_value_win_rate"] = q_quasi_win[i]
            bins_out[i]["quasi_causal"]["q_value_mean_return"] = q_quasi_ret[i]
            bins_out[i]["strong_causal"]["q_value_win_rate"] = q_strong_win[i]
            bins_out[i]["strong_causal"]["q_value_mean_return"] = q_strong_ret[i]
        out[cond] = {
            "closed_trade_count": int(len(rows)),
            "bucket_count": int(len(bins_out)),
            "bins": bins_out,
        }
    return out


def _trade_returns_from_weight_series(
    w: pd.Series,
    ret_exec: pd.Series,
    *,
    cost_bps: float,
    slippage_rate: float,
    exec_price: pd.Series,
    dates: pd.Index,
    eps: float = 1e-12,
) -> dict[str, Any]:
    ww = pd.to_numeric(w, errors="coerce").astype(float).reindex(dates).fillna(0.0)
    rr = (
        pd.to_numeric(ret_exec, errors="coerce")
        .astype(float)
        .reindex(dates)
        .fillna(0.0)
    )
    n = int(len(dates))
    if n <= 0:
        return {"returns": [], "trades": []}
    cost_rate = float(cost_bps) / 10000.0
    px = (
        pd.to_numeric(exec_price, errors="coerce")
        .astype(float)
        .reindex(dates)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )
    slip_spread = float(slippage_rate)
    returns: list[float] = []
    trades: list[dict[str, Any]] = []
    active = False
    start_i = -1
    start_nav = 1.0
    nav_prev = 1.0
    for i in range(n):
        cur = float(ww.iloc[i])
        prev = float(ww.iloc[i - 1]) if i > 0 else 0.0
        r = float(rr.iloc[i]) if np.isfinite(float(rr.iloc[i])) else 0.0
        turnover = abs(float(cur) - float(prev)) / 2.0
        px_i = (
            float(px.iloc[i])
            if np.isfinite(float(px.iloc[i])) and float(px.iloc[i]) > 0.0
            else float("nan")
        )
        slip_ret = (
            float(turnover) * (float(slip_spread) / float(px_i))
            if np.isfinite(px_i) and float(slip_spread) > 0.0
            else 0.0
        )
        day_ret = (
            float(cur) * float(r) - float(turnover) * float(cost_rate) - float(slip_ret)
        )
        if (not active) and (prev <= eps) and (cur > eps):
            active = True
            start_i = int(i)
            start_nav = float(nav_prev)
        nav_cur = float(nav_prev) * (1.0 + float(day_ret))
        # Trade ends on the execution day when position becomes flat (exit cost is booked on this day).
        if active and (prev > eps) and (cur <= eps):
            tr = (
                (float(nav_cur) / float(start_nav) - 1.0)
                if float(start_nav) != 0
                else float("nan")
            )
            returns.append(float(tr))
            trades.append(
                {
                    "entry_date": str(pd.to_datetime(dates[start_i]).date())
                    if start_i >= 0
                    else None,
                    "exit_date": str(pd.to_datetime(dates[i]).date()),
                    "return": float(tr),
                    "closed": True,
                }
            )
            active = False
            start_i = -1
            start_nav = float(nav_cur)
        nav_prev = float(nav_cur)
    # If trade is still open at the end, include mark-to-market return up to last available date.
    if active and start_i >= 0:
        tr = (
            (float(nav_prev) / float(start_nav) - 1.0)
            if float(start_nav) != 0
            else float("nan")
        )
        returns.append(float(tr))
        trades.append(
            {
                "entry_date": str(pd.to_datetime(dates[start_i]).date()),
                "exit_date": str(pd.to_datetime(dates[n - 1]).date()),
                "return": float(tr),
                "closed": False,
            }
        )
    return {"returns": returns, "trades": trades}


def _trade_returns_from_weight_df(
    w: pd.DataFrame,
    ret_exec: pd.DataFrame,
    *,
    cost_bps: float,
    slippage_rate: float,
    exec_price: pd.DataFrame,
    dates: pd.Index,
    eps: float = 1e-12,
) -> dict[str, Any]:
    by_code_returns: dict[str, list[float]] = {}
    by_code_trades: dict[str, list[dict[str, Any]]] = {}
    for c in [str(x) for x in w.columns]:
        one = _trade_returns_from_weight_series(
            w[c] if c in w.columns else pd.Series(dtype=float),
            ret_exec[c] if c in ret_exec.columns else pd.Series(dtype=float),
            cost_bps=float(cost_bps),
            slippage_rate=float(slippage_rate),
            exec_price=exec_price[c]
            if c in exec_price.columns
            else pd.Series(dtype=float),
            dates=dates,
            eps=float(eps),
        )
        by_code_returns[str(c)] = [float(x) for x in (one.get("returns") or [])]
        by_code_trades[str(c)] = list(one.get("trades") or [])
    all_returns: list[float] = []
    all_trades: list[dict[str, Any]] = []
    for c in by_code_returns:
        all_returns.extend(by_code_returns[c])
        all_trades.extend([{**x, "code": str(c)} for x in by_code_trades.get(c, [])])
    return {
        "returns": [float(x) for x in all_returns],
        "trades": all_trades,
        "returns_by_code": by_code_returns,
        "trades_by_code": by_code_trades,
    }


def _pos_from_donchian(close: pd.Series, *, entry: int, exit_: int) -> pd.Series:
    """
    Long/cash Donchian channel:
    - enter long when close > max(close, entry window) excluding today
    - exit to cash when close < min(close, exit window) excluding today
    """
    e = max(2, int(entry))
    x = max(2, int(exit_))
    hi = close.shift(1).rolling(window=e, min_periods=e).max()
    lo = close.shift(1).rolling(window=x, min_periods=x).min()
    pos = np.zeros(len(close), dtype=float)
    in_pos = False
    for i in range(len(close)):
        c = float(close.iloc[i]) if pd.notna(close.iloc[i]) else float("nan")
        if not np.isfinite(c):
            pos[i] = 1.0 if in_pos else 0.0
            continue
        h = hi.iloc[i]
        low_v = lo.iloc[i]
        if (not in_pos) and pd.notna(h) and c > float(h):
            in_pos = True
        elif in_pos and pd.notna(low_v) and c < float(low_v):
            in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=close.index, dtype=float)


def _ema(s: pd.Series, window: int) -> pd.Series:
    w = max(2, int(window))
    return s.ewm(span=w, adjust=False, min_periods=max(2, w // 2)).mean()


def _kama(
    s: pd.Series,
    *,
    er_window: int = 10,
    fast_window: int = 2,
    slow_window: int = 30,
) -> pd.Series:
    """
    Kaufman Adaptive Moving Average (KAMA).
    """
    p = pd.to_numeric(s, errors="coerce").astype(float)
    er_w = max(2, int(er_window))
    fast_w = max(1, int(fast_window))
    slow_w = max(2, int(slow_window))
    if fast_w >= slow_w:
        fast_w = max(1, slow_w - 1)

    change = (p - p.shift(er_w)).abs()
    volatility = p.diff().abs().rolling(window=er_w, min_periods=er_w).sum()
    er = (change / volatility.replace(0.0, np.nan)).clip(lower=0.0, upper=1.0)

    fast_sc = 2.0 / (float(fast_w) + 1.0)
    slow_sc = 2.0 / (float(slow_w) + 1.0)
    sc = ((er * (fast_sc - slow_sc) + slow_sc) ** 2).astype(float)

    out = np.full(len(p), np.nan, dtype=float)
    vals = p.to_numpy(dtype=float)
    sc_vals = sc.to_numpy(dtype=float)

    first_idx = None
    for i, v in enumerate(vals):
        if np.isfinite(v):
            first_idx = i
            break
    if first_idx is None:
        return pd.Series(out, index=p.index, dtype=float)

    out[first_idx] = float(vals[first_idx])
    for i in range(first_idx + 1, len(vals)):
        v = vals[i]
        prev = out[i - 1]
        if not np.isfinite(v):
            out[i] = prev
            continue
        if not np.isfinite(prev):
            out[i] = v
            continue
        alpha = sc_vals[i] if np.isfinite(sc_vals[i]) else (slow_sc * slow_sc)
        out[i] = float(prev + alpha * (v - prev))
    return pd.Series(out, index=p.index, dtype=float)


def _moving_average(
    s: pd.Series,
    *,
    window: int,
    ma_type: str,
    kama_er_window: int = 10,
    kama_fast_window: int = 2,
    kama_slow_window: int = 30,
) -> pd.Series:
    t = str(ma_type or "sma").strip().lower()
    if t == "ema":
        return _ema(s, int(window))
    if t == "wma":
        w = max(2, int(window))
        weights = np.arange(1, w + 1, dtype=float)

        def _wma_window(arr: np.ndarray) -> float:
            if arr.size != w or not np.all(np.isfinite(arr)):
                return float("nan")
            return float(np.dot(arr.astype(float), weights) / weights.sum())

        return (
            pd.to_numeric(s, errors="coerce")
            .astype(float)
            .rolling(window=w, min_periods=w)
            .apply(_wma_window, raw=True)
        )
    if t == "kama":
        return _kama(
            s,
            er_window=int(kama_er_window),
            fast_window=int(kama_fast_window),
            slow_window=int(kama_slow_window),
        )
    w = max(2, int(window))
    return s.rolling(window=w, min_periods=max(2, w // 2)).mean()


def _pos_from_band(
    price: pd.Series,
    center: pd.Series,
    *,
    band: pd.Series,
) -> pd.Series:
    """
    Hysteresis long/cash rule with upper/lower bands around center:
    - enter when close > center + band
    - exit when close < center - band
    - otherwise keep previous state
    """
    px = pd.to_numeric(price, errors="coerce").astype(float)
    cc = pd.to_numeric(center, errors="coerce").astype(float).reindex(px.index)
    bb = (
        pd.to_numeric(band, errors="coerce").astype(float).reindex(px.index).fillna(0.0)
    )
    out = np.zeros(len(px), dtype=float)
    in_pos = False
    for i in range(len(px)):
        p = float(px.iloc[i]) if np.isfinite(float(px.iloc[i])) else float("nan")
        c = float(cc.iloc[i]) if np.isfinite(float(cc.iloc[i])) else float("nan")
        b = max(0.0, float(bb.iloc[i])) if np.isfinite(float(bb.iloc[i])) else 0.0
        if (not np.isfinite(p)) or (not np.isfinite(c)):
            out[i] = 1.0 if in_pos else 0.0
            continue
        upper = c + b
        lower = c - b
        if (not in_pos) and p > upper:
            in_pos = True
        elif in_pos and p < lower:
            in_pos = False
        out[i] = 1.0 if in_pos else 0.0
    return pd.Series(out, index=px.index, dtype=float)


def _stable_code_seed(code: str) -> int:
    v = 0
    for i, ch in enumerate(str(code)):
        v = (v + (i + 1) * ord(ch)) % 2_147_483_647
    return int(v)


def _pos_from_random_entry_hold(
    index: pd.Index, *, hold_days: int, seed: int | None
) -> pd.Series:
    hold_n = max(1, int(hold_days))
    rng = np.random.default_rng() if seed is None else np.random.default_rng(int(seed))
    out = np.zeros(len(index), dtype=float)
    in_pos = False
    days_left = 0
    for i in range(len(index)):
        if in_pos:
            out[i] = 1.0
            days_left -= 1
            if days_left <= 0:
                in_pos = False
            continue
        toss = int(rng.integers(0, 2))
        if toss == 1:
            in_pos = True
            days_left = hold_n
            out[i] = 1.0
            days_left -= 1
            if days_left <= 0:
                in_pos = False
    return pd.Series(out, index=index, dtype=float)


def _pos_from_tsmom(
    mom: pd.Series,
    *,
    entry_threshold: float,
    exit_threshold: float,
) -> pd.Series:
    """
    Long/cash TSMOM with configurable hysteresis thresholds:
    - enter long when momentum > entry_threshold
    - exit to cash when momentum <= exit_threshold
    """
    ent = float(entry_threshold)
    ex = float(exit_threshold)
    pos = np.zeros(len(mom), dtype=float)
    in_pos = False
    for i, v in enumerate(mom.to_numpy(dtype=float)):
        if not np.isfinite(v):
            in_pos = False
            pos[i] = 0.0
            continue
        if not in_pos:
            if v > ent:
                in_pos = True
        elif v <= ex:
            in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=mom.index, dtype=float)


def _apply_atr_stop(
    base_pos: pd.Series,
    *,
    open_: pd.Series | None = None,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    mode: str,
    atr_basis: str,
    reentry_mode: str,
    atr_window: int,
    n_mult: float,
    m_step: float,
    same_day_stop: bool = False,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Apply universal ATR stop-loss overlay on top of base strategy position.

    Long (base_pos > 0): stop below entry; intraday trigger when low <= stop.
    Short (base_pos < 0): stop above entry; intraday trigger when high >= stop.

    Modes:
    - static: stop distance fixed from entry (long: entry - n*ATR(entry); short: entry + n*ATR(entry))
    - trailing: long uses stop_candidate = close - n*ATR(ref), ratcheting stop up only;
      short uses stop_candidate = close + n*ATR(ref), ratcheting stop down only
    - tightening: trailing plus tighten distance using favorable move in ATR units from entry
      (long: rise from entry; short: fall from entry)

    ATR(ref) by atr_basis: entry uses ATR at entry; latest uses current ATR.

    Same-day evaluation: if same_day_stop=False (typical ETF), the entry session does not
    evaluate the stop; if True (e.g. futures T+0), the entry session does.

    Fill: long — touch at stop; gap-down open at or below stop fills at open.
    Short — touch at stop; gap-up open at or above stop fills at open.
    """
    mode_v = str(mode or "none").strip().lower()
    atr_basis_v = str(atr_basis or "latest").strip().lower()
    reentry_v = str(reentry_mode or "reenter").strip().lower()

    def _event_row(
        *,
        idx: Any,
        b: float,
        c: float | None,
        o: float | None,
        l: float | None,  # noqa: E741
        a: float | None,
        stop_before: float | None,
        stop_candidate: float | None,
        stop_after: float | None,
        stop_fill_price: float | None,
        stop_trigger_source: str | None,
        gap_open_triggered: bool | None,
        decision_pos: float,
        in_pos_after: bool,
        wait_lock: bool,
        event_type: str,
        event_reason: str,
        base_entry_event: bool,
        stop_triggered: bool,
    ) -> dict[str, Any]:
        return {
            "date": (idx.date().isoformat() if hasattr(idx, "date") else str(idx)),
            "event_type": str(event_type),  # entry | exit | hold
            "event_reason": str(
                event_reason
            ),  # base_entry_signal | stop_reentry | base_exit_signal | atr_stop
            "base_pos": float(b),
            "base_entry_event": bool(base_entry_event),
            "close": (
                None if c is None else (float(c) if np.isfinite(float(c)) else None)
            ),
            "open": (
                None if o is None else (float(o) if np.isfinite(float(o)) else None)
            ),
            "low": (
                None if l is None else (float(l) if np.isfinite(float(l)) else None)
            ),
            "atr": (
                None if a is None else (float(a) if np.isfinite(float(a)) else None)
            ),
            "stop_before": stop_before,
            "stop_candidate": stop_candidate,
            "stop_after": stop_after,
            "stop_triggered": bool(stop_triggered),
            "stop_fill_price": stop_fill_price,
            "stop_trigger_source": stop_trigger_source,
            "gap_open_triggered": (
                None if gap_open_triggered is None else bool(gap_open_triggered)
            ),
            "decision_pos": float(decision_pos),
            "in_pos_after": bool(in_pos_after),
            "wait_next_entry_lock": bool(wait_lock),
        }

    if mode_v == "none":
        out_none = base_pos.fillna(0.0).astype(float)
        bp = out_none.astype(float)
        trace_rows: list[dict[str, Any]] = []
        prev_b = 0.0
        for i in range(len(bp)):
            b = float(bp.iloc[i]) if np.isfinite(float(bp.iloc[i])) else 0.0
            d = bp.index[i]
            base_entry_long = bool((b > 0.0) and (prev_b <= 0.0))
            base_entry_short = bool((b < 0.0) and (prev_b >= 0.0))
            base_exit_long = bool((b <= 0.0) and (prev_b > 0.0))
            base_exit_short = bool((b >= 0.0) and (prev_b < 0.0))
            base_entry_event = bool(base_entry_long or base_entry_short)
            if base_entry_long or base_entry_short:
                trace_rows.append(
                    _event_row(
                        idx=d,
                        b=b,
                        c=None,
                        o=None,
                        l=None,
                        a=None,
                        stop_before=None,
                        stop_candidate=None,
                        stop_after=None,
                        stop_fill_price=None,
                        stop_trigger_source=None,
                        gap_open_triggered=None,
                        decision_pos=b,
                        in_pos_after=bool(b != 0.0),
                        wait_lock=False,
                        event_type="entry",
                        event_reason="base_entry_signal",
                        base_entry_event=base_entry_event,
                        stop_triggered=False,
                    )
                )
            elif base_exit_long or base_exit_short:
                trace_rows.append(
                    _event_row(
                        idx=d,
                        b=b,
                        c=None,
                        o=None,
                        l=None,
                        a=None,
                        stop_before=None,
                        stop_candidate=None,
                        stop_after=None,
                        stop_fill_price=None,
                        stop_trigger_source=None,
                        gap_open_triggered=None,
                        decision_pos=b,
                        in_pos_after=bool(b != 0.0),
                        wait_lock=False,
                        event_type="exit",
                        event_reason="base_exit_signal",
                        base_entry_event=base_entry_event,
                        stop_triggered=False,
                    )
                )
            prev_b = b
        flat_prev = out_none.shift(1).fillna(0.0)
        entries = int(((out_none != 0.0) & (flat_prev == 0.0)).sum())
        exits = int(((out_none == 0.0) & (flat_prev != 0.0)).sum())
        return out_none, {
            "enabled": False,
            "mode": "none",
            "atr_basis": atr_basis_v,
            "reentry_mode": reentry_v,
            "trigger_count": 0,
            "trigger_dates": [],
            "trigger_events": [],
            "first_trigger_date": None,
            "last_trigger_date": None,
            "entries": entries,
            "exits": exits,
            "trigger_exit_share": 0.0,
            "latest_stop_price": None,
            "latest_stop_date": None,
            "wait_next_entry_lock_active": False,
            "trigger_rule": "long_low_le_stop_short_high_ge_stop",
            "fill_rule": "long_gap_down_open_short_gap_up_open",
            "same_day_stop": bool(same_day_stop),
            "trace_last_rows": trace_rows[-80:],
            "trade_records": [],
        }

    atr = _atr_from_hlc(
        high.astype(float),
        low.astype(float),
        close.astype(float),
        window=int(atr_window),
    ).astype(float)
    bp = base_pos.astype(float).fillna(0.0)
    op = (
        (open_.astype(float) if isinstance(open_, pd.Series) else close.astype(float))
        .reindex(bp.index)
        .astype(float)
    )
    cl = close.astype(float)
    lo = low.astype(float)
    hi = high.astype(float)

    out = np.zeros(len(bp), dtype=float)
    pos_side = 0
    stop_px = float("nan")
    entry_px = float("nan")
    entry_atr = float("nan")
    prev_base = 0.0
    wait_lock_long = False
    wait_lock_short = False
    trigger_dates: list[str] = []
    trigger_events: list[dict[str, Any]] = []
    trade_records: list[dict[str, Any]] = []
    latest_stop_date: str | None = None
    trace_last_rows: list[dict[str, Any]] = []
    entry_decision_date: str | None = None
    initial_stop_price: float = float("nan")

    for i in range(len(bp)):
        b = float(bp.iloc[i]) if np.isfinite(float(bp.iloc[i])) else 0.0
        o = float(op.iloc[i]) if np.isfinite(float(op.iloc[i])) else float("nan")
        c = float(cl.iloc[i]) if np.isfinite(float(cl.iloc[i])) else float("nan")
        l = float(lo.iloc[i]) if np.isfinite(float(lo.iloc[i])) else float("nan")  # noqa: E741
        h = float(hi.iloc[i]) if np.isfinite(float(hi.iloc[i])) else float("nan")
        a = float(atr.iloc[i]) if np.isfinite(float(atr.iloc[i])) else float("nan")
        base_entry_long = bool((b > 0.0) and (prev_base <= 0.0))
        base_entry_short = bool((b < 0.0) and (prev_base >= 0.0))
        base_entry_event = bool(base_entry_long or base_entry_short)

        if pos_side == 0:
            if b > 0.0:
                if (not np.isfinite(c)) or (not np.isfinite(a)) or a <= 0.0:
                    out[i] = 0.0
                    prev_base = b
                    continue
                if (
                    reentry_v == "wait_next_entry"
                    and wait_lock_long
                    and (not base_entry_long)
                ):
                    out[i] = 0.0
                    prev_base = b
                    continue
                pos_side = 1
                wait_lock_long = False
                entry_px = c
                entry_atr = a
                stop_px = entry_px - float(n_mult) * a
                entry_decision_date = (
                    bp.index[i].date().isoformat()
                    if hasattr(bp.index[i], "date")
                    else str(bp.index[i])
                )
                initial_stop_price = (
                    float(stop_px) if np.isfinite(stop_px) else float("nan")
                )
                out[i] = b
                if np.isfinite(stop_px):
                    d_ent = bp.index[i]
                    latest_stop_date = (
                        d_ent.date().isoformat()
                        if hasattr(d_ent, "date")
                        else str(d_ent)
                    )
                trace_last_rows.append(
                    _event_row(
                        idx=bp.index[i],
                        b=b,
                        c=c,
                        o=o,
                        l=l,
                        a=a,
                        stop_before=None,
                        stop_candidate=(
                            float(stop_px) if np.isfinite(stop_px) else None
                        ),
                        stop_after=(float(stop_px) if np.isfinite(stop_px) else None),
                        stop_fill_price=None,
                        stop_trigger_source=None,
                        gap_open_triggered=None,
                        decision_pos=float(out[i]),
                        in_pos_after=True,
                        wait_lock=bool(wait_lock_long or wait_lock_short),
                        event_type="entry",
                        event_reason=(
                            "base_entry_signal" if base_entry_long else "stop_reentry"
                        ),
                        base_entry_event=bool(base_entry_event),
                        stop_triggered=False,
                    )
                )
                if len(trace_last_rows) > 120:
                    trace_last_rows = trace_last_rows[-120:]
                prev_base = b
                if not same_day_stop:
                    continue
            elif b < 0.0:
                if (not np.isfinite(c)) or (not np.isfinite(a)) or a <= 0.0:
                    out[i] = 0.0
                    prev_base = b
                    continue
                if (
                    reentry_v == "wait_next_entry"
                    and wait_lock_short
                    and (not base_entry_short)
                ):
                    out[i] = 0.0
                    prev_base = b
                    continue
                pos_side = -1
                wait_lock_short = False
                entry_px = c
                entry_atr = a
                stop_px = entry_px + float(n_mult) * a
                entry_decision_date = (
                    bp.index[i].date().isoformat()
                    if hasattr(bp.index[i], "date")
                    else str(bp.index[i])
                )
                initial_stop_price = (
                    float(stop_px) if np.isfinite(stop_px) else float("nan")
                )
                out[i] = b
                if np.isfinite(stop_px):
                    d_ent = bp.index[i]
                    latest_stop_date = (
                        d_ent.date().isoformat()
                        if hasattr(d_ent, "date")
                        else str(d_ent)
                    )
                trace_last_rows.append(
                    _event_row(
                        idx=bp.index[i],
                        b=b,
                        c=c,
                        o=o,
                        l=l,
                        a=a,
                        stop_before=None,
                        stop_candidate=(
                            float(stop_px) if np.isfinite(stop_px) else None
                        ),
                        stop_after=(float(stop_px) if np.isfinite(stop_px) else None),
                        stop_fill_price=None,
                        stop_trigger_source=None,
                        gap_open_triggered=None,
                        decision_pos=float(out[i]),
                        in_pos_after=True,
                        wait_lock=bool(wait_lock_long or wait_lock_short),
                        event_type="entry",
                        event_reason=(
                            "base_entry_signal" if base_entry_short else "stop_reentry"
                        ),
                        base_entry_event=bool(base_entry_event),
                        stop_triggered=False,
                    )
                )
                if len(trace_last_rows) > 120:
                    trace_last_rows = trace_last_rows[-120:]
                prev_base = b
                if not same_day_stop:
                    continue
            else:
                out[i] = 0.0
                prev_base = b
                continue

        if pos_side == 1:
            if b <= 0.0:
                stop_before = float(stop_px) if np.isfinite(stop_px) else None
                pos_side = 0
                out[i] = 0.0
                stop_px = float("nan")
                entry_px = float("nan")
                entry_atr = float("nan")
                entry_decision_date = None
                initial_stop_price = float("nan")
                trace_last_rows.append(
                    _event_row(
                        idx=bp.index[i],
                        b=b,
                        c=c,
                        o=o,
                        l=l,
                        a=a,
                        stop_before=stop_before,
                        stop_candidate=None,
                        stop_after=None,
                        stop_fill_price=None,
                        stop_trigger_source=None,
                        gap_open_triggered=None,
                        decision_pos=float(out[i]),
                        in_pos_after=False,
                        wait_lock=bool(wait_lock_long or wait_lock_short),
                        event_type="exit",
                        event_reason="base_exit_signal",
                        base_entry_event=bool(base_entry_event),
                        stop_triggered=False,
                    )
                )
                if len(trace_last_rows) > 120:
                    trace_last_rows = trace_last_rows[-120:]
                prev_base = b
                continue

            stop_before = float(stop_px) if np.isfinite(stop_px) else None
            if np.isfinite(l) and np.isfinite(stop_px) and (l <= stop_px):
                pos_side = 0
                out[i] = 0.0
                d = bp.index[i]
                ds = d.date().isoformat() if hasattr(d, "date") else str(d)
                trigger_dates.append(ds)
                gap_open_triggered = bool(np.isfinite(o) and (o <= stop_px))
                fill_price = (
                    float(o)
                    if gap_open_triggered and np.isfinite(o)
                    else float(stop_px)
                )
                trigger_source = (
                    "gap_open_below_stop" if gap_open_triggered else "low_touch_stop"
                )
                trigger_events.append(
                    {
                        "date": ds,
                        "position_side": "long",
                        "stop_price": (
                            float(stop_px) if np.isfinite(stop_px) else None
                        ),
                        "close_price": (float(c) if np.isfinite(c) else None),
                        "atr_value": (float(a) if np.isfinite(a) else None),
                        "open_price": (float(o) if np.isfinite(o) else None),
                        "low_price": (float(l) if np.isfinite(l) else None),
                        "high_price": None,
                        "fill_price": (
                            float(fill_price) if np.isfinite(fill_price) else None
                        ),
                        "stop_distance_atr": (
                            float((c - stop_px) / a)
                            if np.isfinite(c)
                            and np.isfinite(stop_px)
                            and np.isfinite(a)
                            and a > 0
                            else None
                        ),
                        "trigger_source": trigger_source,
                        "gap_open_triggered": bool(gap_open_triggered),
                    }
                )
                trade_records.append(
                    {
                        "entry_decision_date": entry_decision_date,
                        "entry_execution_date": None,
                        "entry_execution_price": None,
                        "initial_stop_price": (
                            float(initial_stop_price)
                            if np.isfinite(initial_stop_price)
                            else None
                        ),
                        "trigger_stop_price": (
                            float(stop_px) if np.isfinite(stop_px) else None
                        ),
                        "execution_stop_price": (
                            float(fill_price) if np.isfinite(fill_price) else None
                        ),
                        "trigger_date": ds,
                    }
                )
                if reentry_v == "wait_next_entry":
                    wait_lock_long = True
                stop_px = float("nan")
                entry_px = float("nan")
                entry_atr = float("nan")
                entry_decision_date = None
                initial_stop_price = float("nan")
                trace_last_rows.append(
                    _event_row(
                        idx=bp.index[i],
                        b=b,
                        c=c,
                        o=o,
                        l=l,
                        a=a,
                        stop_before=stop_before,
                        stop_candidate=None,
                        stop_after=None,
                        stop_fill_price=(
                            float(fill_price) if np.isfinite(fill_price) else None
                        ),
                        stop_trigger_source=trigger_source,
                        gap_open_triggered=bool(gap_open_triggered),
                        decision_pos=float(out[i]),
                        in_pos_after=False,
                        wait_lock=bool(wait_lock_long or wait_lock_short),
                        event_type="exit",
                        event_reason="atr_stop",
                        base_entry_event=bool(base_entry_event),
                        stop_triggered=True,
                    )
                )
                if len(trace_last_rows) > 120:
                    trace_last_rows = trace_last_rows[-120:]
                prev_base = b
                continue

            stop_candidate: float | None = None
            if (
                mode_v in {"trailing", "tightening"}
                and np.isfinite(c)
                and np.isfinite(a)
                and a > 0.0
            ):
                atr_ref = (
                    float(entry_atr)
                    if (
                        atr_basis_v == "entry"
                        and np.isfinite(entry_atr)
                        and float(entry_atr) > 0.0
                    )
                    else float(a)
                )
                dist_mult = float(n_mult)
                if mode_v == "tightening":
                    rise_in_atr = max(0.0, (c - entry_px) / max(atr_ref, 1e-12))
                    steps = (
                        int(np.floor(rise_in_atr / float(m_step)))
                        if float(m_step) > 0
                        else 0
                    )
                    dist_mult = max(
                        float(m_step), float(n_mult) - steps * float(m_step)
                    )
                stop_candidate_v = c - dist_mult * atr_ref
                stop_candidate = (
                    float(stop_candidate_v) if np.isfinite(stop_candidate_v) else None
                )
                if np.isfinite(stop_candidate_v) and (
                    (not np.isfinite(stop_px)) or (stop_candidate_v > stop_px)
                ):
                    stop_px = stop_candidate_v

            trace_last_rows.append(
                _event_row(
                    idx=bp.index[i],
                    b=b,
                    c=c,
                    o=o,
                    l=l,
                    a=a,
                    stop_before=stop_before,
                    stop_candidate=stop_candidate,
                    stop_after=(float(stop_px) if np.isfinite(stop_px) else None),
                    stop_fill_price=None,
                    stop_trigger_source=None,
                    gap_open_triggered=None,
                    decision_pos=float(b),
                    in_pos_after=True,
                    wait_lock=bool(wait_lock_long or wait_lock_short),
                    event_type="hold",
                    event_reason="atr_update",
                    base_entry_event=bool(base_entry_event),
                    stop_triggered=False,
                )
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]

            out[i] = b
            if np.isfinite(stop_px):
                d = bp.index[i]
                latest_stop_date = (
                    d.date().isoformat() if hasattr(d, "date") else str(d)
                )
            prev_base = b
            continue

        # pos_side == -1 (short)
        if b >= 0.0:
            stop_before = float(stop_px) if np.isfinite(stop_px) else None
            pos_side = 0
            out[i] = 0.0
            stop_px = float("nan")
            entry_px = float("nan")
            entry_atr = float("nan")
            entry_decision_date = None
            initial_stop_price = float("nan")
            trace_last_rows.append(
                _event_row(
                    idx=bp.index[i],
                    b=b,
                    c=c,
                    o=o,
                    l=l,
                    a=a,
                    stop_before=stop_before,
                    stop_candidate=None,
                    stop_after=None,
                    stop_fill_price=None,
                    stop_trigger_source=None,
                    gap_open_triggered=None,
                    decision_pos=float(out[i]),
                    in_pos_after=False,
                    wait_lock=bool(wait_lock_long or wait_lock_short),
                    event_type="exit",
                    event_reason="base_exit_signal",
                    base_entry_event=bool(base_entry_event),
                    stop_triggered=False,
                )
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]
            prev_base = b
            continue

        stop_before = float(stop_px) if np.isfinite(stop_px) else None
        if np.isfinite(h) and np.isfinite(stop_px) and (h >= stop_px):
            pos_side = 0
            out[i] = 0.0
            d = bp.index[i]
            ds = d.date().isoformat() if hasattr(d, "date") else str(d)
            trigger_dates.append(ds)
            gap_open_triggered = bool(np.isfinite(o) and (o >= stop_px))
            fill_price = (
                float(o) if gap_open_triggered and np.isfinite(o) else float(stop_px)
            )
            trigger_source = (
                "gap_open_above_stop" if gap_open_triggered else "high_touch_stop"
            )
            trigger_events.append(
                {
                    "date": ds,
                    "position_side": "short",
                    "stop_price": (float(stop_px) if np.isfinite(stop_px) else None),
                    "close_price": (float(c) if np.isfinite(c) else None),
                    "atr_value": (float(a) if np.isfinite(a) else None),
                    "open_price": (float(o) if np.isfinite(o) else None),
                    "low_price": None,
                    "high_price": (float(h) if np.isfinite(h) else None),
                    "fill_price": (
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    "stop_distance_atr": (
                        float((stop_px - c) / a)
                        if np.isfinite(c)
                        and np.isfinite(stop_px)
                        and np.isfinite(a)
                        and a > 0
                        else None
                    ),
                    "trigger_source": trigger_source,
                    "gap_open_triggered": bool(gap_open_triggered),
                }
            )
            trade_records.append(
                {
                    "entry_decision_date": entry_decision_date,
                    "entry_execution_date": None,
                    "entry_execution_price": None,
                    "initial_stop_price": (
                        float(initial_stop_price)
                        if np.isfinite(initial_stop_price)
                        else None
                    ),
                    "trigger_stop_price": (
                        float(stop_px) if np.isfinite(stop_px) else None
                    ),
                    "execution_stop_price": (
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    "trigger_date": ds,
                }
            )
            if reentry_v == "wait_next_entry":
                wait_lock_short = True
            stop_px = float("nan")
            entry_px = float("nan")
            entry_atr = float("nan")
            entry_decision_date = None
            initial_stop_price = float("nan")
            trace_last_rows.append(
                _event_row(
                    idx=bp.index[i],
                    b=b,
                    c=c,
                    o=o,
                    l=l,
                    a=a,
                    stop_before=stop_before,
                    stop_candidate=None,
                    stop_after=None,
                    stop_fill_price=(
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    stop_trigger_source=trigger_source,
                    gap_open_triggered=bool(gap_open_triggered),
                    decision_pos=float(out[i]),
                    in_pos_after=False,
                    wait_lock=bool(wait_lock_long or wait_lock_short),
                    event_type="exit",
                    event_reason="atr_stop",
                    base_entry_event=bool(base_entry_event),
                    stop_triggered=True,
                )
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]
            prev_base = b
            continue

        stop_candidate = None
        if (
            mode_v in {"trailing", "tightening"}
            and np.isfinite(c)
            and np.isfinite(a)
            and a > 0.0
        ):
            atr_ref = (
                float(entry_atr)
                if (
                    atr_basis_v == "entry"
                    and np.isfinite(entry_atr)
                    and float(entry_atr) > 0.0
                )
                else float(a)
            )
            dist_mult = float(n_mult)
            if mode_v == "tightening":
                fall_in_atr = max(0.0, (entry_px - c) / max(atr_ref, 1e-12))
                steps = (
                    int(np.floor(fall_in_atr / float(m_step)))
                    if float(m_step) > 0
                    else 0
                )
                dist_mult = max(float(m_step), float(n_mult) - steps * float(m_step))
            stop_candidate_v = c + dist_mult * atr_ref
            stop_candidate = (
                float(stop_candidate_v) if np.isfinite(stop_candidate_v) else None
            )
            if np.isfinite(stop_candidate_v) and (
                (not np.isfinite(stop_px)) or (stop_candidate_v < stop_px)
            ):
                stop_px = stop_candidate_v

        trace_last_rows.append(
            _event_row(
                idx=bp.index[i],
                b=b,
                c=c,
                o=o,
                l=l,
                a=a,
                stop_before=stop_before,
                stop_candidate=stop_candidate,
                stop_after=(float(stop_px) if np.isfinite(stop_px) else None),
                stop_fill_price=None,
                stop_trigger_source=None,
                gap_open_triggered=None,
                decision_pos=float(b),
                in_pos_after=True,
                wait_lock=bool(wait_lock_long or wait_lock_short),
                event_type="hold",
                event_reason="atr_update",
                base_entry_event=bool(base_entry_event),
                stop_triggered=False,
            )
        )
        if len(trace_last_rows) > 120:
            trace_last_rows = trace_last_rows[-120:]

        out[i] = b
        if np.isfinite(stop_px):
            d = bp.index[i]
            latest_stop_date = d.date().isoformat() if hasattr(d, "date") else str(d)
        prev_base = b

    out_s = pd.Series(out, index=base_pos.index, dtype=float)
    flat_prev = out_s.shift(1).fillna(0.0)
    entries = int(((out_s != 0.0) & (flat_prev == 0.0)).sum())
    exits = int(((out_s == 0.0) & (flat_prev != 0.0)).sum())
    trigger_count = int(len(trigger_dates))
    stats = {
        "enabled": True,
        "mode": mode_v,
        "atr_basis": atr_basis_v,
        "reentry_mode": reentry_v,
        "trigger_count": trigger_count,
        "trigger_dates": trigger_dates[:200],
        "trigger_events": trigger_events[:200],
        "first_trigger_date": (trigger_dates[0] if trigger_dates else None),
        "last_trigger_date": (trigger_dates[-1] if trigger_dates else None),
        "entries": entries,
        "exits": exits,
        "trigger_exit_share": (
            float(trigger_count) / float(exits) if exits > 0 else 0.0
        ),
        "latest_stop_price": (
            float(stop_px) if (pos_side != 0 and np.isfinite(stop_px)) else None
        ),
        "latest_stop_date": (
            latest_stop_date if (pos_side != 0 and np.isfinite(stop_px)) else None
        ),
        "wait_next_entry_lock_active": bool(wait_lock_long or wait_lock_short),
        "trigger_rule": "long_low_le_stop_short_high_ge_stop",
        "fill_rule": "long_gap_down_open_short_gap_up_open",
        "same_day_stop": bool(same_day_stop),
        "trace_last_rows": trace_last_rows[-80:],
        "trade_records": trade_records,
    }
    return out_s, stats


def _stop_fill_return(
    *,
    exec_price: str,
    open_px: float,
    prev_close_px: float,
    fill_px: float,
) -> float:
    ep = str(exec_price or "close").strip().lower()
    if (not np.isfinite(fill_px)) or fill_px <= 0.0:
        return 0.0
    if ep == "open":
        base = open_px
        if (not np.isfinite(base)) or base <= 0.0:
            return 0.0
        return float(fill_px / base - 1.0)
    if ep == "oc2":
        r1 = 0.0
        r2 = 0.0
        if np.isfinite(open_px) and open_px > 0.0:
            r1 = float(fill_px / open_px - 1.0)
        if np.isfinite(prev_close_px) and prev_close_px > 0.0:
            r2 = float(fill_px / prev_close_px - 1.0)
        return float(0.5 * (r1 + r2))
    base = prev_close_px
    if (not np.isfinite(base)) or base <= 0.0:
        base = open_px
    if (not np.isfinite(base)) or base <= 0.0:
        return 0.0
    return float(fill_px / base - 1.0)


def _apply_intraday_stop_execution_single(
    *,
    weights: pd.Series,
    atr_stop_stats: dict[str, Any],
    exec_price: str,
    open_sig: pd.Series,
    close_sig: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    w0 = weights.astype(float).copy()
    w_adj = w0.copy()
    override = pd.Series(0.0, index=w0.index, dtype=float)
    events = list((atr_stop_stats or {}).get("trigger_events") or [])
    if not events:
        return w_adj, override
    prev_close = close_sig.astype(float).shift(1)
    for e in events:
        d_raw = e.get("date")
        try:
            d = pd.Timestamp(d_raw)
        except Exception:
            continue
        if d not in w_adj.index:
            continue
        wv = float(w_adj.loc[d]) if np.isfinite(float(w_adj.loc[d])) else 0.0
        if abs(wv) <= 1e-12:
            continue
        fill_raw = e.get("fill_price")
        try:
            fill_px = float(fill_raw)
        except (TypeError, ValueError):
            fill_px = float("nan")
        if not np.isfinite(fill_px):
            fill_px = float("nan")
        o = (
            float(open_sig.loc[d])
            if (d in open_sig.index and np.isfinite(float(open_sig.loc[d])))
            else float("nan")
        )
        pc = (
            float(prev_close.loc[d])
            if (d in prev_close.index and np.isfinite(float(prev_close.loc[d])))
            else float("nan")
        )
        day_ret = _stop_fill_return(
            exec_price=exec_price, open_px=o, prev_close_px=pc, fill_px=fill_px
        )
        override.loc[d] = float(override.loc[d] + wv * day_ret)
        w_adj.loc[d] = 0.0
    return w_adj, override


def _apply_intraday_stop_execution_portfolio(
    *,
    weights: pd.DataFrame,
    atr_stop_by_asset: dict[str, dict[str, Any]],
    exec_price: str,
    open_sig_df: pd.DataFrame,
    close_sig_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    w0 = weights.astype(float).copy()
    w_adj = w0.copy()
    override = pd.Series(0.0, index=w0.index, dtype=float)
    prev_close_df = close_sig_df.astype(float).shift(1)
    for c in w_adj.columns:
        stats = atr_stop_by_asset.get(str(c)) or {}
        events = list(stats.get("trigger_events") or [])
        if not events:
            continue
        for e in events:
            d_raw = e.get("date")
            try:
                d = pd.Timestamp(d_raw)
            except Exception:
                continue
            if d not in w_adj.index:
                continue
            wv = float(w_adj.loc[d, c]) if np.isfinite(float(w_adj.loc[d, c])) else 0.0
            if abs(wv) <= 1e-12:
                continue
            fill_raw = e.get("fill_price")
            try:
                fill_px = float(fill_raw)
            except (TypeError, ValueError):
                fill_px = float("nan")
            if not np.isfinite(fill_px):
                fill_px = float("nan")
            o = (
                float(open_sig_df.loc[d, c])
                if (
                    d in open_sig_df.index
                    and c in open_sig_df.columns
                    and np.isfinite(float(open_sig_df.loc[d, c]))
                )
                else float("nan")
            )
            pc = (
                float(prev_close_df.loc[d, c])
                if (
                    d in prev_close_df.index
                    and c in prev_close_df.columns
                    and np.isfinite(float(prev_close_df.loc[d, c]))
                )
                else float("nan")
            )
            day_ret = _stop_fill_return(
                exec_price=exec_price, open_px=o, prev_close_px=pc, fill_px=fill_px
            )
            override.loc[d] = float(override.loc[d] + wv * day_ret)
            w_adj.loc[d, c] = 0.0
    return w_adj, override


def _macd_core(
    close: pd.Series, *, fast: int, slow: int, signal: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, int(fast))
    ema_slow = _ema(close, int(slow))
    macd = (ema_fast - ema_slow).astype(float)
    sig = _ema(macd, int(signal)).astype(float)
    hist = (macd - sig).astype(float)
    return macd, sig, hist


def _atr_from_hlc(
    high: pd.Series, low: pd.Series, close: pd.Series, *, window: int
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    w = max(2, int(window))
    # Wilder-style ATR smoothing (RMA via EWM alpha=1/window).
    return (
        tr.ewm(alpha=1.0 / float(w), adjust=False, min_periods=w).mean().astype(float)
    )


def _normalize_r_take_profit_tiers(
    tiers: list[dict[str, float]] | None,
) -> list[dict[str, float]]:
    raw = tiers if tiers is not None else DEFAULT_R_TAKE_PROFIT_TIERS
    out: list[dict[str, float]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        r_mult = float(item.get("r_multiple"))
        retrace = float(item.get("retrace_ratio"))
        if (not np.isfinite(r_mult)) or r_mult <= 0.0:
            continue
        if (not np.isfinite(retrace)) or retrace <= 0.0 or retrace >= 1.0:
            continue
        out.append({"r_multiple": float(r_mult), "retrace_ratio": float(retrace)})
    if not out:
        out = [dict(x) for x in DEFAULT_R_TAKE_PROFIT_TIERS]
    uniq: dict[float, float] = {}
    for row in out:
        uniq[float(row["r_multiple"])] = float(row["retrace_ratio"])
    rows = [
        {"r_multiple": float(k), "retrace_ratio": float(v)} for k, v in uniq.items()
    ]
    rows.sort(key=lambda x: float(x["r_multiple"]))
    return rows


def _apply_r_multiple_take_profit(
    base_pos: pd.Series,
    *,
    open_: pd.Series | None = None,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    enabled: bool,
    reentry_mode: str,
    atr_window: int,
    atr_n: float,
    tiers: list[dict[str, float]] | None,
    atr_stop_enabled: bool,
) -> tuple[pd.Series, dict[str, Any]]:
    tiers_v = _normalize_r_take_profit_tiers(tiers)
    reentry_v = str(reentry_mode or "reenter").strip().lower()
    if reentry_v not in {"reenter", "wait_next_entry"}:
        reentry_v = "reenter"
    if not bool(enabled):
        out_none = base_pos.fillna(0.0).astype(float)
        return out_none, {
            "enabled": False,
            "reentry_mode": reentry_v,
            "atr_window": int(atr_window),
            "atr_n": float(atr_n),
            "fallback_mode_used": False,
            "initial_r_mode": "disabled",
            "trigger_count": 0,
            "trigger_dates": [],
            "trigger_events": [],
            "first_trigger_date": None,
            "last_trigger_date": None,
            "entries": int(
                ((out_none > 0.0) & (out_none.shift(1).fillna(0.0) <= 0.0)).sum()
            ),
            "exits": int(
                ((out_none <= 0.0) & (out_none.shift(1).fillna(0.0) > 0.0)).sum()
            ),
            "trigger_exit_share": 0.0,
            "wait_next_entry_lock_active": False,
            "tiers": tiers_v,
            "tier_trigger_counts": {},
            "trigger_rule": "low_vs_peak_retrace_same_day_exit",
            "fill_rule": "fill=min(trigger_price,open_price) for long",
            "trace_last_rows": [],
        }
    bp = base_pos.fillna(0.0).astype(float)
    cl = close.astype(float)
    op = (
        (open_.astype(float) if isinstance(open_, pd.Series) else cl.astype(float))
        .reindex(bp.index)
        .astype(float)
    )
    hi = high.astype(float).fillna(cl)
    lo = low.astype(float).fillna(cl)
    atr = _atr_from_hlc(hi, lo, cl, window=int(atr_window)).astype(float)
    out = np.zeros(len(bp), dtype=float)
    trigger_dates: list[str] = []
    trigger_events: list[dict[str, Any]] = []
    tier_trigger_counts: dict[str, int] = {}
    trace_last_rows: list[dict[str, Any]] = []
    in_pos = False
    prev_base = 0.0
    wait_next_entry_lock = False
    entry_px = float("nan")
    initial_r_pct = float("nan")
    peak_profit_pct = float("nan")
    invalid_r_entries = 0
    eps = 1e-12

    for i in range(len(bp)):
        b = float(bp.iloc[i]) if np.isfinite(float(bp.iloc[i])) else 0.0
        o = float(op.iloc[i]) if np.isfinite(float(op.iloc[i])) else float("nan")
        c = float(cl.iloc[i]) if np.isfinite(float(cl.iloc[i])) else float("nan")
        h = float(hi.iloc[i]) if np.isfinite(float(hi.iloc[i])) else float("nan")
        l = float(lo.iloc[i]) if np.isfinite(float(lo.iloc[i])) else float("nan")  # noqa: E741
        a = float(atr.iloc[i]) if np.isfinite(float(atr.iloc[i])) else float("nan")
        d = bp.index[i]
        ds = d.date().isoformat() if hasattr(d, "date") else str(d)
        base_entry_event = bool((b > 0.0) and (prev_base <= 0.0))

        if not in_pos:
            if b <= 0.0 or (not np.isfinite(c)) or c <= 0.0:
                out[i] = 0.0
                prev_base = b
                continue
            if (
                reentry_v == "wait_next_entry"
                and wait_next_entry_lock
                and (not base_entry_event)
            ):
                out[i] = 0.0
                prev_base = b
                continue
            in_pos = True
            wait_next_entry_lock = False
            entry_px = c
            initial_r_pct = (
                float(atr_n) * a / c if (np.isfinite(a) and a > 0.0) else float("nan")
            )
            if (not np.isfinite(initial_r_pct)) or initial_r_pct <= eps:
                invalid_r_entries += 1
                initial_r_pct = float("nan")
            peak_profit_pct = 0.0
            out[i] = b
            trace_last_rows.append(
                {
                    "date": ds,
                    "event_type": "entry",
                    "event_reason": (
                        "base_entry_signal" if base_entry_event else "tp_reentry"
                    ),
                    "base_pos": float(b),
                    "entry_price": float(entry_px),
                    "atr_entry": (float(a) if np.isfinite(a) else None),
                    "initial_r_pct": (
                        float(initial_r_pct) if np.isfinite(initial_r_pct) else None
                    ),
                    "peak_profit_pct": 0.0,
                    "peak_r_multiple": 0.0,
                    "active_tier_r": None,
                    "active_tier_retrace": None,
                    "drawdown_from_peak": 0.0,
                    "tp_triggered": False,
                    "open": (float(o) if np.isfinite(o) else None),
                    "high": (float(h) if np.isfinite(h) else None),
                    "low": (float(l) if np.isfinite(l) else None),
                    "tp_fill_price": None,
                    "tp_trigger_source": None,
                    "gap_open_triggered": None,
                    "stop_fill_price": None,
                    "stop_trigger_source": None,
                    "decision_pos": float(out[i]),
                    "in_pos_after": bool(in_pos),
                    "wait_next_entry_lock": bool(wait_next_entry_lock),
                }
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]
            prev_base = b
            continue

        if b <= 0.0:
            in_pos = False
            out[i] = 0.0
            entry_px = float("nan")
            initial_r_pct = float("nan")
            peak_profit_pct = float("nan")
            trace_last_rows.append(
                {
                    "date": ds,
                    "event_type": "exit",
                    "event_reason": "base_exit_signal",
                    "base_pos": float(b),
                    "entry_price": None,
                    "atr_entry": None,
                    "initial_r_pct": None,
                    "peak_profit_pct": None,
                    "peak_r_multiple": None,
                    "active_tier_r": None,
                    "active_tier_retrace": None,
                    "drawdown_from_peak": None,
                    "tp_triggered": False,
                    "open": (float(o) if np.isfinite(o) else None),
                    "high": (float(h) if np.isfinite(h) else None),
                    "low": (float(l) if np.isfinite(l) else None),
                    "tp_fill_price": None,
                    "tp_trigger_source": None,
                    "gap_open_triggered": None,
                    "stop_fill_price": None,
                    "stop_trigger_source": None,
                    "decision_pos": float(out[i]),
                    "in_pos_after": bool(in_pos),
                    "wait_next_entry_lock": bool(wait_next_entry_lock),
                }
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]
            prev_base = b
            continue

        cur_peak_profit_pct = (
            (h / entry_px - 1.0)
            if (np.isfinite(h) and np.isfinite(entry_px) and entry_px > eps)
            else float("nan")
        )
        cur_low_profit_pct = (
            (l / entry_px - 1.0)
            if (np.isfinite(l) and np.isfinite(entry_px) and entry_px > eps)
            else float("nan")
        )
        if np.isfinite(cur_peak_profit_pct):
            peak_profit_pct = max(float(peak_profit_pct), float(cur_peak_profit_pct))
        peak_r_mult = (
            (float(peak_profit_pct) / float(initial_r_pct))
            if (
                np.isfinite(peak_profit_pct)
                and np.isfinite(initial_r_pct)
                and initial_r_pct > eps
            )
            else float("nan")
        )
        active_tier: dict[str, float] | None = None
        if np.isfinite(peak_r_mult):
            for row in tiers_v:
                if peak_r_mult >= float(row["r_multiple"]):
                    active_tier = row
        dd_from_peak = (
            (float(peak_profit_pct) - float(cur_low_profit_pct))
            / float(peak_profit_pct)
            if (
                np.isfinite(peak_profit_pct)
                and peak_profit_pct > eps
                and np.isfinite(cur_low_profit_pct)
            )
            else float("nan")
        )
        tp_triggered = bool(
            active_tier is not None
            and np.isfinite(dd_from_peak)
            and dd_from_peak >= float(active_tier["retrace_ratio"])
        )
        if tp_triggered:
            in_pos = False
            out[i] = 0.0
            trigger_profit_pct = float(peak_profit_pct) * (
                1.0 - float(active_tier["retrace_ratio"])
            )
            trigger_px = (
                float(entry_px) * (1.0 + float(trigger_profit_pct))
                if np.isfinite(entry_px) and np.isfinite(trigger_profit_pct)
                else float("nan")
            )
            gap_open_triggered = bool(
                np.isfinite(o) and np.isfinite(trigger_px) and (o <= trigger_px)
            )
            fill_price = (
                float(o) if gap_open_triggered and np.isfinite(o) else float(trigger_px)
            )
            trigger_source = (
                "gap_open_below_tp" if gap_open_triggered else "low_touch_tp_retrace"
            )
            if reentry_v == "wait_next_entry":
                wait_next_entry_lock = True
            trigger_dates.append(ds)
            trigger_events.append(
                {
                    "date": ds,
                    "trigger_price": (
                        float(trigger_px) if np.isfinite(trigger_px) else None
                    ),
                    "open_price": (float(o) if np.isfinite(o) else None),
                    "low_price": (float(l) if np.isfinite(l) else None),
                    "fill_price": (
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    "trigger_source": trigger_source,
                    "gap_open_triggered": bool(gap_open_triggered),
                    "peak_r_multiple": (
                        float(peak_r_mult) if np.isfinite(peak_r_mult) else None
                    ),
                    "active_tier_r": float(active_tier["r_multiple"])
                    if active_tier
                    else None,
                    "active_tier_retrace": float(active_tier["retrace_ratio"])
                    if active_tier
                    else None,
                }
            )
            tier_label = (
                f"{float(active_tier['r_multiple']):g}R" if active_tier else "unknown"
            )
            tier_trigger_counts[tier_label] = int(
                tier_trigger_counts.get(tier_label, 0) + 1
            )
            trace_last_rows.append(
                {
                    "date": ds,
                    "event_type": "exit",
                    "event_reason": "r_take_profit",
                    "base_pos": float(b),
                    "entry_price": (float(entry_px) if np.isfinite(entry_px) else None),
                    "atr_entry": None,
                    "initial_r_pct": (
                        float(initial_r_pct) if np.isfinite(initial_r_pct) else None
                    ),
                    "peak_profit_pct": (
                        float(peak_profit_pct) if np.isfinite(peak_profit_pct) else None
                    ),
                    "peak_r_multiple": (
                        float(peak_r_mult) if np.isfinite(peak_r_mult) else None
                    ),
                    "active_tier_r": float(active_tier["r_multiple"])
                    if active_tier
                    else None,
                    "active_tier_retrace": float(active_tier["retrace_ratio"])
                    if active_tier
                    else None,
                    "drawdown_from_peak": (
                        float(dd_from_peak) if np.isfinite(dd_from_peak) else None
                    ),
                    "tp_triggered": True,
                    "open": (float(o) if np.isfinite(o) else None),
                    "high": (float(h) if np.isfinite(h) else None),
                    "low": (float(l) if np.isfinite(l) else None),
                    "tp_fill_price": (
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    "tp_trigger_source": trigger_source,
                    "gap_open_triggered": bool(gap_open_triggered),
                    "stop_fill_price": (
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    "stop_trigger_source": trigger_source,
                    "decision_pos": float(out[i]),
                    "in_pos_after": bool(in_pos),
                    "wait_next_entry_lock": bool(wait_next_entry_lock),
                }
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]
            entry_px = float("nan")
            initial_r_pct = float("nan")
            peak_profit_pct = float("nan")
            prev_base = b
            continue

        out[i] = b
        trace_last_rows.append(
            {
                "date": ds,
                "event_type": "hold",
                "event_reason": "r_take_profit_watch",
                "base_pos": float(b),
                "entry_price": (float(entry_px) if np.isfinite(entry_px) else None),
                "atr_entry": None,
                "initial_r_pct": (
                    float(initial_r_pct) if np.isfinite(initial_r_pct) else None
                ),
                "peak_profit_pct": (
                    float(peak_profit_pct) if np.isfinite(peak_profit_pct) else None
                ),
                "peak_r_multiple": (
                    float(peak_r_mult) if np.isfinite(peak_r_mult) else None
                ),
                "active_tier_r": float(active_tier["r_multiple"])
                if active_tier
                else None,
                "active_tier_retrace": float(active_tier["retrace_ratio"])
                if active_tier
                else None,
                "drawdown_from_peak": (
                    float(dd_from_peak) if np.isfinite(dd_from_peak) else None
                ),
                "tp_triggered": False,
                "open": (float(o) if np.isfinite(o) else None),
                "high": (float(h) if np.isfinite(h) else None),
                "low": (float(l) if np.isfinite(l) else None),
                "tp_fill_price": None,
                "tp_trigger_source": None,
                "gap_open_triggered": None,
                "stop_fill_price": None,
                "stop_trigger_source": None,
                "decision_pos": float(out[i]),
                "in_pos_after": bool(in_pos),
                "wait_next_entry_lock": bool(wait_next_entry_lock),
            }
        )
        if len(trace_last_rows) > 120:
            trace_last_rows = trace_last_rows[-120:]
        prev_base = b

    out_s = pd.Series(out, index=base_pos.index, dtype=float)
    exits = int(((out_s <= 0.0) & (out_s.shift(1).fillna(0.0) > 0.0)).sum())
    trigger_count = int(len(trigger_dates))
    stats = {
        "enabled": True,
        "reentry_mode": reentry_v,
        "atr_window": int(atr_window),
        "atr_n": float(atr_n),
        "fallback_mode_used": bool(not atr_stop_enabled),
        "initial_r_mode": ("atr_stop" if atr_stop_enabled else "virtual_atr_fallback"),
        "trigger_count": trigger_count,
        "trigger_dates": trigger_dates[:200],
        "trigger_events": trigger_events[:200],
        "first_trigger_date": (trigger_dates[0] if trigger_dates else None),
        "last_trigger_date": (trigger_dates[-1] if trigger_dates else None),
        "entries": int(((out_s > 0.0) & (out_s.shift(1).fillna(0.0) <= 0.0)).sum()),
        "exits": exits,
        "trigger_exit_share": (
            float(trigger_count) / float(exits) if exits > 0 else 0.0
        ),
        "wait_next_entry_lock_active": bool(wait_next_entry_lock),
        "tiers": tiers_v,
        "tier_trigger_counts": dict(tier_trigger_counts),
        "invalid_initial_r_entries": int(invalid_r_entries),
        "trigger_rule": "low_vs_peak_retrace_same_day_exit",
        "fill_rule": "fill=min(trigger_price,open_price) for long",
        "trace_last_rows": trace_last_rows[-80:],
    }
    return out_s, stats


def _apply_bias_v_take_profit(
    base_pos: pd.Series,
    *,
    open_: pd.Series | None = None,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    enabled: bool,
    reentry_mode: str,
    ma_window: int,
    atr_window: int,
    threshold: float,
) -> tuple[pd.Series, dict[str, Any]]:
    reentry_v = str(reentry_mode or "reenter").strip().lower()
    if reentry_v not in {"reenter", "wait_next_entry"}:
        reentry_v = "reenter"
    if not bool(enabled):
        out_none = base_pos.fillna(0.0).astype(float)
        return out_none, {
            "enabled": False,
            "reentry_mode": reentry_v,
            "ma_window": int(ma_window),
            "atr_window": int(atr_window),
            "threshold": float(threshold),
            "trigger_count": 0,
            "trigger_dates": [],
            "trigger_events": [],
            "first_trigger_date": None,
            "last_trigger_date": None,
            "entries": int(
                ((out_none > 0.0) & (out_none.shift(1).fillna(0.0) <= 0.0)).sum()
            ),
            "exits": int(
                ((out_none <= 0.0) & (out_none.shift(1).fillna(0.0) > 0.0)).sum()
            ),
            "trigger_exit_share": 0.0,
            "wait_next_entry_lock_active": False,
            "trigger_rule": "high_ge_threshold_same_day_exit",
            "fill_rule": "fill=max(trigger_price,open_price) for long",
            "trace_last_rows": [],
            "trade_records": [],
        }
    bp = base_pos.fillna(0.0).astype(float)
    cl = close.astype(float)
    op = (
        (open_.astype(float) if isinstance(open_, pd.Series) else cl.astype(float))
        .reindex(bp.index)
        .astype(float)
    )
    hi = high.astype(float).fillna(cl)
    lo = low.astype(float).fillna(cl)
    ma = (
        cl.rolling(window=max(2, int(ma_window)), min_periods=max(2, int(ma_window)))
        .mean()
        .astype(float)
    )
    atr = _atr_from_hlc(hi, lo, cl, window=max(2, int(atr_window))).astype(float)
    bias_v = (
        ((cl - ma) / atr.replace(0.0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )

    out = np.zeros(len(bp), dtype=float)
    trigger_dates: list[str] = []
    trigger_events: list[dict[str, Any]] = []
    trade_records: list[dict[str, Any]] = []
    trace_last_rows: list[dict[str, Any]] = []
    in_pos = False
    prev_base = 0.0
    wait_next_entry_lock = False
    entry_decision_date: str | None = None
    initial_tp_price: float = float("nan")

    for i in range(len(bp)):
        b = float(bp.iloc[i]) if np.isfinite(float(bp.iloc[i])) else 0.0
        o = float(op.iloc[i]) if np.isfinite(float(op.iloc[i])) else float("nan")
        c = float(cl.iloc[i]) if np.isfinite(float(cl.iloc[i])) else float("nan")
        h = float(hi.iloc[i]) if np.isfinite(float(hi.iloc[i])) else float("nan")
        l = float(lo.iloc[i]) if np.isfinite(float(lo.iloc[i])) else float("nan")  # noqa: E741
        m = float(ma.iloc[i]) if np.isfinite(float(ma.iloc[i])) else float("nan")
        a = float(atr.iloc[i]) if np.isfinite(float(atr.iloc[i])) else float("nan")
        v = (
            float(bias_v.iloc[i])
            if np.isfinite(float(bias_v.iloc[i]))
            else float("nan")
        )
        d = bp.index[i]
        ds = d.date().isoformat() if hasattr(d, "date") else str(d)
        base_entry_event = bool((b > 0.0) and (prev_base <= 0.0))

        if not in_pos:
            if b <= 0.0 or (not np.isfinite(c)) or c <= 0.0:
                out[i] = 0.0
                prev_base = b
                continue
            if (
                reentry_v == "wait_next_entry"
                and wait_next_entry_lock
                and (not base_entry_event)
            ):
                out[i] = 0.0
                prev_base = b
                continue
            in_pos = True
            wait_next_entry_lock = False
            raw_px = (
                float(m + float(threshold) * a)
                if (np.isfinite(m) and np.isfinite(a))
                else float("nan")
            )
            tp_line_px = float(raw_px) if np.isfinite(raw_px) else float("nan")
            entry_decision_date = ds
            initial_tp_price = (
                float(tp_line_px) if np.isfinite(tp_line_px) else float("nan")
            )
            out[i] = b
            trace_last_rows.append(
                {
                    "date": ds,
                    "event_type": "entry",
                    "event_reason": (
                        "base_entry_signal" if base_entry_event else "tp_reentry"
                    ),
                    "base_pos": float(b),
                    "open": (float(o) if np.isfinite(o) else None),
                    "high": (float(h) if np.isfinite(h) else None),
                    "low": (float(l) if np.isfinite(l) else None),
                    "close": (float(c) if np.isfinite(c) else None),
                    "ma": (float(m) if np.isfinite(m) else None),
                    "atr": (float(a) if np.isfinite(a) else None),
                    "bias_v": (float(v) if np.isfinite(v) else None),
                    "threshold": float(threshold),
                    "tp_trigger_price_raw": (
                        float(raw_px) if np.isfinite(raw_px) else None
                    ),
                    "tp_trigger_price_eff": (
                        float(tp_line_px) if np.isfinite(tp_line_px) else None
                    ),
                    "tp_triggered": False,
                    "tp_fill_price": None,
                    "tp_trigger_source": None,
                    "gap_open_triggered": None,
                    "stop_fill_price": None,
                    "stop_trigger_source": None,
                    "decision_pos": float(out[i]),
                    "in_pos_after": bool(in_pos),
                    "wait_next_entry_lock": bool(wait_next_entry_lock),
                }
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]
            prev_base = b
            continue

        if b <= 0.0:
            in_pos = False
            out[i] = 0.0
            tp_line_px = float("nan")
            entry_decision_date = None
            initial_tp_price = float("nan")
            trace_last_rows.append(
                {
                    "date": ds,
                    "event_type": "exit",
                    "event_reason": "base_exit_signal",
                    "base_pos": float(b),
                    "open": (float(o) if np.isfinite(o) else None),
                    "high": (float(h) if np.isfinite(h) else None),
                    "low": (float(l) if np.isfinite(l) else None),
                    "close": (float(c) if np.isfinite(c) else None),
                    "ma": (float(m) if np.isfinite(m) else None),
                    "atr": (float(a) if np.isfinite(a) else None),
                    "bias_v": (float(v) if np.isfinite(v) else None),
                    "threshold": float(threshold),
                    "tp_trigger_price_raw": None,
                    "tp_trigger_price_eff": None,
                    "tp_triggered": False,
                    "tp_fill_price": None,
                    "tp_trigger_source": None,
                    "gap_open_triggered": None,
                    "stop_fill_price": None,
                    "stop_trigger_source": None,
                    "decision_pos": float(out[i]),
                    "in_pos_after": bool(in_pos),
                    "wait_next_entry_lock": bool(wait_next_entry_lock),
                }
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]
            prev_base = b
            continue

        trigger_px_raw = (
            float(m + float(threshold) * a)
            if (np.isfinite(m) and np.isfinite(a))
            else float("nan")
        )
        if np.isfinite(trigger_px_raw):
            tp_line_px = (
                float(trigger_px_raw)
                if (not np.isfinite(tp_line_px))
                else max(float(tp_line_px), float(trigger_px_raw))
            )
        trigger_px = float(tp_line_px) if np.isfinite(tp_line_px) else float("nan")
        tp_triggered = bool(
            np.isfinite(h) and np.isfinite(trigger_px) and (h >= trigger_px)
        )
        if tp_triggered:
            in_pos = False
            out[i] = 0.0
            gap_open_triggered = bool(
                np.isfinite(o) and np.isfinite(trigger_px) and (o >= trigger_px)
            )
            fill_price = (
                float(o) if gap_open_triggered and np.isfinite(o) else float(trigger_px)
            )
            trigger_source = (
                "gap_open_above_bias_v_tp"
                if gap_open_triggered
                else "high_touch_bias_v_tp"
            )
            if reentry_v == "wait_next_entry":
                wait_next_entry_lock = True
            trigger_dates.append(ds)
            trigger_events.append(
                {
                    "date": ds,
                    "trigger_price": (
                        float(trigger_px) if np.isfinite(trigger_px) else None
                    ),
                    "trigger_price_raw": (
                        float(trigger_px_raw) if np.isfinite(trigger_px_raw) else None
                    ),
                    "trigger_price_eff": (
                        float(trigger_px) if np.isfinite(trigger_px) else None
                    ),
                    "open_price": (float(o) if np.isfinite(o) else None),
                    "high_price": (float(h) if np.isfinite(h) else None),
                    "fill_price": (
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    "trigger_source": trigger_source,
                    "gap_open_triggered": bool(gap_open_triggered),
                    "threshold": float(threshold),
                    "bias_v": (float(v) if np.isfinite(v) else None),
                }
            )
            trade_records.append(
                {
                    "entry_decision_date": entry_decision_date,
                    "entry_execution_date": None,
                    "entry_execution_price": None,
                    "initial_take_profit_price": (
                        float(initial_tp_price)
                        if np.isfinite(initial_tp_price)
                        else None
                    ),
                    "trigger_take_profit_price": (
                        float(trigger_px) if np.isfinite(trigger_px) else None
                    ),
                    "execution_take_profit_price": (
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    "trigger_date": ds,
                }
            )
            trace_last_rows.append(
                {
                    "date": ds,
                    "event_type": "exit",
                    "event_reason": "bias_v_take_profit",
                    "base_pos": float(b),
                    "open": (float(o) if np.isfinite(o) else None),
                    "high": (float(h) if np.isfinite(h) else None),
                    "low": (float(l) if np.isfinite(l) else None),
                    "close": (float(c) if np.isfinite(c) else None),
                    "ma": (float(m) if np.isfinite(m) else None),
                    "atr": (float(a) if np.isfinite(a) else None),
                    "bias_v": (float(v) if np.isfinite(v) else None),
                    "threshold": float(threshold),
                    "tp_trigger_price_raw": (
                        float(trigger_px_raw) if np.isfinite(trigger_px_raw) else None
                    ),
                    "tp_trigger_price_eff": (
                        float(trigger_px) if np.isfinite(trigger_px) else None
                    ),
                    "tp_triggered": True,
                    "tp_fill_price": (
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    "tp_trigger_source": trigger_source,
                    "gap_open_triggered": bool(gap_open_triggered),
                    "stop_fill_price": (
                        float(fill_price) if np.isfinite(fill_price) else None
                    ),
                    "stop_trigger_source": trigger_source,
                    "decision_pos": float(out[i]),
                    "in_pos_after": bool(in_pos),
                    "wait_next_entry_lock": bool(wait_next_entry_lock),
                }
            )
            if len(trace_last_rows) > 120:
                trace_last_rows = trace_last_rows[-120:]
            tp_line_px = float("nan")
            entry_decision_date = None
            initial_tp_price = float("nan")
            prev_base = b
            continue

        out[i] = b
        trace_last_rows.append(
            {
                "date": ds,
                "event_type": "hold",
                "event_reason": "bias_v_take_profit_watch",
                "base_pos": float(b),
                "open": (float(o) if np.isfinite(o) else None),
                "high": (float(h) if np.isfinite(h) else None),
                "low": (float(l) if np.isfinite(l) else None),
                "close": (float(c) if np.isfinite(c) else None),
                "ma": (float(m) if np.isfinite(m) else None),
                "atr": (float(a) if np.isfinite(a) else None),
                "bias_v": (float(v) if np.isfinite(v) else None),
                "threshold": float(threshold),
                "tp_trigger_price_raw": (
                    float(trigger_px_raw) if np.isfinite(trigger_px_raw) else None
                ),
                "tp_trigger_price_eff": (
                    float(trigger_px) if np.isfinite(trigger_px) else None
                ),
                "tp_triggered": False,
                "tp_fill_price": None,
                "tp_trigger_source": None,
                "gap_open_triggered": None,
                "stop_fill_price": None,
                "stop_trigger_source": None,
                "decision_pos": float(out[i]),
                "in_pos_after": bool(in_pos),
                "wait_next_entry_lock": bool(wait_next_entry_lock),
            }
        )
        if len(trace_last_rows) > 120:
            trace_last_rows = trace_last_rows[-120:]
        prev_base = b

    out_s = pd.Series(out, index=base_pos.index, dtype=float)
    exits = int(((out_s <= 0.0) & (out_s.shift(1).fillna(0.0) > 0.0)).sum())
    trigger_count = int(len(trigger_dates))
    stats = {
        "enabled": True,
        "reentry_mode": reentry_v,
        "ma_window": int(ma_window),
        "atr_window": int(atr_window),
        "threshold": float(threshold),
        "trigger_count": trigger_count,
        "trigger_dates": trigger_dates[:200],
        "trigger_events": trigger_events[:200],
        "first_trigger_date": (trigger_dates[0] if trigger_dates else None),
        "last_trigger_date": (trigger_dates[-1] if trigger_dates else None),
        "entries": int(((out_s > 0.0) & (out_s.shift(1).fillna(0.0) <= 0.0)).sum()),
        "exits": exits,
        "trigger_exit_share": (
            float(trigger_count) / float(exits) if exits > 0 else 0.0
        ),
        "wait_next_entry_lock_active": bool(wait_next_entry_lock),
        "trigger_rule": "high_ge_threshold_same_day_exit",
        "fill_rule": "fill=max(trigger_price,open_price) for long",
        "trace_last_rows": trace_last_rows[-80:],
        "trade_records": trade_records,
    }
    return out_s, stats


def compute_trend_backtest(db: Session, inp: TrendInputs) -> dict[str, Any]:
    code = (inp.code or "").strip()
    if not code:
        raise ValueError("code is empty")
    if float(inp.cost_bps) < 0:
        raise ValueError("cost_bps must be >= 0")
    if (not np.isfinite(float(inp.slippage_rate))) or float(inp.slippage_rate) < 0:
        raise ValueError("slippage_rate must be finite and >= 0")
    ep = str(getattr(inp, "exec_price", "open") or "open").strip().lower()
    if ep not in {"open", "close", "oc2"}:
        raise ValueError("exec_price must be one of: open|close|oc2")
    if not np.isfinite(float(inp.risk_free_rate)):
        raise ValueError("risk_free_rate must be finite")
    quick_mode = bool(getattr(inp, "quick_mode", False))

    strat = (inp.strategy or "ma_filter").strip().lower()
    if strat not in {
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
    }:
        raise ValueError(f"invalid strategy={inp.strategy}")
    if int(getattr(inp, "random_hold_days", 20)) < 1:
        raise ValueError("random_hold_days must be >= 1")

    # validate params
    if int(inp.sma_window) < 2:
        raise ValueError("sma_window must be >= 2")
    if int(inp.fast_window) < 2 or int(inp.slow_window) < 2:
        raise ValueError("fast_window/slow_window must be >= 2")
    if int(inp.fast_window) >= int(inp.slow_window):
        raise ValueError("fast_window must be < slow_window")
    strat = str(inp.strategy or "ma_filter").strip().lower()
    ma_type = str(getattr(inp, "ma_type", "sma") or "sma").strip().lower()
    if strat == "ma_cross":
        if ma_type not in {"sma", "ema", "wma"}:
            raise ValueError("ma_type must be one of: sma|ema|wma for ma_cross")
    elif strat == "ma_filter":
        if ma_type not in {"sma", "ema", "kama"}:
            raise ValueError("ma_type must be one of: sma|ema|kama for ma_filter")
    kama_er_window = int(getattr(inp, "kama_er_window", 10) or 10)
    kama_fast_window = int(getattr(inp, "kama_fast_window", 2) or 2)
    kama_slow_window = int(getattr(inp, "kama_slow_window", 30) or 30)
    kama_std_window = int(getattr(inp, "kama_std_window", 20) or 20)
    kama_std_coef = float(getattr(inp, "kama_std_coef", 1.0) or 0.0)
    if kama_er_window < 2:
        raise ValueError("kama_er_window must be >= 2")
    if kama_fast_window < 1:
        raise ValueError("kama_fast_window must be >= 1")
    if kama_slow_window < 2:
        raise ValueError("kama_slow_window must be >= 2")
    if kama_fast_window >= kama_slow_window:
        raise ValueError("kama_fast_window must be < kama_slow_window")
    if kama_std_window < 2:
        raise ValueError("kama_std_window must be >= 2")
    if (not np.isfinite(kama_std_coef)) or kama_std_coef < 0.0 or kama_std_coef > 3.0:
        raise ValueError("kama_std_coef must be in [0,3]")
    if strat == "ma_cross" and ma_type == "kama":
        raise ValueError("ma_type=kama is only supported for ma_filter")
    if not np.isfinite(float(inp.tsmom_entry_threshold)):
        raise ValueError("tsmom_entry_threshold must be finite")
    if not np.isfinite(float(inp.tsmom_exit_threshold)):
        raise ValueError("tsmom_exit_threshold must be finite")
    if float(inp.tsmom_entry_threshold) < float(inp.tsmom_exit_threshold):
        raise ValueError("tsmom thresholds must satisfy: entry >= exit")
    if int(inp.bias_ma_window) < 2:
        raise ValueError("bias_ma_window must be >= 2")
    if not np.isfinite(float(inp.bias_entry)):
        raise ValueError("bias_entry must be finite")
    if not np.isfinite(float(inp.bias_hot)):
        raise ValueError("bias_hot must be finite")
    if not np.isfinite(float(inp.bias_cold)):
        raise ValueError("bias_cold must be finite")
    if not (float(inp.bias_cold) < float(inp.bias_entry) < float(inp.bias_hot)):
        raise ValueError("bias thresholds must satisfy: cold < entry < hot")
    atr_mode = str(getattr(inp, "atr_stop_mode", "none") or "none").strip().lower()
    if atr_mode not in {"none", "static", "trailing", "tightening"}:
        raise ValueError(
            "atr_stop_mode must be one of: none|static|trailing|tightening"
        )
    atr_basis = (
        str(getattr(inp, "atr_stop_atr_basis", "latest") or "latest").strip().lower()
    )
    if atr_basis not in {"entry", "latest"}:
        raise ValueError("atr_stop_atr_basis must be one of: entry|latest")
    atr_reentry_mode = (
        str(getattr(inp, "atr_stop_reentry_mode", "reenter") or "reenter")
        .strip()
        .lower()
    )
    if atr_reentry_mode not in {"reenter", "wait_next_entry"}:
        raise ValueError(
            "atr_stop_reentry_mode must be one of: reenter|wait_next_entry"
        )
    if int(inp.atr_stop_window) < 2:
        raise ValueError("atr_stop_window must be >= 2")
    if (not np.isfinite(float(inp.atr_stop_n))) or float(inp.atr_stop_n) <= 0:
        raise ValueError("atr_stop_n must be finite and > 0")
    if (not np.isfinite(float(inp.atr_stop_m))) or float(inp.atr_stop_m) <= 0:
        raise ValueError("atr_stop_m must be finite and > 0")
    if atr_mode == "tightening" and float(inp.atr_stop_n) <= float(inp.atr_stop_m):
        raise ValueError(
            "atr_stop_n must be > atr_stop_m when atr_stop_mode=tightening"
        )
    rtp_enabled = bool(getattr(inp, "r_take_profit_enabled", False))
    rtp_reentry_mode = (
        str(getattr(inp, "r_take_profit_reentry_mode", "reenter") or "reenter")
        .strip()
        .lower()
    )
    if rtp_reentry_mode not in {"reenter", "wait_next_entry"}:
        raise ValueError(
            "r_take_profit_reentry_mode must be one of: reenter|wait_next_entry"
        )
    rtp_tiers = _normalize_r_take_profit_tiers(
        getattr(inp, "r_take_profit_tiers", None)
    )
    bias_v_tp_enabled = bool(getattr(inp, "bias_v_take_profit_enabled", False))
    bias_v_tp_reentry_mode = (
        str(getattr(inp, "bias_v_take_profit_reentry_mode", "reenter") or "reenter")
        .strip()
        .lower()
    )
    if bias_v_tp_reentry_mode not in {"reenter", "wait_next_entry"}:
        raise ValueError(
            "bias_v_take_profit_reentry_mode must be one of: reenter|wait_next_entry"
        )
    bias_v_tp_ma_window = int(getattr(inp, "bias_v_ma_window", 20) or 20)
    if bias_v_tp_ma_window < 2:
        raise ValueError("bias_v_ma_window must be >= 2")
    bias_v_tp_atr_window = int(getattr(inp, "bias_v_atr_window", 20) or 20)
    if bias_v_tp_atr_window < 2:
        raise ValueError("bias_v_atr_window must be >= 2")
    bias_v_tp_threshold = float(
        getattr(inp, "bias_v_take_profit_threshold", 5.0) or 5.0
    )
    if (not np.isfinite(bias_v_tp_threshold)) or bias_v_tp_threshold <= 0.0:
        raise ValueError("bias_v_take_profit_threshold must be finite and > 0")
    monthly_risk_budget_enabled = bool(
        getattr(inp, "monthly_risk_budget_enabled", False)
    )
    monthly_risk_budget_pct = float(
        getattr(inp, "monthly_risk_budget_pct", 0.06) or 0.06
    )
    monthly_risk_budget_include_new_trade_risk = bool(
        getattr(inp, "monthly_risk_budget_include_new_trade_risk", False)
    )
    if (
        (not np.isfinite(monthly_risk_budget_pct))
        or monthly_risk_budget_pct < 0.01
        or monthly_risk_budget_pct > 0.06
    ):
        raise ValueError("monthly_risk_budget_pct must be in [0.01, 0.06]")
    if int(inp.donchian_entry) < 2 or int(inp.donchian_exit) < 2:
        raise ValueError("donchian_entry/donchian_exit must be >= 2")
    if int(inp.mom_lookback) < 2:
        raise ValueError("mom_lookback must be >= 2")
    if int(inp.bias_ma_window) < 2:
        raise ValueError("bias_ma_window must be >= 2")
    if not np.isfinite(float(inp.bias_entry)):
        raise ValueError("bias_entry must be finite")
    if not np.isfinite(float(inp.bias_hot)):
        raise ValueError("bias_hot must be finite")
    if not np.isfinite(float(inp.bias_cold)):
        raise ValueError("bias_cold must be finite")
    if not (float(inp.bias_cold) < float(inp.bias_entry) < float(inp.bias_hot)):
        raise ValueError("bias thresholds must satisfy: cold < entry < hot")
    bias_mode = (inp.bias_pos_mode or "binary").strip().lower()
    if bias_mode not in {"binary", "continuous"}:
        raise ValueError("bias_pos_mode must be binary|continuous")
    if int(inp.macd_fast) < 2 or int(inp.macd_slow) < 2 or int(inp.macd_signal) < 2:
        raise ValueError("macd_fast/macd_slow/macd_signal must be >= 2")
    if int(inp.macd_fast) >= int(inp.macd_slow):
        raise ValueError("macd_fast must be < macd_slow")
    if int(inp.macd_v_atr_window) < 2:
        raise ValueError("macd_v_atr_window must be >= 2")
    if (not np.isfinite(float(inp.macd_v_scale))) or float(inp.macd_v_scale) <= 0:
        raise ValueError("macd_v_scale must be finite and > 0")
    er_filter = bool(getattr(inp, "er_filter", False))
    er_window = int(getattr(inp, "er_window", 10) or 10)
    er_threshold = float(getattr(inp, "er_threshold", 0.30) or 0.30)
    if er_window < 2:
        raise ValueError("er_window must be >= 2")
    if (not np.isfinite(er_threshold)) or er_threshold < 0.0 or er_threshold > 1.0:
        raise ValueError("er_threshold must be in [0,1]")
    impulse_entry_filter = bool(getattr(inp, "impulse_entry_filter", False))
    impulse_allow_bull = bool(getattr(inp, "impulse_allow_bull", True))
    impulse_allow_bear = bool(getattr(inp, "impulse_allow_bear", False))
    impulse_allow_neutral = bool(getattr(inp, "impulse_allow_neutral", False))
    er_exit_filter = bool(getattr(inp, "er_exit_filter", False))
    er_exit_window = int(getattr(inp, "er_exit_window", 10) or 10)
    er_exit_threshold = float(getattr(inp, "er_exit_threshold", 0.88) or 0.88)
    if er_exit_window < 2:
        raise ValueError("er_exit_window must be >= 2")
    if (
        (not np.isfinite(er_exit_threshold))
        or er_exit_threshold < 0.0
        or er_exit_threshold > 1.0
    ):
        raise ValueError("er_exit_threshold must be in [0,1]")
    ps = str(getattr(inp, "position_sizing", "equal") or "equal").strip().lower()
    if ps not in {"equal", "vol_target", "fixed_ratio", "risk_budget"}:
        raise ValueError(
            "position_sizing must be equal|vol_target|fixed_ratio|risk_budget"
        )
    if int(getattr(inp, "vol_window", 20)) < 2:
        raise ValueError("vol_window must be >= 2")
    if (not np.isfinite(float(getattr(inp, "vol_target_ann", 0.20)))) or float(
        getattr(inp, "vol_target_ann", 0.20)
    ) <= 0:
        raise ValueError("vol_target_ann must be finite and > 0")
    fixed_ratio = float(getattr(inp, "fixed_pos_ratio", 0.04) or 0.04)
    if (not np.isfinite(fixed_ratio)) or fixed_ratio <= 0:
        raise ValueError("fixed_pos_ratio must be finite and > 0")
    fixed_overcap_policy = (
        str(getattr(inp, "fixed_overcap_policy", "skip") or "skip").strip().lower()
    )
    if fixed_overcap_policy not in {"skip", "extend"}:
        raise ValueError("fixed_overcap_policy must be one of: skip|extend")
    fixed_max_holding_n = int(getattr(inp, "fixed_max_holdings", 10) or 10)
    if fixed_max_holding_n < 1:
        raise ValueError("fixed_max_holdings must be >= 1")
    risk_budget_atr_window = int(getattr(inp, "risk_budget_atr_window", 20) or 20)
    if risk_budget_atr_window < 2:
        raise ValueError("risk_budget_atr_window must be >= 2")
    risk_budget_pct = float(getattr(inp, "risk_budget_pct", 0.01) or 0.01)
    if (
        (not np.isfinite(risk_budget_pct))
        or risk_budget_pct < 0.001
        or risk_budget_pct > 0.02
    ):
        raise ValueError("risk_budget_pct must be in [0.001, 0.02]")
    risk_budget_overcap_policy = (
        str(getattr(inp, "risk_budget_overcap_policy", "scale") or "scale")
        .strip()
        .lower()
    )
    if risk_budget_overcap_policy not in {
        "scale",
        "skip_entry",
        "replace_entry",
        "leverage_entry",
    }:
        raise ValueError(
            "risk_budget_overcap_policy must be one of: scale|skip_entry|replace_entry|leverage_entry"
        )
    risk_budget_max_leverage_multiple = float(
        getattr(inp, "risk_budget_max_leverage_multiple", 2.0) or 2.0
    )
    if (
        (not np.isfinite(risk_budget_max_leverage_multiple))
        or risk_budget_max_leverage_multiple < 1.0
        or risk_budget_max_leverage_multiple > 10.0
    ):
        raise ValueError("risk_budget_max_leverage_multiple must be in [1.0, 10.0]")
    vol_regime_risk_mgmt_enabled = bool(
        getattr(inp, "vol_regime_risk_mgmt_enabled", False)
    )
    vol_ratio_fast_atr_window = int(getattr(inp, "vol_ratio_fast_atr_window", 5) or 5)
    vol_ratio_slow_atr_window = int(getattr(inp, "vol_ratio_slow_atr_window", 50) or 50)
    vol_ratio_expand_threshold = float(
        getattr(inp, "vol_ratio_expand_threshold", 1.45) or 1.45
    )
    vol_ratio_contract_threshold = float(
        getattr(inp, "vol_ratio_contract_threshold", 0.65) or 0.65
    )
    vol_ratio_normal_threshold = float(
        getattr(inp, "vol_ratio_normal_threshold", 1.05) or 1.05
    )
    if vol_ratio_fast_atr_window < 2:
        raise ValueError("vol_ratio_fast_atr_window must be >= 2")
    if vol_ratio_slow_atr_window < 2:
        raise ValueError("vol_ratio_slow_atr_window must be >= 2")
    if (not np.isfinite(vol_ratio_expand_threshold)) or vol_ratio_expand_threshold <= 0:
        raise ValueError("vol_ratio_expand_threshold must be > 0")
    if (
        not np.isfinite(vol_ratio_contract_threshold)
    ) or vol_ratio_contract_threshold <= 0:
        raise ValueError("vol_ratio_contract_threshold must be > 0")
    if (not np.isfinite(vol_ratio_normal_threshold)) or vol_ratio_normal_threshold <= 0:
        raise ValueError("vol_ratio_normal_threshold must be > 0")
    if vol_ratio_expand_threshold <= vol_ratio_normal_threshold:
        raise ValueError(
            "vol_ratio_expand_threshold must be > vol_ratio_normal_threshold"
        )
    if vol_ratio_contract_threshold >= vol_ratio_normal_threshold:
        raise ValueError(
            "vol_ratio_contract_threshold must be < vol_ratio_normal_threshold"
        )
    # Price basis consistent with rotation research:
    # - Signal/TA: qfq close
    # - Execution/NAV: none close, with hfq return fallback on corporate-action days to avoid false cliffs
    # - Benchmark (buy&hold): hfq close (total return proxy; non-tradable)
    need_hist = (
        max(
            int(inp.sma_window),
            int(inp.slow_window),
            int(inp.donchian_entry),
            int(inp.mom_lookback),
            20,
        )
        + 60
    )
    ext_start = inp.start - dt.timedelta(days=int(need_hist) * 2)

    close_none = load_close_prices(
        db, codes=[code], start=inp.start, end=inp.end, adjust="none"
    )
    if (
        close_none.empty
        or (code not in close_none.columns)
        or close_none[code].dropna().empty
    ):
        raise ValueError("no execution price data for given range (none)")
    dates = close_none.sort_index().ffill().index

    close_qfq = (
        load_close_prices(db, codes=[code], start=ext_start, end=inp.end, adjust="qfq")
        .sort_index()
        .reindex(dates)
        .ffill()
    )
    close_hfq = (
        load_close_prices(db, codes=[code], start=ext_start, end=inp.end, adjust="hfq")
        .sort_index()
        .reindex(dates)
        .ffill()
    )

    for name, df in [("qfq", close_qfq), ("hfq", close_hfq)]:
        if df.empty or (code not in df.columns) or df[code].dropna().empty:
            raise ValueError(f"missing {name} close data for: {code}")

    px_sig = close_qfq[code].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    px_exec_none = (
        close_none[code].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    )
    px_bh = close_hfq[code].astype(float).replace([np.inf, -np.inf], np.nan).ffill()

    px_sig = px_sig.dropna()
    # align everything to execution calendar
    px_sig = px_sig.reindex(dates).ffill()
    px_exec_none = px_exec_none.reindex(dates).ffill()
    px_bh = px_bh.reindex(dates).ffill()

    if px_exec_none.dropna().empty or px_sig.dropna().empty or px_bh.dropna().empty:
        raise ValueError("no valid price series after alignment")

    high_qfq_df, low_qfq_df = load_high_low_prices(
        db, codes=[code], start=ext_start, end=inp.end, adjust="qfq"
    )
    high_qfq = (
        high_qfq_df[code]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .reindex(dates)
        .ffill()
        if (not high_qfq_df.empty and code in high_qfq_df.columns)
        else px_sig
    )
    low_qfq = (
        low_qfq_df[code]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .reindex(dates)
        .ffill()
        if (not low_qfq_df.empty and code in low_qfq_df.columns)
        else px_sig
    )

    # Returns:
    # - strategy execution: choose open/close/oc2 by exec_price, then apply none->hfq fallback on CA days
    # - benchmark: hfq close
    ohlc_none = load_ohlc_prices(
        db, codes=[code], start=inp.start, end=inp.end, adjust="none"
    )
    ohlc_hfq = load_ohlc_prices(
        db, codes=[code], start=inp.start, end=inp.end, adjust="hfq"
    )
    ohlc_qfq = load_ohlc_prices(
        db, codes=[code], start=inp.start, end=inp.end, adjust="qfq"
    )

    def _pick_series(
        ohlc: dict[str, pd.DataFrame], field: str, fallback: pd.Series
    ) -> pd.Series:
        df = ohlc.get(field, pd.DataFrame())
        if df is None or df.empty or code not in df.columns:
            return fallback.astype(float)
        return (
            pd.to_numeric(df[code], errors="coerce")
            .astype(float)
            .reindex(dates)
            .ffill()
            .combine_first(fallback.astype(float))
        )

    o_none = _pick_series(ohlc_none, "open", px_exec_none)
    c_none = _pick_series(ohlc_none, "close", px_exec_none)
    o_hfq = _pick_series(ohlc_hfq, "open", px_bh)
    c_hfq = _pick_series(ohlc_hfq, "close", px_bh)
    o_qfq = _pick_series(ohlc_qfq, "open", px_sig)
    if ep == "open":
        # Open execution holding return: open[t] -> open[t+1]
        exec_o_none = o_none.combine_first(px_exec_none)
        exec_c_none = c_none.combine_first(px_exec_none)
        exec_o_hfq = o_hfq.combine_first(px_bh)
        exec_c_hfq = c_hfq.combine_first(px_bh)
        ret_none = (
            (exec_o_none.shift(-1).div(exec_o_none) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_exec_hfq = (
            (exec_o_hfq.shift(-1).div(exec_o_hfq) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_intraday_none = (
            (exec_c_none / exec_o_none - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_overnight_none = (
            (exec_o_none.shift(-1).div(exec_c_none) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_intraday_hfq = (
            (exec_c_hfq / exec_o_hfq - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_overnight_hfq = (
            (exec_o_hfq.shift(-1).div(exec_c_hfq) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        px_slip_none = exec_o_none.astype(float)
        px_slip_hfq = exec_o_hfq.astype(float)
    elif ep == "close":
        # Execution-day return when entering at close: close[t+1] / close[t] - 1
        exec_none = c_none.combine_first(px_exec_none)
        exec_hfq = c_hfq.combine_first(px_bh)
        ret_none = (
            (exec_none.shift(-1).div(exec_none) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_exec_hfq = (
            (exec_hfq.shift(-1).div(exec_hfq) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_overnight_none = (
            (o_none.shift(-1).div(c_none) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_intraday_none = (
            (c_none.shift(-1).div(o_none.shift(-1)) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_overnight_hfq = (
            (o_hfq.shift(-1).div(c_hfq) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_intraday_hfq = (
            (c_hfq.shift(-1).div(o_hfq.shift(-1)) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        px_slip_none = exec_none.astype(float)
        px_slip_hfq = exec_hfq.astype(float)
    else:
        # OC2: 50% open-execution(open->open) + 50% close-execution(close->close)
        exec_o_none = o_none.combine_first(px_exec_none)
        exec_c_none = c_none.combine_first(px_exec_none)
        exec_o_hfq = o_hfq.combine_first(px_bh)
        exec_c_hfq = c_hfq.combine_first(px_bh)
        ret_open_none = (
            (exec_o_none.shift(-1).div(exec_o_none) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_close_none = (
            (exec_c_none.shift(-1).div(exec_c_none) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_none = (0.5 * (ret_open_none + ret_close_none)).astype(float)
        ret_open_hfq = (
            (exec_o_hfq.shift(-1).div(exec_o_hfq) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_close_hfq = (
            (exec_c_hfq.shift(-1).div(exec_c_hfq) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_exec_hfq = (0.5 * (ret_open_hfq + ret_close_hfq)).astype(float)
        ret_overnight_none = (
            (0.5 * (o_none.shift(-1).div(c_none) - 1.0))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_intraday_none = (
            (
                0.5 * (c_none / o_none - 1.0)
                + 0.5 * (c_none.shift(-1).div(o_none.shift(-1)) - 1.0)
            )
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_overnight_hfq = (
            (0.5 * (o_hfq.shift(-1).div(c_hfq) - 1.0))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_intraday_hfq = (
            (
                0.5 * (c_hfq / o_hfq - 1.0)
                + 0.5 * (c_hfq.shift(-1).div(o_hfq.shift(-1)) - 1.0)
            )
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        px_slip_none = (0.5 * (exec_o_none + exec_c_none)).astype(float)
        px_slip_hfq = (0.5 * (exec_o_hfq + exec_c_hfq)).astype(float)
    ret_hfq = ret_exec_hfq.astype(float)
    gross_none = (1.0 + ret_none).astype(float)
    gross_hfq = (1.0 + ret_hfq).astype(float)
    corp_factor, ca_mask = corporate_action_mask(gross_none, gross_hfq)
    ret_exec = ret_none.where(~ca_mask.fillna(False), other=ret_exec_hfq).astype(float)
    ret_overnight = ret_overnight_none.where(
        ~ca_mask.fillna(False), other=ret_overnight_hfq
    ).astype(float)
    ret_intraday = ret_intraday_none.where(
        ~ca_mask.fillna(False), other=ret_intraday_hfq
    ).astype(float)
    px_exec_slip = (
        px_slip_none.where(~ca_mask.fillna(False), other=px_slip_hfq)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )

    if strat == "ma_filter":
        ma = _moving_average(
            px_sig,
            window=int(inp.sma_window),
            ma_type=ma_type,
            kama_er_window=kama_er_window,
            kama_fast_window=kama_fast_window,
            kama_slow_window=kama_slow_window,
        )
        if ma_type == "kama":
            kstd = (
                ma.astype(float)
                .rolling(
                    window=int(kama_std_window),
                    min_periods=max(2, int(kama_std_window) // 2),
                )
                .std(ddof=0)
                .fillna(0.0)
            )
            raw_pos = _pos_from_band(
                px_sig.astype(float),
                ma.astype(float),
                band=(float(kama_std_coef) * kstd),
            ).astype(float)
        else:
            raw_pos = (px_sig > ma).astype(float).fillna(0.0)
    elif strat == "ma_cross":
        fast = _moving_average(px_sig, window=int(inp.fast_window), ma_type=ma_type)
        slow = _moving_average(px_sig, window=int(inp.slow_window), ma_type=ma_type)
        raw_pos = (fast > slow).astype(float).fillna(0.0)
    elif strat == "donchian":
        raw_pos = _pos_from_donchian(
            px_sig, entry=int(inp.donchian_entry), exit_=int(inp.donchian_exit)
        ).astype(float)
    elif strat == "linreg_slope":
        # Linear regression slope of log price over window; long if slope > 0.
        n = int(inp.sma_window)
        y = np.log(px_sig.clip(lower=1e-12).astype(float))
        slope = y.rolling(window=n, min_periods=max(2, n // 2)).apply(
            _rolling_linreg_slope, raw=True
        )
        raw_pos = (slope > 0.0).astype(float).fillna(0.0)
    elif strat == "bias":
        # BIAS rising-follow strategy:
        # - Trend filter removed (per research need)
        # - BIAS computed from log-diff to EMA: (ln(C) - ln(EMA(C,N))) * 100 (percent)
        # - Enter when bias > entry; exit on take-profit (bias >= hot) or stop-loss (bias <= cold)
        b_win = int(inp.bias_ma_window)
        ema = px_sig.ewm(
            span=b_win, adjust=False, min_periods=max(2, b_win // 2)
        ).mean()
        ln_c = np.log(px_sig.clip(lower=1e-12).astype(float))
        ln_ema = np.log(ema.clip(lower=1e-12).astype(float))
        bias = (ln_c - ln_ema) * 100.0

        entry = float(inp.bias_entry)
        hot = float(inp.bias_hot)
        cold = float(inp.bias_cold)
        pos_mode = bias_mode

        pos = np.zeros(len(px_sig), dtype=float)
        in_pos = False
        for i, d in enumerate(px_sig.index):
            if not np.isfinite(float(bias.loc[d])):
                in_pos = False
                pos[i] = 0.0
                continue

            b = float(bias.loc[d])
            if not in_pos:
                if b > entry:
                    in_pos = True
            else:
                if (b >= hot) or (b <= cold):
                    in_pos = False

            if not in_pos:
                pos[i] = 0.0
            elif pos_mode == "binary":
                pos[i] = 1.0
            else:
                # Continuous sizing: scale exposure with bias inside [cold, hot]
                # cold -> 0, hot -> 1, clipped.
                w = (b - cold) / (hot - cold)
                pos[i] = float(np.clip(w, 0.0, 1.0))

        raw_pos = pd.Series(pos, index=px_sig.index, dtype=float)
    elif strat == "macd_cross":
        macd, sig, _ = _macd_core(
            px_sig,
            fast=int(inp.macd_fast),
            slow=int(inp.macd_slow),
            signal=int(inp.macd_signal),
        )
        raw_pos = (macd > sig).astype(float).fillna(0.0)
    elif strat == "macd_zero_filter":
        macd, _, _ = _macd_core(
            px_sig,
            fast=int(inp.macd_fast),
            slow=int(inp.macd_slow),
            signal=int(inp.macd_signal),
        )
        raw_pos = (macd > 0.0).astype(float).fillna(0.0)
    elif strat == "macd_v":
        macd, _, _ = _macd_core(
            px_sig,
            fast=int(inp.macd_fast),
            slow=int(inp.macd_slow),
            signal=int(inp.macd_signal),
        )
        atr = _atr_from_hlc(
            high_qfq, low_qfq, px_sig, window=int(inp.macd_v_atr_window)
        )
        macd_v = (macd / atr.replace(0.0, np.nan)) * float(inp.macd_v_scale)
        macd_v_sig = _ema(macd_v, int(inp.macd_signal))
        raw_pos = (macd_v > macd_v_sig).astype(float).fillna(0.0)
    elif strat == "random_entry":
        raw_pos = _pos_from_random_entry_hold(
            px_sig.index,
            hold_days=int(getattr(inp, "random_hold_days", 20)),
            seed=getattr(inp, "random_seed", 42),
        )
    else:
        mom = px_sig / px_sig.shift(int(inp.mom_lookback)) - 1.0
        raw_pos = _pos_from_tsmom(
            mom,
            entry_threshold=float(inp.tsmom_entry_threshold),
            exit_threshold=float(inp.tsmom_exit_threshold),
        ).astype(float)

    impulse_filter_stats_overall = {
        "blocked_entry_count": 0,
        "attempted_entry_count": 0,
        "allowed_entry_count": 0,
        "blocked_entry_count_bull": 0,
        "blocked_entry_count_bear": 0,
        "blocked_entry_count_neutral": 0,
    }
    er_filter_stats_overall = {
        "blocked_entry_count": 0,
        "attempted_entry_count": 0,
        "allowed_entry_count": 0,
    }
    er_exit_filter_stats_overall = {
        "trigger_count": 0,
        "trigger_dates": [],
        "trace_last_rows": [],
    }
    impulse_state: pd.Series | None = _compute_impulse_state(
        px_sig,
        ema_window=13,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
    )
    if impulse_entry_filter and (impulse_state is not None):
        raw_pos, impulse_filter_stats_overall = _apply_impulse_entry_filter(
            raw_pos.astype(float).fillna(0.0),
            impulse_state=impulse_state,
            allow_bull=impulse_allow_bull,
            allow_bear=impulse_allow_bear,
            allow_neutral=impulse_allow_neutral,
        )
    if er_filter:
        er = _efficiency_ratio(px_sig, window=er_window)
        raw_pos, er_filter_stats_overall = _apply_er_entry_filter(
            raw_pos.astype(float).fillna(0.0),
            er=er,
            threshold=er_threshold,
        )
    if er_exit_filter:
        er_exit = _efficiency_ratio(px_sig, window=er_exit_window)
        raw_pos, er_exit_filter_stats_overall = _apply_er_exit_filter(
            raw_pos.astype(float).fillna(0.0),
            er=er_exit,
            threshold=er_exit_threshold,
        )
    base_pos = raw_pos.astype(float).fillna(0.0)
    raw_pos, atr_stop_stats = _apply_atr_stop(
        raw_pos,
        open_=o_qfq.astype(float),
        close=px_sig,
        high=high_qfq,
        low=low_qfq,
        mode=atr_mode,
        atr_basis=atr_basis,
        reentry_mode=atr_reentry_mode,
        atr_window=int(inp.atr_stop_window),
        n_mult=float(inp.atr_stop_n),
        m_step=float(inp.atr_stop_m),
    )
    atr_stop_stats = {
        **(atr_stop_stats or {}),
        **_extract_atr_plan_stops_from_trace(atr_stop_stats or {}),
    }
    raw_pos, bias_v_take_profit_stats = _apply_bias_v_take_profit(
        raw_pos,
        open_=o_qfq.astype(float),
        close=px_sig,
        high=high_qfq,
        low=low_qfq,
        enabled=bias_v_tp_enabled,
        reentry_mode=bias_v_tp_reentry_mode,
        ma_window=int(bias_v_tp_ma_window),
        atr_window=int(bias_v_tp_atr_window),
        threshold=float(bias_v_tp_threshold),
    )
    raw_pos, r_take_profit_stats = _apply_r_multiple_take_profit(
        raw_pos,
        open_=o_qfq.astype(float),
        close=px_sig,
        high=high_qfq,
        low=low_qfq,
        enabled=rtp_enabled,
        reentry_mode=rtp_reentry_mode,
        atr_window=int(inp.atr_stop_window),
        atr_n=float(inp.atr_stop_n),
        tiers=rtp_tiers,
        atr_stop_enabled=bool(atr_mode != "none"),
    )
    raw_pos = raw_pos.astype(float)
    vol_risk_adjust_stats_overall: dict[str, int] = {
        "vol_risk_adjust_total_count": 0,
        "vol_risk_adjust_reduce_on_expand_count": 0,
        "vol_risk_adjust_increase_on_contract_count": 0,
        "vol_risk_adjust_recover_from_expand_count": 0,
        "vol_risk_adjust_recover_from_contract_count": 0,
    }

    # Single-asset sizing overlay (same knobs as portfolio mode).
    sizing_scale = pd.Series(1.0, index=raw_pos.index, dtype=float)
    if ps == "fixed_ratio":
        sizing_scale = pd.Series(float(fixed_ratio), index=raw_pos.index, dtype=float)
    elif ps == "risk_budget":
        atr_rb = (
            _atr_from_hlc(
                high_qfq.astype(float).fillna(px_sig),
                low_qfq.astype(float).fillna(px_sig),
                px_sig.astype(float),
                window=int(risk_budget_atr_window),
            )
            .reindex(raw_pos.index)
            .astype(float)
        )
        atr_fast = (
            _atr_from_hlc(
                high_qfq.astype(float).fillna(px_sig),
                low_qfq.astype(float).fillna(px_sig),
                px_sig.astype(float),
                window=int(vol_ratio_fast_atr_window),
            )
            .reindex(raw_pos.index)
            .astype(float)
        )
        atr_slow = (
            _atr_from_hlc(
                high_qfq.astype(float).fillna(px_sig),
                low_qfq.astype(float).fillna(px_sig),
                px_sig.astype(float),
                window=int(vol_ratio_slow_atr_window),
            )
            .reindex(raw_pos.index)
            .astype(float)
        )
        sizing_scale, vol_risk_adjust_stats_overall = _risk_budget_dynamic_weights(
            raw_pos.astype(float).fillna(0.0),
            close=px_sig.astype(float),
            atr_for_budget=atr_rb,
            atr_fast=atr_fast,
            atr_slow=atr_slow,
            risk_budget_pct=float(risk_budget_pct),
            dynamic_enabled=bool(vol_regime_risk_mgmt_enabled),
            expand_threshold=float(vol_ratio_expand_threshold),
            contract_threshold=float(vol_ratio_contract_threshold),
            normal_threshold=float(vol_ratio_normal_threshold),
        )
    elif ps == "vol_target":
        asset_vol = (
            ret_exec.rolling(
                window=max(2, int(inp.vol_window)),
                min_periods=max(2, int(inp.vol_window)),
            )
            .std()
            .mul(np.sqrt(TRADING_DAYS_PER_YEAR))
        ).replace([np.inf, -np.inf], np.nan)
        sizing_scale = (
            (float(inp.vol_target_ann) / asset_vol)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=0.0, upper=1.0)
            .astype(float)
        )
    raw_pos = (raw_pos.clip(lower=0.0, upper=1.0) * sizing_scale).astype(float)
    atr_gate = (
        _atr_from_hlc(
            high_qfq.astype(float).fillna(px_sig),
            low_qfq.astype(float).fillna(px_sig),
            px_sig.astype(float),
            window=int(inp.atr_stop_window),
        )
        .reindex(raw_pos.index)
        .astype(float)
    )
    monthly_risk_budget_gate_stats: dict[str, Any] = {
        "enabled": False,
        "budget_pct": float(monthly_risk_budget_pct),
        "include_new_trade_risk": bool(monthly_risk_budget_include_new_trade_risk),
        "attempted_entry_count": 0,
        "attempted_entry_count_by_code": {str(code): 0},
        "blocked_entry_count": 0,
        "blocked_entry_count_by_code": {str(code): 0},
    }
    if monthly_risk_budget_enabled:
        gated_w_df, monthly_risk_budget_gate_stats = _apply_monthly_risk_budget_gate(
            raw_pos.to_frame(code),
            close=px_sig.to_frame(code).reindex(raw_pos.index),
            atr=atr_gate.to_frame(code).reindex(raw_pos.index),
            enabled=True,
            budget_pct=float(monthly_risk_budget_pct),
            include_new_trade_risk=bool(monthly_risk_budget_include_new_trade_risk),
            atr_stop_enabled=bool(atr_mode != "none"),
            atr_mode=str(atr_mode),
            atr_basis=str(atr_basis),
            atr_n=float(inp.atr_stop_n),
            atr_m=float(inp.atr_stop_m),
            fallback_position_risk=0.01,
        )
        raw_pos = gated_w_df[code].astype(float)

    # Weights become effective on execution day; ret_exec is already aligned to execution-day semantics.
    w = raw_pos.shift(1).fillna(0.0).astype(float).clip(lower=0.0, upper=1.0)
    w, atr_stop_override_ret = _apply_intraday_stop_execution_single(
        weights=w,
        atr_stop_stats=atr_stop_stats,
        exec_price=str(ep),
        open_sig=o_qfq.reindex(w.index).astype(float).ffill(),
        close_sig=px_sig.reindex(w.index).astype(float).ffill(),
    )
    w, bias_v_take_profit_override_ret = _apply_intraday_stop_execution_single(
        weights=w,
        atr_stop_stats=bias_v_take_profit_stats,
        exec_price=str(ep),
        open_sig=o_qfq.reindex(w.index).astype(float).ffill(),
        close_sig=px_sig.reindex(w.index).astype(float).ffill(),
    )
    w, r_take_profit_override_ret = _apply_intraday_stop_execution_single(
        weights=w,
        atr_stop_stats=r_take_profit_stats,
        exec_price=str(ep),
        open_sig=o_qfq.reindex(w.index).astype(float).ffill(),
        close_sig=px_sig.reindex(w.index).astype(float).ffill(),
    )
    # Match rotation open/oc2 execution semantics (AGENTS strict execution-timing NAV rule):
    # while long (w>0), open-leg day returns use same-day open->close (none; hfq on CA days),
    # not forward open[t]->open[t+1]. Forward legs still apply on flat days (w=0) for series continuity.
    if ep in {"open", "oc2"}:
        same_day_none = (
            (c_none / o_none - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        same_day_hfq = (
            (c_hfq / o_hfq - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        open_fwd_none = (
            (o_none.shift(-1).div(o_none) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        open_fwd_hfq = (
            (o_hfq.shift(-1).div(o_hfq) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        close_fwd_none = (
            (c_none.shift(-1).div(c_none) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        close_fwd_hfq = (
            (c_hfq.shift(-1).div(c_hfq) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        cm = ca_mask.reindex(ret_exec.index).fillna(False)
        w_ix = w.reindex(ret_exec.index).fillna(0.0)
        if ep == "open":
            for d in ret_exec.index:
                if float(w_ix.loc[d]) <= 1e-12:
                    continue
                ret_exec.loc[d] = (
                    float(same_day_hfq.loc[d])
                    if bool(cm.loc[d])
                    else float(same_day_none.loc[d])
                )
        else:
            ret_blend_none = pd.Series(0.0, index=ret_exec.index, dtype=float)
            ret_blend_hfq = pd.Series(0.0, index=ret_exec.index, dtype=float)
            for d in ret_exec.index:
                hold = float(w_ix.loc[d]) > 1e-12
                po_n = (
                    float(same_day_none.loc[d]) if hold else float(open_fwd_none.loc[d])
                )
                po_h = (
                    float(same_day_hfq.loc[d]) if hold else float(open_fwd_hfq.loc[d])
                )
                cn = float(close_fwd_none.loc[d])
                ch = float(close_fwd_hfq.loc[d])
                ret_blend_none.loc[d] = 0.5 * (po_n + cn)
                ret_blend_hfq.loc[d] = 0.5 * (po_h + ch)
            ret_exec = ret_blend_none.where(~cm, ret_blend_hfq).astype(float)
    ret_exec_day = ret_exec.astype(float)
    # Buy-and-hold benchmark: always invested; align daily returns with exec_price (tests + rotation parity).
    cm_bh = ca_mask.reindex(ret_hfq.index)
    if isinstance(cm_bh, pd.DataFrame):
        cm_bh = (
            cm_bh.iloc[:, 0].fillna(False)
            if cm_bh.shape[1]
            else pd.Series(False, index=ret_hfq.index)
        )
    else:
        cm_bh = cm_bh.fillna(False)
    bh_same_none = (
        (c_none / o_none - 1.0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )
    bh_same_hfq = (
        (c_hfq / o_hfq - 1.0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )
    if ep == "close":
        ret_bh = ret_hfq.astype(float)
    elif ep == "open":
        ret_bh = (
            bh_same_none.where(~cm_bh, bh_same_hfq)
            .astype(float)
            .reindex(ret_hfq.index)
            .fillna(0.0)
        )
    else:
        cf_none = (
            (c_none.shift(-1).div(c_none) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        cf_hfq = (
            (c_hfq.shift(-1).div(c_hfq) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        blend_bh_none = (0.5 * (bh_same_none + cf_none)).astype(float)
        blend_bh_hfq = (0.5 * (bh_same_hfq + cf_hfq)).astype(float)
        ret_bh = (
            blend_bh_none.where(~cm_bh, blend_bh_hfq)
            .astype(float)
            .reindex(ret_hfq.index)
            .fillna(0.0)
        )
    cost = _turnover_cost_from_weights(w, cost_bps=float(inp.cost_bps))
    turnover = (w - w.shift(1).fillna(0.0)).abs() / 2.0
    slippage = slippage_return_from_turnover(
        turnover.astype(float),
        slippage_spread=float(inp.slippage_rate),
        exec_price=px_exec_slip.reindex(turnover.index).ffill(),
    ).astype(float)
    atr_stop_override_ret = (
        atr_stop_override_ret.reindex(w.index).fillna(0.0).astype(float)
    )
    bias_v_take_profit_override_ret = (
        bias_v_take_profit_override_ret.reindex(w.index).fillna(0.0).astype(float)
    )
    r_take_profit_override_ret = (
        r_take_profit_override_ret.reindex(w.index).fillna(0.0).astype(float)
    )
    risk_exit_override_ret = (
        atr_stop_override_ret
        + bias_v_take_profit_override_ret
        + r_take_profit_override_ret
    ).astype(float)
    strat_ret = (w * ret_exec_day + risk_exit_override_ret - cost - slippage).astype(
        float
    )
    if not quick_mode:
        atr_stop_stats = {
            **(atr_stop_stats or {}),
            "trade_records": _enrich_trade_records_with_engine_timeline(
                records=list((atr_stop_stats or {}).get("trade_records") or []),
                effective_weight=w.astype(float),
                exec_price_series=px_exec_slip.reindex(w.index).ffill().astype(float),
                slippage_spread=float(inp.slippage_rate),
            ),
        }
        bias_v_take_profit_stats = {
            **(bias_v_take_profit_stats or {}),
            "trade_records": _enrich_trade_records_with_engine_timeline(
                records=list(
                    (bias_v_take_profit_stats or {}).get("trade_records") or []
                ),
                effective_weight=w.astype(float),
                exec_price_series=px_exec_slip.reindex(w.index).ffill().astype(float),
                slippage_spread=float(inp.slippage_rate),
            ),
        }
    else:
        atr_stop_stats = {**(atr_stop_stats or {}), "trade_records": []}
        bias_v_take_profit_stats = {
            **(bias_v_take_profit_stats or {}),
            "trade_records": [],
        }
    turnover_daily = ((w - w.shift(1).fillna(0.0)).abs() / 2.0).astype(float)
    return_decomposition: dict[str, Any] | None = None
    if not quick_mode:
        decomp_overnight = (
            w * ret_overnight.reindex(w.index).astype(float).fillna(0.0)
        ).astype(float)
        decomp_intraday = (
            w * ret_intraday.reindex(w.index).astype(float).fillna(0.0)
        ).astype(float)
        decomp_interaction = (
            w
            * (
                ret_overnight.reindex(w.index).astype(float).fillna(0.0)
                * ret_intraday.reindex(w.index).astype(float).fillna(0.0)
            )
        ).astype(float)
        decomp_atr_stop_override = atr_stop_override_ret.astype(float)
        decomp_bias_v_take_profit_override = bias_v_take_profit_override_ret.astype(
            float
        )
        decomp_r_take_profit_override = r_take_profit_override_ret.astype(float)
        decomp_risk_exit_override = (
            decomp_atr_stop_override
            + decomp_bias_v_take_profit_override
            + decomp_r_take_profit_override
        ).astype(float)
        decomp_cost = (cost + slippage).astype(float)
        decomp_gross = (
            decomp_overnight
            + decomp_intraday
            + decomp_interaction
            + decomp_risk_exit_override
        ).astype(float)
        decomp_net = (decomp_gross - decomp_cost).astype(float)
        return_decomposition = {
            "dates": w.index.date.astype(str).tolist(),
            "series": {
                "overnight": decomp_overnight.astype(float).tolist(),
                "intraday": decomp_intraday.astype(float).tolist(),
                "interaction": decomp_interaction.astype(float).tolist(),
                "atr_stop_override": decomp_atr_stop_override.astype(float).tolist(),
                "bias_v_take_profit_override": decomp_bias_v_take_profit_override.astype(
                    float
                ).tolist(),
                "r_take_profit_override": decomp_r_take_profit_override.astype(
                    float
                ).tolist(),
                "risk_exit_override": decomp_risk_exit_override.astype(float).tolist(),
                "cost": decomp_cost.astype(float).tolist(),
                "gross": decomp_gross.astype(float).tolist(),
                "net": decomp_net.astype(float).tolist(),
            },
            "summary": {
                "ann_overnight": float(
                    decomp_overnight.iloc[1:].mean() * TRADING_DAYS_PER_YEAR
                )
                if len(decomp_overnight) > 1
                else 0.0,
                "ann_intraday": float(
                    decomp_intraday.iloc[1:].mean() * TRADING_DAYS_PER_YEAR
                )
                if len(decomp_intraday) > 1
                else 0.0,
                "ann_interaction": float(
                    decomp_interaction.iloc[1:].mean() * TRADING_DAYS_PER_YEAR
                )
                if len(decomp_interaction) > 1
                else 0.0,
                "ann_atr_stop_override": float(
                    decomp_atr_stop_override.iloc[1:].mean() * TRADING_DAYS_PER_YEAR
                )
                if len(decomp_atr_stop_override) > 1
                else 0.0,
                "ann_bias_v_take_profit_override": float(
                    decomp_bias_v_take_profit_override.iloc[1:].mean()
                    * TRADING_DAYS_PER_YEAR
                )
                if len(decomp_bias_v_take_profit_override) > 1
                else 0.0,
                "ann_r_take_profit_override": float(
                    decomp_r_take_profit_override.iloc[1:].mean()
                    * TRADING_DAYS_PER_YEAR
                )
                if len(decomp_r_take_profit_override) > 1
                else 0.0,
                "ann_risk_exit_override": float(
                    decomp_risk_exit_override.iloc[1:].mean() * TRADING_DAYS_PER_YEAR
                )
                if len(decomp_risk_exit_override) > 1
                else 0.0,
                "ann_cost": float(decomp_cost.iloc[1:].mean() * TRADING_DAYS_PER_YEAR)
                if len(decomp_cost) > 1
                else 0.0,
                "ann_gross": float(decomp_gross.iloc[1:].mean() * TRADING_DAYS_PER_YEAR)
                if len(decomp_gross) > 1
                else 0.0,
                "ann_net": float(decomp_net.iloc[1:].mean() * TRADING_DAYS_PER_YEAR)
                if len(decomp_net) > 1
                else 0.0,
            },
        }
    nav = (1.0 + strat_ret).cumprod()
    if len(nav) > 0:
        nav.iloc[0] = 1.0

    bh_nav = (1.0 + ret_bh).cumprod()
    if len(bh_nav) > 0:
        bh_nav.iloc[0] = 1.0

    active = strat_ret - ret_bh
    excess_nav = (1.0 + active).cumprod()
    if len(excess_nav) > 0:
        excess_nav.iloc[0] = 1.0

    # metrics
    ui_strat = float(_ulcer_index(nav, in_percent=True))
    ui_bh = float(_ulcer_index(bh_nav, in_percent=True))
    ann_strat = float(_annualized_return(nav, ann_factor=TRADING_DAYS_PER_YEAR))
    ann_bh = float(_annualized_return(bh_nav, ann_factor=TRADING_DAYS_PER_YEAR))
    m_strat = {
        "cumulative_return": float(nav.iloc[-1] - 1.0),
        "annualized_return": float(ann_strat),
        "annualized_volatility": float(
            _annualized_vol(strat_ret, ann_factor=TRADING_DAYS_PER_YEAR)
        ),
        "max_drawdown": float(_max_drawdown(nav)),
        "max_drawdown_recovery_days": int(_max_drawdown_duration_days(nav)),
        "sharpe_ratio": float(
            _sharpe(
                strat_ret,
                rf=float(inp.risk_free_rate),
                ann_factor=TRADING_DAYS_PER_YEAR,
            )
        ),
        "sortino_ratio": float(
            _sortino(
                strat_ret,
                rf=float(inp.risk_free_rate),
                ann_factor=TRADING_DAYS_PER_YEAR,
            )
        ),
        "ulcer_index": float(ui_strat),
        "ulcer_performance_index": float(
            (ann_strat - float(inp.risk_free_rate)) / (ui_strat / 100.0)
        )
        if ui_strat > 0
        else float("nan"),
        "avg_daily_turnover": float(turnover_daily.mean())
        if len(turnover_daily)
        else 0.0,
    }
    m_bh = {
        "cumulative_return": float(bh_nav.iloc[-1] - 1.0),
        "annualized_return": float(ann_bh),
        "annualized_volatility": float(
            _annualized_vol(ret_bh, ann_factor=TRADING_DAYS_PER_YEAR)
        ),
        "max_drawdown": float(_max_drawdown(bh_nav)),
        "max_drawdown_recovery_days": int(_max_drawdown_duration_days(bh_nav)),
        "sharpe_ratio": float(
            _sharpe(
                ret_bh, rf=float(inp.risk_free_rate), ann_factor=TRADING_DAYS_PER_YEAR
            )
        ),
        "sortino_ratio": float(
            _sortino(
                ret_bh, rf=float(inp.risk_free_rate), ann_factor=TRADING_DAYS_PER_YEAR
            )
        ),
        "ulcer_index": float(ui_bh),
        "ulcer_performance_index": float(
            (ann_bh - float(inp.risk_free_rate)) / (ui_bh / 100.0)
        )
        if ui_bh > 0
        else float("nan"),
    }
    m_ex = {
        "cumulative_return": float(excess_nav.iloc[-1] - 1.0),
        "annualized_return": float(
            _annualized_return(excess_nav, ann_factor=TRADING_DAYS_PER_YEAR)
        ),
        "information_ratio": float(
            _sharpe(active, rf=0.0, ann_factor=TRADING_DAYS_PER_YEAR)
        ),
    }
    attribution = _compute_return_risk_contributions(
        asset_ret=ret_exec_day.to_frame(code)
        .astype(float)
        .reindex(nav.index)
        .fillna(0.0),
        weights=w.to_frame(code).astype(float).reindex(nav.index).fillna(0.0),
        total_return=float(nav.iloc[-1] - 1.0),
    )
    trade_one = _trade_returns_from_weight_series(
        w,
        ret_exec_day,
        cost_bps=float(inp.cost_bps),
        slippage_rate=float(inp.slippage_rate),
        exec_price=px_exec_slip.reindex(nav.index).ffill(),
        dates=nav.index,
    )
    sample_days = int(len(strat_ret))
    complete_trade_count = int(len(trade_one.get("returns", [])))
    avg_daily_turnover = float(turnover_daily.mean()) if len(turnover_daily) else 0.0
    avg_annual_turnover = float(avg_daily_turnover * TRADING_DAYS_PER_YEAR)
    avg_daily_trade_count = (
        float(complete_trade_count / sample_days) if sample_days > 0 else 0.0
    )
    avg_annual_trade_count = float(avg_daily_trade_count * TRADING_DAYS_PER_YEAR)
    m_strat["avg_daily_turnover"] = float(avg_daily_turnover)
    m_strat["avg_annual_turnover"] = float(avg_annual_turnover)
    m_strat["avg_annual_turnover_rate"] = float(avg_annual_turnover)
    m_strat["avg_daily_trade_count"] = float(avg_daily_trade_count)
    m_strat["avg_annual_trade_count"] = float(avg_annual_trade_count)
    atr_risk = _atr_from_hlc(
        high_qfq.astype(float).fillna(px_sig),
        low_qfq.astype(float).fillna(px_sig),
        px_sig.astype(float),
        window=int(inp.atr_stop_window),
    ).reindex(nav.index)
    trade_r_pack = enrich_trades_with_r_metrics(
        trade_one.get("trades", []),
        nav=nav.astype(float),
        weights=w.astype(float),
        exec_price=px_exec_slip.reindex(nav.index).ffill().astype(float),
        atr=atr_risk.astype(float),
        atr_mult=float(inp.atr_stop_n),
        risk_budget_pct=(float(risk_budget_pct) if ps == "risk_budget" else None),
        cost_bps=float(inp.cost_bps),
        slippage_rate=float(inp.slippage_rate),
        default_code=str(code),
        ulcer_index=float(m_strat.get("ulcer_index"))
        if np.isfinite(float(m_strat.get("ulcer_index", np.nan)))
        else None,
        annual_trade_count=float(avg_annual_trade_count)
        if np.isfinite(float(avg_annual_trade_count))
        else None,
        backtest_years=(float(sample_days) / float(TRADING_DAYS_PER_YEAR))
        if sample_days > 0
        else None,
        score_sqn_weight=0.60,
        score_ulcer_weight=0.40,
    )
    r_stats_out = dict(trade_r_pack.get("statistics") or {})
    r_stats_out.pop("trade_system_score", None)
    trades_with_r = list(trade_r_pack.get("trades") or [])
    if not quick_mode:
        mom_for_entry = (px_sig / px_sig.shift(int(inp.mom_lookback)) - 1.0).astype(
            float
        )
        er_for_entry = _efficiency_ratio(px_sig, window=int(er_window)).astype(float)
        atr_fast_for_entry = _atr_from_hlc(
            high_qfq.astype(float).fillna(px_sig),
            low_qfq.astype(float).fillna(px_sig),
            px_sig.astype(float),
            window=int(vol_ratio_fast_atr_window),
        ).astype(float)
        atr_slow_for_entry = _atr_from_hlc(
            high_qfq.astype(float).fillna(px_sig),
            low_qfq.astype(float).fillna(px_sig),
            px_sig.astype(float),
            window=int(vol_ratio_slow_atr_window),
        ).astype(float)
        vol_ratio_for_entry = (
            atr_fast_for_entry / atr_slow_for_entry.replace(0.0, np.nan)
        ).astype(float)
        condition_bins_by_code_single = {
            str(code): {
                "momentum": _bucketize_momentum_series(
                    mom_for_entry.reindex(nav.index)
                ),
                "er": _bucketize_er_series(er_for_entry.reindex(nav.index)),
                "vol_ratio": _bucketize_vol_ratio_series(
                    vol_ratio_for_entry.reindex(nav.index)
                ),
                "impulse": _bucketize_impulse_series(
                    (
                        impulse_state
                        if impulse_state is not None
                        else pd.Series(index=nav.index, dtype=object)
                    ).reindex(nav.index)
                ),
            }
        }
        trades_with_r = _attach_entry_condition_bins_to_trades(
            trades_with_r,
            condition_bins_by_code=condition_bins_by_code_single,
            dates=nav.index,
            default_code=str(code),
        )
    single_rtp_tier_counts = dict(
        ((r_take_profit_stats or {}).get("tier_trigger_counts") or {})
    )
    mfe_r_distribution = build_trade_mfe_r_distribution(
        trade_one.get("trades", []),
        close=px_sig.astype(float).reindex(nav.index).ffill(),
        high=high_qfq.astype(float).fillna(px_sig).reindex(nav.index).ffill(),
        atr=atr_risk.astype(float).reindex(nav.index),
        atr_mult=float(inp.atr_stop_n),
        default_code=str(code),
    )
    trade_stats = {
        "overall": _trade_stats_from_returns(trade_one.get("returns", [])),
        "by_code": {str(code): _trade_stats_from_returns(trade_one.get("returns", []))},
        "trades": ([] if quick_mode else trades_with_r),
        "trades_by_code": (
            {str(code): []} if quick_mode else {str(code): trades_with_r}
        ),
        "mfe_r_distribution": mfe_r_distribution,
    }
    impulse_attempted_overall = int(
        (impulse_filter_stats_overall or {}).get("attempted_entry_count", 0)
    )
    impulse_blocked_overall = int(
        (impulse_filter_stats_overall or {}).get("blocked_entry_count", 0)
    )
    monthly_attempted_overall = int(
        (monthly_risk_budget_gate_stats or {}).get("attempted_entry_count", 0)
    )
    monthly_blocked_overall = int(
        (monthly_risk_budget_gate_stats or {}).get("blocked_entry_count", 0)
    )
    impulse_block_rate_overall = (
        float(impulse_blocked_overall / impulse_attempted_overall)
        if impulse_attempted_overall > 0
        else 0.0
    )
    monthly_block_rate_overall = (
        float(monthly_blocked_overall / monthly_attempted_overall)
        if monthly_attempted_overall > 0
        else 0.0
    )
    trade_stats["overall"]["atr_stop_trigger_count"] = int(
        (atr_stop_stats or {}).get("trigger_count", 0)
    )
    trade_stats["overall"]["r_take_profit_trigger_count"] = int(
        (r_take_profit_stats or {}).get("trigger_count", 0)
    )
    trade_stats["overall"]["bias_v_take_profit_trigger_count"] = int(
        (bias_v_take_profit_stats or {}).get("trigger_count", 0)
    )
    trade_stats["overall"]["r_take_profit_tier_trigger_counts"] = single_rtp_tier_counts
    trade_stats["overall"]["er_filter_blocked_entry_count"] = int(
        (er_filter_stats_overall or {}).get("blocked_entry_count", 0)
    )
    trade_stats["overall"]["er_filter_attempted_entry_count"] = int(
        (er_filter_stats_overall or {}).get("attempted_entry_count", 0)
    )
    trade_stats["overall"]["er_filter_allowed_entry_count"] = int(
        (er_filter_stats_overall or {}).get("allowed_entry_count", 0)
    )
    trade_stats["overall"]["impulse_filter_blocked_entry_count"] = int(
        (impulse_filter_stats_overall or {}).get("blocked_entry_count", 0)
    )
    trade_stats["overall"]["impulse_filter_attempted_entry_count"] = (
        impulse_attempted_overall
    )
    trade_stats["overall"]["impulse_filter_allowed_entry_count"] = int(
        (impulse_filter_stats_overall or {}).get("allowed_entry_count", 0)
    )
    trade_stats["overall"]["impulse_filter_blocked_entry_rate"] = float(
        impulse_block_rate_overall
    )
    trade_stats["overall"]["impulse_filter_blocked_entry_count_bull"] = int(
        (impulse_filter_stats_overall or {}).get("blocked_entry_count_bull", 0)
    )
    trade_stats["overall"]["impulse_filter_blocked_entry_count_bear"] = int(
        (impulse_filter_stats_overall or {}).get("blocked_entry_count_bear", 0)
    )
    trade_stats["overall"]["impulse_filter_blocked_entry_count_neutral"] = int(
        (impulse_filter_stats_overall or {}).get("blocked_entry_count_neutral", 0)
    )
    trade_stats["overall"]["er_exit_filter_trigger_count"] = int(
        (er_exit_filter_stats_overall or {}).get("trigger_count", 0)
    )
    trade_stats["overall"]["vol_risk_adjust_total_count"] = int(
        (vol_risk_adjust_stats_overall or {}).get("vol_risk_adjust_total_count", 0)
    )
    trade_stats["overall"]["vol_risk_adjust_reduce_on_expand_count"] = int(
        (vol_risk_adjust_stats_overall or {}).get(
            "vol_risk_adjust_reduce_on_expand_count", 0
        )
    )
    trade_stats["overall"]["vol_risk_adjust_increase_on_contract_count"] = int(
        (vol_risk_adjust_stats_overall or {}).get(
            "vol_risk_adjust_increase_on_contract_count", 0
        )
    )
    trade_stats["overall"]["vol_risk_adjust_recover_from_expand_count"] = int(
        (vol_risk_adjust_stats_overall or {}).get(
            "vol_risk_adjust_recover_from_expand_count", 0
        )
    )
    trade_stats["overall"]["vol_risk_adjust_recover_from_contract_count"] = int(
        (vol_risk_adjust_stats_overall or {}).get(
            "vol_risk_adjust_recover_from_contract_count", 0
        )
    )
    trade_stats["overall"]["vol_risk_entry_state_reduce_on_expand_count"] = int(
        (vol_risk_adjust_stats_overall or {}).get(
            "vol_risk_entry_state_reduce_on_expand_count", 0
        )
    )
    trade_stats["overall"]["vol_risk_entry_state_increase_on_contract_count"] = int(
        (vol_risk_adjust_stats_overall or {}).get(
            "vol_risk_entry_state_increase_on_contract_count", 0
        )
    )
    trade_stats["overall"]["monthly_risk_budget_attempted_entry_count"] = (
        monthly_attempted_overall
    )
    trade_stats["overall"]["monthly_risk_budget_blocked_entry_count"] = (
        monthly_blocked_overall
    )
    trade_stats["overall"]["monthly_risk_budget_blocked_entry_rate"] = float(
        monthly_block_rate_overall
    )
    trade_stats["by_code"][str(code)]["atr_stop_trigger_count"] = int(
        (atr_stop_stats or {}).get("trigger_count", 0)
    )
    trade_stats["by_code"][str(code)]["r_take_profit_trigger_count"] = int(
        (r_take_profit_stats or {}).get("trigger_count", 0)
    )
    trade_stats["by_code"][str(code)]["bias_v_take_profit_trigger_count"] = int(
        (bias_v_take_profit_stats or {}).get("trigger_count", 0)
    )
    trade_stats["by_code"][str(code)]["r_take_profit_tier_trigger_counts"] = (
        single_rtp_tier_counts
    )
    trade_stats["by_code"][str(code)]["er_filter_blocked_entry_count"] = int(
        (er_filter_stats_overall or {}).get("blocked_entry_count", 0)
    )
    trade_stats["by_code"][str(code)]["er_filter_attempted_entry_count"] = int(
        (er_filter_stats_overall or {}).get("attempted_entry_count", 0)
    )
    trade_stats["by_code"][str(code)]["er_filter_allowed_entry_count"] = int(
        (er_filter_stats_overall or {}).get("allowed_entry_count", 0)
    )
    trade_stats["by_code"][str(code)]["impulse_filter_blocked_entry_count"] = int(
        (impulse_filter_stats_overall or {}).get("blocked_entry_count", 0)
    )
    trade_stats["by_code"][str(code)]["impulse_filter_attempted_entry_count"] = (
        impulse_attempted_overall
    )
    trade_stats["by_code"][str(code)]["impulse_filter_allowed_entry_count"] = int(
        (impulse_filter_stats_overall or {}).get("allowed_entry_count", 0)
    )
    trade_stats["by_code"][str(code)]["impulse_filter_blocked_entry_rate"] = float(
        impulse_block_rate_overall
    )
    trade_stats["by_code"][str(code)]["impulse_filter_blocked_entry_count_bull"] = int(
        (impulse_filter_stats_overall or {}).get("blocked_entry_count_bull", 0)
    )
    trade_stats["by_code"][str(code)]["impulse_filter_blocked_entry_count_bear"] = int(
        (impulse_filter_stats_overall or {}).get("blocked_entry_count_bear", 0)
    )
    trade_stats["by_code"][str(code)]["impulse_filter_blocked_entry_count_neutral"] = (
        int((impulse_filter_stats_overall or {}).get("blocked_entry_count_neutral", 0))
    )
    trade_stats["by_code"][str(code)]["er_exit_filter_trigger_count"] = int(
        (er_exit_filter_stats_overall or {}).get("trigger_count", 0)
    )
    trade_stats["by_code"][str(code)]["vol_risk_adjust_total_count"] = int(
        (vol_risk_adjust_stats_overall or {}).get("vol_risk_adjust_total_count", 0)
    )
    trade_stats["by_code"][str(code)]["vol_risk_adjust_reduce_on_expand_count"] = int(
        (vol_risk_adjust_stats_overall or {}).get(
            "vol_risk_adjust_reduce_on_expand_count", 0
        )
    )
    trade_stats["by_code"][str(code)]["vol_risk_adjust_increase_on_contract_count"] = (
        int(
            (vol_risk_adjust_stats_overall or {}).get(
                "vol_risk_adjust_increase_on_contract_count", 0
            )
        )
    )
    trade_stats["by_code"][str(code)]["vol_risk_adjust_recover_from_expand_count"] = (
        int(
            (vol_risk_adjust_stats_overall or {}).get(
                "vol_risk_adjust_recover_from_expand_count", 0
            )
        )
    )
    trade_stats["by_code"][str(code)]["vol_risk_adjust_recover_from_contract_count"] = (
        int(
            (vol_risk_adjust_stats_overall or {}).get(
                "vol_risk_adjust_recover_from_contract_count", 0
            )
        )
    )
    trade_stats["by_code"][str(code)]["vol_risk_entry_state_reduce_on_expand_count"] = (
        int(
            (vol_risk_adjust_stats_overall or {}).get(
                "vol_risk_entry_state_reduce_on_expand_count", 0
            )
        )
    )
    trade_stats["by_code"][str(code)][
        "vol_risk_entry_state_increase_on_contract_count"
    ] = int(
        (vol_risk_adjust_stats_overall or {}).get(
            "vol_risk_entry_state_increase_on_contract_count", 0
        )
    )
    trade_stats["by_code"][str(code)]["monthly_risk_budget_attempted_entry_count"] = (
        monthly_attempted_overall
    )
    trade_stats["by_code"][str(code)]["monthly_risk_budget_blocked_entry_count"] = (
        monthly_blocked_overall
    )
    trade_stats["by_code"][str(code)]["monthly_risk_budget_blocked_entry_rate"] = float(
        monthly_block_rate_overall
    )
    if not quick_mode:
        trade_stats["entry_condition_stats"] = {
            "scope": "closed_trades_only",
            "signal_day_basis": "signal_day_before_entry_execution",
            "quasi_causal_method": "uplift + two_proportion_z / welch_t_normal_approx + BH",
            "strong_causal_method": "uplift + stratified_permutation + BH",
            "overall": _build_entry_condition_stats(
                trades_with_r, by_code=False, n_perm=300, seed=20260410
            ),
            "by_code": {
                str(code): _build_entry_condition_stats(
                    trades_with_r, by_code=True, n_perm=200, seed=20260410
                )
            },
        }
    m_strat["r_take_profit_tier_trigger_counts"] = single_rtp_tier_counts
    m_strat["impulse_filter_blocked_entry_count"] = int(
        (impulse_filter_stats_overall or {}).get("blocked_entry_count", 0)
    )
    m_strat["impulse_filter_blocked_entry_count_bull"] = int(
        (impulse_filter_stats_overall or {}).get("blocked_entry_count_bull", 0)
    )
    m_strat["impulse_filter_blocked_entry_count_bear"] = int(
        (impulse_filter_stats_overall or {}).get("blocked_entry_count_bear", 0)
    )
    m_strat["impulse_filter_blocked_entry_count_neutral"] = int(
        (impulse_filter_stats_overall or {}).get("blocked_entry_count_neutral", 0)
    )
    m_strat["monthly_risk_budget_blocked_entry_count"] = int(
        (monthly_risk_budget_gate_stats or {}).get("blocked_entry_count", 0)
    )
    event_study = None
    if not quick_mode:
        event_study = compute_event_study(
            dates=nav.index,
            daily_returns=strat_ret.reindex(nav.index).astype(float),
            entry_dates=entry_dates_from_exposure(w.reindex(nav.index).astype(float)),
        )
    market_regime = build_market_regime_report(
        close=px_sig.to_frame(code).reindex(nav.index).astype(float),
        high=high_qfq.to_frame(code).reindex(nav.index).astype(float),
        low=low_qfq.to_frame(code).reindex(nav.index).astype(float),
        weights=w.to_frame(code).reindex(nav.index).astype(float),
        asset_returns=ret_exec_day.to_frame(code)
        .reindex(nav.index)
        .astype(float)
        .fillna(0.0),
        strategy_returns=strat_ret.reindex(nav.index).astype(float),
        ann_factor=TRADING_DAYS_PER_YEAR,
    )
    weekly = _period_returns(nav, "W-FRI")
    monthly = _period_returns(nav, "ME")
    quarterly = _period_returns(nav, "QE")
    yearly = _period_returns(nav, "YE")
    rolling_out = _rolling_pack(nav)
    entry_exec_price_with_slippage = _latest_entry_exec_price_with_slippage(
        effective_weight=w.reindex(nav.index).astype(float),
        exec_price_series=px_exec_slip.reindex(nav.index).ffill().astype(float),
        slippage_spread=float(inp.slippage_rate),
    )

    out = {
        "meta": {
            "type": "trend_backtest",
            "code": code,
            "start": inp.start.strftime("%Y%m%d"),
            "end": inp.end.strftime("%Y%m%d"),
            "strategy": strat,
            "strategy_execution_description": TREND_STRATEGY_EXECUTION_DESCRIPTIONS.get(
                strat, ""
            ),
            "price_basis": {
                "signal": "qfq close",
                "strategy_nav": "none close preferred; hfq return fallback on corporate-action days",
                "benchmark_nav": {
                    "close": "HFQ close-to-close daily returns (BUY_HOLD line; excess vs strategy uses this series)",
                    "open": "same-day open→close (none; hfq on corporate-action days); BUY_HOLD aligned to open execution",
                    "oc2": "50% same-day open→close + 50% HFQ close-to-close next day; BUY_HOLD aligned to OC2 execution",
                }.get(ep, "unknown exec_price"),
            },
            "params": {
                "sma_window": int(inp.sma_window),
                "fast_window": int(inp.fast_window),
                "slow_window": int(inp.slow_window),
                "ma_type": ma_type,
                "kama_er_window": int(kama_er_window),
                "kama_fast_window": int(kama_fast_window),
                "kama_slow_window": int(kama_slow_window),
                "kama_std_window": int(kama_std_window),
                "kama_std_coef": float(kama_std_coef),
                "donchian_entry": int(inp.donchian_entry),
                "donchian_exit": int(inp.donchian_exit),
                "mom_lookback": int(inp.mom_lookback),
                "tsmom_entry_threshold": float(inp.tsmom_entry_threshold),
                "tsmom_exit_threshold": float(inp.tsmom_exit_threshold),
                "atr_stop_mode": str(atr_mode),
                "atr_stop_atr_basis": str(atr_basis),
                "atr_stop_reentry_mode": str(atr_reentry_mode),
                "atr_stop_window": int(inp.atr_stop_window),
                "atr_stop_n": float(inp.atr_stop_n),
                "atr_stop_m": float(inp.atr_stop_m),
                "r_take_profit_enabled": bool(rtp_enabled),
                "r_take_profit_reentry_mode": str(rtp_reentry_mode),
                "r_take_profit_tiers": rtp_tiers,
                "bias_v_take_profit_enabled": bool(bias_v_tp_enabled),
                "bias_v_take_profit_reentry_mode": str(bias_v_tp_reentry_mode),
                "bias_v_ma_window": int(bias_v_tp_ma_window),
                "bias_v_atr_window": int(bias_v_tp_atr_window),
                "bias_v_take_profit_threshold": float(bias_v_tp_threshold),
                "monthly_risk_budget_enabled": bool(monthly_risk_budget_enabled),
                "monthly_risk_budget_pct": float(monthly_risk_budget_pct),
                "monthly_risk_budget_include_new_trade_risk": bool(
                    monthly_risk_budget_include_new_trade_risk
                ),
                "bias_ma_window": int(inp.bias_ma_window),
                "bias_entry": float(inp.bias_entry),
                "bias_hot": float(inp.bias_hot),
                "bias_cold": float(inp.bias_cold),
                "bias_pos_mode": bias_mode,
                "macd_fast": int(inp.macd_fast),
                "macd_slow": int(inp.macd_slow),
                "macd_signal": int(inp.macd_signal),
                "macd_v_atr_window": int(inp.macd_v_atr_window),
                "macd_v_scale": float(inp.macd_v_scale),
                "random_hold_days": int(getattr(inp, "random_hold_days", 20)),
                "random_seed": (
                    None
                    if getattr(inp, "random_seed", 42) is None
                    else int(getattr(inp, "random_seed", 42))
                ),
                "position_sizing": str(ps),
                "vol_window": int(inp.vol_window),
                "vol_target_ann": float(inp.vol_target_ann),
                "fixed_pos_ratio": float(fixed_ratio),
                "fixed_overcap_policy": str(fixed_overcap_policy),
                "fixed_max_holdings": int(fixed_max_holding_n),
                "risk_budget_atr_window": int(risk_budget_atr_window),
                "risk_budget_pct": float(risk_budget_pct),
                "risk_budget_overcap_policy": str(risk_budget_overcap_policy),
                "risk_budget_max_leverage_multiple": float(
                    risk_budget_max_leverage_multiple
                ),
                "vol_regime_risk_mgmt_enabled": bool(vol_regime_risk_mgmt_enabled),
                "vol_ratio_fast_atr_window": int(vol_ratio_fast_atr_window),
                "vol_ratio_slow_atr_window": int(vol_ratio_slow_atr_window),
                "vol_ratio_expand_threshold": float(vol_ratio_expand_threshold),
                "vol_ratio_contract_threshold": float(vol_ratio_contract_threshold),
                "vol_ratio_normal_threshold": float(vol_ratio_normal_threshold),
                "er_filter": bool(er_filter),
                "er_window": int(er_window),
                "er_threshold": float(er_threshold),
                "impulse_entry_filter": bool(impulse_entry_filter),
                "impulse_allow_bull": bool(impulse_allow_bull),
                "impulse_allow_bear": bool(impulse_allow_bear),
                "impulse_allow_neutral": bool(impulse_allow_neutral),
                "er_exit_filter": bool(er_exit_filter),
                "er_exit_window": int(er_exit_window),
                "er_exit_threshold": float(er_exit_threshold),
                "quick_mode": bool(quick_mode),
                "exec_price": str(ep),
                "cost_bps": float(inp.cost_bps),
                "slippage_rate": float(inp.slippage_rate),
                "risk_free_rate": float(inp.risk_free_rate),
            },
        },
        "nav": {
            "dates": nav.index.date.astype(str).tolist(),
            "series": {
                "STRAT": nav.astype(float).tolist(),
                "BUY_HOLD": bh_nav.reindex(nav.index).astype(float).tolist(),
                "EXCESS": excess_nav.reindex(nav.index).astype(float).tolist(),
            },
        },
        "signals": {
            "base_position": base_pos.reindex(nav.index).astype(float).tolist(),
            "position": raw_pos.reindex(nav.index).astype(float).tolist(),
            "position_effective": w.reindex(nav.index).astype(float).tolist(),
        },
        "period_returns": {
            "weekly": weekly.to_dict(orient="records"),
            "monthly": monthly.to_dict(orient="records"),
            "quarterly": quarterly.to_dict(orient="records"),
            "yearly": yearly.to_dict(orient="records"),
        },
        "rolling": rolling_out,
        "attribution": attribution,
        "trade_statistics": trade_stats,
        "r_statistics": r_stats_out,
        "event_study": event_study,
        "market_regime": market_regime,
        "return_decomposition": return_decomposition,
        "metrics": {"strategy": m_strat, "benchmark": m_bh, "excess": m_ex},
        "risk_controls": {
            "atr_stop": atr_stop_stats,
            "r_take_profit": r_take_profit_stats,
            "bias_v_take_profit": bias_v_take_profit_stats,
            "er_exit_filter": {
                "enabled": bool(er_exit_filter),
                "window": int(er_exit_window),
                "threshold": float(er_exit_threshold),
                "trigger_count": int(
                    (er_exit_filter_stats_overall or {}).get("trigger_count", 0)
                ),
                "trigger_dates": list(
                    (er_exit_filter_stats_overall or {}).get("trigger_dates", [])
                )[:200],
                "trace_last_rows": list(
                    (er_exit_filter_stats_overall or {}).get("trace_last_rows", [])
                ),
            },
            "vol_regime_risk_mgmt": {
                "enabled": bool(ps == "risk_budget" and vol_regime_risk_mgmt_enabled),
                "fast_atr_window": int(vol_ratio_fast_atr_window),
                "slow_atr_window": int(vol_ratio_slow_atr_window),
                "expand_threshold": float(vol_ratio_expand_threshold),
                "contract_threshold": float(vol_ratio_contract_threshold),
                "normal_threshold": float(vol_ratio_normal_threshold),
                "adjust_total_count": int(
                    (vol_risk_adjust_stats_overall or {}).get(
                        "vol_risk_adjust_total_count", 0
                    )
                ),
                "adjust_reduce_on_expand_count": int(
                    (vol_risk_adjust_stats_overall or {}).get(
                        "vol_risk_adjust_reduce_on_expand_count", 0
                    )
                ),
                "adjust_increase_on_contract_count": int(
                    (vol_risk_adjust_stats_overall or {}).get(
                        "vol_risk_adjust_increase_on_contract_count", 0
                    )
                ),
                "adjust_recover_from_expand_count": int(
                    (vol_risk_adjust_stats_overall or {}).get(
                        "vol_risk_adjust_recover_from_expand_count", 0
                    )
                ),
                "adjust_recover_from_contract_count": int(
                    (vol_risk_adjust_stats_overall or {}).get(
                        "vol_risk_adjust_recover_from_contract_count", 0
                    )
                ),
                "entry_state_reduce_on_expand_count": int(
                    (vol_risk_adjust_stats_overall or {}).get(
                        "vol_risk_entry_state_reduce_on_expand_count", 0
                    )
                ),
                "entry_state_increase_on_contract_count": int(
                    (vol_risk_adjust_stats_overall or {}).get(
                        "vol_risk_entry_state_increase_on_contract_count", 0
                    )
                ),
            },
            "monthly_risk_budget": monthly_risk_budget_gate_stats,
        },
        "corporate_actions": (
            [
                {
                    "date": d.date().isoformat(),
                    "none_return": float(ret_none.loc[d]),
                    "hfq_return": float(ret_hfq.loc[d]),
                    "corp_factor": float(corp_factor.loc[d]),
                }
                for d in corp_factor.index[ca_mask.fillna(False)]
            ][:200]
        ),
        "next_plan": {
            "decision_date": (str(nav.index[-1].date()) if len(nav.index) else None),
            "current_effective_weight": (float(w.iloc[-1]) if len(w) else 0.0),
            "target_weight": (
                float(raw_pos.reindex(nav.index).iloc[-1]) if len(nav.index) else 0.0
            ),
            "entry_exec_price_with_slippage_by_asset": (
                {str(code): float(entry_exec_price_with_slippage)}
                if entry_exec_price_with_slippage is not None
                else {}
            ),
            "trace": {
                "strategy": str(strat),
                "atr_stop_mode": str(atr_mode),
                "atr_stop_atr_basis": str(atr_basis),
                "atr_stop_reentry_mode": str(atr_reentry_mode),
                "base_signal_today": (
                    float(base_pos.reindex(nav.index).iloc[-1])
                    if len(nav.index)
                    else 0.0
                ),
                "base_signal_prev": (
                    float(base_pos.reindex(nav.index).iloc[-2])
                    if len(nav.index) > 1
                    else 0.0
                ),
                "base_entry_event_today": bool(
                    (len(nav.index) > 0)
                    and (float(base_pos.reindex(nav.index).iloc[-1]) > 0.0)
                    and (
                        (
                            float(base_pos.reindex(nav.index).iloc[-2])
                            if len(nav.index) > 1
                            else 0.0
                        )
                        <= 0.0
                    )
                ),
                "atr_stop": {
                    "trigger_count": int(
                        (atr_stop_stats or {}).get("trigger_count", 0)
                    ),
                    "last_trigger_date": (atr_stop_stats or {}).get(
                        "last_trigger_date"
                    ),
                    "wait_next_entry_lock_active": bool(
                        (atr_stop_stats or {}).get("wait_next_entry_lock_active", False)
                    ),
                    "latest_stop_price": (atr_stop_stats or {}).get(
                        "latest_stop_price"
                    ),
                    "latest_stop_date": (atr_stop_stats or {}).get("latest_stop_date"),
                    "trace_last_rows": (atr_stop_stats or {}).get(
                        "trace_last_rows", []
                    ),
                },
                "r_take_profit": {
                    "enabled": bool((r_take_profit_stats or {}).get("enabled", False)),
                    "initial_r_mode": (r_take_profit_stats or {}).get("initial_r_mode"),
                    "fallback_mode_used": bool(
                        (r_take_profit_stats or {}).get("fallback_mode_used", False)
                    ),
                    "trigger_count": int(
                        (r_take_profit_stats or {}).get("trigger_count", 0)
                    ),
                    "last_trigger_date": (r_take_profit_stats or {}).get(
                        "last_trigger_date"
                    ),
                    "wait_next_entry_lock_active": bool(
                        (r_take_profit_stats or {}).get(
                            "wait_next_entry_lock_active", False
                        )
                    ),
                    "tiers": (r_take_profit_stats or {}).get("tiers", []),
                    "trace_last_rows": (r_take_profit_stats or {}).get(
                        "trace_last_rows", []
                    ),
                },
                "bias_v_take_profit": {
                    "enabled": bool(
                        (bias_v_take_profit_stats or {}).get("enabled", False)
                    ),
                    "reentry_mode": (bias_v_take_profit_stats or {}).get(
                        "reentry_mode"
                    ),
                    "ma_window": int(
                        (bias_v_take_profit_stats or {}).get(
                            "ma_window", bias_v_tp_ma_window
                        )
                    ),
                    "atr_window": int(
                        (bias_v_take_profit_stats or {}).get(
                            "atr_window", bias_v_tp_atr_window
                        )
                    ),
                    "threshold": float(
                        (bias_v_take_profit_stats or {}).get(
                            "threshold", bias_v_tp_threshold
                        )
                    ),
                    "trigger_count": int(
                        (bias_v_take_profit_stats or {}).get("trigger_count", 0)
                    ),
                    "last_trigger_date": (bias_v_take_profit_stats or {}).get(
                        "last_trigger_date"
                    ),
                    "wait_next_entry_lock_active": bool(
                        (bias_v_take_profit_stats or {}).get(
                            "wait_next_entry_lock_active", False
                        )
                    ),
                    "trace_last_rows": (bias_v_take_profit_stats or {}).get(
                        "trace_last_rows", []
                    ),
                },
                "er_exit_filter": {
                    "enabled": bool(er_exit_filter),
                    "window": int(er_exit_window),
                    "threshold": float(er_exit_threshold),
                    "trigger_count": int(
                        (er_exit_filter_stats_overall or {}).get("trigger_count", 0)
                    ),
                    "trigger_dates": list(
                        (er_exit_filter_stats_overall or {}).get("trigger_dates", [])
                    )[:200],
                    "trace_last_rows": list(
                        (er_exit_filter_stats_overall or {}).get("trace_last_rows", [])
                    ),
                },
            },
        },
    }
    return out


def compute_trend_portfolio_backtest(
    db: Session,
    inp: TrendPortfolioInputs,
    data_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run trend portfolio backtest. When data_override is provided, db is not used
    (can be None); used for OOS bootstrap with synthetic in-sample data.
    """
    codes = list(
        dict.fromkeys([str(c).strip() for c in (inp.codes or []) if str(c).strip()])
    )
    if not codes:
        raise ValueError("codes is empty")
    if float(inp.cost_bps) < 0:
        raise ValueError("cost_bps must be >= 0")
    if (not np.isfinite(float(inp.slippage_rate))) or float(inp.slippage_rate) < 0:
        raise ValueError("slippage_rate must be finite and >= 0")
    ep = str(getattr(inp, "exec_price", "open") or "open").strip().lower()
    if ep not in {"open", "close", "oc2"}:
        raise ValueError("exec_price must be one of: open|close|oc2")
    if not np.isfinite(float(inp.risk_free_rate)):
        raise ValueError("risk_free_rate must be finite")
    quick_mode = bool(getattr(inp, "quick_mode", False))
    ps = str(inp.position_sizing or "equal").strip().lower()
    if ps not in {"equal", "vol_target", "fixed_ratio", "risk_budget"}:
        raise ValueError(
            "position_sizing must be equal|vol_target|fixed_ratio|risk_budget"
        )
    if int(inp.vol_window) < 2:
        raise ValueError("vol_window must be >= 2")
    if (not np.isfinite(float(inp.vol_target_ann))) or float(inp.vol_target_ann) <= 0:
        raise ValueError("vol_target_ann must be finite and > 0")
    if (not np.isfinite(float(getattr(inp, "fixed_pos_ratio", 0.04)))) or float(
        getattr(inp, "fixed_pos_ratio", 0.04)
    ) <= 0:
        raise ValueError("fixed_pos_ratio must be finite and > 0")
    fixed_overcap_policy = (
        str(getattr(inp, "fixed_overcap_policy", "skip") or "skip").strip().lower()
    )
    if fixed_overcap_policy not in {"skip", "extend"}:
        raise ValueError("fixed_overcap_policy must be one of: skip|extend")
    fixed_max_holdings = int(getattr(inp, "fixed_max_holdings", 10) or 10)
    if fixed_max_holdings < 1:
        raise ValueError("fixed_max_holdings must be >= 1")
    risk_budget_atr_window = int(getattr(inp, "risk_budget_atr_window", 20) or 20)
    if risk_budget_atr_window < 2:
        raise ValueError("risk_budget_atr_window must be >= 2")
    risk_budget_pct = float(getattr(inp, "risk_budget_pct", 0.01) or 0.01)
    if (
        (not np.isfinite(risk_budget_pct))
        or risk_budget_pct < 0.001
        or risk_budget_pct > 0.02
    ):
        raise ValueError("risk_budget_pct must be in [0.001, 0.02]")
    risk_budget_overcap_policy = (
        str(getattr(inp, "risk_budget_overcap_policy", "scale") or "scale")
        .strip()
        .lower()
    )
    if risk_budget_overcap_policy not in {
        "scale",
        "skip_entry",
        "replace_entry",
        "leverage_entry",
    }:
        raise ValueError(
            "risk_budget_overcap_policy must be one of: scale|skip_entry|replace_entry|leverage_entry"
        )
    risk_budget_max_leverage_multiple = float(
        getattr(inp, "risk_budget_max_leverage_multiple", 2.0) or 2.0
    )
    if (
        (not np.isfinite(risk_budget_max_leverage_multiple))
        or risk_budget_max_leverage_multiple < 1.0
        or risk_budget_max_leverage_multiple > 10.0
    ):
        raise ValueError("risk_budget_max_leverage_multiple must be in [1.0, 10.0]")
    vol_regime_risk_mgmt_enabled = bool(
        getattr(inp, "vol_regime_risk_mgmt_enabled", False)
    )
    vol_ratio_fast_atr_window = int(getattr(inp, "vol_ratio_fast_atr_window", 5) or 5)
    vol_ratio_slow_atr_window = int(getattr(inp, "vol_ratio_slow_atr_window", 50) or 50)
    vol_ratio_expand_threshold = float(
        getattr(inp, "vol_ratio_expand_threshold", 1.45) or 1.45
    )
    vol_ratio_contract_threshold = float(
        getattr(inp, "vol_ratio_contract_threshold", 0.65) or 0.65
    )
    vol_ratio_normal_threshold = float(
        getattr(inp, "vol_ratio_normal_threshold", 1.05) or 1.05
    )
    if vol_ratio_fast_atr_window < 2:
        raise ValueError("vol_ratio_fast_atr_window must be >= 2")
    if vol_ratio_slow_atr_window < 2:
        raise ValueError("vol_ratio_slow_atr_window must be >= 2")
    if (not np.isfinite(vol_ratio_expand_threshold)) or vol_ratio_expand_threshold <= 0:
        raise ValueError("vol_ratio_expand_threshold must be > 0")
    if (
        not np.isfinite(vol_ratio_contract_threshold)
    ) or vol_ratio_contract_threshold <= 0:
        raise ValueError("vol_ratio_contract_threshold must be > 0")
    if (not np.isfinite(vol_ratio_normal_threshold)) or vol_ratio_normal_threshold <= 0:
        raise ValueError("vol_ratio_normal_threshold must be > 0")
    if vol_ratio_expand_threshold <= vol_ratio_normal_threshold:
        raise ValueError(
            "vol_ratio_expand_threshold must be > vol_ratio_normal_threshold"
        )
    if vol_ratio_contract_threshold >= vol_ratio_normal_threshold:
        raise ValueError(
            "vol_ratio_contract_threshold must be < vol_ratio_normal_threshold"
        )
    strat = str(inp.strategy or "ma_filter").strip().lower()
    ma_type = str(getattr(inp, "ma_type", "sma") or "sma").strip().lower()
    if strat == "ma_cross":
        if ma_type not in {"sma", "ema", "wma"}:
            raise ValueError("ma_type must be one of: sma|ema|wma for ma_cross")
    elif strat == "ma_filter":
        if ma_type not in {"sma", "ema", "kama"}:
            raise ValueError("ma_type must be one of: sma|ema|kama for ma_filter")
    kama_er_window = int(getattr(inp, "kama_er_window", 10) or 10)
    kama_fast_window = int(getattr(inp, "kama_fast_window", 2) or 2)
    kama_slow_window = int(getattr(inp, "kama_slow_window", 30) or 30)
    kama_std_window = int(getattr(inp, "kama_std_window", 20) or 20)
    kama_std_coef = float(getattr(inp, "kama_std_coef", 1.0) or 0.0)
    if kama_er_window < 2:
        raise ValueError("kama_er_window must be >= 2")
    if kama_fast_window < 1:
        raise ValueError("kama_fast_window must be >= 1")
    if kama_slow_window < 2:
        raise ValueError("kama_slow_window must be >= 2")
    if kama_fast_window >= kama_slow_window:
        raise ValueError("kama_fast_window must be < kama_slow_window")
    if kama_std_window < 2:
        raise ValueError("kama_std_window must be >= 2")
    if (not np.isfinite(kama_std_coef)) or kama_std_coef < 0.0 or kama_std_coef > 3.0:
        raise ValueError("kama_std_coef must be in [0,3]")
    if strat == "ma_cross" and ma_type == "kama":
        raise ValueError("ma_type=kama is only supported for ma_filter")
    atr_mode = str(getattr(inp, "atr_stop_mode", "none") or "none").strip().lower()
    if atr_mode not in {"none", "static", "trailing", "tightening"}:
        raise ValueError(
            "atr_stop_mode must be one of: none|static|trailing|tightening"
        )
    atr_basis = (
        str(getattr(inp, "atr_stop_atr_basis", "latest") or "latest").strip().lower()
    )
    if atr_basis not in {"entry", "latest"}:
        raise ValueError("atr_stop_atr_basis must be one of: entry|latest")
    atr_reentry_mode = (
        str(getattr(inp, "atr_stop_reentry_mode", "reenter") or "reenter")
        .strip()
        .lower()
    )
    if atr_reentry_mode not in {"reenter", "wait_next_entry"}:
        raise ValueError(
            "atr_stop_reentry_mode must be one of: reenter|wait_next_entry"
        )
    if int(inp.atr_stop_window) < 2:
        raise ValueError("atr_stop_window must be >= 2")
    if (not np.isfinite(float(inp.atr_stop_n))) or float(inp.atr_stop_n) <= 0:
        raise ValueError("atr_stop_n must be finite and > 0")
    if (not np.isfinite(float(inp.atr_stop_m))) or float(inp.atr_stop_m) <= 0:
        raise ValueError("atr_stop_m must be finite and > 0")
    if atr_mode == "tightening" and float(inp.atr_stop_n) <= float(inp.atr_stop_m):
        raise ValueError(
            "atr_stop_n must be > atr_stop_m when atr_stop_mode=tightening"
        )
    rtp_enabled = bool(getattr(inp, "r_take_profit_enabled", False))
    rtp_reentry_mode = (
        str(getattr(inp, "r_take_profit_reentry_mode", "reenter") or "reenter")
        .strip()
        .lower()
    )
    if rtp_reentry_mode not in {"reenter", "wait_next_entry"}:
        raise ValueError(
            "r_take_profit_reentry_mode must be one of: reenter|wait_next_entry"
        )
    rtp_tiers = _normalize_r_take_profit_tiers(
        getattr(inp, "r_take_profit_tiers", None)
    )
    bias_v_tp_enabled = bool(getattr(inp, "bias_v_take_profit_enabled", False))
    bias_v_tp_reentry_mode = (
        str(getattr(inp, "bias_v_take_profit_reentry_mode", "reenter") or "reenter")
        .strip()
        .lower()
    )
    if bias_v_tp_reentry_mode not in {"reenter", "wait_next_entry"}:
        raise ValueError(
            "bias_v_take_profit_reentry_mode must be one of: reenter|wait_next_entry"
        )
    bias_v_tp_ma_window = int(getattr(inp, "bias_v_ma_window", 20) or 20)
    if bias_v_tp_ma_window < 2:
        raise ValueError("bias_v_ma_window must be >= 2")
    bias_v_tp_atr_window = int(getattr(inp, "bias_v_atr_window", 20) or 20)
    if bias_v_tp_atr_window < 2:
        raise ValueError("bias_v_atr_window must be >= 2")
    bias_v_tp_threshold = float(
        getattr(inp, "bias_v_take_profit_threshold", 5.0) or 5.0
    )
    if (not np.isfinite(bias_v_tp_threshold)) or bias_v_tp_threshold <= 0.0:
        raise ValueError("bias_v_take_profit_threshold must be finite and > 0")
    monthly_risk_budget_enabled = bool(
        getattr(inp, "monthly_risk_budget_enabled", False)
    )
    monthly_risk_budget_pct = float(
        getattr(inp, "monthly_risk_budget_pct", 0.06) or 0.06
    )
    monthly_risk_budget_include_new_trade_risk = bool(
        getattr(inp, "monthly_risk_budget_include_new_trade_risk", False)
    )
    if (
        (not np.isfinite(monthly_risk_budget_pct))
        or monthly_risk_budget_pct < 0.01
        or monthly_risk_budget_pct > 0.06
    ):
        raise ValueError("monthly_risk_budget_pct must be in [0.01, 0.06]")
    if not np.isfinite(float(inp.tsmom_entry_threshold)):
        raise ValueError("tsmom_entry_threshold must be finite")
    if not np.isfinite(float(inp.tsmom_exit_threshold)):
        raise ValueError("tsmom_exit_threshold must be finite")
    if float(inp.tsmom_entry_threshold) < float(inp.tsmom_exit_threshold):
        raise ValueError("tsmom thresholds must satisfy: entry >= exit")
    er_filter = bool(getattr(inp, "er_filter", False))
    er_window = int(getattr(inp, "er_window", 10) or 10)
    er_threshold = float(getattr(inp, "er_threshold", 0.30) or 0.30)
    if er_window < 2:
        raise ValueError("er_window must be >= 2")
    if (not np.isfinite(er_threshold)) or er_threshold < 0.0 or er_threshold > 1.0:
        raise ValueError("er_threshold must be in [0,1]")
    impulse_entry_filter = bool(getattr(inp, "impulse_entry_filter", False))
    impulse_allow_bull = bool(getattr(inp, "impulse_allow_bull", True))
    impulse_allow_bear = bool(getattr(inp, "impulse_allow_bear", False))
    impulse_allow_neutral = bool(getattr(inp, "impulse_allow_neutral", False))
    er_exit_filter = bool(getattr(inp, "er_exit_filter", False))
    er_exit_window = int(getattr(inp, "er_exit_window", 10) or 10)
    er_exit_threshold = float(getattr(inp, "er_exit_threshold", 0.88) or 0.88)
    if er_exit_window < 2:
        raise ValueError("er_exit_window must be >= 2")
    if (
        (not np.isfinite(er_exit_threshold))
        or er_exit_threshold < 0.0
        or er_exit_threshold > 1.0
    ):
        raise ValueError("er_exit_threshold must be in [0,1]")
    group_enforce = bool(getattr(inp, "group_enforce", False))
    group_pick_policy = (
        str(getattr(inp, "group_pick_policy", "highest_sharpe") or "highest_sharpe")
        .strip()
        .lower()
    )
    if group_pick_policy not in {"earliest_entry", "highest_sharpe"}:
        raise ValueError(
            "group_pick_policy must be one of: earliest_entry|highest_sharpe"
        )
    group_max_holdings = int(getattr(inp, "group_max_holdings", 4) or 4)
    if group_max_holdings < 1 or group_max_holdings > 10:
        raise ValueError("group_max_holdings must be in [1,10]")
    group_map = {
        str(k).strip(): str(v).strip()
        for k, v in ((getattr(inp, "asset_groups", None) or {}).items())
        if str(k).strip()
    }

    if strat not in {
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
    }:
        raise ValueError(f"invalid strategy={inp.strategy}")
    if int(getattr(inp, "random_hold_days", 20)) < 1:
        raise ValueError("random_hold_days must be >= 1")

    if data_override is not None:
        dates = data_override["dates"]
        close_qfq = (
            data_override["close_qfq"].reindex(columns=codes).reindex(dates).ffill()
        )
        close_hfq = (
            data_override["close_hfq"].reindex(columns=codes).reindex(dates).ffill()
        )
        open_qfq_df = close_qfq.copy()
        high_qfq_df = data_override.get("high_qfq_df")
        if high_qfq_df is None or high_qfq_df.empty:
            high_qfq_df = pd.DataFrame(index=dates, columns=codes)
        else:
            high_qfq_df = high_qfq_df.reindex(columns=codes).reindex(dates).ffill()
        low_qfq_df = data_override.get("low_qfq_df")
        if low_qfq_df is None or low_qfq_df.empty:
            low_qfq_df = pd.DataFrame(index=dates, columns=codes)
        else:
            low_qfq_df = low_qfq_df.reindex(columns=codes).reindex(dates).ffill()
        ret_exec = (
            data_override["ret_exec"]
            .reindex(columns=codes)
            .reindex(dates)
            .fillna(0.0)
            .astype(float)
        )
        ret_hfq = (
            data_override["ret_hfq"]
            .reindex(columns=codes)
            .reindex(dates)
            .fillna(0.0)
            .astype(float)
        )
        px_exec_slip = pd.DataFrame(1.0, index=dates, columns=codes, dtype=float)
        du_b = bool(getattr(inp, "dynamic_universe", False))
        ch_nb = (
            data_override["close_hfq"]
            .reindex(columns=codes)
            .reindex(dates)
            .astype(float)
        )
        if not du_b:
            ch_nb = ch_nb.ffill()
        if len(codes) == 1:
            bench_ret = hfq_close_buy_hold_returns(ch_nb.iloc[:, 0])
        else:
            bench_ret = hfq_close_daily_equal_weight_returns(
                ch_nb, dynamic_universe=du_b
            )
    else:
        if db is None:
            raise ValueError("db is required when data_override is not set")
        need_hist = (
            max(
                int(inp.sma_window),
                int(inp.slow_window),
                int(inp.donchian_entry),
                int(inp.mom_lookback),
                int(inp.macd_slow),
                int(inp.macd_v_atr_window),
                20,
            )
            + 60
        )
        ext_start = inp.start - dt.timedelta(days=int(need_hist) * 2)
        close_none = (
            load_close_prices(
                db, codes=codes, start=inp.start, end=inp.end, adjust="none"
            )
            .sort_index()
            .ffill()
        )
        close_qfq = load_close_prices(
            db, codes=codes, start=ext_start, end=inp.end, adjust="qfq"
        ).sort_index()
        close_hfq = load_close_prices(
            db, codes=codes, start=ext_start, end=inp.end, adjust="hfq"
        ).sort_index()
        if close_none.empty:
            raise ValueError("no execution price data for given range (none)")
        if not bool(getattr(inp, "dynamic_universe", False)):
            miss = [
                c
                for c in codes
                if c not in close_none.columns or close_none[c].dropna().empty
            ]
            if miss:
                raise ValueError(f"missing execution data (none) for: {miss}")
            first_valid = [
                close_none[c].first_valid_index()
                for c in codes
                if close_none[c].first_valid_index() is not None
            ]
            if not first_valid:
                raise ValueError("no valid first trading date for selected codes")
            common_start = max(first_valid)
            close_none = close_none.loc[common_start:]
            close_qfq = close_qfq.loc[common_start:]
            close_hfq = close_hfq.loc[common_start:]
        dates = close_none.index
        du_b = bool(getattr(inp, "dynamic_universe", False))
        ch_nb = close_hfq.reindex(dates).reindex(columns=codes).astype(float)
        if not du_b:
            ch_nb = ch_nb.ffill()
        if len(codes) == 1:
            bench_ret = hfq_close_buy_hold_returns(ch_nb.iloc[:, 0])
        else:
            bench_ret = hfq_close_daily_equal_weight_returns(
                ch_nb, dynamic_universe=du_b
            )
        close_qfq = close_qfq.reindex(dates).ffill()
        close_hfq = close_hfq.reindex(dates).ffill()
        high_qfq_df, low_qfq_df = load_high_low_prices(
            db, codes=codes, start=ext_start, end=inp.end, adjust="qfq"
        )
        high_qfq_df = (
            high_qfq_df.sort_index().reindex(dates).ffill()
            if not high_qfq_df.empty
            else pd.DataFrame(index=dates, columns=codes)
        )
        low_qfq_df = (
            low_qfq_df.sort_index().reindex(dates).ffill()
            if not low_qfq_df.empty
            else pd.DataFrame(index=dates, columns=codes)
        )
        ohlc_qfq = load_ohlc_prices(
            db, codes=codes, start=inp.start, end=inp.end, adjust="qfq"
        )
        open_qfq_df = (
            ohlc_qfq.get("open", pd.DataFrame())
            .sort_index()
            .reindex(dates)
            .reindex(columns=codes)
            .ffill()
        )

        ohlc_none = load_ohlc_prices(
            db, codes=codes, start=inp.start, end=inp.end, adjust="none"
        )
        ohlc_hfq = load_ohlc_prices(
            db, codes=codes, start=inp.start, end=inp.end, adjust="hfq"
        )
        open_none = (
            ohlc_none.get("open", pd.DataFrame())
            .sort_index()
            .reindex(dates)
            .reindex(columns=codes)
            .ffill()
        )
        close_none_exec = (
            ohlc_none.get("close", pd.DataFrame())
            .sort_index()
            .reindex(dates)
            .reindex(columns=codes)
            .ffill()
        )
        open_hfq = (
            ohlc_hfq.get("open", pd.DataFrame())
            .sort_index()
            .reindex(dates)
            .reindex(columns=codes)
            .ffill()
        )
        close_hfq_exec = (
            ohlc_hfq.get("close", pd.DataFrame())
            .sort_index()
            .reindex(dates)
            .reindex(columns=codes)
            .ffill()
        )
        close_none_f = close_none.reindex(columns=codes).astype(float)
        close_hfq_f = close_hfq.reindex(columns=codes).astype(float)
        if ep == "open":
            exec_o_none = open_none.astype(float).combine_first(close_none_f)
            exec_c_none = close_none_exec.astype(float).combine_first(close_none_f)
            exec_o_hfq = open_hfq.astype(float).combine_first(close_hfq_f)
            exec_c_hfq = close_hfq_exec.astype(float).combine_first(close_hfq_f)
            ret_none = (
                (exec_o_none.shift(-1).div(exec_o_none) - 1.0)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype(float)
            )
            ret_hfq_exec = (
                (exec_o_hfq.shift(-1).div(exec_o_hfq) - 1.0)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype(float)
            )
            px_slip_none = exec_o_none.astype(float)
            px_slip_hfq = exec_o_hfq.astype(float)
        elif ep == "close":
            px_none_exec = close_none_exec.astype(float).combine_first(close_none_f)
            px_hfq_exec = close_hfq_exec.astype(float).combine_first(close_hfq_f)
            ret_none = (
                (px_none_exec.shift(-1).div(px_none_exec) - 1.0)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype(float)
            )
            ret_hfq_exec = (
                (px_hfq_exec.shift(-1).div(px_hfq_exec) - 1.0)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype(float)
            )
            px_slip_none = px_none_exec.astype(float)
            px_slip_hfq = px_hfq_exec.astype(float)
        else:
            exec_o_none = open_none.astype(float).combine_first(close_none_f)
            exec_c_none = close_none_exec.astype(float).combine_first(close_none_f)
            exec_o_hfq = open_hfq.astype(float).combine_first(close_hfq_f)
            exec_c_hfq = close_hfq_exec.astype(float).combine_first(close_hfq_f)
            ret_open_none = (
                (exec_o_none.shift(-1).div(exec_o_none) - 1.0)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype(float)
            )
            ret_close_none = (
                (exec_c_none.shift(-1).div(exec_c_none) - 1.0)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype(float)
            )
            ret_none = (0.5 * (ret_open_none + ret_close_none)).astype(float)
            ret_open_hfq = (
                (exec_o_hfq.shift(-1).div(exec_o_hfq) - 1.0)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype(float)
            )
            ret_close_hfq = (
                (exec_c_hfq.shift(-1).div(exec_c_hfq) - 1.0)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype(float)
            )
            ret_hfq_exec = (0.5 * (ret_open_hfq + ret_close_hfq)).astype(float)
            px_slip_none = (0.5 * (exec_o_none + exec_c_none)).astype(float)
            px_slip_hfq = (0.5 * (exec_o_hfq + exec_c_hfq)).astype(float)
        ret_hfq = ret_hfq_exec.astype(float)
        gross_none = 1.0 + ret_none
        gross_hfq = 1.0 + ret_hfq
        _corp_factor, ca_mask = corporate_action_mask(gross_none, gross_hfq)
        ret_exec = ret_none.where(~ca_mask.fillna(False), other=ret_hfq_exec).astype(
            float
        )
        px_exec_slip = (
            px_slip_none.where(~ca_mask.fillna(False), other=px_slip_hfq)
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
        )
        ret_overnight_none_close = (
            (open_none.shift(-1).div(close_none_exec) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_intraday_none_close = (
            (close_none_exec.shift(-1).div(open_none.shift(-1)) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_overnight_hfq_close = (
            (open_hfq.shift(-1).div(close_hfq_exec) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_intraday_hfq_close = (
            (close_hfq_exec.shift(-1).div(open_hfq.shift(-1)) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_intraday_none_open = (
            (close_none_exec.div(open_none) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_overnight_none_open = (
            (open_none.shift(-1).div(close_none_exec) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_intraday_hfq_open = (
            (close_hfq_exec.div(open_hfq) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_overnight_hfq_open = (
            (open_hfq.shift(-1).div(close_hfq_exec) - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        ret_overnight_close = ret_overnight_none_close.where(
            ~ca_mask.fillna(False), other=ret_overnight_hfq_close
        ).astype(float)
        ret_intraday_close = ret_intraday_none_close.where(
            ~ca_mask.fillna(False), other=ret_intraday_hfq_close
        ).astype(float)
        ret_overnight_open = ret_overnight_none_open.where(
            ~ca_mask.fillna(False), other=ret_overnight_hfq_open
        ).astype(float)
        ret_intraday_open = ret_intraday_none_open.where(
            ~ca_mask.fillna(False), other=ret_intraday_hfq_open
        ).astype(float)

    if data_override is not None:
        if ep == "open":
            ret_overnight = pd.DataFrame(0.0, index=dates, columns=codes, dtype=float)
            ret_intraday = (
                ret_exec.astype(float).reindex(index=dates, columns=codes).fillna(0.0)
            )
        elif ep == "close":
            ret_overnight = (
                ret_exec.astype(float).reindex(index=dates, columns=codes).fillna(0.0)
            )
            ret_intraday = pd.DataFrame(0.0, index=dates, columns=codes, dtype=float)
        else:
            ret_overnight = (
                0.5
                * ret_exec.astype(float).reindex(index=dates, columns=codes).fillna(0.0)
            ).astype(float)
            ret_intraday = (
                0.5
                * ret_exec.astype(float).reindex(index=dates, columns=codes).fillna(0.0)
            ).astype(float)
    else:
        if ep == "open":
            ret_overnight = (
                ret_overnight_open.astype(float)
                .reindex(index=dates, columns=codes)
                .fillna(0.0)
            )
            ret_intraday = (
                ret_intraday_open.astype(float)
                .reindex(index=dates, columns=codes)
                .fillna(0.0)
            )
        elif ep == "close":
            ret_overnight = (
                ret_overnight_close.astype(float)
                .reindex(index=dates, columns=codes)
                .fillna(0.0)
            )
            ret_intraday = (
                ret_intraday_close.astype(float)
                .reindex(index=dates, columns=codes)
                .fillna(0.0)
            )
        else:
            ret_overnight = (
                0.5
                * ret_overnight_close.astype(float)
                .reindex(index=dates, columns=codes)
                .fillna(0.0)
            ).astype(float)
            ret_intraday = (
                0.5
                * ret_exec.astype(float).reindex(index=dates, columns=codes).fillna(0.0)
                + 0.5
                * ret_intraday_close.astype(float)
                .reindex(index=dates, columns=codes)
                .fillna(0.0)
            ).astype(float)

    sig_pos = pd.DataFrame(index=dates, columns=codes, dtype=float)
    sig_score = pd.DataFrame(index=dates, columns=codes, dtype=float)
    cond_momentum_df = pd.DataFrame(index=dates, columns=codes, dtype=float)
    cond_er_df = pd.DataFrame(index=dates, columns=codes, dtype=float)
    cond_vol_ratio_df = pd.DataFrame(index=dates, columns=codes, dtype=float)
    cond_impulse_df = pd.DataFrame(index=dates, columns=codes, dtype=object)
    atr_stop_by_asset: dict[str, dict[str, Any]] = {}
    rtp_by_asset: dict[str, dict[str, Any]] = {}
    bias_v_tp_by_asset: dict[str, dict[str, Any]] = {}
    impulse_filter_by_asset: dict[str, dict[str, int]] = {}
    er_filter_by_asset: dict[str, dict[str, int]] = {}
    er_exit_filter_by_asset: dict[str, dict[str, Any]] = {}
    for c in codes:
        px = close_qfq[c].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
        if px.dropna().empty:
            sig_pos[c] = 0.0
            sig_score[c] = np.nan
            continue
        if strat == "ma_filter":
            ma = _moving_average(
                px,
                window=int(inp.sma_window),
                ma_type=ma_type,
                kama_er_window=kama_er_window,
                kama_fast_window=kama_fast_window,
                kama_slow_window=kama_slow_window,
            )
            if ma_type == "kama":
                kstd = (
                    ma.astype(float)
                    .rolling(
                        window=int(kama_std_window),
                        min_periods=max(2, int(kama_std_window) // 2),
                    )
                    .std(ddof=0)
                    .fillna(0.0)
                )
                pos = _pos_from_band(
                    px.astype(float),
                    ma.astype(float),
                    band=(float(kama_std_coef) * kstd),
                ).astype(float)
            else:
                pos = (px > ma).astype(float)
            score = (px / ma - 1.0).astype(float)
        elif strat == "ma_cross":
            fast = _moving_average(px, window=int(inp.fast_window), ma_type=ma_type)
            slow = _moving_average(px, window=int(inp.slow_window), ma_type=ma_type)
            pos = (fast > slow).astype(float)
            score = (fast / slow - 1.0).astype(float)
        elif strat == "donchian":
            pos = _pos_from_donchian(
                px, entry=int(inp.donchian_entry), exit_=int(inp.donchian_exit)
            )
            hi = (
                px.shift(1)
                .rolling(
                    window=max(2, int(inp.donchian_entry)),
                    min_periods=max(2, int(inp.donchian_entry)),
                )
                .max()
            )
            score = (px / hi - 1.0).astype(float)
        elif strat == "linreg_slope":
            n = int(inp.sma_window)
            y = np.log(px.clip(lower=1e-12).astype(float))
            slope = y.rolling(window=n, min_periods=max(2, n // 2)).apply(
                _rolling_linreg_slope, raw=True
            )
            pos = (slope > 0.0).astype(float)
            score = slope.astype(float)
        elif strat == "bias":
            b_win = int(inp.bias_ma_window)
            ema = _ema(px, b_win)
            bias = (
                np.log(px.clip(lower=1e-12)) - np.log(ema.clip(lower=1e-12))
            ) * 100.0
            entry = float(inp.bias_entry)
            hot = float(inp.bias_hot)
            cold = float(inp.bias_cold)
            pos_mode = (
                str(getattr(inp, "bias_pos_mode", "binary") or "binary").strip().lower()
            )
            if pos_mode not in {"binary", "continuous"}:
                pos_mode = "binary"
            pos_arr = np.zeros(len(px), dtype=float)
            in_pos = False
            for i in range(len(px)):
                b = (
                    float(bias.iloc[i])
                    if np.isfinite(float(bias.iloc[i]))
                    else float("nan")
                )
                if not np.isfinite(b):
                    in_pos = False
                    pos_arr[i] = 0.0
                    continue
                if not in_pos:
                    if b > entry:
                        in_pos = True
                elif (b >= hot) or (b <= cold):
                    in_pos = False
                if not in_pos:
                    pos_arr[i] = 0.0
                elif pos_mode == "binary":
                    pos_arr[i] = 1.0
                else:
                    wv = (b - cold) / (hot - cold)
                    pos_arr[i] = float(np.clip(wv, 0.0, 1.0))
            pos = pd.Series(pos_arr, index=px.index, dtype=float)
            score = bias.astype(float)
        elif strat == "macd_cross":
            macd, sig, _ = _macd_core(
                px,
                fast=int(inp.macd_fast),
                slow=int(inp.macd_slow),
                signal=int(inp.macd_signal),
            )
            pos = (macd > sig).astype(float)
            score = (macd - sig).astype(float)
        elif strat == "macd_zero_filter":
            macd, _, _ = _macd_core(
                px,
                fast=int(inp.macd_fast),
                slow=int(inp.macd_slow),
                signal=int(inp.macd_signal),
            )
            pos = (macd > 0.0).astype(float)
            score = macd.astype(float)
        elif strat == "macd_v":
            macd, _, _ = _macd_core(
                px,
                fast=int(inp.macd_fast),
                slow=int(inp.macd_slow),
                signal=int(inp.macd_signal),
            )
            h = high_qfq_df[c] if (c in high_qfq_df.columns) else px
            low_px = low_qfq_df[c] if (c in low_qfq_df.columns) else px
            atr = _atr_from_hlc(
                h.astype(float).fillna(px),
                low_px.astype(float).fillna(px),
                px,
                window=int(inp.macd_v_atr_window),
            )
            macd_v = (macd / atr.replace(0.0, np.nan)) * float(inp.macd_v_scale)
            macd_v_sig = _ema(macd_v, int(inp.macd_signal))
            pos = (macd_v > macd_v_sig).astype(float)
            score = (macd_v - macd_v_sig).astype(float)
        elif strat == "random_entry":
            seed_base_raw = getattr(inp, "random_seed", 42)
            code_seed = (
                None
                if seed_base_raw is None
                else (int(seed_base_raw) + _stable_code_seed(str(c))) % 2_147_483_647
            )
            pos = _pos_from_random_entry_hold(
                px.index,
                hold_days=int(getattr(inp, "random_hold_days", 20)),
                seed=code_seed,
            )
            score = pos.astype(float)
        else:
            mom = px / px.shift(int(inp.mom_lookback)) - 1.0
            pos = _pos_from_tsmom(
                mom,
                entry_threshold=float(inp.tsmom_entry_threshold),
                exit_threshold=float(inp.tsmom_exit_threshold),
            )
            score = mom.astype(float)
        impulse_state_one: pd.Series | None = _compute_impulse_state(
            px,
            ema_window=13,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
        )
        if impulse_entry_filter and (impulse_state_one is not None):
            pos, one_impulse_stats = _apply_impulse_entry_filter(
                pos.astype(float).fillna(0.0),
                impulse_state=impulse_state_one,
                allow_bull=impulse_allow_bull,
                allow_bear=impulse_allow_bear,
                allow_neutral=impulse_allow_neutral,
            )
            impulse_filter_by_asset[str(c)] = dict(one_impulse_stats or {})
        else:
            impulse_filter_by_asset[str(c)] = {
                "blocked_entry_count": 0,
                "attempted_entry_count": 0,
                "allowed_entry_count": 0,
                "blocked_entry_count_bull": 0,
                "blocked_entry_count_bear": 0,
                "blocked_entry_count_neutral": 0,
            }
        if er_filter:
            er = _efficiency_ratio(px, window=er_window)
            pos, one_er_stats = _apply_er_entry_filter(
                pos.astype(float).fillna(0.0), er=er, threshold=er_threshold
            )
            er_filter_by_asset[str(c)] = dict(one_er_stats or {})
        else:
            er_filter_by_asset[str(c)] = {
                "blocked_entry_count": 0,
                "attempted_entry_count": 0,
                "allowed_entry_count": 0,
            }
        if er_exit_filter:
            er_exit = _efficiency_ratio(px, window=er_exit_window)
            pos, one_er_exit_stats = _apply_er_exit_filter(
                pos.astype(float).fillna(0.0), er=er_exit, threshold=er_exit_threshold
            )
            er_exit_filter_by_asset[str(c)] = dict(one_er_exit_stats or {})
        else:
            er_exit_filter_by_asset[str(c)] = {
                "trigger_count": 0,
                "trigger_dates": [],
                "trace_last_rows": [],
            }
        h = high_qfq_df[c] if (c in high_qfq_df.columns) else px
        low_px = low_qfq_df[c] if (c in low_qfq_df.columns) else px
        pos, one_stop_stats = _apply_atr_stop(
            pos.fillna(0.0).astype(float),
            open_=(
                open_qfq_df[c].astype(float).fillna(px)
                if c in open_qfq_df.columns
                else px
            ),
            close=px,
            high=h.astype(float).fillna(px),
            low=low_px.astype(float).fillna(px),
            mode=atr_mode,
            atr_basis=atr_basis,
            reentry_mode=atr_reentry_mode,
            atr_window=int(inp.atr_stop_window),
            n_mult=float(inp.atr_stop_n),
            m_step=float(inp.atr_stop_m),
        )
        one_stop_stats = {
            **(one_stop_stats or {}),
            **_extract_atr_plan_stops_from_trace(one_stop_stats or {}),
        }
        pos, one_bias_v_tp_stats = _apply_bias_v_take_profit(
            pos.fillna(0.0).astype(float),
            open_=(
                open_qfq_df[c].astype(float).fillna(px)
                if c in open_qfq_df.columns
                else px
            ),
            close=px,
            high=h.astype(float).fillna(px),
            low=low_px.astype(float).fillna(px),
            enabled=bias_v_tp_enabled,
            reentry_mode=bias_v_tp_reentry_mode,
            ma_window=int(bias_v_tp_ma_window),
            atr_window=int(bias_v_tp_atr_window),
            threshold=float(bias_v_tp_threshold),
        )
        pos, one_rtp_stats = _apply_r_multiple_take_profit(
            pos.fillna(0.0).astype(float),
            open_=(
                open_qfq_df[c].astype(float).fillna(px)
                if c in open_qfq_df.columns
                else px
            ),
            close=px,
            high=h.astype(float).fillna(px),
            low=low_px.astype(float).fillna(px),
            enabled=rtp_enabled,
            reentry_mode=rtp_reentry_mode,
            atr_window=int(inp.atr_stop_window),
            atr_n=float(inp.atr_stop_n),
            tiers=rtp_tiers,
            atr_stop_enabled=bool(atr_mode != "none"),
        )
        sig_pos[c] = pos.fillna(0.0)
        sig_score[c] = score.replace([np.inf, -np.inf], np.nan)
        cond_momentum_df[c] = (px / px.shift(int(inp.mom_lookback)) - 1.0).replace(
            [np.inf, -np.inf], np.nan
        )
        cond_er_df[c] = _efficiency_ratio(px, window=int(er_window)).replace(
            [np.inf, -np.inf], np.nan
        )
        atr_fast_for_entry = _atr_from_hlc(
            h.astype(float).fillna(px),
            low_px.astype(float).fillna(px),
            px,
            window=int(vol_ratio_fast_atr_window),
        ).astype(float)
        atr_slow_for_entry = _atr_from_hlc(
            h.astype(float).fillna(px),
            low_px.astype(float).fillna(px),
            px,
            window=int(vol_ratio_slow_atr_window),
        ).astype(float)
        cond_vol_ratio_df[c] = (
            atr_fast_for_entry / atr_slow_for_entry.replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        cond_impulse_df[c] = (
            impulse_state_one.reindex(dates).astype(object)
            if impulse_state_one is not None
            else pd.Series("NA", index=dates, dtype=object)
        )
        atr_stop_by_asset[str(c)] = one_stop_stats
        rtp_by_asset[str(c)] = one_rtp_stats
        bias_v_tp_by_asset[str(c)] = one_bias_v_tp_stats

    atr_gate_df = pd.DataFrame(index=dates, columns=codes, dtype=float)
    for c in codes:
        px = close_qfq[c].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
        h = high_qfq_df[c] if (c in high_qfq_df.columns) else px
        low_px = low_qfq_df[c] if (c in low_qfq_df.columns) else px
        atr_gate_df[c] = _atr_from_hlc(
            h.astype(float).fillna(px),
            low_px.astype(float).fillna(px),
            px,
            window=int(inp.atr_stop_window),
        ).astype(float)

    atr_budget_df = pd.DataFrame(index=dates, columns=codes, dtype=float)
    atr_ratio_fast_df = pd.DataFrame(index=dates, columns=codes, dtype=float)
    atr_ratio_slow_df = pd.DataFrame(index=dates, columns=codes, dtype=float)
    if ps == "risk_budget":
        for c in codes:
            px = close_qfq[c].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
            h = high_qfq_df[c] if (c in high_qfq_df.columns) else px
            low_px = low_qfq_df[c] if (c in low_qfq_df.columns) else px
            atr_budget_df[c] = _atr_from_hlc(
                h.astype(float).fillna(px),
                low_px.astype(float).fillna(px),
                px,
                window=int(risk_budget_atr_window),
            ).astype(float)
            atr_ratio_fast_df[c] = _atr_from_hlc(
                h.astype(float).fillna(px),
                low_px.astype(float).fillna(px),
                px,
                window=int(vol_ratio_fast_atr_window),
            ).astype(float)
            atr_ratio_slow_df[c] = _atr_from_hlc(
                h.astype(float).fillna(px),
                low_px.astype(float).fillna(px),
                px,
                window=int(vol_ratio_slow_atr_window),
            ).astype(float)

    vol_ann = ret_hfq.rolling(
        window=int(inp.vol_window), min_periods=max(3, int(inp.vol_window) // 2)
    ).std(ddof=1) * np.sqrt(252.0)
    sharpe_like = (
        ret_hfq.rolling(
            window=max(20, int(inp.vol_window)),
            min_periods=max(10, int(inp.vol_window) // 2),
        ).mean()
        / ret_hfq.rolling(
            window=max(20, int(inp.vol_window)),
            min_periods=max(10, int(inp.vol_window) // 2),
        )
        .std(ddof=1)
        .replace(0.0, np.nan)
    ) * np.sqrt(252.0)
    w_decision = pd.DataFrame(0.0, index=dates, columns=codes, dtype=float)
    holdings: list[dict[str, Any]] = []
    prev_key: tuple[str, ...] | None = None
    prev_held_set: set[str] = set()
    fixed_ratio = float(getattr(inp, "fixed_pos_ratio", 0.04) or 0.04)
    fixed_max_holding_n = int(getattr(inp, "fixed_max_holdings", 10) or 10)
    ext_events: list[dict[str, Any]] = []
    skip_events: list[dict[str, Any]] = []
    prev_fixed_w = pd.Series(0.0, index=codes, dtype=float)
    prev_rb_w = pd.Series(0.0, index=codes, dtype=float)
    rb_state_by_code: dict[str, str] = {str(c): "FLAT" for c in codes}
    rb_entry_price_by_code: dict[str, float] = {str(c): float("nan") for c in codes}
    rb_entry_seq_by_code: dict[str, int] = {str(c): -1 for c in codes}
    risk_budget_overcap_scale_count = 0
    risk_budget_overcap_scale_by_code: dict[str, int] = {str(c): 0 for c in codes}
    risk_budget_overcap_skip_decision_count = 0
    risk_budget_overcap_skip_episode_count = 0
    risk_budget_overcap_skip_decision_by_code: dict[str, int] = {
        str(c): 0 for c in codes
    }
    risk_budget_overcap_skip_episode_by_code: dict[str, int] = {
        str(c): 0 for c in codes
    }
    risk_budget_overcap_skip_episode_active_by_code: dict[str, bool] = {
        str(c): False for c in codes
    }
    risk_budget_overcap_replace_count = 0
    risk_budget_overcap_replace_out_by_code: dict[str, int] = {str(c): 0 for c in codes}
    risk_budget_overcap_replace_in_by_code: dict[str, int] = {str(c): 0 for c in codes}
    risk_budget_overcap_leverage_usage_count = 0
    risk_budget_overcap_leverage_usage_by_code: dict[str, int] = {
        str(c): 0 for c in codes
    }
    risk_budget_overcap_leverage_max_multiple = 0.0
    risk_budget_overcap_leverage_max_multiple_by_code: dict[str, float] = {
        str(c): 0.0 for c in codes
    }
    risk_budget_overcap_daily_counts: dict[str, dict[str, int]] = {}
    day_seq = 0
    vol_risk_adjust_by_asset: dict[str, dict[str, int]] = {
        str(c): {
            "vol_risk_adjust_total_count": 0,
            "vol_risk_adjust_reduce_on_expand_count": 0,
            "vol_risk_adjust_increase_on_contract_count": 0,
            "vol_risk_adjust_recover_from_expand_count": 0,
            "vol_risk_adjust_recover_from_contract_count": 0,
            "vol_risk_entry_state_reduce_on_expand_count": 0,
            "vol_risk_entry_state_increase_on_contract_count": 0,
        }
        for c in codes
    }
    for d in dates:
        day_seq += 1
        d_key = d.date().isoformat()
        active = sig_pos.loc[d]
        scores = (
            sig_score.loc[d]
            .where(active > 0.0, other=np.nan)
            .replace([np.inf, -np.inf], np.nan)
        )
        active_codes_raw = [
            str(c) for c in scores.dropna().sort_values(ascending=False).index.tolist()
        ]
        active_codes, group_meta = _reduce_active_codes_by_group(
            active_codes=active_codes_raw,
            score_row=scores,
            sharpe_row=sharpe_like.loc[d]
            if d in sharpe_like.index
            else pd.Series(dtype=float),
            group_enforce=group_enforce,
            asset_groups=group_map,
            group_pick_policy=group_pick_policy,
            group_max_holdings=group_max_holdings,
            current_holdings=prev_held_set,
        )
        if ps == "fixed_ratio":
            active_set = set(active_codes)
            w_row = prev_fixed_w.copy().astype(float).reindex(codes).fillna(0.0)
            # Exit when base signal is no longer active.
            for c in codes:
                if (w_row.loc[c] > 1e-12) and (c not in active_set):
                    w_row.loc[c] = 0.0
            # Keep existing active positions at fixed ratio.
            for c in active_set:
                if w_row.loc[c] > 1e-12:
                    w_row.loc[c] = fixed_ratio
            # New entries follow score rank order.
            for c in active_codes:
                if w_row.loc[c] > 1e-12:
                    continue
                cur_total = float(w_row.sum())
                proposed_total = float(cur_total + fixed_ratio)
                cur_count = int((w_row > 1e-12).sum())
                proposed_count = int(cur_count + 1)
                over_weight = bool(proposed_total > 1.0 + 1e-12)
                over_count = bool(proposed_count > fixed_max_holding_n)
                if over_weight or over_count:
                    event = {
                        "date": d.date().isoformat(),
                        "code": str(c),
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
                        skip_events.append(event)
                        continue
                    ext_events.append(event)
                w_row.loc[c] = fixed_ratio
            prev_fixed_w = w_row.copy()
        elif ps == "risk_budget":
            active_set = set(active_codes)
            w_row = prev_rb_w.copy().astype(float).reindex(codes).fillna(0.0)
            skipped_today: set[str] = set()

            def _inc_overcap_daily(kind: str, n: int = 1) -> None:
                nn = int(n)
                if nn <= 0:
                    return
                row = risk_budget_overcap_daily_counts.setdefault(
                    d_key,
                    {
                        "scale": 0,
                        "skip_entry": 0,
                        "replace_entry": 0,
                        "leverage_entry": 0,
                        "leverage_multiple_max": 0.0,
                    },
                )
                row[str(kind)] = int(row.get(str(kind), 0) + nn)

            def _apply_overcap_scale_once(cap_multiple: float = 1.0) -> None:
                nonlocal w_row, risk_budget_overcap_scale_count
                cap_v = (
                    float(cap_multiple)
                    if np.isfinite(float(cap_multiple)) and float(cap_multiple) > 0.0
                    else 1.0
                )
                s_now = float(w_row.sum())
                if s_now <= cap_v + 1e-12:
                    return
                pre_scale = w_row.copy().astype(float)
                w_row = (w_row * (cap_v / s_now)).astype(float)
                risk_budget_overcap_scale_count += 1
                _inc_overcap_daily("scale", 1)
                for cc in w_row.index:
                    key_cc = str(cc)
                    before = (
                        float(pre_scale.loc[cc])
                        if np.isfinite(float(pre_scale.loc[cc]))
                        else 0.0
                    )
                    after = (
                        float(w_row.loc[cc])
                        if np.isfinite(float(w_row.loc[cc]))
                        else 0.0
                    )
                    if before > after + 1e-12:
                        risk_budget_overcap_scale_by_code[key_cc] = int(
                            risk_budget_overcap_scale_by_code.get(key_cc, 0) + 1
                        )

            def _set_new_risk_budget_entry(key: str, base_target: float) -> None:
                nonlocal w_row
                w_row.loc[key] = float(base_target)
                px_now = (
                    float(close_qfq.loc[d, key])
                    if (key in close_qfq.columns and d in close_qfq.index)
                    else float("nan")
                )
                rb_entry_price_by_code[key] = (
                    float(px_now)
                    if np.isfinite(px_now) and px_now > 0.0
                    else float("nan")
                )
                rb_entry_seq_by_code[key] = int(day_seq)
                if bool(vol_regime_risk_mgmt_enabled):
                    af = (
                        float(atr_ratio_fast_df.loc[d, key])
                        if (
                            key in atr_ratio_fast_df.columns
                            and d in atr_ratio_fast_df.index
                        )
                        else float("nan")
                    )
                    aslow = (
                        float(atr_ratio_slow_df.loc[d, key])
                        if (
                            key in atr_ratio_slow_df.columns
                            and d in atr_ratio_slow_df.index
                        )
                        else float("nan")
                    )
                    ratio = (
                        (af / aslow)
                        if (np.isfinite(af) and np.isfinite(aslow) and aslow > 0.0)
                        else float("nan")
                    )
                    if np.isfinite(ratio) and ratio > float(vol_ratio_expand_threshold):
                        rb_state_by_code[key] = "REDUCED"
                        vol_risk_adjust_by_asset[key][
                            "vol_risk_entry_state_reduce_on_expand_count"
                        ] += 1
                    elif np.isfinite(ratio) and ratio < float(
                        vol_ratio_contract_threshold
                    ):
                        rb_state_by_code[key] = "INCREASED"
                        vol_risk_adjust_by_asset[key][
                            "vol_risk_entry_state_increase_on_contract_count"
                        ] += 1
                    else:
                        rb_state_by_code[key] = "NORMAL"
                else:
                    rb_state_by_code[key] = "NORMAL"

            def _select_replace_out_code(new_code: str) -> str | None:
                cand: list[tuple[float, int, str]] = []
                for cc in codes:
                    key_cc = str(cc)
                    if key_cc == str(new_code):
                        continue
                    if float(w_row.loc[key_cc]) <= 1e-12:
                        continue
                    cur_px = (
                        float(close_qfq.loc[d, key_cc])
                        if (key_cc in close_qfq.columns and d in close_qfq.index)
                        else float("nan")
                    )
                    ent_px = float(rb_entry_price_by_code.get(key_cc, float("nan")))
                    ret = (
                        ((cur_px / ent_px) - 1.0)
                        if (
                            np.isfinite(cur_px)
                            and cur_px > 0.0
                            and np.isfinite(ent_px)
                            and ent_px > 0.0
                        )
                        else float("inf")
                    )
                    seq = int(rb_entry_seq_by_code.get(key_cc, 10**9))
                    if seq < 0:
                        seq = 10**9
                    cand.append((float(ret), int(seq), key_cc))
                if not cand:
                    return None
                cand.sort(key=lambda x: (x[0], x[1], x[2]))
                return str(cand[0][2])

            # Exit when base signal is no longer active.
            for c in codes:
                if (w_row.loc[c] > 1e-12) and (c not in active_set):
                    w_row.loc[c] = 0.0
                    key_c = str(c)
                    rb_state_by_code[key_c] = "FLAT"
                    rb_entry_price_by_code[key_c] = float("nan")
                    rb_entry_seq_by_code[key_c] = -1
            # Keep existing active positions at their entry-time risk-budget weight.
            for c in active_codes:
                px = (
                    float(close_qfq.loc[d, c])
                    if (c in close_qfq.columns and d in close_qfq.index)
                    else float("nan")
                )
                a = (
                    float(atr_budget_df.loc[d, c])
                    if (c in atr_budget_df.columns and d in atr_budget_df.index)
                    else float("nan")
                )
                base_target = float("nan")
                if np.isfinite(px) and px > 0.0 and np.isfinite(a) and a > 0.0:
                    base_target = float(risk_budget_pct) * float(px) / float(a)
                has_pos = bool(w_row.loc[c] > 1e-12)
                key = str(c)
                if not has_pos:
                    if np.isfinite(base_target) and base_target > 0.0:
                        proposed_total = float(w_row.sum() + float(base_target))
                        overcap_on_new_entry = bool(proposed_total > 1.0 + 1e-12)
                        if (
                            overcap_on_new_entry
                            and str(risk_budget_overcap_policy) == "skip_entry"
                        ):
                            risk_budget_overcap_skip_decision_count += 1
                            _inc_overcap_daily("skip_entry", 1)
                            risk_budget_overcap_skip_decision_by_code[key] = int(
                                risk_budget_overcap_skip_decision_by_code.get(key, 0)
                                + 1
                            )
                            skipped_today.add(key)
                            if not bool(
                                risk_budget_overcap_skip_episode_active_by_code.get(
                                    key, False
                                )
                            ):
                                risk_budget_overcap_skip_episode_active_by_code[key] = (
                                    True
                                )
                                risk_budget_overcap_skip_episode_count += 1
                                risk_budget_overcap_skip_episode_by_code[key] = int(
                                    risk_budget_overcap_skip_episode_by_code.get(key, 0)
                                    + 1
                                )
                            continue
                        if (
                            overcap_on_new_entry
                            and str(risk_budget_overcap_policy) == "replace_entry"
                        ):
                            out_code = _select_replace_out_code(key)
                            if out_code:
                                w_row.loc[out_code] = 0.0
                                rb_state_by_code[out_code] = "FLAT"
                                rb_entry_price_by_code[out_code] = float("nan")
                                rb_entry_seq_by_code[out_code] = -1
                                risk_budget_overcap_replace_count += 1
                                _inc_overcap_daily("replace_entry", 1)
                                risk_budget_overcap_replace_out_by_code[out_code] = int(
                                    risk_budget_overcap_replace_out_by_code.get(
                                        out_code, 0
                                    )
                                    + 1
                                )
                                risk_budget_overcap_replace_in_by_code[key] = int(
                                    risk_budget_overcap_replace_in_by_code.get(key, 0)
                                    + 1
                                )
                        _set_new_risk_budget_entry(key, float(base_target))
                        if (
                            overcap_on_new_entry
                            and str(risk_budget_overcap_policy) == "replace_entry"
                        ):
                            _apply_overcap_scale_once()
                        elif (
                            overcap_on_new_entry
                            and str(risk_budget_overcap_policy) == "leverage_entry"
                        ):
                            lev_now = float(w_row.sum())
                            if lev_now > 1.0 + 1e-12:
                                risk_budget_overcap_leverage_usage_count += 1
                                _inc_overcap_daily("leverage_entry", 1)
                                landed_lev = float(
                                    min(
                                        float(lev_now),
                                        float(risk_budget_max_leverage_multiple),
                                    )
                                )
                                risk_budget_overcap_leverage_max_multiple = float(
                                    max(
                                        float(
                                            risk_budget_overcap_leverage_max_multiple
                                        ),
                                        landed_lev,
                                    )
                                )
                                row = risk_budget_overcap_daily_counts.setdefault(
                                    d_key,
                                    {
                                        "scale": 0,
                                        "skip_entry": 0,
                                        "replace_entry": 0,
                                        "leverage_entry": 0,
                                        "leverage_multiple_max": 0.0,
                                    },
                                )
                                row["leverage_multiple_max"] = float(
                                    max(
                                        float(
                                            row.get("leverage_multiple_max", 0.0) or 0.0
                                        ),
                                        landed_lev,
                                    )
                                )
                                for cc in codes:
                                    key_cc = str(cc)
                                    if float(w_row.loc[key_cc]) > 1e-12:
                                        risk_budget_overcap_leverage_usage_by_code[
                                            key_cc
                                        ] = int(
                                            risk_budget_overcap_leverage_usage_by_code.get(
                                                key_cc, 0
                                            )
                                            + 1
                                        )
                                        risk_budget_overcap_leverage_max_multiple_by_code[
                                            key_cc
                                        ] = float(
                                            max(
                                                float(
                                                    risk_budget_overcap_leverage_max_multiple_by_code.get(
                                                        key_cc, 0.0
                                                    )
                                                ),
                                                landed_lev,
                                            )
                                        )
                                if (
                                    lev_now
                                    > float(risk_budget_max_leverage_multiple) + 1e-12
                                ):
                                    _apply_overcap_scale_once(
                                        float(risk_budget_max_leverage_multiple)
                                    )
                    continue
                if not bool(vol_regime_risk_mgmt_enabled):
                    continue
                st = str(rb_state_by_code.get(key, "NORMAL") or "NORMAL").upper()
                af = (
                    float(atr_ratio_fast_df.loc[d, c])
                    if (c in atr_ratio_fast_df.columns and d in atr_ratio_fast_df.index)
                    else float("nan")
                )
                aslow = (
                    float(atr_ratio_slow_df.loc[d, c])
                    if (c in atr_ratio_slow_df.columns and d in atr_ratio_slow_df.index)
                    else float("nan")
                )
                ratio = (
                    (af / aslow)
                    if (np.isfinite(af) and np.isfinite(aslow) and aslow > 0.0)
                    else float("nan")
                )
                if not np.isfinite(base_target):
                    continue
                if st == "NORMAL":
                    if np.isfinite(ratio) and ratio > float(vol_ratio_expand_threshold):
                        w_row.loc[c] = float(base_target)
                        rb_state_by_code[key] = "REDUCED"
                        vol_risk_adjust_by_asset[key][
                            "vol_risk_adjust_total_count"
                        ] += 1
                        vol_risk_adjust_by_asset[key][
                            "vol_risk_adjust_reduce_on_expand_count"
                        ] += 1
                    elif np.isfinite(ratio) and ratio < float(
                        vol_ratio_contract_threshold
                    ):
                        w_row.loc[c] = float(base_target)
                        rb_state_by_code[key] = "INCREASED"
                        vol_risk_adjust_by_asset[key][
                            "vol_risk_adjust_total_count"
                        ] += 1
                        vol_risk_adjust_by_asset[key][
                            "vol_risk_adjust_increase_on_contract_count"
                        ] += 1
                elif st == "REDUCED":
                    if np.isfinite(ratio) and ratio < float(vol_ratio_normal_threshold):
                        w_row.loc[c] = float(base_target)
                        rb_state_by_code[key] = "NORMAL"
                        vol_risk_adjust_by_asset[key][
                            "vol_risk_adjust_total_count"
                        ] += 1
                        vol_risk_adjust_by_asset[key][
                            "vol_risk_adjust_recover_from_expand_count"
                        ] += 1
                elif st == "INCREASED":
                    if np.isfinite(ratio) and ratio > float(vol_ratio_normal_threshold):
                        w_row.loc[c] = float(base_target)
                        rb_state_by_code[key] = "NORMAL"
                        vol_risk_adjust_by_asset[key][
                            "vol_risk_adjust_total_count"
                        ] += 1
                        vol_risk_adjust_by_asset[key][
                            "vol_risk_adjust_recover_from_contract_count"
                        ] += 1
            w_row = w_row.clip(lower=0.0)
            if str(risk_budget_overcap_policy) != "leverage_entry":
                _apply_overcap_scale_once()
            for cc in codes:
                key_cc = str(cc)
                if bool(
                    risk_budget_overcap_skip_episode_active_by_code.get(key_cc, False)
                ) and (key_cc not in skipped_today):
                    risk_budget_overcap_skip_episode_active_by_code[key_cc] = False
            prev_rb_w = w_row.copy()
        elif not active_codes:
            w_row = pd.Series(0.0, index=codes, dtype=float)
        elif ps == "equal":
            per = 1.0 / float(len(active_codes))
            w_row = pd.Series(0.0, index=codes, dtype=float)
            for c in active_codes:
                w_row.loc[c] = per
        else:
            inv: dict[str, float] = {}
            for c in active_codes:
                try:
                    av = float(vol_ann.loc[d, c])
                except (TypeError, ValueError, KeyError):
                    av = float("nan")
                if (not np.isfinite(av)) or av <= 0:
                    inv[c] = 0.0
                else:
                    inv[c] = 1.0 / av
            den = float(sum(inv.values()))
            w_row = pd.Series(0.0, index=codes, dtype=float)
            if den > 0:
                raw = {c: float(v) / den for c, v in inv.items()}
                # portfolio target-vol scalar
                port_vol = float(
                    np.sqrt(
                        np.sum(
                            [
                                (raw[c] ** 2)
                                * (
                                    (
                                        float(vol_ann.loc[d, c])
                                        if np.isfinite(float(vol_ann.loc[d, c]))
                                        else 0.0
                                    )
                                    ** 2
                                )
                                for c in active_codes
                            ]
                        )
                    )
                )
                scale = (
                    1.0
                    if port_vol <= 1e-12
                    else min(1.0, float(inp.vol_target_ann) / port_vol)
                )
                for c in active_codes:
                    w_row.loc[c] = raw[c] * scale
            else:
                per = 1.0 / float(len(active_codes))
                for c in active_codes:
                    w_row.loc[c] = per
        w_decision.loc[d] = w_row.to_numpy(dtype=float)
        held_codes = [str(c) for c in w_row.index if float(w_row.loc[c]) > 1e-12]
        key = tuple(sorted(held_codes))
        if key != prev_key:
            holdings.append(
                {
                    "decision_date": d.date().isoformat(),
                    "picks": list(key),
                    "grouped_picks": {
                        str(k): [str(x) for x in (v or [])]
                        for k, v in (
                            (group_meta or {}).get("group_picks", {}) or {}
                        ).items()
                    },
                    "scores": {
                        c: (None if pd.isna(scores.get(c)) else float(scores.get(c)))
                        for c in key
                    },
                    "group_filter": group_meta,
                }
            )
            prev_key = key
        prev_held_set = set(held_codes)

    monthly_risk_budget_gate_stats: dict[str, Any] = {
        "enabled": False,
        "budget_pct": float(monthly_risk_budget_pct),
        "include_new_trade_risk": bool(monthly_risk_budget_include_new_trade_risk),
        "attempted_entry_count": 0,
        "attempted_entry_count_by_code": {str(c): 0 for c in codes},
        "blocked_entry_count": 0,
        "blocked_entry_count_by_code": {str(c): 0 for c in codes},
    }
    if monthly_risk_budget_enabled:
        w_decision, monthly_risk_budget_gate_stats = _apply_monthly_risk_budget_gate(
            w_decision.reindex(index=dates, columns=codes).astype(float).fillna(0.0),
            close=close_qfq.reindex(index=dates, columns=codes).astype(float),
            atr=atr_gate_df.reindex(index=dates, columns=codes).astype(float),
            enabled=True,
            budget_pct=float(monthly_risk_budget_pct),
            include_new_trade_risk=bool(monthly_risk_budget_include_new_trade_risk),
            atr_stop_enabled=bool(atr_mode != "none"),
            atr_mode=str(atr_mode),
            atr_basis=str(atr_basis),
            atr_n=float(inp.atr_stop_n),
            atr_m=float(inp.atr_stop_m),
            fallback_position_risk=0.01,
        )
        # Rebuild holdings timeline from gated decision weights.
        holdings = []
        prev_key = None
        for d in w_decision.index:
            row = w_decision.loc[d]
            key = tuple(
                sorted(
                    [
                        str(c)
                        for c in w_decision.columns
                        if float(row.get(c, 0.0)) > 1e-12
                    ]
                )
            )
            if key != prev_key:
                sc = (
                    sig_score.loc[d] if d in sig_score.index else pd.Series(dtype=float)
                )
                holdings.append(
                    {
                        "decision_date": d.date().isoformat(),
                        "picks": list(key),
                        "grouped_picks": {},
                        "scores": {
                            c: (None if pd.isna(sc.get(c)) else float(sc.get(c)))
                            for c in key
                        },
                        "group_filter": {},
                    }
                )
                prev_key = key

    # Weights become effective on execution day; ret_exec is already execution-day aligned.
    w = w_decision.shift(1).fillna(0.0).astype(float)
    w, atr_stop_override_ret = _apply_intraday_stop_execution_portfolio(
        weights=w,
        atr_stop_by_asset=atr_stop_by_asset,
        exec_price=str(ep),
        open_sig_df=open_qfq_df.reindex(index=w.index, columns=codes)
        .astype(float)
        .ffill(),
        close_sig_df=close_qfq.reindex(index=w.index, columns=codes)
        .astype(float)
        .ffill(),
    )
    w, bias_v_take_profit_override_ret = _apply_intraday_stop_execution_portfolio(
        weights=w,
        atr_stop_by_asset=bias_v_tp_by_asset,
        exec_price=str(ep),
        open_sig_df=open_qfq_df.reindex(index=w.index, columns=codes)
        .astype(float)
        .ffill(),
        close_sig_df=close_qfq.reindex(index=w.index, columns=codes)
        .astype(float)
        .ffill(),
    )
    w, r_take_profit_override_ret = _apply_intraday_stop_execution_portfolio(
        weights=w,
        atr_stop_by_asset=rtp_by_asset,
        exec_price=str(ep),
        open_sig_df=open_qfq_df.reindex(index=w.index, columns=codes)
        .astype(float)
        .ffill(),
        close_sig_df=close_qfq.reindex(index=w.index, columns=codes)
        .astype(float)
        .ffill(),
    )
    ret_exec_day = ret_exec.astype(float)
    bench_ret = bench_ret.reindex(dates).fillna(0.0).astype(float)
    bench_nav = (1.0 + bench_ret).cumprod()
    if len(bench_nav) > 0:
        bench_nav.iloc[0] = 1.0
    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1) / 2.0
    cost = turnover * (float(inp.cost_bps) / 10000.0)
    turnover_by_asset = (w - w.shift(1).fillna(0.0)).abs() / 2.0
    slippage = (
        slippage_return_from_turnover(
            turnover_by_asset.astype(float),
            slippage_spread=float(inp.slippage_rate),
            exec_price=px_exec_slip.reindex(index=w.index, columns=codes).ffill(),
        )
        .sum(axis=1)
        .astype(float)
    )
    if not quick_mode:
        for c in codes:
            key = str(c)
            one_atr = atr_stop_by_asset.get(key) or {}
            atr_stop_by_asset[key] = {
                **one_atr,
                "trade_records": _enrich_trade_records_with_engine_timeline(
                    records=list(one_atr.get("trade_records") or []),
                    effective_weight=w[key].astype(float),
                    exec_price_series=px_exec_slip[key]
                    .reindex(w.index)
                    .ffill()
                    .astype(float),
                    slippage_spread=float(inp.slippage_rate),
                ),
            }
            one_bv = bias_v_tp_by_asset.get(key) or {}
            bias_v_tp_by_asset[key] = {
                **one_bv,
                "trade_records": _enrich_trade_records_with_engine_timeline(
                    records=list(one_bv.get("trade_records") or []),
                    effective_weight=w[key].astype(float),
                    exec_price_series=px_exec_slip[key]
                    .reindex(w.index)
                    .ffill()
                    .astype(float),
                    slippage_spread=float(inp.slippage_rate),
                ),
            }
    else:
        for c in codes:
            key = str(c)
            one_atr = atr_stop_by_asset.get(key) or {}
            atr_stop_by_asset[key] = {**one_atr, "trade_records": []}
            one_bv = bias_v_tp_by_asset.get(key) or {}
            bias_v_tp_by_asset[key] = {**one_bv, "trade_records": []}
    return_decomposition: dict[str, Any] | None = None
    if not quick_mode:
        comp_overnight = (
            (w * ret_overnight.reindex(index=w.index, columns=codes).fillna(0.0))
            .sum(axis=1)
            .astype(float)
        )
        comp_intraday = (
            (w * ret_intraday.reindex(index=w.index, columns=codes).fillna(0.0))
            .sum(axis=1)
            .astype(float)
        )
        comp_interaction = (
            (
                w
                * (
                    ret_overnight.reindex(index=w.index, columns=codes).fillna(0.0)
                    * ret_intraday.reindex(index=w.index, columns=codes).fillna(0.0)
                )
            )
            .sum(axis=1)
            .astype(float)
        )
        decomp_atr_stop_override = (
            atr_stop_override_ret.reindex(w.index).fillna(0.0).astype(float)
        )
        decomp_bias_v_take_profit_override = (
            bias_v_take_profit_override_ret.reindex(w.index).fillna(0.0).astype(float)
        )
        decomp_r_take_profit_override = (
            r_take_profit_override_ret.reindex(w.index).fillna(0.0).astype(float)
        )
        decomp_risk_exit_override = (
            decomp_atr_stop_override
            + decomp_bias_v_take_profit_override
            + decomp_r_take_profit_override
        ).astype(float)
        decomp_cost = (cost + slippage).astype(float)
        decomp_gross = (
            comp_overnight
            + comp_intraday
            + comp_interaction
            + decomp_risk_exit_override
        ).astype(float)
        port_ret = (decomp_gross - decomp_cost).astype(float)
        return_decomposition = {
            "dates": w.index.date.astype(str).tolist(),
            "series": {
                "overnight": comp_overnight.astype(float).tolist(),
                "intraday": comp_intraday.astype(float).tolist(),
                "interaction": comp_interaction.astype(float).tolist(),
                "atr_stop_override": decomp_atr_stop_override.astype(float).tolist(),
                "bias_v_take_profit_override": decomp_bias_v_take_profit_override.astype(
                    float
                ).tolist(),
                "r_take_profit_override": decomp_r_take_profit_override.astype(
                    float
                ).tolist(),
                "risk_exit_override": decomp_risk_exit_override.astype(float).tolist(),
                "cost": decomp_cost.astype(float).tolist(),
                "gross": decomp_gross.astype(float).tolist(),
                "net": port_ret.astype(float).tolist(),
            },
            "summary": {
                "ann_overnight": float(
                    comp_overnight.iloc[1:].mean() * TRADING_DAYS_PER_YEAR
                )
                if len(comp_overnight) > 1
                else 0.0,
                "ann_intraday": float(
                    comp_intraday.iloc[1:].mean() * TRADING_DAYS_PER_YEAR
                )
                if len(comp_intraday) > 1
                else 0.0,
                "ann_interaction": float(
                    comp_interaction.iloc[1:].mean() * TRADING_DAYS_PER_YEAR
                )
                if len(comp_interaction) > 1
                else 0.0,
                "ann_atr_stop_override": float(
                    decomp_atr_stop_override.iloc[1:].mean() * TRADING_DAYS_PER_YEAR
                )
                if len(decomp_atr_stop_override) > 1
                else 0.0,
                "ann_bias_v_take_profit_override": float(
                    decomp_bias_v_take_profit_override.iloc[1:].mean()
                    * TRADING_DAYS_PER_YEAR
                )
                if len(decomp_bias_v_take_profit_override) > 1
                else 0.0,
                "ann_r_take_profit_override": float(
                    decomp_r_take_profit_override.iloc[1:].mean()
                    * TRADING_DAYS_PER_YEAR
                )
                if len(decomp_r_take_profit_override) > 1
                else 0.0,
                "ann_risk_exit_override": float(
                    decomp_risk_exit_override.iloc[1:].mean() * TRADING_DAYS_PER_YEAR
                )
                if len(decomp_risk_exit_override) > 1
                else 0.0,
                "ann_cost": float(decomp_cost.iloc[1:].mean() * TRADING_DAYS_PER_YEAR)
                if len(decomp_cost) > 1
                else 0.0,
                "ann_gross": float(decomp_gross.iloc[1:].mean() * TRADING_DAYS_PER_YEAR)
                if len(decomp_gross) > 1
                else 0.0,
                "ann_net": float(port_ret.iloc[1:].mean() * TRADING_DAYS_PER_YEAR)
                if len(port_ret) > 1
                else 0.0,
            },
        }
    else:
        decomp_atr_stop_override = (
            atr_stop_override_ret.reindex(w.index).fillna(0.0).astype(float)
        )
        decomp_bias_v_take_profit_override = (
            bias_v_take_profit_override_ret.reindex(w.index).fillna(0.0).astype(float)
        )
        decomp_r_take_profit_override = (
            r_take_profit_override_ret.reindex(w.index).fillna(0.0).astype(float)
        )
        decomp_risk_exit_override = (
            decomp_atr_stop_override
            + decomp_bias_v_take_profit_override
            + decomp_r_take_profit_override
        ).astype(float)
        port_ret = (
            (w * ret_exec_day).sum(axis=1).astype(float)
            + decomp_risk_exit_override
            - cost.astype(float)
            - slippage.astype(float)
        )
    nav = (1.0 + port_ret).cumprod()
    if len(nav) > 0:
        nav.iloc[0] = 1.0
    active = (port_ret - bench_ret).astype(float)
    ex_nav = (1.0 + active).cumprod()
    if len(ex_nav) > 0:
        ex_nav.iloc[0] = 1.0

    ui_strat = float(_ulcer_index(nav, in_percent=True))
    ui_bench = float(_ulcer_index(bench_nav, in_percent=True))
    ann_strat = float(_annualized_return(nav, ann_factor=TRADING_DAYS_PER_YEAR))
    ann_bench = float(_annualized_return(bench_nav, ann_factor=TRADING_DAYS_PER_YEAR))
    m_strat = {
        "cumulative_return": float(nav.iloc[-1] - 1.0),
        "annualized_return": float(ann_strat),
        "annualized_volatility": float(
            _annualized_vol(port_ret, ann_factor=TRADING_DAYS_PER_YEAR)
        ),
        "max_drawdown": float(_max_drawdown(nav)),
        "max_drawdown_recovery_days": int(_max_drawdown_duration_days(nav)),
        "sharpe_ratio": float(
            _sharpe(
                port_ret, rf=float(inp.risk_free_rate), ann_factor=TRADING_DAYS_PER_YEAR
            )
        ),
        "sortino_ratio": float(
            _sortino(
                port_ret, rf=float(inp.risk_free_rate), ann_factor=TRADING_DAYS_PER_YEAR
            )
        ),
        "ulcer_index": float(ui_strat),
        "ulcer_performance_index": float(
            (ann_strat - float(inp.risk_free_rate)) / (ui_strat / 100.0)
        )
        if ui_strat > 0
        else float("nan"),
        "avg_daily_turnover": float(turnover.mean()) if len(turnover) else 0.0,
    }
    m_bench = {
        "cumulative_return": float(bench_nav.iloc[-1] - 1.0),
        "annualized_return": float(ann_bench),
        "annualized_volatility": float(
            _annualized_vol(bench_ret, ann_factor=TRADING_DAYS_PER_YEAR)
        ),
        "max_drawdown": float(_max_drawdown(bench_nav)),
        "max_drawdown_recovery_days": int(_max_drawdown_duration_days(bench_nav)),
        "sharpe_ratio": float(
            _sharpe(
                bench_ret,
                rf=float(inp.risk_free_rate),
                ann_factor=TRADING_DAYS_PER_YEAR,
            )
        ),
        "sortino_ratio": float(
            _sortino(
                bench_ret,
                rf=float(inp.risk_free_rate),
                ann_factor=TRADING_DAYS_PER_YEAR,
            )
        ),
        "ulcer_index": float(ui_bench),
        "ulcer_performance_index": float(
            (ann_bench - float(inp.risk_free_rate)) / (ui_bench / 100.0)
        )
        if ui_bench > 0
        else float("nan"),
    }
    m_ex = {
        "cumulative_return": float(ex_nav.iloc[-1] - 1.0),
        "annualized_return": float(
            _annualized_return(ex_nav, ann_factor=TRADING_DAYS_PER_YEAR)
        ),
        "information_ratio": float(
            _information_ratio(active, ann_factor=TRADING_DAYS_PER_YEAR)
        ),
    }
    attribution = _compute_return_risk_contributions(
        asset_ret=ret_exec_day.reindex(index=nav.index, columns=codes)
        .astype(float)
        .fillna(0.0),
        weights=w.reindex(index=nav.index, columns=codes).astype(float).fillna(0.0),
        total_return=float(nav.iloc[-1] - 1.0),
    )
    trade_pack = _trade_returns_from_weight_df(
        w.reindex(index=nav.index, columns=codes).astype(float).fillna(0.0),
        ret_exec_day.reindex(index=nav.index, columns=codes).astype(float).fillna(0.0),
        cost_bps=float(inp.cost_bps),
        slippage_rate=float(inp.slippage_rate),
        exec_price=px_exec_slip.reindex(index=nav.index, columns=codes).ffill(),
        dates=nav.index,
    )
    sample_days = int(len(port_ret))
    complete_trade_count = int(len(trade_pack.get("returns", [])))
    avg_daily_turnover = float(m_strat["avg_daily_turnover"])
    avg_annual_turnover = float(avg_daily_turnover * TRADING_DAYS_PER_YEAR)
    avg_daily_trade_count = (
        float(complete_trade_count / sample_days) if sample_days > 0 else 0.0
    )
    avg_annual_trade_count = float(avg_daily_trade_count * TRADING_DAYS_PER_YEAR)
    m_strat["avg_annual_turnover"] = float(avg_annual_turnover)
    m_strat["avg_annual_turnover_rate"] = float(avg_annual_turnover)
    m_strat["avg_daily_trade_count"] = float(avg_daily_trade_count)
    m_strat["avg_annual_trade_count"] = float(avg_annual_trade_count)
    atr_risk_df = pd.DataFrame(index=nav.index, columns=codes, dtype=float)
    for c in codes:
        px = close_qfq[c].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
        hi = high_qfq_df[c].astype(float).fillna(px) if c in high_qfq_df.columns else px
        lo = low_qfq_df[c].astype(float).fillna(px) if c in low_qfq_df.columns else px
        atr_risk_df[c] = (
            _atr_from_hlc(
                hi,
                lo,
                px,
                window=int(inp.atr_stop_window),
            )
            .reindex(nav.index)
            .astype(float)
        )
    trade_r_pack = enrich_trades_with_r_metrics(
        trade_pack.get("trades", []),
        nav=nav.astype(float),
        weights=w.reindex(index=nav.index, columns=codes).astype(float).fillna(0.0),
        exec_price=px_exec_slip.reindex(index=nav.index, columns=codes)
        .ffill()
        .astype(float),
        atr=atr_risk_df.reindex(index=nav.index, columns=codes).astype(float),
        atr_mult=float(inp.atr_stop_n),
        risk_budget_pct=float(risk_budget_pct)
        if np.isfinite(float(risk_budget_pct))
        else None,
        cost_bps=float(inp.cost_bps),
        slippage_rate=float(inp.slippage_rate),
        ulcer_index=float(m_strat.get("ulcer_index"))
        if np.isfinite(float(m_strat.get("ulcer_index", np.nan)))
        else None,
        annual_trade_count=float(avg_annual_trade_count)
        if np.isfinite(float(avg_annual_trade_count))
        else None,
        backtest_years=(float(sample_days) / float(TRADING_DAYS_PER_YEAR))
        if sample_days > 0
        else None,
        score_sqn_weight=0.60,
        score_ulcer_weight=0.40,
    )
    r_stats_out = dict(trade_r_pack.get("statistics") or {})
    r_stats_out.pop("trade_system_score", None)
    trades_with_r = list(trade_r_pack.get("trades") or [])
    if not quick_mode:
        condition_bins_by_code: dict[str, dict[str, pd.Series]] = {}
        for c in codes:
            ck = str(c)
            condition_bins_by_code[ck] = {
                "momentum": _bucketize_momentum_series(
                    cond_momentum_df[ck].astype(float).reindex(nav.index)
                    if ck in cond_momentum_df.columns
                    else pd.Series(index=nav.index, dtype=float)
                ),
                "er": _bucketize_er_series(
                    cond_er_df[ck].astype(float).reindex(nav.index)
                    if ck in cond_er_df.columns
                    else pd.Series(index=nav.index, dtype=float)
                ),
                "vol_ratio": _bucketize_vol_ratio_series(
                    cond_vol_ratio_df[ck].astype(float).reindex(nav.index)
                    if ck in cond_vol_ratio_df.columns
                    else pd.Series(index=nav.index, dtype=float)
                ),
                "impulse": _bucketize_impulse_series(
                    cond_impulse_df[ck].astype(object).reindex(nav.index)
                    if ck in cond_impulse_df.columns
                    else pd.Series(index=nav.index, dtype=object)
                ),
            }
        trades_with_r = _attach_entry_condition_bins_to_trades(
            trades_with_r,
            condition_bins_by_code=condition_bins_by_code,
            dates=nav.index,
            default_code=None,
        )
    trades_by_code: dict[str, list[dict[str, Any]]] = {str(c): [] for c in codes}
    for tr in trades_with_r:
        c = str(tr.get("code") or "")
        if c in trades_by_code:
            trades_by_code[c].append(tr)
    mfe_r_distribution = build_trade_mfe_r_distribution(
        trade_pack.get("trades", []),
        close=close_qfq.reindex(index=nav.index, columns=codes).astype(float).ffill(),
        high=high_qfq_df.reindex(index=nav.index, columns=codes).astype(float).ffill(),
        atr=atr_risk_df.reindex(index=nav.index, columns=codes).astype(float),
        atr_mult=float(inp.atr_stop_n),
        default_code=None,
    )
    trade_stats = {
        "overall": _trade_stats_from_returns(trade_pack.get("returns", [])),
        "by_code": {
            str(c): _trade_stats_from_returns(
                (trade_pack.get("returns_by_code") or {}).get(str(c), [])
            )
            for c in codes
        },
        "trades": ([] if quick_mode else trades_with_r),
        "trades_by_code": (
            {str(c): [] for c in codes} if quick_mode else trades_by_code
        ),
        "mfe_r_distribution": mfe_r_distribution,
    }
    event_study = None
    if not quick_mode:
        event_study = compute_event_study(
            dates=nav.index,
            daily_returns=port_ret.reindex(nav.index).astype(float),
            entry_dates=entry_dates_from_exposure(
                w.sum(axis=1).reindex(nav.index).astype(float)
            ),
        )
    market_regime = build_market_regime_report(
        close=close_qfq.reindex(index=nav.index, columns=codes).astype(float),
        high=high_qfq_df.reindex(index=nav.index, columns=codes).astype(float),
        low=low_qfq_df.reindex(index=nav.index, columns=codes).astype(float),
        weights=w.reindex(index=nav.index, columns=codes).astype(float).fillna(0.0),
        asset_returns=ret_exec_day.reindex(index=nav.index, columns=codes)
        .astype(float)
        .fillna(0.0),
        strategy_returns=port_ret.reindex(nav.index).astype(float),
        ann_factor=TRADING_DAYS_PER_YEAR,
    )
    weekly = _period_returns(nav, "W-FRI")
    monthly = _period_returns(nav, "ME")
    quarterly = _period_returns(nav, "QE")
    yearly = _period_returns(nav, "YE")
    rolling_out = _rolling_pack(nav)

    total_triggers = int(
        sum(int((v or {}).get("trigger_count", 0)) for v in atr_stop_by_asset.values())
    )
    uniq_trigger_dates = sorted(
        {
            str(d)
            for v in atr_stop_by_asset.values()
            for d in (v or {}).get("trigger_dates", [])
            if str(d).strip()
        }
    )
    total_rtp_triggers = int(
        sum(int((v or {}).get("trigger_count", 0)) for v in rtp_by_asset.values())
    )
    total_bias_v_tp_triggers = int(
        sum(int((v or {}).get("trigger_count", 0)) for v in bias_v_tp_by_asset.values())
    )
    total_impulse_filter_blocked_entries = int(
        sum(
            int((v or {}).get("blocked_entry_count", 0))
            for v in impulse_filter_by_asset.values()
        )
    )
    total_impulse_filter_attempted_entries = int(
        sum(
            int((v or {}).get("attempted_entry_count", 0))
            for v in impulse_filter_by_asset.values()
        )
    )
    total_impulse_filter_allowed_entries = int(
        sum(
            int((v or {}).get("allowed_entry_count", 0))
            for v in impulse_filter_by_asset.values()
        )
    )
    total_impulse_filter_blocked_bull = int(
        sum(
            int((v or {}).get("blocked_entry_count_bull", 0))
            for v in impulse_filter_by_asset.values()
        )
    )
    total_impulse_filter_blocked_bear = int(
        sum(
            int((v or {}).get("blocked_entry_count_bear", 0))
            for v in impulse_filter_by_asset.values()
        )
    )
    total_impulse_filter_blocked_neutral = int(
        sum(
            int((v or {}).get("blocked_entry_count_neutral", 0))
            for v in impulse_filter_by_asset.values()
        )
    )
    total_er_filter_blocked_entries = int(
        sum(
            int((v or {}).get("blocked_entry_count", 0))
            for v in er_filter_by_asset.values()
        )
    )
    total_er_filter_attempted_entries = int(
        sum(
            int((v or {}).get("attempted_entry_count", 0))
            for v in er_filter_by_asset.values()
        )
    )
    total_er_filter_allowed_entries = int(
        sum(
            int((v or {}).get("allowed_entry_count", 0))
            for v in er_filter_by_asset.values()
        )
    )
    total_er_exit_filter_triggers = int(
        sum(
            int((v or {}).get("trigger_count", 0))
            for v in er_exit_filter_by_asset.values()
        )
    )
    total_vol_risk_adjust = int(
        sum(
            int((v or {}).get("vol_risk_adjust_total_count", 0))
            for v in vol_risk_adjust_by_asset.values()
        )
    )
    total_vol_risk_adjust_reduce_expand = int(
        sum(
            int((v or {}).get("vol_risk_adjust_reduce_on_expand_count", 0))
            for v in vol_risk_adjust_by_asset.values()
        )
    )
    total_vol_risk_adjust_increase_contract = int(
        sum(
            int((v or {}).get("vol_risk_adjust_increase_on_contract_count", 0))
            for v in vol_risk_adjust_by_asset.values()
        )
    )
    total_vol_risk_adjust_recover_expand = int(
        sum(
            int((v or {}).get("vol_risk_adjust_recover_from_expand_count", 0))
            for v in vol_risk_adjust_by_asset.values()
        )
    )
    total_vol_risk_adjust_recover_contract = int(
        sum(
            int((v or {}).get("vol_risk_adjust_recover_from_contract_count", 0))
            for v in vol_risk_adjust_by_asset.values()
        )
    )
    total_vol_risk_entry_reduce_expand = int(
        sum(
            int((v or {}).get("vol_risk_entry_state_reduce_on_expand_count", 0))
            for v in vol_risk_adjust_by_asset.values()
        )
    )
    total_vol_risk_entry_increase_contract = int(
        sum(
            int((v or {}).get("vol_risk_entry_state_increase_on_contract_count", 0))
            for v in vol_risk_adjust_by_asset.values()
        )
    )
    total_monthly_risk_budget_attempted_entries = int(
        (monthly_risk_budget_gate_stats or {}).get("attempted_entry_count", 0)
    )
    total_monthly_risk_budget_blocked_entries = int(
        (monthly_risk_budget_gate_stats or {}).get("blocked_entry_count", 0)
    )
    total_impulse_filter_blocked_rate = (
        float(
            total_impulse_filter_blocked_entries
            / total_impulse_filter_attempted_entries
        )
        if total_impulse_filter_attempted_entries > 0
        else 0.0
    )
    total_monthly_risk_budget_blocked_rate = (
        float(
            total_monthly_risk_budget_blocked_entries
            / total_monthly_risk_budget_attempted_entries
        )
        if total_monthly_risk_budget_attempted_entries > 0
        else 0.0
    )
    uniq_er_exit_trigger_dates = sorted(
        {
            str(d)
            for v in er_exit_filter_by_asset.values()
            for d in (v or {}).get("trigger_dates", [])
            if str(d).strip()
        }
    )
    total_rtp_tier_counts: dict[str, int] = {}
    for v in rtp_by_asset.values():
        one = dict(((v or {}).get("tier_trigger_counts") or {}))
        for k, n in one.items():
            kk = str(k).strip()
            if not kk:
                continue
            try:
                nn = int(float(n))
            except (TypeError, ValueError):
                nn = 0
            total_rtp_tier_counts[kk] = int(total_rtp_tier_counts.get(kk, 0) + nn)
    uniq_rtp_trigger_dates = sorted(
        {
            str(d)
            for v in rtp_by_asset.values()
            for d in (v or {}).get("trigger_dates", [])
            if str(d).strip()
        }
    )
    uniq_bias_v_tp_trigger_dates = sorted(
        {
            str(d)
            for v in bias_v_tp_by_asset.values()
            for d in (v or {}).get("trigger_dates", [])
            if str(d).strip()
        }
    )
    trade_stats["overall"]["atr_stop_trigger_count"] = int(total_triggers)
    trade_stats["overall"]["r_take_profit_trigger_count"] = int(total_rtp_triggers)
    trade_stats["overall"]["bias_v_take_profit_trigger_count"] = int(
        total_bias_v_tp_triggers
    )
    trade_stats["overall"]["r_take_profit_tier_trigger_counts"] = dict(
        total_rtp_tier_counts
    )
    trade_stats["overall"]["impulse_filter_blocked_entry_count"] = int(
        total_impulse_filter_blocked_entries
    )
    trade_stats["overall"]["impulse_filter_attempted_entry_count"] = int(
        total_impulse_filter_attempted_entries
    )
    trade_stats["overall"]["impulse_filter_allowed_entry_count"] = int(
        total_impulse_filter_allowed_entries
    )
    trade_stats["overall"]["impulse_filter_blocked_entry_rate"] = float(
        total_impulse_filter_blocked_rate
    )
    trade_stats["overall"]["impulse_filter_blocked_entry_count_bull"] = int(
        total_impulse_filter_blocked_bull
    )
    trade_stats["overall"]["impulse_filter_blocked_entry_count_bear"] = int(
        total_impulse_filter_blocked_bear
    )
    trade_stats["overall"]["impulse_filter_blocked_entry_count_neutral"] = int(
        total_impulse_filter_blocked_neutral
    )
    trade_stats["overall"]["er_filter_blocked_entry_count"] = int(
        total_er_filter_blocked_entries
    )
    trade_stats["overall"]["er_filter_attempted_entry_count"] = int(
        total_er_filter_attempted_entries
    )
    trade_stats["overall"]["er_filter_allowed_entry_count"] = int(
        total_er_filter_allowed_entries
    )
    trade_stats["overall"]["er_exit_filter_trigger_count"] = int(
        total_er_exit_filter_triggers
    )
    trade_stats["overall"]["vol_risk_adjust_total_count"] = int(total_vol_risk_adjust)
    trade_stats["overall"]["vol_risk_adjust_reduce_on_expand_count"] = int(
        total_vol_risk_adjust_reduce_expand
    )
    trade_stats["overall"]["vol_risk_adjust_increase_on_contract_count"] = int(
        total_vol_risk_adjust_increase_contract
    )
    trade_stats["overall"]["vol_risk_adjust_recover_from_expand_count"] = int(
        total_vol_risk_adjust_recover_expand
    )
    trade_stats["overall"]["vol_risk_adjust_recover_from_contract_count"] = int(
        total_vol_risk_adjust_recover_contract
    )
    trade_stats["overall"]["vol_risk_entry_state_reduce_on_expand_count"] = int(
        total_vol_risk_entry_reduce_expand
    )
    trade_stats["overall"]["vol_risk_entry_state_increase_on_contract_count"] = int(
        total_vol_risk_entry_increase_contract
    )
    trade_stats["overall"]["vol_risk_overcap_scale_count"] = int(
        risk_budget_overcap_scale_count
    )
    trade_stats["overall"]["vol_risk_overcap_skip_entry_decision_count"] = int(
        risk_budget_overcap_skip_decision_count
    )
    trade_stats["overall"]["vol_risk_overcap_skip_entry_episode_count"] = int(
        risk_budget_overcap_skip_episode_count
    )
    trade_stats["overall"]["vol_risk_overcap_replace_entry_count"] = int(
        risk_budget_overcap_replace_count
    )
    trade_stats["overall"]["vol_risk_overcap_replace_out_count"] = int(
        risk_budget_overcap_replace_count
    )
    trade_stats["overall"]["vol_risk_overcap_replace_in_count"] = int(
        risk_budget_overcap_replace_count
    )
    trade_stats["overall"]["vol_risk_overcap_leverage_usage_count"] = int(
        risk_budget_overcap_leverage_usage_count
    )
    trade_stats["overall"]["vol_risk_overcap_leverage_max_multiple"] = float(
        risk_budget_overcap_leverage_max_multiple
    )
    trade_stats["overall"]["monthly_risk_budget_attempted_entry_count"] = int(
        total_monthly_risk_budget_attempted_entries
    )
    trade_stats["overall"]["monthly_risk_budget_blocked_entry_count"] = int(
        total_monthly_risk_budget_blocked_entries
    )
    trade_stats["overall"]["monthly_risk_budget_blocked_entry_rate"] = float(
        total_monthly_risk_budget_blocked_rate
    )
    for c in codes:
        code_key = str(c)
        one = trade_stats["by_code"].get(code_key) or {}
        one["atr_stop_trigger_count"] = int(
            (atr_stop_by_asset.get(code_key) or {}).get("trigger_count", 0)
        )
        one["r_take_profit_trigger_count"] = int(
            (rtp_by_asset.get(code_key) or {}).get("trigger_count", 0)
        )
        one["bias_v_take_profit_trigger_count"] = int(
            (bias_v_tp_by_asset.get(code_key) or {}).get("trigger_count", 0)
        )
        one["r_take_profit_tier_trigger_counts"] = dict(
            ((rtp_by_asset.get(code_key) or {}).get("tier_trigger_counts") or {})
        )
        one["impulse_filter_blocked_entry_count"] = int(
            (impulse_filter_by_asset.get(code_key) or {}).get("blocked_entry_count", 0)
        )
        one["impulse_filter_attempted_entry_count"] = int(
            (impulse_filter_by_asset.get(code_key) or {}).get(
                "attempted_entry_count", 0
            )
        )
        one["impulse_filter_allowed_entry_count"] = int(
            (impulse_filter_by_asset.get(code_key) or {}).get("allowed_entry_count", 0)
        )
        one["impulse_filter_blocked_entry_rate"] = (
            float(
                one["impulse_filter_blocked_entry_count"]
                / one["impulse_filter_attempted_entry_count"]
            )
            if int(one["impulse_filter_attempted_entry_count"]) > 0
            else 0.0
        )
        one["impulse_filter_blocked_entry_count_bull"] = int(
            (impulse_filter_by_asset.get(code_key) or {}).get(
                "blocked_entry_count_bull", 0
            )
        )
        one["impulse_filter_blocked_entry_count_bear"] = int(
            (impulse_filter_by_asset.get(code_key) or {}).get(
                "blocked_entry_count_bear", 0
            )
        )
        one["impulse_filter_blocked_entry_count_neutral"] = int(
            (impulse_filter_by_asset.get(code_key) or {}).get(
                "blocked_entry_count_neutral", 0
            )
        )
        one["er_filter_blocked_entry_count"] = int(
            (er_filter_by_asset.get(code_key) or {}).get("blocked_entry_count", 0)
        )
        one["er_filter_attempted_entry_count"] = int(
            (er_filter_by_asset.get(code_key) or {}).get("attempted_entry_count", 0)
        )
        one["er_filter_allowed_entry_count"] = int(
            (er_filter_by_asset.get(code_key) or {}).get("allowed_entry_count", 0)
        )
        one["er_exit_filter_trigger_count"] = int(
            (er_exit_filter_by_asset.get(code_key) or {}).get("trigger_count", 0)
        )
        one["vol_risk_adjust_total_count"] = int(
            (vol_risk_adjust_by_asset.get(code_key) or {}).get(
                "vol_risk_adjust_total_count", 0
            )
        )
        one["vol_risk_adjust_reduce_on_expand_count"] = int(
            (vol_risk_adjust_by_asset.get(code_key) or {}).get(
                "vol_risk_adjust_reduce_on_expand_count", 0
            )
        )
        one["vol_risk_adjust_increase_on_contract_count"] = int(
            (vol_risk_adjust_by_asset.get(code_key) or {}).get(
                "vol_risk_adjust_increase_on_contract_count", 0
            )
        )
        one["vol_risk_adjust_recover_from_expand_count"] = int(
            (vol_risk_adjust_by_asset.get(code_key) or {}).get(
                "vol_risk_adjust_recover_from_expand_count", 0
            )
        )
        one["vol_risk_adjust_recover_from_contract_count"] = int(
            (vol_risk_adjust_by_asset.get(code_key) or {}).get(
                "vol_risk_adjust_recover_from_contract_count", 0
            )
        )
        one["vol_risk_entry_state_reduce_on_expand_count"] = int(
            (vol_risk_adjust_by_asset.get(code_key) or {}).get(
                "vol_risk_entry_state_reduce_on_expand_count", 0
            )
        )
        one["vol_risk_entry_state_increase_on_contract_count"] = int(
            (vol_risk_adjust_by_asset.get(code_key) or {}).get(
                "vol_risk_entry_state_increase_on_contract_count", 0
            )
        )
        one["vol_risk_overcap_scale_count"] = int(
            risk_budget_overcap_scale_by_code.get(code_key, 0)
        )
        one["vol_risk_overcap_skip_entry_decision_count"] = int(
            risk_budget_overcap_skip_decision_by_code.get(code_key, 0)
        )
        one["vol_risk_overcap_skip_entry_episode_count"] = int(
            risk_budget_overcap_skip_episode_by_code.get(code_key, 0)
        )
        one["vol_risk_overcap_replace_out_count"] = int(
            risk_budget_overcap_replace_out_by_code.get(code_key, 0)
        )
        one["vol_risk_overcap_replace_in_count"] = int(
            risk_budget_overcap_replace_in_by_code.get(code_key, 0)
        )
        one["vol_risk_overcap_leverage_usage_count"] = int(
            risk_budget_overcap_leverage_usage_by_code.get(code_key, 0)
        )
        one["vol_risk_overcap_leverage_max_multiple"] = float(
            risk_budget_overcap_leverage_max_multiple_by_code.get(code_key, 0.0)
        )
        one["monthly_risk_budget_attempted_entry_count"] = int(
            (
                (monthly_risk_budget_gate_stats or {}).get(
                    "attempted_entry_count_by_code"
                )
                or {}
            ).get(code_key, 0)
        )
        one["monthly_risk_budget_blocked_entry_count"] = int(
            (
                (monthly_risk_budget_gate_stats or {}).get(
                    "blocked_entry_count_by_code"
                )
                or {}
            ).get(code_key, 0)
        )
        one["monthly_risk_budget_blocked_entry_rate"] = (
            float(
                one["monthly_risk_budget_blocked_entry_count"]
                / one["monthly_risk_budget_attempted_entry_count"]
            )
            if int(one["monthly_risk_budget_attempted_entry_count"]) > 0
            else 0.0
        )
        trade_stats["by_code"][code_key] = one
    if not quick_mode:
        trade_stats["entry_condition_stats"] = {
            "scope": "closed_trades_only",
            "signal_day_basis": "signal_day_before_entry_execution",
            "quasi_causal_method": "uplift + two_proportion_z / welch_t_normal_approx + BH",
            "strong_causal_method": "uplift + stratified_permutation + BH",
            "overall": _build_entry_condition_stats(
                trades_with_r, by_code=False, n_perm=300, seed=20260410
            ),
            "by_code": {
                str(c): _build_entry_condition_stats(
                    trades_by_code.get(str(c), []),
                    by_code=True,
                    n_perm=200,
                    seed=20260410,
                )
                for c in codes
            },
        }
    m_strat["r_take_profit_tier_trigger_counts"] = dict(total_rtp_tier_counts)
    m_strat["r_take_profit_trigger_count"] = int(total_rtp_triggers)
    m_strat["bias_v_take_profit_trigger_count"] = int(total_bias_v_tp_triggers)
    m_strat["atr_stop_trigger_count"] = int(total_triggers)
    m_strat["impulse_filter_blocked_entry_count"] = int(
        total_impulse_filter_blocked_entries
    )
    m_strat["impulse_filter_blocked_entry_count_bull"] = int(
        total_impulse_filter_blocked_bull
    )
    m_strat["impulse_filter_blocked_entry_count_bear"] = int(
        total_impulse_filter_blocked_bear
    )
    m_strat["impulse_filter_blocked_entry_count_neutral"] = int(
        total_impulse_filter_blocked_neutral
    )
    m_strat["vol_risk_overcap_scale_count"] = int(risk_budget_overcap_scale_count)
    m_strat["vol_risk_overcap_skip_entry_decision_count"] = int(
        risk_budget_overcap_skip_decision_count
    )
    m_strat["vol_risk_overcap_skip_entry_episode_count"] = int(
        risk_budget_overcap_skip_episode_count
    )
    m_strat["vol_risk_overcap_replace_entry_count"] = int(
        risk_budget_overcap_replace_count
    )
    m_strat["vol_risk_overcap_leverage_usage_count"] = int(
        risk_budget_overcap_leverage_usage_count
    )
    m_strat["vol_risk_overcap_leverage_max_multiple"] = float(
        risk_budget_overcap_leverage_max_multiple
    )
    m_strat["monthly_risk_budget_blocked_entry_count"] = int(
        (monthly_risk_budget_gate_stats or {}).get("blocked_entry_count", 0)
    )
    ext_dates = sorted(
        {str(e.get("date")) for e in ext_events if str(e.get("date", "")).strip()}
    )
    skip_dates = sorted(
        {str(e.get("date")) for e in skip_events if str(e.get("date", "")).strip()}
    )
    ext_over_weight_count = int(
        sum(1 for e in ext_events if bool(e.get("over_weight")))
    )
    ext_over_count_count = int(sum(1 for e in ext_events if bool(e.get("over_count"))))
    ext_over_both_count = int(
        sum(
            1
            for e in ext_events
            if bool(e.get("over_weight")) and bool(e.get("over_count"))
        )
    )
    skip_over_weight_count = int(
        sum(1 for e in skip_events if bool(e.get("over_weight")))
    )
    skip_over_count_count = int(
        sum(1 for e in skip_events if bool(e.get("over_count")))
    )
    skip_over_both_count = int(
        sum(
            1
            for e in skip_events
            if bool(e.get("over_weight")) and bool(e.get("over_count"))
        )
    )
    exposure_eff = w.sum(axis=1).astype(float)
    group_filter_enabled_days = int(
        sum(
            1
            for h in holdings
            if bool(((h or {}).get("group_filter") or {}).get("enabled"))
        )
    )
    group_filter_effective_days = int(
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
    usage_enabled = bool(ps in {"fixed_ratio", "risk_budget"})
    risk_budget_overcap_daily = [
        {
            "date": str(k),
            "scale": int((v or {}).get("scale", 0)),
            "skip_entry": int((v or {}).get("skip_entry", 0)),
            "replace_entry": int((v or {}).get("replace_entry", 0)),
            "leverage_entry": int((v or {}).get("leverage_entry", 0)),
            "leverage_multiple_max": float(
                (v or {}).get("leverage_multiple_max", 0.0) or 0.0
            ),
        }
        for k, v in sorted(
            (risk_budget_overcap_daily_counts or {}).items(), key=lambda x: str(x[0])
        )
    ]
    usage_quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    if usage_enabled and (len(exposure_eff) > 0):
        util_min = float(exposure_eff.min())
        util_max = float(exposure_eff.max())
        util_mean = float(exposure_eff.mean())
        util_q = {
            f"p{int(q * 100):02d}": float(exposure_eff.quantile(q))
            for q in usage_quantiles
        }
        util_over100_days = int((exposure_eff > 1.0 + 1e-12).sum())
        util_under100_days = int((exposure_eff < 1.0 - 1e-12).sum())
    else:
        util_min = float("nan")
        util_max = float("nan")
        util_mean = float("nan")
        util_q = {f"p{int(q * 100):02d}": float("nan") for q in usage_quantiles}
        util_over100_days = 0
        util_under100_days = 0
    entry_exec_price_with_slippage_by_asset: dict[str, float] = {}
    for c in codes:
        one = _latest_entry_exec_price_with_slippage(
            effective_weight=w[c].reindex(nav.index).astype(float),
            exec_price_series=px_exec_slip[c].reindex(nav.index).ffill().astype(float),
            slippage_spread=float(inp.slippage_rate),
        )
        if one is not None:
            entry_exec_price_with_slippage_by_asset[str(c)] = float(one)

    portfolio_next_plan: dict[str, Any] = {
        "decision_date": (str(nav.index[-1].date()) if len(nav.index) else None),
        "entry_exec_price_with_slippage_by_asset": entry_exec_price_with_slippage_by_asset,
        "position_sizing": str(ps),
        "notes": (
            "effective_weights_last_close：决策日收盘时持仓权重（含盘中止损等）；"
            "decision_weights_next_exec：下一执行时点（通常为下一交易日开盘）目标权重。"
            "二者之差包含波动率目标(vol_target)、风险预算及波动状态机等风控带来的调仓。"
            "若仅波动状态切换而风险预算名义目标未变，可能出现权重差分为零。"
        ),
    }
    if len(nav.index) > 0:
        ld = nav.index[-1]
        w_eff_last = w.loc[ld]
        w_dec_last = w_decision.loc[ld]
        portfolio_next_plan["effective_weights_last_close"] = {
            str(c): float(w_eff_last[c]) for c in codes
        }
        portfolio_next_plan["decision_weights_next_exec"] = {
            str(c): float(w_dec_last[c]) for c in codes
        }
        deltas_m: dict[str, float] = {}
        for c in codes:
            du = float(w_dec_last[c]) - float(w_eff_last[c])
            if abs(du) > 1e-14:
                deltas_m[str(c)] = du
        portfolio_next_plan["weight_delta_next_exec_by_code"] = deltas_m

        if str(ps) == "vol_target":
            active_codes_vt = [str(c) for c in codes if float(w_dec_last[c]) > 1e-12]
            inv_vt: dict[str, float] = {}
            ann_by: dict[str, float] = {}
            for c in active_codes_vt:
                try:
                    av = float(vol_ann.loc[ld, c])
                except (TypeError, ValueError, KeyError):
                    av = float("nan")
                ann_by[str(c)] = av
                if (not np.isfinite(av)) or av <= 0:
                    inv_vt[c] = 0.0
                else:
                    inv_vt[c] = 1.0 / av
            den_vt = float(sum(inv_vt.values()))
            port_vol_est = float("nan")
            scale_vt = 1.0
            if den_vt > 0 and active_codes_vt:
                raw_vt = {
                    c: float(inv_vt[c]) / den_vt
                    for c in active_codes_vt
                    if float(inv_vt.get(c, 0.0)) > 0.0
                }
                s_var = 0.0
                for c in raw_vt:
                    vx = float(vol_ann.loc[ld, c])
                    if np.isfinite(vx):
                        s_var += float(raw_vt[c] ** 2) * float(vx**2)
                port_vol_est = float(np.sqrt(s_var)) if s_var > 0 else float("nan")
                vt_ann = float(inp.vol_target_ann)
                scale_vt = (
                    1.0
                    if (not np.isfinite(port_vol_est)) or port_vol_est <= 1e-12
                    else min(1.0, vt_ann / port_vol_est)
                )
            portfolio_next_plan["vol_target_snapshot"] = {
                "vol_target_ann": float(inp.vol_target_ann),
                "vol_window": int(inp.vol_window),
                "portfolio_vol_annualized_est": port_vol_est,
                "gross_leverage_scalar": float(scale_vt),
                "by_code_ann_vol": ann_by,
            }

        if str(ps) == "risk_budget" and bool(vol_regime_risk_mgmt_enabled):
            by_c_vm: dict[str, dict[str, Any]] = {}
            for c in codes:
                af = (
                    float(atr_ratio_fast_df.loc[ld, c])
                    if (
                        c in atr_ratio_fast_df.columns and ld in atr_ratio_fast_df.index
                    )
                    else float("nan")
                )
                sl = (
                    float(atr_ratio_slow_df.loc[ld, c])
                    if (
                        c in atr_ratio_slow_df.columns and ld in atr_ratio_slow_df.index
                    )
                    else float("nan")
                )
                ratio_vm = (
                    (af / sl)
                    if (np.isfinite(af) and np.isfinite(sl) and sl > 0.0)
                    else float("nan")
                )
                by_c_vm[str(c)] = {
                    "vol_regime_state": str(rb_state_by_code.get(str(c), "FLAT")),
                    "atr_fast": af,
                    "atr_slow": sl,
                    "atr_fast_over_slow": ratio_vm,
                }
            portfolio_next_plan["risk_budget_volatility_regime"] = {
                "enabled": True,
                "fast_atr_window": int(vol_ratio_fast_atr_window),
                "slow_atr_window": int(vol_ratio_slow_atr_window),
                "expand_threshold": float(vol_ratio_expand_threshold),
                "contract_threshold": float(vol_ratio_contract_threshold),
                "normal_threshold": float(vol_ratio_normal_threshold),
                "by_code": by_c_vm,
            }

    return {
        "meta": {
            "type": "trend_portfolio_backtest",
            "codes": codes,
            "start": inp.start.strftime("%Y%m%d"),
            "end": inp.end.strftime("%Y%m%d"),
            "strategy": strat,
            "strategy_execution_description": TREND_STRATEGY_EXECUTION_DESCRIPTIONS.get(
                strat, ""
            ),
            "params": {
                "position_sizing": ps,
                "vol_window": int(inp.vol_window),
                "vol_target_ann": float(inp.vol_target_ann),
                "fixed_pos_ratio": float(fixed_ratio),
                "fixed_overcap_policy": str(fixed_overcap_policy),
                "fixed_max_holdings": int(fixed_max_holding_n),
                "risk_budget_atr_window": int(risk_budget_atr_window),
                "risk_budget_pct": float(risk_budget_pct),
                "risk_budget_overcap_policy": str(risk_budget_overcap_policy),
                "risk_budget_max_leverage_multiple": float(
                    risk_budget_max_leverage_multiple
                ),
                "vol_regime_risk_mgmt_enabled": bool(vol_regime_risk_mgmt_enabled),
                "vol_ratio_fast_atr_window": int(vol_ratio_fast_atr_window),
                "vol_ratio_slow_atr_window": int(vol_ratio_slow_atr_window),
                "vol_ratio_expand_threshold": float(vol_ratio_expand_threshold),
                "vol_ratio_contract_threshold": float(vol_ratio_contract_threshold),
                "vol_ratio_normal_threshold": float(vol_ratio_normal_threshold),
                "selection_mode": "all_active_candidates",
                "group_enforce": bool(group_enforce),
                "group_pick_policy": str(group_pick_policy),
                "group_max_holdings": int(group_max_holdings),
                "asset_groups": group_map,
                "er_filter": bool(er_filter),
                "er_window": int(er_window),
                "er_threshold": float(er_threshold),
                "impulse_entry_filter": bool(impulse_entry_filter),
                "impulse_allow_bull": bool(impulse_allow_bull),
                "impulse_allow_bear": bool(impulse_allow_bear),
                "impulse_allow_neutral": bool(impulse_allow_neutral),
                "er_exit_filter": bool(er_exit_filter),
                "er_exit_window": int(er_exit_window),
                "er_exit_threshold": float(er_exit_threshold),
                "quick_mode": bool(quick_mode),
                "ma_type": ma_type,
                "kama_er_window": int(kama_er_window),
                "kama_fast_window": int(kama_fast_window),
                "kama_slow_window": int(kama_slow_window),
                "kama_std_window": int(kama_std_window),
                "kama_std_coef": float(kama_std_coef),
                "mom_lookback": int(inp.mom_lookback),
                "tsmom_entry_threshold": float(inp.tsmom_entry_threshold),
                "tsmom_exit_threshold": float(inp.tsmom_exit_threshold),
                "random_hold_days": int(getattr(inp, "random_hold_days", 20)),
                "random_seed": (
                    None
                    if getattr(inp, "random_seed", 42) is None
                    else int(getattr(inp, "random_seed", 42))
                ),
                "atr_stop_mode": str(atr_mode),
                "atr_stop_atr_basis": str(atr_basis),
                "atr_stop_reentry_mode": str(atr_reentry_mode),
                "atr_stop_window": int(inp.atr_stop_window),
                "atr_stop_n": float(inp.atr_stop_n),
                "atr_stop_m": float(inp.atr_stop_m),
                "r_take_profit_enabled": bool(rtp_enabled),
                "r_take_profit_reentry_mode": str(rtp_reentry_mode),
                "r_take_profit_tiers": rtp_tiers,
                "bias_v_take_profit_enabled": bool(bias_v_tp_enabled),
                "bias_v_take_profit_reentry_mode": str(bias_v_tp_reentry_mode),
                "bias_v_ma_window": int(bias_v_tp_ma_window),
                "bias_v_atr_window": int(bias_v_tp_atr_window),
                "bias_v_take_profit_threshold": float(bias_v_tp_threshold),
                "monthly_risk_budget_enabled": bool(monthly_risk_budget_enabled),
                "monthly_risk_budget_pct": float(monthly_risk_budget_pct),
                "monthly_risk_budget_include_new_trade_risk": bool(
                    monthly_risk_budget_include_new_trade_risk
                ),
                "exec_price": str(ep),
                "cost_bps": float(inp.cost_bps),
                "slippage_rate": float(inp.slippage_rate),
            },
        },
        "nav": {
            "dates": nav.index.date.astype(str).tolist(),
            "series": {
                "STRAT": nav.astype(float).tolist(),
                "BUY_HOLD_EW": bench_nav.astype(float).tolist(),
                "EXCESS": ex_nav.astype(float).tolist(),
            },
        },
        "weights": {
            "dates": w.index.date.astype(str).tolist(),
            "series": {c: w[c].astype(float).tolist() for c in codes},
        },
        "weights_decision": {
            "dates": w_decision.index.date.astype(str).tolist(),
            "series": {c: w_decision[c].astype(float).tolist() for c in codes},
        },
        "asset_nav_exec": {
            "dates": ret_exec_day.index.date.astype(str).tolist(),
            "series": {
                c: (1.0 + ret_exec_day[c].astype(float))
                .cumprod()
                .astype(float)
                .tolist()
                for c in codes
            },
        },
        "next_plan": portfolio_next_plan,
        "holdings": holdings,
        "period_returns": {
            "weekly": weekly.to_dict(orient="records"),
            "monthly": monthly.to_dict(orient="records"),
            "quarterly": quarterly.to_dict(orient="records"),
            "yearly": yearly.to_dict(orient="records"),
        },
        "rolling": rolling_out,
        "attribution": attribution,
        "trade_statistics": trade_stats,
        "r_statistics": r_stats_out,
        "event_study": event_study,
        "market_regime": market_regime,
        "return_decomposition": return_decomposition,
        "metrics": {"strategy": m_strat, "benchmark": m_bench, "excess": m_ex},
        "risk_controls": {
            "atr_stop": {
                "enabled": bool(atr_mode != "none"),
                "mode": str(atr_mode),
                "atr_basis": str(atr_basis),
                "reentry_mode": str(atr_reentry_mode),
                "trigger_count": total_triggers,
                "trigger_days": int(len(uniq_trigger_dates)),
                "first_trigger_date": (
                    uniq_trigger_dates[0] if uniq_trigger_dates else None
                ),
                "last_trigger_date": (
                    uniq_trigger_dates[-1] if uniq_trigger_dates else None
                ),
                "trigger_dates": uniq_trigger_dates[:200],
                "by_asset": atr_stop_by_asset,
            },
            "r_take_profit": {
                "enabled": bool(rtp_enabled),
                "reentry_mode": str(rtp_reentry_mode),
                "tiers": rtp_tiers,
                "trigger_count": total_rtp_triggers,
                "tier_trigger_counts": dict(total_rtp_tier_counts),
                "trigger_days": int(len(uniq_rtp_trigger_dates)),
                "first_trigger_date": (
                    uniq_rtp_trigger_dates[0] if uniq_rtp_trigger_dates else None
                ),
                "last_trigger_date": (
                    uniq_rtp_trigger_dates[-1] if uniq_rtp_trigger_dates else None
                ),
                "trigger_dates": uniq_rtp_trigger_dates[:200],
                "fallback_mode_used": bool((atr_mode == "none") and rtp_enabled),
                "initial_r_mode": (
                    "atr_stop" if atr_mode != "none" else "virtual_atr_fallback"
                ),
                "by_asset": rtp_by_asset,
            },
            "bias_v_take_profit": {
                "enabled": bool(bias_v_tp_enabled),
                "reentry_mode": str(bias_v_tp_reentry_mode),
                "ma_window": int(bias_v_tp_ma_window),
                "atr_window": int(bias_v_tp_atr_window),
                "threshold": float(bias_v_tp_threshold),
                "trigger_count": int(total_bias_v_tp_triggers),
                "trigger_days": int(len(uniq_bias_v_tp_trigger_dates)),
                "first_trigger_date": (
                    uniq_bias_v_tp_trigger_dates[0]
                    if uniq_bias_v_tp_trigger_dates
                    else None
                ),
                "last_trigger_date": (
                    uniq_bias_v_tp_trigger_dates[-1]
                    if uniq_bias_v_tp_trigger_dates
                    else None
                ),
                "trigger_dates": uniq_bias_v_tp_trigger_dates[:200],
                "by_asset": bias_v_tp_by_asset,
            },
            "er_exit_filter": {
                "enabled": bool(er_exit_filter),
                "window": int(er_exit_window),
                "threshold": float(er_exit_threshold),
                "trigger_count": int(total_er_exit_filter_triggers),
                "trigger_days": int(len(uniq_er_exit_trigger_dates)),
                "first_trigger_date": (
                    uniq_er_exit_trigger_dates[0]
                    if uniq_er_exit_trigger_dates
                    else None
                ),
                "last_trigger_date": (
                    uniq_er_exit_trigger_dates[-1]
                    if uniq_er_exit_trigger_dates
                    else None
                ),
                "trigger_dates": uniq_er_exit_trigger_dates[:200],
                "by_asset": er_exit_filter_by_asset,
            },
            "vol_regime_risk_mgmt": {
                "enabled": bool(ps == "risk_budget" and vol_regime_risk_mgmt_enabled),
                "fast_atr_window": int(vol_ratio_fast_atr_window),
                "slow_atr_window": int(vol_ratio_slow_atr_window),
                "expand_threshold": float(vol_ratio_expand_threshold),
                "contract_threshold": float(vol_ratio_contract_threshold),
                "normal_threshold": float(vol_ratio_normal_threshold),
                "adjust_total_count": int(total_vol_risk_adjust),
                "adjust_reduce_on_expand_count": int(
                    total_vol_risk_adjust_reduce_expand
                ),
                "adjust_increase_on_contract_count": int(
                    total_vol_risk_adjust_increase_contract
                ),
                "adjust_recover_from_expand_count": int(
                    total_vol_risk_adjust_recover_expand
                ),
                "adjust_recover_from_contract_count": int(
                    total_vol_risk_adjust_recover_contract
                ),
                "entry_state_reduce_on_expand_count": int(
                    total_vol_risk_entry_reduce_expand
                ),
                "entry_state_increase_on_contract_count": int(
                    total_vol_risk_entry_increase_contract
                ),
                "overcap_scale_count": int(risk_budget_overcap_scale_count),
                "overcap_policy": str(risk_budget_overcap_policy),
                "overcap_max_leverage_multiple": float(
                    risk_budget_max_leverage_multiple
                ),
                "overcap_skip_entry_decision_count": int(
                    risk_budget_overcap_skip_decision_count
                ),
                "overcap_skip_entry_episode_count": int(
                    risk_budget_overcap_skip_episode_count
                ),
                "overcap_skip_entry_decision_count_by_code": dict(
                    risk_budget_overcap_skip_decision_by_code
                ),
                "overcap_skip_entry_episode_count_by_code": dict(
                    risk_budget_overcap_skip_episode_by_code
                ),
                "overcap_replace_entry_count": int(risk_budget_overcap_replace_count),
                "overcap_replace_out_count_by_code": dict(
                    risk_budget_overcap_replace_out_by_code
                ),
                "overcap_replace_in_count_by_code": dict(
                    risk_budget_overcap_replace_in_by_code
                ),
                "overcap_leverage_usage_count": int(
                    risk_budget_overcap_leverage_usage_count
                ),
                "overcap_leverage_max_multiple": float(
                    risk_budget_overcap_leverage_max_multiple
                ),
                "overcap_leverage_usage_count_by_code": dict(
                    risk_budget_overcap_leverage_usage_by_code
                ),
                "overcap_leverage_max_multiple_by_code": dict(
                    risk_budget_overcap_leverage_max_multiple_by_code
                ),
                "overcap_daily_counts": risk_budget_overcap_daily,
                "by_asset": vol_risk_adjust_by_asset,
            },
            "monthly_risk_budget": monthly_risk_budget_gate_stats,
            "position_extension": {
                "enabled": bool(
                    ps == "fixed_ratio" and fixed_overcap_policy == "extend"
                ),
                "position_sizing": str(ps),
                "fixed_pos_ratio": float(fixed_ratio),
                "overcap_policy": str(fixed_overcap_policy),
                "fixed_max_holdings": int(fixed_max_holding_n),
                "extension_count": int(len(ext_events)),
                "extension_over_weight_count": int(ext_over_weight_count),
                "extension_over_count_count": int(ext_over_count_count),
                "extension_over_both_count": int(ext_over_both_count),
                "extension_days": int(len(ext_dates)),
                "first_extension_date": (ext_dates[0] if ext_dates else None),
                "last_extension_date": (ext_dates[-1] if ext_dates else None),
                "extension_dates": ext_dates[:200],
                "extensions": ext_events[:200],
                "skipped_count": int(len(skip_events)),
                "skipped_over_weight_count": int(skip_over_weight_count),
                "skipped_over_count_count": int(skip_over_count_count),
                "skipped_over_both_count": int(skip_over_both_count),
                "skipped_days": int(len(skip_dates)),
                "first_skipped_date": (skip_dates[0] if skip_dates else None),
                "last_skipped_date": (skip_dates[-1] if skip_dates else None),
                "skipped_dates": skip_dates[:200],
                "skipped": skip_events[:200],
            },
            "position_usage": {
                "enabled": bool(usage_enabled),
                "position_sizing": str(ps),
                "cash_as_residual": bool(
                    usage_enabled
                ),  # <100% exposure is treated as cash at 0 return
                "min_exposure": (
                    None
                    if (not usage_enabled or (not np.isfinite(util_min)))
                    else float(util_min)
                ),
                "max_exposure": (
                    None
                    if (not usage_enabled or (not np.isfinite(util_max)))
                    else float(util_max)
                ),
                "mean_exposure": (
                    None
                    if (not usage_enabled or (not np.isfinite(util_mean)))
                    else float(util_mean)
                ),
                "quantiles": {
                    k: (
                        None
                        if (not usage_enabled or (not np.isfinite(v)))
                        else float(v)
                    )
                    for k, v in util_q.items()
                },
                "over_100pct_days": int(util_over100_days),
                "under_100pct_days": int(util_under100_days),
            },
            "group_filter": {
                "enabled": bool(group_enforce),
                "policy": str(group_pick_policy),
                "max_holdings_per_group": int(group_max_holdings),
                "decision_segments_with_group_filter": int(group_filter_enabled_days),
                "decision_segments_effective": int(group_filter_effective_days),
            },
        },
    }
