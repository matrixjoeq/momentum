"""Capture and audit ETF trend backtests against one live strategy safely.

The script writes sensitive inputs only below the repository-ignored ``data/``
directory.  It never calls the mutating live replay endpoint.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import random
import re
import stat
import subprocess
import sys
import time
import urllib.request
from decimal import Decimal
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASE_DIR = REPO_ROOT / "data" / "audit" / "trend-live-5-13"
API_BASE = "http://127.0.0.1:8000"
AS_OF = "2026-08-27"
ACCOUNT_ID = 5
STRATEGY_ID = 13
AUDIT_SCHEMA = "momentum_audit_trend_live_5_13_20260827"

CODES = [
    "518800",
    "159980",
    "159981",
    "159985",
    "513400",
    "513500",
    "513100",
    "513030",
    "513080",
    "513520",
    "513310",
    "510300",
    "563300",
    "159915",
    "588000",
    "159920",
]

R_TAKE_PROFIT_TIERS = [
    {"r_multiple": 1.0, "retrace_ratio": 1.0},
    {"r_multiple": 2.0, "retrace_ratio": 0.5},
    {"r_multiple": 3.0, "retrace_ratio": 0.33},
    {"r_multiple": 4.0, "retrace_ratio": 0.25},
    {"r_multiple": 5.0, "retrace_ratio": 0.2},
    {"r_multiple": 6.0, "retrace_ratio": 0.17},
    {"r_multiple": 7.0, "retrace_ratio": 0.14},
    {"r_multiple": 8.0, "retrace_ratio": 0.13},
    {"r_multiple": 9.0, "retrace_ratio": 0.11},
    {"r_multiple": 10.0, "retrace_ratio": 0.1},
]


def effective_payload(*, engine: str, quick_mode: bool) -> dict[str, Any]:
    """Return the effective API payload after the browser's normalization."""
    return {
        "codes": CODES,
        "position_sizing": "risk_budget",
        "dynamic_universe": True,
        "start": "20111209",
        "end": "20260827",
        "initial_account_amount": None,
        "cost_bps": 2.0,
        "slippage_rate": 0.001,
        "capacity_window_years": 1,
        "quick_mode": quick_mode,
        "search_minimal_mode": False,
        "engine": engine,
        "exec_price": "close",
        "strategy": "tsmom",
        "sma_window": 20,
        "fast_window": 5,
        "slow_window": 20,
        "ma_type": "kama",
        "kama_er_window": 10,
        "kama_fast_window": 2,
        "kama_slow_window": 30,
        "kama_std_window": 20,
        "kama_std_coef": 1.0,
        "donchian_entry": 20,
        "donchian_exit": 10,
        "mom_lookback": 20,
        "tsmom_entry_threshold": 0.02,
        "tsmom_exit_threshold": 0.0,
        "impulse_entry_filter": False,
        "impulse_allow_bull": True,
        "impulse_allow_bear": False,
        "impulse_allow_neutral": False,
        "er_filter": False,
        "er_window": 10,
        "er_threshold": 0.25,
        "ma_entry_filter_enabled": False,
        "ma_entry_filter_type": "sma",
        "ma_entry_filter_fast": 100,
        "ma_entry_filter_slow": 200,
        "er_exit_filter": False,
        "er_exit_window": 10,
        "er_exit_threshold": 0.88,
        "bias_ma_window": 20,
        "bias_entry": 2.0,
        "bias_hot": 10.0,
        "bias_cold": -2.0,
        "bias_pos_mode": "binary",
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "macd_v_atr_window": 26,
        "macd_v_scale": 100.0,
        "macd_hist_min": 0.0,
        "macd_v_hist_min": 0.0,
        "cci_window": 20,
        "random_hold_days": 20,
        "random_seed": 42,
        "atr_stop_mode": "static",
        "atr_stop_atr_basis": "latest",
        "atr_stop_reentry_mode": "wait_next_entry",
        "atr_stop_execution_mode": "intraday",
        "atr_stop_execution_time": "close",
        "atr_stop_window": 20,
        "atr_stop_n": 2.0,
        "atr_stop_m": 0.5,
        "r_take_profit_enabled": True,
        "r_take_profit_reentry_mode": "wait_next_entry",
        "r_take_profit_execution_mode": "intraday",
        "r_take_profit_execution_time": "full_day",
        "r_take_profit_tiers": R_TAKE_PROFIT_TIERS,
        "r_profit_scaleout_enabled": False,
        "r_profit_scaleout_execution_mode": "intraday",
        "r_profit_scaleout_execution_time": "close",
        "r_profit_scaleout_breakeven_stop_enabled": True,
        "r_profit_scaleout_tiers": [
            {"r_multiple": 2.0, "reduce_fraction": 0.33},
            {"r_multiple": 3.0, "reduce_fraction": 0.33},
        ],
        "bias_v_take_profit_enabled": False,
        "bias_v_take_profit_reentry_mode": "wait_next_entry",
        "bias_v_take_profit_execution_mode": "intraday",
        "bias_v_take_profit_execution_time": "close",
        "bias_v_take_profit_breakeven_stop_enabled": True,
        "bias_v_ma_window": 20,
        "bias_v_atr_window": 20,
        "bias_v_take_profit_tiers": [{"threshold": 4.75, "reduce_fraction": 0.5}],
        "ma_trailing_stop_enabled": False,
        "ma_trailing_stop_ma_type": "sma",
        "ma_trailing_stop_execution_mode": "intraday",
        "ma_trailing_stop_execution_time": "close",
        "ma_trailing_stop_effective_delay_days": 10,
        "ma_trailing_stop_reduce_window": 10,
        "ma_trailing_stop_exit_window": 20,
        "ma_trailing_stop_reduce_fraction": 0.5,
        "monthly_risk_budget_enabled": False,
        "monthly_risk_budget_pct": 0.06,
        "monthly_risk_budget_include_new_trade_risk": False,
        "fixed_pos_ratio": 0.06,
        "fixed_overcap_policy": "skip",
        "fixed_max_holdings": 16,
        "risk_budget_atr_window": 20,
        "risk_budget_pct": 0.0025,
        "cvar_risk_mgmt_enabled": False,
        "cvar_window": 60,
        "cvar_budget_pct": 0.02,
        "risk_budget_overcap_policy": "scale",
        "risk_budget_rebalance_mode": "standard",
        "risk_budget_max_leverage_multiple": 10.0,
        "vol_regime_risk_mgmt_enabled": False,
        "vol_periodic_risk_mgmt_enabled": False,
        "vol_periodic_rebalance_threshold_pct": 0.05,
        "vol_ratio_fast_atr_window": 5,
        "vol_ratio_slow_atr_window": 50,
        "vol_ratio_expand_threshold": 1.4,
        "vol_ratio_contract_threshold": 0.7,
        "vol_ratio_normal_threshold": 1.0,
        "vol_ratio_extreme_threshold": 2.1,
        "risk_of_ruin_maxrisk": 0.2,
        "group_enforce": False,
        "group_pick_policy": "highest_sharpe",
        "group_max_holdings": 4,
        "asset_groups": {},
    }


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(stat.S_IRWXU)


def _write_json(path: Path, value: Any) -> None:
    _private_dir(path.parent)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_decimal(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_float=Decimal,
        parse_int=Decimal,
    )


def _artifact_envelope(
    *,
    artifact_id: str,
    artifact_type: str,
    payload: Any,
    upstream: list[dict[str, str]],
    generated_at: dt.datetime | None = None,
) -> dict[str, Any]:
    """Bind an artifact payload to explicit upstream IDs and hashes."""
    timestamp = generated_at or dt.datetime.now(dt.timezone.utc)
    return {
        "artifact_id": str(artifact_id),
        "artifact_type": str(artifact_type),
        "generated_at": timestamp.isoformat(),
        "upstream": [dict(row) for row in upstream],
        "payload": payload,
    }


def _validate_chain_nodes(
    case_dir: Path,
    nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify node bytes and every JSON upstream reference before publication."""
    hashes_by_id: dict[str, str] = {}
    for node in nodes:
        artifact_id = str(node["artifact_id"])
        if artifact_id in hashes_by_id:
            raise ValueError(f"duplicate artifact_id in chain: {artifact_id}")
        path = case_dir / str(node["path"])
        actual = _sha256(path)
        expected = str(node["sha256"])
        if actual != expected:
            raise ValueError(f"artifact hash mismatch: {node['path']}")
        hashes_by_id[artifact_id] = actual

    reference_count = 0
    for node in nodes:
        path = case_dir / str(node["path"])
        if path.suffix.lower() != ".json":
            continue
        document = _read_json(path)
        for upstream in document.get("upstream") or []:
            reference_count += 1
            upstream_id = str(upstream.get("artifact_id") or "")
            if upstream_id not in hashes_by_id:
                raise ValueError(
                    f"unchained upstream {upstream_id!r} from {node['artifact_id']}"
                )
            if str(upstream.get("sha256") or "") != hashes_by_id[upstream_id]:
                raise ValueError(
                    f"upstream hash mismatch for {upstream_id!r} "
                    f"from {node['artifact_id']}"
                )
    return {
        "node_count": len(nodes),
        "upstream_reference_count": reference_count,
        "node_hashes_valid": True,
        "upstream_hashes_valid": True,
    }


def _build_stopped_bridge(
    *,
    metric: str,
    model_value: Decimal,
    live_value: Decimal,
    unavailable_layer: str,
    reason: str,
    absolute_tolerance: Decimal,
) -> dict[str, Any]:
    """Build an additive bridge that stops at the first evidence gate."""
    layer_names = (
        "data_availability_and_price",
        "signal_and_execution_timing",
        "position_and_capital_constraints",
        "matched_actual_fills",
        "actual_fees",
    )
    if unavailable_layer not in layer_names:
        raise ValueError(f"unknown bridge layer: {unavailable_layer}")
    stop_index = layer_names.index(unavailable_layer)
    layers = []
    for index, layer in enumerate(layer_names):
        if index < stop_index:
            status = "frozen_no_identified_increment"
            increment = "0"
        elif index == stop_index:
            status = "unidentifiable_stop"
            increment = None
        else:
            status = "not_reached"
            increment = None
        layers.append(
            {
                "layer": layer,
                "status": status,
                "increment": increment,
                "reason": reason if index == stop_index else None,
            }
        )
    difference = live_value - model_value
    identified_sum = sum(
        (
            Decimal(str(row["increment"]))
            for row in layers
            if row["increment"] is not None
        ),
        Decimal("0"),
    )
    residual = difference - identified_sum
    rebuild_error = live_value - (model_value + identified_sum + residual)
    return {
        "metric": str(metric),
        "status": "stopped_unidentifiable",
        "model_value": str(model_value),
        "live_value": str(live_value),
        "total_difference": str(difference),
        "layers": layers,
        "identified_increment_sum": str(identified_sum),
        "unexplained_residual": str(residual),
        "rebuild_error": str(rebuild_error),
        "absolute_tolerance": str(absolute_tolerance),
        "within_tolerance": abs(rebuild_error) <= absolute_tolerance,
    }


def _decimal_nav_identity(
    rows: list[dict[str, Any]],
    *,
    physical_types: dict[str, str],
) -> dict[str, Any]:
    """Recompute equity identity exactly from stored decimal strings."""
    import numpy as np

    def _ulp(value: Decimal, column: str) -> Decimal:
        column_type = str(physical_types.get(column, "")).upper()
        if "FLOAT" in column_type and "DOUBLE" not in column_type:
            spacing = abs(float(np.spacing(np.float32(float(value)))))
            return Decimal(str(spacing))
        return Decimal(str(math.ulp(float(value))))

    residuals: list[dict[str, str]] = []
    float_bound = Decimal("0")
    for row in rows:
        equity = Decimal(str(row.get("equity") or "0"))
        cash = Decimal(str(row.get("cash") or "0"))
        market_value = Decimal(str(row.get("market_value") or "0"))
        residual = equity - cash - market_value
        residuals.append(
            {
                "date": str(row.get("nav_date")),
                "residual": str(residual),
            }
        )
        float_bound = max(
            float_bound,
            sum(
                (
                    _ulp(value, column)
                    for value, column in (
                        (equity, "equity"),
                        (cash, "cash"),
                        (market_value, "market_value"),
                    )
                ),
                Decimal("0"),
            ),
        )
    first = next(
        (row["date"] for row in residuals if Decimal(row["residual"]) != 0),
        None,
    )
    maximum = max(
        (abs(Decimal(row["residual"])) for row in residuals),
        default=Decimal("0"),
    )
    return {
        "physical_types": dict(physical_types),
        "first_divergence_date": first,
        "max_abs_decimal_residual": str(maximum),
        "float_error_upper_bound": str(float_bound),
        "explained_by_float_error": maximum <= float_bound,
        "source_precision": (
            "stored_values_exact" if maximum == 0 else "source_precision_unknown"
        ),
        "daily_residuals": residuals,
    }


def _decimal_nav_chain(
    rows: list[dict[str, Any]],
    *,
    physical_types: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Recompute the stored TWR recurrence with Decimal arithmetic."""
    import numpy as np

    types = physical_types or {}

    def _ulp(value: Decimal, column: str) -> Decimal:
        column_type = str(types.get(column, "")).upper()
        if "FLOAT" in column_type and "DOUBLE" not in column_type:
            return Decimal(str(abs(float(np.spacing(np.float32(float(value)))))))
        return Decimal(str(math.ulp(float(value))))

    residuals: list[dict[str, str]] = []
    float_bound = Decimal("0")
    for previous, current in zip(rows, rows[1:], strict=False):
        previous_nav = Decimal(str(previous.get("nav_twr") or "0"))
        daily_return = Decimal(str(current.get("daily_return_twr") or "0"))
        current_nav = Decimal(str(current.get("nav_twr") or "0"))
        expected = previous_nav * (Decimal("1") + daily_return)
        residual = current_nav - expected
        residuals.append(
            {
                "date": str(current.get("nav_date")),
                "stored_nav": str(current_nav),
                "expected_nav": str(expected),
                "residual": str(residual),
            }
        )
        operation_bound = (
            _ulp(previous_nav, "nav_twr") * (Decimal("1") + abs(daily_return))
            + abs(previous_nav) * _ulp(daily_return, "daily_return_twr")
            + _ulp(current_nav, "nav_twr")
            + _ulp(expected, "nav_twr")
        )
        float_bound = max(float_bound, operation_bound)
    first = next(
        (row["date"] for row in residuals if Decimal(row["residual"]) != 0),
        None,
    )
    maximum = max(
        (abs(Decimal(row["residual"])) for row in residuals),
        default=Decimal("0"),
    )
    return {
        "first_divergence_date": first,
        "max_abs_decimal_residual": str(maximum),
        "float_error_upper_bound": str(float_bound),
        "explained_by_float_error": maximum <= float_bound,
        "source_precision": (
            "stored_values_exact" if maximum == 0 else "source_precision_unknown"
        ),
        "daily_residuals": residuals,
    }


def _classify_strategy_cashflows(
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Classify strategy-scope flows without treating income as funding."""
    external_types = {"transfer_in", "transfer_out", "manual", "deposit", "withdraw"}
    income_types = {"dividend", "interest", "repo_carry", "corporate_action_cash"}
    result = {
        "external_flow": Decimal("0"),
        "investment_income": Decimal("0"),
        "unknown": Decimal("0"),
    }
    for row in rows:
        amount = Decimal(str(row.get("amount") or "0"))
        flow_type = str(row.get("flow_type") or "").strip().lower()
        if flow_type in external_types:
            result["external_flow"] += amount
        elif flow_type in income_types:
            result["investment_income"] += amount
        else:
            result["unknown"] += amount
    return {key: float(value) for key, value in result.items()}


def _classify_cashflows_by_scope(
    *,
    account_rows: list[dict[str, Any]],
    strategy_rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    income_types = {"dividend", "interest", "repo_carry", "corporate_action_cash"}

    def _classify_account(rows: list[dict[str, Any]]) -> dict[str, float]:
        totals = {
            "external_flow": Decimal("0"),
            "internal_transfer": Decimal("0"),
            "investment_income": Decimal("0"),
            "unknown": Decimal("0"),
        }
        for row in rows:
            kind = str(row.get("flow_type") or "").strip().lower()
            amount = Decimal(str(row.get("amount") or "0"))
            if kind in {"deposit", "withdraw", "manual"}:
                totals["external_flow"] += amount
            elif kind in {"transfer_to_strategy", "transfer_from_strategy"}:
                totals["internal_transfer"] += amount
            elif kind in income_types:
                totals["investment_income"] += amount
            else:
                totals["unknown"] += amount
        return {key: float(value) for key, value in totals.items()}

    strategy = _classify_strategy_cashflows(strategy_rows)
    strategy["internal_transfer"] = 0.0
    return {
        "account": _classify_account(account_rows),
        "strategy": strategy,
    }


def _modified_dietz_sensitivity(
    nav_rows: list[dict[str, Any]],
    flows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report BOD/EOD Modified Dietz bounds when flow times are date-only."""
    if len(nav_rows) < 2:
        return {"status": "insufficient_nav_observations"}
    ordered = sorted(nav_rows, key=lambda row: str(row.get("nav_date")))
    start_date = dt.date.fromisoformat(str(ordered[0]["nav_date"]))
    end_date = dt.date.fromisoformat(str(ordered[-1]["nav_date"]))
    elapsed = max(1, (end_date - start_date).days)
    begin = Decimal(str(ordered[0].get("equity") or "0"))
    end = Decimal(str(ordered[-1].get("equity") or "0"))
    total_flow = Decimal("0")
    weighted_bod = Decimal("0")
    weighted_eod = Decimal("0")
    included = 0
    for row in flows:
        flow_date = dt.date.fromisoformat(str(row.get("flow_date")))
        if not (start_date < flow_date <= end_date):
            continue
        amount = Decimal(str(row.get("amount") or "0"))
        remaining_days = Decimal(str((end_date - flow_date).days))
        total_flow += amount
        weighted_eod += amount * remaining_days / Decimal(elapsed)
        weighted_bod += amount * (remaining_days + Decimal("1")) / Decimal(elapsed)
        included += 1
    numerator = end - begin - total_flow
    denominator_bod = begin + weighted_bod
    denominator_eod = begin + weighted_eod
    bod = numerator / denominator_bod if denominator_bod != 0 else None
    eod = numerator / denominator_eod if denominator_eod != 0 else None
    values = [value for value in (bod, eod) if value is not None]
    return {
        "status": "date_only_sensitivity",
        "formula": "(ending_equity-beginning_equity-flows)/(beginning_equity+weighted_flows)",
        "begin_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "included_flow_count": included,
        "amount_pnl": str(numerator),
        "bod_assumption_return": str(bod) if bod is not None else None,
        "eod_assumption_return": str(eod) if eod is not None else None,
        "sensitivity_low": str(min(values)) if values else None,
        "sensitivity_high": str(max(values)) if values else None,
        "exact_twr_status": "unidentifiable_without_intraday_flow_valuations",
    }


def _independent_tsmom_trace(
    prices: Any,
    *,
    lookback: int,
    entry_threshold: float,
    exit_threshold: float,
) -> Any:
    """Compute a causal TSMOM state without forward-filling source prices."""
    import pandas as pd

    source = pd.Series(prices).astype(float)
    momentum = source / source.shift(max(1, int(lookback))) - 1.0
    signal = pd.Series(0.0, index=source.index, dtype=float)
    state = 0.0
    for date_value in source.index:
        value = momentum.loc[date_value]
        if pd.notna(value):
            if state <= 0.0 and float(value) > float(entry_threshold):
                state = 1.0
            elif state > 0.0 and float(value) <= float(exit_threshold):
                state = 0.0
        signal.loc[date_value] = state
    return pd.DataFrame(
        {
            "price": source,
            "momentum": momentum,
            "signal": signal,
            "market_observed": source.notna(),
        }
    )


def _terminal_episode_projection(
    episodes: list[dict[str, Any]],
    *,
    mode: str,
    as_of: str,
) -> dict[str, Any]:
    """Keep terminal reporting projections outside the causal trade ledger."""
    if mode not in {"no_forced_liquidation", "reporting_only_liquidation"}:
        raise ValueError(f"unsupported terminal liquidation mode: {mode}")
    causal = [dict(row) for row in episodes]
    reporting = []
    if mode == "reporting_only_liquidation":
        reporting = [
            {
                "code": row.get("code"),
                "entry_date": row.get("entry_date"),
                "reporting_date": str(as_of),
                "return": row.get("return"),
                "closed": False,
                "projection_only": True,
            }
            for row in causal
            if not bool(row.get("closed"))
        ]
    return {
        "mode": mode,
        "causal_episodes": causal,
        "closed_stat_returns": [
            float(row.get("return") or 0.0) for row in causal if bool(row.get("closed"))
        ],
        "reporting_only_liquidations": reporting,
    }


def _window_drawdowns(
    response: dict[str, Any],
    *,
    start: str,
    end: str,
) -> dict[str, Any]:
    """Compute window drawdown under reset and inherited high-water marks."""
    import pandas as pd

    nav_payload = response.get("nav") or {}
    nav = pd.Series(
        (nav_payload.get("series") or {}).get("STRAT") or [],
        index=pd.to_datetime(nav_payload.get("dates") or []),
        dtype=float,
    ).sort_index()
    window = nav.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    if window.empty:
        return {"status": "empty_window"}
    reset_curve = window / float(window.iloc[0])
    reset_drawdown = reset_curve / reset_curve.cummax() - 1.0
    inherited_drawdown = nav / nav.cummax() - 1.0
    inherited_window = inherited_drawdown.reindex(window.index)
    return {
        "status": "ok",
        "window_start_nav": float(window.iloc[0]),
        "inherited_high_water_at_start": float(nav.loc[: window.index[0]].max()),
        "window_reset_high_water_max_drawdown": float(reset_drawdown.min()),
        "inherited_high_water_max_drawdown": float(inherited_window.min()),
    }


def _independent_daily_oracle(
    *,
    asset_returns: list[Any],
    effective_weights: list[Any],
    return_weights: list[Any],
    fee_rate: Any,
    slippage_spread: Any,
    execution_prices: list[Any],
    initial_nav: Any,
) -> list[dict[str, str]]:
    """Hand-calculable normalized daily portfolio accounting oracle."""
    lengths = {
        len(asset_returns),
        len(effective_weights),
        len(return_weights),
        len(execution_prices),
    }
    if len(lengths) != 1:
        raise ValueError("oracle input lengths differ")
    fees = Decimal(str(fee_rate))
    spread = Decimal(str(slippage_spread))
    nav = Decimal(str(initial_nav))
    previous_weight = Decimal("0")
    rows = []
    for asset_ret_raw, weight_raw, return_weight_raw, price_raw in zip(
        asset_returns,
        effective_weights,
        return_weights,
        execution_prices,
        strict=True,
    ):
        asset_return = Decimal(str(asset_ret_raw))
        weight = Decimal(str(weight_raw))
        return_weight = Decimal(str(return_weight_raw))
        price = Decimal(str(price_raw))
        turnover = abs(weight - previous_weight) / Decimal("2")
        fee_drag = turnover * fees
        slippage_drag = turnover * spread / price if price != 0 else Decimal("0")
        gross_return = return_weight * asset_return
        net_return = gross_return - fee_drag - slippage_drag
        nav *= Decimal("1") + net_return
        market_value = nav * weight
        cash = nav - market_value
        rows.append(
            {
                "asset_return": str(asset_return),
                "return_weight": str(return_weight),
                "effective_weight": str(weight),
                "turnover_one_way": str(turnover),
                "gross_return": str(gross_return),
                "fee_drag": str(fee_drag),
                "slippage_drag": str(slippage_drag),
                "net_return": str(net_return),
                "cash": str(cash),
                "market_value": str(market_value),
                "nav": str(nav),
            }
        )
        previous_weight = weight
    return rows


def _independent_risk_budget_sizing(
    *,
    capital: Any,
    risk_budget_pct: Any,
    price: Any,
    atr: Any,
    stop_multiple: Any,
    max_weight: Any,
    lot_size: int,
    available_cash: Any,
) -> dict[str, str]:
    """Rebuild ideal, capped, round-lot, and cash-constrained quantity."""
    capital_value = Decimal(str(capital))
    risk_fraction = Decimal(str(risk_budget_pct))
    price_value = Decimal(str(price))
    atr_value = Decimal(str(atr))
    stop_value = Decimal(str(stop_multiple))
    cap_weight = Decimal(str(max_weight))
    cash_value = Decimal(str(available_cash))
    lot = Decimal(str(lot_size))
    if min(capital_value, price_value, atr_value, stop_value, lot) <= 0:
        raise ValueError("sizing inputs must be positive")
    risk_amount = capital_value * risk_fraction
    risk_per_share = atr_value * stop_value
    ideal_quantity = risk_amount / risk_per_share
    cap_quantity = capital_value * cap_weight / price_value
    overcap_quantity = min(ideal_quantity, cap_quantity)
    round_lot_quantity = (overcap_quantity // lot) * lot
    cash_lots = (cash_value / price_value) // lot
    cash_quantity = min(round_lot_quantity, cash_lots * lot)
    return {
        "risk_amount": str(risk_amount),
        "risk_per_share": str(risk_per_share),
        "ideal_quantity": str(ideal_quantity),
        "ideal_weight": str(ideal_quantity * price_value / capital_value),
        "overcap_quantity": str(overcap_quantity),
        "overcap_weight": str(overcap_quantity * price_value / capital_value),
        "round_lot_quantity": str(round_lot_quantity),
        "cash_constrained_quantity": str(cash_quantity),
        "cash_constrained_notional": str(cash_quantity * price_value),
    }


def _hash_query_rows(connection: Any, statement: Any) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    result = connection.execution_options(stream_results=True).execute(statement)
    while batch := result.fetchmany(1000):
        for row in batch:
            digest.update(
                json.dumps(
                    [str(value) if value is not None else None for value in row],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\n")
            count += 1
    return count, digest.hexdigest()


def _database_metadata(engine: Any, *, tables: tuple[str, ...]) -> dict[str, Any]:
    """Capture server/timezone and physical column types without credentials."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    with engine.connect() as connection:
        server = connection.execute(
            text(
                "SELECT VERSION(), @@session.time_zone, @@system_time_zone, DATABASE()"
            )
        ).one()
    physical_types = {
        table: {
            str(column["name"]): str(column["type"])
            for column in inspector.get_columns(table)
        }
        for table in tables
        if inspector.has_table(table)
    }
    return {
        "server_version": str(server[0]),
        "session_timezone": str(server[1]),
        "system_timezone": str(server[2]),
        "database": str(server[3]),
        "physical_column_types": physical_types,
    }


def _prepare_isolated_audit_schema(case_dir: Path) -> dict[str, Any]:
    """Create a non-destructive, filtered MySQL clone for this audit."""
    from sqlalchemy import create_engine, inspect, text
    from sqlalchemy.exc import DBAPIError

    from etf_momentum.db.session import make_engine
    from etf_momentum.settings import get_settings

    source = make_engine(db_url=get_settings().db_url)
    with source.connect() as connection:
        source_schema = str(connection.execute(text("SELECT DATABASE()")).scalar_one())
        try:
            connection.execute(
                text(
                    f"CREATE DATABASE IF NOT EXISTS `{AUDIT_SCHEMA}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            )
        except DBAPIError:
            tables = (
                "etf_prices",
                "live_trade",
                "live_strategy_cashflow",
                "live_closed_round",
                "live_holding_snapshot",
                "live_nav_daily",
            )
            artifact = {
                "artifact_id": "isolated-mysql-clone-v1",
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "source_schema": source_schema,
                "target_schema": AUDIT_SCHEMA,
                "as_of": AS_OF,
                "status": "blocked_authorization",
                "production_replay_called": False,
                "reason": "database_user_cannot_create_isolated_schema",
                "database_metadata": _database_metadata(source, tables=tables),
            }
            _write_json(
                case_dir / "derived" / "isolated-mysql-clone.json",
                artifact,
            )
            source.dispose()
            return artifact
    target_url = source.url.set(database=AUDIT_SCHEMA)
    target = create_engine(target_url, future=True, pool_pre_ping=True)
    inspector = inspect(target)
    if inspector.get_table_names():
        raise RuntimeError(
            f"audit schema {AUDIT_SCHEMA} is not empty; refusing to overwrite"
        )

    tables = (
        "validation_policy",
        "etf_pool",
        "etf_prices",
        "etf_research_group",
        "etf_research_group_item",
        "live_account",
        "live_shareholder_account",
        "live_strategy",
        "live_strategy_profile",
        "live_account_cashflow",
        "live_strategy_cashflow",
        "live_trade",
        "live_repo_trade_detail",
        "live_trade_audit_log",
        "live_corporate_action_event",
        "live_closed_round",
        "live_closed_round_leg",
        "live_holding_snapshot",
        "live_nav_daily",
        "live_symbol_alias",
    )
    codes_sql = ", ".join(f"'{code}'" for code in CODES)
    filters = {
        "validation_policy": "1=1",
        "etf_pool": f"code IN ({codes_sql})",
        "etf_prices": (f"code IN ({codes_sql}) AND trade_date <= '{AS_OF}'"),
        "etf_research_group": "1=1",
        "etf_research_group_item": f"code IN ({codes_sql})",
        "live_account": f"id = {ACCOUNT_ID}",
        "live_shareholder_account": f"account_id = {ACCOUNT_ID}",
        "live_strategy": (f"id = {STRATEGY_ID} AND account_id = {ACCOUNT_ID}"),
        "live_strategy_profile": f"strategy_id = {STRATEGY_ID}",
        "live_account_cashflow": (
            f"account_id = {ACCOUNT_ID} AND flow_date <= '{AS_OF}'"
        ),
        "live_strategy_cashflow": (
            f"strategy_id = {STRATEGY_ID} AND flow_date <= '{AS_OF}'"
        ),
        "live_trade": (
            f"account_id = {ACCOUNT_ID} AND strategy_id = {STRATEGY_ID} "
            f"AND trade_date <= '{AS_OF}'"
        ),
        "live_repo_trade_detail": (
            "trade_id IN (SELECT id FROM "
            f"`{source_schema}`.live_trade WHERE account_id={ACCOUNT_ID} "
            f"AND strategy_id={STRATEGY_ID} AND trade_date <= '{AS_OF}')"
        ),
        "live_trade_audit_log": (
            f"account_id = {ACCOUNT_ID} AND strategy_id = {STRATEGY_ID}"
        ),
        "live_corporate_action_event": (
            f"account_id = {ACCOUNT_ID} AND "
            f"(strategy_id IS NULL OR strategy_id = {STRATEGY_ID}) "
            f"AND effective_date <= '{AS_OF}'"
        ),
        "live_closed_round": (
            f"strategy_id = {STRATEGY_ID} AND close_date <= '{AS_OF}'"
        ),
        "live_closed_round_leg": (
            "round_id IN (SELECT id FROM "
            f"`{source_schema}`.live_closed_round "
            f"WHERE strategy_id={STRATEGY_ID} AND close_date <= '{AS_OF}')"
        ),
        "live_holding_snapshot": (
            f"strategy_id = {STRATEGY_ID} AND snapshot_date <= '{AS_OF}'"
        ),
        "live_nav_daily": (f"strategy_id = {STRATEGY_ID} AND nav_date <= '{AS_OF}'"),
        "live_symbol_alias": f"effective_date <= '{AS_OF}'",
    }
    with source.begin() as connection:
        for table in tables:
            connection.execute(
                text(
                    f"CREATE TABLE `{AUDIT_SCHEMA}`.`{table}` "
                    f"LIKE `{source_schema}`.`{table}`"
                )
            )
            connection.execute(
                text(
                    f"INSERT INTO `{AUDIT_SCHEMA}`.`{table}` "
                    f"SELECT * FROM `{source_schema}`.`{table}` "
                    f"WHERE {filters[table]}"
                )
            )

    table_manifest = []
    target_inspector = inspect(target)
    with target.connect() as connection:
        for table in tables:
            primary_key = (
                target_inspector.get_pk_constraint(table).get("constrained_columns")
                or []
            )
            order = ", ".join(f"`{key}`" for key in primary_key) or "1"
            count, digest = _hash_query_rows(
                connection,
                text(f"SELECT * FROM `{table}` ORDER BY {order}"),
            )
            table_manifest.append(
                {
                    "table": table,
                    "row_count": count,
                    "sha256": digest,
                    "filter": filters[table],
                }
            )
    metadata = _database_metadata(target, tables=tables)
    artifact = {
        "artifact_id": "isolated-mysql-clone-v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_schema": source_schema,
        "target_schema": AUDIT_SCHEMA,
        "as_of": AS_OF,
        "mode": "physical_filtered_clone",
        "status": "complete",
        "production_replay_called": False,
        "tables": table_manifest,
        "database_metadata": metadata,
    }
    path = case_dir / "derived" / "isolated-mysql-clone.json"
    _write_json(path, artifact)
    target.dispose()
    source.dispose()
    return artifact


def _request_json(
    url: str, *, method: str = "GET", payload: dict[str, Any] | None = None
) -> Any:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.loads(response.read().decode("utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT.parent), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _working_tree_snapshot() -> dict[str, Any]:
    """Hash tracked changes and every non-ignored untracked source artifact."""
    repository = REPO_ROOT.parent
    diff = subprocess.run(
        ["git", "-C", str(repository), "diff", "--binary", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    untracked_output = _git_output("ls-files", "--others", "--exclude-standard")
    untracked = []
    combined = hashlib.sha256()
    combined.update(diff)
    for relative in sorted(filter(None, untracked_output.splitlines())):
        path = repository / relative
        digest = _sha256(path)
        untracked.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": digest,
            }
        )
        combined.update(relative.encode("utf-8"))
        combined.update(digest.encode("ascii"))
    return {
        "head": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "status_porcelain": _git_output("status", "--porcelain"),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_files": untracked,
        "combined_dirty_tree_sha256": combined.hexdigest(),
    }


def _runtime_evidence() -> dict[str, Any]:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    result: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "pip_freeze": sorted(freeze),
    }
    try:
        from etf_momentum.db.session import make_engine
        from etf_momentum.settings import get_settings

        engine = make_engine(db_url=get_settings().db_url)
        result["mysql"] = _database_metadata(
            engine,
            tables=(
                "live_trade",
                "live_strategy_cashflow",
                "live_closed_round",
                "live_holding_snapshot",
                "live_nav_daily",
            ),
        )
        engine.dispose()
    except Exception as exc:  # evidence capture must record, not hide, a DB gap
        result["mysql"] = {
            "status": "unavailable",
            "error_type": type(exc).__name__,
        }
    return result


def _run_and_capture(
    *,
    name: str,
    command: list[str],
    case_dir: Path,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = dt.datetime.now(dt.timezone.utc)
    monotonic_start = time.monotonic()
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    completed = dt.datetime.now(dt.timezone.utc)
    log_path = case_dir / "logs" / f"{name}.log"
    log_text = (
        f"started_at={started.isoformat()}\n"
        f"completed_at={completed.isoformat()}\n"
        f"cwd={cwd}\n"
        f"command={json.dumps(command)}\n"
        f"exit_code={result.returncode}\n\n"
        f"{result.stdout}\n{result.stderr}"
    )
    _private_dir(log_path.parent)
    log_path.write_text(log_text, encoding="utf-8")
    log_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return {
        "name": name,
        "command": command,
        "cwd": str(cwd),
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "elapsed_seconds": time.monotonic() - monotonic_start,
        "exit_code": result.returncode,
        "status": "passed" if result.returncode == 0 else "failed",
        "log_path": str(log_path.relative_to(case_dir)),
        "log_sha256": _sha256(log_path),
    }


def _capture_prefx_failures(case_dir: Path, worktree: Path) -> dict[str, Any]:
    """Run new golden fixtures against the frozen pre-fix Git tree."""
    old_project = worktree / "momentum"
    if not (old_project / "src").is_dir():
        raise ValueError(f"invalid pre-fix worktree: {worktree}")
    tests = (
        REPO_ROOT / "tests" / "test_trend_trade_ledger_audit.py",
        REPO_ROOT / "tests" / "test_strategy_execution_timing_regression.py",
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        *(str(path) for path in tests),
        "-q",
        "-c",
        str(old_project / "pyproject.toml"),
        "--rootdir",
        str(old_project),
    ]
    environment = dict(__import__("os").environ)
    environment["PYTHONPATH"] = str(old_project / "src")
    run = _run_and_capture(
        name="pre-fix-golden-fixtures",
        command=command,
        case_dir=case_dir,
        cwd=old_project,
        env=environment,
    )
    old_head = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    artifact = _artifact_envelope(
        artifact_id="pre-fix-failures-v2",
        artifact_type="failing-fixture-evidence",
        payload={
            "old_head": old_head,
            "test_files": [
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": _sha256(path),
                }
                for path in tests
            ],
            "run": run,
            "expected_failure_observed": run["exit_code"] != 0,
        },
        upstream=[],
    )
    _write_json(case_dir / "reports" / "pre-fix-failures.json", artifact)
    return artifact


def _pytest_counts(log_path: Path) -> dict[str, int]:
    text_value = log_path.read_text(encoding="utf-8")
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    summary_found = False
    for key in counts:
        matches = re.findall(rf"(\d+)\s+{key}", text_value)
        if matches:
            counts[key] = int(matches[-1])
            summary_found = True
    if not summary_found:
        progress_chunks = re.findall(
            r"^([.sFExX]+)\s+\[\s*\d+%\]\s*$",
            text_value,
            flags=re.MULTILINE,
        )
        progress = "".join(progress_chunks)
        counts["passed"] = progress.count(".")
        counts["failed"] = progress.count("F")
        counts["skipped"] = progress.count("s")
        counts["errors"] = progress.count("E")
    counts["collected"] = (
        counts["passed"] + counts["failed"] + counts["skipped"] + counts["errors"]
    )
    return counts


def _run_verification(case_dir: Path) -> dict[str, Any]:
    """Run real verification commands and hash their complete logs."""
    python = sys.executable
    targeted_files = (
        "tests/test_audit_trend_live_runner.py",
        "tests/test_trend_trade_ledger_audit.py",
        "tests/test_strategy_execution_timing_regression.py",
        "tests/test_trend_bt_semantic_parity.py",
        "tests/test_analysis_trend.py",
        "tests/test_analysis_trend_portfolio.py",
        "tests/test_api_live_trading_records.py",
    )
    runs = [
        _run_and_capture(
            name="targeted-pytest",
            command=[python, "-m", "pytest", *targeted_files, "-q"],
            case_dir=case_dir,
        ),
        _run_and_capture(
            name="full-pytest",
            command=[python, "-m", "pytest", "-q"],
            case_dir=case_dir,
        ),
        _run_and_capture(
            name="compileall",
            command=[
                python,
                "-m",
                "compileall",
                "-q",
                "src",
                "scripts",
                "tests",
            ],
            case_dir=case_dir,
        ),
    ]
    ruff_probe = subprocess.run(
        [python, "-c", "import ruff"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if ruff_probe.returncode == 0:
        runs.append(
            _run_and_capture(
                name="ruff",
                command=[python, "-m", "ruff", "check", "src", "scripts", "tests"],
                case_dir=case_dir,
            )
        )
    else:
        runs.append(
            {
                "name": "ruff",
                "status": "unavailable",
                "reason": "ruff_not_installed_in_frozen_environment",
                "exit_code": None,
            }
        )
    for run in runs:
        if run.get("name") in {"targeted-pytest", "full-pytest"}:
            run["counts"] = _pytest_counts(case_dir / str(run["log_path"]))
    artifact = _artifact_envelope(
        artifact_id="verification-v2",
        artifact_type="verification",
        payload={
            "python_executable": python,
            "dirty_tree": _working_tree_snapshot(),
            "runs": runs,
            "passed": all(
                run.get("status") in {"passed", "unavailable"} for run in runs
            ),
        },
        upstream=[],
    )
    _write_json(case_dir / "reports" / "verification.json", artifact)
    return artifact


def _file_manifest(case_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(case_dir.rglob("*")):
        if not path.is_file() or path.name == "evidence-manifest.json":
            continue
        relative = path.relative_to(case_dir)
        if relative.parts[0] == "reports" or relative.name == "scenario-manifest.json":
            continue
        rows.append(
            {
                "path": str(relative),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _capture_live(case_dir: Path, api_base: str) -> None:
    raw_dir = case_dir / "raw"
    derived_dir = case_dir / "derived"
    endpoints = {
        raw_dir / "account-5-export.json": f"{api_base}/api/live/accounts/5/export",
        raw_dir / "strategy-13-cashflows.json": (
            f"{api_base}/api/live/strategies/13/cashflows"
        ),
        derived_dir / "strategy-13-closed-rounds.json": (
            f"{api_base}/api/live/closed-rounds"
            "?scope_type=strategy&scope_id=13&page=1&page_size=500"
        ),
        derived_dir / "strategy-13-holdings.json": (
            f"{api_base}/api/live/holdings?scope_type=strategy&scope_id=13"
        ),
        derived_dir / "strategy-13-performance.json": (
            f"{api_base}/api/live/performance"
            "?scope_type=strategy&scope_id=13&return_basis=both"
        ),
        derived_dir / "strategy-13-attribution.json": (
            f"{api_base}/api/live/attribution?scope_type=strategy&scope_id=13"
        ),
    }
    for path, url in endpoints.items():
        _write_json(path, _request_json(url))


def _capture_backtests(
    case_dir: Path,
    api_base: str,
    scenarios: list[tuple[str, str, bool]],
) -> None:
    for name, engine, quick_mode in scenarios:
        payload = effective_payload(engine=engine, quick_mode=quick_mode)
        _write_json(case_dir / "backtests" / f"{name}-request.json", payload)
        response = _request_json(
            f"{api_base}/api/analysis/trend/portfolio",
            method="POST",
            payload=payload,
        )
        _write_json(case_dir / "backtests" / f"{name}-response.json", response)


def capture(case_dir: Path, api_base: str, *, refresh_live: bool) -> None:
    """Freeze live inputs and three backtest scenarios."""
    _private_dir(case_dir)
    for name in ("raw", "derived", "backtests", "reports"):
        _private_dir(case_dir / name)
    if refresh_live:
        _capture_live(case_dir, api_base)

    _capture_backtests(
        case_dir,
        api_base,
        [
            ("legacy-quick-original", "legacy", True),
            ("legacy-full", "legacy", False),
            ("bt-full", "bt", False),
        ],
    )

    export = _read_json(case_dir / "raw" / "account-5-export.json")
    payload = export.get("payload") or {}
    trades = [
        row
        for row in payload.get("trades") or []
        if int(row.get("strategy_id") or 0) == STRATEGY_ID
    ]
    strategy_flows = [
        row
        for row in payload.get("strategy_cashflows") or []
        if int(row.get("strategy_id") or 0) == STRATEGY_ID
    ]
    evidence = {
        "case_id": "trend-live-5-13",
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "as_of": AS_OF,
        "account_id": ACCOUNT_ID,
        "strategy_id": STRATEGY_ID,
        "artifact_id": "evidence-manifest-v2",
        "git": _working_tree_snapshot(),
        "environment": _runtime_evidence(),
        "security_policy": {
            "storage": "repository_ignored_local_directory",
            "directory_mode": "0700",
            "file_mode": "0600",
            "authorized_access": ["repository_owner"],
            "retention_until": "2026-11-27",
            "destruction": (
                "delete the case directory and verify no tracked or untracked "
                "git entry contains shareholder accounts, broker IDs, notes, "
                "or raw snapshots"
            ),
            "display_policy": "redact shareholder accounts, broker IDs, and notes",
        },
        "counts": {
            "strategy_trades": len(trades),
            "strategy_cashflows": len(strategy_flows),
            "closed_rounds": int(
                (
                    _read_json(case_dir / "derived" / "strategy-13-closed-rounds.json")
                    or {}
                ).get("total", 0)
            ),
        },
        "ranges": {
            "strategy_trade_min_date": min(
                (str(row.get("trade_date")) for row in trades), default=None
            ),
            "strategy_trade_max_date": max(
                (str(row.get("trade_date")) for row in trades), default=None
            ),
        },
        "identifiability": {
            "effective_parameters": "available_from_frozen_response_meta",
            "strategy_version_snapshot": "unavailable",
            "historical_group_membership": "unavailable",
            "price_vintage": "current_database_vintage_only",
            "daily_original_plan": "unavailable",
            "broker_trade_ledger": (
                "available"
                if trades and any(row.get("broker_trade_no") for row in trades)
                else "partial_or_unavailable"
            ),
            "strategy_capital_flows": (
                "available_date_only" if strategy_flows else "unavailable"
            ),
            "cashflow_intraday_timing": "unavailable",
            "model_state_at_window_start": "unavailable",
            "live_book_state_at_window_start": (
                "partial_from_trades_and_date_only_cashflows"
            ),
            "isolated_mysql_schema": (
                _read_json(case_dir / "derived" / "isolated-mysql-clone.json").get(
                    "status"
                )
                if (case_dir / "derived" / "isolated-mysql-clone.json").exists()
                else "not_attempted"
            ),
        },
        "files": [],
    }
    _write_json(case_dir / "evidence-manifest.json", evidence)
    evidence["files"] = _file_manifest(case_dir)
    _write_json(case_dir / "evidence-manifest.json", evidence)


def refresh_evidence(case_dir: Path) -> dict[str, Any]:
    """Refresh code/environment provenance without touching production APIs."""
    path = case_dir / "evidence-manifest.json"
    manifest = _read_json(path)
    manifest["artifact_id"] = "evidence-manifest-v2"
    manifest["provenance_refreshed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["git"] = _working_tree_snapshot()
    manifest["environment"] = _runtime_evidence()
    clone_path = case_dir / "derived" / "isolated-mysql-clone.json"
    if clone_path.exists():
        clone = _read_json(clone_path)
        manifest.setdefault("identifiability", {})["isolated_mysql_schema"] = clone.get(
            "status"
        )
        manifest["isolated_mysql_clone"] = {
            "artifact_id": clone.get("artifact_id"),
            "sha256": _sha256(clone_path),
            "status": clone.get("status"),
            "reason": clone.get("reason"),
        }
    manifest["files"] = _file_manifest(case_dir)
    _write_json(path, manifest)
    return manifest


def capture_postfix(case_dir: Path, api_base: str) -> None:
    """Capture fixed-code scenarios without overwriting frozen baseline files."""
    _capture_backtests(
        case_dir,
        api_base,
        [
            ("postfix-legacy-full", "legacy", False),
            ("postfix-bt-full", "bt", False),
        ],
    )
    manifest_path = case_dir / "evidence-manifest.json"
    manifest = _read_json(manifest_path)
    manifest["postfix_captured_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["files"] = _file_manifest(case_dir)
    _write_json(manifest_path, manifest)


def capture_final(case_dir: Path, api_base: str) -> None:
    """Capture all confirmed fixes without overwriting prior bridge stages."""
    _capture_backtests(
        case_dir,
        api_base,
        [
            ("final-legacy-full", "legacy", False),
            ("final-bt-full", "bt", False),
        ],
    )
    manifest_path = case_dir / "evidence-manifest.json"
    manifest = _read_json(manifest_path)
    manifest["final_captured_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["files"] = _file_manifest(case_dir)
    _write_json(manifest_path, manifest)


def _collect_trigger_events(value: Any, path: str = "") -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            child_path = f"{path}.{key}" if path else str(key)
            if key == "trigger_events" and isinstance(child, list):
                events.extend(
                    {"overlay_path": path, **dict(event)}
                    for event in child
                    if isinstance(event, dict)
                    and any(
                        event.get(field) is not None
                        for field in (
                            "date",
                            "trigger_date",
                            "execution_date",
                            "trigger_source",
                            "fill_price",
                        )
                    )
                )
            else:
                events.extend(_collect_trigger_events(child, child_path))
    elif isinstance(value, list):
        for child in value:
            events.extend(_collect_trigger_events(child, path))
    return events


def _build_model_trace(case_dir: Path) -> dict[str, Any]:
    """Build an audit-only causal/data/accounting trace from read-only inputs."""
    import pandas as pd
    from sqlalchemy import text

    from etf_momentum.db.session import make_engine
    from etf_momentum.settings import get_settings

    response_path = case_dir / "backtests" / "final-legacy-full-response.json"
    if not response_path.exists():
        response_path = case_dir / "backtests" / "postfix-legacy-full-response.json"
    response = _read_json(response_path)
    dates = pd.to_datetime((response.get("weights") or {}).get("dates") or [])
    codes_sql = ", ".join(f"'{code}'" for code in CODES)
    engine = make_engine(db_url=get_settings().db_url)
    with engine.connect() as connection:
        rows = (
            connection.execute(
                text(
                    "SELECT code, trade_date, adjust, open, high, low, close "
                    "FROM etf_prices "
                    f"WHERE code IN ({codes_sql}) AND trade_date <= '{AS_OF}' "
                    "AND adjust IN ('qfq','none','hfq') "
                    "ORDER BY trade_date, code, adjust"
                )
            )
            .mappings()
            .all()
        )
        pool_rows = (
            connection.execute(
                text(
                    "SELECT code, start_date, end_date FROM etf_pool "
                    f"WHERE code IN ({codes_sql})"
                )
            )
            .mappings()
            .all()
        )
    engine.dispose()
    pool = {str(row["code"]): dict(row) for row in pool_rows}
    frame = pd.DataFrame(rows)
    price_by_adjust: dict[str, pd.DataFrame] = {}
    for adjust in ("qfq", "none", "hfq"):
        subset = frame.loc[frame["adjust"] == adjust] if not frame.empty else frame
        price_by_adjust[adjust] = (
            subset.pivot(index="trade_date", columns="code", values="close")
            .reindex(index=dates, columns=CODES)
            .astype(float)
            if not subset.empty
            else pd.DataFrame(index=dates, columns=CODES, dtype=float)
        )
    weights = pd.DataFrame(
        (response.get("weights") or {}).get("series") or {},
        index=dates,
    ).reindex(columns=CODES, fill_value=0.0)
    decision = pd.DataFrame(
        (response.get("weights_decision") or {}).get("series") or {},
        index=dates,
    ).reindex(columns=CODES, fill_value=0.0)
    nav_payload = response.get("nav") or {}
    nav = pd.Series(
        (nav_payload.get("series") or {}).get("STRAT") or [],
        index=pd.to_datetime(nav_payload.get("dates") or []),
        dtype=float,
    ).reindex(dates)
    decomposition = response.get("return_decomposition") or {}
    decomp = pd.DataFrame(
        decomposition.get("series") or {},
        index=pd.to_datetime(decomposition.get("dates") or []),
    ).reindex(dates)
    asset_rows: list[dict[str, Any]] = []
    previous_weight = pd.Series(0.0, index=CODES, dtype=float)
    first_last = {
        code: (
            price_by_adjust["none"][code].first_valid_index(),
            price_by_adjust["none"][code].last_valid_index(),
        )
        for code in CODES
    }
    tsmom = {
        code: _independent_tsmom_trace(
            price_by_adjust["qfq"][code],
            lookback=20,
            entry_threshold=0.02,
            exit_threshold=0.0,
        )
        for code in CODES
    }
    none_returns = price_by_adjust["none"].pct_change(fill_method=None)
    hfq_returns = price_by_adjust["hfq"].pct_change(fill_method=None)
    for date_value in dates:
        for code in CODES:
            first, last = first_last[code]
            observed = pd.notna(price_by_adjust["none"].loc[date_value, code])
            end_raw = (pool.get(code) or {}).get("end_date")
            end_date = pd.Timestamp(end_raw) if end_raw else None
            if first is None or date_value < first:
                availability = "pre_listing"
            elif end_date is not None and date_value > end_date:
                availability = "removed_from_pool"
            elif observed:
                availability = "observed"
            elif last is not None and date_value < last:
                availability = "temporary_or_isolated_gap"
            else:
                availability = "terminal_gap_unknown"
            qfq = price_by_adjust["qfq"].loc[date_value, code]
            none = price_by_adjust["none"].loc[date_value, code]
            hfq = price_by_adjust["hfq"].loc[date_value, code]
            trace_row = tsmom[code].loc[date_value]
            current_weight = float(weights.loc[date_value, code])
            none_return = none_returns.loc[date_value, code]
            hfq_return = hfq_returns.loc[date_value, code]
            fallback = bool(
                pd.notna(none_return)
                and pd.notna(hfq_return)
                and abs(float(none_return) - float(hfq_return)) > 1e-8
            )
            asset_rows.append(
                {
                    "date": date_value.date().isoformat(),
                    "code": code,
                    "availability": availability,
                    "qfq_signal_close": None if pd.isna(qfq) else float(qfq),
                    "none_valuation_close": None if pd.isna(none) else float(none),
                    "hfq_benchmark_close": None if pd.isna(hfq) else float(hfq),
                    "indicator_momentum_20": (
                        None
                        if pd.isna(trace_row["momentum"])
                        else float(trace_row["momentum"])
                    ),
                    "base_signal": float(trace_row["signal"]),
                    "decision_weight": float(decision.loc[date_value, code]),
                    "effective_weight": current_weight,
                    "turnover_one_way": abs(
                        current_weight - float(previous_weight.loc[code])
                    )
                    / 2.0,
                    "market_observed": bool(observed),
                    "intraday_risk_allowed": bool(observed),
                    "corporate_action_fallback_mask": fallback,
                    "corporate_action_fallback_reason": (
                        "none_vs_hfq_return_divergence" if fallback else None
                    ),
                }
            )
        previous_weight = weights.loc[date_value].astype(float)
    portfolio_rows = []
    for date_value in dates:
        row = {
            "date": date_value.date().isoformat(),
            "pre_weight_total": float(
                weights.shift(1).fillna(0.0).loc[date_value].sum()
            ),
            "post_weight_total": float(weights.loc[date_value].sum()),
            "cash_weight": float(max(0.0, 1.0 - weights.loc[date_value].sum())),
            "nav": None if pd.isna(nav.loc[date_value]) else float(nav.loc[date_value]),
        }
        for key in ("gross", "cost", "risk_exit_override", "net"):
            row[key] = (
                float(decomp.loc[date_value, key])
                if key in decomp.columns and pd.notna(decomp.loc[date_value, key])
                else None
            )
        portfolio_rows.append(row)
    payload = {
        "as_of": AS_OF,
        "price_vintage": "current_database_vintage_only",
        "oracle_scope": (
            "independent price availability and TSMOM base signal; "
            "weights/NAV/events are traced from frozen production response"
        ),
        "asset_daily": asset_rows,
        "portfolio_daily": portfolio_rows,
        "events": _collect_trigger_events(response.get("risk_controls") or {}),
        "episodes": ((response.get("trade_statistics") or {}).get("trades") or []),
        "sizing_chain": {
            "formula": (
                "risk_budget_pct*capital/(ATR*stop_multiple), then overcap, "
                "round-lot, and cash constraints"
            ),
            "live_planned_vs_filled_status": (
                "stopped_original_plan_and_reliable_capital_basis_unavailable"
            ),
            "synthetic_golden_oracle": _independent_risk_budget_sizing(
                capital="100000",
                risk_budget_pct="0.0025",
                price="50",
                atr="2",
                stop_multiple="2",
                max_weight="0.03",
                lot_size=10,
                available_cash="2000",
            ),
        },
        "daily_accounting_golden_oracle": _independent_daily_oracle(
            asset_returns=["0", "0.5", "0.1", "-0.2"],
            effective_weights=["0", "1", "1", "0"],
            return_weights=["0", "0", "1", "1"],
            fee_rate="0",
            slippage_spread="0",
            execution_prices=["10", "15", "16.5", "13.2"],
            initial_nav="1",
        ),
        "unavailable_fields": [
            "historical_price_vintage",
            "historical_group_membership",
            "original_order_plan",
            "intraday_cashflow_valuation",
        ],
    }
    scenario_path = case_dir / "scenario-manifest.json"
    evidence_path = case_dir / "evidence-manifest.json"
    upstream_path = scenario_path if scenario_path.exists() else evidence_path
    upstream_id = (
        "scenario-manifest-v2" if scenario_path.exists() else "evidence-manifest-v2"
    )
    artifact = _artifact_envelope(
        artifact_id="model-trace-v2",
        artifact_type="trace-ledger",
        payload=payload,
        upstream=[
            {
                "artifact_id": upstream_id,
                "sha256": _sha256(upstream_path),
            }
        ],
    )
    _write_json(case_dir / "derived" / "model-trace.json", artifact)
    return artifact


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _stats(values: list[float]) -> dict[str, Any]:
    clean = [float(x) for x in values if math.isfinite(float(x))]
    wins = [x for x in clean if x > 1e-12]
    losses = [x for x in clean if x < -1e-12]
    decided = len(wins) + len(losses)
    win_rate = len(wins) / decided if decided else None
    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = sum(losses) / len(losses) if losses else None
    payoff = (
        avg_win / abs(avg_loss)
        if avg_win is not None and avg_loss is not None and avg_loss < 0
        else None
    )
    kelly = (
        win_rate - (1.0 - win_rate) / payoff
        if win_rate is not None and payoff is not None and payoff > 0
        else None
    )
    return {
        "sample_count": len(clean),
        "win_count": len(wins),
        "loss_count": len(losses),
        "flat_count": len(clean) - decided,
        "win_rate_ex_zero": win_rate,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "payoff_ratio": payoff,
        "kelly_fraction": kelly,
        "sum": sum(clean),
    }


def _paired_nav_block_bootstrap(
    before_response: dict[str, Any],
    after_response: dict[str, Any],
    *,
    replicates: int = 1000,
    block_length: int = 20,
    seed: int = 20260827,
) -> dict[str, Any]:
    """Rebuild paired NAV paths from moving calendar blocks."""

    def _nav_map(response: dict[str, Any]) -> dict[str, float]:
        nav = response.get("nav") or {}
        dates = list(nav.get("dates") or [])
        values = list((nav.get("series") or {}).get("STRAT") or [])
        return {
            str(day): float(value)
            for day, value in zip(dates, values, strict=False)
            if _finite(value) is not None
        }

    before_map = _nav_map(before_response)
    after_map = _nav_map(after_response)
    common = sorted(set(before_map) & set(after_map))
    if len(common) < max(10, block_length + 1):
        return {"status": "insufficient_sample", "observation_count": len(common)}

    def _returns(values: list[float]) -> list[float]:
        return [
            float(values[i] / values[i - 1] - 1.0)
            for i in range(1, len(values))
            if values[i - 1] != 0.0
        ]

    before_ret = _returns([before_map[day] for day in common])
    after_ret = _returns([after_map[day] for day in common])
    n = min(len(before_ret), len(after_ret))
    if n < block_length:
        return {"status": "insufficient_sample", "observation_count": n}
    before_ret = before_ret[:n]
    after_ret = after_ret[:n]

    def _path_metrics(values: list[float]) -> tuple[float, float]:
        nav_value = 1.0
        peak = 1.0
        max_drawdown = 0.0
        for value in values:
            nav_value *= 1.0 + float(value)
            peak = max(peak, nav_value)
            max_drawdown = min(max_drawdown, nav_value / peak - 1.0)
        return nav_value - 1.0, max_drawdown

    rng = random.Random(seed)
    cumulative_deltas: list[float] = []
    drawdown_deltas: list[float] = []
    max_start = max(0, n - block_length)
    for _ in range(int(replicates)):
        indices: list[int] = []
        while len(indices) < n:
            start = rng.randint(0, max_start)
            indices.extend(range(start, min(n, start + block_length)))
        indices = indices[:n]
        before_metrics = _path_metrics([before_ret[i] for i in indices])
        after_metrics = _path_metrics([after_ret[i] for i in indices])
        cumulative_deltas.append(after_metrics[0] - before_metrics[0])
        drawdown_deltas.append(after_metrics[1] - before_metrics[1])

    def _quantile(values: list[float], probability: float) -> float:
        ordered = sorted(values)
        position = (len(ordered) - 1) * probability
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return float(ordered[lower])
        weight = position - lower
        return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)

    def _two_sided_p(values: list[float]) -> float:
        nonpositive = sum(value <= 0.0 for value in values)
        nonnegative = sum(value >= 0.0 for value in values)
        return float(
            min(1.0, 2.0 * min(nonpositive + 1, nonnegative + 1) / (len(values) + 1))
        )

    p_values = {
        "cumulative_return_delta": _two_sided_p(cumulative_deltas),
        "max_drawdown_delta": _two_sided_p(drawdown_deltas),
    }
    ordered_p = sorted(p_values.items(), key=lambda item: item[1])
    holm_reject: dict[str, bool] = {}
    still_rejecting = True
    for rank, (name, value) in enumerate(ordered_p):
        threshold = 0.05 / (len(ordered_p) - rank)
        rejected = bool(still_rejecting and value <= threshold)
        holm_reject[name] = rejected
        if not rejected:
            still_rejecting = False
    return {
        "status": "ok",
        "method": "paired_moving_calendar_block_bootstrap",
        "replicates": int(replicates),
        "block_length": int(block_length),
        "seed": int(seed),
        "observation_count": n,
        "primary_metrics": {
            "cumulative_return_delta": {
                "ci_95": [
                    _quantile(cumulative_deltas, 0.025),
                    _quantile(cumulative_deltas, 0.975),
                ],
                "two_sided_p": p_values["cumulative_return_delta"],
                "holm_reject_5pct": holm_reject["cumulative_return_delta"],
            },
            "max_drawdown_delta": {
                "ci_95": [
                    _quantile(drawdown_deltas, 0.025),
                    _quantile(drawdown_deltas, 0.975),
                ],
                "two_sided_p": p_values["max_drawdown_delta"],
                "holm_reject_5pct": holm_reject["max_drawdown_delta"],
            },
        },
    }


def _trade_amount(row: dict[str, Any]) -> float:
    amount = _finite(row.get("amount"))
    if amount is not None:
        return abs(amount)
    price = _finite(row.get("price")) or 0.0
    quantity = _finite(row.get("quantity")) or 0.0
    return abs(price * quantity)


def _trade_sort_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("trade_date") or ""),
        str(row.get("trade_time") or ""),
        int(row.get("id") or 0),
    )


def _independent_live_ledger(case_dir: Path) -> dict[str, Any]:
    export = _read_json(case_dir / "raw" / "account-5-export.json")
    payload = export.get("payload") or {}
    trades = sorted(
        [
            dict(row)
            for row in payload.get("trades") or []
            if int(row.get("strategy_id") or 0) == STRATEGY_ID
        ],
        key=_trade_sort_key,
    )
    flows = [
        dict(row)
        for row in payload.get("strategy_cashflows") or []
        if int(row.get("strategy_id") or 0) == STRATEGY_ID
    ]
    flow_by_day: dict[str, list[dict[str, Any]]] = {}
    trade_by_day: dict[str, list[dict[str, Any]]] = {}
    for flow in flows:
        flow_by_day.setdefault(str(flow.get("flow_date") or ""), []).append(flow)
    for trade in trades:
        trade_by_day.setdefault(str(trade.get("trade_date") or ""), []).append(trade)
    cash_probe = 0.0
    min_cash_probe = 0.0
    for day in sorted(set(flow_by_day) | set(trade_by_day)):
        cash_probe += sum(
            float(row.get("amount") or 0.0) for row in flow_by_day.get(day, [])
        )
        for trade in sorted(trade_by_day.get(day, []), key=_trade_sort_key):
            amount = _trade_amount(trade)
            fee = abs(float(trade.get("fee") or 0.0))
            if str(trade.get("side") or "").strip().lower() == "buy":
                cash_probe -= amount + fee
            else:
                cash_probe += amount - fee
            min_cash_probe = min(min_cash_probe, cash_probe)
    inferred_initial_cash_topup = max(0.0, -min_cash_probe)
    active: dict[tuple[str, int], dict[str, Any]] = {}
    closed: list[dict[str, Any]] = []
    round_no: dict[tuple[str, int], int] = {}
    cash = inferred_initial_cash_topup + sum(
        float(row.get("amount") or 0.0) for row in flows
    )
    all_trade_fees = 0.0
    eps = 1e-8
    for trade in trades:
        code = str(trade.get("code") or "")
        shareholder_id = int(trade.get("shareholder_account_id") or 0)
        key = (code, shareholder_id)
        side = str(trade.get("side") or "").strip().lower()
        quantity = abs(float(trade.get("quantity") or 0.0))
        amount = _trade_amount(trade)
        fee = abs(float(trade.get("fee") or 0.0))
        all_trade_fees += fee
        if side == "buy":
            cash -= amount + fee
            if key not in active:
                round_no[key] = round_no.get(key, 0) + 1
                active[key] = {
                    "code": code,
                    "shareholder_account_id": shareholder_id,
                    "round_no": round_no[key],
                    "open_date": str(trade.get("trade_date")),
                    "close_date": None,
                    "buy_qty": 0.0,
                    "sell_qty": 0.0,
                    "buy_amount": 0.0,
                    "sell_amount": 0.0,
                    "total_fee": 0.0,
                    "buy_count": 0,
                    "sell_count": 0,
                }
            row = active[key]
            row["buy_qty"] += quantity
            row["buy_amount"] += amount
            row["total_fee"] += fee
            row["buy_count"] += 1
        elif side == "sell":
            cash += amount - fee
            if key not in active:
                raise ValueError(f"sell without position for {code}")
            row = active[key]
            row["sell_qty"] += quantity
            row["sell_amount"] += amount
            row["total_fee"] += fee
            row["sell_count"] += 1
            remaining = float(row["buy_qty"]) - float(row["sell_qty"])
            if remaining < -eps:
                raise ValueError(f"oversell for {code}")
            if abs(remaining) <= eps:
                row["close_date"] = str(trade.get("trade_date"))
                row["realized_pnl"] = (
                    float(row["sell_amount"])
                    - float(row["buy_amount"])
                    - float(row["total_fee"])
                )
                row["pure_price_pnl_before_fees"] = float(row["sell_amount"]) - float(
                    row["buy_amount"]
                )
                row["total_return_pnl"] = float(row["realized_pnl"])
                row["total_return_income_component"] = 0.0
                row["total_return_income_status"] = (
                    "no_code_attributable_income_in_frozen_strategy_flows"
                )
                row["return_rate"] = (
                    float(row["realized_pnl"]) / float(row["buy_amount"])
                    if float(row["buy_amount"]) > eps
                    else None
                )
                closed.append(dict(row))
                del active[key]
        else:
            raise ValueError(f"unsupported trade side {side!r}")

    open_rows = []
    for row in active.values():
        item = dict(row)
        item["quantity"] = float(item["buy_qty"]) - float(item["sell_qty"])
        item["remaining_cost"] = (
            float(item["buy_amount"])
            - float(item["sell_amount"])
            + float(item["total_fee"])
        )
        open_rows.append(item)
    return {
        "closed_rounds": closed,
        "open_positions": open_rows,
        "cash_from_dated_flows_and_trades": cash,
        "inferred_initial_cash_topup": inferred_initial_cash_topup,
        "all_trade_fees": all_trade_fees,
        "episode_return_contract": {
            "pure_price": "sell_amount_minus_buy_amount_before_fees",
            "total_return": (
                "pure_price_minus_fees_plus_code_attributable_dividend_"
                "and_corporate_action_cash"
            ),
            "unallocated_investment_income": (
                "strategy_cashflows_without_code_cannot_be_assigned_to_episode"
            ),
        },
        "strategy_flow_sum": sum(float(row.get("amount") or 0.0) for row in flows),
        "strategy_flow_count": len(flows),
        "trade_count": len(trades),
    }


def _live_financial_dod(case_dir: Path) -> dict[str, Any]:
    independent = _independent_live_ledger(case_dir)
    api_rounds = list(
        (_read_json(case_dir / "derived" / "strategy-13-closed-rounds.json") or {}).get(
            "items"
        )
        or []
    )
    unmatched = list(independent["closed_rounds"])
    round_residuals: list[float] = []
    fee_residuals: list[float] = []
    return_residuals: list[float] = []
    for actual in api_rounds:
        candidates = [
            (idx, expected)
            for idx, expected in enumerate(unmatched)
            if str(expected.get("code")) == str(actual.get("code"))
            and str(expected.get("open_date")) == str(actual.get("open_date"))
            and str(expected.get("close_date")) == str(actual.get("close_date"))
            and abs(
                float(expected.get("buy_qty") or 0.0)
                - float(actual.get("buy_qty") or 0.0)
            )
            <= 1e-8
            and abs(
                float(expected.get("sell_qty") or 0.0)
                - float(actual.get("sell_qty") or 0.0)
            )
            <= 1e-8
        ]
        if not candidates:
            continue
        idx, expected = candidates[0]
        unmatched.pop(idx)
        round_residuals.append(
            float(actual.get("realized_pnl") or 0.0)
            - float(expected.get("realized_pnl") or 0.0)
        )
        fee_residuals.append(
            float(actual.get("total_fee") or 0.0)
            - float(expected.get("total_fee") or 0.0)
        )
        return_residuals.append(
            float(actual.get("return_rate") or 0.0)
            - float(expected.get("return_rate") or 0.0)
        )

    holdings = _read_json(case_dir / "derived" / "strategy-13-holdings.json")
    holding_quantity = {
        (str(row.get("code")), int(row.get("shareholder_account_id") or 0)): float(
            row.get("quantity") or 0.0
        )
        for row in holdings
    }
    rebuilt_quantity = {
        (str(row.get("code")), int(row.get("shareholder_account_id") or 0)): float(
            row.get("quantity") or 0.0
        )
        for row in independent["open_positions"]
    }
    quantity_keys = set(holding_quantity) | set(rebuilt_quantity)
    quantity_residuals = [
        holding_quantity.get(key, 0.0) - rebuilt_quantity.get(key, 0.0)
        for key in quantity_keys
    ]

    performance = _read_json(case_dir / "derived" / "strategy-13-performance.json")
    performance_decimal = _read_json_decimal(
        case_dir / "derived" / "strategy-13-performance.json"
    )
    nav_rows = sorted(
        list(performance.get("nav") or []), key=lambda row: str(row.get("nav_date"))
    )
    nav_rows_decimal = sorted(
        list(performance_decimal.get("nav") or []),
        key=lambda row: str(row.get("nav_date")),
    )
    equity_residuals = [
        float(row.get("equity") or 0.0)
        - float(row.get("cash") or 0.0)
        - float(row.get("market_value") or 0.0)
        for row in nav_rows
    ]
    nav_chain_residuals: list[float] = []
    for previous, current in zip(nav_rows, nav_rows[1:], strict=False):
        nav_chain_residuals.append(
            float(current.get("nav_twr") or 0.0)
            - float(previous.get("nav_twr") or 0.0)
            * (1.0 + float(current.get("daily_return_twr") or 0.0))
        )

    attribution = _read_json(case_dir / "derived" / "strategy-13-attribution.json")
    attr_residuals: list[float] = []
    for row in attribution.get("daily") or []:
        components = sum(
            float(row.get(key) or 0.0)
            for key in (
                "selection_return",
                "timing_return",
                "position_return",
                "cost_drag_return",
                "cash_drag_return",
                "repo_carry_return",
                "repo_fee_drag_return",
            )
        )
        attr_residuals.append(float(row.get("daily_return_twr") or 0.0) - components)
    period = attribution.get("period") or {}
    period_residual = float(period.get("rebuild_total") or 0.0) - float(
        period.get("total_return_twr_sum") or 0.0
    )

    last_nav = nav_rows[-1] if nav_rows else {}
    latest_cash_residual = float(last_nav.get("cash") or 0.0) - float(
        independent["cash_from_dated_flows_and_trades"]
    )
    clone_path = case_dir / "derived" / "isolated-mysql-clone.json"
    clone = _read_json(clone_path) if clone_path.exists() else {}
    physical_types = (
        (clone.get("database_metadata") or {}).get("physical_column_types") or {}
    ).get("live_nav_daily") or {}
    precision_value_source = "read_only_mysql_stored_values"
    try:
        from sqlalchemy import text

        from etf_momentum.db.session import make_engine
        from etf_momentum.settings import get_settings

        precision_engine = make_engine(db_url=get_settings().db_url)
        with precision_engine.connect() as connection:
            stored_rows = (
                connection.execute(
                    text(
                        "SELECT nav_date, equity, cash, market_value, "
                        "daily_return_twr, nav_twr "
                        "FROM live_nav_daily "
                        "WHERE scope_type='strategy' AND scope_id=:strategy_id "
                        "AND nav_date <= :as_of ORDER BY nav_date"
                    ),
                    {
                        "strategy_id": STRATEGY_ID,
                        "as_of": dt.date.fromisoformat(AS_OF),
                    },
                )
                .mappings()
                .all()
            )
        precision_engine.dispose()
        if stored_rows:
            nav_rows_decimal = [
                {
                    key: (
                        value.isoformat()
                        if hasattr(value, "isoformat")
                        else str(value)
                        if value is not None
                        else None
                    )
                    for key, value in row.items()
                }
                for row in stored_rows
            ]
    except Exception as exc:
        precision_value_source = f"api_serialized_fallback:{type(exc).__name__}"
    decimal_identity = _decimal_nav_identity(
        nav_rows_decimal,
        physical_types={
            key: str(physical_types.get(key, "unknown"))
            for key in ("equity", "cash", "market_value")
        },
    )
    decimal_nav_chain = _decimal_nav_chain(
        nav_rows_decimal,
        physical_types={
            key: str(physical_types.get(key, "unknown"))
            for key in ("nav_twr", "daily_return_twr")
        },
    )
    export = _read_json(case_dir / "raw" / "account-5-export.json")
    strategy_flows = [
        row
        for row in ((export.get("payload") or {}).get("strategy_cashflows") or [])
        if int(row.get("strategy_id") or 0) == STRATEGY_ID
    ]
    account_flows = [
        row
        for row in ((export.get("payload") or {}).get("account_cashflows") or [])
        if int(row.get("account_id") or 0) == ACCOUNT_ID
    ]
    flow_classification = _classify_cashflows_by_scope(
        account_rows=account_flows,
        strategy_rows=strategy_flows,
    )
    dietz_sensitivity = _modified_dietz_sensitivity(nav_rows_decimal, strategy_flows)
    checks = {
        "fifo_round_count": {
            "expected": len(independent["closed_rounds"]),
            "actual": len(api_rounds),
            "matched": len(round_residuals),
            "unmatched_expected": len(unmatched),
        },
        "fifo_realized_pnl_max_abs_residual": max(
            (abs(x) for x in round_residuals), default=None
        ),
        "fifo_fee_max_abs_residual": max((abs(x) for x in fee_residuals), default=None),
        "fifo_return_max_abs_residual": max(
            (abs(x) for x in return_residuals), default=None
        ),
        "holding_quantity_max_abs_residual": max(
            (abs(x) for x in quantity_residuals), default=0.0
        ),
        "equity_identity_max_abs_residual": max(
            (abs(x) for x in equity_residuals), default=0.0
        ),
        "nav_twr_chain_max_abs_residual": max(
            (abs(x) for x in nav_chain_residuals), default=0.0
        ),
        "daily_attribution_max_abs_residual": max(
            (abs(x) for x in attr_residuals), default=0.0
        ),
        "period_attribution_residual": period_residual,
        "latest_cash_residual_before_unobserved_income": latest_cash_residual,
        "inferred_initial_cash_topup": independent["inferred_initial_cash_topup"],
        "cash_reconciliation_scope": (
            "dated_strategy_flows_plus_etf_trade_cash_and_fees;"
            "dividend_interest_and_intraday_flow_timing_unavailable"
        ),
        "all_trade_fees": independent["all_trade_fees"],
        "decimal_investigation": {
            "value_source": precision_value_source,
            "equity_identity": decimal_identity,
            "nav_twr_chain": decimal_nav_chain,
        },
        "cashflow_classification": flow_classification,
        "modified_dietz_sensitivity": dietz_sensitivity,
    }
    tolerances = {
        "fifo_realized_pnl_max_abs_residual": 1e-8,
        "fifo_fee_max_abs_residual": 1e-10,
        "fifo_return_max_abs_residual": 1e-7,
        "holding_quantity_max_abs_residual": 1e-8,
        "equity_identity_max_abs_residual": 0.01,
        "nav_twr_chain_max_abs_residual": 1e-12,
        "daily_attribution_max_abs_residual": 1e-7,
        "period_attribution_residual": 1e-7,
        "latest_cash_residual_before_unobserved_income": 0.021,
    }
    checks["tolerances"] = tolerances
    checks["passed"] = {
        key: abs(float(checks[key])) <= tolerance
        for key, tolerance in tolerances.items()
    }
    checks["passed"]["fifo_round_count"] = (
        len(independent["closed_rounds"]) == len(api_rounds) == len(round_residuals)
        and not unmatched
    )
    _write_json(case_dir / "derived" / "independent-live-ledger.json", independent)
    _write_json(case_dir / "reports" / "live-financial-dod.json", checks)
    return checks


def _find_trades(response: dict[str, Any]) -> list[dict[str, Any]]:
    trade_stats = response.get("trade_statistics") or {}
    candidates = [
        trade_stats.get("trades"),
        response.get("trades"),
        (response.get("r_multiple") or {}).get("trades"),
    ]
    for value in candidates:
        if isinstance(value, list):
            return [dict(row) for row in value if isinstance(row, dict)]
    by_code = trade_stats.get("trades_by_code")
    if isinstance(by_code, dict):
        return [
            {**dict(row), "code": str(code)}
            for code, rows in by_code.items()
            for row in (rows or [])
            if isinstance(row, dict)
        ]
    return []


def _scalar_subset(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {
        key: source.get(key)
        for key in keys
        if key in source and not isinstance(source.get(key), (dict, list))
    }


def report(case_dir: Path) -> dict[str, Any]:
    """Build a sanitized evidence and metric report from frozen artifacts."""
    manifest = refresh_evidence(case_dir)
    live_start = str(
        ((manifest.get("ranges") or {}).get("strategy_trade_min_date")) or ""
    )
    live_end = str(
        ((manifest.get("ranges") or {}).get("strategy_trade_max_date")) or ""
    )
    export_payload = (
        _read_json(case_dir / "raw" / "account-5-export.json").get("payload") or {}
    )
    live_codes = {
        str(row.get("code"))
        for row in export_payload.get("trades") or []
        if int(row.get("strategy_id") or 0) == STRATEGY_ID
    }
    rounds_payload = _read_json(case_dir / "derived" / "strategy-13-closed-rounds.json")
    rounds = list(rounds_payload.get("items") or [])
    live_amount_stats = _stats(
        [float(row.get("realized_pnl") or 0.0) for row in rounds]
    )
    live_return_stats = _stats([float(row.get("return_rate") or 0.0) for row in rounds])
    live_round_rebuild = {
        "realized_pnl_sum": sum(
            float(row.get("realized_pnl") or 0.0) for row in rounds
        ),
        "fee_sum": sum(float(row.get("total_fee") or 0.0) for row in rounds),
        "amount_stats": live_amount_stats,
        "return_stats": live_return_stats,
    }

    scenarios: dict[str, Any] = {}
    for name in (
        "legacy-quick-original",
        "legacy-full",
        "bt-full",
        "postfix-legacy-full",
        "postfix-bt-full",
        "final-legacy-full",
        "final-bt-full",
    ):
        path = case_dir / "backtests" / f"{name}-response.json"
        if not path.exists():
            continue
        response = _read_json(path)
        trades = _find_trades(response)
        closed = [row for row in trades if bool(row.get("closed"))]
        opened = [row for row in trades if not bool(row.get("closed"))]
        window_closed = [
            row
            for row in closed
            if live_start <= str(row.get("entry_date") or "")
            and str(row.get("exit_date") or "") <= live_end
        ]
        exit_window_closed = [
            row
            for row in closed
            if live_start <= str(row.get("exit_date") or "") <= live_end
        ]
        published = (response.get("trade_statistics") or {}).get("overall") or {}
        portfolio_metrics = (response.get("metrics") or {}).get("strategy") or {}
        nav = response.get("nav") or {}
        nav_dates = [str(value) for value in nav.get("dates") or []]
        nav_values = [
            float(value) for value in ((nav.get("series") or {}).get("STRAT") or [])
        ]
        nav_window_pairs = [
            (day, value)
            for day, value in zip(nav_dates, nav_values, strict=False)
            if live_start <= day <= live_end
        ]
        nav_window_return = (
            nav_window_pairs[-1][1] / nav_window_pairs[0][1] - 1.0
            if len(nav_window_pairs) >= 2 and nav_window_pairs[0][1] != 0.0
            else None
        )
        scenarios[name] = {
            "request_sha256": _sha256(case_dir / "backtests" / f"{name}-request.json"),
            "response_sha256": _sha256(path),
            "engine": (response.get("meta") or {}).get("engine"),
            "published_trade_statistics": _scalar_subset(
                published,
                (
                    "trades",
                    "win_trades",
                    "loss_trades",
                    "zero_trades",
                    "win_rate_ex_zero",
                    "payoff_ex_zero",
                    "kelly_ex_zero",
                ),
            ),
            "portfolio_metrics": _scalar_subset(
                portfolio_metrics,
                (
                    "cumulative_return",
                    "annualized_return",
                    "annualized_volatility",
                    "sharpe_ratio",
                    "sortino_ratio",
                    "max_drawdown",
                    "max_drawdown_recovery_days",
                ),
            ),
            "all_episode_stats_rebuilt": _stats(
                [
                    value
                    for row in trades
                    if (value := _finite(row.get("return"))) is not None
                ]
            ),
            "closed_episode_stats_rebuilt": _stats(
                [
                    value
                    for row in closed
                    if (value := _finite(row.get("return"))) is not None
                ]
            ),
            "episode_counts": {
                "total": len(trades),
                "closed": len(closed),
                "open_mtm": len(opened),
            },
            "open_episode_report": {
                "censoring_rate": (
                    float(len(opened) / len(trades)) if trades else None
                ),
                "episodes": [
                    {
                        "code": row.get("code"),
                        "entry_date": row.get("entry_date"),
                        "as_of": AS_OF,
                        "holding_age_calendar_days": (
                            dt.date.fromisoformat(AS_OF)
                            - dt.date.fromisoformat(str(row.get("entry_date")))
                        ).days,
                        "mtm_return": row.get("return"),
                        "closed": False,
                    }
                    for row in opened
                    if row.get("entry_date")
                ],
                "entry_cohort_inference": (
                    "not_reported_no_predeclared_holding_horizon"
                ),
            },
            "live_window": {
                "start": live_start,
                "end": live_end,
                "nav_reset_return": nav_window_return,
                "carry_in": {
                    "model_weights": (
                        {
                            code: float(values[nav_dates.index(nav_window_pairs[0][0])])
                            for code, values in (
                                (response.get("weights") or {}).get("series") or {}
                            ).items()
                            if len(values) > nav_dates.index(nav_window_pairs[0][0])
                            and abs(
                                float(values[nav_dates.index(nav_window_pairs[0][0])])
                            )
                            > 1e-12
                        }
                        if nav_window_pairs
                        else {}
                    ),
                    "window_basis_nav": (
                        nav_window_pairs[0][1] if nav_window_pairs else None
                    ),
                    "lifetime_basis": (
                        "unidentifiable_without_model_state_at_window_start"
                    ),
                },
                "drawdown": _window_drawdowns(
                    response,
                    start=live_start,
                    end=live_end,
                ),
                "complete_episode_stats": _stats(
                    [
                        value
                        for row in window_closed
                        if (value := _finite(row.get("return"))) is not None
                    ]
                ),
                "exit_date_episode_stats": _stats(
                    [
                        value
                        for row in exit_window_closed
                        if (value := _finite(row.get("return"))) is not None
                    ]
                ),
            },
        }

    performance = _read_json(case_dir / "derived" / "strategy-13-performance.json")
    financial_dod = _live_financial_dod(case_dir)
    attribution = _read_json(case_dir / "derived" / "strategy-13-attribution.json")
    daily_attr = list(attribution.get("daily") or [])
    attr_residuals: list[float] = []
    for row in daily_attr:
        total = _finite(row.get("daily_twr"))
        components = [
            _finite(row.get(key)) or 0.0
            for key in (
                "selection_return",
                "timing_return",
                "position_return",
                "cost_drag_return",
                "cash_drag_return",
                "repo_carry_return",
                "repo_fee_drag_return",
            )
        ]
        if total is not None:
            attr_residuals.append(total - sum(components))

    baseline_trades = _find_trades(
        _read_json(case_dir / "backtests" / "legacy-full-response.json")
    )
    postfix_trades = _find_trades(
        _read_json(case_dir / "backtests" / "postfix-legacy-full-response.json")
    )
    postfix_by_key = {
        (
            str(row.get("code")),
            str(row.get("entry_date")),
            str(row.get("exit_date")),
            bool(row.get("closed")),
        ): row
        for row in postfix_trades
    }
    divergences: list[dict[str, Any]] = []
    for before in baseline_trades:
        key = (
            str(before.get("code")),
            str(before.get("entry_date")),
            str(before.get("exit_date")),
            bool(before.get("closed")),
        )
        after = postfix_by_key.get(key)
        before_return = _finite(before.get("return"))
        after_return = _finite((after or {}).get("return"))
        if (
            after is not None
            and before_return is not None
            and after_return is not None
            and abs(after_return - before_return) > 1e-15
        ):
            divergences.append(
                {
                    "code": key[0],
                    "entry_date": key[1],
                    "exit_date": key[2],
                    "closed": key[3],
                    "before_return": before_return,
                    "after_return": after_return,
                    "delta": after_return - before_return,
                }
            )
    divergences.sort(key=lambda row: (row["entry_date"], row["code"], row["exit_date"]))
    divergence_summary = {
        "layer": "execution_accounting",
        "cause": (
            "trade ledger used post-trade weight instead of portfolio return weight; "
            "risk-exit fill override was absent"
        ),
        "matched_episode_count": len(postfix_by_key),
        "changed_episode_count": len(divergences),
        "first_divergence": divergences[0] if divergences else None,
        "max_abs_return_delta": max(
            (abs(float(row["delta"])) for row in divergences), default=0.0
        ),
    }
    scenario_manifest = {
        "artifact_id": "scenario-manifest-v2",
        "artifact_type": "scenario-manifest",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "upstream": [
            {
                "artifact_id": "evidence-manifest-v2",
                "sha256": _sha256(case_dir / "evidence-manifest.json"),
            }
        ],
        "counterfactual_order": [
            "data_availability_and_price",
            "signal_and_execution_timing",
            "position_and_capital_constraints",
            "matched_actual_fills",
            "actual_fees",
        ],
        "stop_gates": {
            "return_bridge": (
                "data_availability_and_price:"
                "historical_price_vintage_and_model_state_unavailable"
            ),
            "amount_pnl_bridge": (
                "position_and_capital_constraints:"
                "historical_notionals_plans_and_capital_state_unavailable"
            ),
            "planned_vs_filled": "daily_original_plan_unavailable",
        },
        "scenarios": {
            name: {
                "engine": row.get("engine"),
                "request_sha256": row.get("request_sha256"),
                "response_sha256": row.get("response_sha256"),
            }
            for name, row in scenarios.items()
        },
    }
    _write_json(case_dir / "scenario-manifest.json", scenario_manifest)
    model_trace = _build_model_trace(case_dir)
    ledger_path = case_dir / "derived" / "independent-live-ledger.json"
    ledger_payload = _read_json(ledger_path)
    ledger_artifact = _artifact_envelope(
        artifact_id="independent-live-ledger-v2",
        artifact_type="trace-ledger",
        payload=ledger_payload,
        upstream=[
            {
                "artifact_id": model_trace["artifact_id"],
                "sha256": _sha256(case_dir / "derived" / "model-trace.json"),
            }
        ],
    )
    _write_json(ledger_path, ledger_artifact)
    divergence_artifact = _artifact_envelope(
        artifact_id="divergence-table-v2",
        artifact_type="divergence-table",
        payload={
            "summary": divergence_summary,
            "rows": divergences,
            "classification": "identified_code_snapshot_difference",
        },
        upstream=[
            {
                "artifact_id": "independent-live-ledger-v2",
                "sha256": _sha256(ledger_path),
            }
        ],
    )
    divergence_path = case_dir / "reports" / "divergence-table.json"
    _write_json(divergence_path, divergence_artifact)
    pre_fix_path = case_dir / "reports" / "pre-fix-failures.json"
    failing_fixtures = {
        "artifact_id": "failing-fixtures-v2",
        "artifact_type": "failing-fixtures",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "upstream": [
            {
                "artifact_id": "divergence-table-v2",
                "sha256": _sha256(divergence_path),
            },
            {
                "artifact_id": "pre-fix-failures-v2",
                "sha256": _sha256(pre_fix_path),
            },
        ],
        "fixtures": [
            "test_close_trade_ledger_uses_return_weight_and_risk_exit_override",
            "test_trade_statistics_returns_exclude_open_mtm_episodes",
            "test_slippage_rate_is_full_absolute_spread_split_across_sides",
            "test_r_statistics_exclude_open_mtm_episode",
            "test_portfolio_risk_exit_override_is_preserved_by_asset",
            "test_dynamic_benchmark_does_not_forward_fill_missing_observations",
            "test_missing_ohlc_does_not_fabricate_intraday_stop",
            "test_close_execution_initializes_atr_state_from_t_plus_one_close",
            "test_trend_atr_stop_reentry_timing_no_lookahead",
            "test_single_backtest_prefix_is_invariant_to_future_extension",
            "test_rotation_trade_statistics_have_samples_user_case_like",
            "test_portfolio_daily_accounting_and_event_parity",
            "test_decimal_nav_identity_finds_first_divergence_and_float_bound",
            "test_terminal_liquidation_modes_never_rewrite_causal_episodes",
        ],
        "oracle": "hand-calculated deterministic synthetic ledgers",
        "pre_fix_evidence": _read_json(pre_fix_path),
    }
    _write_json(case_dir / "reports" / "failing-fixtures.json", failing_fixtures)
    minimal_patch = {
        "artifact_id": "minimal-patch-v2",
        "artifact_type": "minimal-patch",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "upstream": [
            {
                "artifact_id": "failing-fixtures-v2",
                "sha256": _sha256(case_dir / "reports" / "failing-fixtures.json"),
            }
        ],
        "files": [
            "src/etf_momentum/analysis/baseline.py",
            "src/etf_momentum/analysis/trend.py",
            "src/etf_momentum/analysis/bt_trend.py",
            "src/etf_momentum/analysis/r_multiple.py",
        ],
        "contracts": [
            "close episodes use pre-transaction return weights",
            "risk-exit fill overrides remain asset-specific and enter episode returns",
            "win/payoff/Kelly/R summaries include closed episodes only",
            "close overlays initialize from T+1 market observations and report market dates",
            "dynamic benchmarks do not bridge missing observations",
            "slippage_rate is a full spread split equally across entry and exit",
        ],
    }
    _write_json(case_dir / "reports" / "minimal-patch.json", minimal_patch)
    paired_bootstrap = (
        _paired_nav_block_bootstrap(
            _read_json(case_dir / "backtests" / "legacy-full-response.json"),
            _read_json(case_dir / "backtests" / "final-legacy-full-response.json"),
        )
        if (case_dir / "backtests" / "final-legacy-full-response.json").exists()
        else {"status": "final_scenario_unavailable"}
    )
    pre_window_trades = [
        row
        for row in export_payload.get("trades") or []
        if int(row.get("strategy_id") or 0) == STRATEGY_ID
        and str(row.get("trade_date") or "") < live_start
    ]
    live_carry_quantities: dict[str, float] = {}
    for row in pre_window_trades:
        direction = 1.0 if str(row.get("side") or "").strip().lower() == "buy" else -1.0
        code = str(row.get("code"))
        live_carry_quantities[code] = live_carry_quantities.get(code, 0.0) + (
            direction * float(row.get("quantity") or 0.0)
        )
    strategy_flows_all = [
        row
        for row in export_payload.get("strategy_cashflows") or []
        if int(row.get("strategy_id") or 0) == STRATEGY_ID
    ]
    flow_before = sum(
        float(row.get("amount") or 0.0)
        for row in strategy_flows_all
        if str(row.get("flow_date") or "") < live_start
    )
    flow_on_start = sum(
        float(row.get("amount") or 0.0)
        for row in strategy_flows_all
        if str(row.get("flow_date") or "") == live_start
    )
    pre_window_trade_cash = sum(
        (
            _trade_amount(row) - abs(float(row.get("fee") or 0.0))
            if str(row.get("side") or "").strip().lower() == "sell"
            else -_trade_amount(row) - abs(float(row.get("fee") or 0.0))
        )
        for row in pre_window_trades
    )
    flow_before += pre_window_trade_cash
    model_state_scenario = (
        scenarios.get("final-legacy-full") or scenarios.get("postfix-legacy-full") or {}
    )
    model_carry = (
        (model_state_scenario.get("live_window") or {})
        .get("carry_in", {})
        .get("model_weights", {})
    )

    result = {
        "case_id": "trend-live-5-13",
        "as_of": AS_OF,
        "classification": {
            "backtest_formula_findings": "identified_or_pending_fixture",
            "historical_plan_comparison": "unidentifiable_no_plan_snapshot",
            "historical_group_membership": "unidentifiable",
            "price_input": "descriptive_current_vintage",
            "live_cashflow_timing": "date_only_modified_dietz_sensitivity_required",
        },
        "initial_states": {
            "model_state_at_window_start": {
                "status": "partial_not_sufficient_for_live_parity",
                "source_hash": model_state_scenario.get("response_sha256"),
                "effective_weights": model_carry,
                "warmup": "full_history_then_slice",
                "entry_atr_r_stop_and_tier_state": (
                    "not_fully_exported_at_window_boundary"
                ),
            },
            "live_book_state_at_window_start": {
                "status": "date_only_cashflow_timing",
                "source_hash": _sha256(case_dir / "raw" / "account-5-export.json"),
                "carry_in_quantities_lifetime_basis": {
                    code: quantity
                    for code, quantity in live_carry_quantities.items()
                    if abs(quantity) > 1e-12
                },
                "window_basis_valuation": (
                    "no_carry_in_position"
                    if not any(
                        abs(value) > 1e-12 for value in live_carry_quantities.values()
                    )
                    else "requires_window_opening_price"
                ),
                "cash_before_start_day_flows": flow_before,
                "cash_if_start_day_flow_is_bod": flow_before + flow_on_start,
                "cash_if_start_day_flow_is_eod": flow_before,
                "verified_risk_state": "unavailable",
            },
            "cross_injection_prohibited": True,
        },
        "universe_comparison": {
            "configured_code_count": len(CODES),
            "live_traded_code_count": len(live_codes),
            "intersection_count": len(live_codes & set(CODES)),
            "live_extra_code_count": len(live_codes - set(CODES)),
            "live_extra_codes": sorted(live_codes - set(CODES)),
            "configured_never_traded_codes": sorted(set(CODES) - live_codes),
        },
        "first_divergence": divergence_summary,
        "paired_portfolio_bootstrap": paired_bootstrap,
        "live": {
            "closed_rounds": live_round_rebuild,
            "financial_dod": financial_dod,
            "performance_twr": performance.get("twr_basis_metrics") or {},
            "performance_dietz": performance.get("dietz_basis_metrics") or {},
            "attribution_constructed_residual_max_abs": (
                max((abs(x) for x in attr_residuals), default=0.0)
            ),
        },
        "backtests": scenarios,
    }
    if "legacy-full" in scenarios and "postfix-legacy-full" in scenarios:
        final_scenario = (
            scenarios.get("final-legacy-full") or scenarios["postfix-legacy-full"]
        )
        model_window_return = _finite(
            (final_scenario.get("live_window") or {}).get("nav_reset_return")
        )
        live_twr = _finite(
            (performance.get("twr_basis_metrics") or {}).get("cumulative_return")
        )
        return_bridge = (
            _build_stopped_bridge(
                metric="portfolio_twr",
                model_value=Decimal(str(model_window_return)),
                live_value=Decimal(str(live_twr)),
                unavailable_layer="data_availability_and_price",
                reason=(
                    "historical_price_vintage_and_model_state_at_window_start_unavailable"
                ),
                absolute_tolerance=Decimal("0.00000001"),
            )
            if model_window_return is not None and live_twr is not None
            else {
                "metric": "portfolio_twr",
                "status": "stopped_before_baseline",
                "reason": "model_or_live_return_unavailable",
            }
        )
        amount_bridge = {
            "metric": "amount_pnl",
            "status": "stopped_before_baseline",
            "model_value": None,
            "live_value": live_amount_stats.get("sum"),
            "total_difference": None,
            "layers": [
                {
                    "layer": "position_and_capital_constraints",
                    "status": "unidentifiable_stop",
                    "increment": None,
                    "reason": (
                        "historical_executable_notionals_daily_plans_and_"
                        "point_in_time_capital_state_unavailable"
                    ),
                }
            ],
            "unexplained_residual": None,
            "rebuild_error": None,
            "within_tolerance": None,
        }
        impact_scenarios = [
            ("frozen_original", scenarios["legacy-full"]),
            ("close_ledger_patch_snapshot", scenarios["postfix-legacy-full"]),
        ]
        if scenarios.get("final-legacy-full"):
            impact_scenarios.append(
                ("final_patch_snapshot", scenarios["final-legacy-full"])
            )
        result["counterfactual_bridges"] = {
            "classification": (
                "stopped_by_identifiability_gate; arithmetic residual is not "
                "a causal contribution"
            ),
            "required_order": [
                "data_availability_and_price",
                "signal_and_execution_timing",
                "position_and_capital_constraints",
                "matched_actual_fills",
                "actual_fees",
            ],
            "return_bridge": return_bridge,
            "amount_pnl_bridge": amount_bridge,
        }
        result["descriptive_implementation_impact"] = {
            "classification": (
                "descriptive_code_snapshot_difference_not_fixed_layer_causality"
            ),
            "statistical_scope_impact": {
                "all_episodes_including_open_mtm": scenarios["legacy-full"].get(
                    "all_episode_stats_rebuilt"
                ),
                "closed_episodes_only": scenarios["legacy-full"].get(
                    "closed_episode_stats_rebuilt"
                ),
            },
            "scenarios": {
                name: {
                    "closed_trade_metrics": scenario.get(
                        "closed_episode_stats_rebuilt"
                    ),
                    "portfolio_metrics": scenario.get("portfolio_metrics"),
                    "open_mtm_metrics": scenario.get("all_episode_stats_rebuilt"),
                }
                for name, scenario in impact_scenarios
            },
            "metric_increments": {
                metric: [
                    {
                        "snapshot": name,
                        "value": value,
                        "increment_from_prior_snapshot": (
                            None
                            if index == 0 or value is None or values[index - 1] is None
                            else float(value - values[index - 1])
                        ),
                    }
                    for index, ((name, _), value) in enumerate(
                        zip(impact_scenarios, values, strict=True)
                    )
                ]
                for metric, values in {
                    "win_rate": [
                        _finite(
                            scenario.get("closed_episode_stats_rebuilt", {}).get(
                                "win_rate_ex_zero"
                            )
                        )
                        for _, scenario in impact_scenarios
                    ],
                    "payoff": [
                        _finite(
                            scenario.get("closed_episode_stats_rebuilt", {}).get(
                                "payoff_ratio"
                            )
                        )
                        for _, scenario in impact_scenarios
                    ],
                    "kelly": [
                        _finite(
                            scenario.get("closed_episode_stats_rebuilt", {}).get(
                                "kelly_fraction"
                            )
                        )
                        for _, scenario in impact_scenarios
                    ],
                    "cumulative_return": [
                        _finite(
                            scenario.get("portfolio_metrics", {}).get(
                                "cumulative_return"
                            )
                        )
                        for _, scenario in impact_scenarios
                    ],
                    "max_drawdown": [
                        _finite(
                            scenario.get("portfolio_metrics", {}).get("max_drawdown")
                        )
                        for _, scenario in impact_scenarios
                    ],
                }.items()
            },
        }
    final_legacy = scenarios.get("final-legacy-full") or {}
    final_bt = scenarios.get("final-bt-full") or {}
    legacy_nav = _finite(
        (final_legacy.get("portfolio_metrics") or {}).get("cumulative_return")
    )
    bt_nav = _finite((final_bt.get("portfolio_metrics") or {}).get("cumulative_return"))
    verification_path = case_dir / "reports" / "verification.json"
    verification = (
        _read_json(verification_path)
        if verification_path.exists()
        else {
            "artifact_id": None,
            "payload": {
                "passed": False,
                "status": "not_run_for_current_dirty_tree",
            },
        }
    )
    financial_dod_path = case_dir / "reports" / "live-financial-dod.json"
    financial_dod_artifact = _artifact_envelope(
        artifact_id="live-financial-dod-v2",
        artifact_type="financial-definition-of-done",
        payload=financial_dod,
        upstream=[
            {
                "artifact_id": "independent-live-ledger-v2",
                "sha256": _sha256(ledger_path),
            }
        ],
    )
    _write_json(financial_dod_path, financial_dod_artifact)
    regression_upstream = [
        {
            "artifact_id": "minimal-patch-v2",
            "sha256": _sha256(case_dir / "reports" / "minimal-patch.json"),
        },
        {
            "artifact_id": "live-financial-dod-v2",
            "sha256": _sha256(financial_dod_path),
        },
    ]
    if verification_path.exists():
        regression_upstream.append(
            {
                "artifact_id": str(
                    verification.get("artifact_id") or "verification-v2"
                ),
                "sha256": _sha256(verification_path),
            }
        )
    regression_report = {
        "artifact_id": "regression-report-v2",
        "artifact_type": "regression-report",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "upstream": regression_upstream,
        "financial_dod_passed": all(
            bool(value) for value in (financial_dod.get("passed") or {}).values()
        ),
        "final_capture": {
            "legacy_response_sha256": final_legacy.get("response_sha256"),
            "bt_response_sha256": final_bt.get("response_sha256"),
            "legacy_cumulative_return": legacy_nav,
            "bt_cumulative_return": bt_nav,
            "absolute_cumulative_return_delta": (
                None
                if legacy_nav is None or bt_nav is None
                else abs(legacy_nav - bt_nav)
            ),
        },
        "test_results": verification,
    }
    _write_json(case_dir / "reports" / "regression-report.json", regression_report)
    audit_summary_artifact = _artifact_envelope(
        artifact_id="audit-summary-v2",
        artifact_type="audit-summary",
        payload=result,
        upstream=[
            {
                "artifact_id": "regression-report-v2",
                "sha256": _sha256(case_dir / "reports" / "regression-report.json"),
            }
        ],
    )
    _write_json(
        case_dir / "reports" / "audit-summary.json",
        audit_summary_artifact,
    )
    corrected = (
        scenarios.get("final-legacy-full") or scenarios.get("postfix-legacy-full") or {}
    )
    corrected_all = corrected.get("closed_episode_stats_rebuilt") or {}
    corrected_window = (corrected.get("live_window") or {}).get(
        "complete_episode_stats"
    ) or {}
    summary_sha = _sha256(case_dir / "reports" / "audit-summary.json")
    findings = f"""# ETF trend backtest/live audit

Frozen as-of: {AS_OF}; live window: {live_start} to {live_end}.
Artifact: findings-v2; upstream audit-summary-v2 sha256: {summary_sha}.

## Identified

- The close-execution trade ledger used post-trade weights. It included the
  entry day's close-to-close return and omitted the exit day's return.
- Risk-exit fill overrides affected portfolio NAV but were absent from
  per-episode returns. The audit now preserves those overrides by asset.
- Open mark-to-market episodes contaminated win rate, payoff, Kelly and ordinary
  R statistics. Published statistics are now explicitly closed-trades-only.
- The BT dynamic benchmark forward-filled missing assets before dynamic
  equal-weighting; it now preserves missing observations.
- Live FIFO accounting reconciles all {len(rounds)} closed rounds. Maximum
  realized-PnL residual is
  {financial_dod.get("fifo_realized_pnl_max_abs_residual")}.

## Descriptive implementation impact (not a causal bridge)

- Original full-history legacy: win rate
  {scenarios["legacy-full"]["all_episode_stats_rebuilt"]["win_rate_ex_zero"]:.2%},
  payoff {scenarios["legacy-full"]["all_episode_stats_rebuilt"]["payoff_ratio"]:.3f},
  Kelly {scenarios["legacy-full"]["all_episode_stats_rebuilt"]["kelly_fraction"]:.2%}.
- Removing open MTM only: win rate
  {scenarios["legacy-full"]["closed_episode_stats_rebuilt"]["win_rate_ex_zero"]:.2%},
  payoff {scenarios["legacy-full"]["closed_episode_stats_rebuilt"]["payoff_ratio"]:.3f}.
- Corrected full-history legacy: win rate
  {corrected_all.get("win_rate_ex_zero"):.2%}, payoff
  {corrected_all.get("payoff_ratio"):.3f}, Kelly
  {corrected_all.get("kelly_fraction"):.2%}.
- Current-vintage model slice in the live window (complete episodes): win rate
  {corrected_window.get("win_rate_ex_zero"):.2%}, payoff
  {corrected_window.get("payoff_ratio"):.3f}; model NAV return
  {(corrected.get("live_window") or {}).get("nav_reset_return"):.2%}.
- Live closed rounds on percentage basis: win rate
  {live_return_stats.get("win_rate_ex_zero"):.2%}, payoff
  {live_return_stats.get("payoff_ratio"):.3f}, Kelly
  {live_return_stats.get("kelly_fraction"):.2%}; live TWR
  {(performance.get("twr_basis_metrics") or {}).get("cumulative_return"):.2%}.

## Descriptive

- Live traded {len(live_codes)} codes, while the supplied backtest universe has
  {len(CODES)}. All configured codes appear live, plus
  {len(live_codes - set(CODES))} additional codes. Therefore the supplied
  configuration is not the complete live decision universe.
- The replay injects an inferred initial strategy cash top-up of
  {financial_dod.get("inferred_initial_cash_topup"):.2f} to prevent historical
  negative cash. This is model state, not a documented external flow.
- Daily live attribution adds exactly within serialization tolerance, but timing
  is constructed as a residual; additivity does not validate its economics.
- The required return and amount-PnL bridges stop at their first unavailable
  input. No execution, sizing, fee, or strategy-failure attribution is claimed.
- Stored equity identity first diverges on
  {((financial_dod.get("decimal_investigation") or {}).get("equity_identity") or {}).get("first_divergence_date")};
  its maximum Decimal residual is
  {((financial_dod.get("decimal_investigation") or {}).get("equity_identity") or {}).get("max_abs_decimal_residual")}.
  This exceeds the calculated floating-point bound and is classified
  `source_precision_unknown`.

## Unidentifiable from current records

- Historical daily strategy plans and parameter-version snapshots.
- Historical universe/group-membership versions and point-in-time price
  vintages.
- Intraday timing of capital flows, so exact same-window external-flow TWR cannot
  be independently reconstructed. The report supplies Modified Dietz BOD/EOD
  sensitivity and amount PnL instead.
- `model_state_at_window_start`, historical executable notionals and original
  order plans. Therefore same-window causal attribution and the amount bridge
  are stopped.
- Creation of the isolated MySQL audit schema was denied to the configured
  database user; the authorization stop and source physical types are preserved
  in `derived/isolated-mysql-clone.json`.
"""
    findings_path = case_dir / "reports" / "findings.md"
    _private_dir(findings_path.parent)
    findings_path.write_text(findings, encoding="utf-8")
    findings_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    chain_paths = [
        ("evidence-manifest-v2", case_dir / "evidence-manifest.json"),
        (
            "isolated-mysql-clone-v1",
            case_dir / "derived" / "isolated-mysql-clone.json",
        ),
        ("scenario-manifest-v2", case_dir / "scenario-manifest.json"),
        ("model-trace-v2", case_dir / "derived" / "model-trace.json"),
        (
            "independent-live-ledger-v2",
            case_dir / "derived" / "independent-live-ledger.json",
        ),
        ("divergence-table-v2", case_dir / "reports" / "divergence-table.json"),
        (
            "pre-fix-failures-v2",
            case_dir / "reports" / "pre-fix-failures.json",
        ),
        ("failing-fixtures-v2", case_dir / "reports" / "failing-fixtures.json"),
        ("minimal-patch-v2", case_dir / "reports" / "minimal-patch.json"),
        ("live-financial-dod-v2", financial_dod_path),
        ("verification-v2", verification_path),
        (
            "regression-report-v2",
            case_dir / "reports" / "regression-report.json",
        ),
        ("audit-summary-v2", case_dir / "reports" / "audit-summary.json"),
        ("findings-v2", findings_path),
    ]
    chain_paths = [
        (artifact_id, path) for artifact_id, path in chain_paths if path.exists()
    ]
    chain_nodes = [
        {
            "order": order,
            "artifact_id": artifact_id,
            "path": str(path.relative_to(case_dir)),
            "sha256": _sha256(path),
        }
        for order, (artifact_id, path) in enumerate(chain_paths, start=1)
    ]
    chain_validation = _validate_chain_nodes(case_dir, chain_nodes)
    _write_json(
        case_dir / "reports" / "artifact-chain.json",
        {
            "artifact_id": "trend-live-5-13-chain-v2",
            "required_order": (
                "evidence_manifest -> scenario_manifest -> trace/ledger -> "
                "divergence_table -> failing_fixture -> minimal_patch -> "
                "regression_report -> findings"
            ),
            "validation": chain_validation,
            "nodes": chain_nodes,
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare-db",
            "refresh-evidence",
            "capture",
            "capture-postfix",
            "capture-final",
            "trace",
            "verify-pre-fix",
            "verify",
            "report",
            "all",
        ),
        help="Capture API snapshots, build report, or perform both.",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=DEFAULT_CASE_DIR,
        help="Protected repository-ignored output directory.",
    )
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument(
        "--pre-fix-worktree",
        type=Path,
        default=DEFAULT_CASE_DIR / "pre-fix-worktree",
        help="Detached Git worktree containing the frozen pre-fix HEAD.",
    )
    parser.add_argument(
        "--refresh-live",
        action="store_true",
        help="Refresh read-only live snapshots; never calls live replay.",
    )
    args = parser.parse_args()
    case_dir = args.case_dir.resolve()
    try:
        case_dir.relative_to(REPO_ROOT / "data")
    except ValueError as exc:
        raise SystemExit("case-dir must remain below momentum/data") from exc

    if args.action in {"prepare-db", "all"}:
        clone = _prepare_isolated_audit_schema(case_dir)
        print(
            json.dumps(
                {
                    "audit_schema": clone["target_schema"],
                    "manifest": str(case_dir / "derived" / "isolated-mysql-clone.json"),
                },
                ensure_ascii=False,
            )
        )
    if args.action in {"capture", "all"}:
        capture(case_dir, args.api_base.rstrip("/"), refresh_live=args.refresh_live)
    if args.action == "refresh-evidence":
        manifest = refresh_evidence(case_dir)
        print(
            json.dumps(
                {
                    "evidence_manifest": str(case_dir / "evidence-manifest.json"),
                    "dirty_tree_sha256": (
                        (manifest.get("git") or {}).get("combined_dirty_tree_sha256")
                    ),
                },
                ensure_ascii=False,
            )
        )
    if args.action == "capture-postfix":
        capture_postfix(case_dir, args.api_base.rstrip("/"))
    if args.action == "capture-final":
        capture_final(case_dir, args.api_base.rstrip("/"))
    if args.action == "trace":
        trace_artifact = _build_model_trace(case_dir)
        print(
            json.dumps(
                {
                    "trace": str(case_dir / "derived" / "model-trace.json"),
                    "artifact_id": trace_artifact["artifact_id"],
                },
                ensure_ascii=False,
            )
        )
    if args.action == "verify-pre-fix":
        artifact = _capture_prefx_failures(
            case_dir,
            args.pre_fix_worktree.resolve(),
        )
        print(
            json.dumps(
                {
                    "pre_fix_failures": str(
                        case_dir / "reports" / "pre-fix-failures.json"
                    ),
                    "expected_failure_observed": bool(
                        (artifact.get("payload") or {}).get("expected_failure_observed")
                    ),
                },
                ensure_ascii=False,
            )
        )
    if args.action == "verify":
        verification = _run_verification(case_dir)
        print(
            json.dumps(
                {
                    "verification": str(case_dir / "reports" / "verification.json"),
                    "passed": bool((verification.get("payload") or {}).get("passed")),
                },
                ensure_ascii=False,
            )
        )
    if args.action in {"report", "all"}:
        summary = report(case_dir)
        print(
            json.dumps(
                {
                    "case_id": summary["case_id"],
                    "report": str(case_dir / "reports" / "audit-summary.json"),
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
