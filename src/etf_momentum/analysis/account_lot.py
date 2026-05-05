from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def resolve_max_leverage(
    *,
    position_sizing: str,
    risk_budget_overcap_policy: str | None,
    risk_budget_max_leverage_multiple: float | None,
) -> float:
    ps = str(position_sizing or "").strip().lower()
    pol = str(risk_budget_overcap_policy or "").strip().lower()
    raw = (
        float(risk_budget_max_leverage_multiple)
        if risk_budget_max_leverage_multiple is not None
        else 1.0
    )
    if (
        ps == "risk_budget"
        and pol == "leverage_entry"
        and np.isfinite(raw)
        and raw >= 1.0
    ):
        return float(raw)
    return 1.0


def simulate_lot_account_weights(
    *,
    target_weights: pd.DataFrame,
    exec_price: pd.DataFrame,
    initial_account_amount: float,
    max_leverage_multiple: float,
    lot_size_shares: int = 100,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    w_tgt = target_weights.copy().astype(float).fillna(0.0).clip(lower=0.0)
    px = (
        exec_price.reindex(index=w_tgt.index, columns=w_tgt.columns)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )
    if w_tgt.empty or px.empty:
        return pd.DataFrame(index=w_tgt.index, columns=w_tgt.columns), {
            "enabled": False,
            "reason": "missing_weights_or_prices",
        }
    init_amt = float(initial_account_amount)
    if (not np.isfinite(init_amt)) or init_amt <= 0.0:
        return pd.DataFrame(index=w_tgt.index, columns=w_tgt.columns), {
            "enabled": False,
            "reason": "invalid_initial_account_amount",
        }
    max_lev = (
        float(max_leverage_multiple)
        if np.isfinite(float(max_leverage_multiple))
        and float(max_leverage_multiple) >= 1.0
        else 1.0
    )
    lot = float(max(1, int(lot_size_shares)))
    codes = list(w_tgt.columns)
    shares = pd.Series(0.0, index=codes, dtype=float)
    cash = float(init_amt)
    w_real = pd.DataFrame(index=w_tgt.index, columns=codes, dtype=float)
    shares_hist: dict[str, list[int]] = {str(c): [] for c in codes}
    cash_series: list[float] = []
    gross_lev_series: list[float] = []
    eps = 1e-12
    for d in w_tgt.index:
        p = px.loc[d].astype(float).replace([np.inf, -np.inf], np.nan)
        valid = p > 0.0
        pos_val = float((shares[valid] * p[valid]).sum()) if bool(valid.any()) else 0.0
        equity = float(cash + pos_val)
        if (not np.isfinite(equity)) or equity <= 0.0:
            equity = eps
        desired_notional = (w_tgt.loc[d].astype(float) * equity).astype(float)
        gross_desired = float(desired_notional.sum())
        max_notional = float(max_lev * equity)
        if gross_desired > max_notional and gross_desired > eps:
            desired_notional = (desired_notional * (max_notional / gross_desired)).astype(
                float
            )
        target_shares = shares.copy()
        for c in codes:
            px_c = float(p.get(c, np.nan))
            if (not np.isfinite(px_c)) or px_c <= 0.0:
                continue
            want_shares = float(desired_notional.get(c, 0.0) / px_c)
            want_shares = math.floor(max(0.0, want_shares) / lot) * lot
            target_shares.loc[c] = float(want_shares)
        delta = (target_shares - shares).astype(float)
        buy_notional = (
            float((delta.clip(lower=0.0)[valid] * p[valid]).sum()) if bool(valid.any()) else 0.0
        )
        sell_notional = (
            float(((-delta.clip(upper=0.0))[valid] * p[valid]).sum())
            if bool(valid.any())
            else 0.0
        )
        cash_after = float(cash - buy_notional + sell_notional)
        min_cash_allowed = float(equity - max_notional)
        if cash_after < min_cash_allowed:
            buy_codes = [
                c
                for c in codes
                if float(delta.get(c, 0.0)) > 0.0 and float(p.get(c, np.nan)) > 0.0
            ]
            buy_codes.sort(
                key=lambda c: float(delta.get(c, 0.0)) * float(p.get(c, 0.0)),
                reverse=True,
            )
            while cash_after < min_cash_allowed and buy_codes:
                progressed = False
                for c in list(buy_codes):
                    px_c = float(p.get(c, np.nan))
                    if (not np.isfinite(px_c)) or px_c <= 0.0:
                        buy_codes.remove(c)
                        continue
                    if float(delta.get(c, 0.0)) < lot:
                        buy_codes.remove(c)
                        continue
                    delta.loc[c] = float(delta.get(c, 0.0) - lot)
                    target_shares.loc[c] = float(target_shares.get(c, 0.0) - lot)
                    cash_after += float(lot * px_c)
                    progressed = True
                    if cash_after >= min_cash_allowed:
                        break
                if not progressed:
                    break
        shares = target_shares.astype(float)
        cash = float(cash_after)
        pos_val = float((shares[valid] * p[valid]).sum()) if bool(valid.any()) else 0.0
        equity = float(cash + pos_val)
        if (not np.isfinite(equity)) or equity <= 0.0:
            equity = eps
        w_row = pd.Series(0.0, index=codes, dtype=float)
        if bool(valid.any()):
            w_row.loc[valid] = (shares[valid] * p[valid] / equity).astype(float)
        w_real.loc[d] = w_row
        for c in codes:
            shares_hist[str(c)].append(int(shares.get(c, 0.0)))
        cash_series.append(float(cash))
        gross_lev_series.append(float(pos_val / equity) if equity > eps else 0.0)
    return w_real.astype(float).fillna(0.0), {
        "enabled": True,
        "initial_account_amount": float(init_amt),
        "lot_size_shares": int(lot),
        "max_leverage_multiple": float(max_lev),
        "cash_series": cash_series,
        "gross_leverage_series": gross_lev_series,
        "shares_by_code": shares_hist,
    }


def remap_return_weights(
    *,
    w_eff_before: pd.DataFrame,
    w_eff_after: pd.DataFrame,
    w_ret_before: pd.DataFrame,
    eps: float = 1e-12,
) -> tuple[pd.DataFrame, pd.Series]:
    w0 = (
        w_eff_before.reindex(index=w_ret_before.index, columns=w_ret_before.columns)
        .astype(float)
        .fillna(0.0)
    )
    w1 = (
        w_eff_after.reindex(index=w_ret_before.index, columns=w_ret_before.columns)
        .astype(float)
        .fillna(0.0)
    )
    wr0 = w_ret_before.astype(float).fillna(0.0)
    ratio = pd.DataFrame(0.0, index=wr0.index, columns=wr0.columns, dtype=float)
    nz = w0.abs() > float(eps)
    ratio = ratio.where(
        ~nz,
        (w1 / w0).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float),
    ).clip(lower=0.0)
    wr1 = (wr0 * ratio).astype(float)
    gross_eff_before = w0.sum(axis=1).astype(float)
    gross_eff_after = w1.sum(axis=1).astype(float)
    gross_ret_before = wr0.sum(axis=1).astype(float)
    gross_ret_after = wr1.sum(axis=1).astype(float)
    gross_map = pd.Series(0.0, index=wr0.index, dtype=float)
    nz_eff = gross_eff_before.abs() > float(eps)
    gross_map.loc[nz_eff] = (
        (gross_eff_after.loc[nz_eff] / gross_eff_before.loc[nz_eff])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )
    gross_map = gross_map.clip(lower=0.0)
    gross_target = (gross_ret_before * gross_map).astype(float)
    renorm = pd.Series(0.0, index=wr0.index, dtype=float)
    nz_after = gross_ret_after.abs() > float(eps)
    renorm.loc[nz_after] = (
        (gross_target.loc[nz_after] / gross_ret_after.loc[nz_after])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )
    renorm = renorm.clip(lower=0.0)
    wr1 = wr1.mul(renorm, axis=0).astype(float).fillna(0.0)
    risk_scale = pd.Series(0.0, index=wr0.index, dtype=float)
    nz_ret = gross_ret_before.abs() > float(eps)
    gross_ret_new = wr1.sum(axis=1).astype(float)
    risk_scale.loc[nz_ret] = (
        (gross_ret_new.loc[nz_ret] / gross_ret_before.loc[nz_ret])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )
    risk_scale = risk_scale.clip(lower=0.0)
    return wr1, risk_scale
