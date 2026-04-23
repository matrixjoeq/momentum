"""
Portfolio weight schedules for futures trend research.

Semantics mirror ``analysis/bt_trend.py`` trend *portfolio* sizing: equal weight
among *active* signal names, or ATR risk-budget targets with over-cap policies.
Per-asset returns are still produced by single-asset ``backtesting.py`` runs;
this module only builds the daily weight matrix applied to those return streams.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .trend import _moving_average


def atr_ewm_wilder(
    high: pd.Series, low: pd.Series, close: pd.Series, *, window: int
) -> pd.Series:
    """Wilder-style ATR (fallback without TA-Lib), aligned with bt_trend._atr_from_hlc_fallback."""
    h = pd.to_numeric(high, errors="coerce").astype(float)
    l = pd.to_numeric(low, errors="coerce").reindex(h.index).combine_first(h)  # noqa: E741
    c = pd.to_numeric(close, errors="coerce").reindex(h.index).ffill()
    prev_close = c.shift(1)
    tr = pd.concat(
        [
            (h - l).abs(),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    w = max(2, int(window))
    return (
        tr.ewm(alpha=1.0 / float(w), adjust=False, min_periods=w).mean().astype(float)
    )


def build_ma_panels(
    exec_by_code: dict[str, pd.DataFrame],
    *,
    common_idx: pd.DatetimeIndex,
    fast_ma: int,
    slow_ma: int,
    ma_type: str = "sma",
    trade_direction: str = "long_only",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (score_df fast-slow, sig_direction_df signed MA regime).

    ``sig_df`` values: **+1** long regime, **-1** short regime, **0** flat (ties).
    ``trade_direction`` maps regimes to allowed sides:

    - ``long_only``: +1 when fast > slow, else 0
    - ``short_only``: -1 when fast < slow, else 0
    - ``both``: sign(fast - slow) as +1 / -1 / 0
    """
    cols = sorted(exec_by_code.keys())
    score_df = pd.DataFrame(index=common_idx, columns=cols, dtype=float)
    sig_df = pd.DataFrame(index=common_idx, columns=cols, dtype=float)
    f, s = int(fast_ma), int(slow_ma)
    mt = str(ma_type or "sma").strip().lower()
    td = str(trade_direction or "long_only").strip().lower()
    for code in cols:
        df = exec_by_code[str(code)].reindex(common_idx)
        close = (
            df["SignalClose"].astype(float)
            if "SignalClose" in df.columns
            else df["Close"].astype(float)
        )
        fast = _moving_average(close, window=f, ma_type=mt)
        slow = _moving_average(close, window=s, ma_type=mt)
        score_df[code] = (fast - slow).astype(float)
        if td == "short_only":
            sig_df[code] = -(fast < slow).astype(float)
        elif td == "both":
            diff = fast.astype(float) - slow.astype(float)
            sig_df[code] = np.sign(diff).astype(float)
        else:
            sig_df[code] = (fast > slow).astype(float)
    return score_df, sig_df


def equal_weights_from_signals(sig_direction_df: pd.DataFrame) -> pd.DataFrame:
    """Equal absolute weight 1/k among symbols with non-zero directional signal."""
    w = pd.DataFrame(
        0.0,
        index=sig_direction_df.index,
        columns=sig_direction_df.columns,
        dtype=float,
    )
    for d in sig_direction_df.index:
        row = sig_direction_df.loc[d].astype(float)
        active_s = row[row.abs() > 0.5]
        if active_s.empty:
            continue
        per = 1.0 / float(len(active_s))
        for c, v in active_s.items():
            if c in w.columns:
                w.loc[d, c] = float(v) * float(per)
    return w


def risk_budget_weights(
    *,
    sig_direction_df: pd.DataFrame,
    score_df: pd.DataFrame,
    exec_by_code: dict[str, pd.DataFrame],
    common_idx: pd.DatetimeIndex,
    risk_budget_atr_window: int,
    risk_budget_pct: float,
    policy: str,
    max_leverage_multiple: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Stateful risk-budget loop (vol-regime adjustments disabled).

    ``price_sig`` for entry/ATR uses execution bars (none) High/Low/Close, matching
    traded contract volatility.
    """
    eps = 1e-12
    pol = str(policy or "scale").strip().lower()
    max_lev = float(max_leverage_multiple)
    if (not np.isfinite(max_lev)) or max_lev <= 1.0:
        max_lev = 2.0

    cols = list(sig_direction_df.columns)
    w_out = pd.DataFrame(0.0, index=common_idx, columns=cols, dtype=float)

    atr_budget_df = pd.DataFrame(index=common_idx, columns=cols, dtype=float)
    px_close_df = pd.DataFrame(index=common_idx, columns=cols, dtype=float)
    w_rb = int(risk_budget_atr_window)
    rb = float(risk_budget_pct)
    for c in cols:
        ex = exec_by_code[str(c)].reindex(common_idx)
        atr_budget_df[c] = atr_ewm_wilder(
            ex["High"].astype(float),
            ex["Low"].astype(float),
            ex["Close"].astype(float),
            window=w_rb,
        )
        px_close_df[c] = ex["Close"].astype(float)

    stats: dict[str, Any] = {
        "policy": pol,
        "risk_budget_pct": rb,
        "risk_budget_atr_window": w_rb,
        "max_leverage_multiple": float(max_lev),
        "overcap_scale_total": 0,
        "overcap_skip_decision_total": 0,
        "overcap_replace_total": 0,
        "overcap_leverage_usage_total": 0,
    }

    prev_rb_w = pd.Series(0.0, index=cols, dtype=float)
    rb_entry_price_by_code: dict[str, float] = {str(c): float("nan") for c in cols}
    rb_entry_seq_by_code: dict[str, int] = {str(c): -1 for c in cols}
    rb_side_by_code: dict[str, int] = {str(c): 0 for c in cols}
    rb_state_by_code: dict[str, str] = {str(c): "FLAT" for c in cols}
    day_seq = 0
    overcap_skip_episode_active = {str(c): False for c in cols}

    overcap_skip_decision_by_code = {str(c): 0 for c in cols}
    overcap_replace_out_by_code = {str(c): 0 for c in cols}
    overcap_replace_in_by_code = {str(c): 0 for c in cols}

    for d in common_idx:
        day_seq += 1
        d_key = str(pd.Timestamp(d).date())
        score_row = score_df.loc[d].astype(float).replace([np.inf, -np.inf], np.nan)
        sig_row = (
            sig_direction_df.loc[d]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=-1.0, upper=1.0)
        )
        active_codes = [
            str(c)
            for c in score_row.abs()
            .where(sig_row.abs() > eps, other=np.nan)
            .dropna()
            .sort_values(ascending=False)
            .index.tolist()
        ]
        active_set = set(active_codes)
        w_row = prev_rb_w.copy().astype(float).reindex(cols).fillna(0.0)
        skipped_today: set[str] = set()

        def _inc_overcap_daily(kind: str, n: int = 1) -> None:
            nn = int(n)
            if nn <= 0:
                return
            row_d = stats.setdefault("overcap_daily_counts", {}).setdefault(
                d_key,
                {
                    "scale": 0,
                    "skip_entry": 0,
                    "replace_entry": 0,
                    "leverage_entry": 0,
                },
            )
            row_d[str(kind)] = int(row_d.get(str(kind), 0) + nn)

        def _apply_overcap_scale_once(cap_multiple: float = 1.0) -> None:
            nonlocal w_row
            cap_v = (
                float(cap_multiple)
                if np.isfinite(float(cap_multiple)) and float(cap_multiple) > 0.0
                else 1.0
            )
            s_now = float(w_row.abs().sum())
            if s_now <= cap_v + eps:
                return
            w_row = (w_row * (cap_v / s_now)).astype(float)
            stats["overcap_scale_total"] = int(stats.get("overcap_scale_total", 0)) + 1
            _inc_overcap_daily("scale", 1)

        def _set_new_risk_budget_entry(key: str, signed_target: float) -> None:
            nonlocal w_row
            w_row.loc[key] = float(signed_target)
            px_now = (
                float(px_close_df.loc[d, key])
                if key in px_close_df.columns
                else float("nan")
            )
            rb_entry_price_by_code[key] = (
                float(px_now) if np.isfinite(px_now) and px_now > 0.0 else float("nan")
            )
            rb_entry_seq_by_code[key] = int(day_seq)
            rb_side_by_code[key] = int(np.sign(float(signed_target)))
            rb_state_by_code[key] = "NORMAL"

        def _select_replace_out_code(new_code: str) -> str | None:
            cand: list[tuple[float, int, str]] = []
            for cc in cols:
                key_cc = str(cc)
                if key_cc == str(new_code):
                    continue
                if abs(float(w_row.loc[key_cc])) <= eps:
                    continue
                cur_px = float(px_close_df.loc[d, key_cc])
                ent_px = float(rb_entry_price_by_code.get(key_cc, float("nan")))
                side = int(rb_side_by_code.get(key_cc, 0))
                if side == 0:
                    side = 1 if float(w_row.loc[key_cc]) > 0 else -1
                raw_ret = (
                    ((cur_px / ent_px) - 1.0)
                    if (
                        np.isfinite(cur_px)
                        and cur_px > 0.0
                        and np.isfinite(ent_px)
                        and ent_px > 0.0
                    )
                    else float("inf")
                )
                ret = float(side) * float(raw_ret)
                seq = int(rb_entry_seq_by_code.get(key_cc, 10**9))
                if seq < 0:
                    seq = 10**9
                cand.append((float(ret), int(seq), key_cc))
            if not cand:
                return None
            cand.sort(key=lambda x: (x[0], x[1], x[2]))
            return str(cand[0][2])

        for c in cols:
            key_c = str(c)
            if (float(w_row.loc[c]) > eps) and (key_c not in active_set):
                w_row.loc[c] = 0.0
                rb_state_by_code[key_c] = "FLAT"
                rb_entry_price_by_code[key_c] = float("nan")
                rb_entry_seq_by_code[key_c] = -1
                rb_side_by_code[key_c] = 0

        for c in active_codes:
            px = (
                float(px_close_df.loc[d, c])
                if c in px_close_df.columns
                else float("nan")
            )
            a = (
                float(atr_budget_df.loc[d, c])
                if (c in atr_budget_df.columns and d in atr_budget_df.index)
                else float("nan")
            )
            base_target = float("nan")
            if np.isfinite(px) and px > 0.0 and np.isfinite(a) and a > 0.0:
                base_target = float(rb) * float(px) / float(a)
            sig_v = float(sig_row.get(c, 0.0))
            side_target = 1.0 if sig_v > eps else (-1.0 if sig_v < -eps else 0.0)
            signed_target = (
                float(base_target) * float(side_target)
                if np.isfinite(base_target)
                and base_target > 0.0
                and abs(side_target) > 0
                else float("nan")
            )
            has_pos = bool(abs(float(w_row.loc[c])) > eps)
            key = str(c)
            if not has_pos:
                if np.isfinite(signed_target) and abs(float(signed_target)) > 0.0:
                    proposed_total = float(
                        w_row.abs().sum() + abs(float(signed_target))
                    )
                    overcap_on_new_entry = bool(proposed_total > 1.0 + eps)
                    if overcap_on_new_entry and pol == "skip_entry":
                        stats["overcap_skip_decision_total"] = (
                            int(stats.get("overcap_skip_decision_total", 0)) + 1
                        )
                        _inc_overcap_daily("skip_entry", 1)
                        overcap_skip_decision_by_code[key] = int(
                            overcap_skip_decision_by_code.get(key, 0) + 1
                        )
                        skipped_today.add(key)
                        if not bool(overcap_skip_episode_active.get(key, False)):
                            overcap_skip_episode_active[key] = True
                        continue
                    if overcap_on_new_entry and pol == "replace_entry":
                        out_code = _select_replace_out_code(key)
                        if out_code:
                            w_row.loc[out_code] = 0.0
                            rb_state_by_code[out_code] = "FLAT"
                            rb_entry_price_by_code[out_code] = float("nan")
                            rb_entry_seq_by_code[out_code] = -1
                            stats["overcap_replace_total"] = (
                                int(stats.get("overcap_replace_total", 0)) + 1
                            )
                            _inc_overcap_daily("replace_entry", 1)
                            overcap_replace_out_by_code[out_code] = int(
                                overcap_replace_out_by_code.get(out_code, 0) + 1
                            )
                            overcap_replace_in_by_code[key] = int(
                                overcap_replace_in_by_code.get(key, 0) + 1
                            )
                    _set_new_risk_budget_entry(key, float(signed_target))
                    if overcap_on_new_entry and pol == "replace_entry":
                        _apply_overcap_scale_once()
                    elif overcap_on_new_entry and pol == "leverage_entry":
                        lev_now = float(w_row.abs().sum())
                        if lev_now > 1.0 + eps:
                            stats["overcap_leverage_usage_total"] = (
                                int(stats.get("overcap_leverage_usage_total", 0)) + 1
                            )
                            _inc_overcap_daily("leverage_entry", 1)
                            if lev_now > float(max_lev) + eps:
                                _apply_overcap_scale_once(float(max_lev))
                continue

        if pol != "leverage_entry":
            _apply_overcap_scale_once()
        for key_cc in cols:
            kk = str(key_cc)
            if bool(overcap_skip_episode_active.get(kk, False)) and (
                kk not in skipped_today
            ):
                overcap_skip_episode_active[kk] = False
        w_out.loc[d] = w_row.to_numpy(dtype=float)
        prev_rb_w = w_row.copy()

    stats["overcap_skip_decision_by_code"] = overcap_skip_decision_by_code
    stats["overcap_replace_out_by_code"] = overcap_replace_out_by_code
    stats["overcap_replace_in_by_code"] = overcap_replace_in_by_code
    return w_out, stats


def combine_weighted_returns(
    ret_mat: pd.DataFrame,
    weight_df: pd.DataFrame,
) -> pd.Series:
    """Daily portfolio return using prior-day weights: r_t = sum_i w_{i,t-1} * r_{i,t}."""
    w_prev = weight_df.shift(1).fillna(0.0).astype(float)
    out = (w_prev * ret_mat.astype(float)).sum(axis=1).astype(float)
    return out.fillna(0.0)
