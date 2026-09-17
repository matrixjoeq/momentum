from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etf_momentum.analysis.var_cvar import (
    WEIGHT_EPS,
    build_holding_cvar_overlay,
    collect_hs_portfolio_returns,
    cvar_shrink_scale,
    estimate_cvar_at,
    historical_var_cvar_loss,
)


def test_historical_var_cvar_known_tail_and_quantile_interpolation():
    s = pd.Series([-0.10, -0.08, -0.04, -0.02, 0.01, 0.02, 0.03, 0.00] * 5)
    var_loss, cvar_loss = historical_var_cvar_loss(s, confidence=0.95)
    q = float(s.quantile(0.05))
    assert var_loss == pytest.approx(max(0.0, -q), rel=0.0, abs=1e-12)
    tail = s[s <= q]
    expected_cvar = max(0.0, -float(tail.mean()))
    assert cvar_loss == pytest.approx(expected_cvar, rel=0.0, abs=1e-12)
    assert var_loss is not None and var_loss >= 0.0
    assert cvar_loss is not None and cvar_loss >= 0.0

    pos = pd.Series([0.01, 0.02, 0.03, 0.04])
    v2, c2 = historical_var_cvar_loss(pos, confidence=0.95)
    assert v2 == 0.0
    assert c2 == 0.0


def test_cvar_scale_insufficient_and_non_positive():
    assert cvar_shrink_scale(0.05, 0.02, sample_count=10, window=60) == (
        1.0,
        "insufficient_samples",
    )
    assert cvar_shrink_scale(0.05, 0.02, sample_count=60, window=19) == (
        1.0,
        "insufficient_samples",
    )
    assert cvar_shrink_scale(0.0, 0.02, sample_count=60, window=60) == (
        1.0,
        "non_positive_cvar",
    )
    assert cvar_shrink_scale(None, 0.02, sample_count=60, window=60) == (
        1.0,
        "invalid_cvar",
    )
    scale, status = cvar_shrink_scale(0.08, 0.02, sample_count=60, window=60)
    assert status == "ok"
    assert scale == pytest.approx(0.25, rel=0.0, abs=1e-12)
    assert scale <= 1.0


def test_hs_skips_missing_returns_and_does_not_treat_fill_zero_as_valid():
    idx = pd.bdate_range("2024-01-02", periods=8)
    w = pd.Series({"A": 0.6, "B": 0.4})
    r = pd.DataFrame(
        {
            "A": [np.nan, 0.01, np.nan, 0.02, 0.01, 0.00, 0.01, 0.01],
            "B": [np.nan, 0.01, 0.01, np.nan, 0.01, 0.00, 0.01, 0.01],
        },
        index=idx,
    )
    sample = collect_hs_portfolio_returns(
        w, r, end_loc=len(idx) - 1, window=6, include_end=True
    )
    # Days with any held-name NaN are skipped; 0.00 is a real return and counts.
    assert len(sample) == 5
    filled = r.fillna(0.0)
    sample_filled = collect_hs_portfolio_returns(
        w, filled, end_loc=len(idx) - 1, window=6, include_end=True
    )
    assert len(sample_filled) > len(sample)


def test_sim_scale_excludes_same_day_return_even_if_huge_loss():
    idx = pd.bdate_range("2024-01-02", periods=80)
    w = pd.DataFrame({"A": 1.0}, index=idx)
    r = pd.DataFrame({"A": -0.04}, index=idx)
    r.iloc[0, 0] = np.nan
    r.iloc[-1, 0] = -0.80
    orig = r["A"].fillna(0.0)
    pack = build_holding_cvar_overlay(
        w, r, orig, (1.0 + orig).cumprod(), window=20, budget_pct=0.02
    )
    last = len(idx) - 1
    sim_est = estimate_cvar_at(
        w.iloc[last], r, end_loc=last, window=20, include_end=False, budget_pct=0.02
    )
    prompt_est = estimate_cvar_at(
        w.iloc[last], r, end_loc=last, window=20, include_end=True, budget_pct=0.02
    )
    assert pack["_scales"].iloc[last] == pytest.approx(
        float(sim_est["scale"]), rel=0.0, abs=1e-12
    )
    assert float(sim_est["scale"]) != pytest.approx(
        float(prompt_est["scale"]), rel=0.0, abs=1e-9
    )
    assert float(sim_est["cvar_95"] or 0.0) < float(prompt_est["cvar_95"] or 0.0)


def test_overlay_prefix_matches_when_scale_is_one_and_nav_length():
    idx = pd.bdate_range("2024-01-02", periods=10)
    w = pd.DataFrame({"A": 1.0}, index=idx)
    r = pd.DataFrame({"A": 0.01}, index=idx)
    r.iloc[0, 0] = np.nan
    orig = r["A"].fillna(0.0)
    orig_nav = (1.0 + orig).cumprod()
    orig_nav.iloc[0] = 1.0
    for i in range(1, len(orig_nav)):
        orig_nav.iloc[i] = float(orig_nav.iloc[i - 1]) * (1.0 + float(orig.iloc[i]))
    pack = build_holding_cvar_overlay(w, r, orig, orig_nav, window=60, budget_pct=0.02)
    assert list(pack["_scales"]) == [1.0] * len(idx)
    assert pack["_nav_sim"].tolist() == pytest.approx(orig_nav.astype(float).tolist())
    assert len(pack["sim"]["nav"]) == len(idx)
    assert pack["sim"]["nav"][0] == pytest.approx(1.0)
    assert pack["prompt"]["status"] == "insufficient_samples"
    assert pack["prompt"]["asof"] == idx[-1].date().isoformat()
    assert pack["prompt"]["applies_to"] == "下一交易日"
    assert abs(float(pack["prompt"]["scale"]) - 1.0) <= WEIGHT_EPS


def test_overlay_turnover_counts_implied_cash():
    idx = pd.bdate_range("2024-01-02", periods=40)
    w = pd.DataFrame({"A": 1.0}, index=idx)
    r = pd.DataFrame({"A": -0.04}, index=idx)
    r.iloc[0, 0] = np.nan
    orig = r["A"].fillna(0.0)
    orig_nav = orig.copy()
    orig_nav.iloc[0] = 1.0
    for i in range(1, len(orig_nav)):
        orig_nav.iloc[i] = float(orig_nav.iloc[i - 1]) * (1.0 + float(orig.iloc[i]))
    pack = build_holding_cvar_overlay(w, r, orig, orig_nav, window=20, budget_pct=0.02)
    scales = pack["_scales"].astype(float)
    assert float(scales.min()) < 1.0
    w_scaled = w.mul(scales, axis=0).clip(lower=0.0)
    cash = (1.0 - w_scaled.sum(axis=1)).clip(lower=0.0)
    w_full = w_scaled.copy()
    w_full["__cash__"] = cash
    expected = float(
        ((w_full - w_full.shift(1).fillna(0.0)).abs().sum(axis=1) / 2.0).mean()
    )
    asset_only = float(
        ((w_scaled - w_scaled.shift(1).fillna(0.0)).abs().sum(axis=1) / 2.0).mean()
    )
    assert pack["sim"]["avg_daily_turnover"] == pytest.approx(
        expected, rel=0.0, abs=1e-12
    )
    assert pack["sim"]["avg_daily_turnover"] > asset_only
