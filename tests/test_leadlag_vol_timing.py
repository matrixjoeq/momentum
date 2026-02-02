import numpy as np
import pandas as pd

from etf_momentum.analysis.leadlag import LeadLagInputs, compute_lead_lag


def test_vol_timing_tiered_exposure_by_level_quantile():
    # 120 business days; index level is low then high (0 -> 1)
    dates = pd.date_range("2024-01-02", periods=120, freq="B").date
    # Use strictly positive levels to avoid log(0) in leadlag codepath.
    idx_level = np.concatenate([np.linspace(100.0, 101.0, 60), np.linspace(200.0, 201.0, 60)]).astype(float)
    idx_close = pd.Series(idx_level, index=list(dates))

    # ETF close: mild stochastic drift to keep Granger regression well-behaved.
    rng = np.random.default_rng(0)
    ret = rng.normal(0.0005, 0.01, size=len(dates))
    etf_close = pd.Series(np.exp(np.cumsum(ret)), index=list(dates))

    out = compute_lead_lag(
        LeadLagInputs(
            etf_close=etf_close,
            idx_close=idx_close,
            max_lag=0,
            granger_max_lag=1,
            alpha=0.05,
            trade_cost_bps=0.0,
            enable_threshold=False,
            walk_forward=False,
            vol_timing=True,
            vol_level_quantiles=[0.5],
            vol_level_exposures=[1.0, 0.0],
        )
    )
    assert out["ok"] is True
    vt = (out.get("trade") or {}).get("vol_timing") or {}
    assert vt.get("ok") is True

    # With ~59 low points and ~60 high points after diff/dropna, q50 lands at the first high level.
    thr = (vt.get("thresholds_abs_train") or [None])[0]
    assert 150.0 <= float(thr) <= 201.0

    exp = vt.get("exp") or []
    nav = vt.get("nav_strategy") or []
    assert len(exp) == len(nav)
    assert len(nav) > 50

    # Once exposure becomes 0, NAV should stay flat (no cost, no exposure).
    first_zero = next((i for i, x in enumerate(exp) if abs(float(x) - 0.0) < 1e-12), None)
    assert first_zero is not None
    tail = nav[first_zero:]
    assert max(abs(float(x) - float(tail[0])) for x in tail) < 1e-12


def test_vol_timing_with_rolling_quantile_window_runs():
    # keep it small but >= 120 to have enough history
    dates = pd.date_range("2020-01-02", periods=260, freq="B").date
    idx_level = np.linspace(50.0, 150.0, len(dates)).astype(float)
    idx_close = pd.Series(idx_level, index=list(dates))
    rng = np.random.default_rng(1)
    ret = rng.normal(0.0002, 0.01, size=len(dates))
    etf_close = pd.Series(np.exp(np.cumsum(ret)), index=list(dates))

    out = compute_lead_lag(
        LeadLagInputs(
            etf_close=etf_close,
            idx_close=idx_close,
            max_lag=0,
            granger_max_lag=1,
            alpha=0.05,
            trade_cost_bps=0.0,
            enable_threshold=False,
            walk_forward=False,  # rolling mode disables WF anyway; keep deterministic
            vol_timing=True,
            vol_level_quantiles=[0.8, 0.9],
            vol_level_exposures=[1.0, 0.5, 0.2],
            vol_level_window="1y",
        )
    )
    assert out["ok"] is True
    vt = (out.get("trade") or {}).get("vol_timing") or {}
    assert vt.get("ok") is True
    assert vt.get("vol_level_window") == "1y"

