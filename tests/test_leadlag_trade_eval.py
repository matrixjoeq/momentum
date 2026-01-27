import numpy as np
import pandas as pd

from etf_momentum.analysis.leadlag import LeadLagInputs, compute_lead_lag


def test_trade_eval_includes_threshold_and_walkforward():
    # synthetic: idx leads etf with negative relation
    dates = pd.date_range("2022-01-03", periods=400, freq="B").date
    rng = np.random.default_rng(1)
    idx_ret = rng.normal(0, 0.01, size=len(dates))
    etf_ret = -np.roll(idx_ret, 1) + rng.normal(0, 0.005, size=len(dates))
    etf_ret[0] = 0.0

    idx_close = pd.Series(np.exp(np.cumsum(idx_ret)), index=list(dates))
    etf_close = pd.Series(np.exp(np.cumsum(etf_ret)), index=list(dates))

    out = compute_lead_lag(
        LeadLagInputs(
            etf_close=etf_close,
            idx_close=idx_close,
            max_lag=5,
            granger_max_lag=2,
            alpha=0.05,
            trade_cost_bps=10.0,
            enable_threshold=True,
            threshold_quantile=0.8,
            walk_forward=True,
            train_ratio=0.6,
            walk_objective="sharpe",
        )
    )
    assert out["ok"] is True
    trade = out.get("trade") or {}
    assert "threshold" in trade
    assert "walk_forward" in trade
    assert trade["walk_forward"] is None or isinstance(trade["walk_forward"], dict)

