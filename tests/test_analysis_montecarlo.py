import numpy as np
import pandas as pd

from etf_momentum.analysis.montecarlo import MonteCarloConfig, bootstrap_metrics_from_daily_returns


def test_bootstrap_metrics_from_daily_returns_deterministic_seed():
    # synthetic daily returns with mild drift
    r = pd.Series([0.0] + [0.001] * 200)  # first 0 mimics pct_change fill
    cfg = MonteCarloConfig(n_sims=300, block_size=5, seed=123)
    out1 = bootstrap_metrics_from_daily_returns(r, rf=0.0, cfg=cfg)
    out2 = bootstrap_metrics_from_daily_returns(r, rf=0.0, cfg=cfg)
    assert out1["method"] == "circular_block_bootstrap"
    assert out1["n_sims"] == 300
    assert out1["metrics"]["annualized_return"]["p50"] == out2["metrics"]["annualized_return"]["p50"]


def test_bootstrap_metrics_from_daily_returns_returns_expected_keys():
    rng = np.random.default_rng(0)
    r = pd.Series([0.0] + rng.normal(0.0002, 0.01, size=260).tolist())
    cfg = MonteCarloConfig(n_sims=200, block_size=10, seed=7)
    out = bootstrap_metrics_from_daily_returns(r, rf=0.02, cfg=cfg)
    m = out["metrics"]
    for k in ["cumulative_return", "annualized_return", "annualized_volatility", "max_drawdown", "sharpe_ratio"]:
        assert k in m
        assert "observed" in m[k]
        assert "p05" in m[k] and "p95" in m[k]
        assert "hist" in m[k]
        assert "fit" in m[k]

