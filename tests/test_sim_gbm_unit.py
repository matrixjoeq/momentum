import numpy as np

from etf_momentum.analysis.sim_gbm import SimConfig, simulate_gbm_prices


def test_sim_gbm_prices_positive_and_uncorrelated_ish():
    out = simulate_gbm_prices(start="19900101", end="19901231", cfg=SimConfig(n_assets=4, seed=1))
    assert out["ok"] is True
    codes = out["assets"]["codes"]
    assert len(codes) == 4
    close = out["series"]["close"]
    for c in codes:
        arr = np.asarray(close[c], dtype=float)
        assert arr.shape[0] == len(out["series"]["dates"])
        assert float(np.min(arr)) > 0.0
        assert abs(float(arr[0]) - 1.0) < 1e-12

