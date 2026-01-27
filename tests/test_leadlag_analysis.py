import numpy as np
import pandas as pd

from etf_momentum.analysis.leadlag import (
    LeadLagInputs,
    align_us_close_to_cn_next_trading_day,
    compute_lead_lag,
)


def test_leadlag_detects_leading_relation_by_best_lag():
    # Construct a toy example where idx_ret leads etf_ret by 1 day.
    dates = pd.date_range("2024-01-02", periods=40, freq="D").date
    rng = np.random.default_rng(0)
    idx_ret = rng.normal(0, 0.01, size=len(dates))
    etf_ret = np.roll(idx_ret, 1)  # etf reacts one day later
    etf_ret[0] = 0.0

    idx_close = pd.Series(np.exp(np.cumsum(idx_ret)), index=list(dates))
    etf_close = pd.Series(np.exp(np.cumsum(etf_ret)), index=list(dates))

    out = compute_lead_lag(LeadLagInputs(etf_close=etf_close, idx_close=idx_close, max_lag=5, granger_max_lag=3, alpha=0.2))
    assert out["ok"] is True
    best = out["corr"]["best"]
    assert int(best["lag"]) == 1
    assert out["corr"]["relation"] == "leading"


def test_leadlag_cn_next_trading_day_alignment_moves_friday_to_monday():
    # Cboe DATE is US session date; align to CN next trading day.
    # 2024-01-05 is Friday; next CN trading day is 2024-01-08 (Monday).
    idx_close = pd.Series([10.0], index=[pd.Timestamp("2024-01-05").date()])
    aligned = align_us_close_to_cn_next_trading_day(idx_close)
    assert aligned.index.tolist() == [pd.Timestamp("2024-01-08").date()]
    assert float(aligned.iloc[0]) == 10.0

