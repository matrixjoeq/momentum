from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from etf_momentum.analysis.off_fund_regression import classify_fund_by_regression


def test_classify_handles_mixed_date_index_types() -> None:
    n = 320
    dates_ts = pd.bdate_range("2020-01-01", periods=n)
    dates_date = [d.date() for d in dates_ts]
    x = np.arange(n, dtype=float)
    r1 = 0.0003 + 0.0050 * np.sin(x / 17.0)
    r2 = 0.0002 + 0.0040 * np.cos(x / 19.0)
    rf = 0.65 * r1 + 0.30 * r2
    nav_fund = pd.Series(np.cumprod(1.0 + rf), index=dates_date, dtype=float)
    factor_close = pd.DataFrame(
        {
            "F1": np.cumprod(1.0 + r1),
            "F2": np.cumprod(1.0 + r2),
        },
        index=[dt.datetime.combine(d, dt.time.min) for d in dates_date],
        dtype=float,
    )
    out = classify_fund_by_regression(
        fund_nav=nav_fund,
        factor_close_df=factor_close,
        rolling_window=120,
        min_samples=80,
        dominance_gap=0.08,
        include_series=False,
        max_series_points=0,
    )
    assert out["status"] == "ok"
    assert int(out["sample_days"]) >= 200
