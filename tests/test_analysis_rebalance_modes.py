import datetime as dt

import numpy as np

from etf_momentum.analysis.baseline import BaselineInputs, compute_baseline
from etf_momentum.db.models import EtfPrice


def test_rebalance_none_equals_mean_of_individual_nav(session_factory):
    sf = session_factory
    with sf() as db:
        dates = [dt.date(2024, 1, d) for d in range(1, 6)]
        # asset A doubles; asset B flat
        a = [1.0, 1.25, 1.5, 1.75, 2.0]
        b = [1.0, 1.0, 1.0, 1.0, 1.0]
        for d, ca, cb in zip(dates, a, b, strict=True):
            db.add(EtfPrice(code="A", trade_date=d, close=ca, source="eastmoney", adjust="qfq"))
            db.add(EtfPrice(code="B", trade_date=d, close=cb, source="eastmoney", adjust="qfq"))
        db.commit()

        out = compute_baseline(
            db,
            BaselineInputs(
                codes=["A", "B"],
                start=dates[0],
                end=dates[-1],
                adjust="qfq",
                rebalance="none",
                rolling_weeks=[],
                rolling_months=[],
                rolling_years=[],
            ),
        )

    ew = np.array(out["nav"]["series"]["EW"], dtype=float)
    # expected: mean of [A_nav, B_nav] (both start at 1)
    expected = (np.array(a) / a[0] + np.array(b) / b[0]) / 2.0
    assert np.allclose(ew, expected, rtol=1e-12, atol=1e-12)

