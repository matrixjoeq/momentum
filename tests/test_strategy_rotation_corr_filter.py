import datetime as dt

import pandas as pd

from etf_momentum.db.models import EtfPrice
from etf_momentum.strategy.rotation import RotationInputs, backtest_rotation


def _add_price(db, *, code: str, day: dt.date, close: float) -> None:
    for adj in ("none", "hfq", "qfq"):
        db.add(
            EtfPrice(
                code=code,
                trade_date=day,
                close=float(close),
                low=float(close),
                high=float(close),
                source="eastmoney",
                adjust=adj,
            )
        )


def test_corr_filter_blocks_rebalance_when_new_pick_highly_correlated_with_current(session_factory):
    """
    If corr_filter is enabled and corr(new_pick, current_holding) > threshold, keep holding (no rebalance).
    Correlation uses hfq daily returns over the lookback window.
    """
    sf = session_factory
    with sf() as db:
        # Two weeks of business days including two Fridays (decision days): 2024-01-05 and 2024-01-12
        dates = pd.date_range("2024-01-01", "2024-01-19", freq="B").date
        for i, d in enumerate(dates):
            # Create a shared, high-variance path so return-correlation is meaningful.
            base = 100.0 + (2.0 if (i % 2 == 0) else -2.0) + i * 0.3
            aaa = base
            # BBB tracks the same base very closely (high corr), but:
            # - slightly worse in the first decision window so AAA is held first
            # - slightly better on the 2nd decision date so BBB would be picked (absent the gate)
            bump = 0.0
            if d == dt.date(2024, 1, 5):
                bump = -2.0
            if d == dt.date(2024, 1, 12):
                bump = +0.5
            bbb = base + bump
            _add_price(db, code="AAA", day=d, close=aaa)
            _add_price(db, code="BBB", day=d, close=bbb)
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["AAA", "BBB"],
                start=dates[0],
                end=dates[-1],
                rebalance="weekly",
                top_k=1,
                lookback_days=2,
                tp_sl_mode="none",
                corr_filter=True,
                corr_window=10,
                corr_threshold=0.5,
            ),
        )

    # Segment starting 2024-01-08: first rebalance into AAA (any is fine).
    seg1 = next((p for p in out["holdings"] if p["start_date"] == "2024-01-08"), None)
    assert seg1 is not None
    assert seg1["picks"] == ["AAA"]

    # Segment starting 2024-01-15: would like to switch to BBB due to stronger momentum,
    # but corr gate should block, so it should keep the previous holding.
    seg2 = next((p for p in out["holdings"] if p["start_date"] == "2024-01-15"), None)
    assert seg2 is not None
    assert seg2["corr_filter"]["enabled"] is True
    assert seg2["corr_filter"]["blocked"] is True
    assert seg2["picks"] == seg1["picks"]

