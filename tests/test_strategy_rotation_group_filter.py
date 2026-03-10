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


def test_group_enforce_strongest_score_keeps_one_per_group(session_factory):
    sf = session_factory
    with sf() as db:
        dates = [d.date() for d in pd.date_range("2024-01-01", "2024-01-19", freq="B")]
        for i, d in enumerate(dates):
            # A1/A2 belong to the same group and both trend strongly.
            _add_price(db, code="A1", day=d, close=100.0 + i * 2.0)
            _add_price(db, code="A2", day=d, close=100.0 + i * 1.8)
            # B1 is weaker but in another group.
            _add_price(db, code="B1", day=d, close=100.0 + i * 0.6)
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["A1", "A2", "B1"],
                start=dates[0],
                end=dates[-1],
                rebalance="weekly",
                top_k=2,
                lookback_days=3,
                group_enforce=True,
                group_pick_policy="strongest_score",
                asset_groups={"A1": "CN_EQ", "A2": "CN_EQ", "B1": "US_EQ"},
            ),
        )

    seg = next((p for p in out["holdings"] if p["start_date"] == "2024-01-08"), None)
    assert seg is not None
    assert seg["group_filter"]["enabled"] is True
    assert "A1" in seg["picks"]
    assert "A2" not in seg["picks"]
    assert "B1" in seg["picks"]


def test_group_enforce_earliest_entry_prefers_existing_holding(session_factory):
    sf = session_factory
    with sf() as db:
        dates = [d.date() for d in pd.date_range("2024-01-01", "2024-01-19", freq="B")]
        for i, d in enumerate(dates):
            if d <= dt.date(2024, 1, 5):
                a1 = 100.0 + i * 2.5
                a2 = 100.0 + i * 0.8
            else:
                a1 = 112.0 + (i - 4) * 0.2
                a2 = 103.0 + (i - 4) * 2.2
            _add_price(db, code="A1", day=d, close=a1)
            _add_price(db, code="A2", day=d, close=a2)
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=["A1", "A2"],
                start=dates[0],
                end=dates[-1],
                rebalance="weekly",
                top_k=1,
                lookback_days=3,
                group_enforce=True,
                group_pick_policy="earliest_entry",
                asset_groups={"A1": "CN_EQ", "A2": "CN_EQ"},
            ),
        )

    seg1 = next((p for p in out["holdings"] if p["start_date"] == "2024-01-08"), None)
    seg2 = next((p for p in out["holdings"] if p["start_date"] == "2024-01-15"), None)
    assert seg1 is not None and seg2 is not None
    assert seg1["picks"] == ["A1"]
    assert seg2["group_filter"]["enabled"] is True
    assert seg2["group_filter"]["policy"] == "earliest_entry"
    assert seg2["picks"] == ["A1"]
