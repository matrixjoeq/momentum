import datetime as dt

import pandas as pd

from etf_momentum.analysis.grouping import AssetGroupSuggestInputs, suggest_asset_groups
from tests.helpers.price_seed import add_price_all_adjustments


def _add_price(db, *, code: str, day: dt.date, close: float) -> None:
    add_price_all_adjustments(
        db,
        code=code,
        day=day,
        close=float(close),
        open_price=float(close),
        high=float(close),
        low=float(close),
    )


def test_suggest_asset_groups_returns_mapping(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-03-31", freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            base = 100.0 + i * 0.5
            _add_price(db, code="A1", day=d, close=base)
            _add_price(db, code="A2", day=d, close=base * 1.001)
            _add_price(db, code="B1", day=d, close=90.0 + i * 0.1 + (1 if (i % 2 == 0) else -1))
        db.commit()

        out = suggest_asset_groups(
            db,
            AssetGroupSuggestInputs(
                codes=["A1", "A2", "B1"],
                start=dates[0],
                end=dates[-1],
                adjust="hfq",
                lookback_days=60,
                corr_threshold=0.7,
            ),
        )
    assert "asset_groups" in out
    assert out["asset_groups"]["A1"] == out["asset_groups"]["A2"]
    assert "groups" in out and len(out["groups"]) >= 1
