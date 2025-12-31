import datetime as dt

import pandas as pd
import pytest

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


def test_rr_sizing_scales_exposure_by_bucket(session_factory):
    """
    Enable rr_sizing with a very short window (rr_years ~ 0) and custom thresholds so we can
    deterministically land in a bucket and observe exposure scaling in output.
    """
    sf = session_factory
    with sf() as db:
        dates = pd.date_range("2024-01-01", "2024-02-02", freq="B").date
        for i, d in enumerate(dates):
            # AAA steadily up so strategy has positive trailing return quickly.
            aaa = 100.0 + i * 1.0
            bbb = 100.0
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
                cost_bps=0.0,
                rr_sizing=True,
                rr_years=0.05,  # ~13 trading days
                rr_thresholds=[0.01],  # 1% threshold
                rr_weights=[0.8, 0.6],  # below -> 0.8, above -> 0.6
            ),
        )

    # pick a segment after enough history accumulates
    seg = next((p for p in out["holdings"] if p["start_date"] == "2024-01-15"), None)
    assert seg is not None
    assert seg["rr_sizing"]["enabled"] is True
    # trailing_return should be finite and positive
    assert seg["rr_sizing"]["trailing_return"] is not None
    assert seg["rr_sizing"]["trailing_return"] > 0
    # since trailing_return likely >= 1%, bucket should be 1 and exposure 0.6
    assert seg["rr_sizing"]["exposure"] == pytest.approx(0.6)
    # overall exposure field should reflect scaling (<= 1)
    assert 0.0 <= float(seg["exposure"]) <= 1.0

