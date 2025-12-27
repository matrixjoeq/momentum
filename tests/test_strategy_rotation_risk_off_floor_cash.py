import datetime as dt

import pytest

from etf_momentum.db.models import EtfPrice
from etf_momentum.strategy.rotation import RotationInputs, backtest_rotation


def test_risk_off_floor_triggers_cash_when_no_defensive(session_factory):
    sf = session_factory
    with sf() as db:
        codes = ["AAA", "BBB"]
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(60)]
        for i, d in enumerate(dates):
            # Both trend down so momentum is negative.
            px_a = 100 - i * 0.2
            px_b = 100 - i * 0.1
            for adj in ("hfq", "none"):
                db.add(EtfPrice(code="AAA", trade_date=d, close=px_a, source="eastmoney", adjust=adj))
                db.add(EtfPrice(code="BBB", trade_date=d, close=px_b, source="eastmoney", adjust=adj))
        db.commit()

        out = backtest_rotation(
            db,
            RotationInputs(
                codes=codes,
                start=start,
                end=dates[-1],
                rebalance="monthly",
                top_k=1,
                lookback_days=10,
                skip_days=0,
                risk_off=True,
                defensive_code=None,  # key: no defensive asset
                momentum_floor=0.5,  # high floor: should always trigger risk-off
                cost_bps=0.0,
            ),
        )

    # Should have some holding periods, and at least one should be cash.
    assert out["holdings"]
    assert any((p.get("mode") == "cash" and p.get("picks") == []) for p in out["holdings"])
    # If always cash, NAV should be flat at 1.0 (allow floating noise).
    assert out["nav"]["series"]["ROTATION"][-1] == pytest.approx(1.0)

