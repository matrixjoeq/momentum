import datetime as dt

import pytest

from etf_momentum.db.models import EtfPrice
from etf_momentum.strategy.rotation import RotationInputs, backtest_rotation


def test_backtest_rotation_basic_outputs(session_factory):
    sf = session_factory
    with sf() as db:
        # create minimal none/hfq prices for two codes over 40 days
        codes = ["AAA", "BBB"]
        start = dt.date(2024, 1, 1)
        dates = [start + dt.timedelta(days=i) for i in range(50)]
        for i, d in enumerate(dates):
            # AAA trends up, BBB flat
            for adj in ("hfq", "none"):
                db.add(EtfPrice(code="AAA", trade_date=d, close=100 + i, source="eastmoney", adjust=adj))
                db.add(EtfPrice(code="BBB", trade_date=d, close=100, source="eastmoney", adjust=adj))
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
                risk_off=False,
                cost_bps=0.0,
            ),
        )

    assert out["nav"]["series"]["ROTATION"][0] == pytest.approx(1.0)
    assert "EW_REBAL" in out["nav"]["series"]
    assert "EXCESS" in out["nav"]["series"]
    assert "none" in out["price_basis"]["strategy_nav"]
    assert out["price_basis"]["benchmark_nav"] == "hfq"
    assert out["win_payoff"]["rebalance"] == "monthly"
    assert "kelly_fraction" in out["win_payoff"]
    assert "abs_kelly_fraction" in out["win_payoff"]
    assert "strategy" in out["metrics"]
    assert "excess_vs_equal_weight" in out["metrics"]
    assert "period_returns" in out
    assert "weekly" in out["period_returns"]
    assert "rolling" in out
    assert "returns" in out["rolling"]
    assert "corporate_actions" in out
    if out["period_details"]:
        assert "buys" in out["period_details"][0]
        assert "sells" in out["period_details"][0]

