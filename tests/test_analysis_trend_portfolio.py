import datetime as dt

import pandas as pd

from etf_momentum.analysis.trend import TrendPortfolioInputs, compute_trend_portfolio_backtest
from etf_momentum.db.models import EtfPrice


def _add_price(db, *, code: str, day: dt.date, close: float) -> None:
    for adj in ("none", "hfq", "qfq"):
        db.add(
            EtfPrice(
                code=code,
                trade_date=day,
                open=float(close),
                high=float(close),
                low=float(close),
                close=float(close),
                volume=1.0,
                amount=1.0,
                source="eastmoney",
                adjust=adj,
            )
        )


def test_trend_portfolio_group_enforce_and_outputs(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-03-31", freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            _add_price(db, code="A1", day=d, close=100 + i * 1.2)
            _add_price(db, code="A2", day=d, close=100 + i * 1.0)
            _add_price(db, code="B1", day=d, close=100 + i * 0.8)
        db.commit()

        out = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1", "A2", "B1"],
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=10,
                top_k=2,
                group_enforce=True,
                group_pick_policy="strongest_score",
                asset_groups={"A1": "CN", "A2": "CN", "B1": "US"},
            ),
        )
    assert out["meta"]["type"] == "trend_portfolio_backtest"
    assert "weights" in out and "holdings" in out
    if out["holdings"]:
        one = out["holdings"][0]
        assert "group_filter" in one
        assert one["group_filter"]["enabled"] is True


def test_trend_portfolio_holds_cash_when_candidates_not_greater_than_topk(session_factory):
    sf = session_factory
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-03-31", freq="B")]
    with sf() as db:
        for i, d in enumerate(dates):
            _add_price(db, code="A1", day=d, close=100 + i * 1.0)
            _add_price(db, code="A2", day=d, close=100 + i * 0.9)
        db.commit()

        out = compute_trend_portfolio_backtest(
            db,
            TrendPortfolioInputs(
                codes=["A1", "A2"],
                start=dates[0],
                end=dates[-1],
                strategy="ma_filter",
                sma_window=10,
                top_k=2,
            ),
        )

    strat = out["nav"]["series"]["STRAT"]
    assert strat
    assert all(float(x) == 1.0 for x in strat)
    assert out["holdings"]
    assert all((h.get("picks") or []) == [] for h in out["holdings"])

