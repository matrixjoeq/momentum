import datetime as dt

import pandas as pd

from etf_momentum.db.models import EtfPrice
from etf_momentum.analysis.trend import TrendInputs, compute_trend_backtest


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


def test_trend_ma_filter_smoke(session_factory):
    sf = session_factory
    code = "AAA"
    dates = pd.date_range("2024-01-01", "2024-06-30", freq="B").date  # type: ignore[attr-defined]
    with sf() as db:
        # up then down to create at least one regime change
        for i, d in enumerate(dates):
            px = 100.0 + (i * 0.5 if i < 60 else (60 * 0.5) - (i - 60) * 0.8)
            _add_price(db, code=code, day=d, close=px)
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(code=code, start=dates[0], end=dates[-1], strategy="ma_filter", sma_window=20, cost_bps=0.0),
        )
    assert out["meta"]["type"] == "trend_backtest"
    assert out["meta"]["code"] == code
    assert "nav" in out and "series" in out["nav"]
    s = out["nav"]["series"]
    assert len(out["nav"]["dates"]) == len(s["STRAT"]) == len(s["BUY_HOLD"]) == len(s["EXCESS"])
    # should have some non-trivial positions
    pos = out["signals"]["position"]
    assert any(x > 0 for x in pos)


def test_trend_nav_uses_none_with_hfq_fallback_on_corporate_action_cliff(session_factory):
    sf = session_factory
    code = "AAA"
    dates = pd.date_range("2024-01-01", "2024-01-10", freq="B").date  # type: ignore[attr-defined]
    # Build a fake split-like cliff in none price; hfq remains smooth; qfq used for signals remains smooth.
    none_px = [100, 101, 50, 51, 52, 53, 54, 55]
    hfq_px = [100, 101, 102, 103, 104, 105, 106, 107]
    qfq_px = [100, 101, 102, 103, 104, 105, 106, 107]
    with sf() as db:
        for d, n, h, q in zip(dates, none_px, hfq_px, qfq_px):
            db.add(EtfPrice(code=code, trade_date=d, open=float(n), high=float(n), low=float(n), close=float(n), volume=1.0, amount=1.0, source="eastmoney", adjust="none"))
            db.add(EtfPrice(code=code, trade_date=d, open=float(h), high=float(h), low=float(h), close=float(h), volume=1.0, amount=1.0, source="eastmoney", adjust="hfq"))
            db.add(EtfPrice(code=code, trade_date=d, open=float(q), high=float(q), low=float(q), close=float(q), volume=1.0, amount=1.0, source="eastmoney", adjust="qfq"))
        db.commit()
        out = compute_trend_backtest(
            db,
            TrendInputs(code=code, start=dates[0], end=dates[-1], strategy="ma_filter", sma_window=2, cost_bps=0.0),
        )
    nav = out["nav"]["series"]["STRAT"]
    # If none cliff was applied directly while long, nav would roughly halve; with hfq fallback it should not.
    assert min(nav) > 0.75

