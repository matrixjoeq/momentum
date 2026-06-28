from __future__ import annotations

import datetime as dt
import math

import pandas as pd
import pytest

from etf_momentum.analysis.bt_trend import (
    compute_trend_backtest_bt,
    compute_trend_portfolio_backtest_bt,
)
from etf_momentum.analysis.trend import (
    CASH_MANAGEMENT_PROXY_CODE,
    RISK_EXIT_PROXY_CODE,
    TRADING_COST_PROXY_CODE,
    TrendInputs,
    TrendPortfolioInputs,
    compute_trend_backtest,
    compute_trend_portfolio_backtest,
)
from tests.helpers.price_seed import add_price_all_adjustments


def _add_price(db, *, code: str, day: dt.date, close: float, open_price: float) -> None:
    high = max(float(close), float(open_price)) * 1.002
    low = min(float(close), float(open_price)) * 0.998
    add_price_all_adjustments(
        db,
        code=code,
        day=day,
        close=float(close),
        open_price=float(open_price),
        high=float(high),
        low=float(low),
    )


def _seed_trend_prices(session_factory) -> tuple[dt.date, dt.date]:
    dates = [d.date() for d in pd.date_range("2024-01-02", periods=140, freq="B")]
    with session_factory() as db:
        for i, day in enumerate(dates):
            base_a = 100.0 + 0.08 * i + 3.2 * math.sin(i / 4.0)
            base_b = 82.0 + 0.05 * i + 2.6 * math.cos(i / 5.0)
            open_a = base_a * (1.0 + (0.002 if i % 2 == 0 else -0.0015))
            open_b = base_b * (1.0 + (0.0015 if i % 3 == 0 else -0.001))
            cash = 100.0 + 0.02 * i
            _add_price(db, code="AAA", day=day, close=base_a, open_price=open_a)
            _add_price(db, code="BBB", day=day, close=base_b, open_price=open_b)
            _add_price(db, code="511880", day=day, close=cash, open_price=cash)
        db.commit()
    return dates[0], dates[-1]


def _assert_net_attribution_consistent(out: dict) -> None:
    metrics = ((out.get("metrics") or {}).get("strategy")) or {}
    attribution = (out.get("attribution") or {}).get("return") or {}
    rows = attribution.get("by_code") or []

    cum = float(metrics.get("cumulative_return") or 0.0)
    total = float(attribution.get("total_return") or 0.0)
    row_sum = float(
        sum(float((r or {}).get("return_contribution") or 0.0) for r in rows)
    )

    assert total == pytest.approx(cum, abs=1e-10)
    assert row_sum == pytest.approx(cum, abs=1e-10)

    by_code = {str((r or {}).get("code") or ""): r or {} for r in rows}
    assert CASH_MANAGEMENT_PROXY_CODE in by_code
    assert RISK_EXIT_PROXY_CODE in by_code
    assert TRADING_COST_PROXY_CODE in by_code
    assert (
        float(by_code[TRADING_COST_PROXY_CODE].get("return_contribution") or 0.0)
        <= 1e-12
    )


def test_trend_single_attribution_net_consistency_all_engines(session_factory):
    start, end = _seed_trend_prices(session_factory)
    cases = [
        ("legacy", compute_trend_backtest),
        ("bt", compute_trend_backtest_bt),
    ]
    for _, fn in cases:
        with session_factory() as db:
            out = fn(
                db,
                TrendInputs(
                    code="AAA",
                    start=start,
                    end=end,
                    strategy="ma_filter",
                    sma_window=4,
                    position_sizing="fixed_ratio",
                    fixed_pos_ratio=0.75,
                    exec_price="open",
                    cost_bps=18.0,
                    slippage_rate=0.0018,
                ),
            )
        _assert_net_attribution_consistent(out)


def test_trend_portfolio_attribution_net_consistency_all_engines(session_factory):
    start, end = _seed_trend_prices(session_factory)
    cases = [
        ("legacy", compute_trend_portfolio_backtest),
        ("bt", compute_trend_portfolio_backtest_bt),
    ]
    for _, fn in cases:
        with session_factory() as db:
            out = fn(
                db,
                TrendPortfolioInputs(
                    codes=["AAA", "BBB"],
                    start=start,
                    end=end,
                    strategy="ma_filter",
                    sma_window=4,
                    fixed_max_holdings=1,
                    position_sizing="equal",
                    exec_price="open",
                    cost_bps=20.0,
                    slippage_rate=0.002,
                ),
            )
        _assert_net_attribution_consistent(out)
