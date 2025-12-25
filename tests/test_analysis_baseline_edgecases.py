import datetime as dt

import numpy as np
import pandas as pd
import pytest

import etf_momentum.analysis.baseline as bl
from etf_momentum.analysis.baseline import BaselineInputs, compute_baseline
from etf_momentum.db.models import EtfPrice


def test_helper_metrics_empty_series_return_nan():
    s = pd.Series([], dtype=float)
    assert np.isnan(bl._annualized_return(s))
    assert np.isnan(bl._annualized_vol(s))
    assert np.isnan(bl._sharpe(s))
    assert np.isnan(bl._sortino(s))
    assert np.isnan(bl._information_ratio(s))


def test_helper_metrics_zero_std_returns_nan():
    r = pd.Series([0.0, 0.0, 0.0], dtype=float)
    assert np.isnan(bl._sharpe(r))
    assert np.isnan(bl._sortino(r))
    assert np.isnan(bl._information_ratio(r))


def test_max_drawdown_duration_with_recovery():
    idx = pd.to_datetime([dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)])
    nav = pd.Series([1.0, 0.8, 0.9, 1.1], index=idx)
    assert bl._max_drawdown(nav) == pytest.approx(-0.2)
    assert bl._max_drawdown_duration_days(nav) == 2


def test_compute_baseline_errors(session_factory):
    sf = session_factory
    with sf() as db:
        with pytest.raises(ValueError, match="codes is empty"):
            compute_baseline(db, BaselineInputs(codes=[], start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 2)))
        with pytest.raises(ValueError, match="no price data"):
            compute_baseline(db, BaselineInputs(codes=["AAA"], start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 2)))


def test_compute_baseline_rolling_and_missing_benchmark(session_factory):
    sf = session_factory
    with sf() as db:
        code_a = "510300"
        code_b = "BBB"
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(300)]
        # create a gently trending series
        for i, d in enumerate(dates):
            db.add(EtfPrice(code=code_a, trade_date=d, close=100.0 + i, source="eastmoney", adjust="qfq"))
            db.add(EtfPrice(code=code_b, trade_date=d, close=200.0 + i * 0.5, source="eastmoney", adjust="qfq"))
        db.commit()

        out = compute_baseline(
            db,
            BaselineInputs(
                codes=[code_a, code_b],
                start=dates[0],
                end=dates[-1],
                benchmark_code="MISSING",
                adjust="qfq",
                rolling_weeks=[4],
                rolling_months=[3],
                rolling_years=[1],
            ),
        )

    assert out["metrics"]["benchmark_code"] == "MISSING"
    assert "BENCH:MISSING" in out["nav"]["series"]
    assert out["rolling"]["returns"]["4w"]["values"]
    assert out["rolling"]["max_drawdown"]["4w"]["values"]
    assert out["rolling"]["returns"]["3m"]["values"]
    assert out["rolling"]["max_drawdown"]["3m"]["values"]
    assert out["rolling"]["returns"]["1y"]["values"]
    assert out["rolling"]["max_drawdown"]["1y"]["values"]

