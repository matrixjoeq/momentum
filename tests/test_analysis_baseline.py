import datetime as dt

import pytest

from etf_momentum.analysis.baseline import BaselineInputs, compute_baseline
from etf_momentum.db.models import EtfPrice


def test_compute_baseline_basic_metrics(session_factory):
    sf = session_factory
    with sf() as db:
        code_a = "AAA"
        code_b = "BBB"
        dates = [dt.date(2024, 1, d) for d in range(1, 7)]
        closes_a = [100, 101, 102, 103, 104, 105]
        closes_b = [200, 198, 202, 204, 203, 205]
        for d, ca, cb in zip(dates, closes_a, closes_b, strict=True):
            db.add(
                EtfPrice(
                    code=code_a,
                    trade_date=d,
                    close=float(ca),
                    source="eastmoney",
                    adjust="qfq",
                )
            )
            db.add(
                EtfPrice(
                    code=code_b,
                    trade_date=d,
                    close=float(cb),
                    source="eastmoney",
                    adjust="qfq",
                )
            )
        db.commit()

        out = compute_baseline(
            db,
            BaselineInputs(
                codes=[code_a, code_b],
                start=dates[0],
                end=dates[-1],
                benchmark_code=code_a,
                adjust="qfq",
                rolling_weeks=[],
                rolling_months=[],
                rolling_years=[],
            ),
        )

    assert out["codes"] == [code_a, code_b]
    assert out["date_range"]["common_start"] == "20240101"
    assert out["metrics"]["benchmark_code"] == code_a
    assert out["nav"]["dates"][0] == "2024-01-01"
    assert "EW" in out["nav"]["series"]
    assert out["nav"]["series"]["EW"][0] == pytest.approx(1.0)
    assert out["metrics"]["cumulative_return"] == pytest.approx(out["nav"]["series"]["EW"][-1] - 1.0, rel=1e-12)
    assert out["metrics"]["max_drawdown"] <= 0.0
    assert "ulcer_index" in out["metrics"]
    assert out["metrics"]["ulcer_index"] >= 0.0
    assert "ulcer_performance_index" in out["metrics"]
    assert "holding_weekly_win_rate" in out["metrics"]
    assert "holding_quarterly_payoff_ratio" in out["metrics"]
    assert "holding_yearly_kelly_fraction" in out["metrics"]
    assert "correlation" in out
    assert out["correlation"]["codes"] == [code_a, code_b]
    assert len(out["correlation"]["matrix"]) == 2
    assert len(out["correlation"]["matrix"][0]) == 2


def test_compute_baseline_includes_price_bias_distribution(session_factory):
    sf = session_factory
    with sf() as db:
        code = "AAA"
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(40)]
        closes = [100.0 + float(i) for i in range(40)]
        for d, c in zip(dates, closes, strict=True):
            db.add(
                EtfPrice(
                    code=code,
                    trade_date=d,
                    close=float(c),
                    source="eastmoney",
                    adjust="qfq",
                )
            )
        db.commit()

        out = compute_baseline(
            db,
            BaselineInputs(
                codes=[code],
                start=dates[0],
                end=dates[-1],
                benchmark_code=code,
                adjust="qfq",
                rolling_weeks=[],
                rolling_months=[],
                rolling_years=[],
            ),
        )

    pdist = out["period_distributions"][code]
    assert "daily_bias" in pdist
    bias = pdist["daily_bias"]
    assert bias["count"] > 0
    assert bias["current_date"] == dates[-1].isoformat()
    ma20_last = sum(closes[-20:]) / 20.0
    expected = closes[-1] / ma20_last - 1.0
    assert bias["current"] == pytest.approx(expected, rel=1e-12)

