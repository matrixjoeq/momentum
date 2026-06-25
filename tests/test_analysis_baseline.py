import datetime as dt

import numpy as np
import pandas as pd
import pytest

import etf_momentum.analysis.baseline as baseline_module
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
    assert out["metrics"]["cumulative_return"] == pytest.approx(
        out["nav"]["series"]["EW"][-1] - 1.0, rel=1e-12
    )
    assert out["metrics"]["max_drawdown"] <= 0.0
    assert "ulcer_index" in out["metrics"]
    assert out["metrics"]["ulcer_index"] >= 0.0
    assert "ulcer_performance_index" in out["metrics"]
    assert "avg_daily_turnover" in out["metrics"]
    assert "avg_annual_turnover" in out["metrics"]
    assert "avg_daily_trade_count" in out["metrics"]
    assert "avg_annual_trade_count" in out["metrics"]
    assert "holding_weekly_win_rate" in out["metrics"]
    assert "holding_quarterly_payoff_ratio" in out["metrics"]
    assert "holding_yearly_kelly_fraction" in out["metrics"]
    assert "correlation" in out
    assert out["correlation"]["codes"] == [code_a, code_b]
    assert len(out["correlation"]["matrix"]) == 2
    assert len(out["correlation"]["matrix"][0]) == 2


def test_compute_baseline_supports_dca_metrics(session_factory):
    sf = session_factory
    with sf() as db:
        code = "AAA"
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(8)]
        closes = [100.0 * (1.01**i) for i in range(len(dates))]
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
                rebalance="weekly",
                rolling_weeks=[],
                rolling_months=[],
                rolling_years=[],
                dca_enabled=True,
                dca_base_amount=100.0,
                dca_periodic_amount=20.0,
                dca_frequency="daily",
            ),
        )

    m = out["metrics"]
    assert bool(m.get("dca_enabled")) is True
    expected_invested = 100.0 + 20.0 * (len(dates) - 1)
    assert float(m["dca_total_invested"]) == pytest.approx(
        expected_invested, rel=1e-12
    )
    assert float(m["dca_final_value"]) > expected_invested
    assert float(m["dca_cumulative_return"]) == pytest.approx(
        float(m["dca_final_value"]) / float(m["dca_total_invested"]) - 1.0,
        rel=1e-12,
    )
    dca_series = (out.get("dca") or {}).get("series") or {}
    acct = [float(x) for x in (dca_series.get("account_value") or [])]
    assert len(acct) == len(dates)
    expected_curve_cum = float(acct[-1] / acct[0] - 1.0)
    assert float(m["cumulative_return"]) == pytest.approx(expected_curve_cum, rel=1e-12)
    expected_ann = float((acct[-1] / acct[0]) ** (252.0 / (len(acct) - 1)) - 1.0)
    assert float(m["annualized_return"]) == pytest.approx(expected_ann, rel=1e-12)
    peak = -float("inf")
    mdd = 0.0
    for v in acct:
        peak = max(peak, float(v))
        mdd = min(mdd, float(v) / peak - 1.0)
    assert float(m["max_drawdown"]) == pytest.approx(float(mdd), rel=1e-12)
    strat_cum = float(out["nav"]["series"]["EW"][-1] - 1.0)
    assert float(m["dca_time_weighted_return"]) == pytest.approx(
        strat_cum, rel=1e-12
    )
    assert float(m["cumulative_return"]) != pytest.approx(strat_cum, rel=1e-9, abs=1e-9)
    assert np.isfinite(float(m["dca_money_weighted_return"]))
    dca = out.get("dca") or {}
    assert bool(dca.get("enabled")) is True
    assert len((dca.get("series") or {}).get("dates") or []) == len(
        out["nav"]["dates"]
    )
    dca_by = out.get("dca_by_portfolio") or {}
    assert float(
        (dca_by.get("EW") or {}).get("metrics", {}).get("dca_total_invested")
    ) == pytest.approx(expected_invested, rel=1e-12)


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


def test_compute_baseline_lppl_library_unavailable(
    session_factory, monkeypatch: pytest.MonkeyPatch
):
    sf = session_factory
    with sf() as db:
        code = "AAA"
        dates = [dt.date(2023, 1, 1) + dt.timedelta(days=i) for i in range(220)]
        closes = [100.0 + 0.25 * i + (1.0 if (i % 9) < 4 else -0.6) for i in range(220)]
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
        monkeypatch.setattr(baseline_module, "_LPPLS_MODULE", None)
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
                lppl_enabled=True,
                lppl_lookback_days=180,
                lppl_min_points=60,
            ),
        )
    pdist = out["period_distributions"][code]
    assert "daily_lppl" in pdist
    lppl = pdist["daily_lppl"]
    assert lppl["status"] == "library_unavailable"
    assert "lppls_not_installed" in (lppl.get("reason_codes") or [])


def test_compute_baseline_includes_macd_v_distributions(session_factory):
    sf = session_factory
    with sf() as db:
        code = "AAA"
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(120)]
        closes = [100.0 + 0.2 * i + (1.5 if (i % 7) < 3 else -1.0) for i in range(120)]
        for d, c in zip(dates, closes, strict=True):
            db.add(
                EtfPrice(
                    code=code,
                    trade_date=d,
                    open=float(c),
                    high=float(c * 1.01),
                    low=float(c * 0.99),
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
    assert "daily_macd_v_dif" in pdist
    assert "daily_macd_v_dea" in pdist
    assert "daily_macd_v_h" in pdist
    assert "daily_macd_v_absh" in pdist
    assert "daily_macd_abs" in pdist
    dif = pdist["daily_macd_v_dif"]
    dea = pdist["daily_macd_v_dea"]
    h = pdist["daily_macd_v_h"]
    absh = pdist["daily_macd_v_absh"]
    macd_abs = pdist["daily_macd_abs"]
    assert dif["count"] > 0
    assert dea["count"] > 0
    assert h["count"] > 0
    assert absh["count"] > 0
    assert macd_abs["count"] > 0
    assert dif["current_date"] == dates[-1].isoformat()
    assert dea["current_date"] == dates[-1].isoformat()
    assert h["current_date"] == dates[-1].isoformat()
    assert absh["current_date"] == dates[-1].isoformat()
    assert macd_abs["current_date"] == dates[-1].isoformat()

    idx = pd.to_datetime(dates)
    close_s = pd.Series(closes, index=idx, dtype=float)
    high_s = close_s * 1.01
    low_s = close_s * 0.99
    prev_close = close_s.shift(1)
    tr = pd.concat(
        [
            (high_s - low_s).abs(),
            (high_s - prev_close).abs(),
            (low_s - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr26 = tr.ewm(alpha=1.0 / 26.0, adjust=False, min_periods=26).mean()
    ema12 = close_s.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close_s.ewm(span=26, adjust=False, min_periods=26).mean()
    dif_s = ((ema12 - ema26) / atr26.replace(0.0, pd.NA)).dropna()
    dea_s = dif_s.ewm(span=9, adjust=False, min_periods=9).mean().dropna()
    dif_s = dif_s.reindex(dea_s.index).dropna()
    h_s = 2.0 * (dif_s - dea_s)
    absh_s = h_s.abs()
    std_dif_s = (ema12 - ema26).dropna()
    std_dea_s = std_dif_s.ewm(span=9, adjust=False, min_periods=9).mean().dropna()
    std_dif_s = std_dif_s.reindex(std_dea_s.index).dropna()
    std_macd_abs_s = (2.0 * (std_dif_s - std_dea_s)).abs()
    assert dif["current"] == pytest.approx(float(dif_s.iloc[-1]), rel=1e-12)
    assert dea["current"] == pytest.approx(float(dea_s.iloc[-1]), rel=1e-12)
    assert h["current"] == pytest.approx(float(h_s.iloc[-1]), rel=1e-12)
    assert absh["current"] == pytest.approx(float(absh_s.iloc[-1]), rel=1e-12)
    assert macd_abs["current"] == pytest.approx(
        float(std_macd_abs_s.iloc[-1]), rel=1e-12
    )


def test_compute_baseline_includes_bias_v_distribution(session_factory):
    sf = session_factory
    with sf() as db:
        code = "AAA"
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(120)]
        closes = [100.0 + 0.3 * i + (1.2 if (i % 6) < 3 else -0.8) for i in range(120)]
        for d, c in zip(dates, closes, strict=True):
            db.add(
                EtfPrice(
                    code=code,
                    trade_date=d,
                    open=float(c),
                    high=float(c * 1.01),
                    low=float(c * 0.99),
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
    assert "daily_bias_v" in pdist
    bias_v = pdist["daily_bias_v"]
    assert bias_v["count"] > 0
    assert bias_v["current_date"] == dates[-1].isoformat()

    idx = pd.to_datetime(dates)
    close_s = pd.Series(closes, index=idx, dtype=float)
    high_s = close_s * 1.01
    low_s = close_s * 0.99
    ma20 = close_s.rolling(window=20, min_periods=5).mean()
    prev_close = close_s.shift(1)
    tr = pd.concat(
        [
            (high_s - low_s).abs(),
            (high_s - prev_close).abs(),
            (low_s - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr20 = tr.ewm(alpha=1.0 / 20.0, adjust=False, min_periods=20).mean()
    bias_v_s = ((close_s - ma20) / atr20.replace(0.0, pd.NA)).dropna()
    assert bias_v["current"] == pytest.approx(float(bias_v_s.iloc[-1]), rel=1e-12)


def test_compute_baseline_includes_daily_log_return_acf(session_factory):
    sf = session_factory
    with sf() as db:
        code = "AAA"
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(90)]
        closes = [
            100.0 + 0.15 * i + (0.8 if (i % 9) < 3 else (-0.5 if (i % 9) < 6 else 0.2))
            for i in range(90)
        ]
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
    assert "daily_log_return_acf" in pdist
    acf_stats = pdist["daily_log_return_acf"]
    assert int(acf_stats["sample_size"]) >= 10
    assert float(acf_stats["white_noise_bound"]) > 0.0
    lags = list(acf_stats["lags"])
    acf_vals = list(acf_stats["acf"])
    assert len(lags) == len(acf_vals)
    assert len(lags) >= 1
    assert len(lags) <= 20
    assert lags[0] == 1
    assert lags[-1] == len(lags)

    idx = pd.to_datetime(dates)
    close_s = pd.Series(closes, index=idx, dtype=float)
    ret_s = close_s.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    lr = np.log1p(ret_s).replace([np.inf, -np.inf], np.nan).dropna()
    x = lr.to_numpy(dtype=float)
    x = x - float(np.mean(x))
    expected_lag1 = float(np.dot(x[1:], x[:-1]) / np.dot(x, x))
    assert float(acf_vals[0]) == pytest.approx(expected_lag1, rel=1e-12)

    per_lag = acf_stats["per_lag_significance"]
    assert len(per_lag) == len(lags)
    assert all("conclusion" in row for row in per_lag)
    for row in per_lag:
        conc = str(row["conclusion"])
        if bool(row["significant"]):
            assert conc.startswith("显著")
        else:
            assert conc.startswith("不显著")

    lb_rows = acf_stats["ljung_box"]
    assert len(lb_rows) == len(lags)
    for row in lb_rows:
        p = float(row["p_value"])
        assert 0.0 <= p <= 1.0
        assert "conclusion" in row
