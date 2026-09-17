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
    assert "current_holdings" in out
    assert "current_holdings_by_portfolio" in out
    ew_hold = out["current_holdings_by_portfolio"]["EW"]
    assert isinstance(ew_hold, list)
    assert len(ew_hold) == 2
    assert {str(x["code"]) for x in ew_hold} == {code_a, code_b}
    for row in ew_hold:
        assert "weight" in row
        assert float(row["weight"]) > 0.0
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
                dca_base_amount=0.0,
                dca_periodic_amount=20.0,
                dca_frequency="daily",
            ),
        )

    m = out["metrics"]
    assert bool(m.get("dca_enabled")) is True
    expected_invested = 20.0 * (len(dates) - 1)
    assert float(m["dca_total_invested"]) == pytest.approx(expected_invested, rel=1e-12)
    assert float(m["dca_final_value"]) > expected_invested
    assert float(m["dca_cumulative_return"]) == pytest.approx(
        float(m["dca_final_value"]) / float(m["dca_total_invested"]) - 1.0,
        rel=1e-12,
    )
    dca_series = (out.get("dca") or {}).get("series") or {}
    acct = [float(x) for x in (dca_series.get("account_value") or [])]
    assert len(acct) == len(dates)
    assert acct[0] == pytest.approx(0.0, abs=1e-12)
    # Strategy metrics are time-weighted and must not count deposits as gains.
    strategy_nav = [float(x) for x in out["nav"]["series"]["EW"]]
    strat_cum = float(strategy_nav[-1] / strategy_nav[0] - 1.0)
    assert float(m["cumulative_return"]) == pytest.approx(strat_cum, rel=1e-12)
    expected_ann = float(
        (strategy_nav[-1] / strategy_nav[0]) ** (252.0 / (len(strategy_nav) - 1)) - 1.0
    )
    assert float(m["annualized_return"]) == pytest.approx(expected_ann, rel=1e-12)
    peak = -float("inf")
    mdd = 0.0
    for v in strategy_nav:
        peak = max(peak, float(v))
        mdd = min(mdd, float(v) / peak - 1.0)
    assert float(m["max_drawdown"]) == pytest.approx(float(mdd), rel=1e-12)
    assert float(m["dca_time_weighted_return"]) == pytest.approx(strat_cum, rel=1e-12)
    assert np.isfinite(float(m["dca_money_weighted_return"]))
    dca = out.get("dca") or {}
    assert bool(dca.get("enabled")) is True
    assert len((dca.get("series") or {}).get("dates") or []) == len(out["nav"]["dates"])
    dca_by = out.get("dca_by_portfolio") or {}
    assert float(
        (dca_by.get("EW") or {}).get("metrics", {}).get("dca_total_invested")
    ) == pytest.approx(expected_invested, rel=1e-12)


def test_compute_baseline_includes_price_bias_distribution(session_factory):
    sf = session_factory
    with sf() as db:
        code = "AAA"
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(80)]
        closes = [100.0 + float(i) for i in range(80)]
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
    assert "daily_bias_20" in pdist
    assert "daily_bias_60" in pdist
    bias = pdist["daily_bias"]
    bias20 = pdist["daily_bias_20"]
    bias60 = pdist["daily_bias_60"]
    assert bias["count"] > 0
    assert bias20["count"] > 0
    assert bias60["count"] > 0
    assert bias["current_date"] == dates[-1].isoformat()
    assert bias20["current_date"] == dates[-1].isoformat()
    assert bias60["current_date"] == dates[-1].isoformat()
    ma20_last = sum(closes[-20:]) / 20.0
    ma60_last = sum(closes[-60:]) / 60.0
    expected20 = closes[-1] / ma20_last - 1.0
    expected60 = closes[-1] / ma60_last - 1.0
    assert bias["current"] == pytest.approx(expected20, rel=1e-12)
    assert bias20["current"] == pytest.approx(expected20, rel=1e-12)
    assert bias60["current"] == pytest.approx(expected60, rel=1e-12)
    # Legacy alias must match explicit BIAS(20)
    assert bias["current"] == bias20["current"]
    assert bias["count"] == bias20["count"]


def test_compute_baseline_includes_bias_l_distribution(session_factory):
    sf = session_factory
    with sf() as db:
        code = "AAA"
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(90)]
        closes = [100.0 + 0.4 * i + (1.1 if (i % 5) < 2 else -0.7) for i in range(90)]
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
    assert "daily_bias_l_20" in pdist
    assert "daily_bias_l_60" in pdist
    bias_l20 = pdist["daily_bias_l_20"]
    bias_l60 = pdist["daily_bias_l_60"]
    assert bias_l20["count"] > 0
    assert bias_l60["count"] > 0
    assert bias_l20["current_date"] == dates[-1].isoformat()
    assert bias_l60["current_date"] == dates[-1].isoformat()

    idx = pd.to_datetime(dates)
    close_s = pd.Series(closes, index=idx, dtype=float)
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=10).mean()
    ema60 = close_s.ewm(span=60, adjust=False, min_periods=30).mean()
    expected20 = float(
        ((np.log(close_s) - np.log(ema20.replace(0.0, pd.NA))) * 100.0)
        .dropna()
        .iloc[-1]
    )
    expected60 = float(
        ((np.log(close_s) - np.log(ema60.replace(0.0, pd.NA))) * 100.0)
        .dropna()
        .iloc[-1]
    )
    assert bias_l20["current"] == pytest.approx(expected20, rel=1e-12)
    assert bias_l60["current"] == pytest.approx(expected60, rel=1e-12)


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
    assert "daily_bias_v_20" in pdist
    assert "daily_bias_v_60" in pdist
    bias_v = pdist["daily_bias_v"]
    bias_v20 = pdist["daily_bias_v_20"]
    bias_v60 = pdist["daily_bias_v_60"]
    assert bias_v["count"] > 0
    assert bias_v20["count"] > 0
    assert bias_v60["count"] > 0
    assert bias_v["current_date"] == dates[-1].isoformat()
    assert bias_v20["current_date"] == dates[-1].isoformat()
    assert bias_v60["current_date"] == dates[-1].isoformat()

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
    ma20 = close_s.rolling(window=20, min_periods=5).mean()
    atr20 = tr.ewm(alpha=1.0 / 20.0, adjust=False, min_periods=20).mean()
    bias_v_s = ((close_s - ma20) / atr20.replace(0.0, pd.NA)).dropna()
    ma60 = close_s.rolling(window=60, min_periods=15).mean()
    atr60 = tr.ewm(alpha=1.0 / 60.0, adjust=False, min_periods=60).mean()
    bias_v60_s = ((close_s - ma60) / atr60.replace(0.0, pd.NA)).dropna()
    assert bias_v["current"] == pytest.approx(float(bias_v_s.iloc[-1]), rel=1e-12)
    assert bias_v20["current"] == pytest.approx(float(bias_v_s.iloc[-1]), rel=1e-12)
    assert bias_v60["current"] == pytest.approx(float(bias_v60_s.iloc[-1]), rel=1e-12)
    # Legacy alias must match explicit BIAS-V(20)
    assert bias_v["current"] == bias_v20["current"]
    assert bias_v["count"] == bias_v20["count"]


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


def _assert_cvar_overlay_shape(out):
    ov = out["cvar_overlay"]
    assert ov["confidence"] == pytest.approx(0.95)
    by = ov["by_portfolio"]
    for mode in ("EW", "RP", "IVOL", "CUSTOM"):
        pack = by[mode]
        assert "prompt" in pack and "sim" in pack
        prompt = pack["prompt"]
        assert prompt["status"] in {
            "ok",
            "insufficient_samples",
            "non_positive_cvar",
            "invalid_cvar",
        }
        assert prompt["applies_to"] == "下一交易日"
        sim = pack["sim"]
        dates = out["nav"]["dates"]
        assert len(sim["nav"]) == len(dates)
        assert len(sim["scale"]) == len(dates)
        assert sim["nav"][0] == pytest.approx(1.0)
        m = sim["metrics"]
        for k in (
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "max_drawdown",
        ):
            assert k in m


def test_compute_baseline_cvar_overlay_does_not_change_orig_nav(session_factory):
    sf = session_factory
    with sf() as db:
        code_a = "AAA"
        code_b = "BBB"
        dates = [dt.date(2024, 1, d) for d in range(1, 7)]
        closes_a = [100, 101, 102, 103, 104, 105]
        # Keep the first portfolio return non-zero so the test catches any
        # overlay recurrence that accidentally drops r_orig[0].
        closes_b = [200, 202, 204, 206, 208, 210]
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
        inp = BaselineInputs(
            codes=[code_a, code_b],
            start=dates[0],
            end=dates[-1],
            benchmark_code=code_a,
            adjust="qfq",
            rolling_weeks=[],
            rolling_months=[],
            rolling_years=[],
        )
        out = compute_baseline(db, inp)

    orig_ew = list(out["nav"]["series"]["EW"])
    _assert_cvar_overlay_shape(out)
    assert out["nav"]["series"]["EW"] == orig_ew
    ov = out["cvar_overlay"]["by_portfolio"]["EW"]
    assert ov["prompt"]["status"] == "insufficient_samples"
    assert ov["sim"]["nav"] == pytest.approx(orig_ew, rel=0.0, abs=1e-12)
    holds = out["current_holdings_by_portfolio"]["EW"]
    sug = {x["code"]: float(x["weight"]) for x in ov["prompt"]["suggested_weights"]}
    for row in holds:
        code = str(row["code"])
        assert code in sug
        assert sug[code] == pytest.approx(float(row["weight"]), rel=0.0, abs=1e-9)
    asof = ov["prompt"]["asof"]
    assert asof == dates[-1].isoformat()
    assert asof != inp.end.strftime("%Y%m%d")


def test_compute_baseline_cvar_hs_skips_raw_missing_price_days(session_factory):
    sf = session_factory
    with sf() as db:
        code_a = "AAA"
        code_b = "BBB"
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(21)]
        for i, d in enumerate(dates):
            if i != 10:
                db.add(
                    EtfPrice(
                        code=code_a,
                        trade_date=d,
                        close=float(100.0 + i),
                        source="eastmoney",
                        adjust="qfq",
                    )
                )
            db.add(
                EtfPrice(
                    code=code_b,
                    trade_date=d,
                    close=float(200.0 + i),
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
                benchmark_code=code_b,
                adjust="qfq",
                rolling_weeks=[],
                rolling_months=[],
                rolling_years=[],
                cvar_window=20,
            ),
        )

    prompt = out["cvar_overlay"]["by_portfolio"]["EW"]["prompt"]
    assert prompt["status"] == "insufficient_samples"
    # First row plus the missing row and its following row have no genuine
    # close-to-close return, leaving 18 valid observations.
    assert prompt["sample_count"] == 18


def test_compute_baseline_cvar_rebalance_none_scale_one_matches_nav(session_factory):
    sf = session_factory
    with sf() as db:
        code = "AAA"
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(8)]
        closes = [100.0 + i for i in range(8)]
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
                rebalance="none",
                rolling_weeks=[],
                rolling_months=[],
                rolling_years=[],
            ),
        )
    ew = out["nav"]["series"]["EW"]
    ov = out["cvar_overlay"]["by_portfolio"]["EW"]
    assert all(abs(float(x) - 1.0) <= 1e-12 for x in ov["sim"]["scale"])
    assert ov["sim"]["nav"] == pytest.approx(ew, rel=0.0, abs=1e-12)


def test_compute_baseline_cvar_shrink_and_dca_uses_twr(session_factory):
    sf = session_factory
    with sf() as db:
        code = "AAA"
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(40)]
        closes = [100.0]
        for i in range(1, 40):
            closes.append(closes[-1] * 0.97)
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
                cvar_window=20,
                cvar_budget_pct=0.02,
                dca_enabled=True,
                dca_base_amount=100.0,
                dca_periodic_amount=10.0,
                dca_frequency="daily",
            ),
        )
        out_no_dca = compute_baseline(
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
                cvar_window=20,
                cvar_budget_pct=0.02,
                dca_enabled=False,
            ),
        )
    orig = out["nav"]["series"]["EW"]
    ov = out["cvar_overlay"]["by_portfolio"]["EW"]
    assert orig[0] == pytest.approx(1.0)
    assert ov["sim"]["nav"][0] == pytest.approx(1.0)
    assert min(float(x) for x in ov["sim"]["scale"]) < 1.0
    ov_no_dca = out_no_dca["cvar_overlay"]["by_portfolio"]["EW"]
    assert ov["sim"]["nav"] == pytest.approx(
        ov_no_dca["sim"]["nav"], rel=0.0, abs=1e-12
    )
    assert ov["sim"]["scale"] == pytest.approx(
        ov_no_dca["sim"]["scale"], rel=0.0, abs=1e-12
    )
    dca_acct = ((out.get("dca") or {}).get("series") or {}).get("account_value") or []
    if dca_acct:
        assert ov["sim"]["nav"][-1] != pytest.approx(float(dca_acct[-1]), abs=1e-6)
    prompt = ov["prompt"]
    if prompt["status"] == "ok" and float(prompt["scale"]) < 1.0 - 1e-12:
        holds = out["current_holdings_by_portfolio"]["EW"]
        sug = {x["code"]: float(x["weight"]) for x in prompt["suggested_weights"]}
        for row in holds:
            code = str(row["code"])
            assert sug[code] == pytest.approx(
                float(row["weight"]) * float(prompt["scale"]), rel=0.0, abs=1e-9
            )
