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
    idx = pd.to_datetime(
        [
            dt.date(2024, 1, 1),
            dt.date(2024, 1, 2),
            dt.date(2024, 1, 3),
            dt.date(2024, 1, 4),
        ]
    )
    nav = pd.Series([1.0, 0.8, 0.9, 1.1], index=idx)
    assert bl._max_drawdown(nav) == pytest.approx(-0.2)
    assert bl._max_drawdown_duration_days(nav) == 2


def test_compute_custom_weight_nav_weekly_rebalance_no_numpy_iloc_crash():
    """Regression: custom weights + periodic rebalance must not assume ndarray has .iloc."""
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    daily_ret = pd.DataFrame(
        0.001, index=idx, columns=["510300", "511010", "513100", "518880"]
    )
    tw = pd.Series({"510300": 0.3, "511010": 0.3, "513100": 0.3, "518880": 0.1})
    nav, w = bl._compute_custom_weight_nav_and_weights(
        daily_ret,
        rebalance="weekly",
        target_weights=tw,
    )
    assert len(nav) == 10
    assert w.shape == (10, 4)
    assert np.isfinite(nav.to_numpy(dtype=float)).all()


def test_compute_baseline_errors(session_factory):
    sf = session_factory
    with sf() as db:
        with pytest.raises(ValueError, match="codes is empty"):
            compute_baseline(
                db,
                BaselineInputs(
                    codes=[], start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 2)
                ),
            )
        with pytest.raises(ValueError, match="no price data"):
            compute_baseline(
                db,
                BaselineInputs(
                    codes=["AAA"], start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 2)
                ),
            )


def test_compute_baseline_rolling_and_missing_benchmark(session_factory):
    sf = session_factory
    with sf() as db:
        code_a = "510300"
        code_b = "BBB"
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(300)]
        # create a gently trending series
        for i, d in enumerate(dates):
            db.add(
                EtfPrice(
                    code=code_a,
                    trade_date=d,
                    close=100.0 + i,
                    source="eastmoney",
                    adjust="qfq",
                )
            )
            db.add(
                EtfPrice(
                    code=code_b,
                    trade_date=d,
                    close=200.0 + i * 0.5,
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
    assert out["rolling"]["drawdown"]["4w"]["values"]
    assert out["rolling"]["returns"]["3m"]["values"]
    assert out["rolling"]["drawdown"]["3m"]["values"]
    assert out["rolling"]["returns"]["1y"]["values"]
    assert out["rolling"]["drawdown"]["1y"]["values"]


def test_compute_baseline_dynamic_universe_uses_union_start(session_factory):
    sf = session_factory
    with sf() as db:
        code_a = "AAA"
        code_b = "BBB"
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(120)]
        for i, d in enumerate(dates):
            db.add(
                EtfPrice(
                    code=code_a,
                    trade_date=d,
                    close=100.0 + i,
                    source="eastmoney",
                    adjust="qfq",
                )
            )
            # BBB starts later
            if i >= 80:
                db.add(
                    EtfPrice(
                        code=code_b,
                        trade_date=d,
                        close=50.0 + (i - 80) * 0.7,
                        source="eastmoney",
                        adjust="qfq",
                    )
                )
        db.commit()

        out_inter = compute_baseline(
            db,
            BaselineInputs(
                codes=[code_a, code_b],
                start=dates[0],
                end=dates[-1],
                adjust="qfq",
                dynamic_universe=False,
            ),
        )
        out_union = compute_baseline(
            db,
            BaselineInputs(
                codes=[code_a, code_b],
                start=dates[0],
                end=dates[-1],
                adjust="qfq",
                dynamic_universe=True,
            ),
        )

    assert (
        out_inter["date_range"]["mode_start"] == out_inter["date_range"]["common_start"]
    )
    assert (
        out_union["date_range"]["mode_start"] <= out_union["date_range"]["common_start"]
    )
    # In dynamic mode, early days only one asset is active.
    vals = out_union["active_count"]["values"]
    assert vals and max(vals) >= 2 and 1 in vals


@pytest.mark.parametrize("dynamic_universe", [False, True])
def test_compute_baseline_skips_untradable_candidates_regardless_dynamic_universe(
    session_factory, dynamic_universe: bool
):
    sf = session_factory
    with sf() as db:
        code_a = "AAA"
        missing_code = "159985"
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(80)]
        for i, d in enumerate(dates):
            db.add(
                EtfPrice(
                    code=code_a,
                    trade_date=d,
                    close=100.0 + i,
                    source="eastmoney",
                    adjust="qfq",
                )
            )
        db.commit()

        out = compute_baseline(
            db,
            BaselineInputs(
                codes=[code_a, missing_code],
                start=dates[0],
                end=dates[-1],
                adjust="qfq",
                dynamic_universe=bool(dynamic_universe),
            ),
        )

    skipped = list(out.get("untradable_codes_skipped") or [])
    assert missing_code in skipped
    ew_nav = list(((out.get("nav") or {}).get("series") or {}).get("EW") or [])
    assert ew_nav
    assert float(ew_nav[-1]) > 1.0


def test_compute_baseline_single_asset_portfolios_track_asset_nav(session_factory):
    sf = session_factory
    with sf() as db:
        code = "AAA"
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(60)]
        for i, d in enumerate(dates):
            db.add(
                EtfPrice(
                    code=code,
                    trade_date=d,
                    close=100.0 + i,
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
                adjust="qfq",
                dynamic_universe=False,
            ),
        )

    ew = out["nav"]["series"]["EW"]
    rp = out["nav"]["series"]["RP"]
    assert ew and rp
    expected_last = (100.0 + (len(dates) - 1)) / 100.0
    assert float(ew[-1]) == pytest.approx(expected_last, rel=1e-9)
    assert float(rp[-1]) == pytest.approx(expected_last, rel=1e-9)
    m = out.get("metrics") or {}
    assert float(m.get("cumulative_return") or 0.0) > 0.0


def test_compute_baseline_attribution_respects_exec_price_mode(session_factory):
    sf = session_factory
    with sf() as db:
        code_a = "A"
        code_b = "B"
        dates = [d.date() for d in pd.date_range("2024-01-01", periods=10, freq="B")]
        close_a = [100.0 + 0.2 * i for i in range(len(dates))]
        close_b = [100.0 + 0.2 * i for i in range(len(dates))]
        exec_days = {dates[0], dates[5]}
        for d, ca, cb in zip(dates, close_a, close_b, strict=True):
            oa = float(ca / 1.08) if d in exec_days else float(ca)
            ob = float(cb / 0.95) if d in exec_days else float(cb)
            db.add(
                EtfPrice(
                    code=code_a,
                    trade_date=d,
                    open=oa,
                    close=float(ca),
                    source="eastmoney",
                    adjust="qfq",
                )
            )
            db.add(
                EtfPrice(
                    code=code_b,
                    trade_date=d,
                    open=ob,
                    close=float(cb),
                    source="eastmoney",
                    adjust="qfq",
                )
            )
        db.commit()

        out_open = compute_baseline(
            db,
            BaselineInputs(
                codes=[code_a, code_b],
                start=dates[0],
                end=dates[-1],
                adjust="qfq",
                rebalance="weekly",
                exec_price="open",
                holding_mode="EW",
                rolling_weeks=[],
                rolling_months=[],
                rolling_years=[],
            ),
        )
        out_close = compute_baseline(
            db,
            BaselineInputs(
                codes=[code_a, code_b],
                start=dates[0],
                end=dates[-1],
                adjust="qfq",
                rebalance="weekly",
                exec_price="close",
                holding_mode="EW",
                rolling_weeks=[],
                rolling_months=[],
                rolling_years=[],
            ),
        )

    m_open = (out_open.get("metrics_by_portfolio") or {}).get("EW") or {}
    m_close = (out_close.get("metrics_by_portfolio") or {}).get("EW") or {}
    assert float(m_open.get("cumulative_return") or 0.0) != pytest.approx(
        float(m_close.get("cumulative_return") or 0.0), rel=0.0, abs=1e-9
    )

    a_open = (
        ((out_open.get("attribution_by_portfolio") or {}).get("EW") or {})
        .get("return", {})
        .get("by_code", [])
    )
    a_close = (
        ((out_close.get("attribution_by_portfolio") or {}).get("EW") or {})
        .get("return", {})
        .get("by_code", [])
    )
    share_open = {str(x["code"]): x.get("return_share") for x in a_open}
    share_close = {str(x["code"]): x.get("return_share") for x in a_close}
    assert share_open.get(code_a) is not None
    assert share_close.get(code_a) is not None
    assert abs(float(share_open[code_a]) - float(share_close[code_a])) > 0.05


def test_compute_baseline_attribution_dca_mode_uses_money_weighted_curve(
    session_factory,
):
    sf = session_factory
    with sf() as db:
        code = "AAA"
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(10)]
        closes = [100.0 * (1.01**i) for i in range(len(dates))]
        for d, c in zip(dates, closes, strict=True):
            db.add(
                EtfPrice(
                    code=code,
                    trade_date=d,
                    open=float(c),
                    close=float(c),
                    source="eastmoney",
                    adjust="qfq",
                )
            )
        db.commit()

        out_dca = compute_baseline(
            db,
            BaselineInputs(
                codes=[code],
                start=dates[0],
                end=dates[-1],
                benchmark_code=code,
                adjust="qfq",
                rebalance="weekly",
                holding_mode="EW",
                rolling_weeks=[],
                rolling_months=[],
                rolling_years=[],
                dca_enabled=True,
                dca_base_amount=100.0,
                dca_periodic_amount=20.0,
                dca_frequency="daily",
            ),
        )
        out_lump = compute_baseline(
            db,
            BaselineInputs(
                codes=[code],
                start=dates[0],
                end=dates[-1],
                benchmark_code=code,
                adjust="qfq",
                rebalance="weekly",
                holding_mode="EW",
                rolling_weeks=[],
                rolling_months=[],
                rolling_years=[],
                dca_enabled=False,
            ),
        )

    dca_metrics = (out_dca.get("metrics_by_portfolio") or {}).get("EW") or {}
    dca_attr = ((out_dca.get("attribution_by_portfolio") or {}).get("EW") or {}).get(
        "return", {}
    ) or {}
    dca_rows = dca_attr.get("by_code") or []
    dca_sum = sum(float(x.get("return_contribution") or 0.0) for x in dca_rows)
    assert str(dca_attr.get("method") or "") == "money_weighted_capital_curve"
    assert float(dca_attr.get("total_return") or 0.0) == pytest.approx(
        float(dca_metrics.get("dca_cumulative_return") or 0.0), rel=0.0, abs=1e-12
    )
    assert float(dca_sum) == pytest.approx(
        float(dca_metrics.get("dca_cumulative_return") or 0.0), rel=0.0, abs=1e-12
    )
    assert float(dca_metrics.get("dca_cumulative_return") or 0.0) != pytest.approx(
        float(dca_metrics.get("cumulative_return") or 0.0), rel=0.0, abs=1e-9
    )

    lump_metrics = (out_lump.get("metrics_by_portfolio") or {}).get("EW") or {}
    lump_attr = ((out_lump.get("attribution_by_portfolio") or {}).get("EW") or {}).get(
        "return", {}
    ) or {}
    assert str(lump_attr.get("method") or "") == "log_scaled"
    assert float(lump_attr.get("total_return") or 0.0) == pytest.approx(
        float(lump_metrics.get("cumulative_return") or 0.0), rel=0.0, abs=1e-12
    )
