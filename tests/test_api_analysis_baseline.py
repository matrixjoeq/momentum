import datetime as dt

import pandas as pd
import pytest

from etf_momentum.db.models import EtfPrice
from etf_momentum.db.session import make_session_factory
from etf_momentum.api.routes import _build_trend_capacity_estimate
from etf_momentum.analysis.trend import _apply_intraday_stop_execution_single
from tests.helpers.rotation_case_data import (
    build_rotation_case_series,
    fmt_ymd,
    make_bias_rule,
    make_rotation_base_payload,
    make_trend_rule,
    map_case_series_to_miniprogram_codes,
    post_json,
    post_json_ok,
    seed_prices,
)
from tests.helpers.api_test_client import upsert_and_fetch_etfs


_BASELINE_CODES = ["510300", "511010"]
_BASELINE_NAMES = {"510300": "沪深300", "511010": "国债"}


def _seed_trend_capacity_prices(
    engine,
    *,
    code_to_series: dict[str, list[float]],
    dates: list[dt.date],
    with_amount: bool,
) -> None:
    sf = make_session_factory(engine)
    with sf() as db:
        for code, series in code_to_series.items():
            for i, (d, px) in enumerate(zip(dates, series)):
                vol = float(1_000_000.0 + i * 10_000.0)
                amount = float(vol * px) if with_amount else None
                o = float(px * 0.995)
                h = float(px * 1.01)
                low_px = float(px * 0.99)
                c = float(px)
                for adj in ("none", "hfq", "qfq"):
                    db.add(
                        EtfPrice(
                            code=str(code),
                            trade_date=d,
                            open=o,
                            high=h,
                            low=low_px,
                            close=c,
                            volume=vol,
                            amount=amount,
                            source="eastmoney",
                            adjust=adj,
                        )
                    )
        db.commit()


def _make_next_execution_plan_payload(
    *,
    codes: list[str],
    start: str,
    end: str,
    asof: str,
    rebalance: str = "weekly",
    rebalance_anchor: int = 2,
    top_k: int = 1,
    lookback_days: int = 1,
    skip_days: int = 0,
    risk_off: bool = False,
    exec_price: str = "open",
) -> dict[str, object]:
    return {
        "codes": list(codes),
        "start": str(start),
        "end": str(end),
        "asof": str(asof),
        "rebalance": str(rebalance),
        "rebalance_anchor": int(rebalance_anchor),
        "top_k": int(top_k),
        "lookback_days": int(lookback_days),
        "skip_days": int(skip_days),
        "risk_off": bool(risk_off),
        "exec_price": str(exec_price),
    }


def _assert_semi_variance_stats_shape(stats: dict[str, object]) -> None:
    semi = (stats or {}).get("semi_variance") or {}
    assert isinstance(semi, dict)
    assert "win_rate_ex_zero" in semi
    assert "payoff_ex_zero" in semi
    assert "kelly_ex_zero" in semi
    assert "upside_semivariance" in semi
    assert "downside_semivariance" in semi
    for key in [
        "profit_return_stats",
        "profit_count_stats",
        "loss_return_stats",
        "loss_count_stats",
    ]:
        block = semi.get(key) or {}
        assert isinstance(block, dict)
        assert "max" in block
        assert "min" in block
        assert "mean" in block
        q = block.get("quantiles") or {}
        assert isinstance(q, dict)
        assert "p25" in q
        assert "p50" in q
        assert "p75" in q


def _assert_risk_of_ruin_stats_shape(
    stats: dict[str, object], *, expected_maxrisk: float | None = None
) -> None:
    ror = (stats or {}).get("risk_of_ruin") or {}
    assert isinstance(ror, dict)
    assert str(ror.get("formula") or "") == "((1-P)/P)^(maxrisk/A)"
    assert "probability" in ror
    assert "P" in ror
    assert "A" in ror
    assert "maxrisk" in ror
    if expected_maxrisk is not None:
        assert float(ror.get("maxrisk") or 0.0) == pytest.approx(
            float(expected_maxrisk), rel=0.0, abs=1e-12
        )
        assert str(ror.get("maxrisk_basis") or "") == "configured_risk_of_ruin_maxrisk"


def _assert_dist_stats_shape(stats: dict[str, object]) -> None:
    one = stats or {}
    assert isinstance(one, dict)
    assert "count" in one
    assert "max" in one
    assert "min" in one
    assert "mean" in one
    assert "std" in one
    q = one.get("quantiles") or {}
    assert isinstance(q, dict)
    for k in ["p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]:
        assert k in q


def _assert_trade_extreme_stats_shape(stats: dict[str, object]) -> None:
    one = stats or {}
    assert isinstance(one, dict)
    _assert_dist_stats_shape((one.get("max_possible_return_stats") or {}))
    _assert_dist_stats_shape((one.get("max_possible_return_profit_stats") or {}))
    _assert_dist_stats_shape((one.get("max_possible_return_loss_stats") or {}))
    _assert_dist_stats_shape((one.get("min_possible_return_stats") or {}))
    _assert_dist_stats_shape((one.get("min_possible_return_profit_stats") or {}))
    _assert_dist_stats_shape((one.get("min_possible_return_loss_stats") or {}))
    holding_days = one.get("holding_days_stats") or {}
    assert isinstance(holding_days, dict)
    _assert_dist_stats_shape((holding_days.get("overall") or {}))
    _assert_dist_stats_shape((holding_days.get("profit_trades") or {}))
    _assert_dist_stats_shape((holding_days.get("loss_trades") or {}))
    entry_to_max_days = one.get("entry_to_max_days_stats") or {}
    assert isinstance(entry_to_max_days, dict)
    _assert_dist_stats_shape((entry_to_max_days.get("overall") or {}))
    _assert_dist_stats_shape((entry_to_max_days.get("profit_trades") or {}))
    _assert_dist_stats_shape((entry_to_max_days.get("loss_trades") or {}))
    entry_to_min_days = one.get("entry_to_min_days_stats") or {}
    assert isinstance(entry_to_min_days, dict)
    _assert_dist_stats_shape((entry_to_min_days.get("overall") or {}))
    _assert_dist_stats_shape((entry_to_min_days.get("profit_trades") or {}))
    _assert_dist_stats_shape((entry_to_min_days.get("loss_trades") or {}))
    max_to_exit_days = one.get("max_to_exit_days_stats") or {}
    assert isinstance(max_to_exit_days, dict)
    _assert_dist_stats_shape((max_to_exit_days.get("overall") or {}))
    _assert_dist_stats_shape((max_to_exit_days.get("profit_trades") or {}))
    _assert_dist_stats_shape((max_to_exit_days.get("loss_trades") or {}))
    min_to_exit_days = one.get("min_to_exit_days_stats") or {}
    assert isinstance(min_to_exit_days, dict)
    _assert_dist_stats_shape((min_to_exit_days.get("overall") or {}))
    _assert_dist_stats_shape((min_to_exit_days.get("profit_trades") or {}))
    _assert_dist_stats_shape((min_to_exit_days.get("loss_trades") or {}))
    # Signed semantics: losing-trade max_possible_return should be non-positive.
    loss_n = int(one.get("loss_trades") or 0)
    if loss_n > 0:
        loss_max_stats = one.get("max_possible_return_loss_stats") or {}
        loss_max_v = loss_max_stats.get("max")
        loss_min_v = loss_max_stats.get("min")
        if loss_max_v is not None:
            assert float(loss_max_v) <= 1e-12
        if loss_min_v is not None:
            assert float(loss_min_v) <= 1e-12


def test_api_baseline_analysis_happy_path(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240103",
    )

    data = post_json_ok(
        c,
        "/api/analysis/baseline",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240103",
            "benchmark_code": "510300",
            "adjust": "hfq",
            "rebalance": "yearly",
            "risk_free_rate": 0.02,
            "rolling_weeks": [1],
            "rolling_months": [],
            "rolling_years": [],
            "fft_windows": [20, 10],
        },
    )
    assert data["metrics"]["benchmark_code"] == "510300"
    assert data["metrics"]["rebalance"] == "yearly"
    assert "risk_free_rate" not in data["metrics"]
    assert "ulcer_index" in data["metrics"]
    assert "ulcer_performance_index" in data["metrics"]
    assert "holding_weekly_win_rate" in data["metrics"]
    assert "holding_monthly_payoff_ratio" in data["metrics"]
    assert "holding_yearly_kelly_fraction" in data["metrics"]
    assert "EW" in data["nav"]["series"]
    assert "RP" in data["nav"]["series"]
    assert "IVOL" in data["nav"]["series"]
    assert "510300" in data["nav"]["series"]
    assert "511010" in data["nav"]["series"]
    assert any(k.startswith("BENCH:") for k in data["nav"]["series"].keys())
    assert "quarterly" in data["period_returns"]
    assert "correlation" in data
    assert data["correlation"]["method"] == "pearson_log_return"
    assert data["correlation"]["codes"] == ["510300", "511010"]
    assert len(data["correlation"]["matrix"]) == 2
    assert "fft" in data
    assert data["fft"]["windows"] == [20, 10]
    assert "fft_roll" in data
    assert "ew" in data["fft_roll"]
    assert data["fft_roll"]["ew"]["windows"] == [20, 10]
    assert data["fft_roll"]["ew"]["step"] == 5
    assert "nav_rsi" in data
    assert data["nav_rsi"]["windows"] == [14]


def test_api_baseline_analysis_accepts_dca_payload(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240112",
    )
    data = post_json_ok(
        c,
        "/api/analysis/baseline",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240112",
            "benchmark_code": "510300",
            "adjust": "hfq",
            "rebalance": "weekly",
            "exec_price": "close",
            "holding_mode": "EW",
            "dca_enabled": True,
            "dca_base_amount": 100000.0,
            "dca_periodic_amount": 20000.0,
            "dca_frequency": "weekly",
        },
    )
    data_off = post_json_ok(
        c,
        "/api/analysis/baseline",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240112",
            "benchmark_code": "510300",
            "adjust": "hfq",
            "rebalance": "weekly",
            "exec_price": "close",
            "holding_mode": "EW",
            "dca_enabled": False,
        },
    )
    m = data.get("metrics") or {}
    m_off = data_off.get("metrics") or {}
    assert bool(m.get("dca_enabled")) is True
    assert str(m.get("dca_frequency") or "") == "weekly"
    assert m.get("dca_total_invested") is not None
    assert m.get("dca_final_value") is not None
    assert m.get("dca_cumulative_return") is not None
    assert m.get("dca_money_weighted_return") is not None
    assert m.get("dca_time_weighted_return") is not None
    assert "dca" in data
    assert "dca_by_portfolio" in data
    assert "EW" in (data.get("dca_by_portfolio") or {})
    assert float(m.get("avg_annual_turnover") or 0.0) > float(
        m_off.get("avg_annual_turnover") or 0.0
    )


def test_api_baseline_analysis_lppl_contract(api_client):
    c = api_client
    engine = c.app.state.engine
    code = "LPPL1"
    dates = [dt.date(2023, 1, 1) + dt.timedelta(days=i) for i in range(260)]
    series = {
        code: [100.0 + 0.2 * i + (1.3 if (i % 11) < 5 else -0.7) for i in range(260)]
    }
    seed_prices(engine, code_to_series=series, dates=dates)
    data = post_json_ok(
        c,
        "/api/analysis/baseline",
        {
            "codes": [code],
            "start": fmt_ymd(dates[0]),
            "end": fmt_ymd(dates[-1]),
            "benchmark_code": code,
            "adjust": "qfq",
            "rebalance": "yearly",
            "lppl_enabled": True,
            "lppl_lookback_days": 220,
            "lppl_min_points": 80,
            "lppl_horizon_days": 120,
            "lppl_multistart": 12,
            "lppl_start_mode": "fixed_lookback",
            "lppl_bootstrap_on": True,
            "lppl_bootstrap_reps": 8,
            "lppl_bootstrap_block_size": 4,
        },
    )
    pdist = data["period_distributions"][code]
    assert "daily_lppl" in pdist
    lppl = pdist["daily_lppl"]
    assert "status" in lppl
    assert lppl["status"] in {
        "ok",
        "fit_rejected",
        "fit_failed",
        "library_unavailable",
        "insufficient_data",
    }
    if lppl["status"] in {"ok", "fit_rejected"}:
        assert "params" in lppl
        assert "tc_distribution" in lppl
        assert "tc_horizon_prob" in lppl


def test_api_baseline_analysis_lppl_invalid_bootstrap_block(api_client):
    c = api_client
    payload = {
        "codes": ["510300"],
        "start": "20240102",
        "end": "20240103",
        "benchmark_code": "510300",
        "adjust": "hfq",
        "rebalance": "yearly",
        "lppl_enabled": True,
        "lppl_bootstrap_block_size": 0,
    }
    err = post_json(c, "/api/analysis/baseline", payload, expected_status=422)
    assert isinstance(err, dict)
    detail = err.get("detail") or []
    assert isinstance(detail, list)
    joined = str(detail)
    assert "lppl_bootstrap_block_size" in joined


def test_api_rotation_backtest_happy_path(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240103",
    )

    data = post_json_ok(
        c,
        "/api/analysis/rotation",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240103",
            "rebalance": "monthly",
            "top_k": 1,
            "lookback_days": 1,
            "skip_days": 0,
            "risk_off": False,
            "risk_free_rate": 0.025,
            "cost_bps": 0.0,
        },
    )
    assert "nav" in data and "series" in data["nav"]
    assert "ROTATION" in data["nav"]["series"]
    assert "EW_REBAL" in data["nav"]["series"]
    assert "EXCESS" in data["nav"]["series"]
    assert "RP_REBAL" not in data["nav"]["series"]
    assert "IVOL_REBAL" not in data["nav"]["series"]
    assert "EXCESS_RP" not in data["nav"]["series"]
    assert "EXCESS_IVOL" not in data["nav"]["series"]
    assert (data.get("period_returns_ivol") or {}) == {}
    assert "excess_vs_inverse_vol_rebal" not in (data.get("metrics") or {})
    assert "nav_rsi" in data
    assert data["nav_rsi"]["windows"] == [14]
    assert "win_payoff" in data
    assert "current_holdings" in data
    assert isinstance(data.get("current_holdings"), list)

    data_rp = post_json_ok(
        c,
        "/api/analysis/rotation",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240103",
            "rebalance": "monthly",
            "top_k": 1,
            "lookback_days": 1,
            "skip_days": 0,
            "risk_off": False,
            "risk_free_rate": 0.025,
            "cost_bps": 0.0,
            "benchmark_mode": "RP_REBAL",
        },
    )
    assert "RP_REBAL" in (data_rp.get("nav") or {}).get("series", {})
    assert "EXCESS_RP" in (data_rp.get("nav") or {}).get("series", {})
    assert "excess_vs_risk_parity" in (data_rp.get("metrics") or {})


def test_api_rotation_capacity_estimate_has_three_scenarios(engine, api_client):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=140, freq="B")]
    # Construct alternating momentum leadership to create non-zero turnover windows.
    s1 = [100.0 + ((i % 20) - 10) * 0.8 + i * 0.03 for i in range(len(dates))]
    s2 = [100.0 - ((i % 20) - 10) * 0.8 + i * 0.03 for i in range(len(dates))]
    _seed_trend_capacity_prices(
        engine,
        code_to_series={"RCAP1": s1, "RCAP2": s2},
        dates=dates,
        with_amount=False,
    )
    c = api_client
    out = post_json_ok(
        c,
        "/api/analysis/rotation",
        {
            "codes": ["RCAP1", "RCAP2"],
            "start": fmt_ymd(dates[0]),
            "end": fmt_ymd(dates[-1]),
            "rebalance": "weekly",
            "rebalance_anchor": 5,
            "top_k": 1,
            "lookback_days": 5,
            "skip_days": 0,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    cap = (out or {}).get("capacity_estimate") or {}
    meta = cap.get("meta") or {}
    scenarios = cap.get("scenarios") or []
    assert str(cap.get("method") or "") == "asset_participation_bottleneck_daily"
    assert str(meta.get("status") or "") == "ok"
    assert len(scenarios) == 3
    by_name = {str((x or {}).get("name") or ""): x for x in scenarios}
    assert set(by_name.keys()) == {"conservative", "balanced", "aggressive"}
    assert float(
        (by_name["balanced"] or {}).get("participation_rate") or 0.0
    ) == pytest.approx(0.10, rel=0.0, abs=1e-12)
    src = (meta.get("source_stats") or {}).get("sources") or {}
    assert int(src.get("volume_avg_price") or 0) > 0


def test_api_rotation_weekly5_open_contains_capacity_estimate(api_client, engine):
    dates, src = build_rotation_case_series()
    mapped = map_case_series_to_miniprogram_codes(src)
    seed_prices(engine, code_to_series=mapped, dates=dates)
    c = api_client
    out = post_json_ok(
        c,
        "/api/analysis/rotation/weekly5-open",
        {"start": "20240102", "end": "20240731", "anchor_weekday": 5},
    )
    one = ((out or {}).get("by_anchor") or {}).get("5") or {}
    cap = one.get("capacity_estimate") or {}
    assert str(cap.get("method") or "") == "asset_participation_bottleneck_daily"
    assert "meta" in cap
    assert "scenarios" in cap


def test_api_rotation_backtest_accepts_negative_top_k(api_client) -> None:
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240103",
    )
    data = post_json_ok(
        c,
        "/api/analysis/rotation",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240103",
            "rebalance": "monthly",
            "top_k": -1,
            "lookback_days": 1,
            "skip_days": 0,
            "risk_off": False,
            "risk_free_rate": 0.025,
            "cost_bps": 0.0,
        },
    )
    assert "nav" in data and "ROTATION" in data["nav"]["series"]


def test_api_rotation_backtest_accepts_inverse_vol_position_mode(api_client) -> None:
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20230102",
        end_date="20240131",
    )
    data = post_json_ok(
        c,
        "/api/analysis/rotation",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240131",
            "rebalance": "weekly",
            "top_k": 2,
            "position_mode": "inverse_vol",
            "vol_window": 10,
            "lookback_days": 5,
            "skip_days": 0,
            "risk_free_rate": 0.025,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    assert str(data.get("position_mode") or "") == "inverse_vol"
    assert "nav" in data and "ROTATION" in (data.get("nav") or {}).get("series", {})


def test_api_rotation_backtest_accepts_risk_budget_pct_3_percent(api_client) -> None:
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240110",
    )
    data = post_json_ok(
        c,
        "/api/analysis/rotation",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240110",
            "rebalance": "weekly",
            "top_k": 1,
            "position_mode": "risk_budget",
            "risk_budget_atr_window": 2,
            "risk_budget_pct": 0.03,
            "lookback_days": 2,
            "skip_days": 0,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    assert str(data.get("position_mode") or "") == "risk_budget"
    assert "nav" in data and "ROTATION" in (data.get("nav") or {}).get("series", {})


def test_api_rotation_backtest_rejects_risk_budget_pct_above_3_percent(
    api_client,
) -> None:
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240110",
    )
    err = post_json(
        c,
        "/api/analysis/rotation",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240110",
            "rebalance": "weekly",
            "top_k": 1,
            "position_mode": "risk_budget",
            "risk_budget_atr_window": 2,
            "risk_budget_pct": 0.031,
            "lookback_days": 2,
            "skip_days": 0,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
        expected_status=422,
    )
    assert "0.03" in str(err)


def test_api_rotation_backtest_daily_rebalance_switch(api_client) -> None:
    c = api_client
    engine = c.app.state.engine
    codes = ["RDA", "RDB"]
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=90, freq="B")]
    px_a = 100.0
    px_b = 100.0
    series_a: list[float] = []
    series_b: list[float] = []
    for i, _ in enumerate(dates):
        px_a *= 1.01 if (i % 2 == 0) else 0.99
        px_b *= 0.99 if (i % 2 == 0) else 1.01
        series_a.append(float(px_a))
        series_b.append(float(px_b))
    seed_prices(engine, code_to_series={"RDA": series_a, "RDB": series_b}, dates=dates)
    upsert_and_fetch_etfs(
        c,
        codes=codes,
        names={codes[0]: codes[0], codes[1]: codes[1]},
        start_date=fmt_ymd(dates[0]),
        end_date=fmt_ymd(dates[-1]),
    )
    base = {
        "codes": codes,
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "rebalance": "weekly",
        "rebalance_anchor": 1,
        "top_k": 2,
        "position_mode": "adaptive",
        "lookback_days": 5,
        "skip_days": 0,
        "exec_price": "close",
        "cost_bps": 100.0,
        "slippage_rate": 0.0,
    }
    out_off = post_json_ok(
        c, "/api/analysis/rotation", {**base, "daily_rebalance": False}
    )
    out_on = post_json_ok(
        c, "/api/analysis/rotation", {**base, "daily_rebalance": True}
    )
    m_off = (out_off.get("metrics") or {}).get("strategy") or {}
    m_on = (out_on.get("metrics") or {}).get("strategy") or {}
    assert bool(out_off.get("daily_rebalance")) is False
    assert bool(out_on.get("daily_rebalance")) is True
    assert float(m_on.get("avg_daily_turnover") or 0.0) > float(
        m_off.get("avg_daily_turnover") or 0.0
    )


def test_api_rotation_backtest_rejects_zero_top_k(api_client) -> None:
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240103",
    )
    err = post_json(
        c,
        "/api/analysis/rotation",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240103",
            "rebalance": "monthly",
            "top_k": 0,
            "lookback_days": 1,
            "skip_days": 0,
            "risk_off": False,
            "risk_free_rate": 0.025,
            "cost_bps": 0.0,
        },
        expected_status=422,
    )
    assert isinstance(err, dict)


def test_api_calendar_timing_accepts_risk_budget_pct_3_percent(
    engine, api_client
) -> None:
    c = api_client
    dates = [d.date() for d in pd.date_range("2024-01-02", periods=60, freq="B")]
    seed_prices(
        engine,
        code_to_series={"CTRB1": [100.0 + i * 0.3 for i, _ in enumerate(dates)]},
        dates=dates,
    )
    out = post_json_ok(
        c,
        "/api/analysis/calendar-timing",
        {
            "mode": "single",
            "code": "CTRB1",
            "start": fmt_ymd(dates[0]),
            "end": fmt_ymd(dates[-1]),
            "decision_day": 1,
            "hold_days": 1,
            "position_mode": "risk_budget",
            "risk_budget_atr_window": 2,
            "risk_budget_pct": 0.03,
            "exec_price": "open",
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
            "rebalance_shift": "prev",
        },
    )
    meta = out.get("meta") or {}
    assert str(meta.get("position_mode") or "") == "risk_budget"
    assert float(meta.get("risk_budget_pct") or 0.0) == pytest.approx(0.03)


def test_api_calendar_timing_rejects_risk_budget_pct_above_3_percent(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240110",
    )
    err = post_json(
        c,
        "/api/analysis/calendar-timing",
        {
            "mode": "single",
            "code": "510300",
            "start": "20240102",
            "end": "20240110",
            "decision_day": 1,
            "hold_days": 1,
            "position_mode": "risk_budget",
            "risk_budget_atr_window": 2,
            "risk_budget_pct": 0.031,
            "exec_price": "open",
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
            "rebalance_shift": "prev",
        },
        expected_status=422,
    )
    assert "0.03" in str(err)


def test_api_trend_single_accepts_risk_budget_params(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240103",
    )
    data = post_json_ok(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "sma_window": 2,
            "position_sizing": "risk_budget",
            "risk_budget_atr_window": 2,
            "risk_budget_pct": 0.01,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    params = ((data or {}).get("meta") or {}).get("params") or {}
    assert str(params.get("position_sizing") or "") == "risk_budget"
    assert int(params.get("risk_budget_atr_window") or 0) == 2
    assert float(params.get("risk_budget_pct") or 0.0) == pytest.approx(
        0.01, rel=0.0, abs=1e-12
    )


def test_api_trend_single_accepts_stop_execution_modes(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240103",
    )
    data = post_json_ok(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "sma_window": 2,
            "atr_stop_mode": "static",
            "atr_stop_window": 2,
            "atr_stop_n": 1.0,
            "atr_stop_execution_mode": "next_day",
            "atr_stop_execution_time": "open",
            "r_take_profit_enabled": True,
            "r_take_profit_tiers": [{"r_multiple": 1.5, "retrace_ratio": 0.5}],
            "r_take_profit_execution_mode": "next_day",
            "r_take_profit_execution_time": "close",
            "r_profit_scaleout_enabled": True,
            "r_profit_scaleout_tiers": [{"r_multiple": 1.5, "reduce_fraction": 0.4}],
            "r_profit_scaleout_execution_mode": "intraday",
            "r_profit_scaleout_execution_time": "full_day",
            "bias_v_take_profit_enabled": True,
            "bias_v_ma_window": 2,
            "bias_v_atr_window": 2,
            "bias_v_take_profit_tiers": [
                {"threshold": 0.5, "reduce_fraction": 0.6},
                {"threshold": 0.8, "reduce_fraction": 0.4},
            ],
            "bias_v_take_profit_execution_mode": "next_day",
            "bias_v_take_profit_execution_time": "open",
            "ma_trailing_stop_enabled": True,
            "ma_trailing_stop_execution_mode": "intraday",
            "ma_trailing_stop_execution_time": "full_day",
            "ma_trailing_stop_reduce_window": 2,
            "ma_trailing_stop_exit_window": 2,
            "ma_trailing_stop_effective_delay_days": 1,
            "ma_trailing_stop_reduce_fraction": 0.5,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    params = ((data or {}).get("meta") or {}).get("params") or {}
    assert str(params.get("atr_stop_execution_mode") or "") == "next_day"
    assert str(params.get("atr_stop_execution_time") or "") == "open"
    assert str(params.get("r_take_profit_execution_mode") or "") == "next_day"
    assert str(params.get("r_take_profit_execution_time") or "") == "close"
    assert str(params.get("r_profit_scaleout_execution_mode") or "") == "intraday"
    assert str(params.get("r_profit_scaleout_execution_time") or "") == "full_day"
    assert str(params.get("bias_v_take_profit_execution_mode") or "") == "next_day"
    assert str(params.get("bias_v_take_profit_execution_time") or "") == "open"
    assert str(params.get("ma_trailing_stop_execution_mode") or "") == "intraday"
    assert str(params.get("ma_trailing_stop_execution_time") or "") == "full_day"
    assert "bias_v_take_profit_threshold" not in params
    tiers = list(params.get("bias_v_take_profit_tiers") or [])
    assert tiers == [
        {"threshold": 0.5, "reduce_fraction": 0.6},
        {"threshold": 0.8, "reduce_fraction": 0.4},
    ]
    rc = ((data or {}).get("risk_controls") or {}).get("bias_v_take_profit") or {}
    assert "threshold" not in rc


def test_api_trend_portfolio_accepts_stop_execution_modes(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240103",
    )
    data = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": _BASELINE_CODES,
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "sma_window": 2,
            "atr_stop_mode": "static",
            "atr_stop_window": 2,
            "atr_stop_n": 1.0,
            "atr_stop_execution_mode": "next_day",
            "atr_stop_execution_time": "open",
            "r_take_profit_enabled": True,
            "r_take_profit_tiers": [{"r_multiple": 1.5, "retrace_ratio": 0.5}],
            "r_take_profit_execution_mode": "next_day",
            "r_take_profit_execution_time": "close",
            "r_profit_scaleout_enabled": True,
            "r_profit_scaleout_tiers": [{"r_multiple": 1.5, "reduce_fraction": 0.4}],
            "r_profit_scaleout_execution_mode": "intraday",
            "r_profit_scaleout_execution_time": "full_day",
            "bias_v_take_profit_enabled": True,
            "bias_v_ma_window": 2,
            "bias_v_atr_window": 2,
            "bias_v_take_profit_tiers": [
                {"threshold": 0.5, "reduce_fraction": 0.6},
                {"threshold": 0.8, "reduce_fraction": 0.4},
            ],
            "bias_v_take_profit_execution_mode": "next_day",
            "bias_v_take_profit_execution_time": "open",
            "ma_trailing_stop_enabled": True,
            "ma_trailing_stop_execution_mode": "intraday",
            "ma_trailing_stop_execution_time": "full_day",
            "ma_trailing_stop_reduce_window": 2,
            "ma_trailing_stop_exit_window": 2,
            "ma_trailing_stop_effective_delay_days": 1,
            "ma_trailing_stop_reduce_fraction": 0.5,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    params = ((data or {}).get("meta") or {}).get("params") or {}
    assert str(params.get("atr_stop_execution_mode") or "") == "next_day"
    assert str(params.get("atr_stop_execution_time") or "") == "open"
    assert str(params.get("r_take_profit_execution_mode") or "") == "next_day"
    assert str(params.get("r_take_profit_execution_time") or "") == "close"
    assert str(params.get("r_profit_scaleout_execution_mode") or "") == "intraday"
    assert str(params.get("r_profit_scaleout_execution_time") or "") == "full_day"
    assert str(params.get("bias_v_take_profit_execution_mode") or "") == "next_day"
    assert str(params.get("bias_v_take_profit_execution_time") or "") == "open"
    assert str(params.get("ma_trailing_stop_execution_mode") or "") == "intraday"
    assert str(params.get("ma_trailing_stop_execution_time") or "") == "full_day"
    assert "bias_v_take_profit_threshold" not in params
    tiers = list(params.get("bias_v_take_profit_tiers") or [])
    assert tiers == [
        {"threshold": 0.5, "reduce_fraction": 0.6},
        {"threshold": 0.8, "reduce_fraction": 0.4},
    ]
    rc = ((data or {}).get("risk_controls") or {}).get("bias_v_take_profit") or {}
    assert "threshold" not in rc


def test_api_trend_single_rejects_oc2_exec_price(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240103",
    )
    err = post_json(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "sma_window": 2,
            "exec_price": "oc2",
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
        expected_status=422,
    )
    assert "open" in str(err) and "close" in str(err)


def test_api_trend_single_rejects_next_day_full_day_execution_time(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240103",
    )
    err = post_json(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "sma_window": 2,
            "atr_stop_mode": "static",
            "atr_stop_window": 2,
            "atr_stop_n": 1.0,
            "atr_stop_execution_mode": "next_day",
            "atr_stop_execution_time": "full_day",
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
        expected_status=422,
    )
    assert "full_day" in str(err) and "next_day" in str(err)


def test_api_trend_portfolio_rejects_oc2_exec_price(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240103",
    )
    err = post_json(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": _BASELINE_CODES,
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "sma_window": 2,
            "exec_price": "oc2",
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
        expected_status=422,
    )
    assert "open" in str(err) and "close" in str(err)


def test_api_trend_portfolio_rejects_next_day_full_day_execution_time(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240103",
    )
    err = post_json(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": _BASELINE_CODES,
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "sma_window": 2,
            "atr_stop_mode": "static",
            "atr_stop_window": 2,
            "atr_stop_n": 1.0,
            "atr_stop_execution_mode": "next_day",
            "atr_stop_execution_time": "full_day",
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
        expected_status=422,
    )
    assert "full_day" in str(err) and "next_day" in str(err)


def test_api_trend_single_rejects_risk_budget_pct_above_3_percent(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240103",
    )
    err = post_json(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "sma_window": 2,
            "position_sizing": "risk_budget",
            "risk_budget_atr_window": 2,
            "risk_budget_pct": 0.031,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
        expected_status=422,
    )
    assert "0.03" in str(err)


def test_api_trend_single_accepts_risk_budget_pct_3_percent(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240103",
    )
    out = post_json_ok(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "sma_window": 2,
            "position_sizing": "risk_budget",
            "risk_budget_atr_window": 2,
            "risk_budget_pct": 0.03,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    params = ((out.get("meta") or {}).get("params")) or {}
    assert float(params.get("risk_budget_pct") or 0.0) == pytest.approx(0.03)


def test_api_trend_single_quick_mode_skips_heavy_sections(engine, api_client):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=90, freq="B")]
    seed_prices(
        engine,
        code_to_series={"TQMS1": [100.0 + i * 0.4 for i, _ in enumerate(dates)]},
        dates=dates,
    )
    c = api_client
    out = post_json_ok(
        c,
        "/api/analysis/trend",
        {
            "code": "TQMS1",
            "start": fmt_ymd(dates[0]),
            "end": fmt_ymd(dates[-1]),
            "strategy": "ma_filter",
            "sma_window": 2,
            "quick_mode": True,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    params = ((out or {}).get("meta") or {}).get("params") or {}
    ts = out.get("trade_statistics") or {}
    assert params.get("quick_mode") is True
    assert out.get("return_decomposition") is None
    assert out.get("event_study") is None
    overall_n = int((ts.get("overall") or {}).get("n") or 0)
    trs = list(ts.get("trades") or [])
    if overall_n > 0:
        assert len(trs) == overall_n
        assert (
            len(list(((ts.get("trades_by_code") or {}).get("TQMS1")) or []))
            == overall_n
        )
    assert "entry_condition_stats" not in ts


def test_api_trend_portfolio_quick_mode_skips_heavy_sections(engine, api_client):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=90, freq="B")]
    series = {
        "TQMP1": [100.0 + i * 0.5 for i, _ in enumerate(dates)],
        "TQMP2": [95.0 + i * 0.45 for i, _ in enumerate(dates)],
    }
    seed_prices(engine, code_to_series=series, dates=dates)
    c = api_client
    out = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": ["TQMP1", "TQMP2"],
            "start": fmt_ymd(dates[0]),
            "end": fmt_ymd(dates[-1]),
            "strategy": "ma_filter",
            "sma_window": 2,
            "position_sizing": "equal",
            "quick_mode": True,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    params = ((out or {}).get("meta") or {}).get("params") or {}
    ts = out.get("trade_statistics") or {}
    trades_by_code = ts.get("trades_by_code") or {}
    assert params.get("quick_mode") is True
    assert out.get("return_decomposition") is None
    assert out.get("event_study") is None
    overall_n = int((ts.get("overall") or {}).get("n") or 0)
    trs = list(ts.get("trades") or [])
    if overall_n > 0:
        assert len(trs) == overall_n
        assert (
            len(list(trades_by_code.get("TQMP1") or []))
            + len(list(trades_by_code.get("TQMP2") or []))
            == overall_n
        )
    assert "entry_condition_stats" not in ts


def test_api_trend_single_capacity_estimate_uses_volume_avg_price_fallback(
    engine, api_client
):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=120, freq="B")]
    _seed_trend_capacity_prices(
        engine,
        code_to_series={"CAPS1": [100.0 + i * 0.3 for i in range(len(dates))]},
        dates=dates,
        with_amount=False,
    )
    c = api_client
    out = post_json_ok(
        c,
        "/api/analysis/trend",
        {
            "code": "CAPS1",
            "start": fmt_ymd(dates[0]),
            "end": fmt_ymd(dates[-1]),
            "strategy": "ma_filter",
            "sma_window": 3,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    cap = (out or {}).get("capacity_estimate") or {}
    meta = cap.get("meta") or {}
    scenarios = cap.get("scenarios") or []
    assert str(meta.get("status") or "") == "ok"
    assert len(scenarios) == 3
    names = [str((x or {}).get("name") or "") for x in scenarios]
    assert names == ["conservative", "balanced", "aggressive"]
    assert all(int((x or {}).get("sample_days") or 0) >= 0 for x in scenarios)
    src_stats = meta.get("source_stats") or {}
    src = src_stats.get("sources") or {}
    assert int(src.get("volume_avg_price") or 0) > 0


def test_api_trend_portfolio_capacity_estimate_has_three_scenarios(engine, api_client):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=120, freq="B")]
    _seed_trend_capacity_prices(
        engine,
        code_to_series={
            "CAPP1": [100.0 + i * 0.25 for i in range(len(dates))],
            "CAPP2": [90.0 + i * 0.20 for i in range(len(dates))],
        },
        dates=dates,
        with_amount=True,
    )
    c = api_client
    out = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": ["CAPP1", "CAPP2"],
            "start": fmt_ymd(dates[0]),
            "end": fmt_ymd(dates[-1]),
            "strategy": "ma_filter",
            "sma_window": 3,
            "position_sizing": "equal",
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    cap = (out or {}).get("capacity_estimate") or {}
    scenarios = cap.get("scenarios") or []
    meta = cap.get("meta") or {}
    assert len(scenarios) == 3
    by_name = {str((x or {}).get("name") or ""): x for x in scenarios}
    assert set(by_name.keys()) == {"conservative", "balanced", "aggressive"}
    assert float(
        (by_name["balanced"] or {}).get("participation_rate") or 0.0
    ) == pytest.approx(0.10, rel=0.0, abs=1e-12)
    assert int(meta.get("turnover_positive_days") or 0) > 0
    assert float((by_name["aggressive"] or {}).get("aum_cap_p25") or 0.0) >= float(
        (by_name["conservative"] or {}).get("aum_cap_p25") or 0.0
    )


def test_capacity_estimate_counts_full_cash_to_risk_one_way_turnover(engine):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=3, freq="B")]
    _seed_trend_capacity_prices(
        engine,
        code_to_series={"CAPF1": [100.0, 101.0, 102.0]},
        dates=dates,
        with_amount=True,
    )
    out = {
        "nav": {"dates": [d.strftime("%Y-%m-%d") for d in dates]},
        "weights": {"series": {"CAPF1": [0.0, 1.0, 1.0]}},
    }
    sf = make_session_factory(engine)
    with sf() as db:
        cap = _build_trend_capacity_estimate(db, out)
    by_name = {
        str((x or {}).get("name") or ""): x for x in (cap.get("scenarios") or [])
    }
    assert int((by_name.get("balanced") or {}).get("sample_days") or 0) == 1
    series_bal = (
        ((cap.get("series") or {}).get("by_scenario") or {}).get("balanced")
    ) or {}
    turnover = list(series_bal.get("daily_turnover_one_way") or [])
    liq = list(series_bal.get("daily_liquidity_notional") or [])
    aum = list(series_bal.get("daily_aum_cap") or [])
    assert float(turnover[1]) == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert float(aum[1]) == pytest.approx(float(liq[1]) * 0.10, rel=0.0, abs=1e-8)


def test_capacity_estimate_excludes_untraded_liquidity_from_daily_capacity(engine):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=3, freq="B")]
    _seed_trend_capacity_prices(
        engine,
        code_to_series={
            "CAPL1": [10.0, 10.5, 10.5],
            "CAPL2": [1000.0, 1005.0, 1005.0],
        },
        dates=dates,
        with_amount=True,
    )
    out = {
        "nav": {"dates": [d.strftime("%Y-%m-%d") for d in dates]},
        "weights": {"series": {"CAPL1": [0.2, 0.4, 0.4], "CAPL2": [0.3, 0.3, 0.3]}},
    }
    sf = make_session_factory(engine)
    with sf() as db:
        cap = _build_trend_capacity_estimate(db, out)
    series_bal = (
        ((cap.get("series") or {}).get("by_scenario") or {}).get("balanced")
    ) or {}
    liq_traded = list(series_bal.get("daily_liquidity_notional") or [])
    liq_all = list(series_bal.get("daily_liquidity_notional_all") or [])
    turnover = list(series_bal.get("daily_turnover_one_way") or [])
    aum = list(series_bal.get("daily_aum_cap") or [])
    # Day-2 only CAPL1 trades; CAPL2's high liquidity should not inflate capacity.
    assert float(liq_all[1]) > float(liq_traded[1]) * 50.0
    expected = float(liq_traded[1]) * 0.10 / float(turnover[1])
    assert float(aum[1]) == pytest.approx(expected, rel=0.0, abs=1e-8)


def test_capacity_estimate_drops_days_with_missing_liquidity_on_traded_assets(engine):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=2, freq="B")]
    sf = make_session_factory(engine)
    with sf() as db:
        # Day-1 has liquidity; day-2 intentionally missing amount and volume.
        db.add(
            EtfPrice(
                code="CAPM1",
                trade_date=dates[0],
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1_000_000.0,
                amount=100_000_000.0,
                source="eastmoney",
                adjust="none",
            )
        )
        db.add(
            EtfPrice(
                code="CAPM1",
                trade_date=dates[1],
                open=101.0,
                high=102.0,
                low=100.0,
                close=101.0,
                volume=0.0,
                amount=None,
                source="eastmoney",
                adjust="none",
            )
        )
        db.commit()
    out = {
        "nav": {"dates": [d.strftime("%Y-%m-%d") for d in dates]},
        "weights": {"series": {"CAPM1": [0.0, 1.0]}},
    }
    with sf() as db:
        cap = _build_trend_capacity_estimate(db, out)
    meta = cap.get("meta") or {}
    by_name = {
        str((x or {}).get("name") or ""): x for x in (cap.get("scenarios") or [])
    }
    assert int(meta.get("invalid_traded_liquidity_days") or 0) == 1
    assert int((by_name.get("balanced") or {}).get("sample_days") or 0) == 0


def test_capacity_estimate_tencent_volume_fallback_uses_lot_multiplier(engine):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=3, freq="B")]
    sf = make_session_factory(engine)
    with sf() as db:
        for d in dates:
            db.add(
                EtfPrice(
                    code="TCAP1",
                    trade_date=d,
                    open=10.0,
                    high=10.0,
                    low=10.0,
                    close=10.0,
                    volume=1000.0,  # tencent volume is in lots(手)
                    amount=None,
                    source="tencent",
                    adjust="none",
                )
            )
        db.commit()
    out = {
        "nav": {"dates": [d.strftime("%Y-%m-%d") for d in dates]},
        "weights": {"series": {"TCAP1": [0.0, 1.0, 1.0]}},
    }
    with sf() as db:
        cap = _build_trend_capacity_estimate(db, out)
    series_bal = (
        ((cap.get("series") or {}).get("by_scenario") or {}).get("balanced")
    ) or {}
    liq = list(series_bal.get("daily_liquidity_notional") or [])
    aum = list(series_bal.get("daily_aum_cap") or [])
    # volume*price fallback should convert lot volume into shares by *100.
    assert float(liq[1]) == pytest.approx(1_000_000.0, rel=0.0, abs=1e-8)
    assert float(aum[1]) == pytest.approx(100_000.0, rel=0.0, abs=1e-8)


@pytest.mark.parametrize("runtime_engine", [None, "bt"])
def test_api_trend_account_lot_sizing_enforces_100_share_lots(
    runtime_engine, engine, api_client
):
    dates = [d.date() for d in pd.date_range("2021-01-01", periods=120, freq="B")]
    _seed_trend_capacity_prices(
        engine,
        code_to_series={
            "LOT1": [20.0 + i * 0.05 for i in range(len(dates))],
            "LOT2": [15.0 + i * 0.04 for i in range(len(dates))],
        },
        dates=dates,
        with_amount=True,
    )
    c = api_client
    payload = {
        "codes": ["LOT1", "LOT2"],
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 3,
        "position_sizing": "equal",
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
    }
    if runtime_engine:
        payload["engine"] = runtime_engine
    out_plain = post_json_ok(c, "/api/analysis/trend/portfolio", dict(payload))
    out = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {**payload, "initial_account_amount": 1_000_000},
    )
    meta = (out.get("meta") or {}).get("account_lot_sizing") or {}
    assert bool(meta.get("enabled")) is True
    assert int(meta.get("lot_size_shares") or 0) == 100
    shares_by_code = meta.get("shares_by_code") or {}
    assert isinstance(shares_by_code, dict) and shares_by_code
    for arr in shares_by_code.values():
        for v in list(arr or []):
            assert int(v) % 100 == 0
    strat_with = ((out or {}).get("metrics") or {}).get("strategy") or {}
    strat_plain = ((out_plain or {}).get("metrics") or {}).get("strategy") or {}
    cum_with = float(strat_with.get("cumulative_return") or 0.0)
    cum_plain = float(strat_plain.get("cumulative_return") or 0.0)
    if abs(cum_plain) > 1e-9:
        rel_gap = abs(cum_with - cum_plain) / abs(cum_plain)
        assert rel_gap < 0.10
    else:
        assert abs(cum_with - cum_plain) < 0.02


@pytest.mark.parametrize("runtime_engine", [None, "bt"])
def test_api_trend_scaleout_zero_trigger_is_noop(runtime_engine, engine, api_client):
    dates = [d.date() for d in pd.date_range("2021-01-01", periods=220, freq="B")]
    _seed_trend_capacity_prices(
        engine,
        code_to_series={
            "SCL1": [
                100.0 + i * 0.06 + ((i % 9) - 4) * 0.08 for i in range(len(dates))
            ],
            "SCL2": [
                90.0 + i * 0.05 + ((i % 11) - 5) * 0.07 for i in range(len(dates))
            ],
        },
        dates=dates,
        with_amount=True,
    )
    c = api_client
    payload_base = {
        "codes": ["SCL1", "SCL2"],
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "tsmom",
        "mom_lookback": 20,
        "tsmom_entry_threshold": 0.02,
        "tsmom_exit_threshold": 0.0,
        "position_sizing": "risk_budget",
        "risk_budget_pct": 0.005,
        "risk_budget_overcap_policy": "leverage_entry",
        "risk_budget_max_leverage_multiple": 3.0,
        "exec_price": "close",
        "atr_stop_mode": "none",
        "atr_stop_window": 20,
        "atr_stop_n": 2000.0,
        "r_profit_scaleout_execution_mode": "intraday",
        "r_profit_scaleout_tiers": [
            {"r_multiple": 2.0, "reduce_fraction": 0.5},
            {"r_multiple": 3.0, "reduce_fraction": 0.3},
        ],
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
        "quick_mode": True,
    }
    if runtime_engine:
        payload_base["engine"] = runtime_engine
    out_off = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {**payload_base, "r_profit_scaleout_enabled": False},
    )
    out_on = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {**payload_base, "r_profit_scaleout_enabled": True},
    )
    trig = int(
        (
            ((out_on.get("trade_statistics") or {}).get("overall") or {}).get(
                "r_profit_scaleout_trigger_count"
            )
            or 0
        )
    )
    assert trig == 0
    m_off = ((out_off.get("metrics") or {}).get("strategy")) or {}
    m_on = ((out_on.get("metrics") or {}).get("strategy")) or {}
    assert float(m_on.get("cumulative_return") or 0.0) == pytest.approx(
        float(m_off.get("cumulative_return") or 0.0), rel=0.0, abs=1e-12
    )
    assert float(m_on.get("annualized_return") or 0.0) == pytest.approx(
        float(m_off.get("annualized_return") or 0.0), rel=0.0, abs=1e-12
    )
    assert float(m_on.get("max_drawdown") or 0.0) == pytest.approx(
        float(m_off.get("max_drawdown") or 0.0), rel=0.0, abs=1e-12
    )


@pytest.mark.parametrize("runtime_engine", [None, "bt"])
def test_api_trend_bias_v_tier_zero_trigger_is_noop(runtime_engine, engine, api_client):
    dates = [d.date() for d in pd.date_range("2021-01-01", periods=220, freq="B")]
    _seed_trend_capacity_prices(
        engine,
        code_to_series={
            "BVT1": [
                100.0 + i * 0.05 + ((i % 7) - 3) * 0.07 for i in range(len(dates))
            ],
            "BVT2": [
                95.0 + i * 0.04 + ((i % 11) - 5) * 0.06 for i in range(len(dates))
            ],
        },
        dates=dates,
        with_amount=True,
    )
    c = api_client
    payload_base = {
        "codes": ["BVT1", "BVT2"],
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 10,
        "position_sizing": "equal",
        "exec_price": "close",
        "bias_v_take_profit_reentry_mode": "reenter",
        "bias_v_take_profit_execution_mode": "intraday",
        "bias_v_ma_window": 20,
        "bias_v_atr_window": 20,
        "bias_v_take_profit_tiers": [
            {"threshold": 50.0, "reduce_fraction": 0.5},
            {"threshold": 70.0, "reduce_fraction": 0.5},
        ],
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
        "quick_mode": True,
    }
    if runtime_engine:
        payload_base["engine"] = runtime_engine
    out_off = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {**payload_base, "bias_v_take_profit_enabled": False},
    )
    out_on = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {**payload_base, "bias_v_take_profit_enabled": True},
    )
    trig = int(
        (
            (
                ((out_on.get("trade_statistics") or {}).get("overall") or {}).get(
                    "bias_v_take_profit_trigger_count"
                )
            )
            or 0
        )
    )
    assert trig == 0
    m_off = ((out_off.get("metrics") or {}).get("strategy")) or {}
    m_on = ((out_on.get("metrics") or {}).get("strategy")) or {}
    assert float(m_on.get("cumulative_return") or 0.0) == pytest.approx(
        float(m_off.get("cumulative_return") or 0.0), rel=0.0, abs=1e-12
    )
    assert float(m_on.get("annualized_return") or 0.0) == pytest.approx(
        float(m_off.get("annualized_return") or 0.0), rel=0.0, abs=1e-12
    )
    assert float(m_on.get("max_drawdown") or 0.0) == pytest.approx(
        float(m_off.get("max_drawdown") or 0.0), rel=0.0, abs=1e-12
    )


def test_api_trend_breakeven_addon_defaults_to_enabled(engine, api_client):
    dates = [d.date() for d in pd.date_range("2022-01-03", periods=180, freq="B")]
    _seed_trend_capacity_prices(
        engine,
        code_to_series={
            "BEA1": [100.0 + i * 0.05 + ((i % 9) - 4) * 0.03 for i in range(len(dates))]
        },
        dates=dates,
        with_amount=True,
    )
    out = post_json_ok(
        api_client,
        "/api/analysis/trend",
        {
            "code": "BEA1",
            "start": fmt_ymd(dates[0]),
            "end": fmt_ymd(dates[-1]),
            "strategy": "ma_filter",
            "sma_window": 15,
            "position_sizing": "equal",
            "exec_price": "close",
            "r_profit_scaleout_enabled": False,
            "bias_v_take_profit_enabled": False,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
            "quick_mode": True,
        },
    )
    params = ((out.get("meta") or {}).get("params")) or {}
    assert bool(params.get("r_profit_scaleout_breakeven_stop_enabled")) is True
    assert bool(params.get("bias_v_take_profit_breakeven_stop_enabled")) is True
    addons = ((out.get("risk_controls") or {}).get("addons")) or {}
    rps_addon = addons.get("r_profit_scaleout_breakeven_stop") or {}
    bv_addon = addons.get("bias_v_take_profit_breakeven_stop") or {}
    assert bool(rps_addon.get("enabled")) is True
    assert bool(bv_addon.get("enabled")) is True


@pytest.mark.parametrize("runtime_engine", [None, "bt"])
def test_api_trend_portfolio_breakeven_addon_switch_is_respected(
    runtime_engine, engine, api_client
):
    dates = [d.date() for d in pd.date_range("2022-01-03", periods=220, freq="B")]
    _seed_trend_capacity_prices(
        engine,
        code_to_series={
            "BEP1": [
                100.0 + i * 0.07 + ((i % 9) - 4) * 0.05 for i in range(len(dates))
            ],
            "BEP2": [
                95.0 + i * 0.05 + ((i % 11) - 5) * 0.04 for i in range(len(dates))
            ],
        },
        dates=dates,
        with_amount=True,
    )
    payload = {
        "codes": ["BEP1", "BEP2"],
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 15,
        "position_sizing": "equal",
        "exec_price": "close",
        "atr_stop_mode": "none",
        "r_profit_scaleout_enabled": True,
        "r_profit_scaleout_execution_mode": "intraday",
        "r_profit_scaleout_breakeven_stop_enabled": False,
        "r_profit_scaleout_tiers": [{"r_multiple": 50.0, "reduce_fraction": 0.5}],
        "bias_v_take_profit_enabled": True,
        "bias_v_take_profit_reentry_mode": "reenter",
        "bias_v_take_profit_execution_mode": "intraday",
        "bias_v_take_profit_breakeven_stop_enabled": False,
        "bias_v_ma_window": 20,
        "bias_v_atr_window": 20,
        "bias_v_take_profit_tiers": [{"threshold": 50.0, "reduce_fraction": 0.5}],
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
        "quick_mode": True,
    }
    if runtime_engine:
        payload["engine"] = runtime_engine
    out = post_json_ok(api_client, "/api/analysis/trend/portfolio", payload)
    params = ((out.get("meta") or {}).get("params")) or {}
    assert bool(params.get("r_profit_scaleout_breakeven_stop_enabled")) is False
    assert bool(params.get("bias_v_take_profit_breakeven_stop_enabled")) is False
    addons = ((out.get("risk_controls") or {}).get("addons")) or {}
    rps_addon = addons.get("r_profit_scaleout_breakeven_stop") or {}
    bv_addon = addons.get("bias_v_take_profit_breakeven_stop") or {}
    assert bool(rps_addon.get("enabled")) is False
    assert bool(bv_addon.get("enabled")) is False


@pytest.mark.parametrize("runtime_engine", [None, "bt"])
def test_api_trend_bias_v_deprecated_threshold_is_ignored(
    runtime_engine, engine, api_client
):
    dates = [d.date() for d in pd.date_range("2022-01-03", periods=220, freq="B")]
    _seed_trend_capacity_prices(
        engine,
        code_to_series={
            "BVC1": [
                100.0 + i * 0.08 + ((i % 9) - 4) * 0.09 for i in range(len(dates))
            ],
            "BVC2": [95.0 + i * 0.05 + ((i % 7) - 3) * 0.07 for i in range(len(dates))],
        },
        dates=dates,
        with_amount=True,
    )
    c = api_client
    payload_base = {
        "codes": ["BVC1", "BVC2"],
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 15,
        "position_sizing": "equal",
        "exec_price": "close",
        "bias_v_take_profit_enabled": True,
        "bias_v_take_profit_reentry_mode": "reenter",
        "bias_v_take_profit_execution_mode": "intraday",
        "bias_v_ma_window": 20,
        "bias_v_atr_window": 20,
        "bias_v_take_profit_tiers": [
            {"threshold": 2.0, "reduce_fraction": 0.5},
            {"threshold": 4.0, "reduce_fraction": 0.5},
        ],
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
        "quick_mode": True,
    }
    if runtime_engine:
        payload_base["engine"] = runtime_engine
    out_no_deprecated = post_json_ok(c, "/api/analysis/trend/portfolio", payload_base)
    out_with_deprecated = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {**payload_base, "bias_v_take_profit_threshold": 999.0},
    )

    p0 = ((out_no_deprecated or {}).get("meta") or {}).get("params") or {}
    p1 = ((out_with_deprecated or {}).get("meta") or {}).get("params") or {}
    assert "bias_v_take_profit_threshold" not in p0
    assert "bias_v_take_profit_threshold" not in p1

    rc0 = ((out_no_deprecated or {}).get("risk_controls") or {}).get(
        "bias_v_take_profit"
    ) or {}
    rc1 = ((out_with_deprecated or {}).get("risk_controls") or {}).get(
        "bias_v_take_profit"
    ) or {}
    assert "threshold" not in rc0
    assert "threshold" not in rc1

    m0 = ((out_no_deprecated or {}).get("metrics") or {}).get("strategy") or {}
    m1 = ((out_with_deprecated or {}).get("metrics") or {}).get("strategy") or {}
    assert float(m1.get("cumulative_return") or 0.0) == pytest.approx(
        float(m0.get("cumulative_return") or 0.0), rel=0.0, abs=1e-12
    )
    assert float(m1.get("annualized_return") or 0.0) == pytest.approx(
        float(m0.get("annualized_return") or 0.0), rel=0.0, abs=1e-12
    )
    assert float(m1.get("max_drawdown") or 0.0) == pytest.approx(
        float(m0.get("max_drawdown") or 0.0), rel=0.0, abs=1e-12
    )

    nav0 = (((out_no_deprecated or {}).get("nav") or {}).get("series") or {}).get(
        "STRAT"
    ) or []
    nav1 = (((out_with_deprecated or {}).get("nav") or {}).get("series") or {}).get(
        "STRAT"
    ) or []
    assert len(nav1) == len(nav0)
    for x, y in zip(nav0, nav1):
        assert float(y) == pytest.approx(float(x), rel=0.0, abs=1e-12)


def test_stop_execution_tier_reduce_fraction_is_absolute_on_same_day():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    weights = pd.Series([1.0, 1.0, 1.0], index=idx, dtype=float)
    open_sig = pd.Series([100.0, 100.0, 100.0], index=idx, dtype=float)
    close_sig = pd.Series([100.0, 100.0, 100.0], index=idx, dtype=float)
    stats = {
        "trigger_events": [
            {
                "execution_date": "2024-01-03",
                "execution_mode": "intraday",
                "reduce_fraction": 0.5,
                "fill_price": 110.0,
            },
            {
                "execution_date": "2024-01-03",
                "execution_mode": "intraday",
                "reduce_fraction": 0.5,
                "fill_price": 120.0,
            },
        ]
    }
    w_adj, override = _apply_intraday_stop_execution_single(
        weights=weights,
        atr_stop_stats=stats,
        exec_price="open",
        stop_execution_mode="intraday",
        open_sig=open_sig,
        close_sig=close_sig,
    )
    d = pd.Timestamp("2024-01-03")
    assert float(w_adj.loc[d]) == pytest.approx(0.0, rel=0.0, abs=1e-12)
    assert float(override.loc[d]) == pytest.approx(0.15, rel=0.0, abs=1e-12)


@pytest.mark.parametrize("runtime_engine", [None, "bt"])
def test_api_trend_atr_zero_trigger_is_noop(runtime_engine, engine, api_client):
    dates = [d.date() for d in pd.date_range("2021-01-01", periods=220, freq="B")]
    _seed_trend_capacity_prices(
        engine,
        code_to_series={
            "ATR1": [
                100.0 + i * 0.05 + ((i % 9) - 4) * 0.06 for i in range(len(dates))
            ],
            "ATR2": [96.0 + i * 0.04 + ((i % 7) - 3) * 0.05 for i in range(len(dates))],
        },
        dates=dates,
        with_amount=True,
    )
    c = api_client
    payload_base = {
        "codes": ["ATR1", "ATR2"],
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "tsmom",
        "mom_lookback": 30,
        "tsmom_entry_threshold": 0.02,
        "tsmom_exit_threshold": 0.0,
        "position_sizing": "equal",
        "exec_price": "close",
        "atr_stop_reentry_mode": "reenter",
        "atr_stop_execution_mode": "intraday",
        "atr_stop_window": 20,
        "atr_stop_n": 1000.0,
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
        "quick_mode": True,
    }
    if runtime_engine:
        payload_base["engine"] = runtime_engine
    out_off = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {**payload_base, "atr_stop_mode": "none"},
    )
    out_on = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {**payload_base, "atr_stop_mode": "static"},
    )
    trig = int(
        (
            (
                ((out_on.get("trade_statistics") or {}).get("overall") or {}).get(
                    "atr_stop_trigger_count"
                )
            )
            or 0
        )
    )
    assert trig == 0
    m_off = ((out_off.get("metrics") or {}).get("strategy")) or {}
    m_on = ((out_on.get("metrics") or {}).get("strategy")) or {}
    assert float(m_on.get("cumulative_return") or 0.0) == pytest.approx(
        float(m_off.get("cumulative_return") or 0.0), rel=0.0, abs=1e-12
    )
    assert float(m_on.get("annualized_return") or 0.0) == pytest.approx(
        float(m_off.get("annualized_return") or 0.0), rel=0.0, abs=1e-12
    )
    assert float(m_on.get("max_drawdown") or 0.0) == pytest.approx(
        float(m_off.get("max_drawdown") or 0.0), rel=0.0, abs=1e-12
    )


@pytest.mark.parametrize("runtime_engine", [None, "bt"])
def test_api_trend_r_take_profit_zero_trigger_is_noop(
    runtime_engine, engine, api_client
):
    dates = [d.date() for d in pd.date_range("2021-01-01", periods=220, freq="B")]
    _seed_trend_capacity_prices(
        engine,
        code_to_series={
            "RTP1": [
                100.0 + i * 0.05 + ((i % 8) - 4) * 0.07 for i in range(len(dates))
            ],
            "RTP2": [
                92.0 + i * 0.03 + ((i % 10) - 5) * 0.06 for i in range(len(dates))
            ],
        },
        dates=dates,
        with_amount=True,
    )
    c = api_client
    payload_base = {
        "codes": ["RTP1", "RTP2"],
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 10,
        "position_sizing": "equal",
        "exec_price": "close",
        "atr_stop_mode": "none",
        "atr_stop_window": 20,
        "atr_stop_n": 2.0,
        "r_take_profit_reentry_mode": "reenter",
        "r_take_profit_execution_mode": "intraday",
        "r_take_profit_tiers": [
            {"r_multiple": 1000.0, "retrace_ratio": 0.5},
            {"r_multiple": 2000.0, "retrace_ratio": 0.5},
        ],
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
        "quick_mode": True,
    }
    if runtime_engine:
        payload_base["engine"] = runtime_engine
    out_off = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {**payload_base, "r_take_profit_enabled": False},
    )
    out_on = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {**payload_base, "r_take_profit_enabled": True},
    )
    trig = int(
        (
            (
                ((out_on.get("trade_statistics") or {}).get("overall") or {}).get(
                    "r_take_profit_trigger_count"
                )
            )
            or 0
        )
    )
    assert trig == 0
    m_off = ((out_off.get("metrics") or {}).get("strategy")) or {}
    m_on = ((out_on.get("metrics") or {}).get("strategy")) or {}
    assert float(m_on.get("cumulative_return") or 0.0) == pytest.approx(
        float(m_off.get("cumulative_return") or 0.0), rel=0.0, abs=1e-12
    )
    assert float(m_on.get("annualized_return") or 0.0) == pytest.approx(
        float(m_off.get("annualized_return") or 0.0), rel=0.0, abs=1e-12
    )
    assert float(m_on.get("max_drawdown") or 0.0) == pytest.approx(
        float(m_off.get("max_drawdown") or 0.0), rel=0.0, abs=1e-12
    )


@pytest.mark.parametrize("runtime_engine", [None, "bt"])
def test_api_trend_vol_regime_zero_adjust_is_noop(runtime_engine, engine, api_client):
    dates = [d.date() for d in pd.date_range("2021-01-01", periods=220, freq="B")]
    _seed_trend_capacity_prices(
        engine,
        code_to_series={
            "VOL1": [
                100.0 + i * 0.06 + ((i % 9) - 4) * 0.05 for i in range(len(dates))
            ],
            "VOL2": [
                90.0 + i * 0.05 + ((i % 11) - 5) * 0.04 for i in range(len(dates))
            ],
        },
        dates=dates,
        with_amount=True,
    )
    c = api_client
    payload_base = {
        "codes": ["VOL1", "VOL2"],
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 10,
        "position_sizing": "risk_budget",
        "risk_budget_pct": 0.01,
        "risk_budget_atr_window": 20,
        "risk_budget_overcap_policy": "scale",
        "exec_price": "close",
        "vol_ratio_fast_atr_window": 20,
        "vol_ratio_slow_atr_window": 20,
        "vol_ratio_expand_threshold": 100.0,
        "vol_ratio_contract_threshold": 0.1,
        "vol_ratio_normal_threshold": 50.0,
        "vol_ratio_extreme_threshold": 200.0,
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
        "quick_mode": True,
    }
    if runtime_engine:
        payload_base["engine"] = runtime_engine
    out_off = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {**payload_base, "vol_regime_risk_mgmt_enabled": False},
    )
    out_on = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {**payload_base, "vol_regime_risk_mgmt_enabled": True},
    )
    adj_cnt = int(
        (
            (
                ((out_on.get("trade_statistics") or {}).get("overall") or {}).get(
                    "vol_risk_adjust_total_count"
                )
            )
            or 0
        )
    )
    assert adj_cnt == 0
    m_off = ((out_off.get("metrics") or {}).get("strategy")) or {}
    m_on = ((out_on.get("metrics") or {}).get("strategy")) or {}
    assert float(m_on.get("cumulative_return") or 0.0) == pytest.approx(
        float(m_off.get("cumulative_return") or 0.0), rel=0.0, abs=1e-12
    )
    assert float(m_on.get("annualized_return") or 0.0) == pytest.approx(
        float(m_off.get("annualized_return") or 0.0), rel=0.0, abs=1e-12
    )
    assert float(m_on.get("max_drawdown") or 0.0) == pytest.approx(
        float(m_off.get("max_drawdown") or 0.0), rel=0.0, abs=1e-12
    )


@pytest.mark.parametrize("runtime_engine", [None, "bt"])
def test_api_trend_single_trade_stats_include_semi_variance(
    runtime_engine, engine, api_client
):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=120, freq="B")]
    series = {"SVS1": [100.0 + i * 0.2 + (2.0 if i % 2 else -1.5) for i in range(120)]}
    seed_prices(engine, code_to_series=series, dates=dates)
    c = api_client
    payload = {
        "code": "SVS1",
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 3,
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
    }
    if runtime_engine:
        payload["engine"] = runtime_engine
    out = post_json_ok(c, "/api/analysis/trend", payload)
    ts = out.get("trade_statistics") or {}
    overall = ts.get("overall") or {}
    by_code = (ts.get("by_code") or {}).get("SVS1") or {}
    _assert_semi_variance_stats_shape(overall)
    _assert_semi_variance_stats_shape(by_code)
    _assert_risk_of_ruin_stats_shape(overall, expected_maxrisk=0.30)
    _assert_risk_of_ruin_stats_shape(by_code, expected_maxrisk=0.30)
    _assert_trade_extreme_stats_shape(overall)
    _assert_trade_extreme_stats_shape(by_code)


@pytest.mark.parametrize("runtime_engine", [None, "bt"])
def test_api_trend_portfolio_trade_stats_include_semi_variance(
    runtime_engine, engine, api_client
):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=120, freq="B")]
    series = {
        "SVP1": [100.0 + i * 0.3 + (1.0 if i % 3 else -1.2) for i in range(120)],
        "SVP2": [90.0 + i * 0.25 + (1.3 if i % 4 else -1.0) for i in range(120)],
    }
    seed_prices(engine, code_to_series=series, dates=dates)
    c = api_client
    payload = {
        "codes": ["SVP1", "SVP2"],
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 3,
        "position_sizing": "equal",
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
    }
    if runtime_engine:
        payload["engine"] = runtime_engine
    out = post_json_ok(c, "/api/analysis/trend/portfolio", payload)
    ts = out.get("trade_statistics") or {}
    overall = ts.get("overall") or {}
    by_code = ts.get("by_code") or {}
    _assert_semi_variance_stats_shape(overall)
    _assert_semi_variance_stats_shape(by_code.get("SVP1") or {})
    _assert_semi_variance_stats_shape(by_code.get("SVP2") or {})
    _assert_risk_of_ruin_stats_shape(overall, expected_maxrisk=0.30)
    _assert_risk_of_ruin_stats_shape(by_code.get("SVP1") or {}, expected_maxrisk=0.30)
    _assert_risk_of_ruin_stats_shape(by_code.get("SVP2") or {}, expected_maxrisk=0.30)
    _assert_trade_extreme_stats_shape(overall)
    _assert_trade_extreme_stats_shape(by_code.get("SVP1") or {})
    _assert_trade_extreme_stats_shape(by_code.get("SVP2") or {})


@pytest.mark.parametrize("runtime_engine", [None, "bt"])
def test_api_trend_single_trade_stats_accepts_custom_risk_of_ruin_maxrisk(
    runtime_engine, engine, api_client
):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=100, freq="B")]
    series = {"SVR1": [100.0 + i * 0.18 + (1.5 if i % 2 else -1.1) for i in range(100)]}
    seed_prices(engine, code_to_series=series, dates=dates)
    c = api_client
    payload = {
        "code": "SVR1",
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 3,
        "risk_of_ruin_maxrisk": 0.40,
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
    }
    if runtime_engine:
        payload["engine"] = runtime_engine
    out = post_json_ok(c, "/api/analysis/trend", payload)
    ts = out.get("trade_statistics") or {}
    overall = ts.get("overall") or {}
    _assert_risk_of_ruin_stats_shape(overall, expected_maxrisk=0.40)


def test_api_trend_single_risk_budget_vol_regime_stats_contract(engine, api_client):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=140, freq="B")]
    series = {"RBVS1": [100.0 + i * 0.5 for i, _ in enumerate(dates)]}
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    out = post_json_ok(
        c,
        "/api/analysis/trend",
        {
            "code": "RBVS1",
            "start": fmt_ymd(dates[0]),
            "end": fmt_ymd(dates[-1]),
            "strategy": "ma_filter",
            "sma_window": 2,
            "position_sizing": "risk_budget",
            "risk_budget_atr_window": 20,
            "risk_budget_pct": 0.01,
            "vol_regime_risk_mgmt_enabled": True,
            "vol_ratio_fast_atr_window": 5,
            "vol_ratio_slow_atr_window": 50,
            "vol_ratio_expand_threshold": 1.45,
            "vol_ratio_contract_threshold": 0.65,
            "vol_ratio_normal_threshold": 1.05,
            "vol_ratio_extreme_threshold": 2.0,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    ts = out.get("trade_statistics") or {}
    overall = ts.get("overall") or {}
    by_code = (ts.get("by_code") or {}).get("RBVS1", {})
    assert "vol_risk_adjust_total_count" in overall
    assert "vol_risk_adjust_reduce_on_expand_count" in overall
    assert "vol_risk_adjust_recover_from_contract_count" in overall
    assert "vol_risk_entry_state_reduce_on_expand_count" in overall
    assert "vol_risk_entry_state_increase_on_contract_count" in overall
    assert "vol_risk_adjust_total_count" in by_code
    assert "vol_risk_entry_state_reduce_on_expand_count" in by_code
    rc = (out.get("risk_controls") or {}).get("vol_regime_risk_mgmt") or {}
    assert rc.get("enabled") is True
    assert float(rc.get("extreme_threshold") or 0.0) == pytest.approx(2.0)


def test_api_trend_single_vol_regime_extreme_threshold_default_is_2_2(
    engine, api_client
):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=140, freq="B")]
    series = {"RBVS2": [100.0 + i * 0.5 for i, _ in enumerate(dates)]}
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    out = post_json_ok(
        c,
        "/api/analysis/trend",
        {
            "code": "RBVS2",
            "start": fmt_ymd(dates[0]),
            "end": fmt_ymd(dates[-1]),
            "strategy": "ma_filter",
            "sma_window": 2,
            "position_sizing": "risk_budget",
            "risk_budget_atr_window": 20,
            "risk_budget_pct": 0.01,
            "vol_regime_risk_mgmt_enabled": True,
            "vol_ratio_fast_atr_window": 5,
            "vol_ratio_slow_atr_window": 50,
            "vol_ratio_expand_threshold": 1.45,
            "vol_ratio_contract_threshold": 0.65,
            "vol_ratio_normal_threshold": 1.05,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    rc = (out.get("risk_controls") or {}).get("vol_regime_risk_mgmt") or {}
    assert float(rc.get("extreme_threshold") or 0.0) == pytest.approx(2.2)


def test_api_trend_single_rejects_invalid_vol_regime_threshold_relation(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240103",
    )
    err = post_json(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "sma_window": 2,
            "position_sizing": "risk_budget",
            "risk_budget_atr_window": 20,
            "risk_budget_pct": 0.01,
            "vol_regime_risk_mgmt_enabled": True,
            "vol_ratio_expand_threshold": 1.0,
            "vol_ratio_contract_threshold": 0.65,
            "vol_ratio_normal_threshold": 1.05,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
        expected_status=400,
    )
    assert "vol_ratio_expand_threshold must be > vol_ratio_normal_threshold" in str(err)


def test_api_trend_single_rejects_invalid_vol_regime_extreme_threshold(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240103",
    )
    err = post_json(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "sma_window": 2,
            "position_sizing": "risk_budget",
            "risk_budget_atr_window": 20,
            "risk_budget_pct": 0.01,
            "vol_regime_risk_mgmt_enabled": True,
            "vol_ratio_expand_threshold": 1.45,
            "vol_ratio_contract_threshold": 0.65,
            "vol_ratio_normal_threshold": 1.05,
            "vol_ratio_extreme_threshold": 1.40,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
        expected_status=400,
    )
    assert "vol_ratio_extreme_threshold must be > vol_ratio_expand_threshold" in str(
        err
    )


@pytest.mark.parametrize("runtime_engine", [None, "bt"])
def test_api_trend_single_periodic_risk_mgmt_stats_contract(
    runtime_engine, engine, api_client
):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=140, freq="B")]
    series = {"RBVP1": [100.0 + i * 0.5 for i, _ in enumerate(dates)]}
    seed_prices(engine, code_to_series=series, dates=dates)
    c = api_client
    payload = {
        "code": "RBVP1",
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 2,
        "position_sizing": "risk_budget",
        "risk_budget_atr_window": 20,
        "risk_budget_pct": 0.01,
        "vol_periodic_risk_mgmt_enabled": True,
        "vol_periodic_rebalance_threshold_pct": 0.05,
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
    }
    if runtime_engine:
        payload["engine"] = runtime_engine
    out = post_json_ok(c, "/api/analysis/trend", payload)
    params = (out.get("meta") or {}).get("params") or {}
    assert params.get("vol_periodic_risk_mgmt_enabled") is True
    assert float(
        params.get("vol_periodic_rebalance_threshold_pct") or 0.0
    ) == pytest.approx(0.05)
    rc = (out.get("risk_controls") or {}).get("vol_periodic_risk_mgmt") or {}
    assert rc.get("enabled") is True
    assert float(rc.get("rebalance_threshold_pct") or 0.0) == pytest.approx(0.05)
    overall = (out.get("trade_statistics") or {}).get("overall") or {}
    assert "vol_periodic_rebalance_trigger_count" in overall


@pytest.mark.parametrize("runtime_engine", [None, "bt"])
def test_api_trend_single_periodic_risk_mgmt_lot_stats_contract(
    runtime_engine, engine, api_client
):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=140, freq="B")]
    series = {"RBVP1L": [100.0 + i * 0.5 for i, _ in enumerate(dates)]}
    seed_prices(engine, code_to_series=series, dates=dates)
    c = api_client
    payload = {
        "code": "RBVP1L",
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 2,
        "position_sizing": "risk_budget",
        "risk_budget_atr_window": 20,
        "risk_budget_pct": 0.01,
        "vol_periodic_risk_mgmt_enabled": True,
        "vol_periodic_rebalance_threshold_pct": 0.05,
        "initial_account_amount": 1_000_000.0,
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
    }
    if runtime_engine:
        payload["engine"] = runtime_engine
    out = post_json_ok(c, "/api/analysis/trend", payload)
    rc = (out.get("risk_controls") or {}).get("vol_periodic_risk_mgmt") or {}
    lot_meta = (out.get("meta") or {}).get("account_lot_sizing") or {}
    lot_stats = (lot_meta.get("periodic_rebalance_stats") or {}).get("overall") or {}
    assert bool(lot_meta.get("enabled")) is True
    assert bool(lot_meta.get("periodic_rebalance_enabled")) is True
    assert int(rc.get("evaluated_count") or 0) == int(
        lot_stats.get("periodic_rebalance_evaluated_count") or 0
    )
    assert int(rc.get("trigger_count") or 0) == int(
        lot_stats.get("periodic_rebalance_trigger_count") or 0
    )
    assert int(rc.get("skip_count") or 0) == int(
        lot_stats.get("periodic_rebalance_skip_count") or 0
    )


@pytest.mark.parametrize("runtime_engine", [None, "bt"])
def test_api_trend_single_periodic_risk_mgmt_enabled_false_when_not_risk_budget(
    runtime_engine, engine, api_client
):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=80, freq="B")]
    series = {"RBVP1E": [100.0 + i * 0.3 for i, _ in enumerate(dates)]}
    seed_prices(engine, code_to_series=series, dates=dates)
    c = api_client
    payload = {
        "code": "RBVP1E",
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 2,
        "position_sizing": "equal",
        "vol_periodic_risk_mgmt_enabled": True,
        "vol_periodic_rebalance_threshold_pct": 0.05,
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
    }
    if runtime_engine:
        payload["engine"] = runtime_engine
    out = post_json_ok(c, "/api/analysis/trend", payload)
    rc = (out.get("risk_controls") or {}).get("vol_periodic_risk_mgmt") or {}
    assert rc.get("enabled") is False


def test_api_trend_single_periodic_risk_mgmt_legacy_bt_consistent(engine, api_client):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=140, freq="B")]
    series = {"RBVP1C": [100.0 + i * 0.5 for i, _ in enumerate(dates)]}
    seed_prices(engine, code_to_series=series, dates=dates)
    c = api_client
    payload = {
        "code": "RBVP1C",
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 2,
        "position_sizing": "risk_budget",
        "risk_budget_atr_window": 20,
        "risk_budget_pct": 0.01,
        "vol_periodic_risk_mgmt_enabled": True,
        "vol_periodic_rebalance_threshold_pct": 0.05,
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
    }
    out_legacy = post_json_ok(c, "/api/analysis/trend", dict(payload))
    out_bt = post_json_ok(c, "/api/analysis/trend", {**payload, "engine": "bt"})
    legacy_rc = (out_legacy.get("risk_controls") or {}).get(
        "vol_periodic_risk_mgmt"
    ) or {}
    bt_rc = (out_bt.get("risk_controls") or {}).get("vol_periodic_risk_mgmt") or {}
    assert int(legacy_rc.get("trigger_count") or 0) == int(
        bt_rc.get("trigger_count") or 0
    )
    legacy_overall = (out_legacy.get("trade_statistics") or {}).get("overall") or {}
    bt_overall = (out_bt.get("trade_statistics") or {}).get("overall") or {}
    assert int(legacy_overall.get("vol_periodic_rebalance_trigger_count") or 0) == int(
        bt_overall.get("vol_periodic_rebalance_trigger_count") or 0
    )


def test_api_trend_single_rejects_mutual_vol_regime_and_periodic_risk_mgmt(api_client):
    c = api_client
    err = post_json(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "sma_window": 2,
            "position_sizing": "risk_budget",
            "vol_regime_risk_mgmt_enabled": True,
            "vol_periodic_risk_mgmt_enabled": True,
        },
        expected_status=422,
    )
    assert "cannot both be enabled" in str(err)


@pytest.mark.parametrize("runtime_engine", [None, "bt"])
def test_api_trend_portfolio_periodic_risk_mgmt_stats_contract(
    runtime_engine, engine, api_client
):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=100, freq="B")]
    series = {"RBVP2": [100.0 + i * 0.4 for i, _ in enumerate(dates)]}
    seed_prices(engine, code_to_series=series, dates=dates)
    c = api_client
    payload = {
        "codes": ["RBVP2"],
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 2,
        "position_sizing": "risk_budget",
        "risk_budget_atr_window": 20,
        "risk_budget_pct": 0.01,
        "vol_periodic_risk_mgmt_enabled": True,
        "vol_periodic_rebalance_threshold_pct": 0.05,
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
    }
    if runtime_engine:
        payload["engine"] = runtime_engine
    out = post_json_ok(c, "/api/analysis/trend/portfolio", payload)
    params = (out.get("meta") or {}).get("params") or {}
    assert params.get("vol_periodic_risk_mgmt_enabled") is True
    rc = (out.get("risk_controls") or {}).get("vol_periodic_risk_mgmt") or {}
    assert rc.get("enabled") is True
    assert "trigger_count" in rc
    by_code = ((out.get("trade_statistics") or {}).get("by_code") or {}).get(
        "RBVP2", {}
    )
    assert "vol_periodic_rebalance_trigger_count" in by_code


@pytest.mark.parametrize("runtime_engine", [None, "bt"])
def test_api_trend_portfolio_periodic_risk_mgmt_enabled_false_when_not_risk_budget(
    runtime_engine, engine, api_client
):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=80, freq="B")]
    series = {"RBVP2E": [100.0 + i * 0.4 for i, _ in enumerate(dates)]}
    seed_prices(engine, code_to_series=series, dates=dates)
    c = api_client
    payload = {
        "codes": ["RBVP2E"],
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 2,
        "position_sizing": "equal",
        "vol_periodic_risk_mgmt_enabled": True,
        "vol_periodic_rebalance_threshold_pct": 0.05,
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
    }
    if runtime_engine:
        payload["engine"] = runtime_engine
    out = post_json_ok(c, "/api/analysis/trend/portfolio", payload)
    rc = (out.get("risk_controls") or {}).get("vol_periodic_risk_mgmt") or {}
    assert rc.get("enabled") is False


def test_api_trend_portfolio_rejects_periodic_threshold_above_one(api_client):
    c = api_client
    err = post_json(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": ["510300"],
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "sma_window": 2,
            "position_sizing": "risk_budget",
            "vol_periodic_risk_mgmt_enabled": True,
            "vol_periodic_rebalance_threshold_pct": 1.01,
        },
        expected_status=422,
    )
    assert "vol_periodic_rebalance_threshold_pct" in str(err)


def test_api_trend_portfolio_risk_budget_freezes_weight_after_entry(engine, api_client):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=80, freq="B")]
    series = {"RBP1": [100.0 + i * 0.6 for i, _ in enumerate(dates)]}
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    out = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": ["RBP1"],
            "start": fmt_ymd(dates[0]),
            "end": fmt_ymd(dates[-1]),
            "strategy": "ma_filter",
            "sma_window": 2,
            "position_sizing": "risk_budget",
            "risk_budget_atr_window": 2,
            "risk_budget_pct": 0.005,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    params = ((out or {}).get("meta") or {}).get("params") or {}
    assert str(params.get("position_sizing") or "") == "risk_budget"
    assert int(params.get("risk_budget_atr_window") or 0) == 2
    assert float(params.get("risk_budget_pct") or 0.0) == pytest.approx(
        0.005, rel=0.0, abs=1e-12
    )
    w = [
        float(x)
        for x in ((((out.get("weights") or {}).get("series") or {}).get("RBP1")) or [])
    ]
    positive_w = [x for x in w if x > 1e-12]
    assert len(positive_w) >= 3
    assert max(positive_w) - min(positive_w) <= 1e-12


def test_api_trend_portfolio_risk_budget_overcap_scale_stats_contract(
    engine, api_client
):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=80, freq="B")]
    series = {
        "RBP2A": [100.0 + i * 0.6 for i, _ in enumerate(dates)],
        "RBP2B": [90.0 + i * 0.5 for i, _ in enumerate(dates)],
    }
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    out = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": ["RBP2A", "RBP2B"],
            "start": fmt_ymd(dates[0]),
            "end": fmt_ymd(dates[-1]),
            "strategy": "ma_filter",
            "sma_window": 2,
            "position_sizing": "risk_budget",
            "risk_budget_atr_window": 2,
            "risk_budget_pct": 0.02,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    ts = out.get("trade_statistics") or {}
    overall = ts.get("overall") or {}
    by_code = ts.get("by_code") or {}
    assert int(overall.get("vol_risk_overcap_scale_count") or 0) > 0
    assert int(
        ((by_code.get("RBP2A") or {}).get("vol_risk_overcap_scale_count") or 0) >= 0
    )
    assert int(
        ((by_code.get("RBP2B") or {}).get("vol_risk_overcap_scale_count") or 0) >= 0
    )


def test_api_trend_single_er_filter_contract(engine, api_client):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=80, freq="B")]
    series = {
        "ER1": [100.0 + ((-1.0) ** i) * 0.8 + i * 0.01 for i, _ in enumerate(dates)]
    }
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    base_payload = {
        "code": "ER1",
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 2,
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
    }
    out_no_filter = post_json_ok(
        c, "/api/analysis/trend", {**base_payload, "er_filter": False}
    )
    out_with_filter = post_json_ok(
        c,
        "/api/analysis/trend",
        {**base_payload, "er_filter": True, "er_window": 10, "er_threshold": 0.8},
    )

    pos_no_filter = [
        float(x) for x in ((out_no_filter.get("signals") or {}).get("position") or [])
    ]
    pos_with_filter = [
        float(x) for x in ((out_with_filter.get("signals") or {}).get("position") or [])
    ]
    assert any(x > 0.0 for x in pos_no_filter)
    assert all(x == 0.0 for x in pos_with_filter)
    params = ((out_with_filter or {}).get("meta") or {}).get("params") or {}
    assert params.get("er_filter") is True
    assert int(params.get("er_window") or 0) == 10
    assert float(params.get("er_threshold") or 0.0) == pytest.approx(
        0.8, rel=0.0, abs=1e-12
    )
    ts = out_with_filter.get("trade_statistics") or {}
    assert int((ts.get("overall") or {}).get("er_filter_blocked_entry_count") or 0) > 0
    assert int(
        (ts.get("overall") or {}).get("er_filter_attempted_entry_count") or 0
    ) >= int((ts.get("overall") or {}).get("er_filter_blocked_entry_count") or 0)
    assert int((ts.get("overall") or {}).get("er_filter_allowed_entry_count") or 0) >= 0
    assert (
        int(
            ((ts.get("by_code") or {}).get("ER1") or {}).get(
                "er_filter_blocked_entry_count"
            )
            or 0
        )
        > 0
    )


def test_api_trend_single_ma_entry_filter_contract(engine, api_client):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=80, freq="B")]
    series = {
        "MAAPI1": [100.0 + ((-1.0) ** i) * 0.8 + i * 0.01 for i, _ in enumerate(dates)]
    }
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    base_payload = {
        "code": "MAAPI1",
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 2,
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
    }
    out_no_filter = post_json_ok(
        c, "/api/analysis/trend", {**base_payload, "ma_entry_filter_enabled": False}
    )
    out_with_filter = post_json_ok(
        c,
        "/api/analysis/trend",
        {
            **base_payload,
            "ma_entry_filter_enabled": True,
            "ma_entry_filter_type": "sma",
            "ma_entry_filter_fast": 100,
            "ma_entry_filter_slow": 200,
        },
    )

    pos_no_filter = [
        float(x) for x in ((out_no_filter.get("signals") or {}).get("position") or [])
    ]
    pos_with_filter = [
        float(x) for x in ((out_with_filter.get("signals") or {}).get("position") or [])
    ]
    assert any(x > 0.0 for x in pos_no_filter)
    assert all(x == 0.0 for x in pos_with_filter)
    params = ((out_with_filter or {}).get("meta") or {}).get("params") or {}
    assert params.get("ma_entry_filter_enabled") is True
    assert str(params.get("ma_entry_filter_type") or "") == "sma"
    assert int(params.get("ma_entry_filter_fast") or 0) == 100
    assert int(params.get("ma_entry_filter_slow") or 0) == 200
    ts = out_with_filter.get("trade_statistics") or {}
    assert (
        int((ts.get("overall") or {}).get("ma_entry_filter_blocked_entry_count") or 0)
        > 0
    )
    assert int(
        (ts.get("overall") or {}).get("ma_entry_filter_attempted_entry_count") or 0
    ) >= int((ts.get("overall") or {}).get("ma_entry_filter_blocked_entry_count") or 0)
    assert (
        int((ts.get("overall") or {}).get("ma_entry_filter_allowed_entry_count") or 0)
        >= 0
    )
    assert (
        int(
            ((ts.get("by_code") or {}).get("MAAPI1") or {}).get(
                "ma_entry_filter_blocked_entry_count"
            )
            or 0
        )
        > 0
    )


def test_api_trend_single_er_exit_filter_contract(engine, api_client):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=120, freq="B")]
    series = {"ERX1": [100.0 + i * 0.6 for i, _ in enumerate(dates)]}
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    out_with_exit = post_json_ok(
        c,
        "/api/analysis/trend",
        {
            "code": "ERX1",
            "start": fmt_ymd(dates[0]),
            "end": fmt_ymd(dates[-1]),
            "strategy": "ma_filter",
            "sma_window": 2,
            "er_exit_filter": True,
            "er_exit_window": 10,
            "er_exit_threshold": 0.8,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    params = ((out_with_exit or {}).get("meta") or {}).get("params") or {}
    assert params.get("er_exit_filter") is True
    assert int(params.get("er_exit_window") or 0) == 10
    ts = out_with_exit.get("trade_statistics") or {}
    assert int((ts.get("overall") or {}).get("er_exit_filter_trigger_count") or 0) > 0
    assert (
        int(
            ((ts.get("by_code") or {}).get("ERX1") or {}).get(
                "er_exit_filter_trigger_count"
            )
            or 0
        )
        > 0
    )
    er_exit_rc = (out_with_exit.get("risk_controls") or {}).get("er_exit_filter") or {}
    assert int(er_exit_rc.get("trigger_count") or 0) > 0
    assert isinstance(er_exit_rc.get("trace_last_rows") or [], list)


def test_api_trend_single_impulse_entry_filter_contract(engine, api_client):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=100, freq="B")]
    series = {"IMPAPI1": [100.0 + i * 0.6 for i, _ in enumerate(dates)]}
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    out = post_json_ok(
        c,
        "/api/analysis/trend",
        {
            "code": "IMPAPI1",
            "start": fmt_ymd(dates[0]),
            "end": fmt_ymd(dates[-1]),
            "strategy": "ma_filter",
            "sma_window": 2,
            "impulse_entry_filter": True,
            "impulse_allow_bull": False,
            "impulse_allow_bear": False,
            "impulse_allow_neutral": False,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    pos = [float(x) for x in ((out.get("signals") or {}).get("position") or [])]
    assert all(x == 0.0 for x in pos)
    params = ((out or {}).get("meta") or {}).get("params") or {}
    assert params.get("impulse_entry_filter") is True
    ts = out.get("trade_statistics") or {}
    overall = ts.get("overall") or {}
    blocked = int(overall.get("impulse_filter_blocked_entry_count") or 0)
    blocked_split = (
        int(overall.get("impulse_filter_blocked_entry_count_bull") or 0)
        + int(overall.get("impulse_filter_blocked_entry_count_bear") or 0)
        + int(overall.get("impulse_filter_blocked_entry_count_neutral") or 0)
    )
    assert blocked > 0
    assert blocked_split == blocked
    assert (
        int(
            ((ts.get("by_code") or {}).get("IMPAPI1") or {}).get(
                "impulse_filter_blocked_entry_count"
            )
            or 0
        )
        == blocked
    )


def test_api_trend_single_kama_contract(engine, api_client):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=80, freq="B")]
    series = {
        "KAMA1": [
            100.0 + (i * 0.4 if i < 40 else (40 * 0.4) - (i - 40) * 0.45)
            for i, _ in enumerate(dates)
        ]
    }
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    out = post_json_ok(
        c,
        "/api/analysis/trend",
        {
            "code": "KAMA1",
            "start": fmt_ymd(dates[0]),
            "end": fmt_ymd(dates[-1]),
            "strategy": "ma_filter",
            "ma_type": "kama",
            "sma_window": 20,
            "kama_er_window": 10,
            "kama_fast_window": 2,
            "kama_slow_window": 30,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    params = ((out or {}).get("meta") or {}).get("params") or {}
    assert str(params.get("ma_type") or "") == "kama"
    assert int(params.get("kama_er_window") or 0) == 10
    assert int(params.get("kama_fast_window") or 0) == 2
    assert int(params.get("kama_slow_window") or 0) == 30


def test_api_trend_portfolio_er_filter_contract(engine, api_client):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=90, freq="B")]
    series = {
        "ER1": [100.0 + ((-1.0) ** i) * 0.8 + i * 0.01 for i, _ in enumerate(dates)],
        "ER2": [
            90.0 + ((-1.0) ** (i + 1)) * 0.7 + i * 0.01 for i, _ in enumerate(dates)
        ],
    }
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    base_payload = {
        "codes": ["ER1", "ER2"],
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 2,
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
        "position_sizing": "equal",
    }
    out_no_filter = post_json_ok(
        c, "/api/analysis/trend/portfolio", {**base_payload, "er_filter": False}
    )
    out_with_filter = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {**base_payload, "er_filter": True, "er_window": 10, "er_threshold": 0.8},
    )

    w_no = pd.DataFrame(((out_no_filter.get("weights") or {}).get("series") or {}))
    w_yes = pd.DataFrame(((out_with_filter.get("weights") or {}).get("series") or {}))
    assert not w_no.empty
    assert any(float(v) > 0.0 for v in w_no.to_numpy().ravel())
    assert all(float(v) == 0.0 for v in w_yes.to_numpy().ravel())
    params = ((out_with_filter or {}).get("meta") or {}).get("params") or {}
    assert params.get("er_filter") is True
    assert int(params.get("er_window") or 0) == 10
    assert float(params.get("er_threshold") or 0.0) == pytest.approx(
        0.8, rel=0.0, abs=1e-12
    )
    ts = out_with_filter.get("trade_statistics") or {}
    assert int((ts.get("overall") or {}).get("er_filter_blocked_entry_count") or 0) > 0
    assert int(
        (ts.get("overall") or {}).get("er_filter_attempted_entry_count") or 0
    ) >= int((ts.get("overall") or {}).get("er_filter_blocked_entry_count") or 0)
    assert int((ts.get("overall") or {}).get("er_filter_allowed_entry_count") or 0) >= 0
    by_code = ts.get("by_code") or {}
    assert int(
        ((by_code.get("ER1") or {}).get("er_filter_blocked_entry_count") or 0) >= 0
    )
    assert int(
        ((by_code.get("ER2") or {}).get("er_filter_blocked_entry_count") or 0) >= 0
    )


def test_api_trend_portfolio_ma_entry_filter_contract(engine, api_client):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=90, freq="B")]
    series = {
        "MAP1": [100.0 + ((-1.0) ** i) * 0.8 + i * 0.01 for i, _ in enumerate(dates)],
        "MAP2": [
            90.0 + ((-1.0) ** (i + 1)) * 0.7 + i * 0.01 for i, _ in enumerate(dates)
        ],
    }
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    base_payload = {
        "codes": ["MAP1", "MAP2"],
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 2,
        "position_sizing": "equal",
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
    }
    out_no_filter = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {**base_payload, "ma_entry_filter_enabled": False},
    )
    out_with_filter = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {
            **base_payload,
            "ma_entry_filter_enabled": True,
            "ma_entry_filter_type": "sma",
            "ma_entry_filter_fast": 100,
            "ma_entry_filter_slow": 200,
        },
    )

    w_no = pd.DataFrame(((out_no_filter.get("weights") or {}).get("series") or {}))
    w_yes = pd.DataFrame(((out_with_filter.get("weights") or {}).get("series") or {}))
    assert not w_no.empty
    assert any(float(v) > 0.0 for v in w_no.to_numpy().ravel())
    assert all(float(v) == 0.0 for v in w_yes.to_numpy().ravel())
    params = ((out_with_filter or {}).get("meta") or {}).get("params") or {}
    assert params.get("ma_entry_filter_enabled") is True
    assert str(params.get("ma_entry_filter_type") or "") == "sma"
    assert int(params.get("ma_entry_filter_fast") or 0) == 100
    assert int(params.get("ma_entry_filter_slow") or 0) == 200
    ts = out_with_filter.get("trade_statistics") or {}
    assert (
        int((ts.get("overall") or {}).get("ma_entry_filter_blocked_entry_count") or 0)
        > 0
    )
    assert int(
        (ts.get("overall") or {}).get("ma_entry_filter_attempted_entry_count") or 0
    ) >= int((ts.get("overall") or {}).get("ma_entry_filter_blocked_entry_count") or 0)
    assert (
        int((ts.get("overall") or {}).get("ma_entry_filter_allowed_entry_count") or 0)
        >= 0
    )
    by_code = ts.get("by_code") or {}
    assert int(
        ((by_code.get("MAP1") or {}).get("ma_entry_filter_blocked_entry_count") or 0)
        >= 0
    )
    assert int(
        ((by_code.get("MAP2") or {}).get("ma_entry_filter_blocked_entry_count") or 0)
        >= 0
    )


def test_api_trend_portfolio_impulse_entry_filter_contract(engine, api_client):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=100, freq="B")]
    series = {
        "IMPAP1": [100.0 + i * 0.6 for i, _ in enumerate(dates)],
        "IMPAP2": [120.0 + i * 0.5 for i, _ in enumerate(dates)],
    }
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    out = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": ["IMPAP1", "IMPAP2"],
            "start": fmt_ymd(dates[0]),
            "end": fmt_ymd(dates[-1]),
            "strategy": "ma_filter",
            "sma_window": 2,
            "position_sizing": "equal",
            "impulse_entry_filter": True,
            "impulse_allow_bull": False,
            "impulse_allow_bear": False,
            "impulse_allow_neutral": False,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    w = pd.DataFrame(((out.get("weights") or {}).get("series") or {}))
    assert not w.empty
    assert all(float(v) == 0.0 for v in w.to_numpy().ravel())
    params = ((out or {}).get("meta") or {}).get("params") or {}
    assert params.get("impulse_entry_filter") is True
    ts = out.get("trade_statistics") or {}
    overall = ts.get("overall") or {}
    blocked = int(overall.get("impulse_filter_blocked_entry_count") or 0)
    blocked_split = (
        int(overall.get("impulse_filter_blocked_entry_count_bull") or 0)
        + int(overall.get("impulse_filter_blocked_entry_count_bear") or 0)
        + int(overall.get("impulse_filter_blocked_entry_count_neutral") or 0)
    )
    assert blocked > 0
    assert blocked_split == blocked
    by_code = ts.get("by_code") or {}
    assert int(
        ((by_code.get("IMPAP1") or {}).get("impulse_filter_blocked_entry_count") or 0)
        >= 0
    )
    assert int(
        ((by_code.get("IMPAP2") or {}).get("impulse_filter_blocked_entry_count") or 0)
        >= 0
    )


def test_api_trend_portfolio_er_exit_filter_contract(engine, api_client):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=120, freq="B")]
    series = {
        "ERX1": [100.0 + i * 0.6 for i, _ in enumerate(dates)],
        "ERX2": [120.0 + i * 0.5 for i, _ in enumerate(dates)],
    }
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    out = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": ["ERX1", "ERX2"],
            "start": fmt_ymd(dates[0]),
            "end": fmt_ymd(dates[-1]),
            "strategy": "ma_filter",
            "sma_window": 2,
            "position_sizing": "equal",
            "er_exit_filter": True,
            "er_exit_window": 10,
            "er_exit_threshold": 0.8,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    params = ((out or {}).get("meta") or {}).get("params") or {}
    assert params.get("er_exit_filter") is True
    assert int(params.get("er_exit_window") or 0) == 10
    ts = out.get("trade_statistics") or {}
    assert int((ts.get("overall") or {}).get("er_exit_filter_trigger_count") or 0) > 0
    by_code = ts.get("by_code") or {}
    assert int(
        ((by_code.get("ERX1") or {}).get("er_exit_filter_trigger_count") or 0) >= 0
    )
    assert int(
        ((by_code.get("ERX2") or {}).get("er_exit_filter_trigger_count") or 0) >= 0
    )


def test_api_trend_single_er_filter_rejects_invalid_threshold(api_client):
    c = api_client
    err = post_json(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "sma_window": 2,
            "er_filter": True,
            "er_window": 10,
            "er_threshold": 1.2,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
        expected_status=422,
    )
    assert isinstance(err, dict)


def test_api_trend_single_ma_entry_filter_rejects_invalid_type(api_client):
    c = api_client
    err = post_json(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "sma_window": 2,
            "ma_entry_filter_enabled": True,
            "ma_entry_filter_type": "wma",
            "ma_entry_filter_fast": 100,
            "ma_entry_filter_slow": 200,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
        expected_status=422,
    )
    assert isinstance(err, dict)


def test_api_trend_portfolio_er_filter_rejects_invalid_window(api_client):
    c = api_client
    err = post_json(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "sma_window": 2,
            "position_sizing": "equal",
            "er_filter": True,
            "er_window": 1,
            "er_threshold": 0.3,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
        expected_status=422,
    )
    assert isinstance(err, dict)


def test_api_trend_portfolio_ma_entry_filter_rejects_fast_ge_slow(api_client):
    c = api_client
    err = post_json(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "sma_window": 2,
            "position_sizing": "equal",
            "ma_entry_filter_enabled": True,
            "ma_entry_filter_type": "sma",
            "ma_entry_filter_fast": 200,
            "ma_entry_filter_slow": 100,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
        expected_status=422,
    )
    assert isinstance(err, dict)


def test_api_trend_single_er_exit_filter_rejects_invalid_threshold(api_client):
    c = api_client
    err = post_json(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "sma_window": 2,
            "er_exit_filter": True,
            "er_exit_window": 10,
            "er_exit_threshold": 1.2,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
        expected_status=422,
    )
    assert isinstance(err, dict)


def test_api_trend_portfolio_er_exit_filter_rejects_invalid_window(api_client):
    c = api_client
    err = post_json(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "sma_window": 2,
            "position_sizing": "equal",
            "er_exit_filter": True,
            "er_exit_window": 1,
            "er_exit_threshold": 0.88,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
        expected_status=422,
    )
    assert isinstance(err, dict)


def test_api_trend_single_kama_rejects_fast_ge_slow(api_client):
    c = api_client
    err = post_json(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_filter",
            "ma_type": "kama",
            "sma_window": 20,
            "kama_er_window": 10,
            "kama_fast_window": 30,
            "kama_slow_window": 30,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
        expected_status=400,
    )
    assert isinstance(err, dict)
    assert "kama_fast_window must be < kama_slow_window" in str(
        (err or {}).get("detail") or ""
    )


def test_api_trend_single_ma_cross_rejects_kama_type(api_client):
    c = api_client
    err = post_json(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240103",
            "strategy": "ma_cross",
            "ma_type": "kama",
            "fast_window": 10,
            "slow_window": 30,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
        expected_status=400,
    )
    assert isinstance(err, dict)
    assert "ma_type=kama is only supported for ma_filter" in str(
        (err or {}).get("detail") or ""
    )


def test_api_trend_single_allows_r_take_profit_and_scaleout_together(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=["510300"],
        names={"510300": "沪深300ETF"},
        start_date="20240102",
        end_date="20240110",
    )
    out = post_json_ok(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240110",
            "strategy": "ma_filter",
            "sma_window": 5,
            "r_take_profit_enabled": True,
            "r_profit_scaleout_enabled": True,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    params = ((out.get("meta") or {}).get("params")) or {}
    assert bool(params.get("r_take_profit_enabled")) is True
    assert bool(params.get("r_profit_scaleout_enabled")) is True


def test_api_trend_portfolio_allows_r_take_profit_and_scaleout_together(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=["510300", "159915"],
        names={"510300": "沪深300ETF", "159915": "创业板ETF"},
        start_date="20240102",
        end_date="20240110",
    )
    out = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": ["510300", "159915"],
            "start": "20240102",
            "end": "20240110",
            "strategy": "ma_filter",
            "sma_window": 5,
            "position_sizing": "equal",
            "r_take_profit_enabled": True,
            "r_profit_scaleout_enabled": True,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    params = ((out.get("meta") or {}).get("params")) or {}
    assert bool(params.get("r_take_profit_enabled")) is True
    assert bool(params.get("r_profit_scaleout_enabled")) is True


def test_api_trend_single_bt_engine_contract(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240110",
    )
    data = post_json_ok(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240110",
            "strategy": "ma_filter",
            "sma_window": 2,
            "engine": "bt",
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    assert str((((data or {}).get("meta") or {}).get("engine") or "")).lower() == "bt"
    assert "nav" in data and "series" in data["nav"]
    assert "STRAT" in (data["nav"]["series"] or {})
    assert "BUY_HOLD" in (data["nav"]["series"] or {})
    assert "EXCESS" in (data["nav"]["series"] or {})


def test_api_trend_portfolio_bt_engine_contract(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240110",
    )
    data = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240110",
            "strategy": "ma_filter",
            "sma_window": 2,
            "engine": "bt",
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    assert str((((data or {}).get("meta") or {}).get("engine") or "")).lower() == "bt"
    assert "nav" in data and "series" in data["nav"]
    assert "STRAT" in (data["nav"]["series"] or {})
    assert "BUY_HOLD" in (data["nav"]["series"] or {})
    assert "EXCESS" in (data["nav"]["series"] or {})


def test_api_trend_portfolio_oos_bootstrap_bt_engine_contract(api_client, engine):
    c = api_client
    dates = [d.date() for d in pd.date_range("2023-01-02", periods=180, freq="B")]
    seed_prices(
        engine,
        code_to_series={
            "A": [
                100.0 + i * 0.20 + ((i % 17) - 8) * 0.08 for i, _ in enumerate(dates)
            ],
            "B": [95.0 + i * 0.15 + ((i % 13) - 6) * 0.07 for i, _ in enumerate(dates)],
        },
        dates=dates,
    )
    data = post_json_ok(
        c,
        "/api/analysis/trend/portfolio/oos-bootstrap",
        {
            "codes": ["A", "B"],
            "start": dates[0].strftime("%Y%m%d"),
            "end": dates[-1].strftime("%Y%m%d"),
            "strategy": "ma_filter",
            "n_bootstrap": 5,
            "block_size": 10,
            "oos_ratio": 0.3,
            "exec_price": "close",
            "engine": "bt",
            "param_grid": {
                "sma_window": [20],
                "ma_type": ["sma"],
            },
        },
    )
    assert "error" not in data
    assert str(((data.get("meta") or {}).get("engine") or "")).lower() == "bt"
    assert str(data.get("engine") or "").lower() == "bt"
    assert str(data.get("oos_eval_engine") or "").lower() == "bt"
    assert str(data.get("bootstrap_eval_engine") or "").lower() == "bt"
    assert isinstance(data.get("limitations"), list)


def test_api_trend_engine_uses_server_default_when_request_missing(
    api_client, monkeypatch
):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240110",
    )
    monkeypatch.setenv("MOMENTUM_TREND_BACKTEST_ENGINE", "bt")
    data = post_json_ok(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240110",
            "strategy": "ma_filter",
            "sma_window": 2,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    assert str((((data or {}).get("meta") or {}).get("engine") or "")).lower() == "bt"
    assert (
        str((((data or {}).get("meta") or {}).get("engine_default") or "")).lower()
        == "bt"
    )


def test_api_trend_portfolio_engine_uses_server_default_when_request_missing(
    api_client, monkeypatch
):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240110",
    )
    monkeypatch.setenv("MOMENTUM_TREND_BACKTEST_ENGINE", "bt")
    data = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240110",
            "strategy": "ma_filter",
            "sma_window": 2,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    assert str((((data or {}).get("meta") or {}).get("engine") or "")).lower() == "bt"
    assert (
        str((((data or {}).get("meta") or {}).get("engine_default") or "")).lower()
        == "bt"
    )


def test_api_trend_engine_falls_back_to_legacy_when_server_default_invalid(
    api_client, monkeypatch
):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240110",
    )
    monkeypatch.setenv("MOMENTUM_TREND_BACKTEST_ENGINE", "bad_engine")
    data = post_json_ok(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240110",
            "strategy": "ma_filter",
            "sma_window": 2,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    assert (
        str((((data or {}).get("meta") or {}).get("engine") or "")).lower() == "legacy"
    )
    assert (
        str((((data or {}).get("meta") or {}).get("engine_default") or "")).lower()
        == "legacy"
    )


def test_api_trend_portfolio_engine_falls_back_to_legacy_when_server_default_invalid(
    api_client, monkeypatch
):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240110",
    )
    monkeypatch.setenv("MOMENTUM_TREND_BACKTEST_ENGINE", "bad_engine")
    data = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240110",
            "strategy": "ma_filter",
            "sma_window": 2,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    assert (
        str((((data or {}).get("meta") or {}).get("engine") or "")).lower() == "legacy"
    )
    assert (
        str((((data or {}).get("meta") or {}).get("engine_default") or "")).lower()
        == "legacy"
    )


def test_api_trend_single_rejects_invalid_engine(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240110",
    )
    err = post_json(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240110",
            "strategy": "ma_filter",
            "sma_window": 2,
            "engine": "invalid",
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
        expected_status=400,
    )
    assert "engine must be one of: legacy|bt" in str((err or {}).get("detail") or "")


def test_api_trend_portfolio_rejects_invalid_engine(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240110",
    )
    err = post_json(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240110",
            "strategy": "ma_filter",
            "sma_window": 2,
            "engine": "invalid",
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
        expected_status=400,
    )
    assert "engine must be one of: legacy|bt" in str((err or {}).get("detail") or "")


def test_api_trend_oos_bootstrap_engine_uses_server_default_when_missing(
    api_client, engine, monkeypatch
):
    c = api_client
    dates = [d.date() for d in pd.date_range("2023-01-02", periods=180, freq="B")]
    seed_prices(
        engine,
        code_to_series={
            "A": [
                100.0 + i * 0.20 + ((i % 17) - 8) * 0.08 for i, _ in enumerate(dates)
            ],
            "B": [95.0 + i * 0.15 + ((i % 13) - 6) * 0.07 for i, _ in enumerate(dates)],
        },
        dates=dates,
    )
    monkeypatch.setenv("MOMENTUM_TREND_BACKTEST_ENGINE", "bt")
    data = post_json_ok(
        c,
        "/api/analysis/trend/portfolio/oos-bootstrap",
        {
            "codes": ["A", "B"],
            "start": dates[0].strftime("%Y%m%d"),
            "end": dates[-1].strftime("%Y%m%d"),
            "strategy": "ma_filter",
            "n_bootstrap": 5,
            "block_size": 10,
            "oos_ratio": 0.3,
            "exec_price": "close",
            "param_grid": {
                "sma_window": [20],
                "ma_type": ["sma"],
            },
        },
    )
    assert str(((data.get("meta") or {}).get("engine") or "")).lower() == "bt"
    assert str(((data.get("meta") or {}).get("engine_default") or "")).lower() == "bt"
    assert str(data.get("engine") or "").lower() == "bt"


def test_api_trend_oos_bootstrap_falls_back_to_legacy_when_server_default_invalid(
    api_client, engine, monkeypatch
):
    c = api_client
    dates = [d.date() for d in pd.date_range("2023-01-02", periods=180, freq="B")]
    seed_prices(
        engine,
        code_to_series={
            "A": [
                100.0 + i * 0.20 + ((i % 17) - 8) * 0.08 for i, _ in enumerate(dates)
            ],
            "B": [95.0 + i * 0.15 + ((i % 13) - 6) * 0.07 for i, _ in enumerate(dates)],
        },
        dates=dates,
    )
    monkeypatch.setenv("MOMENTUM_TREND_BACKTEST_ENGINE", "bad_engine")
    data = post_json_ok(
        c,
        "/api/analysis/trend/portfolio/oos-bootstrap",
        {
            "codes": ["A", "B"],
            "start": dates[0].strftime("%Y%m%d"),
            "end": dates[-1].strftime("%Y%m%d"),
            "strategy": "ma_filter",
            "n_bootstrap": 5,
            "block_size": 10,
            "oos_ratio": 0.3,
            "exec_price": "close",
            "param_grid": {
                "sma_window": [20],
                "ma_type": ["sma"],
            },
        },
    )
    assert str(((data.get("meta") or {}).get("engine") or "")).lower() == "legacy"
    assert (
        str(((data.get("meta") or {}).get("engine_default") or "")).lower() == "legacy"
    )
    assert str(data.get("engine") or "").lower() == "legacy"


def test_api_trend_single_request_engine_overrides_invalid_server_default(
    api_client, monkeypatch
):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240110",
    )
    monkeypatch.setenv("MOMENTUM_TREND_BACKTEST_ENGINE", "bad_engine")
    data = post_json_ok(
        c,
        "/api/analysis/trend",
        {
            "code": "510300",
            "start": "20240102",
            "end": "20240110",
            "strategy": "ma_filter",
            "sma_window": 2,
            "engine": "bt",
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    assert str((((data or {}).get("meta") or {}).get("engine") or "")).lower() == "bt"
    assert (
        str((((data or {}).get("meta") or {}).get("engine_default") or "")).lower()
        == "legacy"
    )


def test_api_trend_portfolio_request_engine_overrides_invalid_server_default(
    api_client, monkeypatch
):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240110",
    )
    monkeypatch.setenv("MOMENTUM_TREND_BACKTEST_ENGINE", "bad_engine")
    data = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240110",
            "strategy": "ma_filter",
            "sma_window": 2,
            "engine": "bt",
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
        },
    )
    assert str((((data or {}).get("meta") or {}).get("engine") or "")).lower() == "bt"
    assert (
        str((((data or {}).get("meta") or {}).get("engine_default") or "")).lower()
        == "legacy"
    )


def test_api_trend_oos_request_engine_overrides_invalid_server_default(
    api_client, engine, monkeypatch
):
    c = api_client
    dates = [d.date() for d in pd.date_range("2023-01-02", periods=180, freq="B")]
    seed_prices(
        engine,
        code_to_series={
            "A": [
                100.0 + i * 0.20 + ((i % 17) - 8) * 0.08 for i, _ in enumerate(dates)
            ],
            "B": [95.0 + i * 0.15 + ((i % 13) - 6) * 0.07 for i, _ in enumerate(dates)],
        },
        dates=dates,
    )
    monkeypatch.setenv("MOMENTUM_TREND_BACKTEST_ENGINE", "bad_engine")
    data = post_json_ok(
        c,
        "/api/analysis/trend/portfolio/oos-bootstrap",
        {
            "codes": ["A", "B"],
            "start": dates[0].strftime("%Y%m%d"),
            "end": dates[-1].strftime("%Y%m%d"),
            "strategy": "ma_filter",
            "n_bootstrap": 5,
            "block_size": 10,
            "oos_ratio": 0.3,
            "exec_price": "close",
            "engine": "bt",
            "param_grid": {
                "sma_window": [20],
                "ma_type": ["sma"],
            },
        },
    )
    assert str(((data.get("meta") or {}).get("engine") or "")).lower() == "bt"
    assert (
        str(((data.get("meta") or {}).get("engine_default") or "")).lower() == "legacy"
    )
    assert str(data.get("engine") or "").lower() == "bt"


def test_api_trend_oos_bootstrap_rejects_invalid_engine(api_client, engine):
    c = api_client
    dates = [d.date() for d in pd.date_range("2023-01-02", periods=180, freq="B")]
    seed_prices(
        engine,
        code_to_series={
            "A": [
                100.0 + i * 0.20 + ((i % 17) - 8) * 0.08 for i, _ in enumerate(dates)
            ],
            "B": [95.0 + i * 0.15 + ((i % 13) - 6) * 0.07 for i, _ in enumerate(dates)],
        },
        dates=dates,
    )
    err = post_json(
        c,
        "/api/analysis/trend/portfolio/oos-bootstrap",
        {
            "codes": ["A", "B"],
            "start": dates[0].strftime("%Y%m%d"),
            "end": dates[-1].strftime("%Y%m%d"),
            "strategy": "ma_filter",
            "n_bootstrap": 5,
            "block_size": 10,
            "oos_ratio": 0.3,
            "exec_price": "close",
            "engine": "invalid",
            "param_grid": {
                "sma_window": [20],
                "ma_type": ["sma"],
            },
        },
        expected_status=400,
    )
    assert "engine must be one of: legacy|bt" in str((err or {}).get("detail") or "")


def test_api_trend_oos_bootstrap_bt_legacy_consistency(api_client, engine):
    c = api_client
    # Keep enough history for block bootstrap (block_size 10, oos_ratio 0.3) while avoiding
    # oversized payloads that dominated suite runtime (~200s+ per call before tuning).
    dates = [d.date() for d in pd.date_range("2023-01-02", periods=126, freq="B")]
    seed_prices(
        engine,
        code_to_series={
            "A": [
                100.0 + i * 0.20 + ((i % 17) - 8) * 0.08 for i, _ in enumerate(dates)
            ],
            "B": [95.0 + i * 0.15 + ((i % 13) - 6) * 0.07 for i, _ in enumerate(dates)],
            "C": [90.0 + i * 0.18 + ((i % 11) - 5) * 0.06 for i, _ in enumerate(dates)],
        },
        dates=dates,
    )
    body = {
        "codes": ["A", "B", "C"],
        "start": dates[0].strftime("%Y%m%d"),
        "end": dates[-1].strftime("%Y%m%d"),
        "strategy": "ma_filter",
        "n_bootstrap": 5,
        "block_size": 10,
        "oos_ratio": 0.3,
        "seed": 7,
        "exec_price": "close",
        "cost_bps": 2.0,
        "param_grid": {
            "sma_window": [20, 30],
            "ma_type": ["sma"],
        },
    }
    legacy = post_json_ok(
        c,
        "/api/analysis/trend/portfolio/oos-bootstrap",
        {**body, "engine": "legacy"},
    )
    bt = post_json_ok(
        c,
        "/api/analysis/trend/portfolio/oos-bootstrap",
        {**body, "engine": "bt"},
    )

    assert legacy.get("chosen_params") == bt.get("chosen_params")
    l_m = legacy.get("oos_metrics") or {}
    b_m = bt.get("oos_metrics") or {}
    for key, tol in {
        "cumulative_return": 1e-6,
        "annualized_return": 1e-6,
        "max_drawdown": 1e-6,
        "sharpe_ratio": 1e-1,
    }.items():
        lv = l_m.get(key)
        bv = b_m.get(key)
        assert isinstance(lv, (int, float)) and isinstance(bv, (int, float))
        assert abs(float(bv) - float(lv)) <= float(tol)


def test_api_rotation_backtest_accepts_floating_topk_mode(api_client) -> None:
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240103",
    )
    data = post_json_ok(
        c,
        "/api/analysis/rotation",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240103",
            "rebalance": "monthly",
            "top_k_mode": "floating",
            "floating_benchmark_code": "510300",
            "top_k": 0,
            "lookback_days": 1,
            "skip_days": 0,
            "risk_off": False,
            "risk_free_rate": 0.025,
            "cost_bps": 0.0,
        },
    )
    assert str(data.get("top_k_mode") or "") == "floating"
    assert str(data.get("floating_benchmark_code") or "") == "510300"


def test_api_rotation_next_execution_plan_happy_path(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240103",
    )

    data = post_json_ok(
        c,
        "/api/analysis/rotation/next-execution-plan",
        _make_next_execution_plan_payload(
            codes=_BASELINE_CODES,
            start="20240102",
            end="20240103",
            asof="20240102",
        ),
    )
    assert "next_trading_day" in data
    assert "has_execution_plan" in data
    assert "plan" in data
    if data["has_execution_plan"]:
        assert "target_weights" in data["plan"]


def test_api_rotation_next_execution_plan_keeps_explicit_empty_picks(
    api_client, monkeypatch
):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240103",
    )

    import etf_momentum.strategy.rotation as rot_mod

    def _fake_backtest_rotation(*_args, **_kwargs):
        # Explicitly empty picks should remain empty in API response,
        # and must not be backfilled from target_weights.
        return {
            "holdings": [
                {
                    "start_date": "2024-01-03",
                    "decision_date": "2024-01-02",
                    "mode": "cash",
                    "picks": [],
                    "scores": {},
                }
            ],
            "period_details": [
                {"start_date": "2024-01-03", "buys": [], "sells": [], "turnover": 0.0}
            ],
            "weights_end": {"weights": {"510300": 1.0}},
        }

    monkeypatch.setattr(rot_mod, "backtest_rotation", _fake_backtest_rotation)

    data = post_json_ok(
        c,
        "/api/analysis/rotation/next-execution-plan",
        _make_next_execution_plan_payload(
            codes=_BASELINE_CODES,
            start="20240102",
            end="20240103",
            asof="20240102",
        ),
    )
    assert data["has_execution_plan"] is True
    assert data["plan"]["picks"] == []
    assert (
        data["plan"]["target_weights"]
        and data["plan"]["target_weights"][0]["code"] == "510300"
    )


def test_api_rotation_next_plan_keeps_explicit_empty_picks(api_client, monkeypatch):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=["159915", "511010", "513100", "518880"],
        names={
            "159915": "创业板",
            "511010": "国债",
            "513100": "纳指",
            "518880": "黄金",
        },
        start_date="20240102",
        end_date="20240103",
    )

    import etf_momentum.strategy.rotation as rot_mod

    def _fake_backtest_rotation(*_args, **_kwargs):
        return {
            "holdings": [
                {
                    "start_date": "2024-01-03",
                    "decision_date": "2024-01-02",
                    "mode": "cash",
                    "picks": [],
                    "scores": {},
                }
            ],
            "weights_end": {"weights": {"159915": 1.0}},
        }

    monkeypatch.setattr(rot_mod, "backtest_rotation", _fake_backtest_rotation)

    data = post_json_ok(
        c,
        "/api/analysis/rotation/next-plan",
        {"asof": "20240102", "anchor_weekday": 3},
    )
    assert data["rebalance_effective_next_day"] is True
    assert data["pick_code"] is None
    assert data["pick_name"] == "现金"
    assert float(data["pick_exposure"]) == 0.0


@pytest.mark.parametrize(
    "entry_match_n,entry_backfill,expect_empty,expect_codes",
    [
        (0, False, True, set()),
        (1, False, False, {"A", "C", "D"}),
    ],
)
def test_api_rotation_next_execution_plan_entry_param_matrix(
    api_client,
    engine,
    entry_match_n: int,
    entry_backfill: bool,
    expect_empty: bool,
    expect_codes: set[str],
):
    dates, series = build_rotation_case_series()
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    base = make_rotation_base_payload(
        codes=["A", "B", "C", "D", "E"], dates=dates, rebalance="daily"
    )
    data = post_json_ok(
        c,
        "/api/analysis/rotation/next-execution-plan",
        {
            **base,
            "asof": fmt_ymd(dates[-2]),
            "exec_price": "open",
            "entry_backfill": bool(entry_backfill),
            "entry_match_n": int(entry_match_n),
            "trend_filter": True,
            "bias_filter": True,
            "asset_trend_rules": [make_trend_rule(stage="entry")],
            "asset_bias_rules": [
                make_bias_rule(stage="entry", op="<=", fixed_value=1.5)
            ],
        },
    )
    plan = data["plan"]
    picks = plan.get("picks") or []

    if expect_empty:
        assert picks == []
    else:
        assert picks
        assert any(p in expect_codes for p in picks)


def test_api_rotation_next_execution_plan_entry_backfill_recovers_slots(
    api_client, engine
):
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-07-31", freq="B")]

    def _spike(
        base: float, slope: float, spike_start: int, spike_slope: float
    ) -> list[float]:
        out: list[float] = []
        for i, _ in enumerate(dates):
            v = base + i * slope
            if i >= spike_start:
                v += (i - spike_start) * spike_slope
            out.append(float(v))
        return out

    # A/C: high momentum and high BIAS -> fail strict entry bias
    # D: still in top-3 momentum but lower BIAS -> pass
    # B/E: lower momentum but pass entry bias -> candidates for backfill
    series = {
        "A": _spike(100.0, 0.12, 115, 1.8),
        "C": _spike(100.0, 0.11, 116, 1.7),
        "D": _spike(100.0, 0.10, 117, 0.35),
        "B": [100.0 + i * 0.04 for i, _ in enumerate(dates)],
        "E": [100.0 + i * 0.03 for i, _ in enumerate(dates)],
    }
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    common = {
        **make_rotation_base_payload(
            codes=["A", "B", "C", "D", "E"], dates=dates, rebalance="daily"
        ),
        "asof": fmt_ymd(dates[-2]),
        "exec_price": "open",
        "entry_match_n": 0,  # strict AND on enabled entry filters
        "trend_filter": True,
        "bias_filter": True,
        "asset_trend_rules": [make_trend_rule(stage="entry")],
        "asset_bias_rules": [make_bias_rule(stage="entry", op="<=", fixed_value=1.5)],
    }

    d_off = post_json_ok(
        c,
        "/api/analysis/rotation/next-execution-plan",
        {**common, "entry_backfill": False},
    )
    d_on = post_json_ok(
        c,
        "/api/analysis/rotation/next-execution-plan",
        {**common, "entry_backfill": True},
    )
    picks_off = (d_off.get("plan") or {}).get("picks") or []
    picks_on = (d_on.get("plan") or {}).get("picks") or []
    assert len(picks_on) >= len(picks_off)
    assert len(picks_on) > len(picks_off)


def test_api_rotation_next_execution_plan_trace_includes_scores_and_entry_checks_with_default_rules(
    api_client, engine
):
    dates, series = build_rotation_case_series()
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/rotation/next-execution-plan",
        {
            **make_rotation_base_payload(
                codes=["A", "B", "C", "D", "E"], dates=dates, rebalance="daily"
            ),
            "asof": fmt_ymd(dates[-2]),
            "exec_price": "open",
            "trend_filter": True,
            "trend_sma_window": 5,
            # intentionally do not pass asset_trend_rules to test default-rule trace path
        },
    )
    trace = (data.get("plan") or {}).get("trace") or {}
    m = trace.get("momentum_scores") or {}
    entry = (trace.get("entry_filtering") or {}).get("entry_checks_by_code") or {}
    assert m, "trace.momentum_scores should not be empty"
    assert entry, "trace.entry_filtering.entry_checks_by_code should not be empty"
    one = next(iter(entry.values()))
    by_filter = (one.get("by_filter") or {}) if isinstance(one, dict) else {}
    assert "trend" in by_filter
    assert "ok_gate" in one


def test_api_rotation_next_execution_plan_trace_includes_exit_checks_without_trigger_events(
    api_client, monkeypatch
):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240103",
    )

    import etf_momentum.strategy.rotation as rot_mod

    def _fake_backtest_rotation(*_args, **_kwargs):
        return {
            "holdings": [
                {
                    "start_date": "2024-01-03",
                    "decision_date": "2024-01-02",
                    "mode": "risk_on",
                    "picks": ["510300"],
                    "scores": {"510300": 0.02},
                    "risk_controls": {
                        "reasons": [],
                        "score_by_code": {"510300": 0.02},
                        "candidate_ranked": ["510300"],
                    },
                    "daily_exit": {
                        "gate": {"enabled_count": 2, "required": 2, "mode": "and"},
                        "checks_by_day": [
                            {
                                "decision_date": "2024-01-02",
                                "execution_date": "2024-01-03",
                                "checks": [
                                    {
                                        "code": "510300",
                                        "hit_count": 1,
                                        "required": 2,
                                        "hit_conditions": ["trend_rule"],
                                        "triggered": False,
                                        "by_filter": {
                                            "momentum_rule": False,
                                            "trend_rule": True,
                                            "bias_rule": False,
                                        },
                                    }
                                ],
                            }
                        ],
                        "events": [],
                    },
                }
            ],
            "period_details": [
                {"start_date": "2024-01-03", "buys": [], "sells": [], "turnover": 0.0}
            ],
            "daily_exit_events": [],
            "weights_end": {"weights": {"510300": 1.0}},
        }

    monkeypatch.setattr(rot_mod, "backtest_rotation", _fake_backtest_rotation)

    data = post_json_ok(
        c,
        "/api/analysis/rotation/next-execution-plan",
        _make_next_execution_plan_payload(
            codes=_BASELINE_CODES,
            start="20240102",
            end="20240103",
            asof="20240102",
        ),
    )
    checks = (
        ((data.get("plan") or {}).get("trace") or {}).get("exit_checks") or {}
    ).get("execution_day_checks") or []
    assert checks
    assert checks[0]["code"] == "510300"
    assert checks[0]["triggered"] is False


@pytest.mark.parametrize(
    "entry_match_n,expect_cash",
    [
        (0, True),
        (1, False),
    ],
)
def test_api_rotation_next_plan_entry_param_matrix_mini_program(
    api_client,
    engine,
    entry_match_n: int,
    expect_cash: bool,
):
    dates, src = build_rotation_case_series()
    # Map to fixed mini-program universe codes.
    mapped = map_case_series_to_miniprogram_codes(src)
    seed_prices(engine, code_to_series=mapped, dates=dates)

    c = api_client
    # Ensure next trading day weekday matches anchor tab (Wednesday here).
    asof = dt.date(2024, 7, 2)  # Tue -> next trading day Wed(2)
    data = post_json_ok(
        c,
        "/api/analysis/rotation/next-plan",
        {
            "asof": fmt_ymd(asof),
            "anchor_weekday": 3,
            **make_rotation_base_payload(
                codes=["159915", "511010", "513100", "518880"], dates=dates
            ),
            "entry_backfill": False,
            "entry_match_n": int(entry_match_n),
            "trend_filter": True,
            "bias_filter": True,
            "asset_trend_rules": [make_trend_rule(stage="entry")],
            "asset_bias_rules": [
                make_bias_rule(stage="entry", op="<=", fixed_value=1.5)
            ],
        },
    )
    assert data["rebalance_effective_next_day"] is True
    if expect_cash:
        assert data["pick_code"] is None
        assert data["pick_name"] == "现金"
        assert float(data["pick_exposure"]) == 0.0
    else:
        assert data["pick_code"] is not None
        assert float(data["pick_exposure"]) > 0.0


def test_api_trend_ma_trailing_stop_contract_and_validation(engine, api_client):
    dates = [d.date() for d in pd.date_range("2023-01-03", periods=180, freq="B")]
    _seed_trend_capacity_prices(
        engine,
        code_to_series={
            "MATS1": [
                110.0 - i * 0.08 + ((i % 9) - 4) * 0.03 for i in range(len(dates))
            ]
        },
        dates=dates,
        with_amount=True,
    )
    c = api_client
    payload = {
        "code": "MATS1",
        "start": fmt_ymd(dates[0]),
        "end": fmt_ymd(dates[-1]),
        "strategy": "ma_filter",
        "sma_window": 20,
        "exec_price": "close",
        "position_sizing": "equal",
        "ma_trailing_stop_enabled": True,
        "ma_trailing_stop_ma_type": "ema",
        "ma_trailing_stop_execution_mode": "next_day",
        "ma_trailing_stop_effective_delay_days": 3,
        "ma_trailing_stop_reduce_window": 10,
        "ma_trailing_stop_exit_window": 20,
        "ma_trailing_stop_reduce_fraction": 0.33,
        "cost_bps": 0.0,
        "slippage_rate": 0.0,
        "quick_mode": True,
    }
    out = post_json_ok(c, "/api/analysis/trend", payload)
    params = ((out.get("meta") or {}).get("params")) or {}
    assert bool(params.get("ma_trailing_stop_enabled")) is True
    assert str(params.get("ma_trailing_stop_ma_type") or "") == "ema"
    assert str(params.get("ma_trailing_stop_execution_mode") or "") == "next_day"
    assert int(params.get("ma_trailing_stop_effective_delay_days") or 0) == 3
    rc = ((out.get("risk_controls") or {}).get("ma_trailing_stop")) or {}
    assert bool(rc.get("enabled")) is True
    assert str(rc.get("reduce_fraction_basis") or "") == "initial_position"
    assert int(rc.get("effective_delay_days") or 0) == 3

    bad = dict(payload)
    bad["ma_trailing_stop_ma_type"] = "bad_ma"
    resp = c.post("/api/analysis/trend", json=bad)
    assert resp.status_code == 400
    bad_delay = dict(payload)
    bad_delay["ma_trailing_stop_effective_delay_days"] = 0
    resp_delay = c.post("/api/analysis/trend", json=bad_delay)
    assert resp_delay.status_code == 422


def test_api_trend_allows_r_take_profit_and_scaleout_together(engine, api_client):
    dates = [d.date() for d in pd.date_range("2022-01-03", periods=220, freq="B")]
    _seed_trend_capacity_prices(
        engine,
        code_to_series={
            "OR1": [100.0 + i * 0.06 + ((i % 9) - 4) * 0.04 for i in range(len(dates))],
            "OR2": [95.0 + i * 0.05 + ((i % 11) - 5) * 0.03 for i in range(len(dates))],
        },
        dates=dates,
        with_amount=True,
    )
    c = api_client
    out = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": ["OR1", "OR2"],
            "start": fmt_ymd(dates[0]),
            "end": fmt_ymd(dates[-1]),
            "strategy": "ma_filter",
            "sma_window": 15,
            "position_sizing": "equal",
            "exec_price": "close",
            "atr_stop_mode": "none",
            "r_take_profit_enabled": True,
            "r_take_profit_execution_mode": "intraday",
            "r_take_profit_tiers": [{"r_multiple": 50.0, "retrace_ratio": 0.2}],
            "r_profit_scaleout_enabled": True,
            "r_profit_scaleout_execution_mode": "intraday",
            "r_profit_scaleout_tiers": [{"r_multiple": 50.0, "reduce_fraction": 0.3}],
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
            "quick_mode": True,
        },
    )
    params = ((out.get("meta") or {}).get("params")) or {}
    assert bool(params.get("r_take_profit_enabled")) is True
    assert bool(params.get("r_profit_scaleout_enabled")) is True


def test_api_trend_portfolio_ma_trailing_stop_delay_param(engine, api_client):
    dates = [d.date() for d in pd.date_range("2023-01-03", periods=180, freq="B")]
    _seed_trend_capacity_prices(
        engine,
        code_to_series={
            "MDP1": [
                100.0 + i * 0.03 + ((i % 7) - 3) * 0.04 for i in range(len(dates))
            ],
            "MDP2": [98.0 + i * 0.04 + ((i % 9) - 4) * 0.05 for i in range(len(dates))],
        },
        dates=dates,
        with_amount=True,
    )
    c = api_client
    out = post_json_ok(
        c,
        "/api/analysis/trend/portfolio",
        {
            "codes": ["MDP1", "MDP2"],
            "start": fmt_ymd(dates[0]),
            "end": fmt_ymd(dates[-1]),
            "strategy": "ma_filter",
            "sma_window": 20,
            "position_sizing": "equal",
            "exec_price": "close",
            "ma_trailing_stop_enabled": True,
            "ma_trailing_stop_ma_type": "sma",
            "ma_trailing_stop_execution_mode": "intraday",
            "ma_trailing_stop_effective_delay_days": 3,
            "ma_trailing_stop_reduce_window": 10,
            "ma_trailing_stop_exit_window": 20,
            "ma_trailing_stop_reduce_fraction": 0.33,
            "cost_bps": 0.0,
            "slippage_rate": 0.0,
            "quick_mode": True,
        },
    )
    params = ((out.get("meta") or {}).get("params")) or {}
    assert int(params.get("ma_trailing_stop_effective_delay_days") or 0) == 3
    rc = ((out.get("risk_controls") or {}).get("ma_trailing_stop")) or {}
    assert int(rc.get("effective_delay_days") or 0) == 3
