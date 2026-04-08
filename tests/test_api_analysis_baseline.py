import datetime as dt

import pandas as pd
import pytest

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
    assert data["metrics"]["risk_free_rate"] == 0.02
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
    params = (((data or {}).get("meta") or {}).get("params") or {})
    assert str(params.get("position_sizing") or "") == "risk_budget"
    assert int(params.get("risk_budget_atr_window") or 0) == 2
    assert float(params.get("risk_budget_pct") or 0.0) == pytest.approx(0.01, rel=0.0, abs=1e-12)


def test_api_trend_single_er_filter_contract(engine, api_client):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=80, freq="B")]
    series = {"ER1": [100.0 + ((-1.0) ** i) * 0.8 + i * 0.01 for i, _ in enumerate(dates)]}
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
    out_no_filter = post_json_ok(c, "/api/analysis/trend", {**base_payload, "er_filter": False})
    out_with_filter = post_json_ok(
        c,
        "/api/analysis/trend",
        {**base_payload, "er_filter": True, "er_window": 10, "er_threshold": 0.8},
    )

    pos_no_filter = [float(x) for x in ((out_no_filter.get("signals") or {}).get("position") or [])]
    pos_with_filter = [float(x) for x in ((out_with_filter.get("signals") or {}).get("position") or [])]
    assert any(x > 0.0 for x in pos_no_filter)
    assert all(x == 0.0 for x in pos_with_filter)
    params = (((out_with_filter or {}).get("meta") or {}).get("params") or {})
    assert params.get("er_filter") is True
    assert int(params.get("er_window") or 0) == 10
    assert float(params.get("er_threshold") or 0.0) == pytest.approx(0.8, rel=0.0, abs=1e-12)


def test_api_trend_single_kama_contract(engine, api_client):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=80, freq="B")]
    series = {"KAMA1": [100.0 + (i * 0.4 if i < 40 else (40 * 0.4) - (i - 40) * 0.45) for i, _ in enumerate(dates)]}
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
    params = (((out or {}).get("meta") or {}).get("params") or {})
    assert str(params.get("ma_type") or "") == "kama"
    assert int(params.get("kama_er_window") or 0) == 10
    assert int(params.get("kama_fast_window") or 0) == 2
    assert int(params.get("kama_slow_window") or 0) == 30


def test_api_trend_portfolio_er_filter_contract(engine, api_client):
    dates = [d.date() for d in pd.date_range("2024-01-01", periods=90, freq="B")]
    series = {
        "ER1": [100.0 + ((-1.0) ** i) * 0.8 + i * 0.01 for i, _ in enumerate(dates)],
        "ER2": [90.0 + ((-1.0) ** (i + 1)) * 0.7 + i * 0.01 for i, _ in enumerate(dates)],
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
    out_no_filter = post_json_ok(c, "/api/analysis/trend/portfolio", {**base_payload, "er_filter": False})
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
    params = (((out_with_filter or {}).get("meta") or {}).get("params") or {})
    assert params.get("er_filter") is True
    assert int(params.get("er_window") or 0) == 10
    assert float(params.get("er_threshold") or 0.0) == pytest.approx(0.8, rel=0.0, abs=1e-12)


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
    assert "kama_fast_window must be < kama_slow_window" in str((err or {}).get("detail") or "")


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
    assert "ma_type=kama is only supported for ma_filter" in str((err or {}).get("detail") or "")


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


def test_api_rotation_next_execution_plan_keeps_explicit_empty_picks(api_client, monkeypatch):
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
            "period_details": [{"start_date": "2024-01-03", "buys": [], "sells": [], "turnover": 0.0}],
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
    assert data["plan"]["target_weights"] and data["plan"]["target_weights"][0]["code"] == "510300"


def test_api_rotation_next_plan_keeps_explicit_empty_picks(api_client, monkeypatch):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=["159915", "511010", "513100", "518880"],
        names={"159915": "创业板", "511010": "国债", "513100": "纳指", "518880": "黄金"},
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
    base = make_rotation_base_payload(codes=["A", "B", "C", "D", "E"], dates=dates, rebalance="daily")
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
            "asset_bias_rules": [make_bias_rule(stage="entry", op="<=", fixed_value=1.5)],
        },
    )
    plan = data["plan"]
    picks = plan.get("picks") or []

    if expect_empty:
        assert picks == []
    else:
        assert picks
        assert any(p in expect_codes for p in picks)


def test_api_rotation_next_execution_plan_entry_backfill_recovers_slots(api_client, engine):
    dates = [d.date() for d in pd.date_range("2024-01-01", "2024-07-31", freq="B")]

    def _spike(base: float, slope: float, spike_start: int, spike_slope: float) -> list[float]:
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
        **make_rotation_base_payload(codes=["A", "B", "C", "D", "E"], dates=dates, rebalance="daily"),
        "asof": fmt_ymd(dates[-2]),
        "exec_price": "open",
        "entry_match_n": 0,  # strict AND on enabled entry filters
        "trend_filter": True,
        "bias_filter": True,
        "asset_trend_rules": [make_trend_rule(stage="entry")],
        "asset_bias_rules": [make_bias_rule(stage="entry", op="<=", fixed_value=1.5)],
    }

    d_off = post_json_ok(c, "/api/analysis/rotation/next-execution-plan", {**common, "entry_backfill": False})
    d_on = post_json_ok(c, "/api/analysis/rotation/next-execution-plan", {**common, "entry_backfill": True})
    picks_off = (d_off.get("plan") or {}).get("picks") or []
    picks_on = (d_on.get("plan") or {}).get("picks") or []
    assert len(picks_on) >= len(picks_off)
    assert len(picks_on) > len(picks_off)


def test_api_rotation_next_execution_plan_trace_includes_scores_and_entry_checks_with_default_rules(api_client, engine):
    dates, series = build_rotation_case_series()
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    data = post_json_ok(
        c,
        "/api/analysis/rotation/next-execution-plan",
        {
            **make_rotation_base_payload(codes=["A", "B", "C", "D", "E"], dates=dates, rebalance="daily"),
            "asof": fmt_ymd(dates[-2]),
            "exec_price": "open",
            "trend_filter": True,
            "trend_sma_window": 5,
            # intentionally do not pass asset_trend_rules to test default-rule trace path
        },
    )
    trace = (data.get("plan") or {}).get("trace") or {}
    m = trace.get("momentum_scores") or {}
    entry = ((trace.get("entry_filtering") or {}).get("entry_checks_by_code") or {})
    assert m, "trace.momentum_scores should not be empty"
    assert entry, "trace.entry_filtering.entry_checks_by_code should not be empty"
    one = next(iter(entry.values()))
    by_filter = (one.get("by_filter") or {}) if isinstance(one, dict) else {}
    assert "trend" in by_filter
    assert "ok_gate" in one


def test_api_rotation_next_execution_plan_trace_includes_exit_checks_without_trigger_events(api_client, monkeypatch):
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
                    "risk_controls": {"reasons": [], "score_by_code": {"510300": 0.02}, "candidate_ranked": ["510300"]},
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
                                        "by_filter": {"momentum_rule": False, "trend_rule": True, "bias_rule": False},
                                    }
                                ],
                            }
                        ],
                        "events": [],
                    },
                }
            ],
            "period_details": [{"start_date": "2024-01-03", "buys": [], "sells": [], "turnover": 0.0}],
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
    checks = ((((data.get("plan") or {}).get("trace") or {}).get("exit_checks") or {}).get("execution_day_checks") or [])
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
            **make_rotation_base_payload(codes=["159915", "511010", "513100", "518880"], dates=dates),
            "entry_backfill": False,
            "entry_match_n": int(entry_match_n),
            "trend_filter": True,
            "bias_filter": True,
            "asset_trend_rules": [make_trend_rule(stage="entry")],
            "asset_bias_rules": [make_bias_rule(stage="entry", op="<=", fixed_value=1.5)],
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

