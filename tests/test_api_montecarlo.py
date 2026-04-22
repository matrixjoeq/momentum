from tests.helpers.rotation_case_data import (
    build_rotation_case_series,
    make_bias_rule,
    make_entry_exit_filters_payload,
    make_entry_filters_payload,
    make_rotation_base_payload,
    make_trend_rule,
    mc_metric_mean,
    post_json_ok,
    seed_prices,
)
from tests.helpers.api_test_client import upsert_and_fetch_etfs


_BASELINE_CODES = ["510300", "511010"]
_BASELINE_NAMES = {"510300": "沪深300", "511010": "国债"}


def test_api_baseline_montecarlo_smoke(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240101",
        end_date="20241231",
    )

    data = post_json_ok(
        c,
        "/api/analysis/baseline/montecarlo",
        {
            "codes": ["510300", "511010"],
            "start": "20240101",
            "end": "20241231",
            "benchmark_code": "510300",
            "adjust": "hfq",
            "rebalance": "weekly",
            "risk_free_rate": 0.02,
            "n_sims": 200,
            "block_size": 5,
            "seed": 1,
            "sample_window_days": 2,
        },
    )
    assert data["meta"]["type"] == "baseline"
    assert "mc" in data and "metrics" in data["mc"]
    assert "annualized_return" in data["mc"]["metrics"]


def test_api_rotation_montecarlo_smoke(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240101",
        end_date="20241231",
    )

    data = post_json_ok(
        c,
        "/api/analysis/rotation/montecarlo",
        {
            "codes": ["510300", "511010"],
            "start": "20240101",
            "end": "20241231",
            "rebalance": "weekly",
            "top_k": 1,
            "lookback_days": 20,
            "skip_days": 0,
            "risk_off": False,
            "risk_free_rate": 0.02,
            "cost_bps": 0.0,
            "n_sims": 200,
            "block_size": 5,
            "seed": 1,
            "sample_window_days": 2,
        },
    )
    assert data["meta"]["type"] == "rotation"
    assert "mc" in data and "strategy" in data["mc"] and "excess" in data["mc"]
    assert "annualized_return" in data["mc"]["strategy"]["metrics"]


def test_api_rotation_montecarlo_entry_param_combo_diff(api_client, engine):
    dates, series = build_rotation_case_series()
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    base = {
        **make_rotation_base_payload(
            codes=["A", "B", "C", "D", "E"], dates=dates, rebalance="weekly"
        ),
        "n_sims": 100,
        "block_size": 5,
        "seed": 7,
        "sample_window_days": 120,
        "trend_filter": True,
        "bias_filter": True,
        "asset_trend_rules": [make_trend_rule(stage="entry")],
        "asset_bias_rules": [make_bias_rule(stage="entry", op="<=", fixed_value=1.5)],
    }
    d_and = post_json_ok(
        c, "/api/analysis/rotation/montecarlo", {**base, "entry_match_n": 0}
    )
    d_nofm = post_json_ok(
        c, "/api/analysis/rotation/montecarlo", {**base, "entry_match_n": 1}
    )
    assert "observed_holding_len" in d_and and "observed_holding_len" in d_nofm
    ar_and = mc_metric_mean(d_and, "annualized_return")
    ar_nofm = mc_metric_mean(d_nofm, "annualized_return")
    assert ar_nofm >= ar_and


def test_api_rotation_montecarlo_exit_param_combo_diff(api_client, engine):
    dates, series = build_rotation_case_series()
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    base = {
        **make_rotation_base_payload(
            codes=["A", "B", "C", "D", "E"], dates=dates, rebalance="weekly"
        ),
        "n_sims": 100,
        "block_size": 5,
        "seed": 9,
        "sample_window_days": 120,
        "entry_match_n": 1,
        **make_entry_filters_payload(bias_fixed_value=1.5),
        "asset_trend_rules": [
            make_trend_rule(stage="entry"),
            make_trend_rule(stage="exit"),
        ],
    }
    d_off = post_json_ok(
        c, "/api/analysis/rotation/montecarlo", {**base, "trend_exit_filter": False}
    )
    d_on = post_json_ok(
        c,
        "/api/analysis/rotation/montecarlo",
        {**base, "trend_exit_filter": True, "exit_match_n": 1},
    )
    ar_off = mc_metric_mean(d_off, "annualized_return")
    ar_on = mc_metric_mean(d_on, "annualized_return")
    assert ar_on <= ar_off


def test_api_rotation_montecarlo_entry_exit_nofm_combo_diff(api_client, engine):
    dates, series = build_rotation_case_series()
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    payload = {
        **make_rotation_base_payload(
            codes=["A", "B", "C", "D", "E"], dates=dates, rebalance="weekly"
        ),
        "n_sims": 100,
        "block_size": 5,
        "seed": 11,
        "sample_window_days": 120,
        "entry_match_n": 1,
        **make_entry_exit_filters_payload(
            entry_bias_fixed_value=1.5, exit_bias_fixed_value=99.0
        ),
    }
    d_and = post_json_ok(
        c, "/api/analysis/rotation/montecarlo", {**payload, "exit_match_n": 0}
    )
    d_nofm = post_json_ok(
        c, "/api/analysis/rotation/montecarlo", {**payload, "exit_match_n": 1}
    )
    ar_and = mc_metric_mean(d_and, "annualized_return")
    ar_nofm = mc_metric_mean(d_nofm, "annualized_return")
    assert ar_nofm <= ar_and
