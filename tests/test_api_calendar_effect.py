import pytest

from tests.helpers.rotation_case_data import (
    build_rotation_case_series,
    first_grid_metric,
    make_bias_rule,
    make_entry_exit_filters_payload,
    make_entry_filters_payload,
    make_rotation_base_payload,
    make_trend_rule,
    post_json,
    post_json_ok,
    seed_prices,
)
from tests.helpers.api_test_client import upsert_and_fetch_etfs


_BASELINE_CODES = ["510300", "511010"]
_BASELINE_NAMES = {"510300": "沪深300", "511010": "国债"}


def test_api_baseline_calendar_effect(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240215",
    )

    data = post_json_ok(
        c,
        "/api/analysis/baseline/calendar-effect",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240215",
            "adjust": "hfq",
            "risk_free_rate": 0.02,
            "rebalance": "weekly",
            "anchors": [1, 5],
            "exec_prices": ["close"],
        },
    )
    assert data["meta"]["type"] == "baseline_calendar_effect"
    assert len(data["grid"]) == 2
    assert all("anchor" in x and "exec_price" in x and "ok" in x for x in data["grid"])
    ok = next((x for x in data["grid"] if x.get("ok")), None)
    assert ok is not None
    m = ok.get("metrics") or {}
    for k in [
        "calmar_ratio",
        "sortino_ratio",
        "ulcer_index",
        "ulcer_performance_index",
        "information_ratio",
    ]:
        assert k in m


def test_api_rotation_calendar_effect(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240215",
    )

    data = post_json_ok(
        c,
        "/api/analysis/rotation/calendar-effect",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240215",
            "rebalance": "weekly",
            "top_k": 1,
            "lookback_days": 5,
            "skip_days": 0,
            "risk_off": False,
            "risk_free_rate": 0.02,
            "cost_bps": 0.0,
            "anchors": [1, 5],
            "exec_prices": ["close"],
        },
    )
    assert data["meta"]["type"] == "rotation_calendar_effect"
    assert len(data["grid"]) == 2
    assert all("anchor" in x and "exec_price" in x and "ok" in x for x in data["grid"])
    ok = next((x for x in data["grid"] if x.get("ok")), None)
    assert ok is not None
    m = ok.get("metrics") or {}
    for k in [
        "calmar_ratio",
        "sortino_ratio",
        "ulcer_index",
        "ulcer_performance_index",
        "information_ratio",
    ]:
        assert k in m


def test_api_rotation_calendar_effect_accepts_equity_budget_stop_params(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240215",
    )
    data = post_json_ok(
        c,
        "/api/analysis/rotation/calendar-effect",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240215",
            "rebalance": "weekly",
            "top_k": 1,
            "lookback_days": 5,
            "skip_days": 0,
            "cost_bps": 0.0,
            "stop_scheme": "equity_budget",
            "equity_stop_risk_pct": 0.02,
            "atr_stop_execution_mode": "next_day",
            "atr_stop_execution_time": "close",
            "anchors": [1, 5],
            "exec_prices": ["close"],
        },
    )
    assert data["meta"]["type"] == "rotation_calendar_effect"
    assert len(data["grid"]) == 2


def test_api_rotation_calendar_effect_rejects_equity_budget_non_close_execution(
    api_client,
):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240215",
    )
    err = post_json(
        c,
        "/api/analysis/rotation/calendar-effect",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240215",
            "rebalance": "weekly",
            "top_k": 1,
            "lookback_days": 5,
            "skip_days": 0,
            "cost_bps": 0.0,
            "stop_scheme": "equity_budget",
            "equity_stop_risk_pct": 0.02,
            "atr_stop_execution_mode": "intraday",
            "atr_stop_execution_time": "open",
            "anchors": [1, 5],
            "exec_prices": ["close"],
        },
        expected_status=422,
    )
    assert "equity_budget" in str(err)


def test_api_rotation_calendar_effect_rejects_atr_scheme_with_none_mode(api_client):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240215",
    )
    err = post_json(
        c,
        "/api/analysis/rotation/calendar-effect",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240215",
            "rebalance": "weekly",
            "top_k": 1,
            "lookback_days": 5,
            "skip_days": 0,
            "cost_bps": 0.0,
            "stop_scheme": "atr",
            "atr_stop_mode": "none",
            "atr_stop_execution_mode": "intraday",
            "atr_stop_execution_time": "close",
            "anchors": [1, 5],
            "exec_prices": ["close"],
        },
        expected_status=422,
    )
    assert "stop_scheme=atr requires atr_stop_mode" in str(err)


@pytest.mark.parametrize(
    "patch,err_msg",
    [
        (
            {
                "stop_scheme": "atr",
                "atr_stop_mode": "static",
                "atr_stop_atr_basis": "oops",
            },
            "atr_stop_atr_basis must be one of: entry|latest",
        ),
        (
            {
                "stop_scheme": "atr",
                "atr_stop_mode": "static",
                "atr_stop_reentry_mode": "hold",
            },
            "atr_stop_reentry_mode must be one of: reenter|wait_next_entry",
        ),
        (
            {
                "stop_scheme": "atr",
                "atr_stop_mode": "tightening",
                "atr_stop_n": 0.5,
                "atr_stop_m": 1.0,
            },
            "atr_stop_n must be > atr_stop_m when atr_stop_mode=tightening",
        ),
    ],
)
def test_api_rotation_calendar_effect_rejects_invalid_atr_aux_fields(
    api_client, patch: dict[str, object], err_msg: str
):
    c = api_client
    upsert_and_fetch_etfs(
        c,
        codes=_BASELINE_CODES,
        names=_BASELINE_NAMES,
        start_date="20240102",
        end_date="20240215",
    )
    err = post_json(
        c,
        "/api/analysis/rotation/calendar-effect",
        {
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240215",
            "rebalance": "weekly",
            "top_k": 1,
            "lookback_days": 5,
            "skip_days": 0,
            "cost_bps": 0.0,
            "atr_stop_execution_mode": "intraday",
            "atr_stop_execution_time": "close",
            "anchors": [1, 5],
            "exec_prices": ["close"],
            **patch,
        },
        expected_status=422,
    )
    assert err_msg in str(err)


def test_api_rotation_calendar_effect_entry_param_combo_diff(api_client, engine):
    dates, series = build_rotation_case_series()
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    base = {
        **make_rotation_base_payload(
            codes=["A", "B", "C", "D", "E"], dates=dates, rebalance="weekly"
        ),
        "rebalance": "weekly",
        "anchors": [2],
        "exec_prices": ["close"],
        "trend_filter": True,
        "bias_filter": True,
        "asset_trend_rules": [make_trend_rule(stage="entry")],
        "asset_bias_rules": [make_bias_rule(stage="entry", op="<=", fixed_value=1.5)],
    }

    d_and = post_json_ok(
        c, "/api/analysis/rotation/calendar-effect", {**base, "entry_match_n": 0}
    )
    d_nofm = post_json_ok(
        c, "/api/analysis/rotation/calendar-effect", {**base, "entry_match_n": 1}
    )
    assert ((d_and.get("grid") or [])[0] or {}).get("ok") is True
    assert ((d_nofm.get("grid") or [])[0] or {}).get("ok") is True
    ar_and = first_grid_metric(d_and, "annualized_return")
    ar_nofm = first_grid_metric(d_nofm, "annualized_return")
    assert ar_nofm >= ar_and


def test_api_rotation_calendar_effect_exit_param_combo_diff(api_client, engine):
    dates, series = build_rotation_case_series()
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    base = {
        **make_rotation_base_payload(
            codes=["A", "B", "C", "D", "E"], dates=dates, rebalance="weekly"
        ),
        "rebalance": "weekly",
        "anchors": [2],
        "exec_prices": ["close"],
        "entry_match_n": 1,
        **make_entry_filters_payload(bias_fixed_value=1.5),
        "asset_trend_rules": [
            make_trend_rule(stage="entry"),
            make_trend_rule(stage="exit"),
        ],
    }
    d_off = post_json_ok(
        c,
        "/api/analysis/rotation/calendar-effect",
        {**base, "trend_exit_filter": False},
    )
    d_on = post_json_ok(
        c,
        "/api/analysis/rotation/calendar-effect",
        {**base, "trend_exit_filter": True, "exit_match_n": 1},
    )
    assert ((d_off.get("grid") or [])[0] or {}).get("ok") is True
    assert ((d_on.get("grid") or [])[0] or {}).get("ok") is True
    ar_off = first_grid_metric(d_off, "annualized_return")
    ar_on = first_grid_metric(d_on, "annualized_return")
    assert ar_on <= ar_off


def test_api_rotation_calendar_effect_entry_exit_nofm_combo_diff(api_client, engine):
    dates, series = build_rotation_case_series()
    seed_prices(engine, code_to_series=series, dates=dates)

    c = api_client
    payload = {
        **make_rotation_base_payload(
            codes=["A", "B", "C", "D", "E"], dates=dates, rebalance="weekly"
        ),
        "rebalance": "weekly",
        "anchors": [2],
        "exec_prices": ["close"],
        "entry_match_n": 1,
        **make_entry_exit_filters_payload(
            entry_bias_fixed_value=1.5, exit_bias_fixed_value=99.0
        ),
    }

    d_and = post_json_ok(
        c, "/api/analysis/rotation/calendar-effect", {**payload, "exit_match_n": 0}
    )
    d_nofm = post_json_ok(
        c, "/api/analysis/rotation/calendar-effect", {**payload, "exit_match_n": 1}
    )
    assert ((d_and.get("grid") or [])[0] or {}).get("ok") is True
    assert ((d_nofm.get("grid") or [])[0] or {}).get("ok") is True
    ar_and = first_grid_metric(d_and, "annualized_return")
    ar_nofm = first_grid_metric(d_nofm, "annualized_return")
    assert ar_nofm <= ar_and
