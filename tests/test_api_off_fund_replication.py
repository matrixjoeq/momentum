from __future__ import annotations

import datetime as dt

import numpy as np
from fastapi.testclient import TestClient

from etf_momentum.db.models import EtfPrice, OffFundNav, OffFundPool
from etf_momentum.db.session import make_session_factory


def _cumprod(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + returns)


def _seed_replication_fixture(client: TestClient) -> tuple[str, str]:
    sf = make_session_factory(client.app.state.engine)
    n = 340
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    x = np.arange(n, dtype=float)
    r300 = 0.006 * np.sin(x / 13.0)
    r500 = 0.005 * np.cos(x / 17.0)
    r1000 = 0.007 * np.sin(x / 11.0 + 0.3)
    r2000 = 0.008 * np.cos(x / 9.0 + 0.2)
    target = 0.65 * r300 + 0.20 * r500 + 0.00005
    with sf() as db:
        db.add(OffFundPool(code="REPL1", name="复制样本"))
        for adjust in ("hfq", "none"):
            for date, nav in zip(dates, _cumprod(target)):
                db.add(
                    OffFundNav(
                        code="REPL1",
                        trade_date=date,
                        nav=float(nav),
                        accum_nav=None,
                        source="unit_test",
                        adjust=adjust,
                    )
                )
        prices = {
            "000300": _cumprod(r300),
            "000905": _cumprod(r500),
            "000852": _cumprod(r1000),
            "932000": _cumprod(r2000),
            "SHORT_ONLY": np.where(
                np.arange(n) >= n - 100,
                _cumprod(r300),
                np.nan,
            ),
        }
        for adjust in ("hfq", "none"):
            for code, values in prices.items():
                for date, value in zip(dates, values):
                    db.add(
                        EtfPrice(
                            code=code,
                            trade_date=date,
                            open=float(value),
                            high=float(value),
                            low=float(value),
                            close=float(value),
                            volume=1000.0,
                            amount=float(value * 1000.0),
                            source="unit_test",
                            adjust=adjust,
                        )
                    )
        db.commit()
    return dates[0].strftime("%Y%m%d"), dates[-1].strftime("%Y%m%d")


def test_api_off_fund_replicate_v2_contract(api_client: TestClient) -> None:
    start, end = _seed_replication_fixture(api_client)
    response = api_client.post(
        "/api/analysis/off-fund/replicate",
        json={
            "codes": ["REPL1"],
            "start": start,
            "end": end,
            "template_id": "cn_equity_size",
            "template_version": 2,
            "rolling_window": 120,
            "min_samples": 80,
            "include_weight_series": True,
            "max_series_points": 500,
        },
    )
    assert response.status_code == 200
    out = response.json()
    assert out["ok"] is True
    assert out["meta"]["template_id"] == "cn_equity_size"
    assert {factor["key"] for factor in out["factors"]} == {
        "CSI300",
        "CSI500",
        "CSI1000",
        "CSI2000",
    }
    item = out["items"][0]
    assert item["status"] == "ok"
    assert item["model_version"] == "off_fund_replication_v2"
    assert item["training_days"] == 120
    assert item["oos_days"] == item["oos_metrics"]["sample_days"]
    assert item["estimation_windows"] == item["effective_windows"] + 1
    assert item["effective_windows"] == item["oos_days"]
    assert (
        item["model_parameters"]["calibration_status"] == "independent_walk_forward_v1"
    )
    assert item["asset_weights"]["CSI300"] > 0.5
    assert item["asset_weights"]["CSI500"] > 0.1
    assert abs(sum(item["asset_weights"].values()) + item["cash_weight"] - 1.0) < 1e-8
    assert item["oos_metrics"]["sample_days"] > 20
    assert item["metric_sample"] == "oos_lag1"
    assert item["tracking_error"] == item["oos_metrics"]["tracking_error_annualized"]
    assert item["identifiability"]["group_weight_intervals"]
    assert item["factor_sensitivity"]
    assert item["diagnostics_computed"] is True
    assert out["meta"]["extra_solves_used"] <= 30
    assert len(item["weight_series"]) > 20


def test_api_off_fund_replicate_rejects_ui_invalid_window(
    api_client: TestClient,
) -> None:
    response = api_client.post(
        "/api/analysis/off-fund/replicate",
        json={
            "codes": ["REPL1"],
            "start": "20240101",
            "end": "20241231",
            "rolling_window": 10,
        },
    )
    assert response.status_code == 422


def test_replication_runtime_parameters_survive_research_state_refresh(
    api_client: TestClient,
) -> None:
    saved = api_client.put(
        "/api/off-fund/research/state",
        json={
            "replication_rolling_window": 180,
            "replication_min_samples": 90,
            "replication_include_portfolio": False,
            "replication_drop_short_history_factors": True,
        },
    )
    assert saved.status_code == 200
    loaded = api_client.get("/api/off-fund/research/state")
    assert loaded.status_code == 200
    body = loaded.json()
    assert body["replication_rolling_window"] == 180
    assert body["replication_min_samples"] == 90
    assert body["replication_include_portfolio"] is False
    assert body["replication_drop_short_history_factors"] is True


def test_api_coverage_conflict_returns_http_200_and_ok_false(
    api_client: TestClient,
) -> None:
    start, end = _seed_replication_fixture(api_client)
    response = api_client.post(
        "/api/analysis/off-fund/replicate",
        json={
            "codes": ["REPL1"],
            "start": start,
            "end": end,
            "rolling_window": 80,
            "min_samples": 60,
            "benchmark_factors": [
                {
                    "key": "F300",
                    "aliases": ["000300"],
                    "reporting_group": "large",
                },
                {
                    "key": "F500",
                    "aliases": ["000905"],
                    "reporting_group": "mid",
                },
                {
                    "key": "SHORT_A",
                    "aliases": ["SHORT_ONLY"],
                    "reporting_group": "short_a",
                },
                {
                    "key": "SHORT_B",
                    "aliases": ["SHORT_ONLY"],
                    "reporting_group": "short_b",
                },
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["error"] == "coverage_conflict"
    assert body["items"][0]["status"] == "coverage_conflict"


def test_api_window_budget_returns_item_status(
    api_client: TestClient, monkeypatch
) -> None:
    start, end = _seed_replication_fixture(api_client)
    monkeypatch.setattr(
        "etf_momentum.analysis.off_fund_replication.MAX_WINDOWS_PER_TARGET",
        100,
    )
    response = api_client.post(
        "/api/analysis/off-fund/replicate",
        json={
            "codes": ["REPL1"],
            "start": start,
            "end": end,
            "rolling_window": 120,
            "min_samples": 80,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["error"] == "computation_budget_exceeded"
    assert body["items"][0]["status"] == "computation_budget_exceeded"


def test_api_custom_solver_parameters_require_advanced_mode(
    api_client: TestClient,
) -> None:
    response = api_client.post(
        "/api/analysis/off-fund/replicate",
        json={
            "codes": ["REPL1"],
            "start": "20240101",
            "end": "20241231",
            "lambda_temporal": 0.1,
        },
    )
    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert response.json()["error"] == "advanced_mode_required_for_custom_parameters"


def test_api_target_and_factor_budgets_return_503(
    api_client: TestClient,
) -> None:
    too_many_targets = api_client.post(
        "/api/analysis/off-fund/replicate",
        json={
            "codes": [f"T{idx:02d}" for idx in range(51)],
            "start": "20240101",
            "end": "20241231",
        },
    )
    assert too_many_targets.status_code == 503
    assert too_many_targets.json()["detail"]["max_targets"] == 50

    too_many_with_portfolio = api_client.post(
        "/api/analysis/off-fund/replicate",
        json={
            "codes": [f"T{idx:02d}" for idx in range(50)],
            "start": "20240101",
            "end": "20241231",
            "include_portfolio": True,
        },
    )
    assert too_many_with_portfolio.status_code == 503
    assert too_many_with_portfolio.json()["detail"]["max_targets"] == 50

    at_limit_without_portfolio = api_client.post(
        "/api/analysis/off-fund/replicate",
        json={
            "codes": [f"T{idx:02d}" for idx in range(50)],
            "start": "20240101",
            "end": "20241231",
            "include_portfolio": False,
        },
    )
    assert at_limit_without_portfolio.status_code != 503

    too_many_factors = api_client.post(
        "/api/analysis/off-fund/replicate",
        json={
            "codes": ["REPL1"],
            "start": "20240101",
            "end": "20241231",
            "benchmark_factors": [
                {"key": f"F{idx:02d}", "aliases": ["000300"]} for idx in range(41)
            ],
        },
    )
    assert too_many_factors.status_code == 503
    assert too_many_factors.json()["detail"]["max_factors"] == 40


def test_api_missing_target_is_not_reported_as_missing_benchmark(
    api_client: TestClient,
) -> None:
    response = api_client.post(
        "/api/analysis/off-fund/replicate",
        json={
            "codes": ["MISSING_TARGET"],
            "start": "20240101",
            "end": "20241231",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["error"] == "insufficient_target_samples"
    item = body["items"][0]
    assert item["status"] == "insufficient_target_samples"
    assert item["effective_start"] is None
    assert item["effective_end"] is None


def test_api_mixed_target_outcomes_return_partial_failure(
    api_client: TestClient,
) -> None:
    start, end = _seed_replication_fixture(api_client)
    response = api_client.post(
        "/api/analysis/off-fund/replicate",
        json={
            "codes": ["REPL1", "MISSING_TARGET"],
            "start": start,
            "end": end,
            "rolling_window": 120,
            "min_samples": 80,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["error"] == "partial_failure"
    assert {item["status"] for item in body["items"]} == {
        "ok",
        "insufficient_target_samples",
    }


def test_api_rejects_duplicate_custom_factor_keys(
    api_client: TestClient,
) -> None:
    response = api_client.post(
        "/api/analysis/off-fund/replicate",
        json={
            "codes": ["REPL1"],
            "start": "20240101",
            "end": "20241231",
            "benchmark_factors": [
                {"key": "DUP", "aliases": ["000300"]},
                {"key": "dup", "aliases": ["000905"]},
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert response.json()["error"] == "invalid_factor_configuration"


def test_api_normalizes_raw_and_nfq_adjust_aliases(
    api_client: TestClient,
) -> None:
    start, end = _seed_replication_fixture(api_client)
    response = api_client.post(
        "/api/analysis/off-fund/replicate",
        json={
            "codes": ["REPL1"],
            "start": start,
            "end": end,
            "fund_adjust": "raw",
            "benchmark_adjust": "nfq",
            "rolling_window": 120,
            "min_samples": 80,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["meta"]["fund_adjust"] == "none"
    assert body["meta"]["benchmark_adjust"] == "none"
    assert body["items"][0]["status"] == "ok"
