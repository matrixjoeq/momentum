from __future__ import annotations

from tests.helpers.api_test_client import upsert_and_fetch_etfs


def test_api_baseline_garch_volatility_contract_success(
    api_client, monkeypatch
) -> None:
    import etf_momentum.api.routes as routes

    upsert_and_fetch_etfs(
        api_client,
        codes=["510300"],
        names={"510300": "沪深300ETF"},
        start_date="20240101",
        end_date="20240131",
    )

    def _fake_compute(close, **kwargs):
        _ = (close, kwargs)
        return {
            "ok": True,
            "meta": {
                "n_obs_raw": 200,
                "n_obs_price": 200,
                "n_obs_returns": 199,
                "dropped_obs": 1,
                "min_samples": 120,
                "max_points": 1200,
            },
            "params": {
                "mu": 0.01,
                "omega": 0.02,
                "alpha1": 0.06,
                "gamma1": 0.04,
                "beta1": 0.88,
                "nu": 8.5,
                "persistence": 0.96,
                "unconditional_var_daily": 0.0004,
                "unconditional_vol_daily": 0.02,
                "unconditional_vol_annualized": 0.317,
            },
            "diagnostics": {
                "converged": True,
                "convergence_flag": 0,
                "n_obs_raw": 200,
                "n_obs_price": 200,
                "n_obs_returns": 199,
                "dropped_obs": 1,
                "ann_factor": 252,
                "return_scale": 100.0,
                "loglikelihood": -123.4,
                "aic": 260.1,
                "bic": 278.6,
                "std_resid_mean": 0.0,
                "std_resid_std": 1.0,
                "std_resid_skew": 0.1,
                "std_resid_kurtosis_excess": 0.2,
                "arch_lm_pre": {
                    "ok": True,
                    "lags": 10,
                    "n_obs": 199,
                    "stat": 12.3,
                    "pvalue": 0.02,
                    "significant": True,
                },
                "arch_lm_post": {
                    "ok": True,
                    "lags": 10,
                    "n_obs": 199,
                    "stat": 8.3,
                    "pvalue": 0.61,
                    "significant": False,
                },
            },
            "interpretation": {
                "model_value": "high",
                "value_score": 0.86,
                "summary": "模型稳定，可作为高价值波动参考。",
                "reasons": ["样本充足", "收敛良好"],
            },
            "series": {
                "price_dates": ["2024-01-03", "2024-01-04"],
                "price_close": [1.01, 1.02],
                "vol_dates": ["2024-01-03", "2024-01-04"],
                "cond_vol_daily": [0.012, 0.011],
                "cond_vol_annualized": [0.19, 0.175],
                "log_returns": [0.01, -0.002],
            },
        }

    monkeypatch.setattr(routes, "compute_gjr_garch_volatility", _fake_compute)

    resp = api_client.post(
        "/api/analysis/baseline/garch-volatility",
        json={
            "etf_code": "510300",
            "start": "20240101",
            "end": "20240131",
            "adjust": "hfq",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["error"] is None
    assert data["meta"]["etf_code"] == "510300"
    assert data["interpretation"]["model_value"] == "high"
    assert len(data["series"]["vol_dates"]) == len(
        data["series"]["cond_vol_annualized"]
    )


def test_api_baseline_garch_volatility_end_before_start(api_client) -> None:
    resp = api_client.post(
        "/api/analysis/baseline/garch-volatility",
        json={
            "etf_code": "510300",
            "start": "20240131",
            "end": "20240101",
            "adjust": "hfq",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert data["error"] == "end_before_start"


def test_api_baseline_garch_volatility_empty_series_error(api_client) -> None:
    resp = api_client.post(
        "/api/analysis/baseline/garch-volatility",
        json={
            "etf_code": "NOT_EXISTS",
            "start": "20240101",
            "end": "20240131",
            "adjust": "hfq",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is False
    assert data["error"] == "empty_etf_close"


def test_api_baseline_garch_volatility_rejects_invalid_payload(api_client) -> None:
    resp = api_client.post(
        "/api/analysis/baseline/garch-volatility",
        json={
            "etf_code": "510300",
            "start": "20240101",
            "end": "20240131",
            "adjust": "hfq",
            "max_points": -1,
        },
    )
    assert resp.status_code == 422
