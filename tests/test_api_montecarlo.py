def test_api_baseline_montecarlo_smoke(api_client):
    c = api_client
    c.post("/api/etf", json={"code": "510300", "name": "沪深300", "start_date": "20240101", "end_date": "20241231"})
    c.post("/api/etf", json={"code": "511010", "name": "国债", "start_date": "20240101", "end_date": "20241231"})
    assert c.post("/api/etf/510300/fetch").status_code == 200
    assert c.post("/api/etf/511010/fetch").status_code == 200

    resp = c.post(
        "/api/analysis/baseline/montecarlo",
        json={
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
    assert resp.status_code == 200
    data = resp.json()
    assert data["meta"]["type"] == "baseline"
    assert "mc" in data and "metrics" in data["mc"]
    assert "annualized_return" in data["mc"]["metrics"]


def test_api_rotation_montecarlo_smoke(api_client):
    c = api_client
    c.post("/api/etf", json={"code": "510300", "name": "沪深300", "start_date": "20240101", "end_date": "20241231"})
    c.post("/api/etf", json={"code": "511010", "name": "国债", "start_date": "20240101", "end_date": "20241231"})
    assert c.post("/api/etf/510300/fetch").status_code == 200
    assert c.post("/api/etf/511010/fetch").status_code == 200

    resp = c.post(
        "/api/analysis/rotation/montecarlo",
        json={
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
    assert resp.status_code == 200
    data = resp.json()
    assert data["meta"]["type"] == "rotation"
    assert "mc" in data and "strategy" in data["mc"] and "excess" in data["mc"]
    assert "annualized_return" in data["mc"]["strategy"]["metrics"]

