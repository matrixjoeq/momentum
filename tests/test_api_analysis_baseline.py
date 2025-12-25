def test_api_baseline_analysis_happy_path(api_client):
    c = api_client
    c.post("/api/etf", json={"code": "510300", "name": "沪深300", "start_date": "20240102", "end_date": "20240103"})
    c.post("/api/etf", json={"code": "511010", "name": "国债", "start_date": "20240102", "end_date": "20240103"})
    r1 = c.post("/api/etf/510300/fetch")
    assert r1.status_code == 200
    r2 = c.post("/api/etf/511010/fetch")
    assert r2.status_code == 200

    resp = c.post(
        "/api/analysis/baseline",
        json={
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
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["metrics"]["benchmark_code"] == "510300"
    assert data["metrics"]["rebalance"] == "yearly"
    assert data["metrics"]["risk_free_rate"] == 0.02
    assert "EW" in data["nav"]["series"]
    assert "510300" in data["nav"]["series"]
    assert "511010" in data["nav"]["series"]
    assert any(k.startswith("BENCH:") for k in data["nav"]["series"].keys())

