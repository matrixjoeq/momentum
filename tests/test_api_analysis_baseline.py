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
            "fft_windows": [20, 10],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["metrics"]["benchmark_code"] == "510300"
    assert data["metrics"]["rebalance"] == "yearly"
    assert data["metrics"]["risk_free_rate"] == 0.02
    assert "ulcer_index" in data["metrics"]
    assert "ulcer_performance_index" in data["metrics"]
    assert "holding_weekly_win_rate" in data["metrics"]
    assert "holding_monthly_payoff_ratio" in data["metrics"]
    assert "holding_yearly_kelly_fraction" in data["metrics"]
    assert "EW" in data["nav"]["series"]
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
    assert data["nav_rsi"]["windows"] == [6, 12, 24]


def test_api_rotation_backtest_happy_path(api_client):
    c = api_client
    c.post("/api/etf", json={"code": "510300", "name": "沪深300", "start_date": "20240102", "end_date": "20240103"})
    c.post("/api/etf", json={"code": "511010", "name": "国债", "start_date": "20240102", "end_date": "20240103"})
    assert c.post("/api/etf/510300/fetch").status_code == 200
    assert c.post("/api/etf/511010/fetch").status_code == 200

    resp = c.post(
        "/api/analysis/rotation",
        json={
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
            "timing_rsi_gate": True,
            "timing_rsi_window": 6,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "nav" in data and "series" in data["nav"]
    assert "ROTATION" in data["nav"]["series"]
    assert "EW_REBAL" in data["nav"]["series"]
    assert "EXCESS" in data["nav"]["series"]
    assert "nav_rsi" in data
    assert data["nav_rsi"]["windows"] == [6, 12, 24]
    assert "timing" in data
    assert data["timing"]["enabled"] is True
    assert data["timing"]["threshold"] == 50.0
    assert "win_payoff" in data

