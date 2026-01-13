def test_api_baseline_weekly5_ew_dashboard_smoke(api_client):
    c = api_client
    # create pool entries for the fixed codes and ingest fake prices
    for code, name in [
        ("159915", "创业板ETF"),
        ("511010", "国债ETF"),
        ("513100", "纳指ETF"),
        ("518880", "黄金ETF"),
    ]:
        c.post("/api/etf", json={"code": code, "name": name, "start_date": "20240102", "end_date": "20240103"})
        assert c.post(f"/api/etf/{code}/fetch").status_code == 200

    resp = c.post(
        "/api/analysis/baseline/weekly5-ew-dashboard",
        json={"start": "20240102", "end": "20240103", "risk_free_rate": 0.02, "rebalance_shift": "prev"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["meta"]["type"] == "baseline_weekly5_ew_dashboard"
    assert "by_anchor" in data
    assert set(data["by_anchor"].keys()) == {"0", "1", "2", "3", "4"}

