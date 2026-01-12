def test_api_rotation_weekly5_open_sim(api_client):
    c = api_client
    # Create pool entries for the fixed mini-program universe and ingest tiny fake prices.
    for code, name in [
        ("159915", "创业板ETF"),
        ("511010", "国债ETF"),
        ("513100", "纳指ETF"),
        ("518880", "黄金ETF"),
    ]:
        c.post("/api/etf", json={"code": code, "name": name, "start_date": "20240102", "end_date": "20240103"})
        assert c.post(f"/api/etf/{code}/fetch").status_code == 200

    resp = c.post("/api/analysis/rotation/weekly5-open", json={"start": "20240102", "end": "20240103"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["meta"]["type"] == "rotation_weekly5_open"
    assert data["meta"]["exec_price"] == "open"
    assert data["meta"]["rebalance_shift"] == "prev"
    by = data["by_anchor"]
    assert set(by.keys()) == {"0", "1", "2", "3", "4"}
    # spot-check one result payload shape
    one = by["4"]
    assert "nav" in one and "series" in one["nav"]
    assert "ROTATION" in one["nav"]["series"]
    assert "EW_REBAL" in one["nav"]["series"]
    assert "EXCESS" in one["nav"]["series"]

