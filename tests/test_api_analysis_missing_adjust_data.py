def test_api_analysis_errors_if_any_code_missing_adjust_data(api_client):
    c = api_client

    # add ETFs
    c.post("/api/etf", json={"code": "A", "name": "A", "start_date": "20240101", "end_date": "20240103"})
    c.post("/api/etf", json={"code": "B", "name": "B", "start_date": "20240101", "end_date": "20240103"})

    # fetch will load all adjusts for both A and B
    assert c.post("/api/etf/A/fetch").status_code == 200
    assert c.post("/api/etf/B/fetch").status_code == 200

    # delete B's hfq data to simulate missing adjust
    d = c.delete("/api/etf/B/prices?adjust=hfq")
    assert d.status_code == 200
    assert d.json()["deleted"] > 0

    # analysis on hfq should fail because B lacks hfq
    resp = c.post(
        "/api/analysis/baseline",
        json={"codes": ["A", "B"], "start": "20240102", "end": "20240103", "adjust": "hfq", "rebalance": "yearly"},
    )
    assert resp.status_code == 400
    assert "missing data" in resp.json().get("detail", "")

