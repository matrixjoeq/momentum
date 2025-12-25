def test_api_analysis_errors_if_any_code_missing_adjust_data(api_client):
    c = api_client

    # add ETFs
    c.post("/api/etf", json={"code": "A", "name": "A", "start_date": "20240101", "end_date": "20240103"})
    c.post("/api/etf", json={"code": "B", "name": "B", "start_date": "20240101", "end_date": "20240103"})

    # fetch A under hfq (exists), leave B without hfq (only insert qfq manually via prices CRUD)
    assert c.post("/api/etf/A/fetch").status_code == 200

    # insert qfq prices for B via prices table is not exposed; use delete_prices won't help.
    # We trigger a qfq fetch for B by calling fetch with adjust=qfq so B has only qfq data.
    assert c.post("/api/etf/B/fetch?adjust=qfq").status_code == 200

    # analysis on hfq should fail because B lacks hfq
    resp = c.post(
        "/api/analysis/baseline",
        json={"codes": ["A", "B"], "start": "20240102", "end": "20240103", "adjust": "hfq", "rebalance": "yearly"},
    )
    assert resp.status_code == 400
    assert "missing data" in resp.json().get("detail", "")

