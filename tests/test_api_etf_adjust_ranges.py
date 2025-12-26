def test_etf_list_range_changes_with_adjust(api_client):
    c = api_client
    c.post("/api/etf", json={"code": "510300", "name": "沪深300", "start_date": "20240102", "end_date": "20240103"})

    # default fetch (hfq) creates hfq prices
    r = c.post("/api/etf/510300/fetch")
    assert r.status_code == 200

    # hfq should have range
    resp_h = c.get("/api/etf?adjust=hfq")
    assert resp_h.status_code == 200
    it_h = resp_h.json()[0]
    assert it_h["last_data_start_date"] == "20240102"
    assert it_h["last_data_end_date"] == "20240103"

    # qfq should also have identical range
    resp_q = c.get("/api/etf?adjust=qfq")
    assert resp_q.status_code == 200
    it_q = resp_q.json()[0]
    assert it_q["last_data_start_date"] == "20240102"
    assert it_q["last_data_end_date"] == "20240103"

    # none should also have identical range
    resp_n = c.get("/api/etf?adjust=none")
    assert resp_n.status_code == 200
    it_n = resp_n.json()[0]
    assert it_n["last_data_start_date"] == "20240102"
    assert it_n["last_data_end_date"] == "20240103"

