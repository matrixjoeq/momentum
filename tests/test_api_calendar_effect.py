def test_api_baseline_calendar_effect(api_client):
    c = api_client
    c.post("/api/etf", json={"code": "510300", "name": "沪深300", "start_date": "20240102", "end_date": "20240215"})
    c.post("/api/etf", json={"code": "511010", "name": "国债", "start_date": "20240102", "end_date": "20240215"})
    assert c.post("/api/etf/510300/fetch").status_code == 200
    assert c.post("/api/etf/511010/fetch").status_code == 200

    resp = c.post(
        "/api/analysis/baseline/calendar-effect",
        json={
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240215",
            "adjust": "hfq",
            "risk_free_rate": 0.02,
            "rebalance": "weekly",
            "anchors": [0, 4],
            "exec_prices": ["close"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["meta"]["type"] == "baseline_calendar_effect"
    assert len(data["grid"]) == 2
    assert all("anchor" in x and "exec_price" in x and "ok" in x for x in data["grid"])
    ok = next((x for x in data["grid"] if x.get("ok")), None)
    assert ok is not None
    m = ok.get("metrics") or {}
    for k in ["calmar_ratio", "sortino_ratio", "ulcer_index", "ulcer_performance_index", "information_ratio"]:
        assert k in m


def test_api_rotation_calendar_effect(api_client):
    c = api_client
    c.post("/api/etf", json={"code": "510300", "name": "沪深300", "start_date": "20240102", "end_date": "20240215"})
    c.post("/api/etf", json={"code": "511010", "name": "国债", "start_date": "20240102", "end_date": "20240215"})
    assert c.post("/api/etf/510300/fetch").status_code == 200
    assert c.post("/api/etf/511010/fetch").status_code == 200

    resp = c.post(
        "/api/analysis/rotation/calendar-effect",
        json={
            "codes": ["510300", "511010"],
            "start": "20240102",
            "end": "20240215",
            "rebalance": "weekly",
            "top_k": 1,
            "lookback_days": 5,
            "skip_days": 0,
            "risk_off": False,
            "risk_free_rate": 0.02,
            "cost_bps": 0.0,
            "anchors": [0, 4],
            "exec_prices": ["close"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["meta"]["type"] == "rotation_calendar_effect"
    assert len(data["grid"]) == 2
    assert all("anchor" in x and "exec_price" in x and "ok" in x for x in data["grid"])
    ok = next((x for x in data["grid"] if x.get("ok")), None)
    assert ok is not None
    m = ok.get("metrics") or {}
    for k in ["calmar_ratio", "sortino_ratio", "ulcer_index", "ulcer_performance_index", "information_ratio"]:
        assert k in m


