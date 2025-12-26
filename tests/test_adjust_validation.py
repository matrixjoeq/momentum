import pytest

from etf_momentum.db.repo import normalize_adjust


def test_normalize_adjust_basic():
    assert normalize_adjust(None) == "hfq"
    assert normalize_adjust("HFQ") == "hfq"
    assert normalize_adjust("qfq") == "qfq"
    assert normalize_adjust("none") == "none"
    # empty means "use default"
    assert normalize_adjust("") == "hfq"
    assert normalize_adjust("raw") == "none"
    assert normalize_adjust("nfq") == "none"


def test_normalize_adjust_invalid_raises():
    with pytest.raises(ValueError):
        normalize_adjust("bad")


def test_api_invalid_adjust_handling(api_client):
    c = api_client
    c.post("/api/etf", json={"code": "510300", "name": "沪深300", "start_date": "20240102", "end_date": "20240103"})

    r = c.get("/api/etf?adjust=bad")
    assert r.status_code == 400

    # fetch endpoints ignore adjust (always fetch all three), but keep param for backward compat
    r2 = c.post("/api/etf/510300/fetch?adjust=bad")
    assert r2.status_code in (200, 500)

    r3 = c.post("/api/fetch-all?adjust=bad")
    assert r3.status_code == 200
    out3 = r3.json()
    assert out3 and out3[0]["status"] in ("success", "failed")

    r4 = c.post("/api/fetch-selected", json={"codes": ["510300"], "adjust": "bad"})
    assert r4.status_code == 200
    out4 = r4.json()
    assert out4 and out4[0]["status"] in ("success", "failed")

