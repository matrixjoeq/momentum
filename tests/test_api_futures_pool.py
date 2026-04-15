from __future__ import annotations

from fastapi.testclient import TestClient

from tests.helpers.rotation_case_data import post_json_ok


def test_futures_pool_crud_and_fetch_contract(api_client: TestClient) -> None:
    client = api_client

    up = post_json_ok(
        client,
        "/api/futures",
        {"code": "RB0", "name": "螺纹钢主连", "start_date": "20240101", "end_date": "20241231"},
    )
    assert up["code"] == "RB0"
    assert up["name"] == "螺纹钢主连"

    listed = client.get("/api/futures?adjust=none")
    assert listed.status_code == 200
    items = listed.json()
    assert isinstance(items, list)
    assert any(x["code"] == "RB0" for x in items)

    fetched = post_json_ok(client, "/api/futures/RB0/fetch", {})
    assert fetched["code"] == "RB0"
    assert fetched["status"] == "success"
    assert fetched["inserted_or_updated"] >= 1

    prices = client.get("/api/futures/RB0/prices?adjust=none")
    assert prices.status_code == 200
    rows = prices.json()
    assert isinstance(rows, list)
    assert len(rows) >= 1
    assert rows[0]["code"] == "RB0"
    assert rows[0]["adjust"] == "none"

    bad_adjust = client.get("/api/futures/RB0/prices?adjust=hfq")
    assert bad_adjust.status_code == 400

    rm = client.delete("/api/futures/RB0")
    assert rm.status_code == 200
    body = rm.json()
    assert body["deleted"] is True
    assert body["purged"]["prices"] >= 0


def test_futures_fetch_selected_partial_failure_contract(api_client: TestClient) -> None:
    client = api_client
    client.post("/api/futures", json={"code": "IF0", "name": "股指主连"})
    out = post_json_ok(client, "/api/futures/fetch-selected", {"codes": ["NOPE", "IF0"]})
    assert {x["code"] for x in out} == {"NOPE", "IF0"}
    assert any(x["code"] == "NOPE" and x["status"] == "failed" for x in out)
    assert any(x["code"] == "IF0" and x["status"] in {"success", "failed"} for x in out)
