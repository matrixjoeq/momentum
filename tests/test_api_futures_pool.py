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

    cs = client.get("/api/futures/RB0/contracts/fetch-status")
    assert cs.status_code == 200
    assert isinstance(cs.json(), list)

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


def test_futures_pool_auto_tag_classification_when_tags_empty(api_client: TestClient) -> None:
    client = api_client
    rb = post_json_ok(
        client,
        "/api/futures",
        {
            "code": "RB0",
            "name": "螺纹钢主连",
            "start_date": "20240101",
            "end_date": "20241231",
            "tags": [],
        },
    )
    assert rb["code"] == "RB0"
    assert rb["tags"] == ["黑色系"]

    if0 = post_json_ok(
        client,
        "/api/futures",
        {
            "code": "IF0",
            "name": "股指主连",
            "tags": None,
        },
    )
    assert if0["code"] == "IF0"
    assert if0["tags"] == ["股指期货"]

    listed = client.get("/api/futures?adjust=none")
    assert listed.status_code == 200
    items = listed.json()
    by_code = {x["code"]: x for x in items}
    assert by_code["RB0"]["tags"] == ["黑色系"]
    assert by_code["IF0"]["tags"] == ["股指期货"]


def test_futures_fetch_incremental_fallback_and_full_modes_contract(api_client: TestClient) -> None:
    client = api_client
    post_json_ok(
        client,
        "/api/futures",
        {"code": "RB0", "name": "螺纹钢主连", "start_date": "20240101", "end_date": "20241231"},
    )

    first = post_json_ok(client, "/api/futures/RB0/fetch", {"fetch_type": "incremental"})
    assert first["status"] == "success"
    assert first["inserted_or_updated"] >= 1
    assert "mode=incremental->full" in (first.get("message") or "")

    second = post_json_ok(client, "/api/futures/RB0/fetch", {"fetch_type": "incremental"})
    assert second["status"] == "success"
    assert second["inserted_or_updated"] == 0
    assert "mode=incremental" in (second.get("message") or "")

    third = post_json_ok(client, "/api/futures/RB0/fetch", {"fetch_type": "full"})
    assert third["status"] == "success"
    assert third["inserted_or_updated"] >= 1
    assert "mode=full" in (third.get("message") or "")
