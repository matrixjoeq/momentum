from __future__ import annotations

from fastapi.testclient import TestClient

from tests.helpers.rotation_case_data import post_json_ok


def test_fetch_selected_success(api_client: TestClient) -> None:
    api_client.post("/api/etf", json={"code": "510300", "name": "沪深300ETF"})
    api_client.post("/api/etf", json={"code": "588000", "name": "科创50ETF"})

    data = post_json_ok(api_client, "/api/fetch-selected", {"codes": ["510300"]})
    assert len(data) == 1
    assert data[0]["code"] == "510300"
    assert data[0]["status"] == "success"


def test_fetch_selected_missing_code(api_client: TestClient) -> None:
    api_client.post("/api/etf", json={"code": "510300", "name": "沪深300ETF"})
    data = post_json_ok(
        api_client, "/api/fetch-selected", {"codes": ["NOPE", "510300"]}
    )
    assert {x["code"] for x in data} == {"NOPE", "510300"}
    assert any(x["code"] == "NOPE" and x["status"] == "failed" for x in data)


def test_fetch_selected_parallel_symbol_workers_bounds(api_client: TestClient) -> None:
    api_client.post("/api/etf", json={"code": "510300", "name": "沪深300ETF"})
    r_low = api_client.post(
        "/api/fetch-selected",
        json={
            "codes": ["510300"],
            "fetch_mode": "parallel",
            "parallel_symbol_workers": 1,
        },
    )
    assert r_low.status_code == 422
    r_high = api_client.post(
        "/api/fetch-selected",
        json={
            "codes": ["510300"],
            "fetch_mode": "parallel",
            "parallel_symbol_workers": 6,
        },
    )
    assert r_high.status_code == 422


def test_fetch_selected_parallel_mode(api_client: TestClient) -> None:
    api_client.post("/api/etf", json={"code": "510300", "name": "沪深300ETF"})
    data = post_json_ok(
        api_client,
        "/api/fetch-selected",
        {"codes": ["510300"], "fetch_mode": "parallel", "parallel_symbol_workers": 3},
    )
    assert len(data) == 1
    assert data[0]["code"] == "510300"
    assert data[0]["status"] in ("success", "failed")


def test_fetch_all_parallel_mode(api_client: TestClient) -> None:
    api_client.post("/api/etf", json={"code": "510300", "name": "沪深300ETF"})
    data = post_json_ok(
        api_client,
        "/api/fetch-all",
        {"fetch_mode": "parallel", "parallel_symbol_workers": 4},
    )
    assert len(data) == 1
    assert data[0]["code"] == "510300"
    assert data[0]["status"] in ("success", "failed")


def test_fetch_all_parallel_workers_bounds(api_client: TestClient) -> None:
    r = api_client.post(
        "/api/fetch-all", json={"fetch_mode": "parallel", "parallel_symbol_workers": 1}
    )
    assert r.status_code == 422
