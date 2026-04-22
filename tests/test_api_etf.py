from __future__ import annotations

from fastapi.testclient import TestClient

from tests.helpers.api_test_client import upsert_and_fetch_etfs
from tests.helpers.rotation_case_data import delete_json, get_json_ok, post_json


def test_create_list_delete_etf_and_fetch(api_client: TestClient) -> None:
    client = api_client

    data = post_json(
        client,
        "/api/etf",
        {
            "code": "510300",
            "name": "沪深300ETF",
            "start_date": "20240101",
            "end_date": "20240131",
        },
    )
    # auto-inferred policy should exist
    assert data["validation_policy"]["name"] == "cn_stock_etf_10"

    assert len(get_json_ok(client, "/api/etf")) == 1

    assert post_json(client, "/api/etf/510300/fetch", {})["status"] == "success"

    # fetched data range should be updated
    item = get_json_ok(client, "/api/etf")[0]
    assert item["last_data_start_date"] == "20240102"
    assert item["last_data_end_date"] == "20240103"

    out = delete_json(client, "/api/etf/510300")
    assert out["deleted"] is True
    assert out["purged"] is not None
    assert "prices" in out["purged"]


def test_delete_etf_purge_removes_prices_and_batches(api_client: TestClient) -> None:
    client = api_client
    upsert_and_fetch_etfs(
        client,
        codes=["510300"],
        names={"510300": "沪深300ETF"},
        start_date="20240101",
        end_date="20240131",
    )

    # ensure we have prices and batches before purge
    assert len(get_json_ok(client, "/api/etf/510300/prices?adjust=hfq")) > 0
    assert len(get_json_ok(client, "/api/batches?code=510300")) >= 1

    data = delete_json(client, "/api/etf/510300")
    assert data["deleted"] is True
    assert data["purged"] is not None
    assert data["purged"]["prices"] >= 1
    assert data["purged"]["batches"] >= 1

    # pool removed
    assert get_json_ok(client, "/api/etf") == []

    # data removed
    assert get_json_ok(client, "/api/etf/510300/prices?adjust=hfq") == []
    assert get_json_ok(client, "/api/batches?code=510300") == []
