from __future__ import annotations

from fastapi.testclient import TestClient

from tests.helpers.api_test_client import upsert_and_fetch_etfs
from tests.helpers.rotation_case_data import delete_json, get_json_ok, post_json


def test_prices_crud_via_fetch_read_delete(api_client: TestClient) -> None:
    # Create ETF and fetch (this is our "Create/Update" for prices)
    upsert_and_fetch_etfs(
        api_client,
        codes=["510300"],
        names={"510300": "沪深300ETF"},
        start_date="20240101",
        end_date="20240131",
    )

    # Read prices
    data = get_json_ok(api_client, "/api/etf/510300/prices")
    assert len(data) == 2
    assert data[0]["trade_date"] == "2024-01-02"

    # Delete one day
    assert (
        delete_json(api_client, "/api/etf/510300/prices", params={"end": "20240102"})[
            "deleted"
        ]
        == 1
    )

    assert [
        x["trade_date"] for x in get_json_ok(api_client, "/api/etf/510300/prices")
    ] == ["2024-01-03"]

    # Re-fetch should restore missing day via upsert
    post_json(api_client, "/api/etf/510300/fetch", {})

    assert [
        x["trade_date"] for x in get_json_ok(api_client, "/api/etf/510300/prices")
    ] == ["2024-01-02", "2024-01-03"]
