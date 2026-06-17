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


def test_etf_research_groups_persistence_contract(api_client: TestClient) -> None:
    client = api_client
    post_json(
        client,
        "/api/etf",
        {
            "code": "510300",
            "name": "沪深300ETF",
            "start_date": "20240101",
            "end_date": "20240131",
        },
    )
    post_json(
        client,
        "/api/etf",
        {
            "code": "159915",
            "name": "创业板ETF",
            "start_date": "20240101",
            "end_date": "20240131",
        },
    )

    g_a = post_json(
        client,
        "/api/etf/research/groups",
        {
            "name": "A组",
            "codes": ["510300", "NOT_EXISTS", "159915"],
            "set_active": False,
        },
    )
    assert g_a["name"] == "A组"
    assert g_a["codes"] == ["510300", "159915"]
    assert g_a["is_active"] is False

    g_b = post_json(
        client,
        "/api/etf/research/groups",
        {"name": "B组", "codes": ["159915"], "set_active": True},
    )
    assert g_b["name"] == "B组"
    assert g_b["is_active"] is True

    groups = get_json_ok(client, "/api/etf/research/groups")
    names = {x["name"] for x in groups}
    assert names == {"A组", "B组"}
    active_hit = [x for x in groups if x["is_active"]]
    assert len(active_hit) == 1 and active_hit[0]["name"] == "B组"

    exported = get_json_ok(client, "/api/etf/research/groups-export")
    assert exported["format"] == "etf_momentum_research_candidate_groups"
    assert exported["active_group"] == "B组"
    assert "A组" in exported["groups"]

    imported = post_json(
        client,
        "/api/etf/research/groups-import",
        {
            "groups": {"默认分组": ["510300"]},
            "active_group": "默认分组",
            "replace_all": True,
        },
    )
    assert imported["ok"] is True
    assert imported["active_group"] == "默认分组"

    groups_after = get_json_ok(client, "/api/etf/research/groups")
    assert len(groups_after) == 1
    assert groups_after[0]["name"] == "默认分组"
    assert groups_after[0]["codes"] == ["510300"]
    assert groups_after[0]["is_active"] is True
