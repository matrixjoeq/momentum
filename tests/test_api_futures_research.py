from __future__ import annotations

from fastapi.testclient import TestClient

from tests.helpers.rotation_case_data import post_json_ok


def test_futures_research_groups_state_and_correlation(api_client: TestClient) -> None:
    client = api_client

    # Prepare pool and prices
    post_json_ok(client, "/api/futures", {"code": "RB0", "name": "螺纹钢主连"})
    post_json_ok(client, "/api/futures", {"code": "IF0", "name": "股指主连"})
    post_json_ok(client, "/api/futures/RB0/fetch", {})
    post_json_ok(client, "/api/futures/IF0/fetch", {})

    # Save group as active
    g = post_json_ok(
        client,
        "/api/futures/research/groups",
        {"name": "默认分组", "codes": ["RB0", "IF0"], "set_active": True},
    )
    assert g["name"] == "默认分组"
    assert g["is_active"] is True

    # Update shared state with quick range
    s = client.put(
        "/api/futures/research/state",
        json={
            "start_date": "20240101",
            "end_date": "20241231",
            "dynamic_universe": True,
            "quick_range_key": "1y",
        },
    )
    assert s.status_code == 200
    st = s.json()
    assert st["dynamic_universe"] is True
    assert st["active_group"] == "默认分组"

    # Correlation uses active group by default
    c = client.post("/api/futures/research/correlation", json={"range_key": "all"})
    assert c.status_code == 200
    data = c.json()
    assert data["ok"] is True
    assert len(data["aliases"]) == 2
    assert len(data["matrix"]) == 2
    assert len(data["matrix"][0]) == 2

    cov = client.post("/api/futures/research/coverage-summary", json={"range_key": "all"})
    assert cov.status_code == 200
    cs = cov.json()
    assert cs["ok"] is True
    assert cs["meta"]["union_points"] >= 1
    assert cs["meta"]["intersection_points"] >= 1
    assert cs["meta"]["effective_points"] >= 1
    assert len(cs["symbols"]) == 2

    pick = client.post(
        "/api/futures/research/correlation-select",
        json={"range_key": "all", "mode": "lowest", "score_basis": "mean_abs", "n": 1},
    )
    assert pick.status_code == 200
    ps = pick.json()
    assert ps["ok"] is True
    assert ps["mode"] == "lowest"
    assert ps["score_basis"] == "mean_abs"
    assert ps["effective_n"] == 1
    assert len(ps["items"]) == 1
    assert "avg_corr" in ps["items"][0]
    assert "avg_abs_corr" in ps["items"][0]


def test_futures_research_groups_import_export_overwrite(api_client: TestClient) -> None:
    client = api_client
    post_json_ok(client, "/api/futures", {"code": "RB0", "name": "螺纹钢主连"})
    post_json_ok(client, "/api/futures", {"code": "IF0", "name": "股指主连"})

    post_json_ok(
        client,
        "/api/futures/research/groups",
        {"name": "A组", "codes": ["RB0"], "set_active": True},
    )
    # same-name overwrite
    post_json_ok(
        client,
        "/api/futures/research/groups",
        {"name": "A组", "codes": ["IF0"], "set_active": True},
    )
    all_groups = client.get("/api/futures/research/groups").json()
    hit = [x for x in all_groups if x["name"] == "A组"]
    assert len(hit) == 1
    assert hit[0]["codes"] == ["IF0"]

    exported = client.get("/api/futures/research/groups-export")
    assert exported.status_code == 200
    body = exported.json()
    assert body["format"] == "etf_momentum_futures_groups_v1"
    assert "A组" in body["groups"]

    imported = client.post(
        "/api/futures/research/groups-import",
        json={
            "groups": {"A组": ["RB0", "IF0"], "B组": ["RB0"]},
            "active_group": "B组",
        },
    )
    assert imported.status_code == 200
    out = imported.json()
    assert out["ok"] is True
    assert out["active_group"] == "B组"
