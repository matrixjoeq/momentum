from __future__ import annotations

import json

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text

from etf_momentum.db.schema import ensure_runtime_schema


def test_off_fund_research_state_get_put_roundtrip(api_client: TestClient) -> None:
    client = api_client
    r0 = client.get("/api/off-fund/research/state")
    assert r0.status_code == 200
    s0 = r0.json()
    assert s0["adjust"] in {"hfq", "qfq", "none"}
    assert "start_date" in s0
    assert "rebalance_cycle" in s0
    assert s0["meta"]["contract_version"] == "pair_contract_v1"
    assert isinstance(s0["meta"]["warnings"], list)

    r1 = client.put(
        "/api/off-fund/research/state",
        json={
            "start_date": "20180101",
            "end_date": "20251231",
            "adjust": "none",
            "rf": 0.03,
            "inner_mode": "custom",
            "rp_window": 88,
            "rebalance_cycle": "monthly",
            "drift_rebalance_enabled": False,
            "drift_abs_threshold": 0.12,
            "drift_rel_threshold": 0.31,
            "pair_chart_prefs_json": '{"pair_slot_01":{"base":"CSI300","peer":"CSI500"}}',
        },
    )
    assert r1.status_code == 200
    s1 = r1.json()
    assert s1["start_date"] == "20180101"
    assert s1["end_date"] == "20251231"
    assert s1["adjust"] == "none"
    assert s1["rf"] == 0.03
    assert s1["inner_mode"] == "custom"
    assert s1["rp_window"] == 88
    assert s1["rebalance_cycle"] == "monthly"
    assert s1["drift_rebalance_enabled"] is False
    assert s1["drift_abs_threshold"] == 0.12
    assert s1["drift_rel_threshold"] == 0.31
    assert (
        s1["pair_chart_prefs_json"]
        == '{"pair_slot_01":{"base":"CSI300","peer":"CSI500"}}'
    )
    assert s1["meta"]["contract_version"] == "pair_contract_v1"
    assert s1["meta"]["warnings"] == []

    r2 = client.get("/api/off-fund/research/state")
    assert r2.status_code == 200
    s2 = r2.json()
    assert s2["start_date"] == "20180101"
    assert s2["end_date"] == "20251231"
    assert s2["rebalance_cycle"] == "monthly"
    assert (
        s2["pair_chart_prefs_json"]
        == '{"pair_slot_01":{"base":"CSI300","peer":"CSI500"}}'
    )
    assert s2["meta"]["contract_version"] == "pair_contract_v1"
    assert s2["meta"]["warnings"] == []


def test_off_fund_research_state_rejects_invalid_payload(
    api_client: TestClient,
) -> None:
    client = api_client
    bad_date = client.put(
        "/api/off-fund/research/state",
        json={"start_date": "202401", "end_date": "20251231"},
    )
    assert bad_date.status_code == 400
    assert "YYYYMMDD" in str(bad_date.json().get("detail"))

    bad_bound = client.put(
        "/api/off-fund/research/state",
        json={"drift_abs_threshold": 1.5},
    )
    assert bad_bound.status_code == 422

    too_long = client.put(
        "/api/off-fund/research/state",
        json={"pair_chart_prefs_json": "x" * (16 * 1024 + 1)},
    )
    assert too_long.status_code == 413
    d = too_long.json().get("detail") or {}
    assert d.get("error_code") == "pair_chart_prefs_payload_too_large"
    assert d.get("contract_version") == "pair_contract_v1"
    assert isinstance(d.get("detail"), str)

    bad_json = client.put(
        "/api/off-fund/research/state",
        json={"pair_chart_prefs_json": "{not_json"},
    )
    assert bad_json.status_code == 400
    assert bad_json.json().get("detail") == "pair_chart_prefs_json must be valid JSON"

    bad_type = client.put(
        "/api/off-fund/research/state",
        json={"pair_chart_prefs_json": '["not","object"]'},
    )
    assert bad_type.status_code == 400
    assert (
        bad_type.json().get("detail") == "pair_chart_prefs_json must be a JSON object"
    )


def test_off_fund_research_state_trim_and_order(api_client: TestClient) -> None:
    client = api_client
    too_many = {
        f"pair_slot_{i:02d}": {"base": "CSI300", "peer": "CSI500"} for i in range(1, 23)
    }
    r = client.put(
        "/api/off-fund/research/state",
        json={"pair_chart_prefs_json": json.dumps(too_many)},
    )
    assert r.status_code == 200
    out = r.json()
    assert out["meta"]["contract_version"] == "pair_contract_v1"
    assert out["meta"]["warnings"] == ["prefs_trimmed_to_21"]
    prefs = out["pair_chart_prefs_json"]
    assert isinstance(prefs, str)
    assert '"pair_slot_01"' in prefs
    assert '"pair_slot_03"' in prefs
    assert '"pair_slot_13"' in prefs
    assert '"pair_slot_17"' in prefs
    assert '"pair_slot_18"' in prefs
    assert '"pair_slot_21"' in prefs
    assert '"pair_slot_22"' not in prefs
    assert prefs.index('"pair_slot_01"') < prefs.index('"pair_slot_03"')
    assert prefs.index('"pair_slot_03"') < prefs.index('"pair_slot_13"')
    assert prefs.index('"pair_slot_13"') < prefs.index('"pair_slot_17"')
    assert prefs.index('"pair_slot_17"') < prefs.index('"pair_slot_18"')
    assert prefs.index('"pair_slot_18"') < prefs.index('"pair_slot_21"')


def test_off_fund_state_put_legacy_body_keeps_pair_prefs(
    api_client: TestClient,
) -> None:
    client = api_client
    r0 = client.put(
        "/api/off-fund/research/state",
        json={
            "pair_chart_prefs_json": '{"pair_slot_01":{"base":"CSI300","peer":"CSI500"}}'
        },
    )
    assert r0.status_code == 200
    r1 = client.put(
        "/api/off-fund/research/state",
        json={"start_date": "20190101", "end_date": "20200101"},
    )
    assert r1.status_code == 200
    out = r1.json()
    assert out["start_date"] == "20190101"
    assert out["end_date"] == "20200101"
    assert (
        out["pair_chart_prefs_json"]
        == '{"pair_slot_01":{"base":"CSI300","peer":"CSI500"}}'
    )


def test_off_fund_state_partial_put_keeps_unspecified_fields(
    api_client: TestClient,
) -> None:
    client = api_client
    r0 = client.put(
        "/api/off-fund/research/state",
        json={
            "start_date": "20180101",
            "end_date": "20251231",
            "adjust": "none",
            "rf": 0.03,
            "inner_mode": "custom",
            "rp_window": 88,
            "rebalance_cycle": "monthly",
            "drift_rebalance_enabled": False,
            "drift_abs_threshold": 0.12,
            "drift_rel_threshold": 0.31,
            "pair_chart_prefs_json": '{"pair_slot_01":{"base":"CSI300","peer":"CSI500"}}',
        },
    )
    assert r0.status_code == 200
    r1 = client.put(
        "/api/off-fund/research/state",
        json={"start_date": "20190101", "end_date": "20200101"},
    )
    assert r1.status_code == 200
    out = r1.json()
    assert out["start_date"] == "20190101"
    assert out["end_date"] == "20200101"
    assert out["adjust"] == "none"
    assert out["rf"] == 0.03
    assert out["inner_mode"] == "custom"
    assert out["rp_window"] == 88
    assert out["rebalance_cycle"] == "monthly"
    assert out["drift_rebalance_enabled"] is False
    assert out["drift_abs_threshold"] == 0.12
    assert out["drift_rel_threshold"] == 0.31
    assert (
        out["pair_chart_prefs_json"]
        == '{"pair_slot_01":{"base":"CSI300","peer":"CSI500"}}'
    )


def test_runtime_schema_adds_pair_chart_prefs_column(tmp_path) -> None:
    db_path = tmp_path / "legacy_state.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE etf_pool (id INTEGER PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE ingestion_batch (id INTEGER PRIMARY KEY)"))
        conn.execute(
            text(
                """
                CREATE TABLE off_fund_research_state (
                    id INTEGER PRIMARY KEY,
                    start_date VARCHAR(8),
                    end_date VARCHAR(8),
                    adjust VARCHAR(8) NOT NULL DEFAULT 'hfq',
                    risk_free_rate FLOAT NOT NULL DEFAULT 0.025,
                    inner_mode VARCHAR(32) NOT NULL DEFAULT 'risk_parity_cov',
                    rp_window INTEGER NOT NULL DEFAULT 60,
                    rebalance_cycle VARCHAR(16) NOT NULL DEFAULT 'daily',
                    drift_rebalance_enabled BOOLEAN NOT NULL DEFAULT 1,
                    drift_abs_threshold FLOAT NOT NULL DEFAULT 0.05,
                    drift_rel_threshold FLOAT NOT NULL DEFAULT 0.25
                )
                """
            )
        )
    ensure_runtime_schema(engine)
    cols = {c["name"] for c in inspect(engine).get_columns("off_fund_research_state")}
    assert "pair_chart_prefs_json" in cols
