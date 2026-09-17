from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import inspect, text

from etf_momentum.analysis.off_fund_regression import DEFAULT_CN_STOCK_FACTORS
from etf_momentum.db.base import Base
from etf_momentum.db.init_db import init_db
from etf_momentum.db.models import OffFundRegressionFactorConfig
from etf_momentum.db.off_fund_regression_repo import (
    list_off_fund_factor_configs,
)
from etf_momentum.db.session import make_session_factory, make_sqlite_engine


def test_off_fund_factor_configs_default_and_upsert(api_client: TestClient) -> None:
    c = api_client
    r0 = c.get("/api/off-fund/regression/factor-configs")
    assert r0.status_code == 200
    rows0 = r0.json()
    assert isinstance(rows0, list)
    assert len(rows0) >= 1
    assert any(bool(x.get("is_active")) for x in rows0)
    active0 = next(x for x in rows0 if bool(x.get("is_active")))
    assert active0["is_legacy"] is True
    assert isinstance(active0.get("effective_benchmark_factors"), list)
    assert len(active0["effective_benchmark_factors"]) == len(DEFAULT_CN_STOCK_FACTORS)

    r1 = c.post(
        "/api/off-fund/regression/factor-configs",
        json={
            "name": "测试模板A",
            "set_active": True,
            "benchmark_profile": "cn_stock_core",
            "benchmark_factors": [
                {
                    "key": "F300",
                    "label": "沪深300",
                    "aliases": ["000300", "510300"],
                    "reporting_group": "大盘",
                    "substitution_group": "CSI300",
                    "asset_class": "equity",
                    "future_metadata": {"version": 3},
                },
                {"key": "F500", "label": "中证500", "aliases": ["000905", "510500"]},
            ],
            "solver_params": {
                "lambda_substitution": 0.003,
                "lambda_temporal": 0.03,
            },
        },
    )
    assert r1.status_code == 200
    out1 = r1.json()
    assert out1["name"] == "测试模板A"
    assert out1["is_active"] is True
    assert len(out1["benchmark_factors"]) == 2
    assert len(out1["effective_benchmark_factors"]) == 2
    assert out1["benchmark_factors"][0]["reporting_group"] == "大盘"
    assert out1["benchmark_factors"][0]["substitution_group"] == "CSI300"
    assert out1["benchmark_factors"][0]["future_metadata"] == {"version": 3}
    assert out1["solver_params"]["lambda_temporal"] == 0.03

    r2 = c.get("/api/off-fund/regression/factor-configs")
    assert r2.status_code == 200
    rows2 = r2.json()
    by_name = {x["name"]: x for x in rows2}
    assert by_name["测试模板A"]["is_active"] is True
    assert len(by_name["测试模板A"]["benchmark_factors"]) == 2
    assert by_name["测试模板A"]["benchmark_factors"][0]["future_metadata"] == {
        "version": 3
    }
    assert by_name["测试模板A"]["solver_params"]["lambda_substitution"] == 0.003


def test_off_fund_factor_configs_activate_and_delete(api_client: TestClient) -> None:
    c = api_client
    r1 = c.post(
        "/api/off-fund/regression/factor-configs",
        json={
            "name": "模板1",
            "set_active": False,
            "benchmark_profile": "cn_stock_core",
        },
    )
    assert r1.status_code == 200
    r2 = c.post(
        "/api/off-fund/regression/factor-configs",
        json={
            "name": "模板2",
            "set_active": False,
            "benchmark_profile": "cn_stock_core",
        },
    )
    assert r2.status_code == 200

    ra = c.post("/api/off-fund/regression/factor-configs/模板2/activate")
    assert ra.status_code == 200
    rows = c.get("/api/off-fund/regression/factor-configs").json()
    by = {x["name"]: x for x in rows}
    assert by["模板2"]["is_active"] is True
    assert by["模板1"]["is_active"] is False

    rd = c.delete("/api/off-fund/regression/factor-configs/模板2")
    assert rd.status_code == 200
    rows2 = c.get("/api/off-fund/regression/factor-configs").json()
    names = [x["name"] for x in rows2]
    assert "模板2" not in names
    # deleting active should keep some config active
    assert any(bool(x.get("is_active")) for x in rows2)


def test_off_fund_pair_universe_matches_default_factors(
    api_client: TestClient,
) -> None:
    c = api_client
    r = c.get("/api/off-fund/regression/pair-universe")
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list)
    assert len(rows) == len(DEFAULT_CN_STOCK_FACTORS)
    expected_keys = [spec.key for spec in DEFAULT_CN_STOCK_FACTORS]
    assert [str(x.get("key")) for x in rows] == expected_keys
    assert all(bool(str(x.get("label") or "").strip()) for x in rows)
    assert all(bool(str(x.get("etf_code") or "").strip()) for x in rows)
    by_key = {str(x.get("key")): x for x in rows}
    assert by_key["CSI300"]["etf_code"] == "510300"
    assert by_key["GOLD_SPOT"]["etf_code"] == "518880"


def test_replication_system_templates_are_idempotent_and_do_not_change_active(
    api_client: TestClient,
) -> None:
    first = api_client.get("/api/off-fund/regression/factor-configs")
    second = api_client.get("/api/off-fund/regression/factor-configs")
    assert first.status_code == second.status_code == 200
    first_rows = first.json()
    second_rows = second.json()
    identities = [
        (row["template_id"], row["template_version"])
        for row in first_rows
        if row.get("template_id")
    ]
    assert len(identities) == len(set(identities))
    assert identities == [
        (row["template_id"], row["template_version"])
        for row in second_rows
        if row.get("template_id")
    ]
    size = next(
        row
        for row in first_rows
        if row.get("template_id") == "cn_equity_size"
        and row.get("template_version") == 2
    )
    assert size["is_active"] is False
    assert {factor["key"] for factor in size["benchmark_factors"]} == {
        "CSI300",
        "CSI500",
        "CSI1000",
        "CSI2000",
    }
    overwrite = api_client.post(
        "/api/off-fund/regression/factor-configs",
        json={
            "name": "A股市值模板 v2",
            "set_active": True,
            "benchmark_factors": [
                {"key": "A", "aliases": ["A"]},
                {"key": "B", "aliases": ["B"]},
            ],
        },
    )
    assert overwrite.status_code == 400


def test_init_db_preserves_legacy_name_collision_and_adds_system_identity() -> None:
    engine = make_sqlite_engine()
    Base.metadata.create_all(engine)
    sf = make_session_factory(engine)
    with sf() as db:
        db.add(
            OffFundRegressionFactorConfig(
                name="A股市值模板 v2",
                is_active=True,
                benchmark_profile="cn_stock_core",
                benchmark_factors_json='[{"key":"OLD","aliases":["OLD"],"x":1}]',
            )
        )
        db.commit()
    init_db(engine)
    with sf() as db:
        rows = list_off_fund_factor_configs(db)
        legacy = next(row for row in rows if row.name == "A股市值模板 v2")
        system = next(
            row
            for row in rows
            if row.template_id == "cn_equity_size" and row.template_version == 2
        )
        assert legacy.is_active is True
        assert legacy.benchmark_factors[0]["x"] == 1
        assert system.name != legacy.name
        assert system.is_active is False
    constraints = {
        item["name"]
        for item in inspect(engine).get_unique_constraints(
            "off_fund_regression_factor_config"
        )
    }
    assert "uq_off_fund_factor_template_identity" in constraints


def test_init_db_additively_upgrades_legacy_factor_config_table() -> None:
    engine = make_sqlite_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE off_fund_regression_factor_config ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "name VARCHAR(128) NOT NULL UNIQUE,"
                "is_active BOOLEAN NOT NULL DEFAULT 0,"
                "benchmark_profile VARCHAR(64) NOT NULL,"
                "benchmark_factors_json TEXT,"
                "created_at DATETIME,"
                "updated_at DATETIME"
                ")"
            )
        )
        conn.execute(
            text(
                "INSERT INTO off_fund_regression_factor_config "
                "(name,is_active,benchmark_profile,benchmark_factors_json) "
                "VALUES ('legacy-row',1,'cn_stock_core',:factors)"
            ),
            {"factors": ('[{"key":"OLD","aliases":["OLD"],"future":7}]')},
        )
    init_db(engine)
    columns = {
        item["name"]
        for item in inspect(engine).get_columns("off_fund_regression_factor_config")
    }
    assert {"template_id", "template_version", "solver_params_json"} <= columns
    sf = make_session_factory(engine)
    with sf() as db:
        rows = list_off_fund_factor_configs(db)
        legacy = next(row for row in rows if row.name == "legacy-row")
        assert legacy.is_active is True
        assert legacy.benchmark_factors[0]["future"] == 7
